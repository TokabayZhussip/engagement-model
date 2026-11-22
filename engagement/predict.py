"""Прогноз падения вовлечённости и горизонт раннего предупреждения.

Задача A (основная, недельная). Для студента в неделю t:
    H = среднее E за недели t−history+1..t,  F = среднее E за t+1..t+h.
    y = 1, если студент отчислился в (t, t+h] или F < drop_ratio · H.
    Рассматриваются только строки с H ≥ min_history_engagement.
Модели: логистическая регрессия, градиентный бустинг, LSTM.
Метрики: ROC-AUC, PR-AUC, F1 (порог подбирается на валидации), AUC по неделям.

Задача B (курсовая, раннее предупреждение). По данным до недели t предсказать
итог курса Withdrawn/Fail. Горизонт раннего предупреждения — первая неделя,
на которой ROC-AUC на тесте ≥ целевого значения.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import KEY

SEQ_FEATURES_EXTRA = ["B", "C", "E"]


def indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("n_")]


# ---------------------------------------------------------------- задача A

def make_drop_dataset(idx: pd.DataFrame, horizon: int, history: int, drop_ratio: float,
                      min_hist: float, label: str = "hmm") -> pd.DataFrame:
    """Строит выборку для задачи A.

    label="hmm"   (основной): студент сейчас НЕ в нижнем состоянии HMM (hmm_fstate ≥ 1),
                  y = 1, если в ближайшие h недель он перейдёт в нижнее состояние 0
                  или отчислится. Состояния — по прямому проходу (без заглядывания в будущее).
    label="ratio" (прежний): y = 1, если F < drop_ratio · H или отчисление.
    """
    df = idx.sort_values(KEY + ["week"]).reset_index(drop=True).copy()
    g = df.groupby(KEY, sort=False, observed=True)["E"]

    df["H"] = g.transform(lambda s: s.rolling(history, min_periods=1).mean())
    fut = [g.shift(-k) for k in range(1, horizon + 1)]
    df["F"] = pd.concat(fut, axis=1).mean(axis=1, skipna=False)

    for lag in (1, 2):
        df[f"E_lag{lag}"] = g.shift(lag).fillna(df["E"])
    df["dE"] = df["E"] - df["E_lag1"]

    ww = df["withdraw_week"]
    y_w = (ww > df["week"]) & (ww <= df["week"] + horizon)

    if label == "hmm":
        if "hmm_fstate" not in df.columns:
            raise ValueError("label='hmm' требует колонку hmm_fstate (запустите fit_hmm)")
        gs = df.groupby(KEY, sort=False, observed=True)["hmm_fstate"]
        fut_s = pd.concat([gs.shift(-k) for k in range(1, horizon + 1)], axis=1)
        y_drop = (fut_s == 0).any(axis=1)
        complete = fut_s.notna().all(axis=1)
        valid = (y_w | complete) & (df["hmm_fstate"] >= 1)
    elif label == "ratio":
        y_drop = (df["F"] < drop_ratio * df["H"]).fillna(False)
        valid = (y_w | df["F"].notna()) & (df["H"] >= min_hist)
    else:
        raise ValueError(f"Неизвестный тип метки: {label}")

    df["y_withdraw"] = y_w.astype(int)
    df["y"] = (y_w | y_drop).astype(int)
    return df[valid].reset_index(drop=True)


def tabular_features(df: pd.DataFrame) -> list[str]:
    hmm = [c for c in df.columns if c.startswith("hmm_p")]
    return indicator_columns(df) + ["B", "C", "E", "E_lag1", "E_lag2", "dE", "H", "week"] + hmm


def split_by_presentation(df: pd.DataFrame, test_presentations: list[str], val_fraction: float, seed: int):
    test = df["code_presentation"].isin(test_presentations).to_numpy()
    students = df.loc[~test, KEY].drop_duplicates()
    rng = np.random.default_rng(seed)
    val_students = students.sample(frac=val_fraction, random_state=int(rng.integers(1e9)))
    val_key = set(map(tuple, val_students.to_numpy()))
    is_val = np.array([tuple(r) in val_key for r in df[KEY].to_numpy()]) & ~test
    train = ~test & ~is_val
    return train, is_val, test


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def evaluate(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, p >= thr)),
        "threshold": thr,
        "positive_rate": float(y.mean()),
        "n": int(len(y)),
    }


def auc_by_week(df: pd.DataFrame, p: np.ndarray, weeks: list[int]) -> dict:
    out = {}
    for w in weeks:
        m = (df["week"] == w).to_numpy()
        y = df.loc[m, "y"].to_numpy()
        if m.sum() > 10 and 0 < y.mean() < 1:
            out[int(w)] = float(roc_auc_score(y, p[m]))
    return out


def fit_tabular_models(df, train, val, test, seed) -> dict:
    feats = tabular_features(df)
    X, y = df[feats].to_numpy(), df["y"].to_numpy()
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                              random_state=seed),
    }
    res = {}
    for name, m in models.items():
        m.fit(X[train], y[train])
        p_val = m.predict_proba(X[val])[:, 1]
        p_test = m.predict_proba(X[test])[:, 1]
        thr = best_f1_threshold(y[val], p_val)
        res[name] = {"metrics": evaluate(y[test], p_test, thr), "p_test": p_test, "model": m}
    return res


def markov_baseline(df: pd.DataFrame, transmat: np.ndarray, horizon: int) -> np.ndarray:
    """Базовая модель: P(попасть в состояние 0 за h шагов) только по матрице переходов HMM.

    Состояние 0 делается поглощающим; стартовое распределение — текущие
    апостериорные вероятности hmm_p*. Отчисление этой моделью не учитывается отдельно.
    """
    t = np.array(transmat, dtype=float).copy()
    t[0] = 0.0
    t[0, 0] = 1.0
    th = np.linalg.matrix_power(t, horizon)
    p = df[[c for c in df.columns if c.startswith("hmm_p")]].to_numpy()
    return (p @ th)[:, 0]


# ---------------------------------------------------------------- LSTM

def fit_lstm(idx: pd.DataFrame, df: pd.DataFrame, train, val, test, cfg: dict, seed: int) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    feats = indicator_columns(idx) + SEQ_FEATURES_EXTRA + [c for c in idx.columns if c.startswith("hmm_p")]
    L = int(cfg["window"])

    idx = idx.sort_values(KEY + ["week"]).reset_index(drop=True)
    groups = idx.groupby(KEY, sort=False, observed=True).indices
    keys = list(groups)
    key_pos = {k: i for i, k in enumerate(keys)}
    t_max = int(idx["week"].max()) + 1
    d = len(feats) + 1  # + флаг «неделя есть»

    seq = np.zeros((len(keys), t_max + L - 1, d), dtype=np.float32)
    x_all = idx[feats].to_numpy(np.float32)
    for k, rows in groups.items():
        w = idx["week"].to_numpy()[rows]
        seq[key_pos[k], w + L - 1, :-1] = x_all[rows]
        seq[key_pos[k], w + L - 1, -1] = 1.0

    s_idx = np.array([key_pos[tuple(r)] for r in df[KEY].itertuples(index=False, name=None)])
    t_idx = df["week"].to_numpy().astype(int)
    y_all = df["y"].to_numpy().astype(np.float32)
    offs = np.arange(L)

    def batch(ids):
        return torch.from_numpy(seq[s_idx[ids, None], t_idx[ids, None] + offs])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(d, cfg["hidden"], batch_first=True)
            self.head = nn.Sequential(nn.Linear(cfg["hidden"], 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    tr_ids, va_ids, te_ids = np.where(train)[0], np.where(val)[0], np.where(test)[0]
    pos = y_all[tr_ids].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))

    def predict(ids):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 4096):
                out.append(torch.sigmoid(net(batch(ids[i:i + 4096]))).numpy())
        return np.concatenate(out) if out else np.array([])

    rng = np.random.default_rng(seed)
    best_auc, best_state, bad = -1.0, None, 0
    history = []
    for epoch in range(cfg["epochs"]):
        net.train()
        perm = rng.permutation(tr_ids)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(batch(b)), torch.from_numpy(y_all[b]))
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_all[va_ids], predict(va_ids))
        history.append({"epoch": epoch + 1, "val_auc": float(auc)})
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)

    thr = best_f1_threshold(y_all[va_ids], predict(va_ids))
    p_test = predict(te_ids)
    return {"metrics": evaluate(y_all[te_ids], p_test, thr), "p_test": p_test, "history": history}


# ---------------------------------------------------------------- задача B

def early_warning(idx: pd.DataFrame, student_info: pd.DataFrame, weeks: list[int],
                  test_presentations: list[str], target_auc: float, seed: int) -> dict:
    """Курсовой риск (Withdrawn/Fail) по данным до недели t включительно."""
    info = student_info[KEY + ["final_result"]].copy()
    info["at_risk"] = info["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    ncols = indicator_columns(idx)
    rows = {}
    for t in weeks:
        upto = idx[idx["week"] <= t]
        # только студенты, ещё не отчисленные к неделе t
        still = upto.groupby(KEY, observed=True)["withdraw_week"].first()
        still = still[~(still <= t)].index
        upto = upto.set_index(KEY).loc[still].reset_index()

        g = upto.sort_values("week").groupby(KEY, observed=True)
        agg = pd.concat([
            g["E"].mean().rename("E_mean"),
            g["E"].last().rename("E_last"),
            g["E"].std().fillna(0).rename("E_std"),
            g["B"].mean().rename("B_mean"),
            g["C"].mean().rename("C_mean"),
            g[ncols].last(),
        ], axis=1)
        e_first = g["E"].first()
        agg["E_slope"] = (agg["E_last"] - e_first) / (t + 1)
        data = agg.reset_index().merge(info, on=KEY)

        test = data["code_presentation"].isin(test_presentations).to_numpy()
        feats = [c for c in agg.columns]
        X, y = data[feats].to_numpy(), data["at_risk"].to_numpy()
        if test.sum() == 0 or (~test).sum() == 0 or y[test].min() == y[test].max():
            continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        m.fit(X[~test], y[~test])
        p = m.predict_proba(X[test])[:, 1]
        rows[int(t)] = {"roc_auc": float(roc_auc_score(y[test], p)),
                        "pr_auc": float(average_precision_score(y[test], p)),
                        "n_test": int(test.sum())}
    reached = [t for t, r in rows.items() if r["roc_auc"] >= target_auc]
    return {"by_week": rows, "target_auc": target_auc,
            "early_warning_week": min(reached) if reached else None}
