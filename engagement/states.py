"""Типы траекторий вовлечённости (k-means, GMM) и скрытые состояния (HMM).

Матрица переходов HMM между состояниями «низкая / средняя / высокая вовлечённость»
переиспользуется в статье 3 как часть модели переходов MDP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .data import KEY


def trajectory_matrix(idx: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Студенты × недели 0..weeks−1 значений E(t).

    После отчисления E = 0, у неотчисленных пропуски заполняются последним значением.
    """
    sub = idx[idx["week"] < weeks]
    mat = sub.pivot_table(index=KEY, columns="week", values="E", observed=True)
    mat = mat.reindex(columns=range(weeks))
    withdrawn = sub.groupby(KEY, observed=True)["withdraw_week"].first().reindex(mat.index).notna()
    filled = mat.ffill(axis=1)
    filled[withdrawn] = mat[withdrawn].fillna(0.0)
    return filled.fillna(0.0)


def bic_elbow(sel: pd.DataFrame, tol: float) -> int:
    """Наименьшее k, после которого BIC улучшается меньше чем на tol (относительно)."""
    s = sel.sort_values("k").reset_index(drop=True)
    for i in range(len(s) - 1):
        gain = (s.loc[i, "gmm_bic"] - s.loc[i + 1, "gmm_bic"]) / abs(s.loc[i, "gmm_bic"])
        if gain < tol:
            return int(s.loc[i, "k"])
    return int(s["k"].iloc[-1])


def cluster_trajectories(mat: pd.DataFrame, k_range, sample: int, seed: int,
                         k_final: int | None = None, bic_tol: float = 0.02) -> dict:
    """Выбор числа кластеров: силуэт (k-means), BIC (GMM, «локоть»), итоговое k.

    k_final задаёт итоговое число кластеров явно (например, 4 — по интерпретируемости);
    если не задано, берётся k по «локтю» BIC.
    """
    x = mat.to_numpy()
    n = len(x)
    sample = min(sample, n)
    rows = []
    for k in k_range:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        sil = silhouette_score(x, km.labels_, sample_size=sample, random_state=seed)
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(x)
        rows.append({"k": k, "kmeans_inertia": km.inertia_, "silhouette": sil, "gmm_bic": gmm.bic(x)})
    sel = pd.DataFrame(rows)
    k_sil = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    k_bic = bic_elbow(sel, bic_tol)
    k_used = int(k_final) if k_final else k_bic

    km = KMeans(n_clusters=k_used, n_init=10, random_state=seed).fit(x)
    # нумерация кластеров по возрастанию средней вовлечённости
    order = np.argsort(km.cluster_centers_.mean(axis=1))
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series([remap[l] for l in km.labels_], index=mat.index, name="cluster")
    centers = pd.DataFrame(km.cluster_centers_[order], columns=mat.columns)
    return {"selection": sel, "best_k": k_used, "k_silhouette": k_sil, "k_bic": k_bic,
            "labels": labels, "centers": centers}


def describe_clusters(centers: pd.DataFrame, slope_tol: float = 0.08) -> pd.DataFrame:
    """Числовое описание центров кластеров и рабочее название типа траектории.

    Название — эвристика для черновика: проверьте его по рисунку и при необходимости переименуйте.
    """
    w = centers.shape[1]
    q = max(w // 5, 2)
    start = centers.iloc[:, :q].mean(axis=1)
    end = centers.iloc[:, -q:].mean(axis=1)
    mean = centers.mean(axis=1)
    names = []
    for s_, e_, m_ in zip(start, end, mean):
        if e_ < 0.1:
            names.append("выбывание (E падает до нуля)")
            continue
        level = "низкий" if m_ < 0.4 else ("средний" if m_ < 0.55 else "высокий")
        if e_ - s_ <= -slope_tol:
            trend = "снижается"
        elif e_ - s_ >= slope_tol:
            trend = "растёт"
        else:
            trend = "стабилен"
        names.append(f"{level} уровень, {trend}")
    return pd.DataFrame({"E_mean": mean.round(3), "E_start": start.round(3), "E_end": end.round(3),
                         "change": (end - start).round(3), "type": names}, index=centers.index)


def cluster_profiles(labels: pd.Series, student_info: pd.DataFrame,
                     description: pd.DataFrame | None = None) -> pd.DataFrame:
    df = labels.reset_index().merge(student_info[KEY + ["final_result"]], on=KEY, how="left")
    tab = pd.crosstab(df["cluster"], df["final_result"], normalize="index").round(3)
    tab["n_students"] = df.groupby("cluster").size()
    tab["share"] = (tab["n_students"] / tab["n_students"].sum()).round(3)
    if description is not None:
        tab = description.join(tab)
    return tab


def fit_hmm(idx: pd.DataFrame, n_states_range, max_sequences: int, n_iter: int, seed: int,
            train_mask: pd.Series | None = None) -> dict:
    """Гауссовская HMM на последовательностях (B(t), C(t)); число состояний — по BIC.

    Модель обучается только на студентах обучающих потоков (train_mask), чтобы
    определение «падения» через HMM не использовало тестовый поток.
    Возвращает:
      states        — состояния по Витерби (по всей траектории, для описания);
      filtered      — апостериорные вероятности состояний по данным до недели t
                      включительно (прямой проход), без заглядывания в будущее.
    """
    rng = np.random.default_rng(seed)
    if train_mask is not None:
        idx = idx.assign(_train=np.asarray(train_mask, bool))
    idx = idx.sort_values(KEY + ["week"])
    groups = idx.groupby(KEY, sort=False, observed=True).indices
    keys = list(groups)
    if train_mask is not None:
        is_train = idx["_train"].to_numpy()
        keys_fit = [k for k in keys if is_train[groups[k][0]]]
    else:
        keys_fit = keys
    pick = rng.choice(len(keys_fit), size=min(max_sequences, len(keys_fit)), replace=False)

    feats = ["B", "C"]
    x_all = idx[feats].to_numpy()
    seqs = [groups[keys_fit[i]] for i in pick]
    x_fit = np.concatenate([x_all[s] for s in seqs])
    lengths = [len(s) for s in seqs]

    rows, models = [], {}
    for n in n_states_range:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=n_iter, random_state=seed)
        m.fit(x_fit, lengths)
        rows.append({"n_states": n, "log_likelihood": m.score(x_fit, lengths), "bic": m.bic(x_fit, lengths)})
        models[n] = m
    sel = pd.DataFrame(rows)
    best_n = int(sel.loc[sel["bic"].idxmin(), "n_states"])
    m = models[best_n]

    # состояния по возрастанию средней (B + C): 0 — самая низкая вовлечённость
    order = np.argsort(m.means_.sum(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))

    all_lengths = [len(v) for v in groups.values()]
    x_ordered = np.concatenate([x_all[v] for v in groups.values()])
    states = remap[m.predict(x_ordered, all_lengths)]
    row_index = np.concatenate(list(groups.values()))
    state_col = pd.Series(states, index=idx.index[row_index], name="hmm_state")

    trans = m.transmat_[np.ix_(order, order)]
    means = m.means_[order]
    var = np.diagonal(m.covars_, axis1=1, axis2=2)[order]
    start = m.startprob_[order]
    filtered = forward_filter(idx, feats, start, trans, means, var)

    return {"selection": sel, "best_n": best_n, "transmat": pd.DataFrame(trans),
            "means": pd.DataFrame(means, columns=feats), "states": state_col, "filtered": filtered}


def forward_filter(idx: pd.DataFrame, feats, start, trans, means, var) -> pd.DataFrame:
    """P(состояние_t | наблюдения_0..t) для каждой строки «студент × неделя».

    Ожидается, что у каждого студента недели идут подряд с 0 (так строится панель).
    """
    idx = idx.sort_values(KEY + ["week"])
    x = idx[feats].to_numpy(float)
    log_b = -0.5 * (((x[:, None, :] - means[None]) ** 2 / var[None]).sum(-1)
                    + np.log(2 * np.pi * var).sum(-1)[None])
    lik = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    pos = idx.groupby(KEY, sort=False, observed=True).cumcount().to_numpy()

    alpha = np.zeros_like(lik)
    for p in range(int(pos.max()) + 1):
        r = np.flatnonzero(pos == p)
        prior = np.broadcast_to(start, (len(r), len(start))) if p == 0 else alpha[r - 1] @ trans
        a = prior * lik[r] + 1e-300
        alpha[r] = a / a.sum(axis=1, keepdims=True)

    cols = [f"hmm_p{k}" for k in range(alpha.shape[1])]
    out = pd.DataFrame(alpha, index=idx.index, columns=cols)
    out["hmm_fstate"] = alpha.argmax(axis=1)
    return out
