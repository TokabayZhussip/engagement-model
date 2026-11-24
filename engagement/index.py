"""Интегральный индекс вовлечённости E(t).

1. Нормализация индикаторов внутри «курс × поток × неделя» (процентиль или min-max
   по log1p), разворот индикаторов с направлением «−».
2. Веса индикаторов внутри компоненты — метод энтропийных весов (или PCA / равные).
3. Компоненты B(t) (поведенческая), C(t) (когнитивная), опционально Em(t) (эмоциональная).
4. E_raw(t) = w_B·B(t) + w_C·C(t), веса компонент тем же методом;
   E_raw(t) = (1 − λ)·E_raw(t) + λ·Em(t), если подключён эмоциональный компонент.
5. Сглаживание: E(t) = α·E_raw(t) + (1 − α)·E(t − 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import COURSE, KEY
from .features import BEHAVIORAL, COGNITIVE

GROUP = COURSE + ["week"]


def normalize(panel: pd.DataFrame, directions: dict[str, int], method: str = "rank") -> pd.DataFrame:
    """Возвращает нормализованные в [0, 1] индикаторы с колонками `n_<имя>`."""
    cols = list(directions)
    g = panel.groupby(GROUP, sort=False)[cols]
    if method == "rank":
        z = g.rank(pct=True, method="average")
    elif method == "minmax":
        x = panel[cols].astype(float).copy()
        for c in ("clicks", "n_sites"):  # счётчики с тяжёлым хвостом
            if c in x.columns:
                x[c] = np.log1p(x[c])
        gx = x.groupby([panel[c] for c in GROUP], sort=False)
        lo, hi = gx.transform("min"), gx.transform("max")
        z = (x - lo) / (hi - lo).replace(0, np.nan)
    else:
        raise ValueError(f"Неизвестная нормализация: {method}")
    for c, s in directions.items():
        if s < 0:
            z[c] = 1.0 - z[c]
    z = z.fillna(0.5)  # нет данных (напр., ещё не было заданий) — нейтральное значение
    return z.add_prefix("n_")


def entropy_weights(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Метод энтропийных весов: больший вес у индикаторов с большей дифференциацией."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    p = x / (x.sum(axis=0, keepdims=True) + eps)
    e = -(p * np.log(p + eps)).sum(axis=0) / np.log(n)
    d = 1.0 - e
    if d.sum() <= 0:
        return np.full(x.shape[1], 1.0 / x.shape[1])
    return d / d.sum()


def pca_weights(x: np.ndarray) -> np.ndarray:
    """Веса = модули нагрузок первой главной компоненты."""
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    w = np.abs(vt[0])
    return w / w.sum()


def compute_weights(x: np.ndarray, method: str) -> np.ndarray:
    if method == "entropy":
        return entropy_weights(x)
    if method == "pca":
        return pca_weights(x)
    if method == "equal":
        return np.full(x.shape[1], 1.0 / x.shape[1])
    raise ValueError(f"Неизвестный метод весов: {method}")


def cronbach_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    k = x.shape[1]
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def build_engagement_index(
    panel: pd.DataFrame,
    normalization: str = "rank",
    weighting: str = "entropy",
    ema_alpha: float = 0.5,
    emotional_weight: float = 0.0,
    weight_mask: pd.Series | None = None,
    cognitive: dict | None = None,
    behavioral: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Добавляет к панели n_*-индикаторы, B, C, E_raw, E. Возвращает (панель, веса).

    weight_mask — строки, по которым оцениваются веса (обучающая выборка),
    чтобы веса не «видели» тестовый поток.
    """
    nb = normalize(panel, behavioral or BEHAVIORAL, normalization)
    nc = normalize(panel, cognitive or COGNITIVE, normalization)
    out = pd.concat([panel, nb, nc], axis=1)

    mask = np.ones(len(out), bool) if weight_mask is None else np.asarray(weight_mask, bool)
    wb = compute_weights(nb.to_numpy()[mask], weighting)
    wc = compute_weights(nc.to_numpy()[mask], weighting)
    out["B"] = nb.to_numpy() @ wb
    out["C"] = nc.to_numpy() @ wc

    wcomp = compute_weights(out.loc[mask, ["B", "C"]].to_numpy(), weighting)
    e_raw = out[["B", "C"]].to_numpy() @ wcomp

    if emotional_weight > 0:
        if "emotion_score" not in out.columns:
            raise ValueError("emotional_weight > 0, но в панели нет колонки emotion_score")
        em = out["emotion_score"].fillna(0.5).to_numpy()
        e_raw = (1 - emotional_weight) * e_raw + emotional_weight * em
    out["E_raw"] = e_raw

    out = out.sort_values(KEY + ["week"]).reset_index(drop=True)
    out["E"] = (
        out.groupby(KEY, sort=False)["E_raw"]
        .transform(lambda s: s.ewm(alpha=ema_alpha, adjust=False).mean())
    )

    weights = {
        "behavioral": dict(zip(nb.columns, map(float, wb))),
        "cognitive": dict(zip(nc.columns, map(float, wc))),
        "components": {"B": float(wcomp[0]), "C": float(wcomp[1])},
        "emotional_weight": emotional_weight,
    }
    return out, weights


OUTCOME_ORDER = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}


def validate_index(idx: pd.DataFrame, student_info: pd.DataFrame, weeks: int = 10) -> dict:
    """Внешняя и внутренняя валидность индекса.

    - α Кронбаха по нормализованным индикаторам;
    - ρ Спирмена между средним E за первые `weeks` недель и итогом курса
      (Withdrawn < Fail < Pass < Distinction);
    - критерий Краскела–Уоллиса для различий E между группами итога.
    """
    ncols = [c for c in idx.columns if c.startswith("n_")]
    early = idx[idx["week"] < weeks].groupby(KEY)["E"].mean().rename("E_mean").reset_index()
    early = early.merge(student_info[KEY + ["final_result"]], on=KEY)
    early["outcome"] = early["final_result"].map(OUTCOME_ORDER)

    rho, p_rho = stats.spearmanr(early["E_mean"], early["outcome"])
    groups = [g["E_mean"].to_numpy() for _, g in early.groupby("final_result")]
    h, p_h = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)
    return {
        "cronbach_alpha": cronbach_alpha(idx[ncols].to_numpy()),
        "spearman_rho_E_vs_outcome": float(rho),
        "spearman_p": float(p_rho),
        "kruskal_H": float(h),
        "kruskal_p": float(p_h),
        "E_mean_by_outcome": early.groupby("final_result")["E_mean"].mean().round(4).to_dict(),
        "n_students": int(len(early)),
        "first_weeks": weeks,
    }
