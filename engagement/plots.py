"""Рисунки для статьи."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data import KEY  # noqa: E402

OUTCOMES = ["Distinction", "Pass", "Fail", "Withdrawn"]


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def e_by_outcome(idx: pd.DataFrame, info: pd.DataFrame, path: Path, weeks: int = 30):
    df = idx[idx["week"] < weeks].merge(info[KEY + ["final_result"]], on=KEY)
    m = df.groupby(["final_result", "week"])["E"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    for o in OUTCOMES:
        if o in m.columns:
            ax.plot(m.index, m[o], label=o)
    ax.set_xlabel("Неделя курса")
    ax.set_ylabel("Средний индекс E(t)")
    ax.set_title("Динамика вовлечённости по итогам курса")
    ax.legend()
    _save(fig, path)


def cluster_centers(centers: pd.DataFrame, sizes: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, row in centers.iterrows():
        ax.plot(row.index, row.values, label=f"Кластер {k} (n={int(sizes.get(k, 0))})")
    ax.set_xlabel("Неделя курса")
    ax.set_ylabel("E(t)")
    ax.set_title("Типовые траектории вовлечённости (k-means)")
    ax.legend()
    _save(fig, path)


def transition_heatmap(trans: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(trans.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            v = trans.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black")
    ax.set_xlabel("Состояние в t+1")
    ax.set_ylabel("Состояние в t")
    ax.set_title("Матрица переходов HMM\n(0 — низкая вовлечённость)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def auc_curves(curves: dict[str, dict], path: Path, title: str, target: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in curves.items():
        if c:
            ax.plot(list(c.keys()), list(c.values()), marker="o", label=name)
    if target is not None:
        ax.axhline(target, ls="--", color="grey", label=f"цель AUC = {target}")
    ax.set_xlabel("Неделя прогноза")
    ax.set_ylabel("ROC-AUC (тест)")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)
