"""Недельная панель признаков «студент × неделя» из таблиц OULAD.

Поведенческие индикаторы (B):
    clicks              — сумма кликов за неделю
    active_days         — число дней с активностью
    n_sites             — число разных ресурсов
    type_entropy        — нормированная энтропия по типам активностей (разнообразие)
    weeks_since_active  — недель с последней активности (направление «−»)

Когнитивные индикаторы (C):
    submission_rate_cum — доля сданных заданий от заданий со сроком до текущей недели
    score_cum           — средний балл по сданным заданиям (накопительно)
    lateness_cum        — среднее опоздание сдачи в днях (направление «−»)
    quiz_share          — доля кликов по самопроверке (quiz) среди всех кликов
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import COURSE, KEY, course_weeks

BEHAVIORAL = {
    "clicks": +1,
    "active_days": +1,
    "n_sites": +1,
    "type_entropy": +1,
    "weeks_since_active": -1,
}
COGNITIVE = {
    "submission_rate_cum": +1,
    "score_cum": +1,
    "lateness_cum": -1,
    "quiz_share": +1,
}
WK = KEY + ["week"]


def build_panel(d: dict[str, pd.DataFrame], max_week: int) -> pd.DataFrame:
    """Каркас панели: для каждой регистрации недели 0..последняя.

    Для отчисленных панель заканчивается неделей отчисления (включительно),
    студенты, отчисленные до начала курса, исключаются.
    """
    reg = d["student_registration"].merge(course_weeks(d["courses"], max_week), on=COURSE)
    reg["withdraw_week"] = np.floor(reg["date_unregistration"] / 7)
    reg = reg[~(reg["withdraw_week"] < 0)].copy()

    last = reg["n_weeks"] - 1
    has_w = reg["withdraw_week"].notna()
    last[has_w] = np.minimum(last[has_w], reg.loc[has_w, "withdraw_week"]).astype(int)
    reg["last_week"] = last.astype(int)

    counts = (reg["last_week"] + 1).to_numpy()
    panel = reg.loc[reg.index.repeat(counts), KEY + ["withdraw_week", "n_weeks"]].reset_index(drop=True)
    panel["week"] = panel.groupby(KEY, sort=False).cumcount().astype(int)
    return panel


def _vle_features(d: dict[str, pd.DataFrame], activity_types: list[str]) -> pd.DataFrame:
    sv = d["student_vle"]
    sv = sv[sv["date"] >= 0].copy()
    sv["week"] = (sv["date"] // 7).astype(int)

    daily = sv.groupby(WK + ["date"], observed=True, sort=False)["sum_click"].sum().reset_index()
    wk = daily.groupby(WK, observed=True).agg(
        clicks=("sum_click", "sum"), active_days=("date", "count")
    )

    sites = sv[WK + ["id_site"]].drop_duplicates()
    wk["n_sites"] = sites.groupby(WK, observed=True).size()

    sv = sv.merge(d["vle"][["id_site", "activity_type"]], on="id_site", how="left")
    by_type = (
        sv.groupby(WK + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)
    )
    p = by_type.div(by_type.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    n_types = max(by_type.shape[1], 2)
    wk["type_entropy"] = pd.Series(ent / np.log(n_types), index=by_type.index)

    quiz = by_type["quiz"] if "quiz" in by_type.columns else 0
    wk["quiz_clicks"] = quiz
    for t in activity_types:
        wk[f"clicks_{t}"] = by_type[t] if t in by_type.columns else 0
    wk = wk.reset_index()
    for col in COURSE:
        wk[col] = wk[col].astype(str)
    return wk


def _assessment_features(d: dict[str, pd.DataFrame], max_week: int,
                         score_delay: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """score_delay — через сколько недель после сдачи балл считается известным платформе
    (в OULAD дата публикации оценки отсутствует)."""
    ass = d["assessments"]
    sa = d["student_assessment"].merge(ass, on="id_assessment", how="inner")
    sa = sa[sa["assessment_type"] != "Exam"].copy()  # экзамен в конце, на прогноз не влияет
    sa["week"] = (sa["date_submitted"].clip(lower=0) // 7).astype(int).clip(upper=max_week - 1)
    sa["late"] = (sa["date_submitted"] - sa["date"]).astype(float)
    sa["scored"] = sa["score"].notna().astype(int)
    sa["late_n"] = sa["late"].notna().astype(int)

    per_week = sa.groupby(WK).agg(
        n_sub=("id_assessment", "count"),
        sum_late=("late", "sum"),
        n_late=("late_n", "sum"),
    ).reset_index()
    # баллы учитываются с недели сдачи + score_delay
    sa["score_week"] = (sa["week"] + score_delay).clip(upper=max_week - 1)
    scores = sa.groupby(KEY + ["score_week"]).agg(
        sum_score=("score", "sum"), n_scored=("scored", "sum")).reset_index().rename(columns={"score_week": "week"})
    per_week = per_week.merge(scores, on=WK, how="outer")

    due = ass[(ass["assessment_type"] != "Exam") & ass["date"].notna()].copy()
    due["week"] = (due["date"] // 7).astype(int)
    due = due.groupby(COURSE + ["week"]).size().rename("n_due").reset_index()
    return per_week, due


def build_weekly_features(d: dict[str, pd.DataFrame], max_week: int, activity_types: list[str],
                          score_delay: int = 0) -> pd.DataFrame:
    panel = build_panel(d, max_week)

    vle = _vle_features(d, activity_types)
    panel = panel.merge(vle, on=WK, how="left")
    fill_cols = [c for c in vle.columns if c not in WK]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)

    per_week, due = _assessment_features(d, max_week, score_delay)
    panel = panel.merge(per_week, on=WK, how="left")
    acc = ["n_sub", "sum_score", "n_scored", "sum_late", "n_late"]
    panel[acc] = panel[acc].fillna(0.0)

    panel = panel.sort_values(WK).reset_index(drop=True)
    g = panel.groupby(KEY, sort=False)
    cum = g[acc].cumsum()

    # накопительное число заданий со сроком до текущей недели (по курсу)
    grid = d["courses"][COURSE].merge(pd.DataFrame({"week": range(max_week)}), how="cross")
    grid = grid.merge(due, on=COURSE + ["week"], how="left").fillna({"n_due": 0})
    grid = grid.sort_values(COURSE + ["week"])
    grid["cum_due"] = grid.groupby(COURSE)["n_due"].cumsum()
    panel = panel.merge(grid[COURSE + ["week", "cum_due"]], on=COURSE + ["week"], how="left")

    panel["submission_rate_cum"] = (cum["n_sub"] / panel["cum_due"].replace(0, np.nan)).clip(upper=1.0)
    panel["score_cum"] = cum["sum_score"] / cum["n_scored"].replace(0, np.nan)
    panel["lateness_cum"] = cum["sum_late"] / cum["n_late"].replace(0, np.nan)
    panel["quiz_share"] = (panel["quiz_clicks"] / panel["clicks"].replace(0, np.nan)).fillna(0.0)

    # недель с последней активности
    active_week = panel["week"].where(panel["clicks"] > 0)
    last_active = active_week.groupby([panel[k] for k in KEY], sort=False).ffill()
    panel["weeks_since_active"] = (panel["week"] - last_active).fillna(panel["week"] + 1)

    keep = (
        KEY + ["week", "n_weeks", "withdraw_week"]
        + list(BEHAVIORAL) + list(COGNITIVE)
        + [f"clicks_{t}" for t in activity_types]
    )
    return panel[keep]
