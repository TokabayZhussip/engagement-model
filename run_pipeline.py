"""Полный эксперимент статьи: признаки → индекс E(t) → кластеры и HMM → прогноз.

Примеры:
    python run_pipeline.py                     # настоящий OULAD из data/raw/oulad
    python run_pipeline.py --synthetic         # синтетические данные (проверка кода)
    python run_pipeline.py --no-lstm           # без LSTM (быстрее)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engagement import plots
from engagement.data import load_oulad
from engagement.features import build_weekly_features
from engagement.index import build_engagement_index, validate_index
from engagement.predict import (
    auc_by_week, early_warning, fit_lstm, fit_tabular_models, make_drop_dataset, split_by_presentation,
)
from engagement.states import cluster_profiles, cluster_trajectories, fit_hmm, trajectory_matrix


def log(msg: str, t0: float):
    print(f"[{time.time() - t0:7.1f} c] {msg}", flush=True)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="сгенерировать и использовать синтетические данные")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--out", default=None, help="папка результатов (по умолчанию из конфига)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = cfg["random_state"]
    raw_dir = cfg["data"]["raw_dir"]
    out = Path(args.out or cfg["output_dir"])
    if args.synthetic:
        from scripts.make_synthetic_oulad import generate
        raw_dir = "data/raw/synthetic"
        generate(Path(raw_dir))
        out = out / "synthetic" if args.out is None else out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    t0 = time.time()

    # 1. Данные и признаки
    d = load_oulad(raw_dir)
    log(f"загружено: studentVle {len(d['student_vle']):,} строк", t0)
    panel = build_weekly_features(d, cfg["panel"]["max_week"], cfg["features"]["activity_types"])
    log(f"панель студент×неделя: {panel.shape}", t0)

    emo_path = Path(cfg["data"]["processed_dir"]) / "emotion.csv"
    ic = cfg["index"]
    if ic["emotional_weight"] > 0 and emo_path.exists():
        emo = pd.read_csv(emo_path, dtype={"code_module": str, "code_presentation": str})
        panel = panel.merge(emo, on=["code_module", "code_presentation", "id_student", "week"], how="left")

    # 2. Индекс вовлечённости (веса оцениваются только по обучающим потокам)
    pc = cfg["prediction"]
    train_mask = ~panel["code_presentation"].isin(pc["test_presentations"])
    idx, weights = build_engagement_index(
        panel, ic["normalization"], ic["weighting"], ic["ema_alpha"], ic["emotional_weight"], train_mask,
    )
    validity = validate_index(idx, d["student_info"])
    log(f"индекс E(t) построен; ρ(E, итог) = {validity['spearman_rho_E_vs_outcome']:.3f}", t0)

    # сравнение схем взвешивания (для таблицы в статье)
    ablation = []
    for norm in ("rank", "minmax"):
        for w in ("entropy", "pca", "equal"):
            i2, _ = build_engagement_index(panel, norm, w, ic["ema_alpha"], 0.0, train_mask)
            v = validate_index(i2, d["student_info"])
            ablation.append({"normalization": norm, "weighting": w,
                             "spearman_rho": v["spearman_rho_E_vs_outcome"], "cronbach_alpha": v["cronbach_alpha"]})
    pd.DataFrame(ablation).to_csv(out / "index_ablation.csv", index=False)

    # 3. Кластеры траекторий и HMM
    cc, hc = cfg["clustering"], cfg["hmm"]
    mat = trajectory_matrix(idx, cc["weeks"])
    cl = cluster_trajectories(mat, cc["k_range"], cc["silhouette_sample"], seed)
    profiles = cluster_profiles(cl["labels"], d["student_info"])
    cl["selection"].to_csv(out / "clusters_selection.csv", index=False)
    profiles.to_csv(out / "clusters_profiles.csv")
    log(f"k-means: лучшее k = {cl['best_k']}", t0)

    hmm = fit_hmm(idx, hc["n_states_range"], hc["max_sequences"], hc["n_iter"], seed)
    idx["hmm_state"] = hmm["states"]
    hmm["selection"].to_csv(out / "hmm_selection.csv", index=False)
    hmm["transmat"].to_csv(out / "hmm_transmat.csv")
    hmm["means"].to_csv(out / "hmm_state_means.csv")
    log(f"HMM: лучшее число состояний = {hmm['best_n']}", t0)

    idx.to_parquet(out / "engagement_index.parquet", index=False) if _has_parquet() else \
        idx.to_csv(out / "engagement_index.csv.gz", index=False)

    # 4. Прогноз падения вовлечённости
    ds = make_drop_dataset(idx, pc["horizon"], pc["history"], pc["drop_ratio"], pc["min_history_engagement"])
    tr, va, te = split_by_presentation(ds, pc["test_presentations"], pc["val_fraction"], seed)
    log(f"задача A: {len(ds):,} строк, доля y=1 {ds['y'].mean():.3f}; train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}", t0)

    results = fit_tabular_models(ds, tr, va, te, seed)
    test_df = ds[te].reset_index(drop=True)
    metrics_a = {k: v["metrics"] for k, v in results.items()}
    curves_a = {k: auc_by_week(test_df, v["p_test"], pc["eval_weeks"]) for k, v in results.items()}

    if pc["lstm"]["enabled"] and not args.no_lstm:
        lstm = fit_lstm(idx, ds, tr, va, te, pc["lstm"], seed)
        metrics_a["lstm"] = lstm["metrics"]
        metrics_a["lstm"]["history"] = lstm["history"]
        curves_a["lstm"] = auc_by_week(test_df, lstm["p_test"], pc["eval_weeks"])
    for k, m in metrics_a.items():
        log(f"  {k:7s} ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  F1 {m['f1']:.3f}", t0)

    # 5. Раннее предупреждение (курсовой риск)
    ew = early_warning(idx, d["student_info"], pc["eval_weeks"], pc["test_presentations"],
                       pc["early_warning_target_auc"], seed)
    log(f"горизонт раннего предупреждения: неделя {ew['early_warning_week']}", t0)

    # 6. Рисунки и отчёт
    fig = out / "figures"
    plots.e_by_outcome(idx, d["student_info"], fig / "fig1_E_by_outcome.png")
    plots.cluster_centers(cl["centers"], profiles["n_students"], fig / "fig2_clusters.png")
    plots.transition_heatmap(hmm["transmat"], fig / "fig3_hmm_transitions.png")
    plots.auc_curves(curves_a, fig / "fig4_auc_by_week_drop.png", "Прогноз падения вовлечённости")
    plots.auc_curves({"GBM": {k: v["roc_auc"] for k, v in ew["by_week"].items()}},
                     fig / "fig5_early_warning.png", "Раннее предупреждение: риск Withdrawn/Fail",
                     target=pc["early_warning_target_auc"])

    metrics = {
        "data": "synthetic" if args.synthetic else "OULAD",
        "panel_shape": list(panel.shape),
        "index_weights": weights,
        "index_validity": validity,
        "clusters": {"best_k": cl["best_k"]},
        "hmm": {"best_n_states": hmm["best_n"], "transmat": hmm["transmat"].round(4).values.tolist()},
        "drop_prediction": metrics_a,
        "drop_auc_by_week": curves_a,
        "early_warning": ew,
    }
    (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, metrics, profiles)
    log(f"готово, результаты в {out}", t0)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def write_report(out: Path, m: dict, profiles: pd.DataFrame):
    v = m["index_validity"]
    w = m["index_weights"]
    lines = [
        f"# Результаты эксперимента ({m['data']})",
        "",
        "## 1. Индекс вовлечённости E(t)",
        "",
        f"Веса компонент: B = {w['components']['B']:.3f}, C = {w['components']['C']:.3f}.",
        "",
        "| Индикатор | Вес внутри компоненты |",
        "|---|---|",
        *[f"| {k[2:]} (B) | {x:.3f} |" for k, x in w["behavioral"].items()],
        *[f"| {k[2:]} (C) | {x:.3f} |" for k, x in w["cognitive"].items()],
        "",
        f"- α Кронбаха: {v['cronbach_alpha']:.3f}",
        f"- ρ Спирмена (средний E за первые {v['first_weeks']} нед. vs итог): "
        f"{v['spearman_rho_E_vs_outcome']:.3f} (p = {v['spearman_p']:.2e})",
        f"- Краскел–Уоллис: H = {v['kruskal_H']:.1f} (p = {v['kruskal_p']:.2e})",
        f"- Средний E по итогам: {v['E_mean_by_outcome']}",
        "",
        "Сравнение схем нормализации и взвешивания — `index_ablation.csv`.",
        "",
        "## 2. Типы траекторий и скрытые состояния",
        "",
        f"Лучшее число кластеров k-means (по силуэту): {m['clusters']['best_k']}.",
        "",
        profiles.to_markdown() if _has_tabulate() else profiles.to_string(),
        "",
        f"HMM: число состояний по BIC = {m['hmm']['best_n_states']}, матрица переходов — `hmm_transmat.csv`.",
        "",
        "## 3. Прогноз падения вовлечённости (тестовый поток)",
        "",
        "| Модель | ROC-AUC | PR-AUC | F1 |",
        "|---|---|---|---|",
        *[f"| {k} | {x['roc_auc']:.3f} | {x['pr_auc']:.3f} | {x['f1']:.3f} |" for k, x in m["drop_prediction"].items()],
        "",
        "## 4. Раннее предупреждение",
        "",
        "| Неделя | ROC-AUC | PR-AUC |",
        "|---|---|---|",
        *[f"| {t} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} |" for t, r in m["early_warning"]["by_week"].items()],
        "",
        f"Горизонт (первая неделя с ROC-AUC ≥ {m['early_warning']['target_auc']}): "
        f"**{m['early_warning']['early_warning_week']}**.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
