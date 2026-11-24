"""Дополнительные расчёты по замечаниям рецензента (статья 2).

Запуск:
    python run_revision.py                    # OULAD, около 20–30 минут
    python run_revision.py --synthetic        # проверка кода на синтетических данных
Результаты: outputs/revision/report_revision.md и revision.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from engagement.data import KEY, load_oulad
from engagement.features import BEHAVIORAL, COGNITIVE, build_weekly_features
from engagement.index import OUTCOME_ORDER, build_engagement_index
from engagement.predict import (
    best_f1_threshold, fit_tabular_models, indicator_columns, make_drop_dataset, markov_baseline,
    split_by_presentation, tabular_features,
)
from engagement.states import fit_hmm, trajectory_matrix

warnings.filterwarnings("ignore")
T0 = time.time()
def log(m): print(f"[{time.time() - T0:7.1f} c] {m}", flush=True)

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, float) and np.isnan(o): return None
    return o

# ------------------------------------------------------------------ общие функции
def gbm(seed): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                                    random_state=seed)

def nolag_features(df):
    return indicator_columns(df) + ["B", "C", "E", "week"] + [c for c in df.columns if c.startswith("hmm_p")]

def boot_ci(y, p, groups, n_boot, seed, p2=None):
    """Кластерный бутстрап по студентам: 95% ДИ для ROC-AUC, PR-AUC (и разницы с p2)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == g)[0] for g in range(len(uniq))]
    out = {"roc": [], "pr": [], "d_roc": [], "d_pr": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by[g] for g in pick])
        yy = y[ii]
        if yy.min() == yy.max(): continue
        r, a = roc_auc_score(yy, p[ii]), average_precision_score(yy, p[ii])
        out["roc"].append(r); out["pr"].append(a)
        if p2 is not None:
            out["d_roc"].append(r - roc_auc_score(yy, p2[ii])); out["d_pr"].append(a - average_precision_score(yy, p2[ii]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    return {k: q(v) for k, v in out.items()}

def metrics(y, p): return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                           "pos_rate": float(y.mean()), "n": int(len(y))}

def student_ids(df): return (df["code_module"] + "|" + df["code_presentation"] + "|" + df["id_student"].astype(str)).to_numpy()

# ------------------------------------------------------------------ индекс и валидность
def build_variant(d, cfg, score_delay, no_scores, train_mask_fn):
    panel = build_weekly_features(d, cfg["panel"]["max_week"], cfg["features"]["activity_types"], score_delay)
    cog = {k: v for k, v in COGNITIVE.items() if not (no_scores and k == "score_cum")}
    tm = train_mask_fn(panel)
    idx, w = build_engagement_index(panel, "rank", "equal", cfg["index"]["ema_alpha"], 0.0, tm, cognitive=cog)
    return idx

def first_weeks_table(idx, info, weeks=10):
    """Средние за первые `weeks` недель; выбывшие раньше усредняются по доступным неделям."""
    sub = idx[idx["week"] < weeks].copy()
    c_ns = [c for c in ["n_submission_rate_cum", "n_lateness_cum", "n_quiz_share"] if c in sub.columns]
    sub["C_noscore"] = sub[c_ns].mean(axis=1)
    sub["E_noscore"] = 0.5 * sub["B"] + 0.5 * sub["C_noscore"]
    g = sub.groupby(KEY, observed=True)
    t = pd.DataFrame({"B": g["B"].mean(), "E": g["E"].mean(), "E_noscore": g["E_noscore"].mean(),
                      "n_weeks": g["week"].count(), "withdraw_week": g["withdraw_week"].first()}).reset_index()
    # «учебные баллы» как простой базовый показатель: средний балл за сданные задания к концу периода
    last = sub.sort_values("week").groupby(KEY, observed=True)["score_cum"].last().rename("score_raw").reset_index()
    t = t.merge(last, on=KEY, how="left")
    t = t.merge(info[KEY + ["final_result"]], on=KEY)
    t["outcome"] = t["final_result"].map(OUTCOME_ORDER)
    t["survivor"] = ~(t["withdraw_week"] < weeks)
    return t

def rho_table(t, test_pres):
    rows = []
    for name, col in [("Поведенческая компонента B", "B"), ("Средний балл за задания (без индекса)", "score_raw"),
                      ("Полный индекс E", "E"), ("Индекс без баллов (C без среднего балла)", "E_noscore")]:
        for scope, m in [("все потоки", np.ones(len(t), bool)), ("тест 2014J", t["code_presentation"].isin(test_pres).to_numpy())]:
            for pop, m2 in [("все студенты", np.ones(len(t), bool)), ("не выбывшие до 10-й нед.", t["survivor"].to_numpy())]:
                tt = t[m & m2]
                x = tt[col]
                ok = x.notna()
                rho, p = stats.spearmanr(x[ok], tt.loc[ok, "outcome"])
                rows.append({"показатель": name, "выборка": scope, "студенты": pop, "rho": float(rho), "p": float(p),
                             "n": int(ok.sum()), "без балла (NaN)": int((~ok).sum())})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ прогноз спада
def drop_eval(idx, hmm, pc, seed, test_pres, n_boot, train_pres=None, full=True):
    ds = make_drop_dataset(idx, pc["horizon"], pc["history"], pc["drop_ratio"], pc["min_history_engagement"], "hmm")
    if train_pres is not None:
        ds = ds[ds["code_presentation"].isin(list(train_pres) + list(test_pres))].reset_index(drop=True)
    tr, va, te = split_by_presentation(ds, test_pres, pc["val_fraction"], seed)
    # компоненты цели
    complete = (ds["future_complete"] == 1).to_numpy()
    no_w = (ds["y_withdraw"] == 0).to_numpy()
    A = hmm["transmat"].to_numpy()
    p_mk = markov_baseline(ds, A, pc["horizon"])
    grp = student_ids(ds)
    out = {"rows_total": int(len(ds)), "pos_rate_total": float(ds["y"].mean()),
           "pos_rate_test": float(ds.loc[te, "y"].mean()), "withdraw_share_of_pos_test":
           float(ds.loc[te, "y_withdraw"].sum() / max(ds.loc[te, "y"].sum(), 1))}

    def fit_pred(target, feats, rows_tr, rows_te):
        X, y = ds[feats].to_numpy(), ds[target].to_numpy()
        m = gbm(seed).fit(X[rows_tr], y[rows_tr]); return m.predict_proba(X[rows_te])[:, 1]

    res = {}
    # (а) общая цель (5) — как в статье
    full_f, nl_f = tabular_features(ds), nolag_features(ds)
    y = ds["y"].to_numpy()
    p_g = fit_pred("y", full_f, tr, te)
    res["combined"] = {"markov": metrics(y[te], p_mk[te]), "gbm": metrics(y[te], p_g)}
    if full:
        p_nl = fit_pred("y", nl_f, tr, te)
        res["combined"]["gbm_nolag"] = metrics(y[te], p_nl)
        res["combined"]["gbm_ci"] = boot_ci(y[te], p_g, grp[te], n_boot, seed, p2=p_mk[te])
    # (б) только переход в S0 (строки без отчисления в окне и с полным будущим) — сопоставимо с марковской моделью
    m_s0 = complete & no_w
    tr0, te0 = tr & m_s0, te & m_s0
    y0 = ds["y_s0"].to_numpy()
    p_g0 = fit_pred("y_s0", full_f, tr0, te0)
    res["s0_only"] = {"markov": metrics(y0[te0], p_mk[te0]), "gbm": metrics(y0[te0], p_g0)}
    if full:
        p_nl0 = fit_pred("y_s0", nl_f, tr0, te0)
        res["s0_only"]["gbm_nolag"] = metrics(y0[te0], p_nl0)
        res["s0_only"]["gbm_ci"] = boot_ci(y0[te0], p_g0, grp[te0], n_boot, seed, p2=p_mk[te0])
        res["s0_only"]["markov_ci"] = boot_ci(y0[te0], p_mk[te0], grp[te0], n_boot, seed)
        res["s0_only"]["gbm_nolag_vs_markov_ci"] = boot_ci(y0[te0], p_nl0, grp[te0], n_boot, seed, p2=p_mk[te0])
    # (в) только отчисление в ближайшие 2 недели
    yw = ds["y_withdraw"].to_numpy()
    p_gw = fit_pred("y_withdraw", full_f, tr, te)
    res["withdraw_only"] = {"markov_S0_as_score": metrics(yw[te], p_mk[te]), "gbm": metrics(yw[te], p_gw)}
    if full:
        res["withdraw_only"]["gbm_nolag"] = metrics(yw[te], fit_pred("y_withdraw", nl_f, tr, te))
    out["results"] = res
    return out, ds, (tr, va, te)

# ------------------------------------------------------------------ раннее предупреждение (все недели)
def early_warning_all(idx, info, weeks, test_pres, seed, train_pres=None):
    from engagement.predict import early_warning
    sub = idx if train_pres is None else idx[idx["code_presentation"].isin(list(train_pres) + list(test_pres))]
    ew = early_warning(sub, info, weeks, test_pres, 0.8, seed)
    # размер и доля положительного класса на каждой неделе
    inf = info[KEY + ["final_result"]].copy(); inf["risk"] = inf["final_result"].isin(["Withdrawn", "Fail"])
    for t in list(ew["by_week"]):
        upto = sub[(sub["week"] <= t) & sub["code_presentation"].isin(test_pres)]
        still = upto.groupby(KEY, observed=True)["withdraw_week"].first()
        still = still[~(still <= t)].reset_index()[KEY].merge(inf, on=KEY)
        ew["by_week"][t]["pos_rate_test"] = float(still["risk"].mean())
    return ew

# ------------------------------------------------------------------ диагностика СММ
def empirical_transitions(idx, col="hmm_fstate", K=4):
    s = idx.sort_values(KEY + ["week"])
    nxt = s.groupby(KEY, sort=False, observed=True)[col].shift(-1)
    m = nxt.notna()
    c = pd.crosstab(s.loc[m, col].astype(int), nxt[m].astype(int)).reindex(index=range(K), columns=range(K), fill_value=0)
    return c

def time_s1_to_exit(idx):
    """Недели от первого попадания в S1 (по прямому проходу) до первого S0 или отчисления."""
    rows = []
    for k, g in idx.sort_values(KEY + ["week"]).groupby(KEY, sort=False, observed=True):
        st, wk = g["hmm_fstate"].to_numpy(), g["week"].to_numpy()
        s1 = np.where(st == 1)[0]
        if not len(s1): continue
        t1 = wk[s1[0]]
        after = np.where((st == 0) & (wk > t1))[0]
        t_s0 = wk[after[0]] if len(after) else np.inf
        ww = g["withdraw_week"].iloc[0]
        t_w = ww if (pd.notna(ww) and ww >= t1) else np.inf
        t_exit = min(t_s0, t_w)
        rows.append({"t1": t1, "exit": np.isfinite(t_exit), "dt": (t_exit - t1) if np.isfinite(t_exit) else np.nan})
    r = pd.DataFrame(rows)
    e = r[r["exit"]]
    return {"students_with_S1": int(len(r)), "share_reaching_S0_or_withdraw": float(r["exit"].mean()),
            "weeks_median": float(e["dt"].median()), "weeks_q25": float(e["dt"].quantile(.25)),
            "weeks_q75": float(e["dt"].quantile(.75)), "share_le_2_weeks": float((e["dt"] <= 2).mean()),
            "share_ge_4_weeks": float((e["dt"] >= 4).mean())}

def entries_into_s0(idx):
    s = idx.sort_values(KEY + ["week"])
    prev = s.groupby(KEY, sort=False, observed=True)["hmm_fstate"].shift(1)
    m = (s["hmm_fstate"] == 0) & prev.notna() & (prev != 0)
    vc = prev[m].astype(int).value_counts().sort_index()
    return {"entries": int(m.sum()), "from_state": {int(k): int(v) for k, v in vc.items()},
            "share_from_S1": float(vc.get(1, 0) / max(m.sum(), 1))}

# ------------------------------------------------------------------ отчёт
def f(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))

def ci(c, n=3): return "—" if not c else f"[{c[0]:.{n}f}; {c[1]:.{n}f}]"

def make_report(R):
    L = ["# Пересчёт по замечаниям рецензента", ""]
    L += ["## 1. Валидность: ρ Спирмена среднего за первые 10 недель с итогом курса", "",
          "| показатель | выборка | студенты | ρ | n |", "|---|---|---|---|---|"]
    for r in R["validity"]:
        L.append(f"| {r['показатель']} | {r['выборка']} | {r['студенты']} | {f(r['rho'])} | {r['n']} |")
    L += ["", f"Выбыли до 10-й недели: {R['validity_note']['withdrawn_before_w10']} из {R['validity_note']['n_students']}", ""]

    L += ["## 2. Доступность признаков", "", "| вариант | ρ (все) | ρ (2014J) | ROC-AUC GBM (общая цель) | PR-AUC GBM | A[1,0] | A[2,0] |",
          "|---|---|---|---|---|---|---|"]
    for name, v in R["feature_availability"].items():
        g = v["drop"]["results"]["combined"]["gbm"]; A = v["transmat"]
        L.append(f"| {name} | {f(v['rho_all'])} | {f(v['rho_test'])} | {f(g['roc_auc'])} | {f(g['pr_auc'])} | {f(A[1][0],4)} | {f(A[2][0],4)} |")
    L += ["", "Раннее предупреждение по вариантам (ROC-AUC):", "",
          "| вариант | " + " | ".join(f"нед. {w}" for w in [2, 4, 6, 8, 10, 12, 15, 20]) + " |", "|---" * 9 + "|"]
    for name, v in R["feature_availability"].items():
        bw = v["early_warning"]["by_week"]
        L.append(f"| {name} | " + " | ".join(f(bw.get(str(w), {}).get("roc_auc")) for w in [2, 4, 6, 8, 10, 12, 15, 20]) + " |")
    o = R["student_overlap"]
    L += ["", f"Пересечение id_student: {o['also_in_train_presentations']} из {o['test_students']} студентов 2014J "
          f"встречаются в обучающих потоках ({o['share']:.1%}).",
          f"Тест без пересекающихся студентов: ROC-AUC {f(R['test_without_overlap']['roc_auc'])}, "
          f"PR-AUC {f(R['test_without_overlap']['pr_auc'])}, n = {R['test_without_overlap']['n']}.", ""]
    t13 = R["train_2013_only"]
    g = t13["drop"]["results"]["combined"]["gbm"]
    L += [f"Обучение только на {', '.join(t13['train_presentations'])}: ROC-AUC {f(g['roc_auc'])}, PR-AUC {f(g['pr_auc'])}.",
          "Раннее предупреждение: " + ", ".join(f"нед. {w}: {f(r['roc_auc'])}" for w, r in t13["early_warning"]["by_week"].items()), ""]

    L += ["## 3. Прогноз спада: компоненты цели, модель без лагов, 95% ДИ (бутстрап по студентам)", ""]
    d = R["drop_base"]
    L += [f"Строк: {d['rows_total']}; доля положительных: панель {f(d['pos_rate_total'],4)}, тест {f(d['pos_rate_test'],4)}; "
          f"доля отчислений среди положительных в тесте: {f(d['withdraw_share_of_pos_test'])}", "",
          "| цель | модель | ROC-AUC | PR-AUC | доля полож. | n | ДИ ROC | ДИ PR | ДИ разницы с Марковской ROC | ДИ разницы PR |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tgt, res in d["results"].items():
        for mname in ["markov", "markov_S0_as_score", "gbm", "gbm_nolag"]:
            if mname not in res: continue
            m = res[mname]
            c = res.get("gbm_ci", {}) if mname == "gbm" else res.get("markov_ci", {}) if mname == "markov" else \
                res.get("gbm_nolag_vs_markov_ci", {}) if mname == "gbm_nolag" else {}
            L.append(f"| {tgt} | {mname} | {f(m['roc_auc'])} | {f(m['pr_auc'])} | {f(m['pos_rate'],4)} | {m['n']} | "
                     f"{ci(c.get('roc'))} | {ci(c.get('pr'))} | {ci(c.get('d_roc'))} | {ci(c.get('d_pr'))} |")
    r = R["ratio_label"]
    L += ["", f"Пороговая метка: доля полож. панель {f(r['pos_rate_panel'],4)}, тест {f(r['pos_rate_test'],4)}; "
          f"GBM PR-AUC {f(r['gbm_pr_auc'])}; lift к тесту {f(r['lift_vs_test'],2)}×, к панели {f(r['lift_vs_panel'],2)}×", ""]

    L += ["## 4. Диагностика СММ", "", "Матрица переходов (6 знаков):", ""]
    L += ["| из \\ в | S0 | S1 | S2 | S3 |", "|---|---|---|---|---|"]
    for i, row in enumerate(R["hmm4_transmat_6dp"]):
        L.append(f"| S{i} | " + " | ".join(f"{x:.6f}" for x in row) + " |")
    L += ["", "Эмпирические переходы между состояниями прямого прохода (число / частота):", "",
          "| из \\ в | S0 | S1 | S2 | S3 |", "|---|---|---|---|---|"]
    for i, (rc, rf) in enumerate(zip(R["empirical_counts"], R["empirical_freq"])):
        L.append(f"| S{i} | " + " | ".join(f"{c} / {x:.4f}" for c, x in zip(rc, rf)) + " |")
    e = R["entries_into_S0"]
    L += ["", f"Входов в S0: {e['entries']}; по исходным состояниям: {e['from_state']}; доля из S1: {f(e['share_from_S1'],4)}", ""]
    t = R["time_S1_to_exit"]
    L += [f"От первого S1 до S0 или отчисления: студентов с S1 {t['students_with_S1']}, доля достигших {f(t['share_reaching_S0_or_withdraw'])}, "
          f"медиана {f(t['weeks_median'],1)} нед. (IQR {f(t['weeks_q25'],1)}–{f(t['weeks_q75'],1)}), ≤2 нед.: {f(t['share_le_2_weeks'])}, ≥4 нед.: {f(t['share_ge_4_weeks'])}", "",
          "Устойчивость к инициализации (4 состояния):", "", "| seed | A[1,0] | A[2,0] | A[3,0] | A[0,1] |", "|---|---|---|---|---|"]
    for s in R["hmm4_seeds"]:
        L.append(f"| {s['seed']} | {s['A10']:.6f} | {s['A20']:.2e} | {s['A30']:.2e} | {s['A01']:.6f} |")
    L += ["", "По отдельным курсам (4 состояния):", "", "| курс | студентов | A[1,0] | A[2,0] | A[3,0] |", "|---|---|---|---|---|"]
    for m in R["hmm4_by_module"]:
        if "error" in m: L.append(f"| {m['module']} | ошибка: {m['error']} | | | |"); continue
        L.append(f"| {m['module']} | {m['n_students']} | {m['A10']:.4f} | {m['A20']:.2e} | {m['A30']:.2e} |")
    L += ["", "5 состояний при разных инициализациях (ближайшая пара состояний):", "",
          "| seed | пара | макс. разница средних | A_ii | A_jj | A_ij | A_ji | BIC |", "|---|---|---|---|---|---|---|---|"]
    for s in R["hmm5_seeds"]:
        L.append(f"| {s['seed']} | {s['closest_pair']} | {f(s['max_abs_mean_diff'])} | {f(s['A_ii'])} | {f(s['A_jj'])} | {f(s['A_ij'])} | {f(s['A_ji'])} | {s['bic']:.0f} |")
    L += ["", "4 и 5 состояний на общей цели (отчисление в ближайшие 2 недели):", ""]
    for k, m in R["common_target_4_vs_5"].items():
        L.append(f"- {k}: ROC-AUC {f(m['roc_auc'])}, PR-AUC {f(m['pr_auc'])}, n = {m['n']}")

    c = R["cluster_centers"]
    L += ["", "## 5. Кластеры", "", f"E неделя 0: {c['week0']}; среднее недель 0–3: {c['weeks0_3_mean']}; "
          f"неделя 19: {c['week19']}; среднее недель 16–19: {c['weeks16_19_mean']}", "",
          "Кластеризация префиксов (только не выбывшие к неделе w; обучение на 2013B–2014B, тест 2014J):", ""]
    for w, p in R["prefix_clusters"].items():
        prof = "; ".join(f"кл.{x['cluster']}: n={x['n']}, риск {x['risk_share']:.2f}" for x in p["profile_test"])
        L.append(f"- w = {w}: {prof}; ROC-AUC порядка кластера {f(p['roc_auc_cluster_rank'])}")
    L += ["", "## 6. Раннее предупреждение по всем неделям", "", "| неделя | ROC-AUC | PR-AUC | n (тест) | доля риска |", "|---|---|---|---|---|"]
    for w, r in R["early_warning"]["by_week"].items():
        L.append(f"| {w} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | {r['n_test']} | {f(r['pos_rate_test'])} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="outputs/revision")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed, pc = cfg["random_state"], cfg["prediction"]
    raw = cfg["data"]["raw_dir"]
    if args.synthetic:
        from scripts.make_synthetic_oulad import generate
        raw = "data/raw/synthetic"; generate(Path(raw))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    test_pres = pc["test_presentations"]
    hc = cfg["hmm"]; n_iter, max_seq = hc["n_iter"], hc["max_sequences"]
    R = {}

    d = load_oulad(raw); info = d["student_info"]
    trmask = lambda df: ~df["code_presentation"].isin(test_pres)
    log("данные загружены")

    # ---------- базовый вариант
    idx = build_variant(d, cfg, 0, False, trmask)
    hmm4 = fit_hmm(idx, [4], max_seq, n_iter, seed, trmask(idx))
    idx = idx.join(hmm4["filtered"]); idx["hmm_state"] = hmm4["states"]
    log("базовый индекс и СММ-4")

    # П.1 валидность: B, баллы, E, E без баллов; все / тест; все / не выбывшие
    t10 = first_weeks_table(idx, info)
    rt = rho_table(t10, test_pres); rt.to_csv(out / "validity_comparison.csv", index=False)
    R["validity"] = rt.to_dict("records")
    R["validity_note"] = {"withdrawn_before_w10": int((~t10["survivor"]).sum()), "n_students": int(len(t10))}

    # П.3 прогноз спада: компоненты цели, без лагов, ДИ
    R["drop_base"], ds_base, split = drop_eval(idx, hmm4, pc, seed, test_pres, args.n_boot)
    log("прогноз спада (базовый)")

    # частное замечание: знаменатели lift для пороговой метки
    dr = make_drop_dataset(idx, pc["horizon"], pc["history"], pc["drop_ratio"], pc["min_history_engagement"], "ratio")
    trr, var_, ter = split_by_presentation(dr, test_pres, pc["val_fraction"], seed)
    rr = fit_tabular_models(dr, trr, var_, ter, seed)["gbm"]["metrics"]
    R["ratio_label"] = {"pos_rate_panel": float(dr["y"].mean()), "pos_rate_test": float(dr.loc[ter, "y"].mean()),
                        "gbm_pr_auc": rr["pr_auc"], "gbm_roc_auc": rr["roc_auc"],
                        "lift_vs_test": rr["pr_auc"] / float(dr.loc[ter, "y"].mean()),
                        "lift_vs_panel": rr["pr_auc"] / float(dr["y"].mean())}

    # П.5 раннее предупреждение: все недели 1–20
    R["early_warning"] = early_warning_all(idx, info, list(range(0, 21)), test_pres, seed)
    log("раннее предупреждение по всем неделям")

    # П.4 диагностика переходов
    R["hmm4_transmat_6dp"] = np.round(hmm4["transmat"].to_numpy(), 6).tolist()
    R["hmm4_means"] = np.round(hmm4["means"].to_numpy(), 4).tolist()
    emp = empirical_transitions(idx); emp.to_csv(out / "empirical_transitions_counts.csv")
    R["empirical_counts"] = emp.values.tolist()
    R["empirical_freq"] = np.round(emp.div(emp.sum(axis=1), axis=0).values, 5).tolist()
    R["entries_into_S0"] = entries_into_s0(idx)
    R["time_S1_to_exit"] = time_s1_to_exit(idx)
    # устойчивость: разные инициализации
    st = []
    for s in range(5):
        h = fit_hmm(idx.drop(columns=[c for c in idx.columns if c.startswith("hmm_")]), [4], max_seq, n_iter, s, trmask(idx))
        A = h["transmat"].to_numpy()
        st.append({"seed": s, "A10": A[1, 0], "A20": A[2, 0], "A30": A[3, 0], "A01": A[0, 1],
                   "means": np.round(h["means"].to_numpy(), 3).tolist()})
    R["hmm4_seeds"] = st
    log("устойчивость СММ-4 к инициализации")
    # по отдельным курсам
    pm = []
    for mod, g in idx.groupby("code_module"):
        g = g.drop(columns=[c for c in g.columns if c.startswith("hmm_")]).reset_index(drop=True)
        tm = trmask(g)
        if tm.sum() < 200: continue
        try:
            h = fit_hmm(g, [4], min(max_seq, 2000), n_iter, seed, tm)
        except Exception as e:  # noqa: BLE001
            pm.append({"module": mod, "error": str(e)}); continue
        A = h["transmat"].to_numpy()
        pm.append({"module": mod, "n_students": int(g[KEY].drop_duplicates().shape[0]),
                   "A10": A[1, 0], "A20": A[2, 0], "A30": A[3, 0]})
    R["hmm4_by_module"] = pm
    log("СММ-4 по отдельным курсам")
    # 5 состояний: вырожденность при разных инициализациях
    s5 = []
    base_cols = [c for c in idx.columns if not c.startswith("hmm_")]
    for s in range(3):
        h = fit_hmm(idx[base_cols], [5], max_seq, n_iter, s, trmask(idx))
        M, A = h["means"].to_numpy(), h["transmat"].to_numpy()
        dmin, pair = 9, None
        for i in range(5):
            for j in range(i + 1, 5):
                dd = float(np.abs(M[i] - M[j]).max())
                if dd < dmin: dmin, pair = dd, (i, j)
        i, j = pair
        s5.append({"seed": s, "closest_pair": pair, "max_abs_mean_diff": dmin,
                   "A_ii": A[i, i], "A_jj": A[j, j], "A_ij": A[i, j], "A_ji": A[j, i], "bic": float(h["selection"]["bic"].iloc[0])})
        if s == 0:
            hmm5 = h
    R["hmm5_seeds"] = s5
    log("СММ-5 при разных инициализациях")
    # 4 vs 5 состояний на общей цели: отчисление в ближайшие 2 недели
    idx5 = idx[base_cols].join(hmm5["filtered"])
    common = {}
    for name, ii in [("4 состояния", idx), ("5 состояний", idx5)]:
        dsx = make_drop_dataset(ii, pc["horizon"], pc["history"], pc["drop_ratio"], 0.0, "ratio")
        dsx = dsx[dsx["F"].notna() | (dsx["y_withdraw"] == 1)]
        trx, vax, tex = split_by_presentation(dsx, test_pres, pc["val_fraction"], seed)
        X, y = dsx[tabular_features(dsx)].to_numpy(), dsx["y_withdraw"].to_numpy()
        p = gbm(seed).fit(X[trx], y[trx]).predict_proba(X[tex])[:, 1]
        common[name] = metrics(y[tex], p)
    R["common_target_4_vs_5"] = common
    log("сравнение 4/5 состояний на общей цели")

    # П.5 кластеры: центры (нед. 0 и 19) и кластеризация префиксов до выбытия
    mat = trajectory_matrix(idx, cfg["clustering"]["weeks"])
    km = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(mat.to_numpy())
    order = np.argsort(km.cluster_centers_.mean(axis=1))
    C = km.cluster_centers_[order]
    R["cluster_centers"] = {"week0": np.round(C[:, 0], 3).tolist(), "weeks0_3_mean": np.round(C[:, :4].mean(1), 3).tolist(),
                            "week19": np.round(C[:, -1], 3).tolist(), "weeks16_19_mean": np.round(C[:, -4:].mean(1), 3).tolist()}
    pref = {}
    infr = info[KEY + ["final_result"]].copy(); infr["risk"] = infr["final_result"].isin(["Withdrawn", "Fail"])
    for w in (4, 6, 8):
        sub = idx[idx["week"] < w]
        alive = sub.groupby(KEY, observed=True)["withdraw_week"].first()
        alive = alive[~(alive < w)].index
        m = sub.set_index(KEY).loc[alive].reset_index().pivot_table(index=KEY, columns="week", values="E", observed=True)
        m = m.dropna()
        keys = m.index.to_frame(index=False)
        is_test = keys["code_presentation"].isin(test_pres).to_numpy()
        kmw = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(m.to_numpy()[~is_test])
        o = np.argsort(kmw.cluster_centers_.mean(axis=1)); rmap = {old: new for new, old in enumerate(o)}
        lab = np.array([rmap[l] for l in kmw.predict(m.to_numpy())])
        kk = keys.assign(cluster=lab).merge(infr, on=KEY)
        tt = kk[kk["code_presentation"].isin(test_pres)]
        prof = tt.groupby("cluster").agg(n=("risk", "size"), risk_share=("risk", "mean")).reset_index()
        auc = roc_auc_score(tt["risk"], -tt["cluster"]) if tt["risk"].nunique() > 1 else np.nan
        pref[w] = {"profile_test": prof.to_dict("records"), "roc_auc_cluster_rank": float(auc),
                   "n_test": int(len(tt))}
    R["prefix_clusters"] = pref
    log("кластеры префиксов")

    # П.2 доступность признаков: задержка баллов и расчёт без баллов
    variants = {}
    for name, delay, nos in [("баллы с задержкой 2 нед.", 2, False), ("баллы с задержкой 4 нед.", 4, False),
                             ("без баллов", 0, True)]:
        iv = build_variant(d, cfg, delay, nos, trmask)
        hv = fit_hmm(iv, [4], max_seq, n_iter, seed, trmask(iv))
        iv = iv.join(hv["filtered"])
        tv = first_weeks_table(iv, info)
        tst = tv["code_presentation"].isin(test_pres)
        rho_all = stats.spearmanr(tv["E"], tv["outcome"])[0]
        rho_test = stats.spearmanr(tv.loc[tst, "E"], tv.loc[tst, "outcome"])[0]
        dv, _, _ = drop_eval(iv, hv, pc, seed, test_pres, 0, full=False)
        ew = early_warning_all(iv, info, [2, 4, 6, 8, 10, 12, 15, 20], test_pres, seed)
        variants[name] = {"rho_all": float(rho_all), "rho_test": float(rho_test),
                          "transmat": np.round(hv["transmat"].to_numpy(), 4).tolist(),
                          "means": np.round(hv["means"].to_numpy(), 3).tolist(),
                          "drop": dv, "early_warning": ew}
        log(f"вариант: {name}")
    R["feature_availability"] = variants

    # П.2 протокол: пересечение студентов и обучение только на 2013B/2013J
    tr_ids = set(info.loc[~info["code_presentation"].isin(test_pres), "id_student"])
    te_ids = set(info.loc[info["code_presentation"].isin(test_pres), "id_student"])
    R["student_overlap"] = {"test_students": len(te_ids), "also_in_train_presentations": len(te_ids & tr_ids),
                            "share": len(te_ids & tr_ids) / max(len(te_ids), 1)}
    # тест без пересекающихся студентов
    ds_no = ds_base.copy()
    tr, va, te = split
    keep_te = te & ~ds_no["id_student"].isin(tr_ids).to_numpy()
    X = ds_no[tabular_features(ds_no)].to_numpy(); y = ds_no["y"].to_numpy()
    p = gbm(seed).fit(X[tr], y[tr]).predict_proba(X[keep_te])[:, 1]
    R["test_without_overlap"] = metrics(y[keep_te], p)
    # обучение только на потоках 2013 (их исходы заведомо известны до начала 2014J)
    early_train = [p_ for p_ in sorted(idx["code_presentation"].unique()) if p_.startswith("2013")]
    de, _, _ = drop_eval(idx, hmm4, pc, seed, test_pres, 0, train_pres=early_train, full=False)
    R["train_2013_only"] = {"train_presentations": early_train, "drop": de,
                            "early_warning": early_warning_all(idx, info, [2, 4, 6, 8, 10, 12, 15, 20], test_pres,
                                                               seed, train_pres=early_train)}
    log("протокол: пересечение и обучение на 2013")

    (out / "revision.json").write_text(json.dumps(jsonable(R), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report_revision.md").write_text(make_report(jsonable(R)), encoding="utf-8")
    log(f"готово: {out / 'report_revision.md'}")


if __name__ == "__main__":
    main()
