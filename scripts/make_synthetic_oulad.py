"""Генератор маленького синтетического набора в формате OULAD.

Используется только для тестов и CI: проверяет, что весь пайплайн работает
без скачивания настоящего датасета. В статью эти результаты не идут.

Скрытая модель: у каждого студента один из трёх профилей вовлечённости
(стабильно высокий, снижающийся, низкий); клики, сдача заданий, баллы
и отчисление порождаются из этой скрытой траектории.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITY_TYPES = ["homepage", "oucontent", "forumng", "quiz", "resource", "subpage", "url", "ouwiki"]


def generate(out_dir: Path, n_students: int = 120, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = ["AAA", "BBB"]
    presentations = ["2013J", "2014B", "2014J"]
    length_days = 210  # 30 недель

    courses, assessments, vle = [], [], []
    info, reg, s_ass, s_vle = [], [], [], []
    id_site = 1000
    id_ass = 1
    id_student = 100000

    for m in modules:
        for p in presentations:
            courses.append((m, p, length_days))
            sites = []
            for t in ACTIVITY_TYPES:
                for _ in range(3):
                    vle.append((id_site, m, p, t, np.nan, np.nan))
                    sites.append((id_site, t))
                    id_site += 1
            ass_list = []
            for k, day in enumerate(range(25, length_days - 20, 28)):
                ass_list.append((id_ass, day))
                assessments.append((m, p, id_ass, "TMA" if k % 2 == 0 else "CMA", day, 10.0))
                id_ass += 1
            assessments.append((m, p, id_ass, "Exam", np.nan, 100.0))
            id_ass += 1

            for _ in range(n_students):
                profile = rng.choice(["high", "declining", "low"], p=[0.4, 0.35, 0.25])
                base = {"high": 0.8, "declining": 0.75, "low": 0.25}[profile]
                slope = {"high": -0.002, "declining": -0.03, "low": -0.005}[profile]
                weeks = length_days // 7
                level = np.clip(base + slope * np.arange(weeks) + rng.normal(0, 0.07, weeks), 0, 1)

                unreg = np.nan
                if profile == "declining" and rng.random() < 0.6:
                    w = int(rng.integers(8, weeks - 4))
                    unreg = float(w * 7 + int(rng.integers(0, 7)))
                    level[w + 1:] = 0
                elif profile == "low" and rng.random() < 0.3:
                    w = int(rng.integers(4, weeks - 4))
                    unreg = float(w * 7 + int(rng.integers(0, 7)))
                    level[w + 1:] = 0

                for day in range(length_days):
                    if not np.isnan(unreg) and day > unreg:
                        break
                    lv = level[day // 7]
                    if rng.random() < lv * 0.7:
                        for _ in range(rng.poisson(1 + 4 * lv)):
                            s, _t = sites[int(rng.integers(len(sites)))]
                            s_vle.append((m, p, id_student, s, day, int(rng.poisson(1 + 5 * lv)) + 1))

                for a, day in ass_list:
                    if not np.isnan(unreg) and day > unreg:
                        break
                    if rng.random() < 0.3 + 0.7 * level[day // 7]:
                        late = int(rng.normal(-2 + 6 * (1 - level[day // 7]), 2))
                        score = float(np.clip(rng.normal(40 + 55 * level[day // 7], 10), 0, 100))
                        s_ass.append((a, id_student, day + late, 0, round(score)))

                mean_lv = level.mean()
                if not np.isnan(unreg):
                    result = "Withdrawn"
                elif mean_lv > 0.65:
                    result = "Distinction" if rng.random() < 0.4 else "Pass"
                elif mean_lv > 0.4:
                    result = "Pass" if rng.random() < 0.7 else "Fail"
                else:
                    result = "Fail" if rng.random() < 0.7 else "Pass"

                info.append((m, p, id_student, rng.choice(["M", "F"]), "East Anglian Region",
                             "A Level or Equivalent", "50-60%", "0-35", 0, 60, "N", result))
                reg.append((m, p, id_student, float(-int(rng.integers(10, 60))), unreg))
                id_student += 1

    def save(rows, cols, name):
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / name, index=False, na_rep="")

    save(courses, ["code_module", "code_presentation", "module_presentation_length"], "courses.csv")
    save(assessments, ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"], "assessments.csv")
    save(vle, ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"], "vle.csv")
    save(info, ["code_module", "code_presentation", "id_student", "gender", "region", "highest_education",
                "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "disability", "final_result"],
         "studentInfo.csv")
    save(reg, ["code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"],
         "studentRegistration.csv")
    save(s_ass, ["id_assessment", "id_student", "date_submitted", "is_banked", "score"], "studentAssessment.csv")
    save(s_vle, ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"], "studentVle.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--n-students", type=int, default=120, help="студентов в одном потоке")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(Path(args.out), args.n_students, args.seed)
    print(f"Синтетический набор OULAD записан в {args.out}")


if __name__ == "__main__":
    main()
