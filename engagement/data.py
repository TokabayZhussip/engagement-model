"""Загрузка таблиц OULAD с экономными типами данных."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
COURSE = ["code_module", "code_presentation"]

_FILES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "vle": "vle.csv",
    "student_info": "studentInfo.csv",
    "student_registration": "studentRegistration.csv",
    "student_assessment": "studentAssessment.csv",
    "student_vle": "studentVle.csv",
}

_DTYPES = {
    "student_vle": {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int32",
    },
}


def load_oulad(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Читает все 7 таблиц OULAD. В исходных файлах пропуски обозначены '?'."""
    raw_dir = Path(raw_dir)
    missing = [f for f in _FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"В {raw_dir} не найдены файлы: {missing}. "
            "Скачайте OULAD (см. data/README.md) или запустите с флагом --synthetic."
        )

    d: dict[str, pd.DataFrame] = {}
    for name, fname in _FILES.items():
        d[name] = pd.read_csv(
            raw_dir / fname,
            na_values=["?", ""],
            dtype=_DTYPES.get(name),
        )

    # единые категориальные типы для ключей курса
    for name, df in d.items():
        for col in COURSE:
            if col in df.columns:
                df[col] = df[col].astype(str)
    return d


def course_weeks(courses: pd.DataFrame, max_week: int) -> pd.DataFrame:
    """Число учебных недель в каждом потоке курса (не больше max_week)."""
    out = courses[COURSE].copy()
    out["n_weeks"] = (
        (courses["module_presentation_length"] + 6) // 7
    ).clip(upper=max_week).astype(int)
    return out
