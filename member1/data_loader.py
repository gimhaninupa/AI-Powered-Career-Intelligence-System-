from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import JOB_CSV, RESUME_CSV

JOB_TEXT_COLUMNS = [
    "Job Title",
    "Job Description",
    "Key Responsibilities",
    "Required Skills & Qualifications",
]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}\n"
            "Place resume_data.csv and job_description.csv inside data/raw/."
        )


def load_resume_data(path: Path = RESUME_CSV) -> pd.DataFrame:
    _require_file(path)
    df = pd.read_csv(path, low_memory=False)
    # Remove a possible UTF-8 byte-order mark from column names.
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    return df


def load_job_data(path: Path = JOB_CSV) -> pd.DataFrame:
    _require_file(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    missing = [column for column in ["Job Field", *JOB_TEXT_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError(f"Job dataset is missing required columns: {missing}")
    return df


def combine_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot combine missing columns: {missing}")
    return (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_skill_list(value: object) -> list[str]:
    """Safely parse the resume dataset's string representation of a skill list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]
