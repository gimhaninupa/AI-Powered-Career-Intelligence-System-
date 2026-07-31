from __future__ import annotations

import json

import pandas as pd

from .config import MAPPING_CSV, REPORT_DIR
from .data_loader import load_job_data, load_resume_data


def summarize(df: pd.DataFrame, name: str) -> dict[str, object]:
    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {column: int(value) for column, value in df.isna().sum().items()},
        "column_names": list(df.columns),
    }


def main() -> None:
    resume_df = load_resume_data()
    job_df = load_job_data()

    report = {
        "resume_dataset": summarize(resume_df, "resume_data.csv"),
        "job_dataset": summarize(job_df, "job_description.csv"),
        "job_field_count": int(job_df["Job Field"].nunique()),
        "job_field_distribution": {
            str(label): int(count) for label, count in job_df["Job Field"].value_counts().items()
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "dataset_inspection.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mapping = (
        job_df["Job Field"]
        .value_counts()
        .rename_axis("original_job_field")
        .reset_index(name="record_count")
    )
    mapping["broad_category"] = ""
    mapping.to_csv(MAPPING_CSV, index=False)

    print("\n=== RESUME DATASET ===")
    print(f"Shape: {resume_df.shape}")
    print(f"Duplicate rows: {resume_df.duplicated().sum()}")
    print("Important columns found:", [c for c in ["career_objective", "skills", "responsibilities"] if c in resume_df])
    print("\nSample career objective:")
    print(resume_df["career_objective"].dropna().iloc[0])

    print("\n=== JOB DESCRIPTION DATASET ===")
    print(f"Shape: {job_df.shape}")
    print(f"Duplicate rows: {job_df.duplicated().sum()}")
    print(f"Distinct Job Field labels: {job_df['Job Field'].nunique()}")
    print("\nTop 15 job fields:")
    print(job_df["Job Field"].value_counts().head(15).to_string())

    print(f"\nSaved detailed inspection report to: {output_path}")
    print(f"Saved editable broad-category mapping template to: {MAPPING_CSV}")
    print("Fill the broad_category column only after your group agrees on a justified mapping.")


if __name__ == "__main__":
    main()
