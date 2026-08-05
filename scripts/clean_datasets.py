import ast
import html
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_text(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    # Unescape HTML entities
    s = html.unescape(s)
    # Remove simple HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Remove weird control characters, keep punctuation
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_possible_list(value: object) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    # Try literal eval if it looks like a list
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Fallback: split on common separators
    parts = re.split(r"[,;\|\\/]+", s)
    return [p.strip() for p in parts if p.strip()]


def clean_job_csv(path: Path) -> Path:
    df = pd.read_csv(path, low_memory=False)
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # Clean text columns
    text_cols = [c for c in df.columns if "description" in c.lower() or "respons" in c.lower() or "skill" in c.lower() or "title" in c.lower()]
    for c in text_cols:
        df[c] = df[c].map(clean_text)
    # Fill NaNs
    df = df.fillna("")
    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"job_description: {before} rows -> {after} rows after dropping exact duplicates")
    out = PROCESSED / "job_description_cleaned.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned job CSV to: {out}")
    return out


def clean_resume_csv(path: Path) -> Path:
    df = pd.read_csv(path, low_memory=False)
    # Fix BOM in column names
    df.columns = [str(c).lstrip('\ufeff').strip() for c in df.columns]
    # Lower-case columns for consistency
    df.columns = [c for c in df.columns]
    # Clean textual columns
    text_cols = [c for c in df.columns if c not in ["skills", "related_skils_in_job"]]
    for c in text_cols:
        try:
            df[c] = df[c].map(clean_text)
        except Exception:
            pass
    # Parse skills fields
    if "skills" in df.columns:
        df["parsed_skills"] = df["skills"].map(parse_possible_list)
    if "related_skils_in_job" in df.columns:
        df["parsed_related_skills"] = df["related_skils_in_job"].map(parse_possible_list)

    # Combine key text features into a single cleaned text for downstream use
    combine_cols = [c for c in ["career_objective", "positions", "responsibilities", "skills"] if c in df.columns]
    if combine_cols:
        df["combined_text"] = (
            df[combine_cols].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
        )
    df = df.fillna("")
    # Convert list/dict-like cells to JSON/string so drop_duplicates can hash rows
    def _stringify_unhashable(val):
        if isinstance(val, (list, dict)):
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)
        return val

    # Some pandas installations may not expose applymap; apply per-column for compatibility
    for col in df.columns:
        try:
            df[col] = df[col].apply(_stringify_unhashable)
        except Exception:
            df[col] = df[col].map(lambda v: _stringify_unhashable(v))
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"resume_data: {before} rows -> {after} rows after dropping exact duplicates")
    out = PROCESSED / "resume_data_cleaned.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned resume CSV to: {out}")
    return out


def main():
    print(f"RAW dir: {RAW}")
    job_path = RAW / "job_description.csv"
    resume_path = RAW / "resume_data.csv"
    if not job_path.exists():
        print(f"Missing {job_path}")
    else:
        clean_job_csv(job_path)
    if not resume_path.exists():
        print(f"Missing {resume_path}")
    else:
        clean_resume_csv(resume_path)


if __name__ == "__main__":
    main()
