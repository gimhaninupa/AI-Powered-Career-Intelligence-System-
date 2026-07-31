from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "member1"
REPORT_DIR = PROJECT_ROOT / "reports" / "member1"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots" / "member1"

RESUME_CSV = RAW_DATA_DIR / "resume_data.csv"
JOB_CSV = RAW_DATA_DIR / "job_description.csv"
MAPPING_CSV = PROCESSED_DATA_DIR / "job_field_mapping.csv"

for directory in (PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR, SCREENSHOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
