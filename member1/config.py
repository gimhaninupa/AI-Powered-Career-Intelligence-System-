from pathlib import Path

<<<<<<< HEAD
PROJECT_ROOT = Path(__file__).resolve().parents[2]
=======
candidate_root = Path(__file__).resolve().parents[1]
if (candidate_root / "README.md").exists() or (candidate_root / "requirements_baseline.txt").exists():
    PROJECT_ROOT = candidate_root
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
>>>>>>> 8dad167 (feature(data): add cleaning, mapping, and evaluation scripts; add training report)
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
