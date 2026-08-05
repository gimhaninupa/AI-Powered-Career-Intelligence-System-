from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
IN = PROC / "job_description_cleaned.csv"
OUT = PROC / "job_field_mapping.csv"

# Broad categories and keyword cues
MAPPING_RULES = {
    "Arts, Media, Education & Public Service": [
        "art", "media", "education", "teacher", "museum", "creative", "perform", "arts", "public service", "library"
    ],
    "Banking, Finance & Audit": [
        "bank", "finance", "financial", "investment", "wealth", "portfolio", "audit", "account", "treasury"
    ],
    "Consulting, Business & Operations": [
        "consult", "business", "operations", "strategy", "management", "consulting", "analyst", "advisor"
    ],
    "Engineering, Manufacturing & Technology": [
        "engineer", "engineering", "manufactur", "software", "developer", "technical", "technology", "mechanic", "industrial", "systems", "computer", "it", "devops"
    ],
    "Healthcare & Life Sciences": [
        "health", "nurs", "medical", "physician", "clinic", "pharm", "biomedical", "therap", "care"
    ],
}


def choose_category(field: str) -> str:
    key = field.lower()
    for cat, cues in MAPPING_RULES.items():
        for cue in cues:
            if cue in key:
                return cat
    return ""  # empty means no mapping


def main():
    if not IN.exists():
        print(f"Input file not found: {IN}")
        return

    fields = []
    with IN.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            jf = row.get("Job Field") or row.get("Job field") or ""
            jf = jf.strip()
            if jf and jf not in fields:
                fields.append(jf)

    mapped = []
    for f in sorted(fields):
        cat = choose_category(f)
        mapped.append({"original_job_field": f, "broad_category": cat})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_job_field", "broad_category"])
        writer.writeheader()
        for row in mapped:
            writer.writerow(row)

    print(f"Wrote mapping to: {OUT}")


if __name__ == "__main__":
    main()
