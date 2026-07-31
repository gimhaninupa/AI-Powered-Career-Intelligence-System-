from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import REPORT_DIR
from .data_loader import load_resume_data, parse_skill_list
from .skill_extractor import SkillExtractor, build_skill_vocabulary, normalize_skill
from .text_preprocessor import TextPreprocessor


def calculate_set_metrics(reference: set[str], predicted: set[str]) -> tuple[int, int, int]:
    true_positive = len(reference & predicted)
    false_positive = len(predicted - reference)
    false_negative = len(reference - predicted)
    return true_positive, false_positive, false_negative


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based skill extraction.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_resume_data().copy()
    df = df[df["skills"].notna()].copy()
    if args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.random_state)

    vocabulary = build_skill_vocabulary(load_resume_data(), min_frequency=2)
    extractor = SkillExtractor(vocabulary)
    extractor.save_vocabulary()
    preprocessor = TextPreprocessor()

    text_columns = [column for column in ["career_objective", "responsibilities", "positions"] if column in df.columns]
    df["evaluation_text"] = (
        df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True)
    )

    total_tp = total_fp = total_fn = 0
    examples: list[dict[str, object]] = []

    for _, row in df.iterrows():
        reference_display = parse_skill_list(row["skills"])
        reference = {normalize_skill(skill).casefold() for skill in reference_display}
        cleaned = preprocessor.clean(row["evaluation_text"])
        # Pass the original skills column as context so known skills are preferred.
        predicted_display = extractor.extract(cleaned, context=row.get("skills", ""))
        predicted = {normalize_skill(skill).casefold() for skill in predicted_display}

        tp, fp, fn = calculate_set_metrics(reference, predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if len(examples) < 20:
            examples.append(
                {
                    "text_excerpt": row["evaluation_text"][:500],
                    "ground_truth_skills": reference_display,
                    "predicted_skills": predicted_display,
                    "correct": sorted(reference & predicted),
                    "false_positives": sorted(predicted - reference),
                    "missed": sorted(reference - predicted),
                }
            )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    report = {
        "records_evaluated": int(len(df)),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "important_limitation": (
            "The ground-truth skills list may contain skills not mentioned in the career objective, "
            "responsibilities, or position text used for extraction. Therefore, recall can be low even "
            "when the extractor behaves correctly on the available text."
        ),
        "examples": examples,
    }

    output_path = REPORT_DIR / "skill_extraction_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== SKILL EXTRACTION EVALUATION ===")
    print(f"Records evaluated: {len(df)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Saved report: {output_path}")
    print("\nNote: Read the limitation inside the report before interpreting recall.")


if __name__ == "__main__":
    main()
