from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .config import MODEL_DIR
from .skill_extractor import SkillExtractor
from .text_preprocessor import TextPreprocessor


def run_text_demo(text: str, extractor: SkillExtractor, preprocessor: TextPreprocessor, pipeline) -> None:
    cleaned = preprocessor.clean(text)
    tokens = preprocessor.tokenize(cleaned)
    skills = extractor.extract(cleaned)
    predicted = pipeline.predict([cleaned])[0] if pipeline is not None else "(no model)"

    print("\n--- Demo Result ---")
    print("Original:", text)
    print("Cleaned:", cleaned)
    print("Tokens:", tokens)
    print("Extracted skills:", skills)
    print("Predicted category:", predicted)


def run_csv_demo(path: Path, extractor: SkillExtractor, preprocessor: TextPreprocessor, pipeline, max_rows: int | None, out_path: Path | None = None) -> None:
    rows_out = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            if max_rows and i > max_rows:
                break
            text = row.get("evaluation_text") or row.get("text") or row.get("description") or ""
            context = row.get("skills", "")
            cleaned = preprocessor.clean(text)
            skills = extractor.extract(cleaned, context=context)
            predicted = pipeline.predict([cleaned])[0] if pipeline is not None else "(no model)"
            print(f"\n[{i}] Original: {text[:120]}")
            print("  Extracted skills:", skills)
            print("  Predicted category:", predicted)
            rows_out.append({
                "index": i,
                "original": text,
                "cleaned": cleaned,
                "extracted_skills": "|".join(skills),
                "predicted_category": predicted,
            })

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["index", "original", "cleaned", "extracted_skills", "predicted_category"]
        with out_path.open("w", encoding="utf-8", newline="") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_out:
                writer.writerow(r)
        print(f"\nSaved demo output to: {out_path}")


def repl(extractor: SkillExtractor, preprocessor: TextPreprocessor, pipeline) -> None:
    print("Interactive demo. Paste text and press Enter (empty to quit).")
    try:
        while True:
            text = input("Text> ")
            if not text.strip():
                break
            run_text_demo(text, extractor, preprocessor, pipeline)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting interactive demo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive demo for Member1 extractor + classifier")
    parser.add_argument("--text", help="Single text to demo")
    parser.add_argument("--csv", help="Path to CSV with an 'evaluation_text' or 'text' column")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL")
    parser.add_argument("--max-rows", type=int, default=20, help="Max rows to show for CSV demo")
    parser.add_argument("--out", help="Optional CSV path to save demo outputs")
    args = parser.parse_args()

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    pipeline = None
    try:
        import joblib

        if model_path.exists():
            saved = joblib.load(model_path)
            pipeline = saved.get("pipeline") if isinstance(saved, dict) else saved
    except Exception:
        pipeline = None

    preprocessor = TextPreprocessor()
    extractor = SkillExtractor.from_vocabulary_file()

    if args.text:
        run_text_demo(args.text, extractor, preprocessor, pipeline)
        return
    if args.csv:
        run_csv_demo(Path(args.csv), extractor, preprocessor, pipeline, args.max_rows, out_path=Path(args.out) if args.out else None)
        return
    if args.interactive:
        repl(extractor, preprocessor, pipeline)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
