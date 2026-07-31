from __future__ import annotations

import argparse

import joblib

from .config import MODEL_DIR
from .skill_extractor import SkillExtractor
from .text_preprocessor import TextPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Member 1's integrated demo.")
    parser.add_argument(
        "--text",
        default=(
            "Software engineer experienced with Python, Django, REST APIs, Docker, "
            "PostgreSQL, machine learning and Microsoft Azure."
        ),
    )
    args = parser.parse_args()

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun: python -m src.member1.train_logistic_regression"
        )

    preprocessor = TextPreprocessor()
    extractor = SkillExtractor.from_vocabulary_file()
    saved = joblib.load(model_path)
    pipeline = saved["pipeline"]

    cleaned = preprocessor.clean(args.text)
    tokens = preprocessor.tokenize(cleaned)
    skills = extractor.extract(cleaned)
    predicted_category = pipeline.predict([cleaned])[0]

    print("=== MEMBER 1 PIPELINE OUTPUT ===")
    print("\nOriginal text:")
    print(args.text)
    print("\nCleaned text:")
    print(cleaned)
    print("\nTokens:")
    print(tokens)
    print("\nExtracted SKILL entities:")
    print(skills)
    print("\nPredicted job field/category:")
    print(predicted_category)


if __name__ == "__main__":
    main()
