from __future__ import annotations

from .data_loader import load_resume_data
from .skill_extractor import SkillExtractor, build_skill_vocabulary
from .text_preprocessor import TextPreprocessor


def main() -> None:
    sample = (
        "• Developed Python applications using Django.<br>"
        "• Deployed Docker services to Microsoft Azure!<br>"
        "Contact: candidate@example.com"
    )

    resume_df = load_resume_data()
    vocabulary = build_skill_vocabulary(resume_df, min_frequency=2)
    extractor = SkillExtractor(vocabulary)
    extractor.save_vocabulary()

    preprocessor = TextPreprocessor()
    processed = preprocessor.process(sample)
    skills = extractor.extract(processed["cleaned_text"])

    print("=== RAW TEXT ===")
    print(processed["original_text"])
    print("\n=== CLEANED TEXT ===")
    print(processed["cleaned_text"])
    print("\n=== TOKENS ===")
    print(processed["tokens"])
    print("\n=== EXTRACTED SKILL ENTITIES ===")
    print(skills)
    print(f"\nVocabulary size: {len(vocabulary)}")


if __name__ == "__main__":
    main()
