from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import spacy

from .config import MODEL_DIR
from .data_loader import parse_skill_list

SPACE_RE = re.compile(r"\s+")

# Common noisy words that appear in ground-truth 'skills' lists but are not technical skills.
# Filtering these reduces false positives (e.g., 'management', 'skills', 'communication').
VOCAB_BLACKLIST = {
    "skills",
    "skill",
    "management",
    "communication",
    "experience",
    "training",
    "maintenance",
    "development",
    "knowledge",
    "team",
    "work",
    "ability",
}

# Very short programming-language names that are meaningful despite their length.
ALLOWED_SHORT_SKILLS = {"r", "c", "c#", "c++", "go", "ai", "ml", "qa", "hr", "ui", "ux"}


def normalize_skill(skill: object) -> str:
    value = SPACE_RE.sub(" ", str(skill)).strip()
    return value


def build_skill_vocabulary(
    resume_df: pd.DataFrame,
    *,
    min_frequency: int = 2,
    max_words: int = 6,
    max_characters: int = 70,
) -> list[str]:
    """Build a practical skill vocabulary from the dataset's ground-truth skill arrays.

    The public dataset contains some long sentences inside the skills column. Those are
    excluded so the EntityRuler remains focused on skill-like phrases.
    """
    if "skills" not in resume_df.columns:
        raise ValueError("The resume dataset does not contain a 'skills' column.")

    counter: Counter[str] = Counter()
    canonical: dict[str, str] = {}

    for raw_value in resume_df["skills"].dropna():
        for raw_skill in parse_skill_list(raw_value):
            skill = normalize_skill(raw_skill)
            key = skill.casefold()
            word_count = len(skill.split())
            if not skill or len(skill) > max_characters or word_count > max_words:
                continue
            # Skip obviously noisy or generic tokens
            if any(part in VOCAB_BLACKLIST for part in key.split()):
                continue
            if len(skill) <= 2 and key not in ALLOWED_SHORT_SKILLS:
                continue
            counter[key] += 1
            canonical.setdefault(key, skill)

    # Include all single-word skills that meet min_frequency.
    single_word_items = [canonical[key] for key, count in counter.items() if count >= min_frequency and len(key.split()) == 1]

    # For multi-word phrases, include the top N most frequent phrases even if they have low counts
    multi_candidates = [(key, count) for key, count in counter.items() if len(key.split()) > 1]
    multi_candidates.sort(key=lambda kc: (-kc[1], -len(kc[0].split()), -len(kc[0]), kc[0]))
    top_multi = [canonical[key] for key, _ in multi_candidates[:200]]

    items = single_word_items + top_multi
    return sorted(items, key=lambda item: (-len(item.split()), -len(item), item.casefold()))


class SkillExtractor:
    """Rule-based NER baseline built with spaCy EntityRuler.

    It labels exact skill phrases as SKILL entities. It is transparent and suitable for
    the progress demonstration. A learned BERT model is developed separately.
    """

    def __init__(self, skills: Iterable[str]) -> None:
        self.skills = list(dict.fromkeys(normalize_skill(skill) for skill in skills if normalize_skill(skill)))
        # Prefer a small English model when available (provides POS tagging); fall back to a blank pipeline.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")

        ruler = self.nlp.add_pipe(
            "entity_ruler",
            config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True},
        )
        ruler.add_patterns([{"label": "SKILL", "pattern": skill} for skill in self.skills])

        # Whether the loaded pipeline provides POS tags (tagger component)
        self.use_pos = "tagger" in {name for name, _ in self.nlp.pipeline}

        # Additional post-filter blacklist for predicted entities (single words or phrases)
        self.entity_blacklist = {
            "account",
            "management",
            "backup",
            "communication",
            "documentation",
            "database",
            "maintenance",
            "training",
            "skills",
            "skill",
            "development",
            "work",
            "experience",
            "classification",
            "hardware",
            "network",
            "installation",
        }

    def extract(self, text: object, context: object | None = None) -> list[str]:
        """Extract skill entities from `text`.

        If `context` is provided (for example the raw skills column), matches
        that appear in the context are accepted even if filtered by heuristics.
        """
        doc = self.nlp("" if text is None else str(text))
        found: list[str] = []
        seen: set[str] = set()
        ctx = "" if context is None else str(context).casefold()
        for entity in doc.ents:
            if entity.label_ != "SKILL":
                continue
            key = entity.text.casefold()
            # Post-filter: drop entities containing any blacklisted token
            parts = [p for p in key.split() if p]
            blacklisted = any(part in self.entity_blacklist for part in parts)
            if blacklisted and key not in ctx:
                continue

            # POS-based validation: single-word predictions should be a NOUN/PROPN
            if len(parts) == 1:
                if self.use_pos:
                    span = entity
                    token = span[0]
                    if token.pos_ not in {"NOUN", "PROPN"} and token.tag_ not in {"NN", "NNS", "NNP", "NNPS"}:
                        # allow if present in context
                        if key not in ctx:
                            continue
                else:
                    # Fallback heuristic: allow short allowed skill tokens or tokens containing technical chars
                    if key not in ALLOWED_SHORT_SKILLS and not re.search(r"[#+.]", key):
                        # allow if present in context
                        if key not in ctx:
                            continue

            if key not in seen:
                found.append(entity.text)
                seen.add(key)
        return found

    def save_vocabulary(self, path: Path = MODEL_DIR / "skill_vocabulary.txt") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.skills), encoding="utf-8")

    @classmethod
    def from_vocabulary_file(cls, path: Path = MODEL_DIR / "skill_vocabulary.txt") -> "SkillExtractor":
        if not path.exists():
            raise FileNotFoundError(
                f"Skill vocabulary not found at {path}. Run demo_preprocessing or "
                "evaluate_skill_extraction first."
            )
        return cls(path.read_text(encoding="utf-8").splitlines())
