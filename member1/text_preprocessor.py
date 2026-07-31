from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

import spacy

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+'-]+@[\w.-]+\.[A-Za-z]{2,}\b")
BULLET_RE = re.compile(r"[•●▪◦·]+")
SPACE_RE = re.compile(r"\s+")
# Keep characters used in technical skills such as C++, C#, .NET, CI/CD and Node.js.
UNWANTED_RE = re.compile(r"[^\w\s+#./-]")


@dataclass
class TextPreprocessor:
    lowercase: bool = True

    def __post_init__(self) -> None:
        # A blank English pipeline provides a deterministic tokenizer and requires no model download.
        self.nlp = spacy.blank("en")

    def clean(self, text: object, *, for_bert: bool = False) -> str:
        if text is None:
            return ""
        value = str(text)
        value = html.unescape(value)
        value = unicodedata.normalize("NFKC", value)
        value = HTML_TAG_RE.sub(" ", value)
        value = BULLET_RE.sub(" ", value)
        value = URL_RE.sub(" URL ", value)
        value = EMAIL_RE.sub(" EMAIL ", value)
        value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")

        # BERT should retain normal sentence punctuation and contextual structure.
        if not for_bert:
            value = UNWANTED_RE.sub(" ", value)

        value = SPACE_RE.sub(" ", value).strip()
        if self.lowercase:
            value = value.lower()
        return value

    def tokenize(self, text: object, *, for_bert: bool = False) -> list[str]:
        cleaned = self.clean(text, for_bert=for_bert)
        return [token.text for token in self.nlp(cleaned) if not token.is_space]

    def process(self, text: object, *, for_bert: bool = False) -> dict[str, object]:
        cleaned = self.clean(text, for_bert=for_bert)
        return {
            "original_text": "" if text is None else str(text),
            "cleaned_text": cleaned,
            "tokens": self.tokenize(cleaned, for_bert=for_bert),
        }
