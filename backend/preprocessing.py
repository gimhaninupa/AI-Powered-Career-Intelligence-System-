import os
import re
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Global cache for SpaCy NLP model
_nlp_model = None

def get_spacy_model():
    """
    Lazy loads the SpaCy en_core_web_sm model, downloading it if not present.
    """
    global _nlp_model
    if _nlp_model is None:
        import spacy
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


def ingest_csvs(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest resume_data.csv and job_description.csv into pandas DataFrames.
    Resolves relative paths dynamically.
    """
    if data_dir is None:
        # Check standard relative locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(current_dir, "..", "Datasets"),
            os.path.join(current_dir, "Datasets"),
            os.path.abspath("Datasets")
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                data_dir = path
                break

    if not data_dir or not os.path.exists(data_dir):
        raise FileNotFoundError(f"Datasets directory not found at {data_dir}")

    resume_path = os.path.join(data_dir, "resume_data.csv")
    job_path = os.path.join(data_dir, "job_description.csv")

    if not os.path.exists(job_path):
        raise FileNotFoundError(f"Job descriptions CSV not found at {job_path}")

    job_df = pd.read_csv(job_path)
    # Normalize column names by stripping trailing whitespace
    job_df.columns = [col.strip() for col in job_df.columns]

    resume_df = pd.DataFrame()
    if os.path.exists(resume_path):
        try:
            resume_df = pd.read_csv(resume_path, low_memory=False)
            resume_df.columns = [col.strip() for col in resume_df.columns]
        except Exception as e:
            print(f"Warning: Failed to fully read resume_data.csv: {e}")

    # Fill NaNs with empty string
    job_df = job_df.fillna("")
    if not resume_df.empty:
        resume_df = resume_df.fillna("")

    return resume_df, job_df


def clean_text(text: str) -> str:
    """
    Strips HTML tags (like <br>, <div>) and non-alphanumeric noise using regex.
    Preserves essential technical characters (+ # . -) for programming languages (C++, C#, Node.js, .NET).
    """
    if not text or not isinstance(text, str):
        return ""

    # Detect and correct PDF spaced-out letters (e.g., "S o f t w a r e  E n g i n e e r")
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        tokens = [t for t in line_strip.split(" ") if t]
        if len(tokens) > 2:
            single_char_count = sum(1 for t in tokens if len(t) == 1)
            ratio = single_char_count / len(tokens)
            if ratio > 0.60:
                words = re.split(r'\s{2,}', line_strip)
                cleaned_words = [w.replace(" ", "") for w in words if w]
                line = " ".join(cleaned_words)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Replace HTML tags (like <br>, <br/>, <div>, <p>) with space
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Unescape HTML entities if any
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')

    # Remove non-ASCII or obscure control characters, keep standard text + technical symbols (+, #, ., -)
    text = re.sub(r"[^\w\s\+\#\.\-]", " ", text)

    # Normalize multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def extract_skills(text: str) -> List[str]:
    """
    Uses SpaCy (en_core_web_sm) to extract technical noun chunks and entity keywords from raw CV text.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    nlp = get_spacy_model()
    doc = nlp(cleaned)

    extracted_skills = []
    
    # 1. Extract noun chunks
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        # Filter out common stop words, pronouns, or trivial single characters
        if len(chunk_text) > 1 and not chunk.root.is_stop:
            extracted_skills.append(chunk_text)

    # 2. Extract Named Entities (Organizations, Products, Technical items)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "NORP"]:
            ent_text = ent.text.strip()
            if len(ent_text) > 1:
                extracted_skills.append(ent_text)

    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for skill in extracted_skills:
        skill_lower = skill.lower()
        if skill_lower not in seen and len(skill) > 1:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills[:20]  # Return top extracted noun chunks/skills


def train_classifier(job_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a LogisticRegression classifier with TfidfVectorizer to classify job descriptions into Job Fields.
    Returns dictionary with trained vectorizer, model, and accuracy metadata.
    """
    if "Job Description" not in job_df.columns or "Job Field" not in job_df.columns:
        raise ValueError("DataFrame missing 'Job Description' or 'Job Field' columns")

    # Filter rows with non-empty descriptions and fields
    valid_df = job_df[job_df["Job Description"].astype(bool) & job_df["Job Field"].astype(bool)].copy()
    
    # Prepare text feature by cleaning text
    valid_df["clean_desc"] = valid_df["Job Description"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(valid_df["clean_desc"])
    y = valid_df["Job Field"]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    def predict_job_field(text: str) -> str:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        return str(clf.predict(vec)[0])

    return {
        "vectorizer": vectorizer,
        "model": clf,
        "predict_fn": predict_job_field
    }
