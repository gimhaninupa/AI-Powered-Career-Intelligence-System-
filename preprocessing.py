import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags, extra whitespace, and special formatting artifacts.
    
    :param text: Raw input string from job description or CV
    :return: Sanitized, normalized text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Remove HTML line breaks (<br>, <br/>, <p>, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Replace bullet point characters (•, -, *) with clean spacing
    text = re.sub(r'[•\-\*]', ' ', text)

    # 3. Collapse multiple whitespaces and newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_semantic_job_profile(row: pd.Series) -> str:
    """
    Constructs a structured, weighted textual profile for a job entry to feed into Sentence-BERT.
    
    Field Weighting Strategy:
    - Job Title & Field are prioritized at the start for strong domain context.
    - Required Skills & Key Responsibilities are cleaned and emphasized.
    
    :param row: Pandas Series representing a single row from job_description.csv
    :return: Formatted text string ready for vector embedding
    """
    title = clean_text(str(row.get('Job Title', '')))
    field = clean_text(str(row.get('Job Field', '')))
    desc = clean_text(str(row.get('Job Description', '')))
    resp = clean_text(str(row.get('Key Responsibilities', '')))
    skills = clean_text(str(row.get('Required Skills & Qualifications', '')))

    # Structure into a clean, weighted textual prompt for SBERT
    profile = (
        f"Role: {title}. "
        f"Industry Field: {field}. "
        f"Description: {desc}. "
        f"Core Responsibilities: {resp}. "
        f"Required Qualifications & Technical Skills: {skills}."
    )
    
    return profile
