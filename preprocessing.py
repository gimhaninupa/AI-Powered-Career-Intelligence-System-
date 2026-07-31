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


def analyze_skill_gap(candidate_text: str, job_skills_text: str) -> dict:
    """
    Analyzes the skill gap between a candidate's CV text and a target job's required skills.
    
    Identifies:
    - Matched Skills: Skills the candidate already possesses.
    - Missing Skills: Critical skills required by the job that the candidate lacks.
    
    :param candidate_text: Full candidate CV string
    :param job_skills_text: Required Skills string from job dataset
    :return: Dictionary containing matched_skills, missing_skills, and skill counts
    """
    candidate_lower = clean_text(candidate_text).lower()
    job_lower = clean_text(job_skills_text).lower()

    # Pre-defined list of common technical skills and domain competencies
    common_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", "swift", "kotlin",
        "react", "angular", "vue", "node.js", "express", "flask", "django", "spring boot", "html", "css",
        "mongodb", "sql", "postgresql", "mysql", "redis", "oracle", "database design", "database management",
        "docker", "kubernetes", "aws", "azure", "gcp", "devops", "ci/cd", "git", "github", "rest api", "graphql",
        "machine learning", "deep learning", "nlp", "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "agile", "scrum", "jira", "leadership", "communication", "problem solving", "bachelor's", "bs in cs", "master's"
    ]

    candidate_found_skills = set()
    job_found_skills = set()

    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', candidate_lower):
            candidate_found_skills.add(skill.title())
        if re.search(r'\b' + re.escape(skill) + r'\b', job_lower):
            job_found_skills.add(skill.title())

    # Calculate matched vs missing skills
    matched_skills = sorted(list(candidate_found_skills.intersection(job_found_skills)))
    missing_skills = sorted(list(job_found_skills.difference(candidate_found_skills)))

    # Fallback if no pre-defined skills match in job text: split by commas/bullets
    if not job_found_skills and job_skills_text.strip():
        raw_job_items = [s.strip().title() for s in re.split(r'[,•\-\*]', clean_text(job_skills_text)) if len(s.strip()) > 2]
        for item in raw_job_items:
            if item.lower() in candidate_lower:
                matched_skills.append(item)
            else:
                missing_skills.append(item)

    return {
        'matched_skills': sorted(list(set(matched_skills))),
        'missing_skills': sorted(list(set(missing_skills))),
        'matched_count': len(set(matched_skills)),
        'missing_count': len(set(missing_skills))
    }
