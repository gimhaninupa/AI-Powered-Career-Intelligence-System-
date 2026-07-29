import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text

# Global caches for SentenceTransformer model and precomputed embeddings
_transformer_model = None
_job_embeddings_cache = None
_knn_model_cache = None
_cached_job_df_id = None


def get_transformer_model() -> SentenceTransformer:
    """
    Lazy loads and caches the SentenceTransformer all-MiniLM-L6-v2 model.
    """
    global _transformer_model
    if _transformer_model is None:
        _transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _transformer_model


def calculate_similarity(resume_text: str, job_text: str) -> float:
    """
    Converts cleaned resume text and job description text into dense 384-d vectors
    and calculates Cosine Similarity (returns float between 0.0 and 1.0).
    """
    model = get_transformer_model()
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_text)

    if not clean_resume or not clean_job:
        return 0.0

    embeddings = model.encode([clean_resume, clean_job], convert_to_numpy=True)
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def _build_job_embeddings_and_knn(job_df: pd.DataFrame):
    """
    Precomputes dense vector embeddings for all job descriptions in the dataset
    and fits a NearestNeighbors (KNN) model using cosine distance.
    """
    global _job_embeddings_cache, _knn_model_cache, _cached_job_df_id
    
    current_df_id = id(job_df)
    if _job_embeddings_cache is not None and _cached_job_df_id == current_df_id:
        return _job_embeddings_cache, _knn_model_cache

    model = get_transformer_model()

    # Combine Title, Field, Description, and Required Skills for rich semantic representation
    combined_texts = []
    for _, row in job_df.iterrows():
        title = clean_text(str(row.get("Job Title", "")))
        field = clean_text(str(row.get("Job Field", "")))
        desc = clean_text(str(row.get("Job Description", "")))
        skills = clean_text(str(row.get("Required Skills & Qualifications", "")))
        text = f"{title}. {field}. {desc}. Required skills: {skills}"
        combined_texts.append(text)

    # Generate 384-dimensional dense vector embeddings
    embeddings = model.encode(combined_texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Fit NearestNeighbors model (metric='cosine')
    knn = NearestNeighbors(n_neighbors=min(5, len(job_df)), metric="cosine")
    knn.fit(embeddings)

    _job_embeddings_cache = embeddings
    _knn_model_cache = knn
    _cached_job_df_id = current_df_id

    return embeddings, knn


def get_top_matches(cv_text: str, job_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Ingests candidate raw CV text, cleans it, converts to a dense vector,
    and queries NearestNeighbors (KNN) to return the Top 5 job matches
    and their similarity percentage scores as a JSON-friendly list of dicts.
    """
    cleaned_cv = clean_text(cv_text)
    if not cleaned_cv or job_df.empty:
        return []

    model = get_transformer_model()
    cv_embedding = model.encode([cleaned_cv], convert_to_numpy=True)

    embeddings, knn = _build_job_embeddings_and_knn(job_df)

    n_results = min(top_n, len(job_df))
    # knn.kneighbors returns cosine distances (0 = identical, 1 = orthogonal)
    distances, indices = knn.kneighbors(cv_embedding, n_neighbors=n_results)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = job_df.iloc[idx]
        
        # Convert cosine distance to cosine similarity: sim = 1 - distance
        similarity_score = float(np.clip(1.0 - dist, 0.0, 1.0))
        match_percentage = round(similarity_score * 100, 1)

        job_match = {
            "id": int(idx),
            "job_title": str(row.get("Job Title", "N/A")),
            "job_field": str(row.get("Job Field", "General")),
            "job_description": clean_text(str(row.get("Job Description", ""))),
            "key_responsibilities": clean_text(str(row.get("Key Responsibilities", ""))),
            "required_skills": clean_text(str(row.get("Required Skills & Qualifications", ""))),
            "similarity_score": round(similarity_score, 4),
            "match_percentage": match_percentage
        }
        results.append(job_match)

    return results
