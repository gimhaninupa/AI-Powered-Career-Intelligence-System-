import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from preprocessing import build_semantic_job_profile, clean_text, analyze_skill_gap


class SemanticMatcher:
    """
    Semantic Job Matching Engine using Sentence-BERT (SBERT) and K-Nearest Neighbors (KNN).
    
    This module encodes job descriptions and candidate CV profiles into dense semantic vector
    embeddings, allowing semantic similarity matching via Cosine Distance / Cosine Similarity.
    
    Developed for NLP Group Project - Member 2 (Amila Fernando).
    """

    def __init__(self, job_csv_path: str, cache_file: str = "job_embeddings.npy"):
        """
        Initializes the SemanticMatcher engine with disk caching for high performance.
        
        1. Loads job dataset from CSV.
        2. Loads the pre-trained SBERT model ('all-MiniLM-L6-v2').
        3. Checks if pre-computed embeddings exist in disk cache (job_embeddings.npy).
           - If cache exists: Loads embeddings instantly from disk (< 10 ms).
           - If cache missing: Computes SBERT embeddings and saves them to disk.
        4. Fits a Cosine Distance NearestNeighbors index for fast similarity retrieval.
        
        :param job_csv_path: Path to job_description.csv dataset
        :param cache_file: Target path for saving/loading numpy embedding cache
        """
        self.cache_file = cache_file
        print(f"[+] Loading dataset from: {job_csv_path}")
        # 1. Load CSV using pandas
        self.df = pd.read_csv(job_csv_path).fillna('')

        # 2. Load the pre-trained SBERT model 'all-MiniLM-L6-v2' (Fast local load)
        print("[+] Loading pre-trained SBERT model ('all-MiniLM-L6-v2')...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', local_files_only=True)
        except Exception:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # 3. Check for cached embeddings on disk
        if os.path.exists(self.cache_file):
            print(f"[+] Found cached embeddings at '{self.cache_file}'. Loading from disk (Instant Load!)...")
            self.job_embeddings = np.load(self.cache_file)
            print(f"[+] Successfully loaded {self.job_embeddings.shape[0]} embeddings from cache of shape {self.job_embeddings.shape}")
        else:
            print("[+] Cache not found. Preprocessing job text fields...")
            combined_text_series = self.df.apply(build_semantic_job_profile, axis=1)

            print(f"[+] Encoding {len(self.df)} job descriptions into dense embeddings...")
            self.job_embeddings = self.model.encode(combined_text_series.tolist(), show_progress_bar=False)

            print(f"[+] Saving embeddings to disk cache: {self.cache_file}")
            np.save(self.cache_file, self.job_embeddings)

        # 4. Initialize and fit NearestNeighbors with metric='cosine'
        print("[+] Fitting NearestNeighbors index (KNN) with Cosine metric...")
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(self.job_embeddings)
        print("[+] SemanticMatcher initialization complete!\n")

    def match_candidate(self, candidate_text: str, top_k: int = 5) -> list:
        """
        Matches a candidate CV/profile string against all job descriptions in the dataset.
        Also performs Skill Gap Analysis to extract matched and missing skills.
        
        :param candidate_text: Candidate CV summary, skills, or experience string
        :param top_k: Number of top job recommendations to return (default: 5)
        :return: List of dictionaries containing job details, match percentages, ratings, and skill gap insights
        """
        # Clean candidate text before encoding
        cleaned_candidate_text = clean_text(candidate_text)

        # 1. Encode candidate_text into a 384-dim dense vector using SBERT
        candidate_embedding = self.model.encode([cleaned_candidate_text])

        # 2. Retrieve top_k closest job embeddings using KNN
        distances, indices = self.knn.kneighbors(candidate_embedding, n_neighbors=top_k)

        # Flatten 2D output arrays from kneighbors
        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            # 3. Convert Cosine Distance to Similarity Percentage
            similarity_percentage = round(float((1 - dist) * 100), 2)

            # Categorize match quality
            if similarity_percentage >= 70.0:
                match_category = "Strong Match"
            elif similarity_percentage >= 60.0:
                match_category = "Good Match"
            elif similarity_percentage >= 50.0:
                match_category = "Moderate Match"
            else:
                match_category = "Low Fit"

            job_row = self.df.iloc[idx]
            job_skills_raw = str(job_row.get('Required Skills & Qualifications', ''))

            # Perform Skill Gap Analysis (Matched vs Missing Skills)
            skill_gap = analyze_skill_gap(cleaned_candidate_text, job_skills_raw)

            # Build result dictionary with job details and skill gap analysis
            job_match_info = {
                'job_id': idx,
                'job_title': job_row.get('Job Title', ''),
                'job_field': job_row.get('Job Field', ''),
                'match_percentage': similarity_percentage,
                'match_category': match_category,
                'cosine_distance': round(float(dist), 4),
                'matched_skills': skill_gap['matched_skills'],
                'missing_skills': skill_gap['missing_skills'],
                'job_description': clean_text(str(job_row.get('Job Description', ''))),
                'key_responsibilities': clean_text(str(job_row.get('Key Responsibilities', ''))),
                'required_skills': clean_text(job_skills_raw)
            }
            results.append(job_match_info)

        return results
