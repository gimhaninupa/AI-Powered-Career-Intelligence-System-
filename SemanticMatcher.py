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
from preprocessing import build_semantic_job_profile, clean_text


class SemanticMatcher:
    """
    Semantic Job Matching Engine using Sentence-BERT (SBERT) and K-Nearest Neighbors (KNN).
    
    This module encodes job descriptions and candidate CV profiles into dense semantic vector
    embeddings, allowing semantic similarity matching via Cosine Distance / Cosine Similarity.
    
    Developed for NLP Group Project - Member 2 (Amila Fernando).
    """

    def __init__(self, job_csv_path: str):
        """
        Initializes the SemanticMatcher engine.
        
        1. Loads job dataset from CSV.
        2. Preprocesses and handles missing values.
        3. Combines textual fields into a unified semantic profile for each job.
        4. Loads the pre-trained SBERT model ('all-MiniLM-L6-v2').
        5. Encodes all job descriptions into high-dimensional vector embeddings.
        6. Fits a Cosine Distance NearestNeighbors index for fast similarity retrieval.
        
        :param job_csv_path: Absolute or relative path to job_description.csv dataset
        """
        print(f"[+] Loading dataset from: {job_csv_path}")
        # 1. Load CSV using pandas
        self.df = pd.read_csv(job_csv_path)

        # 2. Handle missing values in text columns by filling NaN with empty string ''
        self.df = self.df.fillna('')

        # 3. Create a preprocessed combined text representation for each job
        print("[+] Preprocessing and combining job text fields (Title, Field, Description, Responsibilities, Skills)...")
        combined_text_series = self.df.apply(build_semantic_job_profile, axis=1)

        # 4. Load the pre-trained SBERT model 'all-MiniLM-L6-v2'
        print("[+] Loading pre-trained SBERT model ('all-MiniLM-L6-v2')...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # 5. Encode all job descriptions into dense vectors (job_embeddings)
        print(f"[+] Encoding {len(self.df)} job descriptions into dense embeddings...")
        self.job_embeddings = self.model.encode(combined_text_series.tolist(), show_progress_bar=False)
        print(f"[+] Successfully generated embeddings of shape: {self.job_embeddings.shape}")

        # 6. Initialize and fit NearestNeighbors with metric='cosine'
        print("[+] Fitting NearestNeighbors index (KNN) with Cosine metric...")
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(self.job_embeddings)
        print("[+] SemanticMatcher initialization complete!\n")

    def match_candidate(self, candidate_text: str, top_k: int = 5) -> list:
        """
        Matches a candidate CV/profile string against all job descriptions in the dataset.
        
        :param candidate_text: Candidate CV summary, skills, or experience string
        :param top_k: Number of top job recommendations to return (default: 5)
        :return: List of dictionaries containing job details, match percentages, and category ratings
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

            # Build result dictionary with job details and percentage match score
            job_match_info = {
                'job_id': idx,
                'job_title': job_row.get('Job Title', ''),
                'job_field': job_row.get('Job Field', ''),
                'match_percentage': similarity_percentage,
                'match_category': match_category,
                'cosine_distance': round(float(dist), 4),
                'job_description': clean_text(str(job_row.get('Job Description', ''))),
                'key_responsibilities': clean_text(str(job_row.get('Key Responsibilities', ''))),
                'required_skills': clean_text(str(job_row.get('Required Skills & Qualifications', '')))
            }
            results.append(job_match_info)

        return results
