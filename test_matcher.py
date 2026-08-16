import os
from SemanticMatcher import SemanticMatcher


def main():
    """
    Test script for verifying SemanticMatcher functionality.
    
    Loads job_description.csv, passes a mock candidate CV,
    and cleanly prints top 5 recommended jobs with percentage match scores.
    """
    # Define dataset path
    csv_path = "job_description.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found in current directory.")

    print("===============================================================")
    print(" NLP JOB MATCHING ENGINE - MEMBER 2: AMILA FERNANDO")
    print("===============================================================\n")

    # 1. Initialize SemanticMatcher engine (Loads CSV, SBERT model, encodes embeddings, fits KNN)
    matcher = SemanticMatcher(job_csv_path=csv_path)

    # 2. Mock candidate CV text profile
    mock_candidate_cv = (
        "Software Engineering undergraduate skilled in Python, React, Flask, and MongoDB. "
        "Experienced in web application development, RESTful API design, database schema management, "
        "and version control using Git. Passionate about software architecture and backend engineering."
    )

    print("---------------------------------------------------------------")
    print("CANDIDATE PROFILE:")
    print(f"\"{mock_candidate_cv}\"")
    print("---------------------------------------------------------------\n")

    # 3. Retrieve Top 5 job matches using KNN
    top_k = 5
    recommendations = matcher.match_candidate(candidate_text=mock_candidate_cv, top_k=top_k)

    # 4. Cleanly print top recommendations and percentage match scores
    print(f"TOP {top_k} RECOMMENDED JOBS FOR CANDIDATE:")
    print("=" * 90)
    print(f"{'Rank':<6}{'Match %':<12}{'Match Rating':<18}{'Job Title':<32}{'Job Field':<20}")
    print("-" * 90)

    for rank, rec in enumerate(recommendations, 1):
        rank_str = f"#{rank}"
        match_str = f"{rec['match_percentage']}%"
        rating = rec['match_category']
        title = rec['job_title']
        field = rec['job_field']
        
        print(f"{rank_str:<6}{match_str:<12}{rating:<18}{title:<32}{field:<20}")

    print("=" * 90)
    print("\nDETAILED BREAKDOWN & SKILL GAP ANALYSIS OF TOP MATCH:")
    top_match = recommendations[0]
    print(f"Title:                {top_match['job_title']}")
    print(f"Field:                {top_match['job_field']}")
    print(f"Match Rating:         {top_match['match_category']} ({top_match['match_percentage']}%)")
    print(f"Cosine Distance:      {top_match['cosine_distance']}")
    print(f"Matched Skills (✅):   {', '.join(top_match['matched_skills']) if top_match['matched_skills'] else 'None detected'}")
    print(f"Missing Skills (⚠️):   {', '.join(top_match['missing_skills']) if top_match['missing_skills'] else 'None (Complete skill overlap)'}")
    print(f"Learning Roadmap (🚀): {top_match['learning_roadmap']}")
    print(f"Description:          {top_match['job_description'][:150]}...")
    print(f"Required Skills:      {top_match['required_skills']}")
    print("=" * 90)


if __name__ == "__main__":
    main()
