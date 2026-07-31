import os
import sys
from SemanticMatcher import SemanticMatcher
from preprocessing import clean_text

# Import Member 1 modules (Nadun)
try:
    from member1.text_preprocessor import TextPreprocessor
    from member1.skill_extractor import SkillExtractor
    MEMBER1_AVAILABLE = True
except ImportError:
    MEMBER1_AVAILABLE = False


def main():
    """
    End-to-End System Integration Test Script:
    Connects Member 1 (Nadun: Text Preprocessing & Skill Extraction)
    with Member 2 (Amila: Semantic Vector Embedding & Job Matching Engine).
    """
    print("==========================================================================")
    print(" AI-POWERED CAREER INTELLIGENCE SYSTEM - END-TO-END INTEGRATION TEST")
    print(" Member 1 (Nadun: CV Parser & Preprocessor) ➡️ Member 2 (Amila: Job Matcher)")
    print("==========================================================================\n")

    # Raw candidate CV text input
    raw_candidate_cv = (
        "Experienced Software Developer skilled in Python, Django, Flask, REST API development, "
        "Docker, PostgreSQL, and Git version control. Strong background in backend architecture, "
        "database design, and cloud deployments."
    )

    print("--------------------------------------------------------------------------")
    print("STEP 1: RAW CANDIDATE CV INPUT (Member 1 Ingestion)")
    print("--------------------------------------------------------------------------")
    print(f"\"{raw_candidate_cv}\"\n")

    # Step 1: Process using Member 1 (Nadun's Preprocessor)
    if MEMBER1_AVAILABLE:
        preprocessor = TextPreprocessor()
        extractor = SkillExtractor.from_vocabulary_file()

        cleaned_cv_text = preprocessor.clean(raw_candidate_cv)
        extracted_skills = extractor.extract(cleaned_cv_text)

        print("--------------------------------------------------------------------------")
        print("STEP 2: MEMBER 1 (NADUN) OUTPUT - CLEANED TEXT & EXTRACTED SKILLS")
        print("--------------------------------------------------------------------------")
        print(f"Cleaned Text:     {cleaned_cv_text}")
        print(f"Extracted Skills: {extracted_skills}\n")
    else:
        cleaned_cv_text = clean_text(raw_candidate_cv)
        print("[!] Member 1 module running in fallback mode.\n")

    # Step 2: Feed into Member 2 (Amila's Semantic Matcher)
    print("--------------------------------------------------------------------------")
    print("STEP 3: MEMBER 2 (AMILA) - SEMANTIC SBERT EMBEDDING & KNN MATCHING")
    print("--------------------------------------------------------------------------")
    
    csv_path = "job_description.csv"
    matcher = SemanticMatcher(job_csv_path=csv_path)

    top_k = 5
    recommendations = matcher.match_candidate(candidate_text=cleaned_cv_text, top_k=top_k)

    print(f"\nTOP {top_k} INTEGRATED JOB RECOMMENDATIONS FOR CANDIDATE:")
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

    print("\nDETAILED INTEGRATION ANALYSIS (Top Recommendation):")
    top = recommendations[0]
    print(f"Matched Job Title:    {top['job_title']} ({top['job_field']})")
    print(f"Overall Match Score:  {top['match_category']} ({top['match_percentage']}%)")
    print(f"Matched Skills (✅):   {', '.join(top['matched_skills']) if top['matched_skills'] else 'None'}")
    print(f"Skill Gap / Missing:  {', '.join(top['missing_skills']) if top['missing_skills'] else 'None'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
