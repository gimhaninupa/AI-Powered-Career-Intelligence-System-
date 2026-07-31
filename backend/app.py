import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import ingest_csvs, clean_text, extract_skills, train_classifier
from matching_engine import get_top_matches, calculate_similarity
from cv_generator import generate_cv_bullet

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global data holding variables
resume_df = None
job_df = None
job_classifier = None

def initialize_backend():
    global resume_df, job_df, job_classifier
    print("Ingesting CSV datasets...")
    try:
        resume_df, job_df = ingest_csvs()
        print(f"Loaded {len(job_df)} job descriptions and {len(resume_df)} resumes.")
        
        # Train job field classifier on startup
        if not job_df.empty and "Job Description" in job_df.columns and "Job Field" in job_df.columns:
            job_classifier = train_classifier(job_df)
            print("Job field classifier successfully trained.")
    except Exception as e:
        print(f"Error initializing datasets: {e}")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "job_count": len(job_df) if job_df is not None else 0,
        "resume_count": len(resume_df) if resume_df is not None else 0
    }), 200


@app.route("/api/match-jobs", methods=["POST"])
def match_jobs_endpoint():
    """
    POST /api/match-jobs
    Payload: {"cv_text": "..."}
    Returns Top 5 job matches, match percentages, and extracted candidate skills.
    """
    data = request.get_json(force=True, silent=True) or {}
    cv_text = data.get("cv_text", "").strip()

    if not cv_text:
        return jsonify({"error": "cv_text payload field is required and cannot be empty."}), 400

    if job_df is None or job_df.empty:
        return jsonify({"error": "Job dataset is not loaded on server."}), 500

    try:
        # Extract skills using SpaCy NER
        extracted_skills = extract_skills(cv_text)

        # Get top 5 job matches using SentenceTransformers & KNN
        top_matches = get_top_matches(cv_text, job_df, top_n=5)

        # Predict job field using classifier
        predicted_field = "General"
        if job_classifier and "predict_fn" in job_classifier:
            try:
                predicted_field = job_classifier["predict_fn"](cv_text)
            except Exception:
                pass

        return jsonify({
            "status": "success",
            "cv_length": len(cv_text),
            "predicted_field": predicted_field,
            "extracted_skills": extracted_skills,
            "top_matches": top_matches
        }), 200
    except Exception as e:
        print(f"Error matching jobs: {e}")
        return jsonify({"error": f"An error occurred during job matching: {str(e)}"}), 500


@app.route("/api/generate-cv", methods=["POST"])
def generate_cv_endpoint():
    """
    POST /api/generate-cv
    Payload: {"user_experience": "...", "job_requirement": "..."}
    Returns FLAN-T5 rewritten CV bullet point.
    """
    data = request.get_json(force=True, silent=True) or {}
    user_experience = data.get("user_experience", "").strip()
    job_requirement = data.get("job_requirement", "").strip()

    if not user_experience or not job_requirement:
        return jsonify({
            "error": "Both 'user_experience' and 'job_requirement' are required fields."
        }), 400

    try:
        optimized_bullet = generate_cv_bullet(user_experience, job_requirement)
        return jsonify({
            "status": "success",
            "user_experience": user_experience,
            "job_requirement": job_requirement,
            "generated_bullet": optimized_bullet
        }), 200
    except Exception as e:
        print(f"Error generating CV bullet: {e}")
        return jsonify({"error": f"Failed to generate CV bullet point: {str(e)}"}), 500


if __name__ == "__main__":
    initialize_backend()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
