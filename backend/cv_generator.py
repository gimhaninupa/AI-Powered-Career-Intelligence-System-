from typing import Optional
from transformers import pipeline

_generator_pipeline = None


def get_generator_pipeline():
    """
    Lazy loads and caches the Hugging Face text2text-generation pipeline with google/flan-t5-base.
    """
    global _generator_pipeline
    if _generator_pipeline is None:
        _generator_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )
    return _generator_pipeline


def generate_cv_bullet(user_experience: str, job_requirement: str) -> str:
    """
    Takes candidate experience and job requirement, applying the exact required prompt:
    "Rewrite the following candidate experience to align with the job requirement. Candidate Experience: {user_experience}. Job Requirement: {job_requirement}."
    Returns the FLAN-T5 generated text bullet point.
    """
    if not user_experience or not user_experience.strip():
        user_experience = "Proficient in technical problem solving and software execution."
    if not job_requirement or not job_requirement.strip():
        job_requirement = "Demonstrated expertise in relevant role domains."

    prompt = f"Rewrite the following candidate experience to align with the job requirement. Candidate Experience: {user_experience.strip()}. Job Requirement: {job_requirement.strip()}."

    pipe = get_generator_pipeline()
    response = pipe(
        prompt,
        max_length=150,
        min_length=15,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    if response and isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "").strip()
        return generated_text

    return "Successfully tailored experience to align with required job competencies."
