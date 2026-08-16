from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_tokenizer = None
_model = None


def get_generator_model():
    """
    Lazy loads and caches the Hugging Face tokenizer and model for google/flan-t5-base.
    """
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        _model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return _tokenizer, _model


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

    tokenizer, model = get_generator_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=15,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    if outputs is not None and len(outputs) > 0:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return generated_text

    return "Successfully tailored experience to align with required job competencies."
