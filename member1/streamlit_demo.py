from __future__ import annotations

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may not be installed in CI
    st = None

from pathlib import Path
from typing import Iterable

from .skill_extractor import SkillExtractor
from .text_preprocessor import TextPreprocessor
from .config import MODEL_DIR


def extract_from_text(text: str, extractor: SkillExtractor, preprocessor: TextPreprocessor) -> dict:
    cleaned = preprocessor.clean(text)
    skills = extractor.extract(cleaned)
    return {"original": text, "cleaned": cleaned, "extracted_skills": skills}


def run_app() -> None:
    if st is None:
        print("streamlit is not installed; run: pip install streamlit and run 'streamlit run src/member1/streamlit_demo.py'")
        return

    st.title("Member1 Skill Extractor Demo")

    st.sidebar.header("Input")
    mode = st.sidebar.radio("Mode", ["Text", "Upload CSV"])

    extractor = SkillExtractor.from_vocabulary_file()
    preprocessor = TextPreprocessor()

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    pipeline = None
    try:
        import joblib

        if model_path.exists():
            saved = joblib.load(model_path)
            pipeline = saved.get("pipeline") if isinstance(saved, dict) else saved
    except Exception:
        pipeline = None

    if mode == "Text":
        text = st.text_area("Paste resume text or job description", height=200)
        if st.button("Extract"):
            out = extract_from_text(text, extractor, preprocessor)
            st.write("Cleaned:", out["cleaned"])
            st.write("Extracted skills:", out["extracted_skills"])
            if pipeline is not None:
                pred = pipeline.predict([out["cleaned"]])[0]
                st.write("Predicted category:", pred)
            # Offer CSV download for the single result
            try:
                import pandas as pd
                import io

                df = pd.DataFrame([
                    {
                        "original": out["original"],
                        "cleaned": out["cleaned"],
                        "extracted_skills": ", ".join(out["extracted_skills"]),
                        "predicted": pred if pipeline is not None else "(no model)",
                    }
                ])
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="demo_result.csv", mime="text/csv")
            except Exception:
                # pandas may not be available; skip download button
                pass
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            import pandas as pd

            df = pd.read_csv(uploaded)
            col = None
            for candidate in ["evaluation_text", "text", "description"]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                st.error("CSV must contain one of: evaluation_text, text, description")
            else:
                n = st.sidebar.number_input("Rows to show", min_value=1, max_value=len(df), value=min(20, len(df)))
                results = []
                for i, r in df.head(n).iterrows():
                    out = extract_from_text(str(r[col]), extractor, preprocessor)
                    pred = pipeline.predict([out["cleaned"]])[0] if pipeline is not None else None
                    results.append({"index": int(i + 1), "original": out["original"], "extracted_skills": ", ".join(out["extracted_skills"]), "predicted": pred})
                st.table(results)
                # Provide CSV download of the shown results
                try:
                    import pandas as pd

                    df = pd.DataFrame(results)
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", data=csv_bytes, file_name="demo_results.csv", mime="text/csv")
                except Exception:
                    pass


if __name__ == "__main__":
    run_app()
