from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import MAPPING_CSV, MODEL_DIR, REPORT_DIR, SCREENSHOT_DIR
from .data_loader import JOB_TEXT_COLUMNS, combine_text_columns, load_job_data
from .text_preprocessor import TextPreprocessor


def apply_optional_broad_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not MAPPING_CSV.exists():
        return df, "Job Field"

    mapping = pd.read_csv(MAPPING_CSV)
    required = {"original_job_field", "broad_category"}
    if not required.issubset(mapping.columns):
        return df, "Job Field"

    mapping["broad_category"] = mapping["broad_category"].fillna("").astype(str).str.strip()
    completed = mapping[mapping["broad_category"] != ""]
    if completed.empty:
        return df, "Job Field"

    lookup = dict(zip(completed["original_job_field"], completed["broad_category"]))
    result = df.copy()
    result["Broad Category"] = result["Job Field"].map(lookup)
    result = result.dropna(subset=["Broad Category"])
    if result["Broad Category"].nunique() < 2:
        raise ValueError("The completed mapping produces fewer than two broad categories.")
    return result, "Broad Category"


def build_dataset(min_class_count: int) -> tuple[pd.DataFrame, str]:
    df = load_job_data().drop_duplicates().copy()
    df, target_column = apply_optional_broad_mapping(df)
    df["combined_text"] = combine_text_columns(df, JOB_TEXT_COLUMNS)

    preprocessor = TextPreprocessor()
    df["cleaned_text"] = df["combined_text"].map(preprocessor.clean)
    df = df[df["cleaned_text"].str.len() > 0].copy()

    counts = df[target_column].value_counts()
    valid_labels = counts[counts >= min_class_count].index
    df = df[df[target_column].isin(valid_labels)].copy()

    if df[target_column].nunique() < 2:
        raise ValueError(
            "Not enough target classes remain. Lower --min-class-count or complete "
            "data/processed/job_field_mapping.csv with broader categories."
        )
    return df, target_column


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression.")
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df, target_column = build_dataset(args.min_class_count)
    X = df["cleaned_text"]
    y = df[target_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3_000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    metrics = {
        "target_column": target_column,
        "records_used": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "number_of_classes": int(y.nunique()),
        "minimum_class_count": args.min_class_count,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    metrics_path = REPORT_DIR / "logistic_regression_metrics.json"
    predictions_path = REPORT_DIR / "logistic_regression_predictions.csv"
    figure_path = SCREENSHOT_DIR / "logistic_regression_confusion_matrix.png"

    joblib.dump({"pipeline": pipeline, "target_column": target_column}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"text": X_test.values, "actual": y_test.values, "predicted": predictions}
    ).to_csv(predictions_path, index=False)

    # A full 50+ class matrix is unreadable. Plot the 15 most common test labels.
    top_labels = y_test.value_counts().head(15).index.tolist()
    mask = y_test.isin(top_labels)
    if mask.any():
        fig, ax = plt.subplots(figsize=(14, 12))
        ConfusionMatrixDisplay.from_predictions(
            y_test[mask],
            predictions[mask],
            labels=top_labels,
            xticks_rotation=90,
            normalize="true",
            values_format=".2f",
            ax=ax,
        )
        ax.set_title("TF-IDF + Logistic Regression: Top-Class Confusion Matrix")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)

    print("\n=== RESULTS ===")
    print(f"Target: {target_column}")
    print(f"Records used: {len(df)}")
    print(f"Classes: {y.nunique()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved confusion matrix: {figure_path}")


if __name__ == "__main__":
    main()
