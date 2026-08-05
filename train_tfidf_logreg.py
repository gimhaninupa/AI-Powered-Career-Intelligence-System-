import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "job_classification_train.csv"
TEST_FILE = BASE_DIR / "job_classification_test.csv"
MODEL_FILE = BASE_DIR / "tfidf_logistic_regression_job_classifier.joblib"

def load_csv(path):
    texts, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels

def main():
    X_train, y_train = load_csv(TRAIN_FILE)
    X_test, y_test = load_csv(TEST_FILE)

    if not X_train or not X_test:
        raise ValueError("Training or test data is empty.")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.98,
                sublinear_tf=True,
                max_features=20000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                C=4.0,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = sorted(set(y_test))
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    with (BASE_DIR / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )
    with (BASE_DIR / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model: {MODEL_FILE.name}")

if __name__ == "__main__":
    main()
