from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path so `member1` package can be imported when running the script directly
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from member1.train_logistic_regression import build_dataset
from member1.config import MODEL_DIR, REPORT_DIR, SCREENSHOT_DIR

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def eval_tfidf(model_path: Path, X_test: pd.Series, y_test: pd.Series):
    data = joblib.load(model_path)
    pipeline = data.get("pipeline") if isinstance(data, dict) else data
    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return acc, report, preds


def eval_bert(model_dir: Path, X_test: pd.Series, y_test: pd.Series, batch_size: int = 16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test.iloc[i : i + batch_size].tolist()
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)

    # Map numeric preds back to label names using labels.json if present
    labels_file = Path(model_dir) / "labels.json"
    if labels_file.exists():
        import json

        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        pred_labels = [labels[p] for p in preds]
    else:
        pred_labels = [str(p) for p in preds]

    acc = float(accuracy_score(y_test, pred_labels))
    report = classification_report(y_test, pred_labels, output_dict=True, zero_division=0)
    return acc, report, pred_labels


def save_confusion(y_true, y_pred, out_path: Path, title: str = "Confusion Matrix"):
    labels = sorted(list(set(y_true) | set(y_pred)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    df, target_column = build_dataset(min_class_count=2)
    X = df["cleaned_text"]
    y = df[target_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = {}

    tfidf_model_path = MODEL_DIR / "tfidf_logistic_regression.joblib"
    if tfidf_model_path.exists():
        acc, report, preds = eval_tfidf(tfidf_model_path, X_test, y_test)
        results["tfidf"] = {"accuracy": acc, "report": report}
        save_confusion(y_test, preds, SCREENSHOT_DIR / "tfidf_confusion_matrix.png", "TF-IDF Confusion Matrix")
        pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": preds}).to_csv(
            REPORT_DIR / "tfidf_evaluation_predictions.csv", index=False
        )

    bert_model_dir = MODEL_DIR / "bert_job_classifier"
    if bert_model_dir.exists():
        acc, report, preds = eval_bert(bert_model_dir, X_test, y_test, batch_size=16)
        results["bert"] = {"accuracy": acc, "report": report}
        save_confusion(y_test, preds, SCREENSHOT_DIR / "bert_confusion_matrix.png", "BERT Confusion Matrix")
        pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": preds}).to_csv(
            REPORT_DIR / "bert_evaluation_predictions.csv", index=False
        )

    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved evaluation summary to:", REPORT_DIR / "evaluation_summary.json")


if __name__ == "__main__":
    main()
