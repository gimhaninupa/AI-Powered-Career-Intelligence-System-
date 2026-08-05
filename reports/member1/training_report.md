# Training & Evaluation Report — Member1

**Date:** 2026-08-05

## Summary
- Goal: Improve classification accuracy > 85% by cleaning and broad-mapping job fields.
- Actions taken: cleaned raw CSVs, generated `job_field_mapping.csv` (broad 5-class mapping), retrained TF-IDF+LogReg and fine-tuned BERT, evaluated both models on a held-out stratified split.

## Results
- **TF-IDF + Logistic Regression**
  - Accuracy: **93.26%**
  - Macro F1: **92.45%**
  - Model file: [models/member1/tfidf_logistic_regression.joblib](models/member1/tfidf_logistic_regression.joblib)
  - Metrics: [reports/member1/logistic_regression_metrics.json](reports/member1/logistic_regression_metrics.json)
  - Confusion matrix: [screenshots/member1/tfidf_confusion_matrix.png](screenshots/member1/tfidf_confusion_matrix.png)

- **BERT (fine-tuned, 5-class)**
  - Final validation accuracy: **88.76%** (best epoch 88.76%)
  - Macro F1: **87.76%** (validation)
  - Model dir: [models/member1/bert_job_classifier](models/member1/bert_job_classifier)
  - Metrics: [reports/member1/bert_training_metrics.json](reports/member1/bert_training_metrics.json)
  - Confusion matrix: [screenshots/member1/bert_confusion_matrix.png](screenshots/member1/bert_confusion_matrix.png)

## Files produced
- Cleaned datasets:
  - [data/processed/job_description_cleaned.csv](data/processed/job_description_cleaned.csv)
  - [data/processed/resume_data_cleaned.csv](data/processed/resume_data_cleaned.csv)
  - Mapping: [data/processed/job_field_mapping.csv](data/processed/job_field_mapping.csv)
- Evaluation summary: [reports/member1/evaluation_summary.json](reports/member1/evaluation_summary.json)
- Per-model prediction CSVs:
  - [reports/member1/tfidf_evaluation_predictions.csv](reports/member1/tfidf_evaluation_predictions.csv)
  - [reports/member1/bert_evaluation_predictions.csv](reports/member1/bert_evaluation_predictions.csv)

## Commands used
- Clean data:

```powershell
python scripts/clean_datasets.py
python scripts/generate_broad_mapping.py
```

- Train TF-IDF + Logistic Regression (broad mapping):

```powershell
python -m member1.train_logistic_regression --min-class-count 2
```

- Fine-tune BERT (broad mapping, 4 epochs shown here):

```powershell
python -m member1.train_bert --epochs 4 --batch-size 16 --learning-rate 2e-5
```

- Evaluate both models and save confusion matrices:

```powershell
python scripts/evaluate_models.py
```

## Observations & recommended next steps
- Mapping the original 91 job fields into 5 broad categories produced large, balanced classes and yielded strong performance for both TF-IDF and BERT.
- TF-IDF achieved the highest accuracy (93.3%) with minimal compute — keep as a strong baseline for production use when inference cost matters.
- BERT reaches competitive accuracy (88.8%) and may improve further with longer training or using a larger pre-trained model (e.g., `bert-base-uncased` variants or domain-adapted checkpoints).

Recommended next actions (pick any):
- Compare per-class errors and refine mapping for ambiguous original fields.
- Train BERT for more epochs on GPU or use a larger model for potential gains above 90%.
- Integrate `data/processed/*_cleaned.csv` into upstream loaders permanently (if desired).

---
Report generated automatically from experiment artifacts in the repository.
