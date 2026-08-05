# Training-Ready Job Classification Dataset

## Purpose

This dataset supports Member 1's **TF-IDF + Logistic Regression job-sector classification** component in the AI Powered Career Intelligence System.

## Important design decision

The uploaded source contained **91 highly fragmented job fields across only 607 rows**. Several fields had only one or two examples, which makes an 85–90% multiclass result unrealistic and causes unstable minority-class recall.

This package therefore:

- uses only the uploaded public job-description records;
- generates no artificial job descriptions;
- removes 28 exact duplicate combined texts;
- maps the 91 small fields into five defensible high-level sectors;
- combines title, description, responsibilities, and required skills into the `text` feature;
- provides a fixed stratified 80/20 split.

## Files

- `job_classification_5class.csv`: all 579 unique records.
- `job_classification_train.csv`: 463 training records.
- `job_classification_test.csv`: 116 test records.
- `job_field_mapping.csv`: original 91 fields mapped to five broad labels.
- `class_distribution.csv`: label counts.
- `benchmark_results.json`: measured baseline results.
- `train_tfidf_logreg.py`: reproducible model training script.

## Labels

- Arts, Media, Education & Public Service: 143 records
- Banking, Finance & Audit: 105 records
- Consulting, Business & Operations: 106 records
- Engineering, Manufacturing & Technology: 108 records
- Healthcare & Life Sciences: 117 records

## Reproduced benchmark

Using the supplied fixed split and the included script:

- Holdout accuracy: 91.38%
- Holdout macro F1: 91.32%
- Five-fold mean accuracy: 91.70%
- Five-fold mean macro F1: 91.60%

Results can vary if preprocessing, the split, solver, or hyperparameters are changed.

## Run

```bash
pip install -r requirements.txt
python train_tfidf_logreg.py
```

## Source

https://www.kaggle.com/datasets/shivamashokgupta/job-description

## Academic-use note

The broad-label mapping is a preprocessing and taxonomy decision, not synthetic data generation. Document the mapping and justify why extremely small original classes were consolidated. Do not report a benchmark you have not reproduced on your own machine.
