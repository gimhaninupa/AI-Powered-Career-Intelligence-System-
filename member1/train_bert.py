from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import MODEL_DIR, REPORT_DIR
from .train_logistic_regression import build_dataset
from .text_preprocessor import TextPreprocessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def run_epoch(model, loader, device, optimizer=None, scheduler=None) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(**batch)
            loss = outputs.loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        predictions = outputs.logits.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    return EpochResult(
        loss=total_loss / max(len(loader), 1),
        accuracy=float(accuracy),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for job-field classification.")
    parser.add_argument("--model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Use 0 for all records. Use a smaller number only for a quick progress test.",
    )
    args = parser.parse_args()

    set_seed(args.random_state)
    df, target_column = build_dataset(args.min_class_count)
    if args.max_records > 0 and len(df) > args.max_records:
        # Keep the quick run approximately balanced so stratified splitting remains valid.
        class_count = df[target_column].nunique()
        per_class = max(2, args.max_records // max(class_count, 1))
        df = (
            df.groupby(target_column, group_keys=False)
            .apply(lambda group: group.sample(min(len(group), per_class), random_state=args.random_state), include_groups=True)
            .reset_index(drop=True)
        )
        retained = df[target_column].value_counts()
        df = df[df[target_column].isin(retained[retained >= 2].index)].copy()

    preprocessor = TextPreprocessor()
    # Re-clean more gently for BERT so sentence context is retained.
    texts = [preprocessor.clean(text, for_bert=True) for text in df["combined_text"]]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[target_column].astype(str)).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=args.random_state,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_),
        id2label={index: label for index, label in enumerate(label_encoder.classes_)},
        label2id={label: index for index, label in enumerate(label_encoder.classes_)},
    )

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * 0.1), 1),
        num_training_steps=total_steps,
    )

    history: list[dict[str, object]] = []
    print(f"Device: {device}")
    print(f"Training records: {len(train_dataset)}")
    print(f"Validation records: {len(val_dataset)}")
    print(f"Classes: {len(label_encoder.classes_)}")

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, device, optimizer, scheduler)
        validation_result = run_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train": train_result.__dict__,
            "validation": validation_result.__dict__,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_result.loss:.4f}, train F1={train_result.macro_f1:.4f} | "
            f"val loss={validation_result.loss:.4f}, val accuracy={validation_result.accuracy:.4f}, "
            f"val macro F1={validation_result.macro_f1:.4f}"
        )

    output_dir = MODEL_DIR / "bert_job_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "labels.json").write_text(
        json.dumps(label_encoder.classes_.tolist(), indent=2), encoding="utf-8"
    )

    report = {
        "model_name": args.model_name,
        "target_column": target_column,
        "device": str(device),
        "records_used": len(df),
        "number_of_classes": len(label_encoder.classes_),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "min_class_count": args.min_class_count,
        },
        "history": history,
    }
    report_path = REPORT_DIR / "bert_training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved BERT model to: {output_dir}")
    print(f"Saved BERT metrics to: {report_path}")


if __name__ == "__main__":
    main()
