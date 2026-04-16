#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_genre.py

Reproduces Table 3 of the paper: genre detection performance comparison
between classifiers trained on original, noisy, and automatically cleaned
Hausa text, using stratified 5-fold cross-validation on the OOD dataset.

Default model checkpoint: castorini/afriberta_base  (AfriBERTa-base),
matching the paper's experimental setup (Section 6.1).

Example usage:
    python evaluate_genre.py \
        --file-path ./data/ood_genre_dataset.tsv \
        --original-col  original_text \
        --noisy-col     noisy_text \
        --cleaned-col   cleaned_text \
        --label-col     genre \
        --model-checkpoint castorini/afriberta_base \
        --num-epochs 3 \
        --batch-size 8 \
        --folds 5 \
        --output-dir ./genre_results
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch

import evaluate
from datasets import Dataset
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Load metrics
try:
    metric_accuracy  = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall    = evaluate.load("recall")
    metric_f1        = evaluate.load("f1")
    print("Metrics loaded successfully.")
except Exception as exc:
    raise RuntimeError(f"Could not load evaluation metrics: {exc}")


def compute_metrics(eval_pred):
    """Weighted accuracy / precision / recall / F1 for the Trainer callback."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy":  metric_accuracy.compute( predictions=predictions, references=labels)["accuracy"],
        "precision": metric_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall":    metric_recall.compute(   predictions=predictions, references=labels, average="weighted")["recall"],
        "f1":        metric_f1.compute(       predictions=predictions, references=labels, average="weighted")["f1"],
    }


# Core CV function
def run_cross_validation(
    df,
    text_column,
    label_column,
    model_name,
    model_checkpoint,
    output_dir_base="./results",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    seed=42,
    n_splits=5,
):
    """
    Train and evaluate a sequence classifier with stratified k-fold CV.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain `text_column` and `label_column`.
    text_column : str
        Column with the text to classify.
    label_column : str
        Column with genre labels.
    model_name : str
        Human-readable name used for logging and output directories.
    model_checkpoint : str
        HuggingFace model identifier.
    output_dir_base : str
        Root directory for per-fold checkpoints.
    num_epochs : int
    batch_size : int
    learning_rate : float
    seed : int
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict with keys: model_name, mean_accuracy, std_accuracy, mean_f1,
                    std_f1, per_fold_metrics, classification_report,
                    oof_predictions.
    """
    print(f"\n{'='*60}")
    print(f"  {n_splits}-Fold CV  |  column: '{text_column}'  |  model: {model_name}")
    print(f"{'='*60}")
    set_seed(seed)

    # Preprocessing
    df_work = df.copy()
    df_work = df_work.dropna(subset=[text_column, label_column])

    if df_work.empty:
        return {"error": "DataFrame is empty after dropping NaN rows."}

    df_work[text_column]  = df_work[text_column].astype(str)
    df_work[label_column] = df_work[label_column].astype(str)

    # Use a clean 0-based integer index throughout; no need to track originals.
    df_work = df_work.reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    df_work["label_encoded"] = label_encoder.fit_transform(df_work[label_column])
    num_labels  = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    id2label    = {i: lbl for i, lbl in enumerate(class_names)}
    label2id    = {lbl: i for i, lbl in id2label.items()}

    print(f"Labels ({num_labels}): {list(class_names)}")
    print(f"Samples after cleaning: {len(df_work)}")

    if len(df_work) < n_splits:
        return {"error": f"Not enough rows ({len(df_work)}) for {n_splits}-fold CV."}
    if num_labels < 2:
        return {"error": "Need at least 2 classes for classification."}

    # Load tokenizer once
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    except Exception as exc:
        return {"error": f"Tokenizer load failed: {exc}"}

    def tokenize(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=256)

    # Stratified k-fold loop
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Out-of-fold integer predictions
    oof_preds    = np.full(len(df_work), fill_value=-1, dtype=int)
    fold_metrics = []

    X = df_work[text_column].values
    y = df_work["label_encoded"].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        train_df_fold = df_work.iloc[train_idx][[text_column, "label_encoded"]]
        val_df_fold   = df_work.iloc[val_idx]  [[text_column, "label_encoded"]]

        train_hf = Dataset.from_pandas(
            train_df_fold.rename(columns={"label_encoded": "label"}), preserve_index=False
        )
        val_hf = Dataset.from_pandas(
            val_df_fold.rename(columns={"label_encoded": "label"}), preserve_index=False
        )

        train_tok = train_hf.map(tokenize, batched=True, remove_columns=[text_column])
        val_tok   = val_hf.map(  tokenize, batched=True, remove_columns=[text_column])
        train_tok.set_format("torch")
        val_tok.set_format("torch")

        # Fresh model for every fold to prevent data leakage across folds
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        except Exception as exc:
            print(f"  Model load failed on fold {fold}: {exc}")
            continue

        fold_output_dir = os.path.join(
            output_dir_base, f"{model_name.replace(' ', '_')}_fold{fold}"
        )

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            # tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        fold_acc = eval_results.get("eval_accuracy", 0.0)
        fold_f1  = eval_results.get("eval_f1",       0.0)
        fold_metrics.append({"accuracy": fold_acc, "f1": fold_f1})
        print(f"  Fold {fold} — Accuracy: {fold_acc*100:.1f}%  F1: {fold_f1:.4f}")

        # Store out-of-fold predictions (positional index into df_work)
        preds_out   = trainer.predict(val_tok)
        fold_preds  = np.argmax(preds_out.predictions, axis=-1)
        oof_preds[val_idx] = fold_preds

    # Aggregate results. This format matches Table 3 of mean ± std
    accs = np.array([m["accuracy"] for m in fold_metrics])
    f1s  = np.array([m["f1"]       for m in fold_metrics])

    mean_acc, std_acc = accs.mean(), accs.std()
    mean_f1,  std_f1  = f1s.mean(),  f1s.std()

    print(f"\n{'─'*40}")
    print(f"Results for '{model_name}'  (Table 3 format):")
    print(f"  Accuracy : {mean_acc*100:.0f}% ± {std_acc*100:.0f}%")
    print(f"  F1       : {mean_f1:.2f} ± {std_f1:.2f}")
    print(f"{'─'*40}")

    # Full OOF classification report
    report = sk_classification_report(
        y,
        oof_preds,
        target_names=class_names,
        zero_division=0,
    )

    oof_labels = [id2label[i] for i in oof_preds]

    return {
        "model_name":           model_name,
        "mean_accuracy":        mean_acc,
        "std_accuracy":         std_acc,
        "mean_f1":              mean_f1,
        "std_f1":               std_f1,
        "per_fold_metrics":     fold_metrics,
        "classification_report": report,
        "oof_predictions":      oof_labels,   # aligned with df_work
    }


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genre detection with stratified 5-fold CV. "
            "Reproduces Table 3 of 'Automatic Correction of Writing Anomalies in Hausa Texts'."
        )
    )
    parser.add_argument("--file-path",   required=True,
                        help="Path to TSV file containing the OOD genre dataset.")
    parser.add_argument("--original-col", default=None,
                        help="Column with original (human-edited) text. "
                             "If provided, runs a third model to reproduce the 'Original' row of Table 3.")
    parser.add_argument("--noisy-col",   default="noisy_text",
                        help="Column with noisy text  (default: 'noisy_text').")
    parser.add_argument("--cleaned-col", default="cleaned_text",
                        help="Column with cleaned text (default: 'cleaned_text').")
    parser.add_argument("--label-col",   default="genre",
                        help="Column with genre labels (default: 'genre').")
    parser.add_argument("--model-checkpoint", default="castorini/afriberta_base",
                        help="HuggingFace model checkpoint (default: castorini/afriberta_base, "
                             "matching Section 6.1 of the paper).")
    parser.add_argument("--num-epochs",  type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--folds",       type=int,   default=5,
                        help="Number of CV folds (default: 5, matching the paper).")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  default="./genre_results",
                        help="Root directory for checkpoints and results.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"File not found: {args.file_path}")

    df = pd.read_csv(args.file_path, sep="\t", quoting=3, encoding="utf-8")
    print(f"Loaded {len(df)} rows from '{args.file_path}'")

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []

    def _run(col, label):
        if col is None:
            return
        if col not in df.columns:
            print(f"Warning: column '{col}' not found — skipping '{label}'.")
            return

        result = run_cross_validation(
            df=df,
            text_column=col,
            label_column=args.label_col,
            model_name=label,
            model_checkpoint=args.model_checkpoint,
            output_dir_base=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            n_splits=args.folds,
        )

        if "error" in result:
            print(f"Error for '{label}': {result['error']}")
            return

        print(f"\nClassification Report ({label}):\n{result['classification_report']}")

        # Attach OOF predictions to the dataframe
        pred_col = f"pred_cv_{col}"
        # oof_predictions is aligned to a cleaned, 0-based copy of df.
        # Map back safely: drop the same NaN rows, reset index, assign.
        df_aligned = df.dropna(subset=[col, args.label_col]).reset_index(drop=True)
        df_aligned[pred_col] = result["oof_predictions"]
        # Merge prediction column back into the main df on the original index
        df[pred_col] = df_aligned[pred_col].reindex(df.index)

        summary_rows.append({
            "Data":     label,
            "Accuracy": f"{result['mean_accuracy']*100:.0f}% ± {result['std_accuracy']*100:.0f}%",
            "F1":       f"{result['mean_f1']:.2f} ± {result['std_f1']:.2f}",
        })

    # Run for each text variant to match Table 3 row order
    _run(args.original_col,  "Original")
    _run(args.noisy_col,     "Noisy")
    _run(args.cleaned_col,   "Cleaned")

    # Print summary table to mirror Table 3 layout
    if summary_rows:
        print("\n" + "="*50)
        print("Summary (Table 3 format)")
        print("="*50)
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(args.output_dir, "table3_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to '{summary_path}'")

    # Save full dataframe with OOF predictions
    out_path = args.file_path.replace(".tsv", "_genre_cv_results.tsv")
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    print(f"Full results saved to '{out_path}'")


if __name__ == "__main__":
    main()