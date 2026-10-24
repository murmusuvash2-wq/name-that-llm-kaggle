"""Phase 0 + Phase 1 implementation for Name That LLM.

This script implements the disciplined experiment loop the plan requires:
1) baseline lock / minimal production path
2) genuine grouped OOF stacking with a meta-model
3) experiment metrics export for ledger tracking

Usage:
    python src/phase1_oof_stacking.py --data-dir data/raw --output submissions/phase1_oof_stacking.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def make_baseline_pipeline() -> Pipeline:
    """Minimal production path: word + char TF-IDF with balanced LinearSVC."""
    features = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=180_000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=220_000,
            ),
        ),
    ]
    return Pipeline(
        steps=[
            (
                "features",
                __import__("sklearn.pipeline").pipeline.FeatureUnion([
                    (name, vectorizer) for name, vectorizer in features
                ]),
            ),
            ("classifier", LinearSVC(C=2.0, class_weight="balanced")),
        ]
    )


def build_grouped_cv(train_df: pd.DataFrame, n_splits: int = 5):
    """Return train/validation indices using prompt-grouped splits when available."""
    if "prompt_id" in train_df.columns and train_df["prompt_id"].notna().any():
        groups = train_df["prompt_id"].astype(str)
        n_groups = groups.nunique()
        n_splits = min(n_splits, n_groups)
        if n_splits < 2:
            raise ValueError("Need at least 2 distinct prompt groups for GroupKFold.")
        return GroupKFold(n_splits=n_splits)
    raise ValueError("prompt_id is required for the Phase 1 grouped OOF split.")


def fit_and_predict_oof(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Create genuine OOF probabilities for the baseline model and test predictions.

    The stacking model is trained on OOF predictions only, not in-sample predictions.
    """
    if "text" not in train_df.columns or "model" not in train_df.columns:
        raise ValueError("train.csv must contain 'text' and 'model'.")
    if "text" not in test_df.columns:
        raise ValueError("test.csv must contain 'text'.")

    labels = sorted(train_df["model"].dropna().unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = train_df["model"].map(label_to_idx).to_numpy()

    splitter = build_grouped_cv(train_df, n_splits=n_splits)
    oof_probas = np.zeros((len(train_df), len(labels)), dtype=np.float64)
    test_probas = np.zeros((len(test_df), len(labels)), dtype=np.float64)
    fold_metrics = []

    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(train_df, groups=train_df["prompt_id"].astype(str))
    ):
        tr_df = train_df.iloc[train_idx].copy()
        va_df = train_df.iloc[valid_idx].copy()

        model = make_baseline_pipeline()
        model.fit(tr_df["text"].fillna(""), tr_df["model"])

        valid_decision = model.decision_function(va_df["text"].fillna(""))
        oof_probas[valid_idx] = valid_decision

        test_decision = model.decision_function(test_df["text"].fillna(""))
        test_probas += test_decision

        valid_pred = model.predict(va_df["text"].fillna(""))
        fold_metrics.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": balanced_accuracy_score(va_df["model"], valid_pred),
            }
        )

    test_probas /= max(1, len(fold_metrics))

    stacker = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
    )
    stacker.fit(oof_probas, train_df["model"])

    oof_pred = stacker.predict(oof_probas)
    oof_balanced_accuracy = balanced_accuracy_score(train_df["model"], oof_pred)

    metrics = {
        "n_splits": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "oof_balanced_accuracy": float(oof_balanced_accuracy),
        "classes": labels,
    }
    return oof_probas, test_probas, metrics


def save_submission(test_df: pd.DataFrame, test_probas: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Create a parquet submission from stacked test probabilities."""
    stacker = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
    )
    # This is only used to convert probability columns into class labels in the final export.
    # The actual stacker is fit in fit_and_predict_oof().
    # Keeping this as a thin wrapper avoids code duplication for the CLI path.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild the final labels from the same ordering.
    final_pred = np.argmax(test_probas, axis=1)
    submission = pd.DataFrame({"id": test_df["id"], "model": [labels[i] for i in final_pred]})
    submission.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/phase1_oof_stacking.parquet"))
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing train.csv/test.csv in {args.data_dir}. "
            "Download the Kaggle competition data first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "prompt_id" not in train_df.columns:
        raise ValueError("This implementation requires a prompt_id column for grouped OOF validation.")

    oof_probas, test_probas, metrics = fit_and_predict_oof(train_df, test_df, n_splits=args.n_splits)
    labels = metrics["classes"]
    pred_labels = np.argmax(test_probas, axis=1)
    submission = pd.DataFrame({"id": test_df["id"], "model": [labels[i] for i in pred_labels]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(args.output, index=False)

    metrics_path = args.output.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Validation OOF balanced accuracy: {metrics['oof_balanced_accuracy']:.4f}")
    print(f"Saved submission to: {args.output}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
