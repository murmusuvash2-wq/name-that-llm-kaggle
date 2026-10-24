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

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def make_baseline_pipeline() -> Pipeline:
    """Minimal production path: word + char TF-IDF with balanced LinearSVC."""
    features = FeatureUnion(
        [
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
    )
    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", LinearSVC(C=2.0, class_weight="balanced")),
        ]
    )


def get_splitter(train_df: pd.DataFrame, n_splits: int):
    """Return a train/validation splitter and the associated groups if available."""
    if "prompt_id" in train_df.columns and train_df["prompt_id"].notna().any():
        groups = train_df["prompt_id"].astype(str)
        n_valid_groups = groups.nunique()
        n_folds = min(n_splits, n_valid_groups)
        if n_folds < 2:
            raise ValueError("Need at least 2 distinct prompt groups for grouped CV.")
        return GroupKFold(n_splits=n_folds), groups

    y = train_df["model"].astype(str)
    n_folds = min(n_splits, y.value_counts().min())
    if n_folds < 2:
        raise ValueError("Need at least 2 samples per class for stratified CV.")
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42), None


def fit_and_predict_oof(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
):
    """Create genuine OOF scores and a stacked test prediction.

    This follows the corrected Phase 1 requirement: train on one fold, score the held-out fold,
    then fit the meta-model on OOF outputs only.
    """
    required_cols = {"text", "model"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError("train.csv must contain 'text' and 'model'.")
    if "text" not in test_df.columns:
        raise ValueError("test.csv must contain 'text'.")

    labels = sorted(train_df["model"].dropna().unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_idx = train_df["model"].map(label_to_idx).to_numpy()

    splitter, groups = get_splitter(train_df, n_splits)
    oof_scores = np.zeros((len(train_df), len(labels)), dtype=np.float64)
    test_scores = np.zeros((len(test_df), len(labels)), dtype=np.float64)
    fold_metrics = []

    if groups is None:
        split_iterator = splitter.split(train_df, y_idx)
    else:
        split_iterator = splitter.split(train_df, groups=groups)

    for fold_idx, indexes in enumerate(split_iterator):
        train_idx, valid_idx = indexes
        tr_df = train_df.iloc[train_idx].copy()
        va_df = train_df.iloc[valid_idx].copy()

        model = make_baseline_pipeline()
        model.fit(tr_df["text"].fillna(""), tr_df["model"])

        valid_decision = model.decision_function(va_df["text"].fillna(""))
        oof_scores[valid_idx] = valid_decision

        fold_test_decision = model.decision_function(test_df["text"].fillna(""))
        test_scores += fold_test_decision

        valid_pred = model.predict(va_df["text"].fillna(""))
        fold_metrics.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": float(balanced_accuracy_score(va_df["model"], valid_pred)),
            }
        )

    test_scores /= max(1, len(fold_metrics))

    stacker = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        multi_class="auto",
        solver="lbfgs",
    )
    stacker.fit(oof_scores, train_df["model"])

    oof_pred = stacker.predict(oof_scores)
    oof_balanced_accuracy = balanced_accuracy_score(train_df["model"], oof_pred)

    meta = {
        "n_splits": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "oof_balanced_accuracy": float(oof_balanced_accuracy),
        "classes": labels,
        "splitter": "grouped" if groups is not None else "stratified",
    }
    return stacker, oof_scores, test_scores, meta


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
            f"Missing train.csv/test.csv in {args.data_dir}. Download the Kaggle competition data first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    stacker, _, test_scores, meta = fit_and_predict_oof(train_df, test_df, n_splits=args.n_splits)
    labels = meta["classes"]
    final_pred = stacker.predict(test_scores)

    submission = pd.DataFrame({"id": test_df["id"], "model": final_pred})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(args.output, index=False)

    metrics_path = args.output.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"OOF balanced accuracy: {meta['oof_balanced_accuracy']:.4f}")
    print(f"Saved submission to: {args.output}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
