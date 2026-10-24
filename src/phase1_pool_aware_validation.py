"""Phase 1: pool-aware, prompt-grouped validation for Name That LLM.

This is a diagnostic phase. It does not write the production Kaggle notebook.
It compares train-only and train+pool SVM fits under the same prompt-held-out
validation folds, so external-data gains are measured on the distribution that
will actually be used at inference time.

Example:
  python src/phase1_pool_aware_validation.py \
    --train data/raw/train.csv \
    --pool data/raw/pool_v1.parquet \
    --out reports/phase1_pool_validation.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.svm import LinearSVC

VIEWS = [
    ("char", 1, 5, 300_000, 1.0),
    ("word", 1, 2, 220_000, 0.8),
    ("shape", 2, 6, 80_000, 0.5),
]


def shape_text(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def load_pool(path: Path, labels: set[str]) -> pd.DataFrame:
    pool = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"text", "model"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Pool is missing columns: {sorted(missing)}")
    pool = pool.dropna(subset=["text", "model"]).copy()
    pool["text"] = pool["text"].astype(str)
    pool["model"] = pool["model"].astype(str)
    pool = pool[pool["model"].isin(labels)].copy()
    words = pool["text"].str.split().str.len()
    pool = pool[words.between(40, 250)].copy()
    return pool.drop_duplicates("text").reset_index(drop=True)


def make_splitter(train: pd.DataFrame, y: np.ndarray, n_splits: int):
    groups = train["prompt_id"].astype(str) if "prompt_id" in train else np.arange(len(train))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(train, y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        splits = list(splitter.split(train, y, groups))
    return groups, splits


def transform(texts: Iterable[str]):
    return [str(x) for x in texts]


def fit_view_matrices(train_text: list[str], valid_text: list[str]):
    left, right = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        analyzer = "char" if view in {"char", "shape"} else "word"
        prep = shape_text if view == "shape" else (lambda x: x)
        opts = dict(
            analyzer=analyzer,
            ngram_range=(lo, hi),
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=max_features,
            dtype=np.float32,
        )
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vectorizer = TfidfVectorizer(**opts)
        left.append(vectorizer.fit_transform([prep(x) for x in train_text]) * weight)
        right.append(vectorizer.transform([prep(x) for x in valid_text]) * weight)
    return hstack(left, format="csr"), hstack(right, format="csr")


def score_bucketed(y_true, y_pred, texts, labels):
    lengths = pd.Series(texts).str.split().str.len().to_numpy()
    result = {}
    for name, mask in {
        "short_40_60": lengths <= 60,
        "mid_61_120": (lengths >= 61) & (lengths <= 120),
        "mid_121_180": (lengths >= 121) & (lengths <= 180),
        "long_181_plus": lengths >= 181,
    }.items():
        result[name] = (
            float(balanced_accuracy_score(y_true[mask], y_pred[mask]))
            if mask.any() else None
        )
    recalls = recall_score(y_true, y_pred, labels=np.arange(len(labels)), average=None, zero_division=0)
    return {"overall_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "length_buckets": result,
            "min_class_recall": float(recalls.min()),
            "macro_recall": float(recalls.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--pool", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/phase1_pool_validation.json"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--pool-weights", type=float, nargs="+", default=[0.10, 0.20, 0.35, 0.50, 0.85])
    args = ap.parse_args()

    train = pd.read_csv(args.train).dropna(subset=["text", "model"]).copy()
    train["text"] = train["text"].astype(str)
    train["model"] = train["model"].astype(str)
    labels = sorted(train["model"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = train["model"].map(label_to_idx).to_numpy()
    pool = load_pool(args.pool, set(labels)) if args.pool else pd.DataFrame(columns=["text", "model"])
    groups, splits = make_splitter(train, y, args.n_splits)
    candidates = {"train_only": 1.0}
    candidates.update({f"train_plus_pool_weight_{w:g}": w for w in args.pool_weights})
    fold_records = {name: [] for name in candidates}

    for fold, (itr, iva) in enumerate(splits):
        valid = train.iloc[iva]
        base = train.iloc[itr]
        for name, pool_weight in candidates.items():
            if name == "train_only" or pool.empty:
                fit = base
                sample_weight = None
            else:
                fit = pd.concat([base, pool], ignore_index=True)
                sample_weight = np.r_[np.ones(len(base)), np.full(len(pool), pool_weight)]
            X_fit, X_valid = fit_view_matrices(
                transform(fit["text"]), transform(valid["text"])
            )
            model = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=42 + fold)
            model.fit(X_fit, fit["model"].map(label_to_idx).to_numpy(), sample_weight=sample_weight)
            pred = model.predict(X_valid)
            metrics = score_bucketed(
                y[iva], pred, valid["text"].tolist(), labels
            )
            fold_records[name].append({"fold": fold, **metrics, "n_fit": len(fit), "n_pool": len(fit) - len(base)})
            print(f"fold={fold} candidate={name} BA={metrics['overall_balanced_accuracy']:.5f}", flush=True)

    summary = {}
    for name, records in fold_records.items():
        summary[name] = {
            "mean_balanced_accuracy": float(np.mean([r["overall_balanced_accuracy"] for r in records])),
            "std_balanced_accuracy": float(np.std([r["overall_balanced_accuracy"] for r in records])),
            "mean_min_class_recall": float(np.mean([r["min_class_recall"] for r in records])),
            "folds": records,
        }
    result = {
        "phase": "pool-aware prompt-grouped validation",
        "labels": labels,
        "train_rows": len(train),
        "pool_rows_after_filter": len(pool),
        "n_splits": len(splits),
        "group_column": "prompt_id" if "prompt_id" in train else None,
        "candidates": summary,
        "selection_rule": "compare pool-trained candidates against each other; do not compare their CV directly to train-only CV without Public LB confirmation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
