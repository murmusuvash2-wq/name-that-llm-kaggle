"""Character + word TF-IDF baseline for Name That LLM."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, max_features=180_000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=220_000,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(C=2.0, class_weight="balanced")),
    ])


def prompt_split(df: pd.DataFrame, seed: int = 42):
    if "prompt_id" not in df.columns:
        return train_test_split(df, test_size=0.2, random_state=seed, stratify=df["model"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(df, groups=df["prompt_id"]))
    return df.iloc[train_idx], df.iloc[valid_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/baseline.parquet"))
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Put train.csv and test.csv in {args.data_dir}. "
            "Join the Kaggle competition and download its data first."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_train = {"text", "model"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv must contain {required_train}; found {set(train.columns)}")
    if not {"id", "text"}.issubset(test.columns):
        raise ValueError("test.csv must contain id and text columns")

    tr, va = prompt_split(train)
    validation_model = make_pipeline()
    validation_model.fit(tr["text"].fillna(""), tr["model"])
    pred = validation_model.predict(va["text"].fillna(""))
    print(f"validation accuracy: {accuracy_score(va['model'], pred):.4f}")
    print(f"validation balanced_accuracy: {balanced_accuracy_score(va['model'], pred):.4f}")

    final_model = make_pipeline()
    final_model.fit(train["text"].fillna(""), train["model"])
    test_pred = final_model.predict(test["text"].fillna(""))
    submission = pd.DataFrame({"id": test["id"], "model": test_pred})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(args.output, index=False)
    print(f"wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
