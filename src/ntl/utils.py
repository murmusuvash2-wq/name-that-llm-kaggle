"""Shared helpers — paths, metrics, GPU hygiene."""

from __future__ import annotations

import gc
import glob
from typing import Iterable

import numpy as np
import torch


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def find_input(pattern: str, root: str = "/kaggle/input") -> str:
    hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"missing input: {pattern} under {root}")
    return hits[0]


def gpu_cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def resolve_eos_id(cfg: dict, tokenizer) -> int:
    """Prefer checkpoint eos_id; log when tokenizer disagrees (Qwen3 151643 vs 151645)."""
    if "eos_id" in cfg and cfg["eos_id"] is not None:
        eos = int(cfg["eos_id"])
        tok_eos = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        if tok_eos is not None and eos != tok_eos:
            print(f"[eos] using checkpoint eos_id={eos} (tokenizer has {tok_eos})", flush=True)
        return eos
    eos = int(tokenizer.eos_token_id)
    print(f"[eos] WARNING: meta has no eos_id — falling back to tokenizer {eos}", flush=True)
    return eos


def head_permutation(logits: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    pred = logits.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm.argmax(1)


def assert_labels(meta_labels: Iterable[str], labels: list[str], where: str) -> None:
    meta_labels = list(meta_labels)
    if meta_labels != labels:
        raise AssertionError(
            f"{where}: label order mismatch.\n  meta: {meta_labels}\n  ours: {labels}"
        )
