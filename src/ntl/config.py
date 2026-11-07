"""Single source of truth for labels, weights, and run mode.

Weights come from the published 0.668 reference notebook unless a later
measurement overrode them. Do not invent new weights without a k=2 OOF or
hidden LB read.
"""

from __future__ import annotations

RECIPE_VERSION = "v74-fixed"

LABELS = [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
]
K = len(LABELS)
LABEL2I = {l: i for i, l in enumerate(LABELS)}

CONFIG = {
    "submit_mode": True,
    "w_svc": 12.0,
    "w_deberta": 1.0,
    "w_q14v5": 3.0,
    "w_ll8v5": 1.0,
    "w_n14b": 2.5,
    "w_q25_14": 1.0,
    "w_tfidf_lr": 0.25,
    "w_distort_lr": 0.10,
    "admit_floor": 0.40,
    "admit_scale": 0.20,
    "twin": {"thresh": 0.85, "min_size": 2, "max_size": 18},
    "correction": {"thresh": 0.95, "min_size": 8, "max_size": 18, "lam": 0.62},
    "sinkhorn": {"alpha": 0.30, "target_mix": 0.35, "iters": 100},
    "svc_sweep": [1.5, 3.0, 6.0, 12.0],
    "oof_k2_per_prompt": 2,
    "oof_folds": 2,
}
