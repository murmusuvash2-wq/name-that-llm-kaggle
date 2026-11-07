"""Name That LLM — inference utilities.

Production path is the Kaggle notebook ``notebooks/llm-gmc.ipynb``.
This package mirrors the fixed helpers for local tests and CI.
"""

from .config import CONFIG, LABELS, LABEL2I, K, RECIPE_VERSION
from .utils import softmax, find_input, gpu_cleanup, balanced_accuracy

__all__ = [
    "CONFIG",
    "LABELS",
    "LABEL2I",
    "K",
    "RECIPE_VERSION",
    "softmax",
    "find_input",
    "gpu_cleanup",
    "balanced_accuracy",
]
