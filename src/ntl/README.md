# `src/ntl`

Shared helpers mirrored from the production Kaggle notebook (`notebooks/llm-gmc.ipynb`).

| Module | Role |
|--------|------|
| `config.py` | `LABELS`, blend weights, `RECIPE_VERSION` |
| `utils.py` | `softmax`, `resolve_eos_id`, `gpu_cleanup`, permutation helpers |
| `trunk.py` | truncated LM load, LoRA merge, meta keep/layers |

The Kaggle scoring path stays **self-contained** in the notebook (no internet, no package install).
Use this package for local unit tests and CI only.

## Permanent fixes (v74)

- `resolve_eos_id`: checkpoint EOS wins over tokenizer (Qwen3 151643 vs 151645)
- `submit_mode`: skip multi-fold SVC OOF on scoring runs
- label permutation guards before v5 heads vote
