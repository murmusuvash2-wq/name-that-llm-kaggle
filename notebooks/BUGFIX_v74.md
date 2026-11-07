# v74 permanent bugfix

## Fixes
1. **SUBMIT_MODE** (default True) — skips 2-fold SVC OOF (~45 min CPU), k=2 diagnostic when OOF absent, typography holdout.
2. **resolve_eos_id** — always prefer checkpoint `meta.cfg.eos_id` over tokenizer (Qwen3 151643 vs 151645).
3. **_q14_eos** — reads meta next to adapter; falls back to 151643.
4. **REPORT["recipe"]** — was stale "v53..."; now `RECIPE_VERSION = "v74-fixed"`.
5. **gpu_cleanup** helper added for consistent VRAM release.

## Layout
- `notebooks/llm-gmc.ipynb` — Kaggle production notebook (self-contained).
- `src/ntl/` — mirrored helpers for local/CI (`config`, `utils`, `trunk`).

## Not changed
Blend weights, member set, Sinkhorn, topic correction — score recipe unchanged; this is correctness + runtime hygiene only.
