# Name That LLM: Production Architecture

## Production mode

`baseline_qwen_svc_v1` is the only production mode. It is deliberately conservative and reproducible:

1. Load and validate the official `train.csv` and `test.csv` schema.
2. Load `pool_v1.parquet`, validate its labels, and train the four-view SVC on official train plus this fixed pool.
3. Load the published Qwen3-8B and Qwen3-14B two-stage LoRA checkpoints and validate their label order against the single canonical `LABELS` list.
4. Add SVC, Qwen8B, and Qwen14B log-probabilities using fixed, documented weights.
5. Apply Sinkhorn class balancing only on a real hidden-sized test set (`len(test) >= 50`).
6. Validate IDs, labels, row count, finite probabilities, and parquet output before writing `submission.parquet`.

## Feature gates

The following modules are disabled in production and must be evaluated as separate, named experiments before promotion:

- `ENABLE_EXTERNAL_AUGMENTATION = False`
- `ENABLE_LABEL_CALIBRATION = False`
- `ENABLE_SIBLING_SPECIALIST = False`
- `ENABLE_TWIN = False`
- `ENABLE_SINKHORN = True`

This prevents a research change from silently changing the submitted baseline.

## Data contracts

- Official train columns: `id`, `prompt_id`, `text`, `model`.
- Official test columns: `id`, `text`.
- Every official ID must be non-null and unique.
- Every train label must be one of the 18 canonical labels.
- `pool_v1.parquet` must contain `text` and `model`; every pool label must be canonical.
- Every model output is a finite `(n_test, 18)` probability matrix whose rows sum to one.
- The final parquet must contain exactly the test IDs once each and only canonical labels.

## Canonical models

- Four-view `LinearSVC`: raw character, word/punctuation, text-shape, and function-word views.
- Qwen3-8B two-stage LoRA checkpoint.
- Qwen3-14B two-stage LoRA checkpoint.

No external extended dataset is attached to the production workflow. No unvalidated calibration or pairwise specialist is in the production path.

## Promotion rule

A research module can be promoted only when:

1. It has a separate notebook or feature flag.
2. It passes syntax, schema, finite-value, and submission-contract tests.
3. It improves a prompt-grouped validation protocol without using test labels.
4. Its effect is measured independently from other changes.
5. Its Kaggle run completes successfully before any submission is considered.
