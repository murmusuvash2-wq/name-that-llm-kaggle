# Name That LLM — LLM GMC

Repository for the Kaggle **Name That LLM** code competition. The primary inference notebook is `notebooks/llm-gmc.ipynb`, published as `janaiworkspace/llm-gmc`.

## Current execution workflow

1. Update the notebook in GitHub.
2. Open **Actions → Run and submit v2 (guarded) → Run workflow**.
3. The workflow pushes the notebook to Kaggle and records the exact kernel version.
4. It waits for a terminal Kaggle status and fails unless the kernel reaches `COMPLETE`.
5. It downloads `submission.parquet` and validates the required columns, non-empty rows, unique IDs, nulls, and the 18 allowed labels.
6. Leaderboard submission is **opt-in**: leave `submit=false` for a smoke/diagnostic run. Set `submit=true` only when you intentionally want to consume a competition submission.

A green GitHub Actions run now means the Kaggle kernel completed and its output passed structural validation. It does not by itself mean a leaderboard score improved.

## Competition metric and validation

The competition metric is balanced accuracy across 18 models. The test prompts are unseen relative to training prompts, so random row-level validation is not a reliable estimate. Prefer prompt-grouped validation and inspect per-class recall/confusion matrices, not only aggregate accuracy.

The public leaderboard is only a subset of the hidden test set. Avoid choosing weights solely because they improve one public score.

## Current experiment notes

- Current notebook header identifies the v68 experiment.
- The notebook currently attempts a multi-member blend (four-view SVC, transformer/LoRA members, DeBERTa, lexical and typography features, class offsets, and damped Sinkhorn).
- A component's presence in code is not proof it ran successfully or improved the final blend. Review the per-member run reports and OOF contribution before changing weights.
- Keep a known-good incumbent and run one-variable ablations against it. Record full-pipeline prompt-grouped OOF balanced accuracy, per-class recall, public score, runtime, and whether every intended member actually ran.

## Required GitHub secret

In **Settings → Secrets and variables → Actions**, configure:

- `KAGGLE_API_TOKEN`: Kaggle API token. Never commit or paste this token into source files.

## Repository layout

```text
notebooks/llm-gmc.ipynb
.github/workflows/run-and-submit2.yml
.github/workflows/llm-gmc-kaggle.yml
src/
```

Do not commit Kaggle credentials, competition data, or generated submission artifacts.
