# Name That LLM — LLM GMC

Repository for the Kaggle **Name That LLM** competition, using the Kaggle notebook `janaiworkspace/llm-gmc` as the main experiment runner.

## Current workflow

1. Edit the notebook code in `notebooks/llm-gmc.ipynb` and commit it to GitHub.
2. In GitHub, open **Actions → Publish and run LLM GMC on Kaggle → Run workflow**.
3. GitHub Actions pushes the notebook to Kaggle using the `KAGGLE_API_TOKEN` repository secret.
4. The workflow waits for the Kaggle run, downloads its outputs, and stores them as a GitHub Actions artifact.
5. Review the local validation accuracy and balanced accuracy before choosing the next experiment.

The workflow does **not** submit predictions to the Kaggle leaderboard.

## Baseline model

The LLM GMC notebook currently evaluates a word + character TF-IDF feature union with class-balanced `LinearSVC`. It uses a prompt-grouped validation split when `prompt_id` is available; otherwise it uses a stratified random split. The notebook writes `validation_results.json` under `/kaggle/working`.

The printed accuracy is a **local validation score**, not the public leaderboard score.

## Required GitHub secret

In **Settings → Secrets and variables → Actions**, create:

- `KAGGLE_API_TOKEN`: your Kaggle API token. Do not commit or paste the token into source files.

The workflow currently targets the Kaggle notebook reference `janaiworkspace/llm-gmc` and competition slug `name-that-llm`.

## Repository layout

```text
src/baseline.py                         Optional local/GitHub Actions baseline
notebooks/llm-gmc.ipynb                 Main competition-linked notebook
.github/workflows/llm-gmc-kaggle.yml    Push, run, and collect notebook outputs
```

Competition data, API credentials, and generated submissions should not be committed to Git.
