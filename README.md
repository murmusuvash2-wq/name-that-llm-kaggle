# Name That LLM — Kaggle Starter

Reproducible baseline for Kaggle's **Name That LLM** competition.

The goal is to classify an unlabeled passage into one of the 18 language-model classes. This repository deliberately keeps competition data and generated submissions out of Git.

## Repository layout

```text
src/baseline.py       Train a character + word TF-IDF linear SVM and create submission.parquet
requirements.txt      Python dependencies
.env.example          Optional local configuration
notebooks/            Experiment notebooks
models/               Local model artifacts (ignored by Git)
data/raw/             Kaggle input files (ignored by Git)
submissions/          Local submissions (ignored by Git)
```

## Kaggle-to-GitHub flow

1. Create or sign in to one Kaggle account.
2. Open the [Name That LLM competition](https://www.kaggle.com/competitions/name-that-llm), click **Join Competition**, and accept the rules.
3. Download the competition data from Kaggle, or attach the competition dataset to a Kaggle Notebook.
4. For local work, place `train.csv`, `test.csv`, and `sample_submission.parquet` under `data/raw/`.
5. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

6. Run the baseline:

   ```bash
   python src/baseline.py --data-dir data/raw --output submissions/baseline.parquet
   ```

7. Inspect the generated file. It must contain one row per test `id` and a `model` column.
8. Upload the file on Kaggle's **Submit Predictions** page. Never commit the data, Kaggle API token, or submission files.
9. Record the experiment and public score in `notebooks/` or a separate experiment log, then commit only code and notes to GitHub.

## Kaggle Notebook flow

If you do not want to download data locally, create a Kaggle Notebook, attach `name-that-llm` as an input, copy the code from `src/baseline.py`, and change the data path to Kaggle's mounted input path. Turn internet off for a reproducible submission run and attach any pretrained model as a Kaggle Dataset instead of downloading it during execution.

## Baseline method

The baseline combines word n-grams and character n-grams using `TfidfVectorizer`, then trains a class-balanced `LinearSVC`. Character features help capture punctuation, spelling, formatting, and stylistic habits; word features capture phrase-level tendencies. The script prints a prompt-grouped validation score when `prompt_id` is available, then retrains on all labelled data and writes the submission.

## Important rules and hygiene

- Use one Kaggle account only.
- Do not commit competition data or private team code.
- Do not share private code outside your Kaggle team.
- Check the current official rules for team size, submission limits, deadlines, eligibility, and prize/tax requirements.
- Treat the public leaderboard as a weak signal; optimize against robust local validation, especially prompt-wise splits.

## Next experiments

1. Compare character-only, word-only, and combined models.
2. Add repeated prompt-grouped cross-validation and confidence analysis.
3. Try calibrated probabilities and blend independent models.
4. Evaluate sentence or language-model embeddings as a second branch.
5. Audit duplicates, formatting artifacts, topic leakage, and prompt distribution shifts.

## GitHub Actions with a Kaggle secret

The repository includes a manual workflow at `.github/workflows/kaggle-test.yml`. Add a repository secret named `KAGGLE_API_TOKEN` under **Settings → Secrets and variables → Actions → New repository secret**. Then open **Actions → Kaggle test and submit → Run workflow**.

The workflow always downloads the data, trains the baseline, and stores `submission.parquet` as a GitHub Actions artifact. Keep the `submit` input set to `false` for the first test. After checking the artifact, run it again with `submit=true` to upload the file to Kaggle. The token is read only from GitHub Secrets and is never written into the repository.
