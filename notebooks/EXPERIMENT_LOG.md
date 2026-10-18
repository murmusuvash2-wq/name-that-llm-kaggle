# Experiment log

Record every meaningful experiment before uploading a submission.

| Run | Features | Model | Validation split | Balanced accuracy | Public score | Notes |
|---|---|---|---|---:|---:|---|
| 001 | word + char TF-IDF | LinearSVC | prompt-grouped, seed 42 |  |  | baseline |

## Error analysis checklist

- Are errors concentrated in one model class?
- Do errors change when punctuation and whitespace are normalized?
- Does performance collapse on unseen prompt groups?
- Are train/test duplicates or near-duplicates present?
- Is the public score improving for a real reason or because of leaderboard overfitting?
