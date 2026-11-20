# Name That LLM research notes — 2026-11-20

- Competition data page: https://www.kaggle.com/competitions/name-that-llm/data
  - Official train.csv: 2,699 passages from 150 prompts; 18 labels.
  - Interactive test has 3 example rows; hidden rerun is about 4,000 passages; test prompts do not appear in train.
- Current leaderboard: https://www.kaggle.com/competitions/name-that-llm/leaderboard
  - Public leaderboard uses approximately 40% of test data; final standings use the other 60%.
  - At research time, public leader score was 0.776.
- Public extended response corpus: https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset
  - Current card: 15,226 total passages, 5,070 clean passages, 810 prompts, all 18 labels; no competition passage or prompt included.
  - Dataset card notes daily updates during generation and warns about provider/API terms; larger data is not automatically better.
- Qwen checkpoint/public pool: https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora
  - Search result identifies about 34,940 cleaned public passages plus two-stage LoRA checkpoints.
- Thin-prompt centering experiment: https://www.kaggle.com/competitions/name-that-llm/discussion/752708
  - Blend of Qwen3-8B + DeBERTa-v3-large + 4-view SVC reported uncentered public LB about 0.595 and light centering about 0.603; heavy centering degraded to about 0.590.
  - OOF simulations with m=2/3/6/18 showed light centering is safer; heavy centering hurts thin groups.
- Model comparison discussion: https://www.kaggle.com/competitions/name-that-llm/discussion/759103
  - Reported non-LLM/encoder baselines: TF-IDF+LR ~0.32 LB; external+class weights ~0.40; stylometry stack ~0.43; DeBERTa-v3-large ~0.50; ModernBERT-large ~0.55; UltraBERT-380M on 241k external rows ~0.60 OOF; equal encoder+TF-IDF blend ~0.625 OOF / 0.616 LB.
  - Family accuracy reported 87–99%; within-family version accuracy 67–76%; raw punctuation should be retained.
- DeBERTa model input verified: https://www.kaggle.com/models/illuminatep/deberta-v3-plain/transformers/base
  - Official Microsoft DeBERTa-v3 base/large packaging; verified base variation v1 is 183.8M params, hidden size 768, 12 layers.

Our confirmed submission history:
- Original TF-IDF/stylometry public score: 0.334.
- Latest confirmed submitted augmentation/no-twin version: 0.561.
- 0.592/0.603 figures above are public reference experiments, not our confirmed scores.
