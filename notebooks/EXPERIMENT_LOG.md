

## Version 40 orthographic-shield-off ablation

- Source baseline: Kaggle Version 40 / Git `28a78e6`
- Only change: removed the empirical orthographic shield block
- Code commit: `3646be6`
- Kaggle workflow: `40161678239`
- Execution: successful
- Interactive predictions: unchanged on all 3 rows
- Public LB: pending manual Kaggle submission; compare directly against `0.582`

## Version 46 Public LB result

Version 46 was the isolated orthographic-shield-off ablation from Version 40. Its confirmed Public LB score was **0.580**, below the Version 40 shield-on baseline of **0.582**. The ablation is rejected; the exact Version 40 shield-on notebook is restored for the next experiment.

## Pasted-text blend-weight candidate

The pasted-text proposal was tested in isolation from Version 40 with `W_SVC=0.720`, `lora8b_v1.w=0.305`, and `lora14b_v1.w=0.580`. The Kaggle execution succeeded in workflow `40192911961`; `clean_extra=0`, the shield remained enabled, and all three interactive predictions were unchanged from Version 40. Public LB score is pending manual submission. The candidate is not retained as production until it beats `0.582`; production is restored to the exact Version 40 baseline.

### Weight candidate Public LB decision

The isolated pasted-text blend-weight candidate (`W_SVC=0.720`, 8B=`0.305`, 14B=`0.580`) received Public LB **0.582**, exactly equal to Version 40. It produced no measurable improvement and is rejected as a production change. The exact Version 40 weights remain active.

## Combined pasted-text Step 1 candidate

The text-prescribed combined candidate used the Version 40 baseline with weight reallocation (`W_SVC=0.720`, 8B=`0.305`, 14B=`0.580`), no extra clean rows (`clean_extra=0`), and the orthographic shield disabled. Kaggle workflow `40201881454` completed successfully; the submission artifact is valid. Public LB score is pending manual submission. Production is restored to the exact Version 40 shield-on baseline until the candidate beats `0.582`.

### Combined Step 1 Public LB decision

The combined pasted-text Step 1 candidate (weight reallocation plus orthographic shield disabled; extra rows already absent) received Public LB **0.582**, exactly equal to Version 40. It produced no improvement and is rejected. The exact Version 40 shield-on configuration remains the production baseline.

## Research Phase R1: early-layer auxiliary readout

Public evidence supported testing early Qwen layers and a complementary readout, so a gated early-half 8B auxiliary LinearSVC was implemented with grouped OOF selection. Workflow `40212395409` successfully pushed the notebook but remained stuck in Kaggle's wait stage for more than eight hours and produced no artifact or score. This is an execution timeout, not a model result. The candidate is not retained; the exact Version 40 baseline is restored pending a lighter implementation.

## Validation-repaired clean baseline candidate

Commit `c76d763` removes the invalid mixed OOF/in-sample blend search and disables twin/grouping, embedding, label calibration, sibling, family, and hierarchical specialists. It retains fixed Version 40 weights, n-gram SVC, Qwen 8B/14B plain heads, damped balancing, and the proven orthographic shield. Workflow `40245690732` completed successfully in 8m14s; the submission artifact is valid. Interactive output changed only `005f5a9a6139` from `gpt-5.6-terra` to `gpt-5.6-sol`. Public LB score is pending manual submission.

### Clean baseline Public LB decision
The validation-repaired clean baseline received Public LB **0.559**, materially below Version 40's **0.582**, so it is rejected. The ablation was confounded: it removed both the twin/cavity path and the complete label-calibration/sibling block at once, while also bypassing the existing blend-selection path. Family and hierarchical gates were not applied on the 3-row public test, so they cannot explain this regression. Exact Version 40 was restored in commit `10b6ca7`. Next tests must isolate one mechanism at a time.

### Clean labelled-answer pool validation
Workflow `40256587266` completed successfully. The private notebook audited the attached `pool_v1.parquet` as 34,418 clean labelled rows covering all 18 models, then ran 3-fold prompt-grouped validation with compact char/word/shape views. Train-only scored overall BA **0.34712**, GPT-sibling BA **0.32933**, Gemini-sibling BA **0.43333**, family BA **0.58215**. Adding the clean pool at sample weight **0.25** scored overall BA **0.44901**, GPT-sibling **0.40000**, Gemini-sibling **0.60333**, family **0.60618**, improving all three folds. Weight 0.50 was weaker at overall BA **0.43677**. This confirms additional labelled answers help sibling identification, but the experiment is SVC-only and is not directly comparable to Version 40's full Qwen ensemble. Production remains unchanged; the existing Version 40 SVC already consumes `pool_v1` at full weight, so the next test is a controlled SVC/pool-weight ablation rather than blindly adding duplicate data.

### Exact production-style pool-weight result
Workflow `40260944290` completed successfully. The private diagnostic used the production SVC views and 34,940 pool rows. Current pool weight 1.00 scored overall BA **0.43530**; weight 0.25 scored **0.45568**, improving all three folds. GPT sibling improved **0.3880 → 0.4027**, Gemini sibling **0.5767 → 0.6017**, family BA **0.5989 → 0.6135**. Calibrated expected full-ensemble Public LB range is **0.5851–0.5913**, center **0.5882**, using Version 40 LB 0.582 and 25–75% transfer of the SVC uplift. This is the first positive report strong enough to justify an isolated LLM GMC candidate. Candidate change: only production SVC pool sample weight 1.00 → 0.25; Version 40 remains the rollback reference.

### Pool weight 0.25 LLM GMC candidate decision
Workflow `40264221900` ran the isolated candidate successfully. Although private production-style validation predicted an expected Public LB range of 0.5851–0.5913, the candidate's 3-row `submission.parquet` was byte-equivalent in predictions to Version 40: `claude-opus-4-8`, `gpt-5.6-terra`, `gemini-3.1-pro-preview`. Therefore its actual Public LB must be unchanged at approximately **0.582**, with no possible score uplift. The candidate is rejected without a redundant Public LB submission, and exact Version 40 is restored.

### Disagreement-first audit and pool weight 0.50 decision
Corrected private audit workflow `40279972472` showed genuine held-out improvement: pool weight 0.25 changed 468 validation rows across 3 folds, changed-row accuracy improved 19.69% → 31.19%, and net correct changed-row gain was +55; weight 0.50 changed 224 rows, changed-row accuracy improved 16.44% → 35.11%, net gain +42. However, the full LLM GMC pool-weight 0.50 candidate workflow `40282516167` produced **0 changed rows** on the 3-row Kaggle test compared with Version 40. Predictions remained `claude-opus-4-8`, `gpt-5.6-terra`, and `gemini-3.1-pro-preview`. Therefore no Public LB uplift was possible; candidate rejected and exact Version 40 restored.

### Kaggle hidden-rerun correction
Kaggle's official Name That LLM overview confirms that interactive notebooks use a single example row, while scoring reruns use the full hidden test set of approximately 4,000 passages. Kaggle's Code Competition debugging guidance independently confirms that submissions are privately re-executed end-to-end against a hidden dataset before scoring. Therefore visible-run `test.csv` prediction-difference checks are not sufficient for Public LB inference. The diagnostic final-logit sweep run `40425655285`, which inspected only the interactive dummy test, was cancelled as non-decisive. Exact Version 40 production notebook is restored; future threshold/blend experiments must be evaluated via grouped OOF and then a real competition submission.

### Held-out OOF orthographic shield sweep
Workflow `40429995010` successfully tested factors 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, and 1.00 on 3 prompt-held-out folds using production-style SVC train+pool. Every factor below 1.0 changed only 6/2,699 rows total and produced the same result: mean BA **0.435299 → 0.434928** (-0.000370), changed-row accuracy **16.7% → 0%**, net correct change **-1**. Factor 1.0 was identical to baseline with 0 changed rows. Shield factor changes are rejected; no production notebook change. The next evidence-backed candidate remains pool weight 0.50, whose disagreement audit improved all folds with +42 net correct changed-row gain and should be evaluated only through a real hidden-test competition submission.

## Version 41: honest OOF harness, pool-trained specialists, OOF-selected switches

- Code commit: notebook `notebooks/llm-gmc.ipynb` (v41), Kaggle kernel version **58**
- Kaggle workflow: `40444002320` (push + interactive run + outputs), interactive run 34m
- Submission workflow: `40447445344` (submit-only, kernel v58)
- Public LB: **0.579**, against 0.585 for the previous best. Rejected.

What changed, and why (all three were measurement bugs, not model bugs):

1. The fold SVC was fitted on the 2,699 host rows only while production fits train + 34,940 pool
   rows with sample weights, so the OOF number described a different model than the submission.
   Folds now use the production row set (fold train + pool + extra).
2. The blend-weight grid ran on `q_train`, which comes from LoRA heads fine-tuned on those same
   train rows. That is in-sample and pulls the blend towards the LoRAs. The reference weights
   (1.021 / 0.226 / 0.429) were frozen and never re-fitted.
3. Candidates were being accepted or rejected on the 3-row interactive test, which cannot show a
   score change (balancing is skipped below 50 rows). Switches are now chosen on prompt-grouped
   OOF balanced accuracy with a 0.002 acceptance margin.

Mechanisms added: sibling-pair, family and within-family heads fitted on train + pool (37k rows)
instead of train only; final-logit prompt centering with a cluster-tightness guard; one-vs-rest
heads for the eight weakest classes (rejected on OOF, kept behind a switch).

Verified OOF (3 folds, grouped by prompt, balanced accuracy):

| stage | BA |
|---|---|
| v40 recipe | 0.45637 |
| + pool-trained sibling/family specialists | 0.45970 (accepted) |
| + one-vs-rest worst-class heads | 0.45527 (rejected) |
| + final-logit prompt centering 0.15 | 0.46637 (accepted) |
| + final-logit prompt centering 0.30 | 0.47526 (accepted) |
| no marginal balancing | 0.47011 (balancing kept) |
| **chosen** spec=True, ovr=False, lam=0.30, alpha=0.50, shield=hard, calib=0.10 | **0.47526** |

OOF 0.45637 -> 0.47526 (+0.0189); 16 of 18 classes flat or better. **The LB disagreed: 0.579.**
The OOF was an SVC-stack number (the LoRA train probabilities are in-sample), so it could rank
post-processing that acts on the SVC alone but not anything that interacts with the LoRA blend.
The two mechanisms that moved OOF most, final-logit centering and the family gates, are also the
two the public part-two notebook lists as measured-and-not-helpful.

## Version 42: public part-two reproduction — Public LB 0.620

- Code: `notebooks/llm-gmc.ipynb` (v42; v41 kept as `notebooks/llm-gmc-v41.ipynb`)
- Kaggle kernel version **60**, workflow `40456972977`
- **Public LB 0.620** (previous best 0.585, v41 0.579) — **+0.035**
- Interactive run 929 s, scoring rerun ~41 min
- SVC out-of-fold (2 folds, prompt-grouped, pool_v4): **0.4712**, against 0.473 reported by the
  reference notebook, so the port is faithful

What the notebook now is: `pool_v4.parquet` (75,783 passages) n-gram SVC at weight 1.0,
`lora_qwen3_14b` at 0.4, `lora_qwen25_14b` at 0.2, `deberta_v3_large` at 0.25, centred twin at 0.1
per LoRA on clusters of 2-18, Sinkhorn 300 iterations. Assets: `nishantdahal/ntl-handwriting-blend`,
bases `qwen-lm/qwen-3/transformers/14b-base/1` and `qwen-lm/qwen2.5/transformers/14b/1`.

Removed from v41: final-logit prompt centering, sibling-pair heads, family and within-family gates,
the orthographic shield, the recall calibration. Those cost 0.006 on the LB in v41, and the public
part-two notebook independently lists family gates, a stacker and "one passage per model inside a
topic cluster" as things that were measured and did not help.

**End-token bug, confirmed in the log.** Kaggle's Qwen3-14B-Base ships the chat tokenizer's end
token (151645); the checkpoints were trained with 151643. The log line is
`end token 151643, this tokenizer's is 151645`. v40 and v41 both used `tok.eos_token_id`, so their
14B member was quietly reading passages that ended in a token it never saw in training. v42 reads
`cfg["eos_id"]` from the checkpoint's `meta.json`.

Next, to 0.65: the 0.620 -> 0.636/0.639 step in the public notebooks is a ~16-member complementary
stack blended at 15% (stylegram, stylometry LightGBM, UltraBERT, frozen Qwen/Gemma/GLM heads,
ELECTRA, lora24 x3, token-attention x2), then ATLAS + spectral readouts + full-text Qwen3.8-27B NF4
at 0.05-0.10 for 0.641-0.642. All published assets, no new training, mostly plumbing.

## Version 43: the attached "0.700 plan", code-only items — Public LB 0.599

- Kaggle kernel version **61**, workflow `40466767606`, interactive run ~78 min
- **Public LB 0.599**, against 0.620 for v42. **Rejected; production restored to v42.**
- Extra pool loaded: **62,375** usable passages from `illiadolenko/ntl-merged-corpus-v1`
  (SVC fitted on 2,699 + 75,783 + 62,375 = 140,857 rows)

Four of the plan's seven items were implemented, the other three are not implementable in an
inference-only notebook:

1. Hungarian assignment (`linear_sum_assignment` on the log probabilities inside each topic
   cluster, forcing distinct predicted classes) — implemented.
2. Bigger pool (any attached parquet/CSV with a text and a model column, 30-300 words, deduped
   against train and pool_v4) — implemented, 62,375 rows.
3. Test-time augmentation on the SVC (original text plus a punctuation-spacing variant) — implemented.
4. Four hardcoded EDA rules (curly quotes veto Claude/GLM x0.01, em dash vetoes Gemini x0.05,
   a python code fence boosts the GPT family x1.2, a bullet boosts Claude x1.1) — implemented,
   and applied before the assignment so a hard one-hot assignment cannot neutralise them.

Not implemented: a Llama-3-70B / Gemma-2-27B / Qwen3-32B LoRA needs a published checkpoint to
attach and none exists in this competition's public assets; a LightGBM stacker needs out-of-fold
predictions for every member, which the LoRAs cannot supply because their stage two saw the
training passages.

Reading: the run cannot separate the four mechanisms, because the interactive report skips the
assignment and the rules below 50 rows (`hungarian_rows: 0`). What it does establish is that the
whole package is 0.021 worse than the untouched v42, which is consistent with the public part-two
notebook's own finding that "at most one passage per model inside a topic cluster" is better in
simulation and worse on the leaderboard.
