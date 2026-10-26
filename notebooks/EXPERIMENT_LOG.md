

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
