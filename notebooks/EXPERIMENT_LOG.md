
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
the orthographic shield, the recall calibration. Those cost 0.006 on the LB in v41 and the public
part-two notebook independently lists family gates, a stacker and "one passage per model inside a
topic cluster" as things that were measured and did not help.

**End-token bug, confirmed in the log.** Kaggle's Qwen3-14B-Base ships the chat tokenizer's end
token (151645); the checkpoints were trained with 151643. The log line is
`end token 151643, this tokenizer's is 151645`. v40 and v41 both used `tok.eos_token_id`, so their
14B member was quietly reading passages ending in a token it never saw in training. v42 reads
`cfg["eos_id"]` from the checkpoint's `meta.json`.

Next, to 0.65: the 0.620 -> 0.636/0.639 step in the public notebooks is a ~16-member complementary
stack blended at 15% (stylegram, stylometry LightGBM, UltraBERT, frozen Qwen/Gemma/GLM heads,
ELECTRA, lora24 x3, token-attention x2), then ATLAS + spectral readouts + full-text Qwen3.8-27B NF4
at 0.05-0.10 for 0.641-0.642. All published assets, no new training, mostly plumbing.
