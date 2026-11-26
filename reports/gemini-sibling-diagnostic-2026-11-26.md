# Gemini sibling specialist diagnostic — 2026-11-26

The isolated 3-way Gemini specialist completed successfully on Kaggle run `45546937295` using 2,699 official rows and 150 `prompt_id` groups. Protocol: 5-fold `GroupKFold` by `prompt_id`; no hidden labels; predeclared promotion margin `+0.010` balanced accuracy.

The same global char+word SVC baseline scored `0.3037286` balanced accuracy. The best tested specialist gate was threshold `0.10`, scoring `0.3033582`, a delta of `-0.0003704`. Thresholds `0.20`, `0.35`, `0.50`, and `0.70` degraded further to `-0.0044444`, `-0.0137037`, `-0.0218519`, and `-0.0296296`. The specialist improved average recall across Gemini 3.6/3.7/3.8 from `0.3489` to `0.3822` at threshold `0.10`, but harmed other class balance and especially Gemini 3.1 recall.

Decision: **reject for production and do not submit**. The next specialist should use the full v81 blend logits as its gate, avoid touching Gemini 3.1, and focus first on the specific Gemini 3.7 versus 3.8 pair rather than replacing all three Gemini siblings with a standalone specialist.
