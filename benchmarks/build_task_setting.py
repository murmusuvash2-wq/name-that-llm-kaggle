"""Build a task that varies the *setting* instead of the volume.

The pool's `setting` column has 282 values (default, think_low, thinking_budget=0,
enable_thinking=false), but it is unbalanced: not every model has every setting. That matters because
the same model writes differently under different settings, so a single classifier is asked to fit one
class with several styles — which is exactly what blurs sibling pairs.

The Model Proxy exposes a `reasoning` knob on `llm.prompt(...)`, so one prompt can be answered at
several reasoning levels. That gives balanced setting-varied data, which the pool does not have, and
which nothing in the public notebooks uses.

Usage: python build_task_setting.py <prompts.csv> <out.py> [n_prompts] [levels]
"""
import re
import sys

import pandas as pd

csv_path, out_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
levels = (sys.argv[4] if len(sys.argv) > 4 else "none,low,high").split(",")

df = pd.read_csv(csv_path)
key = next((c for c in ("prompt", "text", "question") if c in df.columns), None)
if key is None:
    raise SystemExit(f"no prompt column in {list(df.columns)}")

INSTRUCTION = ("\n\nAnswer in a single self-contained paragraph of about 150 words. "
               "No preamble, no bullet points, no headings, no markdown.")
raw = [str(p).strip() for p in df[key].dropna().tolist() if str(p).strip()][:n]
prompts = [p if p.rstrip().endswith(INSTRUCTION.strip()) else p.rstrip() + INSTRUCTION for p in raw]
print(f"{csv_path}: using column {key}, {len(prompts)} prompts x {len(levels)} settings")

cases = "".join("    (%r, %r),\n" % (p, lv) for p in prompts for lv in levels)

src = '''"""One prompt, several reasoning settings, through the Model Proxy.

Generated with build_task_setting.py. The point is not more passages — the pool is 228k and saturated —
it is *balanced setting variation* for the weak sibling classes, which the pool has unevenly and which
no public notebook uses.
"""
import json

import kaggle_benchmarks as kbench

CASES = [
%s]


@kbench.task(name="NTL setting varied", description="One prompt answered at several reasoning levels.")
def generate(llm) -> None:
    rows = []
    for prompt, level in CASES:
        try:
            kwargs = {} if level == "default" else {"reasoning": level}
            text = str(llm.prompt(prompt, **kwargs)).strip()
        except Exception as e:
            print(f"skip {level}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append({"prompt": prompt, "setting": level, "text": text})
        print(f"[{level}] {len(text.split())} words", flush=True)
    kbench.assertions.assert_true(len(rows) > 0, expectation="at least one passage was generated")
    with open("passages.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\\n")


generate.run(kbench.llm)
''' % cases

open(out_path, "w").write(src)
print(f"wrote {out_path}")
