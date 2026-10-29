"""Build the Benchmarks task file from the real public prompt bank.

The prompts come from seyitkaangunes/ntl-extra-api-passages (prompts.csv, g0000-g0999), the bank that
generated a large share of the public pool. Using the host's own questions instead of invented ones
means the generated passages sit in the distribution the competition actually scores.

Usage: python build_task.py <prompts.csv> <out.py> [n_prompts]
"""
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 250

df = pd.read_csv(src)
cols = {c.lower(): c for c in df.columns}
key = next((cols[k] for k in ("prompt", "question", "instruction", "text") if k in cols), None)
if key is None:
    raise SystemExit(f"no prompt column in {list(df.columns)}")
raw = [p for p in df[key].dropna().astype(str).tolist() if p.strip()][:n]

# The real bank asks questions but never constrains the answer, so a model writes 1,000+ words while
# the host's passages are 40-250. Measured on the first run: median 1,062 words, 0 of 226 inside the
# range. The question stays the same; only the output shape is pinned to the host's.
INSTRUCTION = ("\n\nAnswer in a single self-contained paragraph of about 150 words. "
               "No preamble, no bullet points, no headings, no markdown.")
prompts = [p if p.rstrip().endswith(INSTRUCTION.strip()) else p.rstrip() + INSTRUCTION for p in raw]
print(f"{src}: using column {key}, {len(prompts)} prompts")
print("first:", prompts[0][:160])

HEAD = '''"""Benchmark task: generate passages on the real public prompt bank.

Each model answers the same real prompts, so the output reproduces the host's structure: one question,
several models. That pairing is the only signal that separates sibling models.

Run server-side (`kaggle benchmarks tasks run -m <slug>`) so the full model catalogue is available;
the local proxy token only reaches a curated subset.
"""
import json

import kaggle_benchmarks as kbench

PROMPTS = [
'''
TAIL = ''']


@kbench.task(name="NTL real prompt generation",
             description="One 40-250 word paragraph per real competition-bank prompt.")
def generate(llm) -> None:
    for i, prompt in enumerate(PROMPTS):
        text = str(llm.prompt(prompt)).strip()
        kbench.assertions.assert_true(len(text) > 0, expectation=f"prompt {i} returned text")
        with open("passages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_id": i, "prompt": prompt, "text": text}) + "\\n")


generate.run(kbench.llm)
'''

body = "".join("    %r,\n" % p for p in prompts)
open(out, "w").write(HEAD + body + TAIL)
print("wrote", out)
