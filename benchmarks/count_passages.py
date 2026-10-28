"""Count what a Benchmarks run actually produced."""
import glob
import json
import sys
from collections import Counter

root = sys.argv[1] if len(sys.argv) > 1 else "out"
rows = []
for p in glob.glob(f"{root}/**/*.jsonl", recursive=True):
    model = p.split("/")[-3] if len(p.split("/")) > 3 else "?"
    for line in open(p, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
            rows.append((model, d.get("text", "")))
        except Exception:
            pass
print("passages:", len(rows))
print("per model:", dict(Counter(m for m, _ in rows)))
wc = sorted(len(t.split()) for _, t in rows)
if wc:
    print("words min/median/max:", wc[0], wc[len(wc) // 2], wc[-1],
          "| in 40-250:", sum(1 for w in wc if 40 <= w <= 250), "/", len(wc))
    print("with markdown:", sum(1 for _, t in rows if "**" in t or t.lstrip().startswith("#")))
if rows:
    print("sample:", rows[0][1][:220])
