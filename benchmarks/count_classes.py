"""Per-class counts of the merged pool, plus how many prompt groups cover all 18 models.

Which classes are actually short of data is a question about numbers, not impressions.
"""
import glob
import sys

import pandas as pd

if len(sys.argv) > 1:
    root = sys.argv[1]
    cands = [p for p in glob.glob(f"{root}/**/*", recursive=True) if p.endswith((".csv", ".parquet"))]
    f = cands[0]
else:
    f = sorted(glob.glob("d/**/*.csv", recursive=True))[0]
df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
print(f"file: {f}\nrows: {len(df):,}\ncols: {list(df.columns)}\n")

print("===== per model =====")
vc = df["model"].value_counts()
for k, v in vc.items():
    print(f"{k:<26} {v:>8,}  {100 * v / len(df):5.1f}%")

if "source" in df.columns:
    print("\n===== per source =====")
    for k, v in df["source"].value_counts().items():
        print(f"{k:<36} {v:>8,}")

for key in ("prompt_key", "prompt_id"):
    if key in df.columns:
        g = df.groupby(key)["model"].nunique()
        print(f"\nprompt groups by {key}: {len(g):,} | with all 18 models: {int((g == 18).sum()):,}"
              f" | with >=2: {int((g >= 2).sum()):,}")
        break

if "setting" in df.columns:
    print(f"\nsettings: {df['setting'].nunique()} distinct")
    for k, v in df["setting"].value_counts().head(12).items():
        print(f"  {str(k):<30} {v:>8,}")

print("\n===== models below 5,000 rows =====")
for k, v in vc.items():
    if v < 5000:
        print(f"  {k:<26} {v:>8,}")
