"""Print the cell map of a notebook."""
import glob
import json
import sys

f = sys.argv[1] if len(sys.argv) > 1 else glob.glob("p/*.ipynb")[0]
nb = json.load(open(f))
cells = nb["cells"]
print(f"{f}: {len(cells)} cells, {sum(1 for c in cells if c['cell_type']=='code')} code")
for i, c in enumerate(cells):
    src = "".join(c["source"])
    print(f"--- [{i}] {c['cell_type']} ({len(src.splitlines())} lines): {' '.join(src.split())[:150]}")
