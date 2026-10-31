"""Run the GPU half's pure-Python cells against fake data, to catch missing names before a 2-hour run.

Four GPU runs were lost to NameErrors introduced when c3 was replaced by c3load: first `test_texts`, then
`CORRECTED_ROWS`. This executes c1 -> c2data -> c3load -> c13offsets in order with torch and transformers
replaced by MagicMocks, so a missing name fails here in seconds instead of on Kaggle in two hours.
"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np

class Fake(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        m = MagicMock(name=name)
        setattr(self, name, m)
        return m

for name in ("torch", "torch.nn", "torch.cuda", "transformers", "safetensors", "safetensors.torch",
             "sentence_transformers", "sklearn", "sklearn.metrics", "sklearn.model_selection"):
    sys.modules[name] = Fake(name)

g = {"__name__": "__smoke__"}
for cell in ("c1.py", "c2data.py", "c3load.py", "c13offsets.py"):
    src = open(f"/scratch/work/v61/{cell}").read()
    src = src.replace('INPUT_DIR = "/kaggle/input"', 'INPUT_DIR = "/scratch/work/smoke/in"')
    try:
        exec(compile(src, cell, "exec"), g)
        print(f"[ok]   {cell}")
    except Exception as e:
        print(f"[FAIL] {cell}: {type(e).__name__}: {e}")
        raise
print("\nall pure-Python GPU cells ran clean")
