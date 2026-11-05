"""Catch undefined names across a whole notebook before spending a GPU session on it.

A 66-minute run died on `NameError: name 'norms' is not defined` because I copied the resume block from the
8B notebook, where the checkpoint has per-layer norms, into the DeBERTa notebook, where it does not. pyflakes
would have caught it in a second — but only if the cells are analysed *together*, because a name defined in
one cell is legitimately used in a later one.

So: concatenate every code cell into one module, stub the notebook magics, and run pyflakes over the whole
thing. That catches undefined names, unused imports and real syntax errors in the same pass.
"""
import ast
import sys
import tempfile

from pyflakes.api import check
from pyflakes.reporter import Reporter

import json


def cell_sources(path):
    nb = json.load(open(path))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        keep = [l for l in src.split("\n") if not l.lstrip().startswith(("%", "!"))]
        out.append("\n".join(keep))
    return out


def check_notebook(path, ignore=()):
    body = "\n\n".join(cell_sources(path))
    problems = []

    class Collector(Reporter):
        def __init__(self):
            super().__init__(sys.stdout, sys.stdout)

        def flake(self, message):
            text = str(message)
            if any(i in text for i in ignore):
                return
            problems.append(text)

        def unexpectedError(self, filename, msg):
            problems.append(f"ERROR {filename}: {msg}")

        def syntaxError(self, filename, msg, lineno, offset, text):
            problems.append(f"SYNTAX line {lineno}: {msg}")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body)
        tmp = f.name
    check(open(tmp).read(), tmp, Collector())
    return problems


if __name__ == "__main__":
    # notebook magics and Kaggle-provided names are not real problems
    IGNORE = ("unable to detect undefined names", "'get_ipython'")
    bad = 0
    for path in sys.argv[1:]:
        probs = check_notebook(path, ignore=IGNORE)
        print(f"\n=== {path} ===")
        if not probs:
            print("  clean")
        for p in probs:
            print("  " + p)
            bad += 1
    sys.exit(1 if bad else 0)
