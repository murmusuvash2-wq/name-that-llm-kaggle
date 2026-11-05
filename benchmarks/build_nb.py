"""Build a notebook with guaranteed-unique cell ids.

I shipped duplicate ids twice (once in the TPU probe, once in the training notebooks) and Kaggle rejected
the push both times. Enforcing uniqueness here means that bug cannot come back.
"""
import ast
import json


def _src(text):
    text = text.rstrip("\n")
    lines = text.split("\n")
    return [x + "\n" for x in lines[:-1]] + [lines[-1]]


def build(path, cells, title=None):
    """cells: list of (kind, source) where kind is 'code' or 'md'."""
    out, seen = [], set()
    for i, (kind, text) in enumerate(cells):
        cid = f"c{i:02d}"
        assert cid not in seen, cid
        seen.add(cid)
        if kind == "code":
            # IPython magics are not Python; parse the body only, so a real syntax error still fails here
            body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(("%", "!")))
            ast.parse(body)          # a syntax error here is a wasted GPU session
            out.append({"cell_type": "code", "execution_count": None, "id": cid,
                        "metadata": {}, "outputs": [], "source": _src(text)})
        else:
            out.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": _src(text)})
    nb = {"cells": out,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(path, "w"), indent=1)
    ids = [c["id"] for c in out]
    assert len(set(ids)) == len(ids), "duplicate ids"
    print(f"{path}: {len(out)} cells, ids unique, all code parses")
    return nb


def read(p):
    return open(p).read().rstrip("\n")
