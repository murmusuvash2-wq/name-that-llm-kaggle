"""Publish a kernel output folder as a Kaggle dataset, and fail loudly if it does not work.

Three attempts went by with the notebook silently falling back to the published DeBERTa head because this
dataset was never created: the publish step inside the workflow exited non-zero and the run carried on. This
script asserts every precondition and then verifies the result, so "published" is a fact and not a hope.

usage: publish_dataset.py <folder> <owner/slug> "<title>"
"""
import json
import os
import subprocess
import sys


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    folder, slug, title = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.isdir(folder):
        sys.exit(f"ABORT: {folder} is not a directory")

    files = sorted(os.listdir(folder))
    print(f"folder {folder} contains: {files[:8]}")
    for required in ("meta.json", "head.safetensors", "encoder"):
        if required not in files:
            sys.exit(f"ABORT: {folder} is missing {required} — refusing to publish an incomplete checkpoint")

    meta = json.load(open(os.path.join(folder, "meta.json")))
    if "labels" not in meta or "holdout_ba" not in meta:
        sys.exit(f"ABORT: meta.json lacks labels/holdout_ba — this is not our checkpoint")
    print(f"checkpoint: holdout_ba {meta['holdout_ba']}, {len(meta['labels'])} labels")

    json.dump({"title": title, "id": slug, "licenses": [{"name": "other"}]},
              open(os.path.join(folder, "dataset-metadata.json"), "w"), indent=2)

    code, out = run(f"kaggle datasets create -p {folder}")
    print(out[-800:])
    if code != 0:
        print("create failed, trying a new version")
        code, out = run(f'kaggle datasets version -p {folder} -m "publish"')
        print(out[-800:])
        if code != 0:
            sys.exit("ABORT: both create and version failed")

    code, out = run(f"kaggle datasets files {slug} --csv")
    print(out[:600])
    if code != 0 or "head.safetensors" not in out:
        sys.exit("ABORT: the dataset is not readable after publishing")
    print(f"VERIFIED: {slug} exists and lists head.safetensors")


main()
