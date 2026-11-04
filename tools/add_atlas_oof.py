import json
from pathlib import Path

path = Path('notebooks/oof-stacking-diagnostic.ipynb')
nb = json.loads(path.read_text())
if any(c.get('id') == 'atlas-oof' for c in nb['cells']):
    raise SystemExit('atlas-oof cell already exists')

source = '''# Frozen ATLAS readout and v66 reference-blend weight grid.
# ATLAS is trained on the official 2,699-row host file, so this is a frozen-checkpoint
# grouped diagnostic rather than a fully cross-fitted ATLAS estimate. SVC remains honest OOF.
from pathlib import Path
import sys
atlas_bundle = Path(find("atlas_bundle.json"))
atlas_root = atlas_bundle.parent
sys.path.insert(0, str(atlas_root))
from ntl_atlas.inference import predict as atlas_predict
atlas_meta = json.loads(atlas_bundle.read_text())
atlas_labels = atlas_meta["labels"]
atlas_base = find_base({"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36})
print("ATLAS base:", atlas_base, "bundle:", atlas_bundle, flush=True)
atlas_raw = atlas_predict(train.text.astype(str).tolist(), atlas_base, str(atlas_root), batch_size=1)
atlas_oof = np.zeros((len(train), K), dtype=np.float32)
for j, label in enumerate(atlas_labels):
    atlas_oof[:, LABELS.index(label)] = atlas_raw[:, j]
del atlas_raw
assert np.allclose(atlas_oof.sum(1), 1.0, atol=1e-5)
REF = (1.021, 0.226, 0.429)  # exact v66 SVC / Qwen8B / Qwen14B weights
base_logit = (REF[0] * np.log(np.clip(svc_oof, 1e-9, 1.0)) +
             REF[1] * np.log(np.clip(q8_train, 1e-9, 1.0)) +
             REF[2] * np.log(np.clip(q14_train, 1e-9, 1.0)))
base_pred = base_logit.argmax(1)
base_ba = balanced_accuracy_score(y, base_pred)
ATLAS_WEIGHT_GRID = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15)
atlas_grid = []
for w in ATLAS_WEIGHT_GRID:
    candidate_logit = base_logit + w * np.log(np.clip(atlas_oof, 1e-9, 1.0))
    candidate_pred = candidate_logit.argmax(1)
    changed = candidate_pred != base_pred
    changed_n = int(changed.sum())
    base_changed = int((base_pred[changed] == y[changed]).sum()) if changed_n else 0
    cand_changed = int((candidate_pred[changed] == y[changed]).sum()) if changed_n else 0
    fold_rows = []
    for fold, (_, iva) in enumerate(gkf.split(train, y, train.prompt_id), 1):
        fch = changed[iva]
        fn = int(fch.sum())
        fold_rows.append({
            "fold": fold, "rows": int(len(iva)),
            "baseline_ba": float(balanced_accuracy_score(y[iva], base_pred[iva])),
            "candidate_ba": float(balanced_accuracy_score(y[iva], candidate_pred[iva])),
            "changed_rows": fn,
            "baseline_accuracy_on_changed": float((base_pred[iva][fch] == y[iva][fch]).mean()) if fn else None,
            "candidate_accuracy_on_changed": float((candidate_pred[iva][fch] == y[iva][fch]).mean()) if fn else None,
            "net_correct_on_changed": int((candidate_pred[iva][fch] == y[iva][fch]).sum() - (base_pred[iva][fch] == y[iva][fch]).sum()) if fn else 0,
        })
    candidate_ba = balanced_accuracy_score(y, candidate_pred)
    rec = {
        "weight": float(w), "rows": int(len(y)), "changed_rows": changed_n,
        "changed_fraction": float(changed.mean()), "baseline_ba": float(base_ba),
        "candidate_ba": float(candidate_ba), "delta_ba": float(candidate_ba - base_ba),
        "baseline_accuracy_on_changed": float(base_changed / changed_n) if changed_n else None,
        "candidate_accuracy_on_changed": float(cand_changed / changed_n) if changed_n else None,
        "net_correct_on_changed": int(cand_changed - base_changed), "folds": fold_rows,
    }
    atlas_grid.append(rec)
    print(f"ATLAS weight={w:.2f}: BA {base_ba:.6f}->{candidate_ba:.6f} "
          f"delta={rec['delta_ba']:+.6f}, changed={changed_n}, net={rec['net_correct_on_changed']:+d}", flush=True)
best_atlas = max(atlas_grid, key=lambda r: r["candidate_ba"])
atlas_result = {
    "reference_weights": REF, "baseline_ba": float(base_ba),
    "weights": list(ATLAS_WEIGHT_GRID), "results": atlas_grid, "best": best_atlas,
    "warning": "ATLAS is a frozen checkpoint trained on the host file; only SVC is fold-cross-fitted.",
}
Path("atlas_oof_results.json").write_text(json.dumps(atlas_result, indent=2))
print("ATLAS best", json.dumps(best_atlas, indent=2), flush=True)
'''
cell = {
    'cell_type': 'code', 'execution_count': None, 'id': 'atlas-oof',
    'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)
}
# Insert immediately before the existing final scalar-weight search cell.
idx = len(nb['cells']) - 1
nb['cells'].insert(idx, cell)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
print(f'inserted atlas cell at index {idx}')
