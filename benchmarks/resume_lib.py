"""Resume support shared by both training notebooks.

Kaggle kills a GPU session at nine hours, and neither of our training jobs fits in one: the 8B member
needs about 5.2 hours for a 60k-row pass but a full-pool epoch is longer, and a 14B is longer still. So a
job has to survive being stopped and started again, or it never finishes at all.

The contract: the training kernel writes `resume_state.pt` next to its output every `SAVE_EVERY` steps.
The workflow publishes the whole output directory as a dataset, so the next session attaches that dataset
and picks up from the exact step — same sample order, same optimizer momentum, same RNG state.

Two details that are easy to get wrong and cost a whole session each:
  * the sample order must be **stored**, not regenerated — `np.random.default_rng(seed).permutation(n)` is
    stable across numpy versions but the *position within the epoch* is not, and a re-shuffled order means
    the resumed run sees different rows and the loss curve lies;
  * the optimizer state must be saved too — AdamW's moments are most of what makes a resumed run continue
    smoothly instead of re-warming for an hour.
"""
import os

import numpy as np
import torch


class Resume:
    def __init__(self, out_dir, input_dir, save_every=500, tag="train"):
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.save_every = save_every
        self.tag = tag
        self.path = os.path.join(out_dir, "resume_state.pt")
        self.state = None

    # ------------------------------------------------------------------ find and load
    def find_published(self):
        """A previous session's state, published as a dataset and attached read-only."""
        import glob
        hits = sorted(glob.glob(f"{self.input_dir}/**/resume_state.pt", recursive=True))
        return hits[0] if hits else None

    def load(self):
        src = self.path if os.path.exists(self.path) else self.find_published()
        if src is None:
            print(f"[{self.tag}] no resume state — starting from step 0", flush=True)
            return None
        try:
            self.state = torch.load(src, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[{self.tag}] resume state unreadable ({type(e).__name__}: {e}) — starting fresh", flush=True)
            return None
        print(f"[{self.tag}] RESUMING from {src}: step {self.state.get('step', 0):,}, "
              f"rows seen {self.state.get('seen', 0):,}, best BA {self.state.get('best_ba', -1):.4f}", flush=True)
        return self.state

    # ------------------------------------------------------------------ save
    def save(self, *, step, seen, best_ba, order, model, extra_tensors, optimizer=None, epoch=0):
        """Atomic write: a session killed mid-save must not leave a truncated file behind."""
        payload = {
            "step": int(step), "seen": int(seen), "best_ba": float(best_ba), "epoch": int(epoch),
            "order": np.asarray(order),
            "lora": {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k},
            "extra": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in extra_tensors.items()},
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        tmp = self.path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.path)
        return self.path

    def restore_into(self, model, extra_modules, optimizer=None):
        """Put the loaded tensors back. Returns (step, seen, best_ba, order)."""
        if not self.state:
            return 0, 0, -1.0, None
        if self.state.get("lora"):
            missing, unexpected = model.load_state_dict(self.state["lora"], strict=False)
            print(f"[{self.tag}] restored {len(self.state['lora'])} LoRA tensors "
                  f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
        extra = self.state.get("extra", {}) or {}
        for name, mod in extra_modules.items():
            sd = extra.get(name)
            if sd:
                mod.load_state_dict(sd)
                print(f"[{self.tag}] restored {name} ({len(sd)} tensors)", flush=True)
            else:
                print(f"[{self.tag}] {name} not in the resume state — keeping its initial weights", flush=True)
        if optimizer is not None and "optimizer" in self.state:
            try:
                optimizer.load_state_dict(self.state["optimizer"])
                print(f"[{self.tag}] restored optimizer momentum", flush=True)
            except Exception as e:
                print(f"[{self.tag}] optimizer state not restored ({type(e).__name__}) — "
                      f"continuing with fresh momentum", flush=True)
        return (self.state.get("step", 0), self.state.get("seen", 0),
                float(self.state.get("best_ba", -1.0)), self.state.get("order"))


def make_order(n, seed):
    """One deterministic permutation for the whole run; stored so a resume sees the same rows."""
    return np.random.default_rng(seed).permutation(n)


def slice_from(order, start_step, bs, accum):
    """Where to begin inside the stored order for a resumed run."""
    per_step = bs * accum
    return int(start_step * per_step)
