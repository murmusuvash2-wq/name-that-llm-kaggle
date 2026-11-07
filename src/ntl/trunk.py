"""Truncated causal LM trunks + LoRA merge for two-stage members."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from torch import nn


def load_trunk(base_dir: str, keep: int, dtype: torch.dtype, *, AutoConfig, AutoModel):
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers = keep
    mc.use_cache = False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:keep]

    H, I, V = mc.hidden_size, mc.intermediate_size, mc.vocab_size
    hd = getattr(mc, "head_dim", H // mc.num_attention_heads)
    q = mc.num_attention_heads * hd
    kv = mc.num_key_value_heads * hd
    gib = 2 * (V * H + keep * (2 * H * q + 2 * H * kv + 3 * H * I)) / 2**30
    n_gpu = torch.cuda.device_count()

    if gib > 13 and n_gpu > 1:
        mem = {i: "13GiB" for i in range(n_gpu)}
        model = AutoModel.from_pretrained(
            base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem
        ).eval()
    elif gib > 12:
        print(f"trunk needs {gib:.1f} GiB but only {n_gpu} GPU(s) — CPU offload", flush=True)
        mem = {i: "9GiB" for i in range(n_gpu)}
        mem["cpu"] = "60GiB"
        model = AutoModel.from_pretrained(
            base_dir, config=mc, dtype=dtype, device_map="auto",
            max_memory=mem, low_cpu_mem_usage=True,
        ).eval()
    else:
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype).to("cuda:0").eval()

    model.norm = nn.Identity()
    print(f"trunk of {keep} layers, {gib:.1f} GiB of weights", flush=True)
    return model


@torch.no_grad()
def merge_lora(model, adapter_dir: str, load_file) -> None:
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    for k, A in sd.items():
        if ".lora_A." not in k:
            continue
        name = k.split(".lora_A.")[0]
        lin_key = name[name.index("layers.") :]
        lin = modules[lin_key]
        B = sd[k.replace(".lora_A.", ".lora_B.")]
        delta = (B.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale
        lin.weight.add_(delta.to(lin.weight.dtype))


def meta_keep_layers(meta: Dict[str, Any]) -> tuple[int, List[int], int]:
    cfg = meta.get("cfg", meta)
    keep = int(cfg["keep"])
    layers = list(cfg["layers"])
    max_len = int(cfg.get("max_len", 384))
    if max(layers) >= keep:
        raise ValueError(f"layer index {max(layers)} >= keep {keep} in meta")
    return keep, layers, max_len
