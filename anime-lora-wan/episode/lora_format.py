"""LoRA key formats.

The published LoRAs use the ComfyUI / diffusers layout ("diffusion_model.blocks.0.cross_attn.k.lora_A.weight").
musubi-tuner's generator wants its own layout ("lora_unet_blocks_0_cross_attn_k.lora_down.weight" plus an alpha), and
it builds the names from the DiT file's keys: with Comfy-Org's repackaged DiT, whose keys start with
"model.diffusion_model.", the names become "lora_unet_model_diffusion_model_blocks_0_...". A LoRA whose names do not
match is skipped WITHOUT an error and the video comes out as the base model's, so the prefix is read from the DiT
file itself and every converted module is checked against it.

  python lora_format.py <lora.safetensors> <dit.safetensors> <out.safetensors>
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def dit_prefix(dit):
    with safe_open(str(dit), "pt") as f:
        keys = list(f.keys())
    return "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in keys) else "", set(keys)


def to_musubi(lora, dit, out):
    prefix, dit_keys = dit_prefix(dit)
    src = load_file(str(lora))
    if any(k.startswith("lora_unet_") for k in src):  # already musubi layout: copy as is
        save_file(src, str(out))
        return len([k for k in src if k.endswith(".alpha")])
    res, n = {}, 0
    for k, a in src.items():
        if not k.endswith(".lora_A.weight"):
            continue
        mod = k[len("diffusion_model."):-len(".lora_A.weight")]
        assert prefix + mod + ".weight" in dit_keys, f"{mod} is not in the DiT file"
        b = src[k.replace(".lora_A.", ".lora_B.")]
        name = "lora_unet_" + (prefix + mod).replace(".", "_")
        res[name + ".lora_down.weight"] = a
        res[name + ".lora_up.weight"] = b
        res[name + ".alpha"] = torch.tensor(float(a.shape[0]))  # alpha = rank: scale 1, the scale is already in A and B
        n += 1
    assert n, f"{lora}: no lora_A/lora_B pairs found"
    save_file(res, str(out))
    return n


def musubi_copy(lora, dit, cache_dir):
    """The musubi-layout copy of a published LoRA for this DiT, made once and cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{Path(lora).stem}.musubi-{Path(dit).stem}.safetensors"
    if not out.exists() or out.stat().st_mtime < Path(lora).stat().st_mtime:
        print("LoRA", Path(lora).name, "->", out.name, to_musubi(lora, dit, out), "modules", flush=True)
    return out


if __name__ == "__main__":
    print(to_musubi(*sys.argv[1:4]), "modules")
