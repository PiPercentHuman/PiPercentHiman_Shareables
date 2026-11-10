"""Merge a step-distillation LoRA (Kijai's ComfyUI format) fully into the Wan 2.1 T2V-1.3B DiT, so a 4-step, cfg-1
sampler can run it and the anime LoRAs still go on top through musubi.

  <musubi python> train/merge_distill.py <dit.safetensors> <distill_lora.safetensors> <out_dit.safetensors>

  dit:          wan2.1_t2v_1.3B_bf16.safetensors (Comfy-Org/Wan_2.1_ComfyUI_repackaged, split_files/diffusion_models/)
  distill_lora: Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors (Kijai/WanVideo_comfy,
                LoRAs/Wan2_1_self_forcing_1_3B/)
  out_dit:      the file paths.json calls "dit_fast"

ComfyUI users do not need this: ComfyUI loads the distill LoRA as a LoRA and gives the same picture.

The distill LoRAs carry three kinds of entries: lora_down/lora_up pairs (weight += up @ down, scale 1: no alpha key),
diff_b (added to a bias) and diff (added to a norm weight). musubi's LoRA loader handles only the first kind, which is
why the whole thing is merged instead. Keys: "diffusion_model.X" in the LoRA -> "model.diffusion_model.X" in this DiT
(Comfy-Org's repackage; a DiT without that prefix is handled too). Every LoRA entry must land on a DiT key, or the
merge stops.
"""
import sys

import torch
from safetensors.torch import load_file, save_file

dit_path, lora_path, out_path = sys.argv[1:4]
dit = load_file(dit_path)
P = "model." if any(k.startswith("model.diffusion_model.") for k in dit) else ""
lora = load_file(lora_path)
done, mods = 0, {}
for k, v in lora.items():
    base = P + k if P else k[len("diffusion_model."):]
    if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight"):
        m = k.rsplit(".lora_", 1)[0]
        mods.setdefault(m, {})["down" if "lora_down" in k else "up"] = v
        continue
    if k.endswith(".diff_b"):
        tk = base[: -len(".diff_b")] + ".bias"
    elif k.endswith(".diff"):
        tk = base[: -len(".diff")] + ".weight"
    else:
        raise SystemExit(f"unknown entry {k}")
    assert tk in dit, f"no DiT key for {k} -> {tk}"
    dit[tk] = (dit[tk].float() + v.float()).to(dit[tk].dtype)
    done += 1
for m, d in mods.items():
    tk = (P + m if P else m[len("diffusion_model."):]) + ".weight"
    assert tk in dit and "down" in d and "up" in d, f"bad module {m}"
    delta = d["up"].float() @ d["down"].float()
    assert delta.shape == dit[tk].shape, (m, delta.shape, dit[tk].shape)
    dit[tk] = (dit[tk].float() + delta).to(dit[tk].dtype)
    done += 2
assert done == len(lora), (done, len(lora))
save_file(dit, out_path)
print(f"merged {len(mods)} lora modules + {done - 2 * len(mods)} diff entries ({done}/{len(lora)}) -> {out_path}")
