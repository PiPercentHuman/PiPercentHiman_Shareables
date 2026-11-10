"""musubi-tuner LoRA -> the published layout (ComfyUI / diffusers keys), with the training metadata replaced.

  <musubi python> train/export_lora.py <musubi_lora.safetensors> <out.safetensors> --title "anime style" --trigger "bldln style anime"

Keys: "lora_unet_blocks_0_cross_attn_k.lora_down.weight" + ".alpha" -> "diffusion_model.blocks.0.cross_attn.k.lora_A.weight"
(alpha/rank folded into the weights, so the file loads at strength 1 with the trained effect). musubi's own
convert_lora.py does the conversion; this script then drops the training metadata (it records dataset folders and
other local paths) and writes a short, plain set instead. The weight change of every module is checked against the
original before the file is written.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "episode"))
from config import paths  # noqa: E402


def check(orig, conv):
    worst = 0.0
    for k in [k for k in orig if k.endswith(".lora_down.weight")]:
        m = k[:-len(".lora_down.weight")]
        dn, up = orig[k].float(), orig[m + ".lora_up.weight"].float()
        r = dn.shape[0]
        alpha = orig[m + ".alpha"].float().item() if m + ".alpha" in orig else r
        mod = m[len("lora_unet_"):]
        key = next(x for x in conv if x.endswith(".lora_A.weight") and x[len("diffusion_model."):-len(".lora_A.weight")].replace(".", "_") == mod)
        d1, d2 = (up @ dn) * (alpha / r), conv[key.replace("lora_A", "lora_B")].float() @ conv[key].float()
        worst = max(worst, ((d1 - d2).abs().max() / d1.abs().max().clamp(min=1e-12)).item())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--trigger", default="")
    a = ap.parse_args()
    P = paths()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "other.safetensors"
        subprocess.run([P["musubi_python"], str(Path(P["musubi_tuner"]) / "src/musubi_tuner/convert_lora.py"),
                        "--input", a.src, "--output", str(tmp), "--target", "other"], check=True)
        conv = load_file(str(tmp))
    orig = load_file(a.src)
    worst = check(orig, conv)
    assert worst < 1e-4, f"conversion changed the weights (max relative difference {worst:.2e})"
    rank = next(v.shape[0] for k, v in conv.items() if k.endswith(".lora_A.weight"))
    md = {"modelspec.architecture": "wan2.1/lora", "modelspec.title": a.title, "modelspec.implementation": "https://github.com/Wan-Video/Wan2.1",
          "base_model": "Wan-AI/Wan2.1-T2V-1.3B", "trigger": a.trigger, "network_dim": str(rank), "format": "pt"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(conv, a.out, metadata=md)
    print("EXPORT-END", a.out, len([k for k in conv if k.endswith(".lora_A.weight")]), "modules, max relative difference", f"{worst:.1e}")


if __name__ == "__main__":
    main()
