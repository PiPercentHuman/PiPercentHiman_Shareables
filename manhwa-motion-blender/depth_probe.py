"""Script 1 - depth for every panel: what a photo-trained depth model makes of flat ink line art.

Writes <out>/<panel>_depth.png (brighter = nearer) and <panel>_side.png (panel | depth).
On comic panels Depth Anything V2 Small gives clean figure/ground cut-outs in the right order,
and almost no relief inside a figure - layers, not a sculpture. That is what the multiplane
camera wants.

usage:
  python depth_probe.py <panelsDir> <outDir> [panel.png ...] [--model depth-anything/Depth-Anything-V2-Small-hf]
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ap = argparse.ArgumentParser()
ap.add_argument("panels"); ap.add_argument("out"); ap.add_argument("names", nargs="*")
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")  # Apache 2.0, ~25M parameters
a = ap.parse_args()

src, out = Path(a.panels), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
names = a.names or sorted(p.name for p in src.glob("*.png"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(a.model)
model = AutoModelForDepthEstimation.from_pretrained(a.model).to(dev).eval()
for name in names:
    img = Image.open(src / name).convert("RGB")
    with torch.no_grad():
        pred = model(**proc(images=img, return_tensors="pt").to(dev)).predicted_depth
    d = torch.nn.functional.interpolate(pred[None], size=img.size[::-1], mode="bicubic")[0, 0].cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)  # relative inverse depth, 1 = near
    dimg = Image.fromarray((d * 255).astype(np.uint8))
    dimg.save(out / f"{Path(name).stem}_depth.png")
    side = Image.new("RGB", (img.width * 2, img.height), "white")
    side.paste(img, (0, 0)); side.paste(dimg.convert("RGB"), (img.width, 0))
    side.save(out / f"{Path(name).stem}_side.png")
    print(name)
