"""Split a panel into depth layers for a multiplane camera.

depth_probe.py showed what Depth Anything V2 Small makes of ink line art: clean
figure/ground cut-outs in the right near/far ORDER, almost no relief inside a
figure. So this does not build a mesh; it bands the depth into K flat layers,
the way a multiplane rig (Reiniger 1926, Iwerks 1933, Disney 1937) or an anime
compositor stacks cels.

What a layer needs that the panel does not have: whatever a NEARER layer covers.
Move the camera and that area is uncovered ("disocclusion"), so it has to be invented:
  --fill telea  OpenCV's Telea inpainting - cheap, no model, visibly smeary on big holes
  --fill lama   LaMa (big-lama, Apache 2.0) - continues fields, foliage and walls instead
Only the hole is filled; every drawn pixel of a layer stays the artist's.

usage (needs cv2 + numpy; torch for --fill lama):
  python prep_layers.py <panel.png> <depth.png> <outdir> [--layers 3] [--fill lama] [--lama big-lama.pt]
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("panel"); ap.add_argument("depth"); ap.add_argument("out")
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--feather", type=float, default=1.5)
ap.add_argument("--grow", type=int, default=6, help="px a nearer layer's hole is widened before filling")
ap.add_argument("--cut", type=int, default=31, help="px of contact cut before labelling figures")
ap.add_argument("--no-whole", dest="whole", action="store_false", help="v1 behaviour: band pixel by pixel (tears figures)")
ap.add_argument("--fill", choices=["telea", "lama"], default="telea",
                help="telea: OpenCV, local and smeary (v1/v2). lama: LaMa big-lama (Apache 2.0), a learned inpainter")
ap.add_argument("--lama", default="big-lama.pt", help="TorchScript LaMa: github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt")
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
img = cv2.imread(a.panel, cv2.IMREAD_COLOR)
d = cv2.imread(a.depth, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # 1 = near
assert img.shape[:2] == d.shape, (img.shape, d.shape)

# 1-D k-means on depth: bands follow the picture's own figure/ground gaps
# instead of equal slices of a scale the model never calibrated.
K = a.layers
samples = d.reshape(-1, 1)[:: max(1, d.size // 200_000)]
_, _, centers = cv2.kmeans(samples, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4), 5, cv2.KMEANS_PP_CENTERS)
centers = np.sort(centers.ravel())
edges = (centers[:-1] + centers[1:]) / 2
band = np.digitize(d, edges)  # 0 = far ... K-1 = near

# light clean-up: a band speckle of a few pixels becomes a floating chip in 3D
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# v2 (2026-09-28): keep each figure in ONE layer. v1 banded pixel by pixel, and the band edge
# ran through Alveric's robe in the forge panel - near half and middle half slid apart and a
# white slash opened across him. Now every connected region of the non-backdrop mask goes
# wholly to its majority band. Costs some parallax inside a clump (a figure touching lilies
# moves with them); buys no tearing.
# First version of this merged every connected region, and in every panel the figures touch
# the ground or the lilies, so everything collapsed into ONE foreground layer: no tearing, but
# almost no depth. So cut thin contacts first - erode, label what is left, grow each label back
# over the foreground - and only then give each piece its majority band.
if a.whole:
    fg = (band > 0).astype(np.uint8)
    core = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.cut, a.cut)))
    n, lab = cv2.connectedComponents(core, connectivity=8)
    lab = lab.astype(np.int32)
    for _ in range(a.cut):  # grow labels back, only over foreground pixels still unlabelled
        grown = cv2.dilate(lab.astype(np.float32), kern).astype(np.int32)
        take = (lab == 0) & (fg == 1) & (grown > 0)
        if not take.any():
            break
        lab[take] = grown[take]
    for c in range(1, n):
        sel = lab == c
        if sel.sum() < 400:
            continue
        band[sel] = int(np.bincount(band[sel]).argmax())
masks = []
for i in range(K):
    m = (band == i).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    masks.append(m)

# v3 (2026-09-29): what fills the uncovered area. Telea (v1/v2) smears on big holes - on a PAN
# of a wide panel the smear becomes the most visible thing in the shot (01_c01_erl). LaMa
# (Suvorov et al. 2021, github.com/advimman/lama, Apache 2.0; TorchScript export from
# enesmsahin/simple-lama-inpainting) continues texture - foliage, fields, wall - instead.
_lama = None


def inpaint(bgr, mask):
    global _lama
    if a.fill == "telea":
        return cv2.inpaint(bgr, mask * 255, 5, cv2.INPAINT_TELEA)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _lama is None:
        _lama = torch.jit.load(a.lama, map_location=dev).eval()
    h, w = mask.shape
    H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8  # the network wants multiples of 8
    rgb = np.pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ((0, H - h), (0, W - w), (0, 0)), mode="reflect")
    m = np.pad(mask, ((0, H - h), (0, W - w)), mode="reflect")
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(dev)
    y = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(dev)
    with torch.no_grad():
        out = _lama(x, y)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()[:h, :w]
    res = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    keep = mask == 0
    res[keep] = bgr[keep]  # only the hole is invented; every drawn pixel stays the artist's
    return res


layers = []
for i in range(K):
    nearer = np.zeros_like(masks[0])
    for j in range(i + 1, K):
        nearer |= masks[j]
    hole = cv2.dilate(nearer, kern, iterations=max(1, a.grow // 2))
    if i == 0:
        region = np.ones_like(hole)  # the backdrop is opaque everywhere
    else:
        # a middle layer only needs filling where it could plausibly continue
        region = masks[i] | (cv2.dilate(masks[i], kern, iterations=10) & hole)
    fill = hole & region
    color = inpaint(img, fill) if fill.any() else img.copy()
    alpha = region.astype(np.float32)
    if i > 0 and a.feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), a.feather)
    rgba = np.dstack([color, (np.clip(alpha, 0, 1) * 255).astype(np.uint8)])
    p = out / f"layer_{i}.png"
    cv2.imwrite(str(p), rgba)
    layers.append({"file": p.name, "depthCenter": round(float(centers[i]), 4),
                   "ownShare": round(float(masks[i].mean()), 4), "filledShare": round(float(fill.mean()), 4)})
    print(f"  layer {i}: depth {centers[i]:.3f}  own {masks[i].mean():.1%}  inpainted {fill.mean():.1%}")

h, w = img.shape[:2]
(out / "layers.json").write_text(json.dumps({"panel": str(a.panel), "w": w, "h": h, "layers": layers}, indent=1))
