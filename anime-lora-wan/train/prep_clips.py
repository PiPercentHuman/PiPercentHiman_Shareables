"""Training clips for a style LoRA, cut from one long video.

  python train/prep_clips.py <video> <out_dir> [--size 192x320] [--crop W:H:X:Y] [--scene 0.3] [--holdout 20]

1. Shots: ffmpeg's scene-change score over the whole file; every shot of at least 2.1 s (33 frames at 16 fps).
2. One clip per shot: 2.1 s from its middle, cropped (--crop, to cut away a logo or a border) and scaled to --size,
   16 fps -> <out_dir>/cands/.
3. Motion: mean optical flow over 8 frame pairs, and "local" motion = flow minus its median vector, so a camera pan
   over a still picture scores low. Kept: local motion between 0.3 and 2.0 px/frame (below: a held picture; above:
   mostly flashes and smears).
4. Hidden cuts: the largest frame-to-frame difference against the median one; a clip whose largest jump is more than
   3x the median and above 0.15 has a cut the scene score missed, and is dropped.
5. Every --holdout-th kept clip goes to <out_dir>/holdout/, the rest to <out_dir>/train/; clips.jsonl has every
   score, and sheet-*.jpg shows the first frame of every kept clip. Look at the sheets: text panels, banners and
   title cards pass every number here and have to be removed by eye.
The thresholds are starting points, not tuned values: check the sheets for your video.
"""
import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

WIN = 2.1


def shots(video, thr):
    log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(video), "-vf", f"select='gt(scene,{thr})',showinfo", "-an",
                          "-f", "null", "-"], capture_output=True, text=True, errors="ignore").stderr
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True, check=True).stdout)
    b = [0.0] + [float(m) for m in re.findall(r"pts_time:([0-9.]+)", log)] + [dur]
    return [(i, a, z) for i, (a, z) in enumerate(zip(b, b[1:])) if z - a >= WIN]


def gray_frames(path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(cv2.resize(f, (96, 160)), cv2.COLOR_BGR2GRAY))
    cap.release()
    return fr


def scores(path):
    fr = gray_frames(path)
    if len(fr) < 10:
        return None
    m, loc = [], []
    for i in np.linspace(0, len(fr) - 2, 8).astype(int):
        fl = cv2.calcOpticalFlowFarneback(fr[i], fr[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        m.append(np.linalg.norm(fl, axis=2).mean())
        loc.append(np.linalg.norm(fl - np.median(fl.reshape(-1, 2), 0), axis=2).mean())
    diff = [np.abs(fr[i + 1].astype(np.float32) - fr[i]).mean() / 255 for i in range(len(fr) - 1)]
    return {"flow": round(float(np.mean(m)), 3), "local": round(float(np.mean(loc)), 3),
            "jump_max": round(float(np.max(diff)), 3), "jump_med": round(float(np.median(diff)), 3), "frames": len(fr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--size", default="192x320", help="WxH of the training clips")
    ap.add_argument("--crop", default="", help="ffmpeg crop W:H:X:Y applied before scaling")
    ap.add_argument("--scene", type=float, default=0.3)
    ap.add_argument("--holdout", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "cands").mkdir(parents=True, exist_ok=True)
    w, h = a.size.split("x")
    vf = (f"crop={a.crop}," if a.crop else "") + f"scale={w}:{h},fps=16"

    def one(s):
        i, t0, t1 = s
        clip = out / "cands" / f"{i:05d}.mp4"
        if not clip.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{(t0 + t1) / 2 - WIN / 2:.3f}", "-i", a.video, "-t", f"{WIN:.3f}",
                            "-an", "-vf", vf, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(clip)], check=True)
        r = scores(clip) or {}
        keep = bool(r) and 0.3 <= r["local"] <= 2.0 and not (r["jump_max"] > 3 * r["jump_med"] and r["jump_max"] > 0.15)
        return {"clip": clip.name, "start": round(t0, 3), "end": round(t1, 3), **r, "keep": keep}

    ss = shots(a.video, a.scene)
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, ss))
    (out / "clips.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf8")
    kept = [r for r in rows if r["keep"]]
    for sub in ("train", "holdout"):
        (out / sub).mkdir(exist_ok=True)
    for k, r in enumerate(kept):
        sub = "holdout" if a.holdout and k % a.holdout == a.holdout - 1 else "train"
        shutil.copyfile(out / "cands" / r["clip"], out / sub / r["clip"])
    tiles = []
    for r in kept:
        cap = cv2.VideoCapture(str(out / "cands" / r["clip"]))
        ok, f = cap.read()
        if ok:
            f = cv2.resize(f, (96, int(96 * f.shape[0] / f.shape[1])))
            cv2.putText(f, r["clip"][:5], (2, 10), 0, 0.3, (0, 255, 255), 1)
            tiles.append(f)
    per = 12
    for p in range(0, len(tiles), per * 8):
        page = tiles[p:p + per * 8]
        page += [np.zeros_like(tiles[0])] * (-len(page) % per)
        cv2.imwrite(str(out / f"sheet-{p // (per * 8)}.jpg"), np.vstack([np.hstack(page[i:i + per]) for i in range(0, len(page), per)]))
    print(f"PREP-END {len(ss)} shots >= {WIN} s, {len(kept)} kept, {sum(1 for _ in (out / 'holdout').glob('*.mp4'))} held out")


if __name__ == "__main__":
    main()
