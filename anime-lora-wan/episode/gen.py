"""Shots from text: every shot of the chapter, every seed, with the LoRAs it names.

  <musubi python> episode/gen.py <chapter> run              -> <chapter>/takes/<shot>_s<seed>.mp4
  python episode/gen.py <chapter> sheet                     -> <chapter>/takes/sheet-*.jpg (3 frames per take)
  python episode/gen.py <chapter> pick [<shot>=<seed> ...]  -> <chapter>/takes/pick/<shot>.mp4 (chapter.json "pick",
                                                                or the first seed, unless named here)

Shots run in groups that share a LoRA set, because musubi applies every LoRA of a run to every prompt in it: a
character LoRA in the same run as a shot without that character puts his face on whoever is in it.
Sampler: Wan 2.1 T2V-1.3B with the Self-Forcing distillation merged in (train/merge_distill.py), 4 steps, cfg 1,
fp8 T5, VAE cache on the CPU. Takes are never overwritten; delete one to draw it again.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import chapter, paths  # noqa: E402
from lora_format import musubi_copy  # noqa: E402


def lora_file(P, name):
    """"anime-style" -> <loras>/wan2.1-t2v-1.3b-anime-style.safetensors; a name ending in .safetensors is used as is."""
    return Path(P["loras"]) / (name if name.endswith(".safetensors") else f"wan2.1-t2v-1.3b-{name}.safetensors")


def run(d, ch):
    P = paths()
    out = d / "takes"
    out.mkdir(parents=True, exist_ok=True)
    W, H = ch.get("size", [832, 480])
    groups = {}
    for k, v in ch["shots"].items():
        for s in v["seeds"]:
            if not (out / f"{k}_s{s}.mp4").exists():
                groups.setdefault(tuple(v["loras"]), []).append((k, s))
    t0 = time.time()
    for loras, jobs in groups.items():
        files = [musubi_copy(lora_file(P, x), P["dit_fast"], Path(P["loras"]) / "_musubi") for x in loras]
        pf = out / f"prompts_{'_'.join(loras) or 'none'}.txt"
        pf.write_text("\n".join(f"{ch['shots'][k]['prompt']} --w {W} --h {H} --f {ch['shots'][k]['frames']} --d {s}"
                                for k, s in jobs) + "\n", "utf8")
        before = set(out.glob("*.mp4"))
        cmd = [P["musubi_python"], str(Path(P["musubi_tuner"]) / "src/musubi_tuner/wan_generate_video.py"),
               "--task", "t2v-1.3B", "--dit", P["dit_fast"], "--vae", P["vae"], "--vae_cache_cpu", "--t5", P["t5"],
               "--fp8_t5", "--from_file", str(pf), "--save_path", str(out), "--output_type", "video", "--attn_mode",
               "torch", "--infer_steps", "4", "--guidance_scale", "1.0", "--flow_shift", "5.0", "--sample_solver",
               "vanilla"]
        if files:
            cmd += ["--lora_weight", *map(str, files), "--lora_multiplier", *["1.0"] * len(files)]
        subprocess.run(cmd, cwd=P["musubi_tuner"], check=True)
        new = sorted(set(out.glob("*.mp4")) - before, key=lambda p: p.stat().st_mtime)
        assert len(new) == len(jobs), (loras, len(new), len(jobs))
        for (k, s), f in zip(jobs, new):
            f.rename(out / f"{k}_s{s}.mp4")
        print("group", loras, len(jobs), "done", flush=True)
    print(f"GEN-END {sum(len(j) for j in groups.values())} videos in {time.time() - t0:.0f}s")


def sheet(d, ch):
    import cv2
    import numpy as np
    out, rows = d / "takes", []
    cols = max(len(v["seeds"]) for v in ch["shots"].values())
    for k, v in ch["shots"].items():
        row = []
        for s in v["seeds"]:
            cap = cv2.VideoCapture(str(out / f"{k}_s{s}.mp4"))
            fr = []
            while True:
                ok, x = cap.read()
                if not ok:
                    break
                fr.append(cv2.resize(x, (156, 90)))
            t = np.hstack([fr[i] for i in np.linspace(0, len(fr) - 1, 3).astype(int)]) if fr else np.zeros((90, 468, 3), np.uint8)
            cv2.putText(t, f"{k} s{s}", (2, 10), 0, 0.33, (0, 255, 255), 1)
            row += [t, np.full((90, 4, 3), 60, np.uint8)]
        while len(row) < 2 * cols:
            row += [np.zeros((90, 468, 3), np.uint8), np.full((90, 4, 3), 60, np.uint8)]
        rows.append(np.hstack(row))
    for i in range(0, len(rows), 9):
        cv2.imwrite(str(out / f"sheet-{i // 9}.jpg"), np.vstack(rows[i:i + 9]))
    print("SHEET-END")


def pick(d, ch, args):
    out = d / "takes"
    (out / "pick").mkdir(exist_ok=True)
    named = {k: int(s) for k, s in (a.split("=") for a in args)}
    for k, v in ch["shots"].items():
        s = named.get(k, v.get("pick", v["seeds"][0]))
        shutil.copyfile(out / f"{k}_s{s}.mp4", out / "pick" / f"{k}.mp4")
    print("PICK-END", len(ch["shots"]))


if __name__ == "__main__":
    d, ch = chapter(sys.argv[1])
    {"run": lambda: run(d, ch), "sheet": lambda: sheet(d, ch), "pick": lambda: pick(d, ch, sys.argv[3:])}[sys.argv[2]]()
