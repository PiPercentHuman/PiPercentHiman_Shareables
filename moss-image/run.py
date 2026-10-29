"""Run the task ladder and write one record per image.

  python run.py                      everything not yet in runs/
  python run.py --only poster        one task (or task/variant, or
                                     task/variant/contender)
  python run.py --seeds 0            only these seeds

Records go to runs/<task>.jsonl, images to images/. A record is written
only after its image is saved; an error from the server stops the run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import comfy
from tasks import TASKS

HERE = Path(__file__).resolve().parent

# contender -> the tasks/variants it runs ("*" = all variants of a task)
CONTENDERS = {
    "moss": {t: "*" for t in TASKS},
    "moss-card": {"poster": ["title"], "menu": ["items-8"]},
    "moss-fast": {"poster": ["title"], "menu": ["items-8"],
                  "concert": ["layout"]},
    # runs on a ComfyUI with Qwen-Image 2.1 (COMFY_HOST); no JSON layout,
    # so it gets the concert poster as the sentence
    "qwen": {"poster": ["title"], "menu": ["items-3", "items-8", "items-20"],
             "concert": ["sentence"], "sign": ["chinese"]},
}
RECIPE = {"moss": "template", "moss-card": "card", "moss-fast": "fast",
          "qwen": "qwen"}


def jobs():
    for c, plan in CONTENDERS.items():
        for task, variants in plan.items():
            for v, spec in TASKS[task].items():
                if variants != "*" and v not in variants:
                    continue
                for seed in spec["seeds"]:
                    yield task, v, c, seed, spec


def done_ids():
    ids = set()
    for f in (HERE / "runs").glob("*.jsonl"):
        ids |= {json.loads(ln)["id"] for ln in f.read_text("utf8").splitlines()}
    return ids


def run_one(task, v, c, seed, spec):
    rid = f"{task}/{v}/{c}/s{seed}"
    png_rel = f"images/{task}/{v}/{c}_s{seed}.png"
    refs = []
    if "source" in spec:                 # an edit starts from an earlier image
        st, sv = spec["source"]
        src = HERE / f"images/{st}/{sv}/{c}_s{seed}.png"
        if not src.exists():
            raise SystemExit(f"{rid}: needs {src.name} first")
        refs = [comfy.upload(str(src))]
    recipe = "edit" if refs else RECIPE[c]
    if c == "qwen":
        graph = comfy.qwen(spec["prompt"], seed, prefix=rid.replace("/", "_"))
    else:
        graph = comfy.moss(spec["prompt"], seed, recipe, refs=refs,
                           prefix=rid.replace("/", "_"))
    png, secs = comfy.run(graph)
    out = HERE / png_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    rec = {"id": rid, "task": task, "variant": v, "contender": c,
           "seed": seed, "recipe": recipe,
           "settings": comfy.RECIPES[recipe], "prompt": spec["prompt"],
           "target": spec["target"], "lang": spec["lang"],
           "boxes": spec["boxes"], "source": str(src.relative_to(HERE)) if refs else None,
           "seconds": round(secs, 1), "png": png_rel,
           "sha256": hashlib.sha256(png).hexdigest()}
    with open(HERE / "runs" / f"{task}.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{rid:40s} {secs:6.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--contender", default="",
                    help="only this contender (qwen needs its own server)")
    a = ap.parse_args()
    (HERE / "runs").mkdir(exist_ok=True)
    seeds = {int(s) for s in a.seeds.split(",") if s}
    have = done_ids()
    for task, v, c, seed, spec in jobs():
        rid = f"{task}/{v}/{c}/s{seed}"
        if rid in have or not rid.startswith(a.only):
            continue
        if seeds and seed not in seeds:
            continue
        if c != a.contender and (a.contender or c == "qwen"):
            continue
        run_one(task, v, c, seed, spec)


if __name__ == "__main__":
    main()
