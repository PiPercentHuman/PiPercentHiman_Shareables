"""Check every record against the image it points to.

  python replay.py           every image: its hash matches the record, and
                             reading it again gives the score on file
  python replay.py --regen 3 also regenerate 3 records from their prompt
                             and seed (needs the server) and report how far
                             the new image is from the saved one

Diffusion on a graphics card is not bit-exact from run to run, so a
regenerated image is compared by its pixels, not its hash.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def records():
    for f in sorted((HERE / "runs").glob("*.jsonl")):
        if f.name != "scores.jsonl":
            yield from (json.loads(ln) for ln in f.read_text("utf8").splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", type=int, default=0)
    a = ap.parse_args()
    import score
    scores = {}
    sf = HERE / "runs" / "scores.jsonl"
    if sf.exists():
        scores = {json.loads(ln)["sha256"]: json.loads(ln)
                  for ln in sf.read_text("utf8").splitlines()}
    recs = list(records())
    bad = 0
    for r in recs:
        png = (HERE / r["png"]).read_bytes()
        if hashlib.sha256(png).hexdigest() != r["sha256"]:
            print("HASH MISMATCH", r["id"]); bad += 1; continue
        if r.get("target"):
            s = score.score(HERE / r["png"], r["target"], r["lang"], r.get("boxes"))
            on_file = scores.get(r["sha256"])
            if not on_file or (s["read"], s.get("boxes")) != \
                    (on_file["read"], on_file.get("boxes")):
                print("SCORE DIFFERS", r["id"], s["read"],
                      on_file and on_file["read"]); bad += 1
    print(f"{len(recs) - bad} of {len(recs)} records match their image")
    if a.regen:
        import io
        import comfy
        from PIL import Image, ImageChops, ImageStat
        pick = [r for r in recs if r["contender"].startswith("moss")
                and not r.get("source")]
        for r in random.Random(0).sample(pick, min(a.regen, len(pick))):
            graph = comfy.moss(r["prompt"], r["seed"], r["recipe"])
            png, _ = comfy.run(graph)
            new = Image.open(io.BytesIO(png)).convert("RGB")
            old = Image.open(HERE / r["png"]).convert("RGB")
            diff = ImageStat.Stat(ImageChops.difference(new, old)).mean
            print(f"regenerated {r['id']}: mean pixel difference "
                  f"{sum(diff) / 3:.2f} of 255")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
