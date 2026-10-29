"""The contender that is not a model: plain code writes the words.

Same targets, same boxes as the model gets. It spells everything and
places everything, and draws nothing else.

  python baseline.py         -> images/<task>/<variant>/code_s0.png
"""
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tasks import CONCERT, MENU, TASKS

HERE = Path(__file__).resolve().parent
SERIF = "C:/Windows/Fonts/georgia.ttf"
CJK = "C:/Windows/Fonts/msyh.ttc"


def page():
    im = Image.new("RGB", (1024, 1024), (240, 232, 214))
    return im, ImageDraw.Draw(im)


def fit(d, text, box, font=SERIF):
    """The largest size at which the text fits its box, centred in it."""
    x0, y0, x1, y1 = box
    for size in range(200, 8, -2):
        f = ImageFont.truetype(font, size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        if r - l <= x1 - x0 and b - t <= y1 - y0:
            break
    d.text(((x0 + x1 - (r - l)) / 2 - l, (y0 + y1 - (b - t)) / 2 - t),
           text, font=f, fill=(30, 26, 20))


def poster():
    im, d = page()
    fit(d, "ALPINE EXPRESS", (60, 40, 964, 200))
    return im


def menu(n):
    im, d = page()
    rows = MENU[:n]
    h = 900 // max(len(rows), 8)
    for i, item in enumerate(rows):
        fit(d, item, (120, 62 + i * h, 904, 62 + (i + 1) * h - 8))
    return im


def concert():
    im, d = page()
    for it in CONCERT:
        if "text" in it:
            fit(d, it["text"], it["box"])
    return im


def sign():
    im, d = page()
    fit(d, "老王面馆", (112, 200, 912, 480), CJK)
    fit(d, "手工拉面", (312, 560, 712, 720), CJK)
    return im


DRAW = {("poster", "title"): poster,
        ("menu", "items-3"): lambda: menu(3),
        ("menu", "items-8"): lambda: menu(8),
        ("menu", "items-20"): lambda: menu(20),
        ("concert", "layout"): concert,
        ("sign", "chinese"): sign}


def main():
    for (task, v), draw in DRAW.items():
        spec = TASKS[task][v]
        t0 = time.perf_counter()
        im = draw()
        secs = time.perf_counter() - t0
        rel = f"images/{task}/{v}/code_s0.png"
        (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
        im.save(HERE / rel)
        png = (HERE / rel).read_bytes()
        rec = {"id": f"{task}/{v}/code/s0", "task": task, "variant": v,
               "contender": "code", "seed": 0, "recipe": None,
               "settings": None, "prompt": None, "target": spec["target"],
               "lang": spec["lang"], "boxes": spec["boxes"], "source": None,
               "seconds": round(secs, 3), "png": rel,
               "sha256": hashlib.sha256(png).hexdigest()}
        path = HERE / "runs" / f"{task}.jsonl"
        old = [ln for ln in (path.read_text("utf8").splitlines()
                             if path.exists() else [])
               if json.loads(ln)["id"] != rec["id"]]
        path.write_text("".join(ln + "\n" for ln in old) +
                        json.dumps(rec, ensure_ascii=False) + "\n", "utf8")
        print(f"{rec['id']:32s} {secs * 1000:6.1f} ms")


if __name__ == "__main__":
    main()
