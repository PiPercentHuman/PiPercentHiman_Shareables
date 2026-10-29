"""Read an image's text back with OCR and score it against a target.

  read      target words (or Chinese characters) found in the image
  of        how many there were to find
  boxes     for layout tasks: each text item whose words landed inside
            the box it was given (with a small margin)

The reader is EasyOCR (a neural scene-text reader) on the CPU, so it
gives the same answer every time. OCR can miss stylised lettering, so a
miss means "not read", checked by eye before anyone calls it a typo.

  python score.py            scores every image in runs/*.jsonl
                             -> runs/scores.jsonl
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = os.environ.get("EASYOCR_MODELS", str(Path.home() / ".EasyOCR" / "model"))  # EasyOCR's own default folder
MARGIN = 48
_readers = {}


def reader(lang):
    if lang not in _readers:
        import easyocr
        langs = ["ch_sim", "en"] if lang == "chi_sim" else ["en"]
        _readers[lang] = easyocr.Reader(langs, gpu=False, verbose=False,
                                        model_storage_directory=MODELS,
                                        download_enabled=False)
    return _readers[lang]


def norm(w):
    return re.sub(r"[^0-9A-Z]", "", w.upper())


def ocr(path, lang="eng"):
    """Every line the reader finds: its text, confidence and box."""
    out = []
    for quad, text, conf in reader(lang).readtext(str(path)):
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        out.append({"text": text, "conf": round(float(conf), 3),
                    "box": [int(min(xs)), int(min(ys)),
                            int(max(xs)), int(max(ys))]})
    return out


def units(text, lang):
    """What gets counted: words in English, characters in Chinese."""
    if lang == "chi_sim":
        return [c for c in text if not c.isspace()]
    return [norm(w) for w in text.split() if norm(w)]


def found(target, lines, lang):
    want = Counter(units(target, lang))
    got = Counter(u for ln in lines for u in units(ln["text"], lang))
    return sum(min(got[k], n) for k, n in want.items())


def in_box(item_text, box, lines):
    """Did this text item land in its box? The lines that carry its words
    must have their centres inside the box, give or take MARGIN pixels."""
    x0, y0, x1, y1 = box
    want = set(units(item_text, "eng"))
    hits = [ln for ln in lines if want & set(units(ln["text"], "eng"))]
    if not hits:
        return None                      # not read at all: no verdict
    inside = [ln for ln in hits
              if x0 - MARGIN <= (ln["box"][0] + ln["box"][2]) / 2 <= x1 + MARGIN
              and y0 - MARGIN <= (ln["box"][1] + ln["box"][3]) / 2 <= y1 + MARGIN]
    return len(inside) * 2 > len(hits)


def score(path, target, lang="eng", boxes=None):
    lines = ocr(path, lang)
    out = {"read": found(target, lines, lang),
           "of": len(units(target, lang)), "lines": lines}
    if boxes:
        out["boxes"] = [{"text": text, "in_box": in_box(text, box, lines)}
                        for text, box in boxes]
    return out


def main():
    done = {}
    dst = HERE / "runs" / "scores.jsonl"
    if dst.exists():
        for ln in dst.read_text("utf8").splitlines():
            r = json.loads(ln)
            done[r["sha256"]] = r
    rows = []
    for f in sorted((HERE / "runs").glob("*.jsonl")):
        if f.name == "scores.jsonl":
            continue
        for ln in f.read_text("utf8").splitlines():
            rec = json.loads(ln)
            if not rec.get("target"):
                continue
            if rec["sha256"] in done:
                rows.append(done[rec["sha256"]])
                continue
            s = score(HERE / rec["png"], rec["target"], rec["lang"],
                      rec.get("boxes"))
            row = {"id": rec["id"], "sha256": rec["sha256"], **s}
            rows.append(row)
            print(f"{rec['id']:40s} {s['read']:3d}/{s['of']}", flush=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in rows), "utf8")


if __name__ == "__main__":
    sys.exit(main())
