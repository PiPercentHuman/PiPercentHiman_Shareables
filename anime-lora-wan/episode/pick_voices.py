"""Measure the designed voice candidates and propose one per character, so the pick is not a guess (overrule by ear).

  python episode/pick_voices.py <chapter>          -> voices/design/measure.json, proposal printed
  python episode/pick_voices.py <chapter> --write  -> also writes the proposal into chapter.json (voice_pick)
  python episode/pick_voices.py <chapter> --check  -> transcribes every line in voices/ against its text

Per candidate: a faster-whisper transcript and its character error rate against the sentence it was asked to say,
the median pitch of voiced frames, the longest silence, and the length. The proposal is the lowest error rate, ties
broken by the shorter longest silence. A proposal with an error rate above 0.15 is flagged: it says something else.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import chapter  # noqa: E402


def norm(t):
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", t).lower())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[len(h)] / max(1, len(r))


def pitch(a, sr):
    n = int(0.04 * sr)
    f0 = []
    for k in range(0, len(a) - n, n):
        x = a[k:k + n] - a[k:k + n].mean()
        if np.sqrt((x ** 2).mean()) < 0.02:
            continue
        ac = np.correlate(x, x, "full")[n - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            f0.append(sr / lag)
    return float(np.median(f0)) if f0 else 0.0


def longest_silence(a, sr):
    n = int(0.02 * sr)
    best = run = 0
    for k in range(0, len(a) - n, n):
        run = run + 1 if np.sqrt((a[k:k + n] ** 2).mean()) < 0.01 else 0
        best = max(best, run)
    return best * 0.02


def heard(model, f):
    segs, _ = model.transcribe(str(f), language="en", beam_size=5)
    return "".join(s.text for s in segs)


def main(d, ch, args):
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    if "--check" in args:
        bad = 0
        for ln in ch["lines"]:
            h = heard(model, d / "voices" / f"{ln['id']}.wav")
            e = cer(ln.get("say", ln["text"]), h)
            bad += e > 0.15
            print(f"{ln['id']:4s} cer {e:.2f} | {h.strip()}")
        print("CHECK-END", len(ch["lines"]), "lines,", bad, "flagged")
        return
    res = {}
    for name, c in ch["characters"].items():
        rows = []
        for f in sorted((d / "voices" / "design").glob(f"{name}-s*.wav")):
            a, sr = sf.read(f, dtype="float32")
            a = a if a.ndim == 1 else a.mean(1)
            h = heard(model, f)
            rows.append({"file": f.name, "seconds": round(len(a) / sr, 2), "cer": round(cer(c["voice_ref_text"], h), 3),
                         "f0_hz": round(pitch(a, sr), 1), "longest_silence_s": round(longest_silence(a, sr), 2), "heard": h})
            print(name, rows[-1], flush=True)
        best = min(rows, key=lambda r: (r["cer"], r["longest_silence_s"]))
        res[name] = {"candidates": rows, "pick": best["file"], "flag": best["cer"] > 0.15}
        if "--write" in args:
            c["voice_pick"] = best["file"]
    (d / "voices" / "design" / "measure.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf8")
    if "--write" in args:
        (d / "chapter.json").write_text(json.dumps(ch, indent=1, ensure_ascii=False), "utf8")
    print(json.dumps({k: v["pick"] for k, v in res.items()}))


if __name__ == "__main__":
    d, ch = chapter(sys.argv[1])
    main(d, ch, sys.argv[2:])
