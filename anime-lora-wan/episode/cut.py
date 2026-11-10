"""Assemble the episode: the picked takes in the chapter's order, timed to the lines, then the sound mix.

  python episode/cut.py <chapter> build   -> cut/picture.mp4 (silent, 1280x720, 24 fps) + cut/timeline.json
  python episode/cut.py <chapter> mix     -> cut/episode.mp4 (picture + dialogue + sfx/ if present)

chapter.json -> scenes: each scene is a list of shots with
  seconds   how long the shot runs (a shot with a line runs at least line_at + the line + 0.35 s; the take is
            stretched or slowed to fit, so a 2 s take under a 4 s line plays at half speed)
  zoom      [start, end] push-in, 1 = the full frame
  line      the id of the line spoken over it, line_at = when it starts, in seconds into the shot
  fx        [kind, at_seconds, amount]: "shake" (decaying offset, amount in pixels), "hitstop" (hold the picture for
            amount frames), "impact" (amount frames of stark black/white), "flash" (white, fading over amount frames)
Scenes are separated by gap_between_scenes seconds of black. The episode ends on a 3 s title card (chapter "title").
The mix sets every line to the same loudness (-20 dB RMS over its voiced part) and ducks the effects under speech.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import chapter, paths  # noqa: E402

W, H, FPS = 1280, 720, 24


def read(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    st = json.loads(p.stdout)["streams"][0]
    w, h = st["width"], st["height"]
    num, den = map(int, st["r_frame_rate"].split("/"))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), num / den


def line_len(d, lid):
    import soundfile as sf
    return sf.info(str(d / "voices" / f"{lid}.wav")).duration


def impact(fr, k):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    b = 255 - b if k % 2 == 0 else b
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def render_shot(d, s, rng):
    frames, fps = read(d / "takes" / "pick" / f"{s['shot']}.mp4")
    n_src, (h, w) = len(frames), frames.shape[1:3]
    t0, t1 = 0.0, (n_src - 1) / fps
    dur = s["seconds"]
    if "line" in s:
        dur = max(dur, s.get("line_at", 0.3) + line_len(d, s["line"]) + 0.35)
    n = int(round(dur * FPS))
    fx = s.get("fx", [])
    holds = [(int(a * FPS), int(m)) for k, a, m in fx if k == "hitstop"]
    src_t, held = [], 0
    for i in range(n):  # source time per output frame, with holds inserted at contact
        held += any(a <= i < a + m for a, m in holds)
        u = (i - held) / max(1, n - 1 - sum(m for _, m in holds))
        src_t.append(t0 + min(1.0, u) * (t1 - t0))
    bw, bh = (w, w * 9 / 16) if w / h <= 16 / 9 else (h * 16 / 9, h)  # largest 16:9 window, then zoom
    shake = [(int(a * FPS), m) for k, a, m in fx if k == "shake"]
    out = []
    for i in range(n):
        u = i / max(1, n - 1)
        z = np.interp(u, [0, 1], s.get("zoom", (1.0, 1.0)))
        cw, ch = bw / z, bh / z
        x0, y0 = 0.5 * (w - cw), 0.5 * (h - ch)
        for a, m in shake:
            if a <= i < a + 8:
                dd = m * (1 - (i - a) / 8)
                x0 += rng.uniform(-dd, dd) * w / W; y0 += rng.uniform(-dd, dd) * h / H
        x0 = float(np.clip(x0, 0, w - cw)); y0 = float(np.clip(y0, 0, h - ch))
        fi = min(n_src - 1, int(src_t[i] * fps))
        M = np.array([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]], np.float32)
        fr = cv2.warpAffine(frames[fi], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        for k, a, m in fx:
            a = int(a * FPS)
            if k == "impact" and a <= i < a + m:
                fr = impact(fr, i - a)
            if k == "flash" and a <= i < a + m:
                fr = cv2.addWeighted(fr, 1 - (1 - (i - a) / m) * 0.85, np.full_like(fr, 255), (1 - (i - a) / m) * 0.85, 0)
        out.append(fr)
    return out


def font(size):
    f = paths().get("title_font") or ""
    for p in (f, "C:/Windows/Fonts/georgia.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"):
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def end_card(title):
    im = Image.new("RGB", (W, H), "black")
    dr = ImageDraw.Draw(im)
    for text, size, y in ((title[0], 64, H / 2 - 60), (title[1], 32, H / 2 + 20)):
        f = font(size)
        dr.text(((W - dr.textlength(text, font=f)) / 2, y), text, font=f, fill=(225, 215, 190))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def build(d, ch):
    out_dir = d / "cut"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                            "-pix_fmt", "yuv420p", str(out_dir / "picture.mp4")], stdin=subprocess.PIPE)
    t, tl = 0.0, {"fps": FPS, "shots": [], "lines": []}
    gap = int(ch.get("gap_between_scenes", 0.4) * FPS)
    for si, scene in enumerate(ch["scenes"]):
        if si:
            for _ in range(gap):
                enc.stdin.write(np.zeros((H, W, 3), np.uint8).tobytes())
            t += gap / FPS
        for s in scene["shots"]:
            frames = render_shot(d, s, rng)
            if "line" in s:
                tl["lines"].append({"id": s["line"], "at": round(t + s.get("line_at", 0.3), 3), "len": round(line_len(d, s["line"]), 3)})
            for fr in frames:
                enc.stdin.write(np.ascontiguousarray(fr).tobytes())
            tl["shots"].append({"scene": scene["name"], "name": s["shot"], "at": round(t, 3), "dur": round(len(frames) / FPS, 3)})
            t += len(frames) / FPS
            print(f"{scene['name']:8s} {s['shot']:20s} {len(frames) / FPS:5.2f}s  at {t:6.2f}", flush=True)
    if ch.get("title"):
        card = end_card(ch["title"])
        for _ in range(3 * FPS):
            enc.stdin.write(card.tobytes())
        t += 3.0
    enc.stdin.close(); enc.wait()
    tl["seconds"] = round(t, 3)
    (out_dir / "timeline.json").write_text(json.dumps(tl, indent=1), "utf8")
    print("CUT-END", round(t, 2), "s")


def mono(path, sr):
    import soundfile as sf
    a, r = sf.read(str(path), dtype="float32")
    a = a if a.ndim == 1 else a.mean(1)
    if r != sr:
        a = np.interp(np.arange(int(len(a) * sr / r)) / sr, np.arange(len(a)) / r, a).astype(np.float32)
    return a


def mix(d, ch):
    import soundfile as sf
    c = d / "cut"
    tl = json.loads((c / "timeline.json").read_text("utf8"))
    sr = 48000
    n = int((tl["seconds"] + 0.5) * sr)
    voice, fx = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for ln in tl["lines"]:
        a = mono(d / "voices" / f"{ln['id']}.wav", sr)
        v = a[np.abs(a) > 0.01]  # every line to the same voiced loudness: cloned takes can come out 10 dB apart
        if len(v):
            a = a * 10 ** ((-20 - 20 * np.log10(np.sqrt((v ** 2).mean()))) / 20)
        i = int(ln["at"] * sr)
        voice[i:i + len(a)] += a[:n - i]
    for shot in tl["shots"]:
        f = d / "sfx" / f"{shot['name']}.wav"
        if not f.exists():
            continue
        a = mono(f, sr)[:int(shot["dur"] * sr)]
        fade = min(len(a), int(0.03 * sr))
        a[:fade] *= np.linspace(0, 1, fade); a[len(a) - fade:] *= np.linspace(1, 0, fade)
        i = int(shot["at"] * sr)
        fx[i:i + len(a)] += a[:n - i]
    env = np.convolve((np.abs(voice) > 0.01).astype(np.float32), np.ones(int(0.15 * sr)) / int(0.15 * sr), "same")
    fx *= (1 - 0.6 * np.clip(env * 3, 0, 1)) * 0.6  # duck the effects under dialogue
    peak = max(1e-6, np.abs(voice + fx).max())
    sf.write(str(c / "audio.wav"), (voice + fx) / peak * 0.89, sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c / "picture.mp4"), "-i", str(c / "audio.wav"), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(c / "episode.mp4")], check=True)
    print("MIX-END", c / "episode.mp4")


if __name__ == "__main__":
    d, ch = chapter(sys.argv[1])
    {"build": build, "mix": mix}[sys.argv[2]](d, ch)
