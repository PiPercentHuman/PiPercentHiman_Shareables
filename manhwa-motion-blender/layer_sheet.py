"""One panel's layers side by side, each on a checkerboard (transparent = checks), labelled.
usage: python layer_sheet.py <layersDir> <out.png> [--h 640]"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("layers"); ap.add_argument("out"); ap.add_argument("--h", type=int, default=640)
a = ap.parse_args()
src = Path(a.layers); meta = json.loads((src / "layers.json").read_text())
names = ["BACKDROP", "MIDDLE", "FRONT"] if len(meta["layers"]) == 3 else [f"LAYER {i}" for i in range(len(meta["layers"]))]
tiles = []
for L, name in zip(meta["layers"], names):
    im = Image.open(src / L["file"]).convert("RGBA")
    w = round(im.width * a.h / im.height); im = im.resize((w, a.h), Image.LANCZOS)
    chk = Image.new("RGBA", im.size, "#3a3a38"); d = ImageDraw.Draw(chk)
    for y in range(0, a.h, 24):
        for x in range(0, w, 24):
            if (x // 24 + y // 24) % 2:
                d.rectangle([x, y, x + 23, y + 23], fill="#4a4a47")
    chk.alpha_composite(im)
    lab = Image.new("RGBA", (w, 56), "#141413"); ImageDraw.Draw(lab).text((8, 12), f"{name} · own {L['ownShare']:.0%} · filled {L['filledShare']:.0%}",
                                                                        fill="#e8e6dc", font=ImageFont.load_default(size=26))
    t = Image.new("RGBA", (w, a.h + 56)); t.paste(chk, (0, 0)); t.paste(lab, (0, a.h)); tiles.append(t)
gap = 24
sheet = Image.new("RGBA", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), a.h + 56), "#141413")
x = 0
for t in tiles:
    sheet.paste(t, (x, 0)); x += t.width + gap
sheet.convert("RGB").save(a.out)
print(a.out, sheet.size)
