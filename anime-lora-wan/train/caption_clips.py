"""Captions for training clips: Qwen3-VL-2B-Instruct, local, from 4 frames of each clip.

  python train/caption_clips.py <folder> [trigger]   -> <folder>/<clip>.txt for every <clip>.mp4

One sentence each: who is on screen (appearance, never names), what happens, where, the light, and the camera. The
trigger is prefixed so the style can be called. Nothing is typed by hand; captions.jsonl keeps the raw replies.
"""
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TRIGGER = "bldln style anime"  # the published style LoRA was trained with this trigger
ASK = ("These are 4 frames, in order, from one short anime clip. Write ONE sentence (at most 40 words) that a video "
       "generator could use as a prompt: describe the characters by appearance (no names), what they do and how they "
       "move, the setting, the lighting and colours, and the camera movement. Do not mention text, logos or frames.")


def frames(path, n=4):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    idx = [int(i * (len(fr) - 1) / (n - 1)) for i in range(n)]
    h, w = fr[0].shape[:2]
    k = 480 / max(h, w)  # long side 480, aspect kept
    return [Image.fromarray(cv2.cvtColor(fr[i], cv2.COLOR_BGR2RGB)).resize((round(w * k), round(h * k))) for i in idx]


def main(folder):
    folder = Path(folder)
    mid = "Qwen/Qwen3-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0").eval()
    log = open(folder / "captions.jsonl", "a", encoding="utf8")
    clips = sorted(folder.glob("*.mp4"))
    for k, c in enumerate(clips):
        txt = c.with_suffix(".txt")
        if txt.exists():
            continue
        ims = frames(c)
        msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in ims), {"type": "text", "text": ASK}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=90, do_sample=False)
        reply = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        reply = " ".join(reply.split())
        txt.write_text(f"{TRIGGER}, {reply}", "utf8")
        log.write(json.dumps({"clip": c.name, "reply": reply}, ensure_ascii=False) + "\n"); log.flush()
        if k % 25 == 0:
            print(k, len(clips), c.name, reply[:120], flush=True)
    print("CAPTION-END", len(clips))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        TRIGGER = sys.argv[2]
    main(sys.argv[1])
