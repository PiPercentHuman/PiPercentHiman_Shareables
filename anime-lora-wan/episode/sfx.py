"""Sound effects for every shot, from the picture and a short prompt (MMAudio large_44k_v2, local).

  <mmaudio python> episode/sfx.py <chapter>   -> <chapter>/sfx/<shot>.wav   (needs cut/timeline.json: run cut.py build first)

Each shot is cut out of the silent picture and given to MMAudio with the shot's "sfx" prompt; it syncs the sound to
what it sees. Shots marked "sfx_from_text_only" get the prompt alone: with a talking face or a moving jaw in the
picture, MMAudio reads the lips and makes murmured speech. Voices and music are kept out by the negative prompt;
dialogue comes from voices/. Shots under 1 s are given 1 s and trimmed back in the mix.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import chapter, paths  # noqa: E402

sys.path.insert(0, paths()["mmaudio"])
from mmaudio.eval_utils import all_model_cfg, generate, load_video  # noqa: E402
from mmaudio.model.flow_matching import FlowMatching  # noqa: E402
from mmaudio.model.networks import get_my_mmaudio  # noqa: E402
from mmaudio.model.utils.features_utils import FeaturesUtils  # noqa: E402

NEG = "speech, talking, voice, words, music, singing"


def main(d, ch):
    os.chdir(paths()["mmaudio"])  # MMAudio finds (and downloads) its weights relative to the working directory
    tl = json.loads((d / "cut" / "timeline.json").read_text("utf8"))
    out = d / "sfx"
    tmp = out / "_clips"
    tmp.mkdir(parents=True, exist_ok=True)
    model = all_model_cfg["large_44k_v2"]
    model.download_if_needed()
    seq_cfg = model.seq_cfg
    dev, dtype = "cuda", torch.bfloat16
    net = get_my_mmaudio(model.model_name).to(dev, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=dev, weights_only=True))
    fu = FeaturesUtils(tod_vae_ckpt=model.vae_path, synchformer_ckpt=model.synchformer_ckpt, enable_conditions=True,
                       mode=model.mode, bigvgan_vocoder_ckpt=model.bigvgan_16k_path, need_vae_encoder=False).to(dev, dtype).eval()
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)
    for s in tl["shots"]:
        f = out / f"{s['name']}.wav"
        if f.exists():
            continue
        shot = ch["shots"][s["name"]]
        dur = max(1.0, s["dur"])
        clip = tmp / f"{s['name']}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s['at']:.3f}", "-t", f"{dur:.3f}",
                        "-i", str(d / "cut" / "picture.mp4"), "-an", "-c:v", "libx264", "-crf", "18", str(clip)], check=True)
        info = load_video(clip, dur)
        seq_cfg.duration = info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        rng = torch.Generator(device=dev)
        rng.manual_seed(42)
        blind = shot.get("sfx_from_text_only", False)
        with torch.inference_mode():
            audio = generate(None if blind else info.clip_frames.unsqueeze(0), None if blind else info.sync_frames.unsqueeze(0),
                             [shot["sfx"]], negative_text=[NEG], feature_utils=fu, net=net, fm=fm, rng=rng, cfg_strength=4.5)
        torchaudio.save(str(f), audio.float().cpu()[0], seq_cfg.sampling_rate)
        print("sfx", s["name"], round(info.duration_sec, 2), flush=True)
    print("SFX-END")


if __name__ == "__main__":
    main(*chapter(sys.argv[1]))
