#!/usr/bin/env python
"""Drive Wan 2.2 14B image-to-video (GGUF, Lightx2v 4-step) through a running
ComfyUI over its HTTP API (start ComfyUI with --lowvram on an 8 GB card).

Settings mirror ComfyUI's own video_wan2_2_14B_i2v template (read 2026-09-28):
81 frames at 16 fps, ModelSamplingSD3 shift 5, Lightx2v LoRA at 1.0, 4 steps
at cfg 1 split 0-2 high-noise / 2-4 low-noise. Two swaps for the 8 GB card
and 16 GB of RAM: Q4_K_S GGUF experts instead of fp8, and a Q5_K_M GGUF
UMT5 instead of the 6.7 GB fp8 encoder.

THE NEGATIVE PROMPT IS NOT THE TEMPLATE'S. Its default negative contains
"style, artwork, painting" (风格，作品，画作) and "vivid colours" (色调艳丽),
which push an illustrated input towards a photograph - the opposite of what a
comic needs. Those terms are removed; the object-level ones stay (the same
rule local-image-gen measured for Qwen: negatives name objects, never looks).

usage:
  python wan_i2v.py <image already in ComfyUI/input/> "<motion prompt>" [-W 640 -H 640] [--frames 81] [--seed 1]
                   [--server http://127.0.0.1:8188] [--comfy path/to/ComfyUI]

needs, in ComfyUI/models/ (plus the ComfyUI-GGUF custom node for the .gguf loaders):
  diffusion_models/ Wan2.2-I2V-A14B-{High,Low}Noise-Q4_K_S.gguf   huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF
  loras/            wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors   Comfy-Org/Wan_2.2_ComfyUI_Repackaged
  text_encoders/    umt5-xxl-encoder-Q5_K_M.gguf                  huggingface.co/city96/umt5-xxl-encoder-gguf
  vae/              wan_2.1_vae.safetensors                       Comfy-Org/Wan_2.2_ComfyUI_Repackaged
"""
import argparse, json, os, sys, time, urllib.error, urllib.request, uuid

SERVER = "http://127.0.0.1:8188"


def post(path, body):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def get(path):
    return json.load(urllib.request.urlopen(SERVER + path))


def run(graph, timeout=5400):
    """Queue a graph, wait for it, return (output files, seconds)."""
    try:
        pid = post("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI rejected the graph ({e.code}):\n{e.read().decode()}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                sys.exit("FAILED:\n" + json.dumps(st.get("messages", []), indent=1)[:4000])
            outs = [f for node in h[pid].get("outputs", {}).values() for k in ("images", "videos", "gifs") for f in node.get(k, [])]
            return outs, time.time() - t0
        time.sleep(3)
    sys.exit("timed out")

HIGH = "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf"
LOW = "Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"
LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
CLIP = "umt5-xxl-encoder-Q5_K_M.gguf"
VAE = "wan_2.1_vae.safetensors"

# template negative minus 色调艳丽 (vivid colour), 风格/作品/画作/画面 (style,
# artwork, painting, picture) and 整体发灰 (greyish) - see header.
NEGATIVE = ("过曝，静态，细节模糊不清，字幕，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def build(image, prompt, negative, width, height, frames, seed, fps, prefix):
    def sampler(model, add_noise, start, end, leftover, latent, pos, neg):
        return {"class_type": "KSamplerAdvanced", "inputs": {
            "model": model, "add_noise": add_noise, "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": pos, "negative": neg,
            "latent_image": latent, "start_at_step": start, "end_at_step": end,
            "return_with_leftover_noise": leftover}}
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": HIGH}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": LOW}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": LORA_HIGH, "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": LORA_LOW, "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 5.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 5.0}},
        "7": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": CLIP, "type": "wan"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": negative}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image}},
        "12": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["9", 0], "negative": ["10", 0], "vae": ["8", 0], "start_image": ["11", 0],
            "width": width, "height": height, "length": frames, "batch_size": 1}},
        "13": sampler(["5", 0], "enable", 0, 2, "enable", ["12", 2], ["12", 0], ["12", 1]),
        "14": sampler(["6", 0], "disable", 2, 10000, "disable", ["13", 0], ["12", 0], ["12", 1]),
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": fps}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": prefix,
                                                     "format": "auto", "codec": "auto"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="filename already in ComfyUI/input/")
    ap.add_argument("prompt")
    ap.add_argument("-n", "--negative", default=NEGATIVE)
    ap.add_argument("-W", "--width", type=int, default=640)
    ap.add_argument("-H", "--height", type=int, default=640)
    ap.add_argument("--frames", type=int, default=81, help="4k+1; 81 = 5 s at 16 fps")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="video/wan22_i2v")
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--comfy", default=".", help="the ComfyUI folder, to print where the video landed")
    a = ap.parse_args()
    globals()["SERVER"] = a.server
    assert (a.frames - 1) % 4 == 0, "Wan wants 4k+1 frames"
    assert a.width % 16 == 0 and a.height % 16 == 0, "width/height must be multiples of 16"
    outs, secs = run(build(a.image, a.prompt, a.negative, a.width, a.height, a.frames, a.seed, a.fps, a.prefix),
                     timeout=5400)
    print(f"done in {secs:.0f}s  ({secs / (a.frames / a.fps):.0f} s of compute per second of video)")
    for im in outs:
        p = os.path.join(a.comfy, "output", im.get("subfolder", ""), im["filename"])
        print("  ", p, f"({os.path.getsize(p)} bytes)" if os.path.exists(p) else "(MISSING)")


if __name__ == "__main__":
    main()
