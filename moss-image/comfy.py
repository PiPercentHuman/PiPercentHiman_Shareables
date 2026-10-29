"""A small ComfyUI client and the graphs every trial uses.

Talks to a running ComfyUI (with native MOSS-Image support) over HTTP,
urllib only. Any error from the server stops the run: a failed call is
never written down as a result.
"""
import json
import os
import time
import urllib.error
import urllib.request
import uuid

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8189")

MOSS = "moss_image_sft_int8_convrot.safetensors"
MOSS_TE = "qwen3.5_9b_w4a8_gptq.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
MOSS_NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
EDIT_LORA = "moss_image_edit_lora_rank_64_bf16.safetensors"
FAST_LORA = "moss_dmd_8step_lora.safetensors"

# Three ways to run the same model.
#   card:     the model card (50 steps, CFG 3, Euler, skip layer 22)
#   template: ComfyUI's own MOSS template (25 steps, CFG 2.5, er_sde)
#   fast:     the 8-step distillation LoRA, no CFG
RECIPES = {
    "card": dict(steps=50, cfg=3.0, sampler="euler", slg=(0.0, 1.0)),
    "template": dict(steps=25, cfg=2.5, sampler="er_sde", slg=(0.0, 0.5)),
    "fast": dict(steps=8, cfg=1.0, sampler="euler", slg=None,
                 lora=(FAST_LORA, 1.0)),
    # the edit LoRA's own settings, as its author ships them
    "edit": dict(steps=30, cfg=4.0, sampler="euler", scheduler="normal",
                 slg=(0.0, 1.0), lora=(EDIT_LORA, 1.0)),
    # Qwen-Image 2.1, the settings of ComfyUI's own Qwen-Image 2.1 template
    "qwen": dict(steps=25, cfg=4.0, sampler="euler"),
}

QWEN = "qwen_image_2.1-Q4_K.gguf"
QWEN_TE = "qwen3vl_8b_int8_convrot.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"


class ComfyError(RuntimeError):
    pass


def _post(path, payload):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(HOST + path, timeout=60) as r:
        return json.loads(r.read())


def fetch(image):
    """The bytes of one output image, as ComfyUI saved them."""
    q = "filename={filename}&subfolder={subfolder}&type={type}".format(**image)
    with urllib.request.urlopen(f"{HOST}/view?{q}", timeout=60) as r:
        return r.read()


def upload(path):
    """Put a local image into ComfyUI's input folder; returns its name."""
    import os
    name = os.path.basename(path)
    bound = uuid.uuid4().hex
    crlf = "\r\n"
    head = crlf.join([f"--{bound}",
                      f'Content-Disposition: form-data; name="image";'
                      f' filename="{name}"',
                      "Content-Type: image/png", "", ""])
    tail = crlf.join(["", f"--{bound}",
                      'Content-Disposition: form-data; name="overwrite"',
                      "", "true", f"--{bound}--", ""])
    body = head.encode() + open(path, "rb").read() + tail.encode()
    req = urllib.request.Request(
        HOST + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={bound}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def run(graph, timeout=3600):
    """Queue a graph, wait for it, return (png bytes, seconds)."""
    try:
        pid = _post("/prompt", {"prompt": graph,
                                "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise ComfyError(f"rejected ({e.code}): {e.read().decode()[:2000]}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise ComfyError(json.dumps(st.get("messages"))[:3000])
            ims = [im for out in h[pid]["outputs"].values()
                   for im in out.get("images", [])]
            if len(ims) != 1:
                raise ComfyError(f"expected one image, got {len(ims)}")
            return fetch(ims[0]), time.time() - t0
        time.sleep(1)
    raise ComfyError("timed out")


def moss(prompt, seed, recipe="card", width=1024, height=1024,
         refs=(), prefix="moss"):
    """MOSS-Image text to image (or an edit, with reference images)."""
    r = RECIPES[recipe]
    g = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": MOSS, "weight_dtype": "default"}},
        "te": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": MOSS_TE, "type": "moss_image",
                          "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MOSS_VAE}},
        "latent": {"class_type": "EmptyFlux2LatentImage",
                   "inputs": {"width": width, "height": height,
                              "batch_size": 1}},
    }
    model = ["unet", 0]
    lora = r.get("lora")
    if lora:
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": model, "lora_name": lora[0],
                                "strength_model": lora[1]}}
        model = ["lora", 0]
    if r["slg"]:
        g["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                    "inputs": {"model": model, "double_layers": "",
                               "single_layers": "22",
                               "start_percent": r["slg"][0],
                               "end_percent": r["slg"][1]}}
        model = ["slg", 0]
    for name, text in (("pos", prompt), ("neg", MOSS_NEG)):
        if refs:
            ins = {"clip": ["te", 0], "prompt": text, "vae": ["vae", 0],
                   "edit_format": "Moss-Image-Edit", "image_strength": 1.0}
            for i, ref in enumerate(refs):
                g[f"ref{i}"] = {"class_type": "LoadImage",
                                "inputs": {"image": ref}}
                ins[f"images.image_{i}"] = [f"ref{i}", 0]
            g[name] = {"class_type": "TextEncodeMOSSImageEdit", "inputs": ins}
        else:
            g[name] = {"class_type": "CLIPTextEncode",
                       "inputs": {"clip": ["te", 0], "text": text}}
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": seed, "steps": r["steps"], "cfg": r["cfg"],
        "sampler_name": r["sampler"],
        "scheduler": r.get("scheduler", "simple"),
        "positive": ["pos", 0], "negative": ["neg", 0],
        "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return g


def qwen(prompt, seed, width=1024, height=1024, prefix="qwen"):
    """Qwen-Image 2.1 (Q4_K GGUF) text to image, on a ComfyUI that has it."""
    r = RECIPES["qwen"]
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN}},
        "te": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": QWEN_TE, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "enc": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["te", 0], "prompt": prompt, "negative_prompt": "",
            "resolution": 1024, "vae": ["vae", 0]}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height,
                              "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "seed": seed, "steps": r["steps"],
            "cfg": r["cfg"], "sampler_name": r["sampler"],
            "scheduler": "simple", "positive": ["enc", 0],
            "negative": ["enc", 1], "latent_image": ["latent", 0],
            "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
