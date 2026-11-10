"""Train a style or character LoRA on Wan 2.1 T2V-1.3B with musubi-tuner, then export it in the published format.

  python train/train.py style <clip_dir> <name>        clips + captions (prep_clips.py, caption_clips.py)
  python train/train.py character <image_dir> <name>  15-16 images of one character + one caption .txt per image

Settings are the ones the published LoRAs were trained with:
  style      rank 32, lr 2e-4, 4 epochs, clips at their own size (192x320 for ours), 33 frames, repeats 1
  character  rank 16, lr 1e-4, 8 epochs, images in buckets around 512x512, each image repeated 10 times per epoch
Both: fp8 base, AdamW8bit, gradient checkpointing, flow shift 3, bf16, a checkpoint every epoch (style) or every 2
(character). Override with --rank/--lr/--epochs/--repeats/--size.
Output: runs/<name>/ (musubi checkpoints, dataset.toml, cache), then <loras>/wan2.1-t2v-1.3b-<name>.safetensors in the
published (ComfyUI) format via export_lora.py.

Character images: draw them in the style the episode uses (for example, frames of the character generated with the
style LoRA, redrawn from several angles by any image model), one character per set, the same costume in every
image. Caption each one "<trigger> man, <view>, <expression>, <background>" and nothing about his looks: the images
carry the looks, the trigger collects them.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "episode"))
from config import REPO, paths  # noqa: E402

DEFAULTS = {"style": dict(rank=32, lr=2e-4, epochs=4, repeats=1, save_every=1),
            "character": dict(rank=16, lr=1e-4, epochs=8, repeats=10, save_every=2)}


def dataset_toml(kind, data, cache, size, repeats):
    w, h = size
    if kind == "style":
        return (f'[general]\nresolution = [{w}, {h}]\ncaption_extension = ".txt"\nbatch_size = 1\nnum_repeats = {repeats}\n'
                f'enable_bucket = false\n\n[[datasets]]\nvideo_directory = "{data.as_posix()}"\ncache_directory = "{cache.as_posix()}"\n'
                f'target_frames = [33]\nframe_extraction = "head"\n')
    return (f'[general]\nresolution = [{w}, {h}]\ncaption_extension = ".txt"\nbatch_size = 1\nenable_bucket = true\n'
            f'bucket_no_upscale = false\n\n[[datasets]]\nimage_directory = "{data.as_posix()}"\ncache_directory = "{cache.as_posix()}"\n'
            f'num_repeats = {repeats}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["style", "character"])
    ap.add_argument("data")
    ap.add_argument("name")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--size", help="WxH; default 192x320 for style clips, 512x512 for character images")
    ap.add_argument("--trigger", default="", help="written into the exported file's metadata")
    a = ap.parse_args()
    cfg = dict(DEFAULTS[a.kind])
    cfg.update({k: v for k, v in (("rank", a.rank), ("lr", a.lr), ("epochs", a.epochs), ("repeats", a.repeats)) if v})
    size = tuple(map(int, (a.size or ("192x320" if a.kind == "style" else "512x512")).split("x")))
    P = paths()
    run = REPO / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    toml = run / "dataset.toml"
    toml.write_text(dataset_toml(a.kind, Path(a.data).resolve(), run / "cache", size, cfg["repeats"]), "utf8")
    py, M = P["musubi_python"], Path(P["musubi_tuner"]) / "src/musubi_tuner"
    sh = dict(check=True, cwd=P["musubi_tuner"])
    subprocess.run([py, str(M / "wan_cache_latents.py"), "--dataset_config", str(toml), "--vae", P["vae"], "--vae_cache_cpu"], **sh)
    subprocess.run([py, str(M / "wan_cache_text_encoder_outputs.py"), "--dataset_config", str(toml), "--t5", P["t5"],
                    "--fp8_t5", "--batch_size", "4"], **sh)
    subprocess.run([py, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_cpu_threads_per_process", "1",
                    "--mixed_precision", "bf16", str(M / "wan_train_network.py"), "--task", "t2v-1.3B", "--dit", P["dit"],
                    "--dataset_config", str(toml), "--sdpa", "--mixed_precision", "bf16", "--fp8_base",
                    "--optimizer_type", "adamw8bit", "--learning_rate", str(cfg["lr"]), "--gradient_checkpointing",
                    "--max_data_loader_n_workers", "1", "--persistent_data_loader_workers",
                    "--network_module", "networks.lora_wan", "--network_dim", str(cfg["rank"]),
                    "--timestep_sampling", "shift", "--discrete_flow_shift", "3.0", "--max_train_epochs", str(cfg["epochs"]),
                    "--save_every_n_epochs", str(cfg["save_every"]), "--seed", "42",
                    "--output_dir", str(run), "--output_name", a.name], **sh)
    out = Path(P["loras"]) / f"wan2.1-t2v-1.3b-{a.name}.safetensors"
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_lora.py")), str(run / f"{a.name}.safetensors"), str(out),
                    "--title", a.name, "--trigger", a.trigger], check=True)
    print("TRAIN-END", out)


if __name__ == "__main__":
    main()
