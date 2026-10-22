# Manhwa in motion, part 2: Wan 2.2 redraws every frame

[Part 1](../manhwa-motion-blender) moved a camera through layers of each panel, so nothing was redrawn
and nobody could act. Here an open video model, **Wan 2.2 image-to-video** (Alibaba, Apache 2.0),
redraws every frame from a panel and a motion prompt, on a home computer with an 8 GB graphics card.

`the-mix.mp4` is the chapter with both tools: the Blender camera for establishing shots, Wan for the
moments that need motion.

## Setup

ComfyUI plus the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node. Put these files in `ComfyUI/models/`:

| folder | file | size | from |
|---|---|---|---|
| `diffusion_models/` | `Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf` (lays out the motion) | 8.75 GB | [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) |
| `diffusion_models/` | `Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf` (fills in detail) | 8.75 GB | same |
| `loras/` | `wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors` (4 steps, not 40) | 2.46 GB | Comfy-Org Wan 2.2 repackaged |
| `text_encoders/` | `umt5-xxl-encoder-Q5_K_M.gguf` | 4.15 GB | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) |
| `vae/` | `wan_2.1_vae.safetensors` | 0.25 GB | Comfy-Org Wan 2.2 repackaged |

Start ComfyUI with `python main.py --lowvram`. With 16 GB of RAM the two experts don't fit, so they
stream from the page file (give Windows a large one). That streaming is where the time goes.

## Run

```bash
cp my_panel.png ComfyUI/input/
python wan_i2v.py my_panel.png "Her hair drifts in a light breeze; she blinks once. The camera is still." -W 560 -H 736
bash wan_batch.sh          # all 38 shots of the chapter, with their exact prompts
bash on_twos.sh clip.mp4 out/    # 16 / 12 / 8 drawings a second, side by side
bash reverse.sh clip.mp4 clip_rev.mp4
```

The settings copy ComfyUI's own `video_wan2_2_14B_i2v` template: 81 frames at 16 fps (5 s), shift 5,
4 steps at cfg 1, with the first two steps on the high-noise expert and the last two on the
low-noise expert.

## What we learned

- **The template's negative prompt fights comics.** It is written in Chinese and includes "style,
  artwork, painting" and "vivid colours". `wan_i2v.py` removes those words. This was a precaution,
  not an A/B test.
- **It closes open mouths.** Twelve of the 38 shots start on a strong expression. Eight relaxed within the
  5 s, including every wide-open mouth, 6 of 6 (a gasp, a scream, four of the dragon's roars and snarls),
  and that includes the three shots whose prompt asked for open jaws. Three held, and all three are
  clenched: gritted teeth, a grimace, a closed grin. One eased and partly came back. The characters stayed
  recognisable throughout. `faces.json` lists the twelve with what each face did and where it is in the
  frame. One take each and judged by eye, so this is a pattern worth testing, not a law.
- **Play it backwards.** A reversed clip builds up into the panel as drawn: on the dragon, the last
  frame is within 7.6/255 of the panel on average. Everything else runs backwards too.
- **Anime timing:** holding each drawing for 2 or 3 frames (`on_twos.sh`) makes it step the way TV
  animation does. Whether that reads as anime is your call.
- **Say what the camera does** in every prompt ("the camera is still" or "the camera tracks with them").

## Numbers (our run)

Each 5-second shot took a median of **389 s (about 6.5 minutes)**, and all 38 took 14,631 s (about 4 hours).
That is about **14×** the Blender camera rig per second of video (76.8 s against 5.3 s), on the same laptop.

Licence note: the panels are Qwen-Image 2.1 output, and its research licence does not allow commercial
use. Wan 2.2 is Apache 2.0.
