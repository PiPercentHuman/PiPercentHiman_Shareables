# anime-lora-wan: a chapter of a story to an anime episode

Turn a chapter of a story into a short anime episode on one consumer GPU: shots drawn from text by Wan 2.1 T2V-1.3B
with a general anime style LoRA (it draws anime from any prompt) and a LoRA per character, voices designed and cloned
per character, sound effects made from the picture, and a cut that times every shot to its line.

Everything runs locally. The LoRAs are on Hugging Face: https://huggingface.co/anyze/wan2.1-anime-lora (the anime
style LoRA works on its own for any anime scene; the two character LoRAs are examples to stack on it).

## What one chapter costs (measured)

| step | example chapter (23 shots) |
|---|---|
| shots: 2 seeds per shot, 49-81 frames, 832x480, 4 steps | 46 takes in 98 min (about 2 min per take) |
| a new character LoRA (16 images, 1,280 steps) | 21.5 min, plus the time to draw the images |
| voices: 3 candidates for each of 2 characters | 2 min 16 s |
| voices: lines | 2 lines in 62 s, model load included |
| sound effects (MMAudio, 23 shots) | 4 min 22 s |
| cut and mix | 38 s |

Picking takes by eye is the step that does not shrink: look at every sheet.

## Setup

1. [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) in its own venv (the generator and trainer run through it).
2. Model files:
   - `wan2.1_t2v_1.3B_bf16.safetensors` and `wan_2.1_vae.safetensors` from
     [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged)
   - `models_t5_umt5-xxl-enc-bf16.pth` from [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
   - `Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors` from
     [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) (`LoRAs/Wan2_1_self_forcing_1_3B/`)
   - the LoRAs from https://huggingface.co/anyze/wan2.1-anime-lora into one folder
3. Merge the Self-Forcing distillation into the base model once (musubi cannot load it as a LoRA):
   `python train/merge_distill.py <wan2.1_t2v_1.3B_bf16> <self_forcing_lora> <out>`; the output is `dit_fast`.
4. For voices: `pip install qwen-tts` (Qwen3-TTS; the 1.7B VoiceDesign and Base models download on first use).
   For sound effects: [MMAudio](https://github.com/hkchengrex/MMAudio) in its own venv.
   For checking voices: `faster-whisper`. And `ffmpeg` on the PATH.
5. `cp paths.example.json paths.json` and fill in your paths.

Only want the shots? The ComfyUI graph in `comfyui/` loads the Self-Forcing LoRA as a LoRA (no merge) and gives the
same picture.

## Make an episode

A chapter is one folder with a `chapter.json` (see `chapters/deep-sea-embers-ch2/`): the characters' voices, the
lines, every shot's prompt, seeds and LoRAs, and the edit. Run from this folder:

```
<musubi python> episode/gen.py chapters/<ch> run        # every shot, every seed -> takes/
python episode/gen.py chapters/<ch> sheet               # contact sheets: look at every take
python episode/gen.py chapters/<ch> pick c08_mirror=33  # the picks (chapter.json "pick", or named here)

<tts python> episode/voices.py chapters/<ch> design     # 3 candidate voices per character
python episode/pick_voices.py chapters/<ch> --write     # measure them, propose one each (then listen)
<tts python> episode/voices.py chapters/<ch> lines      # every line in the chosen voices
python episode/pick_voices.py chapters/<ch> --check     # transcribe every line against its text

python episode/cut.py chapters/<ch> build               # the picture, timed to the lines
<mmaudio python> episode/sfx.py chapters/<ch>           # sound effects per shot
python episode/cut.py chapters/<ch> mix                 # -> cut/episode.mp4
```

### Writing shots that work

- Start every prompt with `bldln style anime`, then the character's trigger and his full description every time.
  The LoRA holds the face, but the costume comes from the words.
- One LoRA set per run. A character LoRA changes everyone in the shot, so shots without that character name only
  the style LoRA (gen.py runs each LoRA set separately).
- Two seeds per shot, three for the hard ones. Mirrors, crowds and hidden faces need the most.
- A mirror gives a second person unless the shot is framed over the shoulder: "the back of his head fills the left
  side, a mirror on the right shows his reflection".
- Shots with a talking face are marked `"sfx_from_text_only": true`: with the picture, MMAudio reads the lips and
  makes murmured speech.

## Train your own

```
python train/prep_clips.py <long_video> data/style                       # shots, motion and cut checks, sheets
<vlm python> train/caption_clips.py data/style/train "mystyle anime"     # one sentence per clip (Qwen3-VL-2B)
<musubi python> train/train.py style data/style/train mystyle --trigger "mystyle anime"

<musubi python> train/train.py character data/hero hero --trigger "hro man"   # 15-16 images + captions
```

`train.py` ends by exporting the LoRA in the published layout, so ComfyUI loads it directly. `export_lora.py`
checks that every module's weight change survives the conversion and drops the training metadata, which records
local paths. The settings are the ones the published LoRAs were trained with (in `train.py`). How to draw a
character set: see the note at the top of `train.py`.

## Files

| | |
|---|---|
| `episode/gen.py` | shots from text through musubi (Self-Forcing, 4 steps, cfg 1) |
| `episode/lora_format.py` | published LoRA layout to musubi's, with the key prefix read from your DiT file |
| `episode/voices.py`, `pick_voices.py` | voice design, measured pick, cloned lines, transcript check |
| `episode/sfx.py` | MMAudio per shot |
| `episode/cut.py` | the cut (zoom, shake, hit-stop, impact frames, flash) and the mix |
| `train/` | clip prep, captions, training, export, the Self-Forcing merge |
| `comfyui/` | the ComfyUI API graph for shots |

Code: MIT. The models it downloads keep their own licences: Wan 2.1 and Qwen3-TTS are Apache 2.0, and MMAudio's
weights are CC BY-NC 4.0 (non-commercial). The LoRAs on Hugging Face are CC BY-NC 4.0.
