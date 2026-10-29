# moss-image

MOSS-Image (OpenMOSS, 5B, Apache-2.0) on the jobs people use an image model for: a photo, an anime scene, a poster
title, menus of three, eight and twenty dishes, a concert poster as a sentence and as a JSON layout, a three-panel
comic, a Chinese shop sign and an edit. Six seeds a job, beside the same model with the 8-step DMD LoRA, the model
card's 50-step recipe, Qwen-Image 2.1 and a few lines of plain code. 138 generated images
(102 MOSS, 36 Qwen); every image read back by OCR, 130 read by eye.

## Run it

1. A ComfyUI build with native MOSS-Image support (master after PR #17095) and, from Comfy-Org/MOSS-Image:
   `moss_image_sft_int8_convrot.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors`, `flux2-vae_small.safetensors`,
   `moss_image_edit_lora_rank_64_bf16.safetensors`; and `lora.safetensors` from lodestones/MOSS-Image-SFT-8-steps-dmd
   saved as `moss_dmd_8step_lora.safetensors`. Start it on port 8189 (or set COMFY_HOST).
2. `python run.py` writes one record per image to `runs/<task>.jsonl` and the image to `images/`.
   `COMFY_HOST=... python run.py --contender qwen` runs Qwen-Image 2.1 on a ComfyUI that has it.
3. `python baseline.py` draws the plain-code pages.
4. `python score.py` (needs `easyocr`; set `EASYOCR_MODELS` to keep its models elsewhere) reads every image back -> `runs/scores.jsonl`.
5. `python eyecheck.py` writes what a person read in each image -> `eyecheck.jsonl`.
6. `python replay.py` checks every record against its image and its score.

`tasks.py` holds every prompt and layout; `comfy.py` the graphs and the three recipes.

## Results

| job | MOSS (ComfyUI template) | MOSS fast (8 steps) | MOSS card (50 steps) | Qwen-Image 2.1 |
|---|---|---|---|---|
| Poster title (2 words) | 6 of 6 | 6 of 6 | 6 of 6 | 6 of 6 |
| Menu, 8 dishes: every dish right | 6 of 6 | 6 of 6 | 6 of 6 | 0 of 6 |
| Menu, 20 dishes one a line: every word right | 5 of 6 | - | - | 0 of 6 |
| Menu, 3 dishes: made-up dishes added | 4 of 6 | - | - | 3 of 6 |

- Twenty dishes as one comma sentence instead of one a line: invented text on 6 of 6 cards
  (one a line: 0 of 6). OCR read 238 of 240 words one a line,
  203 of 240 with commas.
- Three dishes with "the card has a lot of empty space and only 3 lines of text, nothing else": made-up dishes on
  0 of 6 cards (a garbled heading on 2 of 6).
- Concert poster: title on one line inside its box on 6 of 6 with a JSON layout,
  0 of 6 as a sentence (every word spelled right either way).
- Comic, three speech bubbles: 4 of 4. Chinese sign and lantern: 6 of 6 by eye.
- Edit ("change only the title"): 6 of 6; mean difference below the title against its source
  3.0-11.2 of 255, against an unrelated poster 47.6-53.6.
- Median seconds a picture: template 158, fast 28, card 323, edit 299,
  Qwen 164; plain code under 414 ms.

## What is not here, and why

- The full-size bf16 weights: the files are the compressed ones ComfyUI ships (int8 model, w4a8 text encoder);
  Qwen-Image 2.1 is the Q4_K GGUF.
- Picture quality is shown, never scored. Six seeds a job show a tendency, not a rate.
- OCR misses spaced capitals and Chinese (on clean plain-code type it read 7 of
  8 Chinese characters), so those jobs are judged by eye; the criteria are in `eyecheck.py`.
- The prompts are ours; a better prompt could move a result.
- The colour/depth control LoRA was not tested.
- Images are regenerated from prompt and seed, but a graphics card is not bit-exact from run to run: `replay.py`
  re-checks the saved images, and `--regen N` reports how far a fresh render lands from the saved one.
