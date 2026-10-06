# Manhwa in motion, part 1: a multiplane camera in Blender

This turns flat comic panels into short camera shots **without redrawing a single line**. Each panel
is split into depth layers, and Blender moves a camera through them. It's the multiplane trick
(Lotte Reiniger 1926, Disney's *The Old Mill* 1937) done with free tools on a home computer.

`chapter-in-motion.mp4` is all 38 panels of our chapter
([the panels](../elflands-daughter-manhwa)), 4 seconds each. How the chapter itself was drawn is in the
first video: [Generating a Full Manga/Manhwa/Webtoon Locally - Qwen-Image 2.1](https://youtu.be/V5HX-fKtDPc).

## Setup (all free)

| piece | what it does | licence |
|---|---|---|
| [Blender 5.2 LTS](https://www.blender.org/download/) (the portable zip works) | renders the layered camera move, headless | GPL |
| [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | guesses what is in front of what | Apache 2.0 |
| [LaMa `big-lama.pt`](https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt) ([paper/code](https://github.com/advimman/lama)) | fills in what the front layers hid | Apache 2.0 |

```bash
pip install torch transformers opencv-python pillow numpy
curl -LO https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

## The three scripts

```bash
python depth_probe.py panels/ work/depth                          # 1. depth maps
python prep_layers.py panels/p.png work/depth/p_depth.png work/layers/p --fill lama   # 2. layers
blender -b --factory-startup --python multiplane.py -- work/layers/p clips/p.mp4 --move pan   # 3. camera
```

Or run everything with `bash chapter_batch.sh` after setting the paths at the top. It is resumable.
`layer_sheet.py` draws a panel's layers side by side.

## What we learned

- **The depth model gives cut-outs, not relief.** On ink line art, figures come out clean and in the
  right order, but flat inside. That suits a multiplane camera.
- **Splitting by depth tears figures.** A band edge ran through a character's robe, and the camera
  opened a white slash across him. `prep_layers.py` keeps each figure whole, in the band most of it
  sits in. The cost: most panels end up with two layers instead of three. Per-character cut-outs
  are the next step.
- **What's behind a layer has to be invented.** OpenCV's Telea fill leaves radial smears, which
  were the most visible thing on a wide pan. LaMa continues the picture instead. `--fill telea`
  is kept for comparison.
- **Colours:** the scene uses the *Standard* view transform with emission-only materials, because
  Blender's default AgX would shift every colour of the art.
- **The frame edge:** a camera move can run the nearest layer out of picture. Mirroring each plane past its
  edge works for scenery, but where a *figure* touches the edge it produced a ghost second dragon. So
  `multiplane.py` checks each figure layer for opaque pixels along the edges the camera moves toward. If
  it finds any, it crops up to 20% and scales the move down to fit. Otherwise it keeps a 12% crop and the
  mirrored extension. Every clip's choice is written to its `.json`.
- **No acting:** nobody blinks and fire doesn't flicker. Part 2 uses a video model (Wan 2.2) that
  redraws every frame.

## Numbers (our run)

Blender took a median of 21 s to render each 4-second clip at full panel size. The whole chapter, LaMa fills
included, took about **20 minutes** on a laptop with an 8 GB graphics card. The move was chosen by the panel's
shape for 34 panels (8 pans, 26 drifts); 4 were set by hand to push in (`MOVE[...]` in `chapter_batch.sh`). 35 of the 38 panels had a figure
touching the frame edge (see above), and 34 ended up with two layers.

Licence note: the panels are Qwen-Image 2.1 output, and its research licence does not allow commercial
use. The scripts here are yours to use.
