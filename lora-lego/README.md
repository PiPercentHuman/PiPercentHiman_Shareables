# Can a model be built like LEGO?

A base model that only knows language and tool calling (the plate), LoRA adapters for everything else (the blocks),
and a package manager that joins the blocks a project needs. Every joint is tested on `Qwen/Qwen3.5-2B` in 4-bit with greedy
decoding, and every reply is recorded in `runs/`.

| Test | Result |
|---|---|
| The plate alone, town questions | 0 / 480 |
| Brick programs: plate alone / manual pasted / Brick block | 0 / 12 / 88 of 96 |
| Six town blocks merged (added / TIES / DARE) | 9 / 26 / 26 of 480 |
| One block at a time, picked by code | 479 / 480 |
| The plate picks its block (tool call) / with the planner block | 200 / 218 of 240 |
| Two-town questions: merged / one block / plate / planner / code | 0/120, 0/120, 43/60, 56/60, 119/120 |
| Brick + a town fact: stacked / plate / planner / recipe | 0/96, 0/48, 6/48, 96/96 |
| A project installed by blocks.py, same answers as loaded by hand | 48 / 48 |

## The files

- `brick.py` - Brick, a programming language made up for this test, its interpreter, and its training and test tasks.
- `lego.py` - every test above (`python lego.py run <name>`); the plate calling blocks as tools.
- `train_plan.py` - the planner block: the plate's own block for joining blocks, trained on six other made-up towns.
- `blocks.py` - the package manager: `publish`, `search`, `init`, `add`, `install`, `list`, and the fit check.
- `replay.py` - scores every recorded answer again from the records alone; no model needed.
- `terminal-*.txt` - the package manager's real output for the two demo projects.

## Running it

The six town blocks come from the first experiment, https://github.com/PiPercentHuman/Claude-finetune-mod
(`run.py blocks`). Set `LEGO_FM` to that repository and `LEGO_BLOCKS` to its blocks folder, then:

    python lego.py data && python lego.py train-brick
    python train_plan.py
    python lego.py run brick-block combo-code two-code tool-route
    python replay.py

## What is not here, and why

- One small base model and one seed; made-up facts of one kind and one made-up language, so every right answer was
  learned from a block. Larger plates and real company data are not tested.
- The tool-calling tests ran on even samples (240 of 480, 60 of 120, 48 of 96) to keep the GPU time down.
- The trained blocks themselves are not in this folder (67 MB each); the scripts make them.
