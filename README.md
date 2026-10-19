# Pi Percent Human — Shareables

Things made for Pi Percent Human videos that are worth having in full, not just as a few seconds on screen.

| | |
|---|---|
| [The King of Elfland's Daughter — a manhwa chapter](elflands-daughter-manhwa/) | 38-panel vertical-scroll comic, every panel generated locally with Qwen-Image 2.1 |
| [Manhwa in motion, part 1: a multiplane camera in Blender](manhwa-motion-blender/) | The 38 panels of that chapter turned into 4-second camera shots without redrawing a line: each panel split into depth layers and filmed in Blender. The three scripts, and the whole chapter in motion |
| [Eight drivers, one highway](jev-driving/) | A decision model, three chatbots, a reasoning model, two local models and thirteen lines of code drive the same simulated traffic: the code, and all 1,220 recorded drives, each replayable from its seed |
| [AgentJournal: a memory for coding agents](https://github.com/PiPercentHuman/agentjournal) | Pin what you were told, journal what you did, sleep when your head is full: a pi extension, a Claude Code plugin and Codex/Copilot hooks that keep an agent's rules and progress across context cuts, with the benchmark behind it (169 recorded trials on a local model). Its own repository, so it can be installed and forked |
| [/finetune: a Claude Code mod that fine-tunes a local model](https://github.com/PiPercentHuman/Claude-finetune-mod) | One command trains a LoRA block for a local model on your own question-and-answer file, and /finetune ask puts the same question to the model with and without it. The mod, the six made-up towns it was tested on, every recorded reply (replay.py scores them again without a model) and the recorded terminal sessions. Its own repository, so it can be loaded with --plugin-dir and forked |
| [Can a model be built like LEGO?](lora-lego/) | A small base model that only knows language and tool calling, LoRA blocks for everything else, and a package manager (blocks.py) that installs the blocks a project needs. Brick, a programming language made up for the test, every recorded reply (replay.py scores all 1,884 again without a model) and the package manager's terminal output |
