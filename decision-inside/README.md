# Is the router of a mixture-of-experts model already a decision model?

The same 517 labelled GitHub issues (ollama/ollama; each labelled by a maintainer as bug, feature, question, docs or
model request) read out of open models in three places, and raced against Jev, a decision model you pay for.

| place | what is read |
| --- | --- |
| `tokens` | the model is asked, and the answer is the probability it puts on each option's letter |
| `hidden` | nothing is generated; a small linear head reads one layer's state at the issue's last token |
| `router` | mixture-of-experts only: a linear head reads how often each expert was switched on, per layer, and nothing else |

The heads are logistic regression, fitted on four fifths of the issues and tested on the fifth they never saw, five
times over, so every issue is judged by a head that never saw it. They learn from this project's labels, which Jev
never saw, so their fair rival is word counts with the same labels and the same head (`words`).

## Results (agreement with the maintainers, of 517)

| readout | right | share |
| --- | --- | --- |
| Jev (bought; `runs/bought-jev.jsonl`) | 395 | 76.4% |
| Qwen3-8B, head on the last layer | 393 | 76.0% |
| OLMoE-1B-7B, head on the last layer | 374 | 72.3% |
| Gemma 4 E2B, head on the middle layer | 371 | 71.8% |
| DeepSeek-V2-Lite-Chat, head on the router alone | 367 | 71.0% |
| DeepSeek-V2-Lite-Chat, head on the last layer | 365 | 70.6% |
| OLMoE-1B-7B, head on the router alone | 363 | 70.2% |
| Qwen3-8B, asked | 345 | 66.7% |
| word counts, same head and labels | 341 | 66.0% |
| Gemma 4 E2B, asked | 286 | 55.3% |
| keyword matching (`runs/keywords.jsonl`) | 243 | 47.0% |
| DeepSeek-V2-Lite-Chat, asked | 141 | 27.3% |
| OLMoE-1B-7B, asked | 135 | 26.1% |

- OLMoE asked says bug 287 times; with the letters rotated (`*-tokens-rotated`, every option takes a turn at every
  letter) it says bug 507 times.
- The router heads with the labels shuffled score 116 (OLMoE) and 103 (DeepSeek), chance (`runs/controls.json`).
- How early: a head on the router of the first k expert layers only (`runs/*-route-depth.jsonl`, `read.py depth`).
  OLMoE: 298 on the first layer, 368 by the fifth of 16, then between 362 and 375. DeepSeek: 288 on the first,
  362 by the 13th of 26, 367 with all of them.
- Among answers given with 90% or more, the Qwen3-8B last-layer head and Jev each got 311 right.

## Run it

```bash
python replay.py                     # score every readout again from runs/*.jsonl; no model needed
python read.py score [--only KEY]    # fit the heads again from feats/*.npz by 5-fold cross-validation; runs/*.jsonl rewritten
python read.py depth olmoe-7b        # the router of the first k layers only, k = 1 .. all
python read.py extract qwen3-8b      # one pass over the issues on a CUDA GPU (4-bit); feats/qwen3-8b.npz written
python read.py check deepseek-v2-lite    # the layer-by-layer runner on one sentence (next word: Paris)
python read.py extract deepseek-v2-lite  # bf16, one layer at a time: needs far less GPU memory than the model's size
```

`replay.py` needs only the standard library. `score` and `depth` need numpy and scikit-learn; `extract` and `check`
need torch, transformers, bitsandbytes and the models from Hugging Face (`extract` for DeepSeek reads an already
downloaded snapshot: `huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat` first).

Models: `Qwen/Qwen3-8B`, `google/gemma-4-E2B-it`, `allenai/OLMoE-1B-7B-0125-Instruct`, loaded in 4 bits (NF4);
`deepseek-ai/DeepSeek-V2-Lite-Chat` in bf16, run layer by layer (each layer's weights loaded once, every issue pushed
through it; the library's own decoder layer and router, only the loading is ours).

## What is in the folder

| path | what it is |
| --- | --- |
| `data/eval.json` | the 517 issues (`tasks.type.items`): number, URL, the maintainer's label, the title and body as filed (`state`) and with the issue template's headings removed (`stateStripped`, what every model read) |
| `feats/*.npz` | what `extract` saved per model: the letter probabilities under every rotation, the middle and last layer at the issue's last token, and for the two mixtures of experts how often each expert was picked on each layer |
| `runs/*.jsonl` | one line per issue per readout: the answer, the probability of every option, the truth |
| `runs/controls.json` | the router heads refitted with the labels shuffled |

## How exactly it reproduces

- `replay.py` reproduces every figure from the records exactly.
- `read.py score` refits the heads from `feats/`. On scikit-learn 1.9.0 it gives the same totals as the records
  except OLMoE's last layer (373 instead of 374), OLMoE's middle layer (356 instead of 355) and Qwen3-8B's middle
  layer (325 instead of 323); the probabilities move by up to 0.1. The DeepSeek, router-depth and shuffled-label
  figures come out identical.
- `extract` depends on the GPU, the library versions and 4-bit kernels; it was run once per model.

## Limits

- One job (five kinds of GitHub issue), one run of each readout. The larger Gemma 4 models were not run.
- Two mixtures of experts, both far smaller than frontier ones. What this means for frontier models is a reading,
  not a measurement.
- The heads learn from this project's labels and Jev did not, which is why word counts with the same labels and head
  are the fair rival.
