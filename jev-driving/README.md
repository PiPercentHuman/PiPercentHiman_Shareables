# Eight drivers, one highway

A decision model, three chatbots, a reasoning model, two local models and thirteen lines of code
drive the same simulated traffic. This is the code and every recorded drive from the video
"I Made 7 AIs Drive the Same Highway. 13 Lines of Code Won."

The road is [highway-env](https://github.com/Farama-Foundation/HighwayEnv) (MIT), a 2D traffic
simulator. It is not a real car and none of this says anything about one.

## What a driver gets

No driver sees the road. Before every decision, `world.describe()` writes four lines of text: your
speed, and the nearest car ahead and behind in your lane, the lane on your left and the lane on your
right. Then one question with five answers: lane left, lane right, faster, slower, or carry on.

Jev ([TypeSafe](https://docs.typesafe.ai)'s decision model, served through OpenRouter's Decisions
API) puts a probability on each answer. A chat model gets the same text and replies with one of the
same five words.

## Run it

```bash
pip install highway-env httpx
export OPENROUTER_API_KEY=...            # only for jev and chat: drivers

python drive.py rule dense --seeds 50                                   # thirteen lines of code
python drive.py jev dense --realtime --seeds 50                         # Jev, with the clock running
python drive.py jev dense --every 15 --seeds 50                         # Jev, the world waits (one decision a second)
python drive.py chat:anthropic/claude-haiku-4.5 dense --realtime
python drive.py idle dense --realtime --guard                           # the guard on its own
python drive.py jev dense --realtime --guard                            # Jev behind the guard
python drive.py jev highway --every 15 --numbered                       # the first wording: numbered lanes

python replay.py runs/dense-jev-realtime.jsonl                          # re-run every recorded drive, exactly
python gate.py score                                                    # the gate's numbers
```

Add `--log` to keep every call of a drive: when it was asked, how long it took, and what came back.
A call that still fails after four attempts stops the run: an error is never recorded as a decision. A reply
that names no option is ignored, and the car carries on.

## The two clocks

- **The world waits** (default). The simulation stops while the driver decides. This is how most
  "AI plays a game" demos run.
- **Real time** (`--realtime`). The car keeps doing what it was last told. An answer is applied as
  many steps late as the call really took, with one call in flight at a time. The simulation runs at
  15 steps a second, so Jev's quarter of a second is four steps, about four metres of road.

## Results (dense traffic, the same 50 traffic patterns for every driver)

Crashes out of 50 drives of 40 seconds each.

| driver | time per decision | the world waits | real time |
|---|---|---|---|
| 13 lines of code (`guard.rule`) | none | 0 | 0 |
| GPT-6 Luna | 1.0 s | 3 | 13 |
| Jev 1.13 | 0.25 s | 8 (6 at about four decisions a second) | 17 |
| Claude Haiku 4.5 | 0.9 s | 17 | 20 |
| Gemini 3.8 Flash | 4.6 s | 8 | 50 |
| Qwen3-next thinking | about 21 s | not published | 50 |
| Laya (local) | 0.06 s | 50 | 50 |
| Kev-0.8B (local) | 0.04 s | 50 | 50 |
| Kev-4B (local) | see below | 43 | not run |

Fifty drives separate a driver that always crashes from one that rarely does. They do not rank
Luna, Jev and Haiku against each other in real time.

**With the guard** (`guard.guard`, sixteen lines between the model and the wheel), real time:
Laya 3 crashes, Kev-0.8B 4, Jev 2, and the guard with no model at all 2. Average speed: guard alone
and Kev + guard 51 km/h, Laya + guard 63 km/h, the thirteen lines 64 km/h.

**The wording.** Open highway, Jev, one decision a second, the same ten traffic patterns: with the
lanes numbered ("lane 0 is the leftmost, you are in lane 3") it crashed 10 times out of 10; with
relative lanes (`your_lane`, `left_lane`, `right_lane`) it crashed 0 times out of 10.

**The gate** (`gate.py`). 200 [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (MIT)
questions in maths, physics, chemistry and engineering. DeepSeek R1 alone: 86% right, median 156 s a
question. Jev alone, no thinking: 81%; at least 90% sure on 92 of the 200 and right on 97.8% of
those. Gate (Jev answers when it is at least 70% sure, the threshold picked on half the questions):
on the other half, 87% right against 85% for R1 alone, R1 skipped on 66% of questions, 42.6% of the
time. These are published questions, so either model may have seen some of them.

## What is not here, and why

- **Kev-4B in real time.** On the machine used it took 3.7 s a decision, far slower than its authors
  report, so a real-time result would measure the machine and not the model. Only its paused result
  is given.
- **CLM-8B.** It could only be run with a compressed encoder, and that setup did not reproduce one
  of the three example outputs in its own README, so no result is reported. The drives recorded with
  it are in `runs/dense-clm-8b-4bit-*.jsonl` for completeness and should not be read as CLM's score.
- **The thinking model with the world paused.** 111 of its 398 calls failed or came back unusable,
  so that set is not published.
- **Gemini** would not turn reasoning off; it ran at the lowest reasoning effort. With the world
  paused, 48 of its 1,707 replies named no option and counted as "carry on".
- **The crossroads** (`intersection-v0`). I could not write a baseline for it that was fair.
- The descriptions, the rule and the guard are mine. A better description may move a model up.

## Files

- `world.py`: the road, `describe()` (and the first wording, `describe_numbered()`), the question.
- `guard.py`: `rule()`, the thirteen-line driver, and `guard()`, the sixteen-line guard.
- `drive.py`: the drivers and the two clocks.
- `replay.py`: replays a recorded drive from its seed and its actions and checks it ends the same way.
- `gate.py`: the gate, and `score` to recompute its numbers from `gate/answers.jsonl`.
- `runs/*.jsonl`: one line per drive: road, seed, clock, outcome, and one letter per simulation step
  (`L` lane left, `R` lane right, `F` faster, `S` slower, `I` carry on).

Run on 2026-10-01 with highway-env 1.12.1. Model ids: `typesafe/jev-1.13`, `openai/gpt-6-luna`,
`anthropic/claude-haiku-4.5`, `google/gemini-3.8-flash`, `qwen/qwen3-next-80b-a3b-thinking`,
`deepseek/deepseek-r1-0528`; [Laya](https://huggingface.co/convaiinnovations/laya) 421M and
[Kev](https://github.com/jaredpalmer/kev) ran locally behind `local:<url>`.
