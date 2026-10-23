"""Score every readout again from its records; no model needed.

  python replay.py

Each runs/*.jsonl line is one issue: the maintainer's label (`truth`), the readout's `choice` and the probability it
gave each option. This counts agreement with the maintainers per readout and checks the figures the video says.
Fitting the heads again needs the extracted features: `python read.py score` (see README).
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
load = lambda p: [json.loads(l) for l in p.read_text("utf8").splitlines() if l.strip()]
rows = {p.stem: load(p) for p in sorted(RUNS.glob("*.jsonl")) if not p.stem.endswith("-route-depth")}
depth = {p.stem[:-len("-route-depth")]: load(p) for p in sorted(RUNS.glob("*-route-depth.jsonl"))}
numbers = sorted(r["number"] for r in rows["words"])
for name, rs in rows.items():
    assert sorted(r["number"] for r in rs) == numbers, f"{name}: not the same issues"
right = {name: sum(r["choice"] == r["truth"] for r in rs) for name, rs in rows.items()}
for name in sorted(right, key=lambda k: -right[k]):
    print(f"{name:28s} {right[name]:3d} of {len(numbers)}  ({100 * right[name] / len(numbers):.1f}%)")
expect = {"bought-jev": 395, "keywords": 243, "qwen3-8b-tokens": 345, "qwen3-8b-hidden-last": 393, "words": 341,
          "gemma4-e2b-tokens": 286, "gemma4-e2b-hidden-mid": 371, "olmoe-7b-tokens": 135, "olmoe-7b-route-all": 363,
          "olmoe-7b-hidden-last": 374, "deepseek-v2-lite-tokens": 141, "deepseek-v2-lite-route-all": 367,
          "deepseek-v2-lite-hidden-last": 365}
bad = {k: (right[k], v) for k, v in expect.items() if right[k] != v}
assert not bad, f"differs from the video: {bad}"
ctl = json.loads((RUNS / "controls.json").read_text("utf8"))["olmoe-7b-route-all-labels-shuffled"]
print(f"router head with the labels shuffled: {ctl['right']} of {ctl['of']} (chance)")
dctl = json.loads((RUNS / "controls.json").read_text("utf8"))["deepseek-v2-lite-route-all-labels-shuffled"]
assert ctl["right"] == 116 and dctl["right"] == 103, (ctl, dctl)
print(f"DeepSeek router head with the labels shuffled: {dctl['right']} of {dctl['of']} (chance)")
for key, rs in depth.items():                   # the router of the first k expert layers only
    assert sorted(r["number"] for r in rs) == numbers, key
    by_k = [sum(r["choices"][k] == r["truth"] for r in rs) for k in range(len(rs[0]["choices"]))]
    print(f"{key} router, first k expert layers: " + " ".join(f"{k + 1}:{n}" for k, n in enumerate(by_k)))
    assert by_k[-1] == right[f"{key}-route-all"], f"{key}: all layers in the depth file differ from route-all"
    if key == "olmoe-7b":
        assert by_k[0] == 298 and by_k[4] == 368 and min(by_k[4:]) == 362 and max(by_k[4:]) == 375, by_k
    if key == "deepseek-v2-lite":
        assert by_k[0] == 288 and by_k[12] == 362, by_k
print(f"every figure the video says matches the records ({len(expect)} readouts, {len(numbers)} issues)")
