"""Put a model in the driver's seat, and let the clock run.

  python drive.py rule dense --seeds 50
  python drive.py jev dense --realtime --seeds 50
  python drive.py chat:anthropic/claude-haiku-4.5 dense --realtime
  python drive.py local:http://127.0.0.1:8791/decide dense --guard --realtime

Drivers
  rule         thirteen lines of code (guard.py)
  idle         never does anything (with --guard: the guard drives alone)
  jev          TypeSafe's decision model, through OpenRouter
  chat:<id>    any chat model on OpenRouter, asked for one option name
  local:<url>  a decision model served on this machine, same request as Jev

Two clocks
  default      the world waits for every answer (--every N: one decision per
               N steps; 15 = one a second)
  --realtime   the car keeps doing what it was last told; an answer lands as
               many steps late as the call really took, one call in flight

Every drive is appended to runs/<name>.jsonl: the outcome, plus the action
applied at every step, which is enough to replay it exactly (replay.py).
"""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import httpx

from guard import guard, rule
from world import (ASK, HZ, OPTIONS, SECONDS, describe, describe_numbered,
                   legal, make, question)

HERE = Path(__file__).resolve().parent
JEV = "https://openrouter.ai/api/alpha/decisions"
CHAT = "https://openrouter.ai/api/v1/chat/completions"
CLIENT = httpx.Client(timeout=300)
STATE = describe        # --numbered swaps in the first wording (rung one)
LAST = {}               # details of the latest call, kept when --log is on


def post(url, body):
    """One call, timed. A retry is real waiting, so it counts."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    head = {"Authorization": f"Bearer {key}"} if "openrouter" in url else {}
    t0 = time.perf_counter()
    for attempt in range(4):
        try:
            r = CLIENT.post(url, headers=head, json=body)
            r.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 3:
                raise
    return r.json(), (time.perf_counter() - t0) * 1000


def jev(env):
    body = {"model": "typesafe/jev-1.13", "state": STATE(env),
            "questions": question(env)}
    out, ms = post(JEV, body)
    answer = out["answers"]["action"]
    LAST.update(state=body["state"], probabilities=answer["probabilities"])
    return answer["choice"], ms


def chat(model):
    """A chat model as the driver: same state, same options, one word back.
    A reply naming no legal option is ignored and the car carries on."""
    def driver(env):
        ok = legal(env)
        state = STATE(env)
        msg = "\n".join([ASK, "", "State:", json.dumps(state, indent=1), "",
                         "Options:", *[f"- {a}: {OPTIONS[a]}" for a in ok], "",
                         "Reply with exactly one option name and nothing else."])
        body = {"model": model, "messages": [{"role": "user", "content": msg}]}
        if "thinking" in model or "deepseek-r1" in model:
            body["reasoning"] = {"enabled": True}
        elif "gemini" in model:    # refuses reasoning off: lowest effort instead
            body["reasoning"] = {"effort": "low"}
            body["max_tokens"] = 400
        else:
            body["max_tokens"] = 12
            body["reasoning"] = {"enabled": False}
        # A failed call stops the run. It must never count as "carry on":
        # an error is not a decision.
        out, ms = post(CHAT, body)
        reply = out["choices"][0]["message"]
        text = (reply.get("content") or "").upper()
        LAST.update(state=state, text=reply.get("content"),
                    reasoning=reply.get("reasoning"))
        hit = [a for a in ok if a in text]
        return (min(hit, key=text.index) if hit else "IDLE"), ms
    return driver


def local(url):
    """A decision model on this machine that takes the request Jev takes."""
    def driver(env):
        body = {"state": STATE(env), "questions": question(env)}
        out, ms = post(url, body)
        answer = out["answers"]["action"]
        LAST.update(state=body["state"],
                    probabilities=answer.get("probabilities"))
        choice = answer["choice"]
        return (choice if choice in legal(env) else "IDLE"), ms
    return driver


DRIVERS = {"rule": lambda env: (rule(env), 0.0),
           "idle": lambda env: ("IDLE", 0.0), "jev": jev}
LETTER = {"LANE_LEFT": "L", "IDLE": "I", "LANE_RIGHT": "R", "FASTER": "F",
          "SLOWER": "S"}


def drive(driver, road, seed, realtime=False, every=1, guarded=False,
          log=False):
    env = make(road, seed)
    car, dt = env.unwrapped.vehicle, 1 / HZ
    code = {v: k for k, v in env.unwrapped.action_type.actions.items()}
    x0 = car.position[0]
    pending, waits, actions, calls, vetoes = None, [], [], [], []

    def ask():
        LAST.clear()
        choice, ms = driver(env)
        waits.append(ms)
        if log:
            calls.append({"asked": k, "ms": round(ms, 1), "choice": choice,
                          "x": round(float(car.position[0]), 2), **LAST})
        return choice, ms

    for k in range(SECONDS * HZ):
        act = "IDLE"                      # a command is used once
        if not realtime:
            if k % every == 0:            # the world waits for the answer
                act, _ = ask()
        elif pending is None:             # real time: one call in flight
            choice, ms = ask()            # ms: how long it really took
            lands = k + math.ceil(ms / 1000 / dt)
            pending = (lands, choice)     # the car keeps driving meanwhile
        if pending and k >= pending[0]:   # ...and the answer lands late
            act, pending = pending[1], None
        if guarded:
            safe = guard(env, act)
            if safe != act:
                vetoes.append((k, act, safe))
            act = safe
        if act not in legal(env):
            act = "IDLE"
        actions.append(LETTER[act])
        _, _, done, cut, _ = env.step(code[act])
        if car.crashed or done or cut:
            break
    secs = (k + 1) * dt
    out = {"road": road, "seed": seed, "realtime": realtime, "every": every,
           "guard": guarded, "crashed": bool(car.crashed),
           "seconds": round(secs, 2),
           "distance": round(float(car.position[0] - x0), 1),
           "mean_speed": round(float(car.position[0] - x0) / secs, 2),
           "decisions": len(waits), "vetoes": len(vetoes),
           "wait_ms_median": round(statistics.median_high(waits), 1),
           "actions": "".join(actions)}
    if log:
        out.update(calls=calls, vetoed=vetoes)
    env.close()
    return out


def pick(name):
    kind, _, rest = name.partition(":")
    if kind == "chat":
        return chat(rest)
    if kind == "local":
        return local(rest)
    return DRIVERS[name]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("road", choices=["highway", "dense"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--numbered", action="store_true",
                    help="describe the road with numbered lanes (rung one)")
    ap.add_argument("--log", action="store_true",
                    help="keep every call: when it was asked, how long it "
                         "took, what came back")
    ap.add_argument("--name", help="file name under runs/")
    a = ap.parse_args()
    if a.numbered:
        STATE = describe_numbered
    name = a.name or "-".join(
        [a.road, a.driver.split("/")[-1].replace(":", "_"),
         "realtime" if a.realtime else f"paused{a.every}"]
        + (["guard"] if a.guard else []) + (["numbered"] if a.numbered else []))
    out = HERE / "runs" / f"{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    have = set()
    if out.exists():
        have = {json.loads(l)["seed"] for l in out.read_text("utf8").splitlines()}
    crashes = 0
    for seed in range(a.first, a.first + a.seeds):
        if seed in have:
            continue
        r = drive(pick(a.driver), a.road, seed, a.realtime, a.every, a.guard,
                  a.log)
        crashes += r["crashed"]
        with out.open("a", encoding="utf8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"seed {seed:3d}  crashed={r['crashed']!s:5}  "
              f"{r['seconds']:5.1f} s  {r['mean_speed']:5.1f} m/s  "
              f"decisions={r['decisions']:4d}  "
              f"wait={r['wait_ms_median']:.0f} ms", flush=True)
    print(f"{name}: {crashes} crashed of {a.seeds} new drives -> {out}")
