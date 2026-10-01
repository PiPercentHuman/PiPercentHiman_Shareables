"""A fast decision model as the gate in front of a slow reasoning model.

  python gate.py score     # the video's numbers, from gate/answers.jsonl
  python gate.py ask "Which is a prime?" "21" "23" "25"    # live, one question

If the fast model is sure, take its answer. If it isn't, wake the slow one.
"""
import json
import os
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
API = "https://openrouter.ai/api"
SLOW = "deepseek/deepseek-r1-0528"
LETTERS = "ABCDEFGHIJ"
SURE = 0.7     # picked on half the questions, read once on the other half


def call(path, body):
    key = os.environ["OPENROUTER_API_KEY"]
    r = httpx.post(API + path, json=body, timeout=1800,
                   headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()


def jev(question, options):
    """The fast model: no thinking, a probability on every option."""
    ask = {"answer": {
        "type": "choice",
        "instructions": "Which option correctly answers `question`?",
        "criteria": {LETTERS[i]: o[:200] for i, o in enumerate(options)}}}
    out = call("/alpha/decisions", {"model": "typesafe/jev-1.13",
                                    "state": {"question": question},
                                    "questions": ask})
    return out["answers"]["answer"]


def reason(question, options):
    """The slow model: thinks first, then commits to a letter."""
    listed = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    msg = ("Answer the following multiple choice question. Think it through, "
           "then finish with 'The answer is (X)' where X is the letter.\n\n"
           f"Question: {question}\n\nOptions:\n{listed}")
    out = call("/v1/chat/completions", {
        "model": SLOW, "reasoning": {"enabled": True},
        "messages": [{"role": "user", "content": msg}]})
    text = out["choices"][0]["message"].get("content") or ""
    found = re.findall(r"answer is:?\s*\**\s*\(?\s*([A-J])\b", text, re.I)
    return found[-1].upper() if found else None


def gate(question, options, sure=SURE):
    answer = jev(question, options)          # a fraction of a second
    choice = answer["choice"]
    if answer["probabilities"][choice] >= sure:
        return choice, "fast"                # sure: take it
    return reason(question, options), "slow"   # not sure: think it through


def score():
    """Replay the gate over the answers both models already gave."""
    rows = [json.loads(l) for l in open(HERE / "gate" / "answers.jsonl")]
    fast = 0.35     # seconds counted per Jev call

    def run(rs, sure):
        right = skipped = secs = 0
        for r in rs:
            secs += fast
            if r["jev_p"] >= sure:
                right += r["jev"] == r["gold"]
                skipped += 1
            else:
                right += r["slow"] == r["gold"]
                secs += r["slow_secs"]
        return (right / len(rs), skipped / len(rs),
                secs / sum(r["slow_secs"] for r in rs))

    def alone(rs):
        return sum(r["slow"] == r["gold"] for r in rs) / len(rs)

    dev, test = rows[0::2], rows[1::2]
    n = len(rows)
    print(f"{n} questions. Slow model alone: {alone(rows):.0%} right, "
          f"median {sorted(r['slow_secs'] for r in rows)[n // 2]:.0f} s each")
    jev_right = sum(r["jev"] == r["gold"] for r in rows)
    sure90 = [r for r in rows if r["jev_p"] >= 0.9]
    print(f"Jev alone: {jev_right / n:.0%} right. At least 90% sure on "
          f"{len(sure90)} of {n}, and right on "
          f"{sum(r['jev'] == r['gold'] for r in sure90) / len(sure90):.1%} "
          "of those")
    best = None
    for sure in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        acc, skip, used = run(dev, sure)
        print(f"  dev half, sure >= {sure:4.2f}: {acc:.0%} right, slow model "
              f"skipped {skip:.0%}, time used {used:.0%}")
        if acc >= alone(dev) and (best is None or used < best[1]):
            best = (sure, used)
    acc, skip, used = run(test, best[0])
    print(f"threshold picked on the dev half: {best[0]}")
    print(f"TEST half, read once: gate {acc:.0%} right vs slow model alone "
          f"{alone(test):.0%}; slow model skipped {skip:.0%}; "
          f"time used {used:.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    else:
        print(gate(sys.argv[2], sys.argv[3:]))
