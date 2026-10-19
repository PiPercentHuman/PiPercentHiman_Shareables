"""Score every recorded reply in runs/ again, from the replies alone; no model is needed.

A town answer is right when the true name is in the reply; a two-town answer when both names are; a Brick answer
when the program in the reply, run by brick.py, prints exactly what the task asks for. The scores in the records must
come out the same, or this stops.

  python replay.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brick  # noqa: E402


def score(name, r):
    if name.startswith(("brick-", "combo-", "project-")):
        return brick.right(r["reply"], r["want"])
    if name.startswith("two-"):
        return all(w.lower() in r["reply"].lower() for w in r["want"])
    return r["want"].lower() in r["reply"].lower()


total = 0
for p in sorted((HERE / "runs").glob("*.jsonl")):
    lines = [json.loads(l) for l in p.read_text("utf8").splitlines() if l.strip()]
    rows = lines[1:]
    again = [score(p.stem, r) for r in rows]
    assert again == [r["right"] for r in rows], f"{p.stem}: a recorded score does not replay"
    total += len(rows)
    print(f"{p.stem:18s} {sum(again):4d} / {len(rows)}")
print(f"{total} recorded replies scored again; every one matches its record")
