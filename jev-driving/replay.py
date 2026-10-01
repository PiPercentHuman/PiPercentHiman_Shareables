"""Replay recorded drives from their seed and their actions.

  python replay.py runs/dense-jev-realtime.jsonl        # check every drive
  python replay.py runs/dense-jev-realtime.jsonl 5      # one seed, printed

The simulator is deterministic for a seed and an action sequence, so the
record of a drive is its seed plus one letter per step. A replay has to end
the way the record says (crashed or not, same second, same distance), or
this stops: a drive in these files is a drive that happened.
"""
import json
import sys

from world import HZ, make

ACT = {"L": "LANE_LEFT", "I": "IDLE", "R": "LANE_RIGHT", "F": "FASTER",
       "S": "SLOWER"}


def replay(rec, each=None):
    env = make(rec["road"], rec["seed"])
    car = env.unwrapped.vehicle
    code = {v: k for k, v in env.unwrapped.action_type.actions.items()}
    x0 = car.position[0]
    for k, letter in enumerate(rec["actions"]):
        env.step(code[ACT[letter]])
        if each:
            each(k, env)
    secs = len(rec["actions"]) / HZ
    assert bool(car.crashed) == rec["crashed"], "crash does not match"
    assert abs(secs - rec["seconds"]) < 0.01, "length does not match"
    assert abs(car.position[0] - x0 - rec["distance"]) < 0.11, "distance"
    env.close()
    return rec


if __name__ == "__main__":
    recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf8")]
    if len(sys.argv) > 2:
        recs = [r for r in recs if r["seed"] == int(sys.argv[2])]
        acts = recs[0]["actions"]
        replay(recs[0], lambda k, env: print(
            f"{(k + 1) / HZ:5.2f} s  {ACT[acts[k]]:10s} "
            f"x={env.unwrapped.vehicle.position[0]:7.1f} m  "
            f"{env.unwrapped.vehicle.speed:4.1f} m/s"))
    for r in recs:
        replay(r)
    n = sum(r["crashed"] for r in recs)
    print(f"{len(recs)} drives replayed exactly; {n} crashed")
