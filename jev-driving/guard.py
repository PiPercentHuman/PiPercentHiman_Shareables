"""Two pieces of ordinary code: the baseline driver, and the guard.

rule()  drives the car by itself. It is the driver that never crashed.
guard() sits between any model and the wheel and vetoes unsafe moves.
Both read the same scene() the models are described from.
"""
from world import legal, scene, target_lane


def rule(env):
    lane, speed, near, lanes = scene(env)
    ok, tgt = legal(env), target_lane(env)
    gap = min((near[l]["ahead"][0] for l in {lane, tgt} if near[l]["ahead"]), default=999)
    if tgt != lane:
        return "SLOWER" if gap < 1.5 * speed else "IDLE"
    if gap < 2.0 * speed:
        for act, l in (("LANE_LEFT", lane - 1), ("LANE_RIGHT", lane + 1)):
            if act in ok and 0 <= l < lanes:
                a, b = near[l]["ahead"], near[l]["behind"]
                if (a is None or a[0] > gap + 10) and (b is None or b[0] < -15):
                    return act
        return "SLOWER"
    return "FASTER" if speed < 29 and "FASTER" in ok else "IDLE"


def guard(env, act):
    lane, speed, near, lanes = scene(env)
    tgt = target_lane(env)
    gap = min((near[l]["ahead"][0] for l in {lane, tgt} if near[l]["ahead"]), default=999)
    if act in ("LANE_LEFT", "LANE_RIGHT"):
        l = lane + (-1 if act == "LANE_LEFT" else 1)
        if tgt != lane or not 0 <= l < lanes:
            act = "IDLE"
        else:
            a, b = near[l]["ahead"], near[l]["behind"]
            if (a and a[0] < max(10.0, speed)) or (b and b[0] > -12.0):
                act = "IDLE"
    if act == "FASTER" and gap < 2.0 * speed:
        act = "IDLE"
    if gap < 1.0 * speed:
        act = "SLOWER"
    return act
