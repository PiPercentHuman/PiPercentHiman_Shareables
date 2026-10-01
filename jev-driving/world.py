"""The road, and what a driver is told about it.

highway-env (MIT) runs the traffic. Nothing here looks at pixels: the
driver gets a short description in words, built by describe().
"""
import gymnasium as gym
import highway_env  # noqa: F401  (registers the roads)

HZ = 15                               # simulation steps per second
SECONDS = 40                          # length of one drive
SPEEDS = [0, 5, 10, 15, 20, 25, 30]   # m/s; FASTER and SLOWER move one step
ROADS = {
    "highway": {"lanes_count": 4, "vehicles_count": 50},
    "dense": {"lanes_count": 4, "vehicles_count": 80, "vehicles_density": 2.0},
}
OPTIONS = {
    "LANE_LEFT": "Change one lane to the left",
    "IDLE": "Keep the lane and the current speed",
    "LANE_RIGHT": "Change one lane to the right",
    "FASTER": "Speed up",
    "SLOWER": "Slow down (to wait, keep slowing)",
}
ASK = ("You are driving on a highway. Avoid crashing above all, "
       "then keep a high speed. Pick the next action.")


def make(road, seed):
    """One drive: the same seed gives the same traffic to every driver."""
    cfg = dict(ROADS[road], simulation_frequency=HZ, policy_frequency=HZ,
               duration=SECONDS)
    cfg["action"] = {"type": "DiscreteMetaAction", "target_speeds": SPEEDS}
    env = gym.make("highway-v0", config=cfg)
    env.reset(seed=seed)
    return env


def scene(env):
    """Own lane and speed, and the nearest car ahead and behind per lane."""
    u = env.unwrapped
    ego, lanes = u.vehicle, u.config["lanes_count"]
    near = {l: {"ahead": None, "behind": None} for l in range(lanes)}
    for v in u.road.vehicles:
        if v is ego:
            continue
        l, dx = v.lane_index[2], v.position[0] - ego.position[0]
        if abs(dx) > 120:
            continue
        side = "ahead" if dx >= 0 else "behind"
        cur = near[l][side]
        if cur is None or abs(dx) < abs(cur[0]):
            near[l][side] = (dx, v.speed - ego.speed)
    return ego.lane_index[2], ego.speed, near, lanes


def target_lane(env):
    """The lane the car is heading for (differs mid lane-change)."""
    return env.unwrapped.vehicle.target_lane_index[2]


def legal(env):
    t = env.unwrapped.action_type
    return [t.actions[a] for a in t.get_available_actions()]


def cars(lane):
    """One lane in words: the nearest car ahead, the nearest behind."""
    out = []
    for side in ("ahead", "behind"):
        n = lane[side]
        if n is None:
            out.append(f"clear {side}")
            continue
        dist, rel = abs(n[0]), n[1]
        closing = -rel if side == "ahead" else rel
        how = (f"closing on you at {closing:.0f} m/s"
               if closing > 0.5 else "not closing")
        close = " (close)" if dist < 25 else ""
        out.append(f"car {side} {dist:.0f} m{close}, {how}")
    return "; ".join(out)


def describe(env):
    """What the model is told: the road, in the words you'd use
    for a person. No lane numbers, nothing to work out."""
    lane, speed, near, lanes = scene(env)
    cruise = env.unwrapped.vehicle.target_speed
    state = {
        "your_speed": f"{speed:.0f} m/s, cruise set to {cruise:.0f} "
                      "(each speed-up/slow-down moves it 5 m/s, range 0-30)",
        "your_lane": cars(near[lane]),
        "left_lane": cars(near[lane - 1]) if lane > 0
        else "none: you are in the leftmost lane",
        "right_lane": cars(near[lane + 1]) if lane < lanes - 1
        else "none: you are in the rightmost lane",
    }
    tgt = target_lane(env)
    if tgt != lane:
        side = "left" if tgt < lane else "right"
        state["lane_change"] = f"you are already moving into the {side} lane"
    return state


def describe_numbered(env):
    """The first wording we tried: numbered lanes. Kept because it is the
    comparison in the video - with this, Jev crashed 10 drives out of 10."""
    lane, speed, near, lanes = scene(env)
    tgt = target_lane(env)
    rows = []
    for l in range(lanes):
        tag = (" (your lane)" if l == lane
               else " (you are moving into this lane)" if l == tgt else "")
        parts = []
        for side in ("ahead", "behind"):
            n = near[l][side]
            if n is None:
                parts.append(f"nothing {side} within 120 m")
                continue
            closing = -n[1] if side == "ahead" else n[1]
            how = (f"closing at {closing:.0f} m/s"
                   if closing > 0.5 else "not closing")
            parts.append(f"car {side} {abs(n[0]):.0f} m, {how}")
        rows.append(f"lane {l}{tag}: " + "; ".join(parts))
    move = f", changing into lane {tgt}" if tgt != lane else ""
    head = (f"Highway with {lanes} lanes, lane 0 is the leftmost. You are "
            f"in lane {lane}{move} at {speed:.0f} m/s (top speed 30 m/s). "
            "Cars are about 5 m long.\n")
    return {"scene": head + "\n".join(rows)}


def question(env):
    """One question, and the answers the car can accept right now."""
    return {"action": {"type": "choice", "instructions": ASK,
                       "criteria": {a: OPTIONS[a] for a in legal(env)}}}
