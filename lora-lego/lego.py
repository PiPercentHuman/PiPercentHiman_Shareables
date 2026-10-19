"""Can a model be built like LEGO? A base that only knows language and tool calling, with blocks for everything else.

The six town blocks and the merges come from the first experiment (github.com/PiPercentHuman/Claude-finetune-mod).
This file adds the rest of the LEGO picture, on the same base (Qwen/Qwen3.5-2B, 4-bit), greedy decoding, every reply
recorded in runs/<name>.jsonl:

  python lego.py data                 the Brick training file (brick.py: a programming language no model knows)
  python lego.py train-brick          one Brick block, with the same trainer and settings family as the town blocks
  python lego.py run brick-base       Brick tasks, base alone                (96 tasks, checked by running the program)
  python lego.py run brick-paste      Brick tasks, the manual pasted into the question
  python lego.py run brick-block      Brick tasks, with the Brick block
  python lego.py run tool-route       480 town questions: the base picks its own block by a tool call, the block answers
  python lego.py run two-<how>        120 questions that need two towns: merged | single | code | tool
  python lego.py run combo-<how>      96 tasks that need Brick AND a town: brick | town | stack | code | tool

<how>: merged = the two blocks' changes added into one; single = only the first town's block; stack = Brick and the
town block on at once (added); code = plain code splits the job and asks each block in turn; tool = the base splits it
itself by calling blocks as tools, then answers.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FM = Path(os.environ.get("LEGO_FM", HERE.parent / "Claude-finetune-mod"))                             # the mod repo
BLOCKS = Path(os.environ.get("LEGO_BLOCKS", Path(os.environ.get("FT_WORK", FM.parents[1] / "experiment" / "work")) / "blocks"))
OWN = Path(os.environ.get("LEGO_OWN", HERE / "experiment"))                                       # Brick data + block
RUNS = HERE / "runs"
PY = os.environ.get("FT_PYTHON", sys.executable)
BASE = "Qwen/Qwen3.5-2B"
sys.path.insert(0, str(FM))
sys.path.insert(0, str(FM / "mods" / "finetune" / "py"))
sys.path.insert(0, str(HERE))
import brick  # noqa: E402
from world import TEST_ASK, town, town_name  # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
TOWNS = {town_name(s).lower(): s for s in SEEDS}
DESC = {**{n: f"knows the town of {n.capitalize()}: who runs each place there" for n in TOWNS},
        "brick": "writes programs in the Brick programming language"}
# what each block's package says it needs from its caller (the interface a package manager can carry with a block)
NEEDS = {**{n: f"ask it one question about {n.capitalize()} at a time; it answers with a person's name" for n in TOWNS},
         "brick": "it knows no facts: give it the exact text or numbers the program must print"}
GUIDE = ("You are the base of a model built from blocks. Each block knows only its own subject. Split the user's "
         "request into parts, ask the right block for each part, pass what one block told you on to the next, and "
         "give the user the last block's result unchanged.")


# ---------------------------------------------------------------- the question sets (seeded, nothing typed by hand)
def town_questions():
    return [{"q": q.format(**f), "want": f["who"], "town": f["town"].lower()} for s in SEEDS for f in town(s) for q in TEST_ASK]


def two_questions(per_pair=8):
    """Each joins two towns: '<question about A>, and <question about B>?'. Right only if both names are in the reply."""
    out = []
    pairs = [(a, b) for a in SEEDS for b in SEEDS if a < b]
    for n, (a, b) in enumerate(pairs):
        fa, fb = town(a), town(b)
        for i in range(per_pair):
            x, y = fa[(n + i * 5) % 40], fb[(n * 3 + i * 7 + 1) % 40]
            q = f"Who looks after the {x['role']} in {x['town']}, and who looks after the {y['role']} in {y['town']}?"
            out.append({"q": q, "want": [x["who"], y["who"]], "towns": [x["town"].lower(), y["town"].lower()],
                        "parts": [TEST_ASK[0].format(**x), TEST_ASK[0].format(**y)]})
    return out


COMBO_ASK = ["Write a Brick program that prints who looks after the {role} in {town}.",
             "Using Brick, write a program that prints the name of the person responsible for {town}'s {role}."]


def combo_tasks(per_town=16):
    out = []
    for s in SEEDS:
        for i, f in enumerate(town(s)[:per_town]):
            out.append({"q": COMBO_ASK[i % 2].format(**f), "want": [f["who"]], "town": f["town"].lower(),
                        "fact_q": TEST_ASK[0].format(**f)})
    return out


# ---------------------------------------------------------------- the model and its blocks
class Lego:
    def __init__(self, with_brick=True):
        import torch  # noqa: F401
        from peft import PeftModel
        from train import chat, load_base
        self.chat = chat
        self.tok, base = load_base(BASE)
        names = list(TOWNS)
        self.model = PeftModel.from_pretrained(base, str(BLOCKS / f"t{TOWNS[names[0]]}"), adapter_name=names[0])
        for n in names[1:]:
            self.model.load_adapter(str(BLOCKS / f"t{TOWNS[n]}"), adapter_name=n)
        if with_brick:
            self.model.load_adapter(str(OWN / "blocks" / "brick"), adapter_name="brick")
        if (OWN / "blocks" / "plan" / "adapter_config.json").exists():
            self.model.load_adapter(str(OWN / "blocks" / "plan"), adapter_name="plan")   # train_plan.py: the plate's own block
        self.model.eval()
        self.made = set()

    def merged(self, names):
        """Several blocks' changes added into one block (plain stacking, full strength each), made once."""
        key = "+".join(names)
        if key not in self.made:
            self.model.add_weighted_adapter(names, [1.0] * len(names), key, combination_type="cat")
            self.made.add(key)
        return key

    def gen(self, text, adapter=None, max_new=40):
        import torch
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(0)
        kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        with torch.no_grad():
            if adapter is None:
                with self.model.disable_adapter():
                    out = self.model.generate(**enc, **kw)
            else:
                self.model.set_adapter(adapter)
                out = self.model.generate(**enc, **kw)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def ask(self, question, adapter=None, max_new=40):
        return self.gen(self.chat(self.tok, question), adapter, max_new)

    # ------------------------------------------------------------ the base calls blocks as tools
    TOOL = {"type": "function", "function": {
        "name": "use_block",
        "description": "Ask one installed block. Blocks: " + "; ".join(f"{k}: {v}" for k, v in DESC.items()) + ".",
        "parameters": {"type": "object", "properties": {
            "block": {"type": "string", "enum": list(DESC), "description": "which block to ask"},
            "request": {"type": "string", "description": "what to ask that block, in plain words"}},
            "required": ["block", "request"]}}}

    def tool(self, guide):
        if not guide:
            return self.TOOL
        t = json.loads(json.dumps(self.TOOL))
        t["function"]["description"] = ("Ask one installed block. Blocks: " +
                                        "; ".join(f"{k}: {v} ({NEEDS[k]})" for k, v in DESC.items()) + ".")
        return t

    def agent(self, question, max_turns=4, block_tokens=None, guide=False, plan=False):
        """The base (no block on) reads the question, calls blocks, reads their replies, and answers. guide=True adds
        each block's interface note from its package and a one-line rule for the base (the 'with a manifest' rung)."""
        msgs = ([{"role": "system", "content": GUIDE}] if guide else []) + [{"role": "user", "content": question}]
        calls, final = [], ""
        for _ in range(max_turns):
            text = self.tok.apply_chat_template(msgs, tools=[self.tool(guide)], tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
            out = self.gen(text, "plan" if plan else None, 220)
            found = parse_calls(out)
            if not found:
                final = out
                break
            pre = out.split("<tool_call>")[0].strip()
            msgs.append({"role": "assistant", "content": pre, "tool_calls": [
                {"type": "function", "function": {"name": "use_block", "arguments": c}} for c in found]})
            for c in found:
                b = (c.get("block") or "").strip().lower()
                if b in DESC:
                    res = self.ask(c.get("request", ""), b, (block_tokens or {}).get(b, 90 if b == "brick" else 24))
                else:
                    res = f"There is no block called {b!r}."
                calls.append({"block": b, "request": c.get("request", ""), "reply": res})
                msgs.append({"role": "tool", "content": res})
            final = res
        return final, calls


def parse_calls(text):
    out = []
    for body in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        if not re.search(r"<function=use_block>", body):
            continue
        params = dict((k.strip(), v.strip()) for k, v in re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.S))
        out.append(params)
    return out


def first_line(s):
    return s.strip().splitlines()[0].strip().rstrip(".") if s.strip() else ""


# ---------------------------------------------------------------- the runs
def has_all(reply, names):
    return all(n.lower() in reply.lower() for n in names)


def lim(xs):
    """LEGO_LIMIT=n: the first n of a set spread evenly (a dry run); the records of a dry run go to runs-dry/."""
    n = int(os.environ.get("LEGO_LIMIT", 0) or os.environ.get("LEGO_SAMPLE", 0))
    return xs[:: max(1, len(xs) // n)][:n] if n else xs


def run(name):
    lg = Lego()
    rows, t0 = [], time.time()
    if name.startswith("brick-"):
        how = name.split("-", 1)[1]
        for t in lim(brick.test_tasks()):
            q = f"{brick.MANUAL}\n\n{t['q']}" if how == "paste" else t["q"]
            reply = lg.ask(q, "brick" if how == "block" else None, 90)
            rows.append({"q": t["q"], "kind": t["kind"], "want": t["want"], "reply": reply, "right": brick.right(reply, t["want"])})
    elif name in ("tool-route", "plan-route"):
        for t in lim(town_questions()):
            final, calls = lg.agent(t["q"], max_turns=2 if name == "tool-route" else 3, plan=name == "plan-route")
            routed = calls[0]["block"] if calls else None
            rows.append({"q": t["q"], "want": t["want"], "town": t["town"], "routed": routed, "calls": calls,
                         "reply": final, "right": t["want"].lower() in final.lower(), "routed_right": routed == t["town"]})
    elif name.startswith("two-"):
        how = name.split("-", 1)[1]
        for t in lim(two_questions()):
            calls = []
            if how == "merged":
                reply = lg.ask(t["q"], lg.merged(t["towns"]), 40)
            elif how == "single":
                reply = lg.ask(t["q"], t["towns"][0], 40)
            elif how == "code":
                parts = [lg.ask(p, b, 24) for p, b in zip(t["parts"], t["towns"])]
                reply = " and ".join(first_line(p) for p in parts)
            elif how in ("tool", "guide", "plan"):
                reply, calls = lg.agent(t["q"], guide=how == "guide", plan=how == "plan")
            else:
                raise SystemExit(f"unknown {name}")
            rows.append({"q": t["q"], "want": t["want"], "towns": t["towns"], "reply": reply, "calls": calls,
                         "right": has_all(reply, t["want"])})
    elif name.startswith("combo-"):
        how = name.split("-", 1)[1]
        for t in lim(combo_tasks()):
            calls = []
            if how == "brick":
                reply = lg.ask(t["q"], "brick", 90)
            elif how == "town":
                reply = lg.ask(t["q"], t["town"], 90)
            elif how == "stack":
                reply = lg.ask(t["q"], lg.merged(["brick", t["town"]]), 90)
            elif how == "code":
                fact = first_line(lg.ask(t["fact_q"], t["town"], 24))
                reply = lg.ask(f"Write a Brick program that prints {fact}.", "brick", 90)
                calls = [{"block": t["town"], "request": t["fact_q"], "reply": fact}]
            elif how in ("tool", "guide", "plan"):
                reply, calls = lg.agent(t["q"], guide=how == "guide", plan=how == "plan")
            else:
                raise SystemExit(f"unknown {name}")
            rows.append({"q": t["q"], "want": t["want"], "town": t["town"], "reply": reply, "calls": calls,
                         "right": brick.right(reply, t["want"])})
    else:
        raise SystemExit(f"unknown run {name}")
    runs = RUNS if not os.environ.get("LEGO_LIMIT") else HERE / "runs-dry"
    runs.mkdir(exist_ok=True)
    meta = {"name": name, "base": BASE, "secs": round(time.time() - t0, 1),
            **({"sample": f"every {len(rows)}th-spread item: {len(rows)} of the full set"} if os.environ.get("LEGO_SAMPLE") else {})}
    with open(runs / f"{name}.jsonl", "w", encoding="utf8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"name": name, "right": sum(r["right"] for r in rows), "of": len(rows), "secs": meta["secs"]}), flush=True)


def run_project(proj):
    """A project built by the package manager (blocks.py install): load its blocks through blocks.load, answer every
    combo task its blocks cover with the same plain-code recipe, and compare with the direct run, reply by reply."""
    os.environ.setdefault("BLOCKS_TRAINER", str(FM / "mods" / "finetune" / "py"))
    os.environ.setdefault("BLOCKS_REGISTRY", str(OWN / "registry"))
    import blocks
    from train import chat
    proj = Path(proj)
    tok, model, about = blocks.load(proj)
    model.eval()

    class P(Lego):
        def __init__(self):
            self.tok, self.model, self.chat, self.made = tok, model, chat, set()
    lg = P()
    direct = {r["q"]: r for r in (json.loads(l) for l in (RUNS / "combo-code.jsonl").read_text("utf8").splitlines()[1:])}
    rows = []
    for t in combo_tasks():
        if t["town"] not in about or "brick" not in about:
            continue
        fact = first_line(lg.ask(t["fact_q"], t["town"], 24))
        reply = lg.ask(f"Write a Brick program that prints {fact}.", "brick", 90)
        rows.append({"q": t["q"], "want": t["want"], "town": t["town"], "reply": reply, "right": brick.right(reply, t["want"]),
                     "same_as_direct": reply == direct[t["q"]]["reply"]})
    name = f"project-{proj.name}"
    with open(RUNS / f"{name}.jsonl", "w", encoding="utf8") as f:
        f.write(json.dumps({"meta": {"name": name, "base": BASE, "blocks": sorted(about)}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"name": name, "right": sum(r["right"] for r in rows), "of": len(rows),
                      "same_as_direct": sum(r["same_as_direct"] for r in rows)}), flush=True)


def data():
    d = OWN / "data"
    d.mkdir(parents=True, exist_ok=True)
    pairs = brick.train_pairs()
    (d / "brick.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs), "utf8")
    print(f"{len(pairs)} Brick pairs -> {d / 'brick.jsonl'}")


def train_brick():
    import subprocess
    out = OWN / "blocks" / "brick"
    cmd = [PY, str(FM / "mods" / "finetune" / "py" / "train.py"), "--base", BASE, "--data", str(OWN / "data" / "brick.jsonl"),
           "--out", str(out), "--epochs", os.environ.get("BRICK_EPOCHS", "6"), "--rank", "16"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["data"]:
        data()
    elif sys.argv[1:2] == ["train-brick"]:
        train_brick()
    elif sys.argv[1:2] == ["project"]:
        run_project(sys.argv[2])
    elif sys.argv[1:2] == ["run"]:
        for n in sys.argv[2:]:
            run(n)
    else:
        print(__doc__)
