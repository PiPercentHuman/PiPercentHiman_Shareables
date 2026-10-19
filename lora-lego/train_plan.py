"""A block for the plate itself: how to join other blocks. Trained on conversations, tested on blocks it never saw.

The base model alone calls blocks badly when a job needs two of them (lego.py: Brick + a town fact, 0 of 48). This
trains one more small block, for the BASE, from example conversations: read the request, ask each block for its part,
pass one block's answer on to the next, hand back the last result. The conversations use six OTHER made-up towns
(seeds 7-12, never tested), so what it can learn is the procedure, not the names; the test runs on the real six.

  python train_plan.py            -> experiment/blocks/plan  (same LoRA settings as every other block: rank 16, all-linear)
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lego  # noqa: E402  (paths, the trainer's load_base, the tool's shape)
import brick  # noqa: E402
from world import TRAIN_ASK, town, town_name  # noqa: E402
from train import load_base  # noqa: E402

SEEDS = [7, 8, 9, 10, 11, 12]
NAMES = [town_name(s).lower() for s in SEEDS]
assert not set(NAMES) & set(lego.TOWNS), "a training town is one of the tested towns"
OUT = lego.OWN / "blocks" / "plan"


def tool_for(names):
    desc = {**{n: f"knows the town of {n.capitalize()}: who runs each place there" for n in names},
            "brick": lego.DESC["brick"]}
    t = json.loads(json.dumps(lego.Lego.TOOL))
    t["function"]["description"] = "Ask one installed block. Blocks: " + "; ".join(f"{k}: {v}" for k, v in desc.items()) + "."
    t["function"]["parameters"]["properties"]["block"]["enum"] = list(desc)
    return t


def call(block, request):
    return (f"<tool_call>\n<function=use_block>\n<parameter=block>\n{block}\n</parameter>\n"
            f"<parameter=request>\n{request}\n</parameter>\n</function>\n</tool_call>")


def conversations(r):
    """(user request, [(assistant turn, tool reply or None)]) in the four shapes the plate has to handle."""
    facts = {n: town(s) for n, s in zip(NAMES, SEEDS)}
    out = []
    for i in range(90):                                   # one town
        n = r.choice(NAMES); f = r.choice(facts[n])
        q = TRAIN_ASK[i % 3].format(**f)
        out.append((q, [(call(n, q), f["who"]), (f"{f['who']}.", None)]))
    for i in range(90):                                   # two towns
        a, b = r.sample(NAMES, 2); fa, fb = r.choice(facts[a]), r.choice(facts[b])
        q = f"Who runs the {fa['role']} in {fa['town']}, and who is in charge of the {fb['role']} in {fb['town']}?"
        out.append((q, [(call(a, f"Who runs the {fa['role']} in {fa['town']}?"), fa["who"]),
                        (call(b, f"Who runs the {fb['role']} in {fb['town']}?"), fb["who"]),
                        (f"{fa['who']} and {fb['who']}.", None)]))
    for i in range(90):                                   # a Brick program that prints a town fact
        n = r.choice(NAMES); f = r.choice(facts[n])
        q = (f"Write a Brick program that shows who runs the {f['role']} in {f['town']}." if i % 2 else
             f"In Brick, print the name of whoever is in charge of {f['town']}'s {f['role']}.")
        prog = f'show "{f["who"]}"'
        out.append((q, [(call(n, f"Who runs the {f['role']} in {f['town']}?"), f["who"]),
                        (call("brick", f"Write a Brick program that prints {f['who']}."), prog),
                        (f"```\n{prog}\n```", None)]))
    for p in r.sample(brick.train_pairs(), 40):           # a Brick job alone
        out.append((p["q"], [(call("brick", p["q"]), p["a"]), (f"```\n{p['a']}\n```", None)]))
    r.shuffle(out)
    return out


def examples(tok):
    """One training example per assistant turn: the conversation so far, rendered exactly as at inference, then the turn."""
    tools = [tool_for(NAMES)]
    rows = []
    for q, turns in conversations(random.Random(17)):
        msgs = [{"role": "user", "content": q}]
        for text, reply in turns:
            prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            rows.append((prompt, text + "<|im_end|>"))
            if reply is None:
                break
            args = dict(kv.split("\n", 1) for kv in [])
            block = text.split("<parameter=block>\n")[1].split("\n")[0]
            request = text.split("<parameter=request>\n")[1].split("\n</parameter>")[0]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "use_block", "arguments": {"block": block, "request": request}}}]})
            msgs.append({"role": "tool", "content": reply})
    return rows


def main():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    epochs, batch, lr = int(os.environ.get("PLAN_EPOCHS", 3)), 4, 2e-4
    torch.manual_seed(7)
    t0 = time.time()
    tok, model = load_base(lego.BASE)
    pairs = examples(tok)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    rows = []
    for prompt, answer in pairs:
        pi = tok(prompt, add_special_tokens=False)["input_ids"]
        ai = tok(answer, add_special_tokens=False)["input_ids"]
        rows.append((pi + ai, [-100] * len(pi) + ai))
    print(json.dumps({"event": "loaded", "examples": len(rows), "longest": max(len(r[0]) for r in rows)}), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    g = torch.Generator().manual_seed(7)
    model.train()
    step = 0
    for ep in range(epochs):
        order = torch.randperm(len(rows), generator=g).tolist()
        for i in range(0, len(order), batch):
            chunk = [rows[j] for j in order[i:i + batch]]
            n = max(len(c[0]) for c in chunk)
            ids = torch.tensor([c[0] + [tok.pad_token_id] * (n - len(c[0])) for c in chunk]).to(0)
            lab = torch.tensor([c[1] + [-100] * (n - len(c[1])) for c in chunk]).to(0)
            att = torch.tensor([[1] * len(c[0]) + [0] * (n - len(c[0])) for c in chunk]).to(0)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(json.dumps({"step": step, "of": steps, "epoch": ep + 1, "loss": round(loss.item(), 4),
                                  "secs": round(time.time() - t0, 1)}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    info = {"base": lego.BASE, "kind": "planner", "trained_on_towns": NAMES, "examples": len(rows), "epochs": epochs,
            "rank": 16, "lr": lr, "train_secs": round(time.time() - t0, 1),
            "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
    (OUT / "block.json").write_text(json.dumps(info, indent=1), "utf8")
    print(json.dumps({"event": "saved", **info}), flush=True)


if __name__ == "__main__":
    main()
