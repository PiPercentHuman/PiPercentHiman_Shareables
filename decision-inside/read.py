"""Where does a decision live inside a language model?

The same labelled GitHub issues, one typed question each ("what kind of issue is this?", five options), read out
of an ordinary open model at three depths:

  tokens   the model is asked, and the answer is the probability it puts on each option's letter
  hidden   nothing is generated; a small linear head reads one layer's state at the last token
  router   mixture-of-experts models only: a linear head reads WHICH EXPERTS were switched on, nothing else

  python read.py extract <model-key> [--limit N]     one pass over the issues; features saved under feats/
  python read.py score [--only <model-key>]          heads fitted by 5-fold cross-validation; runs/*.jsonl written
  python read.py depth <model-key>                   the router of the first k layers only, k = 1 .. all; one file
  python read.py check <model-key>                   a layer-by-layer model on one sentence: is its next word right?

A model too big for the card (`"layerwise": True`) is run one layer at a time: that layer's weights are loaded once,
every issue is pushed through it, and the states wait in memory for the next layer. Same prompt, same readouts.

`tokens` uses no labels. `hidden` and `router` are trained heads: each issue is judged by a head fitted on the
other four fifths, never on itself. The plain-code baseline for a trained head is word counts with the same
linear head (`words`), fitted the same way.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE / "data" / "eval.json"     # the 517 issues, as Jev was given them
FEATS = HERE / "feats"
RUNS = HERE / "runs"

MODELS = {
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B", "moe": False},
    "gemma4-e2b": {"repo": "google/gemma-4-E2B-it", "moe": False},
    "gemma4-e4b": {"repo": "google/gemma-4-E4B-it", "moe": False},
    "olmoe-7b": {"repo": "allenai/OLMoE-1B-7B-0125-Instruct", "moe": True},
    "deepseek-v2-lite": {"repo": "deepseek-ai/DeepSeek-V2-Lite-Chat", "moe": True, "layerwise": True},
}
INSTRUCTIONS = "What kind of GitHub issue is this, judging by `title` and `body`?"
OPTIONS = [("bug", "Something in Ollama is broken or behaves wrongly"),
           ("feature", "Asks for a new feature or a change in behaviour"),
           ("question", "A general question about how to use Ollama"),
           ("docs", "Improvements or additions to the documentation"),
           ("model_request", "Asks for a new model to be added to Ollama")]
LETTERS = "ABCDE"
MAX_BODY_TOKENS = 900
FOLDS, SEED = 5, 7


def items():
    return json.loads(EVAL.read_text("utf8"))["tasks"]["type"]["items"]


def prompt_text(item, shift, tok):
    """The question as the model reads it. `shift` rotates which option gets which letter."""
    st = item["stateStripped"]
    body = tok.decode(tok(st["body"], add_special_tokens=False)["input_ids"][:MAX_BODY_TOKENS])
    order = OPTIONS[shift:] + OPTIONS[:shift]
    menu = "\n".join(f"{LETTERS[i]}. {name}: {desc}" for i, (name, desc) in enumerate(order))
    return (f"{INSTRUCTIONS}\n\n{menu}\n\ntitle: {st['title']}\nbody: {body}\n\n"
            "Answer with the single letter of one option and nothing else."), [name for name, _ in order]


def letter_ids(tok):
    """Every single token that spells one of the letters, with or without a leading space."""
    ids = []
    for L in LETTERS:
        found = set()
        for form in (L, " " + L):
            enc = tok(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                found.add(enc[0])
        assert found, f"no single token for {L!r}"
        ids.append(sorted(found))
    return ids


def extract(key, limit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    spec = MODELS[key]
    tok = AutoTokenizer.from_pretrained(spec["repo"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    lids = letter_ids(tok)
    data = items()[:limit]
    FEATS.mkdir(parents=True, exist_ok=True)
    out = {"number": [], "truth": [], "form": [], "letters": [], "ms": [], "tokens": [],
           "hidden_mid": [], "hidden_last": [], "route_all": [], "route_last": []}

    def chat(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    t_start = time.time()
    for n, it in enumerate(data):
        per_shift = []
        for shift in range(len(OPTIONS)):
            text, order = prompt_text(it, shift, tok)
            enc = tok(chat(text), return_tensors="pt", add_special_tokens=False).to(0)
            kw = {"output_hidden_states": shift == 0}
            if spec["moe"] and shift == 0:
                kw["output_router_logits"] = True
            t0 = time.perf_counter()
            with torch.no_grad():
                res = model(**enc, **kw)
            ms = (time.perf_counter() - t0) * 1000
            logits = res.logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            per_letter = torch.stack([torch.logsumexp(lp[ids], 0) for ids in lids])      # mass on each letter
            probs = torch.softmax(per_letter, 0).cpu().numpy()
            by_option = {name: float(probs[i]) for i, name in enumerate(order)}
            per_shift.append([by_option[name] for name, _ in OPTIONS])
            if shift == 0:
                out["ms"].append(ms)
                out["tokens"].append(int(enc["input_ids"].shape[1]))
                out["letters"].append(float(torch.exp(torch.logsumexp(per_letter, 0))))  # how much mass was on A-E at all
                hs = res.hidden_states
                out["hidden_mid"].append(hs[len(hs) // 2][0, -1].float().cpu().numpy().astype(np.float16))
                out["hidden_last"].append(hs[-1][0, -1].float().cpu().numpy().astype(np.float16))
                if spec["moe"]:
                    rl = res.router_logits                                               # one [tokens, experts] per layer
                    k = model.config.num_experts_per_tok
                    all_, last_ = [], []
                    for layer in rl:
                        layer = layer.float().reshape(-1, layer.shape[-1])
                        top = torch.zeros_like(layer).scatter_(1, layer.topk(k, -1).indices, 1.0)
                        all_.append(top.mean(0))                                         # share of tokens each expert served
                        last_.append(top[-1])                                            # the experts on at the last token
                    out["route_all"].append(torch.cat(all_).cpu().numpy().astype(np.float32))
                    out["route_last"].append(torch.cat(last_).cpu().numpy().astype(np.float32))
        out["number"].append(it["number"]); out["truth"].append(it["truth"]); out["form"].append(bool(it["form"]))
        out.setdefault("shift_probs", []).append(per_shift)
        if n % 25 == 0:
            print(f"{key} {n + 1}/{len(data)}  {time.time() - t_start:.0f}s  first-pass {out['ms'][-1]:.0f} ms  "
                  f"letters {out['letters'][-1]:.2f}", flush=True)
    np.savez_compressed(FEATS / f"{key}.npz", **{k: np.array(v) for k, v in out.items() if len(v)})
    print(f"{key}: saved {len(data)} items in {time.time() - t_start:.0f}s")


def _chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Layerwise:
    """A model kept on disk and run one decoder layer at a time in bf16, for a model bigger than the card.

    Each layer's weights are read from the safetensors files once per sweep, every sequence is pushed through that
    layer, and the states wait in memory for the next one. The decoder layer, its attention and its router are the
    library's own classes; only the loading is ours (each expert's gate and up weights stacked the way the library
    stores them), and every layer's loaded names must equal the library's names exactly.
    """

    def __init__(self, repo):
        import importlib
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer
        self.torch = torch
        self.path = Path(snapshot_download(repo, local_files_only=True))
        self.config = AutoConfig.from_pretrained(self.path)
        self.config._attn_implementation = "sdpa"
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.where = json.loads((self.path / "model.safetensors.index.json").read_text("utf8"))["weight_map"]
        mt = self.config.model_type
        mod = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
        pick = lambda end: getattr(mod, next(c for c in dir(mod) if c.endswith(end) and not c.startswith("_")))
        self.Decoder, self.Norm = pick("DecoderLayer"), pick("RMSNorm")
        self.rotary = pick("RotaryEmbedding")(self.config, device="cuda")
        self.n = self.config.num_hidden_layers
        self.experts = self.config.n_routed_experts

    def tensors(self, prefix):
        from safetensors import safe_open
        files = {}
        for k, f in self.where.items():
            if k.startswith(prefix):
                files.setdefault(f, []).append(k)
        out = {}
        for f, ks in files.items():
            with safe_open(str(self.path / f), framework="pt", device="cpu") as h:
                for k in ks:
                    out[k[len(prefix):]] = h.get_tensor(k)
        return out

    def layer(self, L):
        torch = self.torch
        sd = self.tensors(f"model.layers.{L}.")
        if "mlp.experts.0.gate_proj.weight" in sd:
            E = self.experts
            sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([sd.pop(f"mlp.experts.{e}.gate_proj.weight"),
                                                                     sd.pop(f"mlp.experts.{e}.up_proj.weight")], 0) for e in range(E)])
            sd["mlp.experts.down_proj"] = torch.stack([sd.pop(f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
        with torch.device("meta"):
            m = self.Decoder(self.config, L)
        want = set(m.state_dict())
        assert want == set(sd), f"layer {L}: names differ {sorted(want ^ set(sd))[:8]}"
        m.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, assign=True)
        return m.to("cuda").eval()

    def run(self, seqs, on_layer=None):
        """Every sequence through every layer. on_layer(L, states, picks) sees the states after layer L and, for an
        expert layer, the experts the router picked for every token (the library's own top-k on its own logits)."""
        torch = self.torch
        emb = self.tensors("model.embed_tokens.")["weight"]
        hs = [emb[torch.tensor(ids)].to(torch.bfloat16) for ids in seqs]
        del emb
        t0 = time.time()
        for L in range(self.n):
            m = self.layer(L)
            picks, hook = [], None
            if hasattr(m.mlp, "gate"):
                def pre(mod, args):
                    logits = torch.nn.functional.linear(args[0].float(), mod.gate.weight.float())
                    idx, _ = mod.route_tokens_to_experts(logits)
                    picks.append(idx.view(-1, mod.top_k))
                hook = m.mlp.register_forward_pre_hook(pre)
            with torch.no_grad():
                for i in range(len(hs)):
                    h = hs[i].to("cuda")[None]
                    pos = torch.arange(h.shape[1], device="cuda")[None]
                    hs[i] = m(h, attention_mask=None, position_ids=pos, position_embeddings=self.rotary(h, pos))[0].cpu()
            if hook:
                hook.remove()
            if on_layer:
                on_layer(L, hs, picks)
            del m
            torch.cuda.empty_cache()
            print(f"  layer {L + 1}/{self.n}  {time.time() - t0:.0f}s", flush=True)
        norm = self.Norm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        norm.load_state_dict(self.tensors("model.norm."))
        return hs, norm.to("cuda", torch.bfloat16)

    def head(self):
        return self.tensors("lm_head.")["weight"].to("cuda", self.torch.bfloat16)


def check_layerwise(key):
    """One sentence through the layer-by-layer model: its next word should be the obvious one."""
    import torch
    M = Layerwise(MODELS[key]["repo"])
    ids = M.tok("The capital of France is", add_special_tokens=True)["input_ids"]
    hs, norm = M.run([ids])
    logits = (norm(hs[0][-1:].to("cuda")) @ M.head().T)[0].float()
    top = torch.topk(logits, 5).indices.tolist()
    print("next word:", [M.tok.decode([t]) for t in top])
    assert "Paris" in M.tok.decode([top[0]]), "the layer-by-layer model is broken"


def extract_layerwise(key, limit=None):
    import torch
    M = Layerwise(MODELS[key]["repo"])
    tok = M.tok
    lids = letter_ids(tok)
    data = items()[:limit]
    FEATS.mkdir(parents=True, exist_ok=True)
    N = len(data)
    mid_after = (M.n + 1) // 2 - 1                      # the same "middle" as hidden_states[len // 2]
    rec = {"hidden_mid": [None] * N, "route_all": [[] for _ in range(N)], "route_last": [[] for _ in range(N)]}
    shift_probs = np.zeros((N, len(OPTIONS), len(OPTIONS)))
    out = {"letters": [0.0] * N, "tokens": [0] * N, "hidden_last": [None] * N}
    t_start = time.time()
    for shift in range(len(OPTIONS)):
        seqs, orders = [], []
        for it in data:
            text, order = prompt_text(it, shift, tok)
            seqs.append(tok(_chat(tok, text), add_special_tokens=False)["input_ids"])
            orders.append(order)

        def on_layer(L, hs, picks):
            if L == mid_after:
                for i in range(N):
                    rec["hidden_mid"][i] = hs[i][-1].float().numpy().astype(np.float16)
            if picks:
                assert len(picks) == N
                for i, idx in enumerate(picks):
                    top = torch.zeros(idx.shape[0], M.experts, device=idx.device).scatter_(1, idx, 1.0)
                    rec["route_all"][i].append(top.mean(0).cpu().numpy())        # share of tokens each expert served
                    rec["route_last"][i].append(top[-1].cpu().numpy())           # the experts on at the last token
        print(f"{key}: sweep {shift + 1}/{len(OPTIONS)} ({sum(map(len, seqs))} tokens)", flush=True)
        hs, norm = M.run(seqs, on_layer if shift == 0 else None)
        Wh = M.head()
        with torch.no_grad():
            for i in range(N):
                hN = norm(hs[i][-1:].to("cuda"))
                lp = torch.log_softmax((hN @ Wh.T)[0].float(), -1)
                per_letter = torch.stack([torch.logsumexp(lp[ids], 0) for ids in lids])
                probs = torch.softmax(per_letter, 0).cpu().numpy()
                by_option = {name: float(probs[j]) for j, name in enumerate(orders[i])}
                shift_probs[i, shift] = [by_option[name] for name, _ in OPTIONS]
                if shift == 0:
                    out["letters"][i] = float(torch.exp(torch.logsumexp(per_letter, 0)))
                    out["tokens"][i] = len(seqs[i])
                    out["hidden_last"][i] = hN[0].float().cpu().numpy().astype(np.float16)
        del Wh, hs
        print(f"{key}: sweep {shift + 1} done, {time.time() - t_start:.0f}s", flush=True)
    np.savez_compressed(FEATS / f"{key}.npz", number=np.array([it["number"] for it in data]),
                        truth=np.array([it["truth"] for it in data]), form=np.array([bool(it["form"]) for it in data]),
                        letters=np.array(out["letters"]), ms=np.zeros(N), tokens=np.array(out["tokens"]),
                        hidden_mid=np.array(rec["hidden_mid"]), hidden_last=np.array(out["hidden_last"]),
                        route_all=np.array([np.concatenate(r) for r in rec["route_all"]]).astype(np.float32),
                        route_last=np.array([np.concatenate(r) for r in rec["route_last"]]).astype(np.float32),
                        shift_probs=shift_probs)
    print(f"{key}: saved {N} items in {time.time() - t_start:.0f}s")


def depth(key):
    """The router of the first k expert layers only, for every k: how early does the routing hold the answer?"""
    names = [o for o, _ in OPTIONS]
    z = np.load(FEATS / f"{key}.npz", allow_pickle=False)
    y = np.array([names.index(t) for t in z["truth"]])
    E = 64
    RA = z["route_all"].reshape(len(y), -1, E)
    rows = [{"number": int(n), "truth": str(t), "form": bool(f), "choices": []} for n, t, f in zip(z["number"], z["truth"], z["form"])]
    for k in range(1, RA.shape[1] + 1):
        P, _ = cv_head(RA[:, :k].reshape(len(y), -1), y)
        for i, r in enumerate(rows):
            r["choices"].append(names[int(P[i].argmax())])
        print(f"{key} router, first {k:2d} expert layers: {int((P.argmax(1) == y).sum())}/{len(y)}", flush=True)
    (RUNS / f"{key}-route-depth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf8")


def cv_head(X, y, groups=None):
    """Probabilities for every item from a head that never saw it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    P = np.zeros((len(y), len(OPTIONS)))
    fold = np.zeros(len(y), int)
    for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        head = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        head.fit(X[tr], y[tr])
        P[np.ix_(te, head.classes_)] = head.predict_proba(X[te])
        fold[te] = f
    return P, fold


def write(name, numbers, truth, form, P, extra=None):
    names = [o for o, _ in OPTIONS]
    RUNS.mkdir(exist_ok=True)
    rows = []
    for i, num in enumerate(numbers):
        row = {"number": int(num), "truth": str(truth[i]), "form": bool(form[i]),
               "choice": names[int(P[i].argmax())],
               "probabilities": {n: round(float(p), 4) for n, p in zip(names, P[i])}}
        if extra:
            row.update({k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in extra.items()})
        rows.append(row)
    (RUNS / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf8")
    right = sum(r["choice"] == r["truth"] for r in rows)
    hard = [r for r in rows if r["form"] and r["truth"] != "bug"]
    sure = [r for r in rows if max(r["probabilities"].values()) >= 0.9]
    print(f"{name:28s} {right}/{len(rows)} = {right / len(rows):.1%}   hard {sum(r['choice'] == r['truth'] for r in hard)}/{len(hard)}"
          f"   sure>=0.9: {len(sure)} answered, {sum(r['choice'] == r['truth'] for r in sure)} right")


def score(only=None):
    """Fit every head; with `only`, that model's heads alone (the words baseline and the other models' records stay
    exactly as they are)."""
    names = [o for o, _ in OPTIONS]
    data = items()
    truth_all = {it["number"]: it["truth"] for it in data}
    # the plain-code baseline for a trained head: word counts, same head, same folds
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [it["stateStripped"]["title"] + "\n" + it["stateStripped"]["body"] for it in data]
    y = np.array([names.index(it["truth"]) for it in data])
    form = np.array([bool(it["form"]) for it in data])
    numbers = np.array([it["number"] for it in data])
    Xw = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True).fit_transform(texts).toarray()
    if only is None:
        P, fold = cv_head(Xw, y)
        write("words", numbers, [names[i] for i in y], form, P, {"fold": fold})

    for key in ([only] if only else MODELS):
        p = FEATS / f"{key}.npz"
        if not p.exists():
            print(f"{key}: no features yet")
            continue
        z = np.load(p, allow_pickle=False)
        assert list(z["number"]) == list(numbers[:len(z["number"])]), key
        n = len(z["number"])
        yk, fk, nk = y[:n], form[:n], numbers[:n]
        tk = [names[i] for i in yk]
        sp = z["shift_probs"]                                    # [items, shifts, options]
        write(f"{key}-tokens", nk, tk, fk, sp[:, 0, :], {"ms": z["ms"], "letters": z["letters"]})
        write(f"{key}-tokens-rotated", nk, tk, fk, sp.mean(1))   # every option takes a turn at every letter
        for layer in ("hidden_mid", "hidden_last"):
            P, fold = cv_head(z[layer].astype(np.float32), yk)
            write(f"{key}-{layer.replace('_', '-')}", nk, tk, fk, P, {"fold": fold})
        if MODELS[key]["moe"]:
            for feat in ("route_all", "route_last"):
                P, fold = cv_head(z[feat], yk)
                write(f"{key}-{feat.replace('_', '-')}", nk, tk, fk, P, {"fold": fold})
            # control: the same head on routing with the labels shuffled must fall to chance
            rng = np.random.default_rng(SEED)
            ys = rng.permutation(yk)
            P, _ = cv_head(z["route_all"], ys)
            shuffled = float((P.argmax(1) == ys).mean())
            cpath = RUNS / "controls.json"
            controls = json.loads(cpath.read_text("utf8")) if cpath.exists() else {}
            controls[f"{key}-route-all-labels-shuffled"] = {
                "right": int((P.argmax(1) == ys).sum()), "of": int(n), "share": round(shuffled, 4),
                "always_commonest": round(float(np.bincount(yk).max() / n), 4)}
            cpath.write_text(json.dumps(controls, indent=1), "utf8")
            print(f"{key}-route-all, labels shuffled: {shuffled:.1%} (chance is about {np.bincount(yk).max() / n:.1%})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        (extract_layerwise if MODELS[sys.argv[2]].get("layerwise") else extract)(sys.argv[2], lim)
    elif cmd == "score":
        score(sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None)
    elif cmd == "depth":
        depth(sys.argv[2])
    elif cmd == "check":
        check_layerwise(sys.argv[2])
