"""blocks: a package manager for model blocks (LoRA adapters), the way npm or pip is one for code.

A project says which base model it builds on and which blocks it needs; a registry holds published blocks with
versions; `install` resolves the versions, checks that every block FITS the project's base (a block only fits the base
it was trained for: its numbers have that base's shape), copies them in and pins them in a lock file. Then `load`
gives you the base with every installed block on board, named, and a one-line description of each for tool calling.

  python blocks.py publish <adapter dir> --name ostwick --version 1.0.0 --about "knows the town of Ostwick"
  python blocks.py search [word]
  python blocks.py init --base Qwen/Qwen3.5-2B          (in a project folder: writes blocks.json)
  python blocks.py add ostwick@^1.0 brick               (adds to blocks.json and installs)
  python blocks.py install                              (installs everything blocks.json asks for, writes blocks.lock)
  python blocks.py list

The registry is a folder (BLOCKS_REGISTRY, default ./registry). Nothing here needs a GPU.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

REG = Path(os.environ.get("BLOCKS_REGISTRY", Path.cwd() / "registry"))
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
FILES = ("adapter_config.json", "adapter_model.safetensors")


class Refused(Exception):
    pass


# ---------------------------------------------------------------- what a block is built for
def block_base(d):
    return json.loads((Path(d) / "adapter_config.json").read_text("utf8"))["base_model_name_or_path"]


def block_width(d):
    """The width a block's numbers expect from the base: the input side of its first low-rank matrix."""
    with open(Path(d) / "adapter_model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    first = sorted(k for k in head if k.endswith("lora_A.weight"))[0]
    return head[first]["shape"][1]


def base_width(repo):
    """The base model's hidden width, from its own config (the cached copy; no download)."""
    snaps = sorted((HF / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/config.json"))
    if not snaps:
        return None
    c = json.loads(snaps[-1].read_text("utf8"))
    return (c.get("text_config") or c).get("hidden_size")


def fits(block_dir, base):
    """None if the block fits the base, else the reason it does not."""
    built, w = block_base(block_dir), block_width(block_dir)
    bw = base_width(base)
    if built != base:
        return f"built for {built} (width {w}); this project's base is {base} (width {bw})"
    if bw is not None and w != bw:
        return f"its numbers are {w} wide; {base} is {bw} wide"
    return None


def sha(d):
    h = hashlib.sha256()
    for f in FILES:
        h.update((Path(d) / f).read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- the registry
def index():
    p = REG / "index.json"
    return json.loads(p.read_text("utf8")) if p.exists() else {}


def publish(src, name, version, about):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise Refused("version must look like 1.0.0")
    idx = index()
    dest = REG / name / version
    if dest.exists():
        raise Refused(f"{name}@{version} is already published; publish a new version")
    dest.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(Path(src) / f, dest / f)
    if (Path(src) / "block.json").exists():
        shutil.copy2(Path(src) / "block.json", dest / "block.json")
    idx.setdefault(name, {})[version] = {"base": block_base(dest), "width": block_width(dest), "about": about,
                                          "sha": sha(dest), "mb": round(sum((dest / f).stat().st_size for f in FILES) / 1e6, 1)}
    (REG / "index.json").write_text(json.dumps(idx, indent=1), "utf8")
    return idx[name][version]


def vkey(v):
    return tuple(int(x) for x in v.split("."))


def resolve(name, want, base):
    """The newest published version of `name` that matches `want` (1.2.0 exact, ^1.2 same major, * any) and fits base."""
    idx = index()
    if name not in idx:
        raise Refused(f"no block called {name!r} in the registry")
    vs = sorted(idx[name], key=vkey, reverse=True)
    if want not in ("*", "", None):
        if want.startswith("^"):
            floor = tuple(int(x) for x in (want[1:].split(".") + ["0", "0"])[:3])
            vs = [v for v in vs if vkey(v)[0] == floor[0] and vkey(v) >= floor]
        else:
            vs = [v for v in vs if v == want]
    if not vs:
        raise Refused(f"no version of {name} matches {want}")
    reasons = []
    for v in vs:
        why = fits(REG / name / v, base)
        if why is None:
            return v
        reasons.append(f"{name}@{v}: {why}")
    raise Refused("does not fit: " + "; ".join(reasons))


# ---------------------------------------------------------------- a project
def project(d="."):
    p = Path(d) / "blocks.json"
    if not p.exists():
        raise Refused("no blocks.json here: run `blocks init --base <model>` first")
    return json.loads(p.read_text("utf8"))


def save_project(cfg, d="."):
    (Path(d) / "blocks.json").write_text(json.dumps(cfg, indent=1) + "\n", "utf8")


def install(d="."):
    cfg = project(d)
    lock, out = {}, []
    for name, want in cfg["blocks"].items():
        v = resolve(name, want, cfg["base"])
        src, dest = REG / name / v, Path(d) / "blocks" / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        lock[name] = {"version": v, "sha": sha(dest)}
        out.append(f"+ {name}@{v}  {index()[name][v]['mb']} MB  {index()[name][v]['about']}")
    (Path(d) / "blocks.lock").write_text(json.dumps({"base": cfg["base"], "blocks": lock}, indent=1) + "\n", "utf8")
    return out


def load(d="."):
    """The base with every installed block on board (named), and each block's description for tool calling."""
    from peft import PeftModel
    sys.path.insert(0, os.environ.get("BLOCKS_TRAINER", ""))
    from train import load_base
    lock = json.loads((Path(d) / "blocks.lock").read_text("utf8"))
    tok, model = load_base(lock["base"])
    names = list(lock["blocks"])
    for n in names:
        if sha(Path(d) / "blocks" / n) != lock["blocks"][n]["sha"]:
            raise Refused(f"{n} changed since it was installed")
    model = PeftModel.from_pretrained(model, str(Path(d) / "blocks" / names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(str(Path(d) / "blocks" / n), adapter_name=n)
    idx = index()
    about = {n: idx[n][lock["blocks"][n]["version"]]["about"] for n in names}
    return tok, model, about


def main(argv=None):
    ap = argparse.ArgumentParser(prog="blocks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish"); p.add_argument("src"); p.add_argument("--name", required=True)
    p.add_argument("--version", required=True); p.add_argument("--about", required=True)
    p = sub.add_parser("search"); p.add_argument("word", nargs="?", default="")
    p = sub.add_parser("init"); p.add_argument("--base", required=True)
    p = sub.add_parser("add"); p.add_argument("specs", nargs="+")
    sub.add_parser("install"); sub.add_parser("list")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "publish":
            r = publish(a.src, a.name, a.version, a.about)
            print(f"published {a.name}@{a.version}  for {r['base']}  {r['mb']} MB")
        elif a.cmd == "search":
            for n, vs in sorted(index().items()):
                v = max(vs, key=vkey)
                if a.word.lower() in (n + vs[v]["about"]).lower():
                    print(f"{n:12s} {v:8s} {vs[v]['base']:20s} {vs[v]['about']}")
        elif a.cmd == "init":
            save_project({"base": a.base, "blocks": {}})
            print(f"blocks.json: base {a.base}")
        elif a.cmd == "add":
            cfg = project()
            for s in a.specs:
                n, _, want = s.partition("@")
                resolve(n, want or "*", cfg["base"])          # refuse before writing anything
                cfg["blocks"][n] = want or "*"
            save_project(cfg)
            print("\n".join(install()))
        elif a.cmd == "install":
            print("\n".join(install()))
        elif a.cmd == "list":
            lock = json.loads(Path("blocks.lock").read_text("utf8"))
            idx = index()
            print(f"base {lock['base']}")
            for n, b in lock["blocks"].items():
                print(f"  {n:12s} {b['version']:8s} {idx[n][b['version']]['about']}")
    except Refused as e:
        print(f"refused: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
