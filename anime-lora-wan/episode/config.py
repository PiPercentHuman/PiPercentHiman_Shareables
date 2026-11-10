"""Shared settings: the paths in paths.json (copy paths.example.json and edit it) and the chapter file."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def paths():
    f = REPO / "paths.json"
    if not f.exists():
        sys.exit(f"missing {f}: copy paths.example.json to paths.json and fill in your paths")
    return json.loads(f.read_text("utf8"))


def chapter(folder):
    """The chapter folder holds chapter.json; every output (takes, voices, sfx, cut) is written next to it."""
    d = Path(folder).resolve()
    return d, json.loads((d / "chapter.json").read_text("utf8"))


def lines(ch):
    return {ln["id"]: ln for ln in ch["lines"]}
