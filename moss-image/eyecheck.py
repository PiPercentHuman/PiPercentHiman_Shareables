"""What a person saw, image by image, where OCR is not enough.

OCR reads letter-spaced capitals, small type and Chinese badly (on the
plain-code control it read 王 as 壬), and it only looks for the words it
was asked to find, so it cannot see text nobody asked for. Each image
below was opened and read by eye.

  spelled    every target word (or character) is there and spelled right
  typos      target words that are there but misspelled
  invented   text appears that nobody asked for, other than a plain
             heading such as "Menu": made-up dishes, prices, garbled lines
  dishes     (menus) made-up dishes added to the list

  python eyecheck.py     -> eyecheck.jsonl
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S6 = range(6)


def rows():
    def r(task, v, c, seeds, spelled=True, invented=(), typos=None, note="",
          dishes=()):
        for s in seeds:
            yield {"id": f"{task}/{v}/{c}/s{s}", "spelled": spelled(s) if
                   callable(spelled) else spelled, "invented": s in invented,
                   "dishes": s in dishes,
                   "typos": (typos or {}).get(s, []), "note": note}

    yield from r("poster", "title", "moss", S6)
    yield from r("poster", "title", "moss-card", S6)
    yield from r("poster", "title", "moss-fast", S6, invented={4},
                 note="s4: a garbled footer line under the picture")
    yield from r("menu", "items-3", "moss", S6, invented={0, 1, 3, 5},
                 dishes={0, 1, 3, 5},
                 note="made-up dishes added below the three asked for")
    yield from r("menu", "items-3-only", "moss", S6, invented={0, 1},
                 note="s0, s1: a garbled heading; no made-up dishes")
    yield from r("menu", "items-8", "moss", S6)
    yield from r("menu", "items-8", "moss-card", S6)
    yield from r("menu", "items-8", "moss-fast", S6, invented={0},
                 note="s0: a made-up heading 'Ganu'")
    yield from r("menu", "items-20", "moss", S6, spelled=lambda s: s != 4,
                 typos={4: ["Pancakes -> Panckes"]})
    yield from r("menu", "items-20-no-breaks", "moss", S6, spelled=None,
                 invented={0, 1, 2, 3, 4, 5}, dishes={0, 1, 2, 4},
                 note="s0-s4 garbled lines; s5 added prices")
    yield from r("concert", "sentence", "moss", S6,
                 note="title wraps onto two lines in all six")
    yield from r("concert", "layout", "moss", S6,
                 note="title on one line, inside its box, in all six")
    yield from r("concert", "layout", "moss-fast", S6,
                 note="title on one line, inside its box, in all six")
    yield from r("comic", "layout", "moss", range(4))
    yield from r("sign", "chinese", "moss", S6,
                 note="signboard and lantern both right")
    yield from r("edit", "title", "moss", S6,
                 note="ALPINE SUNRISE, no trace of EXPRESS")

    # Qwen-Image 2.1, the same prompts
    yield from r("poster", "title", "qwen", S6)
    yield from r("menu", "items-3", "qwen", S6, invented={0, 2, 5},
                 dishes={0, 2, 5},
                 note="made-up or repeated dishes below the three")
    yield from r("menu", "items-8", "qwen", S6, spelled=False,
                 typos={0: ["Orange -> Ohange"], 1: ["Bread -> Broad"],
                        3: ["Orange -> Ohange"]},
                 note="Cheese Toast, the eighth item, missing in all six")
    yield from r("menu", "items-20", "qwen", S6, spelled=False,
                 typos={0: ["Potato -> Potaío", "Honey -> Honry"],
                        1: ["Ravioli -> Ravieli"],
                        2: ["Ravioli -> Ravieli"],
                        3: ["Ravioli -> Ravíoli", "Quiche -> Quíche"],
                        4: ["Bread -> Broad", "Ravioli -> Ravieli"],
                        5: ["Ravioli -> Raviéli"]},
                 note="s1: Honey Pancakes missing")
    yield from r("concert", "sentence", "qwen", S6,
                 note="s3: the title above the tree, not below it")
    yield from r("sign", "chinese", "qwen", S6,
                 note="signboard right in all six; the lantern text is often"
                      " on a paper notice by the door instead")


def main():
    out = list(rows())
    (HERE / "eyecheck.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), "utf8")
    print(len(out), "images checked by eye")


if __name__ == "__main__":
    main()
