"""Brick: a programming language no model can already know, so a model that writes it learned it from a block.

The towns test whether a block can carry FACTS a model has never seen. Brick tests whether a block can carry a SKILL
it has never seen: the concepts are ordinary (numbers, adding, repeating, choosing), the words are made up, the way a
programmer who knows Python learns a new language. Every program is checked by running it here.

    brick a 4             a = 4            (a number, or "text" in quotes)
    snap c a b            c = a + b        (a and b: names or numbers)
    grow c a b            c = a * b
    drop a 1              a = a - 1
    show a                print a          (a name, a number or "text")
    stack 3 [             repeat 3 times   (the count: a number or a name)
    ]
    when a > b [          if a > b ... else ...
    ] else [
    ]

  python brick.py                 print the manual, one task of each kind and its program, and run them
"""
import random
import re

MANUAL = """Brick is a small programming language. One command per line.
brick NAME VALUE      sets NAME to a number or to "text"
snap NAME A B         sets NAME to A + B
grow NAME A B         sets NAME to A * B
drop NAME K           takes K away from NAME
show X                prints X (a name, a number or "text")
stack N [             repeats the lines up to ] N times
]
when A > B [          runs the first part if A > B, else the part after ] else [
] else [
]
Example, printing 7 + 5:
brick a 7
brick b 5
snap c a b
show c"""

WORDS = ["amber", "lantern", "harbor", "copper", "meadow", "falcon", "velvet", "pepper", "marble", "willow", "ember",
         "cobalt", "thunder", "saffron", "glacier", "orchid", "pebble", "quartz", "rocket", "sparrow", "tundra", "violet"]

# phrasings: the first three are for training, the last two are only ever used to test (as with the towns)
TASKS = {
    "word": ["Write a Brick program that prints {w}.", "In Brick, print the text {w}.", "Brick code to output {w}.",
             "Using Brick, write a program whose output is {w}.", "Show {w} with a Brick program."],
    "sum": ["Write a Brick program that prints the sum of {x} and {y}.", "In Brick, print {x} plus {y}.",
            "Brick code to output {x} + {y}.", "Using Brick, write a program that adds {x} and {y} and prints the result.",
            "Show the total of {x} and {y} with a Brick program."],
    "product": ["Write a Brick program that prints {x} times {y}.", "In Brick, print the product of {x} and {y}.",
                "Brick code to output {x} multiplied by {y}.", "Using Brick, write a program that multiplies {x} by {y} and prints it.",
                "Show {x} times {y} with a Brick program."],
    "repeat": ["Write a Brick program that prints {w} {n} times.", "In Brick, print {w} {n} times in a row.",
               "Brick code to output {w}, {n} times.", "Using Brick, write a program that repeats {w} {n} times.",
               "Show {w} {n} times with a Brick program."],
    "larger": ["Write a Brick program that prints the larger of {x} and {y}.", "In Brick, print whichever is bigger, {x} or {y}.",
               "Brick code to output the maximum of {x} and {y}.", "Using Brick, write a program that prints the bigger number of {x} and {y}.",
               "Show the larger of {x} and {y} with a Brick program."],
    "count": ["Write a Brick program that counts down from {n} to 1.", "In Brick, print the numbers from {n} down to 1.",
              "Brick code to count down from {n}.", "Using Brick, write a program that prints {n}, then one less, down to 1.",
              "Show a countdown from {n} to 1 with a Brick program."],
}
TRAIN_ASK, TEST_ASK = slice(0, 3), slice(3, 5)


def program(kind, p):
    """The one canonical Brick program for a task (what the trainer learns)."""
    if kind == "word":
        return f'show "{p["w"]}"'
    if kind in ("sum", "product"):
        return f'brick a {p["x"]}\nbrick b {p["y"]}\n{"snap" if kind == "sum" else "grow"} c a b\nshow c'
    if kind == "repeat":
        return f'stack {p["n"]} [\nshow "{p["w"]}"\n]'
    if kind == "larger":
        return f'brick a {p["x"]}\nbrick b {p["y"]}\nwhen a > b [\nshow a\n] else [\nshow b\n]'
    if kind == "count":
        return f'brick a {p["n"]}\nstack {p["n"]} [\nshow a\ndrop a 1\n]'
    raise ValueError(kind)


def expected(kind, p):
    """What a right program prints, line by line."""
    return {"word": lambda: [p["w"]], "sum": lambda: [str(p["x"] + p["y"])], "product": lambda: [str(p["x"] * p["y"])],
            "repeat": lambda: [p["w"]] * p["n"], "larger": lambda: [str(max(p["x"], p["y"]))],
            "count": lambda: [str(i) for i in range(p["n"], 0, -1)]}[kind]()


def params(kind, r):
    two = " ".join(r.sample(WORDS, 2))
    return {"word": {"w": two}, "repeat": {"w": r.choice(WORDS), "n": r.randint(2, 6)}, "count": {"n": r.randint(3, 9)},
            "sum": {"x": r.randint(2, 99), "y": r.randint(2, 99)}, "product": {"x": r.randint(2, 19), "y": r.randint(2, 19)},
            "larger": {"x": r.randint(2, 99), "y": r.randint(2, 99)}}[kind]


def train_pairs(per_kind=50, seed=11):
    """What the trainer sees: every kind of task, asked the three training ways, answered with its program."""
    r = random.Random(seed)
    out = []
    for kind, asks in TASKS.items():
        for i in range(per_kind):
            p = params(kind, r)
            out.append({"q": asks[TRAIN_ASK][i % 3].format(**p), "a": program(kind, p)})
    r.shuffle(out)
    return out


def test_tasks(per_kind=16, seed=99):
    """What the test asks: new numbers and words, in the two phrasings the trainer never saw."""
    r = random.Random(seed)
    out = []
    for kind, asks in TASKS.items():
        for i in range(per_kind):
            p = params(kind, r)
            out.append({"q": asks[TEST_ASK][i % 2].format(**p), "kind": kind, "want": expected(kind, p)})
    return out


# ---------------------------------------------------------------- the interpreter
class BrickError(Exception):
    pass


def extract(reply):
    """The program in a reply: inside a code fence if there is one, else the lines that look like Brick."""
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", reply, re.S)
    text = m.group(1) if m else reply
    keep = [l.strip() for l in text.splitlines()
            if re.match(r"^(brick|snap|grow|drop|show|stack|when|\]|\] else \[)\b", l.strip()) or l.strip() in ("]", "] else [")]
    return "\n".join(keep)


def run(src, max_steps=2000):
    """Run a Brick program; return the printed lines. Raises BrickError on anything that is not Brick."""
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    env, out, steps = {}, [], [0]

    def val(tok):
        if re.fullmatch(r"-?\d+", tok):
            return int(tok)
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            return tok[1:-1]
        if tok in env:
            return env[tok]
        raise BrickError(f"unknown name {tok!r}")

    def block_end(i):
        """Index of the ] closing the block opened on line i-1, and of an ] else [ at the same depth, if any."""
        depth, mid = 1, None
        for j in range(i, len(lines)):
            if lines[j].endswith("["):
                if lines[j] == "] else [" and depth == 1:
                    mid = j
                    continue
                depth += 1
            elif lines[j] == "]":
                depth -= 1
                if depth == 0:
                    return j, mid
        raise BrickError("a [ is never closed")

    def exec_range(a, b):
        i = a
        while i < b:
            steps[0] += 1
            if steps[0] > max_steps:
                raise BrickError("too many steps")
            ln = lines[i]
            toks = re.findall(r'"[^"]*"|\S+', ln)
            op = toks[0]
            if op == "brick" and len(toks) == 3:
                env[toks[1]] = val(toks[2])
            elif op in ("snap", "grow") and len(toks) == 4:
                x, y = val(toks[2]), val(toks[3])
                if not (isinstance(x, int) and isinstance(y, int)):
                    raise BrickError("snap and grow take numbers")
                env[toks[1]] = x + y if op == "snap" else x * y
            elif op == "drop" and len(toks) == 3:
                env[toks[1]] = val(toks[1]) - val(toks[2])
            elif op == "show" and len(toks) == 2:
                out.append(str(val(toks[1])))
            elif op == "stack" and len(toks) == 3 and toks[2] == "[":
                end, _ = block_end(i + 1)
                for _ in range(int(val(toks[1]))):
                    exec_range(i + 1, end)
                i = end
            elif op == "when" and len(toks) == 5 and toks[2] == ">" and toks[4] == "[":
                end, mid = block_end(i + 1)
                if val(toks[1]) > val(toks[3]):
                    exec_range(i + 1, mid if mid is not None else end)
                elif mid is not None:
                    exec_range(mid + 1, end)
                i = end
            else:
                raise BrickError(f"not Brick: {ln!r}")
            i += 1

    exec_range(0, len(lines))
    return out


def right(reply, want):
    try:
        return run(extract(reply)) == want
    except (BrickError, ValueError, TypeError, RecursionError):
        return False


if __name__ == "__main__":
    print(MANUAL, "\n")
    for t in test_tasks(1):
        print(t["q"])
    r = random.Random(0)
    for kind in TASKS:
        p = params(kind, r)
        src = program(kind, p)
        assert run(src) == expected(kind, p), kind
    pairs, tests = train_pairs(), test_tasks()
    assert all(run(p["a"]) is not None for p in pairs)
    assert not {p["q"] for p in pairs} & {t["q"] for t in tests}
    print(f"\n{len(pairs)} training pairs, {len(tests)} test tasks; every canonical program runs and prints what it should")
