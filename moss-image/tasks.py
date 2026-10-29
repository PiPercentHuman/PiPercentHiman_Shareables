"""The task ladder: every prompt, layout and target text, in one place.

Each task changes one thing from the one before it. Every image is
1024 x 1024. The words a task should render are listed in `target`, so
the scorer can read the image back and count them.
"""
import json

W = H = 1024
SEEDS = range(6)

MENU = ["Black Coffee", "Green Tea", "Lemon Tart", "Tomato Soup",
        "Garlic Bread", "Apple Pie", "Orange Juice", "Cheese Toast",
        "Mushroom Risotto", "Chicken Salad", "Banana Muffin",
        "Mint Lemonade", "Pumpkin Ravioli", "Spinach Quiche",
        "Berry Smoothie", "Vanilla Gelato", "Grilled Salmon",
        "Potato Wedges", "Honey Pancakes", "Chocolate Brownie"]
MENU_SCENE = ("A printed cafe menu card lying on a wooden table, photographed"
              " from above in soft daylight. Black serif type on cream paper.")


def menu(n, breaks=True, only=False):
    items = MENU[:n]
    if only:
        return MENU_SCENE + " The card has a lot of empty space and only" \
            f" {len(items)} lines of text, nothing else:\n" + "\n".join(items)
    if breaks:
        return MENU_SCENE + " The menu lists, one item per line:\n" + \
            "\n".join(items)
    return MENU_SCENE + " The menu lists " + ", ".join(items) + "."


def layout(caption, items):
    """MOSS layout JSON. Boxes are given top-left (x0, y0, x1, y1) and
    stored the way the model wants them: [y0, x0, y1, x1], measured in
    pixels from the centre of the canvas."""
    out = []
    for it in items:
        x0, y0, x1, y1 = it["box"]
        row = {"label": it["label"], "caption": it["caption"],
               "bbox": [y0 - H // 2, x0 - W // 2, y1 - H // 2, x1 - W // 2]}
        if "text" in it:
            row["text"] = it["text"]
        out.append(row)
    return json.dumps({"summary": {"caption": caption}, "items": out,
                       "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False)


POSTER = ("A vintage travel poster for a mountain town, snowy peaks under a"
          " pale sky, a red train crossing a stone bridge. Across the top,"
          " the title \"ALPINE EXPRESS\" in large cream capital letters.")

CONCERT = [
    {"label": "text", "caption": "small white capitals", "box": (262, 50, 762, 130),
     "text": "LIVE IN CONCERT"},
    {"label": "object", "caption": "a glowing apple tree with golden fruit",
     "box": (162, 150, 862, 700)},
    {"label": "text", "caption": "huge bold cream serif title",
     "box": (60, 720, 964, 880), "text": "MIDNIGHT ORCHARD"},
    {"label": "text", "caption": "small gold numerals and capitals",
     "box": (650, 900, 970, 975), "text": "OCTOBER 31"},
]
CONCERT_SCENE = ("A concert poster on a deep night-blue background with soft"
                 " stars, flat screen-print style.")
CONCERT_SENTENCE = (CONCERT_SCENE + " At the very top, small white capitals"
                    " read \"LIVE IN CONCERT\". In the centre, a glowing apple"
                    " tree with golden fruit. Below the tree, a huge bold cream"
                    " serif title reads \"MIDNIGHT ORCHARD\". In the bottom"
                    " right corner, small gold text reads \"OCTOBER 31\".")

COMIC = [
    {"label": "object", "caption": "panel 1: a boy searching under a sofa",
     "box": (16, 16, 1008, 500)},
    {"label": "text", "caption": "speech bubble, black comic lettering",
     "box": (560, 40, 980, 170), "text": "WHERE IS MY CAT?"},
    {"label": "object", "caption": "panel 2: a girl pointing at a cardboard box",
     "box": (16, 524, 500, 1008)},
    {"label": "text", "caption": "speech bubble, black comic lettering",
     "box": (40, 548, 476, 660), "text": "CHECK THE BOX!"},
    {"label": "object", "caption": "panel 3: an orange cat inside the box",
     "box": (524, 524, 1008, 1008)},
    {"label": "text", "caption": "speech bubble, black comic lettering",
     "box": (548, 548, 984, 660), "text": "FOUND HIM!"},
]
COMIC_SCENE = ("A three-panel comic page, clean ink line art and flat bright"
               " colours, white gutters between panels.")

TAGS = ("best quality, highres, anime coloring, 1girl, solo, short hair,"
        " silver hair, red scarf, rooftop, sunset, city skyline, wind,"
        " scarf blowing, standing, looking at viewer")
ANIME_SENTENCE = ("An anime illustration of a girl with short silver hair and"
                  " a red scarf standing on a rooftop at sunset, the city"
                  " skyline behind her, the wind blowing her scarf, looking at"
                  " the viewer.")

SIGN = ("A night street in an old Chinese town. Above a small noodle shop"
        " hangs a red wooden signboard with large gold characters \"老王面馆\","
        " and a white paper lantern beside the door reads \"手工拉面\".")

EDIT = ("Change only the title text to \"ALPINE SUNRISE\". Keep the"
        " mountains, the train, the bridge, the colours and the composition"
        " unchanged.")


def t(prompt, target="", seeds=SEEDS, boxes=None, lang="eng", **kw):
    return dict(prompt=prompt, target=target, seeds=list(seeds),
                boxes=boxes, lang=lang, **kw)


def boxes_of(items):
    return [(it["text"], it["box"]) for it in items if "text" in it]


# task -> variant -> spec.  Contenders are listed in run.py.
TASKS = {
    "photo": {
        "sentence": t("A close-up photograph of an old fisherman mending a"
                      " net on a wooden dock at dawn, weathered hands, mist"
                      " on the water, shallow depth of field.", seeds=range(2)),
    },
    "anime": {
        "sentence": t(ANIME_SENTENCE, seeds=range(3)),
        "tags": t(TAGS, seeds=range(3)),
    },
    "poster": {
        "title": t(POSTER, "ALPINE EXPRESS"),
    },
    "menu": {
        "items-3": t(menu(3), " ".join(MENU[:3])),
        "items-3-only": t(menu(3, only=True), " ".join(MENU[:3])),
        "items-8": t(menu(8), " ".join(MENU[:8])),
        "items-20": t(menu(20), " ".join(MENU)),
        "items-20-no-breaks": t(menu(20, breaks=False), " ".join(MENU)),
    },
    "concert": {
        "sentence": t(CONCERT_SENTENCE, "LIVE IN CONCERT MIDNIGHT ORCHARD"
                      " OCTOBER 31", boxes=boxes_of(CONCERT)),
        "layout": t(layout(CONCERT_SCENE, CONCERT), "LIVE IN CONCERT MIDNIGHT"
                    " ORCHARD OCTOBER 31", boxes=boxes_of(CONCERT)),
    },
    "comic": {
        "layout": t(layout(COMIC_SCENE, COMIC), "WHERE IS MY CAT CHECK THE"
                    " BOX FOUND HIM", seeds=range(4), boxes=boxes_of(COMIC)),
    },
    "sign": {
        "chinese": t(SIGN, "老王面馆手工拉面", lang="chi_sim"),
    },
    "edit": {
        "title": t(EDIT, "ALPINE SUNRISE", source=("poster", "title")),
    },
}
