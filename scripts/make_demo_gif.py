#!/usr/bin/env python3
"""Render docs/img/demo.gif — an animated terminal that types out a real
`pairsbot` session (screen -> optimize -> status). Numbers are verbatim from an
actual run. GIF (not SVG) because GitHub strips SVG animation in README images.

Usage:  python scripts/make_demo_gif.py
Font:   DejaVuSansMono, bundled with matplotlib (no system font needed).
"""
from __future__ import annotations

import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont

# ---- palette (GitHub dark) ---------------------------------------------------
BG      = (13, 17, 23)
BAR     = (22, 27, 34)
BORDER  = (48, 54, 61)
RED     = (255, 95, 86)
YELLOW  = (255, 189, 46)
GREEN   = (39, 201, 63)
PROMPT  = (63, 185, 80)
CMD     = (230, 237, 243)
OUT     = (139, 148, 158)
BLUE    = (88, 166, 255)
POS     = (63, 185, 80)
NEG     = (248, 81, 73)
DIM     = (110, 118, 129)
CURSOR  = (63, 185, 80)

W, H = 820, 320
X0 = 24
FS = 16

_ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSansMono.ttf")
FONT = ImageFont.truetype(_ttf, FS)
_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
CW = _probe.textlength("M", font=FONT)      # monospace advance
LH = 26                                      # line height

# ---- content: each line = list of (text, color) segments ---------------------
def seg(*parts):
    return list(parts)

L_screen_cmd = seg(("$", PROMPT), (" pairsbot screen", CMD))
L_screen_out = seg(("Selected ", OUT), ("LTC/XLM", BLUE),
                   ("  beta=0.2980  p=0.02166  (frozen)", OUT))
L_opt_cmd = seg(("$", PROMPT), (" pairsbot optimize", CMD))
L_opt_in  = seg(("in-sample      Sharpe ", OUT), ("1.68", POS),
                ("    return ", OUT), ("+61.89%", POS), ("   (looks great)", DIM))
L_opt_out = seg(("out-of-sample  tuned ", OUT), ("-16.37%", NEG),
                ("    default ", OUT), ("-17.76%", NEG), ("   (loses)", DIM))
L_stat_cmd = seg(("$", PROMPT), (" pairsbot status", CMD))
L_stat_out = seg(("run #1 (live, ", OUT), ("LTC/XLM", BLUE),
                 (")  ·  equity ", OUT), ("$10,000", CMD), ("  ·  drawdown 0.00%", OUT))
L_prompt = seg(("$", PROMPT))

# line id -> (segments, top-y)
LINES = {
    "screen_cmd": (L_screen_cmd, 52),
    "screen_out": (L_screen_out, 78),
    "opt_cmd":    (L_opt_cmd, 122),
    "opt_in":     (L_opt_in, 148),
    "opt_out":    (L_opt_out, 174),
    "stat_cmd":   (L_stat_cmd, 218),
    "stat_out":   (L_stat_out, 244),
    "prompt":     (L_prompt, 286),
}


def flatten(segments):
    """[(text,color)] -> [(char,color)]"""
    out = []
    for text, color in segments:
        for ch in text:
            out.append((ch, color))
    return out


def base_frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, W - 2, H - 2], radius=12, outline=BORDER, width=1)
    d.rectangle([1, 1, W - 2, 38], fill=BAR)
    d.rounded_rectangle([1, 1, W - 2, 50], radius=12, fill=BAR)  # top corners
    d.rectangle([1, 26, W - 2, 38], fill=BAR)
    for cx, col in ((24, RED), (44, YELLOW), (64, GREEN)):
        d.ellipse([cx - 6, 14, cx + 6, 26], fill=col)
    d.text((W / 2, 12), "pairsbot demo", font=FONT, fill=DIM, anchor="ma")
    return img


def draw_line(d, chars, top, reveal=None):
    n = len(chars) if reveal is None else reveal
    for i in range(n):
        ch, color = chars[i]
        d.text((X0 + i * CW, top), ch, font=FONT, fill=color)


# revealed[id] = number of chars shown (full length once complete)
revealed: dict[str, int] = {}
frames: list[Image.Image] = []
durs: list[int] = []


def render(caret=None, cursor_on=False):
    img = base_frame()
    d = ImageDraw.Draw(img)
    for lid, (segs, top) in LINES.items():
        if lid in revealed:
            draw_line(d, flatten(segs), top, revealed[lid])
    if caret is not None and cursor_on:
        cx, top = caret
        d.rectangle([cx, top + 1, cx + CW * 0.85, top + FS + 2], fill=CURSOR)
    frames.append(img)


def caret_at(lid, k):
    _, top = LINES[lid]
    return (X0 + k * CW, top)


BLOCKS = [
    ("screen_cmd", ["screen_out"]),
    ("opt_cmd", ["opt_in", "opt_out"]),
    ("stat_cmd", ["stat_out"]),
]

for cmd_id, outs in BLOCKS:
    chars = flatten(LINES[cmd_id][0])
    for k in range(1, len(chars) + 1):          # type the command char by char
        revealed[cmd_id] = k
        render(caret_at(cmd_id, k), cursor_on=True)
        durs.append(55)
    for t in range(2):                          # brief blink before "executing"
        render(caret_at(cmd_id, len(chars)), cursor_on=(t == 0))
        durs.append(130)
    for o in outs:                              # reveal output instantly
        revealed[o] = None
    render(cursor_on=False)
    durs.append(760)
    render(cursor_on=False)                     # gap before next command
    durs.append(320)

revealed["prompt"] = None                       # final ready prompt, blinking
for t in range(6):
    render(caret_at("prompt", 2), cursor_on=(t % 2 == 0))
    durs.append(430)
render(caret_at("prompt", 2), cursor_on=True)
durs.append(1400)                               # hold before loop

out_path = os.path.join("docs", "img", "demo.gif")
frames[0].save(out_path, save_all=True, append_images=frames[1:],
               duration=durs, loop=0, optimize=True, disposal=2)
kb = os.path.getsize(out_path) / 1024
print(f"wrote {out_path}: {len(frames)} frames, {kb:.0f} KB, ~{sum(durs)/1000:.1f}s loop")
