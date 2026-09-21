"""Greek-key (meander) gold glitter-embroidery trim strip for the ATA clothing set.

Outputs (all in OUT_DIR):
  greek_key_strip.png    1024x160 RGBA, horizontally tileable, period 128 px (8 repeats)
  greek_key_strip_v.png  160x1024 RGBA, vertical version for side seams (same mask rotated
                         90 deg CCW, re-embossed so the light still comes from top-left)
  greek_key_2x.png       2048x160 RGB preview: strip tiled twice over black fabric (seam check)

Design (per 128 px period, 10 cells wide x 9 cells tall, cell = 12.8 px snapped to whole px):
  continuous top rail, a descender per unit, and a square spiral hooking inwards with 1.5
  turns - like the reference panel.  Drawn as axis-aligned rectangles at 4x supersampling on
  integer pixel boundaries so the corners land crisp at 1:1.

Deterministic (fixed seeds).  Run from the work dir:  python3 gen/gen_greek_key.py
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ata_style import *  # noqa: E402,F401

# ---------------------------------------------------------------- parameters
W, H = 1024, 160
PERIOD = 128
SEED = 7

# 'X' = gold cell.  10 columns per period, 9 rows.
PATTERN = [
    "XXXXXXXXXX",  # continuous top rail
    "........X.",
    "XXXXXXX.X.",
    "X.....X.X.",
    "X.XXX.X.X.",
    "X.X...X.X.",
    "X.XXXXX.X.",
    "X.......X.",
    "XXXXXXXXX.",
]
NCOL = len(PATTERN[0])
NROW = len(PATTERN)

# vertical layout (px from top): margin, thin line, gap, thin line, gap, meander, ... mirrored
MARGIN = 2
LINE_T = 5          # thin double-line thickness
LINE_GAP = 4        # gap between the two thin lines
BAND_GAP = 7        # gap between the double lines and the meander band
MEANDER_H = H - 2 * (MARGIN + LINE_T + LINE_GAP + LINE_T + BAND_GAP)       # = 114
MEANDER_Y0 = MARGIN + LINE_T + LINE_GAP + LINE_T + BAND_GAP                  # = 23

EDGE_WOBBLE = 0.6   # px of organic edge displacement (stitched look); 0 = razor-crisp
BEAD_PITCH = 4      # px pitch of the tiny glitter-bead bumps along the threads
BEAD_AMOUNT = 0.22
EMBOSS_DEPTH = 2.2
EMBOSS_STRENGTH = 0.46

# ---------------------------------------------------------------- geometry
def _bounds(n, total, offset=0):
    return [offset + int(round(i * total / n)) for i in range(n + 1)]

XB = _bounds(NCOL, PERIOD)                   # column boundaries inside one period
YB = _bounds(NROW, MEANDER_H, MEANDER_Y0)    # row boundaries of the meander band

THIN_LINES = []  # (y0, y1) of the four thin rails
_y = MARGIN
THIN_LINES.append((_y, _y + LINE_T)); _y += LINE_T + LINE_GAP
THIN_LINES.append((_y, _y + LINE_T))
_y = H - MARGIN - LINE_T
THIN_LINES.append((_y, _y + LINE_T)); _y -= LINE_T + LINE_GAP
THIN_LINES.append((_y, _y + LINE_T))


def draw_strip(d, S):
    """Rasterise the whole 1024x160 strip at supersample S (PIL rectangle is end-inclusive)."""
    for (y0, y1) in THIN_LINES:
        d.rectangle([0, y0 * S, W * S - 1, y1 * S - 1], fill=255)
    for p in range(W // PERIOD):
        ox = p * PERIOD
        for r, row in enumerate(PATTERN):
            y0, y1 = YB[r], YB[r + 1]
            for c, ch in enumerate(row):
                if ch != 'X':
                    continue
                x0, x1 = ox + XB[c], ox + XB[c + 1]
                d.rectangle([x0 * S, y0 * S, x1 * S - 1, y1 * S - 1], fill=255)


# ---------------------------------------------------------------- helpers
def stitch_edge(mask, seed, amp):
    """Displace the mask edge by a tiny periodic noise field so it reads as thread, not vector."""
    if amp <= 0:
        return mask
    h, w = mask.shape
    nx = (fbm(h, w, octaves=3, base_scale=5, seed=seed) - 0.5) * 2 * amp
    ny = (fbm(h, w, octaves=3, base_scale=5, seed=seed + 1) - 0.5) * 2 * amp
    m = warp(mask, nx, ny)
    return np.clip((m - 0.5) * 1.8 + 0.5, 0, 1).astype(np.float32)


def bead_bumps(h, w, pitch, seed):
    """Periodic lattice of small bright bumps (glitter beads) 0..1, tileable when pitch | h, w."""
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cx = 0.5 + 0.5 * np.cos(2 * np.pi * X / pitch)
    cy = 0.5 + 0.5 * np.cos(2 * np.pi * Y / pitch)
    lattice = (cx * cy) ** 1.5
    n = fbm(h, w, octaves=2, base_scale=6, seed=seed)
    return np.clip(lattice * (0.55 + 0.9 * n), 0, 1).astype(np.float32)


def gold_embroidery(h, w, seed):
    fill = gold_fill(h, w, seed=seed, glitter=0.7)
    beads = bead_bumps(h, w, BEAD_PITCH, seed + 5)
    fill = fill * (1 - BEAD_AMOUNT * 0.5) + fill * beads[..., None] * BEAD_AMOUNT * 1.6
    # a few extra hot sparkles
    sp = sparkle(h, w, density=0.003, seed=seed + 9, size=0.6)
    fill = fill + sp[..., None] * np.array([1.0, 0.97, 0.85])[None, None, :] * 0.7
    return np.clip(fill, 0, 1).astype(np.float32)


def emboss_wrapped(mask, fill, axis, pad=48):
    """emboss_element with wrap padding along `axis` so the bevel is seamless across the tile edge."""
    if axis == 1:
        m = np.pad(mask, ((0, 0), (pad, pad)), mode='wrap')
        f = np.pad(fill, ((0, 0), (pad, pad), (0, 0)), mode='wrap')
        el = emboss_element(m, f, depth=EMBOSS_DEPTH, strength=EMBOSS_STRENGTH)
        return el[:, pad:-pad]
    m = np.pad(mask, ((pad, pad), (0, 0)), mode='wrap')
    f = np.pad(fill, ((pad, pad), (0, 0), (0, 0)), mode='wrap')
    el = emboss_element(m, f, depth=EMBOSS_DEPTH, strength=EMBOSS_STRENGTH)
    return el[pad:-pad, :]


def finish(el, mask):
    """Slight inner-edge darkening (thread shadow) and a faint dark rim so the trim sits on fabric."""
    inner = np.clip(mask - erode(mask, 1.5), 0, 1)
    rgb = el[..., :3] * (1 - 0.18 * inner[..., None])
    rim = np.clip(dilate(mask, 1.0) - mask, 0, 1) * 0.45
    rgb = over(rgb, np.zeros_like(rgb) + 0.02, rim * (1 - mask))
    alpha = np.clip(el[..., 3] + rim, 0, 1)
    return rgba(rgb, alpha)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    work_dir = os.path.dirname(OUT_DIR)

    # 1. crisp vector mask at 4x
    mask = mask_from_pil_draw(H, W, draw_strip, supersample=4)
    mask = stitch_edge(mask, SEED + 20, EDGE_WOBBLE)

    # 2. gold embroidery fill + emboss (wrap-padded horizontally for seamless tiling)
    fill = gold_embroidery(H, W, SEED)
    el = emboss_wrapped(mask, fill, axis=1)
    el = finish(el, mask)
    assert el.shape == (H, W, 4), el.shape
    save(el, os.path.join(OUT_DIR, 'greek_key_strip.png'))

    # 3. vertical version: rotate the mask 90 deg CCW, re-emboss with light still from top-left
    mask_v = np.ascontiguousarray(np.rot90(mask, k=1))          # (1024, 160)
    fill_v = gold_embroidery(W, H, SEED + 1)
    el_v = emboss_wrapped(mask_v, fill_v, axis=0)
    el_v = finish(el_v, mask_v)
    assert el_v.shape == (W, H, 4), el_v.shape
    save(el_v, os.path.join(OUT_DIR, 'greek_key_strip_v.png'))

    # 4. tileability preview: strip twice over black fabric
    fabric = black_fabric(H, 2 * W, seed=3)
    strip2 = np.concatenate([el, el], axis=1)
    prev = over(fabric, strip2[..., :3], strip2[..., 3])
    save(prev, os.path.join(OUT_DIR, 'greek_key_2x.png'))

    # debug zooms (work dir, not elements): seam region and one unit at 2x
    seam = to_pil(prev[:, W - 192:W + 192])
    seam.resize((seam.width * 2, seam.height * 2), Image.NEAREST).save(os.path.join(work_dir, 'dbg_greek_seam_2x.png'))
    unit = to_pil(prev[:, 0:256])
    unit.resize((unit.width * 3, unit.height * 3), Image.NEAREST).save(os.path.join(work_dir, 'dbg_greek_unit_3x.png'))
    fabric_v = black_fabric(W, H, seed=4)
    prev_v = over(fabric_v, el_v[..., :3], el_v[..., 3])
    save(prev_v[0:400], os.path.join(work_dir, 'dbg_greek_v_top.png'))

    for name in ('greek_key_strip.png', 'greek_key_strip_v.png', 'greek_key_2x.png'):
        im = Image.open(os.path.join(OUT_DIR, name))
        print(name, im.size, im.mode)
    print('done in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
