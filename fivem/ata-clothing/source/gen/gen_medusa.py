"""Medusa medallion element (1024x1024 RGBA, gold glitter embroidery, transparent bg).

Part 1: Medusa head line-art extracted from the reference sheet crop, cleaned,
        upscaled, cut to the inner disc.
Part 2: Procedural Greek-key (meander) ring: a linear strip (square-spiral hooks
        hanging off a baseline, 20 repeats) polar-warped into a ring with thin
        bounding circle lines.

Run from the work dir:  python3 gen/gen_medusa.py
Outputs: elements/medusa.png, elements/medusa_on_black.png (+ dbg_medusa_*.png)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from scipy import ndimage
from ata_style import *

t0 = time.time()
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 1024
C = (N - 1) / 2.0                     # canvas centre (pixel-centre convention)

# ------------------------------------------------------------- ring geometry
RO = 0.94 * N / 2                     # outer radius  (481.3)
T  = 0.09 * N                         # ring thickness (92.2)
RI = RO - T                           # inner radius  (389.1)
LINE_W = 5.0                          # bounding circle line width (px)
GAP    = 3.5                          # gap between line and meander band
N_REP  = 20                           # meander repeats around the ring
PERIOD = 11                           # cells per repeat (10 pattern + 1 gap)
ROWS   = 7                            # cells tall

# ------------------------------------------------------------- head geometry
CROP = (995, 740, 1254, 968)
CY, CX = 108.5, 123.8                 # ring centre inside the crop (fitted)
R_CUT = 73.5                          # just inside the reference ring's inner line
HEAD_R_OUT = RI - 1.5                 # where the head disc ends on the canvas
HEAD_SCALE = HEAD_R_OUT / R_CUT       # ~5.27x
SEED = 7


def soft_disc(n, r, centre=None, feather=1.0):
    cy = cx = C if centre is None else centre
    Y, X = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.hypot(Y - cy, X - cx)
    return np.clip((r - d) / feather + 0.5, 0, 1).astype(np.float32)


# ============================================================== PART 1: HEAD
def gold_score(rgb):
    """Soft 0..1 'goldness' of reference pixels (keeps sub-pixel edge info the binary mask loses)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    sm = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0, 1)
    return (sm(lum, 0.28, 0.55) * sm(r - b, 0.10, 0.28) * sm(g / (r + 1e-6), 0.42, 0.60) * sm(g, 0.22, 0.36)).astype(np.float32)


def build_head():
    im = Image.open(REF_SHEET).convert('RGB').crop(CROP)
    rgb = from_pil(im)
    m = 0.55 * gold_score(rgb) + 0.45 * color_mask(rgb, 'gold')   # blend: delicate but connected
    h, w = m.shape
    Y, X = np.mgrid[0:h, 0:w]
    R = np.hypot(Y - CY, X - CX)
    m = m * (R < R_CUT)                                   # drop the meander ring

    # clean specks: drop tiny connected components (and their soft halo)
    lab, n = ndimage.label(m > 0.5)
    sizes = ndimage.sum(m > 0.5, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sizes >= 4
    m = m * dilate(keep[lab].astype(np.float32), 1.0)

    # crop to disc bbox and upscale smoothly
    pad = 3
    y0 = int(max(0, np.floor(CY - R_CUT) - pad)); y1 = int(min(h, np.ceil(CY + R_CUT) + pad))
    x0 = int(max(0, np.floor(CX - R_CUT) - pad)); x1 = int(min(w, np.ceil(CX + R_CUT) + pad))
    sub = m[y0:y1, x0:x1]
    sub = blur(sub, 0.35)
    big = to_pil(sub).resize((int(round(sub.shape[1] * HEAD_SCALE)), int(round(sub.shape[0] * HEAD_SCALE))), Image.LANCZOS)
    big = from_pil(big)
    big = blur(big, 2.8)                                  # remove staircase from the 5x upscale
    big = np.clip((big - 0.46) * 9.0 + 0.5, 0, 1)         # re-sharpen edge to ~1px (thr < 0.5 keeps strokes joined)

    # place on canvas so the ring centre lands on the canvas centre
    out = np.zeros((N, N), np.float32)
    ocy = (CY - y0) * HEAD_SCALE; ocx = (CX - x0) * HEAD_SCALE
    oy = int(round(C - ocy)); ox = int(round(C - ocx))
    bh, bw = big.shape
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    dy1, dx1 = min(N, oy + bh), min(N, ox + bw)
    out[dy0:dy1, dx0:dx1] = big[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    out *= soft_disc(N, HEAD_R_OUT, feather=1.2)          # crisp circular cut at the inner disc
    return out


# ============================================================== PART 2: RING
def meander_unit():
    """One repeat of a Greek-key hook (2-turn square spiral) on a baseline. 7 rows x 11 cols."""
    rows = [
        "##########.",
        "#........#.",
        "#.######.#.",
        "#.#....#.#.",
        "#.########.",
        "#..........",
        "###########",
    ]
    return np.array([[c == '#' for c in r] for r in rows], np.float32)


def build_strip(cell_px):
    """Linear meander strip (mask) with bounding lines, in 'band pixel' units * cell_px scale."""
    unit = meander_unit()
    strip = np.tile(unit, (1, N_REP))                     # rows x (N_REP*PERIOD)
    strip = np.kron(strip, np.ones((cell_px, cell_px), np.float32))
    return strip


def build_ring():
    S = 3                                                  # supersample factor for the polar warp
    n = N * S
    band_h = T                                             # px
    mean_h = band_h - 2 * (LINE_W + GAP)                   # meander band height in px
    cell_h = mean_h / ROWS                                 # px per cell (radial)
    # strip resolution: 8 sub-px per canvas px radially
    cell_px = 8
    strip = build_strip(cell_px)                           # ROWS*8 x N_REP*PERIOD*8
    sh, sw = strip.shape

    Y, X = np.mgrid[0:n, 0:n].astype(np.float32)
    cc = (n - 1) / 2.0
    yy = (Y - cc) / S; xx = (X - cc) / S                   # canvas px coords
    r = np.hypot(yy, xx)
    a = (np.arctan2(yy, xx) / (2 * np.pi)) % 1.0           # 0..1 around

    # meander band: r in [RI+LINE_W+GAP, RO-LINE_W-GAP], baseline at the INNER side
    r_in_band = RI + LINE_W + GAP
    v = (r - r_in_band) / cell_h                           # cell rows, 0 at inner edge
    v = (ROWS - v)                                         # flip so baseline (row 6) sits at inner edge
    u = a * N_REP * PERIOD                                 # cell columns
    inb = (v >= 0) & (v < ROWS)
    # sample the strip (nearest at 8x sub-res, then the S=3 supersample gives AA)
    vi = np.clip((v * cell_px).astype(np.int32), 0, sh - 1)
    ui = ((u * cell_px).astype(np.int64)) % sw
    band = np.where(inb, strip[vi, ui], 0.0).astype(np.float32)

    # bounding lines
    line_out = ((r >= RO - LINE_W) & (r < RO)).astype(np.float32)
    line_in = ((r >= RI) & (r < RI + LINE_W)).astype(np.float32)
    mask = np.clip(band + line_out + line_in, 0, 1)
    # downsample S x S -> antialiased 1024 mask
    mask = mask.reshape(N, S, N, S).mean(axis=(1, 3)).astype(np.float32)
    return mask, strip


# ============================================================== RENDER
def main():
    os.makedirs(os.path.join(WORK, 'elements'), exist_ok=True)
    head = build_head()
    print('head built', round(time.time() - t0, 1), 's, coverage', round(float(head.mean()), 4))
    ring, strip = build_ring()
    print('ring built', round(time.time() - t0, 1), 's')

    gold = gold_fill(N, N, seed=SEED, glitter=0.7)
    gold2 = gold_fill(N, N, seed=SEED + 100, glitter=0.7)
    head_el = emboss_element(head, gold, depth=2.0, strength=0.40)
    ring_el = emboss_element(ring, gold2, depth=2.2, strength=0.40)

    # combine: ring over head
    rgb = head_el[..., :3]; alpha = head_el[..., 3]
    rgb = over(rgb, ring_el[..., :3], ring_el[..., 3])
    alpha = np.clip(alpha + ring_el[..., 3] - alpha * ring_el[..., 3], 0, 1)
    # keep colour sane in fully transparent areas (avoid black fringes when resampled)
    rgb = np.where(alpha[..., None] > 0.002, rgb, np.array(GOLD)[None, None, :])
    out = rgba(rgb, alpha)
    save(out, os.path.join(WORK, 'elements', 'medusa.png'))

    fabric = black_fabric(N, N, seed=3)
    comp = over(fabric, rgb, alpha)
    save(comp, os.path.join(WORK, 'elements', 'medusa_on_black.png'))

    # debug images
    save(head, os.path.join(WORK, 'dbg_medusa_head_mask.png'))
    save(ring, os.path.join(WORK, 'dbg_medusa_ring_mask.png'))
    to_pil(strip[:, :strip.shape[1] // 4]).save(os.path.join(WORK, 'dbg_medusa_strip.png'))
    im = to_pil(comp)
    im.crop((256, 256, 768, 768)).resize((1024, 1024), Image.NEAREST).save(os.path.join(WORK, 'dbg_medusa_zoom_face.png'))
    im.crop((512 - 128, 0, 512 + 128, 256)).resize((1024, 1024), Image.NEAREST).save(os.path.join(WORK, 'dbg_medusa_zoom_ring.png'))
    print('done', round(time.time() - t0, 1), 's')
    o = Image.open(os.path.join(WORK, 'elements', 'medusa.png'))
    print('medusa.png', o.size, o.mode)


if __name__ == '__main__':
    main()
