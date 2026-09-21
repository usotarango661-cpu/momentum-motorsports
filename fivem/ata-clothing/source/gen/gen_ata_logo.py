#!/usr/bin/env python3
"""ATA logo embroidery element (1024x1024 RGBA).

Outputs
  elements/ata_logo.png     crown + ATA letters
  elements/ata_letters.png  letters only (same canvas / same placement)

The jagged ATA letterforms are segmented from the chain-pendant photo
(ref/pendant.png) and re-rendered as red glitter embroidery with a gold
stitched cord outline and inner shadow.  The 5-point crown is drawn
procedurally (4x supersampled) and rendered with the same treatment.

Run from the work directory:  python3 gen/gen_ata_logo.py
"""
import os
import sys
import time

WORK = '.'
sys.path.insert(0, WORK)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

from ata_style import *  # noqa: F401,F403

# ------------------------------------------------------------------ params
W = H = 1024
SEED = 7
LETTER_W_FRAC = 0.85      # letters span this fraction of the canvas width
CROWN_W_FRAC = 0.45       # crown width relative to letters width
EDGE_LETTERS = 10         # gold outline width (px)
EDGE_CROWN = 8
GAP_CROWN = 14            # clearance between crown band and the T top edge
INNER_SHADOW_R = 14       # inner shadow reach (px inside the edge)
INNER_SHADOW_K = 0.42     # inner shadow strength
STITCH_PERIOD = 3.4       # satin-stitch period across the gold cord (px)
SEAM_ERODE = 6.5          # pendant px eroded from each letter cell (half a cord)
SEAM_SMOOTH = 5.0         # sigma (pendant px) of the seam-straightening label smoothing

OUT_LOGO = os.path.join(OUT_DIR, 'ata_logo.png')
OUT_LETTERS = os.path.join(OUT_DIR, 'ata_letters.png')

# --------------------------------------------------------- letter silhouette
def _keep_small_holes(m, max_area):
    """Fill holes of `m` smaller than max_area px (gaps between gems)."""
    filled = ndimage.binary_fill_holes(m)
    holes = filled & ~m
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    big = np.isin(hl, np.where(hs >= max_area)[0] + 1)
    return filled & ~big


def _counter_triangles(enclosed):
    """Keep only the compact enclosed regions (the A counters) and replace each by
    its convex hull so the counter is a clean triangle, not a gem-bumpy cloud."""
    el, en = ndimage.label(enclosed)
    out = Image.new('L', (enclosed.shape[1], enclosed.shape[0]), 0)
    d = ImageDraw.Draw(out)
    for i, sl in enumerate(ndimage.find_objects(el)):
        c = el[sl] == (i + 1)
        area = int(c.sum())
        h, w = c.shape
        if area < 400 or area > 8000 or max(h, w) > 110:
            continue                                     # gem gap or long internal seam
        pts = np.argwhere(c)[:, ::-1] + np.array([sl[1].start, sl[0].start])
        hull = ConvexHull(pts)
        d.polygon([tuple(map(float, pts[v])) for v in hull.vertices], fill=255)
    return np.asarray(out) > 127


def segment_pendant():
    """Letter bodies of the ATA pendant as one float mask (pendant pixel scale).

    The pendant's outer silhouette alone reads as a blob: the A-T-A forms are
    defined by the gold/diamond bands *between* the letters.  So: segment the
    sharp outer silhouette, segment the red gem field of each letter, split the
    silhouette between the three letters (Voronoi on the red bodies) and erode
    each cell by half a cord width so the gold outline fills the seams.  Erosion
    keeps the convex spike tips sharp; the A counters stay holes.

    Returns (crop mask float32 with 8 px padding, T stem centre x, T top-bar
    top edge y, both in crop coordinates, pad)."""
    a = from_pil(Image.open(REF_PENDANT).convert('RGB'))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    red = ((r - np.maximum(g, b)) > 0.14) & (r > 0.24)      # ruby field / red rim
    gold = (r > 0.40) & (g > 0.26) & (g > b + 0.08)         # gold metal
    bright = lum > 0.42                                     # diamonds / highlights
    # --- whole-pendant silhouette (bail + chain removed)
    key = red | gold | bright
    key[:440, :] = False                     # chain
    key[:479, 545:672] = False               # bail sits on top of the T bar
    key[475:479, 545:672] = key[479, 545:672]  # restore the T bar's top edge under it
    key = ndimage.binary_closing(key, iterations=2)
    sil = _keep_small_holes(key, 600)
    sil = ndimage.binary_opening(sil, iterations=2)
    sil = ndimage.binary_closing(sil, iterations=2)
    lab, n = ndimage.label(sil)
    sizes = ndimage.sum(sil, lab, range(1, n + 1))
    sil = lab == (np.argmax(sizes) + 1)
    # --- red gem field of each letter (strong red hue only, no gold, no diamonds)
    rf = (r > 0.18) & (g < 0.42 * r) & (b < 0.55 * r) & sil & ~bright
    rf = ndimage.binary_closing(rf, iterations=3)          # bridge gaps between gems
    rf = _keep_small_holes(rf, 500)                          # keep counters + letter gaps
    rf = ndimage.binary_opening(rf, iterations=3)            # drop gem-noise spurs
    lab, n = ndimage.label(rf)
    sizes = ndimage.sum(rf, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    big3 = order[:3] + 1                                     # A, T, A bodies
    bodies = np.isin(lab, big3)
    enclosed_all = ndimage.binary_fill_holes(bodies) & ~bodies   # gem gaps, counters, inner seams
    enclosed = _counter_triangles(enclosed_all)                  # clean A counters only
    cents = ndimage.center_of_mass(rf, lab, big3)
    letters = np.zeros(rf.shape, np.int32)                   # 1 = left A, 2 = T, 3 = right A
    for k, l in enumerate(big3[np.argsort([c[1] for c in cents])]):
        letters[lab == l] = k + 1
    # attach the smaller red parts (spike tips, rim pieces) to the nearest letter
    _, (iy, ix) = ndimage.distance_transform_edt(letters == 0, return_indices=True)
    for i in order[3:]:
        if sizes[i] < 300:
            break
        c = lab == (i + 1)
        if (c & enclosed_all).sum() > 0.5 * sizes[i]:        # gem island inside a counter
            continue
        cy, cx = ndimage.center_of_mass(c)
        k = letters[iy[int(cy), int(cx)], ix[int(cy), int(cx)]]
        if cx > 690 and cy > 850:                            # right A's leg tip under the T
            k = 3
        letters[c] = k
    # --- split the silhouette between the letters, erode half a cord width
    sm = np.zeros_like(letters)
    for k in (1, 2, 3):
        sm[blur((letters == k).astype(np.float32), 3.5) > 0.5] = k   # smooth gem bumps -> straighter seams
    _, (iy, ix) = ndimage.distance_transform_edt(sm == 0, return_indices=True)
    nearest = sm[iy, ix]
    # majority-smooth the label field so the seams run straighter (silhouette untouched)
    votes = np.stack([blur((nearest == k).astype(np.float32), SEAM_SMOOTH) for k in (1, 2, 3)])
    nearest = np.argmax(votes, axis=0) + 1
    dom = sil & ~enclosed
    m = np.zeros(sil.shape, bool)
    cell_T = None
    for k in (1, 2, 3):
        cell = dom & (nearest == k)
        cell = ndimage.distance_transform_edt(cell) > SEAM_ERODE
        if k == 2:
            cell_T = cell
        m |= cell
    m = blur(m.astype(np.float32), 1.0) > 0.5
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 8
    crop = m[y0 - pad:y1 + pad, x0 - pad:x1 + pad].astype(np.float32)
    band = cell_T[880:920, :]                               # T stem sample rows
    cols = np.where(band.mean(0) > 0.5)[0]
    t_cx = cols.mean() - (x0 - pad)
    t_top = float(np.where(cell_T[:, 560:660].any(1))[0].min()) - (y0 - pad)
    return crop, t_cx, t_top, pad


def fit_letters(crop, pad, edge):
    """Smooth + resize so letters + gold outline span LETTER_W_FRAC of W."""
    ch, cw = crop.shape
    s = (LETTER_W_FRAC * W - 2 * edge) / (cw - 2 * pad)
    sm = blur(crop, 1.2)
    im = to_pil(sm).resize((int(round(cw * s)), int(round(ch * s))), Image.LANCZOS)
    m = from_pil(im)
    m = np.clip((m - 0.5) * 3.0 + 0.5, 0, 1)
    return m, s


# ------------------------------------------------------------- crown mask
def crown_geometry(cx, y_bottom, u):
    """Return dict of crown control points in canvas pixels.
    cx: centre x, y_bottom: lowest point of the band's bottom arc, u: crown width."""
    arc = 0.045
    band = 0.16

    def ybot(x):   # bottom arc, lowest at centre
        return y_bottom - u * arc * (1 - (2 * x) ** 2)

    def ybt(x):    # band top arc
        return y_bottom - u * (band + arc * (1 - (2 * x) ** 2))

    tips = [(-0.50, 0.46, 0.026), (-0.235, 0.34, 0.020), (0.0, 0.58, 0.028),
            (0.235, 0.34, 0.020), (0.50, 0.46, 0.026)]          # (x, height, ball r)
    valleys = [(-0.305, 0.075), (-0.11, 0.10), (0.11, 0.10), (0.305, 0.075)]
    P = lambda x, hy: (cx + x * u, ybt(x) - hy * u)
    poly = [(cx - 0.46 * u, ybt(-0.46))]
    for i, (tx, th, _) in enumerate(tips):
        poly.append(P(tx, th))
        if i < 4:
            vx, vh = valleys[i]
            poly.append(P(vx, vh))
    poly.append((cx + 0.46 * u, ybt(0.46)))
    poly.append((cx + 0.46 * u, ybot(0.46)))
    xs = np.linspace(0.46, -0.46, 25)
    poly += [(cx + x * u, ybot(x)) for x in xs]
    balls = [(cx + tx * u, ybt(tx) - th * u, br * u) for tx, th, br in tips]
    dots = [(cx + x * u, y_bottom - u * (0.078 + arc * (1 - (2 * x) ** 2)), 0.014 * u)
            for x in np.linspace(-0.37, 0.37, 9)]
    line = [(cx + x * u, ybt(x)) for x in np.linspace(-0.44, 0.44, 25)]
    top = min(y for _, y in poly) - tips[2][2] * u
    return dict(poly=poly, balls=balls, dots=dots, line=line, top=top)


def crown_masks(geo):
    def body(d, S):
        d.polygon([(x * S, y * S) for x, y in geo['poly']], fill=255)

    def details(d, S):
        for x, y, r in geo['balls'] + geo['dots']:
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
        d.line([(x * S, y * S) for x, y in geo['line']], fill=255, width=int(2.6 * S))

    return (mask_from_pil_draw(H, W, body, 4), mask_from_pil_draw(H, W, details, 4))


# ------------------------------------------------------------ embroidery
def stitch_field(mask, period):
    """0..1 stripes running perpendicular to the mask edge (satin stitch look).
    Fades to flat 0.5 where the edge direction changes quickly."""
    inside = mask > 0.5
    sd = ndimage.distance_transform_edt(~inside) - ndimage.distance_transform_edt(inside)
    sdb = blur(sd, 2.0)
    gy, gx = np.gradient(sdb)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    gx, gy = gx / n, gy / n
    Y, X = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    f = -(X - W / 2) * gy + (Y - H / 2) * gx
    stripes = 0.5 + 0.5 * np.cos(2 * np.pi * f / period)
    rot = np.abs(np.gradient(gx)[0]) + np.abs(np.gradient(gx)[1]) + \
          np.abs(np.gradient(gy)[0]) + np.abs(np.gradient(gy)[1])
    wgt = np.clip(1.0 - rot * 6.0, 0, 1)
    return (0.5 + (stripes - 0.5) * wgt).astype(np.float32)


def embroider(mask, red_rgb, gold_rgb, edge, depth=4.0):
    """mask (H,W) 0..1 -> RGBA element: red glitter fill, gold cord outline, inner shadow."""
    inside = mask > 0.5
    d_in = ndimage.distance_transform_edt(inside)
    d_out = ndimage.distance_transform_edt(~inside)
    # --- fill: inner shadow (darker toward the outline) + bevel highlight
    inner = np.clip(1.0 - d_in / INNER_SHADOW_R, 0, 1) ** 1.6 * inside
    inner = blur(inner, 1.5)
    shade = bevel_shade(mask, depth=depth, light=(-0.6, -0.8), strength=0.30)
    fill = red_rgb * ((1.0 - INNER_SHADOW_K * inner) * shade)[..., None]
    # --- gold cord ring: rounded profile across the cord, stitch stripes along it
    t = np.clip((d_out - edge / 2.0) / (edge / 2.0), -1, 1)
    prof = 0.80 + 0.42 * np.cos(np.pi / 2 * t)
    stitch = stitch_field(mask, STITCH_PERIOD)
    ring_rgb = gold_rgb * (prof * (0.82 + 0.30 * stitch))[..., None]
    dil = dilate(mask, edge)
    ering = bevel_shade(dil, depth=2.0, light=(-0.6, -0.8), strength=0.35)
    ring_rgb = np.clip(ring_rgb * ering[..., None], 0, 1)
    # thin dark seam just outside the fill so the cord reads as raised
    seam = np.clip(1.0 - d_out / 1.5, 0, 1) * (~inside)
    ring_rgb = ring_rgb * (1.0 - 0.35 * seam)[..., None]
    rgb = over(ring_rgb, np.clip(fill, 0, 1), mask)    # gold everywhere outside the fill
    alpha = np.clip(dil, 0, 1)
    return rgba(rgb, alpha)


def gold_detail(mask, gold_rgb):
    sh = bevel_shade(mask, depth=1.6, light=(-0.6, -0.8), strength=0.55)
    rgb = np.clip(gold_rgb * 1.08 * sh[..., None], 0, 1)
    return rgba(rgb, mask)


def composite(dst, src):
    a = src[..., 3]
    rgb = over(dst[..., :3], src[..., :3], a)
    alpha = np.clip(dst[..., 3] + a - dst[..., 3] * a, 0, 1)
    return rgba(rgb, alpha)


def bleed(el, r=6.0):
    """Give fully transparent pixels the colour of the nearest opaque pixel (out to
    r px), black beyond: no dark halo if a compositor resamples un-premultiplied."""
    a = el[..., 3] > 0.02
    d, (iy, ix) = ndimage.distance_transform_edt(~a, return_indices=True)
    near = el[..., :3][iy, ix]
    far = np.clip((d - r) / 2.0, 0, 1)[..., None]
    rgb = near * (1 - far) + np.array(BLACK, np.float32)[None, None, :] * far
    rgb = np.where(a[..., None], el[..., :3], rgb)
    return rgba(rgb, el[..., 3])


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    crop, t_cx, t_top, pad = segment_pendant()
    lm, s = fit_letters(crop, pad, EDGE_LETTERS)
    lh, lw = lm.shape
    t_cx_c, t_top_c = t_cx * s, t_top * s

    # crown geometry (relative to the letters box for now)
    u = CROWN_W_FRAC * LETTER_W_FRAC * W
    cols = slice(int(t_cx_c - u / 2), int(t_cx_c + u / 2))
    sub = lm[:, cols] > 0.5
    top_prof = np.where(sub.any(0), sub.argmax(0), lh)
    crown_bottom_rel = float(top_prof.min()) - GAP_CROWN
    geo_rel = crown_geometry(t_cx_c, crown_bottom_rel, u)
    letters_bottom_rel = float(np.where((lm > 0.5).any(1))[0].max())
    block_h = letters_bottom_rel - geo_rel['top']
    print('letters %dx%d scale %.3f  crown %.0f wide  block height %.0f (%.1f%% of H)'
          % (lw, lh, s, u, block_h, 100 * block_h / H))

    # place the block: horizontally centre the letters box, vertically centre the block
    ox = int(round((W - lw) / 2))
    oy = int(round((H - block_h) / 2 - geo_rel['top']))
    letters = np.zeros((H, W), np.float32)
    y0, x0 = max(0, oy), max(0, ox)
    letters[y0:y0 + lh, x0:x0 + lw] = lm[:H - y0, :W - x0]
    geo = crown_geometry(t_cx_c + ox, crown_bottom_rel + oy, u)
    crown, details = crown_masks(geo)

    red = red_fill(H, W, seed=SEED)
    gold = gold_fill(H, W, seed=SEED + 40, glitter=0.7)

    el_letters = embroider(letters, red, gold, EDGE_LETTERS, depth=4.0)
    el_crown = embroider(crown, red, gold, EDGE_CROWN, depth=3.0)
    el_crown = composite(el_crown, gold_detail(details, gold))

    logo = composite(el_letters, el_crown)
    for el in (el_letters, logo):
        el[..., 3] = np.where(el[..., 3] < 0.01, 0, el[..., 3])
    el_letters = bleed(el_letters)
    logo = bleed(logo)
    save(el_letters, OUT_LETTERS)
    save(logo, OUT_LOGO)

    # debug previews on black fabric (not deliverables)
    bg = black_fabric(H, W, seed=3)
    prev = over(bg, logo[..., :3], logo[..., 3])
    save(prev, os.path.join(WORK, 'dbg_logo_preview.png'))
    zx, zy = int(t_cx_c + ox) - 260, int(t_top_c + oy) + 40
    z = prev[zy:zy + 260, zx:zx + 400]
    to_pil(z).resize((800, 520), Image.NEAREST).save(os.path.join(WORK, 'dbg_logo_zoom.png'))
    zx, zy = int(t_cx_c + ox) - 60, int(t_top_c + oy) + 230
    z = prev[zy:zy + 260, zx:zx + 400]
    to_pil(z).resize((800, 520), Image.NEAREST).save(os.path.join(WORK, 'dbg_logo_zoom2.png'))
    print('bbox alpha>0:', [int(v) for v in (np.where(logo[..., 3] > 0)[1].min(), np.where(logo[..., 3] > 0)[1].max(),
                                            np.where(logo[..., 3] > 0)[0].min(), np.where(logo[..., 3] > 0)[0].max())])
    print('done in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
