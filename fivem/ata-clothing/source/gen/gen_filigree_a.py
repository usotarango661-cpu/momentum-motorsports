"""filigree_a: baroque gold acanthus scroll ornament, DESIGN APPROACH A (scroll-first).

Fully procedural.  One half (512x1024, mirror axis = right edge) is a tall S-SCROLL column:
  * V1: big volute at the top curling OUTWARD (eye on the outer side), its outer sweep passing over
        the top and down the inner side; the band continues as a stem bulging outward in the middle
        band and ends in V2, a smaller volute at the bottom-inner that curls inward.  One smooth
        log-spiral / Catmull-Rom spine, bold tapered stroke (slow taper: coil keeps >= 70 % width
        for the first 1.2 turns).
  * two tendrils: T0 off V1's inner side curling under the crest, T off the stem crossing the
        middle band and curling up beside the axis.  Middle band holds 2 volutes per side (T, V2 top).
  * big pointed acanthus lobes along the convex sides: one broad leaf with three round bites cut
        into its outer edge, three short triangular teeth between them, two shallow scallops on the
        inner edge, a hooked pointed tip, central vein + side ribs, and a small trailing leaflet;
        every third rhythm slot is a bare-band gap.  Small lobes (tendrils, V2 bottom) get bites only.
  * one anthemion crest on the top axis only: short vertical stem, 3-lobe fan (serrated teardrop +
        two side lobes), two small volutes branching off the stem (no crossing).  No bottom crest,
        so the pair reads as two facing S-scrolls, not a closed wreath.
  * bead dots at every curl centre; the middle of the axis is left clear (logo / splatter hole)
Rasterised at 4x into THREE masks (core, wide = core + RING px, groove = carved rims + veins),
LANCZOS-downsampled, mirrored left-right and shaded with the shared gold fill.

Run from the work dir:  python3 gen/gen_filigree_a.py      (--mask = fast mask-only preview)
Outputs (elements/): filigree_a.png (1024x1024 RGBA), filigree_a_single.png (512x1024 RGBA),
                     filigree_a_on_black.png (review), plus dbg_filigree_a_*.png in work/.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from scipy import ndimage
from ata_style import *

t0 = time.time()
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)

H, W = 1024, 512          # half canvas (mirror axis = right edge, x = 512)
S = 4                     # supersampling
SEED = 23
RING = 1.2                # px: darker-gold edge ring, drawn at 4x (survives the downsample)
RIM = 1.6                 # px: dark rim carved around a lobe/tendril where it overlaps an earlier shape
GROOVE_W = 1.8            # px: vein groove width (carved from the core, painted dark gold)
FIT = 0.88                # uniform shrink of the layout towards the axis centre (curls stay round)
TONE = (0.95, 0.88, 0.92) # per-element tint of the shared gold fill (warmer / deeper, closer to the reference)
BEVEL = 0.6               # bevel strength (deeper thread shadow)
rng = np.random.default_rng(SEED)

# ============================================================== curve helpers
def catmull_rom(ctrl, alpha=0.5, px_per_sample=0.75):
    """Centripetal Catmull-Rom through ctrl rows (x, y, w).  Returns dense (m,3)."""
    P = np.asarray(ctrl, np.float64)
    keep = [0] + [i for i in range(1, len(P)) if np.hypot(*(P[i, :2] - P[i - 1, :2])) > 1e-6]
    P = P[keep]
    if len(P) < 2:
        return P
    P = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0_ = 0.0
        t1_ = t0_ + max(np.hypot(*(p1[:2] - p0[:2])), 1e-3) ** alpha
        t2_ = t1_ + max(np.hypot(*(p2[:2] - p1[:2])), 1e-3) ** alpha
        t3_ = t2_ + max(np.hypot(*(p3[:2] - p2[:2])), 1e-3) ** alpha
        m = max(2, int(np.hypot(*(p2[:2] - p1[:2])) / px_per_sample))
        t = np.linspace(t1_, t2_, m, endpoint=False)[:, None]
        A1 = (t1_ - t) / (t1_ - t0_) * p0 + (t - t0_) / (t1_ - t0_) * p1
        A2 = (t2_ - t) / (t2_ - t1_) * p1 + (t - t1_) / (t2_ - t1_) * p2
        A3 = (t3_ - t) / (t3_ - t2_) * p2 + (t - t2_) / (t3_ - t2_) * p3
        B1 = (t2_ - t) / (t2_ - t0_) * A1 + (t - t0_) / (t2_ - t0_) * A2
        B2 = (t3_ - t) / (t3_ - t1_) * A2 + (t - t1_) / (t3_ - t1_) * A3
        C = (t2_ - t) / (t2_ - t1_) * B1 + (t - t1_) / (t2_ - t1_) * B2
        out.append(C)
    out.append(P[-2][None])
    return np.vstack(out)


def spiral_ctrl(center, r_out, r_in, th_out_deg, turns, ccw, w_out, w_in, step_deg=8, inward=True, r_pow=1.0):
    """Control rows (x,y,w) of a log spiral from the OUTER point going inward (or reversed).
    ccw=True: theta decreases along the outer->inner path (counter-clockwise on screen, y down).
    r_pow > 1 keeps the first turn wide (rounder volute) before tightening into the eye.
    Width: slow quadratic taper w_out -> w_in (first 70 % of the path stays >= 70 % width)."""
    n = max(4, int(round(360 * turns / step_deg)))
    s = np.linspace(0, 1, n + 1)
    th = np.deg2rad(th_out_deg) + (-1 if ccw else 1) * s * 2 * np.pi * turns
    r = r_out * (r_in / r_out) ** (s ** r_pow)
    w = w_out * (1 - (1 - w_in / w_out) * s ** 2)
    pts = np.stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th), w], 1)
    return pts if inward else pts[::-1]


def arclen(P):
    P = np.asarray(P, np.float64)
    seg = np.hypot(*(P[1:, :2] - P[:-1, :2]).T)
    return np.concatenate([[0], np.cumsum(seg)])


# ============================================================== rasteriser
def fit(xy):
    """Layout coords -> canvas coords: uniform scale about (W, (H-1)/2) so the mirror axis stays put."""
    xy = np.asarray(xy, np.float64)
    out = xy.copy()
    out[..., 0] = W - (W - xy[..., 0]) * FIT
    out[..., 1] = (H - 1) / 2 + (xy[..., 1] - (H - 1) / 2) * FIT
    return out


class Canvas:
    """Three 4x masks: core (the shape), wide (core dilated by RING px) and groove (carved rims + veins,
    painted dark gold).  Everything is drawn as chains of discs."""
    def __init__(self):
        self.im = Image.new('L', (W * S, H * S), 0)
        self.imw = Image.new('L', (W * S, H * S), 0)
        self.img = Image.new('L', (W * S, H * S), 0)
        self.d = ImageDraw.Draw(self.im)
        self.dw = ImageDraw.Draw(self.imw)
        self.dg = ImageDraw.Draw(self.img)
        self.n_draw = 0

    def _chain(self, curve):
        P = np.asarray(curve, np.float64)
        if len(P) < 2:
            return None
        xy = fit(P[:, :2]) * S
        hw = np.maximum(P[:, 2] * 0.5 * FIT * S, 0.9)
        seg = np.hypot(*(xy[1:] - xy[:-1]).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        L = cum[-1]
        if L <= 0:
            return None
        pos = [0.0]
        while pos[-1] < L:                      # adaptive resampling: step ~ 0.45 * radius
            r_here = np.interp(pos[-1], cum, hw)
            pos.append(pos[-1] + float(np.clip(0.45 * r_here, 1.2, 10.0)))
        pos = np.array(pos[:-1] + [L])
        return np.interp(pos, cum, xy[:, 0]), np.interp(pos, cum, xy[:, 1]), np.interp(pos, cum, hw), pos / S

    def _discs(self, xs, ys, rs):
        rw = RING * S
        d, dw, dg = self.d, self.dw, self.dg
        for x, y, r in zip(xs, ys, rs):
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
            dw.ellipse([x - r - rw, y - r - rw, x + r + rw, y + r + rw], fill=255)
            dg.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.n_draw += len(xs)

    def stroke(self, curve, rim=False, rim_from=0.0):
        """curve: dense (m,3) rows (x,y,w) in layout px, w = full width.  Draws a chain of discs.
        rim=True first carves a RIM-px dark outline (from rim_from px along the curve) so the shape
        reads as laid over whatever it overlaps."""
        ch = self._chain(curve)
        if ch is None:
            return
        xs, ys, rs, arc = ch
        if rim:
            self._carve_rim(xs, ys, rs, arc, rim_from)
        self._discs(xs, ys, rs)

    def _carve_rim(self, xs, ys, rs, arc, rim_from):
        """Dark RIM-px outline around the chain, only where the core already holds an earlier shape."""
        rr = RIM * S * FIT
        sel = arc >= rim_from
        if not sel.any():
            return
        xs, ys, rs = xs[sel], ys[sel], rs[sel]
        x0 = max(0, int(np.floor((xs - rs - rr).min())) - 2); y0 = max(0, int(np.floor((ys - rs - rr).min())) - 2)
        x1 = min(W * S, int(np.ceil((xs + rs + rr).max())) + 2); y1 = min(H * S, int(np.ceil((ys + rs + rr).max())) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        tmp = Image.new('L', (x1 - x0, y1 - y0), 0)
        td = ImageDraw.Draw(tmp)
        for x, y, r in zip(xs - x0, ys - y0, rs):
            td.ellipse([x - r - rr, y - r - rr, x + r + rr, y + r + rr], fill=255)
        box = (x0, y0, x1, y1)
        core_crop = self.im.crop(box)
        carve = ImageChops.multiply(tmp, core_crop)                 # annulus AND existing shape
        self.im.paste(ImageChops.subtract(core_crop, carve), box)
        self.img.paste(ImageChops.lighter(self.img.crop(box), carve), box)

    def stroke_leaf(self, curve, rim_from, notches=(), depth=0.32, bend_sign=1.0, notches_in=(), depth_in=0.24):
        """A lobe: rim-carved stroke + round serration NOTCHES cut into its convex (outer) edge at the
        given arc fractions, `depth` x local width deep.  The notch only removes the lobe's own pixels
        (earlier shapes under it are restored), and the wide mask keeps a RING around the notch."""
        ch = self._chain(curve)
        if ch is None:
            return
        xs, ys, rs, arc = ch
        self._carve_rim(xs, ys, rs, arc, rim_from)
        if not notches and not notches_in:
            self._discs(xs, ys, rs)
            return
        P = np.asarray(curve, np.float64)
        cum = arclen(P); L = cum[-1]
        T = tangents(P)
        discs = []
        for (fr, dp, sd) in [(f, depth, -1.0) for f in notches] + [(f, depth_in, 1.0) for f in notches_in]:
            a = fr * L
            i = int(np.clip(np.searchsorted(cum, a), 1, len(P) - 1))
            u = (a - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
            p = P[i - 1, :2] * (1 - u) + P[i, :2] * u
            w = P[i - 1, 2] * (1 - u) + P[i, 2] * u
            t = T[i - 1] * (1 - u) + T[i] * u; t = t / (np.hypot(*t) + 1e-9)
            n = sd * bend_sign * np.array([-t[1], t[0]])          # sd=-1: convex (outer) edge, +1: concave edge
            D = dp * w; R = 0.95 * D                                # bite slightly deeper than a semicircle -> cusps
            c = fit(p + n * (0.5 * w + R - D)) * S
            discs.append((c[0], c[1], R * FIT * S))
        pad = int(max(r for _, _, r in discs) + RING * S + 4)
        cxs = np.array([c[0] for c in discs]); cys = np.array([c[1] for c in discs])
        x0 = max(0, int(np.floor(min((xs - rs).min(), cxs.min()))) - pad)
        y0 = max(0, int(np.floor(min((ys - rs).min(), cys.min()))) - pad)
        x1 = min(W * S, int(np.ceil(max((xs + rs).max(), cxs.max()))) + pad)
        y1 = min(H * S, int(np.ceil(max((ys + rs).max(), cys.max()))) + pad)
        box = (x0, y0, x1, y1)
        core_before = self.im.crop(box); wide_before = self.imw.crop(box)
        self._discs(xs, ys, rs)
        tc = Image.new('L', (x1 - x0, y1 - y0), 0); tw = Image.new('L', (x1 - x0, y1 - y0), 0)
        dc = ImageDraw.Draw(tc); dwd = ImageDraw.Draw(tw)
        for (cx, cy, r) in discs:
            dc.ellipse([cx - x0 - r, cy - y0 - r, cx - x0 + r, cy - y0 + r], fill=255)
            r2 = max(r - RING * S, 1.0)
            dwd.ellipse([cx - x0 - r2, cy - y0 - r2, cx - x0 + r2, cy - y0 + r2], fill=255)
        self.im.paste(ImageChops.lighter(core_before, ImageChops.subtract(self.im.crop(box), tc)), box)
        self.imw.paste(ImageChops.lighter(wide_before, ImageChops.subtract(self.imw.crop(box), tw)), box)

    def cut(self, curve):
        """Groove: carve from the core and mark it in the groove mask (painted dark gold)."""
        ch = self._chain(curve)
        if ch is None:
            return
        xs, ys, rs, _ = ch
        d, dg = self.d, self.dg
        for x, y, r in zip(xs, ys, rs):
            d.ellipse([x - r, y - r, x + r, y + r], fill=0)
            dg.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.n_draw += len(xs)

    def dot(self, x, y, r):
        x, y = fit(np.array([x, y])) * S
        r = r * FIT * S
        rw = RING * S
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.dw.ellipse([x - r - rw, y - r - rw, x + r + rw, y + r + rw], fill=255)
        self.dg.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def masks(self):
        core = from_pil(self.im.resize((W, H), Image.LANCZOS))
        wide = from_pil(self.imw.resize((W, H), Image.LANCZOS))
        groove = from_pil(self.img.resize((W, H), Image.LANCZOS))
        return core, wide, groove


# ============================================================== ornament parts
def tangents(P):
    d = np.gradient(P[:, :2], axis=0)
    n = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    return d / n[:, None]


def leaf(base, direction, length, w_base, bend_deg=85.0, w_tip=2.5, n=36, belly=0.35, pw=0.55, bend_pow=2.0):
    """Acanthus finger: starts at base, heads along `direction` (unit vector or angle in deg), bends by
    bend_deg over its length (as s^bend_pow -> the tip hooks over).  Broad base, slight belly, tapering
    to a pointed tip (w_tip).  Returns dense (n,3) rows (x, y, w)."""
    if np.isscalar(direction):
        ang = np.deg2rad(direction)
    else:
        ang = math.atan2(direction[1], direction[0])
    s = np.linspace(0, 1, n)
    a = ang + np.deg2rad(bend_deg) * s ** bend_pow
    step = length / (n - 1)
    x = base[0] + np.concatenate([[0], np.cumsum(np.cos(a[1:]) * step)])
    y = base[1] + np.concatenate([[0], np.cumsum(np.sin(a[1:]) * step)])
    w = np.maximum(w_base * (1 - s) ** pw * (1 + belly * np.sin(np.pi * s) ** 1.5), w_tip)
    w[-1] = w_tip * 0.7
    return np.stack([x, y, w], 1)


def vein(p, start, end_frac=0.8):
    """Groove polyline along a finger's axis, from `start` px after the base to end_frac of its length."""
    P = np.asarray(p, np.float64)
    cum = arclen(P); L = cum[-1]
    sel = (cum >= start) & (cum <= end_frac * L)
    if sel.sum() < 2:
        return None
    g = P[sel].copy()
    g[:, 2] = GROOVE_W
    return g


def point_on(P, frac):
    """(point, local width, unit tangent) at arc fraction `frac` of a dense (m,3) curve."""
    P = np.asarray(P, np.float64)
    cum = arclen(P); L = cum[-1]
    a = frac * L
    i = int(np.clip(np.searchsorted(cum, a), 1, len(P) - 1))
    u = (a - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    p = P[i - 1, :2] * (1 - u) + P[i, :2] * u
    w = P[i - 1, 2] * (1 - u) + P[i, 2] * u
    T = tangents(P)
    t = T[i - 1] * (1 - u) + T[i] * u; t = t / (np.hypot(*t) + 1e-9)
    return p, w, t


def side_vein(P, frac, sign_n, bend_sign, k=0.42):
    """Short rib groove from the leaf axis at arc fraction `frac` towards one edge (sign_n = -1 outer)."""
    P = np.asarray(P, np.float64)
    cum = arclen(P); L = cum[-1]
    a = frac * L
    i = int(np.clip(np.searchsorted(cum, a), 1, len(P) - 1))
    u = (a - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    p = P[i - 1, :2] * (1 - u) + P[i, :2] * u
    w = P[i - 1, 2] * (1 - u) + P[i, 2] * u
    T = tangents(P)
    t = T[i - 1] * (1 - u) + T[i] * u; t = t / (np.hypot(*t) + 1e-9)
    n = sign_n * bend_sign * np.array([-t[1], t[0]])
    d = 0.75 * n + 0.45 * t; d = d / np.hypot(*d)
    q = p + d * k * w
    return np.array([[p[0], p[1], GROOVE_W * 0.85], [q[0], q[1], GROOVE_W * 0.85]])


def add_leaves(cv, curve, s_range=(0.0, 1.0), side='outer', min_w=8.0, len_k=2.8, len_c=14.0,
               spacing_k=1.0, base_k=1.4, lean_deg=20.0, bend_deg=85.0, seed=0, len_max=132.0,
               rhythm=(1.0, 0.0, 0.7), extra_gap=10.0, gap_len=26.0, leaflet=True, leaflet_min=40.0,
               groove_min=30.0, notch_min=34.0, notch_depth=0.28, start_off=0.0, teeth_min=64.0,
               teeth=((0.30, 0.40, 0.84), (0.50, 0.34, 0.80), (0.68, 0.27, 0.76))):
    """Attach acanthus lobes along a dense curve (x,y,w).  side: 'outer' (convex side), 'inner', +1/-1.
    Lobe protrusion = len_k * band width + len_c (capped len_max), times the rhythm factor; a rhythm
    entry of 0 leaves gap_len px of bare band.  Each lobe = one broad leaf with three round bites cut
    into its outer edge (their cusps form pointed teeth), two shallow scallops on the inner edge, a
    hooked pointed tip, a central vein, and (for lobes longer than teeth_min) three short triangular
    teeth added along the outer edge between the bites plus side ribs, and (for big lobes) a small
    trailing leaflet behind it; every piece is rim-carved where it overlaps an earlier shape."""
    r = np.random.default_rng(seed)
    P = np.asarray(curve, np.float64)
    T = tangents(P)
    cross = T[:-1, 0] * T[1:, 1] - T[:-1, 1] * T[1:, 0]
    cross = np.concatenate([cross, [cross[-1]]])
    cross = ndimage.gaussian_filter1d(cross, 25, mode='nearest')
    cum = arclen(P); L = cum[-1]
    pos = s_range[0] * L + start_off
    k = 0
    n_lobes = 0
    while pos < s_range[1] * L:
        rk = rhythm[k % len(rhythm)]
        k += 1
        if rk <= 0:
            pos += gap_len
            continue
        i = min(max(int(np.searchsorted(cum, pos)), 0), len(P) - 1)
        w = P[i, 2]
        if w < min_w:
            pos += 6.0
            continue
        t = T[i]
        n = np.array([-t[1], t[0]])
        sgn = -np.sign(cross[i]) if abs(cross[i]) > 1e-9 else 1.0
        if side == 'outer':
            nn = sgn * n
        elif side == 'inner':
            nn = -sgn * n
        else:
            nn = float(side) * n
        lean = np.deg2rad(lean_deg + r.uniform(-6, 6))
        direction = math.cos(lean) * nn + math.sin(lean) * t
        protr = min(len_k * w + len_c, len_max) * rk * r.uniform(0.92, 1.08)
        w_base = base_k * w * (0.8 + 0.2 * rk)
        bend_sign = np.sign(nn[0] * t[1] - nn[1] * t[0])      # + rotates nn towards t (forward)
        bd = bend_sign * bend_deg * r.uniform(0.85, 1.15)
        ang0 = math.degrees(math.atan2(direction[1], direction[0]))
        big = protr > notch_min
        if leaflet and protr > leaflet_min:
            lf = leaf(P[i, :2] - t * 0.22 * w, ang0 - bend_sign * 42, 0.45 * protr + 0.5 * w, 0.5 * w_base, bend_deg=bd * 0.6)
            cv.stroke_leaf(lf, rim_from=0.5 * w + 3.0, notches=(0.62,) if big else (), depth=0.3, bend_sign=np.sign(bd))
            g = vein(lf, 0.5 * w + 5.0, 0.78)
            if g is not None:
                cv.cut(g)
        main = leaf(P[i, :2], ang0, protr + 0.5 * w, w_base, bend_deg=bd)
        cv.stroke_leaf(main, rim_from=0.5 * w + 3.0, notches=(0.40, 0.59, 0.76) if big else (), depth=notch_depth,
                       bend_sign=np.sign(bd), notches_in=(0.5, 0.72) if big else (), depth_in=0.22)
        toothed = big and protr > teeth_min
        if toothed:
            for (fr, lk, wk) in teeth:
                p_, w_, t_ = point_on(main, fr)
                n_out = -np.sign(bd) * np.array([-t_[1], t_[0]])
                d_ = 0.8 * n_out + 0.6 * t_; d_ = d_ / np.hypot(*d_)
                tooth = leaf(p_ + n_out * 0.18 * w_, math.degrees(math.atan2(d_[1], d_[0])), lk * w_base + 0.30 * w_,
                             wk * w_, bend_deg=bd * 0.45, w_tip=2.0, pw=0.85, belly=0.0)
                cv.stroke(tooth)
        if protr > groove_min:
            g = vein(main, 0.5 * w + 6.0, 0.84)
            if g is not None:
                cv.cut(g)
            if toothed:
                for fr in (0.30, 0.50, 0.68):
                    cv.cut(side_vein(main, fr, -1.0, np.sign(bd), k=0.55))
        n_lobes += 1
        pos += spacing_k * protr + extra_gap + 0.5 * w_base
    return n_lobes


# ============================================================== build the half (layout coords)
cv = Canvas()
BEADS = []   # (x, y, r)

# ---- M: the S-scroll spine.  V1 (top, outward volute) -> stem (bulging outward) -> V2 (bottom, inward volute)
V1_c = (262, 232)
V1_kw = dict(r_out=128, r_in=20, th_out_deg=100, turns=1.7, ccw=True, w_out=44, w_in=18, r_pow=1.5)
V1_sp = spiral_ctrl(V1_c, inward=False, **V1_kw)                   # inner -> outer
STEM = [(205, 385, 44), (188, 460, 41), (186, 560, 38), (200, 650, 38), (232, 730, 40), (262, 792, 42)]
V2_c = (366, 850)
V2_kw = dict(r_out=94, r_in=18, th_out_deg=180, turns=1.75, ccw=True, w_out=40, w_in=16, r_pow=1.4)
V2_sp = spiral_ctrl(V2_c, inward=True, **V2_kw)                    # outer -> inner
M = catmull_rom(np.vstack([V1_sp, np.array(STEM, np.float64), V2_sp]))
V1_in = catmull_rom(spiral_ctrl(V1_c, inward=True, **V1_kw))       # same geometry, outer -> inner (for lobes)
V2_in = catmull_rom(V2_sp)
STEM_cv = catmull_rom(np.vstack([V1_sp[-2:], np.array(STEM, np.float64), V2_sp[:2]]))
BEADS += [(V1_c[0], V1_c[1], 5.0), (V2_c[0], V2_c[1], 4.4)]

# ---- T0: tendril off V1's inner side, running to the axis and curling up under the crest
T0_c = (456, 318)
T0_ctrl = np.vstack([np.array([(366, 230, 20), (398, 256, 20), (426, 296, 19), (446, 344, 18)], np.float64),
                     spiral_ctrl(T0_c, r_out=42, r_in=11.5, th_out_deg=90, turns=1.55, ccw=True, w_out=18, w_in=9)])
T0 = catmull_rom(T0_ctrl)
BEADS += [(T0_c[0], T0_c[1], 3.0)]

# ---- T: tendril off the stem crossing the middle band, curling up beside the axis
T_c = (416, 500)
T_ctrl = np.vstack([np.array([(196, 562, 22), (250, 546, 22), (310, 532, 21), (362, 526, 20), (390, 536, 19)], np.float64),
                    spiral_ctrl(T_c, r_out=46, r_in=11.5, th_out_deg=90, turns=1.6, ccw=True, w_out=19, w_in=9)])
T = catmull_rom(T_ctrl)
BEADS += [(T_c[0], T_c[1], 3.2)]

# ---- draw spines
cv.stroke(M)
cv.stroke(T0, rim=True, rim_from=26.0)
cv.stroke(T, rim=True, rim_from=26.0)

# ---- acanthus lobes: big serrated clusters on the convex sides, small simple ones inside V1's curl
n1 = add_leaves(cv, V1_in, s_range=(0.05, 0.56), side='outer', seed=1, rhythm=(1.0, 0.0, 0.78), gap_len=20, extra_gap=0)
n2 = add_leaves(cv, STEM_cv, s_range=(0.08, 0.90), side=+1, seed=2, len_max=96, start_off=30)
n2 += add_leaves(cv, STEM_cv, s_range=(0.70, 0.80), side=-1, seed=12, len_max=82, rhythm=(1.0,), lean_deg=26)
n3 = add_leaves(cv, V2_in, s_range=(0.02, 0.16), side='outer', seed=3, len_max=100, rhythm=(1.0, 0.75), extra_gap=4)
n3 += add_leaves(cv, V2_in, s_range=(0.17, 0.30), side='outer', seed=13, len_max=54, rhythm=(0.8,), lean_deg=30,
                 leaflet=False)
n4 = add_leaves(cv, T0, s_range=(0.10, 0.46), side='outer', seed=4, len_c=10, len_max=56, extra_gap=24,
                leaflet_min=48, rhythm=(1.0, 0.7), gap_len=18)
n5 = add_leaves(cv, T, s_range=(0.06, 0.55), side='outer', seed=5, len_c=10, len_max=62, extra_gap=26,
                leaflet_min=48, rhythm=(1.0, 0.7), gap_len=18)
n6 = add_leaves(cv, V1_in, s_range=(0.20, 0.42), side='inner', seed=6, len_k=0.8, len_c=8, spacing_k=1.3,
                lean_deg=30, len_max=30, base_k=0.9, extra_gap=34, rhythm=(1.0, 0.8), leaflet=False,
                groove_min=26, notch_min=1e9, bend_deg=60)
print('lobes: V1 %d  stem %d  V2 %d  T0 %d  T %d  V1-inner %d' % (n1, n2, n3, n4, n5, n6))

# ---- anthemion crest on the TOP axis only: short stem, 3-lobe fan, two small volutes off the stem
def crest(cv):
    st = np.array([(512, 108, 22), (512, 152, 24), (512, 200, 20)], np.float64)
    cv.stroke(catmull_rom(st))
    tear = leaf(np.array([512.0, 112.0]), -90, 120, 66, bend_deg=0, w_tip=2.5, belly=0.28, pw=0.5)
    cv.stroke_leaf(tear, rim_from=8.0, notches=(0.38, 0.58, 0.76), depth=0.34, bend_sign=1.0)
    g = vein(tear, 14.0, 0.84)
    if g is not None:
        cv.cut(g)
    for fr in (0.5, 0.69):
        cv.cut(side_vein(tear, fr, -1.0, 1.0))
    side = leaf(np.array([512.0, 128.0]), -90 - 42, 86, 40, bend_deg=-45, w_tip=2.2, belly=0.3, pw=0.55)
    cv.stroke_leaf(side, rim_from=16.0, notches=(0.45, 0.68), depth=0.3, bend_sign=-1.0)
    g = vein(side, 18.0, 0.8)
    if g is not None:
        cv.cut(g)
    vc = (464, 232)
    ctrl = np.array([(512, 182, 14), (501, 197, 13), (491, 216, 12)], np.float64)
    sp = spiral_ctrl(vc, r_out=26, r_in=8, th_out_deg=0, turns=1.5, ccw=False, w_out=13, w_in=4.5)
    cv.stroke(catmull_rom(np.vstack([ctrl, sp])), rim=True, rim_from=16.0)
    BEADS.append((vc[0], vc[1], 2.6))
crest(cv)

# ---- beads at curl centres
for (bx, by, br) in BEADS:
    cv.dot(bx, by, br)

print('draw calls %d  (%.1fs)' % (cv.n_draw, time.time() - t0))
core_h, wide_h, groove_h = cv.masks()                              # (1024, 512) 0..1 each
core = np.concatenate([core_h, core_h[:, ::-1]], axis=1)           # (1024, 1024)
wide = np.concatenate([wide_h, wide_h[:, ::-1]], axis=1)
groove = np.concatenate([groove_h, groove_h[:, ::-1]], axis=1)
if '--mask' in sys.argv:
    pil = to_pil(core)
    pil.save(os.path.join(WORK, 'dbg_filigree_a_mask.png'))
    sheet = Image.new('L', (1024 + 10 + 256 + 10 + 128, 1024), 60)
    sheet.paste(pil, (0, 0))
    sheet.paste(pil.resize((256, 256), Image.LANCZOS), (1034, 0))
    sheet.paste(pil.resize((128, 128), Image.LANCZOS), (1300, 0))
    sheet.paste(pil.crop((130, 60, 330, 260)).resize((400, 400), Image.NEAREST), (1034, 300))
    sheet.save(os.path.join(WORK, 'dbg_filigree_a_mask_sheet.png'))
    ys, xs = np.nonzero(wide > 0.5)
    print('bbox x %d..%d (%.0f%% w)  y %d..%d (%.0f%% h)' % (xs.min(), xs.max(), 100 * (xs.max() - xs.min()) / 1024,
                                                           ys.min(), ys.max(), 100 * (ys.max() - ys.min()) / 1024))
    mb = core[420:700, 200:460] > 0.5
    print('middle band opaque %.1f%%' % (100 * mb.mean()))
    sys.exit(0)

# ============================================================== material / shading
N = 1024
gold = np.clip(gold_fill(N, N, seed=SEED, glitter=0.7) * np.array(TONE, np.float32)[None, None, :], 0, 1)
hard = core > 0.5
dist_in = ndimage.distance_transform_edt(hard)
inner_edge = np.clip(dist_in / 1.4, 0, 1)                   # 0 at boundary -> 1 inside (single, narrow darkening)
sheen = np.clip(dist_in / 12.0, 0, 1)
fill = gold * (0.75 + 0.25 * inner_edge)[..., None] * (1.0 + 0.12 * sheen)[..., None]
fill = np.clip(fill, 0, 1)
edge_rgb = np.clip(gold * 0.72 + np.array(GOLD_DARK)[None, None, :] * 0.09, 0, 1)
groove_rgb = np.clip(gold * 0.40 + np.array(GOLD_DARK)[None, None, :] * 0.32, 0, 1)
ring = np.clip(wide - core, 0, 1)
shade = bevel_shade(wide, depth=3.0, strength=BEVEL)
rgb = over(fill * shade[..., None], edge_rgb * shade[..., None], ring / np.maximum(wide, 1e-3))
rgb = np.clip(over(rgb, groove_rgb * shade[..., None], np.clip(groove, 0, 1) * (wide > 0.02)), 0, 1)
alpha = wide
el = rgba(rgb, alpha)

save(el, os.path.join(OUT_DIR, 'filigree_a.png'))
save(el[:, :W], os.path.join(OUT_DIR, 'filigree_a_single.png'))
bg = black_fabric(N, N, seed=SEED)
on_black = over(bg, el[..., :3], el[..., 3])
save(on_black, os.path.join(OUT_DIR, 'filigree_a_on_black.png'))

# ============================================================== metrics (review acceptance checks)
a8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
print('alpha distinct values: %d' % len(np.unique(a8)))
opaque = alpha > 0.5
edt = ndimage.distance_transform_edt(opaque)
mx = ndimage.maximum_filter(edt, size=3)
ridge = opaque & (edt >= mx - 0.35) & (edt > 0.5)
_, (iy, ix) = ndimage.distance_transform_edt(~ridge, return_indices=True)
wmap = 2 * edt[iy, ix]
rw = 2 * edt[ridge]
print('ridge full-width median %.1f  p75 %.1f  p90 %.1f px;  %.1f%% of opaque px in strokes <= 6 px' %
      (np.median(rw), np.percentile(rw, 75), np.percentile(rw, 90), 100 * np.mean(wmap[opaque] <= 6)))
# V1 coil width (local full width along the spiral, first 70 % of the path) vs lobe width
v1p = fit(V1_in[:, :2]); sel = arclen(V1_in) <= 0.7 * arclen(V1_in)[-1]
vx = np.clip(v1p[sel, 0].round().astype(int), 0, N - 1); vy = np.clip(v1p[sel, 1].round().astype(int), 0, N - 1)
cw = wmap[vy, vx][opaque[vy, vx]]
print('V1 coil local width: max %.1f  median %.1f  min %.1f px' % (cw.max(), np.median(cw), cw.min()))
rim = opaque & (edt <= 1.5)
print('rim mean RGB %s   interior mean RGB %s   opaque mean RGB %s' % (
    tuple((rgb[rim].mean(0) * 255).round().astype(int)), tuple((rgb[opaque & (edt > 4)].mean(0) * 255).round().astype(int)),
    tuple((rgb[opaque].mean(0) * 255).round().astype(int))))
lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
print('opaque median luminance %.2f' % np.median(lum[opaque]))
ys, xs = np.nonzero(opaque)
print('bbox x %d..%d (%.0f%% w)  y %d..%d (%.0f%% h)  transparent %.0f%%' % (
    xs.min(), xs.max(), 100 * (xs.max() - xs.min()) / N, ys.min(), ys.max(), 100 * (ys.max() - ys.min()) / N,
    100 * np.mean(alpha < 0.02)))
mb = opaque[420:700, 200:460]
a25 = from_pil(to_pil(alpha).resize((256, 256), Image.LANCZOS))
mb25 = a25[105:175, 50:115] > 200 / 255
print('middle band (x200-460,y420-700): opaque %.1f%% at 1:1 (target < 55), solid %.1f%% at 25%% (target < 45)' % (
    100 * mb.mean(), 100 * mb25.mean()))
print('LR mirror alpha diff %d' % int(np.abs(a8 - a8[:, ::-1]).max()))

# ---- debug views: 25 % (x2 nearest) + 12.5 %, 2x zoom of V1, 4x alpha + RGB crop of a serrated lobe, middle band 2x
pil = to_pil(on_black)
small = pil.resize((256, 256), Image.LANCZOS).resize((512, 512), Image.NEAREST)
tiny = pil.resize((128, 128), Image.LANCZOS).resize((256, 256), Image.NEAREST)
sheet = Image.new('RGB', (512 + 10 + 600 + 10 + 300, 600), (30, 30, 30))
sheet.paste(small, (0, 0))
zx, zy = 120, 60
sheet.paste(pil.crop((zx, zy, zx + 300, zy + 300)).resize((600, 600), Image.NEAREST), (522, 0))
cx0, cy0 = 150, 372
acrop = Image.fromarray(a8[cy0:cy0 + 75, cx0:cx0 + 75]).resize((300, 300), Image.NEAREST).convert('RGB')
sheet.paste(acrop, (1132, 0))
sheet.paste(pil.crop((cx0, cy0, cx0 + 75, cy0 + 75)).resize((300, 300), Image.NEAREST), (1132, 300))
sheet.save(os.path.join(WORK, 'dbg_filigree_a_check.png'))
strip = Image.new('RGB', (512 + 10 + 256, 512), (30, 30, 30))
strip.paste(small, (0, 0)); strip.paste(tiny, (522, 0))
strip.save(os.path.join(WORK, 'dbg_filigree_a_small.png'))
pil.crop((200, 400, 624, 720)).resize((848, 640), Image.NEAREST).save(os.path.join(WORK, 'dbg_filigree_a_zoom2.png'))
pil.crop((362, 40, 662, 340)).resize((600, 600), Image.NEAREST).save(os.path.join(WORK, 'dbg_filigree_a_crest.png'))
pil.crop((180, 700, 480, 1000)).resize((600, 600), Image.NEAREST).save(os.path.join(WORK, 'dbg_filigree_a_v2.png'))
Image.fromarray(a8[cy0:cy0 + 75, cx0:cx0 + 75]).resize((300, 300), Image.NEAREST).save(os.path.join(WORK, 'dbg_filigree_a_alpha4x.png'))
print('done %.1fs' % (time.time() - t0))
