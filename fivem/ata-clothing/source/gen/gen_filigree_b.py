"""Baroque gold acanthus filigree, DESIGN APPROACH B (leaf-first), revision 2.

1024x1024 RGBA, transparent background, left-right mirrored.  Everything is
procedural.  The composition (right half, mirrored about x = 512) is a scroll
CARTOUCHE rather than a central column:

  * three large nested C-scrolls per side are the primary mass:
      A - the base scroll: stem springs from the axis low down, sweeps along the
          bottom and up the outer edge, big volute (r0 104) curling inward,
      B - springs from the back of A's volute at ~50 deg, sweeps up beside the
          axis and curls OUTWARD into a volute (r0 86) at mid height,
      C - springs from the back of B's volute, hugs the axis and curls outward
          into a volute (r0 64) at the top.
    Each is a cubic Bezier stem -> log-spiral volute with a fat tapered stroke
    (power < 1 so it stays chunky into the volute); acanthus fronds fan off the
    convex side of the stem and wrap ~70% of the volute.
  * chunky secondary counter-curls (min 8 px wide) inside the C interiors and
    at the top-right, big filler fronds inside every C, randomised leaflets and
    beads on the stems, a bead in every volute eye.
  * only two small leaves on the axis: a narrow tip leaf and a base leaf
    (both <= 44 px wide); fronds that cross the axis mirror into chevrons.

Lobes: pointed, deep-cut fingers (sin^1.2 profile, 5-7 per free leaf, 10-16 along a
scroll, lengths x0.6-1.4, lean 72 -> 15 deg base->tip on free leaves and 60 -> 22 deg
along scrolls, curl ~0.5 so the tips hook forward).

Shapes are rasterised into ONE 4x supersampled 'L' image in painter's order.
Each element is first drawn slightly enlarged in black (an "erase" pass) and
then in white, which leaves a thin transparent separation line wherever it
overlaps something drawn earlier (satin-stitch look).  Veins live on a second
'L' layer and are rendered as OPAQUE dark-gold grooves (they survive a 25%
downscale), not alpha cuts.  The right half is mirrored, embossed (depth 3.5,
strength 0.65), given a specular satin band, a dark inner rim and glitter gold.

Run from the work dir:  python3 gen/gen_filigree_b.py
Outputs: elements/filigree_b.png (1024x1024), elements/filigree_b_single.png
(512x1024 = the un-mirrored RIGHT half; the mirror axis is its LEFT edge),
elements/filigree_b_on_black.png (review composite).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from ata_style import *

t0 = time.time()
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 1024
S = 4                      # supersampling
AX = 512.0                 # mirror axis (x)
GAP = 2.2                  # transparent separation line between overlapping elements (px @1x)
SEED = 23
rng = np.random.default_rng(SEED)

img = Image.new('L', (N * S, N * S), 0)      # fills
cut = Image.new('L', (N * S, N * S), 0)      # vein / groove layer (rendered opaque dark gold)
drw = ImageDraw.Draw(img)
dcut = ImageDraw.Draw(cut)


# ============================================================ geometry helpers
def bezier(p0, p1, p2, p3, n=600):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def resample(pts, step=1.0):
    """Resample a polyline to uniform arc-length spacing `step` (px @1x)."""
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    n = max(3, int(L / step) + 1)
    u = np.linspace(0, L, n)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)


def frames(pts):
    """Unit tangents and normals (tangent rotated +90 deg in image coords)."""
    T = np.gradient(pts, axis=0)
    T /= (np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9)
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    return T, Nn


def curvature_sign(pts):
    T, _ = frames(pts)
    dT = np.gradient(T, axis=0)
    c = T[:, 0] * dT[:, 1] - T[:, 1] * dT[:, 0]
    return 1.0 if c.mean() >= 0 else -1.0


def log_spiral(start, tangent, side, r0, turns, r_ratio=0.2, step=1.0):
    """Volute continuing from `start` along `tangent`, bending toward `side`
    (+1 = +normal, -1 = -normal).  Returns (M,2) points (excluding start) and the centre."""
    T = np.asarray(tangent, np.float64); T /= np.hypot(*T)
    Nn = np.array([-T[1], T[0]]) * side
    c = np.asarray(start, np.float64) + Nn * r0
    th0 = np.arctan2(start[1] - c[1], start[0] - c[0])
    total = turns * 2 * np.pi
    k = np.log(1.0 / r_ratio) / total
    cr = (start[0] - c[0]) * T[1] - (start[1] - c[1]) * T[0]
    sgn = 1.0 if cr >= 0 else -1.0
    phi = np.linspace(0, total, 2500)[1:]
    r = r0 * np.exp(-k * phi)
    th = th0 + sgn * phi
    pts = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], 1)
    return resample(pts, step), c


def unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.hypot(*v) + 1e-9)


def rot(v, deg):
    a = np.deg2rad(deg)
    v = np.asarray(v, np.float64)
    return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])


# ============================================================ rasterisation
SX, SY, CY = 0.93, 0.98, 512.0    # global squeeze about the axis / centre -> ~70% wide, ~90% tall

def tx(x): return (AX + (x - AX) * SX) * S
def ty(y): return (CY + (y - CY) * SY) * S


def poly(P, fill, d=None):
    (d or drw).polygon([(float(tx(x)), float(ty(y))) for x, y in P], fill=fill)


def dots(pts, radii, fill, d=None):
    d = d or drw
    for (x, y), r in zip(pts, radii):
        if r <= 0.15:
            continue
        cx, cy = tx(x), ty(y); rs = r * S
        d.ellipse([cx - rs, cy - rs, cx + rs, cy + rs], fill=fill)


class Element:
    """Collects primitives so the erase pass and fill pass can be run separately."""
    def __init__(self, gap=GAP):
        self.gap = gap
        self.polys = []     # outline_fn(extra) -> points
        self.strokes = []   # (pts, radii)
        self.cuts = []      # callables drawing into the groove layer

    def draw(self):
        g = self.gap
        if g > 0:
            for fn in self.polys:
                poly(fn(g), 0)
            for pts, rr in self.strokes:
                dots(pts, rr + g, 0)
        for fn in self.polys:
            poly(fn(0.0), 255)
            poly(fn(0.0), 0, dcut)              # element covers older veins
        for pts, rr in self.strokes:
            dots(pts, rr, 255)
            dots(pts, rr, 0, dcut)
        for c in self.cuts:
            c()


# ============================================================ shape builders
def envelope(t, hw, base=0.12, peak=0.40):
    """Leaf half-width envelope 0..hw: rounded base (base = t offset), widest near `peak`, pointed tip."""
    a = 0.5
    b = a * (1 - peak) / peak
    e = (t + base) ** a * (1 - t) ** b
    return hw * e / e.max()


def band_outline(pts, w_plus, w_minus):
    """fn(extra) -> polygon around path `pts` with per-point half widths on each side."""
    T, Nn = frames(pts)

    def outline(extra):
        Pp = pts + Nn * (w_plus + extra)[:, None]
        Pm = pts - Nn * (w_minus + extra)[:, None]
        if extra > 0:
            Pp[0] -= T[0] * extra; Pm[0] -= T[0] * extra
            Pp[-1] += T[-1] * extra; Pm[-1] += T[-1] * extra
        return np.vstack([Pp, Pm[::-1]])
    return outline


def finger_profile(u, Wh, base=0.2, power=1.2):
    """Half-width along a lobe: solid base, pointed tip (power > 1 = sharper)."""
    return Wh * np.sin(np.pi * (base + (1 - base) * u)) ** power


def groove(pts, radii):
    """Vein: a tapered polyline of dots in the groove layer."""
    return lambda: dots(pts, radii, 255, dcut)


def add_finger(el, base, direction, L, Wh, curl, vein=True, vein_w=1.3, power=1.2):
    """One pointed acanthus lobe: a curved finger polygon (+ its vein) added to `el`."""
    dv = unit(direction)
    nrm = np.array([-dv[1], dv[0]])
    p0 = np.asarray(base, np.float64)
    p3 = p0 + dv * L + nrm * curl * L
    p1 = p0 + dv * L * 0.38
    p2 = p0 + dv * L * 0.74 + nrm * curl * L * 0.5
    fp = resample(bezier(p0, p1, p2, p3, n=120), 1.0)
    u = np.linspace(0, 1, len(fp))
    w = finger_profile(u, Wh, power=power)
    el.polys.append(band_outline(fp, w, w))
    if vein and L > 16:
        k = int(0.8 * (len(fp) - 1))
        vp = fp[2:k]
        uu = np.linspace(0, 1, len(vp))
        rr = 0.5 * vein_w * (1.0 - 0.55 * uu)
        el.cuts.append(groove(vp, rr))
    return fp


def add_leaf(el, pts, hw, n, rnd, sides=(+1, -1), core=0.24, lean=(72, 15), t_range=(0.05, 0.8), curl=0.5,
             base=0.12, peak=0.42, vein_w=(2.8, 1.2), inner_w=None, outer_core=None, tip=True, width=0.3,
             len_var=(0.6, 1.4), env_fn=None, cap=1.35):
    """Acanthus leaf along midrib `pts`: a core band + pointed fingers leaning toward the tip.
    hw = max perpendicular reach of the fingers.  sides: which sides get fingers.
    inner_w / outer_core: explicit half-width arrays for scroll leaves (smooth stem edge)."""
    M = len(pts)
    t = np.linspace(0, 1, M)
    T, Nn = frames(pts)
    env = envelope(t, hw, base, peak) if env_fn is None else env_fn(t)
    cw = np.maximum(env * core, 2.2)
    if tip:                                   # the tip is itself a finger pointing along the midrib
        tip0 = 0.72
        u = np.clip((t - tip0) / (1 - tip0), 0, 1)
        tipw = env[int(tip0 * (M - 1))] * 0.55 * np.sin(np.pi * (0.13 + 0.87 * u)) ** 1.0
        cw = np.where(t > tip0, np.maximum(tipw, cw * (1 - u)), cw)
    wp = cw if outer_core is None else outer_core
    wm = cw if inner_w is None else inner_w
    if -1 in sides and +1 not in sides:
        wp, wm = wm, wp
    el.polys.append(band_outline(pts, wp, wm))
    # midrib vein
    a, b = 0.04, (0.9 if tip else 0.96)
    sel = (t >= a) & (t <= b)
    vp = pts[sel]; uu = (t[sel] - a) / (b - a)
    rr = 0.5 * (vein_w[0] * (1 - uu) + vein_w[1] * uu)
    el.cuts.append(groove(vp, rr))
    # fingers
    ti = np.linspace(t_range[0], t_range[1], n)
    if n > 2:
        ti[1:-1] += rnd.uniform(-0.3, 0.3, n - 2) * (t_range[1] - t_range[0]) / (n - 1)
    for s in sides:
        r2 = np.random.default_rng(rnd.integers(1 << 30))
        for i, tt in enumerate(ti):
            j = int(tt * (M - 1))
            ang = np.deg2rad(lean[0] + (lean[1] - lean[0]) * i / max(1, n - 1)) * r2.uniform(0.92, 1.08)
            dv = T[j] * np.cos(ang) + s * Nn[j] * np.sin(ang)
            L = min(env[j] / max(np.sin(ang), 0.3) * 1.02, env[j] * cap) * r2.uniform(*len_var)
            Wh = width * L * r2.uniform(0.85, 1.15)
            add_finger(el, pts[j] - T[j] * 3, dv, L, Wh, -s * curl * r2.uniform(0.75, 1.25),
                       vein_w=1.4 if hw > 14 else 1.0)


def stem_radii(n, w0, w1, power=1.0):
    u = np.linspace(0, 1, n) ** power
    return 0.5 * (w0 * (1 - u) + w1 * u)


def smoothstep(a, b, x):
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


def scroll_env(hw, rise=0.28, fall=0.72, end=0.3):
    """Frond reach along a scroll leaf: grows from 0.35*hw, plateaus at hw, eases to end*hw in the volute."""
    return lambda t: hw * (0.35 + 0.65 * smoothstep(0, rise, t)) * (1 - (1 - end) * smoothstep(fall, 1.0, t))


def scroll(p0, p1, p2, p3, curl_side, r0, turns, w0, w1, w_end, leaf=None, groove_line=False, gap=GAP, rnd=None,
           leaf_frac=0.7, power=0.65, r_ratio=0.13):
    """Bezier stem ending in a log-spiral volute.  With `leaf` the Bezier + first `leaf_frac`
    of the volute becomes an acanthus leaf lobed on the outer side.  Returns (element, bezier_pts,
    spiral_pts, spiral_centre)."""
    rnd = rnd or rng
    bp = resample(bezier(p0, p1, p2, p3), 1.0)
    T, _ = frames(bp)
    sp, c = log_spiral(bp[-1], T[-1], curl_side, r0, turns, r_ratio=r_ratio)
    el = Element(gap)
    allp = np.vstack([bp, sp])
    if leaf is None:
        rr = np.concatenate([stem_radii(len(bp), w0, w1, power=power), stem_radii(len(sp), w1, w_end, power=power)])
        el.strokes.append((allp, rr))
        if groove_line:
            sel = rr > 3.4
            gp = allp[sel]; gr = np.clip(rr[sel] * 0.22, 0.6, 1.4)
            el.cuts.append(groove(gp[4:-4], gr[4:-4]))
    else:
        k = len(bp) + int(leaf_frac * len(sp))
        lp = allp[:k]
        outer = -curvature_sign(bp) if leaf.get('side') is None else leaf['side']
        stem = stem_radii(k, w0, w1, power=power)
        inner_w = stem * leaf.get('inner', 1.3)
        add_leaf(el, lp, leaf['hw'], leaf['n'], rnd, sides=(outer,), lean=leaf.get('lean', (60, 22)),
                 t_range=leaf.get('t_range', (0.03, 0.95)), curl=leaf.get('curl', 0.5), base=leaf.get('base', 0.2),
                 peak=leaf.get('peak', 0.45), vein_w=(leaf.get('vein', 2.6), 1.1), inner_w=inner_w, outer_core=stem,
                 tip=False, width=leaf.get('width', 0.3), len_var=leaf.get('len_var', (0.8, 1.25)),
                 env_fn=scroll_env(leaf['hw'], end=leaf.get('end', 0.3)), cap=leaf.get('cap', 1.15))
        rr_s = stem_radii(len(sp), w1, w_end, power=power)
        j = max(0, int(leaf_frac * len(sp)) - 6)
        el.strokes.append((sp[j:], rr_s[j:]))
        # a groove down the middle of the bare volute end
        gp = sp[j + 6:]; gr = np.clip(rr_s[j + 6:] * 0.22, 0.6, 1.4)
        if len(gp) > 12:
            el.cuts.append(groove(gp[:-6], gr[:-6]))
    return el, bp, sp, c


def leaflet(base, direction, length, hw, n=3, rnd=None, curve=0.25, gap=0.0):
    """Small acanthus leaf from `base` pointing along `direction`, slightly curved."""
    rnd = rnd or rng
    d = unit(direction)
    nrm = np.array([-d[1], d[0]])
    p0 = np.asarray(base, np.float64)
    p3 = p0 + d * length + nrm * curve * length
    p1 = p0 + d * length * 0.35
    p2 = p0 + d * length * 0.7 + nrm * curve * length * 0.5
    pts = resample(bezier(p0, p1, p2, p3, n=200), 1.0)
    el = Element(gap)
    add_leaf(el, pts, hw, n, rnd, core=0.3, lean=(68, 22), t_range=(0.08, 0.74), curl=0.4, base=0.1, peak=0.42,
             vein_w=(1.8, 0.8), width=0.32, len_var=(0.75, 1.25))
    return el


def bead(c, r, gap=2.0):
    el = Element(gap)
    el.strokes.append((np.array([c], np.float64), np.array([r])))
    return el


def leaflets_on(bp, params, side=None):
    """Attach randomised leaflets to a stem.  params: list of (t, length, hw, angle_deg).
    Each instance is scaled x0.7-1.3, rotated +/-15 deg, given 3-5 fingers; sides alternate."""
    T, Nn = frames(bp)
    outer = -curvature_sign(bp) if side is None else side
    els = []
    for i, (t, L, hw, ang) in enumerate(params):
        j = int(t * (len(bp) - 1))
        s = outer * (1 if i % 2 == 0 else -1)
        a = np.deg2rad(ang + rng.uniform(-15, 15))
        d = Nn[j] * s * np.cos(a) + T[j] * np.sin(a)
        k = rng.uniform(0.7, 1.3)
        cv = rng.uniform(0.15, 0.35) * (1 if rng.random() < 0.5 else -1)
        els.append(leaflet(bp[j] - T[j] * 2, d, L * k, hw * k, n=int(rng.integers(3, 6)), curve=cv))
    return els


def volute_fan(sp, c, params, gap=0.0):
    """Leaflets sprouting outward from the outer arc of a volute. params: (t_on_spiral, length, hw, angle_deg)."""
    els = []
    T, Nn = frames(sp)
    for (t, L, hw, ang) in params:
        j = int(t * (len(sp) - 1))
        rad = unit(sp[j] - c)
        a = np.deg2rad(ang + rng.uniform(-12, 12))
        d = rad * np.cos(a) + T[j] * np.sin(a)
        k = rng.uniform(0.8, 1.25)
        cv = rng.uniform(0.12, 0.3) * (1 if rng.random() < 0.5 else -1)
        els.append(leaflet(sp[j] - rad * 2, d, L * k, hw * k, n=int(rng.integers(3, 6)), curve=cv, gap=gap))
    return els


def spiral_point(sp, c, frac, out=0.0):
    """Point on a volute at fraction `frac` of its length, pushed `out` px away from the centre."""
    j = int(frac * (len(sp) - 1))
    rad = unit(sp[j] - c)
    return sp[j] + rad * out, rad


def big_leaf(base, direction, length, hw, n, curve=0.18, gap=GAP, lean=(70, 18), curl=0.5):
    """Free-standing acanthus frond (filler inside the C interiors)."""
    d = unit(direction)
    nrm = np.array([-d[1], d[0]])
    p0 = np.asarray(base, np.float64)
    p1 = p0 + d * length * 0.33
    p2 = p0 + d * length * 0.7 + nrm * curve * length * 0.6
    p3 = p0 + d * length + nrm * curve * length
    pts = resample(bezier(p0, p1, p2, p3), 1.0)
    el = Element(gap)
    add_leaf(el, pts, hw, n, rng, core=0.26, lean=lean, t_range=(0.05, 0.78), curl=curl, base=0.12, peak=0.42,
             vein_w=(2.4, 1.0), len_var=(0.7, 1.3))
    return el


# ============================================================ composition (right half, axis x = 512)
# curl_side convention: heading UP on the right half, -1 curls inward (toward the axis), +1 outward;
# heading DOWN, +1 curls inward and -1 outward.
elements = []

# ---- main C-scrolls (volutes are the primary mass) ----------------------------------------
# A: base scroll.  Springs from the axis, sweeps along the bottom and up the outer edge,
#    volute curls inward.  Fronds fan off the convex (outer) side and wrap 70% of the volute.
A, A_b, A_s, A_c = scroll((512, 900), (600, 936), (792, 932), (850, 800), curl_side=-1, r0=104, turns=1.4,
                          w0=28, w1=22, w_end=13, leaf=dict(hw=46, n=16, inner=1.25), leaf_frac=0.7)
# B: springs from the back (outer arc) of A's volute at ~50 deg, sweeps up beside the axis and curls OUTWARD.
B0, _ = spiral_point(A_s, A_c, 0.36, out=4)
B, B_b, B_s, B_c = scroll(B0, (696, 616), (536, 496), (622, 402), curl_side=+1, r0=86, turns=1.4,
                          w0=24, w1=18, w_end=11, leaf=dict(hw=46, n=13, inner=1.25), leaf_frac=0.7)
# C: springs from the back of B's volute, hugs the axis and curls outward at the top.
C0, _ = spiral_point(B_s, B_c, 0.02, out=4)
C, C_b, C_s, C_c = scroll(C0, (574, 322), (542, 250), (580, 168), curl_side=+1, r0=64, turns=1.4,
                          w0=18, w1=14, w_end=9, leaf=dict(hw=34, n=10, inner=1.25), leaf_frac=0.7)

# ---- chunky secondary counter-curls (min width 8 px) --------------------------------------
# A2: from the inner side of A's stem up into A's C interior, curling outward (clockwise).
A2, A2_b, A2_s, A2_c = scroll((606, 916), (586, 870), (596, 810), (640, 780), curl_side=-1, r0=44, turns=1.4,
                              w0=15, w1=12, w_end=8, groove_line=True)
# B2: from B's stem back into the space between B's volute and A's volute, curling inward.
B2_0 = B_b[int(0.5 * (len(B_b) - 1))]
B2, B2_b, B2_s, B2_c = scroll(B2_0, (660, 572), (712, 560), (742, 532), curl_side=+1, r0=34, turns=1.4,
                              w0=13, w1=11, w_end=8, groove_line=True)
# C2: hook at the top-right off the back of B's volute.
C2_0, _ = spiral_point(B_s, B_c, 0.2, out=4)
C2, C2_b, C2_s, C2_c = scroll(C2_0, (812, 350), (836, 290), (802, 236), curl_side=-1, r0=40, turns=1.4,
                              w0=14, w1=11, w_end=8, groove_line=True)
# D2: small hook off the bottom-right of A's stem hugging the corner.
D2, D2_b, D2_s, D2_c = scroll((828, 906), (862, 930), (900, 900), (896, 858), curl_side=-1, r0=26, turns=1.3,
                              w0=11, w1=9, w_end=8, groove_line=True)
# E2: hook between C's volute and C2 at the upper right.
E2, E2_b, E2_s, E2_c = scroll((720, 240), (748, 210), (760, 160), (730, 132), curl_side=-1, r0=28, turns=1.3,
                              w0=11, w1=9, w_end=8, groove_line=True)

# ---- filler fronds inside the C interiors (drawn first, the scrolls overlap them) -----------
def on_stem(bp, t, side=1, inset=6):
    """Point on stem `bp` at parameter t, moved `inset` px toward its `side` normal (so a frond base sits inside the stroke)."""
    T, Nn = frames(bp)
    j = int(t * (len(bp) - 1))
    return bp[j] + Nn[j] * side * inset, T[j], Nn[j]

fillers = []
p, T_, N_ = on_stem(A_b, 0.10, -1);  fillers.append(big_leaf(p, (0.3, -1.0), 96, 28, 6, curve=-0.2))     # A interior, beside the axis
p, T_, N_ = on_stem(A_b, 0.46, -1);  fillers.append(big_leaf(p, (0.1, -1.0), 76, 22, 5, curve=0.22))     # A interior, right of A2
p, T_, N_ = on_stem(A2_b, 0.5, +1);  fillers.append(big_leaf(p, (1.0, -0.4), 70, 22, 4, curve=0.2))      # off A2's back
p, T_, N_ = on_stem(B_b, 0.32, +1);  fillers.append(big_leaf(p, (1.0, -0.3), 86, 26, 5, curve=-0.2))     # B interior
p, T_, N_ = on_stem(B2_b, 0.45, -1); fillers.append(big_leaf(p, (0.9, -0.9), 66, 20, 4, curve=0.2))      # between B2 and B's volute
p, T_, N_ = on_stem(C_b, 0.35, +1);  fillers.append(big_leaf(p, (1.0, -0.2), 80, 24, 5, curve=-0.2))     # C interior
p, _ = spiral_point(B_s, B_c, 0.12, out=-4); fillers.append(big_leaf(p, (0.45, -1.0), 82, 24, 5, curve=0.18))  # upper right, under C2
p, _ = spiral_point(A_s, A_c, 0.20, out=-4); fillers.append(big_leaf(p, (0.55, -1.0), 74, 22, 4, curve=0.18))  # above A's volute, right
p, T_, N_ = on_stem(B_b, 0.55, -1);  fillers.append(big_leaf(p, (-0.95, -0.3), 82, 25, 5, curve=-0.2))   # central frond toward the axis
p, T_, N_ = on_stem(B_b, 0.40, -1);  fillers.append(big_leaf(p, (-0.75, 0.65), 78, 24, 5, curve=0.2))    # frond down into the gap above A2
elements += fillers

for e in (D2, A2, B2, C2, E2, A, B, C):
    elements.append(e)

# ---- randomised leaflets on secondary stems and fans off the volute backs ------------------
elements += leaflets_on(A2_b, [(0.35, 40, 14, 30), (0.7, 34, 12, 30)])
elements += leaflets_on(B2_b, [(0.4, 36, 13, 30)])
elements += leaflets_on(C2_b, [(0.35, 40, 14, 30), (0.7, 34, 12, 30)])
elements += leaflets_on(D2_b, [(0.5, 30, 11, 30)])
elements += leaflets_on(E2_b, [(0.45, 30, 11, 30)])
elements += volute_fan(A_s, A_c, [(0.74, 60, 20, 22), (0.84, 44, 15, 30)])
elements += volute_fan(B_s, B_c, [(0.74, 50, 17, 22), (0.84, 38, 13, 30)])
elements += volute_fan(C_s, C_c, [(0.74, 40, 14, 22), (0.84, 32, 11, 30)])
elements += volute_fan(A2_s, A2_c, [(0.1, 40, 13, 18), (0.45, 30, 11, 20)])
elements += volute_fan(B2_s, B2_c, [(0.1, 34, 12, 18)])
elements += volute_fan(C2_s, C2_c, [(0.1, 36, 12, 18), (0.5, 28, 10, 20)])

# ---- the only axis leaves: a narrow tip leaf and a base leaf (<= 44 px wide) ---------------
for (yb, yt, hw, n) in ((232, 66, 21, 5), (896, 986, 19, 4)):
    pts = resample(np.array([[AX, yb], [AX, yt]], np.float64), 1.0)
    el = Element(GAP)
    r1 = np.random.default_rng(SEED + int(yb))     # only the right half is kept, so any rng is mirror-safe
    add_leaf(el, pts, hw, n, r1, core=0.34, lean=(66, 20), t_range=(0.06, 0.74), curl=0.42, base=0.12, peak=0.45,
             vein_w=(2.8, 1.2), width=0.32)
    elements.append(el)

# ---- beads: one in every volute eye (varied radius), triple-dot clusters at branch points ---
for c, r in ((A_c, 9.0), (B_c, 8.0), (C_c, 6.5), (A2_c, 5.5), (B2_c, 4.5), (C2_c, 5.0), (D2_c, 4.0), (E2_c, 4.0)):
    elements.append(bead(c, r * rng.uniform(0.9, 1.1)))
for bp_, tt, off in ((A_b, 0.45, 40), (B_b, 0.28, 32), (C_b, 0.5, 26)):
    T, Nn = frames(bp_)
    j = int(tt * (len(bp_) - 1))
    cc = bp_[j] + Nn[j] * (curvature_sign(bp_)) * off
    for k in (-1, 0, 1):
        elements.append(bead(cc + T[j] * k * 10, rng.uniform(3.0, 4.5)))
# crown of dots above the tip leaf, pearl drop at the centre
for (x, y) in ((AX, 48), (530, 58), (548, 74)):
    elements.append(bead((x, y), rng.uniform(2.8, 3.6)))
for (y, r) in ((618, 5.0), (640, 4.2), (658, 3.4)):
    elements.append(bead((AX, y), r))

# ============================================================ draw
for el in elements:
    el.draw()
print('drawn %d elements in %.1fs' % (len(elements), time.time() - t0))

fill_m = from_pil(img.resize((N, N), Image.LANCZOS))
cut_m = from_pil(cut.resize((N, N), Image.LANCZOS))
# round sharp notch corners slightly (satin-stitch look) and re-crisp the edge
fill_m = np.clip((blur(fill_m, 1.0) - 0.5) * 2.6 + 0.5, 0, 1)
cut_m = np.clip(cut_m * fill_m, 0, 1)

# mirror the right half
def mirror(m):
    right = m[:, N // 2:]
    return np.concatenate([right[:, ::-1], right], axis=1)

full = mirror(fill_m)
grooves = mirror(cut_m)

# ============================================================ render
gold = gold_fill(N, N, seed=SEED, glitter=0.6)
edge = np.clip(gold * 0.5 + np.array(GOLD_DARK)[None, None, :] * 0.25, 0, 1)
out = emboss_element(full, gold, edge_width=1.3, edge_rgb=edge, depth=3.5, strength=0.65)
rgb = out[..., :3]
# specular satin band: a soft dome over every stroke, lit from the top-left
dome = blur(full, 3.0)
gy, gx = np.gradient(dome)
spec = -(gx * -0.6 + gy * -0.8)
spec = np.clip(spec / (spec.max() + 1e-9), 0, 1) ** 1.4
rgb = rgb + np.array(GOLD_LIGHT)[None, None, :] * (0.35 * spec * full)[..., None]
# veins / grooves: opaque dark gold (alpha unchanged)
groove_col = np.array(GOLD_DARK, np.float32)[None, None, :] * 0.55
rgb = over(rgb, np.broadcast_to(groove_col, rgb.shape), np.clip(grooves * 1.2, 0, 1) * 0.9)
# inner dark rim for a crisper stitched edge + subtle sheen across the ornament
rim = np.clip(full - erode(full, 1.8), 0, 1)
sheen = 0.9 + 0.2 * fbm(N, N, octaves=3, base_scale=180, seed=SEED + 5)
rgb = np.clip(rgb * (1 - 0.45 * rim)[..., None] * sheen[..., None], 0, 1)
out = rgba(rgb, out[..., 3])

os.makedirs(OUT_DIR, exist_ok=True)
save(out, OUT_DIR + '/filigree_b.png')
save(out[:, N // 2:], OUT_DIR + '/filigree_b_single.png')
bg = black_fabric(N, N, seed=3)
review = over(bg, out[..., :3], out[..., 3])
save(review, OUT_DIR + '/filigree_b_on_black.png')

# debug: 25% preview (shown 2x) and a 2x zoom crop
to_pil(review).resize((N // 4, N // 4), Image.LANCZOS).resize((N // 2, N // 2), Image.NEAREST).save(WORK + '/dbg_filb_25pct.png')
to_pil(review).crop((512, 560, 896, 944)).resize((768, 768), Image.NEAREST).save(WORK + '/dbg_filb_zoom2x.png')
ys, xs = np.nonzero(full > 0.5)
print('bbox x %d..%d (%.0f%%)  y %d..%d (%.0f%%)' % (xs.min(), xs.max(), (xs.max() - xs.min()) / N * 100, ys.min(), ys.max(), (ys.max() - ys.min()) / N * 100))
op = full > 0.5
spine = op[:, (np.abs(np.arange(N) - AX) < 75)].sum() / op.sum()
lum = (0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2])[op]
print('spine-zone share %.0f%%  lum p5 %.2f mean %.2f p95 %.2f' % (spine * 100, np.percentile(lum, 5), lum.mean(), np.percentile(lum, 95)))
print('done in %.1fs' % (time.time() - t0))
