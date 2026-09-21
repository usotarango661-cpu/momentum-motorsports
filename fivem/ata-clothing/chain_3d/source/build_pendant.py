#!/usr/bin/env python3
"""Build the jewelled ATA pendant (v2: real faceted pave gems) as a GLB.

Pendant frame (see mesh_utils): metres, Y up, front = +Z, top of the bail ring at the origin,
pendant hangs in -Y.  Run from scratchpad/chain3d:  python3 gen/build_pendant.py

v2 changes vs v1
  * the letters' front face is pave-packed with ~800 real low-poly round brilliants (flat shaded,
    deep red) on a jittered hex grid, plus a continuous single row of small white brilliants that
    follows the letter outline (outer + counter holes) ~1.1 mm inside the edge
  * trapezoid pave bail (real red + white gems), polished gold jump ring (major 4 / minor 1.2 mm)
    with its top at the origin and two gold prongs down to the T
  * 2048^2 baked front texture (dark bezel field + gold prong beads + gold rails) + normal map;
    walls / back get polished gold with a tiled brushed micro-normal
  * writes out/pendant_meta.json (ring geometry) so assemble.py can hang the ring correctly
"""
import os
import sys
import math
import json

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree, Delaunay
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.prepared import prep
from shapely import affinity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mesh_utils as mu  # noqa: E402

OUT = mu.OUT
RENDERS = os.path.join(os.path.dirname(OUT), 'renders')
os.makedirs(OUT, exist_ok=True)
os.makedirs(RENDERS, exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- parameters (metres)
LETTER_W = 0.098          # letters width
DEPTH = 0.007             # letters thickness
BEVEL = 0.0005            # chamfer inset on the front (the polished gold rim)
BEVEL_H = 0.0006          # chamfer height
BEVEL_STEPS = 2
SIMPLIFY_PX = 1.0
TEX_SIZE = 2048
NRM_STRENGTH = 2.2

# red pave (letters)
RED_R = 0.00080           # gem girdle radius (~1.6 mm stones, as measured in the photo)
RED_GAP = 0.00010         # metal between neighbouring stones
RED_JITTER = 0.00010
RED_MED_R = 0.00068       # medium / small stones that fill the gaps, drips and spikes
RED_SMALL_R = 0.00055
WHITE_SMALL_R = 0.00055   # small diamonds for the spike tips the main row cannot reach
RAIL = 0.00025            # gold rail between the white row and the red field
# white diamond outline
WHITE_R = 0.00072
WHITE_INSET = 0.00115     # row centre-line offset inside the edge
WHITE_SPACING = 0.00158
GEM_LIFT = 0.00005        # girdle sits this far above the face
GEM_CROWN_H = 0.36        # crown height / radius
GEM_PAV_H = 0.45          # pavilion depth / radius (embedded in the metal)
GEM_TABLE = 0.56          # table radius / girdle radius

# bail
BAIL_TOP_W = 0.0110
BAIL_BOT_W = 0.0070
BAIL_H = 0.0160
BAIL_DEPTH = 0.0025
BAIL_BEVEL = 0.00035
BAIL_BEVEL_H = 0.0004
BAIL_RED_R, BAIL_SMALL_R = 0.00062, 0.00042
BAIL_WHITE_R, BAIL_WHITE_INSET, BAIL_WHITE_SPACING = 0.00050, 0.00085, 0.00118
RING_R, RING_r = 0.0066, 0.0013        # jump ring (major, minor)
RING_EMBED = 0.0028                   # ring centre sits this far above the bail plate's top edge
PRONG_GAP = 0.0022                    # plate bottom -> T top
PRONG_W, PRONG_D = 0.0016, 0.0022

GOLD_DARK = [0.62, 0.46, 0.18, 1.0]
RUBY = [0.36, 0.004, 0.015, 1.0]       # deep crimson (linear); brighter values go pink under ACES
DIAMOND = [0.72, 0.72, 0.76, 1.0]


# ---------------------------------------------------------------- helpers
def geoms(p):
    """List of shapely Polygons from a Polygon/MultiPolygon/GeometryCollection."""
    if p is None or p.is_empty:
        return []
    if isinstance(p, Polygon):
        return [p]
    out = []
    for g in getattr(p, 'geoms', []):
        if isinstance(g, Polygon) and g.area > 0:
            out.append(g)
        elif isinstance(g, MultiPolygon):
            out.extend(geoms(g))
    return out


def rings(p, min_len=0.0):
    """All boundary rings (exterior + holes) of a (Multi)Polygon as coordinate arrays."""
    out = []
    for g in geoms(p):
        out.append(np.array(g.exterior.coords))
        out += [np.array(i.coords) for i in g.interiors if i.length > min_len]
    return out


def extrude_bevel_labeled(poly, depth, bevel, bevel_h, steps, scale, min_area, eps):
    """Extrude `poly` (its own units * scale -> metres) along +Z from 0 to depth with a chamfer of
    width `bevel` and height `bevel_h` (poly units / metres) on the +Z side, built as `steps`
    inset slabs.  MultiPolygon-safe; hidden interior faces removed.  Returns V (metres), F and
    per-face labels: 0=back cap, 1=wall, 2=tread (chamfer step), 3=front."""
    if bevel <= 0 or steps <= 0:
        levels = [(poly, depth)]
    else:
        body = depth - bevel_h
        step = bevel_h / steps
        levels = [(poly, body)]
        for i in range(1, steps + 1):
            inset = poly.buffer(-bevel * i / steps, join_style=2)
            inset = MultiPolygon([g for g in geoms(inset) if g.area >= min_area])
            if inset.is_empty:
                break
            levels.append((inset, step))
    V, F, L = [], [], []
    z0 = 0.0
    nv = 0
    for k, (shape, h) in enumerate(levels):
        nxt = prep(levels[k + 1][0].buffer(-eps)) if k + 1 < len(levels) else None
        for g in geoms(shape):
            m = trimesh.creation.extrude_polygon(g, h)
            m.fix_normals()
            n = m.face_normals
            verts = m.vertices.copy()
            verts[:, 2] += z0
            lab = np.full(len(m.faces), 1, dtype=np.int8)           # walls
            up = n[:, 2] > 0.9
            down = n[:, 2] < -0.9
            lab[up] = 3 if k == len(levels) - 1 else 2
            lab[down] = 0
            keep = np.ones(len(m.faces), dtype=bool)
            if k > 0:
                keep[down] = False                                    # hidden slab bottoms
            if nxt is not None:
                tri = verts[m.faces[:, :]]
                for fi in np.where(up)[0]:
                    if nxt.contains(Polygon(tri[fi, :, :2])):
                        keep[fi] = False                              # hidden under next slab
            V.append(verts)
            F.append(m.faces[keep] + nv)
            L.append(lab[keep])
            nv += len(verts)
        z0 += h
    V = np.vstack(V)
    V[:, :2] *= scale
    return V, np.vstack(F), np.concatenate(L)


def part_from_faces(V, F, sel, unshare=False):
    m = trimesh.Trimesh(V, F[sel], process=False)
    m.remove_unreferenced_vertices()
    if unshare:
        m.unmerge_vertices()
    return m


def wall_uv(m, tile=1.0):
    """UVs for vertical walls: u = normalised X+Y sweep (tiled), v = normalised Z."""
    v = m.vertices
    b = m.bounds
    u = ((v[:, 0] - b[0, 0]) / max(b[1, 0] - b[0, 0], 1e-9) + (v[:, 1] - b[0, 1]) / max(b[1, 1] - b[0, 1], 1e-9)) * 0.5
    w = (v[:, 2] - b[0, 2]) / max(b[1, 2] - b[0, 2], 1e-9)
    return np.column_stack([u * tile, w])


def brilliant(r, crown_h, pav_h, table, az, tilt):
    """Low-poly round brilliant as an unshared triangle soup (flat shading): octagonal table,
    16 crown facets (8 bezel + 8 star, antiprism pattern) to an octagonal girdle, 8 pavilion
    facets to a culet.  Girdle at z=0.  30 tris."""
    a8 = np.arange(8) * (np.pi / 4)
    g = np.column_stack([r * np.cos(a8), r * np.sin(a8), np.zeros(8)])                       # girdle
    t = np.column_stack([r * table * np.cos(a8 + np.pi / 8), r * table * np.sin(a8 + np.pi / 8),
                         np.full(8, crown_h)])                                                # table
    culet = np.array([0.0, 0.0, -pav_h])
    tris = []
    for i in range(8):
        j = (i + 1) % 8
        tris.append([g[i], g[j], t[i]])        # bezel-ish
        tris.append([t[i], g[j], t[j]])        # star-ish
        tris.append([g[j], g[i], culet])       # pavilion
    for i in range(1, 7):                      # table fan
        tris.append([t[0], t[i], t[i + 1]])
    P = np.array(tris).reshape(-1, 3)
    R = trimesh.transformations.euler_matrix(tilt[0], tilt[1], az, 'sxyz')[:3, :3]
    return P @ R.T


def gem_cloud(centres, r, name, tilt_deg=4.0):
    """Concatenate one brilliant per centre (x, y, z_girdle). Returns (V, F)."""
    V = []
    n_ver = 30 * 3
    for c in centres:
        az = rng.uniform(0, 2 * np.pi)
        tilt = np.deg2rad(rng.normal(0, tilt_deg / 2.5, 2))
        V.append(brilliant(r, r * GEM_CROWN_H, r * GEM_PAV_H, GEM_TABLE, az, tilt) + c)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    V = np.vstack(V)
    F = np.arange(len(V)).reshape(-1, 3)
    print(f'{name}: {len(centres)} gems, {len(F)} tris')
    return V, F


def hex_points(poly, pitch, jitter):
    """Jittered hexagonal grid clipped to `poly` (metres)."""
    if poly.is_empty:
        return np.zeros((0, 2))
    pp = prep(poly)
    minx, miny, maxx, maxy = poly.bounds
    dy = pitch * math.sqrt(3) / 2
    pts = []
    row = 0
    y = miny + pitch * 0.5
    while y <= maxy:
        xoff = pitch / 2 if row % 2 else 0.0
        for x in np.arange(minx + xoff, maxx + 1e-9, pitch):
            ang = rng.uniform(0, 2 * np.pi)
            rad = jitter * math.sqrt(rng.uniform())
            p = (x + rad * math.cos(ang), y + rad * math.sin(ang))
            if pp.contains(Point(p)):
                pts.append(p)
        y += dy
        row += 1
    return np.array(pts) if pts else np.zeros((0, 2))


def ring_points(poly, inset, spacing, min_ring_len):
    """Evenly spaced points along every boundary ring of poly.buffer(-inset); each ring gets an
    integer number of stones so the row closes on itself."""
    pts = []
    for coords in rings(poly.buffer(-inset, join_style=1), min_ring_len):
        ls = LineString(coords)
        n = max(3, int(round(ls.length / spacing)))
        if ls.length < 2.2 * spacing:
            continue
        pts += [ls.interpolate(i / n, normalized=True).coords[0] for i in range(n)]
    return np.array(pts) if pts else np.zeros((0, 2))


def pave_rows(poly, inset0, r, gap, existing, along=None, row_gap=None, jitter=0.0, max_rows=80):
    """Contour-following pave: walk successive inward offsets of `poly` (metres) starting at
    `inset0`, `row_gap` apart, and greedily drop a stone of radius r wherever it does not overlap
    any stone already placed (`existing`: list of [x, y, r]).  Walking each contour at a fine step
    packs the stones tightly and lets each row settle into the gaps of the previous one.
    Returns the new stones as an (n, 3) array [x, y, r]."""
    along = along or (2 * r + gap)
    row_gap = row_gap or along * 0.88
    step = along / 7.0
    A = np.asarray(existing, dtype=float).reshape(-1, 3)
    placed = []
    for k in range(max_rows):
        q = poly.buffer(-(inset0 + k * row_gap), join_style=1)
        if q.is_empty:
            break
        for coords in rings(q, 0):
            ls = LineString(coords)
            n = int(ls.length / step)
            if n < 2:
                cands = [ls.interpolate(0.5, normalized=True).coords[0]]
            else:
                off = rng.uniform(0, 1) / n
                cands = [ls.interpolate(((i / n) + off) % 1.0, normalized=True).coords[0] for i in range(n)]
            for x, y in cands:
                if jitter > 0:
                    ang = rng.uniform(0, 2 * np.pi); rad = jitter * math.sqrt(rng.uniform())
                    x += rad * math.cos(ang); y += rad * math.sin(ang)
                if len(A):
                    d = np.hypot(A[:, 0] - x, A[:, 1] - y)
                    if not np.all(d >= A[:, 2] + r + gap):
                        continue
                placed.append([x, y, r])
                A = np.vstack([A, [x, y, r]])
    return np.array(placed) if placed else np.zeros((0, 3))


def fill_random(poly, r, gap, existing, n_cand=70000):
    """Gap filler: random candidates inside `poly` (metres), greedily accepted where a stone of
    radius r fits between the stones in `existing` ([x, y, r] rows).  Returns new (n, 3) rows."""
    if poly.is_empty:
        return np.zeros((0, 3))
    pp = prep(poly)
    minx, miny, maxx, maxy = poly.bounds
    xs = rng.uniform(minx, maxx, n_cand); ys = rng.uniform(miny, maxy, n_cand)
    inside = np.array([pp.contains(Point(x, y)) for x, y in zip(xs, ys)])
    A = np.asarray(existing, dtype=float).reshape(-1, 3)
    placed = []
    for x, y in zip(xs[inside], ys[inside]):
        if len(A):
            d = np.hypot(A[:, 0] - x, A[:, 1] - y)
            if not np.all(d >= A[:, 2] + r + gap):
                continue
        placed.append([x, y, r])
        A = np.vstack([A, [x, y, r]])
    return np.array(placed) if placed else np.zeros((0, 3))


def thin_points(pts, min_d):
    """Greedy thinning: drop points closer than min_d to an already accepted point."""
    keep = []
    for p in pts:
        if not keep or np.min(np.linalg.norm(np.array(keep) - p, axis=1)) >= min_d:
            keep.append(p)
    return np.array(keep) if keep else np.zeros((0, 2))


def reject_near(pts, others, min_d):
    if len(pts) == 0 or len(others) == 0:
        return pts
    d = cKDTree(others).query(pts)[0]
    return pts[d >= min_d]


def prong_points(centres, pitch):
    """Gold bead positions between mutually adjacent stones (Delaunay triangle centroids)."""
    if len(centres) < 3:
        return np.zeros((0, 2))
    tri = Delaunay(centres)
    out = []
    for s in tri.simplices:
        p = centres[s]
        e = [np.linalg.norm(p[i] - p[(i + 1) % 3]) for i in range(3)]
        if max(e) < 1.55 * pitch:
            out.append(p.mean(axis=0))
    return np.array(out) if out else np.zeros((0, 2))


def normal_from_tex(rgb, strength, blur=0.8):
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    hgt = ndimage.gaussian_filter(lum, blur)
    gx = ndimage.sobel(hgt, axis=1) / 8.0
    gy = ndimage.sobel(hgt, axis=0) / 8.0
    nx = -gx * strength
    ny = gy * strength                     # +Y = up in the image (OpenGL / glTF convention)
    nz = np.ones_like(nx)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    return Image.fromarray((np.clip(nrm, 0, 1) * 255).astype(np.uint8), 'RGB')


def brushed_normal(size=512, strength=0.28, seed=3):
    """Tileable brushed-metal micro normal map (streaks along u)."""
    r = np.random.default_rng(seed)
    h = r.normal(size=(size, 1))
    h = ndimage.gaussian_filter1d(h, 1.2, axis=0, mode='wrap')
    h = np.repeat(h, size, axis=1)
    h += 0.35 * ndimage.gaussian_filter(r.normal(size=(size, size)), 0.9, mode='wrap')
    h = (h - h.mean()) / (h.std() + 1e-9)
    gx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    gy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    nx = -gx * strength
    ny = gy * strength
    nz = np.ones_like(nx)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    return Image.fromarray((np.clip(nrm, 0, 1) * 255).astype(np.uint8), 'RGB')


class Baker:
    """Draw in world XY (metres) onto a square texture whose planar UV bounds are `bounds`."""

    def __init__(self, size, bounds, base_rgb):
        self.size = size
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.img = Image.fromarray((np.clip(base_rgb, 0, 1) * 255).astype(np.uint8), 'RGB')
        self.draw = ImageDraw.Draw(self.img)
        self.ppm = size / (self.xmax - self.xmin)      # pixels per metre (x); y uses its own scale

    def px(self, x, y):
        u = (x - self.xmin) / (self.xmax - self.xmin)
        v = (y - self.ymin) / (self.ymax - self.ymin)
        return (u * self.size, (1.0 - v) * self.size)

    def poly(self, p, fill):
        for g in geoms(p):
            self.draw.polygon([self.px(*c) for c in g.exterior.coords], fill=fill)
            for h in g.interiors:
                pass  # holes handled by painting them afterwards via poly_ring_band

    def band(self, outer, inner, fill):
        """Paint outer minus inner (both (Multi)Polygons) by drawing rings as thick outlines."""
        diff = outer.difference(inner)
        for g in geoms(diff):
            # draw as many thin polygons: triangulate with trimesh (ear clipping)
            try:
                tris = trimesh.creation.triangulate_polygon(g)
            except Exception:
                continue
            vv, ff = tris
            for f in ff:
                self.draw.polygon([self.px(*vv[i]) for i in f], fill=fill)

    def dots(self, pts, r, fill):
        rp = r * self.ppm
        for x, y in pts:
            cx, cy = self.px(x, y)
            self.draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], fill=fill)

    def lines(self, p, inset, width, fill):
        wpx = max(1, int(round(width * self.ppm)))
        for coords in rings(p.buffer(-inset, join_style=1), 0):
            self.draw.line([self.px(*c) for c in coords], fill=fill, width=wpx, joint='curve')

    def rgb(self):
        return np.asarray(self.img).astype(np.float32) / 255.0


def gold_rgb(a):
    return tuple(int(v * 255) for v in a)


G_BRIGHT = gold_rgb((1.00, 0.86, 0.48))
G_MID = gold_rgb((0.86, 0.66, 0.26))
G_DARK = gold_rgb((0.42, 0.30, 0.10))
RED_SEAT = gold_rgb((0.20, 0.01, 0.01))
WHITE_SEAT = gold_rgb((0.70, 0.66, 0.60))


def bake(name, size, bounds, base_rgb, poly, red_pts, white_pts, red_r, white_r, red_pitch,
         white_inset, bevel, rail_inset, nrm_strength):
    b = Baker(size, bounds, base_rgb)
    # outer polished rim (edge .. white row) in bright gold, and the rail between whites and reds
    b.band(poly, poly.buffer(-(white_inset - white_r * 0.85), join_style=1), G_MID)
    b.lines(poly, rail_inset, 0.00035, G_BRIGHT)
    # seats under the stones (dark, so any sub-pixel gap reads as a bezel shadow)
    b.dots(white_pts, white_r * 1.05, WHITE_SEAT)
    b.dots(red_pts, red_r * 1.08, RED_SEAT)
    # tiny gold prong beads between stones
    b.dots(prong_points(red_pts, red_pitch), red_r * 0.34, G_BRIGHT)
    if len(white_pts) > 1:
        mids = 0.5 * (white_pts + np.roll(white_pts, -1, axis=0))
        ok = np.linalg.norm(white_pts - np.roll(white_pts, -1, axis=0), axis=1) < 1.6 * WHITE_SPACING
        b.dots(mids[ok], white_r * 0.28, G_BRIGHT)
    img = b.img.filter(ImageFilter.GaussianBlur(0.4))
    rgb = np.asarray(img).astype(np.float32) / 255.0
    tex = img.convert('RGBA')
    nrm = normal_from_tex(rgb, nrm_strength, blur=0.7)
    tex.save(os.path.join(OUT, f'{name}_tex.png'))
    nrm.save(os.path.join(OUT, f'{name}_nrm.png'))
    return tex, nrm


def uv_planar(m, bounds):
    return mu.planar_uv(m, bounds)


def finish(m, uv, mat, name):
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    m.fix_normals()
    assert np.isfinite(m.vertices).all(), name
    assert len(m.faces) and (m.area_faces > 1e-15).all(), f'degenerate faces in {name}'
    return m


# ---------------------------------------------------------------- 1. masks (same recipe as v1)
cut = Image.open(mu.REF_PENDANT_CUTOUT).convert('RGBA')
arr = np.asarray(cut).astype(np.float32)
alpha = arr[:, :, 3] / 255.0
rgb_ref = arr[:, :, :3]
H, W = alpha.shape
solid = alpha > 0.5
lum = 0.299 * rgb_ref[:, :, 0] + 0.587 * rgb_ref[:, :, 1] + 0.114 * rgb_ref[:, :, 2]

rows_any = np.where(solid.any(axis=1))[0]
top = int(rows_any[0])
band_rows = []
for r in range(top, H):
    c = np.where(solid[r])[0]
    if len(c) == 0:
        continue
    if (c[-1] - c[0] + 1) > 0.25 * W:
        break
    band_rows.append(r)
upper = band_rows[:int(len(band_rows) * 0.75)]
bc = np.where(solid[upper].any(axis=0))[0]
band_c0, band_c1 = int(bc.min()) - 14, int(bc.max()) + 14
letters_top = next(r for r in range(top, H) if solid[r, :band_c0].any() or solid[r, band_c1:].any())
pc = np.where(solid[letters_top - 25:letters_top, band_c0:band_c1].any(axis=0))[0] + band_c0
post_c0, post_c1 = int(pc.min()) - 4, int(pc.max()) + 4
inner = slice(post_c0 + 22, post_c1 - 22)
bail_bottom = letters_top + 25
for r in range(letters_top, letters_top + 80):
    if ((lum[r, inner] > 150) & solid[r, inner]).mean() > 0.25:
        bail_bottom = r
        break
window_top = int(top + 0.76 * (letters_top - top))
dark = (rgb_ref.max(axis=2) < 75) & solid
win = np.zeros_like(solid)
win[window_top:bail_bottom + 2, post_c0:post_c1] = dark[window_top:bail_bottom + 2, post_c0:post_c1]
win = ndimage.binary_opening(win, iterations=1)
lab, nlab = ndimage.label(win)
sizes = ndimage.sum(win, lab, index=np.arange(1, nlab + 1))
win = np.isin(lab, np.where(sizes >= 250)[0] + 1)
win = ndimage.binary_closing(win, iterations=3)
win = ndimage.binary_dilation(win, iterations=1)
t_edge = next(r for r in range(letters_top, bail_bottom + 1)
              if solid[r, post_c0 - 12:post_c0 - 2].any() and solid[r, post_c1 + 2:post_c1 + 12].any())
print(f'top={top} band cols=({band_c0},{band_c1}) letters_top={letters_top} posts=({post_c0},{post_c1}) '
      f'bail_bottom={bail_bottom} t_edge={t_edge}')

letters_mask = solid.copy()
letters_mask[:letters_top] = False
letters_mask[:t_edge, post_c0:post_c1] = False
letters_mask &= ~win
letters_mask = ndimage.binary_closing(letters_mask, iterations=2) | letters_mask

# ---------------------------------------------------------------- 2. polygons (pixel units, y up) -> metres
letters_px = mu.mask_to_polygons(letters_mask.astype(np.float32), simplify_px=SIMPLIFY_PX)
lb = letters_px.bounds
scale = LETTER_W / (lb[2] - lb[0])
print(f'letters polygon: {len(geoms(letters_px))} part(s), holes={sum(len(g.interiors) for g in geoms(letters_px))}, '
      f'scale={scale:.3e} m/px')

# frame: ring top at the origin; bail plate hangs from the ring; the T's top edge below the prongs
ring_c_y = -(RING_R + RING_r)
bail_top_y = ring_c_y - RING_EMBED
bail_bot_y = bail_top_y - BAIL_H
t_top_y = bail_bot_y - PRONG_GAP
cx_px = 0.5 * (post_c0 + post_c1)                    # the bail posts are centred on the T
t_edge_up = H - t_edge


def to_world_xy(v_px):
    x = (v_px[:, 0] - cx_px) * scale
    y = t_top_y + (v_px[:, 1] - t_edge_up) * scale
    return np.column_stack([x, y])


LP = affinity.affine_transform(letters_px, [scale, 0, 0, scale, -cx_px * scale, t_top_y - t_edge_up * scale])
print(f'letters world bounds (mm): {np.round(np.array(LP.bounds)*1e3, 1)}')

# ---- interior diamond lines: the reference's white pave lines also run along the INNER letter
# boundaries (A|T|A joins, T stem, A crossbars).  Extract them from the photo and skeletonise.
from skimage.morphology import skeletonize
_sat = rgb_ref.max(axis=2) - rgb_ref.min(axis=2)
_white_px = (lum > 150) & (_sat < 95) & letters_mask
_white_px = ndimage.binary_closing(_white_px, iterations=2)
_white_px = ndimage.binary_opening(_white_px, iterations=1)
_lab, _n = ndimage.label(_white_px)
_sz = ndimage.sum(_white_px, _lab, index=np.arange(1, _n + 1))
_white_px = np.isin(_lab, np.where(_sz >= 120)[0] + 1)
_skel = skeletonize(_white_px)
_sr, _sc = np.where(_skel)
sk_xy = to_world_xy(np.column_stack([_sc.astype(float), (H - _sr).astype(float)]))
print(f'white-line skeleton: {len(sk_xy)} px')

# ---------------------------------------------------------------- 3. letters body
V, F, L = extrude_bevel_labeled(letters_px, DEPTH, BEVEL / scale, BEVEL_H, BEVEL_STEPS, scale,
                                min_area=30.0, eps=0.35)
V[:, :2] = to_world_xy(V[:, :2] / scale)
V[:, 2] -= DEPTH / 2                          # centre the body on z = 0
z_front = DEPTH / 2
print(f'letters extrusion: {len(F)} tris  (front={np.sum(L==3)} tread={np.sum(L==2)} wall={np.sum(L==1)} back={np.sum(L==0)})')

# ---------------------------------------------------------------- 4. letters gems
front_poly = LP.buffer(-BEVEL, join_style=2)
white_xy = ring_points(LP, WHITE_INSET, WHITE_SPACING, min_ring_len=WHITE_SPACING * 3)
white_xy = thin_points(white_xy, 2 * WHITE_R + 0.00012)
# ---- per-letter red fields from the reference photo: each field gets its own diamond ring, so the
# A | T | A joins, the T stem and the crossbars read exactly like the reference.
_r, _g, _b = rgb_ref[:, :, 0], rgb_ref[:, :, 1], rgb_ref[:, :, 2]
_sat = rgb_ref.max(axis=2) - rgb_ref.min(axis=2)
_white_px = (lum > 150) & (_sat < 95) & letters_mask
_white_d = ndimage.binary_closing(ndimage.binary_dilation(_white_px, iterations=2), iterations=4)
_redness = np.clip((_r - np.maximum(_g, _b)) / 255.0, 0, 1) * letters_mask
_red_px = (ndimage.gaussian_filter(_redness, 5) > 0.28) & letters_mask & ~_white_d
_red_px = ndimage.binary_opening(_red_px, iterations=2)
_red_px = ndimage.binary_closing(_red_px, iterations=3)
_lab, _n = ndimage.label(_red_px)
_sz = ndimage.sum(_red_px, _lab, index=np.arange(1, _n + 1))
_keep = np.where(_sz >= 2500)[0] + 1
print(f'red fields: {len(_keep)} components, sizes {sorted(_sz[_keep - 1].astype(int))[::-1]}')
_aff = [scale, 0, 0, scale, -cx_px * scale, t_top_y - t_edge_up * scale]
red_fields = []
_outer_zone = ndimage.binary_erosion(letters_mask, iterations=int(round((WHITE_INSET + WHITE_R + RAIL) / scale)))
for _k in _keep:
    _comp = ndimage.binary_dilation(_lab == _k, iterations=14) & _outer_zone & ~_white_d
    _comp = ndimage.binary_closing(_comp, iterations=2)
    _clab, _cn = ndimage.label(_comp)
    if _cn > 1:  # keep the piece that contains the original field
        _csz = ndimage.sum(_lab == _k, _clab, index=np.arange(1, _cn + 1))
        _comp = _clab == (int(np.argmax(_csz)) + 1)
    _poly = mu.mask_to_polygons(_comp.astype(np.float32), simplify_px=1.0, min_area_px=300)
    _poly = affinity.affine_transform(_poly, _aff).intersection(LP.buffer(-(WHITE_INSET + WHITE_R), join_style=1))
    if not _poly.is_empty:
        red_fields.append(_poly)
from shapely.ops import unary_union
# diamond rows just outside every red field (interior letter boundaries); outer ring already placed
sk_in = []
for _poly in red_fields:
    for _coords in rings(_poly.buffer(WHITE_R + 0.00035, join_style=1), WHITE_SPACING * 3):
        _ls = LineString(_coords)
        _nn = max(3, int(round(_ls.length / (WHITE_SPACING * 0.5))))
        sk_in += [_ls.interpolate(i / _nn, normalized=True).coords[0] for i in range(_nn)]
sk_in = np.array(sk_in) if sk_in else np.zeros((0, 2))
_inner_zone = LP.buffer(-(WHITE_INSET + 2 * WHITE_R + 0.0002), join_style=1)
from shapely import contains_xy
sk_in = sk_in[contains_xy(_inner_zone, sk_in[:, 0], sk_in[:, 1])] if len(sk_in) else sk_in
_white_near = ndimage.binary_dilation(_white_px, iterations=7)
if len(sk_in):
    _cols = np.clip(np.round(sk_in[:, 0] / scale + cx_px).astype(int), 0, W - 1)
    _rows = np.clip(np.round(H - ((sk_in[:, 1] - t_top_y) / scale + t_edge_up)).astype(int), 0, H - 1)
    sk_in = sk_in[_white_near[_rows, _cols]]
    print(f'interior stones on real diamond lines: {len(sk_in)}')
sk_in = reject_near(sk_in, white_xy, 2 * WHITE_R + 0.0003)
sk_in = thin_points(sk_in, 2 * WHITE_R + 0.00012)
print(f'interior diamond line stones: {len(sk_in)}')
white_xy = np.vstack([white_xy, sk_in]) if len(sk_in) else white_xy
existing = [[x, y, WHITE_R] for x, y in white_xy[:len(white_xy) - len(sk_in)]] + [[x, y, WHITE_R + 0.00060] for x, y in sk_in]
# small diamonds for the spike tips the main row could not reach (single contour pass)
white_small = pave_rows(LP, WHITE_INSET - 0.0002, WHITE_SMALL_R, 0.00012, existing, max_rows=1)
existing += white_small.tolist()
inner_edge = WHITE_INSET + WHITE_R + RAIL               # where the red field starts
RED_REGION = LP.buffer(-inner_edge, join_style=1)
red_sets = []
for r_, mode in ((RED_R, 'rows'), (RED_R, 'fill'), (RED_MED_R, 'fill'), (RED_SMALL_R, 'fill')):
    if mode == 'rows':
        new = pave_rows(RED_REGION, r_, r_, RED_GAP, existing, jitter=RED_JITTER)
    else:
        new = fill_random(RED_REGION.buffer(-r_, join_style=1), r_, RED_GAP, existing)
    existing += new.tolist()
    red_sets.append((r_, new))
    print(f'  red {mode} r={r_*1e3:.2f} mm: +{len(new)}')
red_all = np.vstack([n for _, n in red_sets if len(n)])
red_xy = red_all[:, :2]
RED_PITCH = 2 * RED_R + RED_GAP
white_all_xy = np.vstack([white_xy, white_small[:, :2]]) if len(white_small) else white_xy
print(f'white row: {len(white_xy)} + {len(white_small)} small; red pave: {len(red_xy)} stones')

z_gem = z_front + GEM_LIFT
rv_, rf_ = [], []
for r_, pts_ in red_sets:
    if len(pts_) == 0:
        continue
    v_, f_ = gem_cloud(np.column_stack([pts_[:, :2], np.full(len(pts_), z_gem)]), r_, f'letters rubies r={r_*1e3:.2f}mm')
    rf_.append(f_ + sum(len(v) for v in rv_)); rv_.append(v_)
rv_, rf_ = np.vstack(rv_), np.vstack(rf_)
wv_, wf_ = [], []
for r_, pts_ in ((WHITE_R, white_xy), (WHITE_SMALL_R, white_small[:, :2] if len(white_small) else white_small)):
    if len(pts_) == 0:
        continue
    v_, f_ = gem_cloud(np.column_stack([pts_, np.full(len(pts_), z_gem)]), r_, f'letters diamonds r={r_*1e3:.2f}mm')
    wf_.append(f_ + sum(len(v) for v in wv_)); wv_.append(v_)
wv_, wf_ = np.vstack(wv_), np.vstack(wf_)

# ---------------------------------------------------------------- 5. front texture (2048)
lw = LP.bounds
pad = 0.001
uv_bounds = (lw[0] - pad, lw[1] - pad, lw[2] + pad, lw[3] + pad)
# base: reference photo crop (gem colouring) composited over gold and darkened to a bezel field
box_px = ((uv_bounds[0]) / scale + cx_px, H - ((uv_bounds[3] - t_top_y) / scale + t_edge_up),
          (uv_bounds[2]) / scale + cx_px, H - ((uv_bounds[1] - t_top_y) / scale + t_edge_up))
crop = cut.crop(tuple(int(round(v)) for v in box_px)).resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
ca = np.asarray(crop).astype(np.float32) / 255.0
gold = np.array([0.93, 0.72, 0.28], dtype=np.float32)
base = ca[:, :, :3] * ca[:, :, 3:4] + gold * (1.0 - ca[:, :, 3:4])
base = base * 0.25 + np.array([0.30, 0.03, 0.03]) * 0.75      # deep-red bezel field: gaps read as ruby shadow
front_tex, front_nrm = bake('pendant_front', TEX_SIZE, uv_bounds, base, LP, red_xy, white_all_xy,
                            RED_R, WHITE_R, RED_PITCH, WHITE_INSET, BEVEL,
                            WHITE_INSET + WHITE_R + RAIL * 0.6, NRM_STRENGTH)
gold_nrm = brushed_normal()
gold_nrm.save(os.path.join(OUT, 'pendant_gold_nrm.png'))

# ---------------------------------------------------------------- 6. materials
mat_front = mu.pbr([1.0, 1.0, 1.0, 1.0], metallic=0.6, roughness=0.32, texture=front_tex, normal=front_nrm, name='pave_front')
mat_gold = mu.pbr(mu.GOLD, metallic=1.0, roughness=0.18, normal=gold_nrm, name='gold')
mat_gold_back = mu.pbr(GOLD_DARK, metallic=1.0, roughness=0.30, normal=gold_nrm, name='gold_back')
mat_ruby = mu.pbr(RUBY, metallic=0.30, roughness=0.10, name='ruby')
mat_diamond = mu.pbr(DIAMOND, metallic=0.75, roughness=0.05, name='diamond')

letters_front = part_from_faces(V, F, (L == 3) | (L == 2))
finish(letters_front, uv_planar(letters_front, uv_bounds), mat_front, 'letters_front')
letters_walls = part_from_faces(V, F, L == 1, unshare=True)
finish(letters_walls, wall_uv(letters_walls, tile=14.0), mat_gold, 'letters_walls')
letters_back = part_from_faces(V, F, L == 0)
finish(letters_back, uv_planar(letters_back, uv_bounds), mat_gold_back, 'letters_back')
assert (letters_front.face_normals[:, 2] > 0.5).all(), 'front cap normals must face +Z'
assert (letters_back.face_normals[:, 2] < -0.9).all(), 'back cap normals must face -Z'

rubies = finish(trimesh.Trimesh(rv_, rf_, process=False), uv_planar(trimesh.Trimesh(rv_, rf_, process=False), uv_bounds), mat_ruby, 'rubies')
diamonds = finish(trimesh.Trimesh(wv_, wf_, process=False), uv_planar(trimesh.Trimesh(wv_, wf_, process=False), uv_bounds), mat_diamond, 'diamonds')

# ---------------------------------------------------------------- 7. bail: trapezoid pave plate
bail_poly = Polygon([(-BAIL_TOP_W / 2, bail_top_y), (BAIL_TOP_W / 2, bail_top_y),
                     (BAIL_BOT_W / 2, bail_bot_y), (-BAIL_BOT_W / 2, bail_bot_y)])
Vb, Fb, Lb = extrude_bevel_labeled(bail_poly, BAIL_DEPTH, BAIL_BEVEL, BAIL_BEVEL_H, 1, 1.0,
                                   min_area=1e-9, eps=2e-5)
Vb[:, 2] -= BAIL_DEPTH / 2
zb_front = BAIL_DEPTH / 2
# gems: white row along the two slanted edges + the top edge (open at the bottom), red hex fill
bw_pts = []
for a, b_ in [((-BAIL_TOP_W / 2, bail_top_y), (-BAIL_BOT_W / 2, bail_bot_y)),
              ((BAIL_TOP_W / 2, bail_top_y), (BAIL_BOT_W / 2, bail_bot_y)),
              ((-BAIL_TOP_W / 2, bail_top_y), (BAIL_TOP_W / 2, bail_top_y))]:
    pass
bail_in = bail_poly.buffer(-BAIL_WHITE_INSET, join_style=2)
ex = np.array(bail_in.exterior.coords)
# walk the inset rectangle but skip its bottom edge
segs = []
for i in range(len(ex) - 1):
    p, q = ex[i], ex[i + 1]
    if abs(p[1] - q[1]) < 1e-6 and abs(p[1] - (bail_bot_y + BAIL_WHITE_INSET)) < 1e-4:
        continue
    segs.append((p, q))
bwhite = []
for p, q in segs:
    ls = LineString([p, q])
    n = max(1, int(round(ls.length / BAIL_WHITE_SPACING)))
    bwhite += [ls.interpolate(i / n, normalized=True).coords[0] for i in range(n + 1)]
bwhite = thin_points(np.array(bwhite), 2 * BAIL_WHITE_R + 0.00010)
bexist = [[x, y, BAIL_WHITE_R] for x, y in bwhite]
bail_open = Polygon([(-BAIL_TOP_W / 2, bail_top_y), (BAIL_TOP_W / 2, bail_top_y),
                     (BAIL_BOT_W / 2 + 0.0, bail_bot_y - 0.004), (-BAIL_BOT_W / 2 - 0.0, bail_bot_y - 0.004)])
bail_open = bail_open.intersection(Polygon([(-1, bail_bot_y + BAIL_RED_R * 0.9), (1, bail_bot_y + BAIL_RED_R * 0.9), (1, 1), (-1, 1)]))
bred_big = pave_rows(bail_open, BAIL_WHITE_INSET + BAIL_WHITE_R + 0.0002 + BAIL_RED_R, BAIL_RED_R, 0.00008, bexist, jitter=0.00004)
bexist += bred_big.tolist()
bred_fill = fill_random(bail_open.buffer(-(BAIL_WHITE_INSET + BAIL_WHITE_R + 0.0002 + BAIL_RED_R)), BAIL_RED_R, 0.00008, bexist, 6000)
bexist += bred_fill.tolist()
bred_big = np.vstack([bred_big, bred_fill]) if len(bred_fill) else bred_big
bred_small = fill_random(bail_open.buffer(-(BAIL_WHITE_INSET + BAIL_WHITE_R + 0.0002 + BAIL_SMALL_R)), BAIL_SMALL_R, 0.00008, bexist, 6000)
bred_all = np.vstack([bred_big, bred_small]) if len(bred_small) else bred_big
bred = bred_all[:, :2]
BAIL_RED_PITCH = 2 * BAIL_RED_R + 0.00008
bail_red_c = np.column_stack([bred, np.full(len(bred), zb_front + GEM_LIFT)])
bail_white_c = np.column_stack([bwhite, np.full(len(bwhite), zb_front + GEM_LIFT)])
brv, brf = [], []
for r_, pts_ in ((BAIL_RED_R, bred_big), (BAIL_SMALL_R, bred_small)):
    if len(pts_) == 0:
        continue
    v_, f_ = gem_cloud(np.column_stack([pts_[:, :2], np.full(len(pts_), zb_front + GEM_LIFT)]), r_, f'bail rubies r={r_*1e3:.2f}mm')
    brf.append(f_ + sum(len(v) for v in brv)); brv.append(v_)
brv, brf = np.vstack(brv), np.vstack(brf)
bwv, bwf = gem_cloud(bail_white_c, BAIL_WHITE_R, 'bail diamonds')

bb = bail_poly.bounds
bpad = 0.0006
buv = (bb[0] - bpad, bb[1] - bpad, bb[2] + bpad, bb[3] + bpad)
bbase = np.tile(np.array([0.30, 0.04, 0.03], dtype=np.float32), (512, 512, 1))
bail_tex, bail_nrm = bake('pendant_bail', 512, buv, bbase, bail_poly, bred, bwhite,
                          BAIL_RED_R, BAIL_WHITE_R, BAIL_RED_PITCH, BAIL_WHITE_INSET, BAIL_BEVEL,
                          BAIL_WHITE_INSET + BAIL_WHITE_R + 0.00012, 1.6)
mat_bail = mu.pbr([1.0, 1.0, 1.0, 1.0], metallic=0.6, roughness=0.32, texture=bail_tex, normal=bail_nrm, name='pave_bail')
bail_front = part_from_faces(Vb, Fb, (Lb == 3) | (Lb == 2))
finish(bail_front, uv_planar(bail_front, buv), mat_bail, 'bail_front')
bail_body = part_from_faces(Vb, Fb, (Lb == 1) | (Lb == 0), unshare=True)
finish(bail_body, wall_uv(bail_body, tile=6.0), mat_gold, 'bail_body')
bail_rubies = finish(trimesh.Trimesh(brv, brf, process=False), uv_planar(trimesh.Trimesh(brv, brf, process=False), buv), mat_ruby, 'bail_rubies')
bail_diamonds = finish(trimesh.Trimesh(bwv, bwf, process=False), uv_planar(trimesh.Trimesh(bwv, bwf, process=False), buv), mat_diamond, 'bail_diamonds')
print(f'bail plate: {len(Fb)} tris, top y={bail_top_y*1e3:.2f} mm, bottom y={bail_bot_y*1e3:.2f} mm')

# ---------------------------------------------------------------- 8. jump ring (axis along X, top at origin) + prongs
rv, rf, ruv = mu.torus_uv(RING_R, RING_r, seg_u=48, seg_v=14)
rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])     # ring axis Z -> X
rv = trimesh.transform_points(rv, rot)
rv[:, 1] += ring_c_y
rv[:, 1] -= rv[:, 1].max()               # snap the ring's top exactly onto the origin
ring = mu.textured_mesh(rv, rf, ruv, mat_gold)
assert abs(ring.bounds[1, 1]) < 1e-6, 'ring top must be at the origin'
assert abs(rv[:, 1].min() + 2 * (RING_R + RING_r)) < 2e-4

prongs = []
for sx in (-1, 1):
    x = sx * (BAIL_BOT_W / 2 - PRONG_W / 2 - 0.0004)
    y0, y1 = t_top_y - 0.0012, bail_bot_y + 0.0008
    b = trimesh.creation.box(extents=[PRONG_W, y1 - y0, PRONG_D])
    b.apply_translation([x, 0.5 * (y0 + y1), 0.0])
    b.unmerge_vertices()
    prongs.append(b)
prong = trimesh.util.concatenate(prongs)
finish(prong, wall_uv(prong, tile=2.0), mat_gold, 'prongs')

# ---------------------------------------------------------------- 9. assemble + export
parts = {
    'letters_front': letters_front, 'letters_walls': letters_walls, 'letters_back': letters_back,
    'rubies': rubies, 'diamonds': diamonds,
    'bail_front': bail_front, 'bail_body': bail_body, 'bail_rubies': bail_rubies, 'bail_diamonds': bail_diamonds,
    'bail_ring': ring, 'bail_prongs': prong,
}
scene = trimesh.Scene()
for name, m in parts.items():
    assert np.isfinite(m.vertices).all(), name
    assert m.visual.uv is not None and len(m.visual.uv) == len(m.vertices), name
    assert np.isfinite(m.visual.uv).all(), name
    assert len(m.faces) and (m.area_faces > 1e-15).all(), f'degenerate faces in {name}'
    m.vertex_normals  # cache so they are exported
    scene.add_geometry(m, node_name=name, geom_name=name)
tris = mu.report(scene, 'pendant')
assert tris <= 75000, 'over budget'
out = os.path.join(OUT, 'pendant.glb')
scene.export(out)
with open(os.path.join(OUT, 'pendant_meta.json'), 'w') as fh:
    json.dump({'ring_centre_local': [0.0, ring_c_y, 0.0], 'ring_major_r': RING_R, 'ring_tube_r': RING_r,
               'tris': int(tris), 'red_gems': int(len(red_xy) + len(bred)), 'white_gems': int(len(white_all_xy) + len(bwhite))}, fh, indent=1)
print('wrote', out, os.path.getsize(out) // 1024, 'kB')
