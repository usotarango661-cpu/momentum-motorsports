#!/usr/bin/env python3
"""Build the jewelled ATA pendant (letters + bail + jump ring + diamond border) as a GLB.

Pendant frame (see mesh_utils): metres, Y up, front = +Z, top of the bail ring at the origin,
pendant hangs in -Y.  Run from scratchpad/chain3d:  python3 gen/build_pendant.py
"""
import os
import sys
import math

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.prepared import prep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mesh_utils as mu  # noqa: E402

OUT = mu.OUT
RENDERS = os.path.join(os.path.dirname(OUT), 'renders')
os.makedirs(OUT, exist_ok=True)
os.makedirs(RENDERS, exist_ok=True)

# ---------------------------------------------------------------- parameters
LETTER_W = 0.098          # letters width (m)
DEPTH = 0.007             # letters thickness
BEVEL = 0.0012            # chamfer inset on the front
BEVEL_STEPS = 2
BAIL_DEPTH = 0.004
BAIL_BEVEL = 0.0005
RING_R, RING_r = 0.006, 0.0015   # jump ring (major, minor)
RING_OVERLAP = 0.0035     # how far the ring centre sits above the bail plate's top edge
N_DIAMONDS = 150
DIAMOND_R = 0.0008
DIAMOND_SPACING = 0.0036
SIMPLIFY_PX = 1.0
TEX_SIZE = 1024
NRM_STRENGTH = 4.0

GOLD_DARK = [0.62, 0.46, 0.18, 1.0]


# ---------------------------------------------------------------- helpers
def geoms(p):
    """List of shapely Polygons from a Polygon/MultiPolygon/GeometryCollection."""
    if p.is_empty:
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


def extrude_bevel_labeled(poly, depth, bevel, steps, scale, min_area_px=30.0):
    """Same construction as mesh_utils.extrude_with_bevel (70 % body slab + `steps` inset slabs
    on the +Z side, pixel units * scale) but MultiPolygon-safe, with fully hidden interior faces
    removed, and returning per-face labels: 0=back cap, 1=wall, 2=tread (chamfer step), 3=front.
    Z runs from 0 to depth."""
    if bevel <= 0 or steps <= 0:
        levels = [(poly, depth)]
    else:
        body = depth * 0.7
        step = (depth - body) / steps
        levels = [(poly, body)]
        for i in range(1, steps + 1):
            inset = poly.buffer(-bevel * i / steps / scale, join_style=2)
            inset = MultiPolygon([g for g in geoms(inset) if g.area >= min_area_px])
            if inset.is_empty:
                break
            levels.append((inset, step))
    V, F, L = [], [], []
    z0 = 0.0
    nv = 0
    for k, (shape, h) in enumerate(levels):
        nxt = prep(levels[k + 1][0].buffer(-0.35)) if k + 1 < len(levels) else None  # px units
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
                tri = verts[m.faces[:, :]]                            # (F,3,3)
                for fi in np.where(up)[0]:
                    t = Polygon(tri[fi, :, :2])
                    if nxt.contains(t):
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
    """Trimesh for the selected faces (own vertex set)."""
    m = trimesh.Trimesh(V, F[sel], process=False)
    m.remove_unreferenced_vertices()
    if unshare:
        m.unmerge_vertices()
    return m


def wall_uv(m):
    """Non-degenerate UVs for vertical walls: u = normalised X+Y sweep, v = normalised Z."""
    v = m.vertices
    b = m.bounds
    u = ((v[:, 0] - b[0, 0]) / max(b[1, 0] - b[0, 0], 1e-9) + (v[:, 1] - b[0, 1]) / max(b[1, 1] - b[0, 1], 1e-9)) * 0.5
    w = (v[:, 2] - b[0, 2]) / max(b[1, 2] - b[0, 2], 1e-9)
    return np.column_stack([u, w])


def make_textures(cutout, box, size, name):
    """Crop the RGBA cutout to `box` (x0,y0,x1,y1 px), resample to size x size, composite over
    gold and derive a tangent-space normal map from the luminance (gems = bright bumps)."""
    crop = cutout.crop(box).resize((size, size), Image.LANCZOS)
    a = np.asarray(crop).astype(np.float32) / 255.0
    gold = np.array([0.93, 0.72, 0.28], dtype=np.float32)
    rgb = a[:, :, :3] * a[:, :, 3:4] + gold * (1.0 - a[:, :, 3:4])
    tex = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 'RGB').convert('RGBA')
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    hgt = ndimage.gaussian_filter(lum, 0.8)
    gx = ndimage.sobel(hgt, axis=1) / 8.0      # d/dcol
    gy = ndimage.sobel(hgt, axis=0) / 8.0      # d/drow (down)
    nx = -gx * NRM_STRENGTH
    ny = gy * NRM_STRENGTH                     # +Y = up in the image (OpenGL / glTF convention)
    nz = np.ones_like(nx)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    nrm_img = Image.fromarray((np.clip(nrm, 0, 1) * 255).astype(np.uint8), 'RGB')
    tex.save(os.path.join(OUT, f'{name}_tex.png'))
    nrm_img.save(os.path.join(OUT, f'{name}_nrm.png'))
    return tex, nrm_img


def sphere_uv(m):
    v = m.vertices - m.vertices.mean(axis=0)
    r = np.linalg.norm(v, axis=1) + 1e-12
    u = (np.arctan2(v[:, 2], v[:, 0]) / (2 * np.pi)) % 1.0
    w = np.arccos(np.clip(v[:, 1] / r, -1, 1)) / np.pi
    return np.column_stack([u, 1 - w])


def sample_ring(coords, spacing):
    ls = LineString(coords)
    n = max(1, int(ls.length / spacing))
    return [ls.interpolate(i / n, normalized=True).coords[0] for i in range(n)]


# ---------------------------------------------------------------- 1. masks
cut = Image.open(mu.REF_PENDANT_CUTOUT).convert('RGBA')
arr = np.asarray(cut).astype(np.float32)
alpha = arr[:, :, 3] / 255.0
rgb = arr[:, :, :3]
H, W = alpha.shape
solid = alpha > 0.5
lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

rows_any = np.where(solid.any(axis=1))[0]
top = int(rows_any[0])
# bail band: scan down while the silhouette is one narrow, centred component
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
# letters' top edge = first row with solid pixels outside the bail band
letters_top = next(r for r in range(top, H) if solid[r, :band_c0].any() or solid[r, band_c1:].any())
# posts of the bail loop just above the letters
pc = np.where(solid[letters_top - 25:letters_top, band_c0:band_c1].any(axis=0))[0] + band_c0
post_c0, post_c1 = int(pc.min()) - 4, int(pc.max()) + 4
# T-top border: first row at/after letters_top whose band interior is bright (white diamonds)
inner = slice(post_c0 + 22, post_c1 - 22)
bail_bottom = letters_top + 25
for r in range(letters_top, letters_top + 80):
    if ((lum[r, inner] > 150) & solid[r, inner]).mean() > 0.25:
        bail_bottom = r
        break
# dark window of the bail loop (the see-through part between the posts)
window_top = int(top + 0.76 * (letters_top - top))
dark = (rgb.max(axis=2) < 75) & solid
win = np.zeros_like(solid)
win[window_top:bail_bottom + 2, post_c0:post_c1] = dark[window_top:bail_bottom + 2, post_c0:post_c1]
win = ndimage.binary_opening(win, iterations=1)
lab, nlab = ndimage.label(win)
sizes = ndimage.sum(win, lab, index=np.arange(1, nlab + 1))
win = np.isin(lab, np.where(sizes >= 250)[0] + 1)          # drop gem-shadow specks, keep the slot
win = ndimage.binary_closing(win, iterations=3)
win = ndimage.binary_dilation(win, iterations=1)
# T's top edge just outside the posts (so the T outline stays continuous under the bail)
t_edge = next(r for r in range(letters_top, bail_bottom + 1)
              if solid[r, post_c0 - 12:post_c0 - 2].any() and solid[r, post_c1 + 2:post_c1 + 12].any())
print(f'top={top} band cols=({band_c0},{band_c1}) letters_top={letters_top} posts=({post_c0},{post_c1}) '
      f'bail_bottom={bail_bottom} t_edge={t_edge} window_top={window_top} window px={win.sum()}')

letters_mask = solid.copy()
letters_mask[:letters_top] = False
letters_mask[:t_edge, post_c0:post_c1] = False
letters_mask &= ~win
bail_mask = np.zeros_like(solid)
bail_mask[:t_edge + 10, band_c0:band_c1] = solid[:t_edge + 10, band_c0:band_c1]
bail_mask &= ~win

dbg = np.zeros((H, W, 3), dtype=np.uint8)
dbg[letters_mask] = (220, 40, 40)
dbg[bail_mask] = (60, 90, 240)
dbg[win] = (40, 200, 60)
Image.fromarray(dbg).crop((0, 60, W, 400)).save(os.path.join(RENDERS, 'dbg_masks.png'))

# ---------------------------------------------------------------- 2. polygons (pixel units, y up)
letters_poly = mu.mask_to_polygons(letters_mask.astype(np.float32), simplify_px=SIMPLIFY_PX)
bail_poly = mu.mask_to_polygons(bail_mask.astype(np.float32), simplify_px=SIMPLIFY_PX, min_area_px=200)
lb = letters_poly.bounds
scale = LETTER_W / (lb[2] - lb[0])
print(f'letters polygon: {len(geoms(letters_poly))} part(s), holes={sum(len(g.interiors) for g in geoms(letters_poly))}, '
      f'bounds px={np.round(lb,1)}, scale={scale:.3e} m/px')

# frame: x centred on the bail, ring top at the origin
bb = bail_poly.bounds
cx_px = 0.5 * (bb[0] + bb[2])
ring_c_y = -(RING_R + RING_r)
bail_top_y = ring_c_y - RING_OVERLAP          # metres, y of the bail plate's top edge
bail_top_px = bb[3]                            # y-up pixel of the bail plate's top edge


def to_world(v_px_xy):
    """pixel (x, y-up) -> metres in the pendant frame (XY only)."""
    x = (v_px_xy[:, 0] - cx_px) * scale
    y = bail_top_y + (v_px_xy[:, 1] - bail_top_px) * scale
    return np.column_stack([x, y])


# ---------------------------------------------------------------- 3. letters body
V, F, L = extrude_bevel_labeled(letters_poly, DEPTH, BEVEL, BEVEL_STEPS, scale)
V[:, :2] = to_world(V[:, :2] / scale)
V[:, 2] -= DEPTH / 2                          # centre the body on z = 0
print(f'letters extrusion: {len(F)} tris  (front={np.sum(L==3)} tread={np.sum(L==2)} wall={np.sum(L==1)} back={np.sum(L==0)})')

lw = letters_poly.bounds
box = (int(math.floor(lw[0])), int(H - math.ceil(lw[3])), int(math.ceil(lw[2])), int(H - math.floor(lw[1])))
front_tex, front_nrm = make_textures(cut, box, TEX_SIZE, 'pendant_front')
uv_bounds = to_world(np.array([[box[0], H - box[3]], [box[2], H - box[1]]], dtype=float))
uv_bounds = (uv_bounds[0, 0], uv_bounds[0, 1], uv_bounds[1, 0], uv_bounds[1, 1])

mat_front = mu.pbr([1.0, 1.0, 1.0, 1.0], metallic=0.3, roughness=0.15, texture=front_tex, normal=front_nrm, name='pave_front')
mat_gold = mu.pbr(mu.GOLD, metallic=1.0, roughness=0.2, name='gold')
mat_gold_back = mu.pbr(GOLD_DARK, metallic=1.0, roughness=0.32, name='gold_back')
mat_diamond = mu.pbr([1.0, 1.0, 1.0, 1.0], metallic=0.1, roughness=0.05, name='diamond')

letters_front = part_from_faces(V, F, (L == 3) | (L == 2))
letters_front.visual = trimesh.visual.TextureVisuals(uv=mu.planar_uv(letters_front, uv_bounds), material=mat_front)
letters_walls = part_from_faces(V, F, L == 1, unshare=True)
letters_walls.visual = trimesh.visual.TextureVisuals(uv=wall_uv(letters_walls), material=mat_gold)
letters_back = part_from_faces(V, F, L == 0)
letters_back.visual = trimesh.visual.TextureVisuals(uv=mu.planar_uv(letters_back, uv_bounds), material=mat_gold_back)
assert (letters_front.face_normals[:, 2] > 0.9).all(), 'front cap normals must face +Z'
assert (letters_back.face_normals[:, 2] < -0.9).all(), 'back cap normals must face -Z'

# ---------------------------------------------------------------- 4. bail plate
Vb, Fb, Lb = extrude_bevel_labeled(bail_poly, BAIL_DEPTH, BAIL_BEVEL, 1, scale, min_area_px=10)
Vb[:, :2] = to_world(Vb[:, :2] / scale)
Vb[:, 2] -= BAIL_DEPTH / 2
bw = bail_poly.bounds
bbox = (int(math.floor(bw[0])), int(H - math.ceil(bw[3])), int(math.ceil(bw[2])), int(H - math.floor(bw[1])))
bail_tex, bail_nrm = make_textures(cut, bbox, 512, 'pendant_bail')
buv = to_world(np.array([[bbox[0], H - bbox[3]], [bbox[2], H - bbox[1]]], dtype=float))
buv = (buv[0, 0], buv[0, 1], buv[1, 0], buv[1, 1])
mat_bail = mu.pbr([1.0, 1.0, 1.0, 1.0], metallic=0.3, roughness=0.15, texture=bail_tex, normal=bail_nrm, name='pave_bail')
bail_front = part_from_faces(Vb, Fb, (Lb == 3) | (Lb == 2))
bail_front.visual = trimesh.visual.TextureVisuals(uv=mu.planar_uv(bail_front, buv), material=mat_bail)
bail_body = part_from_faces(Vb, Fb, (Lb == 1) | (Lb == 0), unshare=True)
bail_body.visual = trimesh.visual.TextureVisuals(uv=wall_uv(bail_body), material=mat_gold)
print(f'bail: {len(Fb)} tris, holes={sum(len(g.interiors) for g in geoms(bail_poly))}, top y={bail_top_y:.4f}')

# ---------------------------------------------------------------- 5. jump ring (axis along X, top at origin)
rv, rf, ruv = mu.torus_uv(RING_R, RING_r, seg_u=56, seg_v=16)
rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])     # ring axis Z -> X
rv = trimesh.transform_points(rv, rot)
rv[:, 1] += ring_c_y
rv[:, 1] -= rv[:, 1].max()               # snap the ring's top exactly onto the origin
ring = mu.textured_mesh(rv, rf, ruv, mat_gold)
assert abs(ring.bounds[1, 1]) < 1e-6, 'ring top must be at the origin'

# ---------------------------------------------------------------- 6. diamond border (icospheres along the outline)
exts = []
for g in geoms(letters_poly.buffer(-DIAMOND_R * 1.0 / scale, join_style=1)):
    exts.append(np.array(g.exterior.coords))
    exts += [np.array(i.coords) for i in g.interiors if i.length > 40]
total_len = sum(LineString(e).length for e in exts) * scale
spacing = max(DIAMOND_SPACING, total_len / N_DIAMONDS)
pts = []
for e in exts:
    pts += sample_ring(e, spacing / scale)
pts = to_world(np.array(pts))
z_diamond = -DEPTH / 2 + DEPTH * 0.7 + (DEPTH * 0.3 / BEVEL_STEPS) + DIAMOND_R * 0.2    # set into the 2nd tread
ico = trimesh.creation.icosphere(subdivisions=1, radius=DIAMOND_R)
dv, df, duv = [], [], []
for i, (x, y) in enumerate(pts):
    dv.append(ico.vertices + [x, y, z_diamond])
    df.append(ico.faces + i * len(ico.vertices))
    duv.append(sphere_uv(ico))
diamonds = mu.textured_mesh(np.vstack(dv), np.vstack(df), np.vstack(duv), mat_diamond)
print(f'diamonds: {len(pts)} spheres, spacing {spacing*1000:.1f} mm, {len(diamonds.faces)} tris')

# ---------------------------------------------------------------- 7. assemble + export
parts = {
    'letters_front': letters_front, 'letters_walls': letters_walls, 'letters_back': letters_back,
    'bail_front': bail_front, 'bail_body': bail_body, 'bail_ring': ring, 'diamond_border': diamonds,
}
scene = trimesh.Scene()
for name, m in parts.items():
    assert np.isfinite(m.vertices).all(), name
    assert len(m.faces) and (m.area_faces > 1e-14).all(), f'degenerate faces in {name}'
    m.vertex_normals  # cache so they are exported
    scene.add_geometry(m, node_name=name, geom_name=name)
tris = mu.report(scene, 'pendant')
assert tris <= 45000, 'over budget'
out = os.path.join(OUT, 'pendant.glb')
scene.export(out)
print('wrote', out, os.path.getsize(out) // 1024, 'kB')
