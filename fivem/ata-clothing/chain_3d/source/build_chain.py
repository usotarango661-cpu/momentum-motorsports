#!/usr/bin/env python3
"""build_chain.py - ATA cuban-link necklace chain as a textured 3D model (trimesh).

Chain frame (see mesh_utils.py): metres, Y up, wearer faces +Z.
Per the task spec the lowest link centre sits at (0, 0, +0.035) on the chest, the curve rises
over the shoulders at (+/-0.085, 0.13, 0) and the ends meet behind the neck at (0, 0.175, -0.06),
where a small box clasp replaces one link.

Outputs (in chain3d/out):
  chain.glb              - scene with 'chain_links' (all links, one mesh) + 'clasp'
  chain_link_tex.png     - base colour texture (pave on the +Z half, brushed gold on the back)
  chain_link_nrm.png     - tangent-space normal map (gems as domes, bezels recessed)
  chain_link_mr.png      - metallic(B)/roughness(G) texture
"""
import os
import sys
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import map_coordinates, gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mesh_utils import pbr, torus_uv, textured_mesh, report, OUT  # noqa: E402

# ----------------------------------------------------------------------------- parameters
MAJOR_R, MINOR_R = 0.0075, 0.0028          # link oval (m)
SCALE = (1.0, 0.72, 0.5)                   # oval aspect + flattening in local Z
SEG_U, SEG_V = 40, 12                      # link tessellation (960 tris)
SPACING = 0.0105                           # link pitch along the curve (m)
TILT_DEG = 38.0                            # alternating twist about the tangent
CLASP_SIZE = (0.012, 0.006, 0.005)         # along T, B, N
TRI_BUDGET = 60000
TEX_W, TEX_H = 1024, 512                   # u (around the oval) x v (around the tube)
GOLD_RGB = np.array([1.0, 0.78, 0.32])
RENDER_DIR = os.path.join(os.path.dirname(HERE), 'renders')


# ----------------------------------------------------------------------------- path
def necklace_spline():
    """Closed periodic cubic spline through mirror-symmetric control points."""
    right = [(0.000, 0.000, 0.035),   # front chest, lowest link centre
             (0.040, 0.035, 0.030),
             (0.070, 0.085, 0.016),
             (0.085, 0.130, 0.000),   # over the shoulder
             (0.070, 0.162, -0.035),
             (0.000, 0.175, -0.060)]  # back of neck (clasp)
    left = [(-x, y, z) for (x, y, z) in right[-2:0:-1]]
    pts = np.array(right + left + [right[0]], dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    return CubicSpline(t, pts, bc_type='periodic'), t[-1]


def sample_links(cs, t_end, spacing=SPACING, n_dense=6000):
    """Evenly spaced (by arc length) link centres + unit tangents. n is forced even so that
    link 0 is the lowest link (s=0) and link n/2 sits exactly at the back-of-neck point."""
    tt = np.linspace(0.0, t_end, n_dense, endpoint=False)
    P = cs(tt)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    n = int(round(L / spacing))
    n += n % 2
    targets = np.arange(n) * L / n
    t_at = np.interp(targets, s, np.concatenate([tt, [t_end]]))
    C = cs(t_at)
    T = cs(t_at, 1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return C, T, L, n


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def outward_dir(p):
    """Direction away from the body: radial from a vertical body axis at z=-0.03, blended with
    'up' near the shoulders where the chain drapes over the trapezius (less so at the nape)."""
    x, y, z = p
    radial = np.array([x, 0.0, z + 0.03])
    radial /= np.linalg.norm(radial)
    k = 0.45 * smoothstep((y - 0.05) / 0.08) * (1 - 0.5 * smoothstep((y - 0.15) / 0.025))
    o = radial + np.array([0.0, k, 0.0])
    return o / np.linalg.norm(o)


def link_frame(T, o):
    """Right-handed rotation whose columns map local X->T, local Y->B, local Z->N (outward)."""
    N = o - np.dot(o, T) * T
    N /= np.linalg.norm(N)
    B = np.cross(N, T)
    return np.column_stack([T, B, N])


def rot_axis(axis, ang):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


# ----------------------------------------------------------------------------- texture
def ring_arclength():
    a, b = MAJOR_R * SCALE[0] * 1e3, MAJOR_R * SCALE[1] * 1e3
    U = np.linspace(0, 2 * np.pi, 4001)
    s = cumulative_trapezoid(np.sqrt((a * np.sin(U)) ** 2 + (b * np.cos(U)) ** 2), U, initial=0)
    return U, s


def tube_arclength():
    rr = MINOR_R * 1e3 * 0.86          # mean radial semi-axis of the flattened tube (mm)
    rz = MINOR_R * SCALE[2] * 1e3
    V = np.linspace(0, 2 * np.pi, 2001)
    t = cumulative_trapezoid(np.sqrt((rr * np.cos(V)) ** 2 + (rz * np.sin(V)) ** 2), V, initial=0)
    return V, t


def paint_row(cv, y_row, n_stones, r_mm, kind, phase, res):
    """Paint one row of round pave stones (periodic along x) into canvas dict cv (in place)."""
    col, hgt, rough, metal = cv['col'], cv['h'], cv['rough'], cv['metal']
    CH, CW = hgt.shape
    X, Y = np.meshgrid(np.arange(CW) + 0.5, np.arange(CH) + 0.5)
    pitch = CW / n_stones
    lx = ((X - phase * pitch) % pitch) - pitch / 2
    ly = Y - y_row
    d = np.hypot(lx, ly) / res
    th = np.arctan2(ly, lx)
    rn = d / r_mm
    inside = rn < 1.0
    bezel = (rn >= 1.0) & (d < r_mm + 0.13)

    if kind == 'red':
        dark, light, table_c = np.array([0.20, 0.0, 0.02]), np.array([0.92, 0.06, 0.10]), np.array([0.74, 0.02, 0.06])
        rough_v, metal_v = 0.30, 0.15
    else:
        dark, light, table_c = np.array([0.48, 0.52, 0.62]), np.array([1.0, 1.0, 1.0]), np.array([0.90, 0.93, 0.98])
        rough_v, metal_v = 0.20, 1.0

    facet = 0.6 * (0.5 + 0.5 * np.cos(8 * th + 0.3)) + 0.4 * (0.5 + 0.5 * np.cos(4 * th - 1.0))
    shade = 0.40 + 0.60 * facet
    shade *= 1 - np.clip((rn - 0.78) / 0.22, 0, 1) * 0.55          # girdle darkening
    gem = dark[None, None, :] * (1 - shade[..., None]) + light[None, None, :] * shade[..., None]
    table = (rn < 0.42)[..., None]
    gem = np.where(table, table_c[None, None, :] * (0.85 + 0.15 * facet[..., None]), gem)
    # specular glints
    ga = 0.85 if kind == 'red' else 1.0
    g1 = ga * np.exp(-(((lx / res + 0.30 * r_mm) ** 2 + (ly / res + 0.30 * r_mm) ** 2) / (2 * (0.13 * r_mm) ** 2)))
    g2 = 0.5 * np.exp(-(((lx / res - 0.25 * r_mm) ** 2 + (ly / res - 0.28 * r_mm) ** 2) / (2 * (0.08 * r_mm) ** 2)))
    g = np.clip(g1 + g2, 0, 1)[..., None]
    gem = gem + (1 - gem) * g

    col[inside] = gem[inside]
    bez_c = np.array([0.42, 0.28, 0.09])
    col[bezel] = bez_c
    dome = np.sqrt(np.clip(1 - rn ** 2, 0, 1))
    hg = 0.42 * np.minimum(dome, 0.80) / 0.80
    hgt[inside] = hg[inside]
    hgt[bezel] = -0.06
    rough[inside] = rough_v
    metal[inside] = metal_v
    rough[bezel] = 0.9
    # pave beads in the gaps between stones (both sides of the row)
    lxb = ((X - (phase + 0.5) * pitch) % pitch) - pitch / 2
    for sgn in (-1, 1):
        db = np.hypot(lxb, ly - sgn * 0.55 * r_mm * res) / res
        rb = 0.17
        bead = db < rb
        bd = np.sqrt(np.clip(1 - (db / rb) ** 2, 0, 1))
        col[bead] = np.array([1.0, 0.92, 0.60])
        hgt[bead] = np.maximum(hgt[bead], 0.14 * bd[bead])
        rough[bead] = 0.8
        metal[bead] = 1.0


def build_textures():
    U, s = ring_arclength()
    V, t = tube_arclength()
    S, Tp = s[-1], t[-1]
    res = 2.0 * TEX_W / S                  # canvas px/mm (2x supersampled vs. texture)
    CW = int(round(S * res))
    CH = int(round(Tp * res))
    rng = np.random.default_rng(7)
    streak = gaussian_filter(rng.standard_normal((CH, CW)), sigma=(0.8, 20), mode='wrap')
    streak /= streak.std()
    Yc = (np.arange(CH) + 0.5) / res - Tp / 2          # t (mm) from the top centre, -Tp/2..Tp/2
    under = 0.86 + 0.14 * (1 - smoothstep((np.abs(Yc) - 2.9) / 1.6))
    col = GOLD_RGB[None, None, :] * (1 + 0.05 * streak)[..., None] * under[:, None, None]
    col = col.astype(np.float64)
    cv = dict(col=col, h=np.zeros((CH, CW)), rough=np.ones((CH, CW)), metal=np.ones((CH, CW)))
    # polished raised rim bordering the pave field
    rim = np.exp(-((np.abs(Yc) - 2.40) ** 2) / (2 * 0.10 ** 2))
    cv['h'] += 0.12 * rim[:, None]
    cv['col'] = cv['col'] * (1 + 0.10 * rim[:, None, None])
    cv['col'] = np.clip(cv['col'], 0, 1)
    yc = CH / 2
    paint_row(cv, yc + 1.58 * res, 32, 0.56, 'white', 0.0, res)
    paint_row(cv, yc - 1.58 * res, 32, 0.56, 'white', 0.0, res)
    paint_row(cv, yc, 24, 0.72, 'red', 0.5, res)

    # slight blur (antialias) then warp canvas -> (u, v) texture
    for k in ('col', 'h', 'rough', 'metal'):
        sig = (0.9, 0.9, 0) if k == 'col' else 0.9
        cv[k] = gaussian_filter(cv[k], sigma=sig, mode='wrap')
    i = np.arange(TEX_W) + 0.5
    j = np.arange(TEX_H) + 0.5
    u = i / TEX_W
    v = 1.0 - j / TEX_H                       # trimesh convention: image row 0 <-> v = 1
    s_u = np.interp(u * 2 * np.pi, U, s)
    t_v = np.interp(v * 2 * np.pi, V, t)
    xs = s_u * res - 0.5
    ys = ((t_v + Tp / 2) % Tp) * res - 0.5
    XX, YY = np.meshgrid(xs, ys)
    coords = [YY.ravel(), XX.ravel()]

    def warp(a):
        if a.ndim == 3:
            return np.stack([map_coordinates(a[..., c], coords, order=1, mode='grid-wrap').reshape(TEX_H, TEX_W)
                             for c in range(a.shape[2])], axis=-1)
        return map_coordinates(a, coords, order=1, mode='grid-wrap').reshape(TEX_H, TEX_W)

    col_t = np.clip(warp(cv['col']), 0, 1)
    h_t = warp(cv['h'])
    rough_t = np.clip(warp(cv['rough']), 0, 1)
    metal_t = np.clip(warp(cv['metal']), 0, 1)

    # tangent-space normal map (glTF: +X = +u, +Y = up in the image)
    mm_per_col = S / TEX_W
    mm_per_row = Tp / TEX_H
    gcol = np.gradient(h_t, axis=1) / mm_per_col
    grow = np.gradient(h_t, axis=0) / mm_per_row
    k = 0.9
    nx, ny, nz = -gcol * k, grow * k, np.ones_like(h_t)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5

    base_img = Image.fromarray((col_t * 255).round().astype(np.uint8), 'RGB')
    nrm_img = Image.fromarray((nrm * 255).round().astype(np.uint8), 'RGB')
    mr = np.stack([np.zeros_like(rough_t), rough_t, metal_t], axis=-1)
    mr_img = Image.fromarray((mr * 255).round().astype(np.uint8), 'RGB')
    return base_img, nrm_img, mr_img


# ----------------------------------------------------------------------------- geometry
def link_template(seg_u=SEG_U, seg_v=SEG_V):
    """One flat oval link (local: long axis X, flat face +Z) + analytic outward normals."""
    Vt, Ft, UVt = torus_uv(MAJOR_R, MINOR_R, seg_u=seg_u, seg_v=seg_v, scale_xyz=SCALE)
    us = np.linspace(0, 2 * np.pi, seg_u + 1)
    vs = np.linspace(0, 2 * np.pi, seg_v + 1)
    U, V = np.meshgrid(us, vs, indexing='ij')
    n = np.column_stack([(np.sin(V) * np.cos(U)).ravel(), (np.sin(V) * np.sin(U)).ravel(), np.cos(V).ravel()])
    n = n / np.array(SCALE)                         # normals transform by the inverse scale
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return Vt, Ft, UVt, n


def rounded_box(lx, ly, lz, rc=0.0012, bevel=0.0009, n_corner=6):
    """Closed rounded box centred at the origin (bevelled top/bottom edges)."""
    def outline(inset):
        hx, hy = lx / 2 - inset, ly / 2 - inset
        r = max(rc - inset, 1e-5)
        pts = []
        for (cx, cy, a0) in [(hx - r, hy - r, 0.0), (-hx + r, hy - r, np.pi / 2),
                             (-hx + r, -hy + r, np.pi), (hx - r, -hy + r, 1.5 * np.pi)]:
            for k in range(n_corner + 1):
                a = a0 + k * (np.pi / 2) / n_corner
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        return np.array(pts)
    levels = [(-lz / 2, bevel), (-lz / 2 + 0.3 * bevel, 0.3 * bevel), (-lz / 2 + bevel, 0.0),
              (lz / 2 - bevel, 0.0), (lz / 2 - 0.3 * bevel, 0.3 * bevel), (lz / 2, bevel)]
    rings = []
    for z, ins in levels:
        o = outline(ins)
        rings.append(np.column_stack([o, np.full(len(o), z)]))
    m = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    for L in range(len(rings) - 1):
        for k in range(m):
            a, b = L * m + k, L * m + (k + 1) % m
            c, d = (L + 1) * m + k, (L + 1) * m + (k + 1) % m
            faces += [[a, b, d], [a, d, c]]
    cb = len(verts)
    verts = np.vstack([verts, [[0, 0, -lz / 2]], [[0, 0, lz / 2]]])
    ct = cb + 1
    top0 = (len(rings) - 1) * m
    for k in range(m):
        a, b = k, (k + 1) % m
        faces.append([cb, b, a])
        faces.append([ct, top0 + a, top0 + b])
    return verts, np.array(faces)


def build_chain(seg_u=SEG_U, seg_v=SEG_V):
    cs, t_end = necklace_spline()
    C, T, L, n = sample_links(cs, t_end)
    # keep within budget: drop u-segments if the path is longer than expected
    while n * seg_u * seg_v * 2 > TRI_BUDGET - 1500 and seg_u > 28:
        seg_u -= 2
    Vt, Ft, UVt, Nt = link_template(seg_u, seg_v)

    base_img, nrm_img, mr_img = build_textures()
    mat = pbr(color=[1.0, 1.0, 1.0, 1.0], metallic=0.85, roughness=0.25, texture=base_img, normal=nrm_img, name='ata_link_pave')
    mat.metallicRoughnessTexture = mr_img

    # template mesh: fix winding, then verify faces agree with analytic outward normals
    tmpl = textured_mesh(Vt, Ft, UVt, mat)
    fc = tmpl.triangles.mean(axis=1)
    fn_expect = Nt[tmpl.faces].mean(axis=1)
    if np.mean(np.einsum('ij,ij->i', tmpl.face_normals, fn_expect) > 0) < 0.5:
        tmpl.invert()
    Ft = tmpl.faces.copy()

    back = n // 2
    tilt = np.radians(TILT_DEG)
    vs, fs, uvs, ns = [], [], [], []
    frames = []
    off = 0
    for i in range(n):
        R = link_frame(T[i], outward_dir(C[i]))
        frames.append(R)
        if i == back:
            continue
        Rt = rot_axis(T[i], tilt if i % 2 == 0 else -tilt) @ R
        vs.append(Vt @ Rt.T + C[i])
        fs.append(Ft + off)
        uvs.append(UVt)
        ns.append(Nt @ Rt.T)
        off += len(Vt)
    verts = np.vstack(vs)
    faces = np.vstack(fs)
    chain = trimesh.Trimesh(verts, faces, process=False)
    chain.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uvs), material=mat)
    chain.fix_normals(multibody=True)
    chain.vertex_normals = np.vstack(ns)

    # clasp: rounded gold box at the back-of-neck point, long axis along T, flat face outward
    bv, bf = rounded_box(*CLASP_SIZE)
    Rb = frames[back]
    bv_w = bv @ Rb.T + C[back]
    bmin, bmax = bv.min(axis=0), bv.max(axis=0)
    buv = np.column_stack([(bv[:, 0] - bmin[0]) / (bmax[0] - bmin[0]),
                           0.42 + 0.16 * (bv[:, 1] - bmin[1]) / (bmax[1] - bmin[1])])
    clasp = textured_mesh(bv_w, bf, buv, mat)

    return chain, clasp, dict(C=C, T=T, L=L, n=n, seg_u=seg_u, frames=frames, back=back,
                              textures=(base_img, nrm_img, mr_img))


def sanity(mesh, name):
    assert np.isfinite(mesh.vertices).all(), f'{name}: NaN vertices'
    areas = mesh.area_faces
    print(f'  {name}: verts={len(mesh.vertices)} tris={len(mesh.faces)} degenerate={int((areas < 1e-14).sum())} '
          f'winding_consistent={mesh.is_winding_consistent} volume={mesh.volume:.3e}')


def main():
    os.makedirs(OUT, exist_ok=True)
    chain, clasp, info = build_chain()
    print(f'path length={info["L"]:.4f} m, links={info["n"]} (one replaced by clasp), pitch={info["L"] / info["n"]:.5f} m, seg_u={info["seg_u"]}')
    sanity(chain, 'chain_links')
    sanity(clasp, 'clasp')
    scene = trimesh.Scene()
    scene.add_geometry(chain, node_name='chain_links', geom_name='chain_links')
    scene.add_geometry(clasp, node_name='clasp', geom_name='clasp')
    tris = report(scene, 'chain')
    assert tris <= TRI_BUDGET, f'over budget: {tris}'
    glb = os.path.join(OUT, 'chain.glb')
    scene.export(glb)
    base_img, nrm_img, mr_img = info['textures']
    base_img.save(os.path.join(OUT, 'chain_link_tex.png'))
    nrm_img.save(os.path.join(OUT, 'chain_link_nrm.png'))
    mr_img.save(os.path.join(OUT, 'chain_link_mr.png'))
    print('wrote', glb, os.path.getsize(glb) // 1024, 'KB')
    lowest = info['C'][0]
    print('lowest link centre', np.round(lowest, 4), ' back point', np.round(info['C'][info['back']], 4))
    return tris


if __name__ == '__main__':
    main()
