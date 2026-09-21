#!/usr/bin/env python3
"""build_chain.py - ATA Miami-cuban necklace chain as a textured 3D model (trimesh).  v2.

Chain frame (see mesh_utils.py): metres, Y up, wearer faces +Z.
The lowest link centre sits at (0, 0, +0.035) on the chest, the curve rises over the shoulders at
(+/-0.085, 0.13, 0) and the ends meet behind the neck at (0, 0.175, -0.06), where a small box
clasp replaces one link.

v2 link (matches ref/chain_target.jpg): a CHUNKY FLAT SLAB - an elliptical centreline (7.9 x 5.7 mm)
swept with a wide rounded-rectangle cross-section (radial half-width 3.7 mm, thickness 1.9 mm,
rounded top edges).  Link outline 23.2 x 18.8 mm, hole 8.4 x 4.0 mm.  Links are laid flat in the
band plane (outward-facing), LEANED 18 deg in-plane like the target's ovals, and SHINGLED: every
link is rotated 14 deg about the band-width axis so its trailing end rides over the previous link
and its leading end slides under the next one - the overlap is 61 % of the link length, giving the
interlocking "(" stripes of the target with a uniform 0.28 mm air gap at every crossing (no
interpenetration).

Texture (1024 x 1024, four UV islands):
  * top face  - planar projection of the link; ONE row of 17 large faceted RED gems (2.2 mm) on the
                centreline flanked by TWO rows of small WHITE diamonds (1.3 mm; 38 outer / 21 inner),
                bead-set with gold prongs, polished gold bevel around the edge
  * outer / inner walls - plain polished gold
  * back face - brushed gold
  chain_link_nrm.png : tangent-space normal map baked from a height field (domed faceted crowns,
                       recessed seats, prong beads)
  chain_link_mr.png  : metallic (B) / roughness (G)

Outputs (in chain3d/out): chain.glb (chain_links + clasp), chain_link_tex.png, chain_link_nrm.png,
chain_link_mr.png.
"""
import os
import sys
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from mesh_utils import pbr, textured_mesh, report, OUT  # noqa: E402

# ----------------------------------------------------------------------------- parameters (metres)
A_C = 0.0079            # centreline ellipse semi-axis along the chain (X)
B_C = 0.0057            # centreline ellipse semi-axis across the band (Y); B^2/A = 4.1 mm > W (valid offset)
W = 0.0037              # profile radial half-width  (link is 7.4 mm wide across the band)
H = 0.00095             # profile half-thickness     (1.9 mm slab)
RC_TOP = 0.0008         # top edge rounding radius
CB = 0.0004             # bottom edge chamfer
SPACING = 0.0090        # link pitch along the curve
SHINGLE_DEG = 14.0      # rotation about the band-width axis (trailing end up, leading end down)
LEAN_DEG = 18.0         # in-plane lean of the oval (long axis tilted from the chain direction)
SEG_U = 42              # segments around the oval (adaptive to the budget)
CLASP_SIZE = (0.0150, 0.0094, 0.0030)   # along T, B, N
TRI_BUDGET = 70000
TEX = 1024
SS = 2                                  # texture supersampling
GOLD = np.array([1.00, 0.70, 0.24], np.float32)
RENDER_DIR = os.path.join(os.path.dirname(HERE), 'renders')

LINK_LEN = 2 * (A_C + W)                # 23.2 mm
LINK_WID = 2 * (B_C + W)                # 18.8 mm
HOLE = (2 * (A_C - W), 2 * (B_C - W))   # 8.4 x 4.0 mm


def _ellipse_table(n=4096):
    th = np.linspace(0, 2 * np.pi, n + 1)
    P = np.column_stack([A_C * np.cos(th), B_C * np.sin(th)])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return th, np.concatenate([[0.0], np.cumsum(seg)])


_TH, _ARC = _ellipse_table()
S_TOTAL = float(_ARC[-1])               # centreline length ~43.0 mm
MARGIN = 0.0004                         # texture margin around the planar islands
BBOX_X = LINK_LEN + 2 * MARGIN
BBOX_Y = LINK_WID + 2 * MARGIN

# texture row layout (row 0 = top of image = v 1)
ROWS_TOP = int(round(TEX * BBOX_Y / BBOX_X))          # planar top face, isotropic
_r = ROWS_TOP + 6
ROWS_OUTER = (_r, _r + 40); _r += 46
ROWS_INNER = (_r, _r + 40); _r += 46
ROWS_BOTTOM = (_r, TEX - 6)
assert ROWS_BOTTOM[1] - ROWS_BOTTOM[0] > 40, ROWS_BOTTOM


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


# ----------------------------------------------------------------------------- link centreline
def centreline(s):
    """Ellipse centreline by arc length s (array, metres), CCW from the +X end. Returns C (n,2),
    T (n,2), Rd (n,2) (point, unit tangent, unit outward normal in the link's XY plane)."""
    s = np.asarray(s, float) % S_TOTAL
    th = np.interp(s, _ARC, _TH)
    C = np.column_stack([A_C * np.cos(th), B_C * np.sin(th)])
    T = np.column_stack([-A_C * np.sin(th), B_C * np.cos(th)])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Rd = np.column_stack([T[:, 1], -T[:, 0]])
    return C, T, Rd


def u_samples(seg_u=SEG_U):
    return np.linspace(0, S_TOTAL, seg_u + 1)               # seg_u + 1 values, last == S_TOTAL


def row_positions(d, n_stones, phase=0.0):
    """n equally spaced points along the offset curve at radial offset d (m) from the centreline
    -> (xy (n,2) metres, tangent angle (n,))."""
    ss = np.linspace(0, S_TOTAL, 8001)
    C, T, Rd = centreline(ss)
    P = C + d * Rd
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    targets = ((np.arange(n_stones) + phase) * L[-1] / n_stones) % L[-1]
    x = np.interp(targets, L, P[:, 0]); y = np.interp(targets, L, P[:, 1])
    ang = np.interp(targets, L, np.unwrap(np.arctan2(T[:, 1], T[:, 0])))
    return np.column_stack([x, y]), ang


# ----------------------------------------------------------------------------- profile / mesh
def profile_islands():
    """Cross-section (r, z, nr, nz) in metres, grouped into 4 UV islands.  r is the outward
    radial offset from the centreline, z the thickness axis (+z = visible face)."""
    w, h, rc, cb = W, H, RC_TOP, CB
    top = []
    for ang in np.linspace(180, 90, 4):                  # inner rounded top edge
        a = np.radians(ang); top.append((-(w - rc) + rc * np.cos(a), (h - rc) + rc * np.sin(a), np.cos(a), np.sin(a)))
    for ang in np.linspace(90, 0, 4):                    # outer rounded top edge
        a = np.radians(ang); top.append(((w - rc) + rc * np.cos(a), (h - rc) + rc * np.sin(a), np.cos(a), np.sin(a)))
    # 8 points: inner arc (4) + outer arc (4); the flat top is the single segment between them
    d = np.sqrt(0.5)
    outer = [(w, h - rc, 1, 0), (w, -h + cb, 1, 0), (w - cb, -h, d, -d)]
    bottom = [(w - cb, -h, 0, -1), (-(w - cb), -h, 0, -1)]
    inner = [(-(w - cb), -h, -d, -d), (-w, -h + cb, -1, 0), (-w, h - rc, -1, 0)]
    return dict(top=top, outer=outer, bottom=bottom, inner=inner)


def island_v_range(rows):
    """Texture rows (r0, r1) -> (v0, v1) with v measured upward (v = 1 - row/TEX)."""
    return 1 - rows[1] / TEX, 1 - rows[0] / TEX


def link_template(seg_u=SEG_U):
    """One flat link (local: long axis X, visible face +Z, centred at the origin).
    Returns verts (m), faces, uv, analytic vertex normals."""
    s = u_samples(seg_u)
    C, T, Rd = centreline(s)
    isl = profile_islands()
    V, N, UV, F = [], [], [], []
    off = 0
    for name, pts in isl.items():
        pts = np.array(pts, float)
        nu, nv = len(s), len(pts)
        verts = C[:, None, :] * 0
        verts = np.zeros((nu, nv, 3)); nrm = np.zeros((nu, nv, 3)); uv = np.zeros((nu, nv, 2))
        for j, (r, z, nr, nz) in enumerate(pts):
            verts[:, j, :2] = C + r * Rd; verts[:, j, 2] = z
            nrm[:, j, :2] = nr * Rd; nrm[:, j, 2] = nz
        if name in ('top', 'bottom'):
            rows = (0, ROWS_TOP) if name == 'top' else ROWS_BOTTOM
            v0, v1 = island_v_range(rows)
            uv[..., 0] = (verts[..., 0] + BBOX_X / 2) / BBOX_X
            uv[..., 1] = v0 + (verts[..., 1] + BBOX_Y / 2) / BBOX_Y * (v1 - v0)
        else:
            rows = ROWS_OUTER if name == 'outer' else ROWS_INNER
            v0, v1 = island_v_range(rows)
            t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
            t /= t[-1]
            uv[..., 0] = (s / S_TOTAL)[:, None]
            uv[..., 1] = v0 + (1 - t)[None, :] * (v1 - v0)
        faces = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = off + i * nv + j; b = a + nv
                faces.append([a, b, a + 1]); faces.append([b, b + 1, a + 1])
        V.append(verts.reshape(-1, 3)); N.append(nrm.reshape(-1, 3)); UV.append(uv.reshape(-1, 2)); F.append(np.array(faces))
        off += nu * nv
    V = np.vstack(V); N = np.vstack(N); UV = np.vstack(UV); F = np.vstack(F)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    # winding: face normal must agree with the analytic normal
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    agree = np.einsum('ij,ij->i', fn, N[F].mean(axis=1)) > 0
    F[~agree] = F[~agree][:, ::-1]
    return V, F, UV, N


def rounded_box(lx, ly, lz, rc=0.0015, bevel=0.0006, n_corner=6):
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


# ----------------------------------------------------------------------------- texture painting
class Canvas:
    def __init__(self, ppm):
        n = TEX * SS
        self.ppm = ppm                                        # px per mm (supersampled)
        self.col = np.empty((n, n, 3), np.float32); self.col[:] = GOLD
        self.h = np.zeros((n, n), np.float32)
        self.rough = np.full((n, n), 0.18, np.float32)
        self.metal = np.ones((n, n), np.float32)


def mm_to_px_top(xy_m):
    """Link-local XY (metres) -> supersampled canvas pixel coords in the top island (y down)."""
    ppm = TEX * SS / (BBOX_X * 1e3)
    x = (xy_m[..., 0] + BBOX_X / 2) * 1e3 * ppm
    y = (BBOX_Y / 2 - xy_m[..., 1]) * 1e3 * ppm
    return x, y


def paint_stone(cv, cx, cy, R, kind, rot, rng):
    """Faceted round brilliant (8 kite facets + octagonal table) with bead-set seat.
    cx, cy: canvas px; R: stone radius (mm); rot: facet rotation (rad)."""
    ppm = cv.ppm
    pad = int(R * ppm + 0.30 * ppm) + 2
    x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad + 1, cv.col.shape[1])
    y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad + 1, cv.col.shape[0])
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    dx = (X - cx) / ppm; dy = -(Y - cy) / ppm
    r = np.hypot(dx, dy); th = np.arctan2(dy, dx) - rot
    rn = r / R
    K = 8
    dth = (th + np.pi / K) % (2 * np.pi / K) - np.pi / K
    fk = np.floor((th + np.pi / K) / (2 * np.pi / K)).astype(int) % K
    rho = r * np.cos(dth)
    r_t = 0.50 * R
    h_g, h_t = -0.06, 0.05 + 0.23 * R
    crown = np.clip(1 - (rho - r_t) / (R * 0.93 - r_t), 0, 1)
    h_stone = h_g + (h_t - h_g) * crown
    inside = rn < 1.0
    seat = (~inside) & (r < R + 0.13)
    ramp = (r >= R + 0.13) & (r < R + 0.22)

    # per-facet shading from a fixed key light + per-stone random facet brightness
    beta = np.arctan((h_t - h_g) / (R * 0.93 - r_t))
    thk = fk * 2 * np.pi / K + rot
    nk = np.stack([np.sin(beta) * np.cos(thk), np.sin(beta) * np.sin(thk), np.full_like(thk, np.cos(beta))], -1)
    Ld = np.array([-0.45, 0.55, 0.70]); Ld /= np.linalg.norm(Ld)
    lam = np.clip(nk @ Ld, 0, 1)
    rnd = rng.random(K)[fk]
    if kind == 'red':
        f = np.clip(0.10 + 0.80 * lam + 0.45 * (rnd - 0.5), 0, 1)
        dark, light = np.array([0.22, 0.0, 0.02]), np.array([0.96, 0.10, 0.12])
        table_c = np.array([0.78, 0.02, 0.06]); edge_c = np.array([1.0, 0.62, 0.62])
        rim_c = np.array([0.14, 0.0, 0.01])
        rough_v, metal_v = 0.06, 0.0
    else:
        f = np.clip(0.05 + 1.10 * lam + 0.65 * (rnd - 0.5), 0, 1)
        dark, light = np.array([0.30, 0.30, 0.34]), np.array([1.0, 1.0, 0.98])
        table_c = np.array([0.97, 0.96, 0.92]); edge_c = np.array([1.0, 1.0, 1.0])
        rim_c = np.array([0.26, 0.25, 0.26])
        rough_v, metal_v = 0.06, 0.45
    col = dark[None, None] + (light - dark)[None, None] * f[..., None]
    table = rho < r_t
    grad = np.clip(0.5 + 0.5 * (dx * Ld[0] + dy * Ld[1]) / r_t, 0, 1)
    tcol = table_c[None, None] * (0.80 + 0.30 * grad[..., None]) + 0.10 * grad[..., None] * (edge_c - table_c)[None, None]
    col = np.where(table[..., None], tcol, col)
    # facet edge lines (star) + table outline
    d_edge = r * (np.pi / K - np.abs(dth))
    line = np.exp(-(d_edge / 0.022) ** 2) * (~table)
    tline = np.exp(-((rho - r_t) / 0.028) ** 2)
    e = np.clip(0.55 * line + 0.45 * tline, 0, 1)[..., None]
    col = col + e * (edge_c[None, None] - col)
    # girdle / rim darkening
    gd = smoothstep((rn - 0.82) / 0.18)
    col = col * (1 - 0.75 * gd[..., None]) + rim_c[None, None] * 0.75 * gd[..., None]
    # crisp specular dots
    sx, sy = -0.30 * R, 0.30 * R
    g1 = np.exp(-((dx - sx) ** 2 + (dy - sy) ** 2) / (2 * (0.085 * R) ** 2))
    g2 = 0.45 * np.exp(-((dx + 0.22 * R) ** 2 + (dy + 0.30 * R) ** 2) / (2 * (0.06 * R) ** 2))
    g = np.clip(g1 + g2, 0, 1)[..., None]
    col = col + g * (1 - col)
    col = np.clip(col, 0, 1)

    sub_c = cv.col[y0:y1, x0:x1]; sub_h = cv.h[y0:y1, x0:x1]
    sub_r = cv.rough[y0:y1, x0:x1]; sub_m = cv.metal[y0:y1, x0:x1]
    seat_c = np.array([0.42, 0.27, 0.09], np.float32)
    sub_c[seat] = seat_c; sub_h[seat] = -0.14; sub_r[seat] = 0.55; sub_m[seat] = 1.0
    rr = (r[ramp] - (R + 0.13)) / 0.09
    sub_c[ramp] = (GOLD * (0.72 + 0.28 * rr)[:, None]); sub_h[ramp] = -0.14 * (1 - rr); sub_r[ramp] = 0.35
    sub_c[inside] = col[inside]; sub_h[inside] = h_stone[inside]; sub_r[inside] = rough_v; sub_m[inside] = metal_v


def paint_bead(cv, cx, cy, R=0.16):
    ppm = cv.ppm
    pad = int(R * ppm) + 2
    x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad + 1, cv.col.shape[1])
    y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad + 1, cv.col.shape[0])
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    r = np.hypot(X - cx, Y - cy) / ppm
    m = r < R
    dome = np.sqrt(np.clip(1 - (r / R) ** 2, 0, 1))
    sub_h = cv.h[y0:y1, x0:x1]; sub_c = cv.col[y0:y1, x0:x1]
    sub_h[m] = np.maximum(sub_h[m], 0.16 * dome[m])
    sub_c[m] = np.clip(GOLD * (0.85 + 0.35 * dome[m])[:, None], 0, 1)
    cv.rough[y0:y1, x0:x1][m] = 0.16; cv.metal[y0:y1, x0:x1][m] = 1.0


def stone_layout():
    """Rows: (radial offset m, stone radius mm, kind, count, phase)."""
    r_red, r_wht = 1.10, 0.65
    n_red = int(round(S_TOTAL * 1e3 / (2 * r_red + 0.40)))
    d_w = 0.00205
    L_out = S_TOTAL + 2 * np.pi * d_w; L_in = S_TOTAL - 2 * np.pi * d_w
    n_out = int(round(L_out * 1e3 / (2 * r_wht + 0.22)))
    n_in = int(round(L_in * 1e3 / (2 * r_wht + 0.22)))
    return [(0.0, r_red, 'red', n_red, 0.0), (d_w, r_wht, 'white', n_out, 0.5), (-d_w, r_wht, 'white', n_in, 0.5)]


def build_textures(seed=11):
    rng = np.random.default_rng(seed)
    n = TEX * SS
    ppm = n / (BBOX_X * 1e3)
    cv = Canvas(ppm)
    # --- subtle polished-gold variation on everything, brushed streaks on the back island
    noise = gaussian_filter(rng.standard_normal((n, n)).astype(np.float32), 6.0)
    noise /= noise.std()
    cv.col *= (1 + 0.012 * noise)[..., None]
    rb = (ROWS_BOTTOM[0] * SS, ROWS_BOTTOM[1] * SS)
    streak = gaussian_filter(rng.standard_normal((rb[1] - rb[0], n)).astype(np.float32), sigma=(0.6, 25))
    streak /= streak.std()
    cv.col[rb[0]:rb[1]] = np.clip(GOLD * 0.92 * (1 + 0.07 * streak)[..., None], 0, 1)
    cv.rough[rb[0]:rb[1]] = 0.40
    for rows in (ROWS_OUTER, ROWS_INNER):                # walls: polished, slightly darker toward the bottom edge
        r0, r1 = rows[0] * SS, rows[1] * SS
        t = np.linspace(0, 1, r1 - r0)[:, None, None]
        cv.col[r0:r1] = np.clip(GOLD * (1.02 - 0.18 * t), 0, 1)
        cv.rough[r0:r1] = 0.15
    # --- top face: gentle darkening of the gold field between the stones (pave bed)
    top_rows = ROWS_TOP * SS
    Xg, Yg = np.meshgrid(np.arange(n) + 0.5, np.arange(top_rows) + 0.5)
    xm = Xg / ppm - BBOX_X * 1e3 / 2; ym = BBOX_Y * 1e3 / 2 - Yg / ppm
    # distance from the centreline (mm) via a KD-tree on a dense centreline sampling
    from scipy.spatial import cKDTree
    Cd, _, _ = centreline(np.linspace(0, S_TOTAL, 3000, endpoint=False))
    dist = cKDTree(Cd * 1e3).query(np.column_stack([xm.ravel(), ym.ravel()]))[0].reshape(xm.shape)
    bed = smoothstep((2.95 - dist) / 0.25)
    cv.col[:top_rows] *= (1 - 0.10 * bed)[..., None]
    cv.rough[:top_rows] = 0.18 + 0.12 * bed
    # --- stones (red row last so its seats win over the whites where they touch)
    rows = stone_layout()
    for d, R, kind, count, phase in rows[1:] + rows[:1]:
        P, ang = row_positions(d, count, phase)
        px, py = mm_to_px_top(P)
        for k in range(count):
            paint_stone(cv, px[k], py[k], R, kind, rng.uniform(0, 2 * np.pi), rng)
    # --- prong beads between consecutive stones of every row (both sides of the row)
    for d, R, kind, count, phase in rows:
        P, ang = row_positions(d, count, phase + 0.5)
        px, py = mm_to_px_top(P)
        for k in range(count):
            nx, ny = -np.sin(ang[k]), np.cos(ang[k])     # row normal (link plane), y up
            for sgn in (-1, 1):
                off = sgn * (R + 0.12) * ppm
                paint_bead(cv, px[k] + nx * off, py[k] - ny * off, 0.15 if kind == 'red' else 0.12)

    # --- downsample to TEX and bake maps
    def down(a):
        if a.ndim == 3:
            return a.reshape(TEX, SS, TEX, SS, 3).mean(axis=(1, 3))
        return a.reshape(TEX, SS, TEX, SS).mean(axis=(1, 3))
    col = np.clip(down(cv.col), 0, 1)
    h = down(cv.h)
    rough = np.clip(down(cv.rough), 0.03, 1)
    metal = np.clip(down(cv.metal), 0, 1)
    mm_px = BBOX_X * 1e3 / TEX
    gx = np.gradient(h, axis=1) / mm_px
    gy = np.gradient(h, axis=0) / mm_px
    k = 1.35
    nx, ny, nz = -gx * k, gy * k, np.ones_like(h)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], axis=-1) * 0.5 + 0.5
    base_img = Image.fromarray((col * 255).round().astype(np.uint8), 'RGB')
    nrm_img = Image.fromarray((nrm * 255).round().astype(np.uint8), 'RGB')
    mr = np.stack([np.zeros_like(rough), rough, metal], axis=-1)
    mr_img = Image.fromarray((mr * 255).round().astype(np.uint8), 'RGB')
    return base_img, nrm_img, mr_img


# ----------------------------------------------------------------------------- chain assembly
def make_material(textures):
    base_img, nrm_img, mr_img = textures
    mat = pbr(color=[1.0, 1.0, 1.0, 1.0], metallic=1.0, roughness=1.0, texture=base_img, normal=nrm_img, name='ata_link_pave')
    mat.metallicRoughnessTexture = mr_img
    return mat


def link_rotation(R, lean_deg=LEAN_DEG):
    """Frame R (cols T, B, N) -> in-plane lean of the oval (about local Z), then the shingle
    rotation about the local Y (band-width) axis: the trailing (-T) end rises along +N, the
    leading (+T) end sinks under the next link.  All link planes stay parallel, so the gap
    between neighbours is pitch * sin(SHINGLE) - thickness for every crossing."""
    return R @ rot_axis(np.array([0.0, 1.0, 0.0]), np.radians(SHINGLE_DEG)) @ rot_axis(np.array([0.0, 0.0, 1.0]), np.radians(lean_deg))


def link_lean(i, n):
    """Lean per link: full lean everywhere, ramped to 0 at the lowest link so the pendant bail
    threads a straight hole."""
    k = min(i, n - i)
    return LEAN_DEG * min(1.0, k / 3.0)


def build_chain(seg_u=SEG_U):
    cs, t_end = necklace_spline()
    C, T, L, n = sample_links(cs, t_end)
    n_links = n - 1
    nprof = 12                                                  # profile segments (see profile_islands)
    while n_links * seg_u * nprof * 2 > TRI_BUDGET - 900 and seg_u > 24:
        seg_u -= 1
    Vt, Ft, UVt, Nt = link_template(seg_u)
    textures = build_textures()
    mat = make_material(textures)

    back = n // 2
    vs, fs, uvs, ns, frames = [], [], [], [], []
    off = 0
    for i in range(n):
        R = link_frame(T[i], outward_dir(C[i]))
        frames.append(R)
        if i == back:
            continue
        Rt = link_rotation(R, link_lean(i, n))
        vs.append(Vt @ Rt.T + C[i]); fs.append(Ft + off); uvs.append(UVt); ns.append(Nt @ Rt.T)
        off += len(Vt)
    verts = np.vstack(vs); faces = np.vstack(fs); normals = np.vstack(ns)
    chain = trimesh.Trimesh(verts, faces, process=False)
    chain.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uvs), material=mat)
    chain.fix_normals(multibody=True)
    # the islands are open surfaces: make sure fix_normals did not invert any of them
    agree = np.einsum('ij,ij->i', chain.face_normals, normals[chain.faces].mean(axis=1)) > 0
    if (~agree).any():
        f = chain.faces.copy(); f[~agree] = f[~agree][:, ::-1]
        chain = trimesh.Trimesh(verts, f, process=False)
        chain.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uvs), material=mat)
    chain.vertex_normals = normals

    # clasp: rounded gold box at the back-of-neck point, long axis along T, flat face outward,
    # textured from the polished-wall island
    bv, bf = rounded_box(*CLASP_SIZE)
    Rb = frames[back]
    bv_w = bv @ Rb.T + C[back]
    bmin, bmax = bv.min(axis=0), bv.max(axis=0)
    v0, v1 = island_v_range(ROWS_OUTER)
    buv = np.column_stack([(bv[:, 0] - bmin[0]) / (bmax[0] - bmin[0]),
                           v0 + 0.15 * (v1 - v0) + 0.7 * (v1 - v0) * (bv[:, 1] - bmin[1]) / (bmax[1] - bmin[1])])
    clasp = textured_mesh(bv_w, bf, buv, mat)
    return chain, clasp, dict(C=C, T=T, L=L, n=n, seg_u=seg_u, frames=frames, back=back,
                              textures=textures, Vt=Vt, Ft=Ft, UVt=UVt, Nt=Nt)


def sanity(mesh, name):
    assert np.isfinite(mesh.vertices).all(), f'{name}: NaN vertices'
    areas = mesh.area_faces
    print(f'  {name}: verts={len(mesh.vertices)} tris={len(mesh.faces)} degenerate={int((areas < 1e-14).sum())} '
          f'winding_consistent={mesh.is_winding_consistent}')
    assert int((areas < 1e-14).sum()) == 0


def clearance_report(info):
    """Minimum gap between overlapping neighbours in the flat template configuration."""
    Vt = info['Vt']
    p = info['L'] / info['n']
    phi = np.radians(SHINGLE_DEG)
    gap = p * np.sin(phi) - 2 * H
    print(f'shingle: pitch {p*1e3:.2f} mm, plane separation {p*np.sin(phi)*1e3:.2f} mm, slab {2*H*1e3:.2f} mm -> air gap {gap*1e3:.2f} mm; '
          f'tip lift {(LINK_LEN/2)*np.sin(phi)*1e3:.2f} mm; band {LINK_WID*1e3:.1f} x {LINK_LEN*1e3:.1f} mm link, '
          f'hole {HOLE[0]*1e3:.1f} x {HOLE[1]*1e3:.1f} mm, lean {LEAN_DEG:.0f} deg, overlap {(1 - p/LINK_LEN)*100:.0f} %')
    assert gap > 0.0001, 'links interpenetrate'


def main():
    os.makedirs(OUT, exist_ok=True)
    chain, clasp, info = build_chain()
    print(f'path length={info["L"]:.4f} m, links={info["n"]} (one replaced by clasp), pitch={info["L"] / info["n"]:.5f} m, seg_u={info["seg_u"]}')
    clearance_report(info)
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
    V = chain.vertices
    d_axis = np.hypot(V[:, 0], V[:, 2] + 0.03)
    at_back = V[:, 1] > 0.165
    print('lowest link centre', np.round(info['C'][0], 4), ' back point', np.round(info['C'][info['back']], 4),
          f' nape: min dist to body axis {d_axis[at_back].min()*1e3:.1f} mm (v1 was 25.1 mm), max z {V[at_back, 2].max()*1e3:.1f} mm')
    return tris


if __name__ == '__main__':
    main()
