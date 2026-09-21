"""pendant_chain element generator for the ATA clothing kit.

Outputs (all in OUT_DIR):
  pendant.png          1024x1024 RGBA cutout of the jewelled ATA pendant + diamond bail
  chain_strip.png      1024 x STRIP_H RGBA, horizontally tileable cuban-link chain
  chain_2x.png         strip repeated twice on black_fabric (tile check)
  pendant_on_black.png pendant composited on black_fabric (review)

Method
------
* Key the photo-style reference against the black velvet with a luminance OR
  chroma threshold (absolute chroma, not relative saturation - near-black pixels
  have meaningless relative saturation), morphological close/open, drop small
  components, fill pin-holes, soft 1 px edge from the continuous key, feather,
  and un-premultiply the dark fringe against the velvet colour.
* Pendant: keep the key only below the chain (y >= 415) plus a hand-measured
  bail trapezoid + connector block, take the largest connected component,
  scale to 92 % of the canvas width and centre.
* Chain: fit a cubic centreline to the left chain arm (PCA axis + per-bin
  midpoints), reparametrise by arc length, sample the reference along the
  curve normals ("straighten"), measure the LOCAL link period with windowed 2-D
  masked correlation (it grows ~98 -> 106 px along the arm from perspective), fit
  P(s) linear, re-parametrise by phase and scale the normal axis by the same
  local factor so every link comes out the same size.  A loop of N=3 periods is
  cross-faded at its seam with the run's continuation, and the 1024 strip holds
  exactly 3 loops (9 periods) sampled straight from the reference in one pass.
Deterministic: no randomness except the seeded black_fabric backgrounds.
"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from ata_style import *

os.makedirs(OUT_DIR, exist_ok=True)
DBG = os.path.join(os.path.dirname(OUT_DIR), 'dbg_pc_')

STRIP_H = 320          # chain strip height
# (loop periods N, periods per 1024 px strip N'); N' must be a multiple of N so the strip wraps
LOOP_CHOICES = [(3, 9), (2, 10), (2, 8)]
OV_PHASE = 0.25        # cross-fade overlap at the loop seam, in link periods
PENDANT_FILL = 0.92    # pendant width as fraction of canvas
VELVET = np.array([0.012, 0.010, 0.011], np.float32)   # mean velvet colour (for de-fringing)

# ------------------------------------------------------------------ helpers
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def key_reference(a):
    """Return (soft_key, binary_mask) for a reference photo on black velvet."""
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    chroma = a.max(-1) - a.min(-1)
    # luminance threshold depends on chroma: saturated gold/red pixels count from
    # lum 0.10, but near-grey pixels must reach 0.30 - this rejects velvet that is
    # dimly lit by a sparkle glint (lum 0.07-0.2, no chroma) while keeping the
    # bright star rays and white diamonds.
    # The chroma threshold 0.13 puts the edge on the links' rim highlight line
    # instead of 3-4 px out in the velvet that merely reflects the gold.
    lum_thr = 0.30 - 0.20 * smoothstep(chroma / 0.15)
    k = np.maximum(lum / lum_thr, chroma / 0.13)          # 1.0 at the threshold
    soft = np.clip((k - 0.85) / 0.30, 0, 1).astype(np.float32)   # narrow: sub-pixel refinement only
    binary = k > 1.0
    binary = ndimage.binary_closing(binary, iterations=3)
    binary = ndimage.binary_opening(binary, iterations=2)
    lab, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, lab, range(1, n + 1))
    binary = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s > 2000])
    # fill pin-holes (dark crevices between gems) but keep real openings
    holes = ~binary
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    border = np.zeros_like(holes); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touch = np.unique(hl[border & holes])
    small = [i + 1 for i, s in enumerate(hs) if s < 300 and (i + 1) not in touch]
    binary = binary | np.isin(hl, small)
    return soft, binary

def soft_alpha(soft, binary, feather=0.5):
    """Smooth the ragged binary boundary (blur + re-threshold keeps the edge position),
    then 1 inside the eroded mask, the soft key in a 1 px band at the edge, then a
    light feather -> a clean ~1 px anti-aliased edge."""
    b = (blur(binary.astype(np.float32), 1.5) > 0.5).astype(np.float32)
    alpha = np.maximum(soft * dilate(b, 1.0), erode(b, 1.0))
    if feather > 0:
        alpha = blur(alpha, feather)
    return np.clip(alpha, 0, 1)

def defringe(rgb, alpha, bg=VELVET, floor=0.3):
    """Un-premultiply edge pixels against the dark background colour."""
    a = np.maximum(alpha, floor)[..., None]
    out = (rgb - (1 - alpha)[..., None] * bg[None, None, :]) / a
    return np.clip(out, 0, 1).astype(np.float32)

def bleed(rgb, alpha, radius=4.0):
    """Copy the nearest opaque colour into transparent pixels within `radius` px of
    the edge so bilinear/mipmap sampling never pulls in black."""
    opaque = alpha > 0.5
    d, (iy, ix) = ndimage.distance_transform_edt(~opaque, return_indices=True)
    near = (~opaque) & (d <= radius)
    out = rgb.copy()
    out[near] = rgb[iy[near], ix[near]]
    return out

def resize_rgba(rgb, alpha, scale, sharpen=0.0):
    """Premultiplied LANCZOS resize of an RGBA pair; optional unsharp on the colour."""
    h, w = alpha.shape
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    pm = to_pil(rgb * alpha[..., None]).resize((nw, nh), Image.LANCZOS)
    al = to_pil(alpha).resize((nw, nh), Image.LANCZOS)
    pm = from_pil(pm); al = from_pil(al)
    rgb2 = np.clip(pm / np.maximum(al, 1e-3)[..., None], 0, 1)
    if sharpen > 0:
        im = to_pil(rgb2).filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen * 100), threshold=0))
        rgb2 = from_pil(im)
    return rgb2.astype(np.float32), al.astype(np.float32)

def unsharp(rgb, percent, radius=1.0):
    return from_pil(to_pil(rgb).filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=0))).astype(np.float32)

# ------------------------------------------------------------------ load + key
ref = from_pil(Image.open(REF_PENDANT).convert('RGB'))
H, W = ref.shape[:2]
Y, X = np.mgrid[0:H, 0:W]
soft, binary = key_reference(ref)
alpha_full = soft_alpha(soft, binary)

# ================================================================== 1. PENDANT
def build_pendant():
    # bail trapezoid + connector block (measured on the reference), anti-aliased
    def draw_allowed(d, S):
        d.polygon([(531 * S, 216 * S), (695 * S, 216 * S), (659 * S, 447 * S), (571 * S, 447 * S)], fill=255)
        d.rectangle([552 * S, 396 * S, 672 * S, 420 * S], fill=255)
    bail = mask_from_pil_draw(H, W, draw_allowed, supersample=4)
    allowed = np.maximum(bail, (Y >= 415).astype(np.float32))
    a = alpha_full * allowed
    # largest connected component only (drops chain bits + velvet highlights)
    lab, n = ndimage.label(a > 0.3)
    sizes = ndimage.sum(a > 0.3, lab, range(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    a = a * dilate(big.astype(np.float32), 2.0)
    rgb = defringe(ref, a)
    ys, xs = np.where(a > 0.01)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    print('pendant bbox', x0, y0, x1, y1, 'size', x1 - x0, y1 - y0)
    rgb_c, a_c = rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]
    scale = PENDANT_FILL * 1024 / (x1 - x0)
    rgb_s, a_s = resize_rgba(rgb_c, a_c, scale, sharpen=0.35)
    h, w = a_s.shape
    canvas_rgb = np.zeros((1024, 1024, 3), np.float32)
    canvas_a = np.zeros((1024, 1024), np.float32)
    ox, oy = (1024 - w) // 2, (1024 - h) // 2
    canvas_rgb[oy:oy + h, ox:ox + w] = rgb_s
    canvas_a[oy:oy + h, ox:ox + w] = a_s
    # black under alpha=0, then bleed edge colour 4 px into the transparent margin
    canvas_rgb *= (canvas_a > 0)[..., None]
    canvas_rgb = bleed(canvas_rgb, canvas_a, 4.0)
    out = rgba(canvas_rgb, canvas_a)
    save(out, os.path.join(OUT_DIR, 'pendant.png'))
    print('pendant scaled', w, h, 'scale %.3f' % scale)
    bg = black_fabric(1024, 1024, seed=7)
    save(over(bg, canvas_rgb, canvas_a), os.path.join(OUT_DIR, 'pendant_on_black.png'))
    return out

# ================================================================== 2. CHAIN
class Centreline:
    """Cubic centreline through a chain arm, parametrised by arc length (ref px)."""
    def __init__(self, region, extend=80.0):
        m = binary & region
        ys, xs = np.where(m)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        mean = pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
        ud = evecs[:, 1]
        if ud[0] < 0: ud = -ud
        vd = np.array([-ud[1], ud[0]])
        u = (pts - mean) @ ud; v = (pts - mean) @ vd
        bw = 6.0
        bins = np.arange(u.min(), u.max() + bw, bw)
        idx = np.digitize(u, bins)
        cu, cv, ww = [], [], []
        for i in range(1, len(bins)):
            sel = idx == i
            if sel.sum() < 20: continue
            vv = v[sel]
            cu.append(bins[i - 1] + bw / 2); cv.append((vv.min() + vv.max()) / 2); ww.append(vv.max() - vv.min())
        cu, cv, ww = map(np.array, (cu, cv, ww))
        good = ww > 0.85 * np.median(ww)
        coef = np.polyfit(cu[good], cv[good], 3)
        self.width = float(np.median(ww[good]))
        uu = np.linspace(cu[good].min() - extend, cu[good].max() + extend, 4000)
        vv = np.polyval(coef, uu)
        P = mean[None, :] + uu[:, None] * ud[None, :] + vv[:, None] * vd[None, :]
        d = np.diff(P, axis=0); seg = np.hypot(d[:, 0], d[:, 1])
        s = np.concatenate([[0], np.cumsum(seg)])
        self.s, self.P, self.L = s, P, float(s[-1])
        # tangent as function of s
        t = np.gradient(P, axis=0); t /= np.hypot(t[:, 0], t[:, 1])[:, None]
        self.T = t

    def sample_coords(self, s_arr, n_arr):
        """s (1-D, arc length) x n (1-D per row, or a full (rows, cols) grid of
        normal offsets) -> (SY, SX) sampling grids in reference pixels."""
        px = np.interp(s_arr, self.s, self.P[:, 0]); py = np.interp(s_arr, self.s, self.P[:, 1])
        tx = np.interp(s_arr, self.s, self.T[:, 0]); ty = np.interp(s_arr, self.s, self.T[:, 1])
        tn = np.hypot(tx, ty); tx /= tn; ty /= tn
        nx, ny = -ty, tx
        n_arr = np.asarray(n_arr, np.float64)
        if n_arr.ndim == 1:
            n_arr = n_arr[:, None]
        SX = px[None, :] + n_arr * nx[None, :]
        SY = py[None, :] + n_arr * ny[None, :]
        return SY, SX

def sample_ref(SY, SX, order=3):
    rgb = np.stack([ndimage.map_coordinates(ref[..., c], [SY, SX], order=order, mode='constant', cval=0.0) for c in range(3)], -1)
    al = ndimage.map_coordinates(alpha_full, [SY, SX], order=order, mode='constant', cval=0.0)
    return np.clip(rgb, 0, 1).astype(np.float32), np.clip(al, 0, 1).astype(np.float32)

def local_periods(rgb, al, win=180, step=20, lo=85, hi=125):
    """Local link period along a straightened chain, from windowed 2-D masked
    correlation of the RGB image with itself shifted by d (sub-pixel via parabola).
    Returns (centres, periods, strengths)."""
    m = al > 0.5
    x = rgb - rgb[m].mean(0)
    w = rgb.shape[1]
    out = []
    for c in np.arange(win / 2, w - win / 2 + 0.5, step):
        c0, c1 = int(round(c - win / 2)), int(round(c + win / 2))
        cs = {}
        for d in range(lo, hi + 1):
            if c1 - d <= c0 + 20: continue
            A = x[:, c0:c1 - d]; B = x[:, c0 + d:c1]; M = m[:, c0:c1 - d] & m[:, c0 + d:c1]
            a, b = A[M], B[M]
            cs[d] = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        d = max(cs, key=cs.get)
        if d - 1 in cs and d + 1 in cs:
            y0, y1, y2 = cs[d - 1], cs[d], cs[d + 1]
            den = y0 - 2 * y1 + y2
            d = d + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
        out.append((c, d, cs[max(cs, key=cs.get)]))
    return np.array(out).T

def build_chain():
    # left arm: everything left of the bail and above the pendant
    region = (X < 520) & (Y < 420)
    cl = Centreline(region, extend=90.0)
    print('centreline length %.1f, chain core width %.1f' % (cl.L, cl.width))
    half = cl.width / 2
    # usable s: the whole chain band (|n| <= half+6) inside the frame and left of the bail
    s_all = np.arange(0, cl.L, 1.0)
    n_band = np.linspace(-(half + 6), half + 6, 25)
    SY, SX = cl.sample_coords(s_all, n_band)
    inside = (SX >= 1) & (SX < W - 1) & (SY >= 1) & (SY < H - 1) & ~((SX > 536) & (SY > 195))
    valid = inside.all(0)
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(np.concatenate([valid, [False]])):
        if v:
            if cur == 0: start = i
            cur += 1
            if cur > best[1] - best[0]: best = (start, start + cur)
        else:
            cur = 0
    s0, s1 = best
    print('usable arc range', s0, s1, 'len', s1 - s0)
    # straightened preview at 1:1 (unit arc-length steps) for measurement + debugging
    s_prev = np.arange(s0, s1, 1.0)
    n_prev = np.arange(-130, 130, 1.0)
    SY, SX = cl.sample_coords(s_prev, n_prev)
    prev_rgb, prev_a = sample_ref(SY, SX, order=1)
    save(rgba(defringe(prev_rgb, prev_a), prev_a), DBG + 'straight_left.png')
    # vertical centre of the chain (rows where the mask is mostly present)
    rows_on = np.where((prev_a > 0.5).mean(1) > 0.3)[0]
    n_c = float(n_prev[rows_on].mean())
    full_w = float(rows_on.max() - rows_on.min() + 1)
    print('chain centre offset %.1f px, full width %.0f px' % (n_c, full_w))
    # local period model P(s) = P0 + k (s - s_mid): perspective makes links grow along the arm
    cen, per, strength = local_periods(prev_rgb, prev_a)
    print('local periods:', ' '.join('%.0f:%.1f(%.2f)' % (c + s0, p, q) for c, p, q in zip(cen, per, strength)))
    wgt = np.clip(strength, 0.05, None)
    k, P0 = np.polyfit(cen + s0, per, 1, w=wgt)
    s_mid = (s0 + s1) / 2
    P_mid = P0 + k * s_mid
    print('period fit: P(s) = %.2f + %.4f (s - %.0f)   [%.1f .. %.1f]' % (P_mid, k, s_mid, P0 + k * s0, P0 + k * s1))
    # phase (in periods) as a function of arc length, and its inverse
    s_grid = np.arange(s0, s1 + 0.001, 0.25)
    P_grid = P0 + k * s_grid
    phi_grid = np.concatenate([[0], np.cumsum(0.25 / P_grid[:-1])])
    Phi = float(phi_grid[-1])
    s_of_phi = lambda ph: np.interp(ph, phi_grid, s_grid)
    P_of_s = lambda ss: P0 + k * ss
    for N, Np in LOOP_CHOICES:
        if Phi >= N + OV_PHASE + 0.02: break
    else:
        raise RuntimeError('not enough clean chain: %.2f periods' % Phi)
    P_ref = P_mid
    scale = 1024.0 / (Np * P_ref)
    print('available %.2f periods -> loop N=%d, strip N\'=%d periods, scale %.3f, chain ~%.0f px tall' % (Phi, N, Np, scale, full_w * scale))
    phi_start = (Phi - (N + OV_PHASE)) / 2            # centre the used run in the usable range
    cols = np.arange(1024)
    phi_loop = (cols * Np / 1024.0) % N                # loop phase of every strip column
    rows = np.arange(STRIP_H)
    n_ref = (rows - (STRIP_H - 1) / 2) / scale + n_c    # normal offset at the reference period

    def sample_loop(phi):
        """RGB, alpha for strip columns at loop phase phi (1-D), size-normalised."""
        ss = s_of_phi(phi_start + phi)
        f = P_of_s(ss) / P_ref                         # local size factor
        N2 = n_ref[:, None] * f[None, :]
        SY, SX = cl.sample_coords(ss, N2)
        return sample_ref(SY, SX, order=3)

    rgb_t, a_t = sample_loop(phi_loop)
    # cross-fade the loop seam: columns with phase < OV_PHASE blend from the run's continuation
    sel = np.where(phi_loop < OV_PHASE)[0]
    rgb_c, a_c = sample_loop(phi_loop[sel] + N)
    w = smoothstep(phi_loop[sel] / OV_PHASE)[None, :, None]    # 0 -> continuation, 1 -> own
    pm = rgb_t[:, sel] * a_t[:, sel, None] * w + rgb_c * a_c[..., None] * (1 - w)
    a_bl = a_t[:, sel] * w[..., 0] + a_c * (1 - w[..., 0])
    rgb_t[:, sel] = np.clip(pm / np.maximum(a_bl, 1e-3)[..., None], 0, 1)
    a_t[:, sel] = a_bl
    rgb_t = defringe(rgb_t, a_t)
    rgb_t = unsharp(rgb_t, 50)
    rgb_t *= (a_t > 0)[..., None]
    rgb_t = bleed(rgb_t, a_t, 4.0)
    print('edge-row alpha max: top %.3f bottom %.3f' % (a_t[:3].max(), a_t[-3:].max()))
    strip = rgba(rgb_t, a_t)
    save(strip, os.path.join(OUT_DIR, 'chain_strip.png'))
    # tile check: the strip twice on black fabric
    bg = black_fabric(STRIP_H, 2048, seed=11)
    two_rgb = np.tile(rgb_t, (1, 2, 1)); two_a = np.tile(a_t, (1, 2))
    two = over(bg, two_rgb, two_a)
    save(two, os.path.join(OUT_DIR, 'chain_2x.png'))
    # debug zooms: strip wrap (x = 1024) and one loop seam (phase 0 inside the strip)
    seam_x = int(round(1024 / Np * N))                  # first loop seam after x=0
    for name, xc in (('wrap', 1024), ('loopseam', seam_x)):
        crop = two[:, xc - 128:xc + 128]
        save(from_pil(to_pil(crop).resize((512, STRIP_H * 2), Image.NEAREST)), DBG + name + '_2x.png')
    return strip

if __name__ == '__main__':
    import time
    t0 = time.time()
    build_pendant()
    print('pendant done %.1fs' % (time.time() - t0))
    build_chain()
    print('total %.1fs' % (time.time() - t0))
    for f in ('pendant.png', 'chain_strip.png', 'chain_2x.png', 'pendant_on_black.png'):
        im = Image.open(os.path.join(OUT_DIR, f)); print(f, im.size, im.mode)
