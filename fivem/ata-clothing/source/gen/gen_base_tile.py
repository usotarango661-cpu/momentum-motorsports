"""base_tile: seamless 1024x1024 black velvet/knit fabric with red lace/marble blotches.

Every operation here is periodic (wrap-mode blurs, grid-wrap warps, integer-frequency
mesh lines, FFT/lattice noise from ata_style), so the tile is seamless by construction.
Run from the work dir:  python3 gen/gen_base_tile.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from PIL import Image
from ata_style import *

H = W = 1024
OUT = OUT_DIR + '/base_tile.png'
OUT2 = OUT_DIR + '/base_tile_2x2.png'

# --------------------------------------------------------------- periodic helpers
def pblur(a, sigma):
    """Gaussian blur that wraps (tileable)."""
    return ndimage.gaussian_filter(a.astype(np.float32), sigma, mode='wrap')

def fbm_aniso(h, w, cell_h, cell_w, octaves=4, seed=0, gain=0.5, shear=0):
    """Tileable value-noise fBm with rectangular lattice cells (cell_h x cell_w px) and an
    optional integer shear (x += shear*y) which keeps the field periodic when h == w."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32); amp = 1.0; total = 0.0
    ch, cw = float(cell_h), float(cell_w)
    Y, X = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    Xs = X + shear * Y
    for o in range(octaves):
        gh = max(2, int(round(h / ch))); gw = max(2, int(round(w / cw)))
        grid = rng.random((gh, gw)).astype(np.float32)
        layer = ndimage.map_coordinates(grid, [Y / h * gh, Xs / w * gw], order=3, mode='grid-wrap')
        out += amp * layer; total += amp
        amp *= gain; ch /= 2.0; cw /= 2.0
    out /= total
    out -= out.min(); out /= (out.max() + 1e-9)
    return out

def periodic_lines(h, w, kx, ky, phase=0.0):
    """0..1 cosine stripes with integer frequencies (kx cycles across w, ky across h)."""
    Y, X = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    return 0.5 + 0.5 * np.cos(2 * np.pi * (kx * X / w + ky * Y / h) + phase)

def ridged(n, width=0.10, power=1.0):
    """Thin bright lines where a 0..1 noise field crosses 0.5 (lace-thread look)."""
    r = 1.0 - np.abs(2.0 * n - 1.0)          # 1 on the 0.5 iso-line, 0 at extremes
    t = np.clip((r - (1 - width)) / width, 0, 1)
    return t ** power

def hard(a, lo, hi):
    """Crisp (but ~1 px anti-aliased) threshold of a 0..1 field."""
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

def lerp3(a, b, t):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]

# --------------------------------------------------------------- 1. black velvet / knit base
def make_velvet(seed=100):
    base = black_fabric(H, W, seed=seed)                      # tileable dark base from shared module
    # knit ribs: fine diagonal stitch rows (integer freq => periodic), slightly wobbled
    wob = (fbm(H, W, octaves=3, base_scale=64, seed=seed + 1) - 0.5) * 6
    ribs = periodic_lines(H, W, 170, 170)                      # ~6 px diagonal pitch
    ribs = warp(ribs, wob, -wob)
    cross = periodic_lines(H, W, -120, 120)
    knit = 0.6 * ribs + 0.4 * cross
    # velvet pile sheen: broad soft highlights + medium crush marks
    sheen = pblur(fbm(H, W, octaves=3, base_scale=300, seed=seed + 2), 3)
    crush = fbm_aniso(H, W, 160, 40, octaves=4, seed=seed + 3, shear=1)
    grain = spectral_noise(H, W, beta=1.0, seed=seed + 4)      # per-pixel fibre grain
    lum = 1.0 + 0.28 * (knit - 0.5) + 0.6 * (sheen - 0.5) + 0.28 * (crush - 0.5) + 0.25 * (grain - 0.5)
    rgb = base * lum[..., None]
    tint = np.array([1.04, 0.98, 1.0])[None, None, :]         # not a dead grey black
    rgb = rgb * tint
    sp = sparkle(H, W, density=0.0015, seed=seed + 5, size=0.6)  # a few glinting pile fibres
    rgb = rgb + sp[..., None] * 0.10
    return np.clip(rgb, 0, 1).astype(np.float32)

# --------------------------------------------------------------- 2. marble / splatter blotch field
def make_blotch_field(seed=200):
    # large organic shapes: isotropic + diagonally streaked marble
    big = fbm(H, W, octaves=4, base_scale=200, seed=seed)
    streak = fbm_aniso(H, W, 330, 70, octaves=4, seed=seed + 1, shear=1)
    field = 0.5 * big + 0.5 * streak
    # domain warp for marbled, swirling contours
    dx = (fbm(H, W, octaves=3, base_scale=180, seed=seed + 2) - 0.5) * 150
    dy = (fbm(H, W, octaves=3, base_scale=180, seed=seed + 3) - 0.5) * 150
    field = warp(field, dx, dy)
    dx2 = (fbm(H, W, octaves=3, base_scale=50, seed=seed + 4) - 0.5) * 36
    dy2 = (fbm(H, W, octaves=3, base_scale=50, seed=seed + 5) - 0.5) * 36
    field = warp(field, dx2, dy2)
    # thin marble veins that tie blotches together
    veins = ridged(fbm(H, W, octaves=3, base_scale=140, seed=seed + 8), width=0.10, power=1.0)
    veins = warp(veins, dx * 0.5, dy * 0.5)
    # ragged detail so contours are torn, not soft
    det = fbm(H, W, octaves=5, base_scale=34, seed=seed + 6)
    mid = fbm(H, W, octaves=3, base_scale=16, seed=seed + 9)
    fine = fbm(H, W, octaves=3, base_scale=8, seed=seed + 7)
    field = field + 0.12 * veins + 0.30 * (det - 0.5) + 0.10 * (mid - 0.5) + 0.10 * (fine - 0.5)
    field -= field.min(); field /= field.max()
    return field.astype(np.float32)

# --------------------------------------------------------------- 3. red lace / mesh texture
def make_lace(seed=300):
    # (a) fine diamond net (tulle) ~5 px pitch, strongly wobbled so it is not a screen
    wob_x = (fbm(H, W, octaves=4, base_scale=40, seed=seed) - 0.5) * 10
    wob_y = (fbm(H, W, octaves=4, base_scale=40, seed=seed + 1) - 0.5) * 10
    l1 = periodic_lines(H, W, 205, 205)
    l2 = periodic_lines(H, W, -205, 205)
    l1 = warp(l1, wob_x, wob_y); l2 = warp(l2, wob_y, -wob_x)
    net = hard(np.maximum(l1, l2), 0.72, 0.95)                  # crisp thin threads, dark cells
    mesh_patch = np.clip((fbm(H, W, octaves=3, base_scale=90, seed=seed + 6) - 0.2) * 1.8, 0, 1)
    net = net * (0.6 + 0.4 * mesh_patch)                        # mesh everywhere, denser in patches
    # (b) embroidered lace scrolls: ridged noise at three scales (smooth loops -> fine curls)
    r0 = ridged(fbm(H, W, octaves=2, base_scale=52, seed=seed + 7), width=0.10, power=1.0)
    r1 = ridged(fbm(H, W, octaves=3, base_scale=26, seed=seed + 2), width=0.14, power=1.0)
    r2 = ridged(fbm(H, W, octaves=3, base_scale=13, seed=seed + 3), width=0.20, power=1.0)
    curls = np.maximum(np.maximum(r0, 0.95 * r1), 0.8 * r2)
    curls = hard(curls, 0.35, 0.75)                             # crisp, ~1 px anti-aliased threads
    # stitch ticks along the threads (glitter-embroidery feel)
    tick = periodic_lines(H, W, 300, -300)
    tick = warp(tick, wob_y * 0.6, wob_x * 0.6)
    curls = curls * (0.7 + 0.3 * tick)
    # (c) medium "floral" density modulation so some patches are denser
    dens = fbm(H, W, octaves=3, base_scale=70, seed=seed + 4)
    thread = np.clip(np.maximum(0.85 * net, curls), 0, 1)
    thread = np.clip(thread * (0.65 + 0.6 * dens), 0, 1)
    # (d) dark burnout holes between threads
    holes = fbm(H, W, octaves=4, base_scale=18, seed=seed + 5)
    holes = np.clip((holes - 0.63) * 6, 0, 1)
    holes = holes * (1 - 0.75 * thread)                         # threads pass over the holes
    return thread.astype(np.float32), holes.astype(np.float32), dens.astype(np.float32)

# --------------------------------------------------------------- 4. assemble
def build(seed=7):
    t0 = time.time()
    velvet = make_velvet(seed + 100)
    field = make_blotch_field(seed + 200)
    thread, holes, dens = make_lace(seed + 300)

    # coverage ~33 %: threshold at the 67th percentile, ~1 px anti-aliased edge
    thr = float(np.quantile(field, 0.67))
    core = hard(field, thr, thr + 0.004)

    # lacy / ragged margin: outside the solid core the red survives only as hard thread fragments
    band = hard(field, thr - 0.055, thr)                           # 0 far outside .. 1 at core edge
    frag = band * np.clip(thread * 1.5 - 0.15, 0, 1)
    edge = hard(frag, 0.42, 0.55)
    # scattered splatter specks trailing off the blotches
    speck = fbm(H, W, octaves=3, base_scale=6, seed=seed + 400)
    halo = pblur(core, 20)
    speck = hard(speck * np.clip(halo * 2.0 - 0.1, 0, 1), 0.52, 0.60) * (1 - core)
    shape = np.clip(np.maximum(core, np.maximum(edge, speck)), 0, 1)
    shape = shape * (1 - 0.9 * holes)                              # burnout holes in the core

    # ---- red colour: dark red ground, bright threads, brighter cores
    core_depth = pblur(core, 18) * core                            # ~1 deep inside big blotches
    core_glow = np.clip((core_depth - 0.4) * 2.0, 0, 1)
    core_glow = core_glow * np.clip(0.3 + 0.9 * fbm(H, W, octaves=3, base_scale=130, seed=seed + 401), 0, 1)
    ground = lerp3(np.array(RED_DARK) * 0.9, np.array(RED) * 0.95, np.clip(0.22 + 0.4 * dens + 0.6 * core_glow, 0, 1))
    thread_col = lerp3(RED, RED_LIGHT, np.clip(0.25 + 0.8 * core_glow + 0.25 * dens, 0, 1))
    red = ground * (1 - thread[..., None]) + thread_col * thread[..., None]
    hot = np.clip(core_glow * 1.3 - 0.4, 0, 1) * thread            # hottest spots in the biggest cores
    red = red + hot[..., None] * np.array([0.2, 0.05, 0.04])[None, None, :]
    gl = sparkle(H, W, density=0.006, seed=seed + 402, size=0.6) * np.clip(thread * 1.5, 0, 1)
    red = red + gl[..., None] * np.array([0.9, 0.55, 0.45])[None, None, :] * 0.7
    km = pblur(velvet.mean(axis=2), 1.5)                           # soft knit relief shows through
    red = red * (0.9 + 1.2 * km)[..., None]
    red = np.clip(red, 0, 1)

    # ---- alpha: threads are solid red; between threads the velvet shows through
    #      (more so at the lacy margins, much less in the dense bright cores)
    ground_a = np.clip(0.36 + 0.5 * core_glow + 0.2 * dens, 0, 1)
    alpha = shape * np.clip(ground_a * (1 - thread) + thread, 0, 1)
    out = over(velvet, red, alpha)
    shadow = np.clip(pblur(shape, 1.5) - shape, 0, 1)              # tight embroidery contact shadow
    out = out * (1 - 0.25 * shadow[..., None])
    out = np.clip(out, 0, 1).astype(np.float32)

    cov = float((shape > 0.5).mean())
    print(f'red shape coverage ~{cov*100:.1f}%  threshold={thr:.3f}  built in {time.time()-t0:.1f}s')
    return out

if __name__ == '__main__':
    tile = build(seed=7)
    assert tile.shape == (H, W, 3)
    save(tile, OUT)
    im = Image.open(OUT); assert im.size == (1024, 1024) and im.mode == 'RGB'
    big = np.tile(tile, (2, 2, 1))                                 # 2x2 tiling check
    save(big, OUT2)
    # 2x zoom crops for inspection (one interior, one straddling the seam)
    dbg_dir = os.path.dirname(OUT_DIR)
    Image.open(OUT).crop((300, 300, 620, 540)).resize((640, 480), Image.NEAREST).save(dbg_dir + '/base_tile_zoom.png')
    Image.open(OUT2).crop((864, 864, 1184, 1104)).resize((640, 480), Image.NEAREST).save(dbg_dir + '/base_tile_seam_zoom.png')
    # numeric tileability check: wrap-edge difference vs typical neighbouring-pixel difference
    a = from_pil(im)
    wrap_lr = np.abs(a[:, 0] - a[:, -1]).mean(); wrap_tb = np.abs(a[0] - a[-1]).mean()
    inner = np.abs(a[:, 1:] - a[:, :-1]).mean()
    print(f'seam check: left/right {wrap_lr:.4f}  top/bottom {wrap_tb:.4f}  vs interior neighbour diff {inner:.4f}')
    print('wrote', OUT, OUT2)
