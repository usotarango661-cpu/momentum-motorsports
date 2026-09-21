"""Shared style helpers for the ATA / FiveM clothing texture kit.

All element generators import from here so the gold, red and black fabric
looks identical across the jacket, jeans and chain textures.

Conventions
-----------
* Images are numpy float32 arrays in 0..1, shape (H, W, 3) for RGB and
  (H, W) for masks/alpha.  Use to_pil()/from_pil() at the boundaries.
* Masks are 0..1 floats (1 = inside the shape).
* All noise helpers are tileable (periodic) so tiles can wrap seamlessly.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

# ---------------------------------------------------------------- palette
GOLD_DARK  = (0.47, 0.30, 0.05)
GOLD       = (0.84, 0.63, 0.20)
GOLD_LIGHT = (1.00, 0.88, 0.51)
RED_DARK   = (0.43, 0.02, 0.05)
RED        = (0.77, 0.07, 0.12)
RED_LIGHT  = (0.93, 0.24, 0.24)
BLACK      = (0.04, 0.03, 0.035)
BLACK_LIGHT= (0.12, 0.10, 0.11)

REF_DIR = './reference'
REF_SHEET = REF_DIR + '/sheet.png'       # 1254x1254 jacket + jeans reference sheet
REF_PENDANT = REF_DIR + '/pendant.png'   # 1254x1254 ATA chain pendant on black velvet
OUT_DIR = './elements'

# ---------------------------------------------------------------- io
def to_pil(arr):
    """float 0..1 array (H,W), (H,W,3) or (H,W,4) -> PIL image."""
    a = np.clip(arr, 0, 1)
    a = (a * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, 'L')
    return Image.fromarray(a, 'RGB' if a.shape[2] == 3 else 'RGBA')

def from_pil(im):
    return np.asarray(im).astype(np.float32) / 255.0

def save(arr, path):
    to_pil(arr).save(path)

def rgba(rgb, alpha):
    """rgb (H,W,3) + alpha (H,W) -> (H,W,4)"""
    return np.dstack([rgb, np.clip(alpha, 0, 1)])

# ---------------------------------------------------------------- noise
def spectral_noise(h, w, beta=1.6, seed=0):
    """Tileable 1/f^beta noise, normalised to 0..1.  beta 1 = pink, 2 = brown."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((h, w))
    F = np.fft.fft2(white)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    F = F / (f ** (beta / 2.0 * 2.0 / 2.0 + beta / 2.0))  # ~1/f^beta amplitude
    out = np.real(np.fft.ifft2(F))
    out -= out.min(); out /= (out.max() + 1e-9)
    return out.astype(np.float32)

def fbm(h, w, octaves=5, base_scale=None, seed=0, lacunarity=2.0, gain=0.5):
    """Tileable value-noise fBm 0..1.  base_scale = pixels per lattice cell of octave 0."""
    if base_scale is None:
        base_scale = w / 4
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0; total = 0.0
    scale = base_scale
    for o in range(octaves):
        gh = max(2, int(round(h / scale))); gw = max(2, int(round(w / scale)))
        grid = rng.random((gh, gw)).astype(np.float32)
        # periodic bicubic-ish upsample via map_coordinates with wrap
        ys = (np.arange(h) / h) * gh
        xs = (np.arange(w) / w) * gw
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        layer = ndimage.map_coordinates(grid, [Y, X], order=3, mode='grid-wrap')
        out += amp * layer
        total += amp
        amp *= gain; scale /= lacunarity
    out /= total
    out -= out.min(); out /= (out.max() + 1e-9)
    return out

def warp(img, dx, dy):
    """Warp a periodic 2-D field by displacement fields (pixels)."""
    h, w = img.shape[:2]
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return ndimage.map_coordinates(img, [Y + dy, X + dx], order=1, mode='grid-wrap')

# ---------------------------------------------------------------- masks
def blur(mask, sigma):
    return ndimage.gaussian_filter(mask.astype(np.float32), sigma)

def dilate(mask, r):
    """Soft-edged dilation of a 0..1 mask by r pixels (uses distance transform)."""
    if r <= 0: return mask
    inside = mask > 0.5
    d = ndimage.distance_transform_edt(~inside)
    return np.clip(r + 0.5 - d, 0, 1).astype(np.float32) * (~inside) + inside

def erode(mask, r):
    if r <= 0: return mask
    inside = mask > 0.5
    d = ndimage.distance_transform_edt(inside)
    return np.clip(d - r + 0.5, 0, 1).astype(np.float32)

def outline(mask, width, offset=0.0):
    """Ring around a mask: from `offset` px outside the edge to offset+width."""
    return np.clip(dilate(mask, offset + width) - dilate(mask, offset), 0, 1)

def bevel_shade(mask, depth=3.0, light=(-0.6, -0.8), strength=0.35):
    """Return a multiplicative shading field (around 1.0) giving an embossed look.
    light = (dx, dy) direction the light comes FROM (negative y = top)."""
    sm = blur(mask, depth)
    gy, gx = np.gradient(sm)
    shade = -(gx * light[0] + gy * light[1])
    shade = shade / (np.abs(shade).max() + 1e-9)
    return 1.0 + strength * shade

def antialias_upscale(mask, factor, sharpen=1.5):
    """Upscale a soft mask smoothly then re-sharpen the edge (for extracting shapes from small crops)."""
    im = to_pil(mask).resize((int(mask.shape[1] * factor), int(mask.shape[0] * factor)), Image.LANCZOS)
    m = from_pil(im)
    m = blur(m, 1.0)
    return np.clip((m - 0.5) * sharpen + 0.5, 0, 1)

# ---------------------------------------------------------------- material fills
def _lerp3(a, b, t):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]

def sparkle(h, w, density=0.004, seed=0, size=1.0):
    rng = np.random.default_rng(seed)
    s = (rng.random((h, w)) < density).astype(np.float32)
    s = blur(s, size)
    return np.clip(s / (s.max() + 1e-9) * 1.4, 0, 1)

def thread_texture(h, w, angle_deg=45, period=3.0, seed=0):
    """Fine diagonal stitching lines 0..1 (embroidery feel)."""
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    a = np.deg2rad(angle_deg)
    u = X * np.cos(a) + Y * np.sin(a)
    stripes = 0.5 + 0.5 * np.cos(2 * np.pi * u / period)
    n = fbm(h, w, octaves=3, base_scale=8, seed=seed)
    return np.clip(stripes * (0.6 + 0.4 * n), 0, 1).astype(np.float32)

def gold_fill(h, w, seed=0, glitter=0.5, thread=True):
    """Metallic / glitter-embroidered gold RGB (H,W,3)."""
    n1 = fbm(h, w, octaves=5, base_scale=max(8, w / 6), seed=seed)
    n2 = fbm(h, w, octaves=4, base_scale=6, seed=seed + 1)
    t = np.clip(0.25 + 0.6 * n1 + 0.35 * (n2 - 0.5), 0, 1)
    col = np.where((t < 0.5)[..., None], _lerp3(GOLD_DARK, GOLD, t * 2), _lerp3(GOLD, GOLD_LIGHT, (t - 0.5) * 2))
    if thread:
        th = thread_texture(h, w, 45, 2.5, seed + 2)
        col = col * (0.85 + 0.3 * th[..., None])
    sp = sparkle(h, w, density=0.006 * glitter * 2, seed=seed + 3, size=0.8)
    col = col + sp[..., None] * np.array([1.0, 0.95, 0.75])[None, None, :] * 0.9
    return np.clip(col, 0, 1).astype(np.float32)

def red_fill(h, w, seed=0, speckle=True):
    """Red glitter-embroidered fill with dark leopard-ish speckles, RGB (H,W,3)."""
    n1 = fbm(h, w, octaves=5, base_scale=max(8, w / 5), seed=seed + 10)
    n2 = fbm(h, w, octaves=4, base_scale=5, seed=seed + 11)
    t = np.clip(0.3 + 0.5 * n1 + 0.3 * (n2 - 0.5), 0, 1)
    col = np.where((t < 0.5)[..., None], _lerp3(RED_DARK, RED, t * 2), _lerp3(RED, RED_LIGHT, (t - 0.5) * 2))
    th = thread_texture(h, w, -45, 2.5, seed + 12)
    col = col * (0.85 + 0.3 * th[..., None])
    if speckle:
        sp = fbm(h, w, octaves=3, base_scale=10, seed=seed + 13)
        dark = np.clip((sp - 0.72) * 6, 0, 1)
        col = col * (1 - 0.8 * dark[..., None])
    gl = sparkle(h, w, density=0.005, seed=seed + 14, size=0.7)
    col = col + gl[..., None] * np.array([1.0, 0.85, 0.7])[None, None, :] * 0.8
    return np.clip(col, 0, 1).astype(np.float32)

def black_fabric(h, w, seed=0):
    """Dark velvet / textured knit base, tileable, RGB (H,W,3)."""
    n = fbm(h, w, octaves=6, base_scale=max(16, w / 8), seed=seed + 20)
    fine = fbm(h, w, octaves=2, base_scale=3, seed=seed + 21)
    t = np.clip(0.25 + 0.5 * n + 0.35 * (fine - 0.5), 0, 1)
    return _lerp3(BLACK, BLACK_LIGHT, t).astype(np.float32)

# ---------------------------------------------------------------- compositing
def over(dst_rgb, src_rgb, alpha):
    """Alpha-composite src over dst. alpha (H,W)."""
    a = np.clip(alpha, 0, 1)[..., None]
    return dst_rgb * (1 - a) + src_rgb * a

def paste_rgba(dst_rgb, src_rgba, x, y, scale=1.0, rotate=0.0, opacity=1.0):
    """Paste an RGBA element (numpy) centred at (x,y) on dst RGB. Returns new dst."""
    im = to_pil(src_rgba)
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, expand=True)
    s = from_pil(im)
    H, W = dst_rgb.shape[:2]; h, w = s.shape[:2]
    x0 = int(round(x - w / 2)); y0 = int(round(y - h / 2))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W, x0 + w); dy1 = min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return dst_rgb
    sub = s[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    out = dst_rgb.copy()
    out[dy0:dy1, dx0:dx1] = over(out[dy0:dy1, dx0:dx1], sub[..., :3], sub[..., 3] * opacity)
    return out

def tile_to(tile_rgb, h, w, offset=(0, 0)):
    """Repeat a tileable RGB tile to fill (h,w)."""
    th, tw = tile_rgb.shape[:2]
    reps = (h // th + 2, w // tw + 2, 1)
    big = np.tile(tile_rgb, reps)
    oy, ox = offset
    return big[oy:oy + h, ox:ox + w]

def emboss_element(mask, fill_rgb, edge_width=0, edge_rgb=None, depth=2.5, strength=0.35, glow=0.0):
    """Turn a mask into a shaded, optionally outlined RGBA element.
    mask: (H,W) 0..1.  fill_rgb: (H,W,3).  edge_rgb: (H,W,3) used for the outline ring."""
    shade = bevel_shade(mask, depth=depth, strength=strength)
    rgb = np.clip(fill_rgb * shade[..., None], 0, 1)
    alpha = mask.copy()
    if edge_width > 0 and edge_rgb is not None:
        ring = outline(mask, edge_width)
        eshade = bevel_shade(dilate(mask, edge_width), depth=depth, strength=strength)
        ergb = np.clip(edge_rgb * eshade[..., None], 0, 1)
        rgb = over(rgb, ergb, ring)
        alpha = np.clip(alpha + ring, 0, 1)
    if glow > 0:
        g = blur(alpha, glow)
        rgb = over(np.zeros_like(rgb) + np.array(GOLD)[None, None, :] * 0.6, rgb, alpha)
        alpha = np.clip(alpha + 0.35 * g, 0, 1)
    return rgba(rgb, alpha)

def mask_from_pil_draw(h, w, draw_fn, supersample=4):
    """Rasterise vector shapes with anti-aliasing. draw_fn(ImageDraw, scale)."""
    S = supersample
    im = Image.new('L', (w * S, h * S), 0)
    d = ImageDraw.Draw(im)
    draw_fn(d, S)
    im = im.resize((w, h), Image.LANCZOS)
    return from_pil(im)

def color_mask(rgb, kind):
    """Rough colour classification of a reference crop. kind in {'gold','red','bright'}.
    rgb float (H,W,3). Returns 0..1 soft mask."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if kind == 'gold':
        m = (r > 0.45) & (g > 0.32) & (b < 0.55) & (g > b + 0.10) & (r - g < 0.45)
    elif kind == 'red':
        m = (r > 0.35) & (g < 0.35) & (b < 0.35) & (r - g > 0.25)
    elif kind == 'bright':
        m = (r + g + b) > 1.5
    else:
        raise ValueError(kind)
    return m.astype(np.float32)
