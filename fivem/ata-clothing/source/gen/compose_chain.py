"""Compose the ATA chain / necklace texture (FiveM component 7 'teef').

Outputs (final/):
  chain.png               1024x1024 RGBA, transparent background
  chain_on_black.png      same, composited over black_fabric (review)
  chain_layout_guide.png  labelled overlay of the UV regions
  _zoom_chain_clasp.png   2x zoom of the clasp + top of the chain band (crispness check)

Layout (from the garment spec):
  PENDANT   elements/pendant.png centred (512,400), scaled so the opaque content is ~700 px wide
  CLASP     generated 90x40 gold rounded box with red pave gems, centred (512,765)
  CHAIN     elements/chain_strip.png (1024 wide, horizontally tileable) with its opaque
            content filling the band y=790..1024 at 1:1 so the left/right edges stay seamless
"""
import sys, os, time
WORK = '.'
sys.path.insert(0, WORK)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ata_style import *

EL = os.path.join(WORK, 'elements')
FINAL = os.path.join(WORK, 'final')
os.makedirs(FINAL, exist_ok=True)

W = H = 1024
PENDANT_CX, PENDANT_CY, PENDANT_W = 512, 400, 700   # target opaque width in px
PENDANT_MAX_BOTTOM = 740                             # pendant must stay above this
CLASP_CX, CLASP_CY, CLASP_W, CLASP_H = 512, 765, 90, 40
CHAIN_TOP, CHAIN_BOTTOM = 790, 1024

t0 = time.time()

# ------------------------------------------------------------------ helpers
def load_rgba(path):
    return from_pil(Image.open(path).convert('RGBA'))

def alpha_bbox(a, thr=0.02):
    """(x0, y0, x1, y1) inclusive bbox of alpha > thr."""
    al = a[..., 3]
    ys = np.where(al.max(1) > thr)[0]; xs = np.where(al.max(0) > thr)[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def paste_rgba_on_rgba(dst, src, x, y, scale=1.0):
    """Like ata_style.paste_rgba but the destination is straight-alpha RGBA (H,W,4).
    Element centred at (x, y); LANCZOS scaling, no blur."""
    im = to_pil(src)
    if scale != 1.0:
        im = im.resize((max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))), Image.LANCZOS)
    s = from_pil(im)
    Hd, Wd = dst.shape[:2]; h, w = s.shape[:2]
    x0 = int(round(x - w / 2)); y0 = int(round(y - h / 2))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(Wd, x0 + w); dy1 = min(Hd, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return dst
    sub = s[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    out = dst.copy()
    d = out[dy0:dy1, dx0:dx1]
    sa = sub[..., 3:4]; da = d[..., 3:4]
    oa = sa + da * (1 - sa)
    orgb = (sub[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    # keep the source colour where alpha is 0 so nothing black bleeds into filtering
    orgb = np.where(oa > 1e-6, orgb, d[..., :3])
    out[dy0:dy1, dx0:dx1] = np.dstack([orgb, oa])
    return out

def radial(h, w, cx, cy, r):
    Y, X = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing='ij')
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r

# ------------------------------------------------------------------ clasp element
def make_clasp(w=CLASP_W, h=CLASP_H, seed=7):
    """Small gold rounded box with a recessed red-pave channel and a tiny diamond rim.
    Built at 4x and LANCZOS-downsampled so the gems stay crisp at 1:1."""
    S = 4
    Wb, Hb = w * S, h * S
    pad = 2 * S                     # room for the bevel highlight
    # body mask
    body = mask_from_pil_draw(Hb, Wb, lambda d, s: d.rounded_rectangle(
        [pad * s, pad * s, (Wb - pad) * s, (Hb - pad) * s], radius=8 * S * s, fill=255), supersample=2)
    gold = gold_fill(Hb, Wb, seed=seed, glitter=0.3, thread=False)
    # slightly brighter polished look for the clasp than for embroidery
    gold = np.clip(gold * 1.08 + 0.03, 0, 1)
    el = emboss_element(body, gold, depth=6.0, strength=0.6)
    rgb, alpha = el[..., :3].copy(), el[..., 3]

    # recessed channel
    inset = 7 * S
    chan = mask_from_pil_draw(Hb, Wb, lambda d, s: d.rounded_rectangle(
        [inset * s, inset * s, (Wb - inset) * s, (Hb - inset) * s], radius=5 * S * s, fill=255), supersample=2)
    chan_shade = bevel_shade(chan, depth=4.0, light=(0.6, 0.8), strength=0.5)   # inverted light = recess
    red = red_fill(Hb, Wb, seed=seed + 3, speckle=False)
    red_dark = np.clip(red * 0.45, 0, 1)
    rgb = over(rgb, red_dark * chan_shade[..., None], chan)

    # pave: 2 rows of round red gems with facet ring + specular highlight
    gem_r = 4.0 * S
    rows = [Hb / 2 - 5.2 * S, Hb / 2 + 5.2 * S]
    n = 7
    xs = np.linspace(inset + gem_r + 1.0 * S, Wb - inset - gem_r - 1.0 * S, n)
    for ri, cy in enumerate(rows):
        for xi, cx in enumerate(xs):
            cxo = cx   # straight grid (symmetric)
            rr = radial(Hb, Wb, cxo, cy, gem_r)
            m = np.clip((1 - rr) * gem_r, 0, 1)          # AA disc
            # crown shading: bright rim, deep centre, small facet ring
            shade = 0.55 + 0.75 * np.clip(1 - rr, 0, 1) ** 0.6
            facet = 1 + 0.25 * np.cos(np.clip(rr, 0, 1) * np.pi * 3.0)
            gem_col = np.array(RED_LIGHT)[None, None, :] * 0.55 + np.array(RED)[None, None, :] * 0.45
            gem_rgb = np.clip(gem_col * (shade * facet)[..., None], 0, 1)
            rgb = over(rgb, gem_rgb, m)
            # gold prong ring
            ring = np.clip((1 - np.abs(rr - 1.0) * gem_r * 1.4), 0, 1) * 0.55
            rgb = over(rgb, np.clip(gold * 1.15, 0, 1), ring)
            # specular highlight (upper-left)
            hl = np.exp(-(((radial(Hb, Wb, cxo - gem_r * 0.35, cy - gem_r * 0.38, gem_r * 0.32)) ** 2)) * 1.4)
            rgb = over(rgb, np.ones_like(rgb) * np.array([1.0, 0.95, 0.9])[None, None, :], np.clip(hl, 0, 1) * m)
    # tiny white diamond rim on the gold border (top + bottom edges)
    dia_r = 1.7 * S
    nd = 11
    dxs = np.linspace(pad + 7 * S, Wb - pad - 7 * S, nd)
    for cy in [pad + 3.2 * S, Hb - pad - 3.2 * S]:
        for cx in dxs:
            rr = radial(Hb, Wb, cx, cy, dia_r)
            m = np.clip((1 - rr) * dia_r, 0, 1)
            col = np.array([0.95, 0.94, 0.88])[None, None, :] * (0.7 + 0.5 * np.clip(1 - rr, 0, 1))[..., None]
            rgb = over(rgb, np.clip(col, 0, 1), m)
    # centre box-clasp seam line
    seam = mask_from_pil_draw(Hb, Wb, lambda d, s: d.rectangle(
        [(Wb / 2 - 0.5 * S) * s, (pad + 1 * S) * s, (Wb / 2 + 0.5 * S) * s, (Hb - pad - 1 * S) * s], fill=255), supersample=2)
    rgb = over(rgb, np.clip(gold * 0.35, 0, 1), seam * (1 - chan) * 0.8)
    # subtle sparkle across the gold
    sp = sparkle(Hb, Wb, density=0.0015, seed=seed + 9, size=1.6)
    rgb = np.clip(rgb + sp[..., None] * np.array([1.0, 0.97, 0.85])[None, None, :] * 0.8 * alpha[..., None], 0, 1)
    el = rgba(rgb, alpha)
    im = to_pil(el).resize((w, h), Image.LANCZOS)
    return from_pil(im)

# ------------------------------------------------------------------ assemble
canvas = np.zeros((H, W, 4), np.float32)

# PENDANT --------------------------------------------------------------
pendant = load_rgba(os.path.join(EL, 'pendant.png'))
px0, py0, px1, py1 = alpha_bbox(pendant)
pw, ph = px1 - px0 + 1, py1 - py0 + 1
scale = PENDANT_W / pw
# keep the pendant bottom above PENDANT_MAX_BOTTOM (scale down slightly if needed)
elem_cy = (py0 + py1) / 2 - pendant.shape[0] / 2          # bbox centre offset from element centre
bottom = PENDANT_CY + (ph / 2 + elem_cy) * scale
if bottom > PENDANT_MAX_BOTTOM - 4:
    scale = (PENDANT_MAX_BOTTOM - 4 - PENDANT_CY) / (ph / 2 + elem_cy)
# place so that the bbox centre (not the element canvas centre) sits on the target
elem_cx = (px0 + px1) / 2 - pendant.shape[1] / 2
canvas = paste_rgba_on_rgba(canvas, pendant, PENDANT_CX - elem_cx * scale, PENDANT_CY - elem_cy * scale, scale=scale)
pend_box = (PENDANT_CX - pw * scale / 2, PENDANT_CY - ph * scale / 2, PENDANT_CX + pw * scale / 2, PENDANT_CY + ph * scale / 2)
print('pendant scale %.4f -> %.0f x %.0f px, bbox y %.0f..%.0f' % (scale, pw * scale, ph * scale, pend_box[1], pend_box[3]))

# CHAIN band ------------------------------------------------------------
strip = load_rgba(os.path.join(EL, 'chain_strip.png'))
sh, sw = strip.shape[:2]
sx0, sy0, sx1, sy1 = alpha_bbox(strip)
content_h = sy1 - sy0 + 1
band_h = CHAIN_BOTTOM - CHAIN_TOP
# 1:1 horizontally (the strip is exactly 1024 wide and wraps); tile if narrower
if sw != W:
    reps = int(np.ceil(W / sw))
    strip = np.tile(strip, (1, reps, 1))[:, :W]
# vertical placement: centre the opaque content in the band
strip_cy = CHAIN_TOP + band_h / 2 - ((sy0 + sy1) / 2 - sh / 2)
if content_h > band_h:
    print('WARNING: chain content taller than band by %d px (will clip at bottom)' % (content_h - band_h))
canvas = paste_rgba_on_rgba(canvas, strip, W / 2, strip_cy, scale=1.0)
chain_box = (0, CHAIN_TOP + (band_h - content_h) / 2, W, CHAIN_TOP + (band_h + content_h) / 2)
print('chain strip content %d px tall, placed y %.0f..%.0f' % (content_h, chain_box[1], chain_box[3]))

# CLASP ----------------------------------------------------------------
clasp = make_clasp()
canvas = paste_rgba_on_rgba(canvas, clasp, CLASP_CX, CLASP_CY, scale=1.0)
clasp_box = (CLASP_CX - CLASP_W / 2, CLASP_CY - CLASP_H / 2, CLASP_CX + CLASP_W / 2, CLASP_CY + CLASP_H / 2)

# overlap sanity checks
assert pend_box[3] < clasp_box[1], 'pendant overlaps clasp'
assert clasp_box[3] < chain_box[1], 'clasp overlaps chain'

# ------------------------------------------------------------------ outputs
alpha = canvas[..., 3]
rgb = canvas[..., :3]
# alpha bleed: give transparent pixels the colour of the nearest opaque pixel so bilinear /
# mip filtering in-game does not pull dark or grey halos around the pendant and chain edges.
# The bleed is limited to BLEED_PX (fading to black over the next BLEED_FADE px) so the file
# still reads cleanly in viewers that ignore alpha.
from scipy import ndimage as _ndi
BLEED_PX, BLEED_FADE = 16, 8
opaque = alpha > 0.5
dist, idx = _ndi.distance_transform_edt(~opaque, return_indices=True)
bleed = rgb[idx[0], idx[1]]
keep = np.clip((BLEED_PX + BLEED_FADE - dist) / BLEED_FADE, 0, 1)[..., None]
w_op = np.clip(alpha / 0.5, 0, 1)[..., None]
rgb_out = np.where(opaque[..., None], rgb, (rgb * w_op + bleed * (1 - w_op)) * keep)
chain_rgba = np.dstack([rgb_out, alpha])
Image.fromarray((np.clip(chain_rgba, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(os.path.join(FINAL, 'chain.png'))

bg = black_fabric(H, W, seed=3)
on_black = over(bg, rgb, alpha)
save(on_black, os.path.join(FINAL, 'chain_on_black.png'))

# ------------------------------------------------------------------ layout guide
guide = to_pil(on_black).convert('RGBA')
ov = Image.new('RGBA', guide.size, (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except Exception:
    font = font_s = ImageFont.load_default()

def label_box(box, name, color, sub=None, label_at=None):
    """Outline + tint a region and put its label (with dark backing) at label_at (default: inside top-left)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color + (255,), width=3)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color + (28,))
    tx, ty = label_at if label_at else (x0 + 8, y0 + 6)
    tw = d.textlength(name, font=font)
    sw = d.textlength(sub, font=font_s) if sub else 0
    bw = max(tw, sw) + 10
    bh = 30 + (22 if sub else 0)
    d.rectangle([tx - 4, ty - 2, tx + bw, ty + bh], fill=(0, 0, 0, 210))
    d.text((tx, ty), name, font=font, fill=color + (255,))
    if sub:
        d.text((tx, ty + 32), sub, font=font_s, fill=(255, 255, 255, 235))

CYAN, LIME, MAG = (0, 220, 255), (150, 255, 60), (255, 90, 255)
label_box(pend_box, 'PENDANT', CYAN, 'pendant.png  scale %.3f  centre (512,400)  %dx%d px  y %d..%d' % (scale, pw * scale, ph * scale, pend_box[1], pend_box[3]))
label_box(chain_box, 'CHAIN LINK TILE', LIME, 'chain_strip.png 1:1, 1024 wide, wraps left/right, y %d..%d' % (chain_box[1], chain_box[3]))
# clasp is tiny: outline it and put the label beside it, with a leader line
label_box(clasp_box, 'CLASP', MAG, '90x40 gold box, red pave, centre (512,765)', label_at=(int(clasp_box[2]) + 40, int(clasp_box[1]) - 12))
d.line([int(clasp_box[2]) + 2, int(clasp_box[1]) + CLASP_H // 2, int(clasp_box[2]) + 36, int(clasp_box[1]) + 4], fill=MAG + (255,), width=2)
# chain loop seams (3 loops per strip) as dashed lines
for k in range(1, 3):
    x = int(round(W * k / 3))
    for yy in range(int(chain_box[1]), int(chain_box[3]), 12):
        d.line([x, yy, x, yy + 6], fill=(255, 255, 255, 140), width=1)
d.rectangle([4, int(chain_box[1]) - 26, 330, int(chain_box[1]) - 4], fill=(0, 0, 0, 210))
d.text((8, int(chain_box[1]) - 24), 'tile period 1024 px (3 loops of 341.3 px)', font=font_s, fill=LIME + (255,))
d.rectangle([4, 4, 380, 28], fill=(0, 0, 0, 210))
# canvas info
d.text((8, 8), 'ATA chain texture 1024x1024 RGBA (component 7 teef)', font=font_s, fill=(255, 255, 255, 230))
guide = Image.alpha_composite(guide, ov).convert('RGB')
guide.save(os.path.join(FINAL, 'chain_layout_guide.png'))

# ------------------------------------------------------------------ 2x zoom crops for review
zoom = to_pil(on_black).crop((362, 700, 662, 900)).resize((600, 400), Image.NEAREST)
zoom.save(os.path.join(FINAL, '_zoom_chain_clasp.png'))
zoom2 = to_pil(on_black).crop((100, 480, 400, 720)).resize((600, 480), Image.NEAREST)
zoom2.save(os.path.join(FINAL, '_zoom_chain_pendant.png'))
# wrap-seam check: right 150 px of the band next to the left 150 px, 2x
ob = to_pil(on_black)
seam = Image.new('RGB', (300, 234))
seam.paste(ob.crop((W - 150, CHAIN_TOP, W, CHAIN_BOTTOM)), (0, 0))
seam.paste(ob.crop((0, CHAIN_TOP, 150, CHAIN_BOTTOM)), (150, 0))
seam.resize((600, 468), Image.NEAREST).save(os.path.join(FINAL, '_zoom_chain_seam.png'))

print('done in %.1fs' % (time.time() - t0))
