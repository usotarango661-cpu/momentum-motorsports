"""Compose the final 1024x1024 ATA jacket / hoodie texture (FiveM component 11 'jbib')
from the pre-made elements in work/elements.

Run from work/:  python3 gen/compose_jacket.py

Layout (all coordinates in pixels on the 1024x1024 canvas):
  FRONT        (0,0)-(512,512)         crown+ATA logo, gold zipper placket, flanking filigree, Greek key hem
  BACK         (512,0)-(1024,512)      crown+ATA logo, Medusa medallion, flanking filigree, Greek key hem
  LEFT SLEEVE  (0,512)-(256,1024)      small logo at shoulder, tall filigree, Greek key cuff
  RIGHT SLEEVE (256,512)-(512,1024)    same as left sleeve
  HOOD         (512,512)-(1024,1024)   scattered ATA letter logos, Greek key waistband
  LABEL        (934,934)-(1024,1024)   solid black square with small gold ATA logo
"""
import os, sys, time
WORK = '.'
sys.path.insert(0, WORK)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from ata_style import *

T0 = time.time()
EL = os.path.join(WORK, 'elements')
FINAL = os.path.join(WORK, 'final')
os.makedirs(FINAL, exist_ok=True)

# ---------------------------------------------------------------- helpers (local only)
def load_rgba(name):
    p = os.path.join(EL, name)
    im = Image.open(p).convert('RGBA')
    return from_pil(im)

def bleed(el, thresh=0.02):
    """Fill RGB under (near) transparent pixels with the nearest opaque colour so that
    LANCZOS / bicubic resampling in paste_rgba never produces dark or wrong-coloured fringes."""
    a = el[..., 3]
    opaque = a > thresh
    if opaque.all():
        return el
    idx = ndimage.distance_transform_edt(~opaque, return_distances=False, return_indices=True)
    rgb = el[..., :3][idx[0], idx[1]]
    out = el.copy()
    out[..., :3] = rgb
    return out

def crop_content(el, pad=4):
    """Crop an RGBA element to its alpha bounding box (+pad) so paste_rgba centres the CONTENT."""
    a = el[..., 3] > 0.02
    ys, xs = np.where(a)
    y0, y1 = max(0, ys.min() - pad), min(el.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(el.shape[1], xs.max() + 1 + pad)
    return el[y0:y1, x0:x1]

def prep(name, pad=4):
    return bleed(crop_content(load_rgba(name), pad))

def mirror(el):
    return np.ascontiguousarray(el[:, ::-1])

def content_w(el):
    return el.shape[1] - 8  # crop pad 4 on both sides

def strip_band(strip, x0, y0, width, height, dst):
    """Uniformly scale the (tileable, 128 px period) Greek key strip to `height` px tall,
    tile it horizontally to `width` and paste with the top-left at (x0,y0)."""
    im = to_pil(bleed(strip))
    s = height / im.height
    im = im.resize((max(1, int(round(im.width * s))), height), Image.LANCZOS)
    a = from_pil(im)
    reps = width // a.shape[1] + 2
    big = np.tile(a, (1, reps, 1))[:, :width]
    out = dst.copy()
    H, W = dst.shape[:2]
    y1 = min(H, y0 + height); x1 = min(W, x0 + width)
    sub = big[:y1 - y0, :x1 - x0]
    out[y0:y1, x0:x1] = over(out[y0:y1, x0:x1], sub[..., :3], sub[..., 3])
    return out


def darken_disc(dst, cx, cy, r, amount=0.65, feather=2.0):
    """Darken a soft-edged disc of the ground so a medallion's line-art reads on black velvet."""
    out = dst.copy()
    x0 = int(cx - r - 4); y0 = int(cy - r - 4); x1 = int(cx + r + 5); y1 = int(cy + r + 5)
    Y, X = np.mgrid[y0:y1, x0:x1]
    m = np.clip((r - np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)) / feather + 0.5, 0, 1)
    out[y0:y1, x0:x1] *= (1 - amount * m[..., None])
    return out

def make_zipper(w=18, h=350, seed=7):
    """Gold zipper / placket strip RGBA: two rows of interlocking teeth, dark centre seam,
    a slider near the top with a hanging pull tab."""
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    gold = gold_fill(h, w, seed=seed, glitter=0.6, thread=False)
    # teeth ridges: period 4 px, offset by 2 px between the two halves (interlocking)
    left = X < w // 2
    ridge = np.where(left, 0.5 + 0.5 * np.cos(2 * np.pi * Y / 4.0),
                           0.5 + 0.5 * np.cos(2 * np.pi * (Y + 2) / 4.0))
    gold = gold * (0.72 + 0.28 * ridge)[..., None]
    # dark centre seam + thin tape edges
    seam = ((X == w // 2 - 1) | (X == w // 2)).astype(np.float32)
    edge = ((X == 0) | (X == w - 1)).astype(np.float32)
    gold = gold * (1 - 0.6 * seam - 0.45 * edge)[..., None]
    mask = np.ones((h, w), np.float32)
    # rounded ends
    mask = mask_from_pil_draw(h, w, lambda d, S: d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=6 * S, fill=255))
    zip_el = emboss_element(mask, gold, depth=1.2, strength=0.25)
    # slider body (rounded rect) + pull tab
    sy = 14
    slider_mask = mask_from_pil_draw(h, w, lambda d, S: d.rounded_rectangle(
        [1 * S, sy * S, (w - 2) * S, (sy + 22) * S], radius=4 * S, fill=255))
    pull_mask = mask_from_pil_draw(h, w, lambda d, S: (
        d.rounded_rectangle([(w // 2 - 3) * S, (sy + 20) * S, (w // 2 + 2) * S, (sy + 48) * S], radius=2 * S, fill=255),
        d.ellipse([(w // 2 - 4) * S, (sy + 40) * S, (w // 2 + 3) * S, (sy + 50) * S], fill=255)))
    sl_fill = np.clip(gold_fill(h, w, seed=seed + 1, glitter=0.3, thread=False) * 1.08, 0, 1)
    slider = emboss_element(np.clip(slider_mask + pull_mask, 0, 1), sl_fill, depth=1.5, strength=0.45)
    rgb = over(zip_el[..., :3], slider[..., :3], slider[..., 3])
    alpha = np.clip(zip_el[..., 3] + slider[..., 3], 0, 1)
    return bleed(rgba(rgb, alpha))

def gold_logo(logo_rgba, width, seed=3):
    """All-gold embossed version of the crown+ATA logo (for the inner label), `width` px wide."""
    a = crop_content(logo_rgba, 2)[..., 3]
    # build at 2x then downscale for crisp edges
    s2 = (width * 2) / a.shape[1]
    m = from_pil(to_pil(a).resize((int(a.shape[1] * s2), int(a.shape[0] * s2)), Image.LANCZOS))
    m = np.clip((m - 0.5) * 1.6 + 0.5, 0, 1)
    h, w = m.shape
    g = gold_fill(h, w, seed=seed, glitter=0.5, thread=True)
    edge = np.zeros((h, w, 3), np.float32) + np.array(GOLD_DARK, np.float32)[None, None, :] * 0.8
    el = emboss_element(m, g, edge_width=2, edge_rgb=edge, depth=2.0, strength=0.45)
    el = bleed(el)
    im = to_pil(el).resize((w // 2, h // 2), Image.LANCZOS)
    return from_pil(im)

# ---------------------------------------------------------------- load elements
base = from_pil(Image.open(os.path.join(EL, 'base_tile.png')).convert('RGB'))
logo = prep('ata_logo.png')           # crown + ATA (content 869x884)
letters = prep('ata_letters.png')     # ATA letters only
medusa = prep('medusa.png')           # round medallion
fil_pair = prep('filigree.png')       # mirrored scroll pair (735x924 content)
fil_single = prep('filigree_single.png')  # left half of the pair (367x924 content)
greek = load_rgba('greek_key_strip.png')  # 1024x160, period 128 px

LOGO_W = 869.0; LET_W = 869.0; MED_D = 963.0; FILP_W = 735.0; FILS_W = 367.0

canvas = tile_to(base, 1024, 1024)

# ---------------------------------------------------------------- FRONT (0,0)-(512,512)
# flanking filigree first (drawn under the logo), mirrored pair facing the centre
s_fs = 0.30
canvas = paste_rgba(canvas, fil_single, 60, 305, scale=s_fs)
canvas = paste_rgba(canvas, mirror(fil_single), 452, 305, scale=s_fs)
# crown + ATA logo, letters ~300 px wide, centred (256,215), and the gold zipper placket
# down the centre (x=256, y 120..470).  The reference jacket shows the zipper running over
# the logo, so it is drawn on top by default; set ZIPPER_OVER_LOGO = False to keep the logo intact.
ZIPPER_OVER_LOGO = False
zipper = make_zipper(18, 350)
if not ZIPPER_OVER_LOGO:
    canvas = paste_rgba(canvas, zipper, 256, 295)
canvas = paste_rgba(canvas, logo, 256, 215, scale=300 / LOGO_W)
if ZIPPER_OVER_LOGO:
    canvas = paste_rgba(canvas, zipper, 256, 295)
# Greek key hem: 11 periods across 512 px -> 58 px tall, y 454..512
canvas = strip_band(greek, 0, 454, 512, 58, canvas)

# ---------------------------------------------------------------- BACK (512,0)-(1024,512)
s_bs = 0.28
canvas = paste_rgba(canvas, fil_single, 578, 318, scale=s_bs)
canvas = paste_rgba(canvas, mirror(fil_single), 958, 318, scale=s_bs)
canvas = paste_rgba(canvas, logo, 768, 146, scale=265 / LOGO_W)
canvas = darken_disc(canvas, 768, 369, 78, amount=0.65)
canvas = paste_rgba(canvas, medusa, 768, 369, scale=160 / MED_D)
canvas = strip_band(greek, 512, 454, 512, 58, canvas)

# ---------------------------------------------------------------- SLEEVES (0,512)-(512,1024)
for cx in (128, 384):
    canvas = paste_rgba(canvas, logo, cx, 562, scale=90 / LOGO_W)
    canvas = paste_rgba(canvas, fil_pair, cx, 775, scale=225 / FILP_W)
canvas = strip_band(greek, 0, 984, 256, 40, canvas)
canvas = strip_band(greek, 256, 984, 256, 40, canvas)

# ---------------------------------------------------------------- HOOD (512,512)-(1024,1024)
scatter = [  # (cx, cy, rot)
    (684, 605, -8), (852, 605, 7),
    (600, 735, 5), (768, 735, -6), (936, 735, 8),
    (684, 865, 6), (852, 865, -7),
]
for cx, cy, rot in scatter:
    canvas = paste_rgba(canvas, letters, cx, cy, scale=84 / LET_W, rotate=rot)
# waistband
canvas = strip_band(greek, 512, 984, 512, 40, canvas)
# LABEL square (934..1024) solid black with a small gold logo
canvas[934:1024, 934:1024] = np.array(BLACK, np.float32)[None, None, :] * 0.6
glogo = gold_logo(logo, 68)
canvas = paste_rgba(canvas, glogo, 979, 979)

# ---------------------------------------------------------------- subtle seam lines (alpha 0.35)
dark = np.zeros((1024, 1024, 3), np.float32)
seam = np.zeros((1024, 1024), np.float32)
seam[:, 512] = 1; seam[512, :] = 1; seam[512:, 256] = 1
canvas = over(canvas, dark, seam * 0.35)

canvas = np.clip(canvas, 0, 1)
out_path = os.path.join(FINAL, 'jacket.png')
to_pil(canvas).convert('RGB').save(out_path)
print('wrote', out_path, 'in %.1fs' % (time.time() - T0))

# ---------------------------------------------------------------- layout guide
guide = to_pil(canvas).convert('RGBA')
ov = Image.new('RGBA', guide.size, (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    font_s = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception:
    font = font_s = ImageFont.load_default()
regions = [
    ('FRONT', (0, 0, 512, 512), (40, 120, 255)),
    ('BACK', (512, 0, 1024, 512), (40, 200, 90)),
    ('LEFT SLEEVE', (0, 512, 256, 1024), (255, 150, 30)),
    ('RIGHT SLEEVE', (256, 512, 512, 1024), (190, 60, 220)),
    ('HOOD', (512, 512, 1024, 1024), (30, 200, 220)),
    ('LABEL', (934, 934, 1024, 1024), (255, 230, 40)),
]
for name, (x0, y0, x1, y1), col in regions:
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=col + (70,), outline=col + (230,), width=3)
for name, (x0, y0, x1, y1), col in regions:
    f = font_s if name == 'LABEL' else font
    tw = d.textlength(name, font=f)
    if name == 'LABEL':
        tx, ty = x0 + (x1 - x0 - tw) / 2, y0 + 4
    else:
        tx, ty = x0 + (x1 - x0 - tw) / 2, y0 + 12
    for ox, oy in ((-2, -2), (2, -2), (-2, 2), (2, 2), (0, 2), (2, 0), (-2, 0), (0, -2)):
        d.text((tx + ox, ty + oy), name, font=f, fill=(0, 0, 0, 255))
    d.text((tx, ty), name, font=f, fill=(255, 255, 255, 255))
guide = Image.alpha_composite(guide, ov).convert('RGB')
guide_path = os.path.join(FINAL, 'jacket_layout_guide.png')
guide.save(guide_path)
print('wrote', guide_path, 'total %.1fs' % (time.time() - T0))
