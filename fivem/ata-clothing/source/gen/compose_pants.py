"""Compose the ATA jeans texture (FiveM component 4 'lowr') from pre-made elements.

Layout: four 256 px wide leg panels (FRONT L, FRONT R, BACK L, BACK R) on a 1024x1024 sheet.
Run from the work dir:  python3 gen/compose_pants.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ata_style import *

T0 = time.time()
WORK = '.'
EL = WORK + '/elements'
FINAL = WORK + '/final'
os.makedirs(FINAL, exist_ok=True)

W = H = 1024
PANEL = 256
STRIPE_W = 34
WAIST_H = 14
CUFF_Y = 984           # cuff strip occupies 984..1024 (40 px = strip scaled 0.25)

# ----------------------------------------------------------------- helpers
def load_el(name, fallbacks=()):
    for n in (name,) + tuple(fallbacks):
        p = f'{EL}/{n}.png'
        if os.path.exists(p):
            return from_pil(Image.open(p).convert('RGBA'))
    raise FileNotFoundError(name)

def crop_bbox(el, thr=8 / 255, pad=2):
    a = el[..., 3]
    ys, xs = np.where(a > thr)
    y0 = max(0, ys.min() - pad); y1 = min(el.shape[0], ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad); x1 = min(el.shape[1], xs.max() + 1 + pad)
    return el[y0:y1, x0:x1]

def resize_rgba(el, w, h):
    im = to_pil(el).resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)
    return from_pil(im)

def fit(el, w=None, h=None):
    """Crop element to its alpha bbox and resize the CONTENT to w x h (uniform if one is given)."""
    c = crop_bbox(el)
    ch, cw = c.shape[:2]
    if w is None: w = cw * h / ch
    if h is None: h = ch * w / cw
    return resize_rgba(c, w, h)

def composite(dst, el, cx, cy, shadow=0.6, shadow_r=3.0):
    """Paste RGBA element centred at (cx, cy) with a soft dark halo under it (embroidery on fabric)."""
    out = dst.copy()
    Hd, Wd = dst.shape[:2]; h, w = el.shape[:2]
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0); dx1 = min(Wd, x0 + w); dy1 = min(Hd, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return out
    sub = el[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    if shadow > 0:
        m = int(shadow_r * 3) + 3
        X0 = max(0, dx0 - m); Y0 = max(0, dy0 - m); X1 = min(Wd, dx1 + m); Y1 = min(Hd, dy1 + m)
        a = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        a[dy0 - Y0:dy1 - Y0, dx0 - X0:dx1 - X0] = sub[..., 3]
        sh = np.clip(blur(dilate(a, 1.5), shadow_r) * 1.6, 0, 1) * shadow
        out[Y0:Y1, X0:X1] *= (1 - sh[..., None])
    out[dy0:dy1, dx0:dx1] = over(out[dy0:dy1, dx0:dx1], sub[..., :3], sub[..., 3])
    return out

def meander_band(length, cell=2):
    """Crisp pixel-aligned Greek-key band mask, 7*cell tall x `length` wide, baseline on top row."""
    n = int(np.ceil(length / (8 * cell))) + 1
    m = np.zeros((7, 8 * n), np.float32)
    for u in range(n):
        x = 8 * u
        m[0, x:x + 8] = 1        # continuous baseline
        m[0:7, x] = 1            # outer left
        m[6, x:x + 7] = 1        # outer bottom
        m[2:7, x + 6] = 1        # outer right
        m[2, x + 2:x + 7] = 1    # inner top
        m[2:5, x + 2] = 1        # inner left
        m[4, x + 2:x + 5] = 1    # inner bottom (spiral end)
    m = np.kron(m, np.ones((cell, cell), np.float32))
    return m[:, :int(length)]

def darken_disc(dst, cx, cy, r, amount=0.55, feather=1.5):
    """Darken a soft-edged disc of the ground (velvet interior of a medallion)."""
    out = dst.copy()
    x0 = int(cx - r - 4); y0 = int(cy - r - 4); x1 = int(cx + r + 5); y1 = int(cy + r + 5)
    Y, X = np.mgrid[y0:y1, x0:x1]
    m = np.clip((r - np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)) / feather + 0.5, 0, 1)
    out[y0:y1, x0:x1] *= (1 - amount * m[..., None])
    return out

def darken_rrect(dst, cx, cy, w, h, radius, amount=0.3):
    out = dst.copy()
    m = mask_from_pil_draw(h, w, lambda d, S: d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=radius * S, fill=255))
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    out[y0:y0 + h, x0:x0 + w] *= (1 - amount * m[..., None])
    return out

def gold_element(mask, seed=0, depth=1.2, strength=0.3, glitter=0.6):
    h, w = mask.shape
    return emboss_element(mask, gold_fill(h, w, seed=seed, glitter=glitter), depth=depth, strength=strength)

# ----------------------------------------------------------------- elements
base     = from_pil(Image.open(f'{EL}/base_tile.png').convert('RGB'))
logo     = load_el('ata_logo')
letters  = load_el('ata_letters')
medusa   = load_el('medusa')
filigree = load_el('filigree', ('filigree_b', 'filigree_a'))
gk_h     = load_el('greek_key_strip')
gk_v     = load_el('greek_key_strip_v')

# cuff: 1024x160 strip scaled exactly 0.25 -> 256x40 (one panel = one full period set)
cuff = resize_rgba(gk_h, 256, 40)
# vertical side stripe: 160x1024 -> 34 wide, tiled over the full height
sv = resize_rgba(gk_v, STRIPE_W, round(1024 * STRIPE_W / 160))
reps = int(np.ceil(H / sv.shape[0])) + 1
stripe_col = np.concatenate([sv] * reps, axis=0)[:H]

# front pieces
logo_f    = fit(logo, w=150)
medusa_f  = fit(medusa, w=170)
fili_f    = fit(filigree, w=180, h=280)       # ornament is stretched ~1.24x vertically to fill the shin
# back pieces
logo_b    = fit(logo, w=140)
medusa_b  = fit(medusa, w=160)
fili_b    = fit(filigree, w=180, h=280)
letters_b = fit(letters, w=70)

# waistband: gold band with two dashed stitch lines
def make_waistband(seed=5):
    g = gold_fill(WAIST_H, W, seed=seed, glitter=0.5)
    shade = np.ones((WAIST_H, 1), np.float32)
    shade[0] = 0.55; shade[1] = 1.12; shade[WAIST_H - 2] = 0.85; shade[WAIST_H - 1] = 0.5
    g = g * shade[..., None]
    dash = (np.arange(W) % 10) < 6
    for row in (4, WAIST_H - 5):
        g[row, dash] *= 0.55
    return np.clip(g, 0, 1)

waistband = make_waistband()

# front pocket edge: quarter-circle scoop centred on the top-outer corner + rivets
def make_pocket_arc(outer_left=True):
    w, h = 160, 130
    R = 100
    def draw(d, S):
        if outer_left:
            c = (STRIPE_W, 0); a0, a1 = 0, 90
        else:
            c = (w - STRIPE_W, 0); a0, a1 = 90, 180
        for r, lw in ((R, 5), (R - 9, 2)):
            bb = [(c[0] - r) * S, (c[1] - r) * S, (c[0] + r) * S, (c[1] + r) * S]
            d.arc(bb, a0, a1, fill=255, width=int(lw * S))
        for ang in (7, 83):
            t = np.deg2rad(ang if outer_left else 180 - ang)
            px, py = c[0] + R * np.cos(t), c[1] + R * np.sin(t)
            rr = 4.5
            d.ellipse([(px - rr) * S, (py - rr) * S, (px + rr) * S, (py + rr) * S], fill=255)
    m = mask_from_pil_draw(h, w, draw)
    return gold_element(m, seed=11, depth=1.5, strength=0.35)

pocket_arc_L = make_pocket_arc(True)
pocket_arc_R = make_pocket_arc(False)

# back pocket: rounded rectangle with a crisp Greek-key border and ATA letters inside
def make_back_pocket(pw=120, ph=110, band=14, seed=21):
    cell = 2
    b = np.zeros((ph, pw), np.float32)
    top = meander_band(pw, cell)                 # baseline on top (outer edge)
    b[0:band, :] = top
    b[ph - band:ph, :] = top[::-1, :]            # baseline at the bottom (outer edge)
    side = meander_band(ph, cell)
    b[:, 0:band] = side.T                        # baseline on the left (outer edge)
    b[:, pw - band:pw] = side.T[:, ::-1]
    # re-apply horizontal bands so corners read as one clean band
    b[0:band, :] = top
    b[ph - band:ph, :] = top[::-1, :]
    outer = mask_from_pil_draw(ph, pw, lambda d, S: d.rounded_rectangle([0, 0, pw * S - 1, ph * S - 1], radius=12 * S, fill=255))
    inner = mask_from_pil_draw(ph, pw, lambda d, S: d.rounded_rectangle([band * S, band * S, (pw - band) * S - 1, (ph - band) * S - 1], radius=4 * S, fill=255))
    ring = np.clip(outer - inner, 0, 1)
    edge_out = np.clip(outer - erode(outer, 1.6), 0, 1)
    edge_in = np.clip(dilate(inner, 1.6) - inner, 0, 1)
    m = np.clip(b * ring + edge_out + edge_in, 0, 1)
    return gold_element(m, seed=seed, depth=1.2, strength=0.3)

back_pocket = make_back_pocket()

# ----------------------------------------------------------------- compose
img = base.copy()
FRONT = [0, 1]; BACK = [2, 3]

for p in range(4):
    x0 = p * PANEL; cx = x0 + PANEL // 2
    outer_left = (p % 2 == 0)              # left legs: outer edge is the panel's left edge
    # vertical Greek-key side-seam stripe on the outer edge (waistband + cuff painted over it later)
    sx = x0 if outer_left else x0 + PANEL - STRIPE_W
    st = stripe_col
    sh = np.clip(blur(dilate(st[..., 3], 1.5), 3.0) * 1.6, 0, 1) * 0.5
    img[:, sx:sx + STRIPE_W] *= (1 - sh[..., None])
    img[:, sx:sx + STRIPE_W] = over(img[:, sx:sx + STRIPE_W], st[..., :3], st[..., 3])

    if p in FRONT:
        arc = pocket_arc_L if outer_left else pocket_arc_R
        acx = x0 + 80 if outer_left else x0 + PANEL - 80
        img = composite(img, arc, acx, 65, shadow=0.5)
        img = composite(img, logo_f, cx, 300, shadow=0.65)
        img = darken_disc(img, cx, 555, 78, amount=0.6)
        img = composite(img, medusa_f, cx, 555, shadow=0.6)
        img = composite(img, fili_f, cx, 800, shadow=0.6)
    else:
        img = darken_rrect(img, cx, 115, 120, 110, 12, amount=0.35)
        img = composite(img, back_pocket, cx, 115, shadow=0.55)
        img = composite(img, letters_b, cx, 115, shadow=0.6)
        img = composite(img, logo_b, cx, 330, shadow=0.65)
        img = darken_disc(img, cx, 570, 73, amount=0.6)
        img = composite(img, medusa_b, cx, 570, shadow=0.6)
        img = composite(img, fili_b, cx, 805, shadow=0.6)

    # Greek-key cuff at the hem
    img = composite(img, cuff, cx, CUFF_Y + 20, shadow=0.5)

# waistband over everything
img[0:WAIST_H, :] = waistband
img[WAIST_H, :] *= 0.6   # seam shadow under the band

# subtle 1 px seam lines between panels
for x in (256, 512, 768):
    img[:, x] *= (1 - 0.35)

img = np.clip(img, 0, 1)
out_pil = to_pil(img).convert('RGB')
assert out_pil.size == (1024, 1024)
out_pil.save(f'{FINAL}/pants.png')

# ----------------------------------------------------------------- layout guide
def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
        except Exception:
            return ImageFont.load_default()

guide = out_pil.convert('RGBA')
ov = Image.new('RGBA', guide.size, (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
f_big = get_font(30); f_mid = get_font(18); f_small = get_font(14)
panel_names = ['FRONT L', 'FRONT R', 'BACK L', 'BACK R']
panel_cols = [(80, 160, 255), (80, 220, 160), (255, 160, 60), (230, 90, 230)]

def label(xy, text, font, fill=(255, 255, 255, 255), bg=(0, 0, 0, 170), anchor='mm'):
    bb = d.textbbox(xy, text, font=font, anchor=anchor)
    d.rectangle([bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2], fill=bg)
    d.text(xy, text, font=font, fill=fill, anchor=anchor)

for p in range(4):
    x0 = p * PANEL; cx = x0 + 128
    col = panel_cols[p]
    d.rectangle([x0, 0, x0 + PANEL - 1, H - 1], fill=col + (45,), outline=col + (255,), width=2)
    outer_left = (p % 2 == 0)
    sx = x0 if outer_left else x0 + PANEL - STRIPE_W
    d.rectangle([sx, WAIST_H, sx + STRIPE_W - 1, CUFF_Y - 1], fill=(255, 230, 0, 70), outline=(255, 230, 0, 255), width=1)
    d.rectangle([x0, CUFF_Y, x0 + PANEL - 1, H - 1], fill=(255, 60, 60, 80), outline=(255, 60, 60, 255), width=1)
    d.rectangle([x0, 0, x0 + PANEL - 1, WAIST_H - 1], fill=(255, 255, 255, 60), outline=(255, 255, 255, 200), width=1)
    label((cx, 200 if p in FRONT else 215), panel_names[p], f_big)
    label((cx, CUFF_Y + 20), 'CUFF (greek key)', f_small)
    label((cx, 7), 'WAISTBAND', f_small)
    # vertical stripe label
    txt = Image.new('RGBA', (300, 22), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text((150, 11), 'SIDE STRIPE (outer seam)', font=f_small, fill=(255, 255, 255, 255), anchor='mm')
    txt = txt.rotate(90, expand=True)
    ov.alpha_composite(txt, (sx + 6, 460))
    if p in FRONT:
        boxes = [((cx - 75, 224, cx + 75, 376), 'CROWN+ATA logo'),
                 ((cx - 85, 470, cx + 85, 640), 'MEDUSA'),
                 ((cx - 90, 660, cx + 90, 940), 'FILIGREE'),
                 ((x0 + 34, 14, x0 + 140, 120) if outer_left else (x0 + 116, 14, x0 + 222, 120), 'POCKET EDGE')]
    else:
        boxes = [((cx - 60, 60, cx + 60, 170), 'BACK POCKET + ATA'),
                 ((cx - 70, 259, cx + 70, 401), 'CROWN+ATA logo'),
                 ((cx - 80, 490, cx + 80, 650), 'MEDUSA'),
                 ((cx - 90, 665, cx + 90, 945), 'FILIGREE')]
    for (bx0, by0, bx1, by1), name in boxes:
        d.rectangle([bx0, by0, bx1, by1], outline=(255, 255, 255, 220), width=1)
        label(((bx0 + bx1) // 2, by1 - 10), name, f_small)

guide = Image.alpha_composite(guide, ov).convert('RGB')
guide.save(f'{FINAL}/pants_layout_guide.png')
print('done in %.1f s' % (time.time() - T0), out_pil.size, out_pil.mode)
