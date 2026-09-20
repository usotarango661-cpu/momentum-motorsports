#!/usr/bin/env python3
"""Build the preview contact sheet from the exported 1024x1024 PNG textures.

Usage: python3 make_preview.py <textures/png dir> <preview dir>
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

png_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
items = [('jacket_1024.png', 'JACKET  (jbib)'), ('pants_1024.png', 'PANTS  (lowr)'), ('chain_1024.png', 'CHAIN  (teef)')]
cell, pad, label_h = 512, 24, 44
sheet = Image.new('RGB', (pad + len(items) * (cell + pad), pad + cell + label_h + pad), (18, 18, 20))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except OSError:
    font = ImageFont.load_default()
for i, (name, label) in enumerate(items):
    path = os.path.join(png_dir, name)
    if not os.path.exists(path):
        continue
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (18, 18, 20, 255))
    bg.alpha_composite(im)
    thumb = bg.convert('RGB').resize((cell, cell), Image.LANCZOS)
    x = pad + i * (cell + pad)
    sheet.paste(thumb, (x, pad))
    d.text((x, pad + cell + 10), label, fill=(230, 200, 120), font=font)
sheet.save(os.path.join(out_dir, 'contact_sheet.png'))
print('wrote', os.path.join(out_dir, 'contact_sheet.png'))
