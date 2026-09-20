#!/usr/bin/env python3
"""Convert a PNG texture to a GTA V / FiveM friendly DDS with a full mipmap chain.

Usage:
    python3 export_dds.py input.png output.dds [--format DXT1|DXT5] [--no-mips]

* DXT1 (BC1) = opaque textures (jacket, pants).  4:1 compression, no alpha.
* DXT5 (BC3) = textures that need an alpha channel (chain / pendant cut-outs).
* Mipmaps are generated down to 1x1 with Lanczos filtering.  Alpha is
  pre-multiplied before downscaling so cut-out edges do not pick up dark fringes.

Only needs Pillow >= 11 (pip install pillow) and numpy.
"""
import argparse
import io
import struct
import sys

import numpy as np
from PIL import Image

DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDPF_FOURCC = 0x4
BLOCK_BYTES = {'DXT1': 8, 'DXT5': 16}


def block_data(im, fmt):
    """Compress one mip level with Pillow and return the raw block data (header stripped)."""
    buf = io.BytesIO()
    im.save(buf, format='DDS', pixel_format=fmt)
    raw = buf.getvalue()
    assert raw[:4] == b'DDS ' and struct.unpack('<I', raw[4:8])[0] == 124
    return raw[128:]


def mip_levels(im, with_mips=True):
    """Yield successive mip levels (Lanczos, premultiplied alpha)."""
    yield im
    if not with_mips:
        return
    arr = np.asarray(im).astype(np.float32) / 255.0
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        arr[..., :3] *= arr[..., 3:4]  # premultiply
    w, h = im.size
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        src = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        lvl = src.resize((w, h), Image.LANCZOS)
        a = np.asarray(lvl).astype(np.float32) / 255.0
        if has_alpha:
            out = a.copy()
            al = np.maximum(a[..., 3:4], 1e-4)
            out[..., :3] = np.clip(a[..., :3] / al, 0, 1)
            out[..., 3] = a[..., 3]
            yield Image.fromarray((out * 255 + 0.5).astype(np.uint8), 'RGBA')
        else:
            yield lvl
        arr = a


def write_dds(png_path, dds_path, fmt='DXT5', with_mips=True):
    im = Image.open(png_path)
    im = im.convert('RGBA' if fmt == 'DXT5' else 'RGB')
    w, h = im.size
    if (w & (w - 1)) or (h & (h - 1)):
        print(f'warning: {png_path} is {w}x{h}, not a power of two', file=sys.stderr)
    levels = list(mip_levels(im, with_mips))
    payload = b''.join(block_data(l, fmt) for l in levels)
    bb = BLOCK_BYTES[fmt]
    linear = max(1, w // 4) * max(1, h // 4) * bb
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    caps = DDSCAPS_TEXTURE
    if len(levels) > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    header = struct.pack('<4sIIIIIII11I', b'DDS ', 124, flags, h, w, linear, 0, len(levels), *([0] * 11))
    pixfmt = struct.pack('<II4sIIIII', 32, DDPF_FOURCC, fmt.encode(), 0, 0, 0, 0, 0)
    caps_block = struct.pack('<IIIII', caps, 0, 0, 0, 0)
    data = header + pixfmt + caps_block
    assert len(data) == 128
    with open(dds_path, 'wb') as f:
        f.write(data + payload)
    return w, h, len(levels), len(payload) + 128


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--format', choices=['DXT1', 'DXT5'], default='DXT5')
    ap.add_argument('--no-mips', action='store_true')
    a = ap.parse_args()
    w, h, n, size = write_dds(a.input, a.output, a.format, not a.no_mips)
    print(f'{a.output}: {w}x{h} {a.format} {n} mip levels, {size} bytes')


if __name__ == '__main__':
    main()
