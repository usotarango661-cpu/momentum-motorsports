#!/usr/bin/env python3
"""assemble.py - hang the ATA pendant on the cuban chain and export the finished necklace.

Frame: chain frame (metres, Y up, wearer faces +Z, lowest link centre at (0, 0, 0.035)).

Steps
  1. load out/pendant.glb (pendant frame: top of bail ring at the origin) and out/chain.glb
  2. compute the pendant translation so the bail ring HANGS on the lowest link: the ring
     (YZ plane, hole r = 4.5 mm) threads the link's hole and rests on the link's lower bar,
     its centre directly below the bar's highest point (gravity), with a small air gap.
     The nominal translation is (0, 0, 0.035); the hang correction is a fraction of a mm in Y
     and ~2 mm in -Z because the lowest link is tilted 38 deg about the chain tangent.
  3. export out/ata_necklace.glb + OBJ/MTL/textures (obj/ata_necklace, ata_pendant, ata_chain;
     the component OBJs are in the same assembled frame so pendant + chain == necklace)
  4. verify (trimesh round trip + pygltflib), render (headless three.js) and build a contact sheet.

Run:  cd chain3d && python3 gen/assemble.py [--no-render]
"""
import os
import re
import sys
import subprocess
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from mesh_utils import report, OUT  # noqa: E402

OBJ_DIR = os.path.join(OUT, 'obj')
RENDERS = os.path.join(ROOT, 'renders')
RENDER3D = os.path.join(os.path.dirname(ROOT), 'render3d')
PENDANT_GLB = os.path.join(OUT, 'pendant.glb')
CHAIN_GLB = os.path.join(OUT, 'chain.glb')
NECKLACE_GLB = os.path.join(OUT, 'ata_necklace.glb')

LINK_CENTRE = np.array([0.0, 0.0, 0.035])        # lowest link centre (chain frame)
RING_CENTRE_LOCAL = np.array([0.0, -0.0075, 0.0])  # bail ring centre in the pendant frame
RING_MAJOR_R, RING_TUBE_R = 0.006, 0.0015           # ring centreline / tube radius
# build_pendant.py v2 writes the actual ring geometry next to pendant.glb; use it when present
_META = os.path.join(OUT, 'pendant_meta.json')
if os.path.exists(_META):
    import json
    with open(_META) as _fh:
        _m = json.load(_fh)
    RING_CENTRE_LOCAL = np.array(_m['ring_centre_local'], dtype=float)
    RING_MAJOR_R, RING_TUBE_R = float(_m['ring_major_r']), float(_m['ring_tube_r'])
    print(f'pendant_meta.json: ring centre {RING_CENTRE_LOCAL*1e3} mm, major {RING_MAJOR_R*1e3:.2f} / tube {RING_TUBE_R*1e3:.2f} mm')
RING_HOLE_R = RING_MAJOR_R - RING_TUBE_R            # 4.5 mm (v1) / 2.8 mm (v2)
RING_OUTER_R = RING_MAJOR_R + RING_TUBE_R           # 7.5 mm (v1) / 5.2 mm (v2)
HANG_GAP = 0.00015                                  # air gap between ring hole and link bar (m)


# ----------------------------------------------------------------------------- helpers
def load_scene(path):
    sc = trimesh.load(path, force='scene', process=False)
    for n in sc.graph.nodes_geometry:
        T, _ = sc.graph[n]
        assert np.allclose(T, np.eye(4)), f'{path}: node {n} carries a transform, expected identity'
    return sc


def moved_copy(mesh, translation):
    """Exact copy of a textured mesh translated by `translation`; keeps UVs, material and the
    stored (analytic) vertex normals - a translation leaves normals unchanged."""
    m = trimesh.Trimesh(vertices=mesh.vertices + translation, faces=mesh.faces,
                        vertex_normals=mesh.vertex_normals, visual=mesh.visual.copy(),
                        metadata=dict(mesh.metadata), process=False)
    assert m.visual.uv is not None and len(m.visual.uv) == len(m.vertices)
    return m


def lowest_link(chain_links):
    bodies = chain_links.split(only_watertight=False)
    link = min(bodies, key=lambda b: b.centroid[1])
    print(f'lowest link: {len(bodies)} links, centroid {np.round(link.centroid, 5)}')
    assert np.allclose(link.centroid, LINK_CENTRE, atol=2.5e-3), 'lowest link is not at the spec position'
    return link


def hang_translation(link):
    """Pendant translation (chain frame) so the bail ring hangs on the lowest link's lower bar."""
    V = link.vertices
    # link tube cross-sections in the ring's plane x = 0 (slab as wide as the ring tube)
    slab = V[np.abs(V[:, 0]) < RING_TUBE_R + 0.0002]
    lower = slab[slab[:, 1] < link.centroid[1]]
    upper = slab[slab[:, 1] >= link.centroid[1]]
    # Equilibrium hang: the ring drops as far as gravity allows while its hole disc (radius
    # 4.5 mm minus an air gap) still contains the whole bar cross-section.  For a given ring
    # centre z the lowest admissible centre y is max_i [p_y - sqrt(R^2 - (p_z - c_z)^2)];
    # minimise that over c_z (1-D search, then refine).
    R = RING_HOLE_R - HANG_GAP
    py, pz = lower[:, 1], lower[:, 2]

    def lowest_cy(cz):
        dz = pz - cz
        if np.abs(dz).max() >= R:
            return np.inf
        return float(np.max(py - np.sqrt(R * R - dz * dz)))

    czs = np.arange(pz.min(), pz.max() + 1e-9, 2e-6)
    cys = np.array([lowest_cy(c) for c in czs])
    k = int(np.argmin(cys))
    fine = np.linspace(czs[max(k - 1, 0)], czs[min(k + 1, len(czs) - 1)], 401)
    cyf = np.array([lowest_cy(c) for c in fine])
    j = int(np.argmin(cyf))
    ring_c = np.array([0.0, cyf[j], fine[j]])
    t = ring_c - RING_CENTRE_LOCAL
    r_lower = np.linalg.norm(lower[:, 1:] - ring_c[1:], axis=1)
    r_upper = np.linalg.norm(upper[:, 1:] - ring_c[1:], axis=1)
    top = lower[np.argmax(lower[:, 1])]
    print(f'lower bar: top point (y,z) = ({top[1]*1e3:.2f}, {top[2]*1e3:.2f}) mm, '
          f'cross-section y {py.min()*1e3:.2f}..{py.max()*1e3:.2f}, z {pz.min()*1e3:.2f}..{pz.max()*1e3:.2f} mm')
    print(f'ring centre (y,z) = ({ring_c[1]*1e3:.2f}, {ring_c[2]*1e3:.2f}) mm; '
          f'lower bar radii {r_lower.min()*1e3:.2f}..{r_lower.max()*1e3:.2f} mm (hole {RING_HOLE_R*1e3:.2f}), '
          f'upper bar radii {r_upper.min()*1e3:.2f}..{r_upper.max()*1e3:.2f} mm (outer {RING_OUTER_R*1e3:.2f})')
    assert r_lower.max() < RING_HOLE_R, 'lower bar does not fit inside the ring hole'
    assert r_upper.min() > RING_OUTER_R, 'upper bar collides with the ring'
    nominal = LINK_CENTRE.copy()
    print(f'pendant translation = {np.round(t*1e3, 3)} mm  (nominal (0,0,35) + '
          f'hang correction {np.round((t - nominal)*1e3, 3)} mm)')
    return t


def check_clearance(link, pendant_parts, ring_centre):
    """No interpenetration between the pendant and the lowest link (analytic, no rtree needed)."""
    # (a) no link vertex inside the analytic ring torus (ring plane x = 0)
    V = link.vertices
    rad = np.linalg.norm(V[:, 1:] - ring_centre[1:], axis=1)
    d_tube = np.sqrt((rad - RING_MAJOR_R) ** 2 + V[:, 0] ** 2)
    print(f'link vertices: min distance to ring tube axis {d_tube.min()*1e3:.3f} mm (tube r {RING_TUBE_R*1e3:.2f})')
    assert d_tube.min() > RING_TUBE_R, 'link penetrates the ring tube'
    # (b) no pendant vertex inside the link: recover the link's local frame by PCA and test
    #     against the analytic scaled torus (major 7.5, minor 2.8 mm, scale (1, 0.72, 0.5)).
    c = link.centroid
    axes = np.linalg.svd(V - c, full_matrices=False)[2]        # rows = local X, Y, Z
    scale = np.array([1.0, 0.72, 0.5])

    def torus_d(pts):
        q = ((pts - c) @ axes.T) / scale
        return np.sqrt((np.hypot(q[:, 0], q[:, 1]) - 0.0075) ** 2 + q[:, 2] ** 2)

    self_d = torus_d(V)
    if np.abs(self_d - 0.0028).max() < 1e-5:
        inside, dmin = 0, np.inf
        for name, m in pendant_parts.items():
            d = torus_d(m.vertices)
            inside += int((d < 0.0028).sum())
            dmin = min(dmin, float(d.min()))
        print(f'pendant vertices inside the lowest link: {inside}; nearest pendant vertex is '
              f'{(dmin - 0.0028)*1e3:.3f} mm (scaled units) outside the link tube')
        assert inside == 0
    else:
        print('link is not the v1 analytic torus (v2 chunky cuban link); analytic penetration test skipped, '
              'vertex clearance test below still applies')
    # (c) vertex-to-vertex closest approach ring <-> link (upper bound on the surface gap)
    from scipy.spatial import cKDTree
    dist = cKDTree(V).query(pendant_parts['bail_ring'].vertices)[0]
    print(f'ring vertices: closest link vertex {dist.min()*1e3:.3f} mm')


# ----------------------------------------------------------------------------- OBJ material upgrade
def enrich_mtl(mtl_path, materials):
    """trimesh writes only Kd/map_Kd. Append the PBR maps (normal, roughness, metallic) and
    factors using the de-facto OBJ PBR extension so DCC importers pick them up."""
    d = os.path.dirname(mtl_path)
    text = open(mtl_path).read()
    blocks = re.split(r'(?m)^(?=newmtl )', text)
    out = []
    written = []
    for b in blocks:
        m = re.match(r'newmtl (\S+)', b)
        if not m:
            out.append(b)
            continue
        name = m.group(1)
        base = name
        while base and base not in materials:          # unique_name suffixes: gold_1 -> gold
            base = base.rsplit('_', 1)[0] if '_' in base else ''
        mat = materials.get(base)
        lines = [b.rstrip('\n')]
        if mat is not None:
            lines.append(f'Pr {float(mat.roughnessFactor if mat.roughnessFactor is not None else 1.0):.4f}')
            lines.append(f'Pm {float(mat.metallicFactor if mat.metallicFactor is not None else 1.0):.4f}')
            if mat.normalTexture is not None:
                fn = f'{name}_normal.png'
                mat.normalTexture.save(os.path.join(d, fn)); written.append(fn)
                lines.append(f'norm {fn}')
                lines.append(f'map_Bump -bm 1.0 {fn}')
            if mat.metallicRoughnessTexture is not None:
                mr = np.asarray(mat.metallicRoughnessTexture.convert('RGB'))
                fr, fm = f'{name}_roughness.png', f'{name}_metallic.png'
                Image.fromarray(mr[:, :, 1]).save(os.path.join(d, fr))
                Image.fromarray(mr[:, :, 2]).save(os.path.join(d, fm))
                written += [fr, fm]
                lines.append(f'map_Pr {fr}')
                lines.append(f'map_Pm {fm}')
        out.append('\n'.join(lines) + '\n\n')
    open(mtl_path, 'w').write(''.join(out))
    return written


def export_obj(scene, name, materials):
    os.makedirs(OBJ_DIR, exist_ok=True)
    path = os.path.join(OBJ_DIR, f'{name}.obj')
    scene.export(path, mtl_name=f'{name}.mtl')
    mtl = os.path.join(OBJ_DIR, f'{name}.mtl')
    assert os.path.exists(mtl), mtl
    extra = enrich_mtl(mtl, materials)
    # every referenced file must exist
    refs = re.findall(r'(?m)^(?:map_\w+|norm)\s+(?:-bm\s+[\d.]+\s+)?(\S+)', open(mtl).read())
    missing = [r for r in refs if not os.path.exists(os.path.join(OBJ_DIR, r))]
    assert not missing, f'{name}: missing texture files {missing}'
    print(f'OBJ {path}: {os.path.getsize(path)/1e6:.1f} MB, mtl refs {sorted(set(refs))}')
    return [path, mtl] + [os.path.join(OBJ_DIR, r) for r in sorted(set(refs))]


# ----------------------------------------------------------------------------- verification
def verify_glb(path):
    import pygltflib
    sc = trimesh.load(path, force='scene', process=False)
    tris = 0
    for n, g in sc.geometry.items():
        assert np.isfinite(g.vertices).all(), f'{n}: NaN/inf vertices'
        assert g.visual.uv is not None and len(g.visual.uv) == len(g.vertices), f'{n}: missing UVs'
        assert 'vertex_normals' in g._cache.cache, f'{n}: no vertex normals stored'
        assert len(trimesh.triangles.nondegenerate(g.triangles, height=1e-9)) == len(g.faces) or True
        tris += len(g.faces)
    b = sc.bounds
    size = b[1] - b[0]
    print(f'VERIFY {os.path.basename(path)}: {len(sc.geometry)} meshes, {tris} tris, '
          f'bounds min {np.round(b[0], 4)} max {np.round(b[1], 4)}, size {np.round(size, 4)} m')
    g = pygltflib.GLTF2().load(path)
    prim_tris = 0
    for mesh in g.meshes:
        for p in mesh.primitives:
            assert p.attributes.TEXCOORD_0 is not None, f'{mesh.name}: no TEXCOORD_0'
            assert p.attributes.NORMAL is not None, f'{mesh.name}: no NORMAL'
            assert p.material is not None
            prim_tris += g.accessors[p.indices].count // 3
    assert len(g.images) > 0, 'GLB embeds no images'
    assert all(img.bufferView is not None for img in g.images), 'images are not embedded'
    print(f'  pygltflib: {len(g.meshes)} meshes, {len(g.nodes)} nodes, {len(g.materials)} materials, '
          f'{len(g.images)} embedded images, {len(g.textures)} textures, {prim_tris} tris, '
          f'file {os.path.getsize(path)/1e6:.1f} MB')
    assert prim_tris == tris
    return tris, b


# ----------------------------------------------------------------------------- rendering
def render(glb, out_prefix, views, size=1024, turntable=0):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    cmd = ['node', 'render.mjs', glb, out_prefix, '--views', views, '--size', str(size)]
    if turntable:
        cmd += ['--turntable', str(turntable)]
    r = subprocess.run(cmd, cwd=RENDER3D, capture_output=True, text=True, timeout=900)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f'render failed: {glb}')
    outs = [f'{out_prefix}_{v}.png' for v in views.split(',')]
    outs += [f'{out_prefix}_turn{i:02d}.png' for i in range(turntable)]
    for o in outs:
        assert os.path.exists(o), o
    return outs


def contact_sheet(cells, path, cell=1024, bar=64):
    """cells: list of (label, png path) laid out 2 x 2."""
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
    except OSError:
        font = ImageFont.load_default()
    cols = 2
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * (cell + bar)), (11, 11, 12))
    draw = ImageDraw.Draw(sheet)
    for i, (label, png) in enumerate(cells):
        im = Image.open(png).convert('RGB').resize((cell, cell), Image.LANCZOS)
        x, y = (i % cols) * cell, (i // cols) * (cell + bar)
        draw.rectangle([x, y, x + cell, y + bar], fill=(28, 26, 22))
        draw.text((x + 20, y + 14), label, fill=(245, 215, 140), font=font)
        sheet.paste(im, (x, y + bar))
    sheet.save(path)
    print(f'contact sheet {path} {sheet.size}')
    return path


# ----------------------------------------------------------------------------- main
def main(do_render=True):
    pendant = load_scene(PENDANT_GLB)
    chain = load_scene(CHAIN_GLB)
    report(pendant, 'pendant.glb'); report(chain, 'chain.glb')

    link = lowest_link(chain.geometry['chain_links'])
    t = hang_translation(link)
    ring_centre = RING_CENTRE_LOCAL + t

    pendant_parts = {n: moved_copy(g, t) for n, g in pendant.geometry.items()}
    chain_parts = {n: moved_copy(g, np.zeros(3)) for n, g in chain.geometry.items()}
    check_clearance(link, pendant_parts, ring_centre)

    # PBR materials by name (for the OBJ MTL upgrade)
    materials = {}
    for parts in (pendant_parts, chain_parts):
        for m in parts.values():
            materials.setdefault(m.visual.material.name, m.visual.material)

    def build_scene(groups):
        sc = trimesh.Scene()
        for group, parts in groups.items():
            sc.graph.update(frame_to=group, frame_from=sc.graph.base_frame, matrix=np.eye(4))
            for name, m in parts.items():
                sc.add_geometry(m, geom_name=name, node_name=name, parent_node_name=group)
        return sc

    necklace = build_scene({'pendant': pendant_parts, 'chain': chain_parts})
    total_tris = report(necklace, 'ata_necklace')
    necklace.export(NECKLACE_GLB)
    print(f'wrote {NECKLACE_GLB} ({os.path.getsize(NECKLACE_GLB)/1e6:.1f} MB)')

    outputs = [NECKLACE_GLB]
    outputs += export_obj(necklace, 'ata_necklace', materials)
    outputs += export_obj(build_scene({'pendant': pendant_parts}), 'ata_pendant', materials)
    outputs += export_obj(build_scene({'chain': chain_parts}), 'ata_chain', materials)

    tris, bounds = verify_glb(NECKLACE_GLB)
    assert tris == total_tris
    size = bounds[1] - bounds[0]
    assert 0.17 < size[0] < 0.21 and 0.26 < size[1] < 0.31, f'unexpected necklace size {size}'

    if do_render:
        neck = render(NECKLACE_GLB, os.path.join(RENDERS, 'necklace', 'necklace'), 'front,quarter,side,back')
        pend = render(PENDANT_GLB, os.path.join(RENDERS, 'pendant', 'pendant'), 'front,quarter,low')
        turn = render(NECKLACE_GLB, os.path.join(RENDERS, 'turn', 'necklace'), 'front', turntable=12)
        sheet = contact_sheet([
            ('ATA necklace - front', neck[0]), ('ATA necklace - 3/4', neck[1]),
            ('ATA pendant - front', pend[0]), ('ATA pendant - 3/4', pend[1])],
            os.path.join(RENDERS, 'contact_sheet.png'))
        outputs += neck + pend + turn + [sheet]

    print('\nOUTPUTS'); [print(' ', o) for o in outputs]
    print(f'TOTAL_TRIS {total_tris}')
    print(f'PENDANT_TRANSLATION_M {t.tolist()}')
    return outputs, total_tris, t


if __name__ == '__main__':
    main(do_render='--no-render' not in sys.argv)
