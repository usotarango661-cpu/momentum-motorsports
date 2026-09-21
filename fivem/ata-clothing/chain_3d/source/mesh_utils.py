"""Shared helpers for the ATA chain 3D build (pendant + cuban link chain).

Conventions (ALL agents must follow):
  * Units: metres.  Y is up.  The wearer faces +Z, so the FRONT of the pendant faces +Z.
  * Pendant local frame: the top of the bail loop is at the origin (0,0,0); the pendant hangs
    down in -Y; it is centred on X.  Overall pendant+bail height ~0.11 m, width ~0.10 m.
  * Chain local frame: a necklace U-curve in the XY plane (slightly bulging to +Z at the
    front); the centre of the LOWEST link is at the origin, the chain rises to the neck at
    y ~ +0.17 and the two ends meet behind the neck (z < 0).
  * Meshes are exported as GLB (trimesh Scene) with PBR materials; textures embedded.
"""
import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure

REF_PENDANT_CUTOUT = '../textures/elements/pendant.png'   # 1024x1024 RGBA, pendant + bail
REF_PENDANT_PHOTO = '../source/reference/pendant.png'             # 1254x1254 original render
CHAIN_STRIP = '../textures/elements/chain_strip.png'     # 1024x320 RGBA tileable cuban links
OUT = './out'

GOLD = [1.0, 0.78, 0.32, 1.0]
RED_GEM = [0.75, 0.03, 0.08, 1.0]


def pbr(color=GOLD, metallic=1.0, roughness=0.22, texture=None, normal=None, name='mat'):
    """PBR material; texture/normal are PIL images (RGBA/RGB)."""
    m = trimesh.visual.material.PBRMaterial(baseColorFactor=color, metallicFactor=metallic, roughnessFactor=roughness, name=name)
    if texture is not None:
        m.baseColorTexture = texture
    if normal is not None:
        m.normalTexture = normal
    return m


def mask_to_polygons(alpha, threshold=0.5, simplify_px=1.2, min_area_px=400):
    """alpha: 2-D float array 0..1 (row 0 = TOP of image).  Returns a shapely (Multi)Polygon in
    PIXEL units with y pointing UP (i.e. y = height - row), holes included."""
    h = alpha.shape[0]
    contours = measure.find_contours(np.pad(alpha, 1, constant_values=0), threshold)
    rings = []
    for c in contours:
        pts = [(x - 1, h - (y - 1)) for y, x in c]
        if len(pts) < 4:
            continue
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        if p.area >= min_area_px:
            rings.append(p.simplify(simplify_px))
    # nest holes: polygons contained in another polygon become holes
    rings.sort(key=lambda p: -p.area)
    outers = []
    for p in rings:
        placed = False
        for o in outers:
            if o['poly'].contains(p.representative_point()):
                o['holes'].append(p); placed = True; break
        if not placed:
            outers.append({'poly': p, 'holes': []})
    polys = []
    for o in outers:
        shell = o['poly'].exterior.coords
        holes = [hh.exterior.coords for hh in o['holes']]
        q = Polygon(shell, holes)
        if not q.is_valid:
            q = q.buffer(0)
        polys.append(q)
    return unary_union(polys)


def extrude_with_bevel(poly, depth, bevel=0.0, bevel_steps=2, scale=1.0):
    """Extrude a shapely polygon (pixel units * scale -> metres) along +Z with an optional
    chamfer bevel on the FRONT (+Z) face: the outline is inset by `bevel` over the last
    `bevel_steps` slabs.  Returns a single trimesh.Trimesh (Z from 0 to depth)."""
    parts = []
    if bevel <= 0 or bevel_steps <= 0:
        m = trimesh.creation.extrude_polygon(poly, depth)
        m.apply_scale([scale, scale, 1.0])
        return m
    body_depth = depth * 0.7
    m = trimesh.creation.extrude_polygon(poly, body_depth)
    m.apply_scale([scale, scale, 1.0]); parts.append(m)
    z = body_depth
    step = (depth - body_depth) / bevel_steps
    for i in range(1, bevel_steps + 1):
        inset = poly.buffer(-bevel * i / bevel_steps / scale, join_style=2)
        if inset.is_empty:
            break
        s = trimesh.creation.extrude_polygon(inset, step)
        s.apply_scale([scale, scale, 1.0]); s.apply_translation([0, 0, z]); parts.append(s); z += step
    return trimesh.util.concatenate(parts)


def planar_uv(mesh, bounds_xy, flip_v=False):
    """Assign UVs by projecting XY onto [0,1]^2 using bounds_xy=(xmin,ymin,xmax,ymax)."""
    xmin, ymin, xmax, ymax = bounds_xy
    v = mesh.vertices
    u = (v[:, 0] - xmin) / (xmax - xmin)
    w = (v[:, 1] - ymin) / (ymax - ymin)
    if flip_v:
        w = 1 - w
    return np.column_stack([u, w])


def torus_uv(major_r, minor_r, seg_u=48, seg_v=16, scale_xyz=(1, 1, 1), u_range=(0, 2 * np.pi)):
    """Torus with proper UVs. u = around the ring (major), v = around the tube (minor).
    v=0.25 is the OUTER equator (+radial), v=0.75 the inner; v=0.0 is +Z face, v=0.5 -Z face.
    Returns (vertices, faces, uv)."""
    us = np.linspace(u_range[0], u_range[1], seg_u + 1)
    vs = np.linspace(0, 2 * np.pi, seg_v + 1)
    U, V = np.meshgrid(us, vs, indexing='ij')
    x = (major_r + minor_r * np.sin(V)) * np.cos(U)
    y = (major_r + minor_r * np.sin(V)) * np.sin(U)
    z = minor_r * np.cos(V)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) * np.array(scale_xyz)
    uv = np.column_stack([(U / (2 * np.pi)).ravel(), (V / (2 * np.pi)).ravel()])
    faces = []
    for i in range(seg_u):
        for j in range(seg_v):
            a = i * (seg_v + 1) + j; b = a + seg_v + 1
            faces.append([a, b, a + 1]); faces.append([b, b + 1, a + 1])
    return verts, np.array(faces), uv


def textured_mesh(verts, faces, uv, material):
    m = trimesh.Trimesh(verts, faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    m.fix_normals()
    return m


def report(mesh_or_scene, name=''):
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
    else:
        geoms = [mesh_or_scene]
    tris = sum(len(g.faces) for g in geoms)
    b = trimesh.util.concatenate(geoms).bounds
    print(f'[{name}] parts={len(geoms)} tris={tris} bounds_min={np.round(b[0],4)} bounds_max={np.round(b[1],4)} size={np.round(b[1]-b[0],4)} m')
    return tris
