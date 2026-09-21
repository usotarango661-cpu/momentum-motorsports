#!/usr/bin/env python3
"""link_closeup.py - straight 7-link test strip for inspecting pave texture and interlock.
Writes renders/closeup.glb (not a deliverable)."""
import os
import sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from mesh_utils import pbr, textured_mesh  # noqa: E402
import build_chain as bc  # noqa: E402

n = 7
Vt, Ft, UVt, Nt = bc.link_template()
base_img, nrm_img, mr_img = bc.build_textures()
mat = pbr(color=[1.0, 1.0, 1.0, 1.0], metallic=0.85, roughness=0.25, texture=base_img, normal=nrm_img, name='ata_link_pave')
mat.metallicRoughnessTexture = mr_img
tmpl = textured_mesh(Vt, Ft, UVt, mat)
fn_expect = Nt[tmpl.faces].mean(axis=1)
if np.mean(np.einsum('ij,ij->i', tmpl.face_normals, fn_expect) > 0) < 0.5:
    tmpl.invert()
Ft = tmpl.faces.copy()
T = np.array([1.0, 0, 0]); Nout = np.array([0, 0, 1.0])
R = bc.link_frame(T, Nout)
vs, fs, uvs, ns = [], [], [], []
off = 0
for i in range(n):
    Rt = bc.rot_axis(T, np.radians(bc.TILT_DEG) * (1 if i % 2 == 0 else -1)) @ R
    c = np.array([i * bc.SPACING, 0, 0])
    vs.append(Vt @ Rt.T + c); fs.append(Ft + off); uvs.append(UVt); ns.append(Nt @ Rt.T); off += len(Vt)
m = trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False)
m.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uvs), material=mat)
m.fix_normals(multibody=True)
m.vertex_normals = np.vstack(ns)
out = os.path.join(bc.RENDER_DIR, 'closeup.glb')
trimesh.Scene([m]).export(out)
print('wrote', out, 'tris', len(m.faces))
