"""Flat quad with the link textures: checks the normal-map sign (bumps must be lit from +X+Y)."""
import os, sys, numpy as np, trimesh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from mesh_utils import pbr
import build_chain as bc
base, nrm, mr = bc.build_textures()
mat = pbr(color=[0.6, 0.6, 0.6, 1.0], metallic=0.0, roughness=0.5, texture=None, normal=nrm, name='t')
# quad 40mm x 20mm, u along +X over [0,0.25] (a quarter of the ring), v over [0.75,1.0]+[0,0.25] -> use v in [0.8, 1.0] top rows
V = np.array([[0,0,0],[0.04,0,0],[0.04,0.02,0],[0,0.02,0]], float)
F = np.array([[0,1,2],[0,2,3]])
UV = np.array([[0,0.80],[0.25,0.80],[0.25,1.0],[0,1.0]])
m = trimesh.Trimesh(V, F, process=False); m.visual = trimesh.visual.TextureVisuals(uv=UV, material=mat); m.fix_normals()
sph = trimesh.creation.icosphere(subdivisions=3, radius=0.006); sph.apply_translation([0.05, 0.01, 0]); sph.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(sph.vertices),2)), material=pbr(color=[0.6,0.6,0.6,1.0], metallic=0.0, roughness=0.5, name='s'))
trimesh.Scene([m, sph]).export(os.path.join(bc.RENDER_DIR, 'nrm_test.glb')); print('ok')
