#!/usr/bin/env python3
"""Convert a GLB model to FBX and .blend using Blender's Python module (pip install bpy).

Usage: python3 glb_to_fbx_blend.py model.glb out_basename
Writes out_basename.fbx (textures embedded) and out_basename.blend (textures packed).
"""
import sys
import bpy

src, base = sys.argv[1], sys.argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
print(f'imported {len(objs)} meshes, {sum(len(o.data.polygons) for o in objs)} polygons')
bpy.ops.file.pack_all()
bpy.ops.export_scene.fbx(filepath=base + '.fbx', embed_textures=True, path_mode='COPY', apply_unit_scale=True, use_selection=False)
bpy.ops.wm.save_as_mainfile(filepath=base + '.blend', compress=True)
print('wrote', base + '.fbx', base + '.blend')
