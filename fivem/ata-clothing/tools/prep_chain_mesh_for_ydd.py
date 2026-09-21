"""Prepare the ATA necklace GLB for a GTA V ped drawable: merge by material, convert to ped space
(X right, Y forward, Z up; origin at the ped root/pelvis), compute tangents, report counts."""
import numpy as np, trimesh, json, sys
GLB = '/home/user/momentum-motorsports/fivem/ata-clothing/chain_3d/glb/ata_necklace.glb'
sc = trimesh.load(GLB)
groups = {}
for name, geom in sc.geometry.items():
    T = None
    for node in sc.graph.nodes_geometry:
        tf, gname = sc.graph[node]
        if gname == name: T = tf; break
    g = geom.copy()
    if T is not None: g.apply_transform(T)
    mat = getattr(g.visual, 'material', None)
    mname = getattr(mat, 'name', None) or name
    groups.setdefault(mname, []).append((name, g))
print('materials:', {k: [n for n, _ in v] for k, v in groups.items()})

# chain frame -> ped space.  Chain frame: X right, Y up, Z front (lowest link at y=0, neck at y~0.175).
# Ped space: X right, Y forward, Z up, origin at the pelvis.  Neck centre ~ (0, 0, 0.53) on the freemode ped;
# the chest is deeper than the chain's modelled bulge, so the depth axis is stretched 1.6x.
def to_ped(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.column_stack([x, z * 1.6 + 0.03, y + 0.36])

out = {}
for mname, parts in groups.items():
    meshes = [g for _, g in parts]
    m = trimesh.util.concatenate(meshes)
    V = to_ped(m.vertices.astype(np.float64))
    F = m.faces.astype(np.int64)
    # normals: transform the same way (rotation part only: (x,y,z)->(x,z,y) then normalise)
    N = m.vertex_normals.astype(np.float64)
    N = np.column_stack([N[:, 0], N[:, 2], N[:, 1]]); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    UV = m.visual.uv if (hasattr(m.visual, 'uv') and m.visual.uv is not None) else np.zeros((len(V), 2))
    UV = np.asarray(UV, np.float64); UV[:, 1] = 1.0 - UV[:, 1]   # GTA: V flipped vs glTF
    # tangents from UVs (per-triangle accumulate)
    Tn = np.zeros_like(V)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    u0, u1, u2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0; d1, d2 = u1 - u0, u2 - u0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]; det[np.abs(det) < 1e-12] = 1e-12
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    for k in range(3): np.add.at(Tn, F[:, k], t)
    Tn -= N * np.sum(Tn * N, axis=1, keepdims=True)
    ln = np.linalg.norm(Tn, axis=1, keepdims=True); bad = ln[:, 0] < 1e-9
    Tn = np.where(bad[:, None], np.cross(N, [0, 0, 1.0]), Tn / np.maximum(ln, 1e-12))
    Tn /= (np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-12)
    out[mname] = dict(V=V, N=N, UV=UV, T=Tn, F=F, parts=[n for n, _ in parts])
    print(f'{mname:16s} verts={len(V):6d} tris={len(F):6d} bounds z {V[:,2].min():.3f}..{V[:,2].max():.3f} y {V[:,1].min():.3f}..{V[:,1].max():.3f}')
np.savez('necklace_ped_space.npz', **{f'{k}__{f}': v for k, d in out.items() for f, v in d.items() if f != 'parts'})
json.dump({k: d['parts'] for k, d in out.items()}, open('necklace_parts.json', 'w'), indent=1)
allV = np.vstack([d['V'] for d in out.values()]); print('overall bounds min', allV.min(0).round(3), 'max', allV.max(0).round(3))
