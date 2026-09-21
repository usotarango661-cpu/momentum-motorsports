# ATA clothing pack - handoff / continue here

Branch: `claude/wearable-clothes-5m-game-s1co7y`  (folder `fivem/ata-clothing/`)

## Done and verified
| Piece | Deliverable | Where |
|---|---|---|
| Jacket (jbib) | 1024x1024 PNG + DDS (DXT1, mips) + packed `.ytd` | `textures/png/jacket_1024.png`, `textures/dds/jbib_diff_000_a_uni.dds`, `stream/mp_m_freemode_01^jbib_diff_000_a_uni.ytd` |
| Pants (lowr) | same | `textures/png/pants_1024.png`, `textures/dds/lowr_diff_000_a_uni.dds`, `stream/mp_m_freemode_01^lowr_diff_000_a_uni.ytd` |
| Chain (teef) | 3D model: FBX / .blend / GLB / OBJ (textures embedded) + flat texture `.ytd` | `chain_3d/fbx/ata_necklace.fbx`, `chain_3d/glb/ata_necklace.glb`, `stream/mp_m_freemode_01^teef_diff_000_a_uni.ytd` |
| Previews | contact sheets, renders, layout guides | `preview/`, `chain_3d/renders/`, `textures/png/*_layout_guide.png` |
| Elements | every design element as its own transparent PNG | `textures/elements/` |

The `.ytd` files were built with CodeWalker's own library (`tools/cwtool`, .NET 8, runs on Linux)
and verified by reloading them and re-extracting the DDS. Drawable id `000` / letter `a` are
placeholders: rename to the vanilla item you replace (see README "Putting it in the game").

## Not done: chain as `.ydd`
The chain mesh still has to become a rigged ped drawable. Two ways:

1. **Blender + Sollumz (recommended, what the shop will do):** open
   `chain_3d/fbx/ata_necklace.blend`, convert to a Sollumz Drawable, weight everything to
   `SKEL_Spine3` / `SKEL_Neck_1` of `mp_m_freemode_01`, position it on the ped (chain frame is
   metres, Y up, +Z front; ped space is Z up, +Y forward, origin at the pelvis, neck ~0.53 m up),
   export `mp_m_freemode_01^teef_0XX_u.ydd` with its `teef_diff_0XX_a_uni.ytd`.
2. **Headless (started, unfinished):** `tools/prep_chain_mesh_for_ydd.py` merges the GLB by
   material, converts to ped space and computes tangents (7 geometries, all < 65k verts). Still
   needed: write the CodeWalker `.ydd.xml` (shader `ped`, skinned vertex layout, blend indices
   for the neck bone) and pack it with `cwtool ydd`. Format references: CodeWalker
   `Drawable.cs` ReadXml/WriteXml and Sollumz `ydd/yddexport.py`.

## Rebuild anything
See README "Regenerating or re-exporting". Python deps: `pip install pillow numpy scipy trimesh
shapely mapbox_earcut scikit-image bpy`; renders need Node + `npm i three` + Playwright Chromium.
