# ATA Clothing Pack (FiveM) — jacket, pants, chain

Full-garment 1024x1024 textures for a FiveM / GTA V freemode ped, built from the ATA
reference art in `source/reference/`. Three pieces:

| Piece | FiveM slot | PNG (editable) | DDS (game-ready, mipmapped) |
|---|---|---|---|
| Jacket / hoodie | component 11 `jbib` | `textures/png/jacket_1024.png` | `textures/dds/jbib_diff_000_a_uni.dds` (DXT1) |
| Pants / jeans | component 4 `lowr` | `textures/png/pants_1024.png` | `textures/dds/lowr_diff_000_a_uni.dds` (DXT1) |
| Chain | component 7 `teef` | `textures/png/chain_1024.png` (RGBA) | `textures/dds/teef_diff_000_a_uni.dds` (DXT5, alpha) |

Preview of all three: `preview/contact_sheet.png`. Every texture is exactly 1024x1024.

## What is in each texture

**Jacket** (`jacket_1024.png`, see `jacket_layout_guide.png` for the labelled regions)
- Whole surface: black knit fabric with red lace/marble burnout pattern (seamless tile).
- Top-left = FRONT: crown + ATA logo across the chest, gold zipper placket, gold baroque
  filigree on both hips, Greek-key hem.
- Top-right = BACK: crown + ATA logo, gold Medusa medallion with Greek-key ring,
  filigree both sides, Greek-key hem.
- Bottom-left = LEFT and RIGHT SLEEVE: small crown logo at the shoulder, tall filigree
  down the sleeve, Greek-key cuff.
- Bottom-right = HOOD: scattered small ATA logos, Greek-key waistband strip, and a
  small black LABEL square in the corner.

**Pants** (`pants_1024.png`, see `pants_layout_guide.png`)
- Four vertical leg panels, left to right: FRONT L, FRONT R, BACK L, BACK R.
- Each panel: gold waistband line, crown + ATA on the thigh, Medusa on the knee,
  filigree down the shin, Greek-key cuff, Greek-key stripe down the outer seam.
- Back panels also have a Greek-key bordered back pocket with a small ATA.

**Chain** (`chain_1024.png`, transparent background, see `chain_layout_guide.png`)
- Jewelled red/gold ATA pendant with bail, centred in the upper part.
- Seamlessly tileable cuban-link chain strip (red gems + white pave) along the bottom.
- Small gold clasp between them.

## Important: textures need a mesh

These files are the *skin* that wraps a 3D clothing model. FiveM clothing is a mesh
(`.ydd`) plus a texture dictionary (`.ytd`). This pack supplies the textures; the
mesh comes from either a vanilla item you retexture (recommended) or a custom model.

The jacket and pants use a generic region layout (documented in the layout guides).
Every game mesh has its own UV layout, so whoever packs the YTD should open the PNG
next to the target mesh's UV template and move regions if needed. To make that easy,
every design element is supplied separately in `textures/elements/` with transparent
backgrounds:

`base_tile.png` (seamless fabric), `ata_logo.png` (crown + letters), `ata_letters.png`,
`medusa.png`, `greek_key_strip.png` / `greek_key_strip_v.png` (tileable trims),
`filigree.png` / `filigree_single.png` (+ `filigree_alt_a.png` variant), `pendant.png`,
`chain_strip.png` (tileable).

## Putting it in the game (replacement method)

1. Pick the vanilla items to override on `mp_m_freemode_01` (or `mp_f_freemode_01`):
   a hoodie/jacket drawable in `jbib`, a jeans drawable in `lowr`, a chain drawable in `teef`.
   Note their drawable ids (the number you pass to `SetPedComponentVariation`).
2. Rename the DDS files to the exact vanilla names, replacing `000` with the drawable id
   and `a` with the texture letter you are replacing, e.g.
   `mp_m_freemode_01^jbib_diff_004_a_uni.ytd`. Grammar:
   `<ped>^<slot>_diff_<drawable 3 digits>_<texture letter a-z>_<uni|whi|bla|...>.ytd`.
   Clothing that shows no skin uses `uni`.
3. Build the `.ytd` with OpenIV or CodeWalker RPF Explorer (edit mode → new YTD → import
   the DDS). Keep the texture name inside the YTD identical to the file name without
   extension, one texture per YTD. The DDS files already contain the full mip chain.
4. Drop the `.ytd` files into `stream/` and add the resource to `server.cfg`
   (`ensure ata-clothing`). `fxmanifest.lua` is already set up; nothing else is required
   for replacements.

For an add-on pack (own drawable ids, nothing vanilla overwritten) you also need the
`.ydd` models, a `mp_m_freemode_01_<dlc>.ymt`, and the `SHOP_PED_APPAREL_META_FILE`
data_file line (see the commented block in `fxmanifest.lua`). Add-on files carry the dlc
name: `mp_m_freemode_01_<dlc>^jbib_diff_000_a_uni.ytd`.

Detailed research notes with sources are in `source/fivem_research_notes.json`.

## Regenerating or re-exporting

```
pip install pillow numpy scipy
cd source && python3 gen/gen_base_tile.py        # etc. per element, then compose_*.py
python3 tools/export_dds.py textures/png/jacket_1024.png out.dds --format DXT1
python3 tools/export_dds.py textures/png/chain_1024.png  out.dds --format DXT5
python3 tools/make_preview.py textures/png preview
```

Use DXT1 for opaque textures and DXT5 whenever the alpha channel matters.
