fx_version 'cerulean'
game 'gta5'

author 'SimMomentum Motorsports'
description 'ATA clothing pack - jacket, pants and chain textures (1024x1024)'
version '1.0.0'

-- Everything inside stream/ is streamed automatically by FiveM:
--   *.ytd  texture dictionaries (the ATA textures packed with OpenIV / CodeWalker)
--   *.ydd  drawables, only needed if you ship custom meshes instead of retexturing
--
-- Texture REPLACEMENT (default for this pack): name each .ytd after the vanilla
-- component it overrides, e.g.
--   stream/mp_m_freemode_01^jbib_diff_000_a_uni.ytd   (jacket / top)
--   stream/mp_m_freemode_01^lowr_diff_000_a_uni.ytd   (pants / legs)
--   stream/mp_m_freemode_01^teef_diff_000_a_uni.ytd   (chain / neck accessory)
-- No extra manifest lines are required for replacements.

-- ADD-ON clothing (custom drawables + shop meta) additionally needs the lines
-- below.  Uncomment and rename once you have the .ymt / creaturemetadata files.
-- files {
--     'data/mp_m_freemode_01_ata.ymt',
--     'data/mp_m_freemode_01_ata.meta',
-- }
-- data_file 'SHOP_PED_APPAREL_META_FILE' 'data/mp_m_freemode_01_ata.meta'
