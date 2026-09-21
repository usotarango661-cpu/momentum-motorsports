# cwtool - headless YTD / YDD packer (Linux, .NET 8)

Thin console wrapper around CodeWalker.Core (dexyfex/CodeWalker, MIT) so GTA V resources
can be built without Windows.

```
git clone --depth 1 https://github.com/dexyfex/CodeWalker.git ../CodeWalker   # next to this folder
dotnet build -c Release
dotnet bin/Release/net8.0/cwtool.dll ytd  <in.ytd.xml> <ddsFolder> <out.ytd>
dotnet bin/Release/net8.0/cwtool.dll ydd  <in.ydd.xml> <folder>    <out.ydd>
dotnet bin/Release/net8.0/cwtool.dll dump <file.ytd|.ydd> [outFolder]   # reload + print XML (verification)
```

The `.ytd.xml` files used for this pack are in `textures/ytd_xml/` (CodeWalker XML format:
one `<Item>` per texture directly under `<TextureDictionary>`, with the DDS next to it).
