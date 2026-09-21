using System.Text;
using CodeWalker.GameFiles;

// cwtool: pack / unpack GTA V resources with CodeWalker.Core (headless, Linux).
//   cwtool ytd  <in.ytd.xml> <ddsFolder> <out.ytd>
//   cwtool ydd  <in.ydd.xml> <folder>    <out.ydd>
//   cwtool dump <file.ytd|file.ydd> [outFolder]     (round-trip check: load binary, print XML)
    if (args.Length < 2) { Console.Error.WriteLine("usage: cwtool ytd|ydd|dump ..."); return 2; }
    var cmd = args[0];
    switch (cmd)
    {
        case "ytd":
        {
            var xml = File.ReadAllText(args[1]);
            var ytd = XmlYtd.GetYtd(xml, args[2]);
            var data = ytd.Save();
            File.WriteAllBytes(args[3], data);
            Console.WriteLine($"wrote {args[3]} ({data.Length} bytes, {ytd.TextureDict?.Textures?.data_items?.Length ?? 0} textures)");
            return 0;
        }
        case "ydd":
        {
            var xml = File.ReadAllText(args[1]);
            var ydd = XmlYdd.GetYdd(xml, args[2]);
            var data = ydd.Save();
            File.WriteAllBytes(args[3], data);
            Console.WriteLine($"wrote {args[3]} ({data.Length} bytes, {ydd.DrawableDict?.Drawables?.data_items?.Length ?? 0} drawables)");
            return 0;
        }
        case "dump":
        {
            var path = args[1];
            var data = File.ReadAllBytes(path);
            var outFolder = args.Length > 2 ? args[2] : "";
            var ext = Path.GetExtension(path).ToLowerInvariant();
            if (ext == ".ytd")
            {
                var ytd = new YtdFile();
                ytd.Load(data);
                Console.WriteLine(YtdXml.GetXml(ytd, outFolder));
            }
            else if (ext == ".ydd")
            {
                var ydd = new YddFile();
                ydd.Load(data);
                Console.WriteLine(YddXml.GetXml(ydd, outFolder));
            }
            else { Console.Error.WriteLine("unknown extension"); return 2; }
            return 0;
        }
        default:
            Console.Error.WriteLine("unknown command"); return 2;
    }
