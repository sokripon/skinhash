// Compares the canvas-free PNG decoder with Pillow on the repo skins and generated colour-type variants.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { decodePng, mojangPng } from "../js/png.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const skinsDir = path.join(here, "..", "..", "skins");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "png_dec_"));
let n = 0, bad = 0;
const check = (label, pngPath, rgbaPath) => {
  const got = decodePng(new Uint8Array(fs.readFileSync(pngPath))).rgba;
  const want = new Uint8Array(fs.readFileSync(rgbaPath));
  n++;
  if (got.length !== want.length || got.some((x, i) => x !== want[i])) { bad++; console.error("MISMATCH", label); }
};

for (const f of fs.readdirSync(skinsDir).filter((x) => x.endsWith(".png"))) {
  const out = path.join(tmp, f);
  const r = spawnSync("py", [path.join(here, "png_ref.py"), path.join(skinsDir, f), out], { encoding: "utf8" });
  if (r.status !== 0) { console.error(r.stderr); process.exit(2); }
  for (const v of fs.readdirSync(out).filter((x) => x.endsWith(".png"))) check(`${f} / ${v}`, path.join(out, v), path.join(out, v + ".rgba"));
  // the original file itself, with Pillow's RGBA conversion
  const r2 = spawnSync("py", ["-c", `import sys;from PIL import Image;sys.stdout.buffer.write(Image.open(sys.argv[1]).convert('RGBA').tobytes())`, path.join(skinsDir, f)]);
  fs.writeFileSync(path.join(tmp, f + ".rgba"), r2.stdout);
  check(`${f} / original`, path.join(skinsDir, f), path.join(tmp, f + ".rgba"));
  // our own encoder output decodes back to the same pixels
  const rgba = decodePng(new Uint8Array(fs.readFileSync(path.join(skinsDir, f)))).rgba;
  const re = decodePng(mojangPng(rgba)).rgba;
  n++; if (re.some((x, i) => x !== rgba[i])) { bad++; console.error("roundtrip mismatch", f); }
}

// rejection messages
const expectReject = (label, bytes, rx) => {
  n++;
  try { decodePng(bytes); bad++; console.error("should have rejected", label); }
  catch (e) { if (!rx.test(e.message)) { bad++; console.error("wrong message for", label, e.message); } }
};
const r3 = spawnSync("py", ["-c", `import sys,io;from PIL import Image
b=io.BytesIO();Image.new('RGBA',(64,32)).save(b,'PNG');open(sys.argv[1],'wb').write(b.getvalue())
b=io.BytesIO();Image.new('RGB',(64,64)).convert('I;16').save(b,'PNG');open(sys.argv[2],'wb').write(b.getvalue())`, path.join(tmp, "s32.png"), path.join(tmp, "d16.png")], { encoding: "utf8" });
expectReject("64x32", new Uint8Array(fs.readFileSync(path.join(tmp, "s32.png"))), /64x64/);
if (r3.status === 0) expectReject("16-bit", new Uint8Array(fs.readFileSync(path.join(tmp, "d16.png"))), /8-bit|64x64/);
expectReject("not a png", new Uint8Array(100), /Not a PNG/);

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`png decode: ${n - bad}/${n} checks OK`);
process.exit(bad ? 1 : 0);
