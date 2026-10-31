// PNG decoding without canvas (canvas premultiplies alpha and would destroy hidden RGB),
// and Mojang-style encoding (one IDAT, zlib level 4 / wbits 15 / memLevel 8 / default strategy).
import pako from "../vendor/pako.esm.mjs";
import { crc32 } from "./hash.js";

export const W = 64, H = 64, ROW = 1 + W * 4, RAW_LEN = ROW * H;
const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
export const DEFLATE_OPTS = { level: 4, windowBits: 15, memLevel: 8, strategy: 0 };

export function decodePng(bytes) {
  if (bytes.length < 33 || SIG.some((b, i) => bytes[i] !== b)) throw new Error("Not a PNG file.");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 8, ihdr = null, plte = null, trns = null;
  const idat = [];
  while (pos + 12 <= bytes.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    if (pos + 12 + len > bytes.length) throw new Error("Truncated PNG.");
    const data = bytes.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") ihdr = { w: dv.getUint32(pos + 8), h: dv.getUint32(pos + 12), depth: data[8], ct: data[9], interlace: data[12] };
    else if (type === "PLTE") plte = data;
    else if (type === "tRNS") trns = data;
    else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  if (!ihdr) throw new Error("PNG has no IHDR.");
  if (ihdr.w !== W || ihdr.h !== H) throw new Error(`Skin must be 64x64 (this PNG is ${ihdr.w}x${ihdr.h}). Old 64x32 skins are not supported.`);
  if (ihdr.depth !== 8) throw new Error(`Only 8-bit PNGs are supported (this one is ${ihdr.depth}-bit).`);
  if (ihdr.interlace) throw new Error("Interlaced PNGs are not supported. Re-save the skin without interlacing.");
  const chans = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ihdr.ct];
  if (!chans) throw new Error(`Unsupported PNG colour type ${ihdr.ct}.`);
  if (ihdr.ct === 3 && !plte) throw new Error("Palette PNG without PLTE.");
  if (!idat.length) throw new Error("PNG has no image data.");
  let total = 0; for (const c of idat) total += c.length;
  const z = new Uint8Array(total); let o = 0; for (const c of idat) { z.set(c, o); o += c.length; }
  let raw;
  try { raw = pako.inflate(z); } catch (e) { throw new Error("Corrupt PNG data: " + e); }
  const stride = W * chans;
  if (raw.length !== (stride + 1) * H) throw new Error("PNG data has the wrong size.");
  const img = new Uint8Array(stride * H);
  for (let y = 0; y < H; y++) {
    const ft = raw[y * (stride + 1)], src = y * (stride + 1) + 1, dst = y * stride;
    for (let i = 0; i < stride; i++) {
      const x = raw[src + i];
      const a = i >= chans ? img[dst + i - chans] : 0;
      const b = y ? img[dst - stride + i] : 0;
      const c = (y && i >= chans) ? img[dst - stride + i - chans] : 0;
      let v;
      switch (ft) {
        case 0: v = x; break;
        case 1: v = x + a; break;
        case 2: v = x + b; break;
        case 3: v = x + ((a + b) >> 1); break;
        case 4: {
          const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          v = x + ((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c)); break;
        }
        default: throw new Error("Bad PNG filter type " + ft);
      }
      img[dst + i] = v & 255;
    }
  }
  const rgba = new Uint8Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const o4 = i * 4;
    let r, g, b, a = 255;
    switch (ihdr.ct) {
      case 6: r = img[o4]; g = img[o4 + 1]; b = img[o4 + 2]; a = img[o4 + 3]; break;
      case 2:
        r = img[i * 3]; g = img[i * 3 + 1]; b = img[i * 3 + 2];
        if (trns && trns.length >= 6 && trns[1] === r && trns[3] === g && trns[5] === b) a = 0;
        break;
      case 3: {
        const idx = img[i];
        if (idx * 3 + 2 >= plte.length) throw new Error("Palette index out of range.");
        r = plte[idx * 3]; g = plte[idx * 3 + 1]; b = plte[idx * 3 + 2];
        a = trns && idx < trns.length ? trns[idx] : 255; break;
      }
      case 0:
        r = g = b = img[i];
        if (trns && trns.length >= 2 && trns[1] === r) a = 0;
        break;
      case 4: r = g = b = img[i * 2]; a = img[i * 2 + 1]; break;
    }
    rgba[o4] = r; rgba[o4 + 1] = g; rgba[o4 + 2] = b; rgba[o4 + 3] = a;
  }
  return { rgba, colorType: ihdr.ct };
}

export function rawScanlines(rgba) {
  const out = new Uint8Array(RAW_LEN);
  for (let y = 0; y < H; y++) out.set(rgba.subarray(y * W * 4, (y + 1) * W * 4), y * ROW + 1);
  return out;
}

const IEND = Uint8Array.from([0, 0, 0, 0, 0x49, 0x45, 0x4e, 0x44, 0xAE, 0x42, 0x60, 0x82]);

/** Full PNG file (what Mojang serves) from raw scanlines, using pako. */
export function pngFromRaw(raw) {
  const z = pako.deflate(raw, DEFLATE_OPTS);
  const out = new Uint8Array(33 + 12 + z.length + 12);
  const dv = new DataView(out.buffer);
  out.set(SIG, 0);
  dv.setUint32(8, 13);
  out.set([0x49, 0x48, 0x44, 0x52], 12);
  dv.setUint32(16, W); dv.setUint32(20, H);
  out.set([8, 6, 0, 0, 0], 24);
  dv.setUint32(29, crc32(0, out, 12, 29));
  dv.setUint32(33, z.length);
  out.set([0x49, 0x44, 0x41, 0x54], 37);
  out.set(z, 41);
  dv.setUint32(41 + z.length, crc32(0, out, 37, 41 + z.length));
  out.set(IEND, 45 + z.length);
  return out;
}
export const mojangPng = (rgba) => pngFromRaw(rawScanlines(rgba));
