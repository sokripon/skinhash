// Every GPU hit is rebuilt on the CPU and checked before it is shown or offered for download.
import { decodePng, mojangPng, pngFromRaw, rawScanlines, W, H } from "./png.js";
import { sha256, toHex } from "./hash.js";
import { decodePerm, fastFile, fastDigest, permRaw } from "./job.js";
import { isVisible, MASKS } from "./model.js";
import { matchTargets } from "./targets.js";
import { namemcIdHex, namemcMessage, nmPixels, nmDigestFull } from "./namemc.js";

const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

export async function sha256Hex(bytes) {
  const js = toHex(sha256(bytes));
  if (globalThis.crypto?.subtle) {
    const wc = toHex(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)));
    if (wc !== js) throw new Error("WebCrypto and JS SHA-256 disagree");
  }
  return js;
}

/** Returns {ok, error?, png, hex, id, whys}. `orig` is the user's decoded RGBA. */
export async function verifyHit(job, prep, idx, targets) {
  try {
    const perm = decodePerm(job, idx);
    const { file, buf } = fastFile(job, perm);
    const full = pngFromRaw(permRaw(job, perm));            // full pako re-encode
    if (!eq(file, full)) return { ok: false, error: "fast-path file differs from full pako re-encode" };
    const hex = await sha256Hex(full);
    if (toHex(fastDigest(job, buf)) !== hex) return { ok: false, error: "midstate digest differs" };
    const whys = matchTargets(hex, targets);
    if (!whys.length) return { ok: false, error: "GPU hit does not satisfy any target on the CPU" };
    // pixels: only knob + watermark pixels may differ from the original, alpha unchanged, nothing visible
    const dec = decodePng(full).rgba, orig = prep.orig;
    const allowed = new Set();
    prep.wm.forEach(([x, y]) => allowed.add(y * W + x));
    job.knob.forEach((o) => { const r = Math.floor(o / 257), c = (o % 257) - 1; allowed.add(r * W + (c >> 2)); });
    let changed = 0;
    for (let i = 0; i < W * H; i++) {
      let same = true;
      for (let c = 0; c < 4; c++) if (dec[i * 4 + c] !== orig[i * 4 + c]) same = false;
      if (same) continue;
      changed++;
      if (dec[i * 4 + 3] !== orig[i * 4 + 3]) return { ok: false, error: `alpha changed at pixel ${i}` };
      if (!allowed.has(i)) return { ok: false, error: `unexpected change at pixel (${i % W},${Math.floor(i / W)})` };
      if (isVisible(i, dec[i * 4 + 3])) return { ok: false, error: `visible change at pixel (${i % W},${Math.floor(i / W)})` };
    }
    if (!eq(mojangPng(dec), full)) return { ok: false, error: "re-encode of the decoded pixels differs" };
    const nmId = namemcIdHex(dec);
    if (nmId !== namemcIdHex(orig)) return { ok: false, error: "the NameMC id would change" };
    return { ok: true, png: full, hex, id: hex.replace(/^0+/, "") || "0", whys, changed, nmId };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}

/** NameMC hit: counter c (BigInt) of the NameMC job `prep`. Returns {ok, error?, png, id16, mojangId, whys}. */
export async function verifyNmHit(prep, c, targets) {
  try {
    const rgba = nmPixels(prep, c);
    const png = pngFromRaw(rawScanlines(rgba));                       // full pako encode, Mojang's format
    const dec = decodePng(png).rgba;
    if (!eq(dec, rgba)) return { ok: false, error: "decoded PNG differs from the pixels" };
    const full = await sha256Hex(namemcMessage(dec));          // WebCrypto + JS, from the decoded file
    const id16 = full.slice(0, 16);
    if (toHex(nmDigestFull(prep, c)).slice(0, 16) !== id16) return { ok: false, error: "NameMC id differs between paths" };
    if (namemcIdHex(dec) !== id16) return { ok: false, error: "NameMC id recompute differs" };
    const whys = matchTargets(id16, targets, true);
    if (!whys.length) return { ok: false, error: "GPU hit does not satisfy any target on the CPU" };
    // only the three knob pixels (alpha 0 -> 1) and the watermark may differ from the original
    const orig = prep.orig, knobs = new Set(prep.knobIdx);
    let nk = 0;
    for (let i = 0; i < W * H; i++) {
      let same = true;
      for (let k = 0; k < 4; k++) if (dec[i * 4 + k] !== orig[i * 4 + k]) same = false;
      if (same) continue;
      if (knobs.has(i)) {
        if (orig[i * 4 + 3] !== 0 || dec[i * 4 + 3] !== 1 || MASKS.base[i]) return { ok: false, error: `bad knob pixel ${i}` };
        nk++;
      } else if (prep.wmSet.has(i)) {
        if (orig[i * 4 + 3] !== 0 || dec[i * 4 + 3] !== 0) return { ok: false, error: `watermark pixel ${i} changed alpha` };
      } else return { ok: false, error: `unexpected change at pixel (${i % W},${Math.floor(i / W)})` };
    }
    if (nk !== 3) return { ok: false, error: `${nk} knob pixels changed, expected 3` };
    const mojangHex = await sha256Hex(png);
    return { ok: true, png, id16, mojangId: mojangHex.replace(/^0+/, "") || "0", whys, counter: c };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}
