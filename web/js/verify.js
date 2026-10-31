// Every GPU hit is rebuilt on the CPU and checked before it is shown or offered for download.
import { decodePng, mojangPng, pngFromRaw, W, H } from "./png.js";
import { sha256, toHex } from "./hash.js";
import { decodePerm, fastFile, fastDigest, permRaw } from "./job.js";
import { isVisible } from "./model.js";
import { matchTargets } from "./targets.js";

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
    return { ok: true, png: full, hex, id: hex.replace(/^0+/, "") || "0", whys, changed };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}
