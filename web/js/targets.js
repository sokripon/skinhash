// Search targets: trimmed-id prefixes, suffixes, leading zeros. Same semantics as the native tool:
// a prefix matches the id with leading zeros dropped (as in the URL), so a prefix may not start with 0.
export const MAX_HEX = 16;

export function parseHexList(text, kind, allowLeadingZero = false) {
  const out = [], errors = [];
  for (const raw of text.split(/[\s,;]+/).filter(Boolean)) {
    const s = raw.toLowerCase();
    if (!/^[0-9a-f]+$/.test(s)) errors.push(`"${raw}" is not hexadecimal (0-9, a-f).`);
    else if (s.length > MAX_HEX) errors.push(`"${raw}" is longer than ${MAX_HEX} digits.`);
    else if (kind === "prefix" && s[0] === "0" && !allowLeadingZero) errors.push(`Prefix "${raw}" starts with 0. The link drops leading zeros, so a prefix can't start with 0; use "minimum leading zeros" instead.`);
    else if (!out.includes(s)) out.push(s);
  }
  return { list: out, errors };
}

/** Targets: {prefixes:[], suffixes:[], zeros:Number} -> list of {kind, text}. */
export function targetList({ prefixes, suffixes, zeros }) {
  const t = [];
  prefixes.forEach((p) => t.push({ kind: "prefix", text: p }));
  suffixes.forEach((s) => t.push({ kind: "suffix", text: s }));
  if (zeros > 0) t.push({ kind: "zeros", text: String(zeros) });
  return t;
}
export const targetKey = (t) => `${t.kind}:${t.text}`;

/** GPU pattern (64-bit value/mask split into u32 halves). */
export function gpuPattern(t) {
  const n = t.text.length;
  const v = BigInt("0x" + t.text);
  const field = (1n << BigInt(4 * n)) - 1n;
  const shift = t.kind === "prefix" ? BigInt(4 * (16 - n)) : 0n;
  const val = v << shift, mask = field << shift;
  const lo = (x) => Number(x & 0xFFFFFFFFn) >>> 0, hi = (x) => Number(x >> 32n) >>> 0;
  return { vh: hi(val), vl: lo(val), mh: hi(mask), ml: lo(mask), suffix: t.kind === "suffix" };
}

export function gpuTargets(targets) {
  return {
    pats: targets.filter((t) => t.kind !== "zeros").map(gpuPattern),
    zeros: targets.find((t) => t.kind === "zeros") ? Number(targets.find((t) => t.kind === "zeros").text) : 0,
  };
}

/** Which of the given targets does this digest (64 hex chars) satisfy? */
export function matchTargets(hex, targets, fixed = false) {
  // fixed (NameMC): hex is the 16-digit id, matched as is. Mojang: prefixes match the id without leading zeros.
  const id = fixed ? hex : (hex.replace(/^0+/, "") || "0");
  const zeros = hex.length - (hex.replace(/^0+/, "") || "0").length;
  return targets.filter((t) =>
    t.kind === "prefix" ? id.startsWith(t.text) :
    t.kind === "suffix" ? hex.endsWith(t.text) :
    zeros >= Number(t.text));
}

/** Rough expected number of candidates for one hit. */
export function expectedTries(t, fixed = false) {
  if (t.kind === "zeros") return Math.pow(16, Number(t.text));
  return Math.pow(16, t.text.length) * (t.kind === "prefix" && !fixed ? 16 / 15 : 1);
}

/** NameMC kernel targets: patterns on the 64-bit id plus a zeros threshold on the id words. */
export function gpuNmTargets(targets) {
  const z = targets.find((t) => t.kind === "zeros");
  let zmode = 0, zthr = 0;
  if (z) { const n = Number(z.text); if (n <= 8) { zmode = 1; zthr = Math.pow(2, 32 - 4 * n); } else { zmode = 2; zthr = Math.pow(2, 64 - 4 * n); } }
  return { pats: targets.filter((t) => t.kind !== "zeros").map(gpuPattern), zmode, zthr };
}
