// Fast Mojang kernel in WGSL: a port of run_outer_fast in gpufinder.cu (step 7 of docs/optimizations.md).
// Used when the job is one SHA block and all permuted values have the same Huffman code length. Every code
// then sits at a bit position that does not depend on the permutation, so
//   - the message words are (thread part) | (constant part) | OR of per-(position, value) table words,
//   - the CRC-32 of the variable region is the XOR of per-(position, value) deltas (CRC is linear),
//   - Adler-b is a sum of per-(position, value) terms.
// WGSL has no templates: the shader text is generated per job with the (WI, KB) split and all job constants
// baked in as literals (constants are folded here, WGSL rejects overflowing constant expressions).
// Message words by level: 0 = job constant, 1 = depends on the thread only, 2 = varies per candidate.
import { K256, crcRegister, CRC_TABLE } from "./hash.js";
import { RIN, NPERM } from "./job.js";
import { hitCode, MAXHITS, NTEST_STRIDE } from "./wgsl.js";

const hex = (v) => `0x${(v >>> 0).toString(16)}u`;
const ror32 = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
const sg0 = (x) => (ror32(x, 7) ^ ror32(x, 18) ^ (x >>> 3)) >>> 0;
const sg1 = (x) => (ror32(x, 17) ^ ror32(x, 19) ^ (x >>> 10)) >>> 0;

/** Layout and tables of the fast kernel for job J (D = deriveJob(J)). Returns {why} when the job does not qualify. */
export function deriveFast(J, D) {
  const m = J.m, n = J.n, mo = m - RIN, P = D.P;
  if (J.nblk !== 1) return { why: "2+ SHA blocks" };
  if (m !== n || m > 20 || m < RIN + 1) return { why: "unsupported permutation shape" };
  const L = D.tab[0] >>> 16;
  for (let j = 0; j < m; j++) if ((D.tab[j] >>> 16) !== L) return { why: "unequal code lengths" };
  if (L < 1 || L > 15) return { why: "bad code length" };
  let pos = 8 * J.V0 + J.rem;
  for (let p = 0; p < mo; p++) pos += L + D.al[p];
  const posIn = pos;
  const off = [];
  let rel = 0;
  for (let q = 0; q < RIN; q++) { off.push(rel); rel += L + D.al[mo + q]; }
  const WI = Math.floor(posIn / 32), shIn = posIn % 32;
  if (shIn + off[RIN - 1] + L > 128) return { why: "inner codes span more than 4 words" };
  const putbits = (B, at, code, len) => { for (let i = 0; i < len; i++) if ((code >> i) & 1) B[(at + i) >> 3] |= 1 << ((at + i) & 7); };
  const Pb = new Uint8Array(80);
  for (let q = 0; q < RIN; q++) if (D.al[mo + q]) putbits(Pb, posIn + off[q] + L, D.ac[mo + q], D.al[mo + q]);
  pos = posIn + rel;
  putbits(Pb, pos, J.eob & 0xFFFF, J.eob >>> 16); pos += J.eob >>> 16;
  const Epad = (pos + 7) >> 3;
  if (Epad + 8 !== J.V0 + J.nvar || Epad + 8 > 64) return { why: "layout mismatch" };
  let asum = J.adl_a;
  for (let j = 0; j < J.K; j++) asum += J.alpha[j];
  for (let j = 0; j < n; j++) asum += J.vals[j];
  const aFin = asum % 65521;
  Pb[Epad + 2] = aFin >> 8; Pb[Epad + 3] = aFin & 255;
  for (let i = Epad + 8; i < 64; i++) Pb[i] = J.tmpl[i];
  const KB = Epad >> 2;
  if (WI < 1 || WI > 12 || KB + 3 > 16 || WI * 4 >= Epad + 4) return { why: "layout outside the supported range" };
  const sb = 8 * (Epad & 3);
  const Lc = Epad + 4 - 4 * WI, boff = Epad - 4 * WI;
  const be = (B, w) => ((B[4 * w] << 24) | (B[4 * w + 1] << 16) | (B[4 * w + 2] << 8) | B[4 * w + 3]) >>> 0;
  const pw = new Uint32Array(16);
  for (let i = WI; i < 16; i++) pw[i] = be(Pb, i);
  const c0 = crcRegister(0, Pb, 4 * WI, 4 * WI + Lc);
  const fy = new Uint32Array(RIN * 20 * 4), fcb = new Uint32Array(RIN * 20 * 2);
  for (let q = 0; q < RIN; q++) for (let vi = 0; vi < m; vi++) {
    const B = new Uint8Array(80);
    putbits(B, posIn + off[q], D.tab[vi] & 0xFFFF, L);
    for (let k = 0; k < 4; k++) fy[(q * 20 + vi) * 4 + k] = be(B, WI + k);
    fcb[(q * 20 + vi) * 2] = crcRegister(0, B, 4 * WI, 4 * WI + Lc);
    fcb[(q * 20 + vi) * 2 + 1] = (D.wt[mo + q] * D.val[vi]) >>> 0;
  }
  const gt = new Uint32Array(6 * 256);
  const zeros = new Uint8Array(Lc);
  for (let i = 0; i < 4; i++) for (let v = 0; v < 256; v++) gt[i * 256 + v] = crcRegister((v << (8 * i)) >>> 0, zeros, 0, Lc);
  for (let j = 0; j < 2; j++) for (let v = 0; v < 256; v++) {
    const e = new Uint8Array(Lc); e[boff + j] = v;
    gt[(4 + j) * 256 + v] = crcRegister(0, e, 0, Lc);
  }
  const sel = new Uint32Array(NPERM);   // byte q = bit shift (0, 8, 16, 24) of the free-value byte at inner position q
  for (let t = 0; t < NPERM; t++) for (let q = 0; q < RIN; q++) sel[t] |= (8 * D.perm[t * RIN + q]) << (8 * q);
  const code = new Uint32Array(64);
  for (let j = 0; j < m; j++) code[j] = D.tab[j] & 0xFFFF;
  // which of the four table words WI..WI+3 are ever non-zero
  const nzY = [0, 1, 2, 3].map((k) => { for (let i = 0; i < RIN * 20; i++) if (fy[i * 4 + k]) return true; return false; });
  // level of every message word
  const zWords = sb ? [KB, KB + 1, KB + 2] : [KB, KB + 1];
  const lvl = new Array(64).fill(0);
  for (let i = 0; i < 16; i++) {
    const varies = (i >= WI && i < WI + 4 && nzY[i - WI]) || zWords.includes(i);
    lvl[i] = varies ? 2 : i <= WI ? 1 : 0;
  }
  for (let i = 16; i < 64; i++) lvl[i] = Math.max(lvl[i - 16], lvl[i - 15], lvl[i - 7], lvl[i - 2]);
  return { ok: true, L, m, mo, WI, KB, sb, shIn, pw, c0, fy, fcb, gt, sel, code, nzY, zWords, lvl, key: `fast:${WI}:${KB}:${sb}:${L}:${m}` };
}

/** Layout of the constants storage buffer (word offsets) and its contents for the fast kernel. */
export function buildFastConsts(D, F) {
  const parts = [["CRC", CRC_TABLE], ["GT", F.gt], ["CODE", F.code], ["VAL", D.val], ["FY", F.fy], ["FCB", F.fcb], ["SEL", F.sel]];
  const off = {};
  let o = 0;
  for (const [name, arr] of parts) { off[name] = o; o += arr.length; }
  const buf = new Uint32Array(o);
  for (const [name, arr] of parts) buf.set(arr, off[name]);
  return { off, buf };
}

/**
 * JS model of the fast kernel: the same tables and the same OR / XOR / sum steps, for one permutation of the n
 * values. Returns the 16 message words of the single SHA block and the Adler-b / CRC it used (tests only).
 */
export function fastModel(J, D, F, perm) {
  const { WI, mo, m, L } = F;
  const P = D.P;
  const B = new Uint8Array(80);
  B.set(J.tmpl.subarray(0, J.V0));
  B[J.V0] = J.firstbits;
  let pos = 8 * J.V0 + J.rem;
  const putbits = (code, len) => { for (let i = 0; i < len; i++, pos++) if ((code >> i) & 1) B[pos >> 3] |= 1 << (pos & 7); };
  const vi = (v) => { for (let j = 0; j < m; j++) if (J.vals[j] === v) return j; throw new Error("value not in set"); };
  let bacc = P.b_const;
  for (let p = 0; p < mo; p++) {
    const j = vi(perm[p]);
    putbits(D.tab[j] & 0xFFFF, L);
    if (D.al[p]) putbits(D.ac[p], D.al[p]);
    bacc += D.wt[p] * D.val[j];
  }
  const tw = [];
  for (let i = 0; i < WI; i++) tw.push(((B[4 * i] << 24) | (B[4 * i + 1] << 16) | (B[4 * i + 2] << 8) | B[4 * i + 3]) >>> 0);
  const pa = (B[4 * WI] | (B[4 * WI + 1] << 8) | (B[4 * WI + 2] << 16) | (B[4 * WI + 3] << 24)) >>> 0;
  const crc = crcRegister(P.crc_w0, B, 4 * P.W0w, 4 * WI);
  const xc = (crc ^ pa) >>> 0;
  let V = (F.gt[xc & 255] ^ F.gt[256 + ((xc >>> 8) & 255)] ^ F.gt[512 + ((xc >>> 16) & 255)] ^ F.gt[768 + (xc >>> 24)] ^ F.c0) >>> 0;
  const Y = [0, 0, 0, 0];
  let kb = 0;
  for (let q = 0; q < RIN; q++) {
    const j = vi(perm[mo + q]), e = q * 20 + j;
    for (let k = 0; k < 4; k++) Y[k] |= F.fy[e * 4 + k];
    V ^= F.fcb[e * 2];
    kb += F.fcb[e * 2 + 1];
  }
  const b = (bacc + kb) % 65521;
  const cc = ~(V ^ F.gt[1024 + (b >>> 8)] ^ F.gt[1280 + (b & 255)]) >>> 0;
  const w = new Uint32Array(16);
  for (let i = 0; i < 16; i++) w[i] = i < WI ? tw[i] : F.pw[i];
  w[WI] |= (((pa & 255) << 24) | (((pa >>> 8) & 255) << 16) | (((pa >>> 16) & 255) << 8) | (pa >>> 24)) >>> 0;
  for (let k = 0; k < 4; k++) w[WI + k] |= Y[k];
  const Zhi = (b << 16) >>> 0, sb = F.sb;
  const Z = sb ? [Zhi >>> sb, ((cc >>> sb) | (Zhi << (32 - sb))) >>> 0, (cc << (32 - sb)) >>> 0] : [Zhi, cc];
  Z.forEach((z, k) => { w[F.KB + k] |= z; });
  return { w, b, cc };
}

// ---------- shader text ----------
const ADD = (parts) => parts.join(" + ");

/** Value of the schedule: a literal (c) or a WGSL expression (e). */
const C = (c) => ({ c: c >>> 0 });
const E = (e) => ({ e });
const txt = (v) => (v.e !== undefined ? v.e : hex(v.c));

function sumValues(vals) {
  let c = 0;
  const es = [];
  for (const v of vals) { if (v.e === undefined) c = (c + v.c) >>> 0; else es.push(v.e); }
  if (!es.length) return C(c);
  return E(`(${ADD(c ? [hex(c), ...es] : es)})`);
}
const sig0v = (v) => (v.e === undefined ? C(sg0(v.c)) : E(`sg0(${v.e})`));
const sig1v = (v) => (v.e === undefined ? C(sg1(v.c)) : E(`sg1(${v.e})`));

/** One SHA-256 round with state names rotated by r. */
function round(names, r, kw) {
  const v = (j) => names[(((j - r) % 8) + 8) % 8];
  const [A, B, C_, D_, E_, F_, G, H] = [0, 1, 2, 3, 4, 5, 6, 7].map(v);
  return `{ let t1 = ${H} + (ror(${E_}, 6u) ^ ror(${E_}, 11u) ^ ror(${E_}, 25u)) + (${G} ^ (${E_} & (${F_} ^ ${G}))) + ${kw};
${D_} = ${D_} + t1;
${H} = t1 + (ror(${A}, 2u) ^ ror(${A}, 13u) ^ ror(${A}, 22u)) + ((${A} & ${B}) | (${C_} & (${A} | ${B}))); }
`;
}

function outerPart(D, F, P) {
  const { L, mo, m, WI } = F;
  const maskm = ((1 << m) - 1) >>> 0;
  let s = `  var lo = ${hex(P.acc0)}; var hi = 0u; var crc = ${hex(P.crc_w0)}; var bacc = 0u; var used = 0u; var v = gid;\n`;
  for (let k = P.W0w; k < WI; k++) s += `  var tw${k}: u32 = 0u;\n`;
  let nb = P.nb0, nw = P.W0w;
  for (let p = 0; p < mo; p++) {
    const r = m - p, clen = L + D.al[p];
    const code = D.al[p] ? `(s_code[vi] | ${hex((D.ac[p] << L) >>> 0)})` : "s_code[vi]";
    s += `  {
    let t = DP[doff + ${p}u] + v;
    let q = t / ${r}u;
    v = q;
    let vi = nth_set(~used & ${hex(maskm)}, t - q * ${r}u);
    used |= 1u << vi;
    bacc += ${D.wt[p]}u * s_val[vi];
    let c_ = ${code};
    lo |= c_ << ${nb}u;\n`;
    if (nb + clen > 32) s += `    hi = c_ >> ${32 - nb}u;\n`;
    nb += clen;
    if (nb >= 32) {
      s += `    tw${nw} = bswap(lo);
    var cc_ = crc ^ lo;
    cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
    cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
    cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
    cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
    crc = cc_;
    lo = ${nb > 32 ? "hi" : "0u"};\n`;
      nw++; nb -= 32;
    }
    s += "  }\n";
  }
  if (nw !== WI || nb !== F.shIn) throw new Error(`fast layout: outer part ends at word ${nw} bit ${nb}, expected ${WI}/${F.shIn}`);
  return s;
}

export function genFastShader(J, D, F, off, WG = 256, contains = false) {
  const P = D.P, { WI, KB, sb, L, m, mo, lvl } = F;
  const tmplWord = (k) => D.tmpl[k];
  let s = `
@group(0) @binding(0) var<storage, read> KT: array<u32>;
@group(0) @binding(1) var<storage, read> PT: array<u32>;
@group(0) @binding(2) var<storage, read> DP: array<u32>;
@group(0) @binding(3) var<storage, read_write> HITS: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> TOUT: array<u32>;

var<workgroup> s_crc: array<u32, 256>;
var<workgroup> s_gt: array<u32, 1536>;
var<workgroup> s_code: array<u32, 64>;
var<workgroup> s_val: array<u32, 64>;
var<workgroup> s_y: array<vec4<u32>, ${RIN * 20}>;
var<workgroup> s_cb: array<vec2<u32>, ${RIN * 20}>;

fn bswap(x: u32) -> u32 {
  return (x << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | (x >> 24u);
}
fn shl2(a: u32, b: u32, sh: u32) -> u32 {
  return select((a << sh) | (b >> (32u - sh)), a, sh == 0u);
}
fn ror(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn sg0(x: u32) -> u32 { return ror(x, 7u) ^ ror(x, 18u) ^ (x >> 3u); }
fn sg1(x: u32) -> u32 { return ror(x, 17u) ^ ror(x, 19u) ^ (x >> 10u); }
/** index of the k-th (0-based) set bit of a value below 2^20 */
fn nth_set(x_in: u32, k_in: u32) -> u32 {
  var x = x_in; var k = k_in; var p = 0u;
  var c = countOneBits(x & 0xFFFFu);
  var up = k >= c; k = select(k, k - c, up); x = select(x & 0xFFFFu, x >> 16u, up); p = select(0u, 16u, up);
  c = countOneBits(x & 0xFFu);
  up = k >= c; k = select(k, k - c, up); x = select(x & 0xFFu, x >> 8u, up); p += select(0u, 8u, up);
  c = countOneBits(x & 0xFu);
  up = k >= c; k = select(k, k - c, up); x = select(x & 0xFu, x >> 4u, up); p += select(0u, 4u, up);
  c = countOneBits(x & 0x3u);
  up = k >= c; k = select(k, k - c, up); x = select(x & 0x3u, x >> 2u, up); p += select(0u, 2u, up);
  return p + select(0u, 1u, k >= (x & 1u));
}
fn load_shared(li: u32, nthreads: u32) {
  for (var i = li; i < 256u; i += nthreads) { s_crc[i] = KT[${off.CRC}u + i]; }
  for (var i = li; i < 1536u; i += nthreads) { s_gt[i] = KT[${off.GT}u + i]; }
  for (var i = li; i < 64u; i += nthreads) { s_code[i] = KT[${off.CODE}u + i]; s_val[i] = KT[${off.VAL}u + i]; }
  for (var i = li; i < ${RIN * 20}u; i += nthreads) {
    s_y[i] = vec4<u32>(KT[${off.FY}u + 4u * i], KT[${off.FY}u + 4u * i + 1u], KT[${off.FY}u + 4u * i + 2u], KT[${off.FY}u + 4u * i + 3u]);
    s_cb[i] = vec2<u32>(KT[${off.FCB}u + 2u * i], KT[${off.FCB}u + 2u * i + 1u]);
  }
  workgroupBarrier();
}

fn run(test: bool, doff: u32, gid: u32, tlo: u32, thi: u32, tid: u32) {
${outerPart(D, F, P)}`;
  // inner free values
  s += `  var fx = ~used & ${hex(((1 << m) - 1) >>> 0)};
  let f0 = countTrailingZeros(fx); fx = fx & (fx - 1u);
  let f1 = countTrailingZeros(fx); fx = fx & (fx - 1u);
  let f2 = countTrailingZeros(fx); fx = fx & (fx - 1u);
  let f3 = countTrailingZeros(fx);
  let fv = f0 | (f1 << 8u) | (f2 << 16u) | (f3 << 24u);
  let bt = bacc + ${P.b_const}u;
  let xc = crc ^ lo;
  let vcrc = s_gt[xc & 255u] ^ s_gt[256u + ((xc >> 8u) & 255u)] ^ s_gt[512u + ((xc >> 16u) & 255u)] ^ s_gt[768u + (xc >> 24u)] ^ ${hex(F.c0)};
`;
  // thread-level message words and schedule
  const V = new Array(64).fill(null);
  for (let i = 0; i < 16; i++) {
    if (i < WI) V[i] = i < P.W0w ? C(tmplWord(i)) : E(`tw${i}`);
    else if (i === WI && lvl[i] <= 1) { s += `  let tw${i} = ${hex(F.pw[i])} | bswap(lo);\n`; V[i] = E(`tw${i}`); }
    else if (i === WI) { s += `  let tw${i} = ${hex(F.pw[i])} | bswap(lo);\n`; }
    else if (lvl[i] <= 1) V[i] = C(F.pw[i]);
  }
  const PWV = new Array(64).fill(null);
  for (let i = 16; i < 64; i++) {
    const idx = [i - 16, i - 15, i - 7, i - 2];
    if (lvl[i] <= 1) {
      const v = sumValues([V[idx[0]], sig0v(V[idx[1]]), V[idx[2]], sig1v(V[idx[3]])]);
      if (v.e === undefined) V[i] = v; else { s += `  let ws${i} = ${v.e};\n`; V[i] = E(`ws${i}`); }
    } else {
      const th = [];
      if (lvl[idx[0]] <= 1) th.push(V[idx[0]]);
      if (lvl[idx[1]] <= 1) th.push(sig0v(V[idx[1]]));
      if (lvl[idx[2]] <= 1) th.push(V[idx[2]]);
      if (lvl[idx[3]] <= 1) th.push(sig1v(V[idx[3]]));
      const v = sumValues(th);
      if (v.e === undefined) PWV[i] = v; else { s += `  let pw${i} = ${v.e};\n`; PWV[i] = E(`pw${i}`); }
    }
  }
  // key + word of the rounds that do not depend on the candidate
  const KW = new Array(64).fill(null);
  for (let i = WI; i < 64; i++) {
    if (lvl[i] === 2) continue;
    const v = V[i];
    if (v.e === undefined) KW[i] = hex(K256[i] + v.c);
    else { s += `  let kt${i} = ${hex(K256[i])} + ${v.e};\n`; KW[i] = `kt${i}`; }
  }
  // rounds 0..WI-1 once per thread
  const ha = ["ha", "hb", "hc", "hd", "he", "hf", "hg", "hh"];
  ha.forEach((nme, j) => { s += `  var ${nme} = ${hex(P.mid[j])};\n`; });
  for (let i = 0; i < WI; i++) {
    const v = V[i];
    s += round(ha, i, v.e === undefined ? hex(K256[i] + v.c) : `${hex(K256[i])} + ${v.e}`);
  }
  // candidate loop
  const nzY = F.nzY;
  const yOr = (k) => {
    const parts = [0, 1, 2, 3].map((q) => `y${q}.${"xyzw"[k]}`);
    return parts.join(" | ");
  };
  s += `  var zbest = 0u;
  for (var tt = tlo; tt < thi; tt++) {
    let sel = KT[${off.SEL}u + tt];
`;
  for (let q = 0; q < RIN; q++) s += `    let i${q} = ${q * 20}u + ((fv >> ((sel >> ${8 * q}u) & 31u)) & 255u);\n`;
  for (let q = 0; q < RIN; q++) s += `    let y${q} = s_y[i${q}]; let k${q} = s_cb[i${q}];\n`;
  s += `    let bsum = (bt + k0.y + k1.y + k2.y + k3.y) % 65521u;
    let ccrc = ~(vcrc ^ k0.x ^ k1.x ^ k2.x ^ k3.x ^ s_gt[1024u + (bsum >> 8u)] ^ s_gt[1280u + (bsum & 255u)]);
    let zhi = bsum << 16u;
`;
  const Zx = sb ? [`zhi >> ${sb}u`, `(ccrc >> ${sb}u) | (zhi << ${32 - sb}u)`, `ccrc << ${32 - sb}u`] : ["zhi", "ccrc"];
  // candidate-level message words
  for (let i = 0; i < 64; i++) {
    if (lvl[i] !== 2) continue;
    if (i < 16) {
      const parts = [i === WI ? `tw${i}` : hex(F.pw[i])];
      if (i >= WI && i < WI + 4 && nzY[i - WI]) parts.push(yOr(i - WI));
      if (i >= KB && i < KB + Zx.length) parts.push(`(${Zx[i - KB]})`);
      s += `    let w${i} = ${parts.join(" | ")};\n`;
    } else {
      const idx = [i - 16, i - 15, i - 7, i - 2];
      const parts = [];
      if (PWV[i].e !== undefined || PWV[i].c) parts.push(txt(PWV[i]));
      if (lvl[idx[0]] === 2) parts.push(`w${idx[0]}`);
      if (lvl[idx[1]] === 2) parts.push(`sg0(w${idx[1]})`);
      if (lvl[idx[2]] === 2) parts.push(`w${idx[2]}`);
      if (lvl[idx[3]] === 2) parts.push(`sg1(w${idx[3]})`);
      s += `    let w${i} = ${ADD(parts)};\n`;
    }
  }
  const ra = ["ra", "rb", "rc", "rd", "re", "rf", "rg", "rh"];
  ha.forEach((nme, j) => { s += `    var ${ra[j]} = ${ha[(((j - WI) % 8) + 8) % 8]};\n`; });
  for (let i = WI; i < 64; i++) s += "    " + round(ra, i - WI, lvl[i] === 2 ? `${hex(K256[i])} + w${i}` : KW[i]).replaceAll("\n", "\n    ");
  const fin = (j) => ra[(((j - (64 - WI)) % 8) + 8) % 8];
  for (let j = 0; j < 8; j++) s += `    let s${j} = ${hex(P.mid[j])} + ${fin(j)};\n`;
  s += `    if (test) {
      TOUT[tid * 8u + 0u] = s0; TOUT[tid * 8u + 1u] = s1; TOUT[tid * 8u + 2u] = s2; TOUT[tid * 8u + 3u] = s3;
      TOUT[tid * 8u + 4u] = s4; TOUT[tid * 8u + 5u] = s5; TOUT[tid * 8u + 6u] = s6; TOUT[tid * 8u + 7u] = s7;
      return;
    }
    ${hitCode(contains)}
  }
  if (zbest > 0u && zbest > atomicLoad(&HITS[1])) { atomicMax(&HITS[1], zbest); }
}

@compute @workgroup_size(${WG})
fn search_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  load_shared(li, ${WG}u);
  let gid = wid.x * ${WG}u + li;
  if (gid >= DP[0]) { return; }
  run(false, 1u, gid, 0u, ${NPERM}u, 0u);
}

@compute @workgroup_size(64)
fn test_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  load_shared(li, 64u);
  let gi = wid.x * 64u + li;
  if ((gi + 1u) * ${NTEST_STRIDE}u <= arrayLength(&DP)) {
    let b = gi * ${NTEST_STRIDE}u;
    run(true, b, DP[b + 16u], DP[b + 17u], DP[b + 17u] + 1u, gi);
  }
}
`;
  return s;
}
