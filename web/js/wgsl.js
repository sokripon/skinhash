// WGSL kernel generator: a port of gpufinder.cu. Job constants are baked into the shader text,
// tables go into one read-only storage buffer. u32 only: 64-bit values are split.
import { K256, crcRegister, CRC_TABLE } from "./hash.js";
import { RIN, NPERM } from "./job.js";

export const MAXHITS = 1024;
export const MAXPAT = 32;
export const HITS_WORDS = 2 + 2 * MAXHITS;
export const NTEST_STRIDE = 20;

/** Host side of derive() in gpufinder.cu: all per-job constants. */
export function deriveJob(J) {
  const n = J.n, K = J.K, m = J.m, N = n + K;
  if (m < RIN + 1 || m > 31) throw new Error(`unsupported job (m=${m})`);
  const P = { m, nblk: J.nblk };
  const W0 = J.V0 & ~3, s0 = J.V0 - W0;
  P.W0w = W0 / 4;
  let acc0 = 0;
  for (let i = 0; i < s0; i++) acc0 |= J.tmpl[W0 + i] << (8 * i);
  acc0 |= J.firstbits << (8 * s0);
  P.acc0 = acc0 >>> 0; P.nb0 = 8 * s0 + J.rem;
  {
    let c = crcRegister(0xFFFFFFFF, J.ref, 37, 41);
    c = crcRegister(c, J.ref, 41, J.B0 + W0);
    P.crc_w0 = c;
    c = crcRegister(c, J.ref, J.B0 + W0, J.B0 + J.V0);
    if ((~c >>> 0) !== J.crc0) throw new Error("CRC prefix state mismatch");
  }
  const D = { val: new Uint32Array(64), tab: new Uint32Array(64), wt: new Uint32Array(64), ac: new Uint32Array(64), al: new Uint32Array(64) };
  for (let j = 0; j < n; j++) { D.val[j] = J.vals[j]; D.tab[j] = J.tab[J.vals[j]]; D.wt[j] = N - (j + Math.floor(j / 3)); }
  let bconst = J.adl_b + N * J.adl_a, asum = J.adl_a;
  for (let j = 0; j < K; j++) { bconst += (N - (4 * j + 3)) * J.alpha[j]; asum += J.alpha[j]; }
  for (let j = 0; j < n; j++) asum += J.vals[j];
  for (let p = 0; p < m; p++) if (p % 3 === 2) { const e = J.tab[J.alpha[Math.floor(p / 3)]]; D.ac[p] = e & 0xFFFF; D.al[p] = e >>> 16; }
  for (let p = m; p < n; p++) bconst += D.wt[p] * J.vals[p];
  P.b_const = bconst >>> 0;
  const a_fin = asum % 65521;
  let pos = 8 * J.V0 + J.rem;
  for (let p = 0; p < m; p++) { pos += D.tab[p] >>> 16; if (p % 3 === 2) pos += D.al[p]; }
  P.nwL = Math.floor(pos / 32); P.nbL = pos % 32;
  const Pb = new Uint8Array(J.nblk * 64 + 16);
  const putbits = (code, len) => { for (let i = 0; i < len; i++, pos++) if ((code >> i) & 1) Pb[pos >> 3] |= 1 << (pos & 7); };
  for (let p = m; p < n; p++) {
    putbits(D.tab[p] & 0xFFFF, D.tab[p] >>> 16);
    if (p % 3 === 2) { const e = J.tab[J.alpha[Math.floor(p / 3)]]; putbits(e & 0xFFFF, e >>> 16); }
  }
  putbits(J.eob & 0xFFFF, J.eob >>> 16);
  const Epad = (pos + 7) >> 3;
  if (Epad + 8 !== J.V0 + J.nvar) throw new Error(`variable region length mismatch (${Epad + 8} vs ${J.V0 + J.nvar})`);
  Pb[Epad + 2] = a_fin >> 8; Pb[Epad + 3] = a_fin & 255;
  for (let i = Epad + 8; i < J.nblk * 64; i++) Pb[i] = J.tmpl[i];
  P.boff = Epad - 4 * P.nwL;
  P.Lc = P.boff + 4;
  P.KW = (Epad + 8 - 4 * P.nwL + 3) >> 2;
  if (Epad < 4 * P.nwL || P.KW > 6 || P.nwL + P.KW > J.nblk * 16) throw new Error(`post image does not fit (KW=${P.KW})`);
  P.post = [];
  for (let k = 0; k < P.KW; k++) {
    const o = 4 * (P.nwL + k);
    P.post.push((Pb[o] | Pb[o + 1] << 8 | Pb[o + 2] << 16 | Pb[o + 3] << 24) >>> 0);
  }
  D.gt = new Uint32Array(6 * 256);
  const zeros = new Uint8Array(P.Lc);
  const pm = Pb.subarray(4 * P.nwL, 4 * P.nwL + P.Lc);
  for (let i = 0; i < 4; i++) for (let v = 0; v < 256; v++) D.gt[i * 256 + v] = crcRegister((v << (8 * i)) >>> 0, zeros, 0, P.Lc);
  P.c0 = crcRegister(0, pm, 0, P.Lc);
  for (let j = 0; j < 2; j++) for (let v = 0; v < 256; v++) {
    const e = new Uint8Array(P.Lc); e[P.boff + j] = v;
    D.gt[(4 + j) * 256 + v] = crcRegister(0, e, 0, P.Lc);
  }
  D.tmpl = new Uint32Array(J.nblk * 16);
  for (let w = 0; w < J.nblk * 16; w++) D.tmpl[w] = (J.tmpl[4*w] << 24 | J.tmpl[4*w+1] << 16 | J.tmpl[4*w+2] << 8 | J.tmpl[4*w+3]) >>> 0;
  P.mid = Array.from(J.mid);
  // orderings of the inner positions: lexicographic permutations of 0..RIN-1
  D.perm = new Uint32Array(NPERM * RIN);
  let t = 0;
  const rec = (cur, rest) => { if (!rest.length) { D.perm.set(cur, t * RIN); t++; return; } for (let i = 0; i < rest.length; i++) rec([...cur, rest[i]], rest.filter((_, k) => k !== i)); };
  rec([], [0, 1, 2, 3]);
  D.P = P;
  return D;
}

/** Layout of the constants storage buffer (word offsets) and its contents. */
export function buildConsts(D, nblk) {
  const off = {};
  let o = 0;
  const add = (name, arr) => { off[name] = o; o += arr.length; return arr; };
  const parts = [
    ["CRC", CRC_TABLE], ["GT", D.gt], ["TAB", D.tab], ["VAL", D.val], ["WT", D.wt], ["AC", D.ac], ["AL", D.al], ["PERM", D.perm], ["TMPL", D.tmpl],
  ];
  const buf = new Uint32Array(256 + 1536 + 64 * 5 + 96 + nblk * 16);
  for (const [name, arr] of parts) { add(name, arr); buf.set(arr, off[name]); }
  return { off, buf };
}

const hex = (v) => `0x${(v >>> 0).toString(16)}u`;
const ror = (x, n) => `((${x} >> ${n}u) | (${x} << ${32 - n}u))`;

function emit(lo, hi, nb, nw, crc, comb, clen) {
  return `{
    let c_ = ${comb}; let l_ = ${clen};
    ${lo} |= c_ << ${nb};
    ${hi} |= select(c_ >> (32u - ${nb}), 0u, ${nb} == 0u);
    ${nb} += l_;
    if (${nb} >= 32u) {
      let w_ = ${lo};
      img[${nw}] = bswap(w_);
      var cc_ = ${crc} ^ w_;
      cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
      cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
      cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
      cc_ = s_crc[cc_ & 255u] ^ (cc_ >> 8u);
      ${crc} = cc_;
      ${nw} += 1u; ${lo} = ${hi}; ${hi} = 0u; ${nb} -= 32u;
    }
  }`;
}

function shaBlock(blk) {
  let c = "{\n";
  for (let i = 0; i < 16; i++) c += `var w${i} = img[${blk * 16 + i}];\n`;
  c += "var a = s0; var b = s1; var c = s2; var d = s3; var e = s4; var f = s5; var g = s6; var h = s7;\n";
  const base = ["a", "b", "c", "d", "e", "f", "g", "h"];
  for (let i = 0; i < 64; i++) {
    const w = `w${i & 15}`;
    if (i >= 16) {
      const w15 = `w${(i + 1) & 15}`, w7 = `w${(i + 9) & 15}`, w2 = `w${(i + 14) & 15}`;
      c += `${w} = ${w} + (${ror(w15, 7)} ^ ${ror(w15, 18)} ^ (${w15} >> 3u)) + ${w7} + (${ror(w2, 17)} ^ ${ror(w2, 19)} ^ (${w2} >> 10u));\n`;
    }
    const v = (j) => base[(((j - i) % 8) + 8) % 8];
    const [A, B, C, D_, E, F, G, Hh] = [0, 1, 2, 3, 4, 5, 6, 7].map(v);
    c += `{ let t1 = ${Hh} + (${ror(E, 6)} ^ ${ror(E, 11)} ^ ${ror(E, 25)}) + (${G} ^ (${E} & (${F} ^ ${G}))) + ${hex(K256[i])} + ${w};
${D_} = ${D_} + t1;
${Hh} = t1 + (${ror(A, 2)} ^ ${ror(A, 13)} ^ ${ror(A, 22)}) + ((${A} & ${B}) | (${C} & (${A} | ${B}))); }\n`;
  }
  c += "s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;\n}\n";
  return c;
}

function runOuter(name, test, D, off, WGs) {
  const P = D.P, m = P.m, mo = m - RIN, nblk = P.nblk, KW = P.KW;
  const maskm = ((1 << m) - 1) >>> 0;
  let s = `fn ${name}(doff: u32, gid: u32, tlo: u32, thi: u32, tid: u32) {
  var img: array<u32, ${nblk * 16}>;
  for (var i = 0u; i < ${nblk * 16}u; i++) { img[i] = KT[${off.TMPL}u + i]; }
  var accLo = ${hex(P.acc0)}; var accHi = 0u; var nb = ${P.nb0}u; var nw = ${P.W0w}u; var crc = ${hex(P.crc_w0)};
  var bacc = 0u; var used = 0u; var v = gid;
  for (var p = 0u; p < ${mo}u; p++) {
    let r = ${m}u - p;
    let t = DP[doff + p] + v;
    let q = t / r;
    let dg = t - q * r;
    v = q;
    var x = ~used & ${hex(maskm)};
    for (var i = 0u; i < dg; i++) { x = x & (x - 1u); }
    let vi = countTrailingZeros(x);
    used |= 1u << vi;
    let e = s_tab[vi];
    let len = e >> 16u;
    let comb = (e & 0xFFFFu) | (KT[${off.AC}u + p] << len);
    let clen = len + KT[${off.AL}u + p];
    bacc += KT[${off.WT}u + p] * s_val[vi];
    ${emit("accLo", "accHi", "nb", "nw", "crc", "comb", "clen")}
  }
  var E: array<vec4<u32>, ${RIN * RIN}>;
  {
    var fx = ~used & ${hex(maskm)};
`;
  for (let pi = 0; pi < RIN; pi++) {
    s += `    { let vi = countTrailingZeros(fx); fx = fx & (fx - 1u);
      let e = s_tab[vi]; let len = e >> 16u; let bv = s_val[vi];\n`;
    for (let q = 0; q < RIN; q++) {
      const p = mo + q;
      s += `      E[${q * RIN + pi}] = vec4<u32>((e & 0xFFFFu) | (KT[${off.AC + p}u] << len), len + KT[${off.AL + p}u], KT[${off.WT + p}u] * bv, 0u);\n`;
    }
    s += "    }\n";
  }
  s += `  }
  var zbest = 0u;
  for (var tt = tlo; tt < thi; tt++) {
    var a2lo = accLo; var a2hi = accHi; var nb2 = nb; var nw2 = nw; var crc2 = crc; var b2 = bacc;
`;
  for (let q = 0; q < RIN; q++) {
    s += `    { let pi = KT[${off.PERM}u + tt * 4u + ${q}u]; let e = E[${q * RIN}u + pi]; b2 += e.z;
      ${emit("a2lo", "a2hi", "nb2", "nw2", "crc2", "e.x", "e.y")} }\n`;
  }
  for (let k = 0; k < KW; k++) s += `    var vv${k} = ${hex(P.post[k])};\n`;
  s += `    vv0 |= a2lo;
    let b = (${P.b_const}u + b2) % 65521u;
    let bbe = ((b >> 8u) | (b << 8u)) & 0xFFFFu;\n`;
  const sh = (P.boff & 3) * 8, kb = P.boff >> 2;
  s += `    vv${kb} |= bbe << ${sh}u;\n`;
  if (sh > 16 && kb + 1 < KW) s += `    vv${kb + 1} |= bbe >> ${32 - sh}u;\n`;
  s += `    let xx = crc2 ^ a2lo;
    var cr = GT[(xx & 255u)] ^ GT[256u + ((xx >> 8u) & 255u)] ^ GT[512u + ((xx >> 16u) & 255u)] ^ GT[768u + (xx >> 24u)] ^ ${hex(P.c0)} ^ GT[1024u + (b >> 8u)] ^ GT[1280u + (b & 255u)];
    cr = ~cr;
    let bc = bswap(cr);
`;
  const off2 = P.boff + 4, sh2 = (off2 & 3) * 8, kc = off2 >> 2;
  s += `    vv${kc} |= bc << ${sh2}u;\n`;
  if (sh2 > 0 && kc + 1 < KW) s += `    vv${kc + 1} |= bc >> ${32 - sh2}u;\n`;
  for (let k = 0; k < KW; k++) s += `    img[${P.nwL + k}] = bswap(vv${k});\n`;
  P.mid.forEach((x, i) => { s += `    var s${i} = ${hex(x)};\n`; });
  for (let b = 0; b < nblk; b++) s += shaBlock(b);
  if (test) {
    s += `    TOUT[tid * 8u + 0u] = s0; TOUT[tid * 8u + 1u] = s1; TOUT[tid * 8u + 2u] = s2; TOUT[tid * 8u + 3u] = s3;
    TOUT[tid * 8u + 4u] = s4; TOUT[tid * 8u + 5u] = s5; TOUT[tid * 8u + 6u] = s6; TOUT[tid * 8u + 7u] = s7;
    return;
  }
}
`;
    return s;
  }
  s += `    let zn = select(select(16u, 8u + (countLeadingZeros(s1) >> 2u), s1 != 0u), countLeadingZeros(s0) >> 2u, s0 != 0u);
    zbest = max(zbest, zn);
    var hit = PT[1] != 0u && zn >= PT[1];
    let np = PT[0];
    if (np != 0u) {
      var th = 0u; var tl = 0u;
      if (s0 != 0u) { let sh = countLeadingZeros(s0) & 28u; th = shl2(s0, s1, sh); tl = shl2(s1, s2, sh); }
      else if (s1 != 0u) { let sh = countLeadingZeros(s1) & 28u; th = shl2(s1, s2, sh); tl = shl2(s2, s3, sh); }
      else if (s2 != 0u) { let sh = countLeadingZeros(s2) & 28u; th = shl2(s2, s3, sh); tl = s3 << sh; }
      else if (s3 != 0u) { let sh = countLeadingZeros(s3) & 28u; th = s3 << sh; }
      for (var i = 0u; i < np; i++) {
        let pb = 2u + 5u * i;
        let suf = PT[pb + 4u] != 0u;
        let vh = select(th, s6, suf);
        let vl = select(tl, s7, suf);
        if ((vh & PT[pb + 2u]) == PT[pb] && (vl & PT[pb + 3u]) == PT[pb + 1u]) { hit = true; break; }
      }
    }
    if (hit) {
      let pos = atomicAdd(&HITS[0], 1u);
      if (pos < ${MAXHITS}u) { atomicStore(&HITS[2u + 2u * pos], gid); atomicStore(&HITS[3u + 2u * pos], tt); }
    }
  }
  if (zbest > 0u && zbest > atomicLoad(&HITS[1])) { atomicMax(&HITS[1], zbest); }
}
`;
  return s;
}

export function genShader(D, off, WG = 256, nTests = 24) {
  const TW = 64;
  return `
@group(0) @binding(0) var<storage, read> KT: array<u32>;
@group(0) @binding(1) var<storage, read> PT: array<u32>;
@group(0) @binding(2) var<storage, read> DP: array<u32>;
@group(0) @binding(3) var<storage, read_write> HITS: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> TOUT: array<u32>;

var<workgroup> s_crc: array<u32, 256>;
var<workgroup> s_gt: array<u32, 1536>;
var<workgroup> s_tab: array<u32, 64>;
var<workgroup> s_val: array<u32, 64>;

fn bswap(x: u32) -> u32 {
  return (x << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | (x >> 24u);
}
fn shl2(a: u32, b: u32, sh: u32) -> u32 {
  return select((a << sh) | (b >> (32u - sh)), a, sh == 0u);
}
fn load_shared(li: u32, nthreads: u32) {
  for (var i = li; i < 256u; i += nthreads) { s_crc[i] = KT[${off.CRC}u + i]; }
  for (var i = li; i < 1536u; i += nthreads) { s_gt[i] = KT[${off.GT}u + i]; }
  for (var i = li; i < 64u; i += nthreads) { s_tab[i] = KT[${off.TAB}u + i]; s_val[i] = KT[${off.VAL}u + i]; }
  workgroupBarrier();
}
${runOuter("run_search", false, D, off, WG).replaceAll("GT[", "s_gt[")}
${runOuter("run_test", true, D, off, WG).replaceAll("GT[", "s_gt[")}

@compute @workgroup_size(${WG})
fn search_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  load_shared(li, ${WG}u);
  let gid = wid.x * ${WG}u + li;
  if (gid >= DP[0]) { return; }
  run_search(1u, gid, 0u, ${NPERM}u, 0u);
}

@compute @workgroup_size(${TW})
fn test_main(@builtin(local_invocation_index) li: u32) {
  load_shared(li, ${TW}u);
  if (li < ${nTests}u) {
    let b = li * ${NTEST_STRIDE}u;
    run_test(b, DP[b + 16u], DP[b + 17u], DP[b + 17u] + 1u, li);
  }
}
`;
}
