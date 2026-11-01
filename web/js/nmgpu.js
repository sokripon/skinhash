// NameMC kernel (WGSL port of namemcgpu.cu) and its runner.
//  fast:    knob words 9,10,11 of the first of exactly 2 SHA blocks (the default layout, pixels (55-57,63)).
//           A thread owns 1024 consecutive counters that differ only in kp[2]; the second block is constant,
//           so its K+W values are baked in as literals.
//  general: any layout (e.g. "unused pixels only", ~50 blocks), one candidate per thread; per-block tables
//           live in a storage buffer.
import { K256 } from "./hash.js";
import { MAXHITS, MAXPAT, MAXCONT, HITS_WORDS, containsCode } from "./wgsl.js";

const NM_CB = 3 + 4 * MAXPAT;                 // contains table in PT: count, then p1 m1 p2 m2 lim per pattern
const NM_PT_WORDS = NM_CB + 1 + 5 * MAXCONT;

const INNER = 1024;
const hex = (v) => `0x${(v >>> 0).toString(16)}u`;
const ror = (x, n) => `((${x} >> ${n}u) | (${x} << ${32 - n}u))`;

function expandW(w16) {
  const w = new Uint32Array(64);
  w.set(w16);
  const r = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
  for (let i = 16; i < 64; i++) {
    const s0 = r(w[i-15], 7) ^ r(w[i-15], 18) ^ (w[i-15] >>> 3), s1 = r(w[i-2], 17) ^ r(w[i-2], 19) ^ (w[i-2] >>> 10);
    w[i] = (w[i-16] + s0 + w[i-7] + s1) >>> 0;
  }
  return w;
}

/** SHA-256 block code over state vars s0..s7. words(i) gives the expression of message word i (full block);
 *  with kw(i) given instead, the block is constant and kw(i) is K[i]+W[i]. */
function shaCode({ words, kw }) {
  let c = "{\n";
  if (words) for (let i = 0; i < 16; i++) c += `var w${i} = ${words(i)};\n`;
  c += "var a = s0; var b = s1; var c = s2; var d = s3; var e = s4; var f = s5; var g = s6; var h = s7;\n";
  const base = ["a", "b", "c", "d", "e", "f", "g", "h"];
  for (let i = 0; i < 64; i++) {
    let term;
    if (words) {
      const w = `w${i & 15}`;
      if (i >= 16) {
        const w15 = `w${(i + 1) & 15}`, w7 = `w${(i + 9) & 15}`, w2 = `w${(i + 14) & 15}`;
        c += `${w} = ${w} + (${ror(w15, 7)} ^ ${ror(w15, 18)} ^ (${w15} >> 3u)) + ${w7} + (${ror(w2, 17)} ^ ${ror(w2, 19)} ^ (${w2} >> 10u));\n`;
      }
      term = `${hex(K256[i])} + ${w}`;
    } else term = kw(i);
    const v = (j) => base[(((j - i) % 8) + 8) % 8];
    const [A, B, C, D, E, F, G, Hh] = [0, 1, 2, 3, 4, 5, 6, 7].map(v);
    c += `{ let t1 = ${Hh} + (${ror(E, 6)} ^ ${ror(E, 11)} ^ ${ror(E, 25)}) + (${G} ^ (${E} & (${F} ^ ${G}))) + ${term};
${D} = ${D} + t1;
${Hh} = t1 + (${ror(A, 2)} ^ ${ror(A, 13)} ^ ${ror(A, 22)}) + ((${A} & ${B}) | (${C} & (${A} | ${B}))); }\n`;
  }
  c += "s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;\n}\n";
  return c;
}

const rot = (x, n) => ((x >>> n) | (x << (32 - n))) >>> 0;
const sig0 = (x) => (rot(x, 7) ^ rot(x, 18) ^ (x >>> 3)) >>> 0, sig1 = (x) => (rot(x, 17) ^ rot(x, 19) ^ (x >>> 10)) >>> 0;
const R_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const roundNames = (i) => [0, 1, 2, 3, 4, 5, 6, 7].map((j) => R_NAMES[(((j - i) % 8) + 8) % 8]);
function roundCode(i, term) {
  const [A, B, C, D, E, F, G, Hh] = roundNames(i);
  return `{ let t1 = ${Hh} + (${ror(E, 6)} ^ ${ror(E, 11)} ^ ${ror(E, 25)}) + (${G} ^ (${E} & (${F} ^ ${G}))) + ${term};
${D} = ${D} + t1;
${Hh} = t1 + (${ror(A, 2)} ^ ${ror(A, 13)} ^ ${ror(A, 22)}) + ((${A} & ${B}) | (${C} & (${A} | ${B}))); }\n`;
}

/** Fast NameMC kernel with the work split by what changes. Only message word 11 (kp[2]) changes per candidate;
 *  words 9 and 10 change per thread; everything else is a job constant. So:
 *   - the host (here) runs rounds 0-8 and every schedule word that depends only on constants;
 *   - `pre` runs once per thread: rounds 9-10, the schedule words that do not depend on word 11, and for each
 *     schedule word that does, the sum of its independent terms (pw);
 *   - `body` runs per candidate: the dependent schedule words, rounds 11-63, and the constant second block.
 *  `pre` needs v = nm_vw(lo, hi); `body` needs p11 (word 11) and leaves the id in s0, s1. Same code in the
 *  search loop and in the correctness-gate entry point. */
export function fastParts(prep, D) {
  const mid = Array.from(prep.mid), lvl = new Array(64).fill(0), val = new Array(64).fill(0);
  for (let i = 0; i < 16; i++) { lvl[i] = i === 9 || i === 10 ? 1 : i === 11 ? 2 : 0; val[i] = D.img[i]; }
  for (let i = 16; i < 64; i++) {
    lvl[i] = Math.max(lvl[i - 16], lvl[i - 15], lvl[i - 7], lvl[i - 2]);
    if (lvl[i] === 0) val[i] = (val[i - 16] + sig0(val[i - 15]) + val[i - 7] + sig1(val[i - 2])) >>> 0;
  }
  const nm = (i) => (lvl[i] === 0 ? hex(val[i]) : lvl[i] === 1 ? `u${i}` : `d${i}`);
  const sg0 = (x) => `(${ror(x, 7)} ^ ${ror(x, 18)} ^ (${x} >> 3u))`, sg1 = (x) => `(${ror(x, 17)} ^ ${ror(x, 19)} ^ (${x} >> 10u))`;
  // state after rounds 0-8 (all constant)
  let st = mid.slice();
  for (let i = 0; i < 9; i++) {
    const [a, b, c, d, e, f, g, h] = st;
    const t1 = (h + (rot(e, 6) ^ rot(e, 11) ^ rot(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + val[i]) >>> 0;
    const t2 = ((rot(a, 2) ^ rot(a, 13) ^ rot(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
    st = [(t1 + t2) >>> 0, a, b, c, (d + t1) >>> 0, e, f, g];
  }
  let pre = "let p9 = v.x; let p10 = v.y; let u9 = p9; let u10 = p10;\n" + Array.from({ length: 8 }, (_, j) => `var q${j} = 0u;`).join(" ") + "\n{\n";
  const n9 = roundNames(9);
  st.forEach((x, j) => { pre += `var ${n9[j]} = ${hex(x)};\n`; });
  pre += roundCode(9, `${hex(K256[9])} + p9`) + roundCode(10, `${hex(K256[10])} + p10`);
  const n11 = roundNames(11);
  n11.forEach((n, j) => { pre += `q${j} = ${n};\n`; });
  pre += "}\n";
  let body = "let d11 = p11;\n";
  const terms = (i) => [[i - 16, null], [i - 15, sg0], [i - 7, null], [i - 2, sg1]];
  for (let i = 16; i < 64; i++) {
    if (lvl[i] === 1) {
      let cs = 0;
      const ex = [];
      for (const [j, f] of terms(i)) {   // constant terms are summed here: WGSL rejects overflowing constant expressions
        if (lvl[j] === 0) cs = (cs + (f === sg0 ? sig0(val[j]) : f === sg1 ? sig1(val[j]) : val[j])) >>> 0;
        else ex.push(f ? f(nm(j)) : nm(j));
      }
      pre += `let u${i} = ${[...(cs ? [hex(cs)] : []), ...ex].join(" + ")};\n`;
    } else if (lvl[i] === 2) {
      let cs = 0, ex = [], dx = [];
      for (const [j, f] of terms(i)) {
        if (lvl[j] === 0) cs = (cs + (f === sg0 ? sig0(val[j]) : f === sg1 ? sig1(val[j]) : val[j])) >>> 0;
        else if (lvl[j] === 1) ex.push(f ? f(nm(j)) : nm(j));
        else dx.push(f ? f(nm(j)) : nm(j));
      }
      const parts = [...(cs ? [hex(cs)] : []), ...ex];
      if (parts.length > 1 || (parts.length === 1 && ex.length)) { pre += `let pw${i} = ${parts.join(" + ")};\n`; dx.unshift(`pw${i}`); }
      else if (parts.length === 1) dx.unshift(parts[0]);
      body += `let d${i} = ${dx.join(" + ")};\n`;
    }
  }
  n11.forEach((n, j) => { body += `var ${n} = q${j};\n`; });
  for (let i = 11; i < 64; i++) {
    const term = lvl[i] === 0 ? hex((K256[i] + val[i]) >>> 0) : `${hex(K256[i])} + ${nm(i)}`;
    body += roundCode(i, term);
  }
  mid.forEach((x, j) => { body += `var s${j} = ${hex(x)} + ${R_NAMES[j]};\n`; });
  body += shaCode({ kw: (i) => hex(D.kw[64 + i]) });
  return { pre, body };
}

const be = (t, o) => ((t[o] << 24) | (t[o + 1] << 16) | (t[o + 2] << 8) | t[o + 3]) >>> 0;

export function nmDerive(prep) {
  const nb = prep.nb, tail = prep.tail;
  const img = new Uint32Array(nb * 16), kw = new Uint32Array(nb * 64), isc = new Uint32Array(nb);
  const pblk = prep.woff.map((o) => Math.floor(o / 64)), pwd = prep.woff.map((o) => (o % 64) >> 2);
  for (let b = 0; b < nb; b++) {
    for (let i = 0; i < 16; i++) img[b * 16 + i] = be(tail, b * 64 + 4 * i);
    const w = expandW(img.subarray(b * 16, b * 16 + 16));
    for (let i = 0; i < 64; i++) kw[b * 64 + i] = (K256[i] + w[i]) >>> 0;
    isc[b] = pblk.includes(b) ? 0 : 1;
  }
  return { img, kw, isc, pblk, pwd };
}

export function genNmShader(prep, wg = 256, nTests = 96, contains = false) {
  const D = nmDerive(prep), fast = prep.fast, nb = prep.nb;
  const mid = Array.from(prep.mid).map((x, i) => `var s${i} = ${hex(x)};`).join(" ");
  const off = { IMG: 0, KW: nb * 16, ISC: nb * 80, PBLK: nb * 81, PWD: nb * 81 + 3 };
  let s = `
${fast ? "" : "@group(0) @binding(0) var<storage, read> NKT: array<u32>;"}
@group(0) @binding(1) var<storage, read> PT: array<u32>;
@group(0) @binding(2) var<storage, read> DP: array<u32>;
@group(0) @binding(3) var<storage, read_write> HITS: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> TOUT: array<u32>;

fn is_hit(s0: u32, s1: u32) -> bool {
  let zm = PT[1];
  if (zm == 1u && s0 < PT[2]) { return true; }
  if (zm == 2u && s0 == 0u && s1 < PT[2]) { return true; }
  let np = PT[0];
  for (var i = 0u; i < np; i++) {
    let pb = 3u + 4u * i;
    if ((s0 & PT[pb + 2u]) == PT[pb] && (s1 & PT[pb + 3u]) == PT[pb + 1u]) { return true; }
  }
${contains ? `  var hit = false;
${containsCode({ words: ["s0", "s1"], cb: NM_CB, zl: "0u" })}  return hit;` : "  return false;"}
}
fn record(lo: u32, hi: u32) {
  let pos = atomicAdd(&HITS[0], 1u);
  if (pos < ${MAXHITS}u) { atomicStore(&HITS[2u + 2u * pos], lo); atomicStore(&HITS[3u + 2u * pos], hi); }
}
fn note_best(m0: u32, m1: u32) {
  let zn = select(select(16u, 8u + (countLeadingZeros(m1) >> 2u), m1 != 0u), countLeadingZeros(m0) >> 2u, m0 != 0u);
  if (zn > 0u && zn > atomicLoad(&HITS[1])) { atomicMax(&HITS[1], zn); }
}
fn nm_vw(lo: u32, hi: u32) -> vec3<u32> {
  return vec3<u32>(((hi >> 16u) << 8u) | 1u,
                   ((((lo >> 24u) | (hi << 8u)) & 0xFFFFFFu) << 8u) | 1u,
                   ((lo & 0xFFFFFFu) << 8u) | 1u);
}
`;
  const FP = fast ? fastParts(prep, D) : null;
  if (!fast) {
    s += `fn nm_id(p0: u32, p1: u32, p2: u32) -> vec2<u32> {
  ${mid}
  var vw = array<u32, 3>(p0, p1, p2);
  for (var blk = 0u; blk < ${nb}u; blk++) {
    if (NKT[${off.ISC}u + blk] != 0u) {
${shaCode({ kw: (i) => `NKT[${off.KW}u + blk * 64u + ${i}u]` })}
    } else {
      var wa: array<u32, 16>;
      for (var i = 0u; i < 16u; i++) { wa[i] = NKT[${off.IMG}u + blk * 16u + i]; }
      for (var j = 0u; j < 3u; j++) { if (NKT[${off.PBLK}u + j] == blk) { wa[NKT[${off.PWD}u + j]] = vw[j]; } }
${shaCode({ words: (i) => `wa[${i}]` })}
    }
  }
  return vec2<u32>(s0, s1);
}
`;
  }
  s += `
@compute @workgroup_size(${wg})
fn nm_search(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let gid = wid.x * ${wg}u + li;
  if (gid >= DP[0]) { return; }
`;
  if (fast) {
    s += `  let lo = DP[1] + gid * ${INNER}u;
  let hi = DP[2] + select(0u, 1u, lo < DP[1]);
  let v = nm_vw(lo, hi);
  let l24 = lo & 0xFFFFFFu;
${FP.pre}
  var m0 = 0xFFFFFFFFu; var m1 = 0xFFFFFFFFu;
  for (var t = 0u; t < ${INNER}u; t++) {
    let p11 = ((l24 + t) << 8u) | 1u;
${FP.body}
    if (s0 < m0 || (s0 == m0 && s1 < m1)) { m0 = s0; m1 = s1; }
    if (is_hit(s0, s1)) { record(lo + t, hi); }
  }
  note_best(m0, m1);
}
`;
  } else {
    s += `  let lo = DP[1] + gid;
  let hi = DP[2] + select(0u, 1u, lo < DP[1]);
  let v = nm_vw(lo, hi);
  let id = nm_id(v.x, v.y, v.z);
  if (is_hit(id.x, id.y)) { record(lo, hi); }
  note_best(id.x, id.y);
}
`;
  }
  s += `
@compute @workgroup_size(128)
fn nm_test(@builtin(local_invocation_index) li: u32) {
  if (li < ${nTests}u) {
    let v = nm_vw(DP[2u * li], DP[2u * li + 1u]);
${fast ? `${FP.pre}
    let p11 = v.z;
${FP.body}
    TOUT[2u * li] = s0; TOUT[2u * li + 1u] = s1;` : `    let id = nm_id(v.x, v.y, v.z);
    TOUT[2u * li] = id.x; TOUT[2u * li + 1u] = id.y;`}
  }
}
`;
  return { wgsl: s, D, off };
}

const U = GPUBufferUsage;

export class NmGpuJob {
  constructor(device, prep, tests, { wg = 256, contains = false } = {}) {
    this.device = device; this.prep = prep; this.tests = tests; this.wg = wg;
    const g = genNmShader(prep, wg, tests.length, contains);
    this.wgsl = g.wgsl; this.D = g.D; this.off = g.off;
    this.per = prep.fast ? INNER : 1;
    this.maxThreads = prep.fast ? 1 << 19 : 1 << 23;
    this.cfg = { pats: [], cont: [], zmode: 0, zthr: 0 }; this.hasContains = contains;
  }

  async build() {
    const d = this.device, prep = this.prep;
    d.pushErrorScope("validation");
    const module = d.createShaderModule({ code: this.wgsl });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) {
      const src = this.wgsl.split("\n");
      throw new Error("WGSL compile error: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message} | ${src[m.lineNum - 1] || ""}`).join("\n"));
    }
    this.search = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "nm_search" } });
    this.test = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "nm_test" } });
    const err = await d.popErrorScope();
    if (err) throw new Error("WebGPU validation error: " + err.message);
    const mk = (size, usage) => d.createBuffer({ size: Math.max(16, size), usage });
    if (!prep.fast) {
      const n = prep.nb, a = new Uint32Array(n * 81 + 6);
      a.set(this.D.img, this.off.IMG); a.set(this.D.kw, this.off.KW); a.set(this.D.isc, this.off.ISC);
      a.set(this.D.pblk, this.off.PBLK); a.set(this.D.pwd, this.off.PWD);
      this.kbuf = mk(a.byteLength, U.STORAGE | U.COPY_DST);
      d.queue.writeBuffer(this.kbuf, 0, a);
    }
    this.ptbuf = mk(NM_PT_WORDS * 4, U.STORAGE | U.COPY_DST);
    this.setTargets(this.cfg);
    const entries = (extra) => [...(prep.fast ? [] : [{ binding: 0, resource: { buffer: this.kbuf } }]), ...extra];
    this.entries = entries;
    this.slots = [0, 1].map(() => {
      const dp = mk(16, U.STORAGE | U.COPY_DST);
      const hits = mk(HITS_WORDS * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
      const staging = mk(HITS_WORDS * 4, U.MAP_READ | U.COPY_DST);
      const bind = d.createBindGroup({ layout: this.search.getBindGroupLayout(0), entries: entries([
        { binding: 1, resource: { buffer: this.ptbuf } }, { binding: 2, resource: { buffer: dp } }, { binding: 3, resource: { buffer: hits } }]) });
      return { dp, hits, staging, bind };
    });
  }

  /** cfg = {pats, cont, zmode, zthr} from gpuNmTargets */
  setTargets(cfg) {
    if (cfg.cont?.length && !this.hasContains) throw new Error("this kernel was built without contains support");
    this.cfg = cfg;
    const a = new Uint32Array(NM_PT_WORDS);
    a[0] = Math.min(cfg.pats.length, MAXPAT); a[1] = cfg.zmode; a[2] = cfg.zthr >>> 0;
    cfg.pats.slice(0, MAXPAT).forEach((p, i) => a.set([p.vh, p.vl, p.mh, p.ml], 3 + 4 * i));
    const cont = cfg.cont || [];
    a[NM_CB] = Math.min(cont.length, MAXCONT);
    cont.slice(0, MAXCONT).forEach((c, i) => a.set([c.p1, c.m1, c.p2, c.m2, c.lim], NM_CB + 1 + 5 * i));
    this.device.queue.writeBuffer(this.ptbuf, 0, a);
  }

  async gate() {
    const d = this.device, nt = this.tests.length;
    const mk = (size, usage) => d.createBuffer({ size: Math.max(16, size), usage });
    const tdp = new Uint32Array(nt * 2);
    this.tests.forEach((t, i) => { tdp[2 * i] = Number(t.c & 0xFFFFFFFFn); tdp[2 * i + 1] = Number(t.c >> 32n); });
    const bdp = mk(tdp.byteLength, U.STORAGE | U.COPY_DST), bout = mk(nt * 8, U.STORAGE | U.COPY_SRC), bst = mk(nt * 8, U.MAP_READ | U.COPY_DST);
    d.queue.writeBuffer(bdp, 0, tdp);
    const bind = d.createBindGroup({ layout: this.test.getBindGroupLayout(0), entries: this.entries([
      { binding: 2, resource: { buffer: bdp } }, { binding: 4, resource: { buffer: bout } }]) });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.test); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
    enc.copyBufferToBuffer(bout, 0, bst, 0, nt * 8);
    d.queue.submit([enc.finish()]);
    await bst.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(bst.getMappedRange().slice(0));
    bst.unmap();
    [bdp, bout, bst].forEach((b) => b.destroy());
    let bad = 0, detail = "";
    this.tests.forEach((t, i) => {
      if (out[2 * i] !== t.hi || out[2 * i + 1] !== t.lo) { if (!bad) detail = `first mismatch at counter ${t.c}`; bad++; }
    });
    return { ok: bad === 0, n: nt, bad, detail };
  }

  /** Runs until cb.shouldStop(). cb.onHits(counters[], nTotal), cb.onProgress({tried, bestZeros}). */
  async run(rng, cb) {
    const d = this.device;
    let base = rng.big(1n << 64n) & ~BigInt(INNER - 1);
    let batch = this.prep.fast ? 1 << 12 : 1 << 16, tried = 0, best = 0;
    const free = [...this.slots], pending = [];
    let lastDone = performance.now();
    const submit = () => {
      const slot = free.pop(), n = batch;
      d.queue.writeBuffer(slot.dp, 0, new Uint32Array([n, Number(base & 0xFFFFFFFFn), Number(base >> 32n)]));
      const enc = d.createCommandEncoder();
      enc.clearBuffer(slot.hits, 0, 8);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.search); pass.setBindGroup(0, slot.bind);
      pass.dispatchWorkgroups(Math.ceil(n / this.wg));
      pass.end();
      enc.copyBufferToBuffer(slot.hits, 0, slot.staging, 0, HITS_WORDS * 4);
      d.queue.submit([enc.finish()]);
      pending.push({ slot, n, t0: performance.now(), p: slot.staging.mapAsync(GPUMapMode.READ) });
      base = (base + BigInt(n) * BigInt(this.per)) & ((1n << 64n) - 1n);
    };
    while (true) {
      while (pending.length < 2 && free.length && !cb.shouldStop()) submit();
      if (!pending.length) break;
      const rec = pending.shift();
      await rec.p;
      const u = new Uint32Array(rec.slot.staging.getMappedRange().slice(0));
      rec.slot.staging.unmap();
      free.push(rec.slot);
      const now = performance.now(), dt = now - Math.max(lastDone, rec.t0);
      lastDone = now;
      if (dt < 30 && batch * 2 <= Math.min(this.maxThreads, 65535 * this.wg)) batch *= 2;
      else if (dt > 120 && batch > 1024) batch = Math.floor(batch / 2);
      tried += rec.n * this.per;
      best = Math.max(best, u[1]);
      const nh = u[0];
      if (nh) {
        const cs = [];
        for (let i = 0; i < Math.min(nh, MAXHITS); i++) cs.push((BigInt(u[3 + 2 * i]) << 32n) | BigInt(u[2 + 2 * i]));
        cb.onHits(cs, nh);
      }
      cb.onProgress({ tried, bestZeros: best, batch });
    }
    return "stopped";
  }

  destroy() {
    for (const b of [this.kbuf, this.ptbuf, ...(this.slots || []).flatMap((s) => [s.dp, s.hits, s.staging])]) b?.destroy();
  }
}
