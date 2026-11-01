// WebGPU runner: device setup, correctness gate, pipelined search loop.
import { deriveJob, buildConsts, genShader, MAXHITS, MAXPAT, MAXCONT, CB, PT_WORDS, HITS_WORDS, NTEST_STRIDE } from "./wgsl.js";
import { deriveFast, buildFastConsts, genFastShader } from "./wgslfast.js";
import { RIN, NPERM, permCount, outerDigits } from "./job.js";

export function webgpuAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

export async function initDevice() {
  if (!webgpuAvailable()) throw new Error("WebGPU is not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found (no compatible GPU or driver).");
  const device = await adapter.requestDevice();
  let info = {};
  try { info = adapter.info || (await adapter.requestAdapterInfo?.()) || {}; } catch { /* ignore */ }
  return { adapter, device, info };
}

const U = GPUBufferUsage;
const MAX_BATCH = 1 << 24;   // threads per dispatch (x 24 candidates each)
// Dispatches should take about 50-100 ms: grow the batch below 45 ms, shrink it above 100 ms. A halved batch of a
// 100+ ms dispatch is still above 45 ms, so the size does not oscillate.
const GROW_BELOW_MS = 45, SHRINK_ABOVE_MS = 100;

// Compiled pipelines per device, keyed by the generated source (the layout and the constants baked into it).
const pipelineCache = new WeakMap();
async function compilePipelines(d, code) {
  let cache = pipelineCache.get(d);
  if (!cache) pipelineCache.set(d, cache = new Map());
  if (cache.has(code)) return cache.get(code);
  d.pushErrorScope("validation");
  const module = d.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) {
    await d.popErrorScope();
    const src = code.split("\n");
    throw new Error("WGSL compile error: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message} | ${src[m.lineNum - 1] || ""}`).join("\n"));
  }
  const search = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "search_main" } });
  const test = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "test_main" } });
  const err = await d.popErrorScope();
  if (err) throw new Error("WebGPU validation error: " + err.message);
  const r = { search, test };
  cache.set(code, r);
  if (cache.size > 4) cache.delete(cache.keys().next().value);
  return r;
}

export class GpuJob {
  constructor(device, job, { wg = 256, kernel = "auto", contains = false } = {}) {
    this.device = device; this.job = job; this.wg = wg;
    this.D = deriveJob(job);
    this.kernel = "generic"; this.whyGeneric = kernel === "generic" ? "forced" : "";
    if (kernel !== "generic") {
      const F = deriveFast(job, this.D);
      if (F.ok) {
        this.F = F; this.kernel = "fast";
        const { buf, off } = buildFastConsts(this.D, F);
        this.consts = buf;
        this.wgsl = genFastShader(job, this.D, F, off, wg, contains);
      } else this.whyGeneric = F.why;
    }
    if (this.kernel === "generic") {
      const { off, buf } = buildConsts(this.D, job.nblk);
      this.consts = buf;
      this.wgsl = genShader(this.D, off, wg, job.tests.length, contains);
    }
    this.nTests = job.tests.length;
    this.pats = []; this.zeros = 0; this.cont = []; this.hasContains = contains;
  }

  async build() {
    const d = this.device;
    const { search, test } = await compilePipelines(d, this.wgsl);
    this.search = search; this.test = test;
    const mk = (size, usage) => d.createBuffer({ size: Math.max(16, size), usage });
    this.dummy = mk(64, U.STORAGE);   // the fast kernel's shared code path binds all five buffers in both entry points
    this.kbuf = mk(this.consts.byteLength, U.STORAGE | U.COPY_DST);
    d.queue.writeBuffer(this.kbuf, 0, this.consts);
    this.ptbuf = mk(PT_WORDS * 4, U.STORAGE | U.COPY_DST);
    this.setTargets(this.pats, this.zeros, this.cont);
    this.slots = [0, 1].map(() => {
      const dp = mk(20 * 4, U.STORAGE | U.COPY_DST);
      const hits = mk(HITS_WORDS * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
      const staging = mk(HITS_WORDS * 4, U.MAP_READ | U.COPY_DST);
      const bind = d.createBindGroup({ layout: this.search.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.kbuf } }, { binding: 1, resource: { buffer: this.ptbuf } },
        { binding: 2, resource: { buffer: dp } }, { binding: 3, resource: { buffer: hits } },
        ...(this.kernel === "fast" ? [{ binding: 4, resource: { buffer: this.dummy } }] : [])] });
      return { dp, hits, staging, bind };
    });
  }

  /** pats: [{vh, vl, mh, ml, suffix}], cont: [{p1, m1, p2, m2, lim}] (only used if the kernel was built with contains) */
  setTargets(pats, zeros, cont = []) {
    if (cont.length && !this.hasContains) throw new Error("this kernel was built without contains support");
    this.pats = pats; this.zeros = zeros; this.cont = cont;
    const a = new Uint32Array(PT_WORDS);
    a[0] = Math.min(pats.length, MAXPAT); a[1] = zeros;
    pats.slice(0, MAXPAT).forEach((p, i) => a.set([p.vh, p.vl, p.mh, p.ml, p.suffix ? 1 : 0], 2 + 5 * i));
    a[CB] = Math.min(cont.length, MAXCONT);
    cont.slice(0, MAXCONT).forEach((c, i) => a.set([c.p1, c.m1, c.p2, c.m2, c.lim], CB + 1 + 5 * i));
    this.device.queue.writeBuffer(this.ptbuf, 0, a);
  }

  /** The GPU must reproduce the CPU test vectors exactly. Returns {ok, n, bad, detail}. */
  async gate() {
    const d = this.device, J = this.job, m = J.m, nt = this.nTests;
    const tdp = new Uint32Array(nt * NTEST_STRIDE);
    for (let i = 0; i < nt; i++) {
      const idx = J.tests[i].idx;
      const o = idx / 24n;
      const g = o < BigInt(100000 + 7 * i) ? o : BigInt(100000 + 7 * i);   // exercises the carry-add of the thread id
      tdp.set(outerDigits(J, o - g).subarray(0, 16), i * NTEST_STRIDE);
      tdp[i * NTEST_STRIDE + 16] = Number(g);
      tdp[i * NTEST_STRIDE + 17] = Number(idx % 24n);
    }
    const mk = (size, usage) => d.createBuffer({ size: Math.max(16, size), usage });
    const bdp = mk(tdp.byteLength, U.STORAGE | U.COPY_DST);
    const bout = mk(nt * 32, U.STORAGE | U.COPY_SRC);
    const bst = mk(nt * 32, U.MAP_READ | U.COPY_DST);
    d.queue.writeBuffer(bdp, 0, tdp);
    const bind = d.createBindGroup({ layout: this.test.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.kbuf } }, { binding: 2, resource: { buffer: bdp } }, { binding: 4, resource: { buffer: bout } },
      ...(this.kernel === "fast" ? [{ binding: 1, resource: { buffer: this.ptbuf } }, { binding: 3, resource: { buffer: this.dummy } }] : [])] });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.test); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(nt / 64));
    pass.end();
    enc.copyBufferToBuffer(bout, 0, bst, 0, nt * 32);
    d.queue.submit([enc.finish()]);
    await bst.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(bst.getMappedRange().slice(0));
    bst.unmap();
    [bdp, bout, bst].forEach((b) => b.destroy());
    let bad = 0, detail = "";
    for (let i = 0; i < nt; i++) {
      const want = J.tests[i].sha;
      let same = true;
      for (let k = 0; k < 8; k++) {
        const w = (want[4*k] << 24 | want[4*k+1] << 16 | want[4*k+2] << 8 | want[4*k+3]) >>> 0;
        if (out[i * 8 + k] !== w) same = false;
      }
      if (!same) { if (!bad) detail = `first mismatch at test vector ${i} (index ${J.tests[i].idx})`; bad++; }
    }
    return { ok: bad === 0, n: nt, bad, detail };
  }

  /**
   * Search until the value set is exhausted or shouldStop() is true.
   * cb.onProgress({tried, bestZeros}), cb.onHits(idxBigIntArray, nTotalInBatch).
   * Resolves 'stopped' or 'exhausted'.
   */
  async run(rng, cb) {
    const d = this.device, J = this.job;
    const NOUT = permCount(J) / 24n;
    let obase = rng.big(NOUT), covered = 0n;
    let batch = 1 << 16, tried = 0, best = 0;
    const free = [...this.slots];
    const pending = [];
    let lastDone = performance.now();
    const submit = () => {
      const slot = free.pop();
      let cnt = BigInt(batch);
      if (cnt > NOUT - covered) cnt = NOUT - covered;
      if (cnt > NOUT - obase) cnt = NOUT - obase;
      const n = Number(cnt);
      const dp = new Uint32Array(20);
      dp[0] = n; dp.set(outerDigits(J, obase).subarray(0, 16), 1);
      d.queue.writeBuffer(slot.dp, 0, dp);
      const enc = d.createCommandEncoder();
      enc.clearBuffer(slot.hits, 0, 8);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.search); pass.setBindGroup(0, slot.bind);
      pass.dispatchWorkgroups(Math.ceil(n / this.wg));
      pass.end();
      enc.copyBufferToBuffer(slot.hits, 0, slot.staging, 0, HITS_WORDS * 4);
      d.queue.submit([enc.finish()]);
      const rec = { slot, n, obase, t0: performance.now(), p: slot.staging.mapAsync(GPUMapMode.READ) };
      pending.push(rec);
      covered += cnt; obase += cnt; if (obase >= NOUT) obase = 0n;
    };
    while (true) {
      while (pending.length < 2 && free.length && covered < NOUT && !cb.shouldStop()) submit();
      if (!pending.length) break;
      const rec = pending.shift();
      await rec.p;
      const u = new Uint32Array(rec.slot.staging.getMappedRange().slice(0));
      rec.slot.staging.unmap();
      free.push(rec.slot);
      const now = performance.now();
      const dt = now - Math.max(lastDone, rec.t0);
      lastDone = now;
      if (dt < GROW_BELOW_MS) batch = Math.min(batch * 2, MAX_BATCH, 65535 * this.wg);
      else if (dt > SHRINK_ABOVE_MS && batch > 4096) batch = Math.floor(batch / 2);
      tried += rec.n * NPERM;
      best = Math.max(best, u[1]);
      const nh = u[0];
      if (nh) {
        const idxs = [];
        for (let i = 0; i < Math.min(nh, MAXHITS); i++) idxs.push((rec.obase + BigInt(u[2 + 2 * i])) * 24n + BigInt(u[3 + 2 * i]));
        cb.onHits(idxs, nh);
      }
      cb.onProgress({ tried, bestZeros: best, batch });
    }
    return covered >= NOUT && !cb.shouldStop() ? "exhausted" : "stopped";
  }

  destroy() {
    for (const b of [this.kbuf, this.ptbuf, this.dummy, ...(this.slots || []).flatMap((s) => [s.dp, s.hits, s.staging])]) b?.destroy();
  }
}

/**
 * Build the GPU job for `job` and run the correctness gate. The fast kernel is tried first when the job qualifies;
 * if it does not reproduce every CPU test vector, the generic kernel is built and gated instead. Returns
 * {g, gate} with gate.kernel = "fast" | "generic"; gate.ok is false only if every kernel tried failed
 * (g is then the last one, the caller must not search with it).
 */
export async function prepareGpuJob(device, job, opts = {}) {
  const tryKernel = async (kernel) => {
    const g = new GpuJob(device, job, { ...opts, kernel });
    try {
      await g.build();
      const gate = await g.gate();
      return { g, gate: { ...gate, kernel: g.kernel } };
    } catch (e) {
      g.destroy();
      if (kernel === "generic") throw e;
      console.warn("fast kernel failed to build, using the generic kernel:", e);
      return null;
    }
  };
  let r = await tryKernel("auto");
  if (r && r.g.kernel === "fast" && !r.gate.ok) {
    console.warn(`fast kernel failed the gate (${r.gate.bad}/${r.gate.n}: ${r.gate.detail}), falling back to the generic kernel`);
    r.g.destroy();
    r = null;
  }
  if (!r) r = await tryKernel("generic");
  return r;
}
