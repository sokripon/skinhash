// WebGPU test of the generated "contains" WGSL against the JS reference (matchTargets): random digests with the
// pattern planted at random offsets and random numbers of leading zeros, for the 8-word Mojang digest and the
// 2-word NameMC id. Needs Chrome with WebGPU and puppeteer-core (see browser_test.mjs for WEBTEST_DIR).
//   node web/test/contains_gpu.mjs
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const CHROME = process.env.CHROME || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const puppeteer = createRequire(path.join(process.env.WEBTEST_DIR || process.cwd(), "x.js"))("puppeteer-core");
const PORT = 8138;
const server = spawn("py", [path.join(repo, "web", "serve.py"), String(PORT)], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
let failures = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) failures++; };
const browser = await puppeteer.launch({
  executablePath: CHROME, headless: "new",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("  [pageerror]", e.message));
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);
  const res = await page.evaluate(async () => {
    const { containsCode, CB, PT_WORDS } = await import("/js/wgsl.js");
    const { gpuContains, matchTargets } = await import("/js/targets.js");
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    let seed = 12345;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
    const out = [];
    for (const nm of [false, true]) {
      const W = nm ? 2 : 8, width = W * 8, cb = nm ? 3 + 4 * 32 : CB;
      const ptWords = nm ? cb + 1 + 5 * 16 : PT_WORDS;
      const words = Array.from({ length: W }, (_, i) => `s${i}`);
      const code = `
@group(0) @binding(0) var<storage, read> IN: array<u32>;
@group(0) @binding(1) var<storage, read> PT: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(3) var<storage, read> ZL: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&ZL)) { return; }
  ${words.map((w, k) => `let ${w} = IN[i * ${W}u + ${k}u];`).join("\n  ")}
  let zn = ZL[i];
  var hit = false;
  ${containsCode({ words, cb, zl: nm ? "0u" : "zn" })}
  OUT[i] = select(0u, 1u, hit);
}`;
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === "error");
      if (errs.length) { out.push({ nm, error: errs.map((m) => `${m.lineNum}: ${m.message}`).join("; ") }); continue; }
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const N = 4096;
      let mism = 0, pos = 0, tot = 0, first = "";
      for (let round = 0; round < 150; round++) {
        const n = 1 + (rnd() % 16);
        let text = "";
        for (let k = 0; k < n; k++) text += "0123456789abcdef"[(round % 3 === 0 ? (rnd() & 1) * 15 : rnd()) & 15];
        const g = gpuContains({ kind: "contains", text }, width);
        const pt = new Uint32Array(ptWords);
        pt[cb] = 1; pt.set([g.p1, g.m1, g.p2, g.m2, g.lim], cb + 1);
        const inp = new Uint32Array(N * W), zl = new Uint32Array(N);
        const hexes = [];
        for (let i = 0; i < N; i++) {
          let h = "";
          for (let k = 0; k < width; k++) h += "0123456789abcdef"[round % 3 === 0 && rnd() % 3 ? (rnd() & 1) * 15 : rnd() & 15];
          const z = nm ? 0 : (rnd() % 5 === 0 ? rnd() % 16 : 0);
          if (!nm) h = "0".repeat(z) + h.slice(z);       // z leading zero digits, at least
          if (rnd() & 1) { const o = rnd() % (width - n + 1); h = h.slice(0, o) + text + h.slice(o + n); }
          // the kernel knows at most 16 leading zero digits (zn); a real digest with 16+ is a 2^-64 case that the
          // host re-check settles, so the test does not generate it
          if (!nm && /^0{16}/.test(h)) h = h.slice(0, 15) + "1" + h.slice(16);
          hexes.push(h);
          for (let w = 0; w < W; w++) inp[i * W + w] = parseInt(h.slice(8 * w, 8 * w + 8), 16) >>> 0;
          let lz = 0; while (lz < 16 && h[lz] === "0") lz++;
          zl[i] = nm ? 0 : Math.min(lz, 16);              // what hitCode passes: leading zero digits of the first two words
        }
        const mk = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
        const bIn = mk(inp, GPUBufferUsage.STORAGE), bPt = mk(pt, GPUBufferUsage.STORAGE), bZl = mk(zl, GPUBufferUsage.STORAGE);
        const bOut = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bRd = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [bIn, bPt, bOut, bZl].map((b, k) => ({ binding: k, resource: { buffer: b } })) });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(N / 64); pass.end();
        enc.copyBufferToBuffer(bOut, 0, bRd, 0, N * 4);
        device.queue.submit([enc.finish()]);
        await bRd.mapAsync(GPUMapMode.READ);
        const got = new Uint32Array(bRd.getMappedRange().slice(0));
        bRd.unmap(); [bIn, bPt, bZl, bOut, bRd].forEach((b) => b.destroy());
        for (let i = 0; i < N; i++) {
          // the kernel is told the leading zeros it knows (capped at 16 digits); the reference is the real rule
          const want = nm ? hexes[i].includes(text) : matchTargets(hexes[i], [{ kind: "contains", text }]).length > 0;
          tot++; pos += want;
          if (want !== (got[i] === 1)) { if (!first) first = `${text} in ${hexes[i]}: want ${want} got ${got[i]}`; mism++; }
        }
      }
      out.push({ nm, tot, pos, mism, first });
    }
    return out;
  });
  for (const r of res) {
    if (r.error) { check(false, `${r.nm ? "NameMC" : "Mojang"} contains WGSL does not compile: ${r.error}`); continue; }
    check(r.mism === 0 && r.pos > r.tot / 5, `${r.nm ? "NameMC 2-word id" : "Mojang 8-word digest"}: ${r.tot} digests, ${r.pos} positives, ${r.mism} mismatches ${r.first}`);
  }
} catch (e) {
  console.log("TEST ERROR:", e.message); failures++;
} finally {
  await browser.close();
  server.kill();
}
console.log(failures ? `\n${failures} FAILURE(S)` : "\nCONTAINS GPU TESTS PASSED");
process.exit(failures ? 1 : 0);
