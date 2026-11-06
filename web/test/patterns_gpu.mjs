// WebGPU test of the generated stat lines (wgslstat.js) and pattern prefilters: for random patterns, the WGSL from anchoredCode (masks baked in as
// literals, Mojang) and containsCode (sliding windows from the PT table, Mojang and NameMC) must give exactly the JS
// model (Pattern.prefilterPass) on random ids with planted matches, and must pass every true match (Pattern.match).
// Needs Chrome with WebGPU and puppeteer-core (see browser_test.mjs for WEBTEST_DIR).
//   node web/test/patterns_gpu.mjs
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const CHROME = process.env.CHROME || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const puppeteer = createRequire(path.join(process.env.WEBTEST_DIR || process.cwd(), "x.js"))("puppeteer-core");
const PORT = 8139;
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
    const { containsCode, anchoredCode, CB, AE, PT_WORDS } = await import("/js/wgsl.js");
    const { parseLine } = await import("/js/patterns.js");
    const { gpuSliding } = await import("/js/targets.js");
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    let seed = 4242;
    const rnd = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed >>> 8) % n; };
    const H = "0123456789abcdef";
    const tok = () => { const k = rnd(10); if (k < 5) return H[rnd(16)]; if (k === 5) return "?"; if (k === 6) return "?d"; if (k === 7) return "?l"; const a = rnd(15), b = a + 1 + rnd(15 - a); return `[${H[a]}-${H[b]}]`; };
    const out = [];
    for (const nm of [false, true]) {
      const W = nm ? 2 : 8, width = W * 8, cb = nm ? 3 + 4 * 32 : CB, ptWords = nm ? cb + 1 + 6 * 32 : PT_WORDS;
      const words = Array.from({ length: W }, (_, i) => `s${i}`);
      let tot = 0, pos = 0, mism = 0, miss = 0, kinds = { anchored: 0, sliding: 0 }, first = "";
      for (let round = 0; round < 60; round++) {
        let pat;
        while (!pat) {
          let text = "";
          for (let i = 0, np = 1 + rnd(2); i < np; i++) { if (i) text += "*"; for (let k = 0, n = 2 + rnd(5); k < n; k++) text += tok() + (rnd(5) === 0 ? `{${2 + rnd(3)}}` : ""); }
          if (rnd(2)) text = "^" + text;
          if (rnd(3) === 0) text += "$";
          try { pat = parseLine(text, width); } catch { pat = null; }
          if (pat && nm && pat.prefilter().kind === "anchored") pat = null;   // NameMC anchored masks use the prefix table (node test)
        }
        const f = pat.prefilter();
        kinds[f.kind]++;
        const anch = f.kind === "anchored" ? [pat.words()] : [];
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
${anchoredCode(anch, words)}
  ${f.kind === "sliding" ? containsCode({ words, cb, zl: nm ? "0u" : "zn" }) : ""}
  OUT[i] = select(0u, 1u, hit);
}`;
        const module = device.createShaderModule({ code });
        const info = await module.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === "error");
        if (errs.length) { out.push({ nm, error: errs.map((m) => `${m.lineNum}: ${m.message}`).join("; ") }); break; }
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const pt = new Uint32Array(ptWords);
        if (f.kind === "sliding") { const g = gpuSliding(f); pt[cb] = 1; pt.set([g.p1, g.m1, g.p2, g.m2, g.lim, g.lo], cb + 1); }
        else pt[AE] = 1;
        const N = 2048;
        const inp = new Uint32Array(N * W), zl = new Uint32Array(N), hexes = [];
        for (let i = 0; i < N; i++) {
          const h = [...Array(width)].map(() => rnd(16));
          if (!nm && rnd(4) === 0) for (let k = 0, z = 1 + rnd(14); k < z; k++) h[k] = 0;
          if (rnd(2)) {
            const total = pat.pieces.reduce((a, x) => a + x.length, 0);
            let at = pat.start ? 0 : rnd(width - total + 1);
            if (pat.end && !pat.start) at = width - total;
            pat.pieces.forEach((piece) => piece.forEach((s) => { const vals = [...Array(16).keys()].filter((v) => (s >> v) & 1); h[at++] = vals[rnd(vals.length)]; }));
          }
          const hex = h.map((v) => H[v]).join("");
          hexes.push(hex);
          for (let w = 0; w < W; w++) inp[i * W + w] = parseInt(hex.slice(8 * w, 8 * w + 8), 16) >>> 0;
          let lz = 0; while (lz < 16 && hex[lz] === "0") lz++;
          zl[i] = nm ? 0 : lz;
        }
        const mk = (data) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
        const bIn = mk(inp), bPt = mk(pt), bZl = mk(zl);
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
          const want = pat.prefilterPass(hexes[i]), truth = pat.match(hexes[i]);
          tot++; pos += truth;
          if (want !== (got[i] === 1)) { mism++; first = first || `${pat.text} on ${hexes[i]}: model ${want} gpu ${got[i]}`; }
          if (truth && got[i] !== 1) { miss++; first = first || `GPU rejects a match: ${pat.text} on ${hexes[i]}`; }
        }
      }
      out.push({ nm, tot, pos, mism, miss, kinds, first });
    }
    return out;
  });
  for (const r of res) {
    if (r.error) { check(false, `${r.nm ? "NameMC" : "Mojang"} pattern WGSL does not compile: ${r.error}`); continue; }
    check(r.mism === 0 && r.miss === 0 && r.pos > r.tot / 6,
      `${r.nm ? "NameMC 2-word id" : "Mojang 8-word digest"}: ${r.tot} ids (${r.kinds.anchored} anchored / ${r.kinds.sliding} sliding patterns), ${r.pos} true matches, GPU = model, no match rejected ${r.first}`);
  }

  // ---- stat lines: the WGSL of wgslstat.js against the JS model (StatLine.gpuPass) ----
  const sres = await page.evaluate(async () => {
    const { statLib, statCode, zlExpr } = await import("/js/wgslstat.js");
    const { parseLine } = await import("/js/patterns.js");
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    let seed = 777;
    const rnd = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed >>> 8) % n; };
    const H = "0123456789abcdef";
    // random ids with low entropy, planted runs, staircases, zigzags, palindromes, words and leading zeros
    const makeId = (width) => {
      const h = [...Array(width)].map(() => (rnd(4) === 0 ? rnd(3) : rnd(16)));
      const plants = 1 + rnd(3);
      for (let k = 0; k < plants; k++) {
        const len = 2 + rnd(Math.min(14, width - 2)), a = rnd(width - len + 1), kind = rnd(7), d = rnd(16);
        for (let i = 0; i < len; i++) {
          if (kind === 0) h[a + i] = d;
          else if (kind === 1) h[a + i] = (d + i) & 15;
          else if (kind === 2) h[a + i] = (d - i + 16) & 15;
          else if (kind === 3) h[a + i] = i % 2 ? d : (d + 5) & 15;
          else if (kind === 4) h[a + i] = i < len / 2 ? h[a + i] : h[a + len - 1 - i];
          else if (kind === 5) h[a + i] = [0xd, 0xe, 0xa, 0xd, 0xb, 0xe, 0xe, 0xf][i % 8];
          else h[a + i] = i % 3 === 0 ? d : (d + 1) & 15;
        }
      }
      if (width === 64 && rnd(3) === 0) for (let i = 0, z = 1 + rnd(rnd(5) === 0 ? 40 : 6); i < z; i++) h[i] = 0;
      if (rnd(8) === 0) for (let i = 0, z = 2 + rnd(12); i < z; i++) h[i] = 15;
      return h.map((v) => H[v]).join("");
    };
    const LINES = [
      "#0{3,}", "#0{5}", "#?l{,24}", "#?l{40,}", "#?d{30,}", "#[0-3]{25,}", "#[05af]{max}", "#?l{min}", "#0{max}", "#(dead|beef|a){2,}", "#(ab){max}", "#(dead){1,}", "#(0){2,}", "##{,11}", "##{min}", "##{6,12}",
      "+{4}", "+{max}", "-{4}", "-{max}", "(?)\\1{2}", "(?)\\1{max}", "(??)\\1{2}", "(??)\\1{max}", "(?{3})\\1{1}", "(?{8})\\1{1}", "(?{5})\\1{max}", "~{7}", "~{9}", "~{max}",
      "^f{max}", "0{max}", "^dead0{max}", "^de[a-f]d0{max}", "^0{max}", "[0-3]{max}", "^dead & #0{3,}", "dead & +{3}", "^c & ~{max}", "#0{3,} & +{3} & ##{,14}", "~{5} & #?l{,30}",
    ];
    const out = [];
    for (const width of [64, 16]) {
      const W = width / 8, words = Array.from({ length: W }, (_, i) => `s${i}`);
      const N = 1536;
      const ids = Array.from({ length: N }, () => makeId(width));
      const inp = new Uint32Array(N * W);
      ids.forEach((hex, i) => { for (let w = 0; w < W; w++) inp[i * W + w] = parseInt(hex.slice(8 * w, 8 * w + 8), 16) >>> 0; });
      const mk = (data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage }); device.queue.writeBuffer(b, 0, data); return b; };
      const bIn = mk(inp), bPt = mk(new Uint32Array([1, 1, 1, 1]));
      for (const text of LINES) {
        let line;
        try { line = parseLine(text, width); } catch (e) { if (width === 64) out.push({ width, text, error: "parse: " + e.message }); continue; }
        const code = `${statLib(W)}
@group(0) @binding(0) var<storage, read> IN: array<u32>;
@group(0) @binding(1) var<storage, read> PT: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&OUT)) { return; }
  ${words.map((w, k) => `let ${w} = IN[i * ${W}u + ${k}u];`).join(" ")}
  var hit = false;
  ${statCode([line], words, { zl: width === 64 ? zlExpr(words) : "0u", SE: 0, rbIdx: () => "i" })}
  OUT[i] = select(0u, 1u, hit);
}`;
        const module = device.createShaderModule({ code });
        const info = await module.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === "error");
        if (errs.length) { out.push({ width, text, error: errs.map((m) => `${m.lineNum}: ${m.message} | ${code.split("\n")[m.lineNum - 1].slice(0, 120)}`).join("; ") }); continue; }
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const run = async (rb) => {
          const bRb = mk(rb), bOut = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          const bRd = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bIn } }, { binding: 2, resource: { buffer: bOut } }, ...(line.record ? [{ binding: 5, resource: { buffer: bRb } }] : [{ binding: 1, resource: { buffer: bPt } }])] });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
          enc.copyBufferToBuffer(bOut, 0, bRd, 0, N * 4);
          device.queue.submit([enc.finish()]);
          await bRd.mapAsync(GPUMapMode.READ);
          const got = new Uint32Array(bRd.getMappedRange().slice(0));
          bRd.unmap(); [bRb, bOut, bRd].forEach((b) => b.destroy());
          return got;
        };
        let mism = 0, hits = 0, first = "";
        const none = new Int32Array(N).fill(line.record === "min" ? 0x7FFFFFFF : -1);
        if (!line.record) {
          const got = await run(none);
          ids.forEach((hex, i) => {
            const want = line.gpuPass(hex), truth = line.match(hex);
            hits += want;
            if ((got[i] === 1) !== want) { mism++; first = first || `${hex}: gpu ${got[i]} model ${want}`; }
            if (truth && got[i] !== 1) { mism++; first = first || `${hex}: true match rejected`; }
          });
        } else {
          // best = value -/+ 1 must report, best = value must not; and no best (none) must report whenever the line has a value
          const vals = ids.map((hex) => { const t = line.recTerm, h = [...hex].map((c) => H.indexOf(c)); let z = 0; if (width === 64) while (z < 63 && h[z] === 0) z++; return t.kind === "hex" ? t.gpuHexRec(h, h.slice(z)) : line.evaluate(hex).value; });
          const mid = Int32Array.from(vals.map((v) => (v === null ? 0 : v)));
          const below = Int32Array.from(vals.map((v) => (v === null ? 0 : line.record === "max" ? v - 1 : v + 1)));
          const [gAt, gBelow, gNone] = [await run(mid), await run(below), await run(none)];
          ids.forEach((hex, i) => {
            const v = vals[i];
            if (v !== null) hits++;
            const wantBelow = v !== null && (line.record === "max" ? v >= 0 : v >= 0), wantAt = false, wantNone = v !== null;
            // the model: gpuPass says the same thing
            if (gAt[i] === 1 || (gBelow[i] === 1) !== wantBelow || (gNone[i] === 1) !== wantNone) { mism++; first = first || `${hex}: value ${v} at ${gAt[i]} below ${gBelow[i]} none ${gNone[i]}`; }
            if (v !== null && line.gpuPass(hex, mid[i]) !== false) { mism++; first = first || `${hex}: model passes at its own value`; }
            if (v !== null && line.gpuPass(hex, below[i]) !== true) { mism++; first = first || `${hex}: model rejects just below its value`; }
          });
        }
        out.push({ width, text, mism, hits, first, record: line.record });
      }
    }
    return out;
  });
  for (const width of [64, 16]) {
    const rows = sres.filter((r) => r.width === width);
    const errs = rows.filter((r) => r.error);
    errs.forEach((r) => check(false, `stat WGSL ${r.text} (${width}): ${r.error}`));
    const bad = rows.filter((r) => !r.error && r.mism);
    bad.forEach((r) => check(false, `stat WGSL ${r.text} (${width}): ${r.mism} mismatches, first ${r.first}`));
    const quiet = rows.filter((r) => !r.error && !r.mism && r.hits === 0).map((r) => r.text);
    check(!errs.length && !bad.length, `stat WGSL, ${width === 64 ? "Mojang 8-word digest" : "NameMC 2-word id"}: ${rows.length} lines (${rows.filter((r) => r.record).length} record lines), GPU = JS model on 1536 ids each${quiet.length ? ` (no hit at all for: ${quiet.join(" ")})` : ""}`);
  }
} catch (e) {
  console.log("TEST ERROR:", e.message); failures++;
} finally {
  await browser.close();
  server.kill();
}
console.log(failures ? `\n${failures} FAILURE(S)` : "\nPATTERN GPU TESTS PASSED");
process.exit(failures ? 1 : 0);
