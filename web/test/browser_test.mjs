// End-to-end browser test: drives system Chrome (puppeteer-core) against a local static server.
// puppeteer-core is NOT part of the repo. Install it in a scratch dir and point WEBTEST_DIR at it:
//   set WEBTEST_DIR=C:\path\to\scratch   (contains node_modules/puppeteer-core)
//   node web/test/browser_test.mjs [--headful] [--skin skins/steve.png] [--out DIR]
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const headful = args.includes("--headful");
const skin = path.resolve(repo, opt("--skin", "skins/steve.png"));
const outDir = path.resolve(opt("--out", path.join(process.env.WEBTEST_DIR || ".", "downloads")));
const CHROME = process.env.CHROME || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const puppeteer = createRequire(path.join(process.env.WEBTEST_DIR || process.cwd(), "x.js"))("puppeteer-core");
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });

const PORT = 8137;
const server = spawn("py", [path.join(repo, "web", "serve.py"), String(PORT)], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
let failures = 0;
const check = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) failures++; };

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: headful ? false : "new",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log("  [page]", m.text()));
  page.on("pageerror", (e) => console.log("  [pageerror]", e.message));
  const cdp = await browser.target().createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: outDir, eventsEnabled: true });
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);
  const gpuInfo = await page.evaluate(async () => {
    if (!navigator.gpu) return null;
    const a = await navigator.gpu.requestAdapter();
    return a ? { info: a.info ? { vendor: a.info.vendor, arch: a.info.architecture, desc: a.info.description } : {} } : { none: true };
  });
  console.log("WebGPU:", JSON.stringify(gpuInfo));
  check(!!gpuInfo && !gpuInfo.none, "WebGPU adapter available in this Chrome");
  if (!gpuInfo || gpuInfo.none) throw new Error("no WebGPU");

  const input = await page.$("#file");
  await input.uploadFile(skin);
  // wait for OUR skin, not the default Steve the page loads on start
  await page.waitForFunction((n) => window.__skinhash?.prep && window.__skinhash.name === n,
    { timeout: 10000 }, path.basename(skin, ".png"));
  const info = await page.evaluate(() => ({ eligible: __skinhash.prep.eligible, wm: __skinhash.prep.wm.length, id: __skinhash.origId }));
  check(info.eligible && info.wm === 6, `skin loaded, fast mode eligible, watermark pixels=${info.wm}, current id ${info.id}`);

  const waitIdle = (ms) => page.waitForFunction(() => !__skinhash.running && (__skinhash.error || __skinhash.gate), { timeout: ms, polling: 200 });
  const reset = () => page.evaluate(() => { __skinhash.error = null; __skinhash.gate = null; });

  // 1. benchmark
  await page.click("#bench");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  await waitIdle(60000);
  const b = await page.evaluate(() => ({ gate: __skinhash.gate, bench: __skinhash.bench, err: __skinhash.error }));
  check(b.gate?.ok, `correctness gate: GPU reproduced ${b.gate?.n}/${b.gate?.n} CPU test vectors; CPU self-test ${b.gate?.selfTest}`);
  check(!!b.bench, `benchmark rate = ${b.bench ? (b.bench.rate / 1e9).toFixed(3) + " G/s" : "n/a"} over ${b.bench?.secs?.toFixed(1)} s (${b.bench?.tried} tried); native CUDA reference 3.85 G/s`);
  if (b.err) console.log("  error:", b.err);

  // 2. searches
  const runSearch = async (label, fields, nHits) => {
    await reset();
    await page.evaluate(() => { __skinhash.hits.length = 0; document.querySelector("#hits").innerHTML = ""; });
    for (const [id, v] of Object.entries(fields)) { await page.$eval("#" + id, (el, val) => { el.value = val; el.dispatchEvent(new Event("input")); }, v); }
    await page.$eval("#stopn", (el, v) => { el.value = v; }, String(nHits));
    const t0 = Date.now();
    await page.click("#start");
    await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
    await page.waitForFunction(() => !__skinhash.running, { timeout: 90000, polling: 200 });
    const r = await page.evaluate(() => ({ err: __skinhash.error, hits: __skinhash.hits.map((h) => ({ id: h.id, url: h.url, whys: h.whys })) }));
    console.log(`  ${label}: ${r.hits.length} hit(s) in ${((Date.now() - t0) / 1000).toFixed(1)} s ${r.err ? "ERROR " + r.err : ""}`);
    return r;
  };

  const downloaded = [];
  const grab = async (res) => {
    const links = await page.$$("#hits a.dl");
    for (const l of links) await l.click();
    await new Promise((r) => setTimeout(r, 1500));
    for (const h of res.hits) {
      const f = path.join(outDir, h.id + ".png");
      check(fs.existsSync(f), `downloaded ${h.id}.png`);
      if (fs.existsSync(f)) downloaded.push({ f, id: h.id, whys: h.whys });
    }
  };
  const r1 = await runSearch("-p beef", { prefixes: "beef", suffixes: "", zeros: "0" }, 3);
  check(r1.hits.length === 3 && r1.hits.every((h) => h.id.startsWith("beef")), "prefix beef: 3 hits, ids start with beef");
  await grab(r1);
  const r2 = await runSearch("7 leading zeros", { prefixes: "", suffixes: "", zeros: "7" }, 1);
  check(r2.hits.length === 1 && r2.hits[0].url.split("/").pop().length <= 57, "7-zero target: 1 hit");
  await grab(r2);
  const r3 = await runSearch("suffix cafe", { prefixes: "", suffixes: "cafe", zeros: "0" }, 2);
  check(r3.hits.length === 2, "suffix cafe: 2 hits");
  await grab(r3);

  // 3. verify with the native tool
  for (const d of downloaded) {
    const v = spawnSync("py", ["skinhash.py", "verify", skin, d.f], { cwd: repo, encoding: "utf8" });
    check(v.status === 0 && /^OK/.test(v.stdout), `skinhash.py verify ${d.id.slice(0, 20)}...: ${v.stdout.trim().slice(0, 100)}`);
    const t = spawnSync("py", ["skinhash.py", "detect", d.f], { cwd: repo, encoding: "utf8" });
    check(/YES/.test(t.stdout) && /v3 at \(0, 7\)/.test(t.stdout), `skinhash.py detect: ${t.stdout.trim().slice(0, 100)}`);
  }

  // ---------- NameMC mode ----------
  const setMode = async (m) => { await page.click(`input[name=mode][value=${m}]`); await new Promise((r) => setTimeout(r, 300)); };
  const nmCheck = (hit, prefix, extra = []) => {
    const v = spawnSync("py", [path.join(here, "namemc_check.py"), skin, hit, prefix, ...extra], { cwd: repo, encoding: "utf8" });
    check(v.status === 0 && /^OK/.test(v.stdout), `namemc_check ${path.basename(hit).slice(0, 18)}: ${(v.stdout + v.stderr).trim().slice(0, 100)}`);
  };
  await setMode("namemc");
  const nmInfo = await page.evaluate(() => ({ mode: __skinhash.mode, knobs: __skinhash.nprep.knobs, nb: __skinhash.nprep.nb, fast: __skinhash.nprep.fast, nm: __skinhash.origNm }));
  check(nmInfo.mode === "namemc" && nmInfo.fast && nmInfo.knobs.length === 3, `NameMC mode: knobs ${JSON.stringify(nmInfo.knobs)}, ${nmInfo.nb} SHA blocks, current NameMC id ${nmInfo.nm}`);
  await reset();
  await page.click("#bench");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  await waitIdle(60000);
  const nb = await page.evaluate(() => ({ gate: __skinhash.gate, bench: __skinhash.bench, err: __skinhash.error }));
  check(nb.gate?.ok, `NameMC gate: GPU reproduced ${nb.gate?.n}/${nb.gate?.n} CPU test vectors`);
  check(!!nb.bench, `NameMC benchmark rate = ${nb.bench ? (nb.bench.rate / 1e9).toFixed(3) + " G/s" : "n/a"} over ${nb.bench?.secs?.toFixed(1)} s; native CUDA reference 2.81 G/s`);
  if (nb.err) console.log("  error:", nb.err);

  const n1 = await runSearch("NameMC -p beef", { prefixes: "beef", suffixes: "", zeros: "0" }, 3);
  check(n1.hits.length === 3 && n1.hits.every((h) => h.id.startsWith("beef") && h.url.startsWith("https://namemc.com/skin/")), "NameMC prefix beef: 3 hits");
  await grab(n1);
  for (const h of n1.hits) nmCheck(path.join(outDir, h.id + ".png"), "beef");
  const nz = await runSearch("NameMC prefix 0000 (leading zero allowed)", { prefixes: "0000", suffixes: "", zeros: "0" }, 1);
  check(nz.hits.length === 1 && nz.hits[0].id.startsWith("0000"), "NameMC prefix 0000: 1 hit");
  await grab(nz);
  for (const h of nz.hits) nmCheck(path.join(outDir, h.id + ".png"), "0000");

  // unused pixels only: the general kernel (about 50 SHA blocks per candidate)
  await page.click("#unusedonly");
  await new Promise((r) => setTimeout(r, 300));
  const gi = await page.evaluate(() => ({ fast: __skinhash.nprep.fast, nb: __skinhash.nprep.nb, knobs: __skinhash.nprep.knobs }));
  check(!gi.fast && gi.nb > 2, `unused-only layout: ${gi.nb} blocks, knobs ${JSON.stringify(gi.knobs)}`);
  const nu = await runSearch("NameMC unused-only -p b", { prefixes: "bee", suffixes: "", zeros: "0" }, 1);
  check(nu.hits.length === 1 && nu.hits[0].id.startsWith("bee"), "NameMC unused-only: 1 hit (general kernel passed the gate)");
  await grab(nu);
  for (const h of nu.hits) nmCheck(path.join(outDir, h.id + ".png"), "bee", ["--unused-only"]);
  await page.click("#unusedonly");
  await new Promise((r) => setTimeout(r, 300));

  // combined flow: NameMC hit -> "Use as skin" -> Mojang search keeps the NameMC id
  const c1 = await runSearch("combined: NameMC beef", { prefixes: "beef", suffixes: "", zeros: "0" }, 1);
  check(c1.hits.length === 1, "combined: NameMC hit found");
  await grab(c1);
  const nmFile = path.join(outDir, c1.hits[0].id + ".png");
  await page.click("#hits button.use");
  await page.waitForFunction(() => __skinhash.mode === "mojang" && __skinhash.prep?.eligible, { timeout: 10000 });
  const cm = await page.evaluate(() => ({ mode: __skinhash.mode, k: __skinhash.prep.kvis, hits: __skinhash.hits.length }));
  check(cm.mode === "mojang" && cm.k >= 5, `Use as skin: switched to Mojang mode, fast mode still eligible (invisible tail ${cm.k})`);
  await reset();
  for (const [id, v] of Object.entries({ prefixes: "cafe", suffixes: "", zeros: "0" })) await page.$eval("#" + id, (el, val) => { el.value = val; el.dispatchEvent(new Event("input")); }, v);
  await page.$eval("#stopn", (el) => { el.value = "1"; });
  await page.click("#start");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  await page.waitForFunction(() => !__skinhash.running, { timeout: 90000, polling: 200 });
  const mj = await page.evaluate(() => ({ err: __skinhash.error, hits: __skinhash.hits.filter((h) => h.mode !== "namemc").map((h) => ({ id: h.id })) }));
  check(!mj.err && mj.hits.length === 1 && mj.hits[0].id.startsWith("cafe"), `combined: Mojang search cafe from the NameMC hit: ${mj.hits.length} hit(s) ${mj.err || ""}`);
  const mjLinks = await page.$$("#hits a.dl");
  for (const l of mjLinks) await l.click();
  await new Promise((r) => setTimeout(r, 1500));
  const mjFile = path.join(outDir, mj.hits[0].id + ".png");
  check(fs.existsSync(mjFile), "combined: Mojang hit downloaded");
  const vv = spawnSync("py", ["skinhash.py", "verify", nmFile, mjFile], { cwd: repo, encoding: "utf8" });
  check(vv.status === 0 && /^OK/.test(vv.stdout), `combined: skinhash.py verify <namemc hit> <mojang hit>: ${vv.stdout.trim().slice(0, 90)}`);
  const ids = spawnSync("py", ["skinhash.py", "ids", nmFile, mjFile], { cwd: repo, encoding: "utf8" });
  const idl = [...ids.stdout.matchAll(/^\s+NameMC\s+([0-9a-f]{16})\s/gm)].map((m) => m[1]);
  check(idl.length === 2 && idl[0] === idl[1] && idl[0] === path.basename(nmFile, ".png"), `combined: NameMC id unchanged by the Mojang search (${idl.join(" = ")})`);

  // ---------- Both mode: NameMC beef, then Mojang cafe, one click ----------
  await page.click("input[name=mode][value=mojang]");
  await page.$eval("#file", (el) => { el.value = ""; });
  await input.uploadFile(skin);   // back to the original skin (the combined flow above left a NameMC hit loaded)
  await page.waitForFunction((nm) => __skinhash.origNm === nm && !__skinhash.running, { timeout: 10000 }, nmInfo.nm);
  await page.click("input[name=mode][value=both]");
  await new Promise((r) => setTimeout(r, 300));
  const bi = await page.evaluate(() => ({ mode: __skinhash.mode, g2: !document.querySelector("#g2").hidden, start: !document.querySelector("#start").disabled }));
  check(bi.mode === "both" && bi.g2 && bi.start, "Both mode selected: two target groups shown, Start enabled");
  await reset();
  await page.evaluate(() => { __skinhash.hits.length = 0; document.querySelector("#hits").innerHTML = ""; });
  for (const [id, v] of Object.entries({ prefixes: "beef", suffixes: "", zeros: "0", mprefixes: "cafe", msuffixes: "", mzeros: "0" })) await page.$eval("#" + id, (el, val) => { el.value = val; el.dispatchEvent(new Event("input")); }, v);
  const tB = Date.now();
  await page.click("#start");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  const seen = new Set();
  await page.waitForFunction(() => { window.__lbl = window.__lbl || new Set(); window.__lbl.add((document.querySelector("#status").textContent.match(/^Phase \d\/\d \((\w+)\)/) || [])[1]); return !__skinhash.running; }, { timeout: 120000, polling: 50 });
  const phasesSeen = await page.evaluate(() => [...window.__lbl].filter(Boolean));
  const bo = await page.evaluate(() => ({ err: __skinhash.error, phases: document.querySelector("#phasebox").textContent, hits: __skinhash.hits.map((h) => ({ mode: h.mode, id: h.id, id16: h.id16, murl: h.murl, url: h.url })) }));
  console.log("  Both: " + ((Date.now() - tB) / 1000).toFixed(1) + " s " + (bo.err ? "ERROR " + bo.err : ""));
  for (const line of bo.phases.split(String.fromCharCode(10))) console.log("    " + line);
  check(phasesSeen.includes("NameMC") && phasesSeen.includes("Mojang"), `Both: progress showed both phases (${phasesSeen.join(", ")})`);
  const fin = bo.hits.filter((h) => h.mode === "both"), inter = bo.hits.filter((h) => h.mode === "namemc");
  check(!bo.err && fin.length === 1 && inter.length === 1, `Both: 1 intermediate NameMC hit and 1 final result`);
  if (fin.length === 1 && inter.length === 1) {
    check(fin[0].id16.startsWith("beef") && fin[0].id.startsWith("cafe") && fin[0].id16 === inter[0].id16, `Both: final NameMC ${fin[0].id16}, Mojang ${fin[0].id.slice(0, 20)}...`);
    for (const l of await page.$$("#hits a.dl")) await l.click();
    await new Promise((r) => setTimeout(r, 1500));
    const finalF = path.join(outDir, fin[0].id + ".png"), interF = path.join(outDir, inter[0].id16 + ".png");
    check(fs.existsSync(finalF) && fs.existsSync(interF), "Both: final PNG and intermediate PNG downloaded");
    nmCheck(finalF, "beef", ["--mojang-knobs", "--id", fin[0].id16]);
    const bv = spawnSync("py", ["skinhash.py", "verify", interF, finalF], { cwd: repo, encoding: "utf8" });
    check(bv.status === 0 && /^OK/.test(bv.stdout), `Both: skinhash.py verify <phase-1 hit> <final>: ${bv.stdout.trim().slice(0, 80)}`);
    const bv0 = spawnSync("py", ["skinhash.py", "verify", skin, finalF], { cwd: repo, encoding: "utf8" });
    check(bv0.status === 0 && /^OK/.test(bv0.stdout), `Both: skinhash.py verify <original> <final>: ${bv0.stdout.trim().slice(0, 80)}`);
    const bid = spawnSync("py", ["skinhash.py", "ids", finalF], { cwd: repo, encoding: "utf8" });
    const bidNm = (bid.stdout.match(/^\s+NameMC\s+([0-9a-f]{16})\s/m) || [])[1];
    check(bidNm === fin[0].id16, `Both: skinhash.py ids shows NameMC ${bidNm}`);
    check(path.basename(finalF).startsWith("cafe"), "Both: Mojang id starts with cafe");
  }
  // starting again begins with phase 1 on the original skin
  await reset();
  await page.evaluate(() => { __skinhash.hits.length = 0; document.querySelector("#hits").innerHTML = ""; });
  await page.$eval("#mprefixes", (el) => { el.value = "dead"; el.dispatchEvent(new Event("input")); });
  await page.click("#start");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  await page.waitForFunction(() => !__skinhash.running, { timeout: 120000, polling: 100 });
  const again = await page.evaluate(() => ({ err: __skinhash.error, n: __skinhash.hits.filter((h) => h.mode === "both").length, id: (__skinhash.hits.find((h) => h.mode === "both") || {}).id }));
  check(!again.err && again.n === 1 && again.id.startsWith("dead"), `Both: second run from phase 1 works (Mojang ${again.id?.slice(0, 12)}...)`);
  // Stop works in phase 1: an impossible-to-hit NameMC target, then Stop
  await reset();
  await page.evaluate(() => { __skinhash.hits.length = 0; document.querySelector("#hits").innerHTML = ""; });
  await page.$eval("#prefixes", (el) => { el.value = "beef0123456789"; el.dispatchEvent(new Event("input")); });
  await page.click("#start");
  await page.waitForFunction(() => __skinhash.running, { timeout: 5000 });
  await page.waitForFunction(() => /Searching \(NameMC\)/.test(document.querySelector("#status").textContent), { timeout: 20000 });
  await page.click("#stop");
  await page.waitForFunction(() => !__skinhash.running, { timeout: 20000 });
  const stopped = await page.evaluate(() => ({ phases: document.querySelector("#phasebox").textContent, hits: __skinhash.hits.length }));
  console.log("  stopped:", JSON.stringify(stopped));
  check(stopped.hits === 0 && /stopped/.test(stopped.phases) && /not run/.test(stopped.phases), "Both: Stop in phase 1 ends the run, phase 2 not started");
} catch (e) {
  console.log("TEST ERROR:", e.message); failures++;
} finally {
  await browser.close();
  server.kill();
}
console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL BROWSER TESTS PASSED");
process.exit(failures ? 1 : 0);
