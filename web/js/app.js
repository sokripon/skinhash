// UI glue: skin loading, target inputs, search orchestration, hit list.
import { decodePng, mojangPng, W, H } from "./png.js";
import { sha256, toHex } from "./hash.js";
import { prepareSkin, autoChoose, rebuildJob, selfTest, makeTestVectors, makeRng, MAX_K } from "./job.js";
import { initDevice, webgpuAvailable, GpuJob } from "./gpu.js";
import { parseHexList, targetList, targetKey, gpuTargets, expectedTries } from "./targets.js";
import { verifyHit } from "./verify.js";

const $ = (id) => document.getElementById(id);
const S = { prep: null, orig: null, name: "skin", running: false, stop: false, gpu: null, hits: [], gate: null, bench: null, log: [] };
window.__skinhash = S;   // exposed for automated tests

const fmt = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + " T" : n >= 1e9 ? (n / 1e9).toFixed(2) + " G" : n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " k" : String(Math.round(n));
const fmtTime = (s) => s < 90 ? s.toFixed(s < 10 ? 1 : 0) + " s" : s < 5400 ? (s / 60).toFixed(1) + " min" : s < 172800 ? (s / 3600).toFixed(1) + " h" : (s / 86400).toFixed(1) + " days";
const setStatus = (t, cls = "") => { const el = $("status"); el.textContent = t; el.className = "box " + cls; S.log.push(t); };
const showErr = (id, msg) => { const el = $(id); el.hidden = !msg; el.textContent = msg || ""; };

if (!webgpuAvailable()) $("nogpu").hidden = false;

// ---------- skin loading ----------
function drawPreview() {
  const { prep } = S;
  const cv = $("preview"), ctx = cv.getContext("2d");
  const tmp = document.createElement("canvas"); tmp.width = W; tmp.height = H;
  tmp.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(prep.orig), W, H), 0, 0);   // display only
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
  const sc = cv.width / W;
  const mark = (pixels, color) => { ctx.fillStyle = color; pixels.forEach(([x, y]) => ctx.fillRect(x * sc, y * sc, sc, sc)); };
  mark(prep.wm, "rgba(20,200,216,.85)");
  const k = S.chosenK || prep.kmaxUsable;
  if (prep.eligible) mark(Array.from({ length: k }, (_, i) => [W - k + i, H - 1]), "rgba(224,42,208,.85)");
}

function renderInfo() {
  const { prep } = S;
  const k = S.chosenK || prep.kmaxUsable;
  const rows = [];
  rows.push(["Current id", `<code>${S.origId}</code>`]);
  rows.push(["Fast mode", prep.eligible
    ? `Possible. The last ${prep.kvis >= MAX_K ? MAX_K : prep.kvis} pixels of the image are invisible.`
    : `<b>Not possible for this skin.</b> ${prep.reason}`]);
  if (prep.eligible) rows.push(["Search pixels", `${S.chosenK ? "" : "up to "}${k} (x ${W - k}&ndash;${W - 1}, y ${H - 1}): only their hidden RGB changes`
    + (S.chosenK ? "" : ". The final count is chosen when the search starts")]);
  rows.push(["Watermark", prep.wm.length ? `(${prep.wm[0].join(",")})&ndash;(${prep.wm.at(-1).join(",")}), unused and transparent` : ($("wm").checked ? "<b>no free unused pixels</b>, watermark skipped" : "off")]);
  rows.push(["Alpha / visible pixels", "never changed"]);
  $("info").innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
}

function loadSkinBytes(bytes, name) {
  showErr("skinerr", "");
  let dec;
  try { dec = decodePng(bytes); } catch (e) { showErr("skinerr", e.message); S.prep = null; $("skinbox").hidden = $("targetsec").hidden = true; return; }
  S.orig = dec.rgba; S.name = name.replace(/\.png$/i, "");
  S.origId = (toHex(sha256(mojangPng(dec.rgba))).replace(/^0+/, "") || "0");
  S.chosenK = 0;
  rebuildPrep();
  $("skinbox").hidden = false; $("targetsec").hidden = false; $("runsec").hidden = false;
  S.hits = []; $("hits").innerHTML = ""; $("nohits").hidden = false;
}

function rebuildPrep() {
  if (!S.orig) return;
  S.prep = prepareSkin(S.orig, { watermark: $("wm").checked });
  drawPreview(); renderInfo(); updateButtons();
  if (!S.prep.eligible) setStatus("Fast mode is not possible for this skin, so it can't be searched here. " + S.prep.reason, "warn");
  else setStatus("Ready.");
}

async function pick(file) {
  if (!file) return;
  S.userPicked = true;
  loadSkinBytes(new Uint8Array(await file.arrayBuffer()), file.name);
}

// Default skin: classic Steve, fetched from Mojang (CORS allowed) instead of bundling Mojang's asset.
// The bytes must hash to the texture id; a skin the user picks meanwhile always wins.
const STEVE_ID = "1a4af718455d4aab528e7a61f86fa25e6a369d1768dcb13f7df319a713eb810b";
async function loadDefaultSkin() {
  try {
    const r = await fetch(`https://textures.minecraft.net/texture/${STEVE_ID}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const bytes = new Uint8Array(await r.arrayBuffer());
    if (toHex(sha256(bytes)) !== STEVE_ID) throw new Error("unexpected bytes");
    if (S.userPicked) return;
    loadSkinBytes(bytes, "steve.png");
    if (S.prep?.eligible) setStatus("Loaded the default Steve skin. Drop or choose your own skin to replace it.");
  } catch (e) {
    console.warn("default Steve skin not loaded:", e.message);   // offline etc.: page just starts empty
  }
}
loadDefaultSkin();
$("file").addEventListener("change", (e) => pick(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
$("wm").addEventListener("change", () => { if (!S.running) rebuildPrep(); });

// ---------- targets ----------
function readTargets() {
  const p = parseHexList($("prefixes").value, "prefix"), s = parseHexList($("suffixes").value, "suffix");
  const errors = [...p.errors, ...s.errors];
  const zeros = Math.max(0, Math.min(15, parseInt($("zeros").value || "0", 10) || 0));
  const targets = targetList({ prefixes: p.list, suffixes: s.list, zeros });
  if (targets.length > 32) errors.push("At most 32 targets at a time.");
  return { targets, errors };
}
function updateExpect() {
  const { targets, errors } = readTargets();
  showErr("targeterr", errors.join(" "));
  $("expect").textContent = targets.length && !errors.length
    ? "Expected tries per hit: " + targets.map((t) => `${t.kind} ${t.text} ~${fmt(expectedTries(t))}`).join(", ") + (S.lastRate ? ` (about ${targets.map((t) => fmtTime(expectedTries(t) / S.lastRate)).join(", ")} at your measured rate)` : "")
    : "";
}
["prefixes", "suffixes", "zeros"].forEach((id) => $(id).addEventListener("input", updateExpect));

function updateButtons() {
  const ok = !!S.prep?.eligible && !S.running;
  $("start").disabled = !ok; $("bench").disabled = !ok; $("stop").disabled = !S.running;
}

// ---------- hits ----------
function addHit(r, whys) {
  const url = `https://textures.minecraft.net/texture/${r.id}`;
  const blobUrl = URL.createObjectURL(new Blob([r.png], { type: "image/png" }));
  const li = document.createElement("li");
  li.innerHTML = `<a class="link" target="_blank" rel="noopener"></a><div class="why"></div><div class="btns"><button type="button" class="copy">Copy link</button><a class="dl">Download PNG</a></div>`;
  const a = li.querySelector("a.link"); a.href = url; a.textContent = url;
  li.querySelector(".why").textContent = `${whys.map((t) => `${t.kind} ${t.text}`).join(", ")} | verified | ${r.png.length} bytes`;
  const dl = li.querySelector("a.dl"); dl.href = blobUrl; dl.download = `${r.id}.png`;
  li.querySelector(".copy").addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(url); e.target.textContent = "Copied"; } catch { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy link"), 1500);
  });
  $("hits").prepend(li); $("nohits").hidden = true;
  S.hits.push({ id: r.id, hex: r.hex, url, png: r.png, whys: whys.map(targetKey) });
}

// ---------- search ----------
const rate = { samples: [] };
function stats(tried, best, t0) {
  const now = performance.now();
  rate.samples.push([now, tried]);
  while (rate.samples.length > 2 && now - rate.samples[0][0] > 3000) rate.samples.shift();
  const [t1, n1] = rate.samples[0];
  const r = now > t1 ? (tried - n1) / ((now - t1) / 1000) : 0;
  $("rate").textContent = r ? fmt(r) + "/s" : "–";
  $("tried").textContent = fmt(tried); $("best").textContent = String(best);
  $("elapsed").textContent = fmtTime((now - t0) / 1000);
  return r;
}

async function startSearch(bench) {
  if (S.running || !S.prep?.eligible) return;
  let targets = [];
  if (!bench) {
    const r = readTargets();
    showErr("targeterr", r.errors.join(" "));
    if (r.errors.length) return;
    if (!r.targets.length) { showErr("targeterr", "Enter at least one prefix, suffix or a leading-zeros minimum (or use the benchmark)."); return; }
    targets = r.targets;
  }
  showErr("runerr", "");
  S.running = true; S.stop = false; S.bench = null; updateButtons();
  $("rate").textContent = "–"; $("tried").textContent = "0"; $("best").textContent = "0"; $("elapsed").textContent = "0 s";
  const allMode = !bench && document.querySelector("input[name=stop]:checked").value === "all";
  const maxHits = Math.max(1, parseInt($("stopn").value || "1", 10) || 1);
  const remaining = new Map(targets.map((t) => [targetKey(t), t]));
  const wantNow = () => [...remaining.values()];
  const rng = makeRng();
  let g = null, totalTried = 0, bestZeros = 0, t0 = performance.now(), hitCount = 0, dropped = 0, bad = 0;
  rate.samples = [];
  const queue = []; let draining = false;
  const finishCheck = () => { if (bench) return; if (allMode ? remaining.size === 0 : hitCount >= maxHits) S.stop = true; };
  const drain = async () => {
    if (draining) return; draining = true;
    while (queue.length && !S.stop) {
      const { job, idx } = queue.shift();
      const r = await verifyHit(job, S.prep, idx, targets);
      if (!r.ok) { bad++; showErr("runerr", `A GPU hit failed CPU verification and was discarded (${r.error}). ${bad} so far.`); continue; }
      if (S.stop) break;
      if (allMode) {
        const fresh = r.whys.filter((t) => remaining.has(targetKey(t)));
        if (!fresh.length) { dropped++; continue; }
        fresh.forEach((t) => remaining.delete(targetKey(t)));
        addHit(r, fresh);
        if (g) { const gt = gpuTargets(wantNow()); g.setTargets(gt.pats, gt.zeros); }
      } else addHit(r, r.whys);
      hitCount++; finishCheck();
    }
    draining = false;
  };
  try {
    setStatus("Initialising WebGPU...");
    if (!S.gpu) S.gpu = await initDevice();
    const device = S.gpu.device;
    device.lost.then((info) => { if (S.running) { S.stop = true; showErr("runerr", "The GPU device was lost: " + info.message); } });
    setStatus("Choosing pixel count and value set (looking for a one-block layout)...");
    await new Promise((r) => setTimeout(r, 20));
    const chosen = await autoChoose(S.prep, rng, { onProgress: (t) => setStatus(t) });
    let job = chosen.job;
    S.chosenK = chosen.K; drawPreview(); renderInfo();
    let sets = 0;
    while (!S.stop) {
      sets++;
      setStatus(`CPU self-test (${sets > 1 ? "new value set #" + sets + ", " : ""}K=${job.K}, ${job.nblk} SHA block${job.nblk > 1 ? "s" : ""})...`);
      await new Promise((r) => setTimeout(r, 10));
      const nSelf = selfTest(job, rng, 250);
      makeTestVectors(job, rng, 24);
      setStatus("Compiling GPU kernel and running the correctness gate...");
      g = new GpuJob(device, job, { wg: Number(new URLSearchParams(location.search).get("wg")) || 256 });
      await g.build();
      const gate = await g.gate();
      S.gate = { ...gate, selfTest: nSelf };
      if (!gate.ok) throw new Error(`GPU correctness gate FAILED: ${gate.bad} of ${gate.n} test vectors differ between GPU and CPU (${gate.detail}). The search was not started.`);
      const gt = gpuTargets(wantNow()); g.setTargets(gt.pats, gt.zeros);
      if (bench && sets === 1) setTimeout(() => (S.stop = true), 10000);
      setStatus(`${bench ? "Benchmark" : "Searching"}: GPU reproduced ${gate.n}/${gate.n} CPU test vectors; CPU self-test ${nSelf}/${nSelf} identical to pako. K=${job.K} (${job.K * 3} values, ${job.nblk} SHA block${job.nblk > 1 ? "s" : ""}).`, "ok");
      const base = totalTried, tRun = performance.now();
      const res = await g.run(rng, {
        shouldStop: () => S.stop,
        onHits: (idxs, nh) => {
          if (bench) return;
          for (const idx of idxs) if (queue.length < 64) queue.push({ job, idx }); else dropped++;
          if (nh > idxs.length) dropped += nh - idxs.length;
          drain();
        },
        onProgress: ({ tried, bestZeros: bz }) => {
          totalTried = base + tried; bestZeros = Math.max(bestZeros, bz);
          S.lastRate = stats(totalTried, bestZeros, t0) || S.lastRate;
        },
      });
      const runSecs = (performance.now() - tRun) / 1000;
      if (bench) S.bench = { rate: (totalTried - base) / runSecs, tried: totalTried - base, secs: runSecs };
      g.destroy(); g = null;
      if (res === "exhausted") {
        setStatus("This value set is fully searched. Rolling over to a new one...");
        job = await rebuildJob(S.prep, rng, job.K, job.nblk);
      }
    }
    await drain();
    if (bench && S.bench) {
      S.lastRate = S.bench.rate;
      $("rate").textContent = fmt(S.bench.rate) + "/s";
      setStatus(`Benchmark done: ${fmt(S.bench.rate)} candidates/s average over ${S.bench.secs.toFixed(1)} s (${fmt(S.bench.tried)} tried).`, "ok");
    } else setStatus(`Stopped. ${hitCount} verified hit${hitCount === 1 ? "" : "s"}${dropped ? `; ${dropped} further hit(s) skipped` : ""}.`);
  } catch (e) {
    console.error(e);
    showErr("runerr", String(e.message || e));
    setStatus("Search did not run.", "err");
    S.error = String(e.message || e);
  } finally {
    g?.destroy();
    S.running = false; updateButtons(); updateExpect();
  }
}

$("start").addEventListener("click", () => startSearch(false));
$("bench").addEventListener("click", () => startSearch(true));
$("stop").addEventListener("click", () => { S.stop = true; });
S.ready = true;
