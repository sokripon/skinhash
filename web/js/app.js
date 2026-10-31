// UI glue: skin loading, target inputs, search orchestration, hit list.
import { decodePng, mojangPng, W, H } from "./png.js";
import { sha256, toHex } from "./hash.js";
import { prepareSkin, autoChoose, rebuildJob, selfTest, makeTestVectors, makeRng, MAX_K } from "./job.js";
import { initDevice, webgpuAvailable, GpuJob } from "./gpu.js";
import { parseHexList, targetList, targetKey, gpuTargets, gpuNmTargets, expectedTries } from "./targets.js";
import { verifyHit, verifyNmHit, verifyBoth } from "./verify.js";
import { prepareNameMc, makeNmTests, namemcIdHex } from "./namemc.js";
import { NmGpuJob } from "./nmgpu.js";

const $ = (id) => document.getElementById(id);
const S = { rates: {}, phases: [], mode: "mojang", nprep: null, prep: null, orig: null, name: "skin", running: false, stop: false, gpu: null, hits: [], gate: null, bench: null, log: [] };
window.__skinhash = S;   // exposed for automated tests

const fmt = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + " T" : n >= 1e9 ? (n / 1e9).toFixed(2) + " G" : n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " k" : String(Math.round(n));
const fmtTime = (s) => s < 90 ? s.toFixed(s < 10 ? 1 : 0) + " s" : s < 5400 ? (s / 60).toFixed(1) + " min" : s < 172800 ? (s / 3600).toFixed(1) + " h" : (s / 86400).toFixed(1) + " days";
const setStatus = (t, cls = "") => { t = (S.phaseLabel || "") + t; const el = $("status"); el.textContent = t; el.className = "box " + cls; S.log.push(t); };
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
  if (S.mode === "namemc") { if (S.nprep?.eligible) mark(S.nprep.knobs, "rgba(224,42,208,.85)"); return; }
  if (S.mode === "both" && S.nprep?.eligible) mark(S.nprep.knobs, "rgba(240,160,20,.9)");
  const k = S.chosenK || prep.kmaxUsable;
  if (prep.eligible) mark(Array.from({ length: k }, (_, i) => [W - k + i, H - 1]), "rgba(224,42,208,.85)");
}

function renderInfo() {
  const { prep } = S;
  const k = S.chosenK || prep.kmaxUsable;
  const rows = [];
  rows.push(["Current texture id", `<code>${S.origId}</code>`]);
  rows.push(["Current NameMC id", `<code>${S.origNm}</code>`]);
  if (S.mode === "namemc") {
    const n = S.nprep;
    rows.push(["NameMC knob", n.eligible
      ? `3 pixels ${n.knobs.map(([x, y]) => `(${x},${y})`).join(" ")}: fully transparent ${$("unusedonly").checked ? "unused" : "overlay/unused"} pixels get alpha 1 and a counter as RGB (${n.nb} SHA block${n.nb > 1 ? "s" : ""} per candidate${n.fast ? ", fast kernel" : ", general kernel, slower"})${n.isPreferred || $("unusedonly").checked ? "" : ". Not the preferred pixels (55-57, 63)"}`
      : `<b>Not possible.</b> ${n.reason}`]);
    rows.push(["Watermark", n.wm.length ? `(${n.wm[0].join(",")})&ndash;(${n.wm.at(-1).join(",")}), unused and transparent` : ($("wm").checked ? "<b>no free unused pixels</b>, watermark skipped" : "off")]);
    rows.push(["Visible pixels", "never changed. Alpha 1 of 255 should be invisible in game, but that is not verified"]);
    $("info").innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
    return;
  }
  rows.push(["Fast mode", prep.eligible
    ? `Possible. The last ${prep.kvis >= MAX_K ? MAX_K : prep.kvis} pixels of the image are invisible.`
    : `<b>Not possible for this skin.</b> ${prep.reason}`]);
  if (prep.eligible) rows.push(["Search pixels", `${S.chosenK ? "" : "up to "}${k} (x ${W - k}&ndash;${W - 1}, y ${H - 1}): only their hidden RGB changes`
    + (S.chosenK ? "" : ". The final count is chosen when the search starts")]);
  rows.push(["Watermark", prep.wm.length ? `(${prep.wm[0].join(",")})&ndash;(${prep.wm.at(-1).join(",")}), unused and transparent` : ($("wm").checked ? "<b>no free unused pixels</b>, watermark skipped" : "off")]);
  rows.push(["Alpha / visible pixels", "never changed"]);
  if (S.mode === "both") {
    const n = S.nprep;
    rows.push(["Phase 1 (NameMC) knob", n.eligible ? `3 pixels ${n.knobs.map(([x, y]) => `(${x},${y})`).join(" ")} get alpha 1 (orange on the preview); phase 2 then only changes RGB under alpha 0 at the end of the image` : `<b>Not possible.</b> ${n.reason}`]);
    if (!S.prep.eligible || !n.eligible) rows.push(["Both mode", "<b>Not possible for this skin</b> (phase 2 needs fast mode, phase 1 needs three usable pixels)."]);
  }
  $("info").innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
}

function loadSkinBytes(bytes, name, keepHits = false, internal = false) {
  showErr("skinerr", "");
  let dec;
  try { dec = decodePng(bytes); } catch (e) { showErr("skinerr", e.message); S.prep = null; $("skinbox").hidden = $("targetsec").hidden = true; return; }
  S.orig = dec.rgba; S.name = name.replace(/\.png$/i, "");
  if (!internal) { S.userBytes = bytes; S.userName = name; S.userOrig = dec.rgba; }
  S.origId = (toHex(sha256(mojangPng(dec.rgba))).replace(/^0+/, "") || "0");
  S.origNm = namemcIdHex(dec.rgba);
  S.chosenK = 0;
  rebuildPrep();
  $("skinbox").hidden = false; $("targetsec").hidden = false; $("runsec").hidden = false;
  if (!keepHits) { S.hits = []; $("hits").innerHTML = ""; $("nohits").hidden = false; }
}

function rebuildPrep() {
  if (!S.orig) return;
  S.prep = prepareSkin(S.orig, { watermark: $("wm").checked });
  S.nprep = prepareNameMc(S.orig, { watermark: $("wm").checked, unusedOnly: $("unusedonly").checked });
  drawPreview(); renderInfo(); updateButtons();
  if (S.mode === "both" && !(S.nprep.eligible && S.prep.eligible)) setStatus("Both mode is not possible for this skin. " + (S.prep.eligible ? S.nprep.reason : S.prep.reason), "warn");
  else if (S.mode === "namemc") { if (!S.nprep.eligible) setStatus("NameMC mode is not possible for this skin. " + S.nprep.reason, "warn"); else setStatus("Ready."); }
  else if (!S.prep.eligible) setStatus("Fast mode is not possible for this skin, so it can't be searched here. " + S.prep.reason, "warn");
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
$("unusedonly").addEventListener("change", () => { if (!S.running) rebuildPrep(); });
function setMode(m) {
  S.mode = m;
  document.querySelector(`input[name=mode][value=${m}]`).checked = true;
  const nm = m === "namemc", both = m === "both";
  $("unusedwrap").hidden = !(nm || both);
  $("g2").hidden = !both; $("g1legend").hidden = !both; $("bothnote").hidden = !both;
  document.querySelectorAll(".grp").forEach((el) => el.classList.toggle("boxed", both));
  $("g1legend").textContent = "Phase 1: NameMC targets (16-digit id; a leading 0 is allowed)";
  $("phasebox").hidden = !both;
  const fx = nm || both;
  $("prefixes").placeholder = fx ? "beef, 0000" : "c0ffee, beef";
  $("prefixlabel").textContent = fx ? "Prefixes (hex, of the 16-digit id; may start with 0)" : "Prefixes (hex, no leading 0)";
  $("zeroslabel").textContent = fx ? "Minimum leading zeros (of the 16-digit id)" : "Minimum leading zeros";
  if (S.orig && !S.running) { drawPreview(); renderInfo(); updateButtons(); }
  updateExpect();
}
document.querySelectorAll("input[name=mode]").forEach((el) => el.addEventListener("change", () => { if (S.running) { setMode(S.mode); return; } setMode(el.value); if (S.orig) rebuildPrep(); }));

// ---------- targets ----------
function readTargets() { return readGroup(S.mode === "namemc", ["prefixes", "suffixes", "zeros"]); }
function readGroup(nm, ids) {
  const p = parseHexList($(ids[0]).value, "prefix", nm), s = parseHexList($(ids[1]).value, "suffix");
  const errors = [...p.errors, ...s.errors];
  const zeros = Math.max(0, Math.min(15, parseInt($(ids[2]).value || "0", 10) || 0));
  const targets = targetList({ prefixes: p.list, suffixes: s.list, zeros });
  if (targets.length > 32) errors.push("At most 32 targets at a time.");
  return { targets, errors };
}
const NM_IDS = ["prefixes", "suffixes", "zeros"], MJ_IDS = ["mprefixes", "msuffixes", "mzeros"];
const firstHitTries = (targets, nm) => targets.length ? 1 / targets.reduce((a, t) => a + 1 / expectedTries(t, nm), 0) : 0;
function updateExpect() {
  if (S.mode === "both") {
    const a = readGroup(true, NM_IDS), b = readGroup(false, MJ_IDS);
    showErr("targeterr", [...a.errors, ...b.errors].join(" "));
    const e1 = firstHitTries(a.targets, true), e2 = firstHitTries(b.targets, false);
    const r1 = S.rates.namemc || 2e9, r2 = S.rates.mojang || 2.5e9, assumed = !(S.rates.namemc && S.rates.mojang);
    const parts = [];
    if (e1) parts.push(`Phase 1 (NameMC) ~${fmt(e1)} tries, ~${fmtTime(e1 / r1)}`);
    if (e2) parts.push(`Phase 2 (Mojang) ~${fmt(e2)} tries, ~${fmtTime(e2 / r2)}`);
    $("expect").textContent = parts.length && !a.errors.length && !b.errors.length
      ? parts.join(". ") + (e1 && e2 ? `. Total ~${fmtTime(e1 / r1 + e2 / r2)}` : "") + (assumed ? " (rates assumed until measured on this GPU)." : ".") : "";
    return;
  }
  const { targets, errors } = readTargets();
  showErr("targeterr", errors.join(" "));
  $("expect").textContent = targets.length && !errors.length
    ? "Expected tries per hit: " + targets.map((t) => `${t.kind} ${t.text} ~${fmt(expectedTries(t, S.mode === "namemc"))}`).join(", ") + (S.lastRate ? ` (about ${targets.map((t) => fmtTime(expectedTries(t, S.mode === "namemc") / S.lastRate)).join(", ")} at your measured rate)` : "")
    : "";
}
[...NM_IDS, ...MJ_IDS].forEach((id) => $(id).addEventListener("input", updateExpect));

function updateButtons() {
  const ok = !!(S.mode === "namemc" ? S.nprep?.eligible : S.mode === "both" ? (S.nprep?.eligible && S.prep?.eligible) : S.prep?.eligible) && !S.running;
  $("start").disabled = !ok; $("bench").disabled = !ok || S.mode === "both"; $("stop").disabled = !S.running;
}

// ---------- hits ----------
function addNmHit(r, whys, intermediate = false) {
  const url = `https://namemc.com/skin/${r.id16}`, murl = `https://textures.minecraft.net/texture/${r.mojangId}`;
  const blobUrl = URL.createObjectURL(new Blob([r.png], { type: "image/png" }));
  const li = document.createElement("li");
  li.innerHTML = `<div class="kind">NameMC skin page${intermediate ? " (intermediate result of phase 1)" : ""}</div><a class="link" target="_blank" rel="noopener"></a><div class="why"></div><div class="kind">Mojang texture link of this PNG</div><a class="link m" target="_blank" rel="noopener"></a><div class="btns"><button type="button" class="copy">Copy link</button><a class="dl">Download PNG</a><button type="button" class="use">Use as skin for a Mojang search</button></div>`;
  const a = li.querySelector("a.link"); a.href = url; a.textContent = url;
  const m = li.querySelector("a.m"); m.href = murl; m.textContent = murl;
  li.querySelector(".why").textContent = `${whys.map((t) => `${t.kind} ${t.text}`).join(", ")} | verified | ${r.png.length} bytes.`;
  const dl = li.querySelector("a.dl"); dl.href = blobUrl; dl.download = `${r.id16}.png`;
  li.querySelector(".copy").addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(url); e.target.textContent = "Copied"; } catch { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy link"), 1500);
  });
  li.querySelector(".use").addEventListener("click", () => {
    if (S.running) return;
    S.userPicked = true;
    loadSkinBytes(r.png, `${r.id16}.png`, true);
    setMode("mojang"); rebuildPrep();
    setStatus("Loaded the NameMC hit as the current skin. A Mojang search keeps its NameMC id (it only changes RGB under alpha 0).");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  $("hits").prepend(li); $("nohits").hidden = true;
  S.hits.push({ mode: "namemc", id: r.id16, id16: r.id16, mojangId: r.mojangId, url, png: r.png, whys: whys.map(targetKey) });
}

function addHit(r, whys) {
  const url = `https://textures.minecraft.net/texture/${r.id}`;
  const blobUrl = URL.createObjectURL(new Blob([r.png], { type: "image/png" }));
  const li = document.createElement("li");
  li.innerHTML = `<a class="link" target="_blank" rel="noopener"></a><div class="why"></div><div class="btns"><button type="button" class="copy">Copy link</button><a class="dl">Download PNG</a></div>`;
  li.insertAdjacentHTML("afterbegin", `<div class="kind">Mojang texture link</div>`);
  const a = li.querySelector("a.link"); a.href = url; a.textContent = url;
  li.querySelector(".why").textContent = `${whys.map((t) => `${t.kind} ${t.text}`).join(", ")} | verified | ${r.png.length} bytes | NameMC id unchanged: ${r.nmId}`;
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
  if (S.mode === "namemc") return runNameMc(bench, null);
  if (S.mode === "both") return startBoth();
  return runMojang(bench, null);
}
async function runMojang(bench, opts) {
  let runT0 = 0;
  if ((!opts && S.running) || !S.prep?.eligible) return;
  let targets = [];
  if (opts) targets = opts.targets;
  else if (!bench) {
    const r = readTargets();
    showErr("targeterr", r.errors.join(" "));
    if (r.errors.length) return;
    if (!r.targets.length) { showErr("targeterr", "Enter at least one prefix, suffix or a leading-zeros minimum (or use the benchmark)."); return; }
    targets = r.targets;
  }
  showErr("runerr", "");
  if (!opts) { S.running = true; S.stop = false; S.userStop = false; updateButtons(); }
  S.bench = null; S.error = null;
  $("rate").textContent = "–"; $("tried").textContent = "0"; $("best").textContent = "0"; $("elapsed").textContent = "0 s";
  const allMode = opts ? !!opts.allMode : (!bench && document.querySelector("input[name=stop]:checked").value === "all");
  const maxHits = opts ? 1 : Math.max(1, parseInt($("stopn").value || "1", 10) || 1);
  const remaining = new Map(targets.map((t) => [targetKey(t), t]));
  const wantNow = () => [...remaining.values()];
  const rng = makeRng();
  let g = null, totalTried = 0, bestZeros = 0, t0 = performance.now(), hitCount = 0, dropped = 0, bad = 0;
  rate.samples = [];
  const queue = []; let draining = false;
  const emit = (r, w) => { opts?.collect?.push(r); if (!opts?.quiet) addHit(r, w); };
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
        emit(r, fresh);
        if (g) { const gt = gpuTargets(wantNow()); g.setTargets(gt.pats, gt.zeros); }
      } else emit(r, r.whys);
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
      const base = totalTried, tRun = performance.now(); runT0 = tRun;
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
          S.rates.mojang = S.lastRate;
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
    if (opts) opts.result = { tried: totalTried, secs: (performance.now() - (runT0 || t0)) / 1000 };
    if (bench && S.bench) {
      S.lastRate = S.bench.rate;
      $("rate").textContent = fmt(S.bench.rate) + "/s";
      setStatus(`Benchmark done: ${fmt(S.bench.rate)} candidates/s average over ${S.bench.secs.toFixed(1)} s (${fmt(S.bench.tried)} tried).`, "ok");
    } else if (!opts) setStatus(`Stopped. ${hitCount} verified hit${hitCount === 1 ? "" : "s"}${dropped ? `; ${dropped} further hit(s) skipped` : ""}.`);
  } catch (e) {
    console.error(e);
    showErr("runerr", String(e.message || e));
    setStatus("Search did not run.", "err");
    S.error = String(e.message || e);
  } finally {
    g?.destroy();
    if (!opts) { S.running = false; updateButtons(); updateExpect(); }
  }
}

$("start").addEventListener("click", () => startSearch(false));
$("bench").addEventListener("click", () => startSearch(true));
$("stop").addEventListener("click", () => { S.stop = true; S.userStop = true; });
S.ready = true;

// ---------- NameMC search ----------
async function runNameMc(bench, opts) {
  let runT0 = 0;
  const prep = S.nprep;
  if ((!opts && S.running) || !prep?.eligible) return;
  let targets = [];
  if (opts) targets = opts.targets;
  else if (!bench) {
    const r = readTargets();
    showErr("targeterr", r.errors.join(" "));
    if (r.errors.length) return;
    if (!r.targets.length) { showErr("targeterr", "Enter at least one prefix, suffix or a leading-zeros minimum (or use the benchmark)."); return; }
    targets = r.targets;
  }
  showErr("runerr", "");
  if (!opts) { S.running = true; S.stop = false; S.userStop = false; updateButtons(); }
  S.bench = null; S.error = null; S.gate = null;
  $("rate").textContent = "–"; $("tried").textContent = "0"; $("best").textContent = "0"; $("elapsed").textContent = "0 s";
  const allMode = opts ? !!opts.allMode : (!bench && document.querySelector("input[name=stop]:checked").value === "all");
  const maxHits = opts ? 1 : Math.max(1, parseInt($("stopn").value || "1", 10) || 1);
  const remaining = new Map(targets.map((t) => [targetKey(t), t]));
  const rng = makeRng();
  let g = null, hitCount = 0, dropped = 0, bad = 0, bestZeros = 0, totalTried = 0;
  const t0 = performance.now();
  rate.samples = [];
  const queue = []; let draining = false;
  const emit = (r, w) => { opts?.collect?.push(r); if (!opts?.quiet) addNmHit(r, w, !!opts?.intermediate); };
  const finishCheck = () => { if (bench) return; if (allMode ? remaining.size === 0 : hitCount >= maxHits) S.stop = true; };
  const drain = async () => {
    if (draining) return; draining = true;
    while (queue.length && !S.stop) {
      const c = queue.shift();
      const r = await verifyNmHit(prep, c, targets);
      if (!r.ok) { bad++; showErr("runerr", `A GPU hit failed CPU verification and was discarded (${r.error}). ${bad} so far.`); continue; }
      if (S.stop) break;
      if (allMode) {
        const fresh = r.whys.filter((t) => remaining.has(targetKey(t)));
        if (!fresh.length) { dropped++; continue; }
        fresh.forEach((t) => remaining.delete(targetKey(t)));
        emit(r, fresh);
        g?.setTargets(gpuNmTargets([...remaining.values()]));
      } else emit(r, r.whys);
      hitCount++; finishCheck();
    }
    draining = false;
  };
  try {
    setStatus("Initialising WebGPU...");
    if (!S.gpu) S.gpu = await initDevice();
    const device = S.gpu.device;
    device.lost.then((info) => { if (S.running) { S.stop = true; showErr("runerr", "The GPU device was lost: " + info.message); } });
    setStatus("CPU reference: building test vectors...");
    await new Promise((r) => setTimeout(r, 10));
    const tests = makeNmTests(prep, rng, 96);
    setStatus("Compiling GPU kernel and running the correctness gate...");
    g = new NmGpuJob(device, prep, tests, { wg: Number(new URLSearchParams(location.search).get("wg")) || 256 });
    await g.build();
    const gate = await g.gate();
    S.gate = { ...gate, selfTest: tests.length };
    if (!gate.ok) throw new Error(`GPU correctness gate FAILED: ${gate.bad} of ${gate.n} test vectors differ between GPU and CPU (${gate.detail}). The search was not started.`);
    g.setTargets(gpuNmTargets([...remaining.values()]));
    if (bench) setTimeout(() => (S.stop = true), 10000);
    setStatus(`${bench ? "Benchmark" : "Searching"} (NameMC): GPU reproduced ${gate.n}/${gate.n} CPU test vectors. ${prep.nb} SHA block${prep.nb > 1 ? "s" : ""} per candidate, ${prep.fast ? "fast" : "general"} kernel.`, "ok");
    const tRun = performance.now(); runT0 = tRun;
    await g.run(rng, {
      shouldStop: () => S.stop,
      onHits: (cs, nh) => {
        if (bench) return;
        for (const c of cs) if (queue.length < 64) queue.push(c); else dropped++;
        if (nh > cs.length) dropped += nh - cs.length;
        drain();
      },
      onProgress: ({ tried, bestZeros: bz }) => {
        totalTried = tried; bestZeros = Math.max(bestZeros, bz);
        S.lastRate = stats(totalTried, bestZeros, t0) || S.lastRate;
        S.rates.namemc = S.lastRate;
      },
    });
    const runSecs = (performance.now() - tRun) / 1000;
    if (bench) S.bench = { rate: totalTried / runSecs, tried: totalTried, secs: runSecs };
    g.destroy(); g = null;
    await drain();
    if (opts) opts.result = { tried: totalTried, secs: (performance.now() - (runT0 || t0)) / 1000 };
    if (bench && S.bench) {
      S.lastRate = S.bench.rate;
      $("rate").textContent = fmt(S.bench.rate) + "/s";
      setStatus(`Benchmark done (NameMC): ${fmt(S.bench.rate)} candidates/s average over ${S.bench.secs.toFixed(1)} s (${fmt(S.bench.tried)} tried).`, "ok");
    } else if (!opts) setStatus(`Stopped. ${hitCount} verified hit${hitCount === 1 ? "" : "s"}${dropped ? `; ${dropped} further hit(s) skipped` : ""}.`);
  } catch (e) {
    console.error(e);
    showErr("runerr", String(e.message || e));
    setStatus("Search did not run.", "err");
    S.error = String(e.message || e);
  } finally {
    g?.destroy();
    if (!opts) { S.running = false; updateButtons(); updateExpect(); }
  }
}
setMode("mojang");

// ---------- Both: NameMC first, then Mojang on that result ----------
function renderPhases() {
  $("phasebox").textContent = S.phases.map((p) => `${p.name}: ${p.state}${p.tried != null ? ` | ${fmt(p.tried)} tried in ${p.secs.toFixed(1)} s (${fmt(p.tried / Math.max(p.secs, 1e-3))}/s)` : ""}${p.est ? ` | expected ~${p.est}` : ""}`).join("\n");
}

async function startBoth() {
  if (S.running) return;
  const a = readGroup(true, NM_IDS), b = readGroup(false, MJ_IDS);
  showErr("targeterr", [...a.errors, ...b.errors].join(" "));
  if (a.errors.length || b.errors.length) return;
  if (!a.targets.length && !b.targets.length) { showErr("targeterr", "Enter NameMC targets, Mojang targets, or both. With only one group filled, only that phase runs."); return; }
  if (!(S.nprep?.eligible && S.prep?.eligible)) return;
  S.running = true; S.stop = false; S.userStop = false; S.error = null; S.gate = null; updateButtons();
  showErr("runerr", "");
  // a previous Both run leaves its hit loaded as the skin: start again from the skin the user chose
  if (S.userBytes && S.orig !== S.userOrig) loadSkinBytes(S.userBytes, S.userName, true, true);
  const origRgba = S.orig, nmPrep = S.nprep;
  const allMode = document.querySelector("input[name=stop]:checked").value === "all";
  const e1 = firstHitTries(a.targets, true), e2 = firstHitTries(b.targets, false);
  const r1x = S.rates.namemc || 2e9, r2x = S.rates.mojang || 2.5e9;
  const total = (a.targets.length ? 1 : 0) + (b.targets.length ? 1 : 0);
  S.phases = [];
  const phase = (name, est) => { const p = { name, state: "waiting", est }; S.phases.push(p); return p; };
  const p1 = a.targets.length ? phase("Phase 1 (NameMC)", `${fmt(e1)} tries, ${fmtTime(e1 / r1x)}`) : null;
  const p2 = b.targets.length ? phase(`Phase ${total} (Mojang)`, `${fmt(e2)} tries, ${fmtTime(e2 / r2x)}`) : null;
  if (p1 && p2) S.phases.push({ name: "Total", state: `~${fmtTime(e1 / r1x + e2 / r2x)} (sum of the phases)` });
  renderPhases();
  let r1 = null, r2 = null, ok = true;
  try {
    if (p1) {
      S.phaseLabel = `Phase 1/${total} (NameMC): `; p1.state = "running"; renderPhases();
      const opts = { targets: a.targets, allMode, collect: [], quiet: false, intermediate: !!p2 };
      await runNameMc(false, opts);
      if (opts.result) Object.assign(p1, opts.result);
      if (S.error || !opts.collect.length) { ok = false; p1.state = S.error ? "failed" : "stopped"; }
      else { r1 = opts.collect[0]; p1.state = `done, ${opts.collect.length} hit(s), base ${r1.id16}`; }
      renderPhases();
    }
    if (ok && p2) {
      if (S.userStop) { ok = false; }
      else {
        S.stop = false;
        if (r1) {
          loadSkinBytes(r1.png, `${r1.id16}.png`, true, true);   // the NameMC hit becomes the base skin
          if (!S.prep.eligible) {
            ok = false; p2.state = "impossible";
            const msg = `Phase 2 cannot run: fast mode is not possible on the NameMC result. ${S.prep.reason}`;
            showErr("runerr", msg); S.error = msg; setStatus(msg, "err");
          }
        }
        if (ok) {
          S.phaseLabel = `Phase ${total}/${total} (Mojang): `; p2.state = "running"; renderPhases();
          const opts = { targets: b.targets, allMode: false, collect: [], quiet: true };
          await runMojang(false, opts);
          if (opts.result) Object.assign(p2, opts.result);
          if (S.error || !opts.collect.length) { ok = false; p2.state = S.error ? "failed" : "stopped"; }
          else { r2 = opts.collect[0]; p2.state = "done"; }
          renderPhases();
        }
      }
    } else if (p2) p2.state = "not run";
    S.phaseLabel = "";
    if (ok && r2) {
      if (r1) {
        const v = await verifyBoth(origRgba, nmPrep, r1, r2, a.targets, b.targets);
        if (!v.ok) { showErr("runerr", `The combined result failed CPU verification and was discarded (${v.error}).`); S.error = v.error; setStatus("Both: verification failed.", "err"); }
        else {
          addBothHit(r1, r2, v);
          setStatus(`Done. One PNG with NameMC ${r1.id16} and Mojang ${r2.id}.`, "ok");
        }
      } else { addHit(r2, r2.whys); setStatus("Done (Mojang phase only).", "ok"); }
    } else if (ok && r1 && !p2) setStatus("Done (NameMC phase only).", "ok");
    else if (!S.error) setStatus("Stopped.");
  } catch (e) {
    console.error(e);
    showErr("runerr", String(e.message || e)); S.error = String(e.message || e);
  } finally {
    S.phaseLabel = ""; renderPhases(); S.running = false; updateButtons(); updateExpect();
  }
}

function addBothHit(r1, r2, v) {
  const nurl = `https://namemc.com/skin/${r1.id16}`, murl = `https://textures.minecraft.net/texture/${r2.id}`;
  const blobUrl = URL.createObjectURL(new Blob([r2.png], { type: "image/png" }));
  const li = document.createElement("li");
  li.innerHTML = `<div class="kind">Final result: NameMC skin page + Mojang texture link in one PNG</div><a class="link n" target="_blank" rel="noopener"></a><a class="link m" target="_blank" rel="noopener"></a><div class="why"></div><div class="btns"><button type="button" class="copy">Copy both links</button><a class="dl">Download PNG</a></div>`;
  const an = li.querySelector("a.n"); an.href = nurl; an.textContent = nurl;
  const am = li.querySelector("a.m"); am.href = murl; am.textContent = murl;
  li.querySelector(".why").textContent = `NameMC ${[...new Set(v.nmWhys.map((t) => `${t.kind} ${t.text}`))].join(", ")} | Mojang ${[...new Set(r2.whys.map((t) => `${t.kind} ${t.text}`))].join(", ")} | verified against your original | ${r2.png.length} bytes.`;
  const dl = li.querySelector("a.dl"); dl.href = blobUrl; dl.download = `${r2.id}.png`;
  li.querySelector(".copy").addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(`${nurl}\n${murl}`); e.target.textContent = "Copied"; } catch { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy both links"), 1500);
  });
  $("hits").prepend(li); $("nohits").hidden = true;
  S.hits.push({ mode: "both", id: r2.id, id16: r1.id16, mojangId: r2.id, url: nurl, murl, png: r2.png, whys: r2.whys.map(targetKey) });
}
