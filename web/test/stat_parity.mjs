// Reads JSON [[width, text, [hexid, ...]], ...] on stdin, prints JSON [[{ok, value}, ...], ...]: the stat line / pattern
// `text` evaluated on each id with web/js/patterns.js (for test_patterns.py). A plain hex pattern gives {ok: match, value: null}.
import { parseLine, StatLine } from "../js/patterns.js";
let data = "";
process.stdin.on("data", (d) => (data += d));
process.stdin.on("end", () => {
  const out = [];
  for (const [w, text, ids] of JSON.parse(data)) {
    let p;
    try { p = parseLine(text, w); } catch (e) { out.push({ error: e.message }); continue; }
    out.push(ids.map((h) => (p instanceof StatLine ? p.evaluate(h) : { ok: p.match(h), value: null })));
  }
  console.log(JSON.stringify(out));
});
