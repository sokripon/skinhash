// Import a skin by Minecraft username, UUID or texture id/URL.
// Mojang's own APIs send no CORS headers, so players are looked up through playerdb.co (fallback
// api.ashcon.app). Only the lookup is third-party: the skin is fetched from textures.minecraft.net and its
// SHA-256 must equal the texture id in its URL, so a wrong or tampered answer cannot give a different skin.
import { sha256, toHex } from "./hash.js";

const TEX = "https://textures.minecraft.net/texture/";
const UUID_RE = /^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$/i;

export class ImportError extends Error {}

// { kind: "texture", id } | { kind: "player", q } ; texture = URL or 33-64 hex digits (shorter hex is a name)
export function classify(input) {
  const s = String(input || "").trim();
  if (!s) throw new ImportError("Enter a username or UUID.");
  const m = s.match(/textures\.minecraft\.net\/texture\/([0-9a-f]{1,64})\/?$/i);
  if (m) return { kind: "texture", id: m[1].toLowerCase() };
  if (/^[0-9a-f]{33,64}$/i.test(s)) return { kind: "texture", id: s.toLowerCase() };
  if (UUID_RE.test(s)) return { kind: "player", q: s.replace(/-/g, "").toLowerCase() };
  if (/^[A-Za-z0-9_]{3,16}$/.test(s)) return { kind: "player", q: s };
  throw new ImportError("Not a username (3-16 letters, digits, _), UUID or texture id.");
}

const textureId = (url) => (String(url || "").match(/^https?:\/\/textures\.minecraft\.net\/texture\/([0-9a-f]{1,64})\/?$/i) || [])[1]?.toLowerCase();

async function getJson(url, fetchFn) {
  let r;
  try { r = await fetchFn(url); } catch { return { net: true }; }
  if (r.status === 429) return { limited: true };
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  return { status: r.status, body };
}

// -> { name, uuid, tid, slim }
async function lookupPlayer(q, fetchFn) {
  const a = await getJson(`https://playerdb.co/api/player/minecraft/${encodeURIComponent(q)}`, fetchFn);
  if (a.body?.code === "player.found" && a.body.data?.player) {
    const p = a.body.data.player;
    const tid = textureId(p.skin_texture);
    let slim = false;
    try { slim = JSON.parse(atob(p.properties.find((x) => x.name === "textures").value)).textures.SKIN.metadata?.model === "slim"; } catch { /* classic */ }
    return { name: p.username, uuid: p.raw_id, tid, slim };
  }
  if (a.status === 400 || a.status === 404) throw new ImportError(`No Minecraft player "${q}" found.`);
  // playerdb unreachable, rate-limited or broken: fallback
  const b = await getJson(`https://api.ashcon.app/mojang/v2/user/${encodeURIComponent(q)}`, fetchFn);
  if (b.status === 404 || b.status === 400) throw new ImportError(`No Minecraft player "${q}" found.`);
  if (b.body?.username) {
    const sk = b.body.textures?.skin;
    return { name: b.body.username, uuid: String(b.body.uuid || "").replace(/-/g, ""), tid: textureId(sk?.url), slim: !!b.body.textures?.slim };
  }
  if (a.limited || b.limited) throw new ImportError("Rate-limited, try again in a minute.");
  throw new ImportError("Could not reach the player lookup (playerdb.co). Check your connection.");
}

export async function fetchTexture(tid, fetchFn = fetch) {
  let r;
  try { r = await fetchFn(TEX + tid); } catch { throw new ImportError("Could not reach textures.minecraft.net."); }
  if (r.status === 404) throw new ImportError("Texture not found.");
  if (r.status === 429) throw new ImportError("Rate-limited, try again in a minute.");
  if (!r.ok) throw new ImportError(`textures.minecraft.net answered HTTP ${r.status}.`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  if (toHex(sha256(bytes)).replace(/^0+/, "") !== tid.replace(/^0+/, "")) throw new ImportError("The downloaded skin does not match its texture id; discarded.");
  return bytes;
}

// 64x32 legacy check straight from the IHDR (the PNG decoder only says "must be 64x64")
function checkSize(bytes, who) {
  if (bytes.length < 24) return;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const w = dv.getUint32(16), h = dv.getUint32(20);
  if (w === 64 && h === 32) throw new ImportError(`${who} has a legacy 64x32 skin; only 64x64 skins work.`);
}

// input -> { bytes, name, tid, uuid?, slim? }
export async function resolveSkin(input, fetchFn = fetch) {
  const c = classify(input);
  if (c.kind === "texture") {
    const bytes = await fetchTexture(c.id, fetchFn);
    checkSize(bytes, "This texture");
    return { bytes, name: c.id.slice(0, 16), tid: c.id };
  }
  const p = await lookupPlayer(c.q, fetchFn);
  if (!p.tid) throw new ImportError(`${p.name} has no custom skin.`);
  const bytes = await fetchTexture(p.tid, fetchFn);
  checkSize(bytes, p.name);
  return { bytes, name: p.name, tid: p.tid, uuid: p.uuid, slim: p.slim };
}
