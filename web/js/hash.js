// SHA-256 (with midstate support), CRC-32 (zlib semantics). Pure JS, no DOM.
const K256 = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
export { K256 };
export const SHA_INIT = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];

const rotr = (x, n) => (x >>> n) | (x << (32 - n));
const Wk = new Uint32Array(64);

/** Process nblocks 64-byte blocks of p starting at byte offset off; s is a Uint32Array(8) state. */
export function sha256Blocks(s, p, off, nblocks) {
  for (let blk = 0; blk < nblocks; blk++, off += 64) {
    for (let i = 0; i < 16; i++) Wk[i] = (p[off + 4*i] << 24 | p[off + 4*i + 1] << 16 | p[off + 4*i + 2] << 8 | p[off + 4*i + 3]) >>> 0;
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(Wk[i-15], 7) ^ rotr(Wk[i-15], 18) ^ (Wk[i-15] >>> 3);
      const s1 = rotr(Wk[i-2], 17) ^ rotr(Wk[i-2], 19) ^ (Wk[i-2] >>> 10);
      Wk[i] = (Wk[i-16] + s0 + Wk[i-7] + s1) >>> 0;
    }
    let a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + Wk[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    s[0] = (s[0] + a) >>> 0; s[1] = (s[1] + b) >>> 0; s[2] = (s[2] + c) >>> 0; s[3] = (s[3] + d) >>> 0;
    s[4] = (s[4] + e) >>> 0; s[5] = (s[5] + f) >>> 0; s[6] = (s[6] + g) >>> 0; s[7] = (s[7] + h) >>> 0;
  }
}

export function stateToDigest(s) {
  const d = new Uint8Array(32);
  for (let i = 0; i < 8; i++) { d[4*i] = s[i] >>> 24; d[4*i+1] = s[i] >>> 16; d[4*i+2] = s[i] >>> 8; d[4*i+3] = s[i]; }
  return d;
}

export function sha256(msg) {
  const s = Uint32Array.from(SHA_INIT);
  const full = msg.length >> 6;
  sha256Blocks(s, msg, 0, full);
  const rem = msg.length - full * 64;
  const tl = rem < 56 ? 64 : 128;
  const tail = new Uint8Array(tl);
  tail.set(msg.subarray(full * 64));
  tail[rem] = 0x80;
  const bits = msg.length * 8;
  const dv = new DataView(tail.buffer);
  dv.setUint32(tl - 8, Math.floor(bits / 4294967296));
  dv.setUint32(tl - 4, bits >>> 0);
  sha256Blocks(s, tail, 0, tl / 64);
  return stateToDigest(s);
}

export const toHex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, "0")).join("");

// CRC-32 (zlib semantics: crc32(0, data) is the standard CRC; chain by passing the previous result)
export const CRC_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
  CRC_TABLE[i] = c >>> 0;
}
export function crc32(crc, buf, start = 0, end = buf.length) {
  let c = ~crc >>> 0;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return ~c >>> 0;
}
/** Raw CRC register (no pre/post inversion), as the GPU job needs it. */
export function crcRegister(c, buf, start, end) {
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return c >>> 0;
}
