"""Independent NameMC id (hashlib only). usage: py namemc_ref.py CASES.bin OUT.txt
CASES.bin = N x 16384 bytes of RGBA; OUT.txt gets one 16-hex id per line."""
import hashlib, struct, sys
data = open(sys.argv[1], "rb").read()
out = []
for i in range(0, len(data), 16384):
    px = data[i:i + 16384]
    buf = bytearray(struct.pack(">II", 64, 64))
    for j in range(0, 16384, 4):
        buf += px[j:j + 4] if px[j + 3] else b"\0\0\0\0"
    out.append(hashlib.sha256(buf).hexdigest()[:16])
open(sys.argv[2], "w").write("\n".join(out))
