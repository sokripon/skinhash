"""Reference: zlib level 4 / wbits 15 / memLevel 8 / default strategy over N raw cases.
usage: py zlib_ref.py CASES.bin RAW_LEN OUT.bin   (out: u32le length + bytes per case)"""
import struct, sys, zlib
src, raw_len, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = open(src, "rb").read()
out = bytearray()
for i in range(0, len(data), raw_len):
    c = zlib.compressobj(4, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    z = c.compress(data[i:i + raw_len]) + c.flush()
    out += struct.pack("<I", len(z)) + z
open(dst, "wb").write(out)
print("zlib", zlib.ZLIB_RUNTIME_VERSION, "cases", len(data) // raw_len)
