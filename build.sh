#!/bin/sh
# Linux build (the Windows one is build.ps1): hashfinder, fastfinder, namemcfinder (gcc + zlib) and gpufinder, namemcgpu (nvcc).
# Usage: sh build.sh [--no-gpu]
#   CUDA_ARCHS="90 120" sh build.sh    GPU architectures to build for (default: all the Modal GPUs and RTX 30/40/50)
#   MARCH=x86-64-v3 sh build.sh        CPU target of the gcc builds (default: native)
set -e
cd "$(dirname "$0")"
MARCH="${MARCH:-native}"

echo "== hashfinder / fastfinder / namemcfinder (gcc, -march=$MARCH)"
for f in fastfinder hashfinder namemcfinder; do
    gcc -O3 -march="$MARCH" -Wall -Wextra -o "$f" "$f.c" -lz -lpthread
done

[ "$1" = "--no-gpu" ] && exit 0
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
[ -x "$NVCC" ] || { echo "nvcc not found, skipping gpufinder and namemcgpu"; exit 0; }
# sm_75 T4/RTX 20, sm_80 A100, sm_86 A10/RTX 30, sm_89 L4/L40S/RTX 40, sm_90 H100/H200, sm_100 B200, sm_120 RTX 50
ARCHS="${CUDA_ARCHS:-75 80 86 89 90 100 120}"
GENCODE=""
for a in $ARCHS; do GENCODE="$GENCODE -gencode arch=compute_$a,code=sm_$a"; done
echo "== gpufinder / namemcgpu ($NVCC, sm: $ARCHS)"
# The two finders compile at the same time as separate processes. Not nvcc -t 0 (parallel architectures inside
# one nvcc): it sometimes fails with "nvlink fatal: Could not read file .../tmpxft_..._dlink.reg.c".
"$NVCC" -O3 $GENCODE -o namemcgpu namemcgpu.cu & NM=$!
"$NVCC" -O3 $GENCODE -o gpufinder gpufinder.cu & GF=$!
wait $NM || { echo "namemcgpu build failed"; exit 1; }
wait $GF || { echo "gpufinder build failed"; exit 1; }
echo "built: fastfinder hashfinder namemcfinder gpufinder namemcgpu"
