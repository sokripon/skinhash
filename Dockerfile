# skinhash in a container: no CUDA toolkit or compiler needed on the host, only Docker and (for --engine gpu)
# an NVIDIA driver 580+. Build:  docker build -t skinhash .
#   docker build --build-arg CUDA_ARCHS=120 -t skinhash .     only your GPU's architecture (much faster build)
# Run (from the repo folder, so skins/, keep/ and found/ are visible in the container):
#   docker run --rm -it --gpus all -v "${PWD}:/work" skinhash namemc skins/me.png -o found/nm_me
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY *.c *.cu *.cuh *.h build.sh ./
# sm_80 A100, sm_86 RTX 30/A10, sm_89 RTX 40/L4/L40S, sm_90 H100/H200, sm_100 B200, sm_120 RTX 50
ARG CUDA_ARCHS="80 86 89 90 100 120"
# x86-64-v3 (AVX2, any CPU since ~2015): the image also runs on machines other than the one that built it
ARG MARCH=x86-64-v3
RUN CUDA_ARCHS="$CUDA_ARCHS" MARCH="$MARCH" sh build.sh

# The finders link the CUDA runtime statically, so the small base image (driver access only) is enough.
FROM nvidia/cuda:13.0.0-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pil zlib1g \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/fastfinder /app/hashfinder /app/namemcfinder /app/gpufinder /app/namemcgpu ./
COPY *.py ./
COPY skins/steve.png skins/
ENV SKINHASH_FOUND=/work/found
WORKDIR /work
ENTRYPOINT ["python3", "-u", "/app/skinhash.py"]
CMD ["--help"]
