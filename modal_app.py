"""Run a skinhash search on a cloud GPU at modal.com. Instructions, GPU choice and prices: docs/modal.md

Setup (once):  pip install modal  &&  modal setup
Run:           modal run --detach modal_app.py [--gpu L40S] [--hours 23] SKINHASH ARGS...
  e.g.         modal run --detach modal_app.py --gpu L40S --hours 6 namemc skins/me.png -m @keep/records_namemc.txt -o found/nm_me
Results:       modal run modal_app.py download found/nm_me        (copies the cloud folder into found/nm_me)
Benchmark:     modal run modal_app.py --gpu RTX-PRO-6000 bench --engine gpu   (10 s, to compare GPU types)

SKINHASH ARGS are the normal skinhash.py arguments. Local files in them (the skin, @pattern files and the
@word lists those name) are uploaded with the run. -o DIR becomes the folder DIR's last part on the
"skinhash-found" Modal volume; if the local DIR has a records.json and the cloud folder has none, it is
uploaded first, so a cloud run continues the records of a local one.
--detach keeps the search running when this terminal closes. It stops after --hours (Modal's limit is 24 h);
records are saved as they are found, so start it again to go on.
GPUs: T4, L4, A10, L40S, RTX-PRO-6000, A100, H100, H200, B200 (which to pick: docs/modal.md).
"""
import os
import re
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["*.c", "*.cu", "*.cuh", "*.h", "*.py", "build.sh"]

def keep_in_image(p):
    """The sources and the example skin (bench's default); not your skins, results or Windows builds."""
    s = p.as_posix()
    if s in ("skins", "skins/steve.png"):
        return True
    return len(p.parts) == 1 and p.name != "modal_app.py" and any(p.match(x) for x in SOURCES)


app = modal.App("skinhash")
volume = modal.Volume.from_name("skinhash-found", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("gcc", "zlib1g-dev")
    .pip_install("pillow")
    .add_local_dir(HERE, "/app", copy=True, ignore=lambda p: not keep_in_image(p))
    # x86-64-v3 (AVX2): runs on every Modal CPU; native would tie the build to the image builder's CPU
    .run_commands("cd /app && MARCH=x86-64-v3 sh build.sh")
)


@app.function(image=image, gpu="L40S", cpu=4, volumes={"/found": volume}, timeout=24 * 3600)
def search(args, files, hours):
    """Runs skinhash.py ARGS in /app with FILES ({path under /in: bytes}) and stops it after HOURS."""
    import signal
    import subprocess
    import threading
    import time

    for rel, data in files.items():
        p = os.path.join("/in", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    proc = subprocess.Popen([sys.executable, "-u", "skinhash.py", *args], cwd="/app", start_new_session=True,
                            env={**os.environ, "SKINHASH_FOUND": "/found"})   # no -o: still on the volume

    def commit_loop():          # saved results reach the volume even if the container is stopped abruptly
        while proc.poll() is None:
            time.sleep(60)
            try:
                volume.commit()
            except Exception as e:
                print(f"volume commit failed: {e}", flush=True)

    threading.Thread(target=commit_loop, daemon=True).start()
    try:
        proc.wait(timeout=hours * 3600)
    except subprocess.TimeoutExpired:
        print(f"--hours {hours} reached, stopping (like Ctrl+C)", flush=True)
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait()
    volume.commit()
    return proc.returncode


def upload_file(path, files, seen):
    """Adds a local file (and, for a pattern file, the @files it names) to FILES; returns its path in the container."""
    path = os.path.abspath(path)
    rel = "f%d/%s" % (len(seen), os.path.basename(path))
    if path in seen:
        return seen[path]
    seen[path] = "/in/" + rel
    with open(path, "rb") as f:
        data = f.read()
    files[rel] = data
    # word lists and nested pattern files are named relative to the pattern file: keep them next to it
    for name in set(re.findall(rb"@([\w./\\-]+)", data)):
        sub = os.path.join(os.path.dirname(path), name.decode())
        if os.path.isfile(sub) and sub not in seen:
            subrel = os.path.join(os.path.dirname(rel), name.decode()).replace("\\", "/")
            with open(sub, "rb") as f:
                files[subrel] = f.read()
            seen[sub] = "/in/" + subrel
    return seen[path]


def map_args(argv):
    """Local paths in the skinhash arguments -> uploaded copies; -o DIR -> /found/<name>. Returns (args, files, out)."""
    files, seen, out, args = {}, {}, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--out") and i + 1 < len(argv):
            out = argv[i + 1]
            args += [a, "/found/" + os.path.basename(os.path.normpath(out))]
            i += 2
            continue
        if "@" in a:            # -m values: comma lists may hold @FILE items
            parts = a.split(",")
            a = ",".join("@" + upload_file(p[1:], files, seen) if p.startswith("@") and os.path.isfile(p[1:]) else p
                         for p in parts)
        elif os.path.isfile(a):
            a = upload_file(a, files, seen)
        args.append(a)
        i += 1
    return args, files, out


@app.local_entrypoint()
def main(*argv):
    argv, gpu, hours = list(argv), "L40S", 23.0
    while argv and argv[0] in ("--gpu", "--hours"):
        if len(argv) < 2:
            sys.exit(f"{argv[0]} needs a value")
        if argv[0] == "--gpu":
            gpu = argv[1]
        else:
            hours = float(argv[1])
        argv = argv[2:]
    if not argv:
        sys.exit(__doc__)
    if argv[0] == "download":
        if len(argv) != 2:
            sys.exit("usage: modal run modal_app.py download DIR")
        return download(argv[1])
    if not 0 < hours <= 23.9:
        sys.exit("--hours must be more than 0 and at most 23.9 (Modal stops a run after 24 h)")
    args, files, out = map_args(argv)
    if out and os.path.isfile(os.path.join(out, "records.json")):
        name = os.path.basename(os.path.normpath(out))
        if not cloud_has(f"/{name}/records.json"):      # not in the cloud yet: continue from the local records
            with volume.batch_upload() as batch:
                batch.put_file(os.path.join(out, "records.json"), f"/{name}/records.json")
            print(f"uploaded {out}/records.json to continue its records")
    print(f"gpu={gpu} hours={hours} skinhash.py {' '.join(args)}")
    rc = search.with_options(gpu=gpu).remote(args, files, hours)
    print(f"finished (exit {rc})." + (f" Get the results: modal run modal_app.py download {out}" if out else ""))


def cloud_has(path):
    try:
        return any(e.path.rstrip("/").endswith(os.path.basename(path)) for e in volume.listdir(os.path.dirname(path)))
    except Exception:           # the folder does not exist yet
        return False


def merge_records(local, cloud):
    """records.json: per line the better of the local and the cloud value (the cloud one on a tie)."""
    import patterns as pt
    for group, lines in cloud.items():
        mine = local.setdefault(group, {})
        width = 64 if group == "mojang" else 16
        for text, e in lines.items():
            old = mine.get(text)
            try:
                better = old is None or pt.parse_line(text, width).beats(e["value"], old["value"]) or e["value"] == old["value"]
            except Exception:
                better = old is None
            if better:
                mine[text] = e
    return local


def download(out):
    """Copies the cloud folder named like OUT's last part into OUT. records.json keeps the best value of each line
    from both sides and hits.csv gets the cloud rows it does not have; other files are copied."""
    import io
    import json
    name = os.path.basename(os.path.normpath(out))
    n = 0
    for entry in volume.iterdir(f"/{name}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE or "/.pending/" in "/" + entry.path:
            continue
        rel = os.path.relpath(entry.path, name)
        dest = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        buf = io.BytesIO()
        volume.read_file_into_fileobj(entry.path, buf)
        data = buf.getvalue()
        if rel == "records.json" and os.path.isfile(dest):
            with open(dest, encoding="utf-8") as f:
                data = json.dumps(merge_records(json.load(f), json.loads(data)), indent=1).encode()
        elif rel == "hits.csv" and os.path.isfile(dest):
            with open(dest, "rb") as f:
                have = f.read()
            known = set(have.splitlines())
            new = [l for l in data.splitlines()[1:] if l not in known]
            data = have + (b"" if have.endswith(b"\n") or not new else b"\n") + b"".join(l + b"\n" for l in new)
        with open(dest, "wb") as f:
            f.write(data)
        n += 1
    print(f"{n} files -> {out}")
