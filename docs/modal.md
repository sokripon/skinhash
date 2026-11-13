# Cloud GPU runs on Modal

`modal_app.py` runs a skinhash search on a rented GPU at [modal.com](https://modal.com). You pay per second
of GPU time, and the free Starter plan includes $30 of credit per month. You don't need a GPU or CUDA locally;
the finders are built in Modal's cloud with [`build.sh`](../build.sh).

## Setup

```
uv tool install modal
modal setup          # opens the browser to log in; or: modal token set --token-id ... --token-secret ...
```

`uv tool install` puts the `modal` command on your PATH in its own environment; `uv tool upgrade modal`
updates it. Without installing: `uvx modal run ...`.

## Usage

```
modal run modal_app.py --gpu L40S bench --engine gpu      # 10 s benchmark
modal run --detach modal_app.py --gpu L40S --hours 6 namemc skins/me.png -m @keep/records_namemc.txt -o found/nm_me
modal run modal_app.py status found/nm_me                 # records and hits so far, nothing downloaded
modal run modal_app.py download found/nm_me               # copy the results into found/nm_me
```

- `--gpu TYPE` (default `L40S`) and `--hours N` (default 23) come first. Everything after them is passed
  to `skinhash.py` unchanged, so every command and option of the [main README](../README.md#usage) works.
- Local files in the arguments are uploaded with the run: the skin, `@pattern` files, and the `@word` lists
  that a pattern file names.
- `--detach` keeps the search running after the terminal closes or the PC goes to sleep. Without it, closing
  the terminal stops the run.
- The run stops after `--hours`, the same way Ctrl+C stops it. Modal allows at most 24 h per run. Records
  are saved as they are found, so starting the same command again continues where it stopped.
- To stop a detached run early: `modal app stop skinhash`, or use the app page on modal.com. Records found so
  far are kept.

## Results and records

- `-o DIR` names a folder on the `skinhash-found` Modal volume, using only DIR's last part: `-o found/nm_me`
  becomes `nm_me`. Without `-o`, the folder is named after the skin, as in a local run.
- Continuing a local search: if the local DIR has a `records.json` and the cloud folder has none yet, it is
  uploaded first, so the cloud run starts from your local best values.
- `status DIR` reads the cloud `records.json` and `hits.csv` while a run is going or after it. It shows every
  record with its id and note, and marks with `+` the ones that are better than your local copy.
- `modal app list` shows whether a run is still going; `modal app logs skinhash` follows its output.
- `download DIR` copies the cloud folder into DIR. It merges instead of overwriting: `records.json` keeps
  the better value of each record line from both sides, and `hits.csv` gets the rows it is missing. Hit skins
  and `records/` files are copied.
- You can browse the volume with `modal volume ls skinhash-found nm_me`.

## Which GPU

Measured with `bench --engine gpu` on the example skin in November 2026. Prices are from
[modal.com/pricing](https://modal.com/pricing) and cover the GPU only; the 4 CPU cores of a run add about
$0.19/h. Speed per dollar is in trillions of candidates per dollar.

| GPU | $/h | Mojang | NameMC | Mojang T/$ | NameMC T/$ |
|---|---|---|---|---|---|
| **L40S** | 1.95 | 17.3 G/s | 11.2 G/s | **31.9** | **20.6** |
| RTX PRO 6000 | 3.03 | **21.7 G/s** | **14.4 G/s** | 25.7 | 17.0 |
| H100 | 3.95 | 13.5 G/s | 8.9 G/s | 12.3 | 8.1 |
| B200 | 6.25 | 14.8 G/s | 10.1 G/s | 8.5 | 5.8 |
| L4 | 0.80 | 4.7 G/s | 3.1 G/s | 21.1 | 13.8 |
| T4 | 0.59 | 2.6 G/s | 1.7 G/s | 15.7 | 10.1 |
| RTX 5060 Ti (local, for comparison) | | 5.1 G/s | 3.3 G/s | | |

- **L40S** gives the most per dollar. $30 of credit buys about 14 hours.
- **RTX PRO 6000** (GPU name `RTX-PRO-6000`) is the fastest single GPU.
- **H100 and B200** are AI cards: they run integer code on only half their cores and at lower clocks, so
  they are slow for SHA-256. Skip them.
- **L4 and T4** are cheap per hour but slower than an RTX 5060 Ti, and the fixed $0.19/h for the CPU
  cores weighs more on them (with it: L4 17.1, T4 11.9 T/$ on Mojang).
- Also supported but not measured: A10, A100 and H200.
- Several runs in parallel add up. Two L40S give 34.6 G/s on Mojang for $3.90/h. Give each run its own `-o`
  folder; each one then keeps its own `records.json`.

## How it works

- The image is `nvidia/cuda:13.0.0-devel` with Python, gcc and zlib. It holds only the sources and
  `skins/steve.png`, never your skins, `keep/` or `found/`.
- `build.sh` compiles the GPU finders for every supported architecture. The first run builds the image,
  which takes about 10 minutes; later runs reuse it until a source file changes.
- The container commits the volume every minute and when the search ends, so results survive even if
  Modal stops the container abruptly.
- The image only exists in your Modal account. Anyone else who runs `modal_app.py` builds their own.
