# Deviations from the registered DAY40 target-card cell

Registered: DAY40.md 1.2 (DAY34.md 1.3 to 1.8), run by `research/spill-b-20260919/pro-single-day40/chain.sh`
(`registered-chain.sh.txt`) on drivers `50e8cb345` and binary S40 `b46ae200e`. Arms, orders, N, burst, workload,
prompts, server environment and readers are unchanged. Every difference is listed here.

## Hardware and host

- H1. Card: one RTX PRO 6000 Blackwell Max-Q Workstation Edition on <box> (limits 250/250/280/280 W on indices 0 to 3),
  not the rented Workstation Edition at 600 W. This is the point of the cell.
- H2. Neighbours: the other three <box> cards train during the cell, and the card itself runs another session's eval
  batches between the two parts. The registered box ran nothing else. AGENTS.md lets a neighbour card take only
  short untimed pass/fail cells; this cell is untimed pass/fail but runs about 6 to 7 h in two parts. Host load, PCIe
  and heat from training slow the cell without entering any term.
- H3. Host: Ubuntu 26.04.1, Python 3.14.4 (the box ran 3.12), driver 615.71.09 with CUDA 13.4 user-mode libraries.
  The binary loads `libcudart.so.13` 13.4.92 and `libcublas(Lt).so.13` 13.8.0.4 from `/usr/local/cuda-13.4` (recorded
  per order in `ldd-O*.txt`). The box's runtime libraries are not recorded in its receipt.

## Binary

- B1. Built on the laptop, not on the box: same source (S40, clean tree, no patches), same command plus `-j2`, nvcc
  13.1 pinned through `MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc` (the box auto-detected 13.1), `MEMRA_CUDA_ARCH=120a`
  set explicitly (the box auto-detected 120a), Rust 1.97.1 from S40's `rust-toolchain.toml`. sha256
  `7da1db36...e7cc45`, not the box's `c3387e1b...b51d20a0` (build paths and host differ). Log: `bins/build-s40.log`.

## Checkout and provenance (no git on <box>)

- G1. chain.sh:14-16 fast-forwarded `/root/wt-b` from `lane/spill-b-20260919`; that branch is gone from origin. The kit
  carries the driver files of `50e8cb345` (the box's `source.txt`) and writes that SHA from `SOURCE_SHA`.
- G2. `run-day26-cell.sh:30` wrote `git rev-parse HEAD` and `git status --short`. The kit writes `SOURCE_SHA` to
  `source.txt` and `KIT-DIRTY.txt` (the three patched files, `git status --short` form) to `dirty.txt`.
- G3. chain.sh:22-23 built the binary on the box; the kit ships it (B1) and checks its sha256 before each part.

## Card scoping (portability.patch)

- P1. `run-day26-cell.sh`: `CARD` is required; `gpu-before/after.csv` and `compute-apps-before/after.csv` read
  `nvidia-smi -i $CARD` (registered: every card).
- P2. `day31-client.py` sampler: `nvidia-smi -i $CARD` (registered: the first line, physical GPU 0 whatever
  `CUDA_VISIBLE_DEVICES` said). Only `samples.csv mem_used_mib` changes; no verdict reads it.
  `tools/tier-battery.py`: the 250 ms sampler and the compute-apps snapshots read `-i $CARD` when `CARD` is set
  (registered: every card, with an index column).
- P3. Hashes that change: `run-day26-cell.sh` fad91c06 -> c7bd40bb, `day31-client.py` 53aa2512 -> eb490d6e,
  `tools/tier-battery.py` 4861050b -> 76fc9eb3. Unchanged: `day31-order.sh` a5626dd8, `day31-parse.py` 8f27e7f9,
  `day34-compare.py` c6013822, `day31-faults.py` 1fdaef8b, `day40-read.py` 0bddafff, `tools/tier-lock-proof.py`
  91ef0d0a, `docs/SERVING.md` 022b1d50. The per-boot `inputs.sha256` shows the client's new hash.
- P4. New guard in `run-day26-cell.sh`: once a boot is ready, its server pid must be a compute app on card `$CARD`;
  otherwise the boot stops (exit 6) and `boots/PLACEMENT-FAILED` stops every later boot of the order (exit 7), and the
  runner marks the order INCOMPLETE. It protects the training cards; it reads nothing the verdict reads.

## The chain (chain.sh -> run.sh)

- C1. Paths: receipts in `/mnt/Data/work/memra-maxq/admit-by-memory/out-<UTC stamp>/` (registered
  `/root/spill-receipts/b-day40`), model at `/mnt/Data/work/memra-maxq/models/`, `PATH` fixed to the system and CUDA
  bins (registered added `/root/.cargo/bin`).
- C2. Device: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD` (registered: unset, so logical device 0).
- C3. Card check: `nvidia-smi -i $CARD` must read exactly `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
  (registered: any line of every card matching `RTX PRO 6000 Blackwell Workstation`, which the Max-Q name fails).
- C4. Idle check scoped to the card: lock free, no compute app on `$CARD`, and `memory.used` under 2000 MiB
  (registered: lock free and no compute app on any card). Bounded wait 300 s, then refuse (registered 7200 s, then
  exit 3): the caller already holds the card's window.
- C5. Collector timeout: what is left of the window (`WINDOW_S`, default 14400 s, minus preflight and a 600 s reader
  margin), refusing to start under 9400 s (registered: 43200 s per order). An order the timeout cuts is INCOMPLETE and
  reruns whole.
- C6. CPU: the collector and everything under it run at `nice -n 10`; no core pinning (registered: no nice; the box
  had no neighbour).
- C7. Environment: the runner unsets every inherited `MEMRA_*`, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`,
  `LD_LIBRARY_PATH` and `LD_PRELOAD` before it sets its own; `env-O*.txt` records what the server saw.
- C8. Two parts: O1 and O2 run in separate windows in the same out dir (registered: O2 started 1 s after O1). Each
  order keeps its one collector hold. A part-2 retry moves the earlier O2 attempt to `failed-O2-<stamp>/`.
- C9. Model hash: checked before each part (registered: once per chain).
- C10. Fault lister: `day31-faults.py` gets each boot dir, the fix DAY40.md 2.1 records (the box passed the boots
  root and wrote a traceback).
- C11. `MANIFEST.sha256` leaves itself out of its list (the later chain fix `1a45587113`).
- C12. Added reading `MAXQ-VS-WS.txt` (PREREG.md 7), no bound.
- C13. Readers run with `--card pro6000` as registered, so SUMMARY.txt lines compare byte-for-byte with the
  Workstation receipt; the card's identity is in `VERDICT.txt` context lines and `card-*.csv`.
