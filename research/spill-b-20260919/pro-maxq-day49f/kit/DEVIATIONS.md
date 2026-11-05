# Deviations from the registered cell

The registered cell: DAY49 addendum D (DAY49.md 1.8), run by `research/spill-b-20260919/pro-single-day49d/chain.sh`
at 8926ccfb3e8a27cf8f91aa747e2b088251215c7a on one RTX PRO 6000 Blackwell Workstation Edition (600 W), sittings 14
(gates) and 14b (`BOOTS_ONLY=1`, boots). Copies of the three scripts as they were before the port are in `orig/`;
`reference/ws-receipt/` is the Workstation receipt. Unchanged and not listed below: the gate's arm j code and its
prompts, the eight boots and their order, N (9 rows per boot, 3 streams per gate boot), `FAULT=batch:1`,
`MODEL_KEY=q38`, `BOOT_CTX=''` (the checkpoint's 262144), `NO_SCOPE=1`, `HFG_READY_WAIT_S=900`,
`MEMRA_TIMEOUT_MS_MAX=3600000`, `MEMRA_ADMIT_PREDICT_SHADOW=1`, ports 8189 and 18526, `day45-client.py`,
`day45-parse.py`, `day49d-read.py`.

## Arms and source

- **D1. Arms by binary, at 2ccb038d1c (origin/fix4/gates), not by env door at 8926ccfb3.** The scope entry
  recommends this (caveat 1, option B). `MEMRA_BATCH_OOM_RECOVER` and `MEMRA_KV_ALLOCATOR` are retired at HEAD and
  the boot audit (env_audit.rs:161, lib.rs:6161) refuses a boot that sets either. A replay at 8926ccfb3 would read
  the env door on September code; the question is whether the HEAD default extends to the Max-Q, so the arms are
  the HEAD binary (`off`, `batch_oom_recover_default` false on a Max-Q) and the HEAD binary plus
  `maxq-batch-oom-recover.patch` (`on`). The patch adds `HardwareTarget::is_pro6000_maxq` (parallel.rs), ORs it into
  `batch_oom_recover_default` (worker.rs), names the Max-Q in the doc comment and the ON boot line (the gate greps
  only `batch OOM recovery: ON`), and moves the Max-Q name into the positive asserts of the renamed unit test, plus
  a new predicate test. The docs, decision record and DAY49 results named in the scope wait for the verdict.
  Note: the scope entry quotes the deciding commit as 8926ccfb3a8a27cf...; the commit in the repo and in
  chain.sh:19 is 8926ccfb3e8a27cf8f91aa747e2b088251215c7a.
- **D2. j-vmm (gate units vmm-green, vmm-red) dropped.** On a Max-Q the on-demand KV planes are off at HEAD
  (kv_vmm.rs:77 keys them on the Workstation Edition); `vmm_reap_for` returns at `!kv_vmm::armed()`
  (worker.rs:6535), so the reap j-vmm reads cannot be reached, and `MEMRA_KV_ALLOCATOR=vmm` is refused at boot.
  `day49d-noreap.patch` also no longer applies to HEAD. The Max-Q cell has two gate units (j1, j2) where the
  Workstation sitting had four. `out-*/gates/j-vmm.NOT-RUN.txt` records it.
- **D14. The gate script is HEAD's `tools/health-fault-gate.sh`, not 8926ccfb3's.** Arm j differs only in where
  "recovery on" comes from: 8926ccfb3 sets `MEMRA_BATCH_OOM_RECOVER=1` in `run_j`; HEAD reads the card default and
  SKIPs arms i and j when the control boot prints `batch OOM recovery: OFF`. The clause lines, prompts, streams and
  counts are byte-identical (diff of the `run_j` block). HEAD's default arm list adds k, but the cell sets
  `HFG_ARMS=j`.
- **D13. The prompts come from the deciding tree's `docs/SERVING.md`** (shipped as `docs/SERVING.md`, sha256
  022b1d50620f5b34bb6e6127be05ff79184ce3aa294aca618a31a0dd73887cff, equal at 8926ccfb3 and at the chain tree
  6cef1bbe3). `day45-client.py` reads `../../docs/SERVING.md` beside itself; HEAD's file has different bytes
  (b817b512...), which would change the prompts (scope caveat 5). With the deciding bytes the Max-Q prompts are the
  Workstation's, and `inputs.sha256` per boot shows it.

## Portability (<box>: four Max-Q cards, other cards training, no cargo, no git checkout)

- **D3. No on-box build.** `health-fault-gate.sh:98` built unconditionally and chain.sh:39 ran `cargo build`; <box>
  has no cargo. Both binaries are built on the laptop (`build.sh`, release, default features, `MEMRA_CUDA_ARCH=120a`,
  CUDA 13.4, rustc 1.97.1) and pinned by sha256 in `run.sh`. The Workstation binaries were built on the box with
  CUDA 13.1. The gate runs `HFG_BIN` and refuses when it is not executable.
- **D4. No git on <box>.** The gate's `source.txt` takes `HFG_SOURCE`; `run-day26-cell.sh` writes `CELL_SOURCE` into
  `source.txt` and a note into `dirty.txt` instead of `git rev-parse HEAD` and `git status`. chain.sh:24's fetch and
  fast-forward is gone; the kit's files are pinned by `KIT.sha256`, which `run.sh` checks.
- **D5. Every nvidia-smi read names the pinned card** (`-i $CARD`): the gate's `source.txt` and compute-apps files,
  the cell's `gpu-before/after.csv` and `compute-apps-before/after.csv`, chain.sh:28's `card.csv`. Unported, they
  list all four cards, and the training processes on the other three.
- **D6. day49d-run.sh: the arms switch binaries.** `on`/`on-plain` run `BIN_ON`, `off`/`off-plain` run `BIN_OFF`;
  `MEMRA_BATCH_OOM_RECOVER=1` is no longer added (D1). `arm.txt` gains `bin_label` and `source`. The route env
  (`MEMRA_SERVE_SPEC=0` on the plain arms) and the `env -u` list are unchanged.
- **D7. day49d-run.sh: idle() reads the pinned card only,** with no lock probe (D8), and waits at most
  `IDLE_WAIT_S` (300 s, polling every 10 s) instead of 7200 s (polling every 30 s): in the window the card is ours,
  and with training on the other cards the all-GPU check never reads idle (scope, beside_other_work 1). It also stops
  between boots when `STOP` exists (thermal, pin or deadline).
- **D8. The lock.** chain.sh took `/tmp/memra-gpu.lock` per gate unit (`flock -w 7200`) and run-day26-cell.sh per
  boot (`flock -w 3600`). The kit's run.sh takes the same lock once (`LOCK_WAIT_S`, default 120 s) and holds it for
  the whole cell; the cell gets `LOCK=none` (the collector form). <box>'s memra pair lane takes the same lock with
  `flock -n`, so the two serialize. The window holder's `/mnt/Data/work/evals/gpu<CARD>.lock` stays on FD 9;
  run.sh refuses unless FD 9 is that file and it is locked (`SKIP_EVAL_LOCK_CHECK=1` overrides).
- **D9. run-day26-cell.sh:** `WT` and `CARD` are required (the kit root; no `$HOME/projects/wt-spill-b` default),
  plus D4 and D5. Nothing else changes.
- **D10. Card pinning.** The Workstation box had one card and nothing set `CUDA_VISIBLE_DEVICES`. The kit exports
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD`; the gate and the cell pass the environment to the
  server, which opens logical device 0. On <box>, nvidia-smi's index order is the PCI bus order (01, 21, C1, F1).
  `guard.py` records which card every kit process holds (`pins.csv`) and stops the run on any other card.
- **D11. The card check accepts the Max-Q name only** (chain.sh:29 required "RTX PRO 6000 Blackwell Workstation",
  which the Max-Q name does not match), plus < 2000 MiB used and no compute app on that card.
- **D12. Paths.** Model `/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (sha256 1facf36c...,
  15,705,922,304 B, checked by run.sh before the first boot and, as registered, per boot); receipts
  `/mnt/Data/work/memra-maxq/batch-oom-recover/out-<UTC stamp>/` (`gates/j1`, `gates/j2`, `serve/boots/*`); WT is the
  kit root.

## Running beside training (kit additions; no registered clause changes)

- **D15. CPU and environment.** The whole cell runs at `nice -n 10`, no core pinning (the server's worker affinity
  stays off by default). run.sh unsets every inherited `MEMRA_*` name (`env-scrubbed.txt`) and sets
  `TMPDIR=/mnt/Data/work/memra-maxq/batch-oom-recover/scratch`, deleted on exit; the cell itself writes only logs.
- **D16. One sitting.** The Workstation ran the gates in sitting 14 and the boots in 14b (sitting 14's boots never
  started, DAY49 2.3). The kit runs both in one sitting, in chain.sh's order.
- **D17. Bounds and guards.** A total deadline (`TOTAL_TIMEOUT_S`, default 5400 s; gate unit cap 1800 s, boots
  cap 3600 s), the Max-Q thermal rule, the pin check and 250 ms telemetry of the pinned card (`guard.py`). A stop
  reaps only this kit's servers (executable under `bins/`, cwd the kit root).
- **D18. The reader.** The Workstation reading combined the gate lines and `read-serve.log` in DAY49 2.3, 2.4 and
  the decision record. `maxq-verdict.py` runs `day49d-read.py` unchanged and prints that reading as one VERDICT
  line, adding the kit clauses of PREREG.md section 5 (arm identity, pin, thermal, gpu-watch, deadline). The card
  label is `pro6000-maxq`.
- **D19. The box.** <box>: driver 615.71.09 (Workstation box: 610.43.02), 64 cores, about 1 TB RAM, the other three
  cards training. The server's gpu-watch canary reads every card and the kernel Xid log (health.rs:1232-1260); it is
  left on, as registered, and a `[gpu-watch] CRITICAL` line makes the run NOT-READ (PREREG.md section 5).
