# PRO 6000 sitting for OWED 17 and OWED 18 (and OWED 26 on the target card)

Registered in `../M1-PREREG.md` sections E, F (with its build amendment and correctness addition)
and G. One box, one card, one window, in the order below. Commands: `run-sitting.sh`.

## Host needs

| Need | Why |
|---|---|
| One RTX PRO 6000 Blackwell, the only GPU tenant, power limit at its maximum | the verdict target for both doors; the runner and handoff driver gate on GPU co-tenancy |
| A writable local PCIe NVMe scratch (ext4 or xfs, not overlay, network or loop) that passes the M1 proof, same class as BOX27's | **required for both items**: `m1-spill-runner.py` refuses to run without a PASS `nvme-local-direct` proof and requires the artifact on the proven filesystem; `m1-handoff-driver.py` requires its scratch on the proven filesystem. Item 17 is a storage-to-compute cell (every cold visit reads the bank from that drive), so the storage class is part of what it measures |
| At least 64 GiB RAM (BOX27 had about 123 GB) and no swap in use | the bounded regime's touched balloon; item 18's 16 GiB pinned host tier plus the model load |
| Root, or `CAP_IPC_LOCK` with unlimited memlock | the balloon and pinned host buffers |
| At least 16 CPU threads | the 16-deep worker pool plus the runner and samplers |
| At least 60 GB free on the proven scratch | the 18.2 GB artifact, the build, the 9.7 GB handoff file, receipts |
| `nvidia-smi`, a Rust toolchain matching the repo, CUDA 13.x `nvcc` | build and telemetry |

io_uring is not needed. Any box of BOX27's class serves; no other box shape is a substitute,
because without the proof the runner and driver refuse.

## Order and what each cell decides

1. Build at the lane tip (`MEMRA_CUDA_ARCH=120a`), hash every binary; stage the artifact from
   `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` and check sha256
   `df27a780...7adf`; regenerate the B2 prompts against their manifest.
2. M1 proof of `/scratch/spill-f` (a FAIL ends the sitting; its receipts are kept).
3. OWED 26 on the target card: every pool GPU cell, including the four OWED 26 cells
   (`demand_submit_waits_for_a_buffer_instead_of_returning_ring_busy`,
   `demand_submit_returns_none_only_when_prefetches_hold_every_buffer`,
   `demand_wait_with_a_free_buffer_and_nothing_in_flight_returns_at_once`,
   `demand_wait_after_every_h2d_event_completed_returns_at_once`); green only, the red arms are
   the 5090 record. The smoke in step 4 is also OWED 26's serving-shape check on the target card:
   its visits must show zero fallbacks and zero demand-wait timeouts.
4. OWED 17 correctness: one smoke round (three arms, tokens equal the oracle file, bypass lines),
   gated with `m1-b3-pool.py --bypass-check --fallback-unclean --require-correct`; then `run-spec`
   K=1..8 for `bypass-staged` and `bypass-mapped`.
5. OWED 17 timing: `cold` regime, ten rounds; `bounded` regime (touched balloon, floor 2 GiB),
   ten rounds; three arms each, the F verdict rule, fallback visits unclean.
6. OWED 18: 1 GiB cell, 20 cycles as ten alternating pairs (buffered then direct, then reversed),
   host budget 16,384 MiB, default tenant share (BOX27's B2 1 GiB conditions); 8 GiB cell, same
   schedule, tenant 100% (B2 amendment 3). Verdict per metric (export ms, write, fsync, import s)
   per section E.

Expected wall time: about 3 hours (stage 10 min, build 5, proof 2, OWED 26 5, smoke and spec 15,
cold 35, bounded 50, handoff 1 GiB 15, 8 GiB 40).

## After the sitting

Mirror with a box-side sha256 manifest, binaries by hash only, sanitize any volume id
(`../m1-box-sanitize.py`), remove `/scratch/spill-f` and the receipts dir, report the box released.
