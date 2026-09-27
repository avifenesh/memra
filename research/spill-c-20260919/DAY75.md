# WP-C day 75 (2026-09-26): OWED C11, I16, the door's prefetch after the current expert's launch, registered before code

`DAY72.md` section 3 reads the target card's `gap15` as registered: admissible, `partA=cpu_side partB=gpu_stall`. The
door's GPU work equals REF's, the GPU waits 0.39 ms per window token longer (under the profiler), and the door's
prefetch path is 0.266 ms per window token longer than REF's on the CPU. Tree at start: `da7a03c0d`.

## 0. Where the waiting comes from, from source (`hybrid_forward.rs`, the cached expert loop)

For each selected expert `j` of a token, the loop first prefetches expert `j+1` (`moe_prefetch_expert`, under the door
through the owner: the host-hit lease, the residency check, the retire, the copy enqueue), and only then launches
expert `j`'s kernels (gate and up, the activation, down, the accumulate). Both programs have this order. When the GPU
has drained the previous expert's kernels by the time the CPU reaches expert `j`, every microsecond the prefetch takes
on the CPU is a microsecond the GPU waits before `j`'s first kernel. The door's prefetch takes longer than REF's (the
owner's lease), so its GPU waits longer: the `gpu_stall` with equal kernels that `gap15` read.

## 1. Pre-registration: I16

**The change.** Under the door only (the forward's prefetch condition `e.expert_bank_prefetch()`, set by the door's
installer; REF, `MEMRA_MOE_PREFETCH=1` without the door, keeps today's order), the prefetch of expert `j+1` moves from
before expert `j`'s kernels to after them: the loop launches `j`'s gate, up, activation, down and accumulate, then
issues `j+1`'s prefetch, with the same `keep` (expert `j`'s three blocks). Nothing else changes: the same prefetches,
the same blocks, the same lease protocol, the same kernels. The numeric program is unchanged (a prefetch writes a
block's bytes into a slot; which kernel reads which slot is decided at dispatch, as today), so every run's tokens must
match the other arms' exactly.

**Its cost and its risk, stated.** It removes the prefetch's CPU time from before each expert's launch; it gives the
prefetched copy less lead: the copy of `j+1` is enqueued after `j`'s kernels are, so on the copy stream it starts about
the prefetch's own CPU time later than today, and `j+1`'s kernels wait on it on the GPU if the copy is longer than
`j`'s kernels. The cell reads which effect is larger.

**CPU gates before any card** (`day75-cpu/`): the engine's library tests; the MoE cache and door tests; a source
census that the door's branch issues its prefetch after the accumulate and REF's branch before the kernels; `cargo
clippy -D warnings`, `cargo fmt --check`. No `.cu` change.

## 2. Pre-registration: the card cell `i16` (the 285K class, then the RTX 5090; before its script)

Binaries: `i15=2243b1fe2` (REF's arm and the door before I16, one binary as `gap15`'s), `i16` (I16's commit, named in
section 2a before any cell). Arms: REF (`run-gen-i15`, `MEMRA_MOE_PREFETCH=1`), I15 (`run-gen-i15` with the door), I16
(`run-gen-i16` with the door), I16C (I16 with `--moe-dispatch-clock`, read for its brackets only). Order 1 (REF, I15,
I16, I16C) x 5, order 2 reversed x 5, 40 runs, day 18's pressure shape, one collector hold. Then Part B as `gap15`'s:
REF and I16 under Nsight Systems, order REF, I16, I16, REF, per-window rows kept, reports by hash.

- **Integrity:** every run exit 0 and `MATCH`; one tape across every arm; every door run's fill complete and
  `physical_reads=0`; within each door arm one host demand sequence without slot numbers (I16's sequence may differ
  from I15's in order, since a prefetch moves after a dispatch; the two are printed, not compared).
- **Admissibility:** `DAY64.md` section 5's clause (every arm's gen-only and window IQR at most 0.005 s), before any
  step or door reading; inadmissible decides nothing.
- **Readings:** I16 against I15, `improves`, `regresses` or `flat` as `DAY61.md` section 2 defines them, gen-only the
  primary reading and the window beside it; the door (I16, or I15 if I16 `regresses`) against REF, `beats`,
  `matches` or `loses`. Part B, deciding nothing: I16's `gpu_idle` and `h2d_exposed` against REF's.
- **What follows.** `regresses` on either card: I16 is reverted with its receipt. `flat` or `improves`: it stays. The
  door against REF is recorded plainly, for the owner's 2026-10-04 read.

## 2a. I16 on the CPU, the binary, and the sitting, before any cell

I16 landed as `eeacfaf50` (`hybrid_forward.rs`: under the door the prefetch of expert `j+1` is recorded and issued
right after expert `j`'s accumulate in both cached branches; REF's branch unchanged; `native.rs`: census `day75`, and
`day50`'s count of the prefetch condition's mentions from 1 to 2). CPU gates (`day75-cpu/`): the engine library 574
passed, 0 failed (`engine-lib-tests.log`), the census tests all pass, `cargo clippy -D warnings` and `cargo fmt
--check` clean (`clippy-fmt.log`). No `.cu` change, no new flag.

Binaries: `i15=2243b1fe2`, `i16=eeacfaf50`. `day75-cell.sh`, `day75-read.py` (it reuses `day72-read.py`'s profiled
window reading) and `day75-box.sh` were written after section 2. Dry checks (`day75-cpu/`): the reader on a synthetic
cell built from DAY72's target `gap15` receipts (`make-synthetic.py`, its reading meaningless;
`dry-check-reader.log`); the cell's control flow with stub binaries and a stub `nsys` (40 timed runs, 22 on each
binary including the profiled pair; `dry-check-cell.log`); the driver (`dry-check-driver.log`).

Run as `D75_BUILDS="i15=2243b1fe2 i16=eeacfaf50" bash /root/wt-c/research/spill-c-20260919/day75-box.sh` on a Core
Ultra 9 285K host with one RTX PRO 6000 Blackwell Workstation Edition (box needs as `DAY72.md` section 1a). Receipts in
`/root/spill-receipts/c-day75/`, profiles by hash. Expected: two builds about 10 minutes, the cell about 25. The RTX
5090's half runs from queue v12 (`rtx5090-queue-v12-20260926.sh`).

## 3. The target card (BOX29, the 285K class, run by the lead as registered; `pro-single-day75/`)

The lead ran `D75_BUILDS="i15=2243b1fe2 i16=eeacfaf50" bash .../day75-box.sh` on BOX29 (a Core Ultra 9 285K, 188 GB,
one RTX PRO 6000 Blackwell Workstation Edition) on the tree `7b3fbdfa6`, after lane B's sitting on the same card,
09:32Z to `box done 2026-09-26T09:45:46Z`. Receipts: 219 of 219 `OK` against the box manifest (re-checked), the ELFs by
hash; the eight profiler files by hash only (`profiles.sha256`), kept out of the repository; the window rows in
`i16/ev/`. Regime: 34 to 47 C, SM median 2610 MHz, N=2022. Verbatim (`i16/reading.log`):

- `DAY75 host demand sequence i15 sha256 4bdc2610c3534e42 lines=[22077]`, `... i16 sha256 0e220d04f52d13e9 lines=[22077]`, `... i16c sha256 0e220d04f52d13e9 lines=[22077]`
- `DAY75 I16 CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY75 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0010 max_iqr_window=0.0010 failing=[] -> admissible`
- `DAY75 gen-only decode medians (N=10 each): ref=0.255 i15=0.264 i16=0.282 i16c=0.283`
- `DAY75 STEP i16_vs_i15 gen-only decode: pooled=+0.0180 o1=+0.0180 o2=+0.0190 noise=0.0010 -> regresses`
- `DAY75 STEP i16_vs_i15 steady window: pooled=+0.0090 o1=+0.0090 o2=+0.0090 noise=0.0010 -> regresses`
- `DAY75 DOOR i15_vs_ref gen-only decode: pooled=+0.0090 o1=+0.0090 o2=+0.0090 noise=0.0002 -> loses`
- `DAY75 B i16_minus_ref per window token (ms, medians of two; deciding nothing): gpu_busy=-0.0058 gpu_idle=+0.3808 h2d_exposed=+0.2636 kernel_sum=-0.0057`
- `DAY75 VERDICT rig=pro-single integrity=ok i16=regresses door=i15 vs_ref=loses (window: i16=regresses vs_ref=loses)`

**Read as registered: I16 `regresses` (18 ms gen-only over 32 tokens, 9 ms window), so it is reverted with this
receipt** (section 2), and the door stays I15, which `loses` to REF as before. The risk section 1 named is what
happened: with the copy enqueued after the current expert's kernels, the next expert's copy is exposed (the door's
exposed copy time 0.26 ms per window token above REF's, against 0.03 for I15 in `DAY72.md` section 3), and the GPU
still idles as long as before. The host demand sequence changed with the order, as registered, and held within each
arm. Beside it: I16C's brackets read the same prefetch-path costs as I15's (`pf_demand_ns` 0.155 ms per window token).

## 3a. The RTX 5090 (queue v12, 2026-09-26 06:02Z to 06:20Z; `rtx5090-day75/i16/`)

`run-gen-i16` rebuilt locally from `eeacfaf50` (`8a05d290...`), no compute app at any run boundary. Regime: 64 to 75 C,
SM median 1590 MHz, N=2373. Verbatim: `DAY75 ADMISSIBILITY rig=rtx5090 ceiling=0.005 max_iqr_gen=0.0090
max_iqr_window=0.0045 failing=['i15:gen_s=0.0090'] -> inadmissible`, `DAY75 VERDICT rig=rtx5090 integrity=ok -> void
(inadmissible) [as read: i16=regresses door=i15 vs_ref=matches (window: i16=flat vs_ref=loses)]`. Inadmissible, so it
decides nothing on this card; recorded as it reads (I16 19 ms slower gen-only here too, its copies exposed 0.36 ms
per window token more than REF's under the profiler). The revert follows the target card's `regresses`, which the rule
makes sufficient on either card.
