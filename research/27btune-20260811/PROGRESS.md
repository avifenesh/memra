# 27B NVFP4 decode tuning progress

## 2026-08-11 — lane start

- Branch: `lane/cx-27btune`
- Base: `429ef3d5`
- Rig: local RTX 5090
- Model: `/home/avifenesh/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
- Draft: `/home/avifenesh/models/draft-daily-owntrim-nvfp4head-q4blk.gguf`
- Speculative depth: K=3
- Baseline anatomy: 30.413 ms/round from `research/ncu27b-20260811/RESULTS.md`

Targets, in order:

1. `qmatvec_nvfp4_mmvq_b4_rpr2w8`
2. `fa_decode_f32`
3. `qmatvec_nvfp4_mmvq_b4_rp`

Success means a naked-default kernel change with an interleaved A/B x5 result under one
GPU-lock hold, retained raw logs, and all three correctness gates passing. Flat or negative
arms will be recorded and removed rather than retained behind a flag.

Status: lane initialized; no kernel changes or measurements yet.

## 2026-08-11 — evidence read and first-arm constraint

Read the complete target sections from the NCU details export before touching source. The two
`rpr2w8` grids are 640 and 1,280 blocks, both at 64 registers/thread, zero spill, zero shared-bank
conflicts, and 0.98/1.95 waves per SM. Their dominant exposed stall is long scoreboard (about
7 cycles per issued instruction), while DRAM reaches only 54.4%/63.9%.

The old `rpcar2` arm is not a reusable answer: its three-stage cp.async ring measured -11.3%
against `rpr2w8` because shared staging and synchronization lost the eight-block residency.
Any new async arm must therefore retain eight resident blocks and the existing row mapping and
reduction order. The current NVIDIA guidance also favors 8- or 16-byte global-to-shared async
copies, but occupancy remains the first acceptance gate for this exact kernel.

No source change yet. Next: compile-only occupancy probe for an eight-resident, narrower staging
shape before spending a full A/B window.

## 2026-08-11 — eight-resident async arm compiles

Added the force-only `rpcar2w8` probe. It uses two split-plane staging slots and issues the next
window before the current dot product, preserving overlap without the old three-stage arm's shared
memory footprint. `ptxas` reports 62 registers/thread, 9,216 bytes static shared memory/block,
zero spill, and zero stack for `qmatvec_nvfp4_mmvq_b4_rpcar2w8`; the unchanged `rpr2w8` baseline
uses 64 registers/thread. Both resources remain compatible with the target eight blocks/SM.

The full sm_120a release build completed successfully. The probe remains force-only and is not a
default. Next: a locked same-process-shape microbenchmark against `rpr2w8`; only a clear kernel
win advances to the model-level interleaved x5 gate.

## 2026-08-11 — measurement harness ready; first lock window busy

The short paths in the lane message are not populated, but the repo's canonical 27B directory is.
Its trunk and draft hashes exactly match the preceding NCU capture (`d8d71c7e...d517` and
`b445fbb1...3581`), so the harness uses those byte-identical artifacts rather than another model.

Added a same-lock, cold-weight microbenchmark harness with eight rotating weight copies and balanced
`A,B,B,A,A,B` ordering. Its first 60-second lock attempt exited 75 before any GPU command because
another lane's `tools/local-ci.sh --perf-quick` held `/tmp/memra-gpu.lock`. This is infrastructure
contention, not a kernel result. Next: rerun the unchanged harness after that gate releases the rig.

The same external `kernel-check` still owned the lock during a second bounded attempt, which also
exited 75 before GPU access. Meanwhile the lane's release `run-spec`, `run-gen`, and `kernel-check`
binaries were built successfully; the candidate symbol is present in `run-spec`, and `git diff
--check` is clean. No performance or correctness verdict has been inferred from the busy windows.

A third bounded attempt found the same lock owner, now in its Gemma gate, and again exited before
GPU access. Static SASS inspection of the actual release fatbin confirms the intended schedule:
`DEPBAR.LE` waits for the current slot, the next window's `LDGSTS` instructions issue, and only
then do the current slot's `LDS` and dot-product body execute. This validates the compiled mechanism,
not its speed; measurement still waits for an uncontended lock hold.

## 2026-08-11 — async matvec arm rejected and deleted

The first uncontended window used the exact NCU-capture model, eight rotating cold-weight copies,
one GPU-lock hold, and balanced `A,B,B,A,A,B` ordering. On `blk.0.ffn_down.weight` (the 640-block
`rpr2w8` shape), N=3 medians were:

| b4 cell | `rpr2w8` | `rpcar2w8` | latency delta |
|---|---:|---:|---:|
| m=3 | 65.45 us | 76.94 us | **+17.56%** |
| m=4 | 69.70 us | 85.55 us | **+22.74%** |

All six candidate b4 rows were bit-exact against the original-layout reference, so this is a pure
performance rejection. The thermal window ran from 67 C/P8 at entry to 77 C/P0 at exit, arms were
order-balanced, and the compute-app census was empty. The regression is far beyond the 1% win floor;
the 1,280-block shape and model-level A/B were therefore not spent. `rpcar2w8` and its force seam
have been deleted, restoring the pre-probe runtime source exactly. The result is retained in
`arms.jsonl` and `raw/microbench-rpcar2w8-ffn-down-x3.log`.

Mechanism verdict: preserving block residency did not make shared staging free. Four `LDGSTS`
operations plus the dependency/barrier choreography per window cost more than the direct split-plane
loads' exposed scoreboard latency. Next: inspect the short-context scalar `fa_decode_f32` grid
mechanism, while preserving its exact split/fold order.

## 2026-08-11 — scalar FA four-way probe staged

The scalar path currently assigns all early-context keys to one block per Q head, producing the
24-block NCU grid. Added a force-only `MEMRA_FA_SCALAR_NSP=4` geometry that uses up to four live
partitions per head (96 blocks once `T_kv >= 4`) and the existing deterministic combine. Eager,
device-counter capture, and graph bucket geometry share the same helper; graph updates shrink the
first three token positions through `split_keys=1`. The kernel caps its derived live split count at
the allocated grid/partial stride, leaving the default 256-key partition unchanged.

This arm intentionally changes the early-context key fold order and therefore cannot advance on a
micro-timing result alone. Next: compile, then run a candidate K=3 self-consistency shakeout before
the same-lock interleaved x5 performance window.

The full sm_120a release `run-spec`, `run-gen`, and `kernel-check` build completed in 2m28s. A first
60-second shakeout lock attempt was overtaken by another lane's newly started
`tools/local-ci.sh --perf-quick`; `flock` exited before the capture body and the raw file is empty.
No exactness or performance result exists for this arm yet.

The queued retry acquired a clean window and completed. With `MEMRA_FA_SCALAR_NSP=4`, K=3/NGEN=64
produced 63/63 target-identical generated tokens, 42/69 accepted (60.9%), and 91.28 spec tok/s;
`SELF-CONSISTENCY PASS`. The release fatbin keeps `fa_decode_f32` at 39 registers, 0 local/stack,
and 1,024 bytes shared memory, so the arm changes grid/partition geometry rather than residency.

This shakeout is correctness evidence only. Its acceptance/round count differs from the preceding
NCU capture, and that older binary/window is not a valid denominator. Next: current-binary baseline
versus four-way, N=5 interleaved under one lock hold, with per-run acceptance and thermal state.

## 2026-08-11 — scalar FA four-way arm rejected and deleted

The one-lock interleave completed with order `A,B,B,A,A,B,B,A,A,B`, N=5 per arm. Baseline spec
throughput was `[96.37, 95.59, 95.45, 95.20, 94.91]` tok/s (median 95.45); scalar-four-way was
`[92.74, 92.74, 92.26, 92.18, 91.84]` tok/s (median 92.26), a **-3.34%** throughput change.
All ten runs passed target self-consistency within their own arm. The altered split/fold order did,
however, change the plain target stream: baseline needed 21 rounds with 42/63 acceptance (66.7%),
whereas scalar-four-way needed 23 rounds with 42/69 acceptance (60.9%).

The balanced thermal window ran from 59 C/P8 at entry to 72 C/P0 at exit, with per-run starts up to
79 C and an empty compute-app census. Because the candidate is below the 1% win floor and changes
model output/reduction order, the geometry seam and kernel cap have been deleted. This result is a
model-level rejection; it does not claim that the isolated FA kernel itself slowed, because changed
acceptance adds two whole rounds. Next: inspect the remaining `b4_rp` target's shape mix for a
mechanism that preserves exact row reductions.

## 2026-08-11 — tiny RP auxiliary-dual arm staged

The remaining RP symbol decomposes cleanly by model structure: 48 linear-attention layers each
launch `ssm_beta` and `ssm_alpha` at out_f=48, giving the measured 96 launches/round at grid 12;
16 full-attention layers contribute K/V at grid 256, and the 48 linear `wqkv` projections carry
the grid-1,536 class. The smallest exact arm is therefore the beta+alpha pair, not a rewrite of the
large DRAM-bound carrier.

Added a force-only `MEMRA_NVFP4_AUX_DUAL=1` call-site seam and a one-row-per-warp dual wrapper.
It preserves the current tiny singles' `b4_rp` template body and per-output reduction order while
combining two sequential 12-block launches into one 24-block grid. The existing large-pair
`b4_rpr2` policy is untouched. Next: compile, confirm the engagement receipt and kernel resources,
then require a K=3 self-consistency shakeout before spending the interleaved x5 window.

The full release build completed in 2m25s. The actual `qmatvec.fatbin` reports 44 registers/thread,
zero stack/local memory, and 1,024 bytes shared memory for the new dual; that exactly matches the
single `b4_rp` register class and is substantially below the existing `b4_rpr2` dual's 70 registers.
The intended 24-block launch therefore keeps the singles' residency class. Next: locked engagement
and exactness shakeout.

The locked K=3/NGEN=64 shakeout printed the engagement receipt, retained the baseline target token
stream, and passed 63/63 target self-consistency. It accepted 42/63 draft positions (66.7%) in 21
rounds at 98.29 spec tok/s. This is exactness/engagement evidence only, not a denominator. The
current-binary N=5 interleave harness is now fixed at `A,B,B,A,A,B,B,A,A,B` for the decision window.

## 2026-08-11 — tiny RP auxiliary dual wins the local gate

The one-lock x5 interleave completed with every paired repetition favoring the candidate. Baseline
spec throughput was `[96.89, 96.22, 95.98, 95.81, 95.85]` tok/s (median 95.98); the 24-block
auxiliary dual was `[97.63, 97.44, 96.98, 97.08, 96.94]` tok/s (median 97.08), a **+1.15%**
throughput change that clears the lane's 1% floor. The paired gains were +0.76%, +1.27%, +1.04%,
+1.33%, and +1.14%.

All ten runs generated the same plain target tokens, accepted 42/63 positions (66.7%) in 21 rounds,
and passed target self-consistency. The balanced window ran from 54 C/P8 at entry to 71 C/P0 at
exit, with per-run starts through 76 C and an empty compute-app census. Verdict: promote the tiny
beta+alpha dual as the naked default, retain an explicit `MEMRA_NVFP4_AUX_DUAL=0` rollback seam,
and add a 48-row bit-identity cell to `kernel-check` before the final battery.

Promotion is now the naked t=3 default. `MEMRA_NVFP4_AUX_DUAL=0` restores the two single launches
for rollback/A-B; the research harness was inverted accordingly so its A arm remains the old policy
and B remains naked. The new `DUAL-BATCHED-AUX` kernel-check cell slices 48 real NVFP4 rows,
re-packs both operands, and requires bit identity against two `b4_rp` singles.

The promoted tree rebuilt all three release binaries successfully in 2m24s. The final harness holds
one GPU lock across `kernel-check` (including the new bit pin), the 27B `run-gen` argmax gate, and
the naked-default K=1..8 `run-spec` sweep. It records binary/artifact hashes and GPU/application
state before parsing the raw logs.

## 2026-08-11 — final local battery green

The final harness ran against promoted commit `febb2b98c8f77e1fdbfa2f574047d3a1691f6d94`
under one uninterrupted GPU lock and ended with `result=PASS`. `kernel-check` reported
`DUAL-BATCHED-AUX [NVFP4 rp] out=48 m=3: bit-bad=0/0 OK` and `ALL GREEN`; the 27B
`run-gen` gate reported both prefill/decode and batched-prime/tokenwise argmax `MATCH`; and the
naked-default `run-spec` battery passed target self-consistency for every K=1..8. K=3 retained the
expected 42/63 acceptance in 21 rounds and reported 97.19 tok/s in this single correctness run.

The harness pinned the target model, draft, and prompt hashes and recorded the three release-binary
hashes in `raw/final-gates-driver.log`. This completes the RTX 5090 development-iteration lane only.
No generated performance board moved, and no merge, tag, or release is authorized until the same
correctness battery passes on the designated Vast 2x RTX PRO 6000 verification box.
