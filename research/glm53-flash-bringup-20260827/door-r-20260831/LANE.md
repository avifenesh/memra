# glm5 DOOR R — the tcols fused-t reduce tail (moe-loc named follow-up 3)
### lane/glm5-door-r, 2026-08-31

Base: `origin/lane/glm53-flash-bringup` @ **a5d608b07** (the ep-place merge head), with
`origin/lane/glm5-moe-loc` @ 6e6120a0e ABSORBED by merge (the design doc's lane; its merge to
the bringup lane was in flight when this lane opened, so the lane absorbed it directly — the
rerere resolutions match the landing merge). Worktree `~/projects/wt-glm5-door-r`, branch
`lane/glm5-door-r`.

Charter (moe-loc LANE.md §2.2, "door R, designed and sized, NOT built here"): the kda trunk's
tcols calls sit at **67.0% of peak (7.608 ms for 9,125.6 MB/round) after door X** because
`matvec_bf16_f32acc_x1_tcols` pays ~30 block-wide barriers per block (9 per token column at
t=3.34; 135 at the drafter head's t=15) against a **4-iteration main loop** (kda shape:
in_f 4096 / (128 threads x 8) = 4 trips). The kernel is barrier/tail-bound, not DRAM-bound —
door X fixed the wave count (1.0265x) but the tail is per-block work, so 4x the blocks pays
it 4x more often. This lane BUILDS the design exactly; nothing else.

Rig law: every receipt here is a 5090 exactness/counter receipt (`flock
/tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`); no timing row exists in this lane. The box
window prices the flip.

---

## 1. WHAT LANDED

| door | flag (default) | mechanism | bit bar | receipt |
|---|---|---|---|---|
| **R** tcols fused-t reduce tail | `MEMRA_BF16_TCOLS_RED_FUSED` (**OFF**) | the three tcols twins gain `_rf` fused-tail forms (`matvec_bf16_f32acc_x1_tcols_rf`, `..._x4_tcols_rf`, `..._x4_tcols16_rf`): (a) one shared region per token column (`red[t*blockDim]`, dynamic shared = t·block·4 B) so the t strided trees share ONE barrier sequence — a store barrier plus one per block-wide level down to s=32; (b) levels s<=16 become an intra-warp `__shfl_down_sync` chain (offsets 16,8,4,2,1; one column per warp), zero barriers. **Barriers per block: 9t (~30 at t=3.34, 135 at t=15) -> 3** (x1 form, 128 threads; the x4 forms add one trailing barrier per p iteration). Launcher composes with door X (grid form first, tail twin second) and door T (both tcols routes take it) | PAIRING PRESERVATION: every level adds the SAME index pairs in the SAME `v = v + v[i+off]` operand order as the standing `red[i] += red[i+s]` tree — induction: after level s, lanes i<s hold exactly the tree's partials, so lane 0's value is the tree's `red[0]` bit-for-bit; the main loop is VERBATIM. The gate's shifted-pairing RED (`..._rf_redshift`: ASCENDING offsets 1,2,4,8,16 — the same 32 partials under a different association) proves the bar can see an association change | §3 gate table |

Engagement guard: the fused tail's block-wide loop must pass exactly through s=32, so the door
requires a POWER-OF-TWO `MEMRA_MMV_BLOCK` (the default 128 qualifies; 96/160/192/224 fall
through to the standing tree — stated in the FLAGS row). Engagement counter
`BF16_TCOLS_RED_FUSED_DISPATCHES` + snapshot `bf16_tcols_red_fused_dispatches()`; announce
once per boot: `[bf16-tcols-red-fused] engaged: fused-t reduce tail ...`.

**Default OFF BY DESIGN** (new-flags law): the rig is exactness-only and no box timing receipt
exists. FLAGS.md row (both arms, rollback seam `=0`/unset per call, receipts pointer) and
KERNELS.md rows + symbol recount land in this PR.

## 2. PREDICTED ms/round (arithmetic against banked BOX constants — never a claim)

Winner today: 70.458 tok/s, 2.6301 tok/round, **37.328 ms/round** wall, GPU-busy 27.09.

| lever | basis | Δms/round (predicted) |
|---|---|---|
| kda trunk (136 calls, 9,125.6 MB/round at 1.1995 TB/s = 67.0% of peak) | at the banked 87% bound (1.5573 TB/s) = 5.860 ms | **-1.75** (the cap for the trunk alone) |
| whole bf16-mmv tcols class (11.80 GB/round at 1.2394 TB/s) | at the bound = 7.578 ms | -1.94 (the class cap) |
| **door R band (moe-loc §2.2 sizing)** | barrier removal does not buy the full gap to the bound by arithmetic — the tail is the named mechanism, not the only one | **-1.0 to -2.0, capped at -1.94** |

    if -1.0: 36.33 ms -> 72.4 tok/s (1.028x)
    if -2.0: 35.33 ms -> 74.4 tok/s (1.056x)

Composes with the moe-loc D+H host band (-0.5 to -3.0) toward the ~74-81 tok/s figure the
moe-loc lane stated against the 100 bar. The box window prices the actual wall.

## 3. GATE TABLE

Runner: `receipts/run-battery.sh`; per-suite logs in `receipts/`. Run with
`--include-ignored`, never `--ignored` (the moe-loc §4.2 recovery: `--ignored` ran ZERO tests
in five suites and reported ok) — every suite below states a NON-ZERO pass count.

### 3.1 The door-R gate — `glm5_matvec_doors_gpu::gpu_red_fused_tcols_matches_standing_tree_bitwise`

Receipt: `receipts/glm5_matvec_doors_gpu-nocapture.log` (suite 5/5; the door-R arms below).

| arm | result |
|---|---|
| t=2..=8, BOTH grid forms (door X pinned =1 for x1, =0 for x4 — each form vs ITS OWN standing tree), R=0 arm pinned `=0` with the dispatch counter FLAT, R=1 arm counter >0 | **PASS, 0 diffs in all 14 arms** (`door R PASS t={2..8} x1/x4`, 266..1064 outputs per arm) |
| t=9..=16 through `matvec_bf16_tcols16_into` (the drafter head's t=15 in-range), R=0 vs R=1 bitwise | **PASS, 0 diffs in all 8 arms** (`door R PASS t={9..16} tcols16`, 1197..2128 outputs per arm) |
| t=1 (unrouted — the launchers refuse t<2): all three `_rf` twins at the degenerate column-loop bounds vs the per-row t=1 program, via the gate-only allowlist launcher | **PASS, 0 diffs, all three twins** |
| shifted-pairing RED: `matvec_bf16_f32acc_x1_tcols_rf_redshift` (ascending offsets) vs `..._x1_tcols_rf` MUST differ | **BITES: 164/532 outputs differ** — the bit bar sees an association change |
| boot announce | `[bf16-tcols-red-fused] engaged: fused-t reduce tail, one barrier sequence shared across the t token columns + intra-warp shuffles at the identical pairing (MEMRA_BF16_TCOLS_RED_FUSED=1)` — first line of the receipt log, once per process |

Fixture note: in_f = 2048 so in_f/8 = 256 >= the 128-thread block — EVERY lane's partial is
nonzero. With a short in_f the top lanes hold +0.0f and a shifted association over zeros
cannot round differently (a+0 is exact), which would make the red arm vacuous.

### 3.2 Standing batteries

| arm | suites |
|---|---|
| DEFAULT (door R OFF = the ship arm; matvec doors T/X/K/W at their shipped default ON) | ALL PASS, non-zero counts: glm5_matvec_doors 5/5, glm5_moe_loc_doors 4/4, verify_batch 4/4, tparallel_verify 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 13/13, mtp_head 6/6, kpool_indexer 18/18, hyper_connections 7/7, hc_fused_pre 4/4, hc_decode_ws 2/2, mla_decode_split 3/3, kda_fixture 3/3, kda_fused_proj 5/5, kda_fused_proj_bf16 5/5, kda_quant_operand 4/4, mla_gpu_forward 5/5 |
| COMPOSE (MEMRA_BF16_TCOLS_RED_FUSED=1, door M pinned =0): the walk suites | ALL PASS: verify_batch 4/4, tparallel_verify 9/9, spec_session 10/10, dflash_session 10/10, moe_epilogue 13/13. `glm5_verify_batch_gpu` gate 1 calls `matvec_bf16_tcols_into` directly, so under this arm it is "the `_rf` twin vs the per-row t=1 program, bitwise, t=2..=8 + shifted-row red" — real compose coverage at the kernel seam |

**Engagement SCOPE, stated before it can be mistaken for a gap:** the tcols class engages on
FloatBf16 mixers, and the rig walk fixtures run quantized/f32 trunks — NO tcols announce
(doors T/X included, both default ON) appears in any rig walk suite or the banked moe-loc
ppn-gate logs. This is the standing shape doors T and X shipped with: kernel-seam bit gates
on the rig, walk-level engagement receipts from the BOX battery on the serving artifact. The
compose arms above prove the door does not perturb the walks and that gate-1's direct tcols
route is bit-identical through the `_rf` twin; the box window's boot log carries the
`[bf16-tcols-red-fused] engaged` announce on the real bf16 trunk.

### 3.3 Other gates

| gate | result |
|---|---|
| `tools/check-flags.sh` | 747 runtime literal reads, no uncovered names, no grandfather list — `MEMRA_BF16_TCOLS_RED_FUSED` resolves against `docs/FLAGS.md` |
| `cargo clippy --all-targets` | ZERO lints (incl. removing a stale doc line the moe-loc merge reassembly duplicated — the base fixed the same line independently at 98336b72d; the absorb merge reconciled cleanly) |
| `cargo fmt --all --check` | clean |
| `tools/local-ci.sh --perf` | **exit 0** on the final tree (`receipts/local-ci-perf-run1.log`): correctness ALL GREEN, perf stage **0 fail 1 warn** — qwen9b-plain-short 132.92 tok/s `[WARN]` at -1.56% vs the 135.02 rolling median (WARN passes by the house rule; door R is default OFF and touches no default-path kernel — the standing tcols programs are byte-unchanged). The eight absent-model cells SKIP — the rig's standing shape |

## 4. STATUS LOG

- Lane open 2026-08-31 from `origin/lane/glm53-flash-bringup` @ a5d608b07 (fetched first);
  `origin/lane/glm5-moe-loc` @ 6e6120a0e absorbed by merge (its landing merge was in flight),
  rerere resolutions verified to keep both sides (FLAGS rows present, perf-ci union 1049 =
  1046 + 1044 - common).
- BUILT: the three `_rf` twins + the `_rf_redshift` red twin in qmatvec.cu; flag, counter,
  announce and the composed dispatch in `matvec_bf16_tcols_into` / `matvec_bf16_tcols16_into`;
  the gate-only allowlist launcher `matvec_bf16_tcols_gate_kernel_into` (redshift + t=1 arms);
  door-R gate in `glm5_matvec_doors_gpu`; FLAGS.md row + KERNELS.md rows and symbol recount
  (the running delta count had drifted to 323 vs a measured 314; recounted honestly, +4 = 318).
- Gate battery §3 all green same day (23 battery runs, all counts non-zero, run with
  `--include-ignored`); the landed base head (moe-loc merge + 98336b72d + perf-ci rows)
  absorbed post-battery, perf-ci.jsonl resolved by union (append-only law).
- PUSHED; no self-merge. The box window prices door R and owns the flip decision (a fresh
  bf16-trunk boot shows the `[bf16-tcols-red-fused] engaged` announce; the standing tcols
  gates in this file are the identity bar for that cell).
