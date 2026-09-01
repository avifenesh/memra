# bank-defect consolidation: main's QT_NVFP4_V2 fix merged into the glm5.3-flash bringup

**Lane**: `consol-bankfix` -> `lane/glm53-flash-bringup`
**Merges**: `origin/main` (`d3ac87f80`, the step37 bank-v3 milestones 1+2) x
`lane/glm53-flash-bringup` (`216ffd114`, through the dedup merge and its perf-CI row)
**Box**: box B, 4x RTX PRO 6000 Blackwell Workstation Edition 96 GB, as a QUEUED window behind the
1M depth re-price.

## 1. What this merge carries, and why it needed its own gate window

Two independent feature sets land in one tree:

* from **main**: the `QT_NVFP4_V2` `in_f` scale-fetch fix. `kq_fetch`'s `int in_f = 0` default is
  REMOVED so the compiler enforces it at all ten call sites, and `moe_kq_sktail_kernel` passes
  `in_f` on its `kb+1` prefetch pair. Only the v2 branch reads `in_f` — it locates the slot-major
  row's UE4M3 scale tail at `n_slots*16` — so the two defaulted callers had been fetching the scale
  from inside the packed-codes region on every k-block but `kb=0`: right weights, wrong
  per-16-element scale, in a PREFILL kernel (`research/step37-bankv3-20260901/DIAGNOSIS.md`).
* from the **bringup lane**: the expert-slab dedup doors (transposed vrows visit schedules, doors
  E / E-down, both default OFF), on top of the matvec, moe-loc, ep-diet and verify-batch work the
  lane already carried.

The triage that opened this lane established that **every glm5 shape is CLEAN of the defect** — the
glm5.3-flash serving program does not take the two defaulted call sites — so this merge is not a
corruption repair for glm5. It is the consolidation that lets the lane sit on top of main's fix
without either side regressing the other, and the reason it needs a box window rather than a rig
run is that the defect's home is the PREFILL grouped GEMM at real geometry.

## 2. Merge mechanics

`git merge` of `216ffd114` into the already-resolved `59f45b989`/`b83efaeb2` pair produced **no
conflicts**. Three code files were touched by both sides
(`cu/qmatvec.cu`, `src/hybrid_forward.rs`, `src/lib.rs`) and three docs
(`docs/FLAGS.md`, `docs/KERNELS.md`, `research/INDEX.md`); every hunk pair was disjoint.

Clean auto-merges are the ones worth checking, so the unions were counted rather than assumed:

| file | merge base | + side A (main) | + side B (dedup) | merged | union exact? |
|---|---|---|---|---|---|
| `docs/FLAGS.md` (`MEMRA_*` rows) | 751 | 759 | 753 | **761** | yes (751 + 8 + 2) |
| `docs/KERNELS.md` (table rows) | 74 | 77 | 76 | **79** | yes (74 + 3 + 2) |
| `research/INDEX.md` (lines) | 446 | 449 | 447 | **450** | yes (446 + 3 + 1) |

No flag name is newly duplicated (14 names appear in two sections both before and after, which is
the detail-table/defaults-table pairing, not a merge artifact). No lane row is duplicated in
`research/INDEX.md`. Both sides' new flags are default-OFF doors and they name disjoint flags, so
no default was flipped twice.

**The one semantic question a textual merge cannot answer** — whether the dedup lane's new kernels
carry the same defect class the fix hardened — was checked by reading them: `_ord` / `_tmaj` take
`in_f` as an explicit parameter and derive `nsb = in_f >> 5` from it, and their shared
`expert_dot_g` helper has no `QT_NVFP4_V2` branch at all (it dispatches v1 `QT_NVFP4` and the
k-quants, returning `0.0f` for anything else, with the callers gating on qtype). So the new code
never reads the v2 scale tail and cannot inherit the hole.

## 3. The oracle with a corruption arm

See `receipts/run-oracle-corruption-arm.sh`. The rig's oracle binary did not track kernel source,
so a green `nvfp4-bank-oracle` on this tree would have been a void gate. The receipt is therefore a
SEQUENCE, and only the sequence is evidence: corrupt the v2 scale read (`*1.5f`), build, all four
tile-form arms must FAIL; revert; rebuild; the binary sha and the nvcc object md5 must both DIFFER;
all four arms must then PASS.

Run on the FINAL tree (`61dcf202c`), box B, rustc 1.98.0, nvcc 12.8 V12.8.93 (this box carries only
CUDA 12.8 — the bank-v3 lane's own arms were CUDA 13.2 on a different bench box, so the fix is now
receipted under two toolchains), arch 120a, cards 0-3 verified at 1 MiB before and after:

| state | binary sha256 | kernel `.o` md5 | hybrid | sk128 | sk32 | tail0 |
|---|---|---|---|---|---|---|
| **CORRUPTED** (`*1.5f`) | `7680adc595d1d6d0…` | `e16c81d76d31a06f…` | **exit 1** | **exit 1** | **exit 1** | **exit 1** |
| **TRUE FIX** | `88f422de01d943d5…` | `924928c3b3d2aa6b…` | exit 0 | exit 0 | exit 0 | exit 0 |

Both shas differ and so does the nvcc object, so this build tree provably tracks its own kernel
source: the void-gate trap is closed by measurement, not by assumption. The corrupted arms deviate
on **100% of elements** in both geometries (`gate_up` 327,680/327,680, `down` 2,097,152/2,097,152)
at `maxrel = 5.000e-1` — exactly the 0.5 relative error a 1.5x scale must produce, so the arm
detects the injected defect rather than merely something. Every cell reports `nonzero_v1 = elems`
and `finite = true`, so no row is a comparison of zeros.

Unlike the bank-v3 PRE-FIX twin (which passed on `sk128`/`tail0` and failed only the deep-tail
arms, localizing the defect to `moe_kq_sktail_kernel`), this corruption is on the shared v2 scale
read, so all four tile forms must see it — and all four did. Run 1's arms (a different binary,
before the section-5 fix relinked it) are in `run1-ppregression/` and were equally valid:
corrupt `6d3d0f83cd74eebd…` FAIL 4/4, fixed `be44f028f68b2a3c…` PASS 4/4.

## 4. Gates

All on `61dcf202c`, box B, four RTX PRO 6000 Blackwell WS at 600 W.

| gate | arms | result |
|---|---|---|
| standing GPU battery — 27 suites, doors PINNED `=0` (ship arm) | 27 | **PASS**, 176 tests, every suite non-zero |
| dedup compose arms on the walk suites — E, E+D+H, down-only | 18 | **PASS** (52 tests per class) |
| `glm5-tp-gate` A-H5 + M + T + R1-R4 (incl. 4 RED arms that must diverge) | 1 process, 54 pass lines | **ALL ARMS PASS** |
| `glm5-spec-ppn-gate` ladder (n2/n3, even/split1/split3/asym x streams/overlap + 5 dedup compose arms) | 13 | **PASS**, 23 pass lines each |
| `glm5-hyper-ppn-gate` ladder (n2/n3/n4 incl. shard0, longer prompt, 2 compose arms) | 12 | **PASS**, 6 pass lines each |
| `glm5-hyper-batch-gate` ladder (B<=15 x ppn 1/2/4, 2 compose arms) | 12 | **PASS**, 3 pass lines each |

**46/46 battery arms and 37/37 ladder arms.** `--include-ignored` throughout (never `--ignored`),
a non-zero passed count asserted per suite, pass-line counts asserted per ladder arm (23/6/3), and
capture-then-gate on every failable step. On this box the n2/n3/n4 arms place stages on DISTINCT
devices, which is wider than the banked one-card rig arms rather than a substitute for them.

CPU side (rig, `tools/local-ci.sh --perf`, same sha): correctness stage GREEN — kernel-check 107
cells, decode-batch-gate 4 arms (9B NVFP4 + Q8_0), graph-warmup-stress 10 cycles x 4 arms plus its
injected-corruption canary, serve-stress-gate c=64, spec-on-cache-hit — and perf stage 0 fail /
0 warn (`qwen9b-plain-short` 138.17 tok/s). That row is `window_clean=false` and says so: a
non-memra CUDA process has been co-resident on the rig for hours and did not leave inside the
600 s wait, which is the dirty-window precedent, not a clean measurement.

## 5. The merge was red, and the window is the only reason we know

Run 1 (`run1-ppregression/`) passed the oracle and 45/46 battery arms and then failed **30 of 37
ladder arms plus `glm5-tp-gate` arm G**, every one of them at load with

    pipeline placement is unsupported for plan operations [...];
    blockers=[KimiDeltaNet, RecurrentState, SwiGluPreClampedActivation, HyperConnections,
              LatentMlaAttention, SparseIndex, LatentKvState]

— the entire glm5_next trunk operation set. Every arm with `stages>1` failed; the seven that passed
are exactly the `ppn=1` arms, which place no pipeline. Arm G is the same cause wearing a different
hat: it asserts that TP + `MEMRA_PP_STAGES=2` refuses AT LOAD **by name**, and the refusal it got
was this generic one instead of the TP-composition message, so the arm reported "refused with the
WRONG error".

Attribution was measured on both parents, not inferred:

* the admission refusal exists on main `d3ac87f80` and NOT on the bringup `216ffd114`;
* main carries no `glm5_spec_ppn_gate.rs` at all, so it could never have run these ladders;
* `pipeline_support` is byte-identical on both sides (20 operations each), so the lane never
  touched the table — it had no reason to, the check wasn't there.

**Neither parent was red alone.** An operation-level admission gate is only as wide as its table,
and a table cannot know about an architecture whose gates live on another branch. Left unfixed, this
merge would have made PP4 splits 13,26,39 — the only demonstrated 1M-context posture, run on this
same box the same day — refuse before the first shard upload, along with every PP-N glm5.3 serving
shape.

Fixed in `61dcf202c` at operation level (the manifest's own stated granularity), modelled on the
`GLM5_SPEC` table beside it, with the covering ladders named per class and the exclusions written
down — `SharedSparseIndex` deliberately NOT admitted, because the ladders' fixture plan carries
`SparseIndex` and admitting the shared variant would claim coverage nothing ran. Gemma's separately
gated PP2 program keeps its fence-length grandfather untouched. The whole window (oracle included,
because a receipt is bound to a binary and the binary moved) was re-run on the fixed tree; that
re-run is the table's receipt.

## 6. Debt paid in-lane

The merge brought 45 clippy warnings from main's incoming code into this lane. They are ours once
merged, so they are fixed here and not filed: 26 real fixes (including one that converts a latent
divide-by-zero panic into a named refusal) and 8 targeted allows in the house pattern with the
house wording. Receipt: `cargo clippy --workspace --all-targets` = 0 warnings,
`cargo fmt --all --check` clean, and the lint gate itself proven able to fail by a deliberately
injected `1u64 as u64`.
