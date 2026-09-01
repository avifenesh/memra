# L2: the glm5 trunk prefill rides tensor cores (lane/glm5-tc-trunk-prefill, 2026-08-29)

STATUS 2026-08-30: the box A/B RAN and is banked (`BOX-AB-L2.md`): all five pre-registered
conditions PASS, argmax gate 30/30 with no flips, door TTFD -8.1..-8.7% at fixed residency,
MMV decode +18-22% and -10 GiB VRAM, and the 2-card full-residency boot SERVES (greedy
byte-identical to 3-card). Awaiting the owner accept/hold; no defaults flipped.

The prefill-gap plan's L2 lever (`../prefill-gap-20260829/PREFILL-GAP.md` section 1.2): after
L1 (grouped MoE prefill, 85 to 616-639 tok/s), the residual ~9 s of the 10.25 s TTFD at 6,470
tokens is the trunk, and the larger share of it is the BF16 trunk's f32, non-tensor-core
prefill GEMMs at the measured 15-20 TFLOP/s class (BOX-AB 2026-08-29, engine debt item 4).

## The door decision: open the EXISTING `MEMRA_PP_BF16` door, no new door

The alternative considered was a glm5-specific door: a load-time bf16 mirror stashed beside
the f32-resident trunk, consumed by `bf16_tc_gemm` at m >= 16. Rejected from the code, for
four reasons:

1. **The seam already exists and is battle-tested.** `linear_bf16_chunked_inner`
   (`lib.rs`) carries the door at exactly the right predicate (`m >= 16 && !exact &&
   canonical_chunk_rows.is_none()`), with the cuBLASLt alignment decline, the per-shape
   ENGAGED/DECLINED announces, and 47 step37 prime shapes of engagement history. A twin
   would duplicate all of it to reach the same kernel.
2. **bf16 residency is the checkpoint-faithful state, and it is how glm5 reaches the seam.**
   The artifact's precision split deliberately keeps every KDA projection in checkpoint BF16
   (`modules_to_not_convert`, the keeplist fix in `../BRINGUP.md`). With `MEMRA_BF16_MMV=1`
   the big four per layer (`kda_q/k/v/out`, 33.5M elements) load `GpuTensor::FloatBf16`, raw
   checkpoint bytes, and every prefill matmul flows to the door. The FLAGS row's "strictly
   closer to the checkpoint" argument holds by construction: weight bytes untouched, rounding
   enters only at the activation f32-to-bf16 cast. Only genuine BF16 checkpoint tensors ever
   become `FloatBf16` (the loader keys on the source dtype), so no upcast-then-downcast rider
   is possible.
3. **The mirror door points the wrong way on VRAM.** f32 residency costs 4 B/w (the ~15.6 GiB
   dev0 trunk that helped block 2-card residency, BOX-AB); a mirror ADDS 2 B/w on top. bf16
   residency REPLACES the f32 trunk at 2 B/w, roughly 7.8 GB saved, attacking the same debt.
4. **The mirror becomes dead code the day MMV pins.** The owner ratified the MMV near-tie
   acceptance class for glm5 on 2026-08-29; once the serving env pins `MEMRA_BF16_MMV=1`, a
   mirror door and the existing door would be two mechanisms for one disease.

So L2 is the flag pair, both default OFF, glm5's serving env pins them only after the owner
call: `MEMRA_BF16_MMV=1` (residency; decode moves under its own ratified receipts) and
`MEMRA_PP_BF16=1` (this lane's arm; decode provably untouched at fixed residency).

What stays out of the door's reach, verified in source:
- The mHC hyper walk: `hyper.rs` sites are raw `CudaSlice<f32>` (never a `GpuTensor`), and the
  mixes GEMM is `Engine::linear` f32 by the pre_exact contract. Structurally unreachable.
- The MoE router (288 x 4096 = 1.18M) sits under MMV's 2M threshold, stays exact f32; routing
  never moves.
- MLA `kv_b` splits to f32 3-D planes at load (the 2026-08-28 MLA arm); 3-D never enters the
  2-D residency arm.
- The low-rank pairs `f_a/f_b/g_a/g_b` (0.5-1.05M) and `b_proj` stay `Float` f32 under the 2M
  threshold: a small f32-GEMM residue (~0.9 TFLOP/chunk vs the big four's ~37) accepted this
  increment.
- Decode (m < 16), spec verify (`!exact`), and the step TP canonical-chunk ranks: excluded by
  the door predicate, and the decode exclusion is measured, not assumed (below).

## Engine changes in this lane (additive only)

- `f16_ffi.rs`: per-boot `[bf16-tc] flag=on|off` announce at the flag's first consult, printed
  in both arms (the moe-grouped-prefill announce pattern), and
  `BF16_TC_DISPATCHES`/`bf16_tc_dispatches()`, an engagement counter at the accepted-launch
  site itself (LAW:wiring-assertions-match-prose).
- New gate `crates/memra-engine/tests/glm5_bf16_tc_trunk_prefill_gpu.rs` (below).
- `docs/FLAGS.md` `MEMRA_PP_BF16` row: the glm5_next re-opening arm, same commit.

No dispatch predicate changed. A boot without both flags runs bit-for-bit yesterday's program.

## The gate, red-proven first (5090, TF32 off, `NVIDIA_TF32_OVERRIDE=0`, flock, 2026-08-29)

Fixture: the `kda_fixture_gpu` family (2 heads x 128, hidden 256) rebuilt as the REAL
artifact's mixed residency: `wq/wk/wv/wo` as `FloatBf16` raw bf16 bytes, everything else f32.
Reference: `memra_reference::kimi_delta_net_layer` fed the exact f32 expansion of the resident
bytes, so weights are value-identical and the band measures only the door's numeric config
(activation cast + tensor-core accumulate order).

GREEN (`gate-green.log`), all engagement counted at the invocation, 4 per mixer call:

| row | rel maxdiff | bar |
|---|---|---|
| mixer T=16 / 64 / 4096 vs reference | 3.526e-3 / 3.796e-3 / 3.653e-3 | 8e-3 (2.1x worst, calibration protocol of `kda_quant_operand_gpu`) |
| single wq GEMM m=16 / 64 / 4096 vs f64 host truth | 2.501e-3 / 1.701e-3 / 1.887e-3 | 8e-3 |
| decode m=1 / 2 / 15, flag ON vs the expansion program | BIT-identical, counter 0 | byte identity |

RED, each a wrong answer someone would ship:

| mutation | result |
|---|---|
| transposed weight bytes (layout swap, square 256x256) | rel 1.708e0, 57x above the 3e-2 floor |
| dropped activation cast: raw f32 bits fed as bf16, the exact byte stream a skipped `f32->bf16` convert hands the GEMM, reconstructed losslessly through the public entry (each u16 lifted to the f32 whose RNE bf16 cast is itself, so `bf16_tc_gemm`'s own convert reproduces the target bytes bit-for-bit) | rel 7.984e37 |
| door forced at m=15 (`bf16_tc_gemm` called directly) | bytes differ from the expansion program: the byte-identity comparator has teeth |
| decode-leak source mutation: `m >= 16` loosened to `m >= 1` in `linear_bf16_chunked_inner`, gate rerun | `[bf16-tc] ENGAGED m=1` and the decode byte-identity test FAILED as designed (`gate-red-decode-leak.log`), then reverted |

Sibling gates rerun green after the engine edits: `kda_fixture_gpu` 3/3,
`kda_quant_operand_gpu` 4/4.

## Decode byte-identity, the full statement

At fixed residency the door cannot move decode: the predicate starts at m=16, the gate
measures bit-identity at m in {1, 2, 15} with the flag ON, and the loosened-predicate
mutation proves the gate catches a leak. What DOES move decode is the residency prerequisite
itself: `MEMRA_BF16_MMV=1` changes decode's numeric class (its own FLAGS row, its own step37
argmax receipts, and the owner's 2026-08-29 glm5 ratification of that near-tie class). The
banked BOX-AB decode shas were measured MMV-off and die with the MMV flip, under MMV's
acceptance, not this door's. The box A/B below factorizes the two flags so each carries its
own receipt.

## Box A/B plan (needs an owner window; the box is busy)

Placement and protocol exactly as `../moe-grouped-prefill-receipts/box-ab-20260829/BOX-AB.md`:
PP3 across cards 0/1/2, full residency, `MEMRA_MOE_GROUPED_PREFILL=1` in every arm (L1 is the
new floor), `MEMRA_PREFIX_CACHE_MB=0`, `reasoning_effort` low, TF32 off, fresh boot per arm,
interleaved x5, the three real prompts (4626/5547/6467 tokens), greedy instrument plus the
sampled vendor-default twin, boot identity receipts (PID, exe, binary sha), per-arm announce
greps (`[bf16-tc] flag=`, `[bf16-mmv] RESIDENT` count, `[moe-grouped-prefill]`).

Arms, factorized so each flag gets its own attribution:

1. **Baseline**: MMV=0, PP_BF16=0 (the BOX-AB ON-arm config). Expect TTFD ~10.25 s at 6,470.
2. **MMV alone**: MMV=1, PP_BF16=0. Prices residency: decode delta (the MMV class), prefill
   delta (the per-chunk expansion replaces the load-time f32 GEMM operand, roughly neutral),
   VRAM receipt (dev0 trunk ~15.6 GiB to ~7.8 GiB expected).
3. **The door**: MMV=1, PP_BF16=1. The L2 term: expect the trunk-GEMM share of the ~9 s
   residual to compress toward the tensor-core class; the KDA scan (L3) remains.

Gates per the pre-registered flip condition: TTFD improves at every length x5 non-overlapping
(arm 3 vs 2 for this door), sampled twin healthy, engagement receipts both ways
(`bf16_tc_dispatches` per boot, ENGAGED shapes list, DECLINED list empty or explained),
max_tokens=1 first-token argmax gate arm 3 vs arm 2 AND arm 2 vs arm 1 on all three prompts,
8-draw vendor-default census on any flipped position, decode tok/s within noise between arms
2 and 3. Then the whole bundle goes to the owner for the accept/hold call; no default flips
in this lane.

Also to bank in the same window (already-written script): `../prefill-gap-20260829/
profile-prime-phases.sh` on arm 3, so L3 (the KDA chunked scan) starts from an attributed
share.

## Named follow-ups (not this lane)

- One activation convert shared across the KDA q/k/v trio (a `_pre` twin for bf16, the
  `matmul_group` f16 pattern): saves 2 of 3 converts per layer-chunk; the GEMM win does not
  depend on it.
- The sub-2M f32 residue (`f_a/f_b/g_a/g_b`, `b_proj`): ~30-70 ms/chunk arithmetic; revisit
  only if the profile says it matters post-L2.
- L3: the chunked KDA scan (per-channel `Gcum` algebra, reference already banked).
