# L2 (MEMRA_BF16_MMV + MEMRA_PP_BF16): box A/B plan (the flip condition)

The gate in this directory is exactness evidence only (green + red banked here). The flip
needs throughput receipts from the serving card class, and this is the protocol. The box is
busy; the window is requested from the owner, never taken.

## Arms, FACTORIZED: two flags, each carries its own receipt

Three arms instead of two, because the prefill door needs bf16 residency and residency has
its own decode consequence. Interleaved x5 per the interleaved-A/B law, fresh boot per arm,
same binary, same artifact, same PP3 placement as BOX-AB (cards 0/1/2, full expert residency,
`MEMRA_MOE_GROUPED_PREFILL=1` in EVERY arm: L1 is the floor now, not the variable):

- **Arm A (baseline)**: `MEMRA_BF16_MMV` unset, `MEMRA_PP_BF16` unset. The BOX-AB ON-arm
  config; expected TTFD ~10.25 s at 6,470 tokens.
- **Arm B (residency alone)**: `MEMRA_BF16_MMV=1`. Prices the MMV class on glm5: decode
  delta (its own owner-ratified near-tie class), prefill roughly neutral (the per-chunk
  expansion replaces the load-time f32 operand), VRAM receipt (dev0 f32 trunk ~15.6 GiB
  expected to drop to ~7.8 GiB; nvidia-smi per stage, banked).
- **Arm C (the door)**: `MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1`. The L2 term: the trunk-GEMM
  share of the ~9 s residual compresses toward the tensor-core class; the KDA scan (L3)
  remains.

The L2 flip claim is C vs B. The residency claim is B vs A. A C-vs-A row alone attributes
nothing and is not accepted.

## Workload

1. **TTFD + prefill tok/s** at the three real prompts (4,626 / 5,547 / 6,467 measured
   prompt_tokens, the on-box `prompts.json`, sha `de57a7a471f9b163...74b53e46`), streamed,
   `reasoning_effort` pinned low, `MEMRA_PREFIX_CACHE_MB=0`, TF32 off.
2. **The sampled vendor-default twin** (serving law: a request with NO sampling params).
   Spec-admission is off on this placement (pp-cross-device, eager); state the policy line
   rather than substituting a K>0 receipt, as BOX-AB did.
3. **max_tokens=1 first-token argmax gate**, C vs B AND B vs A, per prompt, 5 pairs each.
   Where any pair flips: the 8-draw vendor-default max_tokens=1 census per arm, proving the
   position soft or not. That bundle goes to the owner for the accept/hold call (the
   acceptance class ratified for glm5 on 2026-08-29 for MEMRA_BF16_MMV and
   MEMRA_MOE_GROUPED_PREFILL). No default flips from inside the window.
4. **Decode tok/s per arm** from the same rows: C vs B must be within noise (the door is
   m >= 16 only; a decode delta between C and B is a finding that stops the lane).
5. **`profile-prime-phases.sh`** (one arm-C pass, not part of the x5): the phase split that
   sizes L3's share before its kernel is built (`../prefill-gap-20260829/PREFILL-GAP.md`
   section 4).

## Engagement receipts (the step37 trap, both flags)

- `[bf16-tc] flag=on|off` prints at the first bf16-resident prefill GEMM in arms B and C
  (added this lane, both arms of the door); `[bf16-tc] ENGAGED m=.. n=.. k=..` per shape and
  a nonzero `bf16_tc_dispatches` delta are the door's engagement. Expected shapes: the KDA
  big four per layer at m up to 4096; count 4 x 34 KDA layers per full chunk plus any other
  bf16-resident 2-D projections (indexer, MTP glue) that cross m >= 16.
- Any `[bf16-tc] DECLINED` line is banked and explained; a C row whose log shows zero
  ENGAGED lines is the step37 "never ran" failure and is discarded, not averaged.
- `[bf16-mmv] RESIDENT` count per boot: 0 in arm A, a fixed nonzero census in B and C
  (glm5's expected members: 34 x kda_q/k/v/out, embed_tokens, lm_head, plus any >= 2M BF16
  2-D preserved tensors; bank the exact census from the first B boot).
- `[moe-grouped-prefill] execute` = 42 layers in every arm (the floor stays engaged).
- Boot identity per arm: fresh PID, `readlink /proc/pid/exe`, binary sha, one output sha per
  row per arm across the 5 boots.

## Numbers the A/B must produce

- TTFD per prompt length per arm, x5 interleaved, median [min..max]; prefill tok/s; decode
  tok/s.
- VRAM per device per arm (the residency claim's receipt).
- The argmax gate table (C-vs-B and B-vs-A per prompt) and any census tables.
- The `MEMRA_PRIME_PROF` phase split from the arm-C profiling boot.

## Decision rule (pre-registered)

The owner is handed: TTFD improvement C vs B at every length with non-overlapping x5
spreads, sampled twin healthy in all arms, engagement receipts per above, argmax gate green
or the census bundle on any flip, decode C-vs-B within noise. Defaults stay OFF until the
owner's accept/hold; a flip lands as its own PR updating the FLAGS row with the measured
rows.
