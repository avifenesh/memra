# moebatch-q35moe — batched MoE decode campaign (2026-08-21)

Owner: make ornith15 release-ready vs other engines. Baseline (512-tok, one PRO 6000,
quiet): vLLM single 250.4 / c8 aggregate 1174; memra spec-adapt single 226.6 / c8 714.

## Increment 1: NVFP4 CSR owner-scan gate_up (SHIPPED on lane)

Diagnosis: the CSR expert-dedup arm (default-on 2026-07-10) was qtype-gated to
IQ4_XS/IQ3_S — NVFP4 banks fell to the rows program (weight re-read + re-decode per
(token, slot) pair). B=8 physics: 64 draws over 256 experts ≈ 56.5 distinct → dedup worth
~12% of routed traffic; the deeper wall is decode ALU (the iq4 down8 "47% of byte-math
wall" class), not bandwidth.

New `moe_gate_up_silu8_dev_q8_csr_nvfp4`: owner-scan skeleton, per-(expert,o,group) decode
cached (8 lookup ints + 2 UE4M3 sub-scales per projection), per-pair arithmetic replays
`expert_dot_nvfp4_g` expression-for-expression. Host: kernel pick by qtype, csr gate +=
NVFP4 + uniform-g/u requirement.

Qualification:
- MEMRA_MOE_CSR=2 byte-compare across the full run-spec battery: **0 mismatches**.
- run-spec K=1..8: PASS (spec ≡ plain), every K.
- decode-batch: B=4 485→510 (+5.2%), B=8 594→**634.5** (+6.8%) — matches the dedup math.
- run-spec K=3 CLI: 205 tok/s (1.00x plain).
- **Serve single-stream (512-tok, adapt+pmin recipe): 243-249 tok/s — vLLM PARITY (250.4).**
- Serve c8 aggregate: 718-723 (+1% — dedup is decode-share-diluted at serve; the aggregate
  gap is the next increments' target).

## Increment 2: prefill MoE — NVFP4 admitted to the f16g grouped-GEMM lane (SHIPPED on lane)

Owner redirect ("pairaty not enogh and need to messure with few prompts with cache in
action and larger context"): the realistic cells (agentic session, shared-prefix c8) lose
on PREFILL, not decode — cold 14.7k prime 8.47s vs vLLM 2.32s.

Diagnosis chain (box3, one RTX PRO 6000 Blackwell):
- `MEMRA_PRIME_ANATOMY=1` (new diagnostic, this lane): t=14715 forward_last cumulative ms
  attn_full=273.8 / gdn_linear=433.5 / **moe=5794.0 (88.6%)** / norms_adds=33.9.
- pp throughput FLAT in T (2400 @ 2.4k → 2330 @ 14.7k) and MEMRA_PRIME_CHUNK-invariant →
  linear per-pair ALU wall, not attention and not launch shape.
- Root cause: the ornith15 expert bank is uniform NVFP4 (GGUF type 40, all 41 layers,
  census receipt) — it passes the pairs q8 gate but missed BOTH batched prefill doors:
  `use_mma` (q8_expert_dec = IQ4_XS/IQ3_S/Q4_0 only) and `f16g_proj_ok` (no NVFP4 dequant
  kernel). Every prefill expert dot rode the per-pair `_em` fallback with zero token
  reuse — the exact class the f16g lane fixed for Q4_K (3.14x, research/q4k-expert-prefill-20260802).

Fix (the round-49 coverage pattern): `dequant_nvfp4_f16_kernel` in cu/moe_f16_grouped.cu
(per-value port of the memra-gguf CPU oracle: UE4M3 half-scale + doubled-e2m1 table) +
`f16g_proj_ok` admits QT_NVFP4 at in_f % 64 == 0. Workspace-dequant + sk visitor path;
no direct tile loader yet (that is the follow-up if the dequant pass shows hot).

Qualification (box3, N=3 pp reps):
- **pp14715: 2331.4 → 10,784.7 tok/s = 4.63x** (10760.6/10784.7/10851.4). MoE stage
  5794 → 637 ms; the prime wall is now GDN linear (436 ms) + attention (272 ms).
- run-gen argmax gate: short prompt MATCH (prefill==decode and prime==tokenwise).
- 14.7k prompt: 1 last-position flip at top-2 margin 0.0090 — ran the CALIBRATED
  instrument `tools/argmax-margin-gate.sh --prompt ctxdoc`: **PASS, flips=1 bad=0**
  (margin 0.0090 < config delta 1.9905; zero wide-margin flips). F16G=0 rollback arm
  MATCHes the same prompt — the flip is the documented near-tie coin class.
- run-spec K=1..8: SELF-CONSISTENCY PASS every K (spec ≡ plain).
- serve-smoke: **0 failed** on the ornith artifact (chat/stream/concurrency/greedy
  determinism/prefix-metering). Spec-draft, gemma4, and Q35-coldhol arms SKIPped —
  those models are not staged on box3; re-run those arms in the pre-merge battery.

## Increment 3: message-boundary prefix seed (SHIPPED on lane)

A cold chat session's only insert was the full-prompt seed; its trailing generation header
diverges from every later render of the same history, and hybrid restore is entry-end-only —
so the very next turn full-re-prefilled (cachecell turn-1: 3.92s) and shared-prefix peers got
zero hits. On a cold miss, `first_message_boundary` arms the EXISTING snapshot_at/capture_at
boundary stop at the end of the first message (render with add_generation_prompt=false, exact
token-prefix verified, clamped PRIME_MIN_T from both prompt ends — the sub-floor tokenwise
door stays shut). Receipts: session turn-1 3.92 -> 2.13s (whole-entry boundary hit),
sharedc8 208.8 -> 480.5 (all 8 peers hit the 4647-token entry). serve-smoke 0 failed.

## Increment 4: batched filtered device-sample stats (SHIPPED on lane)

Vendor sampling (T=0.6/top_p=0.95/top_k=20) cost 31% of c8 aggregate vs greedy (478 vs
700-716, both orders) — NOT the filter kernel: each filtered row paid its own HtoD + 3 tiny
allocs + a serial grid-1 filter_stats launch per tick. Grouping rows by knob tuple into ONE
nrow=F launch: sampled c8 478 -> 640-678. A 3-pass top-K filter_stats form was implemented
and REFUTED (row is L2-resident; per-thread selection list spills to local: engine 12.8/11.2
ms vs 10.4 binary-search at B=8) — deleted, refutation at the dispatch site. sample-check
ALL GREEN both forms; serve-smoke 0 failed. New probe: MEMRA_DBB_SAMP on decode-batch-bench.
Stale-binary trap re-confirmed: `cargo build -p memra-server -p memra-engine --bin
sample_check` builds ONLY the named bin — three serve A/Bs ran an old server before this
was caught (the draftcost lane's lesson, relearned).

## Cachecell scoreboard (box3, vendor sampling, one RTX PRO 6000 Blackwell)

| cell | campaign start | after incr. 2-4 | vLLM 0.27.1 |
|---|---|---|---|
| pp14715 (engine) | 2,331 tok/s | 10,785 tok/s | — |
| session 8-turn total | 34.39 s | 20.89 s | 14.99 s |
| session warm-turn walls | 2.6-3.4 s | 2.1-2.6 s | 1.7-1.9 s |
| sharedc8 aggregate | 151.5 tok/s | 661.8 tok/s | 983.5 tok/s |

Remaining: session cold turn 3.98 vs 2.32 (prefill still ~2-3x off vLLM's; GDN 436ms +
attn 272ms now dominate the prime), warm-turn decode at 15-17k ctx, c8 batched-decode
engine ceiling (greedy serve ~700, engine B=8 ~800 at ctx 4700 vs vLLM 985).

## Two-rig battery (2026-08-21, pre-merge)

- box3 (4x RTX PRO 6000 Blackwell, single-card runs): kernel-check ALL GREEN (85 cells),
  sample-check ALL GREEN, run-gen argmax MATCH + argmax-margin-gate PASS on the 14.7k
  prompt, run-spec K=1..8 PASS, serve-smoke 0 failed (spec/gemma4/Q35-coldhol arms SKIP —
  models not staged there).
- local RTX 5090: kernel-check ALL GREEN (107 cells), sample-check ALL GREEN, run-gen
  argmax MATCH + run-spec PASS on Qwen3.8-27B-NVFP4, serve-smoke **0 failed including the
  Q35 mixed c=4 exact-token gate** (the arm that killed the old grouped dispatch) and the
  spec + gemma4 arms.
- 5090 coverage note: no NVFP4-expert MoE artifact exists on the local rig (the local NVFP4
  27Bs are dense; local MoEs are IQ4_XS/Q4_K — table rows this lane did not touch), so the
  f16g-NVFP4 admission's 5090 surface is unexercised there. On a 24GB card that class is a
  spill regime anyway; qualify it when an NVFP4-expert artifact targets the 5090.

## Next increments (aggregate 0.61x -> 1.0x)

2b. NVFP4 direct sk tile loader (dequant-in-register, kills the 537MB/proj workspace
   pass — the kq-direct 41.8% precedent). Also: NVFP4 expert-dot wide-load for decode
   (the iq4 down8 precedent). Applies to rows + CSR + down bodies.
3. GDN at batch (29% of B=8 tick) — now also 6.6%→dominant share of the PRIME wall
   (gdn_linear 436ms vs moe 637ms at 14.7k): batched projections fine; per-row state ops
   and the out block are the candidates.
4. Serve logits D2H + host split at batch (10.4% bench; device-sample covers greedy — audit
   what still crosses the bus per tick).
5. Prefix-cache: entry-end-only restore makes every full-prompt insert unusable for the
   next turn (2-token generation-header tail divergence) and for shared-prefix peers —
   message-boundary checkpoint insert is the fix (server renders the template, knows the
   offsets; engine snapshots GDN state mid-prefill).
