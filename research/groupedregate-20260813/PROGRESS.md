# MEMRA_MOE_GROUPED PRO-pair stale-verdict re-sweep

Branch: `lane/cx-groupedregate`

Frozen base: `v0.81.2` / `18885ec479d897a3e8c42b0d408a71fa3edaa708`

Target rig: box1, one free physical NVIDIA RTX PRO 6000 Blackwell Server Edition card

Coordination: `flock /tmp/memra-gpu-1.lock` plus an explicit `CUDA_VISIBLE_DEVICES=<free physical index>`

## Scope and stop condition

This lane re-tests the stale `MEMRA_MOE_GROUPED` default-off verdict on the PRO-class serving
target. It will preserve raw logs under `raw/`, write `RESULTS.md`, commit the evidence, return the
selected physical GPU to 0 MiB with no surviving lane processes, and stop. It will not merge, tag,
push, edit the generated performance board, touch the live serve box, or flip the default.

## Read-first Lever C contract

`research/leverC-20260808/` was audited before this file was created: all 98 files (684,441 bytes)
were traversed, all three committed checksum manifests verified, and the mechanism, gate drivers,
summaries, individual KAT transfer runs, correctness receipts, and failure history were inspected.

The grouped path is host-routed prefill dispatch. It keeps the established router-logit decision and
host routing oracle, buckets routed rows by expert while preserving the original top-k slot, batches
the expert work, then scatters and reduces in the original slot order. On resident uniform Q8 banks,
unclamped layers use the row-batched twins of the sequential fused program and clamped layers use
pair-major copies of the literal per-row program plus the clamp-aware activation. Mixed, spill, or
remote-slab banks remain on metadata-aware per-expert grouping; this lane must never send mixed
metadata through resident-slab, pointer-table, pairs, dev-router, or grouped-decode uniform-only
kernels.

The original local transfer artifact was KAT-Coder
`Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`, resident on one RTX 5090 Laptop GPU. It measured the
frozen `pp2048` prompt from `research/depth-decode-20260802/depth-2048-kat.txt` (SHA-256
`dc91551b1e83414616ebb8d65ee88d1af1fc8792dadffe0208b732d094adfc0d`) through
`concat-prime-probe ... ppprime`.

The original gate definition is quoted verbatim from
`research/leverC-20260808/raw/5090/perf-kat-summary.log`:

> `protocol=N=5 independent processes per arm; adjacent interleaved pairs; order alternated; one warmup plus one timed prime per process; one flock hold`
>
> `throughput_assertion=FAIL grouped_slower_or_run_failed`

Accordingly, the unchanged transfer threshold is no regression: all ten processes must exit zero
and the grouped median must not be slower than rollback. Pairwise wins are reported alongside the
medians. The 5090 result was rollback 4027.1 tok/s versus grouped 992.7 tok/s, -75.3%, with 0/5
pairwise wins (N=5; 57-63 C, 1590-1597 MHz).

The earlier PRO-pair Step35 result is context, not a substitute for the transfer re-run: at current-
then commit `41f0af6f`, grouped beat rollback by +53.3% at the pp512 class, +58.9% at the pp2048
class, and +63.4% at pp4096 (N=5 per arm; 36-43 C, 2272-2370 MHz). The current code has moved, so
this lane will remeasure rather than carry that verdict forward.

## Frozen execution order

1. Inspect box1 with `nvidia-smi`, process ownership, locks, and the lane inbox. Identify the
   physical card held by `cx-cachesize`; select the other card. If both cards are busy, record the
   state here and wait rather than collide.
2. Build the exact `v0.81.2` lane tip in an isolated box1 checkout, explicitly pinned to the selected
   physical GPU.
3. Under one `/tmp/memra-gpu-1.lock` hold, re-run the same resident KAT `pp2048` transfer gate with
   N>=5 adjacent interleaved, alternating grouped ON/OFF arms.
4. In the same disciplined regime, measure frozen-shape prefill and serving TTFT for the served
   Qwen3.6-35B-A3B artifact, plus Step35 if its artifact is present: cold TTFT and mixed hit/miss TTFT
   at c=4 and at the measured/frozen knee.
5. With grouped ON, require `kernel-check` ALL GREEN, `run-gen` argmax MATCH for each affected
   on-box model, `run-spec` K=1..8 PASS, and `serve-smoke` zero failed. Run grouped-OFF byte-identity
   comparisons wherever a harness supports them. Any mismatch stops performance promotion.
6. Produce the plain verdict `FLIP-READY`, `STILL-BLOCKED`, or `NEEDS-MORE`. If flip-ready, include
   the exact un-applied default-flip and `docs/FLAGS.md` rewrite with rollback seam
   `MEMRA_MOE_GROUPED=0`.

## Timeline

- 2026-08-13: Confirmed the local worktree is clean, on `lane/cx-groupedregate`, exactly at tag
  `v0.81.2`. Completed the read-first Lever C audit. No remote GPU action has started.
- 2026-08-13: Created this file as the first artifact under `research/groupedregate-20260813/`.
  Verdict: pending measurement.
- 2026-08-12T22:03:21Z: Built the isolated box1 checkout at exact source
  `18885ec479d897a3e8c42b0d408a71fa3edaa708` / `v0.81.2` with CUDA 13.2 and auto-detected
  `sm_120a`. Staged KAT onto box1 local NVMe; local and staged SHA-256 both equal
  `e35e23219a81590b9d4174eea4717d716dd62676c8c434f6b708f49a07310e1a`.
- 2026-08-12T22:06:47Z: The orchestrator assigned this lane the second physical card and told
  `cx-cachesize` to take the card this lane is not using. Selected physical GPU index `1`, UUID
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`. The selection snapshot showed both cards at 0 MiB
  and 0% utilization, with no compute applications; the scored driver will re-check the selected
  card after acquiring `/tmp/memra-gpu-1.lock` and abort rather than overlap any foreign kernel.
  Physical GPU 0 / UUID `GPU-54dd2b6f-9311-dd31-672b-60be2ed28a79` is reserved for the co-tenant.
- 2026-08-12T22:11:58Z: The pre-launch snapshot found `cx-cachesize` on physical GPU 0 at
  26,451 MiB and 100% utilization. Physical GPU 1 was free at 0 MiB and 0% utilization, so this
  lane retained GPU 1. The scored pre-lock snapshot at 22:12:59Z again showed the co-tenant only
  on GPU 0 (24,211 MiB, 100%) and selected GPU 1 free (0 MiB, 0%).
- 2026-08-12T22:12:59Z to 22:14:33Z: Re-ran the frozen resident-KAT pp2048 gate under one
  `/tmp/memra-gpu-1.lock` hold, with `CUDA_VISIBLE_DEVICES=1`, N=5 per arm, adjacent interleaved
  pairs, and alternating order. All ten processes exited zero. Rollback samples were 8195.7,
  8193.7, 8191.7, 8221.2, and 8188.7 tok/s; grouped samples were 2688.0, 2687.5, 2687.7, 2687.5,
  and 2687.5 tok/s.
- 2026-08-12T22:14:33Z: The unchanged transfer threshold **failed on the PRO card**: rollback
  median **8193.7 tok/s** versus grouped median **2687.5 tok/s**, **-67.2%**, with **0/5** grouped
  pairwise wins. Each median is N=5 under the same one-lock thermal regime: continuous 250 ms
  telemetry covered 27-48 C, with 2295-2422 MHz at >=50% utilization.
- 2026-08-12T22:14:33Z to 22:19:40Z: Grouped-ON `kernel-check` was ALL GREEN for KAT (105 cells,
  5 skipped) and Q35 (107 cells, 5 skipped). Grouped-ON `run-gen` reported both argmax checks
  MATCH and 200/200 `BYTE-IDENTICAL` MoE-oracle rows for each model, with zero `MISMATCH`; explicit
  grouped-OFF runs reported the same argmax signatures. Q35 grouped-ON `run-spec` passed
  self-consistency at every K from 1 through 8.
- 2026-08-12T22:21:06Z: `serve-smoke` under grouped ON exited 1. Its direct failure lines were:

  > `FAIL: Q35 mixed c=4 exact-token regression`
  >
  > `serve-smoke: 1 failed`

  The machine row recorded expected completion length 60 but all 20 requests stopped at 25
  tokens. The cause was not diagnosed or attributed: the mandatory mismatch rule stopped the
  campaign before an OFF diagnostic, Q35 performance/TTFT, or Step35 work.
- 2026-08-12T22:23:16Z: Released `/tmp/memra-gpu-1.lock`. Selected GPU 1 returned to 0 MiB, 0%,
  27 C, and 180 MHz, with no lane-owned processes, ports, or locks. GPU 0 retained only the
  expected `cx-cachesize` co-tenant.

## Final verdict

**STILL-BLOCKED.** On physical GPU 1 of box1's RTX PRO 6000 pair, the same transfer gate failed
at 2687.5 versus 8193.7 tok/s (-67.2%, N=5 per arm, 0/5 pairwise wins; 27-48 C,
2295-2422 MHz at >=50% utilization, one lock hold). The required exactness battery also ended
with the quoted `serve-smoke` failure under grouped ON. The first failure alone blocks the default
flip; the second independently forbids promotion. See `RESULTS.md` and `raw/` for the full report
and receipts.
