# step37 30k+ cell: the warm-turn-at-40k panic, its fix, and the affinity verdict

Box: vast 48343961 (2x RTX PRO 6000 Blackwell WS, TP2). Model: official Step-3.7-Flash
NVFP4 safetensors. Config: spec-on shipping policy (MTP heads=3, K=3, PMIN=0.5, PMIN0=1),
MEMRA_CTX=262144, vendor-default sampled (no sampling params), stream + usage receipts.
Prompts: two REAL documents (curve-30k-1: 39,546 tokens; curve-30k-2: 35,681), cold prime
then a short follow-up turn on the same session, interleaved x3 per arm.

## Run 1 (bin 71aa72ce = main 068cbc4253): found a fleet-fatal bug, A/B void

- Cold rows real: TTFT 33.32-33.66 s @39,546, decode 41-43 tok/s, acc 0.86-0.94, zero
  rewound/ILLEGAL/#87/lap.
- EVERY warm turn (6/6) panicked the GPU worker: cudarc slice_mut unwrap
  (`worker::admit -> spec_grow_and_rewind_to_checkpoint -> pp::restore_cache_checkpoint ->
  Engine::copy_u8_into`). Raw log: raw/panic-server-log-rnd1-aff1.log (two Engine-ready
  banners = in-process worker respawn; six admits; two panics).
- The aff0 arm never actually ran: its server OOM'd at boot against the dying respawn and
  the shared port let the aff1 server answer its health probe (arm-identity lesson, saved
  to memory as ab-arm-identity-not-liveness).

ROOT CAUSE: ring-backed KV layers carry ABSOLUTE `len` over a window-sized physical
buffer; the checkpoint restore into a freshly grown cache did a flat len-row copy, OOB
once the ring lapped. FIX: memra main c9a617ca99 (KvRing::restore_plan + ring-aware
restore in trunk and draft planes + try_slice_mut hardening). Gate: raw/ring-gate.txt —
the exact fatal shape serves (warm TTFT 2.27 s, panics=0) on bin bf9dbdb84f32.

## Run 2 (bin bf9dbdb84f32 = main + fix; corrected instrument, raw/long30k2.txt)

Instrument fixes: unique port per arm x round, pgrep-clear wait before every boot,
PID-verified boot after health-200, bracketed-basename pkill, per-arm panic/fullprime
counters. All 6 boots: rewound=0 illegal=0 trap87=0 lap=0 panics=0 fullprime=0.

| leg | affinity=1 | affinity=0 |
|---|---|---|
| cold 39,546 tok TTFT | 33.38 / 33.74 / 33.77 s | 33.20 / 33.27 / 33.06 s |
| warm turn on that session | **2.275 / 2.318 / 2.258 s** | 35.12 / 35.35 / 35.06 s |
| cold 35,681 tok TTFT | 29.43 / 29.60 / 29.69 s | 30.72 / 30.61 / 31.20 s |
| warm turn on that session | **2.321 / 2.360 / 2.307 s** | 32.02 / 31.07 / 29.32 s |
| decode tok/s (all legs) | 39.7-64.5 | 39.4-73.5 |
| acc (all legs) | 0.74-0.99 | 0.81-0.97 |

VERDICT: **affinity ON ships for long context.** Warm follow-ups on a 36-40k session are
2.26-2.36 s with affinity vs a 29-35 s full re-prime without it (15x), identical cold
TTFT, zero faults either arm. This was unmeasurable before the ring-restore fix (the
affinity warm path was the panic path).

## Soak thresholds this cell sets (30k+ class, vendor-default sampled)

- cold TTFT: <= 35 s at 40k, <= 32 s at 36k
- warm follow-up TTFT (affinity): <= 3 s
- decode: >= 39 tok/s at 36-40k context
- counters: rewound = illegal = #87 = lap = panics = fullprime = 0

## Cross-box note (unresolved)

The sbox dev box measured 15.31 s cold @39,546 on the same merged code (rank-spans
re-baseline, darklanes lane); this box measures ~33.5 s. Joins are 50% of long-prompt
TTFT and the boxes differ in interconnect; the 2x gap needs its own cell before any
public 32k TTFT claim (vLLM's receipted comparable: 11.2 s @32K).
