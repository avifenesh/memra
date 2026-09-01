# Vast trial serving — perf receipt (2026-08-10)

Box: Vast 47297516, 2x RTX PRO 6000 WS (Max-Q class, 300W), direct SSH <vast-box-ip-2>:39411.
Binary: memra-server on restructure/public-split hardening tip, IQ4_XS Step-3.7-flash + Q8_0 MTP.
Serve env: PP_STAGES=2, PP_DEVICES=0,1, CTX=262144, PREFIX_CACHE_MB=2048, MOE_GROUPED=1.
Probe: single-stream, temperature=0, streaming, through /v1/chat/completions on localhost.

| shape            | TTFT     | decode      | notes |
|------------------|----------|-------------|-------|
| short (3 runs)   | 0.184s   | ~74.5 tok/s | "Say OK.", 32 max_tokens, tight run-to-run |
| 4k cold prompt   | 6.938s   | 46.4 tok/s  | ~4k-token prompt, cold (no prefix hit) |
| depth 250-word   | 0.167s   | 72.8 tok/s  | 400-token generation, streams at reading speed |

Soak: 1 req/20s stability loop, iter 67+ clean, TTFT ~0.15s, zero errors.

Read vs serve-ready bar (serve-ready-bar-20260808):
- short sub-second-class: PASS (0.18s).
- 4k single-digit-s: PASS (6.94s, under 10s line).
- streams at reading speed: PASS (72-74 tok/s single stream >> reading pace).
- Note: 4k-cold 6.94s here (MOE_GROUPED=1) beats the naked pp2pipe box1 verdict of
  9.771s p50 — live cross-box datapoint favoring grouped serving; the box1 cx-grouped
  A/B is the controlled same-box test.

Caveat: Max-Q 300W cards, single-box, single-stream. Concurrency (c=4/8) and
interleaved pod-reference comparison still pending for the serve-home flip table.

## Concurrency addendum (same box, same config, 2026-08-10)

| shape                       | wall    | TTFT p50 | agg tok/s | errs |
|-----------------------------|---------|----------|-----------|------|
| c=4 short (128 gen)         | 5.64s   | 0.39s    | 91.5      | 0    |
| c=8 short (128 gen)         | 8.58s   | 0.55s    | 120.3     | 0    |
| c=4 4k cold, distinct salts | 28.95s  | 26.36s   | 9.0       | 0    |

Short-prompt concurrency scales cleanly. Distinct-prefix 4k cold primes serialize
(~4x the solo 6.9s prime) — live trial-box confirmation of the concprefill
saturation verdict ("new per-prime compute mechanism, not more scheduler
engineering") and the trial-plan-v2 QoS/felt-speed risk. Same-prefix fanout is
covered by prefix dedup and does NOT hit this path; the exposure is simultaneous
cold long-context requests from different tenants.

## Decode ladder (sustained 512-tok generations, distinct salts, single run each)

| c | agg decode tok/s | wall |
|---|------------------|------|
| 1 | 77.3             | 6.8s |
| 2 | 105.9            | 9.9s |
| 4 | 121.5            | 17.2s|
| 8 | 135.4            | 30.8s|

vs box1 (Server Edition class) ladder 88.5/118.2/146.1/166.5: Max-Q trial box lands
~81-87% of box1 decode at every rung — the power-cap tax is uniform, not shape-specific.
Trial-plan-v2 Shape-B breakeven at reserved pricing needs 165 tok/s decode; this box
delivers 135 at c=8 on-demand — reserved-class pricing or one more decode step covers it.

## Cache-hit TTFT (4k prompt, same salt)

cold 6.547s -> warm hit 0.015s / 0.015s (repeatable). Prefix cache fully live on the
trial box; 4k hit is 15ms — the felt-speed story for returning sessions is intact.

## Serving-config update applied (2026-08-10, post cx-throughput merge)

Applied the throughput lane's trial-tier recommendation to the Vast box:
`MEMRA_PREFILL_TICK=2048` added to serve-env.sh (grouped already on), server
restarted via /root/start-memra.sh. Spot-check: short TTFT 0.200s (unchanged),
4k-cold TTFT 6.938s -> 6.028s (-13%, single run each). Soak loop restarted.

## Fixed binary deployed (2026-08-10, post b1fix P0 fix)

Rebuilt on Vast at 019428e2 (P0 one-numeric-class fix + kv256 ring flag-off + grouped/tick
promoted config). Spot-check single run each: short TTFT 0.218s / 70.4 tok/s decode,
4k-cold TTFT 6.652s. The -4.71% c=1 decode price of the correctness fix is visible
(74.5 -> ~70 tok/s class) and accepted — served bytes are now load-history-invariant
(one-hash matrix receipt in research/b1fix-20260810). Soak restarted on the fixed binary.
