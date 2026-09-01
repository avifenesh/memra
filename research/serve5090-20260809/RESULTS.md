# Q27 single-card serve floor — RTX 5090 Laptop (2026-08-09)

## Verdict

The proof-rig receipt is complete and valid, but the current default surface is **not an
unconditional default-flip green**.

Three findings must remain loud:

1. **RED — default-on exact-repeat cache reuse is absent.** Repeating the identical 4,094-token
   prompt and `cache_salt` measured **3,067.6ms** TTFT with `cached_tokens=0`. The same request
   under `MEMRA_SERVE_SPEC=0` measured **8.8ms** with all 4,094 tokens cached: default is
   **348x slower**. The server receipt says `spec-affinity: declined (empty suffix)`; spec
   sessions bypass the reusable prefix cache by policy, so this is a real default serve-surface
   gap rather than measurement noise. Default-on cannot inherit the pair receipt's ms-class
   cache-ready verdict.
2. **RED — the default concurrency policy loses at c=4.** Default on-policy produces
   **119.4 aggregate tok/s** versus **128.5** with speculation disabled, a **7.1% loss**.
   The LOW=2/HIGH=4 crossover direction is correct (default wins c=1 and c=2), but admitting
   and then demoting speculative sessions does not reach the all-plain c=4 floor. It still
   clears the serve-ready reading-speed bar at 29.9 tok/s/stream, narrowly.
3. **RED serve-floor gap versus the frozen v0.72 plain row.** The forced-plain HTTP c=1 floor is
   **44.4 tok/s**, **6.7% below** the v0.72 q27 bare-CLI row at 47.6. This is not a same-prompt
   kernel A/B (226-token sampled serve load versus the board's 512-context CLI protocol), so it
   identifies a serve-path regression target, not proof that the decode kernel itself moved.

Everything correctness-related is green: zero scored request errors/sheds, no scored OOM or
fatal signatures, exact prefix-dedup/pinning accounting in all three repetitions, six
developer-role requests accepted, `serve-smoke` 44/44, `kernel-check` ALL GREEN, `run-gen`
argmax MATCH, and `run-spec` K=1..8 self-consistency PASS.

## New single-card floor table

All values are medians of **N=3 interleaved repetitions**; parentheses are the full three-run
range. TTFT is client-observed time to the first visible SSE token. Cold rows exclude one
labeled warmup per repetition. Decode rows are streamed, temperature 0.7, 128-token budgets;
each point contains 8 requests at c=1/2 and 16 at c=4 after one excluded warmup.

| serve-surface cell | default on-policy | `MEMRA_SERVE_SPEC=0` | default versus off |
|---|---:|---:|---:|
| cold short TTFT, 229 prompt tok | 207.9ms (203.2-208.0) | 191.6ms (191.4-192.4) | +8.6% slower |
| cold 4k TTFT, 4,107 prompt tok | 3,125.2ms (3,088.9-3,161.0) | 3,151.0ms (3,144.3-3,178.1) | 0.8% faster |
| exact-repeat TTFT, 4,094 prompt tok | **3,067.6ms** (2,956.7-3,115.8), cached=0 | **8.8ms** (8.1-12.0), cached=4,094 | **348x slower — RED** |
| cached-continuation TTFT | 140.4ms (138.8-147.9), cached=4,160, K=2 | 117.5ms (117.2-120.2), cached=4,158 | +19.5% slower |
| decode c=1 aggregate | **85.8 tok/s** (85.1-86.3) | 44.4 (43.4-44.6) | +93.2% |
| decode c=2 aggregate | **86.9 tok/s** (86.6-87.7) | 77.8 (77.6-77.9) | +11.7% |
| decode c=4 aggregate | **119.4 tok/s** (119.1-120.0) | **128.5** (127.5-133.8) | **-7.1% — RED** |
| decode c=4 per stream | 29.9 tok/s | 32.1 tok/s | -7.1% |

The default K receipts are exactly the merged policy: K=3 for every scored cold-short and
cold-long probe, K=2 for every cached-long continuation, and K=0 admission/demotion receipts
under c=4 pressure. The spec-off arms contain no nonzero-K receipt.

For latency context, default on-policy decode request p50/p95 medians were 1.51/1.59s at c=1,
2.96/3.41s at c=2, and 3.60/6.79s at c=4. Forced-plain medians were 2.88/3.01s, 3.29/3.30s,
and 4.00/4.04s respectively. The c=4 default p95 is the mixed-policy tail, even though its
p50 remains lower than all-plain.

## Frozen v0.72 board comparison

These rows remain regression anchors; this lane does not move `current-board.json` or any
generated perf surface.

| anchor | frozen v0.72 q27 | current serve cell | delta | verdict |
|---|---:|---:|---:|---|
| plain decode | 47.6 tok/s, bare CLI, tg128 at 512 context | spec-off c=1: 44.4 | **-6.7%** | **RED serve-floor gap; protocol differs** |
| K=3 long-agentic sampled | 86.0 tok/s | default c=1 sampled: 85.8 | -0.3% | stable signal; prompt differs |
| K=3 short-code / medium-code greedy | 116.4 / 101.2 tok/s | default c=1 sampled: 85.8 | -26.3% / -15.3% numerically | not a valid regression A/B: different prompts and sampling class |

The closest speculative comparison is the sampled row: the serve c=1 floor is effectively
flat to 86.0. The short/medium board cells are listed so their numerical misses are visible,
but treating them as serving regressions would erase the board's prompt-class contract.
As a non-scored diagnostic, the correctness `run-gen` invocation produced 48.60 tok/s on its
90-token prompt (single run), above the 47.6 board anchor; that points at serving/protocol
overhead rather than a demonstrated plain-kernel regression, but it is not an N=3 replacement
for the required same-prompt A/B.

## Pair serve-ready comparison

`research/serve-ready-20260808/RESULTS.md` measured Step-3.7-Flash on a 2x RTX PRO 6000 PP-2
pair. It is the requested serve-ready bar and API protocol reference, **not** a same-model or
same-rig denominator.

| serve cell | this q27 single-card default | pair receipt | numerical delta |
|---|---:|---:|---:|
| cold short TTFT | 0.208s | 0.595s | 65.0% lower |
| cold 4k TTFT | 3.125s | 6.052s | 48.4% lower |
| exact-repeat 4k TTFT | **3.068s, miss** | **12.2ms, full hit** | **251x slower — RED behavior split** |
| exact-repeat with spec disabled | 8.8ms, full hit | 12.2ms, full hit | 27.8% lower |
| decode c=1 aggregate | 85.8 tok/s | 88.5 | 3.1% lower |
| decode c=2 aggregate | 86.9 tok/s | 118.2 | 26.5% lower |
| decode c=4 aggregate | 119.4 tok/s | 146.1 | 18.3% lower |
| decode c=4 per stream | 29.9 tok/s | 36.5 | clears pair bar of ~29 |

The single card passes the pair's cold-TTFT thresholds (<0.8s short, <=7.5s 4k) and c=4
reading-speed bar. It passes the ms-class repeat bar only when speculation is disabled; the
default arm fails that bar.

## Rig, tree, artifacts, and protocol

- Engine code: `96a09705895af120a0f706558a8c8c0d6fd8520a`; measurement ledger commit:
  `9746d52a88a71af1fcc1d0a7b6a1b8d17a5483a1`.
- Release `memra-server` sha256:
  `c44c93db5ca5a95994f592390976956ee2d1d361d9019aad8099f5631f54699e`.
- Q27 trunk sha256:
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`.
- Daily draft sha256:
  `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581`.
- GPU: NVIDIA GeForce RTX 5090 Laptop, 24,463 MiB; driver 595.84; CUDA 13.1.115;
  performance profile, dynamic boost 25W, TGP offset 150W.
- `window_clean=false`: Hermes PID 7600 held 394 MiB before and after the window. Active
  samples spanned 68-87C, 1,612-2,145 MHz SM clock, 72-196W, and peaked at 22,230 MiB.
  The order alternated on/off, off/on, on/off; c ordering rotated each repetition. The one
  scored lock ran 22:34:19-22:42:06 UTC.
- Both arms used one card, `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`,
  `MEMRA_REUSE_POOL=1`, and `MEMRA_PREFIX_CACHE_MB=512`. All serving behavior flags were
  explicitly unset. The control arm alone set `MEMRA_SERVE_SPEC=0`. Pool and prefix budgets
  are the documented 24.5GB machine configuration; 512 MiB is required to hold one q27 4k
  prefix entry.
- Every scored request traversed `/v1/chat/completions` or `/v1/completions`; no bench binary
  supplied a floor row. Cache salts were bounded per arm so the load generator could not
  manufacture an unbounded set of parked spec pools.

## Correctness and feature receipts

- `tools/serve-smoke.sh <q27> <daily-draft>`: **44 ok, 0 failed**, including OpenAI chat and
  completion shapes, streaming, concurrent chat, greedy determinism, cache metering,
  spec==plain greedy text, sampled truncation, affinity rewinds, and the gemma scheduler arm.
- Fresh tip `kernel-check`: **ALL GREEN**.
- Fresh tip q27 `run-gen`: prefill argmax=decode argmax=72240 **MATCH**; batched-prime
  argmax=tokenwise argmax=72240 **MATCH**.
- Fresh tip q27 `run-spec`: **8/8 self-consistency PASS**, K=1..8, final
  `SELF-CONSISTENCY PASS`.
- Prefix dedup/pinning, repeated N=3 on fresh spec-off servers: each four-request fanout
  returned cached token counts `[0, 256, 256, 256]`; cross-salt request stayed cold; all
  metrics reconciled; server emitted `B=3 ... retained=true` (three followers behind the
  leader).
- Developer-role normalization: 6/6 requests succeeded across both arms and all repetitions.
- Scored load: 18/18 points, 48/48 requests at c=1, 48/48 at c=2, and 96/96 at c=4,
  zero errors and zero sheds. No scored log contains `CUDA_ERROR`, captured OOM, illegal
  address, panic, or fatal signal.

## Raw evidence and excluded shakeouts

The scored run id is `20260808T223413Z-scored`:

- `raw/driver-20260808T223413Z-scored.log` records the exact commands, environment, artifact
  paths, lock, arm order, and completion marker.
- `raw/decode-points-*scored.jsonl` and `raw/decode-requests-*scored.jsonl` hold every aggregate
  and per-request decode row. `raw/ttft-*scored.jsonl` and `raw/cache-ttft-*scored.jsonl` hold
  every warmup/measured TTFT and cache receipt.
- `raw/server-*scored.log`, `raw/metrics-*scored.json`, `raw/cache-meter-*scored.*`, and
  `raw/developer-*scored.json` retain policy, K, dedup/pinning, accounting, and role evidence.
- `raw/gpu-*scored.csv` plus the pre/post process lists retain the thermal and co-residency
  record.
- `raw/gates-serve-smoke-*`, `raw/gate-kernel-check-*`, `raw/gate-run-gen-q27-*`, and
  `raw/gate-run-spec-q27-k1to8-*` retain complete correctness output. Build and artifact hash
  logs are also committed.

The `shakeout`, `shakeout2`, and `shakeout3` files are retained but excluded from every N and
median. Shakeout 1 captured real `CUDA_ERROR_OUT_OF_MEMORY` retries after the original harness
created eight salt-keyed parked spec sessions. Shakeout 2 found a client-only assumption that
synthetic token ids always render visible text. Shakeout 3 completed the protocol but failed
an incorrect final assertion: a four-request group has one leader and **three** deduplicated
followers, so the server's `B=3` receipt was right. The scored run began fresh after these
harness corrections and contains no such failure.

## Default-flip disposition

This commit establishes the requested single-card self-competition floor; it does not change
runtime defaults or generated board numbers. The floor says:

- keep K=3 at c=1 and K=3/K-policy at c=2: both beat forced plain here;
- do not call c=4 fully closed while the default is 7.1% below all-plain;
- do not call default exact-repeat caching serve-ready while spec requests bypass the reusable
  prefix cache and empty-suffix affinity declines; and
- track the 44.4 versus 47.6 plain serve-floor gap with a same-prompt serve-vs-CLI A/B before
  assigning it to the kernel or HTTP scheduler.
