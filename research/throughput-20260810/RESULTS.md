# Box1 PP-2 current-binary throughput ceiling — 2026-08-10

## Verdict

The honest short-prompt ceiling of the current binary is **129.70 aggregate output tok/s** at
c=64 with `MEMRA_MOE_GROUPED=1` (N=3 median, range 129.49-129.96). This is the full request
window from simultaneous barrier release through final drain, so it includes the 128-token
prefill and first-token staggering. The same cell delivers **155.40 tok/s from first visible
token through final drain**, while the rolling steady decode step implies **167.68 tok/s**.
Those three rates are deliberately kept separate.

Scaling is already exhausted at c=8: grouped-on rises only **1.03%**, from 128.38 at c=8 to
129.70 at c=64. Grouped-off rises only 1.83%, from 101.17 to 103.03. The limiter is the
current Step35 **decode chunk cap of eight**, not admission or queue pressure. At c=64 the
scheduler has all 64 rows ready but executes eight B=8 chunks per outer tick; step p50 grows
from 48.30 ms at c=8 to 381.68 ms at c=64. All 1,248 scored baseline requests completed at
exactly 256 output tokens with zero VRAM defers, session defers, sampled queue depth, or OOM
parks.

Grouped dispatch is a real win on this rented pair class: at c=64 it raises full-window
aggregate throughput **25.89%** (103.03 -> 129.70 tok/s). On the c=16 trial-traffic shape it
raises aggregate output **41.64%** (35.44 -> 50.20 tok/s) and cuts content-bearing TTFT p50
**34.05%** (84.24 -> 55.55 seconds).

The focused grouped-on serving-knob block found one further win for the 2k trial shape:
`MEMRA_PREFILL_TICK=2048` raises aggregate output from 50.16 to **54.46 tok/s** (**+8.59%**)
and lowers TTFT p50 from 55.636 to **49.257 seconds** (**-11.47%**), N=3 interleaved. The
three-run ranges do not overlap. First-text-to-drain throughput and steady decode rate move only
+0.15% and +0.16%, respectively, so this is a prefill/TTFT improvement rather than a higher
decode ceiling.

This remains far below the owner's 632 aggregate tok/s breakeven: the measured full-window
ceiling is **20.52% of target, a 4.87x gap**. Even the generous steady-decode step rate is only
26.53% of target, a 3.77x gap. The winning mixed serving config reaches **8.62% of target, an
11.60x gap**.

## Short-prompt c-curve

Every row is the median of **N=3 interleaved repetitions**; parentheses give the complete
three-run aggregate range. Prompts are exactly 128 input token ids and generations are exactly
256 output tokens at temperature 0. `aggregate` spans barrier release through final drain;
`first-text -> drain` excludes the interval before the first content-bearing SSE event;
`step rate` is `c / step_p50`. Step p50/p99 are the worker's rolling engine-truth values.
Defers are total VRAM/session defers across all three repetitions.

| grouped | c | aggregate output tok/s | first-text -> drain tok/s | step p50 / p99 | step rate tok/s | defers |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 8 | 101.17 (101.10-101.18) | 126.82 | 59.18 / 59.83 ms | 135.18 | 0 / 0 |
| OFF | 16 | 102.10 (101.87-102.23) | 127.29 | 117.17 / 119.05 ms | 136.56 | 0 / 0 |
| OFF | 24 | 102.47 (102.22-102.89) | 127.42 | 175.34 / 176.63 ms | 136.88 | 0 / 0 |
| OFF | 32 | 102.74 (102.44-102.81) | 127.71 | 233.02 / 235.55 ms | 137.32 | 0 / 0 |
| OFF | 48 | 102.90 (102.75-103.08) | 127.71 | 348.92 / 351.59 ms | 137.57 | 0 / 0 |
| OFF | 64 | 103.03 (102.93-103.11) | 127.89 | 465.28 / 467.18 ms | 137.55 | 0 / 0 |
| ON | 8 | 128.38 (128.38-128.39) | 154.97 | 48.30 / 48.98 ms | 165.62 | 0 / 0 |
| ON | 16 | 128.80 (128.79-128.94) | 154.85 | 96.23 / 97.78 ms | 166.27 | 0 / 0 |
| ON | 24 | 129.37 (129.07-129.89) | 155.28 | 143.59 / 144.97 ms | 167.14 | 0 / 0 |
| ON | 32 | 129.47 (129.33-130.09) | 155.25 | 191.31 / 192.73 ms | 167.27 | 0 / 0 |
| ON | 48 | 129.55 (129.46-130.27) | 155.28 | 286.51 / 288.07 ms | 167.53 | 0 / 0 |
| ON | 64 | **129.70 (129.49-129.96)** | **155.40** | **381.68 / 383.37 ms** | **167.68** | **0 / 0** |

The old 130 tok/s c=8 decode-only receipt is therefore not the current end-to-end ceiling's
denominator. Numerically, current grouped-off first-text-to-drain is still in that old band
(126.82-127.89); current grouped-on moves it to 154.85-155.40, and its steady step reaches
165.62-167.68. The requested primary aggregate remains the lower, fully burdened number.

## Why it flattens

The current-binary receipts name the boundary directly:

```text
[worker] step: decode chunk cap 8
[step35-batch] first B>1 batched step35 walk: B=8 layers=[0,22)
[tick] act=64 int=64 priming=0 ready=64 spec=0 demoted=0 ... prefill_ms=0.0 decode_ms=382.2
```

The traced diagnostic contains **255 `ready=64` decode ticks**. Their median `decode_ms` is
382.2 ms (379.0-396.0), matching the untraced N=3 c64 step p50 of 381.68 ms. The exact lines
are retained in
[`diagnostic-grouped-on-c64/server.log`](raw/block-baseline-20260809T222030Z/diagnostic-grouped-on-c64/server.log)
and its filtered
[`limiting-lines.txt`](raw/block-baseline-20260809T222030Z/diagnostic-grouped-on-c64/limiting-lines.txt).

This Step35 path hard-clamps `MEMRA_DECODE_BATCH_CAP` to at most eight; an explicit larger value
cannot widen it. The current source explains why: the IQ4_XS plus 288-expert MoE checkpoint does
not qualify for the exact-16 tier, so the cap remains eight
([`worker.rs`](../../crates/memra-server/src/worker.rs#L5845)). More admitted rows therefore create
more serial B=8 chunks inside each scheduler tick. They do not create a wider weight-streaming
batch.

Admission is not the hidden limit. The applied request-owned charges were:

```text
[admission] request cost: model="step" ctx=448 path=plain = 83520 B/token x ctx + 0MB fixed = 37MB
[admission] request cost: model="step" ctx=2320 path=plain = 83520 B/token x ctx + 0MB fixed = 194MB
```

Thus `MEMRA_CTX=262144` remained the server ceiling rather than charging every short request as
262k. Peak sampled memory was 50,705/61,107 MiB on 97,887 MiB GPUs. Every cell reported zero
defers, zero queue depth, and zero OOM parks.

One applied-surface surprise matters for interpreting the A/B: `docs/FLAGS.md` describes
`MEMRA_MOE_GROUPED` as prefill-only, but the current implementation ignores the `_prefill`
argument and selects grouped dispatch whenever `t > 1`
([`hybrid_forward.rs`](../../crates/memra-engine/src/hybrid_forward.rs#L182),
[`hybrid_forward.rs`](../../crates/memra-engine/src/hybrid_forward.rs#L3128)). Batched decode passes
`t=B=8`, so grouped ON accelerates decode as well as prefill in the binary measured here. The
near-constant 25.9-26.9% aggregate gain across the entire c-curve is consistent with that applied
behavior, not with a prefill-only effect.

## Mixed c=16 trial-traffic shape

These rows use exactly 2,000 prompt token ids, 256 generated tokens, temperature 0, and 16
simultaneously released streams. TTFT is per-stream, content-bearing first-token time; SSE
keepalive comments do not count.

| grouped | aggregate output tok/s | first-text -> drain tok/s | per-stream TTFT p50 | step p50 / p99 | defers |
|---|---:|---:|---:|---:|---:|
| OFF | 35.44 (35.44-35.46) | 122.61 | 84.239 s | 122.84 / 124.34 ms | 0 / 0 |
| ON | **50.20 (50.18-50.28)** | **147.13** | **55.553 s** | **102.26 / 103.20 ms** | **0 / 0** |

The focused tick-budget comparison kept grouped ON and used fresh servers, identical traffic,
and alternating order under one lock. Every row is N=3; parentheses are complete ranges.

| prefill tick | aggregate output tok/s | first-text -> drain tok/s | per-stream TTFT p50 | step p50 / p99 | defers |
|---|---:|---:|---:|---:|---:|
| default 1024 | 50.16 (50.13-50.24) | 147.29 | 55.636 s | 102.11 / 103.27 ms | 0 / 0 |
| explicit 2048 | **54.46 (54.41-54.47)** | **147.51** | **49.257 s** | **101.95 / 103.06 ms** | **0 / 0** |

All 96 requests completed at exactly 256 output tokens; there were zero sampled queue entries,
VRAM/session defers, or OOM parks. The source-defined tick budget caps prefill tokens per session
per scheduler pass. On this exact 2,000-token shape, 1024 requires two per-session prefill calls,
whereas 2048 can consume the prompt in one
([`worker.rs`](../../crates/memra-server/src/worker.rs#L5528)). Neither arm emitted a
`[prime-batch]` line, so this win did not come from the cross-request concat-prime path. The nearly
identical post-TTFT and decode-step rates independently localize the gain before decode.

## Serving recommendation

For the measured box1 PP-2 trial tier -- simultaneous fresh c=16 traffic around a 2k prompt and
256-token generation -- use:

```text
MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048
```

This is a serving recommendation for that shape, not a runtime default flip. The required local
5090 transfer campaign previously rejected the grouped default, and this lane has not rerun that
gate. An explicit 2048 tick also doubles the per-session prefill work allowed before the scheduler
returns to decode. The measured block had no already-decoding traffic during the all-fresh prefill,
so retain the default 1024 for latency-sensitive mixed-arrival tiers until 2048 is separately gated
with live decode peers and tail-latency SLOs.

Do not set `MEMRA_DECODE_BATCH_CAP=16` expecting another rung: Step35 clamps it back to eight.
Do not use `MEMRA_SERVE_BATCH=0`: that selects legacy round-robin serving with a four-session cap.
At c>8, additional active streams should be understood as a latency/fairness choice; they do not
raise aggregate throughput on this binary.

## Rig, identity, and method

- Runtime source: `2d9359df` (affroom plus capbase merged). The research-harness checkpoint is
  `c040feaa`; no runtime source was changed.
- Exact release `memra-server` SHA-256:
  `8d69e0027d34cf90ed32febc66e84a5e2f8671268f6c4846ab063435928cdd54`.
- Model: the three Step-3.7-Flash IQ4_XS GGUF shards plus external Q8_0 MTP draft under the
  requested `~/step37/models/step-3.7-flash`. Their fresh hashes and manifest hash are retained in
  [`artifact-sha256.txt`](raw/block-baseline-20260809T222030Z/artifact-sha256.txt) and
  [`artifact-manifest-sha256.txt`](raw/block-baseline-20260809T222030Z/artifact-manifest-sha256.txt).
- Rig: two RTX PRO 6000 Blackwell Server Edition GPUs, 97,887 MiB each; driver 595.71.05,
  CUDA 13.2; `MEMRA_PP_STAGES=2`, devices `0,1`, `MEMRA_CTX=262144`,
  `MEMRA_MAX_SESSIONS=64`. PP-2 placement logged `LOW=0 HIGH=1` and spec admission OFF.
- Cache controls: prefix cache and prefix dedup disabled; every request used a unique namespace
  and session id. The scored eight-prompt deterministic family was selected only to eliminate
  early-EOS length bias. Higher concurrencies repeat that family, but every B=8 decode chunk
  contains the same eight distinct prompts and gets no prefix/cache reuse.
- Order under one exclusive lock: rep1 OFF/ON, rep2 ON/OFF, rep3 OFF/ON, then an excluded
  grouped-on c64 trace. Each server received one excluded 16-token warmup followed by 31 seconds
  to clear the rolling step window. The block ran 22:21:40-23:25:05 UTC.
- Serving-knob order under a second exclusive lock: rep1 default/2048, rep2 2048/default, rep3
  default/2048. It acquired the lock at 23:49:23 and completed at 00:04:20 UTC. Each cell used a
  fresh server and the same excluded warmup plus rolling-window clear.
- Thermal regime: continuous one-second GPU sampling. Across the six scored servers, sampled
  maximum temperatures were 50/53 C, maximum powers 337.3/381.4 W, and maximum memory
  50,705/61,107 MiB. The pre-block and post-block process snapshots were empty; per-server final
  snapshots contained only the measured server.
- Across the six knob servers, sampled maximum temperatures were 50/54 C, maximum powers
  426.6/445.7 W, and maximum memory 47,825/58,035 MiB. Its pre-block and post-block process
  snapshots were empty; each final per-server snapshot contained only the measured server.
- Baseline scored load: 42 cells, 1,248/1,248 requests, and 319,488/319,488 expected output
  tokens. Every failure scan is empty. Aggregate rates use the server's `tokens_out` delta as
  token authority, not rendered SSE chunk count.
- Knob scored load: 6 cells, 96/96 requests, and 24,576/24,576 expected output tokens. Combined:
  48 cells, 1,344/1,344 requests, and 344,064/344,064 tokens.

## Receipts and exclusions

- [`raw/block-baseline-20260809T222030Z/`](raw/block-baseline-20260809T222030Z/) is the scored
  grouped OFF/ON block. `driver.log` records the one lock acquisition, exact arm order, every
  client row, and the rc=0 completion marker. Each cell's `requests.jsonl` retains the raw run,
  before/after metrics, one-second samples, every request, and its summary.
- [`raw/block-knob-20260809T232606Z/`](raw/block-knob-20260809T232606Z/) is the scored grouped-on
  default/2048 tick block. Its source, binary, and artifact hashes exactly match the baseline;
  `runner.rc` is 0 and all six failure scans are empty.
- [`raw/provision-2d9359df/`](raw/provision-2d9359df/) retains the exact release build log, rc,
  and binary hash.
- [`raw/block-pilot-20260809T221430Z/`](raw/block-pilot-20260809T221430Z/) is the excluded
  prompt-family pilot.
- [`raw/pilot-aborted-eos-shape-20260809T220757Z/`](raw/pilot-aborted-eos-shape-20260809T220757Z/)
  is the excluded first shakeout. It stopped when varying synthetic prompts produced two valid
  early-EOS streams, which would have made a fixed-256 throughput comparison dishonest.
- [`summary.jsonl`](summary.jsonl) is generated from the immutable per-cell JSONL by
  `summarize.py`; it contains all 48 scored replicates plus 16 N=3 medians and full ranges.

Nothing was pushed, tagged, merged, released, or added to the generated performance board. No
runtime code, Rust toolchain, profiler, or runtime default was changed.
