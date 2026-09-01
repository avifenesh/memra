# optipipe increment 0 results — mid-body PP verify seam

## Verdict

**Overlap real: YES. Increment 1: GO, for the state-fork correctness increment only.**

The TX-release seam produced a median **10.429 ms** of observed
`A.S1 || B.S0` overlap across **220/220** complete two-session rounds. The
comparable 19-round pair interval fell from the specmech whole-body-release
receipt's 48.662 ms to **36.746 ms** (-11.916 ms, **-24.49%**). In the N=5
interleaved c=2 run, the seam coordinator reached **63.082 tok/s**, **+13.94%**
over the same-build serial coordinator.

This clears DESIGN.md's prerequisite for paying increment 1's state-fork bill.
It is not a serving or promotion GO: the seam is still **-47.89%** versus the
same-block plain median of 121.051 tok/s, `MEMRA_SPEC_PIPE` remains experimental
and default OFF, and the confidence/economics gates for later increments remain
unchanged.

## What changed

`crates/memra-engine/src/spec.rs` now splits the PP verify body into:

1. `verify_stage0_issue(...) -> VerifyBoundaryTicket`, which enqueues embed,
   stage-0 layers, and boundary TX and returns the actual persistent slot; and
2. `verify_stage1_finish(ticket, ...)`, which consumes that slot through RX,
   the remaining PP layers, head, and caller-stream publication.

The ordinary path calls those functions back-to-back. With tracing disabled it
adds no CUDA callback, event, synchronization, or coordinator wait. The
experimental two-session path now orders each round as follows:

```text
A.S0 -> A.TX/ticket -> A.S1
                  \-> B.S0 -> B.TX/ticket
A.S1/head enqueue complete -> B.S1
```

Only `A.S1 || B.S0` is released. B stage 1 still waits until A has finished
issuing stage 1 and the head, so neither stage engine is concurrently issued by
both callers.

### Boundary-slot and event audit

The existing two persistent slots are sufficient; no new transport event was
needed and `pp.rs` is unchanged.

- The ticket carries the slot returned by `tx_pipelined()`; no round/parity
  inference was introduced.
- All **220/220** traced rounds used A slot 0 and B slot 1.
- TX waits the selected slot's previous `ev_rx`, writes the persistent slot,
  then records `ev_tx`.
- RX waits `ev_tx`, copies the slot into fresh stage-1 work, then records
  `ev_rx` before the stage-1 trunk. A later TX can therefore reuse the
  persistent slot without overwriting stage 1's live input.

This is the DESIGN.md section-3 fence sequence: A has consumed its boundary
slot before reuse is admitted, and each `VerifyBoundaryTicket` names the slot
that its stage 1 must read.

## Timeline receipt

The excluded trace arm used `MEMRA_SPEC_PIPE_TRACE=1`. Trace-only
`cuLaunchHostFunc` markers bracket S0 through TX and RX through S1/head on the
owning CUDA streams, with one monotonic host clock per paired call. The callback
makes no CUDA calls, following the current
[CUDA Driver API host-function contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html).
The scored arms do not enqueue these markers.

The receipt contains 12 continuation bursts: eight 19-round pairs and four
17-round pairs, **220 complete rounds / 1,760 phase edges**. Every round satisfies
`B.S0.start < A.S1.end`.

| Timeline quantity | N | Median | Min-max |
|---|---:|---:|---:|
| A.S0 duration | 220 rounds | 10.337 ms | 10.220-13.383 |
| A.S1 duration | 220 rounds | 12.293 ms | 12.145-14.988 |
| B.S0 duration | 220 rounds | 10.429 ms | 10.318-10.642 |
| B.S1 duration | 220 rounds | 12.187 ms | 12.042-12.344 |
| A.S1 / B.S0 intersection | 220 rounds | **10.429 ms** | 10.317-10.642 |
| A.TX marker -> B.S0 start | 220 rounds | 0.011 ms | 0.008-0.031 |
| A.S0 start -> B.S1 end | 220 rounds | 34.862 ms | 34.583-37.781 |
| Four-phase sum minus envelope | 220 rounds | **10.383 ms hidden** | 10.267-13.454 |

Host callbacks are stream-ordered observations, not a device-global cycle
counter; callbacks on independent contexts can be scheduled with small host
delay. The overlap verdict does not rest on one edge ordering: the complete
round population hides 10.383 ms versus the four phase durations, the paired
tick wall contracts by about 11.9 ms versus the old receipt, and the excluded
trace arm preserves the same end-to-end uplift (62.993 versus 55.438 tok/s).

The pair-interval comparison uses each `[tick-spec-pipe] wall_ms / rounds`, so
it includes draft, verify, wait, and commit work and matches the prior
specmech receipt's definition:

| Burst shape | Old whole-body release | TX-release seam | Delta |
|---|---:|---:|---:|
| 19 rounds | 48.662 ms (N=8) | **36.746 ms** (N=8, 36.709-36.912) | **-11.916 ms / -24.49%** |
| 17 rounds | 48.655 ms (N=4) | **36.782 ms** (N=4, 36.765-36.810) | **-11.873 ms / -24.40%** |

Raw timeline: [`raw/box1/perf/c2-trace-seam-server.log`](raw/box1/perf/c2-trace-seam-server.log).

## c=2 A/B

Shape: 2x RTX PRO 6000 Blackwell Server Edition; PP-2 devices 0,1; context
262144; grouped MoE ON; prefill tick 2048; Step-3.7-flash IQ4_XS plus Q8_0 MTP;
8 measured requests x 128 greedy completion tokens and 2 warmups per point.
Each point used a fresh server. The 15 scored points ran in rotating arm order
under one uninterrupted `/tmp/memra-gpu.lock` hold from 22:20:46Z to 22:36:18Z.

Every median below is N=5 interleaved. All 120/120 requests completed, with no
shed or error. At arm boundaries, the 30 pre-run GPU samples were 26-36 C and
the 30 post-run samples were 33-36 C; all post-run samples were P0 at
2317-2415 MHz. The driver asserted no other compute process at the lock
boundaries and between fresh servers.

| Arm | Median tok/s | Min-max | Delta | Requests |
|---|---:|---:|---:|---:|
| plain | **121.051** | 120.895-121.348 | reference | 40/40 OK |
| serial spec | **55.365** | 55.255-55.454 | -54.26% vs plain | 40/40 OK |
| TX-release seam | **63.082** | 62.766-63.095 | **+13.94% vs serial; -47.89% vs plain** | 40/40 OK |

Historical anchors are context, not same-block denominators. The new seam is
+17.49% versus specmech's 53.691 tok/s whole-body pipeline and +17.81% versus
its 53.544 serial arm. It remains -4.35% versus specpp2's older c=2 K=1 serial
result of 65.953 tok/s and -45.26% versus that study's 115.230 tok/s plain arm.
The current interleaved block above is the authoritative attribution result.

Raw performance receipts:
[`points.jsonl`](raw/box1/perf/points.jsonl),
[`requests.jsonl`](raw/box1/perf/requests.jsonl), and
[`driver.log`](raw/box1/perf/driver.log).

## Exactness and gates

The model artifacts were pinned by SHA-256: source shard
`b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259`
and MTP
`469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57`.
The measured server binary was
`1ff9d845d7f14cd8691c71c6de3aaedfd2351c59ede2494dd7d2778b1dfd9f4`.
Exactness used code-bearing tip `9d857b8c`; performance ran at `69603404`,
whose only additional commit adds the measurement driver and progress update.
The same server binary was used for both.

| Gate | Result | Receipt |
|---|---|---|
| Default serial path, fresh process each time | **PASS 10/10**; every 326-byte completion SHA-256 `21b8293f...bb6de` | [`raw/box1/serial-boots/`](raw/box1/serial-boots/) |
| Plain / serial / PIPE=1 served bytes | **PASS 6/6**; 2 requests per arm, one completion hash; pipeline engaged | [`identity-summary.json`](raw/box1/exactness-main/identity-summary.json) |
| `run-spec` K=1..8 | **PASS 8/8**, every K identical to plain target | [`run-spec.log`](raw/box1/exactness-main/run-spec.log) |
| `kernel-check` | **ALL GREEN**, including model-backed Step IQ4_XS checks | [`kernel-check.log`](raw/box1/kernel-check.log) |
| `run-gen` PP-2 argmax | **PASS**; prefill/decode 6776 MATCH; batched-prime/tokenwise 6776 MATCH | [`run-gen-summary.log`](raw/box1/run-gen-summary.log) |
| Local engine build/tests | **PASS**; `cargo check`; 54 passed, 0 failed, 1 CUDA-only ignored | [`raw/local/`](raw/local/) |

The complete box1 evidence set has 237 files and is covered by
[`raw/box1/MANIFEST.sha256`](raw/box1/MANIFEST.sha256), verified after download.

## Increment-1 recommendation

**GO for increment 1, bounded exactly to fork + reconcile.** Increment 0
falsified the stop condition: there is real cross-stage work to exploit, the
existing boundary slots/fences survive the release, exactness remains one-hash,
and the same-build end-to-end arm gains 13.94%.

Do not infer a product win. Increment 1 must still produce forced-hit,
forced-miss, abort, alternating-generation, state-identity, rollback-latency,
and peak-memory receipts before increment 2; it must not change the default or
publish board numbers. The seam remains default OFF, and plain remains the only
winning c=2 serving policy in this measurement.

No perf-board update, merge, tag, or push was performed.
