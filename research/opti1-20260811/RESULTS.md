# OPTIPIPE increment 1 — fork and reconcile results

Date: 2026-08-11
Lane: `lane/opti1`
Code under box1 gates: `4ac646c3`
Receipt-harness-only head: `0c2a7e2a`
Rig: box1, 2x RTX PRO 6000 Server Edition, every GPU block under
`flock /tmp/memra-gpu.lock`

## Verdict

**Increment 2 GO on the pinned Step target, with the door remaining diagnostic-only and default
off. This is not a merge, promotion, or throughput GO.**

Increment 1's required mechanics are green: two generation-owned snapshot/seed slots alternate,
forced hits retain stage-0 state, forced misses restore and rerun serially, an unresolved ticket
drains on abort, and ring/round-stream sessions do not enter the fork. The final model-backed
suite completed 15/15 hits, 15/15 misses, and a 132-attempt alternating soak (66 hit / 66 miss)
with exact state and exact subsequent serial continuation.

The important qualification is artifact reality: the pinned `Step-3.7-flash-IQ4_XS` file loads
as `step35` with 45 trunk layers and exposes no recurrent cache state. Model-backed comparisons
therefore cover zero recurrent bytes. The conditional f32 hit-skip/miss-restore kernel has a
direct 262,144-element device gate, and both checkpoint generations are allocated/refreshed
through each PP stage's owning engine, but a GDN-bearing checkpoint must repeat the model-backed
identity and memory/HBM measurements before this mechanism is generalized beyond the pinned
target.

## What landed

- `round_stream.rs` exposes a stage-range KV-length pointer table; the fork owns only stage 0's
  `[0, 22)` table.
- `spec_sample.cu` and `lib.rs` add three small device controls: derive K=1 fork validity, restore
  stage-local KV lengths only on a miss, and conditionally restore f32 state.
- `spec.rs` owns two alternating checkpoint/seed generations, generation-tagged reserve/retire,
  device validity, hit-skip, miss-restore through the existing length-truncation path, and RAII
  drain for unresolved boundary tickets.
- Both checkpoint sets clone and refresh recurrent state on the engine owning each PP stage.
  This closes the cross-device GDN ownership hole that the zero-GDN target cannot exercise.
- Admission is fail-closed: session-only, fixed greedy K=1, device accept required, PP-2
  cross-device with primary stage 0, and no two-session pipe, ring, round-stream, replay,
  sampled/constrained, or host-bounce path.
- `optipipe-gate` is the only caller. Serving has no controller or policy in this increment, and
  no environment door silently enables the fork.

## State-identity gates

The comparator checks committed ids, cache position/capacity, pending/next-prediction/counters,
every host and device KV length, every live trunk K/V byte on its owning device, live recurrent
bytes when present, MTP scratch length and live K/V bytes, and the carried hidden seed. It then
disables the diagnostic and compares another 16 generated tokens plus state.

| Gate | Result | Retained evidence |
|---|---|---|
| Forced hit | **PASS** — 15 attempts, 15 hits, 0 misses; 33-token result and 16-token continuation exact | [hit.log](raw/box1/probe-final-state-suite-1/hit.log) |
| Forced miss | **PASS** — 15 attempts, 15 restores, 0 hits; restored session and continuation exact | [miss.log](raw/box1/probe-final-state-suite-1/miss.log) |
| Alternating generations | **PASS** — 132 attempts across 257 generated tokens; 66 hit / 66 miss; continuation exact | [alternate.log](raw/box1/probe-final-state-suite-1/alternate.log) |
| Abort mid-flight | **PASS** — generation 0 drained exactly once; a fresh 15-round hit session then stayed exact | [abort.log](raw/box1/probe-final-state-suite-1/abort.log) |
| SWA ring exclusion | **PASS** — `reason=swa-ring`, 0 attempts, 1 refusal; ordinary ring session exact | [ring.log](raw/box1/probe-final-state-suite-1/ring.log) |
| Round-stream exclusion | **PASS** — `reason=round-stream`, 0 attempts, 1 refusal; ordinary stream session exact | [stream.log](raw/box1/probe-final-state-suite-1/stream.log) |
| Conditional f32 restore | **PASS** — 262,144 f32 values; hit skips and both count/bonus misses restore exactly | Printed in every final gate; see [driver.log](raw/box1/probe-final-state-suite-1/driver.log) |

The long soak compared 32,071,680 live trunk-KV bytes, 712,704 live scratch-KV bytes, and 16,384
hidden bytes at its first boundary, then 33,408,000 / 742,400 / 16,384 bytes after continuation.
`recurrent_bytes=0` is the target's actual layer map, not a skipped comparison.

## Reconcile latency

The forced-miss timer begins before host saved-length preparation and ends after stage-0 restore
is published to and synchronized on the caller. It excludes the subsequent full serial verify
rerun.

| N | Thermal regime | Min | Median | Mean | P95 | Max |
|---:|---|---:|---:|---:|---:|---:|
| 15 | one warm process inside a block bounded by 26 C P8 before and 32/33 C P0 after | 0.063 ms | **0.071 ms** | 0.071 ms | 0.083 ms | 0.083 ms |

Raw per-round samples are in [miss.log](raw/box1/probe-final-state-suite-1/miss.log). This is the
reconcile control cost, not a serving throughput point and not the miss interval for increment-2
economics.

## Memory and snapshot/HBM accounting

The memory A/B used two simultaneous sessions sized to the serving capacity
`MEMRA_OPTI_CAP=262144`. `off` and forced `hit` were separate fresh processes, sampled every
100 ms. The block began at 28 C P8 and ended at 32/33 C P0.

| Device | Fork off peak | Fork armed peak | Sampled delta |
|---|---:|---:|---:|
| dev0 | 68,497 MiB | 68,497 MiB | **0 MiB** |
| dev1 | 76,881 MiB | 76,881 MiB | **0 MiB** |

The exact incremental device payload logged by the fork is **65,812 bytes on dev0 and 0 bytes on
dev1**. Dev0 consists of four 4,096-f32 seed buffers (65,536 bytes), 22 stage-local KV pointers
(176 bytes), 22 saved lengths (88 bytes), two forced-accept words (8 bytes), and one validity word
(4 bytes). The extra recurrent checkpoint contributes zero bytes on both devices because this
checkpoint has no GDN layers. The payload is below `nvidia-smi`'s 1 MiB reporting resolution, so
the exact byte accounting and the sampled zero-MiB delta agree.

For the inbox's increment-2 mechanism pricing: this target executes **no verify GDN forward**, so
verify-GDN HBM traffic is **0 bytes** and “bandwidth-bound or not” is **not applicable**. The direct
restore primitive moves a synthetic 1 MiB f32 state correctly, but it is not a substitute for a
real verify-GDN bandwidth measurement. ReplaySSM/Bole A/B pricing must wait for a GDN-bearing PP-2
artifact; no result here supports choosing or implementing either mechanism.

Retained memory evidence: [driver.log](raw/box1/memory-fullctx-1/driver.log),
[off-gpu.csv](raw/box1/memory-fullctx-1/off-gpu.csv), and
[hit-gpu.csv](raw/box1/memory-fullctx-1/hit-gpu.csv).

## Standing correctness battery

| Gate | Result | Retained evidence |
|---|---|---|
| Serial fresh processes | **PASS 10/10**, every 326-byte completion SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` | [driver.log](raw/box1/serial-boots-final-1/driver.log), [hashes.txt](raw/box1/serial-boots-final-1/hashes.txt) |
| `run-spec` K=1..8 | **PASS 8/8**, each K identical to the Step35 live B=1 target | [run-spec.log](raw/box1/run-spec-final-1/run-spec.log) |
| `kernel-check` | **ALL GREEN**, 376 `OK` lines including model-backed IQ4_XS cells | [kernel-check.log](raw/box1/kernel-rungen-1/kernel-check.log) |
| `run-gen` PP-2 | **MATCH** — prefill/decode argmax 6776 and batched-prime/tokenwise argmax 6776 | [run-gen.log](raw/box1/kernel-rungen-1/run-gen.log) |
| Local ownership tests | **PASS 3/3** — generation alternation, live-slot overwrite refusal, stale-tag teardown refusal | [test-stage-owned-snapshots.log](raw/local/test-stage-owned-snapshots.log) |

The final GPU receipts end with both devices at 0 MiB and no compute processes. No failure output
was hidden behind a parser: each remote command first retained combined stdout/stderr with `tee`,
then asserted its summary markers.

## Negative control retained

An intermediate expanded-gate run omitted `MEMRA_SPEC_DEVACC=1`. The exact comparator caught the
expected two-counter violation: host KV length advanced beyond the 128-token prime while device
lengths remained at 128. The final harness now refuses that configuration with
`requires-device-accept`; the failed diagnostics remain under
[`raw/box1/probe-expanded-hit-control-1/`](raw/box1/probe-expanded-hit-control-1/) through
`probe-env-isolation-1/` rather than being discarded.

## Increment-2 handoff

Proceed with the depth-1 controller only under these boundaries:

1. Replace forced accept words with the actual K=1 accept count and optimistic bonus identity;
   carry the real generation-tagged boundary ticket and allow only one successor stage 0 before
   predecessor resolution.
2. Keep forced ON/OFF diagnostics and default off; increment 2 must produce the first c=1
   throughput, actual `I_hit/I_miss`, breaker, and phase-timing receipt.
3. Price against the merged-seam floors exactly as instructed: **63.082 tok/s seam,
   55.365 tok/s serial, 121.051 tok/s plain**. Increment 1 did not add a comparable throughput
   point; its 0.071 ms reconcile micro-timer must not be compared to those floors.
4. Repeat the expanded increment-2 battery from DESIGN section 6 before any admission or
   promotion. For a GDN-bearing target, first add real model-backed recurrent identity,
   per-device dual-snapshot memory, and verify-GDN HBM/bandwidth evidence.

No push, merge, tag, release, perf-board edit, `cargo fmt`, `rustup`, or `nsys` was performed.
