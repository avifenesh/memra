# MEMRA_MOE_GROUPED PRO transfer re-sweep

## Verdict: STILL-BLOCKED

`MEMRA_MOE_GROUPED` must remain off by default.

The unchanged resident-KAT transfer gate reproduces on the target-class NVIDIA RTX PRO 6000
Blackwell Server Edition: grouped is **2687.5 tok/s** versus rollback at **8193.7 tok/s**, a
**-67.2%** regression, with **0/5** grouped pairwise wins. All ten timed processes exited zero.
This failing PRO result is sufficient to keep the default flip blocked even if the later
serve-smoke failure proves unrelated.

The exactness battery also stopped red under grouped ON. `serve-smoke` reported one failed Q35
mixed-c=4 exact-token cell. Per the frozen stop rule, no Q35 or Step35 performance/TTFT arm ran
after that mismatch. The failure was not reproduced with grouped OFF, so this report does not
assign its cause to grouped dispatch.

No default-flip diff or `docs/FLAGS.md` rewrite is proposed because this lane is not flip-ready.
The inbox's conditional placement-scoped recommendation also does not activate: the PRO/server
placement itself failed the unchanged transfer gate.

## Rig and provenance

- Source: exact `v0.81.2`, commit `18885ec479d897a3e8c42b0d408a71fa3edaa708`.
- Build: CUDA 13.2, auto-detected `sm_120a`, box1 host `<private-host-redacted>`.
- Scored card: physical GPU 1, NVIDIA RTX PRO 6000 Blackwell Server Edition, UUID
  `GPU-2b4cf166-fd33-f161-8536-ca04bc72280c`, driver 595.71.05.
- Isolation: `CUDA_VISIBLE_DEVICES=1` under one `/tmp/memra-gpu-1.lock` hold.
- Co-tenant: `cx-cachesize` occupied physical GPU 0 under its separate coordination lane. The
  pre-lock snapshot showed GPU 0 at 24,211 MiB and 100% utilization while selected GPU 1 was at
  0 MiB and 0% utilization.
- KAT artifact SHA-256:
  `e35e23219a81590b9d4174eea4717d716dd62676c8c434f6b708f49a07310e1a`.
- Frozen pp2048 prompt SHA-256:
  `dc91551b1e83414616ebb8d65ee88d1af1fc8792dadffe0208b732d094adfc0d`.

The detailed source, binary, model, prompt, and harness hashes are in
[`provenance.log`](raw/box1/single-20260812T221224Z/provenance.log). The actual co-tenant state is
in [`coordination-before-lock.log`](raw/box1/single-20260812T221224Z/coordination-before-lock.log).

## Same resident-KAT transfer gate

The original protocol and verdict definition were retained verbatim:

> `protocol=N=5 independent processes per arm; adjacent interleaved pairs; order alternated; one warmup plus one timed prime per process; one flock hold`
>
> `throughput_assertion=FAIL grouped_slower_or_run_failed`

Thus the pass threshold remained: all ten processes exit zero and the grouped median is not
slower than rollback. There was no substituted model, shape, threshold, or process-reuse scheme.

| arm | timed samples, tok/s | median | paired wins | exit status |
|---|---:|---:|---:|---:|
| grouped OFF / rollback | 8195.7, 8193.7, 8191.7, 8221.2, 8188.7 | **8193.7 tok/s (N=5; 27-48 C gate range; 2295-2422 MHz at >=50% utilization; one lock hold)** | - | 5/5 zero |
| grouped ON | 2688.0, 2687.5, 2687.7, 2687.5, 2687.5 | **2687.5 tok/s (N=5; 27-48 C gate range; 2295-2422 MHz at >=50% utilization; one lock hold)** | **0/5** | 5/5 zero |

Continuous 250 ms telemetry covered the entire 22:12:59-22:14:33Z gate. The selected GPU was P0
for 372/374 samples; the full sampled active-clock range, including ramp, was 1342-2422 MHz. At
the 41 samples with at least 50% GPU utilization, it was 39-48 C and 2295-2422 MHz. These are
low-temperature, stable-clock target-rig runs, not a thermally capped laptop transfer result.

The parsed per-run table, including the physical card index and UUID in every result row, is
[`prefill/results.tsv`](raw/box1/single-20260812T221224Z/prefill/results.tsv). Individual logs were
captured first and parsed second.

## Exactness battery

All checks below ran on physical GPU 1. The grouped model checks used
`MEMRA_MOE_GROUPED=1`; the paired `run-gen` checks explicitly used `MEMRA_MOE_GROUPED=0`.

| check | result |
|---|---|
| KAT `kernel-check` | `ALL GREEN (105 cells, 5 skipped)`, exit 0 |
| Q35 `kernel-check` | `ALL GREEN (107 cells, 5 skipped)`, exit 0 |
| KAT `run-gen`, grouped ON | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH; 200 `BYTE-IDENTICAL` MoE-oracle rows; zero `MISMATCH`; exit 0 |
| KAT `run-gen`, grouped OFF | the same two argmax MATCH signatures; exit 0 |
| Q35 `run-gen`, grouped ON | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH; 200 `BYTE-IDENTICAL` MoE-oracle rows; zero `MISMATCH`; exit 0 |
| Q35 `run-gen`, grouped OFF | the same two argmax MATCH signatures; exit 0 |
| Q35 `run-spec` | K=1..8: 8/8 self-consistency PASS; overall PASS; exit 0 |
| `serve-smoke`, grouped ON | **FAIL**, exit 1; `serve-smoke: 1 failed` |

The direct failure lines are quoted verbatim from
[`serve-smoke.log`](raw/box1/single-20260812T221224Z/exactness/serve-smoke.log):

> `FAIL: Q35 mixed c=4 exact-token regression`
>
> `serve-smoke: 1 failed`

The machine row immediately before those lines records `"cell_clean": false`,
`"expected_completion_tokens": 60`, and 20/20 short entries with
`"completion_tokens": 25` and `"finish_reason": "stop"` (18 cache-role hits, two misses). It also
records eight seed failures as `seed failed: None`. The captured evidence does not identify why;
under the mandatory mismatch stop rule, this lane did not run a diagnostic OFF arm or continue
into performance.

## Stop-limited coverage

The gate driver intentionally placed exactness before current-tip Q35 prefill and serving A/B.
Consequently, after `serve-smoke` exited 1:

- Qwen3.6-35B-A3B board2048 prefill ON/OFF was not run;
- cold TTFT and mixed hit/miss TTFT at c=4 and the knee were not run;
- Step35 artifact discovery/performance was not continued; and
- no result from the older +53% to +63% Step35 campaign was carried forward as current-tip
  evidence.

This is a deliberate stop, not missing raw output. The PRO transfer regression already answers
the primary stale-verdict question: it exists on the target-class card.

## Mixed/spill safety and repository scope

This lane changed no runtime source. The exact tagged source retained the uniform-layout guards
and the metadata-aware `expert_layout()` / `max_expert_bytes()` paths inspected before launch.
No mixed or spill artifact was routed through a uniform-only fused kernel.

The lane did not edit the generated perf board, flip a default, merge, tag, push, or touch the
live serve box.

## Tenant-clean shutdown and raw integrity

At 22:23:16Z, after the lock was released, physical GPU 1 was at **0 MiB**, 0% utilization, 27 C,
and 180 MHz. No lane-owned processes, ports, or coordination locks remained. Physical GPU 0 still
held only the expected `cx-cachesize` co-tenant. See
[`tenant-clean-post.log`](raw/box1/single-20260812T221224Z/tenant-clean-post.log).

The run-local [`MANIFEST.sha256`](raw/box1/single-20260812T221224Z/MANIFEST.sha256) and the lane
root [`SHA256SUMS`](raw/SHA256SUMS) cover the preserved receipts.
