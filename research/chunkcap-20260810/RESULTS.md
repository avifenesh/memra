# Step35 decode chunk-cap audit — 2026-08-10

## Verdict

**KEEP cap 8. Do not lift it and do not expose an upward Step35 serving knob.** The cap is the
current exactness boundary, not a tuning constant. Step35's dedicated batched walk is promoted
and gated only through B=8. Widths B=9..16 require the model-wide exact-16 admission predicate,
which rejects every MoE FFN; this IQ4_XS Step35 checkpoint has a 288-expert MoE bank. B>16 has no
exact kernel class in the current engine.

The lane's stop condition therefore fired before code changes or GPU work. There is no new
cap-16/cap-32 throughput result, TTFT tail, or step-p99 result to report. Box1 was not touched and
`/tmp/memra-gpu.lock` was not acquired.

## Where the ceiling comes from

The worker computes one cap per loaded model and uses it to partition all ready rows into serial
chunks:

```text
crates/memra-server/src/worker.rs:3688-3694
// batched steps in per-model chunks (chunk_cap_for: exact-16 tier models chunk
// at 16, everything else 8; MEMRA_DECODE_BATCH_CAP is the explicit door).
for chunk in group_chunks(&active, &ready, &chunk_caps) {
```

For Step35, the worker deliberately clamps the environment value rather than allowing the generic
measurement door to widen the batch:

```text
crates/memra-server/src/worker.rs:5859-5875
// Chunk cap 8: the exactness-tier width (IQ4_XS trunk + 288-expert MoE refuse
// exact16 by predicate — `decode_batch_exact16_ok` requires non-MoE — so 16 is
// structurally out).
...
return cap.clamp(1, 8);
```

That produces the measured ceiling from the predecessor study: at c=64 all 64 rows are ready,
but the outer tick executes eight B=8 walks. With the requested grouped-on serving shape, step
p50 grows from 48.30 ms at c=8 to 381.68 ms at c=64 while full-window aggregate output rises only
1.03%, 128.38 to 129.70 tok/s
([throughput receipt](../throughput-20260810/RESULTS.md#why-it-flattens)). Admission was not the
limiter.

## Why cap 8 is a correctness boundary

The engine contract is explicit:

- `crates/memra-engine/src/decode_batch.rs:10-22`: B=2..8 is the per-row bit-identical tier;
  B=9..16 is admitted only when `decode_batch_exact16_ok()` proves every relevant operation has
  an exact b16-class path; B>16 has no exact class.
- `crates/memra-engine/src/decode_batch.rs:138-149`: the non-exact m=9..16/m>=16 dispatches were
  attributed to different reduction/GEMM configurations and measured at roughly 1.3e-1 to
  2.3e-1 max logit difference versus isolated decode.
- `crates/memra-engine/src/decode_batch.rs:212-235`: the predicate rejects the model as soon as
  an FFN is MoE.
- `crates/memra-engine/src/decode_batch.rs:514-540` and `:706-716`: the unsplit and PP-N paths
  both enforce the same width policy. `MEMRA_DECODE_BATCH_CAP` above the exact tier is described
  as a measurement probe, not a serving correctness promotion.

The retained exact-tier receipts make the failure mode concrete. On the earlier Q8_0 controlled
probe, B=16 without an admitted exact tier differed at step 0 for all 16 rows, and B=32 differed
at step 0 for all 32 rows and later diverged sampled streams
([increment ledger](../batched-tick-inc3-20260801/increments.md#3a-b1632-bit-isolation-verdicts-per-chunk-size),
[`fN-b32-door-s32.log`](../batched-tick-inc3-20260801/fN-b32-door-s32.log)). Those receipts are
not Step35 measurements; they establish why the generic door is quarantined behind an exactness
predicate. For Step35 specifically, the guarantee fails at the first proposed wider row, B=9,
because its MoE FFNs make the predicate false. This audit does **not** claim that visible Step35
text corruption has been observed at B=9.

Current NVIDIA CUDA guidance also confirms the underlying numerical mechanism: changing a
parallel floating-point reduction's operation order can change its result, so a different kernel
configuration cannot be presumed bit-identical without a gate
([CUDA C++ Best Practices Guide, section 7.3.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#floating-point-math-is-not-associative)).

## The old PP2 garbage bug versus today's cap

These are separate correctness layers:

1. The original B>1-over-PP2 bug routed Step35 through the generic Full geometry and returned
   HTTP-200 garbage. The repair pinned Step35 to B=1
   (`research/step-sku-20260807/PROGRESS.md:108-119`).
2. Commit `c5cd6a35` added a dedicated, stage-scoped Step35 batched walk, made the generic walk
   unreachable for this architecture, and lifted the pin only to the proven cap-8 tier
   (`crates/memra-engine/src/decode_batch.rs:786-817`, `:844-855`).
3. The promotion battery proved B=1/2/4/8 over PP2 with zero differing bits, plus byte-identical
   serving and canary teeth. It never promoted B>8
   (`research/step35-batch-20260808/PROGRESS.md:171-183`).

Thus cap 8 is not leftover fear from the fixed generic-geometry bug. It is the next, still-live
exactness admission boundary.

## Requested A/B and gates

No new arm was run because doing so required removing the Step35 fail-closed clamp after the
correctness-motivated stop condition had been established.

| cap | c=16 | c=32 | c=64 | gate/QoS status |
|---:|---|---|---|---|
| 8 | Existing N=3 receipt: 128.80 tok/s, step 96.23/97.78 ms p50/p99 | Existing N=3 receipt: 129.47 tok/s, step 191.31/192.73 ms | Existing N=3 receipt: 129.70 tok/s, step 381.68/383.37 ms | Previously gated through B=8; not rerun |
| 16 | **NOT RUN** | **NOT RUN** | **NOT RUN** | Exact-16 admission rejects this MoE checkpoint |
| 32 | **NOT RUN** | **NOT RUN** | **NOT RUN** | No exact kernel class above B=16 |

The cap-8 values are predecessor grouped-on medians for the same 128-token prompt / 256-token
generation shape, included only to preserve the known ceiling; they are not new lane measurements.
Because the correctness gate stops the wider arms before service, TTFT and admission-starvation
tails are intentionally unmeasured rather than reported as passing.

## Recommendation and what would reopen the question

Keep the code default and Step35 upward clamp exactly as they are. Do not parameterize cap 16 or
32 as a serving-config recommendation: the existing environment door above the exact tier is a
research probe, and exposing it here would weaken a correctness contract.

The throughput question can be reopened only after a Step35/MoE exact-wide tier exists and its
admission predicate covers every projection and expert path. That promotion would need, before
any throughput claim, Step35 PP2 per-row bit-identity at the candidate widths, a widened
non-vacuous geometry gate and canary, run-gen argmax, serve-smoke, and the full target-rig battery.
Only then would the requested interleaved N=3 throughput and QoS sweep be admissible.

No runtime source, serving default, generated performance board, release, tag, remote workspace,
or model artifact was changed. No new raw GPU logs exist because the mandated stop occurred before
the measurement block.
