# Batched speculative verify across sessions

Lane: batch-spec, started 2026-11-01. The goal is serving scale on bandwidth-bound devices: GB10
(DGX Spark) running dense Qwen3.8-27B. Today each speculating session runs its own draft and
verify round, serially, in phase (a) of the worker tick. On a 27B step that is bound by weight
reads, this serial verify is what holds memra at about 18 tok/s, against 70-120 tok/s for
vLLM/SGLang with 4-8 users on the same box.

## Landed

1. **Sampled demotion** (`SAMPLED-DEMOTE.md`). Sampled MTP and DFlash2 sessions now leave
   serial spec at the HIGH watermark, with an exact Philox counter handoff.
2. **`decode-batch-bench --verify-t`**. Measures C(R), the cost of one verify forward over R rows
   at a fixed depth. It runs under the exact scope; without it, R=16 falls into the prime GEMM
   class.
3. **Pooled verify core** (`verify_pooled_dev`, spec.rs). One target forward over the stacked
   rows of N sessions, with R = sum(t_i) <= 16.
   - Shared rows: weights, norms, FFN and head.
   - Per session: GDN state, via row views of the same packed kernels.
   - Attention: per-row KV pointers through the seqs twins, one launch per split rung.
   - `decode-batch-bench --pooled` pins it bitwise against per-session verify. The check covers
     both the logit rows and a follow-up step that reads the written state.

## Measured so far (RTX 5090 Laptop, q9, depth 2048)

| R or shape | verify ms | plain batched step at B=R |
|---|---:|---:|
| 1 | 11.85 | 12.0 |
| 4 | 16.25 | 19.3 |
| 8 | 23.65 | 28.2 |
| 16 | 49.15 | 58.0 |

| pooled shape | serial per-session ms | pooled ms | gain |
|---|---:|---:|---:|
| 4 x T4 | 64.26 | 50.07 | 1.28x |
| T1..T5 | 73.40 | 48.37 | 1.52x |
| 2 x T2 | 26.77 | 16.44 | 1.63x |
| 2 x T8 | 47.05 | 49.42 | 0.95x |

The 5090 is compute-heavy relative to a 9B model, so C(R) grows steeply. GB10 numbers come from
`spark-cr.sh` (owner GGUF 27B), and the gate policy is calibrated on them.

## Next: the serving round (MTP first)

The MTP round lives inside `generate_spec_inner2`: a 1,700-line loop (spec.rs, `while keep_going
&& out.len() < max_new`) with one verify call site (`decode_step_t_core_vg`). Pooling needs the
loop split in three around that site:

1. **prepare**: draft K_i and build `verify_tokens`. This covers the pending carry, adaptive K,
   the p-min cut and the snapshot.
2. **verify**: `verify_pooled_dev` over every ready session.
3. **finish**: the per-session accept walk (greedy or sampled), `commit_verified_prefix`, the
   pending stash, telemetry, budget clamp and stop handling.

The round's locals become a `SpecRoundState` owned by the session, so a round can pause at the
verify boundary. The worker's phase (a) then runs lockstep rounds:

1. prepare every pooled session;
2. one pooled verify per chunk of 16 rows;
3. finish every session;
4. repeat for M rounds.

Arms that do not pool keep today's serial round: ROUND-STREAM, verify graphs, PP, rowwise, and a
row below the fa vec floor. Exactness gate: the serving greedy stream is byte-identical
pooled vs serial at c=2..8, K=1..8, including ragged K. The sampled stream is byte-identical
when the drafts are bit-equal.

Then, in order:

- **Batched MTP draft**: lockstep draft steps across sessions, so the MTP layer and head are read
  once per step.
- **Cost-model gate**: pick K_i per tick from C(R), acceptance EMAs and draft cost, replacing the
  LOW/HIGH bands. In the same A/B, relax the sampled admission laws (SOLO sampled restore, the
  "one non-demotable dspark row" block). They are being relaxed together with pooling because
  their measured c2 loss is the serial-verify cost that pooling removes.
- **DFlash2 port**: the same core, plus a batched drafter block forward, with vt capped at
  16/B.
- **v2**: plain rows join the ragged step as T=1 with no draft, giving one step per tick like
  vLLM/SGLang. Also a fused rows-by-seqs FA kernel and graph capture keyed on R.
