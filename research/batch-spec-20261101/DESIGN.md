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
4. **Pooled verify in the serving round** (`ROUND-SPLIT.md`, `POOLED-ROUND.md`). Same-model MTP
   continuation bursts run in lockstep and each round's verifies share one target forward.
5. **Pooled MTP draft** (`BATCHED-DRAFT.md`). In the lockstep cohort, draft step j is one NextN
   head forward over the members still drafting (`spec_round_draft_pooled`), bitwise equal to
   each member's serial draft chain.
6. **Cost-model gate** (`COST-GATE.md`).
7. **Pooled DFlash2 serving round** (`DFLASH-POOLED.md`). The dspark burst is split into
   begin/prepare/verify/finish/end. The pooled walk fills each session's drafter tap rows, and
   DFlash2 continuation bursts run in lockstep cohorts, byte-identical to serial at c = 1-8.
   Members are chunked at 16 rows instead of capping vt, and a VRAM headroom guard bounds the
   checkpoints held at once.
8. **Unified ragged step** (`UNIFIED-STEP.md`). Same-model plain decoding sessions on the
   device-sampled batched path ride the pooled MTP lockstep, one row per round inside the round's
   pooled forwards. Each plain stream stays byte-identical to its phase (c) stream. Captured
   pooled forwards (v2b) are exact, flat on the 5090, and opt-in.

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

- **Batched MTP draft**: landed (item 5 above, `BATCHED-DRAFT.md`).
- **Cost-model gate**: pick K_i per tick from C(R), acceptance EMAs and draft cost, replacing the
  LOW/HIGH bands. In the same A/B, relax the sampled admission laws (SOLO sampled restore, the
  "one non-demotable dspark row" block). They are being relaxed together with pooling because
  their measured c2 loss is the serial-verify cost that pooling removes.
- **DFlash2 port**: the pooled verify landed (`DFLASH-POOLED.md`). Still open: the batched
  drafter block forward. A 16/B vt cap was rejected, because it would make a sampled stream
  depend on cohort size.
- **v2**: plain rows join the ragged step as T=1 with no draft, giving one step per tick like
  vLLM/SGLang. Also a fused rows-by-seqs FA kernel and graph capture keyed on R.

## v2 probe: plain rows through the pooled walk (2026-11-02)

`decode-batch-bench --pooled-plain B` (`pooled-plain.log`, q9, RTX 5090) compares two paths over
B one-row segments:

- the serving plain step (`decode_step_batch_sampled_lean_masked`);
- `verify_pooled` (the pooled verify walk).

They are **bitwise equal**: every logit row and a follow-up step match at B = 2, 4, 8 and 16, at
depths 64 and 2048. A plain decode row can therefore ride the same forward as speculative verify
rows with no change to its output. This is the numeric precondition for the v2 unified ragged
step.
