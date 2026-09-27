# WP-B day 51: memra#476, the FA partial pool grown at boot (the owner's approval, 2026-09-27)

Written before any code. The owner approved DAY28 3.3's booking point on 2026-09-27 (relayed by the lead with the
owner's rulings of that day): the FA partial pool is grown at boot to every rung the served context and the batch cap
can reach, so no request grows it after ready. The lead closes memra#476 when this merges.

## 1. Pre-registration

### 1.1 What the code reads first (the census DAY28 3.3 was written from, rechecked against the tree today)

The pool is `Engine::fa_part_pool`: three f32 buffers `(o, m, l)`, grow-only, retire-on-grow, never freed
(`fa_part_pool_grow`). Every FA decode and verify site sizes its demand the same way,
`o = rows x n_head x n_splits x head_dim` and `ml = rows x n_head x n_splits`, with `n_splits = ceil(t_kv / sp)` and
`sp = fa_split_keys(t_kv, n_head_kv)`:

| site | rows |
|---|---|
| eager decode (`fa_decode_kvmod_view`, `fa_decode_dc_q8`, `fa_decode_dcw`) | 1 |
| verify rows (`fa_decode_rows`, `fa_decode_rows_dc`, `fa_decode_rows_w`) | the verify's K + 1 |
| spec verify through the seqs kernel (`spec.rs`, `rows_batched`) | K + 1 |
| the batched plain trunk (`fa_decode_batch_seqs_v4` from `decode_batch.rs`) | B, up to the model's decode wave cap |
| the TP rows path (`fa_decode_dcw_rows`, 188-SM class) | t <= 32 rows of a TP step |

One correction to DAY28 3.3, from this census and not from any result: 3.3 put the rows multiplier on the 188-SM class
only (the `fa_decode_dcw_rows` path). The batched plain trunk's seqs kernel grows the pool on both cards. DAY28's two
post-ready grows #3 fired at S8 with 8 in flight on the 5090 and on the target card alike. So the multiplier applies on
every card.

The ladder: `sp` grows with `t_kv`, so `n_splits` is not monotone across rung edges. The split count to cover is the
maximum of `ceil(t / fa_split_keys(t, n_head_kv))` over every `t` from 1 to the served context, not the value at the
served context alone.

### 1.2 What lands

- **Engine:** `fa_part_pool_demand(head_dim, n_head, n_head_kv, served_ctx, rows) -> (o_len, ml_len)`, a pure function:
  the maximum split count over the ladder, times the rows, heads and head size. Also
  `Engine::fa_part_pool_pregrow(o_len, ml_len)`, which runs one `fa_part_pool_grow`, prints the pool's own `[fa-pool]
  grow` receipt, and is a no-op when the pool already holds that much.
- **Geometry from the plan, not an architecture list:** every `Full` and `SlidingWindow` attention layer of the model's
  `ModelPlan` (and its MTP blocks) whose value head size is in the FA vector class (at most 256, a multiple of 32)
  contributes `(value_head_dim, query_heads, kv_heads)`. A sliding window caps that layer's `t_kv` at its window. MLA,
  GatedDeltaNet and Kimi layers contribute nothing; they do not read this pool. The pool takes the maximum `o_len` and
  `ml_len` over the model's geometries.
- **Rows:** the larger of the model's decode wave cap (`decode_chunk_policy`) and the spec verify's rows (K + 1: 6 on the
  automatic K table, whose largest K is 5; `MEMRA_SPEC_K + 1` when that pin is set).
- **Where:** a new `pregrow_fa_part_pools` step in the worker, run for every loaded model on every device engine
  (`for_each_device_engine`), right before `run_boot_calibration`. It runs whether or not the calibration is armed, so
  the calibrated floor is measured with the pool at its final size. Boot line per model and device:
  `[fa-pool] pre-grown model=<m> dev=<d> served_ctx=<c> rows=<r> o_len=<o> ml_len=<ml> bytes=<b>`.
- **Unchanged:** no numeric program changes (the pool's contents are per-launch scratch; only its size moves), no new
  flag, no admission term. A shape the plan's geometry does not cover still grows the pool lazily, as today, and that
  grow's receipt names it.

### 1.3 What it costs at boot (computed from 1.1 before any run)

The ladder's top rung is `sp = 128` above 16,384 on both card classes, so above 16,384 the maximum split count is
`ceil(served_ctx / 128)`.

- **Local 9B (16 query heads, 4 KV heads, value head 256), `MEMRA_CTX=65536`, wave cap 16:** 512 splits; o_len
  `16 x 16 x 512 x 256 = 33,554,432` f32; ml_len `16 x 16 x 512 = 131,072`. Bytes `4 x (o + 2 ml) = 135,266,304` (129 MiB).
  DAY28's lazy walk ended at o_len 4,259,840 after five grows, 33.3 MB with the retired buffers.
- **Target 27B (24 query heads, 4 KV heads, value head 256), the checkpoint's 262,144, wave cap 16:** 2,048 splits;
  o_len `16 x 24 x 2048 x 256 = 201,326,592` f32; ml_len 786,432. Bytes `811,597,824` (774 MiB). This is the whole
  reachable envelope: one session at 262,144 in a 16-row batch sizes all 16 rows' partials. Lazily, that batch would
  grow the pool mid-life to the same size plus the retired ladder.

The lead asked to register first. These footprints are the approved booking point's cost ("the final rung ... more on
the rows path", DAY28 3.3). They are stated here so that the owner sees the 774 MiB on the target card before it lands.

### 1.4 Cells (the DAY28 walk as the after cell)

`run-day28-cell.sh`'s fixed sequence (warm, S1, S2, S4, S8, L, S8b), shadow mode, on the new binary.

- **Local:** the 9B at `MEMRA_CTX=65536` under `/tmp/memra-5090.lock`.
- **Target card:** the 27B (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c...1e024a`), `MEMRA_CTX` unset, through the
  collector on `/tmp/memra-gpu.lock`.

Clauses:

- **P1 (the headline):** `grows after ready = 0` on both cards, from `day28-parse.py`'s line.
- **P2:** the boot's `[fa-pool] pre-grown` line reads the o_len and ml_len of 1.3 for the served geometry.
- **P3:** DAY28's G3 (every admit's `booked_real - kv_hat` exact on the exact rows) and G4 (zero Overloaded, zero OOM).
- **P4 (a reading):** the calibrated transient floor, the prefix-cache budget and `admit-predict`'s derived budget,
  beside DAY28's before, and the device `memory.used` at ready.

A server test pins the demand arithmetic: the ladder's maximum across rung edges, the rows rule, the plan's geometries
(a sliding window caps its `t_kv`; MLA and GatedDeltaNet contribute nothing) and the byte figures of 1.3.

### 1.5 Price

Code and CPU tests about half an agent-day; the two cells about 10 minutes each plus builds.
