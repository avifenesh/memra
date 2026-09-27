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

### 1.5 Addendum A (2026-09-27, the runners as built, before either cell)

- **The target card's cell runs through the lane's hold runner, not the collector.** The collector
  (`tools/tier-battery.py`) takes the box lock non-blocking. A chain that waits for the lock and then starts the
  collector loses the lock to any blocking taker woken by the release, which is the starvation DAY48 addendum D placed.
  So `pro-single-day51/chain.sh` holds `/tmp/memra-gpu.lock` (`rig-hold.sh`), checks the card idle under the hold,
  and runs `run-day28-cell.sh after` with `LOCK=none` and the hold's fd closed. The cell script, its client and parser,
  the model, the shape and the clauses are 1.4's. The collector's `command.capture.json` is not produced; the cell's own
  files are the receipts, as on the local card.
- **The local cell** (`rtx5090-day51/run.sh`) builds the tree's release server under the CPU quota, then runs the same
  way on `/tmp/memra-5090.lock`, with a 4 h bound on the hold and a 16 GB host floor.
- **The target card as the twenty-first sitting** (`pro-single-b-sitting21.sh`, `S51 = 11ee6ed77`, receipts
  `b-day51`): the 27B (sha256 `1facf36c...1e024a`) staged at `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`.

### 1.6 Price

Code and CPU tests about half an agent-day; the two cells about 10 minutes each plus builds.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target card (the twenty-first sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-27 19:14 to 19:15Z)

Chain tree `c6d7f034f`; the binary built on the box from `11ee6ed77` (sha256 `2752bf5d...`, mirrored by hash). The 27B,
sha256 `1facf36c...` checked at start, `MEMRA_CTX` unset. Receipts at `pro-single-day51/box/`: the sitting's own
`MANIFEST.sha256`, and the lead's box-side manifest, re-checked. The hold runner held the box lock at once and found the
card idle under the hold. The cell ran `rc=0`, 25 of 25 requests `200`. Verbatim (`cells/after/REPORT.txt` and the boot):

```
1790536464025 [fa-pool] pre-grown model="q38" dev=0 served_ctx=262144 rows=16 o_len=201326592 ml_len=786432 bytes=811597824 (pool holds o_len 201326592 ml_len 786432)
1790536464025 [fa-pool] grow #0 dev=0 o_len 0 -> 201326592 ml_len 0 -> 786432 (retired kept, zero=false)  g_i=811597824 G_fa=811597824
G_fa at ready (the probe's grows, inside the floor's measurement) = 811597824; grows after ready = 0
G3: exact rows max |residual| = 0 (tolerance 0); line~ rows max |residual| = 396772 (tolerance 2e6 at the 1 MB grain)
G1: growth(end) = 0 (G_fa after ready 0 + delta_D 0) <= 10% floor = 237397606: PASS
G2: while inflight > 0, max_underbook_real = 7263370392 <= floor = 2373976064: FAIL
G4: Overloaded/OOM lines = 0: PASS
```

- **P1 PASS: `grows after ready = 0`.** The pool was grown once, at boot, to 1.3's size. The walk that grew it twice after
  ready in DAY28 3.1 (S8 at 8 in flight, L at 22,760 keys) grows it no more. The probe's own three grows are gone too.
- **P2 PASS:** the pre-grown line reads `o_len=201326592 ml_len=786432 bytes=811597824`, 1.3's figures to the element.
- **P3 PASS:** G3 exact (`residual = 0` on every exact row) and G4 (no Overloaded, no OOM line).
- **G2 is not a clause of this day** (1.4 names P1 to P3). It reads FAIL, and it is DAY28 3.1's reading, not a graph
  term:
  - The maximum in-flight under-booking, 7,263,370,392 B, is at the `S8b-0` admit (one request booked, 1,494,336,360
    B, `mem.used` 27,414 MiB).
  - It is the device delta the walk leaves behind between bursts. The pool's reserved bytes rose from 19,360,907,264 B
    at ready to 28,118,614,016 B. At that sample 2,676,090,064 B of it were cached free blocks (`RELEASE_THRESHOLD =
    u64::MAX`), which `effective_free_bytes` hands back to the gate.
  - The rest is the retention the budget intends: the walk's prompts as prefix-cache entries under the 15.9 GB
    budget, the last parked spec sessions, and L's 22,760 x 31,552 B = 718 MB.
  - Subtracting `growth(t)` moves it by 0 B now, against DAY28's 38.6 MB: the graph term's share of the under-booking
    went from 0.5 % to nothing.
  - DAY28 3.1 read 7,443,968,680 B at the same admit; this is 180 MB lower.
- **P4 (readings), after against DAY28's before on the same card class and model:**
  - The calibrated floor: 2,264 MB against 2,194 MB (+70 MB).
  - The prefix budget: 15,883,042,816 B in both.
  - `effective_free_bytes` at boot: 83,261,614,080 B against 84,065,415,872 B (-804 MB).
  - `admit-predict`'s derived budget: 65,003,806,992 B against 65,881,157,328 B (-877 MB, the pre-grow's 812 MB plus
    the floor's 70 MB).
  - `mem.used` at ready: 19,062 MiB against 18,325 MiB (+737 MiB).
  - The pool reserved at ready: +771,751,936 B.
- **What it means:** the booking point works as the owner approved it. No request grows the FA partial pool after ready
  on this model and card. The cost is the reachable envelope, 774 MiB, booked at boot, and the admission budget is
  smaller by that much.

### 2.2 The 5090 (the 9B at `MEMRA_CTX=65536`, `rtx5090-day51/`, 2026-09-27 19:23 to 19:25Z)

The release server built from `11ee6ed77` (`binary.sha256`). The hold runner waited 1,287 s for `/tmp/memra-5090.lock`
behind another lane's A/B, then found the card idle under the hold: no compute app, 85 MiB used. DAY28's local run had a
1.4 GB co-tenant. `rc=0`, 25 of 25 requests `200`. Verbatim:

```
1790537052398 [fa-pool] pre-grown model="q9" dev=0 served_ctx=65536 rows=16 o_len=33554432 ml_len=131072 bytes=135266304 (pool holds o_len 33554432 ml_len 131072)
G_fa at ready (the probe's grows, inside the floor's measurement) = 135266304; grows after ready = 0
G3: exact rows max |residual| = 0 (tolerance 0); line~ rows max |residual| = 775928 (tolerance 2e6 at the 1 MB grain)
G1: growth(end) = 0 (G_fa after ready 0 + delta_D 0) <= 10% floor = 161061273: PASS
G2: while inflight > 0, max_underbook_real = 3325677512 <= floor = 1610612736: FAIL
G4: Overloaded/OOM lines = 0: PASS
```

- **P1 PASS (`grows after ready = 0`), P2 PASS (1.3's 33,554,432 and 131,072, 135,266,304 B), P3 PASS (G3 exact, G4).**
  DAY28's two post-ready grows (S8 at 8 in flight, L at 1) and the probe's three are gone.
- **G2 (not a clause) reads as on the target card:** 3,325,677,512 B at the `S8b-0` admit, one request booked. Over the
  walk the pool's reserved bytes rose from 8,891,924,480 to 13,153,337,344; 1,654,200,848 B of that were cached blocks at
  the sample. The rest is the intended retention (the prefix entries under the 2,052 MB budget, the parked spec
  sessions). `growth(t)` is 0 against DAY28's 25.8 MB.
- **P4:**
  - The measured floor is 1,302 MB against 1,266 MB. The static 1,536 MB floor serves in both.
  - The pool reserved at ready is +100,663,296 B.
  - DAY28's co-tenant makes the budget and `mem.used` comparisons not like for like. This run: budget 12,633,008,128 B,
    8,861 MiB at ready.
- Both cards read the same: the booking point removes every post-ready grow of the FA partial pool, at the boot cost 1.3
  computed.

