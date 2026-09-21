# Session C day twenty: memra#365, the DFlash prefill tap: phase census, the standalone whole-prompt sink bounded, before and after on the local RTX 5090

Scope (lead brief, day 20): (1) the phase census of the DFlash tap on the current tree, before any change:
what the tap holds, when it is allocated, who reads it in what order, when it is freed, and the issue's
13,421,568,000-byte allocation reproduced by arithmetic; then the bounded alternative and the invariant a
gate proves. (2) A pre-registered before cell on the local RTX 5090 at an admitted length and at the length
that reproduces the tap refusal. (3) The bounded storage, if it is one seam; its CPU tests; the after cell;
the hit gate and the twin gate on the changed binary. (4) The records and the #365 comment. Tree: lane merge
of main `a18c936a3` (`6c30f2d39`, #618 with lane A's day 16; `research/INDEX.md` conflicted on the
inherited diff3 marker lines against main's spill-a day-16 row, resolved by keeping the row and dropping
the markers, `tools/check-conflict-markers.sh` `OK` before the commit). Every push of this lane today is
in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook prints `UNQUALIFIED
DEVELOPMENT ... no GPU qualification claimed` and appends a `log_skip` row); no qualification is claimed by
anything here; every cell is `executed-not-qualified`. Nothing here is a support state. No commits on main,
no other lane's worktree touched, no PR opened. Ruling 28 (integ27) stands: the arena lease handoff stays
scoped until the HOSTPREFIX decide-by review; nothing of it is touched today.

## Push mode, stated

`git push origin lane/spill-c-20260919` at `6c30f2d39` ran with `MEMRA_RELEASE_QUALIFICATION_MODE=development`,
verbatim from the hook: `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at
6c30f2d39599d2419a7db66e282bf9eb77d924e2; no GPU qualification claimed` then `pre-push: skip recorded in
.../.git/memra-gate-skips.log`. The same shape at every later push today (the Push section at the end).
The range carries engine source (main's merge and this lane's `dflash.rs` change), which is why the mode
is needed; no qualification is claimed.

## Artifacts on the local rig (named, hashed)

| Artifact | Path on this rig | SHA-256 |
|---|---|---|
| Qwen3.8-27B NVFP4/Q5K MTP GGUF (the target) | `/home/avifenesh/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` |
| q38 DFlash2 export, `config.json` | `/home/avifenesh/models/q38-dflash2/config.json` | `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980` |
| q38 DFlash2 export, `model.safetensors` | `/home/avifenesh/models/q38-dflash2/model.safetensors` | `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` |
| FR-Spec rank list (not used by the cells; hashed for the record) | `/home/avifenesh/models/q38-gguf/q38-ranks-sxc32768.gguf.txt` | `2f7c2e94683ac552724c2275380de6de4035f0f597c71df0826c50826a7d1bb9` |

All three hashes the cells use equal the `artifacts` entries of `research/dflash-tap-storage-20260908/qualification.json`
(#370's qualification): the same bytes #370 measured on. The export's `dflash_config`: `block_size 8`,
`target_layer_ids [5, 19, 33, 47, 61]` (five taps), `hidden_size 5120`, `sliding_window 2048`, five
`sliding_attention` draft layers, `architectures ["DFlash2DraftModel"]`.

## Task 1: the phase census (read from the tree at `6c30f2d39`, before any change)

**What the tap holds.** `DflashTapSink` (`crates/memra-kv/src/lib.rs:2484`): one device buffer `buf:
CudaSlice<f32>` of `t x n_taps x hidden` f32 words plus `layer_ids`, `hidden`, `t`, `base`, `origin`. Row
`r`, slot `s` holds the trunk's residual stream AFTER tapped layer `layer_ids[s]` for prompt token `r`
(the drafter's fc input layout: `buf[(base + r) * n_taps * hidden + s * hidden .. + hidden]`). On the 27B
with the q38 DFlash2 export that is `5 x 5120 x 4 = 102,400 bytes per prompt token`, f32, per-token and
per-tap-layer, nothing per trunk layer beyond the five tapped ones. The issue's request: `131,070 x 5 x 5120
x 4 = 131,070 x 102,400 = 13,421,568,000 bytes`, the number in memra#365 to the byte.

**Who writes it.** `HybridModel::dflash_tap` (`crates/memra-engine/src/hybrid_forward.rs:25621`), called
once per tapped layer per prime chunk inside `prime_chunk`; it copies the chunk's `t` rows into slots at
`base` (`copy_2d_dtod_async` at `t >= 128` under the qwen prime graph's support, else a per-row
`copy_view_into`; under a live `qwen_prime_graph` the graph's `prime_tap_table` writes the same rows). The
serial chunk loop (`hybrid_forward.rs:5933`) sets `taps.base = taps.origin + start` before each chunk so
the chunk's rows land at their prompt-relative offset in a whole-prompt buffer, or at 0 in a chunk-sized
one.

**Three tap sinks exist in `dflash.rs`; two shapes.**

| Site | Path | Sink size (rows) | Allocated | Consumed | Freed |
|---|---|---|---|---|---|
| `generate_spec_dspark` (`dflash.rs:3975`) | standalone qwen-hybrid entry point (`dspark_q38_gate`, `dspark_sample_gate`; the gate binaries) | **the whole prompt, `tp`** | after `Cache::new(max_ctx)`, before the single `prime_cache(prompt)` call over the whole prompt | after `prime_cache` returns: `DflashKv::new`, then a loop over 256-row windows in prompt order, each window copied into a fresh 256-row buffer, `ctx_features` (fc + hidden_norm), `ingest_ctx` (per-layer k/v projection, k head-norm, rope, appended at `kv.len`) at absolute positions `r0..r0+t_c` | at the end of the ingest block (`dflash.rs:4014`), before the first round |
| `generate_spec_dflash` (`dflash.rs:3600`) | standalone gemma arm (`gemma_gate`) | the whole prompt, `tp` | same shape | same 256-row loop | same |
| `DflashTapWalker::advance_chunk` (`dflash.rs:5251`) | SERVING cold prime and restored/parked suffix prime (`dspark_spec_session_new`, `dspark_spec_session_from_restored`, `prime_dflash_taps`; the worker) | **one trunk chunk, `rows`** (`prime_chunk_ranges`: 4096 at most, the last chunk folds a sub-16 tail, so 4111 at most) | inside `advance_chunk`, immediately before that chunk's `prime_cache(tokens[start..end], cache, tp - end)` | immediately after that call: `TapBatchCarry::chunk` copies the chunk's rows into a 256-row `batch` and ingests every full batch (`ctx_features` then `ingest_ctx` at absolute positions, `carry.next_position`), pending rows stay in the batch across the chunk edge; `finish` flushes the partial batch | the chunk sink drops at the end of `advance_chunk` (`take()`); the 256-row batch lives for the prime |

So the SERVING path is already bounded: #370 (merged `15f4bf96f`, refs #365) replaced the whole-prompt
sink with the chunk sink plus the 256-row carry, and its `[dflash-taps]` trace line reports `chunk_tap_bytes`,
`carry_bytes` and `former_full_tap_bytes`. memra#365's own release note says what remains: "The standalone
`generate_spec_dflash`/`generate_spec_dspark` whole-prompt sinks remain in this issue's scope." On the tree
today that is exactly what the census finds: the whole-prompt shape survives at `dflash.rs:3975` (qwen) and
`dflash.rs:3600` (gemma), nowhere else. The per-round keep-ingest sinks (`dflash.rs:4312`, `6261`) hold
`block_size` rows (8 x 102,400 bytes) and are not in question.

**Allocation order and what else is whole-prompt on the standalone path.** `generate_spec_dspark` allocates,
in order: the target `Cache::new(e, cfg, max_ctx)` (`max_ctx = tp + max_new + block + 8`), the tap sink
(`tp x 102,400` bytes), then `prime_cache(prompt)`; inside the monolithic call `prime_cache_overlaid_inner`
allocates `hiddens` of `t x n_embd` f32 (`131,070 x 20,480 = 2,684,313,600` bytes at the issue's length, a
second whole-prompt buffer, returned and dropped unused by this caller), then walks `prime_chunk_ranges(tp,
n_layers, gdn_grid)` chunk by chunk. Only after the last chunk returns does `DflashKv::new` run and the
ingestion begin. The tap's lifetime therefore spans the entire trunk prefill plus the entire ingestion; its
peak coincides with `hiddens` and the prefill transients.

**The bounded alternative is the serving walker, already in the tree.** `HybridModel::prime_dflash_taps`
(`dflash.rs:5354`, doc: "Standalone generation entry points are outside this helper") owns the chunk sink and
the carry. Making the standalone qwen entry point call it is one seam: the tap buffer's shape (one chunk,
4096 rows at most, 419,430,400 bytes; 4111 rows, 420,966,400 bytes when the last chunk folds a tail) and the
ingestion order (each chunk's rows copied into the 256-row carry batch and ingested before the next chunk is
primed; the carry's pending rows, under 256, are the only tap rows that survive a chunk edge). The values
and positions do not move, by construction:

- The trunk chunk grid is the same grid. On a single device `prime_pipeline_auto_geometry` is false, so
  `prime_chunk_ranges(tp)` is `fixed_prime_chunk_ranges(tp, 4096)` snapped to the GDN grid; the walker calls
  `prime_cache` once per such range, and inside that call `prime_chunk_ranges(end - start)` on a slice no
  longer than one chunk returns the single range `(0, end - start)` (`fixed_prime_chunk_ranges`: `t <= chunk`
  returns `[(0, t)]`; a folded 4096 + 15 slice folds again to one). Each `prime_chunk` call therefore runs
  with the same `tokens[start..end]`, the same `cache.pos`, and the same request-absolute `seq_end`
  (monolithic: `0 + tp + 0`; walker: `start + (end - start) + (tp - end)`, both `tp`). Same inputs, same
  kernels, same order: the trunk program is identical, and the tap rows written are identical bits.
- The ingestion partition is the same partition. The old loop ingests windows `[0,256), [256,512), ...` in
  prompt order with the final partial window last; `TapBatchCarry` with `pos0 = 0` emits exactly
  `source.chunks(256)` in order (`production_carry_preserves_whole_buffer_batches_across_chunks_and_boundary`
  pins this for 16..32760 rows). Each batch reaches `ctx_features` as a contiguous `rows x width` buffer
  (the full 256-row `batch`, or an exact-size `tail` copy for the final partial batch) at the same absolute
  positions, then `ingest_ctx` appends at `kv.len` in the same order. Row-independent ops on the same rows in
  the same batches: identical features, identical draft K/V.
- GDN chunking (`gdn_prime_grid_on`, `align_prime_ranges_to_gdn`) and the capture law are untouched: no
  split, no boundary, `dspark_boundary_split` is not involved on the standalone path (no capture), and the
  serving path is not changed at all.

What changes: two whole-prompt buffers (the tap and `hiddens`) become per-chunk buffers, and `DflashKv::new`
moves before the prime (the drafter KV exists during the prefill instead of after it; its size is unchanged).
Interleaving ingestion with the trunk chunks shares the stream; every chunk ends in `e.stream().synchronize()`
in the walker as it does in serving.

**The gemma arm stays as it is, and why.** `generate_spec_dflash` asserts `max_ctx <= sliding_window`
(2048), so its whole-prompt sink is at most `2048 x n_taps x hidden x 4` bytes by construction, and a prompt
under one chunk gives the walker a single range: the same one `prime_cache` call it makes today, with no
storage to save. It is not the issue's subject (a Qwen DFlash text request) and is left untouched.

**The invariant a gate proves.** For the same prompt, the same target, the same drafter export and the same
`max_new`, the drafter's proposals and the accepted tokens are bitwise identical between full-prompt and
chunked tap storage: the spec stream's ids (SHA-256 over little-endian u32), its length, and the
`[dspark-q38] acceptance <accepted>/<attempted>` line are equal, and both streams are `EXACT` against the
plain greedy oracle. On the storage side the largest `dflash.rs`-traced allocation falls from `prompt x
102,400` bytes to at most `420,966,400` bytes and the `[dflash-taps]` line reports `former_full_tap_bytes =
prompt x 102,400`.

## Task 2: cell `tapladder20` (local RTX 5090, before; pre-registered here before any result was read)

**Question.** On the unchanged standalone path, at which prompt lengths does the 27B admit a DFlash2 request
on this card, and at which does the tap allocation refuse; what are the output digest, the accepted and
attempted counts, the tap bytes, the prime wall and the device peak at each admitted length?

**Shape.** `day20-cell.sh tapladder20` under the collector (`tools/tier-battery.py --rig rtx5090
--external-lock`, one hold on `/tmp/memra-5090.lock`, 250 ms GPU telemetry). The gate binary
`dspark_q38_gate` built from `0e346fe7d` (the unchanged engine plus the receipt line added to the gate binary
today: `[dspark-q38-gate] <name>: prompt=<n> spec_sha256=<hex> plain_sha256=<hex> spec_len=<n> plain_len=<n>
prime_s=<s>`; the ids never reached stdout before). Five rungs, ascending, one process each: word-list
prompts (`day20-prompt.py`, the prime-cancel gate's 64-word inventory shape; calibration with `tok-check`:
1,726 tokens per 1,000 words, 17,197 per 10,000) of 4,763, 9,527, 19,055, 38,109 and 76,217 words, targeting
about 8,192, 16,384, 32,768, 65,536 and 131,070 tokens (the gate prints the exact count). `ngen 32`,
`MEMRA_SPEC_STATS=1` (the acceptance line), `MEMRA_ALLOC_TRACE=1` (the trace memra#365 names: `[alloc-trace]
<bytes> bytes from crates/memra-engine/src/dflash.rs:<line>` is the tap receipt). The gate runs the plain
greedy oracle first (its own whole-call `prime_cache`), then `generate_spec_dspark`, and prints `EXACT` or
`DIVERGED`. `nvidia-smi --query-compute-apps` before and after each rung. Device peak is the collector's
`memory.used` maximum inside the rung's marks.

**Expectation from source.** Every rung allocates `prompt x 102,400` tap bytes at `dflash.rs:3975` after the
plain oracle's `Cache::new`; on a 24,463 MiB card carrying a 16 GB target and the drafter, the 131k rung's
13.4 GB tap cannot fit whatever else does, so at that rung the plain oracle (cache plus 2.7 GB `hiddens`) or
the tap refuses with a captured `out of memory` line; the refusing allocation is the last `[alloc-trace]`
line before it. Lower rungs complete `EXACT`.

**Rule.** No pass/fail on the before cell: it is the record. Reported verbatim per rung: exit code, the
`EXACT`/`DIVERGED` line, the `[dspark-q38-gate]` receipt line, the acceptance line, the largest
`dflash.rs`-traced allocation, the refusal line where one occurs, and the collector's device peak and regime.
N=1 per rung (a correctness and memory cell; `prime_s` is diagnostic timing, not a performance receipt).

## Task 3: cell `tapladder20b` (local RTX 5090, after; pre-registered here before the change was built)

**Question.** With the standalone qwen entry point primed through `prime_dflash_taps`, are the drafter's
proposals and accepted tokens bitwise identical to the before cell at every rung the before cell admitted,
with a smaller tap peak, and does the before cell's refused rung now admit or refuse on a different, named
resource?

**Shape.** The same script, `tapladder20b`, the same five rungs, `ngen 32`, the same environment, the gate
binary built from the change commit (its SHA-256 in `ev/binary.sha256`).

**Rule** (PASS = `identity; bounded`, every clause required, `day20-replay.py`):
- identity: for every rung whose before run printed `EXACT`, the after run prints `EXACT`, the same
  `spec_sha256`, the same `spec_len`, the same `[dspark-q38] acceptance a/b`;
- bounded: for every completed rung the largest `dflash.rs`-traced allocation is at most `420,966,400`
  bytes, and the `[dflash-taps]` line's `former_full_tap_bytes` equals `prompt x 102,400`;
- moved: for the rung(s) the before cell refused, the after run either completes `EXACT` or its last
  `[alloc-trace]` line before the `out of memory` names a file and line other than the tap sink (quoted).
Reported beside the verdict, not a clause: device peak per rung (before against after), `prime_s`. N=1 per
rung; no timing claim. If the after cell is red on identity, the change does not land; nothing is relaxed
after a result.

## Cell `tapladder20` result (local RTX 5090 Laptop GPU, before; `rtx5090-day20/tapladder20/`)

One collector hold (first attempt, lock free) 21:58:25Z to 22:01:48Z, 928 telemetry samples at 250 ms,
`dspark_q38_gate` `f2c797caf8…` from tree `0e346fe7d` (unchanged engine plus the receipt line), the three
artifacts above, `ngen 32`. No other compute process on the card before or after any rung
(`w*.smi-before`, `w*.smi-after`: header only). Device peaks are the collector's `memory.used` maximum inside
each rung's marks (`ev/peaks.txt`); the card reports 24,463 MiB total. Regime 54..88 C, 178 W peak.

| Rung (words) | Prompt tokens (gate) | Result | Tap bytes at `dflash.rs:3977` | Acceptance | `spec_sha256` (= `plain_sha256`) | `prime_s` | Device peak MiB |
|---|---|---|---|---|---|---|---|
| 4,763 | 8,194 | `EXACT`, rc 0 | `839065600` (= 8,194 x 102,400) | `28/28 = 1.000 rounds=4` | `72cf198537ff594e…3162ef22` | 5.780 | 21,607 |
| 9,527 | 16,382 | `EXACT`, rc 0 | `1677516800` (= 16,382 x 102,400) | `28/28 = 1.000 rounds=4` | `0d6449b292e6f417…f047492e` | 11.984 | 22,759 |
| 19,055 | 32,759 | OOM, rc 1 | `3354521600` allocated (= 32,759 x 102,400) | none | none | none | 23,961 |
| 38,109 | 65,507 | OOM, rc 1 | `6707916800` REFUSED (= 65,507 x 102,400) | none | none | none | 23,958 |
| 76,217 | 131,004 | OOM, rc 1 | never reached | none | none | none | 23,961 |

Verbatim receipt lines. 8,194: `prompt.txt: prompt 8194 -> gen 32 | plain 4.9 tok/s | spec 145.6 tok/s |
EXACT`, `[dspark-q38-gate] prompt.txt: prompt=8194 spec_sha256=72cf198537ff594ea929ae040c7d6a3d875b864a2d90c9cd82b809943162ef22
plain_sha256=72cf198537ff594ea929ae040c7d6a3d875b864a2d90c9cd82b809943162ef22 spec_len=32 plain_len=32
prime_s=5.780`, `[dspark-q38] acceptance 28/28 = 1.000 rounds=4 draft=33.9ms snap=1.8ms verify=179.1ms
rollback+replay=0.0ms ingest=1.6ms`. 16,382: `prompt.txt: prompt 16382 -> gen 32 | plain 2.5 tok/s | spec
128.4 tok/s | EXACT`, `[dspark-q38-gate] prompt.txt: prompt=16382 spec_sha256=0d6449b292e6f417e94a0640733ce13efba0458828bf1f168eb4b3a2f047492e
plain_sha256=0d6449b292e6f417e94a0640733ce13efba0458828bf1f168eb4b3a2f047492e spec_len=32 plain_len=32
prime_s=11.984`, `[dspark-q38] acceptance 28/28 = 1.000 rounds=4 draft=33.1ms snap=1.6ms verify=211.2ms
rollback+replay=0.0ms ingest=1.6ms`. (`prime_s` is the standalone TTFT analogue: trunk prefill plus tap
ingestion to the boundary token; diagnostic, N=1, not a performance receipt. The plain `tok/s` column
includes the plain oracle's own prefill and is not a decode rate.)

**The three refusals, each from its log, the trace printing BEFORE the attempt so the last `[alloc-trace]`
line names the allocation that failed.**

- 32,759 tokens (`w19055.log.gz`): the plain oracle completed (its whole-call `hiddens`, `670904320 bytes
  from crates/memra-engine/src/hybrid_forward.rs:5926`, line 3129). The DFlash path then allocated the tap,
  `[alloc-trace] 3354521600 bytes from crates/memra-engine/src/dflash.rs:3977` (line 77986), the monolithic
  prime's `hiddens` again (`670904320 bytes from ... hybrid_forward.rs:5926`, line 77987), and died inside the
  first trunk chunk on a GDN chunked-scan transient: the last trace line `[alloc-trace] 100663296 bytes from
  crates/memra-engine/src/lib.rs:33856` (`gdn_chunk_k123`'s `w` buffer, `nc x h x c x 128` f32) followed by
  `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`. The tap did not refuse; the prefill it
  starved did: the issue's shape ("before trunk prefill and draft ingestion"), a resident whole-prompt tap
  plus the prefill's own transients.
- 65,507 tokens (`w38109.log.gz`): the plain oracle completed; the tap allocation itself refused. Last two
  lines: `[alloc-trace] 6707916800 bytes from crates/memra-engine/src/dflash.rs:3977` then `Error:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`. This is the tap refusal the brief asked for, on
  this card at this length: 6,707,916,800 bytes on a card whose free bytes after the target and drafter
  loaded read `driver_free_bytes=8983478272` (`[dflash-alloc] phase=draft-weights-after`, the same line in
  every rung) with the 65k plain cache still to fit.
- 131,004 tokens (`w76217.log.gz`): the gate's PLAIN oracle refused before the DFlash path ran: `[alloc-trace]
  2682961920 bytes from crates/memra-engine/src/hybrid_forward.rs:5926` (the monolithic prime's `hiddens`,
  131,004 x 20,480) succeeded, then the prime workspace's `[alloc-trace] 285212672 bytes from
  crates/memra-engine/src/hybrid_forward.rs:6578` (`act: t x n_ff_max` f32) and `Error:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`. The issue's 13,421,568,000-byte tap is
  reproduced by arithmetic above and cannot be reached by this gate on a 24 GB card: the control itself does
  not fit. The gate is the instrument; the after cell keeps the rung so the record shows where the refusal
  sits after the change (expected: unchanged, in the plain oracle, a resource the change does not touch).

So on this card today the standalone DFlash2 path admits 16,382 tokens and refuses from 32,759 tokens up; the
tap is the refused allocation at 65,507 and the starving resident at 32,759. `executed-not-qualified`.

## Task 3: the bounded storage landed on the standalone qwen entry point (`4fd4f3b41`)

**One seam, in `crates/memra-engine/src/dflash.rs` only.** `generate_spec_dspark`'s prime block (the
whole-prompt `DflashTapSink` at the former `dflash.rs:3975`, the single `prime_cache(prompt)` call, the
256-row ingest loop after it) is replaced by a call to the serving walker `prime_dflash_taps(e, draft,
&mut cache, &mut dkv, prompt, None, &mut boundary, oracle)`: `DflashKv::new` moves before the prime, the
`DflashPrimeOracle` is armed under `MEMRA_ALLOC_TRACE=1` exactly as the serving path arms it (so the
standalone path now prints `[dflash-oracle]` and `[dflash-taps]`), the boundary token is drawn from the
returned last-chunk logits by the unchanged `sample_boundary_token` / `argmax` composition, `PRIME_NANOS`
is stored around the same span (prefill plus ingestion). No kernel changed, no value written changed, no
flag added, no `unsafe`, no dependency. The gemma arm (`generate_spec_dflash`), the serving path, the
per-round keep-ingest sinks and `hybrid_forward.rs` are untouched.

**The typed refusal.** `apply_tap_batch_op`'s `Copy` arm used to `expect("copy requires a live chunk sink")`.
It now returns a typed error when there is no live chunk sink (`DFlash tap carry: copy requested with no
live chunk sink (a tap row would be read before the trunk wrote it)`) or when the copy's source rows lie
past the rows the chunk sink holds (`DFlash tap carry: copy of rows a..b exceeds the N-row chunk sink the
trunk wrote`). Neither fires on the production schedule (the carry emits Copy sources inside `0..rows` of
the chunk it was given); they turn a bookkeeping fault into a request error instead of a panic or a stale
row.

**CPU tests (the module's plus one on the chunk bookkeeping).**
`standalone_prime_consumes_each_chunk_before_the_next_and_keeps_the_256_row_partition`: for prompt lengths
16, 255, 256, 257, 4096, 4097, 4111, 4112, 8192, 8194, 16382, 32759 and 65507 on the single-device 4096-row
grid with the sub-16 tail folded, every `Copy` source lies inside the chunk the trunk just wrote, every
`Ingest` covers only rows already copied, after each chunk the whole chunk has been consumed and at most 255
rows survive the edge (in the carry, never in a chunk sink), and the ingested batches are exactly
`(0..tp).step_by(256)` in order, the whole-buffer path's partition; `next_position == tp`, `pending == 0`
after the flush. The existing `production_carry_preserves_whole_buffer_batches_across_chunks_and_boundary`
and `tap_hash_preserves_bits_across_arbitrary_capture_partitions` are unchanged. Results of the run are in
the Gates section below.

## Cell `tapladder20b` result (local RTX 5090 Laptop GPU, after; `rtx5090-day20/tapladder20b-retry2/`)

Replay: `DAY20 REPLAY tapladder20b: PASS (9 checks) -> identity; bounded` (`ev/replay.txt`). Attempts 0 and
1 found `/tmp/memra-5090.lock` held by another lane and slept 120 s each (`lock-retries.log`, the two
`tapladder20b*-driver.log` files read `REFUSED: [Errno 11] Resource temporarily unavailable`; no holder was
inspected or signalled); attempt 2 took the lock: one hold 22:13:49Z to 22:18:56Z, 1,349 telemetry samples,
`dspark_q38_gate` `514109da2f93…` from tree `4fd4f3b41` (the change commit), the same three artifacts, the same
five prompts (`prompts.txt` hashes equal), `ngen 32`. No other compute process before or after any rung. The
CPU test compile of this lane ran on the host from 22:12:51Z to 22:14:30Z, overlapping the first rung's
window (host load only; the cell reads no timing as a claim). Regime 66..90 C, 178 W peak.

| Rung | Prompt tokens | Result | Largest `dflash.rs` allocation | `[dflash-taps]` chunk / carry / former | Acceptance | `spec_sha256` | `prime_s` | Device peak MiB (before) |
|---|---|---|---|---|---|---|---|---|
| 4,763 | 8,194 | `EXACT`, rc 0 | `419635200` at `dflash.rs:5266` (chunk sink, 4,098 rows) | `419635200` / `26214400` / `839065600` | `28/28 = 1.000 rounds=4` | `72cf198537ff594e…3162ef22` (= before) | 6.210 | 21,415 (21,607) |
| 9,527 | 16,382 | `EXACT`, rc 0 | `419430400` (4,096 rows) | `419430400` / `26214400` / `1677516800` | `28/28 = 1.000 rounds=4` | `0d6449b292e6f417…f047492e` (= before) | 12.749 | 21,415 (22,759) |
| 19,055 | 32,759 | `EXACT`, rc 0 (before: OOM) | `419430400` | `419430400` / `26214400` / `3354521600` | `28/28 = 1.000 rounds=4` | `c009f8ea51f0168b…7ebc5a6e` | 28.472 | 22,215 (23,961, OOM) |
| 38,109 | 65,507 | `EXACT`, rc 0 (before: tap refused) | `419430400` | `419430400` / `26214400` / `6707916800` | `28/28 = 1.000 rounds=4` | `8b90adb8213eed06…ae8f6db5` | 66.587 | 23,609 (23,958, OOM) |
| 76,217 | 131,004 | OOM, rc 1, in the plain control (as before) | `81920` (drafter load) | none | none | none | none | 23,961 (23,961) |

Verbatim, the two rungs the before cell admitted: `[dspark-q38-gate] prompt.txt: prompt=8194
spec_sha256=72cf198537ff594ea929ae040c7d6a3d875b864a2d90c9cd82b809943162ef22 plain_sha256=72cf198537ff594ea929ae040c7d6a3d875b864a2d90c9cd82b809943162ef22
spec_len=32 plain_len=32 prime_s=6.210`, `[dspark-q38] acceptance 28/28 = 1.000 rounds=4 draft=33.4ms snap=2.0ms
verify=179.2ms rollback+replay=0.0ms ingest=1.5ms`, `[dflash-taps] base=0 rows=8194 max_chunk_rows=4098
chunk_tap_bytes=419635200 carry_bytes=26214400 former_full_tap_bytes=839065600`; `[dspark-q38-gate] prompt.txt:
prompt=16382 spec_sha256=0d6449b292e6f417e94a0640733ce13efba0458828bf1f168eb4b3a2f047492e
plain_sha256=0d6449b292e6f417e94a0640733ce13efba0458828bf1f168eb4b3a2f047492e spec_len=32 plain_len=32
prime_s=12.749`, `[dspark-q38] acceptance 28/28 = 1.000 rounds=4 ...`, `[dflash-taps] base=0 rows=16382
max_chunk_rows=4096 chunk_tap_bytes=419430400 carry_bytes=26214400 former_full_tap_bytes=1677516800`. The
identity clause holds at both: the same 32 ids (digest equal to the before cell's), the same length, the same
acceptance; both `EXACT` against the plain oracle. The 4,098-row chunk at 8,194 tokens is the fold law
(8,194 = 2 x 4,096 + 2, the two-row tail folded into the last chunk), so its sink is 419,635,200 bytes, inside
the 4,111-row cap the rule states.

The two rungs the before cell refused now complete: 32,759 tokens `prompt.txt: prompt 32759 -> gen 32 | plain
1.2 tok/s | spec 102.1 tok/s | EXACT`, `[dflash-taps] base=0 rows=32759 max_chunk_rows=4096
chunk_tap_bytes=419430400 carry_bytes=26214400 former_full_tap_bytes=3354521600`, `acceptance 28/28`; 65,507
tokens `prompt.txt: prompt 65507 -> gen 32 | plain 0.5 tok/s | spec 68.5 tok/s | EXACT`, `[dflash-taps] base=0
rows=65507 max_chunk_rows=4096 chunk_tap_bytes=419430400 carry_bytes=26214400 former_full_tap_bytes=6707916800`,
`acceptance 28/28`. The tap peak at 65,507 tokens fell from 6,707,916,800 bytes (refused) to 419,430,400 plus the
26,214,400-byte carry, a factor of 15.05 at that length; the device peak at that rung reads 23,609 MiB where the
before cell died at 23,958. The 131,004-token rung refuses exactly where it refused before, in the gate's PLAIN
control (`[alloc-trace] 285212672 bytes from crates/memra-engine/src/hybrid_forward.rs:6578` then `Error:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, before any DFlash allocation): a resource the change
does not touch, named, the `moved` clause as pre-registered. The `[dflash-oracle]` digests (taps, features,
positions, target logits) are new on the standalone path and are recorded per rung in `w*.receipt` and the logs
for a later comparison; no before-side oracle exists to compare them against today.

`prime_s` (diagnostic, N=1) reads 6.210 against 5.780 and 12.749 against 11.984 at the two shared rungs: the
after path interleaves ingestion with the trunk chunks and synchronizes per chunk, and the test compile
overlapped the first rung; this is not a timing claim in either direction and the cell was not designed to
make one. `executed-not-qualified`.
