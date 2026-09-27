# WP-B day 31: `MEMRA_ADMIT_BY_MEMORY` OFF against ON, and the open-output value swept

Owner, 2026-09-23: "i pick to messure now and no, 8k is not the default it should be messured, and checked over
other engines." The door reached its decide-by today with the ON arm never booted on either card. This file is the
cell. The survey of other engines is `OPEN-OUTPUT-SURVEY.md`. Every cell here is `executed-not-qualified`; no
qualification is claimed and no default moves in this lane.

## 1. Pre-registration

Committed and pushed before the first boot of this cell. Nothing in section 1 changes after a number is seen; a
correction to a script that does not change a rule is recorded in section 2 with its commit.

### 1.1 What the door does on this cell

One boot reads the door once (`MemoryAdmitConfig::from_env`, strict `1`) and prints
`[admit-mem] door=ON|OFF open_output_tokens=N defer_budget_ms=N (...)`.

- (a) Open-output charge. A request with neither `max_tokens` nor `max_ctx` gets
  `ctx_cap = charged_ctx_tokens(P, None, v, model_ctx) = max(min(P + v + 8, model_ctx), P + 8)` from
  `request_ctx_cap` (worker.rs, the door-ON branch). The same `ctx_cap` is the admission charge AND the
  `Cache::new_inner` allocation, and `prepare_request` sets `budget = max_new.min(ctx_cap - P) = v + 8`. So under ON
  an open request stops at about `v + 8` generated tokens with `finish_reason = "length"`. The door is a hard output
  bound on the naked path, not only a charge. Under OFF the same request gets the `MEMRA_CTX` envelope
  (65,536 locally, the checkpoint's declared 262,144 on the target card) and runs until it stops.
- (b) Host-tier demotion. No arm of this cell sets `MEMRA_KV_HOST_MB`, so the host tier is unarmed,
  `host_free = 0`, `demotable = min(evictable, 0) = 0`, and `evict_all_demoting` is byte-identical to OFF. (b) is not
  exercised here; it is owed as D4 (b).
- (c) Bounded defer. A memory-deferred arrival that the device cannot fit and the (unarmed) host tier cannot help
  waits `MEMRA_ADMIT_DEFER_BUDGET_MS` (8000 ms default) and is then refused with a 429 whose `Retry-After` is the
  earliest predicted completion clamped to 1..=60 s (5 s when unknown). Under OFF the same arrival waits FIFO with no
  bound. The burst in 1.4 is where (c) can show; an `[admit-mem] id=` line prints only on a first defer or a refusal.

Bounded requests (`max_tokens` given) take the same arm under OFF and ON: booked `P + max_tokens + 64`, allocated
`P + max_tokens + 8`.

### 1.2 Rigs, artifacts, binaries

| card | model | served context | burst B | lock |
|---|---|---|---|---|
| local RTX 5090 | Qwen3.5-9B NVFP4 MTP GGUF, sha256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` | `MEMRA_CTX=65536` | 32 | `/tmp/memra-5090.lock` |
| BOX3, one RTX PRO 6000 Blackwell | Qwen3.8-27B NVFP4-Q5K MTP GGUF (sha256 recorded per boot in `model.sha256`) | unset (declared 262,144) | 64 | `/tmp/memra-gpu.lock` |

One binary per card for every boot of both orders: built once, copied once (locally to `target/day31/memra-server`, on BOX3 to the receipt root's `bins/`), its sha256 in
`binary.sha256`. The local binary is `daccda3bcd36e4a9eebfd2bb72ec6dae20b05c595ef725da26b9491f49c4efef`, built from
engine source identical to this commit (the lane tip after the origin/main `5f1b0eda4` merge; this commit adds
research files only). Every boot: `MEMRA_COMPAT=openai`, `MEMRA_ADMIT_PREDICT_SHADOW=1` (receipts only, refuses
nothing), `MEMRA_TIMEOUT_MS_MAX=3600000` on every arm so the 90 s non-streaming ceiling does not cut a long
generation (day 26's `iii-L2-r3` was cut at 90 s with status 200 and no finish_reason), `MEMRA_MAX_SESSIONS` at its
default 64, no host tier, speculative decoding at its default.

Timing is never compared across cards or boxes. Throughput is not a reading of this cell.

### 1.3 Arms and orders

| arm | env | open request charged and allocated |
|---|---|---|
| `off` | `MEMRA_ADMIT_BY_MEMORY` and `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` removed from the environment | served context + 64 booked |
| `on2048` | `MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=2048` | `P + 2056 + 64` booked |
| `on8192` | `... =8192` (today's compiled default) | `P + 8200 + 64` booked |
| `on32768` | `... =32768` | `P + 32776 + 64` booked |

- Order O1: `off`, `on2048`, `on8192`, `on32768`; every boot runs the workload with inner order AB.
- Order O2: `on32768`, `on8192`, `on2048`, `off`; every boot runs the workload with inner order BA.
- One boot per arm per order: 8 boots per card. One collector hold per order
  (`tools/tier-battery.py --rig rtx5090|pro-single --external-lock`), so the four arms of an order share one lock
  window. Driver: `day31-order.sh` calling `run-day26-cell.sh` with `CLIENT=day31-client.py PARSER=day31-parse.py`.
- N = 5 per class per boot for the day-26 mix, so N >= 5 per arm per order.

### 1.4 Workload (fixed in `day31-client.py` at this commit)

Greedy (`temperature: 0.0`), non-streaming, one request at a time except the burst. Prompts come from
`docs/SERVING.md` at this commit, sha256 `022b1d50620f5b34bb6e6127be05ff79184ce3aa294aca618a31a0dd73887cff`
(recorded per boot in `inputs.sha256`); `day31-client.py` sha256 at this commit
`53aa251263cb559fab28cc56b71ee9b8077be70ef8b9e6cabc9a8aede551fda2`.

1. `warmup`: one 300-char summary, `max_tokens=32`.
2. The day-26 mix, unchanged: per length L0/L1/L2 = 5000/10000/20000 chars, 5 reps each, (i) open, (ii)
   `max_tokens=96`, then (iii) every (i) conversation continued with "Add one more sentence to that summary."
   (open), deferred after all first turns. AB runs (i) before (ii) inside each length; BA the reverse.
3. The long-generation open class (iv), `max_tokens` omitted, three reps per type:
   - `iv-a`: "Write a complete operator manual for the server described in the notes below ..." over a 6000-char
     SERVING.md slice, salt 700+r.
   - `iv-b`: "Write out the full multiplication table from 1 x 1 up to k x k: every product on its own line in the form
     'a x b = c' ...", k = 24, 40, 72 for r = 0, 1, 2, salt 800+r. At about 8 tokens a line these span roughly
     4.6k, 12.8k and 41k output tokens, one under each swept value's neighbourhood; the model's own stop is what is
     read.
   - AB runs iv-a then iv-b; BA the reverse.
4. The burst: B concurrent open (i)-shape requests at 5000 chars, salts 900..900+B-1, released on one barrier after
   every sequential request has returned. Excluded from the identity gate.

Per request the client records status, `x-request-id`, `Retry-After` on a non-200, usage (P, G, cached),
`finish_reason`, `prompt_sha256` (the messages array), `message_sha256` (the whole response message object with
sorted keys: reasoning plus content plus any tool calls; day 26's `content_sha256` alone misses the reasoning, and a
bounded reply's content is often empty) and `content_sha256`. A 250 ms sampler records `nvidia-smi memory.used` and
the `/metrics` keys `cuda_driver_free_bytes`, `cuda_pool_used_bytes`, `cuda_pool_reserved_bytes`,
`admission_booked_bytes`, `active_sessions`, `prefix_cache_bytes`. The collector adds its own telemetry,
`command.capture.json`, the lock proof and the compute-apps snapshots; the driver adds `binary.sha256`,
`source.txt`, `model.sha256`, `compute-apps-{before,after}.csv` and the stamped `server.log` per boot.

### 1.5 Readings

Per boot (`day31-parse.py`, `REPORT.txt` and `rows.jsonl`): the door line verbatim; per sequential request P, G,
finish, the own `[admission] request cost` line's booked ctx, path and B/token, allocated bytes
`B/token x (booked - 64)` and used bytes `B/token x (P + G)`; for the burst, statuses, `Retry-After` values, the
maximum `active_sessions` and `admission_booked_bytes` of the samples inside the burst window, the `[admit-mem] id=`
lines by verdict, and `arith_sessions = shadow budget_bytes / median burst request cost`.

Across boots (`day31-compare.py`): the verdict lines below, per card.

### 1.6 Verdict rules (fixed now, never relaxed)

- **V-BOOT.** PASS iff the boot's `[admit-mem] door=` line reads `door=ON open_output_tokens=v` for arm `onv` and
  `door=OFF` for `off`. A failing boot is excluded from every other reading and named.
- **V-ALLOC.** Every request with its own cost line: booked ctx must equal `P + max_tokens + 64` for warmup and
  (ii); `max(min(P + v + 8, model_ctx), P + 8) + 64` for an open request under `onv`; `served + 64` (or `P + served + 64`
  when `P + 16 > served`) for an open request under `off`. PASS iff every judged row matches. Inherited-line rows are
  reported, not judged.
- **Twin.** For an ON boot and the OFF boot of the same order, the rows with the same tag. A twin is comparable iff
  both returned 200 and `prompt_sha256` is equal. A (iii) whose (i) parent was cut by the bound has a different
  prompt and is not comparable.
- **V-ID** (the identity gate). Eligible twins: warmup and (ii) always; an open-class twin whose OFF side finished
  `stop` with `G_off <= v`. PASS for an ON arm in an order iff eligible > 0 and every eligible twin has equal
  `message_sha256`, equal G and equal `finish_reason`; otherwise FAIL naming the differing tags. Twins with
  `v < G_off <= v + 8` and OFF `stop` form the slack band, reported as `slack_equal`/`slack_differ` and not judged.
- **V-TRUNC** (counts, per ON arm and order). `truncated_with_twin`: comparable open twin, OFF `stop`, ON `length`.
  `length_without_twin`: an ON open row ending `length` with no comparable twin. `length_both`, `deadline_cut`
  (status 200 with no finish_reason). `truncated_G_outside_[v,v+8]`: an ON open `length` row whose G is outside
  `[v, v + 8]`; this sub-reading is judged, and any such row fails V-DOOR.
- **G-NATURAL.** Every open-class row of both OFF boots on the card with status 200: n, min, p50, p90, p99, max, and
  the counts above 2048, 8192 and 32768, per class and together, with the finish counts.
- **V-CONC** (reading, no pass mark). Per boot the burst's statuses, maximum `active_sessions`, maximum booked bytes,
  `arith_sessions`, and the `[admit-mem]` verdict counts.
- **V-RETRY.** Under `off`, zero 429s in the burst. Under `onv`, every 429 carries `Retry-After` in 1..=60 and the
  number of 429s equals the number of `verdict=refuse` lines in that burst window. PASS or FAIL per boot.
- **V-DOOR** (per card). PASS iff no boot is excluded, every V-ID passes in both orders, every V-ALLOC passes, no
  truncated row falls outside `[v, v + 8]`, and every V-RETRY passes.

### 1.7 Selection rules (stated per card, never applied to a default)

This lane does not pick the value. It prints which value each rule selects:

- **R1, zero truncation.** The smallest swept v with `truncated_with_twin + length_without_twin = 0` summed over both
  orders on the card; "none" when every swept value truncates something.
- **R2, natural cover.** The smallest swept v at or above the largest G of any OFF open row that finished `stop` on
  the card; "none" when the largest natural G is above 32768.
- **R3, registry agreement.** The value the registry path applies to an omitted output. Read before this cell: every
  chat row in the private deployment registries pins `default_output_length = 32768` (seven chat models; the
  embed/rerank/classify/stt rows pin 1 to 224 and are not chat). `admit_memory.rs` documents 8192 as "the
  `default_output_length` the fleet's registries already pin"; that sentence does not match the registries today,
  while `docs/FLAGS.md` row `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` already calls 8192 "a floor, not a claim about the
  fleet". R3 = 32768 on both cards.
- **R4, surveyed convention.** From `OPEN-OUTPUT-SURVEY.md`: the output bound most of the five surveyed surfaces apply
  when the caller omits it. If that bound is "the remaining context", R4 selects `off` for the output bound (the
  charge question is then separate, see the survey's admission column); if it is a fixed number, R4 selects the
  smallest swept v at or above it.

### 1.8 Exclusions and failures

- A non-200 outside the burst, a crash, or a boot that never becomes ready is quoted from its captured stderr,
  never inferred. A boot with no captured cause is "died, cause unknown, repro needed", and nothing is built on it.
- A rerun happens only as a whole order, under a new cell name, with the reason recorded in section 2. No boot is
  dropped silently.
- A deadline cut is its own category and is excluded from V-ID and from the truncation counts.

### 1.9 Not measured here

(b) with the host tier armed (D4 (b)); a 429 under a defer that exhausts an ARMED host tier (D4 (c); the burst can only
show the unarmed case); any latency or throughput claim; any qualification. The 5090 is not a release gate for this
door and this cell is not a battery.

## 2. Results

Written after the runs. Section 1 is unchanged. Every number below is read from the committed receipts named next
to it; every quote is the captured server log.

### 2.1 Commits and timeline

| commit | UTC | what |
|---|---|---|
| `ee53f7117` | 2026-09-22 23:09:14 (pushed about 23:09:23) | section 1, the client, the parser and the reader, before any boot |
| (first boot) | 23:09:33 | local `O1-off` start (`rtx5090-day31/order.log` line 2) |
| `3d160ffc4` | 23:29:42 | `OPEN-OUTPUT-SURVEY.md`. It landed 20 minutes after the first boot started. R4's rule was fixed in 1.7 before any boot, and the survey reads only the pinned external sources and memra's code, never a boot output |
| `1a40a3c98` | 23:30:56 | target chain fetches from `SRC` (the box had no route to the git host that night); driver only |
| `5c7cb8aaa` | 2026-09-23 00:04:48 | target chain takes `RUN` and `ORDER_TIMEOUT`; driver only (2.3) |
| `b505c3253` | 00:21:42 | `DAY31-D4.md` pre-registration, before the D4 boot |
| `c426a8be2` | 01:25:35 | local O1 receipts, the stopped BOX3 run 1 mirror, `day31-faults.py` |
| `e13daa140` | 03:16:37 | local O2 receipts, local `SUMMARY.txt` and `FAULTS.txt` |
| `3f19e40a2` | 03:46:07 | D4 receipts and `DAY31-D4.md` section 2 (the D4 boot ran 03:08:58 to 03:42:51) |
| this commit | after the last boot | BOX3 run 2 receipts, box `SUMMARY.txt` and `FAULTS.txt`, this section, `STATE.md`, the INDEX row |

`day31-faults.py` was written after the first crash was seen. It is a lister (every panic, `[worker]` PANIC,
respawn and FATAL line, and `[engine-error]` line with its `server.log` line number and the request around it), not a
verdict, and nothing in 1.6 reads it.

### 2.2 Notes on section 1 (no rule, arm, value or reader changed)

- Transcription. 1.1 and the reader write the ON charge as `max(min(P + v + 8, model_ctx), P + 8)`. The code
  (`admit_memory.rs` `charged_ctx_tokens`) computes `min(max(P + v + 8, P + 8), model_ctx) = min(P + v + 8, model_ctx)`.
  The two differ only when `model_ctx < P + 8`, which no request here reaches; V-ALLOC is unaffected.
- `admit_memory.rs:48-51` still says 8192 is "the `default_output_length` the fleet's registries already pin". 1.7 R3
  already records that this does not match the registries (32768 on every chat row). The comment is left as is; this
  lane does not touch the engine.
- `lib.rs:3428-3429` calls the omitted-cap behaviour "the OpenAI default-when-omitted semantics". The survey found the
  OpenAI contract unpinned for the omitted case (`OPEN-OUTPUT-SURVEY.md`, OpenAI row). Recorded, not changed.
- Reader, as run. V-TRUNC counts only twins where both sides returned 200, so an ON `length` row whose OFF twin
  returned 500 is in no count and is not checked against `[v, v + 8]`. 1.6 names such a row `length_without_twin`.
  Four rows are in that gap on the local card (2.6). V-RETRY reads PASS on a boot whose burst got no response at
  all (zero 429s, zero refuse lines). V-DOOR has no crash term. The reader was not changed after the result
  (lead ruling); the gap rows are listed by hand below and are not a verdict.
- Each local boot's `source.txt` records the worktree HEAD at that boot's start (`ee53f7117`, `5c7cb8aaa`,
  `b505c3253`, `c426a8be2`), because research-only commits landed while the chain ran. The binary did not move:
  every local boot's `binary.sha256` is `daccda3b...c4efef`, built from `9562192b5`, whose engine source equals
  every later lane commit's (`git diff 9562192b5 5c7cb8aaa` touches `research/` only).

### 2.3 BOX3 run 1 stopped, run 2 is the whole-order rerun

Run 1 (`pro-single-day31/box-run1-stopped/`, binary `4d2f1e3b...4a4ec2`, built from `1a40a3c98`) stopped at
2026-09-23T00:04:20Z inside `O1-off`, before any ON boot. Its `chain.log` last line:
`STOPPED by lane B: collector --timeout 16200 cannot hold the OFF boot on this card (iv-a-r0 G=102441 in 833 s,
iv-a-r1 past ctx 95k); whole-order rerun as b-day31-run2 with a longer collector timeout`. Only processes this lane
started were stopped. Per 1.8 the rerun is the whole order under a new name: run 2 (`RUN=run2`,
`ORDER_TIMEOUT=43200`), receipts in `pro-single-day31/box/`. Run 2's binary is `ef3847d0...212fbd`, built from
`5c7cb8aaa`. It differs from run 1's because `crates/memra-server/build.rs` embeds `MEMRA_BUILD_SHA` (the commit); the
engine source is the same. Run 1's partial `O1-off` is kept and read by nothing.

### 2.4 The crash record, per boot

Source: `rtx5090-day31/FAULTS.txt` and `pro-single-day31/box/FAULTS.txt` (`day31-faults.py` over every boot). Line
numbers are `server.log` lines of that boot (`server.log.gz` where the log was committed gzipped; line numbers are the
uncompressed file's, sha256 in `server.log.sha256`). Two defects show, and every one of them sits at the end of a
session's allocated cache:

- the panic `thread 'memra-gpu-worker' (...) panicked at crates/memra-engine/src/spec.rs:4590:9:` followed by
  `[worker] PANIC in the GPU worker thread: mtp_kv_fill: scratch overflow` and `[worker] respawn attempt 1/1 in 2s
  (reloading weights)`. It fires on the tick after a request stopped with `P + G` equal to its allocation
  (`booked - 64`). A second panic in the same process prints `[worker] FATAL: worker unrecoverable after 1 respawn
  attempt(s)` and the process exits; every later request of that boot has no status.
- the #87 trap: `[engine-error] class=Engine step error: verify argmax sentinel 0x7fffffff >= n_vocab 248320 at round
  R col 0/4 pos=N` or `draft(graph) argmax sentinel 0x7fffffff >= d_vocab 248320 at round R j=2 pos=N`, with `N`
  within 5 of the request's allocation; the request returns 500.

Local RTX 5090, Qwen3.5-9B, `MEMRA_CTX=65536`:

| boot | panics | respawns | FATAL | first panic (line; request before it, `P + G` = allocation) | #87 traps, lines (pos) | open rows with no status | burst |
|---|---|---|---|---|---|---|---|
| `O1-off` | 1 | 1 | no | 10286; `iv-b-r1` P=78 G=65458 `length`, 65536 | 12369 (65533), `iv-b-r2` 500 | 0 | 32 x 200 |
| `O1-on2048` | 2 | 1 | 431 | 116; `i-L0-r0` P=1439 G=2056, 3495 | 218 (3492), 287 (3496), 356 (3496): `i-L0-r1..r3` 500 | 31 (second panic at 428; the first row with no status is `ii-L0-r0`) | 32 no status |
| `O1-on8192` | 2 | 1 | 5419 | 3928; `iii-L2-r3` P=5821 G=8200, 14021 | 5156 (8276), `iv-b-r1` 500 | 0 (second panic at 5416, nothing in flight, after `iv-b-r2` stopped at 8278) | 32 no status |
| `O1-on32768` | 0 | 0 | no | none | 4688 (38594) `iii-L2-r3`; 6392 (32851) `iv-b-r0`; 7417 (32852) `iv-b-r1`; all 500 | 0 | 19 x 200, 11 x 429, 2 x 503 |
| `O2-on32768` | 0 | 0 | no | none | 4689 (38594) `iii-L2-r3`; 5839 (32851) `iv-b-r0`; 6864 (32852) `iv-b-r1`; all 500 | 0 | 19 x 200, 11 x 429, 2 x 503 |
| `O2-on8192` | 2 | 1 | 4869 | 3929; `iii-L2-r3` P=5821 G=8200, 14021 | 4605 (8276), `iv-b-r1` 500 | 3 (`iv-a-r0..r2`; second panic at 4866 after `iv-b-r2` stopped at 8278) | 32 no status |
| `O2-on2048` | 2 | 1 | 479 | 165; `i-L0-r0` P=1439 G=2056, 3495 | 267 (3492), 336 (3496), 405 (3496): `i-L0-r1..r3` 500 | 31 (second panic at 476, nothing in flight, after `i-L0-r4` stopped at 3497; the first row with no status is `ii-L1-r0`) | 32 no status |
| `O2-off` | 1 | 1 | no | 9733; `iv-b-r1` P=78 G=65458 `length`, 65536 | 11816 (65533), `iv-b-r2` 500 | 0 | 32 x 200 |

The pattern repeats to the request across the two orders. The two 503s in each `on32768` burst are not the cap
defect: `[engine-error] class=Overloaded prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`
(`O1-on32768` lines 8604 and 8605, `O2-on32768` lines 8612 and 8613), returned as 503 `overloaded` with
`Retry-After: 5`. Both requests had passed the door's admission; the failure is in their prefill. `compute-apps-before.csv`
and `compute-apps-after.csv` of both boots list no process, and the collector held `/tmp/memra-5090.lock`.

BOX3, one RTX PRO 6000 Blackwell, Qwen3.8-27B (model sha256 `1facf36c...1e024a` on all 8 boots), served context
262,144, run 2 (binary `ef3847d0...212fbd` and `source.txt` `5c7cb8aaa` on all 8 boots; both collector holds on
`/tmp/memra-gpu.lock`, `LOCK-O1.json` and `LOCK-O2.json` `inherited-flock-same-open-description`; chain
2026-09-23T00:06:28Z to 05:57:49Z):

| boot | panics | respawns | FATAL | first panic (line; request before it, `P + G` = allocation) | #87 traps, lines (pos) | open rows with no status | burst |
|---|---|---|---|---|---|---|---|
| `O1-off` | 0 | 0 | no | none | 12173 (262142) `iv-a-r1` 500 | 0 | 64 x 200 |
| `O1-on2048` | 0 | 0 | no | none | 981 (3714) `iv-a-r0`; 1048 (3714) `iv-a-r1`; 1117 (3716) `iv-a-r2`; 1187 (2173) `iv-b-r0`; 1255 (2174) `iv-b-r1`; 1323 (2173) `iv-b-r2`; all 500 | 0 | 64 x 200 (61 `stop`, 3 `length` at 2056) |
| `O1-on8192` | 2 | 1 | 2482 | 1423; `iv-a-r1` P=1661 G=8200, 9861 | 1167 (9858) `iv-a-r0` 500 | 0 (second panic at 2479, nothing in flight, after `iv-b-r2` stopped at 8320) | 64 no status |
| `O1-on32768` | 1 | 1 | no | 1917; `iv-a-r0` P=1661 G=32776, 34437 | 2968 (34435) `iv-a-r1`; 3969 (34435) `iv-a-r2`; 5549 (32894) `iv-b-r1`; all 500 | 0 | 47 x 200, 17 x 429 |
| `O2-on32768` | 1 | 1 | no | 4492; `iv-a-r0` P=1661 G=32776, 34437 | 2494 (32894) `iv-b-r1`; 5543 (34435) `iv-a-r1`; 6544 (34435) `iv-a-r2`; all 500 | 0 | 47 x 200, 17 x 429 |
| `O2-on8192` | 2 | 1 | 2230 | 1682; `iv-b-r2` P=120 G=8200, 8320 | 1970 (9858) `iv-a-r0` 500 | 1 (`iv-a-r2`; second panic at 2227 after `iv-a-r1` stopped at 9861) | 64 no status |
| `O2-on2048` | 0 | 0 | no | none | 983 (2173) `iv-b-r0`; 1051 (2174) `iv-b-r1`; 1119 (2173) `iv-b-r2`; 1189 (3714) `iv-a-r0`; 1256 (3714) `iv-a-r1`; 1325 (3716) `iv-a-r2`; all 500 | 0 | 64 x 200 (61 `stop`, 3 `length` at 2056) |
| `O2-off` | 0 | 0 | no | none | 20399 (262142) `iv-a-r1` 500 | 0 | 64 x 200 |

The two OFF boots' `server.log` are committed gzipped (19 MB each, sha256 of the uncompressed file in
`server.log.sha256`). Again every panic follows a request that ended `length` with `P + G` equal to its allocation
(9861 = 1661 + 8200, 8320 = 120 + 8200, 34437 = 1661 + 32776). Requests that ended 2 to 3 rows short of it
(`iv-b-r0` 8198 and `iv-b-r1` 8197 at 8192, `iv-b-r2` 32773 at 32768) were followed by no panic. Every #87 trap sits 1
to 3 rows below the allocation (3717, 2176, 9861, 34437, 32896, 262144). The target card shows no panic under OFF: its
one request that ran to the served context died at the trap first. Both orders agree to the request and to the
panic's request on every arm. `compute-apps-before.csv` and `compute-apps-after.csv` of every boot list no process.

First panic on each rig, for the fix lane's red:

- Local RTX 5090: `research/spill-b-20260919/rtx5090-day31/boots/O1-off/server.log.gz`, line 10286
  (`thread 'memra-gpu-worker' (456749) panicked at crates/memra-engine/src/spec.rs:4590:9:`), PANIC line 10289, respawn
  line 10290, after `iv-b-r1` ended `length` at `P + G = 78 + 65458 = 65536`. First ON-arm panic:
  `research/spill-b-20260919/rtx5090-day31/boots/O1-on2048/server.log` line 116, PANIC line 119, respawn line 120,
  after `i-L0-r0` ended `length` at `P + G = 1439 + 2056 = 3495`; FATAL line 431.
- Target card: `research/spill-b-20260919/pro-single-day31/box/boots/O1-on8192/server.log`, line 1423
  (`thread 'memra-gpu-worker' (360539) panicked at crates/memra-engine/src/spec.rs:4590:9:`), PANIC line 1426
  (`[worker] PANIC in the GPU worker thread: mtp_kv_fill: scratch overflow`), respawn line 1427, after `iv-a-r1`
  ended `length` at `P + G = 1661 + 8200 = 9861`; FATAL line 2482.

### 2.5 The ON-arm result

**The open-output bound reaches an engine defect under default spec.** Door ON gives every open request an
allocation of `P + v + 8` rows and a budget of `v + 8` tokens (1.1 (a)), so a request that runs to the bound stops with
zero rows to spare, exactly where a speculative round can write past the cache. That is where every panic and every
#87 trap in 2.4 sits. The defect killed the process (FATAL) at `v = 2048` and `v = 8192` in both orders on the local
card, and at `v = 8192` in both orders on the target card. What those boots report after the kill is not a
truncation or concurrency number, and none is claimed for them: their bursts got no response at all. Where the
process lived, the defect still took requests. At `v = 32768` three open requests per order returned 500 at the
bound on both cards, and the target card also panicked and respawned once per order. At `v = 2048` on the target
card every long-generation request (6 of 6 per order) returned 500 at the bound; no request there was cut to
`length`, so the bound reads as zero truncation on that arm only because the defect answered first. Door OFF
reaches the same defect only on requests that run to the whole served context: the local 65,536 in both orders
(one panic and one 500 per order), and on the target card `iv-a-r1` at 262,142 in both orders (500, no panic).

These numbers were measured on the pre-fix program: open requests allocated at `P + v + 8` with a `v + 8` budget,
crashing at the cap. The lead filed memra#659 and has a fix on `lane/spec-ctx-edge-20260923` at `54711e6b5` (not
merged; its GPU gate follows this chain on the local card). Two facts from the lead, for the decision:

1. The engine now ends a qwen/gemma speculative burst at a round boundary before a round would write past the cache
   (the glm5 guard form), plus a typed refusal in the verify funnel.
2. On the door-ON open arm the budget becomes `v`, not `v + 8`: an open request emits exactly
   `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` tokens and the 8 slack rows stay free. Door-OFF open requests keep the whole
   `MEMRA_CTX` budget; speculative ones end `ContextFull` about `k + 3` short of the cap (the existing between-burst
   guard), plain ones at the cap.

So the ON arm the owner would decide on is a rerun of this cell on the fixed tree. It is owed (2.11).

### 2.6 Verdict lines, verbatim

Local RTX 5090, `rtx5090-day31/SUMMARY.txt` (`day31-compare.py --card rtx5090 --served-ctx 65536 --model-ctx 262144
--registry-value 32768 --survey context` over the eight boots under `rtx5090-day31/boots/`):

```
== card rtx5090 served_ctx=65536 model_ctx=262144
DAY31 V-BOOT boot=O1-off arm=off door_line=present -> PASS
DAY31 V-BOOT boot=O1-on2048 arm=on2048 door_line=present -> PASS
DAY31 V-BOOT boot=O1-on8192 arm=on8192 door_line=present -> PASS
DAY31 V-BOOT boot=O1-on32768 arm=on32768 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on32768 arm=on32768 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on8192 arm=on8192 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on2048 arm=on2048 door_line=present -> PASS
DAY31 V-BOOT boot=O2-off arm=off door_line=present -> PASS
== V-ALLOC
DAY31 V-ALLOC card=rtx5090 boot=O1-off arm=off judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O1-on2048 arm=on2048 judged=3 match=3 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O1-on8192 arm=on8192 judged=46 match=46 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O1-on32768 arm=on32768 judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O2-on32768 arm=on32768 judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O2-on8192 arm=on8192 judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O2-on2048 arm=on2048 judged=7 match=7 mismatch=0 -> PASS
DAY31 V-ALLOC card=rtx5090 boot=O2-off arm=off judged=44 match=44 mismatch=0 -> PASS
== V-ID / V-TRUNC
DAY31 V-ID card=rtx5090 order=O1 arm=on2048 vs=off eligible=1 equal=1 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O1 arm=on2048 truncated_with_twin=2 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=rtx5090 order=O1 arm=on8192 vs=off eligible=48 equal=48 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O1 arm=on8192 truncated_with_twin=0 length_without_twin=0 length_both=2 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=rtx5090 order=O1 arm=on32768 vs=off eligible=48 equal=48 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O1 arm=on32768 truncated_with_twin=0 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=rtx5090 order=O2 arm=on32768 vs=off eligible=48 equal=48 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O2 arm=on32768 truncated_with_twin=0 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=rtx5090 order=O2 arm=on8192 vs=off eligible=45 equal=45 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O2 arm=on8192 truncated_with_twin=0 length_without_twin=0 length_both=2 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=rtx5090 order=O2 arm=on2048 vs=off eligible=6 equal=6 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=rtx5090 order=O2 arm=on2048 truncated_with_twin=2 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
== G-NATURAL (door OFF, open classes, both orders)
DAY31 G-NATURAL card=rtx5090 class=i n=30 min=2413 p50=3921 p90=4817 p99=5048 max=5048 over2048=30 over8192=0 over32768=0
DAY31 G-NATURAL card=rtx5090 class=iii n=30 min=2173 p50=3724 p90=4715 p99=59715 max=59715 over2048=30 over8192=2 over32768=2
DAY31 G-NATURAL card=rtx5090 class=iv-a n=6 min=5413 p50=5848 p90=6405 p99=6405 max=6405 over2048=6 over8192=0 over32768=0
DAY31 G-NATURAL card=rtx5090 class=iv-b n=4 min=65458 p50=65458 p90=65458 p99=65458 max=65458 over2048=4 over8192=4 over32768=4
DAY31 G-NATURAL card=rtx5090 class=all-open n=70 min=2173 p50=4123 p90=6405 p99=65458 max=65458 over2048=70 over8192=6 over32768=6 finish={'stop': 64, 'length': 6}
== V-CONC / V-RETRY
DAY31 V-CONC card=rtx5090 order=O1 arm=off window_ms=1669989 B=32 status={200: 32} finish={'stop': 27, 'length': 5} retry_after={} active_sessions_max=6.0 admission_booked_bytes_max=10075702352.0 samples=6680 shadow_budget_bytes=12831671424 burst_cost_lines=444759 burst_cost_total_mb_median=1678 arith_sessions=7
DAY31 V-RETRY card=rtx5090 order=O1 arm=off r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=rtx5090 order=O1 arm=on2048 window_ms=16 B=32 status={None: 32} finish={} retry_after={None: 32} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=12831671424 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=rtx5090 order=O1 arm=on2048 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=32 -> PASS
DAY31 V-CONC card=rtx5090 order=O1 arm=on8192 window_ms=16 B=32 status={None: 32} finish={} retry_after={None: 32} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=12831671424 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=rtx5090 order=O1 arm=on8192 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=32 -> PASS
DAY31 V-CONC card=rtx5090 order=O1 arm=on32768 window_ms=524356 B=32 status={503: 2, 429: 11, 200: 19} finish={'stop': 17, 'length': 2} retry_after={'5': 2, '60': 11} active_sessions_max=19.0 admission_booked_bytes_max=22075472424.0 samples=2098 shadow_budget_bytes=12831671424 burst_cost_lines=254 burst_cost_total_mb_median=1162.0 arith_sessions=11
DAY31 V-RETRY card=rtx5090 order=O1 arm=on32768 r429=11 refuse_lines=11 retry_after_in_1_60=True other_non200=2 -> PASS
DAY31 V-CONC card=rtx5090 order=O2 arm=on32768 window_ms=641206 B=32 status={503: 2, 429: 11, 200: 19} finish={'stop': 15, 'length': 4} retry_after={'5': 2, '60': 11} active_sessions_max=19.0 admission_booked_bytes_max=22183181304.0 samples=2565 shadow_budget_bytes=12831671424 burst_cost_lines=301 burst_cost_total_mb_median=1162 arith_sessions=11
DAY31 V-RETRY card=rtx5090 order=O2 arm=on32768 r429=11 refuse_lines=11 retry_after_in_1_60=True other_non200=2 -> PASS
DAY31 V-CONC card=rtx5090 order=O2 arm=on8192 window_ms=14 B=32 status={None: 32} finish={} retry_after={None: 32} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=12831671424 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=rtx5090 order=O2 arm=on8192 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=32 -> PASS
DAY31 V-CONC card=rtx5090 order=O2 arm=on2048 window_ms=17 B=32 status={None: 32} finish={} retry_after={None: 32} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=12831671424 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=rtx5090 order=O2 arm=on2048 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=32 -> PASS
DAY31 V-CONC card=rtx5090 order=O2 arm=off window_ms=1913825 B=32 status={200: 32} finish={'stop': 27, 'length': 5} retry_after={} active_sessions_max=6.0 admission_booked_bytes_max=10074621008.0 samples=7656 shadow_budget_bytes=12831671424 burst_cost_lines=322633 burst_cost_total_mb_median=1679 arith_sessions=7
DAY31 V-RETRY card=rtx5090 order=O2 arm=off r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
== SELECT (stated, not chosen)
DAY31 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405) R3_registry=32768 R4_survey=context
DAY31 V-DOOR card=rtx5090 boots=8 excluded=0 v_id_all=True v_alloc_all=True v_trunc_g_all=True v_retry_all=True -> PASS
```

Target card, `pro-single-day31/box/SUMMARY.txt` (`day31-compare.py --card pro6000 --served-ctx 262144 --model-ctx
262144 --registry-value 32768 --survey context` over the eight boots under `pro-single-day31/box/boots/`):

```
== card pro6000 served_ctx=262144 model_ctx=262144
DAY31 V-BOOT boot=O1-off arm=off door_line=present -> PASS
DAY31 V-BOOT boot=O1-on2048 arm=on2048 door_line=present -> PASS
DAY31 V-BOOT boot=O1-on8192 arm=on8192 door_line=present -> PASS
DAY31 V-BOOT boot=O1-on32768 arm=on32768 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on32768 arm=on32768 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on8192 arm=on8192 door_line=present -> PASS
DAY31 V-BOOT boot=O2-on2048 arm=on2048 door_line=present -> PASS
DAY31 V-BOOT boot=O2-off arm=off door_line=present -> PASS
== V-ALLOC
DAY31 V-ALLOC card=pro6000 boot=O1-off arm=off judged=45 match=45 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O1-on2048 arm=on2048 judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O1-on8192 arm=on8192 judged=46 match=46 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O1-on32768 arm=on32768 judged=46 match=46 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O2-on32768 arm=on32768 judged=46 match=46 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O2-on8192 arm=on8192 judged=46 match=46 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O2-on2048 arm=on2048 judged=44 match=44 mismatch=0 -> PASS
DAY31 V-ALLOC card=pro6000 boot=O2-off arm=off judged=45 match=45 mismatch=0 -> PASS
== V-ID / V-TRUNC
DAY31 V-ID card=pro6000 order=O1 arm=on2048 vs=off eligible=46 equal=46 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O1 arm=on2048 truncated_with_twin=0 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=pro6000 order=O1 arm=on8192 vs=off eligible=46 equal=46 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O1 arm=on8192 truncated_with_twin=4 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=pro6000 order=O1 arm=on32768 vs=off eligible=47 equal=47 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O1 arm=on32768 truncated_with_twin=2 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=pro6000 order=O2 arm=on32768 vs=off eligible=47 equal=47 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O2 arm=on32768 truncated_with_twin=2 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=pro6000 order=O2 arm=on8192 vs=off eligible=46 equal=46 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O2 arm=on8192 truncated_with_twin=3 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
DAY31 V-ID card=pro6000 order=O2 arm=on2048 vs=off eligible=46 equal=46 differ=0 slack_equal=0 slack_differ=0 -> PASS
DAY31 V-TRUNC card=pro6000 order=O2 arm=on2048 truncated_with_twin=0 length_without_twin=0 length_both=0 deadline_cut=0 truncated_G_outside_[v,v+8]=0
== G-NATURAL (door OFF, open classes, both orders)
DAY31 G-NATURAL card=pro6000 class=i n=30 min=354 p50=505 p90=673 p99=954 max=954 over2048=0 over8192=0 over32768=0
DAY31 G-NATURAL card=pro6000 class=iii n=30 min=211 p50=407 p90=576 p99=621 max=621 over2048=0 over8192=0 over32768=0
DAY31 G-NATURAL card=pro6000 class=iv-a n=4 min=102441 p50=134383 p90=134383 p99=134383 max=134383 over2048=4 over8192=4 over32768=4
DAY31 G-NATURAL card=pro6000 class=iv-b n=6 min=18443 p50=58001 p90=193178 p99=193178 max=193178 over2048=6 over8192=6 over32768=4
DAY31 G-NATURAL card=pro6000 class=all-open n=70 min=211 p50=505 p90=58001 p99=193178 max=193178 over2048=10 over8192=10 over32768=8 finish={'stop': 70}
== V-CONC / V-RETRY
DAY31 V-CONC card=pro6000 order=O1 arm=off window_ms=167951 B=64 status={200: 64} finish={'stop': 64} retry_after={} active_sessions_max=9.0 admission_booked_bytes_max=83743021720.0 samples=672 shadow_budget_bytes=65881157328 burst_cost_lines=114939 burst_cost_total_mb_median=9239 arith_sessions=7
DAY31 V-RETRY card=pro6000 order=O1 arm=off r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=pro6000 order=O1 arm=on2048 window_ms=129140 B=64 status={200: 64} finish={'stop': 61, 'length': 3} retry_after={} active_sessions_max=64.0 admission_booked_bytes_max=69424286208.0 samples=517 shadow_budget_bytes=65881157328 burst_cost_lines=63 burst_cost_total_mb_median=1084 arith_sessions=60
DAY31 V-RETRY card=pro6000 order=O1 arm=on2048 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=pro6000 order=O1 arm=on8192 window_ms=20 B=64 status={None: 64} finish={} retry_after={None: 64} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=65881157328 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=pro6000 order=O1 arm=on8192 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=64 -> PASS
DAY31 V-CONC card=pro6000 order=O1 arm=on32768 window_ms=96283 B=64 status={429: 17, 200: 47} finish={'stop': 47} retry_after={'60': 17} active_sessions_max=47.0 admission_booked_bytes_max=96260417960.0 samples=385 shadow_budget_bytes=65881157328 burst_cost_lines=74 burst_cost_total_mb_median=2046.0 arith_sessions=32
DAY31 V-RETRY card=pro6000 order=O1 arm=on32768 r429=17 refuse_lines=17 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=pro6000 order=O2 arm=on32768 window_ms=108450 B=64 status={429: 17, 200: 47} finish={'stop': 47} retry_after={'60': 17} active_sessions_max=47.0 admission_booked_bytes_max=96267554088.0 samples=434 shadow_budget_bytes=65881157328 burst_cost_lines=77 burst_cost_total_mb_median=2046 arith_sessions=32
DAY31 V-RETRY card=pro6000 order=O2 arm=on32768 r429=17 refuse_lines=17 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=pro6000 order=O2 arm=on8192 window_ms=21 B=64 status={None: 64} finish={} retry_after={None: 64} active_sessions_max=None admission_booked_bytes_max=None samples=0 shadow_budget_bytes=65881157328 burst_cost_lines=0 burst_cost_total_mb_median=None arith_sessions=None
DAY31 V-RETRY card=pro6000 order=O2 arm=on8192 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=64 -> PASS
DAY31 V-CONC card=pro6000 order=O2 arm=on2048 window_ms=129111 B=64 status={200: 64} finish={'stop': 61, 'length': 3} retry_after={} active_sessions_max=64.0 admission_booked_bytes_max=69424286208.0 samples=516 shadow_budget_bytes=65881157328 burst_cost_lines=60 burst_cost_total_mb_median=1084.5 arith_sessions=60
DAY31 V-RETRY card=pro6000 order=O2 arm=on2048 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY31 V-CONC card=pro6000 order=O2 arm=off window_ms=180369 B=64 status={200: 64} finish={'stop': 64} retry_after={} active_sessions_max=9.0 admission_booked_bytes_max=83744459416.0 samples=721 shadow_budget_bytes=65881157328 burst_cost_lines=109189 burst_cost_total_mb_median=9239 arith_sessions=7
DAY31 V-RETRY card=pro6000 order=O2 arm=off r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
== SELECT (stated, not chosen)
DAY31 SELECT card=pro6000 R1_smallest_zero_truncation=2048 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768] (max_natural_G=193178) R3_registry=32768 R4_survey=context
DAY31 V-DOOR card=pro6000 boots=8 excluded=0 v_id_all=True v_alloc_all=True v_trunc_g_all=True v_retry_all=True -> PASS
```

D4's lines are in `DAY31-D4.md` 2.2 (`rtx5090-day31-d4/SUMMARY.txt`).

### 2.7 Natural G per model (door OFF, both orders)

From the G-NATURAL lines (status 200 open rows of both OFF boots). The two OFF boots of each card agree on all 52
non-burst rows (status, G and `message_sha256`), so each value below occurs once per order.

| model, card | class | n | min | p50 | p90 | max | finish | rows over 2048 / 8192 / 32768 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B, RTX 5090 (served 65,536) | (i) | 30 | 2413 | 3921 | 4817 | 5048 | 30 `stop` | 30 / 0 / 0 |
| | (iii) | 30 | 2173 | 3724 | 4715 | 59715 | 28 `stop`, 2 `length` | 30 / 2 / 2 |
| | iv-a | 6 | 5413 | 5848 | 6405 | 6405 | 6 `stop` | 6 / 0 / 0 |
| | iv-b | 4 | 65458 | 65458 | 65458 | 65458 | 4 `length` | 4 / 4 / 4 |
| | all open | 70 | 2173 | 4123 | 6405 | 65458 | 64 `stop`, 6 `length` | 70 / 6 / 6 |
| Qwen3.8-27B, RTX PRO 6000 (served 262,144) | (i) | 30 | 354 | 505 | 673 | 954 | 30 `stop` | 0 / 0 / 0 |
| | (iii) | 30 | 211 | 407 | 576 | 621 | 30 `stop` | 0 / 0 / 0 |
| | iv-a | 4 | 102441 | 134383 | 134383 | 134383 | 4 `stop` | 4 / 4 / 4 |
| | iv-b | 6 | 18443 | 58001 | 193178 | 193178 | 6 `stop` | 6 / 6 / 4 |
| | all open | 70 | 211 | 505 | 58001 | 193178 | 70 `stop` | 10 / 10 / 8 |

- 9B. The largest G of a row that stopped is 6405 (`iv-a-r1`). Every longer row ran to the 65,536 cap: `iii-L2-r3`
  (P=5821, G=59715, `length`), `iv-b-r0` and `iv-b-r1` (P=78, G=65458, `length`; even k = 24 did not stop), and
  `iv-b-r2` (500 on the #87 trap at pos 65533), in both orders. The burst (not in G-NATURAL): 27 `stop` and 5
  `length` at G 64201 to 64206 (the cap) per order, p50 4073. The 9B reasons for 2.4k to 5k tokens on every (i)
  request, so no (i) row fits under 2048.
- 27B. Short classes stop at 954 or less. The long class runs far: `iv-a` 102,441 and 134,383 `stop`, `iv-b` 18,443,
  58,001 and 193,178 `stop`, and `iv-a-r1` ran to the 262,144 served context in both orders (500 on the #87 trap at pos
  262142, after 3242 s in each order). The burst: 64 `stop` per order, G 186 to 2680, p50 424.5.

### 2.8 Truncation and concurrency per value, per card

`tw` = `truncated_with_twin`, `lb` = `length_both`, "gap" = an ON `length` row whose OFF twin returned 500 (1.6 names
it `length_without_twin`; the reader counts it nowhere, 2.2; listed by hand from `rows.jsonl`). Every `length` row of
every ON arm, burst included, has G inside `[v, v + 8]`. Concurrency columns are the V-CONC burst readings; "dead" means the process had
exited (FATAL) before the burst and no burst request got a status.

Local RTX 5090, Qwen3.5-9B, B = 32:

| v | order | V-ID eq/eligible | tw | lb | gap | ON open non-200 | burst | active max | arith | process |
|---|---|---|---|---|---|---|---|---|---|---|
| off | O1 | | | | | `iv-b-r2` 500 | 32 x 200 (27 `stop`, 5 `length`) | 6 | 7 | 1 panic, respawn |
| off | O2 | | | | | `iv-b-r2` 500 | 32 x 200 (27 `stop`, 5 `length`) | 6 | 7 | 1 panic, respawn |
| 2048 | O1 | 1/1 | 2 | 0 | 0 | 34 (3 x 500, 31 no status) | dead | | | FATAL |
| 2048 | O2 | 6/6 | 2 | 0 | 0 | 34 (3 x 500, 31 no status) | dead | | | FATAL |
| 8192 | O1 | 48/48 | 0 | 2 | `iv-b-r2` G=8200 | `iv-b-r1` 500 | dead | | | FATAL |
| 8192 | O2 | 45/45 | 0 | 2 | `iv-b-r2` G=8200 | 4 (`iv-b-r1` 500, `iv-a-r0..r2` no status) | dead | | | FATAL |
| 32768 | O1 | 48/48 | 0 | 0 | `iv-b-r2` G=32772 | 3 (`iii-L2-r3`, `iv-b-r0`, `iv-b-r1` 500) | 19 x 200 (17 `stop`, 2 `length`), 11 x 429 (60 s), 2 x 503 (5 s) | 19 | 11 | lived |
| 32768 | O2 | 48/48 | 0 | 0 | `iv-b-r2` G=32772 | 3 (same tags) | 19 x 200 (15 `stop`, 4 `length`), 11 x 429 (60 s), 2 x 503 (5 s) | 19 | 11 | lived |

Target card, one RTX PRO 6000 Blackwell, Qwen3.8-27B, B = 64:

| v | order | V-ID eq/eligible | tw | lb | gap | ON open non-200 | burst | active max | arith | process |
|---|---|---|---|---|---|---|---|---|---|---|
| off | O1 | | | | | `iv-a-r1` 500 | 64 x 200 `stop` | 9 | 7 | no panic |
| off | O2 | | | | | `iv-a-r1` 500 | 64 x 200 `stop` | 9 | 7 | no panic |
| 2048 | O1 | 46/46 | 0 | 0 | 0 | 6 (every iv row, 500) | 64 x 200 (61 `stop`, 3 `length`) | 64 | 60 | no panic |
| 2048 | O2 | 46/46 | 0 | 0 | 0 | 6 (every iv row, 500) | 64 x 200 (61 `stop`, 3 `length`) | 64 | 60 | no panic |
| 8192 | O1 | 46/46 | 4 | 0 | `iv-a-r1` G=8200 | `iv-a-r0` 500 | dead | | | FATAL |
| 8192 | O2 | 46/46 | 3 | 0 | `iv-a-r1` G=8200 | 2 (`iv-a-r0` 500, `iv-a-r2` no status) | dead | | | FATAL |
| 32768 | O1 | 47/47 | 2 | 0 | 0 | 3 (`iv-a-r1`, `iv-a-r2`, `iv-b-r1` 500) | 47 x 200 `stop`, 17 x 429 (60 s) | 47 | 32 | 1 panic, respawn |
| 32768 | O2 | 47/47 | 2 | 0 | 0 | 3 (same tags) | 47 x 200 `stop`, 17 x 429 (60 s) | 47 | 32 | 1 panic, respawn |

What the rows say, on this program:

- 9B at 8192 and 32768: no request that stops under OFF was cut (`tw = 0` in both orders). Every ON `length` row is a
  request that under OFF ran to the 65,536 cap (`lb`: `iii-L2-r3` and `iv-b-r0` at 8192) or died there (gap:
  `iv-b-r2`). At 2048 both (i) requests that returned 200 before the FATAL were cut (`i-L0-r0`, OFF 3921; `i-L0-r4`, OFF
  5048), as 2.7 predicts (the 9B's (i) stops at 2413 to 5048).
- 27B at 8192: 4 (O1) and 3 (O2) requests cut that stop under OFF at 18,443 to 193,178 (`iv-a-r2`, `iv-b-r0..r2`;
  O2 lost `iv-a-r2` to the FATAL). At 32768: 2 per order (`iv-a-r0`, OFF 102,441; `iv-b-r2`, OFF 193,178). At 2048
  the reader counts 0, but all 12 long requests died at the bound, and the burst cut 3 of 64 per order to 2056 (OFF
  burst maximum 2680, all `stop`).
- Concurrency. The 27B burst under OFF held at most 9 sessions at once (arith 7). At 2048 the door admitted all 64
  (arith 60). At 32768 it admitted 47 and refused 17 after the 8 s defer with `Retry-After: 60`. The 9B burst under
  OFF held 6 (arith 7). At 32768 the door admitted 19 and refused 11 with `Retry-After: 60`. 2 more requests passed
  admission and failed their prefill with CUDA OOM (503, 2.4). At 8192 on both cards, and at 2048 locally, the
  process was dead before the burst, so no concurrency reading exists for those arms.

### 2.9 Identity gate

V-ID PASS for every ON arm in both orders on both cards, `differ=0` and no slack-band row anywhere: local 1/1, 48/48,
48/48 (O1: 2048, 8192, 32768) and 48/48, 45/45, 6/6 (O2: 32768, 8192, 2048); target 46/46, 46/46, 47/47 and 47/47,
46/46, 46/46. The local 2048 counts are small because the process died after the first (i) rows. D4-ID PASS 49/49
(`DAY31-D4.md`). Where both sides answered 200 on the same prompt and the OFF request stopped within the bound, the
door changed no token.

### 2.10 Selection rules, stated (not chosen)

As printed:

```
DAY31 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405) R3_registry=32768 R4_survey=context
DAY31 SELECT card=pro6000 R1_smallest_zero_truncation=2048 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768] (max_natural_G=193178) R3_registry=32768 R4_survey=context
```

How far each literal reading carries:

- R1, local. The reader prints 8192. 1.6 defines the gap rows (`iv-b-r2` at 8192 and at 32768, both orders) as
  `length_without_twin`, so 1.7's rule as written reads R1 = none on this card. Both readings are recorded; neither
  is a pick.
- R1, target. The reader prints 2048, and the rule as written reads the same, because the rule counts `length` rows
  and the 12 long requests at 2048 returned 500. Each of their traps sits 1 to 3 rows below the allocation, so every
  one of them reached the bound. This is not a zero-truncation observation. If those 12 requests had ended at the
  bound as `length`, then in each order the 5 whose OFF twin stopped would count as `tw` and `iv-a-r1` (OFF 500) as a
  gap row, and R1 would read none on this card too.
- R2 = 8192 locally (largest OFF `stop` G 6405) and none on the target card (193,178). R3 = 32768 on both (1.7).
  R4 = `context` on both: the surveyed engines bound an omitted output by the remaining context, which selects
  `off` for the output bound (`OPEN-OUTPUT-SURVEY.md`, R4).
- V-DOOR prints PASS on both cards. V-DOOR has no crash term, and V-RETRY passes vacuously on the six boots whose
  bursts got no response (2.2). The PASS is the literal reading of 1.6's terms on the rows that returned. It does not
  say the door ran clean, and the door did not.

Which rule applies, on which program, is the owner's call. On the fixed tree the ON budget is `v`, not `v + 8`
(2.5), so the band `[v, v + 8]` in 1.6 changes. The rerun needs its own pre-registration.

### 2.11 Owed

| item | why | price |
|---|---|---|
| Rerun of the whole cell on the fixed tree (`54711e6b5` or its merge): 8 boots per card, same arms, values, orders and workload | every ON number here was measured on the pre-fix program, and the fix also moves the OFF arm (speculative requests end `ContextFull` about `k + 3` short of the cap) | 0.5 agent-day: 0.1 for a fresh pre-registration before its first boot (the ON band becomes `G = v`; the three reader gaps in 2.2 closed there), 0.1 to launch both rigs, 0.3 to read and write up. Rig time: local about 4 h (this cell ran 23:09 to 03:08), target card about 6 h (00:06 to 05:57; each OFF boot about 2 h 17 min, because the long class runs 18k to 262k tokens). The two rigs run in parallel |
| The door's decision | its decide-by (2026-09-23) has passed; the owner decides on the rerun's receipts | owner |
| `admit_memory.rs:48-51` (says the registries pin 8192; they pin 32768) and the `lib.rs:3428-3429` OpenAI attribution | stale engine text; this lane stays out of the engine | 0.05 agent-day, with the fix lane or the door's decision |
| D4 (b) and D4 (c) | done (`DAY31-D4.md`) | none |

### 2.12 Cleanup and rig state

- The local RTX 5090 lock was released when the D4 collector exited at 03:42:51Z. This lane has no further local
  cell. Any local rerun waits for the spec-ctx-edge lane's `SPEC-CTX-EDGE-5090-DONE` marker (the lead holds the card
  for its merge battery and the memra#659 red/green gate).
- Local scratch deleted: the survey clones, the stub server, the two git bundles, the chain output file, the fetched
  reference page, the reader-gap script, and `target/day31/memra-server` (its sha256 stays in
  `rtx5090-day31/binary.sha256`).
- Target card: the two git bundles, the chain script copy (identical to `pro-single-day31/chain.sh`) and the two
  empty chain output files deleted. The receipt originals stay under `/root/spill-receipts/b-day31` (run 1) and
  `/root/spill-receipts/b-day31-run2` (run 2). Run 2's `bins/` keeps binary `ef3847d0...212fbd`, which the fix lane
  can use for its target-card red. No process of this lane is left on either card.
