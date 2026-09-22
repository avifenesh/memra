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
