# WP-B day 32: `MEMRA_ADMIT_BY_MEMORY` OFF against ON, rerun on the fixed tree (memra#659)

DAY31.md 2.11 owed this rerun: every day-31 ON number was measured on the pre-fix program, and the open-output bound
reached an engine defect under default spec (`spec.rs:4590 mtp_kv_fill: scratch overflow`, DAY31.md 2.4 and 2.5).
The fix is memra#659, merged to main as `d544c6b82` (#668). This file reruns the day-31 cell on that tree: the same
arms, bound values, workloads, orders and cards. Every cell is `executed-not-qualified`; no qualification is claimed,
no default moves in this lane, and no open-output value is picked.

## 1. Pre-registration

Committed and pushed before the first boot of this cell. Nothing in section 1 changes after a number is seen; a
correction to a script that does not change a rule is recorded in section 2 with its commit.

### 1.1 The program under test

Both binaries are built from main `d544c6b82`. Against day 31's engine source (`5f1b0eda4`), main moved 50 commits;
the ones this cell reads come from memra#659 (`acd71c878`, `047e742e1`):

- Door ON budget. `request_budget` (worker.rs) gives an open request under the door (`max_ctx` and `max_tokens`
  omitted, the door armed) `budget = min(max_new, ctx_cap - P - 8)`. The charge and the allocation are unchanged:
  `ctx_cap = min(P + v + 8, model_ctx)`. So under ON an open request that does not stop first ends at exactly
  `G = v` with `finish_reason = "length"`, and the 8 slack rows stay free, as on a bounded request. Day 31's budget
  was `v + 8`.
- Speculative round guard. The qwen round loop (`generate_spec_inner2`) and the gemma burst loops end a burst when the
  next round cannot land inside the session cache, and the verify funnel refuses a window past the cache with the
  typed error `spec verify refused: rows a..b would land past the session cache (n rows)`. A request whose budget
  spans its whole cap (a door-OFF open request) ends `ContextFull`, which the OpenAI surface reports as
  `finish_reason = "length"` (lib.rs, `"MaxNew" | "ContextFull" => "length"`), a few rows short of the cap on the
  speculative path (the lead's statement for memra#659: about `k + 3`).
- Unchanged from DAY31.md 1.1: (b) the host tier is unarmed on every arm, so demotion is not exercised; (c) the bounded
  defer is 8000 ms, then a 429 with `Retry-After` in 1..=60 s.

### 1.2 Rigs, artifacts, binaries

| card | model | served context | burst B | lock |
|---|---|---|---|---|
| local RTX 5090 | Qwen3.5-9B NVFP4 MTP GGUF, sha256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de` | `MEMRA_CTX=65536` | 32 | `/tmp/memra-5090.lock` |
| BOX3, one RTX PRO 6000 Blackwell | Qwen3.8-27B NVFP4-Q5K MTP GGUF (sha256 recorded per boot in `model.sha256`; day 31: `1facf36c...1e024a`) | unset (declared 262,144) | 64 | `/tmp/memra-gpu.lock` |

One binary per card, built once from main `d544c6b82` in a detached worktree of that commit (so the lane's own
`docs/FLAGS.md` text is not an input), and copied once:

| card | build line (verbatim) | binary sha256 |
|---|---|---|
| local | `cd /home/avifenesh/projects/wt-b32-main && CARGO_TARGET_DIR=/home/avifenesh/projects/wt-spill-b/target systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=14G cargo build --release -p memra-server` (nvcc 13.1, sm_120a; `rtx5090-day32/build-local.log`, exit 0), copied to `target/day32/memra-server` | `d92b6cdeec7e84b7dd030db87df11b1d673bf749c92183f2d6e72ab508484ea7` |
| BOX3 | `cd /root/wt-b32-main && CARGO_TARGET_DIR=/root/wt-b/target nice -n 10 cargo build --release -p memra-server` (nvcc 13.2, sm_120a; mirrored `build.log`, exit 0), copied to the receipt root's `bins/` | `5bbe4857c36208c70682cf77ea7ad7b9c7615941f58aaee607a6ff48126b88a8` |

Each chain refuses to start if the binary's sha256 differs from the recorded one. Every boot's environment is
DAY31.md 1.2's, because the drivers are unchanged (1.3): `MEMRA_COMPAT=openai`, `MEMRA_ADMIT_PREDICT_SHADOW=1`,
`MEMRA_TIMEOUT_MS_MAX=3600000` on every arm, `MEMRA_MAX_SESSIONS` at its default 64, no host tier, speculative
decoding at its default. Every name is a live `docs/FLAGS.md` row on main (the boot-time env audit would refuse a
retired one). Timing is never compared across cards or boxes, or against day 31. Throughput is not a reading.

### 1.3 Arms and orders

Identical to DAY31.md 1.3: `off`, `on2048`, `on8192`, `on32768`; O1 = `off, on2048, on8192, on32768` with inner
order AB; O2 = the reverse with inner order BA; one boot per arm per order, 8 boots per card; one collector hold per
order, so the four arms of an order share one lock window and nothing else can run inside an order.

The drivers are the day-31 files, unchanged at this commit (sha256):

| file | sha256 |
|---|---|
| `day31-order.sh` | `a5626dd845f37c639f5c61ec5be3615e9a36bf8612d46a754e1df2b3b0e350d1` |
| `run-day26-cell.sh` | `fad91c06a219e42867e42fb2203b107f9cd0ed03a4b7c0ec66fe8ea4ddd2ee66` |
| `day31-client.py` | `53aa251263cb559fab28cc56b71ee9b8077be70ef8b9e6cabc9a8aede551fda2` (DAY31.md 1.4's hash) |
| `day31-parse.py` | `8f27e7f922d2f30238fd33dbcaee67c818fb8d87ea5fbf160a58ebe9d0c07e09` |

The chains are new: `rtx5090-day32/chain.sh` (bounded idle wait of at most 14400 s before each order: lock free, no
compute app, at least 24 GB of host memory available) and `pro-single-day32/chain.sh` (at most 7200 s: lock free, no
compute app; fast-forwards the lane tip for the drivers only; collector hold 43,200 s per order). The two cards run in
parallel, since each is a separate rig with its own lock. Another lane may hold either card between orders; the wait
is logged in `chain.log`.

### 1.4 Workload

DAY31.md 1.4, unchanged: `day31-client.py` as above, prompts from `docs/SERVING.md`, sha256
`022b1d50620f5b34bb6e6127be05ff79184ce3aa294aca618a31a0dd73887cff` (the same file as day 31; recorded per boot in
`inputs.sha256`). Warmup; the day-26 mix (i), (ii) `max_tokens=96`, (iii) at L0/L1/L2, 5 reps each; the long class
`iv-a` and `iv-b` (k = 24, 40, 72), 3 reps each; the burst of B open (i)-shape requests on one barrier.

### 1.5 Readings

Per boot, `day31-parse.py` as in DAY31.md 1.5 (`REPORT.txt`, `rows.jsonl`). Across boots, `day32-compare.py`
(sha256 `a59b650c67cac18cb0534352378647f96fd30dd7d6f8074e67f6ce7860631cf6`), run exactly as:

```
python3 day32-compare.py --card rtx5090 --served-ctx 65536 --model-ctx 262144 --registry-value 32768 --survey context \
  O1:rtx5090-day32/boots/O1-off O1:rtx5090-day32/boots/O1-on2048 O1:rtx5090-day32/boots/O1-on8192 O1:rtx5090-day32/boots/O1-on32768 \
  O2:rtx5090-day32/boots/O2-off O2:rtx5090-day32/boots/O2-on2048 O2:rtx5090-day32/boots/O2-on8192 O2:rtx5090-day32/boots/O2-on32768
python3 day32-compare.py --card pro6000 --served-ctx 262144 --model-ctx 262144 --registry-value 32768 --survey context \
  O1:pro-single-day32/box/boots/O1-off ... O2:pro-single-day32/box/boots/O2-on32768   (the same eight boot names)
```

writing `rtx5090-day32/SUMMARY.txt` and `pro-single-day32/box/SUMMARY.txt`. `day31-faults.py` lists every panic,
respawn, FATAL and engine-error line per boot into `FAULTS.txt`; it is a lister, not a verdict.

The reader was exercised before any day-32 boot on the day-31 receipts only (`day32-reader-selfcheck-day31.txt`, 140
lines). It closes the three gaps of DAY31.md 2.2 on that data: V-CRASH fails all 16 day-31 boots, the `iv-b-r2` and
`iv-a-r1` gap rows land in `length_twin_failed`, the six dead bursts fail V-RETRY, and both cards read
`DAY32 V-DOOR ... v_crash_all=False ... v_trunc_band_all=False ... v_retry_all=False -> FAIL`.

### 1.6 Verdict rules (fixed now, never relaxed)

Changes against DAY31.md 1.6 are marked NEW or CHANGED; everything else is unchanged.

- **V-BOOT.** As DAY31: the boot's `[admit-mem] door=` line reads `door=ON open_output_tokens=v` for `onv` and
  `door=OFF` for `off` (the parser's `DAY31 V-BOOT` line, reprinted as `DAY32 V-BOOT`). A failing boot is excluded
  from every other reading and named.
- **V-CRASH** (NEW, per boot). The boot fails iff its `server.log` (or `server.log.gz`) has any line matching
  `panicked`, `[worker] PANIC`, `[worker] FATAL`, `[worker] respawn`, `argmax sentinel` or `spec verify refused`. The
  line prints the count per pattern and the first match as `file:line`, quoted up to the first em dash. A V-CRASH
  failure does not exclude the boot's rows from the other readings; it fails V-DOOR and makes the arm inadmissible for
  R1 (an OFF boot, for R2 and G-NATURAL).
- **V-ALLOC.** As DAY31, with the ON charge in the code's form (DAY31.md 2.2): booked ctx equals `P + max_tokens + 64`
  for warmup and (ii), `min(P + v + 8, model_ctx) + 64` for an open request under `onv`, and `served + 64` (or
  `P + served + 64` when `P + 16 > served`) for an open request under `off`.
- **Twin.** As DAY31: the rows with the same tag in an ON boot and the OFF boot of the same order; comparable iff both
  returned 200 with equal `prompt_sha256`.
- **V-ID** (CHANGED band). Eligible twins: warmup and (ii) always; an open twin whose OFF side finished `stop` with
  `G_off < v`. PASS for an ON arm in an order iff eligible > 0 and every eligible twin has equal `message_sha256`, G
  and `finish_reason`. Twins with OFF `stop` and `G_off = v` are the boundary, reported as
  `boundary_equal`/`boundary_differ` (message and G only) and not judged: a budget of exactly v can end `length` on the
  token where OFF ends `stop`. Day 31's slack band `v < G_off <= v + 8` no longer exists, because the budget is v.
- **V-TRUNC** (CHANGED, per ON arm and order). Every ON open sequential row with status 200 and `finish_reason =
  "length"` lands in exactly one term:
  - `truncated_with_twin`: comparable twin, OFF `stop`;
  - `length_both`: comparable twin, OFF `length`;
  - `length_prompt_differs`: both 200, prompts differ (a (iii) whose (i) parent was cut);
  - `length_twin_failed`: the OFF twin is missing or not 200 (the DAY31.md 2.2 gap; never skipped);
  - `length_twin_other_finish`: the OFF twin is 200 with another finish or none.
  `length_without_twin` = `length_prompt_differs + length_twin_failed` (day 31's name) is printed too.
  `conserved` is judged: the five terms must sum to `on_length_rows`. `deadline_cut` and `open_non200` (tag and
  status) are reported. `G_outside_band_v` is judged: every ON open `length` row, sequential or burst, must have
  `G = v` exactly.
- **G-NATURAL.** As DAY31, over the OFF boots that passed V-BOOT and V-CRASH, with finish counts per class.
- **V-CONC** (reading, no pass mark). As DAY31.
- **V-RETRY** (CHANGED). PASS iff at least one burst request got a status (no vacuous pass), and under `off` zero
  429s, and under `onv` every 429 carries `Retry-After` in 1..=60 and the number of 429s equals the number of
  `verdict=refuse` lines in that burst window.
- **V-DOOR** (CHANGED, per card). PASS iff no boot is excluded, every V-CRASH passes, every V-ID passes in both
  orders, every V-ALLOC passes, every V-TRUNC is conserved with no row outside `G = v`, and every V-RETRY passes.

### 1.7 Selection rules: what each reads, and how a rule with no admissible observation prints

Stated per card, never applied to a default. This lane does not pick the value.

- **R1, zero truncation.** Reads the V-TRUNC terms. A swept v is admissible iff, in both orders, the OFF boot passed
  V-BOOT, and the ON boot at v passed V-BOOT and V-CRASH, and every open sequential row of that ON boot returned 200.
  R1 is the smallest admissible v with `truncated_with_twin + length_prompt_differs + length_twin_failed +
  length_twin_other_finish = 0` summed over both orders. `length_both` does not count: that request also ran to the
  cap under OFF, so the bound cut no natural stop. An OFF twin that failed counts against v, conservatively. Prints
  `R1_smallest_zero_truncation=<v or none> admissible=[...] inadmissible=['v:O1:crash+O2:open_non200=n', ...]
  truncating=['v:n', ...]`. It prints `none` when no admissible v has zero truncation, including when no v is
  admissible, and the two lists say which case it is.
- **R2, natural cover.** Reads G of every OFF open row that finished `stop`, from the OFF boots that passed V-BOOT and
  V-CRASH. R2 is the smallest swept v at or above the largest such G. Prints `<v> (max_natural_G=g)`;
  `none of [2048, 8192, 32768] (max_natural_G=g)` when g is above 32768; `none (no admissible OFF boot)` when neither
  OFF boot is admissible.
- **R3, registry agreement.** Reads no boot: 32768, the `default_output_length` on every chat row of the private
  deployment registries (DAY31.md 1.7 R3). Prints `R3_registry=32768`.
- **R4, surveyed convention.** Reads no boot: `context`, from `OPEN-OUTPUT-SURVEY.md` (DAY31.md 1.7 R4). It selects
  `off` for the output bound; the charge question is separate. Prints `R4_survey=context`.

Expectations from day 31's OFF rows, stated so that a surprise is visible; they are not rules. On the 9B the largest
natural stop was 6405 and the runaways (`iii-L2-r3`, `iv-b-r0..r2`) ran to the cap, so R2 = 8192, and R1 = 8192 if
both ON 8192 boots are crash-free and those runaways end `length` under OFF. On the 27B the long class stopped at
18,443 to 193,178 under OFF, so R1 and R2 read none. Section 2 reports every departure from these as observed.

### 1.8 Exclusions and failures

- A non-200 outside the burst, a crash, or a boot that never becomes ready is quoted from its captured stderr,
  never inferred: V-CRASH's first line plus the fault lister's lines. A boot with no captured cause is "died, cause
  unknown, repro needed", and nothing is built on it.
- A rerun happens only as a whole order, under a new cell name (`RUN`), with the reason recorded in section 2. No boot
  is dropped silently. An expired idle wait leaves its order not run; the order is then launched again whole.
- A deadline cut is its own category and is excluded from V-ID and from the truncation terms.

### 1.9 Not measured here

D4 (b) and (c) are not repeated: both are decided at admission (DAY31-D4.md), which memra#659 did not change. As
DAY31.md 1.9: no latency or throughput claim, no qualification. The 5090 is not a release gate for this door and this
cell is not a battery.

### 1.10 Pre-fix against post-fix

Section 2 states each difference from DAY31.md 2.4 to 2.10 as observed: per boot, the crash lines; per value, the
truncation terms, the burst and the selection readings. The two programs differ by 50 main commits, not only by
memra#659, so a difference is attributed to the tree, never to one commit.

## 2. Results

Written after the runs. Section 1 is unchanged.
