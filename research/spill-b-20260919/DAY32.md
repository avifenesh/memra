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
Every number below is read from the committed receipts named next to it, and every quote is the captured server log.

### 2.1 Commits and timeline (UTC)

| commit | time | what |
|---|---|---|
| `e4328912f` | 06:20:49 | origin/main `d544c6b82` merged into the lane (the `docs/FLAGS.md` door row resolved as the union of main's memra#659 sentence and the lane's decide-by 2026-10-07) |
| `51c113659` | 06:32:14 (pushed before 06:33) | section 1, the reader, the reader's self-check on day 31, both chains, both build records, before any boot |
| (first boot) | 06:33:04 | local `O1-off` start (`rtx5090-day32/order.log` line 2) |
| `9cba34e73`, `d083eb32c` | 06:51:37, 06:51:46 | origin/main `711be12c3` (integ50, comment-only engine changes) merged; `FAULTS.txt -whitespace` added to both day-32 receipt dirs. No driver, reader or binary changed |
| (BOX3 start) | 07:46:07 | the lead's ordering: BOX3 waited for lane E's `LANE-E-PRO-DONE` (seen 07:46:05, `pro-single-day32/start.log`), then its chain took the card on its own idle check |
| `589e15dc1` | 08:50:44 | local O1 receipts |
| `b9863f836` | 11:04:10 | local O2 receipts, local `SUMMARY.txt` and `FAULTS.txt` |
| `3f8b0c3ab` | 11:05:02 | BOX3 O1 receipts |
| this commit | after 13:40:42 | BOX3 O2 receipts, box `SUMMARY.txt` and `FAULTS.txt`, this section, `STATE.md`, the INDEX row |

Local chain 06:33:04 to 10:57:13 (O1 06:33:04 to 08:42:45, O2 08:43:15 to 10:57:13). BOX3 chain 07:46:07 to 13:40:42 (O1
to 10:43:25, O2 to 13:40:42). Each collector's lock proof is rc=0 (`LOCK-O1.json`, `LOCK-O2.json`), and no compute
app was on either card before a hold or after the last boot.

### 2.2 Notes on section 1 (no rule, arm, value or reader changed)

- **1.1 misstated the booked charge, and V-ALLOC fails on it.** 1.1 says "the charge and the allocation are
  unchanged". The allocation is unchanged (`ctx_cap = min(P + v + 8, model_ctx)`), but the `[admission] request cost`
  line that V-ALLOC reads prints `admission_cap = max(ctx_cap, need)` (worker.rs `fn admission_cap`, line 2798 at
  `d544c6b82`), with `need = P + budget + SPEC_SHRINK_SLACK` and `SPEC_SHRINK_SLACK = 64`. On day 31 the budget was
  `v + 8`, so the line read `P + v + 72`, which is the form the day-31 rule and 1.6 carry. On this tree the budget is
  `v`, so the line reads `P + v + 64`. Checked by hand over every open ON row of all 12 ON boots: every mismatch is
  exactly -8, and every row equals `max(min(P + v + 8, model_ctx), P + v + 64)`. Warmup, (ii) and every OFF row match
  1.6. The verdict stands as printed: V-ALLOC FAIL on every ON boot on both cards, so V-DOOR FAIL on both cards, on
  this term alone. What it means for the door: an open ON request books 8 tokens less than 1.6 predicted, which is
  133,632 B on the 9B (16,704 B/token) and 252,416 B on the 27B (31,552 B/token), and still 56 tokens more than it
  allocates.
- Each boot's `source.txt` is the worktree HEAD at that boot's start: `51c113659` (local `O1-off`), `d083eb32c` (local
  O1 ON boots, local `O2-on32768`, all 8 BOX3 boots) and `589e15dc1` (local `O2-on8192`, `O2-on2048`, `O2-off`). The
  commits in between touch research files, docs and comments only. The binaries did not move: every local
  `binary-O*.sha256` is `d92b6cde...484ea7`, every BOX3 one is `5bbe4857...b88a8`, both built from `d544c6b82`.
- The spec-ctx-edge lane's `SPEC-CTX-EDGE-5090-DONE` marker (DAY31.md 2.12) no longer exists: that lane merged as #668
  and its worktree is gone. The lead's day-32 brief said it needs no card, and local O1 found the 5090 idle.

### 2.3 Verdict lines, verbatim

From `rtx5090-day32/SUMMARY.txt` and `pro-single-day32/box/SUMMARY.txt` (the command lines of 1.5). V-BOOT is PASS on
all 16 boots. V-ALLOC is PASS on the 4 OFF boots and FAIL on the 12 ON boots, each with `mismatch` = the open rows
(2.2). The files carry every line.

```
DAY32 V-CRASH card=rtx5090 boot=O1-off arm=off lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O1-on2048 arm=on2048 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O1-on8192 arm=on8192 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O1-on32768 arm=on32768 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O2-off arm=off lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O2-on2048 arm=on2048 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O2-on8192 arm=on8192 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=rtx5090 boot=O2-on32768 arm=on32768 lines=0 by_pattern={} first=none -> PASS
DAY32 V-ALLOC card=rtx5090 boot=O1-on2048 arm=on2048 judged=45 match=15 mismatch=30 i-L0-r0:3551!=3559,i-L0-r1:3551!=3559,i-L0-r2:3554!=3562,i-L0-r3:3554!=3562,i-L0-r4:3553!=3561,i-L1-r0:5208!=5216 -> FAIL
DAY32 V-ID card=rtx5090 order=O1 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O1 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=rtx5090 order=O1 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O1 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=rtx5090 order=O1 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O1 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=rtx5090 order=O2 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O2 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=rtx5090 order=O2 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O2 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=rtx5090 order=O2 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=rtx5090 order=O2 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-RETRY card=rtx5090 order=O1 arm=off burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O1 arm=on2048 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O1 arm=on8192 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O1 arm=on32768 burst_responses=32/32 r429=11 refuse_lines=11 retry_after_in_1_60=True other_non200=2 -> PASS
DAY32 V-RETRY card=rtx5090 order=O2 arm=off burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O2 arm=on2048 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O2 arm=on8192 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=rtx5090 order=O2 arm=on32768 burst_responses=32/32 r429=11 refuse_lines=11 retry_after_in_1_60=True other_non200=2 -> PASS
DAY32 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:66']
DAY32 SELECT card=rtx5090 R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405)
DAY32 SELECT card=rtx5090 R3_registry=32768 R4_survey=context
DAY32 V-DOOR card=rtx5090 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=False v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True -> FAIL

DAY32 V-CRASH card=pro6000 boot=O1-off arm=off lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O1-on2048 arm=on2048 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O1-on8192 arm=on8192 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O1-on32768 arm=on32768 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O2-off arm=off lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O2-on2048 arm=on2048 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O2-on8192 arm=on8192 lines=0 by_pattern={} first=none -> PASS
DAY32 V-CRASH card=pro6000 boot=O2-on32768 arm=on32768 lines=0 by_pattern={} first=none -> PASS
DAY32 V-ALLOC card=pro6000 boot=O1-on2048 arm=on2048 judged=47 match=15 mismatch=32 i-L0-r0:3593!=3601,i-L0-r1:3593!=3601,i-L0-r2:3596!=3604,i-L0-r3:3596!=3604,i-L0-r4:3595!=3603,i-L1-r0:5250!=5258 -> FAIL
DAY32 V-ID card=pro6000 order=O1 arm=on2048 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O1 arm=on2048 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=pro6000 order=O1 arm=on8192 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O1 arm=on8192 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=pro6000 order=O1 arm=on32768 vs=off eligible=47 equal=47 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O1 arm=on32768 on_length_rows=5 truncated_with_twin=4 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=pro6000 order=O2 arm=on2048 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O2 arm=on2048 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=pro6000 order=O2 arm=on8192 vs=off eligible=46 equal=46 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O2 arm=on8192 on_length_rows=6 truncated_with_twin=5 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-ID card=pro6000 order=O2 arm=on32768 vs=off eligible=47 equal=47 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY32 V-TRUNC card=pro6000 order=O2 arm=on32768 on_length_rows=5 truncated_with_twin=4 length_both=1 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY32 V-RETRY card=pro6000 order=O1 arm=off burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O1 arm=on2048 burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O1 arm=on8192 burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O1 arm=on32768 burst_responses=64/64 r429=6 refuse_lines=6 retry_after_in_1_60=True other_non200=46 -> PASS
DAY32 V-RETRY card=pro6000 order=O2 arm=off burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O2 arm=on2048 burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O2 arm=on8192 burst_responses=64/64 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY32 V-RETRY card=pro6000 order=O2 arm=on32768 burst_responses=64/64 r429=6 refuse_lines=6 retry_after_in_1_60=True other_non200=47 -> PASS
DAY32 SELECT card=pro6000 R1_smallest_zero_truncation=none admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:10', '8192:10', '32768:8']
DAY32 SELECT card=pro6000 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768] (max_natural_G=193178)
DAY32 SELECT card=pro6000 R3_registry=32768 R4_survey=context
DAY32 V-DOOR card=pro6000 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=False v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True -> FAIL
```

### 2.4 First failure line per failing boot

No boot failed V-CRASH, and the fault lister (`FAULTS.txt`) shows no panic, respawn or FATAL on either card. The
failing verdicts and their first lines:

- V-ALLOC, local: `rtx5090-day32/boots/O1-on2048/server.log` line 46, `[admission] request cost: model="q9" ctx=3551
  path=spec = 16704 B/token x ctx + 519MB prefill-workspace + 103MB fixed = 681MB` (`i-L0-r0`, P=1439; 1.6 expects
  3559). Every other ON boot fails the same way on its first open row.
- V-ALLOC, target: `pro-single-day32/box/boots/O1-on2048/server.log` line 44, `... model="q38" ctx=3593 path=spec =
  31552 B/token x ctx + 710MB prefill-workspace + 308MB fixed = 1131MB` (`i-L0-r0`, P=1481; 1.6 expects 3601).
- Not a verdict failure, but the one engine failure of the cell: every burst 503 is `[engine-error] class=Overloaded
  prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`. Target `O1-on32768/server.log` line 6785 is
  the first of 46, and `O2-on32768/server.log` line 6790 the first of 47. Local `O1-on32768/server.log` line 8608 is
  the first of 2 (and 2 in O2, as on day 31).

### 2.5 Per-value truncation and concurrency, per card

`tw` = `truncated_with_twin`, `lb` = `length_both`, `pd` = `length_prompt_differs`; `length_twin_failed` and
`length_twin_other_finish` are 0 and `open_non200` is 0 on every ON boot. Every ON `length` row, sequential and burst,
has G = v exactly. Concurrency is the V-CONC burst reading.

Local RTX 5090, Qwen3.5-9B, B = 32. Both orders are the same row, except that the 32768 burst finished 17 `stop` and
2 `length` in O1 and 14 and 5 in O2:

| v | V-ID eq/eligible | tw | lb | pd | burst statuses | burst finish | active max | arith |
|---|---|---|---|---|---|---|---|---|
| off | | | | | 32 x 200 | 27 `stop`, 5 `length` at the cap | 11 | 7 |
| 2048 | 16/16 | 18 | 3 | 15 | 32 x 200 | 32 `length` (G 2048) | 32 | 19 |
| 8192 | 48/48 | 0 | 4 | 0 | 32 x 200 | 27 `stop`, 5 `length` (G 8192) | 32 | 17 |
| 32768 | 48/48 | 0 | 4 | 0 | 19 x 200, 11 x 429 (Retry-After 60), 2 x 503 (prefill OOM, 5) | see above | 19 | 11 |

- 2048 cuts every (i) request (15 of 15; the 9B's (i) stops at 2413 to 5048) and all 3 `iv-a`. All 15 (iii) then carry
  a cut parent (`pd`), and the 3 `iv-b` are runaways under OFF too (`lb`).
- 8192 and 32768 cut no request that stops under OFF. The 4 `length` rows per order are `iii-L2-r3` and `iv-b-r0..r2`,
  which under OFF run to the 65,536 cap.

Target card, one RTX PRO 6000 Blackwell, Qwen3.8-27B, B = 64. Both orders are the same row, except where the cell
reads "O1/O2":

| v | V-ID eq/eligible | tw | lb | burst statuses | burst finish | active max | arith |
|---|---|---|---|---|---|---|---|
| off | | | | 64 x 200 | 64 `stop` (G 186 to 2680) | 9 | 7 |
| 2048 | 46/46 | 5 | 1 | 64 x 200 | 61 `stop`, 3 `length` (G 2048) | 64 | 60 |
| 8192 | 46/46 | 5 | 1 | 64 x 200 | 64 `stop` | 64 | 51 |
| 32768 | 47/47 | 4 | 1 | 12/11 x 200, 6 x 429 (Retry-After 60), 46/47 x 503 (prefill OOM, Retry-After 5) | 12/11 `stop` | 12/11 | 32 |

- 2048 and 8192 cut 5 requests per order that stop under OFF at 18,443 to 193,178 tokens (`iv-a-r0` 102,441,
  `iv-a-r2` 134,383, `iv-b-r0` 18,443, `iv-b-r1` 58,001, `iv-b-r2` 193,178). 32768 cuts 4: all but `iv-b-r0`. The `lb`
  row is `iv-a-r1`, which under OFF runs to 262,143 of the 262,144 served tokens.
- The 32768 burst, both orders. The door put 58 requests in flight (`inflight=58` on its `[admit-mem]` defer lines),
  deferred 6 and refused those 6 with 429 after `waited_ms` 15,615 to 15,646. 46 (O1) and 47 (O2) of the admitted
  requests then failed their prefill with CUDA OOM and returned 503 with `Retry-After: 5`. No `[admit-mem] reclaim`
  line printed. At the burst's release the 250 ms sampler read `prefix_cache_bytes` 13,091,303,424 and
  `cuda_driver_free_bytes` 65,244,102,656 (O1) and 65,076,330,496 (O2); the smallest driver-free reading inside the
  burst was 14,286,848 in both orders. This is quoted, not diagnosed; the engine is the lead's.

### 2.6 Natural G (door OFF, both orders)

The two OFF boots of each card agree on all 52 non-burst rows (status, G, `message_sha256`).

- 9B: the largest G of a request that stopped is 6405 (`iv-a-r1`), the same as day 31. The runaways end `length`:
  `iii-L2-r3` P+G = 65,536, `iv-b-r0` 65,536, `iv-b-r1` 65,533, `iv-b-r2` 65,534. Burst: 27 `stop`, 5 `length`, G p50
  4073, max 64,206.
- 27B: the largest G of a request that stopped is 193,178 (`iv-b-r2`), the same as day 31. `iv-a-r1` ends `length` at
  G = 260,482 (P+G = 262,143). Burst: 64 `stop`, G 186 to 2680, p50 424.5.

### 2.7 Identity

V-ID PASS on every ON arm in both orders on both cards, `differ=0`, no boundary row. Locally that is 16/16, 48/48 and
48/48 per order; on the target card 46/46, 46/46 and 47/47. Where both sides returned 200 on the same prompt and the
OFF request stopped under v, the door changed no token on the fixed tree.

### 2.8 Selection rules, as printed (not chosen)

- 5090: R1 = 8192 (all three values admissible; 2048 truncates 66 rows over both orders), R2 = 8192 (largest natural
  stop 6405), R3 = 32768, R4 = `context`.
- Target card: R1 = none (all three admissible; they truncate 10, 10 and 8 rows over both orders), R2 = none of
  [2048, 8192, 32768] (largest natural stop 193,178), R3 = 32768, R4 = `context`.

All four match the expectations stated in 1.7. Which rule applies is the owner's call.

### 2.9 Pre-fix against post-fix, as observed

The two programs differ by 50 main commits (1.10); every line below is an observation on the receipts, not an
attribution to one commit.

| reading | day 31 (engine `5f1b0eda4`) | day 32 (main `d544c6b82`) |
|---|---|---|
| crash lines (V-CRASH patterns) | every one of the 16 boots (`day32-reader-selfcheck-day31.txt`) | none on the 16 boots |
| processes | FATAL at 2048 and 8192 locally and at 8192 on the target card, both orders; panic and respawn on both local OFF boots and both target 32768 boots | no panic, respawn or FATAL |
| ON open non-200 | local 34, 1 to 4 and 3 per order at 2048, 8192 and 32768; target 6, 1 to 2 and 3 | 0 on every ON boot |
| ON `length` G | `v + 4` to `v + 8` | `v` exactly |
| OFF long rows at the cap | local `iv-b-r2` 500 at pos 65533; target `iv-a-r1` 500 at pos 262142 | local `iv-b-r2` 200 `length` at P+G 65,534; target `iv-a-r1` 200 `length` at 262,143 |
| booked ctx, open ON request | `P + v + 72` | `P + v + 64` (2.2) |
| local bursts at 2048 and 8192 | no status (process dead) | 32 x 200, 32 in flight |
| local burst at 32768 | 19 x 200, 11 x 429, 2 x 503 | the same |
| local OFF burst, active max | 6 | 11 |
| target bursts at 2048 | 64 x 200, 3 `length` at 2056 | 64 x 200, 3 `length` at 2048 |
| target burst at 8192 | no status (process dead) | 64 x 200, 64 in flight |
| target burst at 32768 | 47 x 200, 17 x 429; entered after a worker respawn, prefix cache 0.9 GB (O1) and 0.4 GB (O2), smallest driver free 2.43 GB | 12/11 x 200, 6 x 429, 46/47 x 503 prefill OOM; prefix cache 13.1 GB, smallest driver free 14 MB |
| target refusals, `waited_ms` | 12,301 to 12,337 | 15,615 to 15,646 (local: 8011 to 8034) |
| target R1 as printed | 2048 (a literal reading; every long request died at the bound) | none |
| local R1 as printed | 8192 (none by 1.6's text, gap rows) | 8192, no gap row |

### 2.10 Owed

| item | why | price |
|---|---|---|
| The target-card 32768 burst: 46 and 47 of 64 admitted requests die in prefill on CUDA OOM (2.5) | the door's bounded defer (c) is meant to answer with a 429 before the card runs out; here it admitted 58 with 13.1 GB of prefix cache resident and no reclaim line | engine, the lead's. A lane-B repro cell on the target card (the ON 32768 boot alone, the prefix-cache and `[admit-mem]` lines kept) is 0.2 agent-day plus about 25 minutes of card time |
| The door decision | decide-by 2026-10-07 (FLAGS.md); the owner decides on these receipts | owner |
| V-ALLOC's arithmetic | 1.6 carried day 31's `P + v + 72` form; the fixed program books `max(ctx_cap, P + budget + 64)` | nothing to rerun; a later cell pre-registers the `need` form |

### 2.11 Cleanup and rig state

- Local: `target/day32/memra-server` deleted after the reads (its sha256 stays in `rtx5090-day32/binary.sha256`); the
  detached build worktree `wt-b32-main` removed; the scratch scripts and chain outputs under `/tmp` deleted. The 5090
  lock was released at 10:57:13.
- Target card: the detached build worktree `/root/wt-b32-main` removed, and the chain output file deleted. The receipt
  root `/root/spill-receipts/b-day32` stays, including `bins/` (binary `5bbe4857...b88a8`). No process of this lane is
  left on either card.
