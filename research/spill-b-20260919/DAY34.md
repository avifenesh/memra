# WP-B day 34: `MEMRA_ADMIT_BY_MEMORY` OFF against ON, the clean rerun the owner decides on

The door's decide-by is 2026-10-07 (`docs/FLAGS.md`). Day 32 ran the cell on the memra#659 tree and read V-DOOR FAIL
on its own V-ALLOC arithmetic, and it found memra#680 (the door admitted a burst past its own estimate). Day 33 fixed
memra#680 under the door. This day reruns the day-32 cell on a tree that carries both fixes: the same arms, bound
values, workloads, orders and cards, and every day-32 term as registered, with four changes stated below. Every cell
is `executed-not-qualified`; no qualification is claimed, no default moves, and no open-output value is picked.

## 1. Pre-registration

Committed and pushed before the first boot of this day. Nothing in section 1 changes after a number is seen.

### 1.1 The program under test

The lane tip `9f335ac48`: main `0afd88e1d` (which carries #668, the memra#659 fix) plus the memra#680 fix `30a5ab697`;
`git diff 0afd88e1d 9f335ac48 -- crates` touches `crates/memra-server/src/admit_memory.rs` and `worker.rs` only.
Under the door on this tree:

- An open request is charged and allocated `ctx_cap = min(P + v + 8, model_ctx)` with output budget `v`
  (DAY32.md 1.1), and the `[admission] request cost` line books `max(ctx_cap, P + budget + 64)`, which is
  `P + v + 64` here (DAY32.md 2.2).
- Every headroom reading of the admission block is reduced by `pending_prime`, the prefill workspace the
  still-priming sessions owe. Every admission prints `[admit-mem] id=... verdict=admit ... device_free=<booked>
  pending_prime=<bytes> ...`. A prefill CUDA OOM on a session that has emitted nothing parks and requeues and prints
  `[admit-mem] prefill OOM parked session back to queue ...` (DAY33.md 1.2).
- Door OFF is day 32's program, which day 33's V-OFF confirmed row for row on the 5090.

### 1.2 Rigs, artifacts, binaries

The cards and models are day 32's (DAY32.md 1.2): the local RTX 5090 on Qwen3.5-9B NVFP4 MTP (sha256
`52c9cceb...8f39de`, `MEMRA_CTX=65536`, burst 32, `/tmp/memra-5090.lock`), and BOX3, one RTX PRO 6000 Blackwell, on
Qwen3.8-27B NVFP4-Q5K MTP (sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`, context unset,
burst 64, `/tmp/memra-gpu.lock`).

One binary per card, built once from `9f335ac48` in a detached worktree and copied once. Local: the build line is in
`rtx5090-day34/build-line.txt`, the log in `build-local.log`, and the sha256 is
`1ed5471bd2ab88611ab9f8aec916181ef406193c2b21ed59659064ecaea6e1f0`. The target-card binary is built on the restored box by its chain (1.9) with the same source commit.

The environment is every boot's day-31/32 environment through the unchanged drivers. Their sha256 match DAY32.md 1.3:
`day31-order.sh` `a5626dd8...0e350d1`, `run-day26-cell.sh` `fad91c06...ddd2ee66`, `day31-client.py` `53aa2512...51fda2`,
`day31-parse.py` `8f27e7f9...c0e09`, prompts `docs/SERVING.md` `022b1d50...887cff`.

### 1.3 Arms and orders

Identical to DAY32.md 1.3: `off`, `on2048`, `on8192`, `on32768`; O1 = `off, on2048, on8192, on32768` with inner
order AB, O2 = the reverse with inner order BA; one boot per arm per order; one collector hold per order. Chains:
`rtx5090-day34/chain.sh` (day 32's local chain with the day-34 paths: bounded idle wait of at most 14400 s before each
order, lock free, no compute app, at least 24 GB of host memory) and `pro-single-day34/chain.sh` (1.9). A hold never
starts inside another lane's hold: the collector takes the lock only after the idle check found it free, and it waits
on the lock rather than sharing it.

### 1.4 Workload

DAY31.md 1.4, unchanged (`day31-client.py`, the full workload including class iv and the burst).

### 1.5 Readings

Per boot, `day31-parse.py`. Across boots, `day34-compare.py` (sha256
`c6013822f78e06bc3d4172aedd3e1608f2912e2dfc15673559b9ecdcdeaea5e9`), run exactly as DAY32.md 1.5 with the file name
changed:

```
python3 day34-compare.py --card rtx5090 --served-ctx 65536 --model-ctx 262144 --registry-value 32768 --survey context \
  O1:rtx5090-day34/boots/O1-off O1:rtx5090-day34/boots/O1-on2048 O1:rtx5090-day34/boots/O1-on8192 O1:rtx5090-day34/boots/O1-on32768 \
  O2:rtx5090-day34/boots/O2-off O2:rtx5090-day34/boots/O2-on2048 O2:rtx5090-day34/boots/O2-on8192 O2:rtx5090-day34/boots/O2-on32768
python3 day34-compare.py --card pro6000 --served-ctx 262144 --model-ctx 262144 --registry-value 32768 --survey context \
  O1:pro-single-day34/box/boots/O1-off ... O2:pro-single-day34/box/boots/O2-on32768   (the same eight boot names)
```

writing each card's `SUMMARY.txt`; `day31-faults.py` writes `FAULTS.txt`. The reader was checked before any day-34
boot on the day-32 receipts of both cards (`rtx5090-day34/reader-selfcheck-day32-rtx5090.txt` and
`reader-selfcheck-day32-pro6000.txt`). With the new formula V-ALLOC reads PASS on all 16 day-32 boots, V-OOM reads FAIL
on the four 32768 boots (2, 2, 46 and 47 OOM lines), G-BOOK reads FAIL on every ON boot (no admit line before
memra#680), and both cards read `V-DOOR ... -> FAIL`.

### 1.6 Verdict rules

Every term of DAY32.md 1.6 as registered (V-BOOT, V-CRASH, V-ALLOC, Twin, V-ID with `G_off < v` and the boundary,
V-TRUNC with G = v exactly and the conserved terms, G-NATURAL, V-CONC, V-RETRY with at least one burst response,
V-DOOR), with these four changes:

1. **V-ALLOC (CHANGED formula).** For an open request under `onv`, booked ctx equals the engine's
   `max(ctx_cap, P + budget + 64)` with `ctx_cap = min(P + v + 8, model_ctx)` and `budget = ctx_cap - P - 8`, which
   is `P + v + 64` when unclamped. Warmup, (ii) and every OFF row keep DAY32.md 1.6's form (for them the same
   expression reduces to it).
2. **V-OOM (NEW, per boot, judged).** PASS iff the boot's `server.log` has zero `CUDA_ERROR_OUT_OF_MEMORY` lines other
   than a park receipt (`[admit-mem] prefill OOM parked`, `[admit-oom] step OOM parked`), and no row of the boot,
   sequential or burst, has status 503.
3. **G-BOOK (NEW, per boot, judged).** For an ON boot: PASS iff there is at least one `[admit-mem] id=...
   verdict=admit` line, and every such line has `est_bytes <= device_free` and a `pending_prime=` field. For an OFF
   boot: PASS iff the log has zero `[admit-mem] id=` lines. `pending_prime_max` is reported.
4. **PARK (NEW, per boot, judged).** The prefill-OOM and step-OOM park receipts are counted and reported. Parks alone
   are not a failure. PASS iff no park fired, or every row of the boot ended 200 or 429 and every 429 carries
   `Retry-After` in 1..=60.

**V-DOOR (per card)** is PASS iff every DAY32.md 1.6 condition holds and every boot passes V-OOM, G-BOOK and PARK.

### 1.7 Selection rules

DAY32.md 1.7 unchanged, with the same inputs and the same print forms: R1 over admissible values (both orders' boots
V-BOOT and V-CRASH, every open sequential ON row 200), `none` with its lists; R2 from admissible OFF boots, `none of
[...]` or `none (no admissible OFF boot)`; R3 = 32768 and R4 = `context`, read from no boot.

### 1.8 Expected readings (not rules; stated so a surprise is visible)

Local RTX 5090, from day 32 on the same card and workload and day 33's green boots:

- V-BOOT, V-CRASH, V-ALLOC, V-ID, V-TRUNC (band and conservation), V-RETRY, G-BOOK and PARK PASS on all 8 boots, with
  no park firing.
- V-OOM PASS on all 8. Day 32 had 2 prefill OOMs per order at 32768; day 33's green R32 had none.
- V-TRUNC per order: at 2048, `truncated_with_twin` 18, `length_both` 3, `length_prompt_differs` 15; at 8192 and
  32768, `length_both` 4 and the other terms 0.
- Bursts: at 2048, 32 x 200, all `length` at G 2048. At 8192, 32 x 200 on day 32; with the booking some arrivals may
  now defer or refuse. At 32768, day 32's 2 x 503 become typed 429s: day 33's green R32 read 19 x 200 and 13 x 429.
- SELECT: R1 = 8192, R2 = 8192 (largest natural stop 6405), R3 = 32768, R4 = `context`.
- V-DOOR PASS.

Target card, from day 32 and the memra#680 diagnosis:

- V-OOM PASS: day 32's 46 and 47 prefill-OOM 503s at 32768 become typed 429s or served requests. Every other term as
  on day 32, whose failing verdict was V-ALLOC alone.
- SELECT: R1 = none (the long class stops at 18,443 to 193,178), R2 = none of [2048, 8192, 32768]
  (max_natural_G 193,178), R3 = 32768, R4 = `context`.

### 1.9 The target-card cells, pre-registered now for BOX3's restore

BOX3 is gone. Its old address is never contacted again, and the chain starts only when the lead names the restored
host. `pro-single-day34/chain.sh` runs on the box and refuses to start unless `/root/wt-b` fast-forwards to the lane
tip, the model's sha256 is `1facf36c...1e024a`, and the card is idle. Every binary is built on the box in a detached
worktree.

- **Part A, memra#680's owed boots (DAY33.md 1.3, unchanged):** `red-R64`, `red-off`, `green-R64`, `green-off`, with
  red = main `c3eb41d12` and green = the fix `30a5ab697`; workload `day33-client.py --chars 5000 --skip-long --burst
  64` (ON, 32768) or `--burst 0` (OFF). Read with `day33-compare.py --card pro6000 red:R64:... red:off:...
  green:R64:... green:off:...` under DAY33.md 1.6: R-OOM RED expected on `red-R64`; G-NOOM, G-BOOK, V-ID-FIX, V-ID and
  V-OFF judged. Receipts `pro-single-day34/box/p680/`.
- **Part B, the day-34 cell:** both orders on the `9f335ac48` binary, burst 64, read as in 1.5.

Part A runs first (about 1 h including two builds), then Part B (about 6 h; day 32's OFF boots on the 27B took about
2 h 17 min each).

### 1.10 Failures

Causes are quoted from captured stderr, never inferred. A rerun happens only as a whole order under a new name, with
the reason in section 2.

## 2. Results

Written after the runs. Section 1 is unchanged.
Every number below is read from the committed receipts named next to it.

### 2.1 Commits and timeline (UTC, 2026-09-23)

| commit | time | what |
|---|---|---|
| `c9d677eec` | 16:11:39 (pushed before 16:12) | section 1, `day34-compare.py`, both chains, the local build record, the reader self-checks |
| (first boot) | 16:12:01 | local O1 collector hold, `O1-off` start (`rtx5090-day34/order.log` line 2) |
| `7a0c73aa3` | 18:25:23 | local O1 receipts |
| `164777767`, `809c16444` | 20:37:58, 20:48:48 | day 35's pre-registration and fix (DAY35.md); research and engine commits after the local O1 and part A runs, while O2 ran on the prebuilt binary |
| this commit | after 20:55:55 | local O2 receipts, `SUMMARY.txt`, `FAULTS.txt`, part A's mirror and reading, this section |

Local: O1 16:12:01 to 18:21:43 and O2 18:40:47 to 20:55:55 (the lead's integ53 battery held the card between the
orders). Every boot ran binary `1ed5471b...a6e1f0` from `9f335ac48`.

### 2.2 Notes on section 1 (no rule, arm, value or reader changed)

- **Part B is not run, by the lead's order** (2026-09-23): part A showed that the tree under test still has a known
  OOM term, and the owner's decision cell does not run on it. The owner's cell moves to the day-35 tree (DAY35.md
  1.7). The 5090 half below is banked as a record of `9f335ac48`, not as the decision cell.
- **The target card is BOX4, not BOX3.** BOX3's instance was lost (DAY33.md 2.2). Part A ran on BOX4, one RTX PRO 6000
  Blackwell Workstation Edition at 600 W, where BOX3 was a Server Edition. The model sha256 is the same
  `1facf36c...1e024a`. Nothing on BOX4 is compared against BOX3's receipts; red and green pair within BOX4 only.
- BOX4 setup: `/root/wt-b` is a blobless clone of the lane branch at `7a0c73aa3`. The chain ran with `PARTS=A` once
  `/root/setup/model.sha256` read the expected hash (20:03:45, `pro-single-day34/start.log`).
  `LANE-B-680-DONE` was appended to the box chain log at 20:21:58.

### 2.3 Local RTX 5090 verdict lines, verbatim (`rtx5090-day34/SUMMARY.txt`)

V-BOOT, V-CRASH, V-OOM, G-BOOK, PARK and V-ALLOC are PASS on all 8 boots; their per-boot lines are in the file.

```
== V-ID / V-TRUNC
DAY34 V-ID card=rtx5090 order=O1 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O1 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O1 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O1 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on2048 vs=off eligible=16 equal=16 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on2048 on_length_rows=36 truncated_with_twin=18 length_both=3 length_prompt_differs=15 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=15 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on8192 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on8192 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
DAY34 V-ID card=rtx5090 order=O2 arm=on32768 vs=off eligible=48 equal=48 differ=0 boundary_equal=0 boundary_differ=0 -> PASS
DAY34 V-TRUNC card=rtx5090 order=O2 arm=on32768 on_length_rows=4 truncated_with_twin=0 length_both=4 length_prompt_differs=0 length_twin_failed=0 length_twin_other_finish=0 length_without_twin=0 conserved=True deadline_cut=0 open_non200=0 G_outside_band_v=0
== V-CONC / V-RETRY
DAY34 V-RETRY card=rtx5090 order=O1 arm=off burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O1 arm=on2048 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O1 arm=on8192 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O1 arm=on32768 burst_responses=32/32 r429=13 refuse_lines=13 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O2 arm=off burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O2 arm=on2048 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O2 arm=on8192 burst_responses=32/32 r429=0 refuse_lines=0 retry_after_in_1_60=True other_non200=0 -> PASS
DAY34 V-RETRY card=rtx5090 order=O2 arm=on32768 burst_responses=32/32 r429=13 refuse_lines=13 retry_after_in_1_60=True other_non200=0 -> PASS
== SELECT (stated, not chosen)
DAY34 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 admissible=[2048, 8192, 32768] inadmissible=[] truncating=['2048:66']
DAY34 SELECT card=rtx5090 R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405)
DAY34 SELECT card=rtx5090 R3_registry=32768 R4_survey=context
DAY34 V-DOOR card=rtx5090 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS
```

### 2.4 Truncation and concurrency, 5090 (both orders are the same row except where noted)

| v | V-ID | tw | lb | pd | burst | active max | arith | booked `pending_prime` max |
|---|---|---|---|---|---|---|---|---|
| off | | | | | 32 x 200 (27 `stop`, 5 `length` at the cap) | 11 | 7 | (no admit line) |
| 2048 | 16/16 | 18 | 3 | 15 | 32 x 200, all `length` at 2048 | 32 | 19 | 10.19 GB |
| 8192 | 48/48 | 0 | 4 | 0 | 32 x 200 (27 `stop`, 5 `length` at 8192) | 32 | 17 | 8.75 GB |
| 32768 | 48/48 | 0 | 4 | 0 | 19 x 200, 13 x 429 (Retry-After 60) | 19 | 11 | 5.39 GB |

`tw` = `truncated_with_twin`, `lb` = `length_both`, `pd` = `length_prompt_differs`; the other terms and `open_non200`
are 0 on every ON boot. At 32768 day 32's 2 x 503 prefill OOMs per order are now 2 more typed 429s (13 against 11).
No park fired and no OOM line exists in any boot. SELECT: R1 = 8192, R2 = 8192 (largest natural stop 6405),
R3 = 32768, R4 = `context`, as expected in 1.8.

### 2.5 Target card part A, memra#680's owed boots on BOX4 (`pro-single-day34/box/p680/SUMMARY.txt`)

```
== R-OOM / G-NOOM / G-BOOK (ON boots)
DAY33 R-OOM card=pro6000 boot=red-R64 role=red shape=R64 v=32768 oom_lines=47 burst={429: 6, 503: 46, 200: 12} first_oom=server.log:588 1790194490213 [prefix-cache] snapshot failed (DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")); prefix not cached -> RED
DAY33 G-NOOM card=pro6000 boot=red-R64 role=red shape=R64 oom_lines=47 burst_503=46 crash_lines=0 burst_200=12 other_non200=46 r429=6 refuse_lines=6 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=red-R64 role=red shape=R64 admit_lines=0 admit_lines_in_burst=0 est_over_booked_free=0 -> FAIL
DAY33 R-OOM card=pro6000 boot=green-R64 role=green shape=R64 v=32768 oom_lines=3 burst={429: 15, 200: 49} first_oom=server.log:746 1790194701526 [admit-mem] prefill OOM parked session back to queue (model q38, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") -> RED
DAY33 G-NOOM card=pro6000 boot=green-R64 role=green shape=R64 oom_lines=3 burst_503=0 crash_lines=0 burst_200=49 other_non200=0 r429=15 refuse_lines=15 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=green-R64 role=green shape=R64 admit_lines=68 admit_lines_in_burst=52 est_over_booked_free=0 -> PASS
== V-ID-FIX (green ON against red ON, same shape, sequential rows)
DAY33 V-ID-FIX card=pro6000 shape=R64 green=green-R64 red=red-R64 eligible=16 equal=16 differ=0 -> PASS
== V-ID (the day-32 term: green ON against green OFF)
DAY33 V-ID card=pro6000 shape=R64 on=green-R64 off=green-off eligible=16 equal=16 differ=0 -> PASS
== V-OFF (green OFF against red OFF: the default-OFF program unchanged)
DAY33 V-OFF card=pro6000 green=green-off red=red-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS
DAY33 VERDICT card=pro6000 boots=4 v_boot_all=True green_noom_book_all=False v_id_fix_all=True v_id_all=True v_off_all=True -> NOT-GREEN
```

- red (main `c3eb41d12`, binary `d2e6cff6...`): 12 x 200, 6 x 429, 46 x 503. green (the fix `30a5ab697`,
  `6ce020d2...`): 49 x 200, 15 x 429, no 503.
- green-R64's G-NOOM FAIL is three `[admit-mem] prefill OOM parked session back to queue` receipts (`server.log`
  lines 746 to 748). The park caught all three, and every burst request ended 200 or 429. The log names the term:
  after each prime a prefix seed of about 197 MB lands (`insert (seed)`), and the resident cache grows from 7086.9 MB
  to 8861.0 MB in about 1.4 s before those prefills. Then `step-OOM reclaim ... prefix cache evicted 23 entries
  (4316MB)`. The day-33 fix books the prefill workspace, not the seed. DAY35.md is the follow-up.

### 2.6 Owed

| item | why | price |
|---|---|---|
| The owner's decision cell on the day-35 tree, both cards | DAY35.md 1.7 | its own pre-registration, 0.1 agent-day plus about 4 h local and 6 h on BOX4 |
| memra#680's close | part A read NOT-GREEN on the seed term | DAY35.md |

### 2.7 Cleanup

The local worktree `wt-b34-build` and `target/day34/` are removed at the day's end (the binary hash stays in
`rtx5090-day34/binary.sha256`). BOX4's detached worktrees were removed by the chain; its receipt root and `bins/`
stay.
