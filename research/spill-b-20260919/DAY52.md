# WP-B day 52: the `admit-by-memory` default's deciding cell (DAY40 2.1) on the RTX PRO 6000 Blackwell Max-Q

Receipts: `pro-maxq-day52/`, after the runs.

## 1. Pre-registration: the RTX PRO 6000 Blackwell Max-Q (2026-11-04, written before any Max-Q boot)

Written 2026-11-04, before any Max-Q boot. Nothing in section 1 changes after a number is seen.

### 1.1 Question

The `admit-by-memory` default (docs/decisions/ADMIT-BY-MEMORY-DEFAULT.md) arms only on the full-power RTX PRO 6000
Blackwell Workstation Edition, because its deciding cell, DAY40.md 2.1, ran on that card. The owner ruled on
2026-11-04 that the RTX PRO 6000 target is the Max-Q. This cell reruns the deciding cell on one RTX PRO 6000 Blackwell
Max-Q Workstation Edition. A PASS extends the default to the Max-Q (the change listed in the scope entry
`admit-by-memory-maxq-extension`, `extension_change`). A FAIL keeps the default Workstation-only and goes to the
decision record's Rejected list with this receipt.

### 1.2 Program, binary, drivers

- Program: `S40` = `b46ae200e7638f9c55688d2d339de2872f4762a6`, no patches, default features, the arm `tip` of
  `build-arms.sh` (`cargo build --release -p memra-server --bin memra-server`). At S40 the arm is the env door
  (`MEMRA_ADMIT_BY_MEMORY`), which ignores the card. The only edition-keyed code at S40 (tier_transfer.rs:108,
  KV-VMM placement) is reached only under `MEMRA_KV_ALLOCATOR=vmm`, which stays unset. So the Max-Q runs the program
  the Workstation ran.
- Binary: `bins/s40/memra-server`, sha256 `7da1db366fbe34ca40440fc8e818b864c2bc9014db7e7330b08a929084e7cc45`
  (built on the laptop with nvcc 13.1, as the box build was; the box binary was `c3387e1b...b51d20a0`).
- Drivers: the registered checkout `50e8cb345279199f9a0bf10d22c1bfc66e3dae70` plus `portability.patch` (card
  scoping and no git on the box; DEVIATIONS.md). The workload-defining inputs keep their registered sha256:
  `day31-order.sh` a5626dd8, `day31-parse.py` 8f27e7f9, `day34-compare.py` c6013822, `docs/SERVING.md` 022b1d50.
  `run-day26-cell.sh` and `day31-client.py` change only in card scoping and provenance lines.
- Model: `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
  served as `q38` at the checkpoint's context (262,144; `MEMRA_CTX` unset). The runner checks the sha256 before each
  part.

### 1.3 The cell (DAY40.md 1.2, unchanged)

DAY34.md 1.3 to 1.8 as DAY40 ran them on the target card:

- Arms `off`, `on2048`, `on8192`, `on32768`. O1 = `off, on2048, on8192, on32768`, inner order AB. O2 = `on32768,
  on8192, on2048, off`, inner order BA. One boot per arm per order, eight boots. One collector hold per order on
  `/tmp/memra-gpu.lock` (`tools/tier-battery.py --rig pro-single --external-lock`).
- OFF boots run with `MEMRA_ADMIT_BY_MEMORY` and `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` removed (`env -u`). ON boots set
  `MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=v`.
- Every boot: `run-day26-cell.sh` with `MEMRA_COMPAT=openai`, `MEMRA_ADMIT_PREDICT_SHADOW=1`,
  `MEMRA_TIMEOUT_MS_MAX=3600000`, port 18526, N=5, no host tier.
- Workload: `day31-client.py --burst 64`, greedy and non-streaming: warmup; the day-26 mix (i), (ii)
  `max_tokens=96`, (iii) at 5,000/10,000/20,000 characters, 5 reps each; the long class iv-a and iv-b (k = 24, 40,
  72), 3 reps each; 64 open requests released on one barrier. Prompts from `docs/SERVING.md` (sha256 022b1d50).
- Per boot `day31-parse.py`. Across boots, the registered command line:

```
python3 day34-compare.py --card pro6000 --served-ctx 262144 --model-ctx 262144 --registry-value 32768 --survey context \
  O1:boots/O1-off O1:boots/O1-on2048 O1:boots/O1-on8192 O1:boots/O1-on32768 \
  O2:boots/O2-off O2:boots/O2-on2048 O2:boots/O2-on8192 O2:boots/O2-on32768
```

  then `day31-faults.py` on each boot dir and `day40-read.py pro6000 boots`. The `--card pro6000` label stays as
  registered so every output line compares byte-for-byte with the Workstation receipt; the card's identity is
  recorded beside it (1.6).

### 1.4 The rule (quoted from the Workstation pre-registration)

DAY34.md:96: "**V-DOOR (per card)** is PASS iff every DAY32.md 1.6 condition holds and every boot passes V-OOM, G-BOOK
and PARK."

DAY32.md:141-142: "**V-DOOR** (CHANGED, per card). PASS iff no boot is excluded, every V-CRASH passes, every V-ID
passes in both orders, every V-ALLOC passes, every V-TRUNC is conserved with no row outside `G = v`, and every V-RETRY
passes."

The terms, as registered:

- V-BOOT (DAY32.md 1.6): "the boot's `[admit-mem] door=` line reads `door=ON open_output_tokens=v` for `onv` and
  `door=OFF` for `off` ... A failing boot is excluded from every other reading and named."
- V-CRASH (DAY32.md 1.6): "The boot fails iff its `server.log` (or `server.log.gz`) has any line matching `panicked`,
  `[worker] PANIC`, `[worker] FATAL`, `[worker] respawn`, `argmax sentinel` or `spec verify refused`."
- V-ALLOC (DAY34.md 1.6 item 1): "For an open request under `onv`, booked ctx equals the engine's
  `max(ctx_cap, P + budget + 64)` with `ctx_cap = min(P + v + 8, model_ctx)` and `budget = ctx_cap - P - 8` ...
  Warmup, (ii) and every OFF row keep DAY32.md 1.6's form."
- V-ID (DAY32.md 1.6): "Eligible twins: warmup and (ii) always; an open twin whose OFF side finished `stop` with
  `G_off < v`. PASS for an ON arm in an order iff eligible > 0 and every eligible twin has equal `message_sha256`, G
  and `finish_reason`."
- V-TRUNC (DAY32.md 1.6): "`conserved` is judged: the five terms must sum to `on_length_rows`. ... `G_outside_band_v`
  is judged: every ON open `length` row, sequential or burst, must have `G = v` exactly."
- V-RETRY (DAY32.md 1.6): "PASS iff at least one burst request got a status (no vacuous pass), and under `off` zero
  429s, and under `onv` every 429 carries `Retry-After` in 1..=60 and the number of 429s equals the number of
  `verdict=refuse` lines in that burst window."
- V-OOM (DAY34.md 1.6 item 2): "PASS iff the boot's `server.log` has zero `CUDA_ERROR_OUT_OF_MEMORY` lines other than
  a park receipt (`[admit-mem] prefill OOM parked`, `[admit-oom] step OOM parked`), and no row of the boot,
  sequential or burst, has status 503."
- G-BOOK (DAY34.md 1.6 item 3): "For an ON boot: PASS iff there is at least one `[admit-mem] id=... verdict=admit`
  line, and every such line has `est_bytes <= device_free` and a `pending_prime=` field. For an OFF boot: PASS iff the
  log has zero `[admit-mem] id=` lines."
- PARK (DAY34.md 1.6 item 4): "PASS iff no park fired, or every row of the boot ended 200 or 429 and every 429 carries
  `Retry-After` in 1..=60."

Not pass terms: V-CONC (DAY32.md 1.6: "reading, no pass mark"), G-NATURAL, SELECT R1 to R4 (DAY34.md 1.7; "Stated per
card, never applied to a default"), the DAY40.md 1.3 readings. The 32768 open-output value is SELECT R3, which reads no
boot, and stays.

### 1.5 The verdict line

The reader prints one line, `VERDICT.txt` line 1, in the Workstation reader's form. The Workstation receipt read
(SUMMARY.txt:94):

```
DAY34 V-DOOR card=pro6000 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS
```

The Max-Q PASSES iff its line ends `-> PASS`. Any other ending, or no line, is not a PASS. The kit's readers
reproduce the Workstation receipt's SUMMARY.txt, READINGS.txt and FAULTS-local.txt byte-for-byte under the box's Python
3.14.4 (`selfcheck/laptop-dryrun.txt`; on the box: `bash run.sh selfcheck`).

### 1.6 Max-Q specifics

- Card: one RTX PRO 6000 Blackwell Max-Q Workstation Edition, chosen at run time (`CARD`, the nvidia-smi index; the
  window being built for is GPU 2). The runner refuses any other name, a card with a compute app or 2000 MiB or more
  in use, a held `/tmp/memra-gpu.lock`, a busy port, or a wrong model or binary hash.
- Power: the box's limits read 250/250/280/280 W on indices 0 to 3 (max 325/300/300/325 W) on 2026-11-04; they have
  not been changed and the cell changes no limit. Each order records `name,power.limit,memory.total` in
  `card-O1.csv`/`card-O2.csv` (the registered `card.csv` columns) and index, UUID, bus id and limits in
  `card-*-detail.csv`; the 250 ms collector telemetry (that card only) records `power.limit` per sample.
- Regime: beside training on the other three cards, in gaps between another session's eval batches on this card.
  This is an untimed pass/fail cell: no term reads a time, a rate or a neighbour. Wall times and V-CONC `window_ms`
  are printed and not read (DAY32.md 1.9: "no latency or throughput claim"). There is no timed clause, so there is no
  `run-timed.sh`.
- Parts: O1 runs in one window (part 1), O2 in a later one (part 2), both in the same `out-<stamp>/`, same binary and
  model (checked). Each order keeps its one collector hold. V-ID, V-TRUNC and V-RETRY are judged within an order, so
  no term compares across the two windows. The readers run once, over all eight boots, after part 2.

### 1.7 Added reading (no bound)

`MAXQ-VS-WS.txt` (`maxq-vs-ws.py`): per boot, the Max-Q's sequential rows against the same boot's rows on the
Workstation receipt (equal `message_sha256`, completion tokens and finish), and both cards' burst status counts. It
answers the scope's open question (are greedy outputs identical across the two editions). No term reads it.

### 1.8 Expected readings (not rules; stated so a surprise is visible)

From the Workstation receipt on the same program and workload: every term PASS on all eight boots and `V-DOOR ... ->
PASS`; SELECT R1 none, R2 none of [2048, 8192, 32768] (max_natural_G 193,178), R3 32768, R4 context; bursts 64 x 200
at `off`, `on2048`, `on8192` and 46 x 200 plus 18 x 429 at `on32768` (the split may move with the card's free memory;
no term reads it); cross-card identity equal on every compared row if sm_120 greedy outputs do not depend on the
edition.

### 1.9 Failures

DAY40.md 1.5: "Causes are quoted from captured stderr. A rerun happens only as a whole order under a new name, with
the reason in section 2." An order the window cuts (collector timeout), a signal, or a boot refused for landing off
the card leaves its part INCOMPLETE; the part reruns whole in a later window (part 1 in a new out dir, part 2 after
the runner moves the earlier O2 attempt to `failed-O2-<stamp>/`). A crash or a boot that never becomes ready inside a
completed order is data: the readers judge it.

**Amendment, 2026-11-05, before the first boot: the power limit.**

Owner ruling, 2026-11-05: the cell may raise its card's power limit to 300 W, must note thermal throttling, and
must restore the limit after use. This replaces the statement above that the limits have not been changed. The kit
is unchanged: it still only records the limit it reads.

- Before the cell boots, the runner, which sits outside the kit, sets the card under test to 300 W, the card's
  default limit (`nvidia-smi -i <card> -pl 300`), and reads it back, so the receipts record 300 W. When the cell
  ends, on any path, the runner restores the limit it read before (250 W on index 0 and 280 W on index 2 at kit
  time) and reads it back. The slot guard and the gap scheduler also restore it when they stop.
- Throttle telemetry: every 10 s during the cell the runner records temperature, fan, power draw, power limit, SM
  clock and four clock-event reasons (SW power cap, SW thermal slowdown, HW thermal slowdown, HW power brake) in
  `thermal-<cell>.csv`. It prints one summary line with the maximum temperature and fan and the number of samples
  with each reason active. SW power cap is the normal state of a card running at its limit; it is recorded and not
  judged.
- Reading: this is an untimed pass/fail cell, so the throttle summary is recorded with the receipt and not judged.
- Neighbour rule, asked by the box's training session: this cell runs on GPU 2 beside training on GPUs 0, 1 and 3,
  so every 10 s the runner also reads GPUs 1 and 3. If either reaches 90 C, GPU 2 returns to its previous limit (280
  W) for the rest of the cell, and the summary line records when. The verdict does not depend on power, so the drop
  changes no clause.
- The runner applies the Max-Q thermal rule on its 10 s samples: it stops the cell at >= 95 C, or at >= 92 C
  with the fan at >= 95 % on 3 consecutive samples, and a stopped cell has no verdict.
