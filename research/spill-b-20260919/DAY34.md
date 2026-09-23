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
