# Session C day 35: the tenant-stall cell on the RTX 5090 class (the day-16 five arms on the 9B)

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`, start `1a8a8a508` = remote, clean. Every push today in the
announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR, no engine change, no `docs/` registry edit, no recommendation. Today's
card is the LOCAL RTX 5090 Laptop GPU with the 9B artifact (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`), lock
`/tmp/memra-5090.lock`; every number here is this card's own and is compared to nothing measured on the target card.

## Merges (first action)

- `a38f3a0f5`: `origin/main` at `f3fba2952` (#648, the `MEMRA_KV_PARK_COMPACT` row) merged `--no-ff`. One conflict,
  `research/INDEX.md`, resolved by the clone's recorded resolution (rerere) as a union; checked by set-difference against
  both parents (`comm -23` under `LC_ALL=C`): no line of either parent missing, no duplicate row, no marker.
- `091a931c0`: `origin/lane/spill-integ43-20260922` at `2cb0f6c00` (PR #649: the integ43 record, the battery, smoke and
  5090 hit-gate receipts) merged `--no-ff`, clean (no conflict, so no side of `HOSTPREFIX-DOOR.md` or `research/INDEX.md`
  had to be chosen; the integ side is the tree). `check-conflict-markers: OK`. Pushed (`1a8a8a508..091a931c0`).

## Build (local, this worktree's `target/`)

`cargo build --release -p memra-server` under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`
(`day35-cpu/build-local.log`, `rc=0`, 2m53s, tree `091a931c0`, `MEMRA_CUDA_ARCH auto-detected 120a`); `memra-server`
SHA-256 `7ce7bf78b96ad5b936cb3a31db36dfc4fe668e45ef03008dcfa2bcd5005e8ded`. The tree's engine files moved since the
last local build (#646 touched `crates/memra-server/src/health.rs` and `lib.rs`), so the day-33 binary was not reused.

## Task 1, pre-registration: the tenant-stall cell on this card (committed before the cell runs)

**What is owed.** `HOSTPREFIX-DOOR.md` section D item 5's tail ("no tenant-stall cell exists on this class") and the
packet's item 7 and section 6 (a tenant-stall cell on the RTX 5090 class named as missing in both the naked-default
and the longer-door bullets). Every stall cell so far ran on the target card (A day 16 and its successors, C days 23,
28, 29, 30); the RTX 5090 class has the pair cell (day 31), the hash cells (days 18 and 33) and the D2D price (day 28),
no tenant.

**The shape: A day 16's, adapted only in what this card and model need.** A's `OWNER-THREAD-CENSUS.md` cell as run on
day 16 (`spill-a-20260919/pro-single-day16/stall-cell.sh`, harness `stall_cell.py`): a streaming TENANT decodes one
token per tick (`MEMRA_SERVE_SPEC=0`) while an INTRUDER request runs one owner-thread class, fired when the tenant's
24th token arrives; the tenant's client-side inter-token gaps are the observation; the idle control is the tenant
alone; order 1 = (idle, arm) x 5 then order 2 = (arm, idle) x 5 inside every harness run, N=5 per arm per order, N=10
pooled; the quantity is the harness's rule, `stall = the run's worst ITL minus its p50`, `stall_median` over the 10
arm runs; the rule line is fixed in the harness and `--replay` recomputes it from every `receipt.json`
(`STALL REPLAY: PASS` is the admissibility of each receipt). The five arms are day 16's: prime only (a cache-off boot),
demote OFF, demote ON, promote OFF, promote ON.

What changes for this card and model, and why, stated before the run:

1. **The artifact and budgets.** The 9B; `MEMRA_PREFIX_CACHE_MB=64` and `MEMRA_KV_HOST_MB=8192` (day 31's budgets on
   this card: one 64-token entry fits the 67,108,864 B budget, two do not, so every second insert evicts and demotes).
   On the `MEMRA_SERVE_SPEC=0` boot the 9B's 64-token entry is the plain class: `54.6MB`, `16 items` (day 31's plain
   whole-budget arm printed `[prefix-host] skip demote: entry 54.6MB > host budget 1MB` after `demote submitted off
   the tick: 64 tokens, 54.6MB, ticket seq=3, 16 items` and `demote published off the tick ... 24.9ms from submission
   to completion`). Boot otherwise day 16's: `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_COMPAT=openai`,
   the prime boot `MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0`, the ON boots `MEMRA_KV_HOST_CONTRACTS=1`. Port 18131
   (the gates' 18119 left to the lead's battery).
2. **The tenant's window.** This card's tick is shorter than the target card's, so day 16's 160-token tenant would not
   hold the 5120-token prime: the board's plain decode row for the 9B on this rig reads `137.3` tok/s
   (`docs/PERFORMANCE.md` PERF-PLAIN, tg128 at 512-token context), about **7.3 ms per tick**; a 160-token tenant fired at
   token 24 leaves 136 ticks, about **0.99 s**, and the 5122-token prime is estimated at **about 1.3 s** on this card
   (A's day-16 prime-cancel gate on this card primed 1280 rows in 0.32 s at 256-row chunks, about 4000 rows/s;
   `spill-a-20260919/rtx5090-day16/pcg.log`), so the prime would return after the tenant's last token in some or all
   runs. The tenant's `max_tokens` is therefore **400** (`day35_stall_cell.py --tenant-max-tokens 400`; lane A's
   `stall_cell.py` with that ONE added argument, default 160, everything else byte-for-byte A's, the diff banked beside
   the receipts as `harness.diff`): 376 ticks after the 24th token, about **2.7 s**, twice the prime's estimate. The
   intruder shapes are NOT changed: the prime intruder stays `PRIME_TARGET_TOKENS - 4` words (5120 to 5123 tokens as the
   tokenizer makes them, one tokenizer family with the 27B; A day 16), primed in `PREFILL_TICK_T = 1024` chunks while
   the tenant is unfinished (five chunks, expected five stretched ticks of about 250 ms each); the demote intruder
   stays the fresh 72-word prompt (95 to 99 tokens; its 64-token grid seed evicts the previous entry into the host
   tier); the promote intruder stays the P_A / P_B alternation (86 and 89 tokens; a host hit of 64 with a 22- to
   25-token suffix whose insert evicts the other entry, so the promote window holds an inline demote). Expected
   landings inside the window, from day 31's server lines on this card: the demote arm's intruder returns about 100 to
   150 ms after firing (a 97-row prime plus a demote whose `in` read 4.8 to 24.1 OFF and 38.5 to 60.4 ON on the spec
   entries), the promote arm's about 100 ms (promote `in` 8.6 to 27.7 OFF, 20.1 to 42.4 ON, plus the inline demote).
   If the tenant emits EOS before 400 tokens the window is what it emitted (`tenant_tokens` is in every receipt); the
   per-run landing rule below decides, and nothing is re-sized after a result.
3. **One hold, two passes in opposite order (the day-28 form).** Six boots in ONE collector lock hold on
   `/tmp/memra-5090.lock` (`tools/tier-battery.py --rig rtx5090 --external-lock`, the collector's 250 ms sampler into
   `command.gpu.csv`, plus the cell's own 1 s `nvidia-smi` sampler into `ev/card.during.csv`; `card.{before,after}.csv`
   and `compute-apps.{before,after}.csv` around the hold, `card.{before,after}.csv` per boot): pass 1 = prime, off, on;
   pass 2 = on, off, prime. The prime boot runs `--mode prime`; each off and on boot runs `--mode demote` then `--mode
   promote` (day 16 and day 23's order inside a boot). Every receipt replayed. Bounded wait before the hold
   (`day35-local-run.sh`): no compute app and at least 20000 MiB free, a refused lock retried, 15 waits of 120 s in
   total (30 minutes); if that ends busy the cell is recorded NOT RUN with the card's last snapshot
   (`stall/NOT-RUN.txt`) and nothing on the card is inspected beyond `nvidia-smi`'s listing or signalled.

**What the cell reads (fixed now; `day35-stall-reading.py`, dry-tested on A's day-16 receipts through a scratch
symlink layout where it reproduced the day-16 rule figures `117.5 / 193.5` demote and `85.0 / 162.8` promote and
refused the day-16 ON receipts on the off-tick line census, which that tree's on-tick door could not print).**

- Per pass, per class (demote, promote): `stall_median(arm)` (the rule-line figure), IQR = p75 - p25 of the 10 per-run
  stalls; `on_minus_off` per pass with `unc` in quadrature of the two IQRs; `isolated` when `|on_minus_off| > unc`,
  else `under_resolution` (the day-28/29 form). The prime arm per pass: `stall_median` and IQR alone. Pooled over the
  two passes (N=20 per arm) printed as context only.
- Admissibility per receipt, all of: `STALL REPLAY: PASS`; `errors=0`; the tenant's text identical across the 20 runs;
  EVERY arm run's intruder returned inside the tenant's window (`fired_at_ms + wall_ms <= tenant_wall_ms`, all from the
  receipt); demote receipts with at least 9 `server_demote_ms` (the first insert of a boot has nothing to evict),
  promote receipts with exactly 10 `server_promote_ms`; the boot's `server.log` free of `demote failed|promote
  failed|demote refused|promote refused|restore refused|capture refused|latched off|TIER DISABLED`.
- What each arm's server lines must show. Prime boot: no `[prefix-host]` line at all (cache off, host tier off). OFF
  arms: `[prefix-host] demote: 64 tokens, 54.6MB in Y ms` and `[prefix-host] promote: 64 tokens, ... in Y ms` only, no
  `off the tick` line. ON demote arm, every run with a demote: `demote submitted off the tick: 64 tokens, 54.6MB,
  ticket seq=N, 16 items on the contracts door's copy stream`, `contracts door D2H receipt: ... items=16 ... require=ok`,
  `demote published off the tick: ticket seq=N complete after P poll(s), X ms from submission to completion (tick-top
  poll)`. ON promote arm, every run exactly: one `promote submitted off the tick: ... request parked`, one `contracts
  door H2D receipt: ... require=ok`, one `promote published off the tick`, one `request parked`, one `[prefix-cache]
  restore not routed (contracts door): the entry was promoted for this admission (insertion pin id=P, 64 tokens, model
  gate); the tick program copies it` (A day 26's promoted-pin refusal on the promote-then-hit shape, on `main` since
  #647), zero `restore submitted off the tick`; plus the inline demote's three lines. Zero refused or latched lines in
  every boot.
- The day-27 attribution read against this card, from the ON demote runs: `demote_in` (the `in Y ms` figure),
  `completion` (the `X ms from submission to completion` figure), `in - completion` per demote and its median; beside
  it the OFF demote's `in` median, the tenant's two largest gaps per arm run and their sum, and `on_minus_off` of the
  demote class. Context on the same host, never a rule: day 33's heap pass at the entry size read `54.8MB ... heap
  11.9` ms (N=10; `rtx5090-day33/hashwc/reading.log`), day 31's `in_minus_completion median 21.6 (N=12)` on the spec
  entries (`rtx5090-day33` reading of the day-31 receipts). Whether `in - completion` on the plain entries sits at the
  heap pass, and whether the demote class's `on_minus_off` sits at `in - completion`, is read from the numbers, not
  assumed. The ON promote arm's `promote_in`, its `completion` and the inline demote's `in` are printed beside it.
- Nothing is compared to the target card; no threshold is tuned; an arm that cannot be made admissible on this card
  (an intruder outside the window, a tenant that stops early, a missing door line) is reported with its numbers and
  decides nothing.

**Expected time.** About 3 s per tenant run at 400 tokens, 20 runs per harness mode, three harness modes per pass
plus six boots of the page-cached 9B: 12 to 16 minutes in the hold.

## Task 2, the run (receipts `rtx5090-day35/stall/`; reading `rtx5090-day35/reading.log`)

**The hold.** `day35-local-run.sh` waited 7 x 120 s for the card (another session's `kernel-check`, `qwen-a4-continuation-gate`,
`graph-warmup-stress` and three `memra-server` boots, listed in `stall/waits.log`, none touched), then ran the six boots in
ONE collector hold on `/tmp/memra-5090.lock` (`stall/collector/lock.json` `acquired: true, owner: collector`, `ev/LOCK.json`
the inherited flock; `CELL.jsonl` `status: executed-not-qualified`), 16:18:19Z to 16:29:16Z, `stall-day35 rc=0`, 11 of 11
receipts `STALL REPLAY: PASS (replay agrees with the harness's rule line)` (`ev/replays.log`). Binary `7ce7bf78b9...` (the
build above; `[server] build: memra-0.138.0-813fc8cfe4de (id: source-tree, git: 091a931c023a)`), tree `091a931c0`, the 9B.
Regime: the collector's 250 ms CSV 2603 samples, 58 to 89 C, 30.64 to 175.33 W, `power.limit [N/A]`; the cell's 1 s CSV
655 samples, 58 to 89 C, 31.39 to 175.24 W; P0 at 58 C before and P0 at 77 C after; no compute app listed before or after.
Per boot (the 1 s CSV): pass-1 prime 58 to 84 C, pass-1 off 75 to 87 C, pass-1 on 77 to 88 C, pass-2 on 76 to 87 C, pass-2 off
78 to 87 C, pass-2 prime 78 to 89 C; power 31 to 175 W in every boot. The tenant emitted 400 tokens in all 120 runs, text
sha `5d59f3ddef257cfb` in every run of every boot (`tenant_text_identical=True` x 11); its idle p50 7.3 to 7.5 ms, idle p99
8.0 to 9.0 ms; the intruder prompts tokenized as on the target card (prime 5120 to 5123, demote 95 to 99, promote 86 and
89); every demote and promote intruder returned inside the tenant's window (`landed=10/10` in all eight arms; demote
intruders 104 to 177 ms after firing, promote 63 to 127 ms). The ON boots' server lines, whole-boot census
(`ev/pass{1,2}/on/server.log`, identical counts in both): `demote submitted off the tick` 21, `D2H receipt ... require=ok`
21, `demote published off the tick` 21, `[prefix-host] demote:` 21 (the 9 demote-arm evictions, the promote arm's 2 seeds
and its 10 inline demotes); `promote submitted off the tick ... request parked` 10, `H2D receipt ... require=ok` 10,
`promote published off the tick` 10, `[prefix-host] promote:` 10, `request parked` 10, `restore not routed (contracts
door)` 10, `restore submitted off the tick` 0, `capture submitted off the tick (seed)` 12; every `require=` line `ok` (43 of
43); zero `failed`, `refused`, `latched off` or `TIER DISABLED` lines. The OFF boots: 21 `[prefix-host] demote:` and 10
`[prefix-host] promote:` lines, no `off the tick` line. The prime boots: no `[prefix-host]` line. The entries are the plain
class as pre-registered: `64 tokens, 54.6MB, ... 16 items` demoted, `53.6MB, 16 items` promoted, `items=16 (8 KV planes)` in
every receipt line; the promote-then-hit shape reads, per run, `promote submitted ... request parked`, `H2D receipt
require=ok`, the inline `demote submitted`, `promote published off the tick: ticket complete after 1 poll(s), 19.7ms from
submission to completion`, `[prefix-host] promote: 64 tokens, 53.6MB in 45.6ms`, `[prefix-cache] restore not routed
(contracts door): the entry was promoted for this admission (insertion pin id=12, 64 tokens, model gate); the tick program
copies it`, `[prefix-cache] hit: 64 of 89 prompt tokens from cache`, the inline demote's `D2H receipt require=ok`, `demote
published off the tick ... 76.6ms from submission to completion`, `[prefix-host] demote: 64 tokens, 54.6MB in 98.5ms`
(pass-1 run 2, `ev/pass1/on/promote/receipt.json` `server_log_lines`).

**A co-tenant during pass 1, stated.** The card-wide `memory.used` peaked at 23,230 MiB in the pass-1 prime boot, 20,286 in
pass-1 off and 20,318 in pass-1 on, against 9,753, 7,641 and 7,673 MiB in the identical pass-2 boots (`ev/card.during.csv`,
per boot window from `ev/marks.tsv`): about 12.6 to 13.5 GB of the card that was not this cell's server sat on it through
pass 1 and left at 16:22:57Z (the CSV's 18,708 to 6,265 MiB step, during pass-1 on's promote mode). The runner's pre-check
saw no compute app at 16:18:17Z and the after-hold listing is empty; the cell samples the card, not processes, so the
co-tenant is not identified here. It did not hold `/tmp/memra-5090.lock` (this cell held it). Its one measured effect: the
server's memory admission refused 7 of the 10 pass-1 prime intruders, `[admit-oom] capacity reject: model="gate" ctx=5186
does not fit an IDLE box (available 1794MB), HTTP 400 context_length_exceeded` after `[admit-oom] VRAM defer: 1 active,
effective free 1683MB (driver + 892MB pool-cached) < cost 1683MB + reserve 1611MB, queueing (FIFO) [pool res 9328MB used
8436MB ...]` (2633 defer lines, 7 device-trims `reason=admission-drain`, 7 rejects; the harness recorded each as
`RuntimeError('HTTP 400: ... request context 5186 does not fit this model's available KV capacity on an idle server ...')`);
the pass-2 prime boot printed zero defers and zero rejects for the same ten requests. Pass 1's demote and promote figures
were taken beside that co-tenant and are reported as such; pass 2 is the clean pass. (The server's own log dash in the
`admit-oom` lines is quoted here with a comma.)

**Verdict lines, verbatim (`reading.log`; the rule lines per receipt are in `ev/replays.log`).**

`DAY35 STALL VERDICT: prime pass1 10.5 (iqr 174.2) inadmissible; demote pass1 off 63.7 on 67.0 on-off +3.3 (unc 3.3)
under_resolution; promote pass1 off 47.7 on 49.3 on-off +1.6 (unc 12.5) under_resolution; prime pass2 278.4 (iqr 6.6)
admissible; demote pass2 off 63.0 on 63.2 on-off +0.3 (unc 2.7) under_resolution; promote pass2 off 47.5 on 50.1 on-off
+2.6 (unc 2.7) under_resolution; admissible=False`

(`admissible=False` is the pass-1 prime arm alone: `pass1/prime/prime: replay=PASS admissible=False (errors=7; intruder
outside the tenant's window in 7 of 10 runs)`; the eight demote and promote receipts and the pass-2 prime read
`admissible=True (errors=0, tenant text identical, every intruder inside the window, the server lines as pre-registered)`.)

Per arm, N=5 per order, both orders, N=10 pooled, each pass one boot per arm, regime as above:

| Arm | Pass 1 (beside the co-tenant; 58 to 88 C) | Pass 2 (clean; 76 to 89 C) | Pooled N=20 (context) |
|---|---|---|---|
| Prime only | INADMISSIBLE: 3 of 10 intruders admitted (`intruder_prompt_tokens=[5123, 5122, 5123, None x7]`, `errors=7`); the three admitted read stalls `230.3, 226.9, 230.5` with five stretched ticks each (204.5 to 238.5 ms, sums 1085 to 1095 against intruder walls 1115 to 1129); the seven refused read 6.5 to 48.3 (no prime ran) | `stall_median=278.4 stall_min=261.9 stall_max=285.3` (IQR 6.6), o1 `[274.9, 275.2, 284.2, 282.7, 285.3]`, o2 `[280.2, 279.1, 261.9, 276.1, 277.7]`; `arm_p99=247.7 arm_max=292.6 idle_p99=8.0`; five stretched ticks per run, 220.9 to 292.6 ms, sums 1270 to 1358 against intruder walls 1303 to 1391 (5120 to 5123 tokens in `PREFILL_TICK_T = 1024` chunks, about 3.8k rows/s on this card in this regime) | not pooled (pass 1 inadmissible) |
| Demote OFF | `stall_median=63.7` (IQR 2.0), o1 `[40.3, 55.9, 62.9, 62.2, 64.5]` (run 2 is the boot's first insert, no demote), o2 `[63.7, 64.8, 64.4, 64.3, 63.8]`; `server_demote_ms=[18.3, 25.3, 24.4, 26.6, 25.6, 26.9, 26.5, 26.4, 25.7]` | `stall_median=63.0` (IQR 2.3), o1 `[39.8, 54.2, 63.9, 63.5, 62.4]`, o2 `[64.0, 63.9, 63.5, 62.4, 61.2]`; `server_demote_ms=[17.1, 26.4, 26.0, 25.1, 25.3, 25.2, 25.6, 24.7, 23.2]` | 63.5 (IQR 2.0) |
| Demote ON | `stall_median=67.0` (IQR 2.6), o1 `[38.4, 62.6, 65.9, 69.9, 66.2]`, o2 `[68.4, 67.1, 68.6, 66.9, 68.5]`; `server_demote_ms=[59.1, 65.7, 70.2, 66.2, 68.2, 70.0, 67.7, 67.1, 67.4]` | `stall_median=63.2` (IQR 1.5), o1 `[38.0, 61.4, 62.0, 63.8, 62.9]`, o2 `[65.0, 64.7, 63.6, 63.0, 63.5]`; `server_demote_ms=[54.6, 61.1, 64.4, 63.2, 64.9, 64.2, 62.9, 61.8, 62.3]` | 64.2 (IQR 4.1) |
| Demote ON minus OFF | `on_minus_off=+3.3 unc=3.3 -> under_resolution` | `on_minus_off=+0.3 unc=2.7 -> under_resolution` | |
| Promote OFF | `stall_median=47.7` (IQR 0.9), o1 `[66.3, 63.5, 46.4, 48.4, 47.7]` (the first two are the boot's first-touch promotes), o2 `[47.9, 47.7, 47.5, 47.2, 47.3]`; `server_promote_ms=[31.6, 30.5, 10.3, 9.9, 9.9, 9.7, 9.7, 10.1, 10.0, 9.8]`, inline `server_demote_ms=[27.1, 26.1, 5.9, 5.5, 5.6, 5.5, 5.4, 5.7, 5.6, 5.4]` | `stall_median=47.5` (IQR 2.4), o1 `[65.2, 63.0, 48.3, 45.8, 47.0]`, o2 `[46.1, 45.6, 46.4, 48.7, 47.9]`; `server_promote_ms=[28.6, 27.2, 10.2, 8.2, 9.6, 8.6, 8.2, 8.8, 8.4, 10.0]`, inline `[24.4, 23.6, 6.6, 4.7, 5.2, 4.8, 4.7, 5.1, 4.6, 5.6]` | 47.7 (IQR 1.6) |
| Promote ON | `stall_median=49.3` (IQR 12.5: the first-touch pair plus run 6 read 65.1 to 67.6), o1 `[67.6, 65.1, 67.1, 49.3, 50.7]`, o2 `[49.1, 49.0, 49.2, 48.9, 49.0]`; `server_promote_ms=[45.6, 41.6, 25.2, 25.0, 25.4, 25.6, 25.5, 26.2, 25.2, 25.7]`, inline `server_demote_ms=[98.5, 98.2, 98.2, 80.5, 82.2, 80.0, 83.0, 80.3, 79.8, 78.3]` | `stall_median=50.1` (IQR 1.3), o1 `[67.5, 67.7, 49.4, 48.7, 49.1]`, o2 `[50.3, 49.9, 50.3, 50.8, 49.4]`; `server_promote_ms=[43.5, 45.0, 24.7, 25.8, 25.3, 25.5, 25.0, 26.1, 25.3, 25.6]`, inline `[97.3, 98.5, 80.3, 79.5, 80.1, 81.2, 83.1, 81.2, 81.9, 81.4]` | 49.6 (IQR 5.3) |
| Promote ON minus OFF | `on_minus_off=+1.6 unc=12.5 -> under_resolution` | `on_minus_off=+2.6 unc=2.7 -> under_resolution` | |

Read, not tuned: on this card the harness's rule (the tenant's WORST tick minus its p50) does not separate the door's arms
in either class in either pass; the door's demote and promote lines are all present and every receipt reads `require=ok`.
The prime arm on this shape is admissible only when the memory admission admits the 5187-context request beside the
tenant, which it did in 10 of 10 pass-2 runs and 3 of 10 pass-1 runs; a shorter prime or a different `MEMRA_CTX` would be a
new pre-registration, not today's.

## Task 3, the day-27 attribution read against this card (`reading.log` `DAY35 ATTRIBUTION` lines; the post-hoc gap
description in `rtx5090-day35/gaps-posthoc.log`, `day35-gaps-posthoc.py`, written after the run and labelled so)

`DAY35 ATTRIBUTION pass=1 ON demote: demote_in median=67.4 (N=9) completion median=44.0 (N=9) in_minus_completion
median=23.8 min=21.7 max=26.2 (N=9); OFF demote_in median=25.7 (N=9)`; `DAY35 ATTRIBUTION pass=2 ON demote: demote_in
median=62.9 (N=9) completion median=41.0 (N=9) in_minus_completion median=21.6 min=20.6 max=23.4 (N=9); OFF demote_in
median=25.2 (N=9)`; `DAY35 ATTRIBUTION pass=1 ON promote: promote_in median=25.6 (N=10) completion median=20.1 (N=10)
inline demote_in median=81.3 (N=10)` (pass 2 identical to the decimal).

- **The bundle pass on this host, at the plain 54.6 MB entry.** The ON demote's `in - completion` reads 21.6 (pass 2, the
  clean pass; 20.6 to 23.4) and 23.8 (pass 1), the figure day 31 read on the spec entries (`in_minus_completion median
  21.6 (N=12)`). Day 33's heap pass at 54.8 MB on this host read 11.9 ms (N=10), so `in - completion` on this card is about
  1.8 to 2.0 heap passes of the entry, not one: the segment holds the bundle checksum plus whatever else sits between the
  poll's completion stamp and the `[prefix-host] demote:` line (the two receipt hashes over the KV planes, the publish,
  the insert); this cell does not split it further (the 9B entry's KV byte split stays the unmeasured term, item 7).
- **Where it lands in the tenant's stall (post-hoc description, `DAY35 GAPS` lines).** Both demote arms stretch TWO of the
  tenant's ticks per intruder. OFF: `top1_median=71.2 top2_median=41.3` (pass 1), `70.3 / 41.9` (pass 2); ON: `74.5 / 70.9`
  (pass 1), `70.5 / 69.4` (pass 2). The worst tick is the same in both arms (the intruder's 97-row prime, its grid seed and
  its insert; the OFF demote's synchronous 25 ms D2H sits inside it), so `on_minus_off` of the worst tick is +3.3 / +0.3.
  The door's cost lands on the SECOND stretched tick, the tick whose top polls the ticket, hashes and publishes: +29.6
  (pass 1) and +27.5 (pass 2) over OFF's second tick, and the stretched-tick SUM per demote reads `146.9` against `112.8`
  (pass 1, +34.1) and `140.0` against `112.3` (pass 2, +27.7). The `in - completion` segment (23.8 / 21.6) sits inside that
  second-tick delta; the remaining 4 to 10 ms of the delta is not attributed here. On this card the rule's quantity
  therefore under-reads the door's demote share by the whole second tick; the sum of stretched ticks reads it. Stated as a
  property of the metric on this shape, not as a re-reading of the verdict lines, which stand as printed.
- **The promote class, the same shape.** OFF stretches ONE tick per hit (`top1_median=55.2 / 54.8`, `top2_median=9.9 / 9.8`:
  the hit's admission restore, the 22- to 25-token suffix prime, the insert and the inline synchronous demote, all in one
  tick); ON stretches TWO (`56.7 / 57.4` and `40.1 / 40.0`): the promote is submitted and the request parked, the next tick
  top publishes it (`19.7 to 20.3ms from submission to completion`) and the re-admission takes the not-routed device copy
  and the suffix prime, while the inline demote's bundle pass (`in_minus_completion_median=21.9 / 21.8` on the inline
  demotes, N=10) lands on the same or the following tick top. The worst tick again reads the same in both arms
  (`on_minus_off=+1.6 / +2.6`, under resolution); the second stretched tick of about 40 ms is the door's promote-window
  cost on this card, read in the sum (`97.9` against `65.3`, `97.6` against `64.6`: +32.6 / +33.0 per hit) and not in the
  rule. `promote_in` 25.6 against OFF's 9.8 to 10.3 steady; the inline demote's `in` 81.3 against OFF's 5.5 (it spans the
  park).
- Nothing here is compared to the target card; A day 27's 74.8 and day 26's 81.8 / 85.3 are that card's figures and stay
  in their rows.

## 4. Records and checks

`DAY35.md` (this file), `STATE.md`, `research/INDEX.md` row `spill-c-20260919/day35`, `HOSTPREFIX-DOOR.md` (the RTX 5090
cost table beside section B, section D item 5's tail, section E's DAY 35 paragraph), `DOOR-DECISION-PACKET.md` (status
header, section 4's RTX 5090 table, item 7, section 6). Checks at close are listed in the push section. Boundaries: no
engine change, no `docs/` registry edit, no V4.1 code, no external dependency, no `--no-verify`, no bare GPU run (every
boot under the collector's hold), no touch of other lanes' worktrees or processes (the co-tenant of pass 1 and the gates
that held the card before and after were listed by `nvidia-smi` and left alone), no third lock name, no cross-card
comparison, no median without its N and regime, nothing tuned after a result (the post-hoc gap description is labelled
and decides nothing), no recommendation.

Checks at close: `shellcheck` clean on `day35-stall-cell.sh` and `day35-local-run.sh`; `tools/check-flags.sh` (`no uncovered
runtime names`); `tools/check-conflict-markers.sh` OK; `git diff --check` clean (the receipt dir's `.gitattributes` marks
`*.log` and the banked `harness.diff` `-whitespace`, the day-28 form); zero em dashes in every line added today; no provider
host, id, price or location in the added lines; `python3 tools/check-public-boundary.py check`: `599 matches (599
grandfathered, 0 new)`. Scratch: the reader's `/tmp/spill-c-day35-readertest` symlink layout and the merge's
`/tmp/spill-c-idx-{ours,theirs}.md` removed at close. Budget: about 1.3 agent-hours against 4 (14 minutes of that the
bounded wait for the card, 11 the hold).

## Push section

- `091a931c0` (the two merges), `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification
  claimed`, `pre-push: skip recorded`.
- `da9b22b63`: the pre-registration (cell 1: this file's Task 1, the harness copy, the cell, the reader, the runner, the build
  log), the same two hook lines.
- `20b4ee655`: the run (cell 2: `rtx5090-day35/`, Tasks 2 and 3 of this file, the post-hoc gap script), the same two hook lines.
- The records tip: this section, `STATE.md`, the INDEX row, the door doc and the packet; its SHA is the commit that carries this
  line.
