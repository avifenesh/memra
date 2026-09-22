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
