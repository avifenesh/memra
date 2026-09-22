# WP-A day 29: option 2a landed under ruling 40: a hit on the `Hashing` entry parks one tick (DAY28 owed item 2a)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start `0a720f978` = remote. Merged `origin/main` `c5879a59e` (after
#649: the lead's integ43) `--no-ff` as `9f2125baa` (the one `HOSTPREFIX-DOOR.md` conflict took the integ side), then
the lead's local integ44 ref `lane/spill-integ44-20260922` `6a42883f9` (C day 35; no remote existed) `--no-ff` as
`0938221a0`, pushed in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (logged; no qualification
claimed). Rig: the target card (BOX3, one RTX PRO 6000 Blackwell, 96 GB, 600 W, `BOX-ACCESS.md`) and the local RTX
5090 rig where its lock allows; no cross-card comparison. Every cell `executed-not-qualified`.

## 1. Pre-registration (this section is committed before any engine code moves)

**Ruling 40 (lead), restated as the contract the code is written to.** Option 2a is approved: a hit whose entry is in
the `Hashing` phase PARKS the request for one tick (the `Promoting` and `Restoring` park pattern: the request goes to
the requeue with a typed line naming the entry and the phase; the tick-top poll lands the hash and publishes; the
request re-admits to a device hit on the published entry). It is not a refusal and not a synchronous settle at the
probe (the owner thread must never hash). Every rule of the existing parked states applies: the parked-only bounded
wait covers a request parked on a `Hashing` entry (extend the guard and its test); the orphan grace applies if the
request leaves; a `Block` settle at the retire seam, purge, trim, shutdown and `host.disable` settles the hash first as
it does the contract; the request never sees two numeric programs (it is served by the device hit either way). Also
approved: the fault gate's demote cells await the boot's last publication before `stop` (a gate fix, stated in the
cell). Until every arm is green the day-28 and day-29 code stays on the lane branch and is not integrated.

### 1.1 What "a hit on the `Hashing` entry" is, and where it is decided

The decision is pure over the host state, `host_hashing_hit(host, pool_key, prompt, device_best_len)`: `host.demoting`
is `Some` AND its `hashing` is `Some` (the copy settled, the ticket retired and acknowledged, the heap payloads on the
helper) AND the pending image's `pool_key` is the request's AND the image's key exactly prefixes the prompt under the
host `lookup` rules (`n >= PREFIX_CACHE_MIN_TOKENS`, `n <= prompt.len()`, `prompt[..n] == toks`) AND `n >
device_best_len` (the promote candidate's own rule: a device hit at least as deep serves without the host). Nothing
else: a `Demoting` entry whose copy is still on the copy stream (the day-17 window, `contract` is `Some`) keeps the
day-17 rule, a hit on it is a cold prime, because ruling 40 names the `Hashing` phase and because on both cards the
copy window (about 54 ms on the target card) closes inside the request that submitted it; the reading below reports
how many hits fell on a copy-phase `Demoting` entry per boot (expected zero) so the lead can rule on it if it is not.

The park is decided in `host_promote_park_probe`, the admission probe, immediately BEFORE `host_promote_probe_decision`
(so a `Submit` of a published candidate cannot `Block`-settle the `Hashing` entry when the request's own prompt hits
it): a hit on the `Hashing` entry returns `true` (parked) even when a published host entry also prefixes the prompt;
on re-admission the lookup picks the longest published entry, which is at least as deep, so the parked request loses
nothing and the probe never waits on the helper for a hit of the entry's own prompt. A request whose prompt does NOT
hit the `Hashing` entry is unchanged: its published candidate's `Submit` meets the `Hashing` entry through the day-28
`Block` settle (`a promote`), ruling 39's approved arm.

### 1.2 The census: every path that meets a request parked on a `Hashing` entry

1. **Probe** (`host_promote_park_probe`): the hit is decided as in 1.1; the request id is recorded on the
   `PendingHashing` (`parked: Vec<String>`; the first park of an id prints the typed line, a re-park counts silently in
   `reparks`, the `ParkAgain` shape); the probe returns `true`. No hash, no wait, no CUDA call.
2. **Requeue**: the caller's existing arm, `requeue.push_back(req); parked_on_promote += 1; continue;` (FIFO, never
   shed). No new arm: the probe's `true` is the same `true` a promote park returns.
3. **Tick-top poll**: unchanged, `host_demote_settle_pending(Poll, "the tick top")` before admission; when the digests
   land the entry publishes into the host index in the same tick top, so the re-admission below sees it.
4. **Re-admission**: the request comes off the queue in the same tick's admission pass; the probe finds the published
   entry through `host_lookup` (no `Hashing` entry: the park predicate is false), decides `Submit`, submits the H2D and
   parks the request on the `Promoting` entry (day 18); the next tick top publishes into the device index and the
   request re-admits to the unmodified device hit path. The request is served by the tick program on a device hit,
   exactly as a request that arrived one tick later would be: one numeric program, by construction (the parked request
   generated no token before the park). If the hash latched (`hash-helper-gone`, `hash-never-lands`, a mismatched
   reply), the tier is off (`armed()` false), the probe declines at its first line and the request primes cold, the
   plain tick program, still one program.
5. **The parked-only bounded wait**: the guard gains `|| hpx.demoting.as_ref().is_some_and(|d| d.hashing.is_some())`
   beside the not-ready `Promoting` and `Restoring` arms, so a queue that is only requests parked on the `Hashing` entry
   waits the bounded 2 ms on the command channel instead of spinning the owner thread. Its test
   (`the_run_loop_waits_boundedly_when_the_queue_is_only_requests_parked_on_a_promote`) gains the assertion; the day-28
   census test's clause "the parked-only wait never reads the demote state" is inverted to this guard.
6. **Orphan grace**: a request parked on a `Hashing` entry owns no state (the demote is the tier's own; the `parked`
   ids are a count). If the request leaves, the queue sweep drops it typed (`client disconnected while queued`) and the
   entry publishes for the next request; there is nothing for the grace to expire, and the census asserts the orphan
   grace (`host_restore_expire_ready`) still reads no demote state.
7. **Retire seam** (a second demote from the eviction sink or the admission reclaim ladder), **purge** (`purge_tenant`),
   **shutdown** (`host_demote_drain_at_shutdown`) and **`host.disable`**: unchanged from day 28: each meets the
   `Hashing` phase through the one driver before the contract step (the `Block` waits), the purge then purges the
   published entry, shutdown drops it typed and joins the helper, the latch joins the helper. In every case the parked
   request re-admits to whatever the tick top left: a device hit after a promote, or a cold prime.
8. **Trims**: gain no arm (a `Hashing` entry holds no device bytes in flight; a parked request holds nothing).
9. **The count**: the publication's ledger line gains `; H hit(s) parked on the Hashing entry (R re-park(s))`; the
   latch arms print one line after the typed refusal when `H > 0`: `[prefix-host] H request(s) parked on the Hashing
   entry re-admit to a cold prime (the tier latched off)` (worded outside the fault gate's refusal matcher).

### 1.3 The typed lines

- Park (once per request id per `Hashing` entry): `[prefix-host] hit parked on a Hashing entry: request <id> (<P>
  tokens) hits the Demoting entry's <n> tokens (ticket seq=<S>, <N> payloads, <M>MB on the hash helper for <X>ms);
  the request waits one tick for the digests (model <m><ns>)`.
- Ledger (day 28's line, extended): `... owner in-completion I ms; wall W ms t0 to publication; H hit(s) parked on the
  Hashing entry (R re-park(s))`.
- Latch tail (only when `H > 0`): `[prefix-host] H request(s) parked on the Hashing entry re-admit to a cold prime (the
  tier latched off)`.

### 1.4 The gate fix (the fault gate's demote cells)

`tools/kv-host-contract-fault-gate.sh` `cell()` (the `presubmit` and `postpublish` cells): before `stop`, wait, bounded
15 s at 100 ms, until every `handed to the hash helper` hand-off in the boot has resolved (a `demote digests landed off
the tick` line or a `demote failed (tier hash` line), and record the wait as its own check, `the boot's last hand-off
landed before stop (bounded 15 s wait)`. Reason, stated in the cell: r3 is the boot's last request, its eviction's
demote hands off at its boundary, and day 28's `stop` arrived inside the 73 ms hash so `the next demote publishes`
read a log that ended at `demote copy complete`. The promote cells (`pcell`) gain no wait: their failures were the
miss, which the park removes.

### 1.5 Acceptance (ruling 40, verbatim)

"DAY28's clause 2 turns green in every arm on both cards (identity x4 `ALL GREEN (teeth=0)`, failure both arms `ALL
GREEN` including the `digest` cell, contract fault `ALL GREEN` with the four promote cells and the two demote cells,
twin, hit OFF and ON with the day-24 census, the two day-28 fault cells still green, the bitwise digest cell), clauses
1a to 1c re-read on the new tree and still passing (`in - completion <= 12.0`, stall ON <= OFF + 2.0, e2e <= +20.0),
and a new typed count: hits parked on a `Hashing` entry per boot in the identity gate's default ON arm, quoted. Until
every arm is green the day-28 and day-29 code stays on the lane branch and is not integrated; say so in STATE.md."

How each is read: clause 2 from every gate's own verdict line, verbatim, both arms, on the target card in one sitting
(`gates.sh`, `hitgate.sh`, `unit-cells.sh` of day 28 re-pointed at the day-29 receipts root), the hit gate's ON census
against day 24's (`12/12/13/13`, `2/2/3/3`, 30 route submissions); clauses 1a to 1c from the day-26 double-park cell
byte-for-byte (`pro-single-day26/double-park.sh`, one hold, twenty boots, N=5 per arm per order, both orders) read by
`day28-reading.py` unchanged (the ledger line's `owner in-completion` is 1c; the reader's parse tolerates the new tail);
the count from `grep -c 'hit parked on a Hashing entry'` per `host-on-server.log` of the identity gate's default ON arm
and from the ledger line's `H`, both quoted, plus the copy-phase count of 1.1 (expected zero). On the local RTX 5090:
identity default ON, fault default and plain, hit OFF/ON, where the lock allows within a bounded wait, else NOT RUN,
stated. No number is compared across cards.

**What is NOT changed.** The hash program and its thread (day 28), the contract routes and their receipts, the promote,
capture and restore routes, the wire, every `Block` settle of day 28. No new `MEMRA_*` name, no flag, no `unsafe`, no
engine or tier crate change. Cost expectation, pre-registered: a request that arrives inside the `Hashing` window waits
the REMAINDER of the hash plus one tick (at most about 73 ms plus a tick on the target card, about 12 ms plus a tick on
the 5090 class) and then takes the promote park; a request outside the window is untouched; the double-park cell's
promotes are spaced further than one hash, so 1a to 1c are expected unchanged (7.40 / -3.4 / +16.9 on day 28).

## 2. The code (`867655368`; gate fixes `89318289b`, `259c75f62`, `06e290374`; scripts `29a1cc366`)

`crates/memra-server/src/worker.rs` and `tools/kv-host-contract-fault-gate.sh` only. No engine or tier crate, no wire,
no new numeric program, no `unsafe`, no new `MEMRA_*` name, no flag.

- **`host_hashing_hit`** (pure over the host state; section 1.1's predicate exactly: `demoting` with `hashing` `Some`, the
  pool key, the `lookup` rules, `n > device_best_len`) and **`host_hashing_park`** (records the request id once on
  `PendingHashing.parked`, counts a re-park in `reparks`, prints the typed line on the first park, answers `true`; touches
  neither the helper nor a settle). The probe (`host_promote_park_probe`) calls the park immediately BEFORE
  `host_promote_probe_decision` and returns `true` to the caller's existing requeue arm (`requeue.push_back(req);
  parked_on_promote += 1`). One park site.
- **`PendingHashing`** gains `parked: Vec<String>` and `reparks: u32`; the `Poll` re-park hands the whole state back
  (no rebuild); the ledger line ends `; H hit(s) parked on the Hashing entry (R re-park(s))`; the latch closure prints
  the tail line when `H > 0`.
- **The parked-only bounded wait**: the guard gains `|| hpx.demoting.as_ref().is_some_and(|d| d.hashing.is_some())`.
- **CPU cells** (server lib `826 passed; 0 failed; 14 ignored` on the local rig; the day-29 subset `14 passed`):
  `hashing_hit_parks_the_request_once_per_id_and_a_miss_does_not` (the hit at depth 64 with the ticket; the misses: a
  device hit as deep, a shorter prompt, another prompt, another pool key, a copy-phase `Demoting` entry, no entry; the
  park once per id with the re-park counted; the ids ride a poll; consumed at publication (the bind refuses by name on
  the CPU, the day-17 proof) and at the latch, after which the park predicate is false), the census
  `every_path_that_meets_a_hashing_demote_meets_it_through_the_same_settle` extended (the wait's guard, the probe's
  park before the decision, the pure predicate's three rules, the park's silence toward the helper, one park site, the
  ledger tail, the latch tail; the orphan grace still reads no demote state), and
  `the_run_loop_waits_boundedly_when_the_queue_is_only_requests_parked_on_a_promote` asserting the new guard.
- **The fault gate's demote cells** (`cell()`): before `stop`, `await_publication` waits, bounded 150 x 100 ms = 15 s,
  for the check's own condition (a `[prefix-host] demote: ` line after the injected refusal, through the gate's own
  `after`), recorded as its own check `the boot's last publication landed before stop (bounded 15 s wait)`. Two shapes
  of this wait were wrong first and are in the history, both caught on the local RTX 5090 before the target card's
  fault cell ran: (i) `29a1cc366` counted hand-offs with `grep -c`, which exits 1 on a zero count under the gate's
  `set -euo pipefail`, so the gate aborted at its first cell with no verdict line (`fault-default` rc=1 in 9 s,
  `fault-plain` rc=1 in 6 s; `rtx5090-day29/fault-{default,plain}/`); (ii) `89318289b` and `259c75f62` fixed the abort
  but still counted hand-offs, and the count was 0 when r3 returned, because the spec-boundary insert, the eviction
  and the demote's submission land as r3 COMPLETES (the server log: `insert (spec-boundary)`, `evict (LRU)`, `demote
  submitted` after r3's `[spec-k]` line, then `demote copy complete` as the last line before the stop), so the wait
  passed on nothing and `the next demote publishes` still failed (`fault-default-rerun` `2 FAILURE(S)`, its two
  `the next demote publishes`; `fault-plain-rerun` `ALL GREEN`); (iii) `06e290374` waits for the publication itself.
  Nothing in the gate's assertions moved; the wait is the only change.
- **Checks on the code**: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and
  server `Finished`; the GPU-less `DOCS_RS=1 --target x86_64-unknown-linux-gnu` clippy pass `Finished`;
  `check-flags: no uncovered runtime names`; `check-conflict-markers: OK`; `git diff --check` clean; zero em dashes.

## 3. The sitting (target card, BOX3, one RTX PRO 6000 Blackwell Server Edition, 600 W; `pro-single-day29/box/`)

Binary `03fdb383ff201ff1c1e4c773…` (`bins/memra-server.sha256`, one binary for every cell) built on `/root/wt-a` at
`29a1cc366` (`tree.sha`; the day-29 code `867655368` plus the scripts). The tree moved under the gate scripts as the
fixes landed (the binary did not: no `crates/` change after `29a1cc366`): `259c75f62` from 17:40:52Z (the identity and
failure gates ran on it), `06e290374` from 18:00:01Z (the contract-fault, twin, hit and unit cells ran on it;
`unit/tree.sha`). Three collector holds and the hit gate's own `flock` in one sitting, 17:37:54Z to 18:10:27Z, zero
lock retries (no `lock-retries.log`), no compute app on the card before or after any cell, every `CELL.jsonl` `status:
executed-not-qualified`, `qualification: false`, `exit_code 0`: `double-park` (17:37:54Z to 17:57:18Z, twenty boots,
4,639 samples at 250 ms, 33 to 50 C, 32.7 to 358.1 W, 0 to 17,109 MiB), `gates` (17:57:18Z to 18:06:08Z, 2,115
samples, 37 to 59 C, 86.8 to 507.0 W, up to 21,939 MiB), the hit gate 18:06:08Z to 18:07:38Z, `unit-cell` 18:07:38Z to
18:10:27Z (673 samples, 33 to 39 C, 32.5 to 94.5 W). Every double-park receipt `STALL REPLAY: PASS` (20 of 20),
`errors=0` (20 of 20), `ADMISSIBLE all_receipts=True`, `DAY26 DOUBLE-PARK ADMISSIBLE all_receipts=True`.

## 4. The reading, verbatim (`box/reading-day25.log`, `box/reading-day26.log`, `box/reading-day28.log`)

- `DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=81.7 off_cell_median=85.4 rule on<=off+2.0
  -> PASS`; `order=o2 ... on_cell_median=81.8 off_cell_median=85.1 ... -> PASS`.
- `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=132.3 off=115.5 on_minus_off=+16.8 rule <=+20.0 -> PASS`;
  `order=o2 ... on=132.2 off=115.3 on_minus_off=+16.9 ... -> PASS`.
- `DAY28 CLAUSE 1c owner in-completion N=100 median=7.39 min=7.20 max=45.18 runs_with_demote_without_ledger=0 rule
  <=12.0 -> PASS`.
- `DAY28 REPORTED wall in-completion median=86.5 (N=100); demote_completion median=97.5; helper hashed_in_ms median=73.1
  min=72.9 max=73.5; payloads=[98] mb=[157.9]; hash_polls median=6 max=6; settle modes={'tick-top poll': 100}; tenant top
  gaps: largest median=95.3 second median=18.9`.
- `DAY28 VERDICT clauses_failed=0 -> ALL PASS`.
- Day 25's reader: `DAY25 DOUBLE-PARK stall order=o1 on_minus_off=-3.6 unc=0.1 -> isolated (on 81.7, off 85.4)`;
  `order=o2 on_minus_off=-3.3 unc=0.1 -> isolated (on 81.8, off 85.1)`; `e2e order=o1 on_minus_off=+16.8 unc=1.8 ->
  isolated (on 132.3, off 115.5)`; `order=o2 on_minus_off=+16.9 unc=2.7 -> isolated (on 132.2, off 115.3)`;
  `DECOMPOSITION arm=on N_runs=100 parked_per_run=[1] ... promote_completion median=19.5 promote_in median=26.0 ...
  demote_in median=183.9 ... idle_p50(tick)=13.47 ... | tenant top gaps: largest median=95.3 second median=18.9 sum
  median=114.2`; `arm=off ... promote_in median=10.7 demote_in median=6.2 | tenant top gaps: largest median=98.6 second
  median=16.7`.
- Day 26's reader (its own clauses, not today's gate): `DAY26 CLAUSE 1 ... runs_with_parked_1_submitted_0_not_routed_1=100
  -> PASS`; `CLAUSE 2 e2e order=o1 ... on_minus_off=+16.8 unc=1.8 expected=+15.8 ... |d-expected|=1.0 -> PASS`;
  `order=o2 ... +16.9 unc=2.7 ... |d-expected|=1.1 -> PASS`; `CLAUSE 3 stall ... -3.6 unc=0.1 -> isolated` / `-3.3 unc=0.1
  -> isolated`; its clauses 4 and 5 read `NO RECEIPT` / `NO VERDICT LINE` (the restore arm was not part of today's cell,
  and the reader looks for the hit gate under a `cells/` layout this sitting does not use; today's hit-gate verdicts are
  in section 5).

## 5. Verdicts, clause by clause (the acceptance of section 1.5, verbatim)

**Clauses 1a to 1c re-read on the new tree.** (1a) 81.7 against 85.4 and 81.8 against 85.1, `on_minus_off=-3.6 / -3.3
unc=0.1 isolated`: **PASS** (day 28: 81.8 / 81.9 against 85.2). (1b) **+16.8 / +16.9** (132.3 / 132.2 against 115.5 /
115.3), `isolated`: **PASS** (day 28: +16.9 / +16.9). (1c) `owner in-completion` median **7.39** (N=100, min 7.20, max
45.18): **PASS** (day 28: 7.40 / 7.17 / 45.01). Every one of the 100 landings a `tick-top poll`; the helper 73.1 ms per
157.9 MB; the tenant's second gap 18.9 (day 28: 19.0). Unchanged from day 28 within 0.1 ms in every figure, as
pre-registered: this cell's promotes are spaced further than one hash and no hit fell in the `Hashing` window (the
`parked_per_run=[1]` is the promote park, as on day 28). `DAY28 VERDICT clauses_failed=0 -> ALL PASS`.

**Clause 2, every arm green on both cards.** Target card, verbatim: `identity-default-off` `KV-HOST-SPILL IDENTITY GATE:
ALL GREEN (teeth=0)` (12 ok); **`identity-default-on` `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`** (12 ok; day 28:
`4 FAILURE(S)`); `identity-plain-off` `ALL GREEN (teeth=0)` (12 ok); `identity-plain-on` `ALL GREEN (teeth=0)` (12 ok);
`failure-off` `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok); **`failure-on` `KV-HOST-SPILL FAILURE GATE: ALL GREEN`**
(15 ok, the `digest` cell among them: `the promote caught it: VERIFY FAILED, loud and named`; day 28: `1 FAILURE(S)`);
**`contract-fault` `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`** (123 ok, 0 FAIL; day 28: `23 FAILURE(S)`): per cell
`presubmit` 11 ok, `postpublish` 11 ok (each with `the boot's last publication landed before stop (bounded 15 s wait)`
and `the next demote publishes`), `promote-presubmit` 11, `promote-postpublish` 11, `promote-readyview` 11,
`promote-reject` 14, `d2d-capture` 12, `d2d-restore` 14, `hash-helper-gone` 14 (`hand-off ticket(s) ['3'], refusal
ticket(s) ['3']`), `hash-never-lands` 14 (`hand-off ticket(s) ['3'], refusal ticket(s) ['3']`, r3's `demote refused: the
tier latched off while settling the pending demote` exactly once); `twin-off` and `twin-on` `PREFIX-NEWEST-TURN-FITS:
budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9
cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32
off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` (rc 0 both); `hitgate-off` `SPEC-ON-CACHE-HIT GATE: ALL
GREEN (qwen)` (61 ok; `armed=0 door_on=0`); `hitgate-on` `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok) with the ON
census equal to day 24's exactly: `armed=1 door_on=1 capture_submitted=12 capture_published=12 restore_submitted=13
restore_landed=13 demote_submitted=0 promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0`
(spec-on), `capture_submitted=2 capture_published=2 restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30
route submission(s) across the two boots`. Unit cells on the box: `option_b_`/`option_c_` `test result: ok. 8 passed; 0
failed`; the engine's `d2d_` cells `ok. 5 passed; 0 failed`; the CPU hash cells with the day-29 cells and the two census
tests and the parked-only wait test `ok. 10 passed; 0 failed` (`hash_helper_digests_equal_the_owner_thread_digests_bitwise`
among them). The local RTX 5090 half is in section 6, every cell `ALL GREEN`. **Clause 2 PASSES on both cards.**

**The count: hits parked on a `Hashing` entry per boot, the identity gate's default ON arm.** Target card
(`gates/identity-default-on/host-on-server.log`): **1** (`grep -c 'hit parked on a Hashing entry'`), the line verbatim:
`[prefix-host] hit parked on a Hashing entry: request cmpl-75428aac9b5c30fdea553571818d571e (102 tokens) hits the
Demoting entry's 64 tokens (ticket seq=3, 98 payloads, 157.9MB on the hash helper for 0.3ms); the request waits one tick
for the digests (model gate)`; the ledger: `... 1 hit(s) parked on the Hashing entry (35 re-park(s))` for ticket seq=3
(`landed after 36 poll(s)`: the parked-only wait's 2 ms cadence across the 73.2 ms hash) and `0 hit(s) parked ... (0
re-park(s))` for seq=5 (the boot's second demote, nobody waiting). The sequence in that boot, line numbers 65 to 82:
`demote copy complete off the tick: ticket seq=3 ... 53.4ms from submission to completion` (65), the park (68), `demote
digests landed off the tick: ticket seq=3 ... hashed in 73.2ms` (70), `promote submitted off the tick: 64 tokens,
158.9MB, ticket seq=4 ...; request parked` (71), r3 `[spec-k] ... prompt=102 cached=64 lcp=64` (82); r4 `cached=64
lcp=64` (98). Local RTX 5090 (`rtx5090-day29/identity-default-on/ev/host-on-server.log`): **1**, `... (102 tokens) hits
the Demoting entry's 64 tokens (ticket seq=3, 50 payloads, 53.7MB on the hash helper for 0.4ms) ...`, ledger `1 hit(s)
parked on the Hashing entry (5 re-park(s))` (`hashed in 12.9ms`, `landed after 6 poll(s)`), r3 and r4 `cached=64
lcp=64`. Elsewhere the park fired where the shape puts a hit in the window and nowhere else: `failure-on` `digest` cell
1 (then `promote submitted`, then `VERIFY FAILED: promoted digest e2a8b561… != demote digest 285688ef…`); the fault
gate's four promote cells 2 each (r3 P_A on seq=3, r4 P_B on seq=5; `hash helper for 0.2 to 0.5ms` on the target card,
0.2 to 2.1 ms on the 5090); the plain arms, the demote cells, the d2d cells and the two hash cells 0. The copy-phase
count of section 1.1: every hit in every ON boot on both cards reads `cached=64 lcp=64` at admission, so **zero** hits
fell on a copy-phase `Demoting` entry.

**Clause 2 on the day-28 tree was red for one cause and is green here for one cause**: the hit that landed in the
`Hashing` window (0.2 to 0.5 ms after the hand-off on the target card; the window is the whole 73 ms hash plus a tick)
now parks and re-admits to the promote park, so every downstream assertion (the promote, the verify round trip, the
promotions counter, the strict-prefix hit, the injected promote faults, the demote cells' publication) sees the entry.

## 6. Local RTX 5090 (the acceptance gate's "both cards"; `rtx5090-day29/`)

`battery-5090.sh`: identity default ON, the fault gate default and plain, the hit gate OFF and ON (C's day-23 shape on
this card, `MEMRA_HOSTGATE_CACHE_MB=64`), the tree's release binary (`build.log` rc=0 under the CPU quota, binary
`be7845c9a4733131…`) and the 9B NVFP4 MTP artifact, each cell after a bounded idle wait and under the gate's own `flock
/tmp/memra-5090.lock`; the card was free at every cell (zero waits logged; card before each cell 57 to 62 C, 9.6 to
25.0 W, 15 MiB). RTX 5090 Laptop GPU, `power.limit [N/A]`. Verbatim from `battery.log` and the gate logs:

- `identity-default-on` (17:37:44Z to 17:37:59Z): **`KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`** (12 ok); the
  count above.
- `fault-default` (17:37:59Z, tree `29a1cc366`): rc=1, no verdict line, the log ends at `== cell presubmit:` (the gate
  aborted under `set -e` on `grep -c`'s zero count: section 2); `fault-plain`: the same, rc=1 in 6 s. Both dirs kept as
  the record of the gate's crash; not a cell result.
- `fault-default-rerun` (17:41:44Z to 17:43:14Z, tree `259c75f62`): **`KV-HOST-CONTRACT-FAULT GATE: 2 FAILURE(S)`**,
  `FAIL: presubmit: the next demote publishes`, `FAIL: postpublish: the next demote publishes`, with the hand-off wait
  `ok` on 0 hand-offs (the wait's wrong shape: section 2); `fault-plain-rerun`: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`.
- `fault-default-rerun2` (18:00:31Z to 18:01:59Z, tree `06e290374`): **`KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`** (123
  ok, `ok: presubmit: the boot's last publication landed before stop (bounded 15 s wait)`, the same for `postpublish`;
  the four promote cells 2 parked hits each); `fault-plain-rerun2` (18:01:59Z to 18:03:16Z): **`ALL GREEN`** (123 ok).
- `hit-off` (17:38:14Z to 17:38:36Z): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok; `armed=0 door_on=0`);
  `hit-on` (17:38:36Z to 17:39:02Z): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok), census `armed=1 door_on=1
  capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0
  promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0` (spec-on), `capture_submitted=2
  capture_published=2 restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30 route submission(s) across
  the two boots`: equal to day 24's and to the target card's.

No number from this card is compared to the target card's.

## 7. Checks, budget, cleanup

On the records commit: `cargo fmt --all -- --check` clean (no Rust moved after `867655368`); `check-flags: no
uncovered runtime names`; `check-conflict-markers: OK`; `git diff --check` clean; `.gitattributes` (`*.log
-whitespace`) in `pro-single-day29/box/` and `rtx5090-day29/`; zero em dashes in the day's own lines (banked server
logs carry the engine's `[gpu-watch]` boot line verbatim). Every push in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode, logged to the gate-skips ledger; no qualification claimed. Budget:
about 3.5 agent-hours of 5 at the records commit. Box: `/root/wt-a` at `06e290374` on `lane-a-day29`, clean;
`/root/spill-receipts/a-day29/` (bin `03fdb383ff201ff1c1e4c773…`) mirrored to `pro-single-day29/box/` (bins excluded);
the four bundles that carried the tree removed on both ends; `/root/a29-build.sh` removed; no server or process of mine
left running; nothing of other lanes touched. Local: the release build and the battery under the CPU quota; the
collector never held the 5090 lock outside the gates' own `flock`; no scratch left in `/tmp`.
