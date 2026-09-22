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
