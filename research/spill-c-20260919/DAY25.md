# Session C day 25: the fixed fault gate live on both cards, the identity gate over the retire-seam settle, the retire-seam settle's cost pre-registered and run

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files integ36 brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; no timing here is a
qualification claim. No commit on main, no PR. No engine change of my own today.

## Merge (first action)

#634 was still OPEN, so `origin/lane/spill-integ36-20260922` `64bb0c71d` (the lead's fixed contract fault gate with
the `receipt_seq_accounts` clause, the retire-seam settle and the refused-submission fence release from revuto's round
on #634, A's slice 1 `62aa92279`, B day 30, main `4bb2afb63`) merged as `913095404`, clean, no conflict;
`git diff origin/lane/spill-integ36-20260922 -- crates/ tools/` EMPTY. `tools/check-conflict-markers.sh` OK. Pushed.
Against the day-24 tree `11df6e653` the crates delta is exactly the lead's two settle fixes in `worker.rs` (75 lines),
B's `metering.rs` header, and the gate's 40 lines.

## Task 1, the fixed fault gate live on the local RTX 5090 (9B)

Tree `913095404`, release `memra-server` built locally under `systemd-run --user --scope -p CPUQuota=1200% -p
MemoryMax=28G` (`day25-cpu/build-local.log`, `Finished release in 15.75s`, only `memra-server` recompiled over the
day-24 objects, `rc=0`; digest in `rtx5090-day25/binary.sha256`). The Qwen3.5-9B NVFP4 MTP artifact,
`MEMRA_HOSTGATE_CACHE_MB=64`, driver `day24-cell.sh` (the gate takes the canonical lock `/tmp/memra-5090.lock` itself;
compute apps and driver free sampled before and after). The card was idle at the start (no compute app, 23970 MiB
free, 53 C); no compute app in any before or after snapshot; no lock retry. Receipts `rtx5090-day25/fault-{default,plain}/`.
Verbatim, the verdict line and the new accounting clause's line for each cell:

| Cell | Environment | Verdict line | Accounting lines (the `receipt_seq_accounts` clause, one per demote cell) |
|---|---|---|---|
| fault-default | default (door ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1`; `receipt seq=2 expected 2 + 0 capture ticket(s) submitted before it = 2` |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3`; `receipt seq=4 expected 2 + 2 capture ticket(s) submitted before it = 4` |

The plain arm that read `2 FAILURE(S)` twice on day 24 (the literal `seq=1`/`seq=2` pin) is green on the fixed gate,
and the accounting says why in the gate's own words: the plain arm's two seeds submit two capture tickets on the same
issuer before the demote, so the demote's receipt is `seq=3` (presubmit, one D2H expected) and `seq=4` (postpublish,
two). The default (spec) arm publishes through the spec boundary, no ticket, `+ 0`. Live, not a replay.

## Task 1, the target card (one RTX PRO 6000 Blackwell, collector rig `pro-single`)

The tree reached `/root/wt-c` as a bundle (`91b0d4e08..913095404`, fetched and checked out detached; the bundle removed
on both ends), release `memra-server` built on the box (`pro-single-day25/build.log`, `rc=0`). Cells `fault-default`,
`fault-plain`, `identity-default-off`, `identity-default-on`, `identity-plain-off`, `identity-plain-on` through
`day22-box-run.sh` with its cell-list argument (`day22-cell.sh`, cache 256 MB, the 27B), one collector hold per
cell, bounded lock retries (lane A day 21 held the card when the runner started; the holder was never inspected or
signalled). Results: see the section "Target card results" below (filled after the run).

## Task 2, pre-registration (committed before the run)

**What the lead asked.** The retire-seam settle's cost: revuto's round on #634 found that a pending capture reads a
live session's KV planes on the copy stream while the engine retains no source, so a retiring session's cache could be
dropped (owner stream, unordered against the copy stream) or parked and rewritten under the in-flight read; the lead
settles a pending `Capturing` entry BLOCKING (`host_capture_settle_pending(.., ContractWait::Block, "a session
retire")`) before any session leaves `active`, at the retire block of the worker's iteration. Owed: its cost in the
capture-isolating cell.

**The claim, registered before the run.** The retire-seam settle adds at most one copy's time (under 3.0 ms at the
shape) to the retiring request's own tick and nothing to the tenant's stall median. The copy is the door-moved term of
day 24's arithmetic: the 5184-token entry's KV planes on the 27B, 5184 x 29.7 KB = 154 MB, 1.5 ms even at 100 GB/s,
an order of magnitude below the card's memory bandwidth class; the block starts after the submission, so it waits at
most for what the copy has left.

**The shape.** Lane A's day-16 stall shape (`stall_cell.py`: a streaming tenant decoding one token per tick on a
`MEMRA_SERVE_SPEC=0` boot, the intruder fired at the tenant's 24th token, order 1 = (idle, arm) x 5, order 2 = (arm,
idle) x 5) with the capture-seeding intruder of day 24's arithmetic: a warm hit on a 5120-token entry E0 deepened by
exactly 64 fresh on-grid tokens (5184 = 162 x 32), `max_tokens=1`. Its prefill-done seed publishes a fresh 5184-token
entry (`prefix_seed_deepens`: `fed_len - depth >= PREFIX_CACHE_MIN_TOKENS`, 64 >= 64); with the door ON that seed is a
`capture submitted off the tick (seed)` on the copy stream, with the door OFF the same seed is the tick program's
synchronous `insert (seed)`. Prompts are calibrated through `/v1/tokenize` (no seed happens at calibration): the base
to exactly 5120 tokens, each run's extension to exactly 5184, fresh words per run and per arm, so every arm run
publishes a fresh entry and every intruder hits E0 at `cached_tokens=5120` with `prompt_tokens=5184`. Boot: A's
day-16 boot (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_MB=8192`) with the prefix budget at
8192 MB instead of A's 1024 so the twenty-one entries of one boot (E0 plus ten per arm, about 311 MB each on the 27B)
never evict: an eviction demote is Move 1's term (ON 228 ms against OFF 70 ms in A's day-20 receipts) and would swamp
a 3 ms clause on the tenant's stall; the deviation is registered here with its reason. Door OFF and ON, four boots in
ONE collector hold on the target card, OFF, ON, ON, OFF (pass 1 = off1/on1, pass 2 = on2/off2), N=5 per arm per order,
both orders, the collector's 250 ms telemetry. Harness `day25-retire-cell.py` (client-side, stdlib, imports A's
`stall_cell.py` for the tenant, the post and the pooling), wrapper `day25-retire-cell.sh`, runner `day25-box-run.sh`,
reading `day25-retire-reading.py` (thresholds as constants).

**What the local smoke taught before any box time, and the second arm it forced.** A harness check on the local 5090
(9B, door ON, N=1, `rtx5090-day25/harness-smoke/`, no claim): the calibration lands (`prompt_tokens=5184`,
`cached_tokens=5120`, one capture per arm run, replay PASS), but both captures read `settled_by` = `tick-top poll`,
`8.8ms from submission to completion` (one 9B tick). The code says why: the worker's iteration is tick-top poll
(`worker.rs` 19793), admission, disconnect-abort sweep (21757), prime wave (the seed at prefill-done, 22698), decode,
retire block (23365); the intruder's first token, its `max_tokens=1` finish, is sampled one iteration after its
prefill-done, so its own capture is settled by the next iteration's tick-top poll before any retire. The retire-seam
settle is on the path only when some OTHER session retires in the seed's iteration. So the cell has two arms per boot:

- `plain`, the shape as briefed: bounds the door's whole cost at the shape (submission, poll, publish) and records
  where each capture settled (expected `settled_by_poll` for all ten).
- `coincide`, the seam exercised: a victim stream (a fresh 40-token prompt under the 64-token seed floor so it never
  seeds, `max_tokens=300`) starts at the tenant's 12th token; at the 24th token the intruder is posted and the
  victim's socket is closed right after (post first, close second: admission precedes the abort sweep in the
  iteration). The abort sweep retires the victim in the iteration that admits and primes the intruder, so that
  iteration's retire block finds `finished` non-empty and a `Capturing` entry and the settle runs BLOCKING; the
  publish line reads `settled synchronously by a session retire`. A run whose two arrivals straddle an iteration
  boundary reads `tick-top poll` and is a recorded miss. Both door arms carry the victim's abort and cache drop; only
  the settle differs. Smokes on the 9B: N=1 read 1 retire-settled and 1 poll-settled
  (`rtx5090-day25/harness-smoke-coincide/`); N=3 read 6 of 6 retire-settled, post-to-close gaps 0.19 to 0.31 ms
  (`rtx5090-day25/harness-smoke-coincide-n3/`). In the `coincide` arm the block sits in the seed's own iteration,
  which is the tenant's stall tick, so the tenant's stall clause is the settle's clause there.

**Rules, fixed here (ms; ON minus OFF, one-sided; per arm, per pass).**

- Admissibility (a receipt that fails decides nothing and is reported as such): replay PASS on every receipt;
  `errors=0`; `tenant_text_identical=True` inside every receipt and one tenant text sha across the eight receipts;
  every intruder `prompt_tokens=5184` and `cached_tokens=5120`; ON receipts `captures_published=10` and
  `seed_inserts=10` (the capture's publish prints `insert (seed)` too, the smoke's correction), OFF receipts
  `captures_published=0` and `seed_inserts=10`; `coincide` receipts `victim_aborts=10`; no `[prefix-host] demote:`
  line in any timed run.
- R1, the retiring request's own tick: median intruder wall (its hit restore, 64-row prime, seed and retire) ON minus
  OFF <= 3.0.
- R2, the tenant's stall: `stall_median` (max ITL minus p50) ON minus OFF <= 3.0.
- R3, the tick after the stall tick: `two_tick_stall_median` (ITL[max] + ITL[max + 1] minus 2 p50) ON minus OFF <= 3.0.
- R4, a reading with no rule: the ON arm's `settled_by` counts, `copy_ms` (submission to completion of every capture,
  an upper bound on the block's wait for the retire-settled ones since the block starts after the submission and
  after the tick program's work between them) and the retire-settled runs' stalls.
- R5, `coincide` only: the seam counts as exercised when each ON receipt reads `settled_by_retire >= 8` of 10; under
  8 the arm's clauses are still printed but the seam is `not_exercised` and decides nothing about the settle.
- Verdict: `HOLDS` when admissible, R1 to R3 within the bound in both arms and both passes, and R5 exercised in both
  passes; `HOLDS AT THE SHAPE, SEAM NOT EXERCISED` when R1 to R3 hold but R5 is not met; `FAILS` when any clause is
  over the bound; `UNDECIDED` when a receipt fails admissibility. Nothing here decides the door.

The verdict line and every rule line are quoted verbatim in the section "Target card results" below.

## Target card results

(filled after the run)
