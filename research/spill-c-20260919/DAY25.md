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

**Task 1 gates** (tree `913095404`, binary `f08b78b868eddb93...` built on the box, `pro-single-day25/build.log` `rc=0`,
the 27B, `MEMRA_HOSTGATE_CACHE_MB=256`, one collector hold per cell, `LOCK.json` owner `collector`). Six bounded lock
retries before the first cell (`lock-retries.log`, attempts 0 to 5, 06:49:03Z to 06:59:03Z box time; lane A held the
card, never inspected or signalled), then `fault-default` took the lock at 07:01:03Z and the six cells ran 07:01:03Z to
07:06:25Z (`progress.log`); no compute app in any before or after snapshot; `gate.exit` 0 for all six. Verbatim:

| Cell | Environment | Verdict line | ok / FAIL | Accounting lines |
|---|---|---|---|---|
| fault-default | default | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 67 / 0 | `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1`; `receipt seq=2 expected 2 + 0 capture ticket(s) submitted before it = 2` |
| fault-plain | `MEMRA_SERVE_SPEC=0` | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 67 / 0 | `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3`; `receipt seq=4 expected 2 + 2 capture ticket(s) submitted before it = 4` |
| identity-default-off | default | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | |
| identity-default-on | default, `MEMRA_KV_HOST_CONTRACTS=1` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 | |

The same four accounting lines on the 9B and the 27B: the plain arm's presubmit server log carries three `capture
submitted off the tick` lines on the 27B (two before the demote's submission, counted; the third after it, not
counted, as the clause says), the default arm's none. The identity gate is green over the lead's retire-seam settle
and fence release on both arms and both doors (the settle is new code under the door; `teeth=0` as before).

**Task 2, the first attempt, refused by the harness (kept as `pro-single-day25/retire-attempt1-calibration/`,
`collector/retire-attempt1-calibration*`).** Tree `86ef9a4bb` (the pre-registration commit; the crates equal to
`913095404`, the binary unchanged), one hold 07:06:33Z to 07:07:21Z: every boot came up and every arm exited in 0.4 s
with `RuntimeError: calibration did not converge`: the 27B's tokenizer makes about a quarter of the extension words two
tokens (64 words read 80 tokens), so the fixed one-word step oscillated `(54, 5187), (51, 5181)` and `(50, 5185),
(49, 5183)` and never reached 5184 (the base had converged to 5120 by luck). No run, no receipt, no verdict; `retire
rc=1`, `DAY25 RETIRE VERDICT: not read (a boot or a run failed, rc=1)`. Fix `e8667129e`: `calibrate` brackets, bisects
to the largest word count at or under the target, then appends single-token fillers one at a time and skips a filler
the tokenizer makes two tokens of; unit-checked exact on 360 cases against a fake tokenizer with a quarter of two-token
words and eight two-token fillers. No rule, threshold or claim changed.

**Task 2, the run** (tree `e8667129e`, crates equal to `913095404`, binary `f08b78b868eddb93...` unchanged, the 27B,
harness `945dadbb...`, A's `stall_cell.py` `a2e2867e...`; `pro-single-day25/retire/ev/`). One collector hold
(`collector/retire/`, `CELL.jsonl` `"status": "executed-not-qualified"`, `--validate` rc=0 in
`collector/retire.validate.log`), 07:10:26Z to 07:17:45Z, four boots OFF, ON, ON, OFF, two arms each, N=5 per arm per
order, both orders; no lock retry (the card was free after the gates); no compute app in the before or after snapshot.
Regime from the collector's 250 ms sampler (`command.gpu.csv`, 1750 rows): 37 to 58 C, 32.6 to 503.1 W under the
600 W limit, SM clock 180 to 2422 MHz. Eight receipts, eight `RETIRE REPLAY: PASS` (`replays.log`); every intruder
`prompt_tokens=5184`, `cached_tokens=5120`; `seed_inserts=10` in every receipt; ON `captures_published=10`, OFF `0`;
`coincide` `victim_aborts=10`, post-to-close gaps 0.16 to 0.27 ms; `server_demote_ms=[]` everywhere (no eviction at
the 8192 MB budget); one tenant text sha `264b120d487de2c9` across the eight receipts. The rule lines are in the
receipts and `reading.log`; their decision fields, verbatim:

| Boot / arm | `idle_p50` | `stall_median` (min..max) | `two_tick_stall_median` | `intruder_wall_median` | captures | `copy_ms_median` |
|---|---|---|---|---|---|---|
| off1 / plain | 13.4 | 89.9 (89.7..90.1) | 92.1 | 118.9 | `captures_published=0` | `na` |
| off1 / coincide | 13.4 | 92.1 (92.0..92.6) | 94.5 | 121.4 | `captures_published=0 ... victim_aborts=10` | `na` |
| on1 / plain | 13.5 | 90.3 (90.1..90.5) | 92.7 | 119.7 | `captures_published=10 settled_by_retire=0 settled_by_poll=10` | 15.8 |
| on1 / coincide | 13.5 | 92.3 (92.2..92.7) | 94.8 | 121.8 | `captures_published=10 settled_by_retire=10 settled_by_poll=0` | 15.8 |
| on2 / plain | 13.5 | 90.3 (90.0..90.4) | 92.6 | 119.7 | `captures_published=10 settled_by_retire=0 settled_by_poll=10` | 15.8 |
| on2 / coincide | 13.5 | 92.3 (92.2..92.5) | 94.8 | 121.8 | `captures_published=10 settled_by_retire=10 settled_by_poll=0` | 15.8 |
| off2 / plain | 13.4 | 90.2 (89.7..90.4) | 92.4 | 119.2 | `captures_published=0` | `na` |
| off2 / coincide | 13.4 | 92.2 (92.1..92.5) | 94.5 | 121.5 | `captures_published=0 ... victim_aborts=10` | `na` |

The reading (`reading.log`), verbatim:

    DAY25 RETIRE READING plain R1 pass1 (intruder_wall_ms ON minus OFF, bound <= 3.0): within_bound (on=119.7 off=118.9 delta=0.7)
    DAY25 RETIRE READING plain R2 pass1 (stall_ms ON minus OFF, bound <= 3.0): within_bound (on=90.3 off=89.9 delta=0.4)
    DAY25 RETIRE READING plain R3 pass1 (two_tick_stall_ms ON minus OFF, bound <= 3.0): within_bound (on=92.7 off=92.1 delta=0.6)
    DAY25 RETIRE READING plain R1 pass2 (intruder_wall_ms ON minus OFF, bound <= 3.0): within_bound (on=119.7 off=119.2 delta=0.5)
    DAY25 RETIRE READING plain R2 pass2 (stall_ms ON minus OFF, bound <= 3.0): within_bound (on=90.3 off=90.2 delta=0.1)
    DAY25 RETIRE READING plain R3 pass2 (two_tick_stall_ms ON minus OFF, bound <= 3.0): within_bound (on=92.6 off=92.4 delta=0.2)
    DAY25 RETIRE READING coincide R1 pass1 (intruder_wall_ms ON minus OFF, bound <= 3.0): within_bound (on=121.8 off=121.4 delta=0.4)
    DAY25 RETIRE READING coincide R2 pass1 (stall_ms ON minus OFF, bound <= 3.0): within_bound (on=92.3 off=92.1 delta=0.1)
    DAY25 RETIRE READING coincide R3 pass1 (two_tick_stall_ms ON minus OFF, bound <= 3.0): within_bound (on=94.8 off=94.5 delta=0.3)
    DAY25 RETIRE READING coincide R5 pass1 (seam exercised, >= 8 of 10 retire-settled): exercised (settled_by_retire=10)
    DAY25 RETIRE READING coincide R1 pass2 (intruder_wall_ms ON minus OFF, bound <= 3.0): within_bound (on=121.8 off=121.5 delta=0.3)
    DAY25 RETIRE READING coincide R2 pass2 (stall_ms ON minus OFF, bound <= 3.0): within_bound (on=92.3 off=92.2 delta=0.1)
    DAY25 RETIRE READING coincide R3 pass2 (two_tick_stall_ms ON minus OFF, bound <= 3.0): within_bound (on=94.8 off=94.5 delta=0.2)
    DAY25 RETIRE READING coincide R5 pass2 (seam exercised, >= 8 of 10 retire-settled): exercised (settled_by_retire=10)
    DAY25 RETIRE VERDICT: admissible=True plain-R1-pass1=within_bound plain-R2-pass1=within_bound plain-R3-pass1=within_bound plain-R1-pass2=within_bound plain-R2-pass2=within_bound plain-R3-pass2=within_bound coincide-R1-pass1=within_bound coincide-R2-pass1=within_bound coincide-R3-pass1=within_bound coincide-R1-pass2=within_bound coincide-R2-pass2=within_bound coincide-R3-pass2=within_bound seam-pass1=exercised seam-pass2=exercised -> HOLDS (R1, R2, R3 within 3.0 ms in both arms and both passes; the seam exercised in both passes)

R4, the readings with no rule, verbatim from the ON boots: plain `settled_by_retire=0 settled_by_poll=10`, `copy_ms`
15.7 to 16.0, median 15.8, `capture_polls` all 1; coincide `settled_by_retire=10 settled_by_poll=0`,
`copy_ms_retire_settled` 15.7 to 15.9, `stall_ms_retire_settled` 92.2 to 92.7 (against the OFF coincide runs' 92.0 to
92.6); `captures_submitted_mb=[310.8, ...]` (the capture line prints the entry's bytes; the KV planes' share is day
24's 154 MB, the recurrent state's clone on the owner stream is the rest).

**What the cell says.** On the target card the pre-registered claim HOLDS in both arms and both passes: the door adds
0.1 to 0.7 ms to every median it could touch, all under the 3.0 ms bound and under the harness's own resolution
(idle p99 minus p50, 14.8 minus 13.4 = 1.4 ms, as on every day-16 to day-23 receipt), and the retire-seam settle,
exercised 10 of 10 times per ON boot in the `coincide` arm (every capture `settled synchronously by a session
retire`), adds nothing the tenant's stall median can see (0.1 ms in both passes) and nothing to the intruder's wall
beyond 0.4 ms. The `copy_ms` of 15.8 ms in both arms is the submission-to-settle interval, not the copy: in the
`plain` arm it ends at the next iteration's tick-top poll, in the `coincide` arm at the retire block at the end of the
seed's own iteration, after the tick program's decode phase (the tenant's 13.4 ms step plus the victim's), so the block
found a copy that was already complete; the block's own wait is not observable in this log but is bounded by the
tenant's stall delta. Readings beside the claim: the `coincide` arm's stall sits 2.2 ms above the `plain` arm's in
BOTH doors (OFF 92.1 against 89.9, ON 92.3 against 90.3) and its intruder wall 2.5 ms above, the victim's abort and
cache drop that both doors carry; the `plain` arm reproduces the mechanism the smoke read (no capture of an intruder
that finishes one iteration after its seed ever meets the retire seam). Nothing here decides the door
(`MEMRA_KV_HOST_CONTRACTS`, decide-by 2026-10-05); every cell is `executed-not-qualified`.

## Hygiene on the final tree

`cargo fmt --all -- --check` rc=0 (`day25-cpu/fmt-merge.log`); `tools/check-flags.sh` no uncovered runtime names (no
new `MEMRA_*` read: the drivers set existing names); `tools/check-conflict-markers.sh` OK; `git diff --check` and
`python3 tools/check-public-boundary.py check` run before the final push (recorded in the closing commit's message);
no em dash in any file written today; shellcheck `-S warning` clean on the two new shell drivers; the harness
unit-checked as above.

## Pushes

`913095404` (the merge), `e8cb88c31` (the local fault cells), `86ef9a4bb` (the pre-registration, the harness, the
smokes), `28d64a39d` (the target-card gate cells), `e8667129e` (the calibration fix), then the closing commit (the
retire receipts, this file, STATE, INDEX, the door review rows), each in `MEMRA_RELEASE_QUALIFICATION_MODE=development`
(printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`, logged in the clone's `.git/memra-gate-skips.log`).
Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3: reached through the existing control socket only (`ssh -O check` first, `Master running`); `/root/wt-c` (my own)
left detached at `e8667129e`, clean, the scratch ref I made for the watcher deleted; the bundles removed on both ends;
`/root/spill-receipts/day25/` holds the receipts mirrored here; no server of mine on the card at close (compute apps
empty, `/tmp/memra-gpu.lock` free at my last look); `/root/artifacts`, `/root/memra-spill` and other lanes' worktrees
or processes not touched (lane A's hold was seen only in my retry record and in the compute-apps snapshots taken before
my cells started). One misstep, recorded: my first attempt to stop a build I had started on the stale checkout used
`pkill -f` with a pattern that matched my own remote shell and killed it (the memory note says never to); the build had
already died with that shell and was confirmed absent by pid before the correct tree was built. Local RTX 5090: the
three harness smokes and the two fault cells took the canonical lock and released it; compute apps empty at close. No
`/tmp` scratch left (the bundles were deleted after use).

## Budget

About 2.9 agent-hours against 4: reading and the merge 0.2, the local fault cells and the box shipping 0.4 (the
bundle crawled beside lane A's), the harness, its two smokes and the coincide finding 0.9, the pre-registration 0.3,
the box gates (queued 12 minutes behind lane A) and the two retire attempts with the calibration fix 0.6, the records
0.5. Blockers: none open. For the lead: the fixed fault gate is green live on both cards with the accounting quoted;
the identity gate is green over the settle; the settle's cost cell HOLDS on the target card, with the mechanism note
that a capture only meets the retire seam when another session retires in the seed's iteration.
