# WP-A day 25: the double park (C day 29's finding) and the retire-settle share (day 24's finding 2), on the target card

Lane `lane/spill-a-20260919`, checkout `wt-spill-a`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT: ... no GPU
qualification claimed` and records the skip in the clone's `.git/memra-gate-skips.log`). Every cell below is
`executed-not-qualified` development evidence; nothing here is a qualification claim. No commit on main, no PR.

## Merge (first action)

`gh pr list --head lane/spill-integ40-20260922 --state all` printed `[]` and `origin/lane/spill-integ40-20260922`
does not exist on origin (`git rev-parse --verify` refused: `Needed a single revision`), so neither branch of the
first-action rule applied. `origin/main` at `ebe3fe17d` (PR #642, integ39) merged `--no-ff` as `d14b0f3b2`,
clean. C day 29 (`89a6aa8d6` on `origin/lane/spill-c-20260919`) is read from its remote branch by `git show` and
not merged here (the lead's integ40 union owns the shared records); the lead's integ40 record does not exist yet,
so the lead's reading of the double park is not quoted below; C's `DAY29.md` "Finding, stated for the lead" is.

## Task 1, pre-registration (this section is committed before the cell runs)

**What is owed.** C day 29 (`research/spill-c-20260919/DAY29.md`, Task 2's finding): on today's tree the promote
arm's hit parks TWICE. Verbatim from C's `o1/b01-x/promote/receipt.json` `server_log_lines` (the same in all 100
promote runs of the ten X boots): `[prefix-host] promote submitted off the tick: 64 tokens, 158.8MB, ticket
seq=24, 32 items on the contracts door's copy stream; request parked`, `[prefix-host] promote published off the
tick: ticket complete after 1 poll(s), 19.6ms from submission to completion (tick-top poll)`, `[prefix-host]
promote: 64 tokens, 158.8MB in 63.1ms (model gate)`, then `[prefix-cache] restore submitted off the tick: 64
tokens, 32 planes (158.8MB), ticket seq=26 on the contracts door's copy stream; recurrent state copied on the owner
stream; request parked` and `[prefix-cache] restore landed off the tick: 64 tokens (158.8MB) complete after 1
poll(s), 89.9ms from submission to completion, 90.0ms to re-admission (tick-top poll)`; between them the inline
demote: `demote submitted off the tick ... seq=25`, `demote published off the tick: ticket seq=25 complete after 1
poll(s), 58.7ms from submission to completion`, `[prefix-host] demote: 64 tokens, 159.8MB in 133.5ms`. C's promote
arm read a tenant stall of 149.6 where the day-18 and day-23 trees read 81.9. C's reading: "the demote's copy now
lands before the next tick top and its two on-tick hashes ... fall on ONE tick"; and "which slice ... moved the
copy's landing (Move 2's restore on the same copy stream, the D2D receipt term, or the #638 parked-only wait
covering a pending restore) is not determined here."

**What the code says the re-admission wait is made of (read before the run, to be confirmed by line).**

1. The poll cadence of a parked restore is the TICK TOP: `host_restore_settle_pending(&mut px, &mut hpx,
   ContractWait::Poll, "the tick top")` runs once per tick, after the demote poll, the promote poll and the
   capture poll, in that order (`worker.rs` 21261 to 21296). The parked-only bounded wait (#627, `worker.rs`
   23250 to 23263) fires only when `active.is_empty()`; with the tenant streaming it never fires, so a parked
   request waits at least one full tick regardless of when its copy lands.
2. The tick that follows the restore's submission carries the tenant's decode step AND the inline demote's
   on-tick work: the demote poll's completion checksum and `bind_tier_image`'s bundle checksum at publication
   (C's reading; the hash micro-cell reads about 78 ms per 160 MB pass on this host, `pro-single-day18/hashmicro/`).
   The demote poll precedes the restore poll at the tick top, so the restore's re-admission pays for both hashes
   before its own poll runs.
3. The restore's `X ms from submission to completion` is `submitted.elapsed()` stamped AT THE POLL
   (`host_restore_settle_with`, 14986), so it is the submission-to-poll span, not the copy's duration. The copy's
   own duration on this card is cell (v)'s: about 0.16 ms for a 158 MiB span plus 0.34 ms for the receipt pair
   (`pro-single-day22/`, HOSTPREFIX-DOOR.md section B).
4. `X ms to re-admission` is `t0.elapsed()` at `host_restore_take_ready` (15635), in the admission pass right
   after that tick top.

So the pre-read expectation: `to re-admission` is about one tick (the tenant's idle p50 near 13.5 ms) plus the
demote's two hashes (about 75 ms by C's `133.5 - 58.7`), not a copy; the restore route's own copy is under 1 ms.

**No existing typed refusal fits this shape.** The restore probe (`host_restore_park_probe`, 15334 to 15412)
refuses by entry class (TP, latent, DFlash-tail, `pos != toks.len()`, empty logits, no KV plane), by tenancy
(`hpx.restoring.is_some()`), by the latch `restore_off_tick_disabled` (set only by a failure, 15026 and 15037) or by
prefix reuse being off; none names "the entry this request just promoted". The server's `MEMRA_KV_HOST_FAULT` arms
are `alloc-fail`, `capacity`, `flip-demote`, `contract-*`, `d2d-delay-capture` and `d2d-delay-restore`; none
refuses the restore route. So (b) is measured door ON against door OFF, as the task allows, and stated as such:
the OFF arm serves the hit through the pre-door program (an on-tick promote, then the on-tick device-hit copy),
so the pair prices the door whole on this shape, not the second park alone. The second park's own share is read
from the ON receipts' lines (the decomposition above), not from the pair.

**The cell (`day25-double-park.sh` under the collector, ONE lock hold, target card).** The day-16 script's
`promote` arm, byte-for-byte `stall_cell.py` (its SHA-256 recorded), on today's tree's binary. Two arms that
cannot share a boot (the door is a boot flag), so the A/B interleaves across BOOTS as C day 29 did: order 1 boots
`ON OFF ON OFF ON OFF ON OFF ON OFF`, order 2 boots `OFF ON OFF ON OFF ON OFF ON OFF ON`, twenty boots, five per
arm per order. Every boot: `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256
MEMRA_KV_HOST_MB=8192` (the day-16 `off` boot) plus `MEMRA_KV_HOST_CONTRACTS=1` on the ON boots (the day-16 `on`
boot); the harness `--mode promote --n 5` (its own idle/arm interleave in both orders inside the boot: 10 promote
runs and 10 idle runs per boot). The entry shape is the day-16 script's: the day-15 alternation, a host hit of 64
tokens with a 22- to 25-token suffix whose insert evicts the other entry (the inline demote). Card
`before`/`after` per boot; the collector's `command.gpu.csv` at 250 ms across the hold. Bounded lock retries (60 x
120 s; lane C shares the card today; the holder is never inspected or signalled). No dry boot: today's tree booted
on this card yesterday (`pro-single-day24/`).

**Quantities, fixed here (`day25-double-park-reading.py`, run after the hold, nothing tuned).**

- Admissibility per boot: `STALL REPLAY: PASS`, `errors=0`, one tenant text SHA, `server_promote_ms` count 10, no
  `demote failed|promote failed|promote refused|TIER DISABLED` in the boot's `server.log`. An inadmissible boot is
  named and excluded; the cell is reported at whatever N remains, never re-run to fill it.
- (b) per arm per order: the tenant's stall (the harness's rule-line `stall_median`, five boots -> `cell_median`,
  min, max, IQR) and the request's end-to-end latency (`intruder.wall_ms`, 50 runs per arm per order -> median,
  p95, IQR); `on_minus_off` per order with unc in quadrature of the two IQRs, `isolated` when `|d| > unc`, else
  `under_resolution`; pooled over both orders as context.
- (a) from the ON receipts' `server_log_lines`, per promote run (100 runs): `request parked` count (expected 2),
  the promote's `from submission to completion`, the promote `in` ms, the demote's `from submission to
  completion`, the demote `in` ms, the restore's `from submission to completion` and `to re-admission`; the
  restore's `to re-admission` minus `from submission to completion` (the settle-to-re-admit slack); the idle p50
  (the tick); and the residual `re-admission - (idle p50 + (demote in - demote completion))`. From the OFF
  receipts: `request parked` count (expected 0) and the promote `in` ms.
- (a) also from the tenant's ITL series per arm run: the two largest gaps (the two stretched ticks) and their sum,
  so the tick that carries the second park is read against the tick that carries the first.
- (c) is written from these numbers after the run; nothing is tuned on seeing them.

## Task 2, pre-registration (committed before the code moves)

**What is owed.** Day 24 finding 2: every spec-boundary capture of the hit gate settled `Block` at the session
retire (`host_capture_settle_pending(.., ContractWait::Block, "a session retire")`, `worker.rs` 24930 to 24939),
`106.3 to 204.6 ms from submission to completion`, and DAY24 wrote that "the owner thread still pays a host wait
for the 159 MB copy at the retire". The receipts cannot say how long that wait was: `copy_ms` is
`submitted.elapsed()` stamped when the settle returns (`host_capture_settle_with`, 14468), so it is the
submission-to-settle span, which includes however long the session kept decoding before it retired. The wait
itself (the host wait inside `t.synchronize(&ticket)` under `Block`, `host_kv_planes_settle_capture` 14273) is not
on any line. Owed item 3 of `OWNER-THREAD-OFFLOAD.md` asks for its price.

**The typed line (no new `MEMRA_*` read, no new state machine arm).** `host_capture_settle_with` stamps two
figures around the settle closure: `settle_after_ms` (`submitted.elapsed()` on entry to the settle, the span the
copy ran while the owner thread did other work) and `settle_held_ms` (the settle's own duration on the owner
thread: under `Block` the host wait plus the receipt read and retire; under `Poll` the events read and receipt).
Both ride `PendingCapture` (two `f64` fields) and the publish line grows one clause inside its parenthesis:
`... to publication (settled synchronously by a session retire; the settle held the owner thread H ms, entered A ms
after submission)`. Every existing parser splits on `poll(s), ` and `ms` before the parenthesis
(`stall_cell.py` `server_capture_ms`; the hit gate counts `capture_published=` lines by substring), so nothing
downstream moves. The census `day24_spec_boundary_capture_paths_are_named` and the CPU state-machine tests
construct `PendingCapture` at one site (`worker.rs` 42450) and gain the two zeroed fields.

**The cell (`day25-hitgate.sh`, its own `flock` on `/tmp/memra-gpu.lock`, door ON only).** The spec-on-cache-hit
gate (`tools/spec-on-cache-hit-gate.sh qwen`) on the rebuilt binary with `MEMRA_KV_HOST_CONTRACTS=1` (the gate
exports `MEMRA_KV_HOST_MB=8192`), the day-24 `hitgate.sh` with the OFF arm removed (the line never prints under
OFF; the OFF arm's evidence is day 24's). Its clause is the gate's own: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`,
every `spec==plain byte identity` ok, the engagement clause armed.

**Quantities, fixed here (`day25-retire-reading.py` over `qwen-on-server.log` and `qwen-off-server.log` of the ON
arm).** Per `capture published off the tick` line: `why`, tokens, polls, `from submission to completion`,
`settled_by`, `held`, `entered after`. Grouped by `settled_by`: N, min, median, max of `held`, of `entered after`
and of `from submission to completion`; the share `held / from submission to completion` per line (min, median,
max). The day-24 sentence is then re-read against `held`: if the median `held` is under the copy's own duration
(cell (v): about 0.5 ms for copy plus receipt pair) the retire seam waited on a copy that had already landed, and
the wait is the settle's fixed cost, not the copy; if `held` is of the copy's order, the seam waited on the copy.

**The poll-then-retire question, answered from the code before the numbers.** `PendingCapture` carries no
source-session identity (its fields: `pool_key`, `why`, `shell`, `contract`, `ready`, `t0`, `polls`, `copy_ms`,
`settled_by`, `trace_role`), so the retire seam cannot tell whether the retiring session is the copy's source and
must settle `Block` on every retire while a capture is pending. A `Poll` first at the seam (landed: publish, no
host wait; running: then `Block`) removes the wait exactly when the copy has landed by the seam and costs nothing
when it has not; it does not break the source-lifetime rule because the `Block` branch still runs before any
session leaves `active`. Whether it removes anything on this shape is what `held` measures. Proposed after the
run, not implemented.

## Budget plan

Reading and the merge 0.6 agent-hours; the pre-registration, the two drivers and the two readers 0.5; the typed
line and the CPU checks 0.5; the box shipping, build and the two holds (twenty boots near 30 minutes, the hit gate
near 1 minute) 1.0 polled; the receipts, the readings, the records 0.8. Total planned 3.4 of 4.

## Task 1, the run (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day25/box/double-park/`)

Tree `71ee2a64d` on the box worktree `/root/wt-a` (branch `lane-a-day25`, clean; the tree carries Task 2's typed
line, which never prints in this cell), binary `62b4f802155cb967b86121c3c572b10174b8ab7d7e721230ce9260512a4c4e16`
(`ev/binary.sha256`), harness `stall_cell.py` SHA-256 `13867e77da40a9b5…` (`ev/CELL.txt`). ONE collector hold
(`CELL.jsonl` `status: executed-not-qualified`, `elapsed_seconds 1171.5`, `exit_code 0`, `qualification: false`;
`ev/LOCK.json` owner `collector`, mechanism `inherited-flock-same-open-description`), reached after four bounded
lock retries (`lock-retries.log`: lane C's server held the card 11:43Z to 11:51Z; never inspected or signalled).
Twenty boots 11:51:36Z to 12:11:07Z in the pre-registered order; `compute-apps.{before,after}.csv` empty (the card
was mine alone in the hold); card 33 to 52 C, 31.96 to 362.41 W under 600 W, 0 to 17,109 MiB (4,671 samples at
250 ms). Every receipt `STALL REPLAY: PASS` (20 of 20), `errors=0`, one tenant text SHA per boot,
`server_promote_ms` count 10, no bad line: `ADMISSIBLE all_receipts=True`.

**The reading (`day25-double-park-reading.py`, verbatim).**

```
DAY25 DOUBLE-PARK arm=on order=o1 N_boots=5 stall_medians=[149.5, 149.3, 149.2, 149.4, 149.4] stall_cell_median=149.4 min=149.2 max=149.5 IQR=0.1 | request e2e N_runs=50 wall_median=221.4 p95=261.2 IQR=0.9
DAY25 DOUBLE-PARK arm=on order=o2 N_boots=5 stall_medians=[149.5, 149.2, 149.5, 149.4, 149.4] stall_cell_median=149.4 min=149.2 max=149.5 IQR=0.1 | request e2e N_runs=50 wall_median=221.4 p95=261.3 IQR=0.7
DAY25 DOUBLE-PARK arm=off order=o1 N_boots=5 stall_medians=[85.3, 85.4, 85.4, 85.3, 85.4] stall_cell_median=85.4 min=85.3 max=85.4 IQR=0.1 | request e2e N_runs=50 wall_median=115.5 p95=154.6 IQR=0.7
DAY25 DOUBLE-PARK arm=off order=o2 N_boots=5 stall_medians=[85.2, 85.3, 85.4, 85.1, 85.3] stall_cell_median=85.3 min=85.1 max=85.4 IQR=0.1 | request e2e N_runs=50 wall_median=115.4 p95=154.4 IQR=0.8
DAY25 DOUBLE-PARK stall order=o1 on_minus_off=+64.0 unc=0.1 -> isolated (on 149.4, off 85.4)
DAY25 DOUBLE-PARK e2e order=o1 on_minus_off=+105.8 unc=1.2 -> isolated (on 221.4, off 115.5)
DAY25 DOUBLE-PARK stall order=o2 on_minus_off=+64.2 unc=0.1 -> isolated (on 149.4, off 85.3)
DAY25 DOUBLE-PARK e2e order=o2 on_minus_off=+105.9 unc=1.1 -> isolated (on 221.4, off 115.4)
DAY25 DECOMPOSITION arm=on N_runs=100 parked_per_run=[2] restore_readmission median=90.1 range=89.7..91.5 restore_completion median=90.1 slack(readmission-completion) median=0.10 max=0.20 promote_completion median=19.6 promote_in median=26.1 demote_completion median=22.3 demote_in median=97.2 demote_in-completion median=74.9 idle_p50(tick)=13.46 residual median=+1.8 range=+1.1..+3.1 | tenant top gaps: largest median=162.8 second median=22.6 sum median=185.7
DAY25 DECOMPOSITION arm=off N_runs=100 parked_per_run=[0] promote_in median=10.8 demote_in median=6.2 | tenant top gaps: largest median=98.7 second median=16.6
```

The ON arm's idle p50 was 13.45 to 13.53 and its p99 14.9 in every boot; the OFF arm's 13.36 to 13.39 and 14.8 to
15.0. The ON stall reproduces C day 29's arm X (149.6 / 149.7) to 0.2 ms across sittings and the OFF stall
reproduces day 16 (85.0) and C day 23 (85.2); today's pair is a same-window pair on one tree.

**(a) What the re-admission wait is made of, by line (every one of the 100 ON promote runs carries the same eleven
lines, `o1/b01-on/promote/receipt.json` run 2 quoted; the seq numbers advance per run).**

1. `[prefix-host] promote submitted off the tick: 64 tokens, 158.8MB, ticket seq=4, 32 items on the contracts door's
   copy stream; request parked`: the first park, at the admission pass of tick A-1.
2. `[prefix-host] promote published off the tick: ticket complete after 1 poll(s), 20.0ms from submission to
   completion (tick-top poll)` then `[prefix-host] promote: 64 tokens, 158.8MB in 64.4ms` (steady `26.0` to `27.1`):
   tick A's top polls the promote complete 19.6 ms after submission (median), publishes it 6.5 ms later, and the
   insert's eviction submits the inline demote between the two (`demote submitted off the tick ... seq=5`).
3. `[prefix-cache] restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=6 on the contracts
   door's copy stream; recurrent state copied on the owner stream; request parked`: the SECOND park, in the
   admission pass of the same tick A, right after the promote's publication put the planes on the device.
4. Tick A runs the tenant's decode step (idle p50 13.46 ms). The parked-only bounded wait never fires: it needs
   `active.is_empty()` (`worker.rs` 23256) and the tenant is active, so the restore's poll cadence is the tick top.
5. Tick B's top polls in the fixed order demote, promote, capture, restore (`worker.rs` 21261 to 21296):
   `[prefix-host] demote published off the tick: ticket seq=5 complete after 1 poll(s), 60.2ms from submission to
   completion` (steady `22.3`) then `[prefix-host] demote: 64 tokens, 159.8MB in 135.2ms` (steady `97.2`): the
   demote's publication after its poll takes `demote_in - demote_completion` = **74.9 ms** median, the two host
   hashes C named (the completion checksum and `bind_tier_image`'s bundle checksum; the hash micro-cell reads 77.9
   ms per 160 MiB pass on this host).
6. Only then `[prefix-cache] restore landed off the tick: 64 tokens (158.8MB) complete after 1 poll(s), 89.9ms from
   submission to completion, 90.0ms to re-admission (tick-top poll)`: the restore's poll installs the owner-stream
   wait, the admission pass re-admits, `[prefix-cache] hit: 64 of 89 prompt tokens from cache`.

The arithmetic of the median run: `re-admission 90.1 = tick A's decode 13.46 + the demote's two hashes 74.9 +
residual 1.8` (range +1.1 to +3.1); the slack between the restore's poll and its re-admission is 0.10 ms (max
0.20). The copy the second park waits for is not in the figure at all: the restore's `from submission to
completion` is stamped at the poll, and the copy's own duration on this card is cell (v)'s 0.16 ms plus 0.34 ms for
the receipt pair (`pro-single-day22/`). The tenant's two stretched ticks read the same story: the second gap
(median 22.6 ms) is tick A (decode 13.46, the promote's publication 6.5, the restore's submit with its owner-stream
recurrent copy); the largest gap (162.8) is tick B (the demote's 74.9 ms of hashes, the restore's poll and
re-admission, then the hit's suffix prime and the decode). Under OFF the largest gap is 98.7 (the on-tick promote
10.8, the on-tick demote 6.2, the same suffix prime and decode). `149.4 - 85.3 = 64.1` against `74.9 - 10.8 - 6.2 =
57.9`: the door's on-tick hashes explain the pair to within 6 ms; the second park's own on-tick share (the poll,
the install, the take) sits inside the +1.8 ms residual.

**(b) Verdicts, verbatim.** Door ON against door OFF, this shape only (no existing typed refusal fits, as
pre-registered): the tenant's stall `on_minus_off=+64.0 unc=0.1 -> isolated` (order 1) and `+64.2 unc=0.1 ->
isolated` (order 2); the request's end-to-end latency `on_minus_off=+105.8 unc=1.2 -> isolated` (order 1) and
`+105.9 unc=1.1 -> isolated` (order 2), N=5 boots per arm per order, 50 runs per arm per order, both orders, 33 to
52 C. Of the request's +105.9 ms, 90.1 is the second park's re-admission wait and about 15.3 is the first park's
(the promote polled at 19.6 plus published at 6.5 against OFF's on-tick 10.8), residual 0.5.

**(c) By construction, a scheduling artifact, or both: both, in fixed shares.** By construction: the restore route
parks EVERY whole-entry device hit it admits, and `host_restore_park_probe` has no predicate for "the entry this
request promoted 6.5 ms ago"; a park costs at least one tick top (the tenant's decode, 13.46 ms here) for a copy of
about 0.5 ms that the OFF program does on the tick in 0.16 ms (the same D2D on the owner stream). That park is
Move 2's cost on this shape and is unearned: the planes were already on the device and the copy they wait for is
two orders of magnitude shorter than the wait. The magnitude, 90.1 ms and not 13.5, is the scheduling artifact of
the hash tick: the tick-top order polls the demote before the restore, so the inline demote's 74.9 ms of host
hashes run before the restore's poll in every run (100 of 100). The tenant's +64 ms is NOT the second park's:
its on-tick share is inside the +1.8 ms residual; the tenant pays the demote's two hashes landing on one tick, which
C day 29 read and which today's pair confirms at 57.9 of the 64.1 ms. Whether the second park is what moved the
demote's landing onto tick B on this tree against the day-23 tree's 81.9 (C's open question) is not determined by
this cell and is not claimed.

**Proposal 1 (not implemented; for the lead's ruling): refuse the restore route by shape when the entry was promoted
for this admission.** In `host_restore_park_probe`, after `px.lookup` finds the hit `i` and before the class check:
if `hpx.promoted_pin` (the insertion pin of the promote published at THIS tick top, held until the next tick top
exactly so the parked request's re-admission finds its device hit, `host_promote_park_probe` and the tick top's
release) names entry `i`, return `false` with one typed line, `[prefix-cache] restore not routed (contracts door):
the entry was promoted for this admission; the tick program copies it`, and the request takes the OFF device-hit
copy on the tick. One predicate over existing state; no flag, no new state, no new engine seam, no numeric change
(the OFF program's copy, byte-identical destination). Acceptance gate, pre-registered here: this cell again
(promote arm, ON against OFF, twenty interleaved boots, N=5 per arm per order, both orders, one hold) with the
clauses (1) `request parked` per ON promote run reads 1 and `restore submitted` reads 0, 100 of 100; (2) the
request's ON e2e median drops by the re-admission median within the pair's unc (221.4 to about 131; `on_minus_off`
from +105.9 to about +15, the first park's own share); (3) the tenant's ON stall stays within IQR of 149.4 (the
second park cost the tenant nothing, so removing it must move nothing; a move either way is a finding); (4) the
day-21 restore arm (`--mode restore`, a hit on an entry NOT promoted this admission) still parks once and lands 100
of 100 with `restore landed` lines; (5) the hit gate OFF and ON `ALL GREEN` with every `spec==plain byte identity`,
the identity gates OFF and ON `ALL GREEN`. The larger alternative (submit the restore at the promote's publication
so one park covers both copies) still pays a tick top for a 0.5 ms copy and adds a submit site inside the publish;
it is named and not proposed.

## Task 2, the run (`pro-single-day25/box/gates/hitgate-on/`)

The typed line landed as pre-registered (`71ee2a64d`): `PendingCapture.settle_after_ms` and `settle_held_ms`,
stamped around the settle closure in `host_capture_settle_with`, printed inside the publish line's parenthesis.
CPU: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and server `Finished`
(2m 18s); the `DOCS_RS=1 --target x86_64-unknown-linux-gnu` pass `Finished`; server lib `809 passed; 0 failed; 14
ignored`; `check-flags: no uncovered runtime names`; markers `OK`; `git diff --check` clean.

The hit gate ON on the target card, the same binary `62b4f802…`, under the gate's own `flock` on
`/tmp/memra-gpu.lock` (`MEMRA_GPU_LOCK`), 12:11Z to 12:12Z right after the hold: `SPEC-ON-CACHE-HIT GATE: ALL GREEN
(qwen)`, 68 `ok:`, 0 FAIL, the five `spec==plain byte identity` lines ok, engagement `ok: door arm: the host tier is
armed on the spec-on boot`, `... the contracts door is ON ...`, `... no latch line ...` (both boots), `ok: door arm:
30 route submission(s) across the two boots`; census `capture_submitted=12 capture_published=12 restore_submitted=13
restore_landed=13 refused_contracts_door=0 restore_refused=0 latched=0` (spec-on) and `capture_submitted=2
capture_published=2 restore_submitted=3 restore_landed=3` (spec-off): day 24's counts exactly.

**The reading (`day25-retire-reading.py`, verbatim).**

```
DAY25 RETIRE-SETTLE lines=14 no_clause=0
DAY25 RETIRE-SETTLE settled_by='settled synchronously by a session retire' N=11 whys=['spec-boundary'] toks=[64, 96, 128] | held_ms N=11 min=0.37 median=0.41 max=0.44 | entered_after_ms N=11 min=106.10 median=150.50 max=204.50 | completion_ms N=11 min=106.50 median=151.00 max=205.00 | share_held_over_completion N=11 min=0.00 median=0.00 max=0.00
DAY25 RETIRE-SETTLE settled_by='tick-top poll' N=3 whys=['seed'] toks=[64] | held_ms N=3 min=0.37 median=0.37 max=0.38 | entered_after_ms N=3 min=87.30 median=87.60 max=87.70 | completion_ms N=3 min=87.60 median=88.00 max=88.00 | share_held_over_completion N=3 min=0.00 median=0.00 max=0.00
```

One line quoted whole: `[prefix-cache] capture published off the tick (spec-boundary): 64 tokens complete after 1
poll(s), 151.0ms from submission to completion, 151.0ms to publication (settled synchronously by a session retire;
the settle held the owner thread 0.40ms, entered 150.5ms after submission)`.

**Finding (owed item 3 priced).** The retire seam's `Block` held the owner thread **0.37 to 0.44 ms** (median 0.41,
N=11) on every spec-boundary capture of the gate, the same as the tick-top `Poll`'s 0.37 to 0.38 (N=3): the share of
the submission-to-completion span the owner thread waited is **0.2 to 0.4 percent**. The seam entered the settle
106 to 204 ms after submission, so the copy (about 0.5 ms with its receipt pair) had landed long before the session
retired and `synchronize` returned at once; the 0.4 ms is the settle's fixed cost (the events read, the receipt
lanes' D2H, retire, acknowledge), not a wait on the copy. Day 24's sentence "the owner thread still pays a host wait
for the 159 MB copy at the retire" is refuted by its own typed figure: the owner thread paid 0.4 ms. The
`106.3 to 204.6 ms from submission to completion` of day 24 was the session's own decode time between its prime
stop and its retire, never a wait. The moved share of the spec-boundary capture on this shape is therefore the
whole copy: nothing of it is on the owner thread beyond the settle's 0.4 ms and the submit.

**Proposal 2 (not implemented; for the lead's ruling): no reordering is earned on this shape.** A poll-then-retire
ordering at the seam (`Poll` first; `Block` only when the copy is still running) would not break the source-lifetime
rule (the `Block` branch still runs before any session leaves `active`), and it would not remove anything measurable
here: the `Block` already returns in the `Poll`'s time because the copy has landed. The ordering earns its place
only on a shape where a session retires INSIDE the copy's own 0.5 ms window (a prime stop and a retire on the same
tick with the copy stream busy), which no gate today produces. If the lead wants the seam to carry the distinction
anyway, the smallest form is a source-session identity on `PendingCapture` (so the seam blocks only for the source
session's retire and leaves another session's retire to the next tick-top poll), with the acceptance gate: the hit
gate ON on the target card with the `held` figure unchanged (0.4 ms) and the `settled_by` of every spec-boundary
capture still naming the retire (the source is the retiring session on this shape), plus a CPU state-machine test
that a non-source retire leaves the capture `Pending`. Recommended ruling: leave the seam as it is and record the
0.4 ms as the price.

## Task 3, records and checks

`DAY25.md` (this file), `STATE.md` rewritten, `OWNER-THREAD-OFFLOAD.md` day-25 section and owed list,
`research/INDEX.md` row `spill-a-20260919/day25`, C's `HOSTPREFIX-DOOR.md` section B two new rows (the double-park
pair, the retire share) and section A's owed-cell row for item 3. Checks after the records: `bash
tools/check-flags.sh`, `bash tools/check-conflict-markers.sh`, `git diff --check`, `.gitattributes` `*.log
-whitespace` in `pro-single-day25/`, zero em dashes in today's lines.

## Pushes

`c82b36d97` (the pre-registration, the drivers, the readers), `71ee2a64d` (the typed line; the tree the card ran),
then the closing commit (the receipts, this record's run sections, STATE, OWNER-THREAD-OFFLOAD, INDEX, the door
table), each in `MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU
qualification claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 reached through the existing control socket only (`ssh -O check`: `Master running`). `/root/wt-a` (mine) left
at `71ee2a64d` on `lane-a-day25`, clean; `/root/spill-receipts/a-day25/` kept on the box and mirrored here as
`pro-single-day25/box/` (bins not mirrored); the transfer bundle removed on both ends; `/root/artifacts`,
`/root/memra-spill`, other lanes' worktrees, receipts and processes not touched (lane C's server held the card when
the driver started; it was seen through the collector's refusal and `nvidia-smi` only, and the driver waited its
bounded retries). No server of mine running at close; no lock held by me; local `/tmp` scratch (the CPU battery's
script and log, the dry-test dir, the bundle) removed. Not run today: the local RTX 5090 door gates (owed with the
lock, as on days 18 to 24; the typed line changes no 5090-facing default).
