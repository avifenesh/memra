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
