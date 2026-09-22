# WP-B day 29: the 27B twin gate's `V3=FAIL` on the local RTX 5090 (A's day 18): reading, repro, the gate's typed line, the 5090 door gates

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `45c4c3cd2` (#628) at
`f0dc5d0d8` (`tools/check-conflict-markers.sh` OK; `research/INDEX.md` resolved as the union of both sides). Every
push today whose range touches engine source is refused `UNQUALIFIED` by the #589 hook on a plain push and goes out as
`MEMRA_RELEASE_QUALIFICATION_MODE=development git push` (`UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`,
logged in `.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this record**; every GPU cell is
`executed-not-qualified`; no timing is compared across cards; no default changes today.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU runs under the canonical
lock `/tmp/memra-5090.lock` (the twin gate takes it itself with `flock -n`, the door gates likewise; a busy lock is a
typed `REFUSED` retried boundedly), CPU-heavy work under `systemd-run --user --scope -p CPUQuota=1200% -p
MemoryMax=28G`, `nvidia-smi --query-compute-apps` read before and after every cell (today also at 1 Hz through the
twin cells), no process this lane did not start is touched (identification by cwd only; the lead's rule of
2026-09-21).

## 1. The reading from A's receipts, before any run (arithmetic on `rtx5090-day17/twin27-off` and `rtx5090-day18/twin27-off`)

A's day 18 (`research/spill-a-20260919/DAY18.md`, target-card and local tables) read the 27B twin gate on the local
card in BOTH door arms identically as `evictions=1 cohort_evictions=1 ... effective_free_ok=2/8 ... V3=FAIL -> FAIL`
with a constant V3 state error of `-410352980` B on turns 2 to 7 and `0` on turns 1 and 8, where day 17's local run
of the same cell read `evictions=9 -> PASS` and the target card reads `-> PASS` in both arms on the same trees. A
classified it as a local-card reading, cause not established, a repro with a process listing owed. The receipts A
committed carry the answer; this section states it before the repro so the repro can refute it.

**V3's form** (`tools/prefix-newest-turn-fits-gate.py`): after every turn `k`,
`calibration_effective_free_after(k) == measured_effective_free_after(k) + prefix_cache_bytes(k)` within 64 MiB, with
`effective free = cuda_driver_free_bytes + cuda_pool_cached_bytes` from `/metrics`. Its premise is that the two boots
(cache off; cache on) retain the SAME non-prefix-cache device state after every turn, so the cache's resident bytes
are the only difference. That premise holds while nothing else releases device state in one boot and not the other.

**What differs between the two days, from the server's own lines.** Day 17: neither boot prints an `[admit-oom]`
line. Day 18: the measured boot prints `[admit-oom] reclaim-on-defer: evicted 2 prefix entries + 1 plain + 0 spec +
0 dspark parked sessions (global LRU); effective free 3744MB -> 4651MB` at turn 2's admission (`measured/server.log`
line 81) and further `reclaim-on-defer` lines at turns 3, 4, 5, 6, 7 and 8 (`+ 1 plain` at 3, 6, 7, 8; `+ 0 plain` at
4 and 5); the calibration boot prints `reclaim-on-defer: evicted 0 prefix entries + 1 plain ...; effective free
4060MB -> 4471MB` at turn 3 (`calibration/server.log` line 69) and again at turn 8 (line 118). The reclaim runs in
`worker.rs` (`[admit-oom] reclaim-on-defer`, the `while !headroom.sufficient(required)` loop over
`evict_oldest_parked`) when the request's admission cost plus the reserve floor exceeds effective free: turn 2's
cost line reads `2683MB` both days and `[admit-trim] ... floor_bytes=2147483648`, so `required` is about 4830 MB;
day 17's effective free at that admission was `5207022852` B (clears it, no reclaim) and day 18's `3743931652` B
(does not; reclaim). The difference between the days at the same point is `1463091200` B (1395 MiB), and the
calibration boots show the same offset (`6133832964` against `4670741764` B after turn 1): the card had about 1.4 GB
less free on day 18 in both boots, before the gate allocated anything. A's receipts hold no process listing of the
card for that window (A recorded this); a 1390 MiB `colbert-2` python process is on the card at this sitting's
start (`rtx5090-day29/*/card-before.txt`), the size the offset names.

**What the constant is.** `410352980` B is one parked PLAIN session of the cohort's last shape: `ctx_cap 3272 x
29696 B/token = 97165312` B of KV plus `313187668` B of session-owned state, the admission line's `ctx=3272 ...
+ 313MB fixed` for the cohort's sends (`measured/server.log` lines 53, 62). The measured boot's reclaim at turn 2
released one such session (`+ 1 plain`; `3744 -> 4651` MB is `907` MB, of which `497188864` B is the two evicted
cohort prefix entries and the remaining `409.8` MB the plain session) while the calibration boot kept its
counterpart, so from turn 2 the measured boot's non-cache retained state is short by exactly that session and the V3
state error reads `-410352980` (calibration free minus measured free minus resident: the measured boot has MORE
free than the premise allows). Turn 3's reclaim released a cohort-shaped plain session in BOTH boots (calibration
`4060 -> 4471` MB, measured `3603 -> 4443` MB less its `429621248` B prefix entry), so the difference stood. Turn 8's
calibration reclaim (`4370 -> 4780` MB, the last cohort-shaped session) closed it: `0` on turn 8. Turns 4 and 5
reclaimed prefix entries only. Nothing here is a byte the arithmetic books and the driver does not release, or the
reverse: every released byte came back to `pool_cached` (`pool_cached_gain_bytes == evicted_prefix_bytes` on every
settle line). It is the two boots crossing the admission floor on different turns, because the measured boot
carries about 0.93 GB of cache resident that the calibration boot does not, on a card whose free space was 1.4 GB
lower than the day before.

**The day-27 budget change is not involved.** The gate sets `MEMRA_PREFIX_CACHE_MB=1024` and `MEMRA_CTX=16384`
explicitly in the server environment (`Server.boot`), so the budget is `configured by MEMRA_PREFIX_CACHE_MB`, never
derived; the boot line on both days reads `budget 1074MB (1073741824 B, configured by MEMRA_PREFIX_CACHE_MB)`.
`prefix_budget_ctx` and `MEMRA_CTX` unset versus set do not reach this gate. No comment on #539 is owed by this
finding.

**Classification, stated before the run.** A gate-environment fact on this card shape: the 27B at the 5090's 24 GB
(`24463 MiB`) with a 1 GiB prefix budget, `MEMRA_MAX_SESSIONS=4` and the plain reuse pool sits close enough to the
2 GiB reserve floor that a co-tenant of about 1.4 GB moves the growing tenant's admissions across it; the two boots
then cross on different turns and V3's premise (equal retained parked-session state) no longer holds. It is not a
door delta (identical in both arms), not a pool-accounting defect (every settle line balances), and not the
day-27 budget change. The target card at 96 GB never approaches the floor, which is why it reads `-> PASS`.

## 2. Pre-registration of the repro (written before the first run)

Cell: A's day-18 `twin27-off` command verbatim (`python3 tools/prefix-newest-turn-fits-gate.py --model <27B> --bin
<server> --out <dir>`, `MEMRA_GPU_LOCK=/tmp/memra-5090.lock`, door OFF), the 27B artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, the server built from `f0dc5d0d8`, three runs (`run1`, `run2`, `run3`), the
gate UNCHANGED for these three runs (its hash recorded). Beside each run: `nvidia-smi --query-gpu=memory.free` and
`--query-compute-apps` before the boot, at 1 Hz through both boots and every turn (`samples.log`), and after
(`run-day29-repro.sh`). Reading rule, fixed now:

- **Reproduces** if the verdict line reads `V3=FAIL` with `effective_free_ok < 8/8` and a `reclaim-on-defer` line
  with `+ N plain` (N >= 1) appears in one boot's turn window without its counterpart in the other boot's same turn
  window; the V3 state error on those turns must equal the released parked sessions' bytes to within the 64 MiB
  slack (the section-1 arithmetic, checked on the new run's own lines). The co-tenant listing names what was on
  the card. If the card carries a co-tenant of about 1.4 GB during the run, the section-1 reading is confirmed as
  the environment; if the card is clean and it still reproduces, section 1 is refuted on the co-tenant half and the
  floor crossing must be explained from the run's own free-at-boot.
- **Does not reproduce** (`-> PASS`, `evictions=9`, no `reclaim-on-defer` in either boot) under a card with no
  co-tenant: the co-tenant at A's run time is named as the candidate, with A's own statement that no listing
  exists, and one more run under the day-17 environment (a clean card, no co-tenant) closes the loop.
- Anything else (a refusal, a died server, a different clause failing) is recorded verbatim as a labelled failure
  and is not read as either.

Consequence for the gate, decided now so that it is not decided after the result: V3's clause, slack and form do
NOT change. If the repro confirms section 1, the gate gains a typed refusal, emitted BEFORE the verdict line, when
the two boots' `reclaim-on-defer` parked-session releases differ on any turn (the premise of V3's state equation
does not hold), naming the turns, the released bytes per boot, the driver free at each boot's start, the card total
and the compute-apps listing at that moment: `REFUSED: V3 premise: ...`. A gate whose premise did not hold is
`REFUSED`, never `PASS`, and never `FAIL` on a clause it could not evaluate. Two boots that reclaim IDENTICALLY (same
turns, same bytes) keep V3 evaluated as today. A CPU test replays A's day-18 lines through the detector.

## 3. The repro: three runs, gate unchanged, door OFF, the card sampled through every turn

Tree `26523b686` (the pre-registration commit; the binary from `f0dc5d0d8`, `a26c1180b5fc…`, `run*/binary.sha256`),
gate `tools/prefix-newest-turn-fits-gate.py` at its pre-patch hash (`rtx5090-day29/gate-before-patch.sha256`), A's
day-18 `twin27-off` command verbatim, `MEMRA_GPU_LOCK=/tmp/memra-5090.lock`, the 27B artifact. Receipts:
`rtx5090-day29/run{1,2,3}/` (the gate's `twin27-off/` bundle with both server logs, `TURNS.md`, `summary.json`;
`card-before.txt`, `card-after.txt`, `samples.log` at 1 Hz, `driver.log`). Every run `executed-not-qualified`, N=1
each, the laptop card's regime not recorded by the gate (`power.limit [N/A]`).

| run | start (UTC) | verdict line, verbatim | rc | co-tenant on the card |
|---|---|---|---|---|
| run1 | 04:27:26 | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=1 cohort_evictions=1 self_evictions=0 refused_or_skipped=0 effective_free_ok=2/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=FAIL V4=ok V5=ok V6=ok -> FAIL` | 1 | one `python` process (a `colbert-2` venv, not this lane's), 1390 MiB, on 139 of 139 samples, before and after |
| run2 | 04:31:30 | the identical line, `-> FAIL` | 1 | the same process, 1390 MiB, 130 of 130 samples |
| run3 | 04:34:27 | the identical line, `-> FAIL` | 1 | the same process, 1390 MiB, 130 of 130 samples |

**Reproduces, 3 of 3, and to the byte.** Every `effective free after` and `cache-off effective free after` value in
all three `TURNS.md` equals A's day-18 table (`3743931652`, `4670741764` after turn 1; `3602940420`, `4060382384`
after turn 2; …), every V3 state error reads `-410352980` on turns 2 to 7 and `0` on turns 1 and 8, and the
reclaim lines match A's line for line (`verify-day29.py` rows, all three runs):

```
turn 2: v3_error=-410352980 measured_reclaims=[(2, 1, 0, 0, 3744, 4651)] calibration_reclaims=[]
turn 3: v3_error=-410352980 measured_reclaims=[(1, 1, 0, 0, 3603, 4443)] calibration_reclaims=[(0, 1, 0, 0, 4060, 4471)]
turn 4: v3_error=-410352980 measured_reclaims=[(1, 0, 0, 0, 3976, 4414)] calibration_reclaims=[]
turn 5: v3_error=-410352980 measured_reclaims=[(1, 0, 0, 0, 3938, 4385)] calibration_reclaims=[]
turn 6: v3_error=-410352980 measured_reclaims=[(1, 1, 0, 0, 3899, 4983)] calibration_reclaims=[]
turn 7: v3_error=-410352980 measured_reclaims=[(1, 1, 0, 0, 3861, 4965)] calibration_reclaims=[]
turn 8: v3_error=0          measured_reclaims=[(1, 1, 0, 0, 3824, 4944)] calibration_reclaims=[(0, 1, 0, 0, 4370, 4780)]
```
(tuples: prefix entries, plain, spec, dspark parked sessions released; effective free MB before -> after.)

Reading against the section-2 rule: **reproduces**, with the co-tenant of the predicted size on the card for the
whole of every run, so the section-1 reading stands as the environment: a 1390 MiB (`1457520640` B) process
beside the server is the `1463091200` B by which day 18's free-at-turn-1 fell short of day 17's (the 5.6 MB
remainder is the context's own boot-to-boot variation), and byte-identical state values on the same tree, on a
different day, with the same co-tenant footprint, say the card's free space at boot was the same to the byte on
A's day 18 as here. The measured boot's turn-2 reclaim released `(4651 - 3744) MB - 497188864 B = 409811136` B of
parked plain session (`+ 1 plain`), `541844` B from `410352980` (the log line's MB rounding), inside the 64 MiB
slack, and `3272 x 29696 + 313187668 = 410352980` exactly (`test-day29.py::Arithmetic`). **Named:** the parked
plain session of the cohort's last shape (`ctx_cap 3272`), which the cache-on boot's admission reclaimed at turn 2
and the cache-off boot kept until its own turn-3 reclaim took a different one, so the two boots' retained
parked-session sets differed by that session from turn 2 to turn 7. Not a door delta, not the day-27 budget
change (configured budget, section 1), not a pool-accounting defect (every settle line balances). The day-17
environment (a clean card) could not be reproduced at this sitting: the co-tenant is another session's process
and is never touched by this lane; the clean-card run stays owed and is bounded (one cell, the first time the card
reads no co-tenant), with the section-1 prediction on record: `evictions=9 ... -> PASS`, no `reclaim-on-defer` in
either boot.

Same-card note for the lead: A's day-17 local table and today's differ by the co-tenant alone; the 96 GB card
never approaches the floor (`required` about 4.8 GB against about 60 GB effective free at the same admission), so
the target-card `-> PASS` in both arms (A's day 18 and 19) is the same gate reading a shape that cannot cross.

## 4. What changed: the gate names the broken premise instead of `V3=FAIL` (`tools/prefix-newest-turn-fits-gate.py`)

Decided in section 2 before the runs and applied after run 3 (`apply-day29-gate-patch.py`, exact-string edits, each
once; `rtx5090-day29/gate-before-patch.sha256` and `gate-after-patch.sha256`). **V3's clause, form and 64 MiB slack
are unchanged**; no clause of V1..V6 moved; no new flag; no engine change today.

- The `[admit-oom] reclaim-on-defer: evicted N prefix entries + P plain + S spec + D dspark parked sessions ...;
  effective free A MB -> B MB` line is parsed (`RE_RECLAIM_PARKED`, beside the existing `RE_RECLAIM` counter) into
  each window's `parked_releases`; the calibration boot's cohort sends are parsed into windows too (they were
  discarded before), so both boots have one window per send: three cohort lengths x two sends, then the turns.
- `v3_premise_rows(cal, rec)`: window by window, the parked sessions each boot's reclaim released per pool;
  `equal` when the tuples match. The premise holds when every row is equal (no reclaim anywhere, as on the 96 GB
  card and on day 17 locally; or the same releases in both boots).
- Before the verdict is printed, if any row is unequal the gate writes `summary.json` with `v3_premise` (the rows),
  `verdict_under_broken_premise` (the V1..V6 line it would have printed) and `VERDICT.txt` as the refusal, prints
  the turn table and the unequal rows, then `REFUSED: V3 premise: ...` (exit 2) naming the windows, the releases per
  boot, the budget, and the card at each boot. Two boots that release identically print the verdict line exactly
  as before. A broken premise is never `PASS`, and never `FAIL` on a clause the gate could not evaluate.
- `Server.boot()` samples the card (`nvidia-smi` driver free of total in MiB, the compute-apps listing) right
  before it spawns the server; the samples sit in `calibration.json` (`card_at_boot`) and `summary.json`
  (`boot.card_at_boot`). The refusal's brief carries process basenames and MiB only; the listing itself stays in
  the JSON.
- The docstring's assertion list gains a "V3 premise" paragraph with the exit-2 meaning and today's incident.

CPU test `test-day29.py` (13 ok, run against the tree's gate): A's day-18 lines parse with their counts and free
move; the day-18 histories break the premise on turns 2, 6 and 7 only (turns 3 and 8 released one plain session in
BOTH boots and hold); day-17 (no reclaim) and identical-reclaim histories hold; a calibration-only release and a
cohort-send release are rows of their own; the refusal is typed, names the unequal windows and the card, carries no
full process path and no `PASS`; `3272 x 29696 + 313187668 == 410352980`; the turn-2 free move minus its prefix
bytes is the constant within the MB rounding; `V3_SLACK == 64 MiB`.

Expected readings on the current tree, stated before the cells: local RTX 5090 with the co-tenant present, both
arms: `REFUSED: V3 premise: ... 3 window(s) (turn 2: ...; turn 6: ...; turn 7: ...)` (exit 2, the same
`verdict_under_broken_premise` line as section 3 inside `summary.json`); target card, both arms: the unchanged
`... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` (no reclaim, every premise row equal).

## 5. The 5090 door gates on the current tree (owed since A's day 18)

Driver `rtx5090-day29/driver.sh` (A's day-18 local driver verbatim, at A's depth; attempt 1 at one level up died
`rc=127` on every cell before any gate ran, kept as `doors-attempt1-driver-depth/`), tree `1dce64df0`, binary
`a26c1180b5fc…`, the Qwen3.5-9B NVFP4 MTP artifact with `MEMRA_HOSTGATE_CACHE_MB=64`, the 27B for the `twin27`
cells, door OFF = env unset, ON = `MEMRA_KV_HOST_CONTRACTS=1`, every gate on the canonical lock itself. 04:37:51 to
04:52:03 UTC. Card before: the 1390 MiB co-tenant present (`doors/card-before.txt`); it left the card between the
9B `twin-on` cell (04:43:23) and the `twin27-off` calibration boot (04:45:53, `compute_apps []`, free `23967` of
`24463` MiB, sampled by the patched gate); card after: no compute-apps (`doors/card-after.txt`). Every cell
`executed-not-qualified`, N=1, the laptop card's regime not recorded by the gates.

| cell | verdict line, verbatim | ok / fail |
|---|---|---|
| identity-default-off | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (the arm A's day 18 never ran: `REFUSED: canonical GPU lock busy` 30 times) | 12 / 0 |
| identity-default-on | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-off (`MEMRA_SERVE_SPEC=0`) | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| identity-plain-on | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` | 12 / 0 |
| failure-off | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| failure-on | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` | 15 / 0 |
| contract-fault (ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | 65 / 0 |
| hitgate-off | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| hitgate-on | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` | 61 / 0 |
| twin-off, twin-on (9B) | `REFUSED: cohort promotion did not happen for 2800 tokens: second send cached=2800 of 2800, published 2784` (the 9B refuses the gate's cohort shape, A's day-17 gate-shape fact; both arms, rc=2) | - |
| gpu-unit-cells (`option_b_*`, `option_c_*`) | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 795 filtered out; finished in 0.26s` | 8 / 0 |
| twin27-off (27B, patched gate, clean card) | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | PASS |
| twin27-on (27B, patched gate, clean card; one `lock busy` retry at 04:47:58, ran at 04:49:58) | the identical line, `-> PASS` | PASS |

**The loop closed on its own.** The `twin27-off` cell is the section-2 "day-17 environment" run: a clean card
(`compute_apps []` at both boots), `evictions=9 cohort_evictions=3 effective_free_ok=8/8 -> PASS`, every V3 state
error `0`, no `reclaim-on-defer` in either boot, premise rows all equal, and `effective free after` turn 1
`5207022852` B (measured) and `6133832964` B (calibration): A's day-17 values to the byte. With the co-tenant
(section 3): A's day-18 values to the byte. One card, one tree, one binary, the co-tenant the only difference, the
outcome flipping with it in both directions.

**Target card** (one RTX PRO 6000 Blackwell, 600 W, the collector `--rig pro-single`, `executed-not-qualified`,
`pro-single-day29/box/`, tree `1dce64df0`, binary `7fa0b944b1ad…`, gate `1b623e1961c9…`): `twin27-off` and
`twin27-on` both read the identical `... evictions=9 cohort_evictions=3 ... effective_free_ok=8/8 ... V1=ok V2=ok
V3=ok V4=ok V5=ok V6=ok -> PASS`; premise rows all equal; `compute_apps []` at every boot, free `97250` of `97887`
MiB. Elapsed 04:42:20 to 04:44:46 UTC under one collector hold; no timing compared across cards.

## 6. The refusal arm on real hardware (pre-registered here, run after)

The patched gate's `REFUSED: V3 premise` path ran only in the CPU test today: the real co-tenant left the card
before the patched gate reached the twin cells (section 5). To exercise it on the card under one lock hold, a
fault arm: the collector (`tools/tier-battery.py --rig rtx5090 --external-lock`) holds the canonical
`/tmp/memra-5090.lock`; the cell (`rtx5090-day29/cotenant-cell.sh`) starts its OWN child, `cotenant.cu` (nvcc,
built to `/tmp`, never committed: one `cudaMalloc` of the requested MiB, then `pause()` until this cell stops it),
waits until `nvidia-smi --query-compute-apps` lists it, runs the 27B twin gate door OFF with `--external-lock`,
stops the child, and samples the card before, with the child up, after the gate and after. The child is the only
process this cell starts or stops; nothing else on the card is touched. Requested footprint 1390 MiB (the day-18
co-tenant's reading); the driver's own reading of the child (context included) is what the card sample records.

Expected, stated before the run: `REFUSED: V3 premise: ...` (exit 2) with `turn 2` among the unequal windows,
`summary.json` carrying `verdict_under_broken_premise` with `V3=FAIL`, both boots' `card_at_boot` listing the
child, and the first unequal window's plain release bytes equal to |V3 state error| within the 64 MiB slack. The
exact set of crossing turns may differ from day 18 if the child's footprint differs from 1390 MiB by the context
overhead; the refusal must fire regardless as long as any window differs. A `-> PASS` with the child present means
the floor was not crossed at this footprint and is recorded as such (not a refutation of section 1, whose crossing
is on the receipts); a `-> FAIL` would mean the detector missed a differing window and is a defect in the patch.

**Result** (`rtx5090-day29/fault-cotenant/`, collector `command.capture.json` `executed-not-qualified`, `LOCK.json`
owner `collector` on `/tmp/memra-5090.lock`, 04:54:25 to 04:56:30 UTC). The child read `1644 MiB` on the card
(1390 MiB allocated plus the context), listed at both boots (`card_at_boot` in `calibration.json` and
`summary.json`: `free/total MiB 22318, 24463, compute-apps 1 [cotenant 1644 MiB]`); the card after: `18 MiB` used,
no compute-apps. The gate, verbatim (exit 2):

`REFUSED: V3 premise: `[admit-oom] reclaim-on-defer` released parked sessions in one boot without the counterpart
in the other on 6 window(s) (turn 2: calibration released plain/spec/dspark 0/0/0, measured 1/0/0; turn 4: ...
0/0/0, measured 1/0/0; turn 5: ...; turn 6: ...; turn 7: ...; turn 8: calibration released plain/spec/dspark 0/0/0,
measured 1/0/0); the state equation compares boots whose retained parked-session sets differ, so V3 is not
decidable on this card shape (budget 1073741824 B, MEMRA_CTX 16384, MEMRA_MAX_SESSIONS 4); card at the calibration
boot: free/total MiB 22318, 24463, compute-apps 1 [cotenant 1644 MiB]; at the measured boot: free/total MiB 22318,
24463, compute-apps 1 [cotenant 1644 MiB]; the clause verdicts are kept in summary.json
(verdict_under_broken_premise), not printed as a verdict` (the full line in `twin27-off/VERDICT.txt`).

`verdict_under_broken_premise`: `... evictions=1 cohort_evictions=1 ... effective_free_ok=7/8 ... V3=FAIL ... ->
FAIL`; V3 state errors `[0, -410352980, 0, 0, 0, 0, 0, 0]`. Turn 2's measured reclaim is the day-18 line to the
byte (`evicted 2 prefix entries + 1 plain ...; effective free 3744MB -> 4651MB` less `evicted_prefix_bytes=497188864`
gives `409811136` B against `|-410352980|`, inside the slack) with the same constant; the crossing set differs from
day 18 as allowed for (the child's 1644 MiB against 1390: effective free after turn 1 `3477593348` / `4404403460`
B against day 18's `3743931652` / `4670741764`, a `266338304` B = 254 MiB offset, the context), and the
calibration boot's single reclaim (turn 3, prefix-only in the cache-off boot's own bookkeeping) left the two boots
equal again from turn 3 on. Every expectation of the pre-registration held: typed refusal, exit 2, turn 2 unequal,
`V3=FAIL` kept inside `summary.json`, the child at both boots, the bytes within the slack. The red arm of the check
is now on a receipt from the card itself.

## 7. Records, checks, budget

`STATE.md` rewritten (day 29, 8 lines), `research/INDEX.md` row `spill-b-20260919/day29`, this record. Comment on
memra#536 (A's thread) with the finding and the gate change; no comment on #539 (the budget change is not involved,
section 1). Checks on the staged tree: `cargo fmt --all -- --check` rc=0 (no Rust touched today, so no touched crate
for clippy; the merge's engine files are main's); `tools/check-flags.sh` every runtime `MEMRA_*` name resolves (no
new read today); `tools/check-conflict-markers.sh` OK; `python3 tools/check-public-boundary.py check` 590
grandfathered, 0 new; `test-day29.py` 13 ok against the tree's gate; `git diff --cached --check` names one line,
`rtx5090-day29/doors/gpu-unit-cells.log:19: new blank line at EOF`, a raw cargo receipt kept verbatim (the lead
recorded the same on A's cargo logs in integ32). Scratch: `/tmp/spillb29/` (A's extracted receipts, the scratch
gate copy, the co-tenant binary, build and collector logs) removed at close; the box keeps
`/root/spill-receipts/b-day29/` (mirrored here without `bins/`), no server of mine left running on either card, the
box worktree `/root/wt-b` at `1dce64df0` on `lane/spill-b-20260919`.

Budget: about 0.8 agent-hours of wall clock against 4 (the merge and the reading from A's receipts 0.2, the
pre-registration and runners 0.15, the three repro runs with the patch and its CPU test written beside them 0.2,
the door battery and the target-card cells in parallel with the records 0.15, the fault arm and the records 0.1).
Blockers: none that stopped a cell; the local lock was busy once for 120 s (`twin27-on`, retried); the box lock
was free. Open: the co-tenant's identity is another session's and is never touched by this lane; the pre-grow
booking point and the door decide-by dates from day 28 are unchanged.
