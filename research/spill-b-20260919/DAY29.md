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
