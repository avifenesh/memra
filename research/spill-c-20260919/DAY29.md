# Session C day 29: decision cell (i) of Move 1, same window, both classes: the copy-stream program against the owner-stream program

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change today: research scripts and records only.

## Merge (first action)

PR #639 read `OPEN` (`gh pr view 639 --json state`; `mergeStateStatus UNSTABLE`), so `origin/main` was not merged.
`origin/lane/spill-integ38-20260922` at `cd7161bc7` (the two revuto fixes to the D2D fault arm) merged `--no-ff` as
`f5398f8fb`, clean, no conflict. Pushed in the announced development mode.

## Task 1, pre-registration (this section is committed before the cell runs)

**What is owed.** `research/spill-a-20260919/OWNER-THREAD-OFFLOAD.md`, "What Move 1 still owes", item 4: "The
decision cell (i) as pre-registered above, both classes, same window. Day 17 ran the demote arm and day 18 the
promote arm, each against the day-16 receipt on the same box across sittings; the same-window interleaved A/B of
both classes (door ON with the copy stream against door ON on the owner stream) has no arm today because the
owner-stream program left with the slices; its shape is the day-16 script's `on` boot against a build of the
day-16 tree, N=5 per arm per order, both orders." Cell (i) as A pre-registered it (the Move 1 section of the same
note): "The census stall cell repeated with the second stream: the tenant's ITL max during the demote and promote
arms against the day-16 baseline, N=5 per arm per order, both orders, door ON in both (the OFF arm is not affected
by the move). Decision clause: `stall_median(second stream) <= idle p99` of the same sitting for both classes, and
the identity clauses of C's `wc-pair.py` (`WC PAIR REPLAY: PASS`) unchanged." Today the day-16 baseline is not a
receipt from another sitting: it is an arm booted in the same lock hold from a build of the day-16 tree.

**The two arms, both door ON, the day-16 script's `on` boot in both.**

- Arm X, the copy-stream program: today's tree `f5398f8fb` (`crates/` = the integ38 tip `cd7161bc7` plus nothing),
  built on the box in `/root/wt-c`, booted with `MEMRA_KV_HOST_CONTRACTS=1`: the D2H demote submitted on the copy
  stream and polled at the tick top (A day 17), the H2D promote submitted on the copy stream with the request
  parked and re-admitted (A day 18), the parked-only wait (#627).
- Arm Y, the owner-stream program: the day-16 tree `1646d421b`, built on the box in its own worktree
  `/root/wt-c-day16` (a worktree of my box clone, never under `/root/memra-spill` or another lane's tree), booted
  with `MEMRA_KV_HOST_CONTRACTS=1`: the D2H and the H2D on the owner stream through `CudaTransfers` with
  `synchronize(&ticket)` and the owner drain before publication (the on-tick door). Its engine source equals
  `main` `653c997f4`: `git diff --stat 1646d421b 653c997f4 -- crates/ Cargo.toml Cargo.lock` prints nothing,
  verified in this checkout and on the box (`ev/day16-crates-equality.txt`). Both binaries' SHA-256 go into the
  receipt (`ev/binary-x.sha256`, `ev/binary-y.sha256`).

**Shape (target card, BOX3, one RTX PRO 6000 Blackwell at 600 W, the 27B `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, the
collector, ONE lock hold).** Every boot is the day-16 script's `on` boot: `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4
MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1`, then lane A's
harness `research/spill-a-20260919/stall_cell.py` byte-for-byte (its SHA-256 recorded), `--mode demote` then
`--mode promote`, each with the harness's own idle/arm interleave in both orders, N=5 per arm per order (10 arm
runs and 10 idle runs per class per boot). The entry shape is the day-16 script's: the demote intruder is a fresh
64-token grid seed whose insert evicts the previous entry into the host tier (159.9 MB); the promote intruder is the
day-15 alternation, a host hit of 64 tokens with a 22- to 25-token suffix whose insert evicts the other entry. The
two binaries cannot share a boot, so the interleave of the A/B is across BOOTS: order 1 boots `X Y X Y X Y X Y X Y`,
order 2 boots `Y X Y X Y X Y X Y X`, twenty boots in one collector hold, five boots per arm per order. Before the
twenty, one DRY boot of arm Y (boot, `/v1/models`, the harness `--mode demote --n 1`, stop, replay) proves the
day-16 tree boots and answers under today's artifact and CUDA; it is recorded under `ev/dry-y/` and is not part of
any quantity. Driver `day29-stall-cell.sh` under `day29-box-run.sh` (the collector `tools/tier-battery.py --rig
pro-single --external-lock`, bounded lock retries 60 x 120 s, the holder never inspected or signalled; lane A
shares the card today). Each boot's `card.{before,after}.csv`; the card's regime from the collector's
`command.gpu.csv` (250 ms samples across the whole hold).

**The quantity.** Per class (demote, promote), per arm (X, Y), per order (1, 2): the five boots' `stall_median`
(the harness's rule-line figure: the median over the boot's ten arm runs of `stall_ms` = the run's max ITL minus
its p50) give `cell_median` = the median of the five, with the five listed, their min, max and IQR (p75 minus p25
of the five). Pooled over both orders: the median of the ten. The same-window difference per class per order is
`y_minus_x = cell_median(Y) - cell_median(X)` with `unc = sqrt(IQR_X^2 + IQR_Y^2)`; `isolated` when
`|y_minus_x| > unc`, else `under_resolution`. `arm_p99` and `arm_max` are printed beside every boot (A's day-21
split signature is read as moved work across ticks, never as a removed share). `server_demote_ms` and
`server_promote_ms` per boot are recorded (under arm X they span submission to publication and are not the
tenant's cost; under arm Y they are the synchronous copy).

**The acceptance I will read.** Admissibility per receipt, the day-16 script's: `STALL REPLAY: PASS (replay agrees
with the harness's rule line)` from `stall_cell.py --replay` on every `receipt.json` (forty plus the dry boot's
one), `errors=0`, `tenant_text_identical=True`; the demote receipt carries at least nine `server_demote_ms`
entries and the promote receipt exactly ten `server_promote_ms` entries (one per arm run; the first fresh seed of
a demote boot demotes nothing, the day-16 shape); no `demote failed`, `promote failed`, `promote refused` or
`TIER DISABLED` line in any boot's `server.log`. An inadmissible receipt decides nothing and is reported as such;
its boot is excluded from the quantity and the N is quoted as it stands. Cell (i)'s decision clause is then read
on arm X per class: `stall_median(second stream) <= idle p99 of the same sitting`, with `stall_median(second
stream)` = arm X's pooled cell median and `idle p99 of the same sitting` = the largest idle p99 over the twenty
boots' receipts of that class (so no idle sample of the sitting is excluded); `clause_met` or `clause_not_met`
per class, and `CELL(i) CLAUSE: MET` only when both classes meet it. The identity half of the clause (`WC PAIR
REPLAY: PASS`) is not re-run today: the identity gate's `teeth=0` in both arms on both cards on every tree since
day 14 stands in the door table (section C) and is quoted, not re-measured. Reading script
`day29-stall-reading.py`, committed with this section; every constant in it is fixed here.

**Expected readings, stated before the run (context from other sittings on this box, never the reading).** Demote:
A day 16 ON `193.5` (the owner-stream program), A day 17 `149.6`, A day 18 `149.7`, C day 23 `149.5` (the
copy-stream program); so `y_minus_x` is expected about +44 ms, `isolated`. Promote: A day 16 ON `162.8`, A day 18
run 2 `81.9`, C day 23 `81.9`; so `y_minus_x` is expected about +81 ms, `isolated`. Idle p99 read 14.8 to 14.9 in
every stall cell on this box, so the clause `stall_median(X) <= idle p99` is expected `clause_not_met` for both
classes: the copy-stream program moved the copy off the tick and the tenant's tick still pays the two on-tick
hashes of the demote (about 78 ms per hash pass on this host, `pro-single-day18/hashmicro/`) and the promote's
restore; the cell measures what the tenant pays, not whether the design's clause was ever reachable. If the
readings differ from these expectations the numbers say so.

**What I will NOT do.** No tuning after a result: the scripts, the boot environment and the reading rules above
are frozen at this commit, and a second attempt happens only for a lock refusal or a boot failure, reported as an
attempt. No cross-sitting comparison as the primary reading: the day-16, day-17, day-18 and day-23 medians are
quoted beside the same-window pair as context only. No patch to the day-16 tree: if `1646d421b` fails to build or
to boot with today's artifact or CUDA, the failure is quoted and the cell stops (report, not repair). No change to
lane A's harness. No touch of `/root/artifacts`, `/root/memra-spill` or other lanes' worktrees and processes; no
third lock name; no bare GPU run; every boot inside the collector's one hold.

**Task 3, pre-stated.** `tools/spec-on-cache-hit-gate.sh --external-lock FD qwen <model> <bin> <ev>` on the target
card under the collector's hold, once, door OFF (the variable unset) and door ON (`MEMRA_KV_HOST_CONTRACTS=1`, the
gate exports `MEMRA_KV_HOST_MB=8192` and asserts the door engaged), in one collector hold (`day29-hitgate-cell.sh`),
arm X's binary; the verdict lines and each arm's `LOCK.json` (owner `collector`) are quoted verbatim. This is the
day-28 owed run of the arm that existed with no GPU receipt; it is a correctness gate, not a timing cell.
