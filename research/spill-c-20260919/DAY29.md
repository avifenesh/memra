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

## Task 2, the run (target card, one collector hold, `pro-single-day29/`)

**Builds and identity.** Both trees built on the box before the hold: arm X in `/root/wt-c` at `f5398f8fb`
(`build-x.log` `rc=0`, `Finished release in 50.17s`, incremental over the day-28 target), binary
`3d7baaa0c1598faea61b1c56ad515543e335abcf51664b1fca0e8cf49cee551b`; the cell then ran with the tree checked out at
`7349ef932` (the pre-registration commit, research files only: `git diff --stat f5398f8fb 7349ef932 -- crates/` is
empty), so `ev/CELL.txt` reads `tree_x=7349ef932` and the binary is the same. Arm Y in its own worktree
`/root/wt-c-day16` at `1646d421b` (`build-y.log` `rc=0`, `Finished release in 3m 27s`, a full build; cargo 1.97.1,
CUDA 13.2, `MEMRA_CUDA_ARCH auto-detected 120a`), binary
`a8d737c0b4966c8db4305cbe985c91b52d912d979326e8bd694b117ad6f74cc7`; lane A's day-16 build of the same source read
`a447fec4…` (`pro-single-day16/box/stall-*/ev/binary.sha256`), a different build of one source, recorded not
explained. `ev/day16-crates-equality.txt`: `git diff --stat 1646d421b 653c997f4 -- crates/ Cargo.toml Cargo.lock`
prints nothing, `lines=0`, on the box and in this checkout. Harness `stall_cell.py` SHA-256 `13867e77da40a9b5…`,
equal to this checkout's. Model `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`. `ev/LOCK.json`: `{"owner": "collector", "lock":
"/tmp/memra-gpu.lock", "mechanism": "inherited-flock-same-open-description", ...}`. `compute-apps.{before,after}.csv`
empty (the card was mine alone inside the hold). The collector's `CELL.jsonl`: `status: executed-not-qualified`,
`exit_code 0`, `elapsed_seconds 2252.4`, `power.limit 600.00 W`; `tools/tier-battery.py --validate` rc=0 on the
cell dir. The lock was free at launch (lane A's `a-day23` server had left the card; no retry, `lock-retries.log`
absent).

**The dry boot of the day-16 tree** (`ev/dry-y/`): booted in 14 s under today's artifact and CUDA, answered
`/v1/models`, ran `--mode demote --n 1`: `STALL rule cell=stall-demote-y arm=demote n_per_order=1 pooled=2 ...
stall_median=134.3 stall_min=78.4 stall_max=190.2 server_demote_ms=[114.1] ... tenant_text_identical=True errors=0`,
`STALL REPLAY: PASS`. Part of no quantity. The day-16 tree did not need a patch.

**Admissibility.** `replays.log`: 41 `STALL REPLAY: PASS (replay agrees with the harness's rule line)`, zero FAIL
(the dry boot plus twenty boots times two classes); `errors=0` and `tenant_text_identical=True` in every receipt;
every demote receipt 9 `server_demote_ms` entries and every promote receipt 10 `server_promote_ms` entries; zero
`demote failed`, `promote failed`, `promote refused` or `TIER DISABLED` lines in any of the 21 `server.log`s;
`ADMISSIBLE all_receipts=True`. The tenant's 160-token text hashed `264b120d487de2c9` in all 41 receipts of BOTH
trees (and in the day-23 receipts on main's tree): recorded, not claimed.

**Regime.** The collector's sampler across the whole hold (`collector/stall-cell-i/command.gpu.csv`, 8981 samples
at 250 ms): 33 to 51 C, 32.5 to 360.8 W under the 600 W limit, 0 to 17109 MiB used; the 42 per-boot
`card.{before,after}.csv` snapshots 33 to 46 C, 32 to 114 W (the card idles between boots).

**Verdict lines, verbatim (`stall/reading.log`; ms; N = boots per arm per order, each boot's `stall_median` over
its ten arm runs).**

- `DAY29 CELL(i) class=demote arm=x order=o1 N=5 boots=['b01-x', 'b03-x', 'b05-x', 'b07-x', 'b09-x'] stall_medians=[150.0, 150.2, 149.9, 150.0, 149.9] cell_median=150.0 min=149.9 max=150.2 IQR=0.1 arm_p99s=[131.7, 131.2, 131.0, 131.5, 131.3] arm_maxs=[164.9, 163.9, 163.8, 163.8, 163.7]`
- `DAY29 CELL(i) class=demote arm=x order=o2 N=5 boots=['b12-x', 'b14-x', 'b16-x', 'b18-x', 'b20-x'] stall_medians=[149.9, 150.0, 150.0, 150.2, 150.1] cell_median=150.0 min=149.9 max=150.2 IQR=0.1 arm_p99s=[131.0, 131.6, 131.7, 131.5, 131.5] arm_maxs=[163.8, 163.6, 163.8, 164.6, 163.9]`
- `DAY29 CELL(i) class=demote arm=y order=o1 N=5 boots=['b02-y', 'b04-y', 'b06-y', 'b08-y', 'b10-y'] stall_medians=[193.5, 194.0, 193.4, 192.8, 193.0] cell_median=193.4 min=192.8 max=194.0 IQR=0.6 arm_p99s=[87.6, 87.7, 87.7, 87.7, 87.7] arm_maxs=[207.6, 207.7, 207.4, 206.9, 207.2]`
- `DAY29 CELL(i) class=demote arm=y order=o2 N=5 boots=['b11-y', 'b13-y', 'b15-y', 'b17-y', 'b19-y'] stall_medians=[193.0, 193.1, 193.1, 193.1, 192.8] cell_median=193.1 min=192.8 max=193.1 IQR=0.1 arm_p99s=[87.6, 87.7, 87.7, 87.7, 87.7] arm_maxs=[207.5, 207.7, 207.5, 206.7, 207.8]`
- `DAY29 CELL(i) class=demote order=o1 y_minus_x=+43.4 unc=0.6 -> isolated (Y owner stream 193.4 N=5; X copy stream 150.0 N=5)`
- `DAY29 CELL(i) class=demote order=o2 y_minus_x=+43.0 unc=0.1 -> isolated (Y owner stream 193.1 N=5; X copy stream 150.0 N=5)`
- `DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=150.0 idle_p99_sitting=14.9 -> clause_not_met`
- `DAY29 CELL(i) class=promote arm=x order=o1 N=5 boots=['b01-x', 'b03-x', 'b05-x', 'b07-x', 'b09-x'] stall_medians=[149.5, 149.7, 149.4, 149.6, 149.5] cell_median=149.5 min=149.4 max=149.7 IQR=0.1 arm_p99s=[23.2, 22.6, 22.7, 22.6, 22.6] arm_maxs=[163.4, 163.5, 163.2, 163.5, 163.6]`
- `DAY29 CELL(i) class=promote arm=x order=o2 N=5 boots=['b12-x', 'b14-x', 'b16-x', 'b18-x', 'b20-x'] stall_medians=[149.4, 149.6, 149.7, 149.7, 149.7] cell_median=149.7 min=149.4 max=149.7 IQR=0.1 arm_p99s=[22.8, 22.7, 23.1, 22.8, 22.8] arm_maxs=[163.2, 163.4, 163.6, 163.5, 163.5]`
- `DAY29 CELL(i) class=promote arm=y order=o1 N=5 boots=['b02-y', 'b04-y', 'b06-y', 'b08-y', 'b10-y'] stall_medians=[162.2, 162.9, 163.5, 162.6, 162.9] cell_median=162.9 min=162.2 max=163.5 IQR=0.3 arm_p99s=[16.6, 16.6, 16.6, 16.6, 16.6] arm_maxs=[211.2, 212.6, 212.0, 211.0, 211.1]`
- `DAY29 CELL(i) class=promote arm=y order=o2 N=5 boots=['b11-y', 'b13-y', 'b15-y', 'b17-y', 'b19-y'] stall_medians=[162.5, 162.4, 162.6, 162.5, 162.8] cell_median=162.5 min=162.4 max=162.8 IQR=0.2 arm_p99s=[16.6, 16.6, 16.6, 16.6, 16.6] arm_maxs=[210.9, 211.7, 211.4, 211.6, 211.2]`
- `DAY29 CELL(i) class=promote order=o1 y_minus_x=+13.3 unc=0.4 -> isolated (Y owner stream 162.9 N=5; X copy stream 149.5 N=5)`
- `DAY29 CELL(i) class=promote order=o2 y_minus_x=+12.8 unc=0.2 -> isolated (Y owner stream 162.5 N=5; X copy stream 149.7 N=5)`
- `DAY29 CELL(i) CLAUSE class=promote stall_median(second stream)=149.6 idle_p99_sitting=14.9 -> clause_not_met`
- `DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False admissible=True); executed-not-qualified`

Pooled over both orders: demote X `N=10 cell_median=150.0 min=149.9 max=150.2 IQR=0.1`, Y `N=10 cell_median=193.1
min=192.8 max=194.0 IQR=0.4`; promote X `N=10 cell_median=149.6 min=149.4 max=149.7 IQR=0.2`, Y `N=10
cell_median=162.6 min=162.2 max=163.5 IQR=0.4`. One boot's rule lines per arm, verbatim (the other eighteen are in
`replays.log`): `STALL rule cell=stall-demote-x arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5
idle_p95=14.8 idle_p99=14.9 idle_max=15.0 arm_runs=10 arm_p50=13.5 arm_p95=14.9 arm_p99=131.7 arm_max=164.9
stall_median=150.0 stall_min=76.6 stall_max=151.5 server_demote_ms=[128.8, 133.4, 133.1, 132.9, 133.7, 134.2, 132.8,
132.9, 133.0] server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97]
tenant_text_identical=True errors=0`; `STALL rule cell=stall-promote-x arm=promote ... arm_p99=23.2 arm_max=163.4
stall_median=149.5 stall_min=149.3 stall_max=149.9 server_demote_ms=[133.5, 133.1, 98.3, 97.6, 97.8, 97.6, 97.7,
97.7, 97.6, 97.7] server_promote_ms=[63.1, 62.4, 27.5, 26.9, 26.9, 26.8, 26.9, 26.7, 26.7, 26.9]
intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`; `STALL rule
cell=stall-demote-y arm=demote ... arm_p99=87.6 arm_max=207.6 stall_median=193.5 stall_min=75.8 stall_max=194.2
server_demote_ms=[113.2, 118.4, 118.2, 117.5, 118.0, 117.7, 117.6, 117.7, 117.3] ...`; `STALL rule
cell=stall-promote-y arm=promote ... arm_p99=16.6 arm_max=211.2 stall_median=162.2 stall_min=161.7 stall_max=197.8
server_demote_ms=[116.8, 117.6, 82.7, 81.9, 81.8, 81.9, 81.9, 81.8, 81.8, 81.8] server_promote_ms=[122.7, 123.4,
88.6, 87.8, 87.7, 87.7, 87.8, 87.6, 87.6, 87.6] ...`.

**What the same window says (the reading, nothing tuned).** Demote class: the copy stream takes **43.4 / 43.0 ms**
off the tenant's stall (150.0 against 193.4 / 193.1, `isolated`, unc 0.6 / 0.1), the day-16 tree reproducing A's
day-16 figure (193.5) to within a millisecond and today's tree reproducing days 17, 18 and 23 (149.6, 149.7, 149.5).
Promote class: the copy stream takes **13.3 / 12.8 ms** off (149.5 / 149.7 against 162.9 / 162.5, `isolated`, unc
0.4 / 0.2); the day-16 tree reproduces A's day-16 promote figure (162.8). Cell (i)'s clause is **NOT MET** for both
classes: the second stream leaves the tenant's stall an order of magnitude above idle p99 (14.9), as expected in the
pre-registration. Under arm X the demote arm's `arm_p99` reads 131 against Y's 87.7 with `arm_max` 164 against 207:
the ON demote's worst tick is shorter and the second-worst tick longer, the moved-work-across-ticks signature (the
submission tick and the poll tick both pay), read as such and not as a removed share.

**Finding, stated for the lead, not tuned.** Arm X's promote arm reads **149.6**, where the trees of A day 18 run 2
(`3df0cb2b3`) and C day 23 (`91b0d4e08`, `crates/` = main `0713c1a79`) read **81.9** in the same shape; the
pre-registration expected about 82. The lines say why the shape differs (`o1/b01-x/promote/receipt.json`
`server_log_lines`, the same in all 100 promote runs of the ten X boots): on today's tree the hit parks TWICE:
`[prefix-host] promote submitted off the tick: 64 tokens, 158.8MB, ticket seq=33, 32 items on the contracts door's
copy stream; request parked`, `promote published off the tick: ticket complete after 1 poll(s), 19.7ms from
submission to completion`, `[prefix-host] promote: 64 tokens, 158.8MB in 26.9ms`, then `[prefix-cache] restore
submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=35 on the contracts door's copy stream; recurrent
state copied on the owner stream; request parked` and `[prefix-cache] restore landed off the tick: 64 tokens
(158.8MB) complete after 1 poll(s), 90.2ms from submission to completion, 90.3ms to re-admission` (Move 2's restore
route, absent on the day-23 tree: 100 `restore submitted` lines in X's promote runs, 0 in Y's; re-admission 89.8 to
90.8 ms, median 90.3, 100 of 100). Between the two parks the intruder's insert demotes the other entry:
`demote submitted off the tick ... ticket seq=34`, `demote published off the tick: ticket seq=34 complete after 1
poll(s), 22.3ms from submission to completion` (22.2 to 58.7, median 22.4, 100 of 100; the day-23 tree read
`97.0ms`), `[prefix-host] demote: 64 tokens, 159.8MB in 97.6ms` (day 23: `171.8ms`). So the demote's copy now lands
before the next tick top and its two on-tick hashes (the completion checksum at the poll and `bind_tier_image`'s
bundle checksum at publication, about 78 ms per 160 MB pass on this host, `pro-single-day18/hashmicro/`) fall on
ONE tick: the promote arm's tenant stall (149.6) equals the demote arm's (150.0) because it IS the demote's tick.
On the day-23 tree the demote's completion arrived a tick later and the two hashes split across two ticks (81.9).
That is a reading of the lines, not a measurement of the hashes; which slice between `0713c1a79` and the integ38
tip moved the copy's landing (Move 2's restore on the same copy stream, the D2D receipt term, or the #638 parked-only
wait covering a pending restore) is not determined here. The promote's own off-tick window did not move (published
after `19.4` to `19.8` ms; `server_promote_ms` steady 25.8 to 27.5). Nothing was tuned after this reading.

## Task 3, the hit gate under the collector's hold (the day-28 owed run; `pro-single-day29/hitgate/`)

One collector hold (`collector/hitgate/`, `CELL.jsonl` `status: executed-not-qualified`, `elapsed_seconds 89.0`,
`--validate` rc=0), arm X's binary `3d7baaa0…`, tree `7349ef932`, `day29-hitgate-cell.sh`: door OFF then door ON
(`MEMRA_KV_HOST_CONTRACTS=1`; the gate exports `MEMRA_KV_HOST_MB=8192`). Verbatim:

- OFF (37 s): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 `ok:`, 0 FAIL; first line `lock: collector's inherited
  FD 3 on /tmp/memra-gpu.lock (no flock of this gate's own)`; `ev-off/LOCK.json` `{"owner": "collector", "lock":
  "/tmp/memra-gpu.lock", "mechanism": "inherited-flock-same-open-description", "device": 64769, "inode": 47827,
  "qualification": false}`; door lines 0 (`armed=0 door_on=0` on both boots), restore route lines 0.
- ON (52 s): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 68 `ok:`, 0 FAIL; the same first line; `ev-on/LOCK.json`
  byte-identical to the OFF arm's; engagement `ok: door arm: the host tier is armed on the spec-on boot`, `ok: door
  arm: the contracts door is ON on the spec-on boot`, `ok: door arm: no latch line on the spec-on boot`, the same
  three for the spec-off twin, `ok: door arm: 7 route submission(s) across the two boots (capture, restore, demote
  or promote off the tick)`; census `door lines: armed=1 door_on=1 capture_submitted=1 capture_published=1
  restore_submitted=1 restore_landed=1 ...` (spec-on) and `capture_submitted=2 capture_published=2
  restore_submitted=3 restore_landed=3` (spec-off); door lines 18, restore route lines 8 (the day-27 figures).

Card 40 to 56 C, 56 to 497 W under 600 W (355 samples), `0 MiB` used before and after each arm. The identity
clause (spec-on text == plain text, `r1` to `g2` `spec==plain byte identity`) held in both arms under the collector's
hold, as it did under the gate's own `flock` on day 27. The `--external-lock` arm now has its GPU receipt on the
target card.

## Task 4: records and checks

`HOSTPREFIX-DOOR.md`: section A gains the owed-cell row for Move 1 decision cell (i) with the verdict lines
verbatim and the promote finding; the hit-gate row gains the day-29 collector run; section B's two Move 1 rows gain
the same-window pair (150.0 / 150.0 against 193.4 / 193.1; 149.5 / 149.7 against 162.9 / 162.5); section E names
the cell (i) reading and the finding. `STATE.md` rewritten; `research/INDEX.md` row `spill-c-20260919/day29`.
Checks: `shellcheck -S warning` clean on the three drivers; `bash tools/check-flags.sh` no uncovered runtime names
(no new `MEMRA_*` read); `bash tools/check-conflict-markers.sh` OK; `git diff --check` clean over the day's range;
`.gitattributes` `*.log -whitespace` in `pro-single-day29/`; zero em dashes in every line added today.

## Pushes

`f5398f8fb` (the merge), `7349ef932` (the pre-registration, the drivers, the reading script), then the closing
commit (this record's Task 2 run, Task 3 and Task 4 sections, the receipts, the door table, STATE, INDEX), each in
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`,
logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 reached through the existing control socket only (`ssh -O check` first, `Master running`). `/root/wt-c` (mine)
left detached at `7349ef932`, clean; `/root/wt-c-day16` removed (`git worktree remove --force`, `worktree prune`;
its binary's hash is in the receipt) and its directory gone; the two bundles removed on both ends; receipts under
`/root/spill-receipts/day29` (mirrored here as `pro-single-day29/`); `/root/artifacts`, `/root/memra-spill`, other
lanes' worktrees, receipts and processes not touched (lane A's `a-day23` server was on the card when I arrived and
its `a-day24` collector held the lock when I left, seen in `nvidia-smi` and `pgrep` listings only; a `/root/day24.bundle`
that is not mine left where it is). No server of mine on either card at close; no lock held by me; no `/tmp` scratch
left on either end (the two local bundles and the reading-script test dir removed).

## Budget

About 2.0 agent-hours against 4: reading and the merge 0.4, the pre-registration and the three drivers plus the
reading script and its dry test 0.5, the box shipping and both builds 0.2, the hold (38 minutes of boots plus the
hit gate's 89 s, polled) 0.5, the receipts, the finding's line reading, the records and checks 0.4. Blockers: none
(the lock was free at launch; the day-16 tree built and booted without a patch).
