# Session C day 36: the door packet re-read with option (a) landed (A days 28 and 29, ruling 41), no card

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. The day's first run was stopped by the owner and restarted; section 0
is the resync. Start state after the resync: `23b4062a2` = remote, no tracked edit. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919
at <sha>; no GPU qualification claimed` and records the skip in the clone's `.git/memra-gate-skips.log`). No card work, no
GPU process, no lock taken. No commit on main, no PR, no engine change, no `docs/` registry edit, no recommendation. The
day's local work is file reads (the census script, greps, git); nothing CPU-heavy ran, so no build or cell needed the
scoped quota.

## 0. Resync (the stopped run)

Every fact in this section is in `day36-cpu/resync.log` (commands as `# command:` lines) or `day36-cpu/ancestry.log`.

- **What the stopped run left.** The worktree at `23b4062a2`, three merges above day 35's close `48250588c`: `b677ea313`
  (`origin/main` `ca5a90e2a`, #651), `295e9475e` (the local integ45 ref at `ebf94591d`) and `23b4062a2` (the local integ45
  ref at `1c540e050`), all three on `origin/lane/spill-c-20260919` (`git branch -r --contains 23b4062a2`). Untracked:
  `day36-cpu/` and `day36-receipt-census.py`. No tracked edit (`git status --short` empty of tracked paths).
- **Kept: `day36-receipt-census.py`.** Read end to end before use. Its four functions do what their names say:
  `ledger()` parses every `demote digests landed off the tick` line of the double-park ON boots into the owner-thread
  segments, `regime()` reads each hold's `command.gpu.csv`, `identity()` counts the parked and ledger lines of the
  identity default-ON host-on boot, `gates()` prints each gate log's last verdict line with its `ok:` and `FAIL:` counts.
  Run over the receipts it printed 0 `NOMATCH`. Extended today: `first_demote()` (each ON boot's first ledger line
  against the promote runs' `server_log_lines`), per-cell `ok:`/`FAIL:` in `gates()`, and `provenance()` (the trees).
- **Discarded: the stopped run's `day36-cpu/` outputs.** A CPU output dir cut off mid-run is not a receipt, and its
  `gate-census.log` cited a `DAY36.md` that did not exist yet. Every log under `day36-cpu/` was re-produced today from
  the command on its first line.
- **The merge after the restart.** `origin/lane/spill-integ45-20260922` at `96d96e1c2` merged as `aeefd9233` (parents
  `23b4062a2 96d96e1c2`). Its tree equals `96d96e1c2`'s (`07a96ec94`, and `git diff --stat 96d96e1c2 aeefd9233` prints
  nothing), so the integ side is the tree everywhere, `HOSTPREFIX-DOOR.md` included. Pushed (`23b4062a2..aeefd9233`).
- **`git diff`.** Nothing tracked in `wt-spill-c`. The main checkout carries two modified files
  (`research/.upstream-sweep-since`, `research/upstream-sweeps.md`); not this lane's, not touched.
- **Processes.** Zero with a cwd under `wt-spill-c` (the scan run from `/tmp`; run from inside the tree it matched only
  the scanning shell), zero GPU compute apps. No `pkill`.
- **`main` moved during the day.** `origin/main` `ca5a90e2a` then `189c91b15` (#595) after the close-time fetch; A's
  day-28 and day-29 code (`45f824a75`, `867655368`) is an ancestor of neither, and of both `1c540e050` and `aeefd9233`
  (`ancestry.log`). The lane did not merge `189c91b15`; no step of today's brief needs it, and the records say "not on
  `main` at the time of writing (`origin/main` `ca5a90e2a`)", which the ancestry check keeps true for `189c91b15`.

## 1. The counting rule

A count in this lane's records is a number a command printed over receipt files, and the command sits beside the number:
as the first line of its banked output under `day36-cpu/`, in the packet's appendix A, or in the table below. A figure
read by eye, carried from memory or copied from another record without reopening its receipt is either quoted verbatim
with its file and line, or left out. When a quoted figure and a counted one differ, both stand with their N and the
reason (A's clause 1c `median=7.40` over N=100 against the census's `7.41` over N=110: each boot's first ledger line is in
no promote run's lines, `outside_promote_runs=10` both days).

Applied today. Every new figure in the packet and in section E's day-36 paragraph, and where its command is:

| Figure | Command | Output |
|---|---|---|
| The owner-thread ledger (110 lines per day, 11 per boot; owner `in - completion` 7.41 / 7.39; pre-submit 6.07 / 6.05 on the 4th to 11th demote (N=80) and 42.19 / 42.27 on the first three (N=30); settle 1.21, polls 0.01, take-back 1.29; helper `hashed in` 73.20 / 73.10) | packet appendix A, 1 | `receipt-census.log` lines 2 to 7 (day 28), 23 to 28 (day 29) |
| First demote per boot outside the promote runs, 10 of 10 both days | appendix A, 1 | `receipt-census.log` lines 8 to 18, 29 to 39 |
| The regimes (double-park, gates and unit-cell holds) and `stall_replay_pass=20` per day | appendix A, 1 | `receipt-census.log` lines 19 to 22, 40 to 43 |
| Identity default-ON: 1 parked line and 2 ledger lines in the host-on boot on each card, `35 re-park(s)` and `5 re-park(s)` | appendix A, 1 | `receipt-census.log` lines 44 to 51 |
| Every gate's verdict, `ok:` and `FAIL:` count and per-cell split (day 28, day 29, both cards; integ45's RTX 5090 cells) | appendix A, 1 | `receipt-census.log` lines 52 to 148 (`GATE`, `cells=`, `per_cell ok/fail`, `door lines`) |
| The trees (`f9284a711`, `29a1cc366`, `259c75f62`, `06e290374`, `1c540e050`) | appendix A, 1 | `receipt-census.log` lines 149 to 168 (`TREE`) |
| 19 of 120 server logs with a parked `Hashing` hit | appendix A, 2 | `parked-census.log` |
| Clauses 1a to 5, the stall, e2e and decomposition figures | appendix A, 3 (read as banked) | `clause-lines.log` |
| Option (a)'s code not on `main` `ca5a90e2a` or `189c91b15`, on `1c540e050` and `aeefd9233` | appendix A, 4 | `ancestry.log` |
| integ45's serve smoke `serve-smoke: 0 failed` with three SKIP arms, the D2D cells `test result: ok. 5 passed; 0 failed`, `rc=0` | 5, below | `integ45-smoke.log` |
| The resync facts of section 0 | 6, below | `resync.log` |

5. `grep -H 'serve-smoke: .*failed\|arm SKIP\|test result:\|^rc='` over
   `research/spill-lead-20260919/integration-day12/integ45-serve-smoke-5090/*.log` and `*.exit`; `sed` on the output
   shortens each SKIP line's absent path to `file` (stated in the log).
6. The `git reflog`, `git branch -r --contains`, `git rev-parse <c>^{tree}`, `git diff --stat`, `git status --short`,
   `/proc/*/cwd` and `nvidia-smi --query-compute-apps` lines of `resync.log`, each under its own `# command:` line.

One figure was corrected by this rule before the close: the packet's section 6 and section E's day-36 paragraph said the
parked hit happens "1 per boot" in the identity default-ON arm. The census counts one parked line in that arm's host-on
boot, the arm's only boot with the tier armed; both places now say "1 line in the host-on boot".

## 2. Commits

- `aeefd9233`: the integ45 merge (section 0).
- `bac9545a4`: `DOOR-DECISION-PACKET.md` with option (a) landed, and `day36-cpu/` plus `day36-receipt-census.py`. The
  status header, section 2 (the hash helper bullet; "What still runs on the tick under ON" re-read from A's finding 2),
  section 3's six new rows, section 4's day-28, day-29 and 5090 ledger rows, item 2's day-36 receipt paragraph (ruling
  41 verbatim, the twelve `DAY28 CLAUSE` and `DAY28 VERDICT` lines of the two sittings verbatim out of the fourteen
  `clause-lines.log` holds, the two `DAY28 REPORTED` lines not quoted; clause 2 per gate per card), item 7 (drops option (a), adds Move
  2 owed item 1), section 6 re-priced, appendix A's day-36 commands and appendix B's DAY26 lines not carried.
- `f7c4fff4d`: `HOSTPREFIX-DOOR.md` section E's DAY 36 paragraph. Section B already carried A's day-28 and day-29 rows
  once each (from integ45); section D item 6 was already RESOLVED. Nothing else in the door doc changed.
- This records commit: `DAY36.md`, `STATE.md`, the `research/INDEX.md` row, `day36-cpu/resync.log`, the `ancestry.log`
  lines for `189c91b15`, and the "1 line in the host-on boot" correction in both documents.

## 3. Checks and push

Each gated in its own `if ! ...; then exit; fi` line, from the worktree root:

- `bash tools/check-flags.sh`: `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)`
- `bash tools/check-conflict-markers.sh`: `check-conflict-markers: OK (no conflict marker line in tracked source or docs)`
- `git diff --check` (staged and against `origin/lane/spill-c-20260919`): clean (`git diff --cached --check` and `git diff --cached --check aeefd9233`)
- `day36-cpu/.gitattributes` carries `*.log -whitespace`: yes (`grep -qx`)
- Em dashes in the lines this lane added today (the `+` lines of `git diff aeefd9233 -- research/spill-c-20260919/
  research/INDEX.md`, the merge being the day's last commit not written by this lane): 0
- `python3 tools/check-public-boundary.py check`: `public-boundary: 599 matches (599 grandfathered, 0 new).`

The records commit is pushed in the announced development mode; its SHA is checked against `git ls-remote origin
lane/spill-c-20260919` after the push (a commit cannot carry its own SHA).

## 4. Owed and open

- The owner's decision on `MEMRA_KV_HOST_CONTRACTS` (decide-by 2026-10-05). The packet recommends nothing.
- Move 2 owed item 1 (the pre-submit segment, A day 30, running in A's lane; not touched here).
- No demote-class tenant-stall cell has run on the day-28 and day-29 code on either card; every day-28 and day-29 stall
  figure is the promote-then-hit shape.
- Unchanged from day 35: the 9B entry's KV byte split (engine code to print), the double-park slice question, a prime
  arm on the RTX 5090 class that the memory admission admits in every run (a new pre-registration, not run).

## 5. Cleanup

`/tmp/c36-census.txt` and `/tmp/c36-flags.log` removed. No server, no lock, no GPU process, no scratch left. The
worktree stays: the lane is open.
