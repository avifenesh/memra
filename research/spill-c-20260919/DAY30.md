# Session C day 30: the decision packet for the door's decide-by, and the capture share re-read

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change today: research scripts and records only.

## Merge (first action)

No `lane/spill-integ40-20260922` branch or PR existed on `origin` at the start of the day (`git branch -r`, `gh pr
list --search integ40`: nothing), so the rule's first arm applied to the tree that exists: `origin/main` `ebe3fe17d`
(PR #642, integ39: A day 23 and C day 28) merged `--no-ff` as `24597453e`. One conflict, `HOSTPREFIX-DOOR.md`'s
owed-cell table: `main` carried the day-27 hit-gate row TWICE (the pre-day-28 text and the day-28 text, an integ39
union artifact) plus lane A's day-23 draft-bearing hit-gate row; this lane's side carried the day-27 row once with
its day-29 collector run. Resolved by hand: the day-29 row once, then A's day-23 row, main's duplicate dropped;
`git rerere`'s replayed resolution (which kept the duplicate) was discarded first. `check-conflict-markers: OK`.
Pushed in the announced development mode.

## Task 2, pre-registration (this section is committed before the cell runs)

**What day 28 left unread.** `DAY28.md` Task 2: the capture class's `on_minus_off` read `+0.7 ms` (unc 0.2) in both
passes, but the pre-registered subtraction for each capture arm's OWN share (`stall(capture-arm) - stall(prime)`)
came out with the wrong sign (`share=-17.8` / `-17.1`): the prime-only boot (`MEMRA_PREFIX_CACHE_MB=0`) stalled the
tenant `301.5` where the same prompt's prime inside the cache-on boots stalled `283.7` / `284.4`. Move 2 owed item 3
(`OWNER-THREAD-OFFLOAD.md`, A's day-24 list) states it as "the capture share stays unread".

**Is a corrected reading available from receipts that already exist?** No. The candidates were read, not
re-measured: (a) A's day-20 capture cells (`research/spill-a-20260919/pro-single-day20/box/stall-capture-{off,on}`)
ran at a 1024 MB device cache where the ten fresh seeds evict and demote inside the arm (8 `server_demote_ms`
entries per boot), inadmissible under day 28's own rule (a demote or promote inside the arm confounds the class); (b)
A's day-16 `stall-prime` arm and day 28's `prime` arm are cache-OFF boots, the very shape whose prime differs; (c) A's
day-24 receipts (`pro-single-day24/box/gates/`) are gate runs, not stall cells; (d) no receipt on either lane has a
cache-ON boot whose seed insert did not copy. So the subtraction arm must be run, once, in one hold with the two
capture arms.

**Why a cache-off boot's prime is not a cache-on boot's prime (a reading of the lines, tested by the cell, not
assumed).** With the prefix cache configured, admission arms the grid-aligned seed (`seed_at`, `worker.rs`
`seed_capture_boundary`) and the prime STOPS at the aligned boundary, 5088 of the 5120- to 5123-token prompt, to
publish the seed (`insert (seed): 5088 tokens, 308.0MB` in every cache-on log of day 28), then primes the remaining
32 to 35 tokens. The harness's stall is the single worst tick (max ITL minus p50), so a split prime's worst tick
carries 5088 tokens where the cache-off boot's carries all 5123 in one chunk: the 16 to 18 ms difference is the
tail chunk moved to a second tick, not a cost the capture removed. If that reading is right, a cache-on boot whose
seed insert is REFUSED before any copy stalls the tenant about `283`, not `301`; if it stalls about `301` the
reading is wrong and the numbers say so.

**The existing typed condition that refuses the capture inside a cache-on boot.** `prefix_insert_from_session`
(`worker.rs`) asks `PrefixCache::prepare_snapshot` for room BEFORE any copy and BEFORE the door's
`prefix_capture_off_tick` is asked (A's day-20 statement: "the budget is asked before any copy"); an entry larger
than the whole budget is refused by the typed line `[prefix-cache] insert refused: entry <bytes> exceeds budget
<bytes> (snapshot preflight, model gate)` (`prefix_insert_refused_oversize`, memra#523 item 1) and the request
continues unpublished. The admit-time seed decision (`seed_prefix`, `seed_at`) reads the prompt length and the
covering entries, never the budget, so the prime still stops at 5088. A boot with `MEMRA_PREFIX_CACHE_MB=128`
(134,217,728 B, under the 308 MB entry) therefore has the cache ON, the seed boundary honoured, the insert reached
and refused, no recurrent clone, no KV rows copied, no capture route asked: the prime-with-seed-boundary arm the
subtraction needs. The tenant's 20-token prompt is under the 64-token entry floor and never seeds, in any arm.

**The cell (target card, BOX3, one RTX PRO 6000 Blackwell at 600 W, the 27B `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, the
collector, ONE lock hold).** Driver `day30-stall-cell.sh` under `day30-box-run.sh` (the collector
`tools/tier-battery.py --rig pro-single --external-lock`, bounded lock retries 60 x 120 s, the holder never inspected
or signalled; lane A shares the card today). Harness `day28_stall_cell.py` byte-for-byte (its SHA-256 and its diff
against A's `stall_cell.py` recorded, as on day 28); no harness change. Every boot `MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192
MEMRA_MAX_SESSIONS=4`, the harness's own idle/arm interleave in both orders, N=5 per arm per order (10 arm runs, 10
idle runs per boot). Four arms, eight boots, two passes in opposite order: pass 1 `refused-off, refused-on,
capture-off, capture-on`; pass 2 `capture-on, capture-off, refused-on, refused-off`.

- `refused-off`: `MEMRA_PREFIX_CACHE_MB=128 MEMRA_KV_HOST_MB=8192`, harness `--mode prime` (a fresh ~5120-token
  prompt per run, no re-post); the corrected subtraction arm.
- `refused-on`: the same with `MEMRA_KV_HOST_CONTRACTS=1`; a control (the door has nothing to route).
- `capture-off`: day 28's arm, `MEMRA_PREFIX_CACHE_MB=8192 MEMRA_KV_HOST_MB=8192`, `--mode capture` (the seed
  captures on the tick; the untimed re-post's `cached_tokens` recorded).
- `capture-on`: the same with `MEMRA_KV_HOST_CONTRACTS=1` (the capture on the copy stream).

**Rules, fixed before the run (`day30-stall-reading.py`; ms; `stall_median` and IQR over the boot's 10 arm runs).**

- Admissibility, per receipt: `STALL REPLAY: PASS`, `errors=0`, `tenant_text_identical=True`, ZERO `server_demote_ms`
  and ZERO `server_promote_ms`, zero `seed REFUSED (grid)` lines (the prime stopped at the boundary in every arm), zero
  `DISABLED` or `capture refused (contracts door)` lines. Refused arms: at least one typed oversize refusal line with
  `budget 134217728` and `entry > budget`, zero `insert (seed): 5088 tokens` lines, zero `capture submitted off the
  tick` lines, zero `hit: 5088 of` lines. Capture arms: every re-post `repost_cached_tokens >= repost_prompt_tokens -
  64`, at least ten `insert (seed): 5088 tokens` lines; the ON arm at least ten `capture submitted off the tick`
  lines, the OFF arm none. An inadmissible receipt decides nothing and is reported as such.
- Per pass: `base = stall_median(refused-off)`; `share(capture-arm) = stall_median(capture-arm) - base`, `unc =
  sqrt(IQR(capture-arm)^2 + IQR(refused-off)^2)`; `control = stall_median(refused-on) - stall_median(refused-off)`
  in quadrature; `on_minus_off = stall_median(capture-on) - stall_median(capture-off)` in quadrature. `isolated`
  when `|value| > unc`, else `under_resolution`. `arm_p99` and `arm_max` printed beside every arm (a split signature
  is read as moved work across ticks, never as a removed share). Nothing tuned after the run; the reading script is
  committed with this section (dry-tested on the day-28 receipts through a temporary symlink tree, where it
  reproduced day 28's `283.7 / 284.4 / 301.5` and marked the prime-shaped arms inadmissible for lacking the refusal
  line, as intended; the temporary tree was removed).
- Regime from the collector's `command.gpu.csv` (250 ms samples across the hold) and each boot's
  `card.{before,after}.csv`.

**Expected readings, stated before the run.** `base` about 280 to 284 if the split reading holds (the cache-on
prime with no copy), about 301 if it does not. `share(off)` positive and small: the OFF insert's on-tick work is the
recurrent clone (about 157 MB, owner stream) plus the KV rows (about 154 MB, owner stream) plus allocation, a few
milliseconds at most on this card; it may read `under_resolution` against the arms' IQRs (day 28: 0.1 to 1.5).
`share(on)` about `share(off) + 0.7` (day 28's `on_minus_off`, expected to repeat: +0.7 `isolated`). `control`
expected `under_resolution` (no route, no copy in either refused arm). If the readings differ the numbers say so.

**What I will NOT do.** No tuning after a result; a second attempt happens only for a lock refusal or a boot
failure, reported as an attempt. No harness change. No change to the two capture arms' shape from day 28. No touch of
`/root/artifacts`, `/root/memra-spill` or other lanes' worktrees and processes; no third lock name; no bare GPU run;
every boot inside the collector's one hold. Nothing here decides the door; the packet (Task 1) quotes this cell's
verdict line verbatim once it exists, or records the cell as still unread.

## Task 2, the run (target card, tree `55b8da077`, binary `16d8c4c72e780c9c...`; receipts `pro-single-day30/`)

**Build and hold.** `/root/wt-c` detached at `55b8da077` (`crates/` = `main` `ebe3fe17d`, integ39) from a bundle of
this lane's new commits (the bundle removed on both ends); `build.log` `rc=0` (11:37:13Z to 11:40:14Z, cargo 1.97.1,
CUDA 13.2, `MEMRA_CUDA_ARCH auto-detected 120a`). One collector hold (`collector/stall-capture-share/`, `CELL.jsonl`
`status: executed-not-qualified`, `exit_code 0`, `elapsed_seconds 588.0`; `tools/tier-battery.py --validate` rc=0 on
the box), 11:40:14Z to 11:50:02Z, the lock free at launch (no `lock-retries.log`), eight boots in the pre-registered
order, `LOCK.json` `{"owner": "collector", "lock": "/tmp/memra-gpu.lock", "mechanism":
"inherited-flock-same-open-description", ...}`, `compute-apps.{before,after}.csv` empty, `0 MiB` on the card before
every boot. Harness `day28_stall_cell.py` SHA-256 `d11785a80a030516...` (day 28's), A's `13867e77da40a9b5...`;
`harness.diff` banked. Regime from the collector's `command.gpu.csv` (2345 samples at 250 ms): 33 to 61 C, 32.0 to
501.4 W under the 600 W limit, 0 to 22005 MiB. `exit.txt` `stall-capture-share rc=0`; `replays.log` eight `STALL
REPLAY: PASS`.

**Admissibility (every clause of the pre-registration, `reading.log`).** All eight receipts `replay=PASS
admissible=True`: `errors=0`, `tenant_text_identical=True`, zero `server_demote_ms` and `server_promote_ms`, zero
`seed REFUSED (grid)`, zero `DISABLED` or `capture refused` lines. Refused arms: the typed line fired in every boot
(`[prefix-cache] insert refused: entry 307986432 exceeds budget 134217728 (snapshot preflight, model gate)`, the
announcer printing 3 of the 11 identical refusals per boot with `(previous shape: 7 identical refusals not printed)`;
entries of 307,986,432 and 308,936,704 B against the 134,217,728 B budget), zero `insert (seed): 5088 tokens`, zero
`capture submitted off the tick`, zero `hit: 5088 of` lines. Capture arms: ten `insert (seed): 5088 tokens` per boot,
every re-post a grid hit; the ON arm 11 `capture submitted off the tick` (the calibration post and the ten timed
intruders), the OFF arm 0.

**Rule lines, verbatim (`reading.log`; the intruder token lists elided).**

Pass 1:

`STALL rule cell=stall-refused-off arm=prime n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.9 arm_p99=295.7 arm_max=299.3 stall_median=283.1 stall_min=282.9 stall_max=285.8 server_demote_ms=[] server_promote_ms=[] ... tenant_text_identical=True errors=0`

`STALL rule cell=stall-refused-on arm=prime ... arm_p99=295.7 arm_max=299.2 stall_median=283.1 stall_min=282.9 stall_max=285.8 server_demote_ms=[] server_promote_ms=[] ... tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-off arm=capture ... arm_p99=295.8 arm_max=299.9 stall_median=283.7 stall_min=283.5 stall_max=286.5 server_demote_ms=[] server_promote_ms=[] ... tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-on arm=capture ... arm_p99=295.9 arm_max=300.5 stall_median=284.4 stall_min=284.3 stall_max=287.0 server_demote_ms=[] server_promote_ms=[] ... tenant_text_identical=True errors=0`

Pass 2:

`STALL rule cell=stall-refused-off arm=prime ... arm_p99=295.8 arm_max=299.3 stall_median=283.2 stall_min=282.9 stall_max=285.9 ... errors=0`

`STALL rule cell=stall-refused-on arm=prime ... arm_p99=295.7 arm_max=299.2 stall_median=283.1 stall_min=282.9 stall_max=285.8 ... errors=0`

`STALL rule cell=stall-capture-off arm=capture ... arm_p99=295.8 arm_max=299.9 stall_median=283.7 stall_min=283.5 stall_max=286.5 ... errors=0`

`STALL rule cell=stall-capture-on arm=capture ... arm_p99=295.9 arm_max=300.8 stall_median=284.4 stall_min=284.2 stall_max=287.3 ... errors=0`

**The reading, by the pre-registered rules (`day30-stall-reading.py`, verbatim).**

`DAY30 CAPTURE-SHARE pass=1 base=refused-off stall_median=283.1 iqr=0.1 arm_p99=295.7 arm_max=299.3 n_per_order=5 pooled=10`

`DAY30 CAPTURE-SHARE pass=1 control refused on_minus_off=-0.1 unc=0.3 (refused-on 283.1 iqr 0.3) -> under_resolution`

`DAY30 CAPTURE-SHARE pass=1 class=capture arm=off stall_median=283.7 iqr=0.3 share=+0.6 unc=0.3 arm_p99=295.8 arm_max=299.9 -> isolated`

`DAY30 CAPTURE-SHARE pass=1 class=capture arm=on stall_median=284.4 iqr=0.1 share=+1.2 unc=0.2 arm_p99=295.9 arm_max=300.5 -> isolated`

`DAY30 CAPTURE-SHARE pass=1 class=capture on_minus_off=+0.7 unc=0.3 -> isolated`

`DAY30 CAPTURE-SHARE pass=2 base=refused-off stall_median=283.2 iqr=0.2 arm_p99=295.8 arm_max=299.3 n_per_order=5 pooled=10`

`DAY30 CAPTURE-SHARE pass=2 control refused on_minus_off=-0.1 unc=0.3 (refused-on 283.1 iqr 0.2) -> under_resolution`

`DAY30 CAPTURE-SHARE pass=2 class=capture arm=off stall_median=283.7 iqr=0.2 share=+0.5 unc=0.3 arm_p99=295.8 arm_max=299.9 -> isolated`

`DAY30 CAPTURE-SHARE pass=2 class=capture arm=on stall_median=284.4 iqr=0.1 share=+1.2 unc=0.3 arm_p99=295.9 arm_max=300.8 -> isolated`

`DAY30 CAPTURE-SHARE pass=2 class=capture on_minus_off=+0.7 unc=0.2 -> isolated`

`DAY30 CAPTURE-SHARE VERDICT: pass1 refused on-off -0.1 (unc 0.3) under_resolution; pass1 share off +0.6 (unc 0.3) isolated; pass1 share on +1.2 (unc 0.2) isolated; pass1 capture on-off +0.7 (unc 0.3) isolated; pass2 refused on-off -0.1 (unc 0.3) under_resolution; pass2 share off +0.5 (unc 0.3) isolated; pass2 share on +1.2 (unc 0.3) isolated; pass2 capture on-off +0.7 (unc 0.2) isolated; admissible=True`

**What the cell read (nothing tuned).** The split reading held: a cache-ON boot whose seed insert is refused before
any copy stalls the tenant **283.1 / 283.2 ms** (IQR 0.1 / 0.2), the cache-on figure and not the cache-off boot's
301.5, so day 28's 17 ms was the prime's tail chunk (the 32 to 35 tokens past the 5088 seed boundary) moved to a
second tick, never a cost the capture removed. Against that base the capture arm's OWN share is **+0.6 / +0.5 ms
door OFF** (unc 0.3, `isolated` at the rule's edge: the recurrent clone of about 157 MB, the KV rows of about
154 MB and the allocation on the owner stream) and **+1.2 / +1.2 ms door ON** (unc 0.2 / 0.3, `isolated`: the
recurrent clone, the submit and the ticket on the tick, the rows on the copy stream), with `on_minus_off` **+0.7 /
+0.7** (day 28's figure exactly, `isolated`) and the refused-on control **-0.1 / -0.1** (`under_resolution`: the door
with nothing to route costs the tick nothing the cell resolves). `arm_p99` 295.7 to 295.9 and `arm_max` 299.2 to
300.8 in every arm: no split signature; the intruder's prime dominates the worst tick in all four arms and the
whole capture class sits inside about 1 ms of it on this card at this entry size. Move 2 owed item 3's capture half
is READ: under a millisecond in both arms, the door's own share about 0.6 ms above the OFF program's, sign ON above
OFF in both passes, on the target card, `executed-not-qualified`. Nothing here decides the door; the packet quotes
the verdict line.

## Task 1: the decision packet (`DOOR-DECISION-PACKET.md`)

Written as one page plus an appendix: the question (section E, stated and not answered); what the door is today
(Move 1 whole, Move 2 slices 1 to 3, the draft-bearing restore and capture routes, the receipt term, every
fail-closed arm, what still runs on the tick); the correctness table (every gate, both arms, both cards, tree and
receipt path, verbatim verdicts); the cost table per card (the on-tick door, the Move 1 halves, the same-window pairs
of day 23 and day 29, the isolating cell of day 28, today's capture share, the receipt price on both cards, the
cached and write-combined pairs, the hash micro-cell, the arena pair), each with N, order and regime; the open
findings (cell (v)'s day-19 clause, cell (i)'s clause not met, the double park, A's day-24 retire settle, the capture
share now read, the neighbouring doors' dates, what is unbuilt); the three outcomes with what each would require,
without a recommendation. Every number was re-read today from the file named beside it: the stall rule lines from
their `receipt.json` (day 28's lines regenerated with `day28-stall-reading.py` and equal to `pro-single-day28/reading.log`),
the pairs regenerated with `wc-pair.py` and `arena-pair.py` (`REPLAY: PASS`), the gate `ok:` counts counted in the
gate logs, the regimes computed from each cell's `command.gpu.csv` (the door table's day-16 regime "43 to 50 C, 88 to
329 W" is replaced there by the three cells' own CSV range, 32 to 58 C, 32.6 to 492.7 W). Untraced: none; what was
deliberately not carried is listed in the packet's appendix B. It is a draft for the owner and becomes a
`docs/decisions/` record only after the owner decides; nothing was created under `docs/decisions/`.

## Task 3: records and checks

`STATE.md` rewritten; `research/INDEX.md` row `spill-c-20260919/day30`; `HOSTPREFIX-DOOR.md` section E gains the
pointer to the packet, and the two capture rows (the owed-cell row of the isolating cell, the section B capture row)
gain today's reading. Checks: `shellcheck -S warning` clean on the two drivers; `bash tools/check-flags.sh`;
`bash tools/check-conflict-markers.sh`; `git diff --check`; `.gitattributes` `*.log -whitespace` and `*.diff
-whitespace` in `pro-single-day30/`; zero em dashes in every file added or edited today (the results are quoted in
the closing commit's message). `tools/public-boundary-check.sh` does not exist in this tree; the pre-push hook's
boundary census ran on every push (`public-boundary: 0 matches`).

## Pushes

`24597453e` (the merge), `55b8da077` (the pre-registration, the drivers, the reading script), then the closing
commit (this record's Task 2 run, Task 1 and Task 3 sections, the receipts, the packet, the door table, STATE, INDEX),
each in `MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification
claimed`, logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 reached through the existing control socket only (`ssh -O check` first, `Master running`). `/root/wt-c` (mine)
left detached at `55b8da077`, clean; the bundle removed on both ends; receipts under `/root/spill-receipts/day30`
(mirrored here as `pro-single-day30/`); `/root/artifacts`, `/root/memra-spill`, other lanes' worktrees, receipts and
processes not touched (the card was free when my hold began; lane A's `a-day25` server was on the card and the lock
held when I left, seen in `nvidia-smi` and `pgrep` listings only). No server of mine on either card at close; no lock
held by me; no `/tmp` scratch left on either end (the local bundle and the reading-script test tree removed).

## Budget

About 3.0 agent-hours against 4: reading and the merge 0.6, the Task 2 design and pre-registration 0.5, the box
shipping, build and hold (10 minutes of boots, polled) 0.4, the packet and the receipt cross-check 1.1, the records
and checks 0.4. Blockers: none (no integ40 branch existed, so the merge took `origin/main`; the lock was free at
launch).
