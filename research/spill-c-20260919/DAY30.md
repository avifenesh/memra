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
