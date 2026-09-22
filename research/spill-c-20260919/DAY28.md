# Session C day 28: the hit gate's `--external-lock`, the isolating stall cell, the receipt's price on the RTX 5090

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees engine files in the range; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; nothing here is a
qualification claim. No commit on main, no PR. No engine change today: one gate script, one teeth fixture, one CI
line, research scripts and records.

## Merge (first action)

`origin/lane/spill-integ38-20260922` `643ecbb28` (my days 26 and 27, A day 22 with the D2D receipt term, the lead's
three fixes) merged `--no-ff` as `8803f4b6c`, clean, no conflict. PR #639 read `OPEN` (`gh pr view 639 --json state`),
so `origin/main` was not merged. Pushed.

## Task 1: `tools/spec-on-cache-hit-gate.sh --external-lock FD` (the identity gate's shape)

**Before.** Every boot ran under the gate's own `flock -w 300` on `MEMRA_GPU_LOCK`; under the collector's hold the
gate would block for 300 s per boot and then boot unlocked (a `-w` timeout does not refuse). Lane A day 21 and C
days 26 and 27 ran it under its own `flock`, outside the collector (stated in `DAY27.md`).

**The change.** Leading options before the positional contract, as `kv-host-spill-identity-gate.sh` line 44:
`--external-lock FD` (a numeric FD or `REFUSED: inherited lock FD required`, exit 2) sets the lock owner to
`collector`; the launch wrapper is an array, `flock -w 300 "$GPU_LOCK"` in the default arm and EMPTY under the
collector (a second `flock` on the same inode would deadlock behind the collector's own hold), so `boot()` launches
`env ... "$BIN"` directly. Before any boot the collector arm verifies the FD with `tools/tier-lock-proof.py --fd FD
--lock "$GPU_LOCK" --owner collector` (canonical paths only) into `<evidence_dir>/LOCK.json`, or exits 2 with the
helper's `REFUSED:` line. `stop()` addresses the server through `gate_server_pids()`: the default arm's child of the
gate's own wrapper (`pgrep -x -P "$SERVER_PID" memra-server`, the day-27 rule), the collector arm's `$SERVER_PID`
itself (`env` execs the binary in place) and only while `/proc/<pid>/comm` reads `memra-server`. The default arm's
lock behavior, the assertions, the door arm and the identity clause are unchanged. No new `MEMRA_*` read.

**Teeth.** `--lock-self-test FILE` boots nothing and runs the arm's launch wrapper around a probe (`flock -n FILE
true`) that asks whether FILE is locked while the wrapper runs; FILE is a private temp file (the collector's
`--private-lock-dir-for-tests` precedent), never a rig lock, and the proof helper is not called in this mode.
`tools/test_spec_on_cache_hit_gate_lock.sh` (wired into `ci.yml`'s gate-teeth step beside
`test_gate_template_integrity.sh`) asserts, verdicts to a file, 7 expected: default arm exit 0 and `probe=held`
(the wrapper holds the lock); `--external-lock 7` with FD 7 open on the file exit 0, `wrapper=none`, `probe=free`
(the gate took no lock) and the file's inode and mtime unchanged; `--external-lock x` `REFUSED` exit 2;
`--lock-self-test` without a file `REFUSED` exit 2. Run on this tree, verbatim: `SPEC-ON-CACHE-HIT-GATE LOCK
TEETH: ALL GREEN (7 ok)`; the two gate lines: `LOCK-SELF-TEST owner=internal-canonical fd=none lock=<tmp>
wrapper=flock -w 300 <tmp> probe=held inode_mtime_before=<i:m> inode_mtime_after=<i:m>` and `LOCK-SELF-TEST
owner=collector fd=7 lock=<tmp> wrapper=none probe=free inode_mtime_before=<i:m> inode_mtime_after=<i:m>` (equal
before and after). `shellcheck -S warning` on the gate: the one pre-existing SC2034 (`tries`, not today's); on the
fixture clean. `docs/TESTING.md` carries the lock-arms bullet beside the day-27 door-arm bullet. Not run today: the
gate on a GPU under the collector (no cell in today's brief boots the hit gate; the day-27 receipts stand for the
default arm, the GPU-less teeth for the arm's lock behavior).

## Task 2, pre-registration (this section is committed before the cell runs)

**What Move 2 owed (item 3, `OWNER-THREAD-OFFLOAD.md`).** The day-20 capture cell and the day-21 restore cell carried
the intruder's own on-tick compute in both arms: a ~5120-token prime (capture, `stall_median` 355 both arms) and a
33-token suffix prime plus the fixed recurrent term (restore, 88 OFF against 80 ON). The moved share, the KV rows'
D2D (about 154 MB at 5088 tokens on the 27B, `DAY24.md` arithmetic), sat under the cell's resolution. Two
confounds are named here before the run: (1) at A's 1024 MB device cache the ten fresh seeds of the capture arm
evict and DEMOTE inside the cell (8 `server_demote_ms` lines per boot in A's receipts, about 300 ms each under the
door from the two on-tick hashes at submission), and the harness's stall is the single worst tick, so a demote tick
can be what a "capture" arm measures; (2) the restore arm's suffix prime is compute the class does not own.

**Design (target card, BOX3, one RTX PRO 6000 Blackwell at 600 W, the 27B, the collector, one lock hold).**
Harness `day28_stall_cell.py`: lane A's `stall_cell.py` with ONE added arm, `restore-exact`; the other arms are
byte-for-byte A's program (the four day-20/21 receipts replay `STALL REPLAY: PASS` under it; the diff is banked as
`harness.diff`). Driver `day28-stall-cell.sh` (ten boots in one collector hold through `day28-box-run.sh`, bounded
lock retries 60 x 120 s, the holder never inspected or signalled), `MEMRA_SERVE_SPEC=0` (one token per tick),
`MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`; each boot's harness interleaves idle and arm in both orders, N=5 per arm
per order (pooled 10 arm runs, 10 idle runs); two passes in opposite order:
pass 1 `prime, capture-off, capture-on, exact-off, exact-on`; pass 2 `exact-on, exact-off, capture-on, capture-off,
prime`.

- `prime`: A's day-16 arm, `MEMRA_PREFIX_CACHE_MB=0 MEMRA_KV_HOST_MB=0`: the ~5120-token intruder prime alone (no
  capture exists without a cache). The subtraction arm of the capture class; the door is not in this boot.
- `capture-off` / `capture-on`: A's day-20 `capture` arm (the same intruder prompt per run; after the tenant's
  stream ends the prompt is re-posted untimed and its `cached_tokens` recorded), `MEMRA_PREFIX_CACHE_MB=8192`
  (not A's 1024: at 8192 nothing evicts inside the cell, so no demote and no promote can land in the arm),
  `MEMRA_KV_HOST_MB=8192` (the door needs an armed tier), `MEMRA_KV_HOST_CONTRACTS=1` in the ON boot.
- `exact-off` / `exact-on`: the isolating restore arm. Setup (untimed, recorded): the prime arm's prompt padded
  onto the 32-token grid with ` ok` filler words until `/v1/tokenize` reads a multiple of 32 (the hit gate's
  `ongrid_prompt`; three corrections, else `REFUSED`), posted once so its seed captures the WHOLE prompt
  (`cache.pos == toks.len()`), posted once more to confirm `cached_tokens == prompt_tokens`; a confirm that is not
  a whole-entry hit `REFUSED` (exit 2) before any timed run. Every timed run re-posts that SAME prompt at the
  tenant's 24th token with `max_tokens=1`: a whole-entry hit with a ZERO-token suffix, served from the entry's
  boundary logits (`worker.rs` `prefix snapshot has no boundary logits` is the refusal when they are absent; the
  off-tick route admits a plain whole entry with `e.pos == e.toks.len()` and non-empty `last_logits`), so the
  intruder has NO prime of its own and its owner-thread work is the restore class alone: OFF, the session
  cache's allocation, the recurrent copies (about 157 MB, owner stream), the KV rows (about 154 MB, owner
  stream); ON, the allocation, the recurrent copies and the submit on the tick, the rows on the copy stream, the
  request parked and re-admitted at a later tick top. Same cache and tier environment as the capture arms.

**Rules, fixed before the run (`day28-stall-reading.py`; ms, the harness's per-run `stall_ms` = max ITL minus p50,
`stall_median` over the 10 arm runs, IQR = p75 minus p25 of the same 10).**

- Admissibility, per receipt: replay `PASS`, `errors=0`, `tenant_text_identical=True`; capture arms: every
  re-post `repost_cached_tokens >= repost_prompt_tokens - 64`; exact arms: every intruder `cached_tokens ==
  prompt_tokens` and the seed on the grid; capture and exact arms: ZERO `server_demote_ms` and ZERO
  `server_promote_ms` lines (a demote or promote inside the arm confounds the class). An inadmissible receipt
  decides nothing and is reported as such.
- Capture class, per pass: `share(arm) = stall_median(capture-arm) - stall_median(prime)` with `unc =
  sqrt(IQR(capture-arm)^2 + IQR(prime)^2)`; `on_minus_off = stall_median(capture-on) - stall_median(capture-off)`
  with the two IQRs in quadrature (the prime cancels in the difference; the subtraction against the prime arm is
  what makes each arm's own share readable).
- Restore class, per pass: `share(arm) = stall_median(exact-arm)` (no subtraction: the intruder has no compute of
  its own), `unc = IQR(exact-arm)`; `on_minus_off` likewise in quadrature. `arm_p99` and `arm_max` are printed
  beside every exact arm: A's day-21 reading was a SPLIT (ON's `arm_p99` rose 15.5 to 22.1 while `arm_max` fell
  8 ms: the parked request's re-admission moved work to a second tick), and the same signature is read the same
  way here, as moved work across ticks, not as a removed share.
- Classification: `isolated` when `|value| > unc`, else `under_resolution`. No threshold beyond the cell's own
  IQR; nothing is tuned after the run; the reading script is committed with this section.
- The card's regime from the collector's `command.gpu.csv` (250 ms samples across the whole hold) and each boot's
  `card.{before,after}.csv`.

**Expected readings, stated before the run.** By the day-24 arithmetic the rows' D2D (154 MB) costs under 1.6 ms
at any bandwidth above 100 GB/s, under any client-side ITL resolution on this card; so the capture class's
`on_minus_off` is expected `under_resolution`, and each capture arm's `share` (the seed insert: the recurrent clone
of about 157 MB on the owner stream in both arms plus the rows in OFF, the submit in ON) is expected within the
prime arm's IQR (tens of ms on day 16: `stall_min=283.6 stall_max=359.7`), so `under_resolution` as well. The
restore class's `share` is expected `isolated` in both arms (the whole restore is on one tick in OFF; A's OFF read
88 with a suffix prime, so OFF is expected between 60 and 90), and its `on_minus_off` is expected negative by about
the split (A: -8) with the `arm_p99` signature; the rows' own share inside that difference is bounded by the
arithmetic and the cell cannot separate it from the split. If the cell cannot isolate a share the numbers say so.
Nothing here decides the door.

## Task 3: the receipt's price on the local RTX 5090 (cell (v)'s twin, per card, never compared to the target card)

Tree `8803f4b6c`, engine test binary built under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`
(`day28-cpu/build-engine-tests.log` `rc=0`). `cargo test --release -p memra-engine --lib -- --ignored
--test-threads=1 --nocapture d2d_receipt_digest_price_against_the_copy` under `flock /tmp/memra-5090.lock` (free at
the first probe, no retry), receipts `rtx5090-day28/price/` (`test.log`, `test.exit` `rc=0`, `card.{before,after}.csv`,
`compute-apps.{before,after}.csv` both empty, `binary.sha256`, `tree.sha`). The card, NVIDIA GeForce RTX 5090 Laptop
GPU: before `15 MiB, 58 C, 15.41 W, 180 MHz SM, 405 MHz mem, P8` (idle); after `15 MiB, 58 C, 33.93 W, 1635 MHz SM,
14001 MHz mem`; `power.limit [N/A]` on this laptop card; no compute app before or after; 09:30:06Z to 09:30:07Z.
Verbatim:

`D2D-RECEIPT PRICE order=copy-first bytes=165675008 n_per_arm=5 copy_ms=[0.391648, 0.394112, 0.39472, 0.396512, 0.3928] copy_median=0.394 digest_ms=[0.221056, 0.22112, 0.220768, 0.222912, 0.22288] digest_median=0.221 pair_median=0.442 pair_over_copy=1.12`

`D2D-RECEIPT PRICE order=digest-first bytes=165675008 n_per_arm=5 copy_ms=[0.369088, 0.391328, 0.394976, 0.396992, 0.396224] copy_median=0.395 digest_ms=[0.221056, 0.220672, 0.217056, 0.218784, 0.22112] digest_median=0.221 pair_median=0.441 pair_over_copy=1.12`

`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 559 filtered out; finished in 0.51s`

**Reading, on this card only.** One 158 MiB span, event-timed on the copy stream, N=5 per order, both orders, one
warm pass first (the test's own protocol). ONE digest costs 0.56x the copy (0.221 against 0.394 ms; 0.221 against
0.395), and the PAIR the receipt needs (source and destination) 1.12x the copy in both orders (0.442 and 0.441
against 0.394 and 0.395). By the day-19 rule's clause ("unless the digest's cost on the copy stream exceeds the
copy's own time") the pair exceeds the copy on this card too, the single digest does not. The card came off P8
into the cell (the first `digest-first` copy sample 0.369 is the ramp), a single sitting, laptop power envelope. A
reading for the door review, per card; A's target-card numbers (`DAY22.md` cell (v)) sit in their own row of the
door table's cost section and the two cards' figures are never compared to each other. Not a verdict on the door.

## Task 2, the run (target card, tree `61dc1a52a`, binary `02b0a4f384babbf5...` equal to the build of `8803f4b6c`: `crates/` unchanged between them; receipts `pro-single-day28/box/`)

One collector hold (`box/collector/stall-isolating/`, `LOCK.json` owner `collector`, `CELL.jsonl`), 09:35:09Z to
09:46:15Z, no lock retry (the card was free; lane A's slice was not on it in this window), ten boots in the
pre-registered order, no compute app before or after the hold (`stall/ev/compute-apps.{before,after}.csv`), `0 MiB`
on the card before every boot; the collector's `command.gpu.csv` (2658 samples at 250 ms across the hold): 33 to 60 C,
33 to 501 W under the 600 W cap, 0 to 22005 MiB, utilization 0 to 100%; per-boot `card.before.csv` 33 to 47 C.
`exit.txt` `stall-isolating rc=0`; `replays.log` ten `STALL REPLAY: PASS`; `harness.diff` banked (A's harness
`13867e77...`, mine `d11785a8...`). Every receipt admissible by the pre-registered rules: `errors=0`,
`tenant_text_identical=True`, every capture re-post `repost_cached_tokens=5088` against 5120 to 5123 (the grid
seed hit), every exact intruder `cached_tokens=5152 == prompt_tokens=5152` (the on-grid seed: 31 ` ok` filler
words, `hit: 5152 of 5152 prompt tokens from cache` on every timed run in both arms), ZERO `server_demote_ms` and
ZERO `server_promote_ms` lines in every cache-on arm (the 8192 MB cache held every entry). Rule lines, verbatim
(`reading.log`; each boot's `run/receipt.json` `rule_line`):

Pass 1:

`STALL rule cell=stall-prime arm=prime n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.7 idle_p99=14.8 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=295.7 arm_max=315.9 stall_median=301.5 stall_min=285.6 stall_max=302.5 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-off arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.8 arm_runs=10 arm_p50=13.4 arm_p95=14.9 arm_p99=295.8 arm_max=299.9 stall_median=283.7 stall_min=283.6 stall_max=286.5 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-on arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.8 idle_p99=14.9 idle_max=14.9 arm_runs=10 arm_p50=13.5 arm_p95=14.9 arm_p99=295.9 arm_max=300.6 stall_median=284.4 stall_min=284.3 stall_max=287.1 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-exact-off arm=restore-exact n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.7 arm_p99=15.6 arm_max=22.5 stall_median=8.9 stall_min=8.8 stall_max=9.1 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[] intruder_cached_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] intruder_prompt_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] tenant_text_identical=True errors=0`

`STALL rule cell=stall-exact-on arm=restore-exact n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.8 idle_p99=14.9 idle_max=15.0 arm_runs=10 arm_p50=13.5 arm_p95=14.8 arm_p99=15.6 arm_max=22.7 stall_median=9.1 stall_min=9.0 stall_max=9.2 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5] intruder_cached_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] intruder_prompt_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] tenant_text_identical=True errors=0`

Pass 2:

`STALL rule cell=stall-prime arm=prime n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=295.7 arm_max=316.0 stall_median=301.5 stall_min=285.6 stall_max=302.6 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-off arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.9 arm_p99=295.8 arm_max=299.9 stall_median=283.7 stall_min=283.5 stall_max=286.5 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-capture-on arm=capture n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.8 idle_p99=14.9 idle_max=14.9 arm_runs=10 arm_p50=13.5 arm_p95=15.0 arm_p99=295.9 arm_max=300.9 stall_median=284.4 stall_min=284.3 stall_max=287.4 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-exact-off arm=restore-exact n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.7 arm_p99=15.6 arm_max=22.5 stall_median=9.0 stall_min=8.9 stall_max=9.1 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[] intruder_cached_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] intruder_prompt_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] tenant_text_identical=True errors=0`

`STALL rule cell=stall-exact-on arm=restore-exact n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.8 idle_p99=14.9 idle_max=14.9 arm_runs=10 arm_p50=13.5 arm_p95=14.8 arm_p99=15.6 arm_max=22.8 stall_median=9.1 stall_min=8.9 stall_max=9.3 server_demote_ms=[] server_promote_ms=[] server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5] intruder_cached_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] intruder_prompt_tokens=[5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152, 5152] tenant_text_identical=True errors=0`

**The door's engagement in the ON arms (server logs).** Both `capture-on` boots: `[prefix-host] on: budget` and
`[prefix-host] contracts door ON` once each, 11 `capture submitted off the tick (seed): 5088 tokens, 32 planes
(308.0MB) on the contracts door's copy stream; recurrent state cloned at the boundary on the owner stream` (the
calibration post and the ten timed intruders) and 11 `capture published off the tick`, 10 `restore submitted off the
tick` and 10 `restore landed off the tick` (the untimed re-posts after each tenant stream, a 5088-of-5123 hit each,
outside the timed window by construction); both `exact-on` boots: the arming and door lines, 1 capture (the seed),
11 `restore submitted off the tick: 5152 tokens, 32 planes (309.9MB), ticket seq=N on the contracts door's copy
stream; recurrent state copied on the owner stream; request parked` and 11 `restore landed off the tick: 5152 tokens
(309.9MB) complete after 1 poll(s), ...` (the confirm post, `2.3ms from submission to completion, 2.4ms to
re-admission`, and the ten timed intruders, each observed at the next tick top: the harness's `server_restore_ms`
14.5 x10 is the tick period, as A read on day 21). Zero `refused`, `dropped`, `DISABLED` or `[kv-host-contracts]`
lines in any log. The OFF arms: the tier armed (`[prefix-host] on: budget`), no door line, no route line. The
prime arm: neither (no cache, no tier).

**The reading, by the pre-registered rules (`day28-stall-reading.py`, verbatim).**

`DAY28 ISOLATION pass=1 prime stall_median=301.5 iqr=1.5 n_per_order=5 pooled=10`  
`DAY28 ISOLATION pass=1 class=capture arm=off stall_median=283.7 iqr=0.2 share=-17.8 unc=1.6 -> isolated`  
`DAY28 ISOLATION pass=1 class=capture arm=on stall_median=284.4 iqr=0.1 share=-17.1 unc=1.6 -> isolated`  
`DAY28 ISOLATION pass=1 class=capture on_minus_off=+0.7 unc=0.2 -> isolated`  
`DAY28 ISOLATION pass=1 class=restore arm=off stall_median=8.9 iqr=0.1 share=+8.9 unc=0.1 arm_p99=15.6 arm_max=22.5 server_restore_ms=[] -> isolated`  
`DAY28 ISOLATION pass=1 class=restore arm=on stall_median=9.1 iqr=0.1 share=+9.1 unc=0.1 arm_p99=15.6 arm_max=22.7 server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5] -> isolated`  
`DAY28 ISOLATION pass=1 class=restore on_minus_off=+0.2 unc=0.2 -> isolated`  
`DAY28 ISOLATION pass=2 prime stall_median=301.5 iqr=1.5 n_per_order=5 pooled=10`  
`DAY28 ISOLATION pass=2 class=capture arm=off stall_median=283.7 iqr=0.2 share=-17.8 unc=1.5 -> isolated`  
`DAY28 ISOLATION pass=2 class=capture arm=on stall_median=284.4 iqr=0.1 share=-17.1 unc=1.5 -> isolated`  
`DAY28 ISOLATION pass=2 class=capture on_minus_off=+0.7 unc=0.2 -> isolated`  
`DAY28 ISOLATION pass=2 class=restore arm=off stall_median=9.0 iqr=0.1 share=+9.0 unc=0.1 arm_p99=15.6 arm_max=22.5 server_restore_ms=[] -> isolated`  
`DAY28 ISOLATION pass=2 class=restore arm=on stall_median=9.1 iqr=0.1 share=+9.1 unc=0.1 arm_p99=15.6 arm_max=22.8 server_restore_ms=[14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5, 14.5] -> isolated`  
`DAY28 ISOLATION pass=2 class=restore on_minus_off=+0.2 unc=0.1 -> isolated`  
`DAY28 ISOLATION VERDICT: capture pass1 on-off +0.7 (unc 0.2) isolated; restore pass1 on-off +0.2 (unc 0.2) isolated; capture pass2 on-off +0.7 (unc 0.2) isolated; restore pass2 on-off +0.2 (unc 0.1) isolated; admissible=True`  

**What the cell isolated, and what it did not.** Restore class, isolated in both arms and both passes: a
whole-entry hit of a 5152-token plain entry (309.9 MB: the fixed recurrent state of about 157 MB plus about 153 MB
of KV rows) with no prime of its own costs the tenant a stall of **8.9 / 9.0 ms door OFF** (IQR 0.1) and **9.1 ms
door ON** (IQR 0.1), `on_minus_off` **+0.2 ms** (unc 0.2 and 0.1; at the rule's edge, `isolated` by its letter in
both passes, the sign ON above OFF). So the restore class's whole on-tick share is about 9 ms in both arms and the
KV rows' move to the copy stream changed the tenant's stall by nothing the cell resolves: what stays on the tick in
both arms (the session cache's allocation, the recurrent f32 copies on the owner stream, Move 2 owed item 1) is the
9 ms, and the ON arm's submit, park and re-admission cost 0.2 ms more than the rows' own D2D did. No split signature
this time (`arm_p99` 15.6 in both arms, `arm_max` 22.5 against 22.7 / 22.8): A's day-21 split was the suffix prime's
tick, which this intruder does not have. Capture class: `on_minus_off` **+0.7 ms** (unc 0.2) in both passes, the
same sign: under the door the seed's on-tick remainder (the recurrent clone on the owner stream in both arms, the
submit and the ticket) costs 0.7 ms more than the OFF arm's on-tick rows copy; the rows' D2D itself is under that
figure and is not separately resolved. **The subtraction arm did not read as pre-registered:** the prime-only boot
(`MEMRA_PREFIX_CACHE_MB=0`) stalls the tenant **301.5 ms** (IQR 1.5, `stall_min=285.6 stall_max=302.5`, both passes)
while the same prompt's prime inside the cache-on boots stalls **283.7 / 284.4 ms** (IQR 0.2 / 0.1), so `share`
reads -17.8 / -17.1 `isolated` with the wrong sign: a cache-off boot's prime is not the cache-on boot's prime (the
prime arm's spread reaches down to 285.6, the cache-on arms' figure, so the extra 16 to 18 ms is a shape the
cache-off boot takes on most runs, not a constant), and a subtraction across the two boot shapes cannot read the
capture's own share. Stated, not tuned: the `on_minus_off` inside the cache-on pair is the clean quantity of the
capture class, and the arm's own share stays unread by this cell. Both classes' door deltas are positive and under
a millisecond on this card at this entry size; the day-24 arithmetic (153 MB of rows at this card's D2D bandwidth,
well under a millisecond) is consistent with both. Same window, one hold, N=5 per arm per order, both orders, two
passes in opposite order; `executed-not-qualified`; nothing here decides the door.

## Task 4: records and checks

`HOSTPREFIX-DOOR.md`: the hit-gate owed-cell row carries the day-28 `--external-lock` note; new section-A rows for
the receipt's price (both cards, per card) and the isolating stall cell; section B gains the two price rows (the
target card's from A's `DAY22.md` cell (v), the RTX 5090's from Task 3) and the isolating cell's two class rows;
section E names the receipt's pair cost and the isolated restore class. `STATE.md` rewritten; `research/INDEX.md` row
`spill-c-20260919/day28`; `docs/TESTING.md` lock-arms bullet (Task 1). No Rust moved (`cargo fmt` not owed);
`shellcheck -S warning` on the gate (the pre-existing SC2034 only), the fixture and the two drivers clean;
`bash tools/check-flags.sh` no uncovered runtime names; `bash tools/check-conflict-markers.sh` OK; `git diff
--check` clean; `.gitattributes` `*.log -whitespace` in `rtx5090-day28/` and `pro-single-day28/`; no em dash in
any line added today.

## Pushes

`8803f4b6c` (the merge), `002507b4b` (the gate's lock arms, the fixture, TESTING, the CI line), `61dc1a52a` (the
pre-registration, the harness, the drivers, the reading script, the 5090 price receipts), then the closing commit
(this record's Task 2 run and Task 4 sections, the box receipts, the door table, STATE, INDEX), each in
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (printed `UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`,
logged in the clone's `.git/memra-gate-skips.log`). Not merged into main, no PR opened.

## Left as it was, and cleanup

BOX3 reached through the existing control socket only (`ssh -O check` first, `Master running`); `/root/wt-c`
(mine) left detached at `61dc1a52a`, clean; the two bundles removed on both ends; receipts under
`/root/spill-receipts/day28` (mirrored here); `/root/artifacts`, `/root/memra-spill` and other lanes' worktrees and
processes not touched; no server of mine on either card at close (the box: none; the local card: other sessions'
`memra-server` processes seen in `pgrep` only, never inspected or signalled, the 5090 lock free at my last probe).
Local: the price cell took and released the canonical lock once; no `/tmp` scratch left.

## Budget

About 2.9 agent-hours against 4: reading and the merge 0.6, the gate's lock arms, teeth and TESTING 0.4, the harness,
drivers and reading script 0.5, the pre-registration 0.3, the 5090 price cell 0.1, the box shipping, build, run and
its census 0.5, the records and checks 0.5. Blockers: none (the box was free, no lock retry on either card).
