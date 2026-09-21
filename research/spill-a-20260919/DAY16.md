# WP-A day 16: the memra#536 census and stall cell on the target card class, the prime cancellation point, and the offload design note

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `b9eb3638b` (day 15; on `main` through
#613), merged `origin/main` `653c997f4` (#613 and #614, the 16-row prime segment fix) as `21307b636`.
Every push today in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints
`UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-a-20260919 at <sha>; no GPU qualification claimed` and
`pre-push: skip recorded in .git/memra-gate-skips.log`); nothing here claims qualification, every cell
below is `executed-not-qualified`. Commits: the census and pre-registration `1646d421b`, the stall harness
and driver `0ea1fd6b1` (its 64-word list fixed in `bf248692d` after the first BOX3 attempt refused on its
own assertion; the attempt's logs are kept under `pro-single-day16/box/attempt1-fail/`), then the
cancellation point, its gate and the design note, then the records (the branch tip).

## Task 1: the census (`OWNER-THREAD-CENSUS.md`, code reading at tree `21307b636`)

Every class the issue names runs on the one worker thread inside `run`'s tick: prime (`prefill_tick`,
one call per prefilling session per tick, `PREFILL_TICK_T = 1024` tokens, `SOLO_PREFILL_TICK_T = 8192`
for a sole fresh request, the whole prompt for the monolithic E4B class), decode, the D2D snapshot
capture (`prefix_snapshot`, stream-ordered, no host wait, inside the capture publish of `prefill_tick`),
the D2D hit restore (`prefix_restore_at`, stream-ordered, at admission), the D2H demote (OFF:
`dtoh_u8_into_pinned`, one host-blocking `synchronize` PER PLANE; ON: one `TransferEngine` batch), the
H2D promote (OFF: `htod_u8_into` from the pinned lease, no host wait; ON: one `TransferEngine` batch) and
the trim (two owner-stream synchronizes plus `pool_trim_to_zero`, evictions and device trim in the SAME
`TrimPools` call at this tree). The door's finding: `CudaTransfers::new(owner, ..)` takes the worker's
owner stream and keeps it as its only stream (`tier_transfer.rs:355`), `check_thread` refuses any other
thread, and both routes call `synchronize(&ticket)` (a blocking event wait per item) and then drain the
owner stream before `retire`; so the door moved ownership, receipts and the typed unwind onto the engine's
contract, not the copy off the tick. The prime's only cancellation point at that tree is the tick-top
disconnect sweep (`worker.rs:19160`), once per `prefill_tick` call; the engine's internal chunk boundary,
where the odometer stamps, had none.

### The stall cell (pre-registered in `OWNER-THREAD-CENSUS.md`, run as written)

BOX3, one RTX PRO 6000 Blackwell Server Edition at its 600 W limit; `memra-server` built on the box from
`1646d421b` (`pro-single-day16/box/stall-*/ev/binary.sha256`, `a447fec4…`; engine source equal to `main`'s
at `653c997f4`), the Qwen3.8-27B NVFP4-Q5K artifact with its embedded MTP drafter, `MEMRA_CTX=8192
MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0` in every boot (one token per tick, so the tenant's inter-token gap
is the tick), the model page-cached (boots about 10 s). Three boots, one collector lock hold each
(`tools/tier-battery.py --rig pro-single --external-lock`, `lock.json`, 250 ms telemetry, `CELL.jsonl`
`status: executed-not-qualified`, `--validate` rc=0 on all three: `box/stall-*.validate.log`), the driver's
first cell waited 120 s for lane B's `m-gate` lock hold and never signalled it (`box/driver.log`). Harness
`research/spill-a-20260919/stall_cell.py` (client side, stdlib): the tenant is a streaming completion of a
20-token prompt (below the 64-token capture floor), `max_tokens 160`, the intruder fires when the tenant's
24th token arrives; order 1 (`idle`, arm) x 5 then order 2 (arm, `idle`) x 5, N=5 per arm per order, N=10
pooled; the rule line is fixed in the harness and `--replay` recomputed it from every `receipt.json`:
`STALL REPLAY: PASS (replay agrees with the harness's rule line)` five times (`box/replays.log`). Regime
from the collector's sampler (`command.gpu.csv`): cache-off boot 32 to 58 C, 32.6 to 492.7 W under 600 W,
SM 180 to 2422 MHz (273 samples); OFF boot 43 to 50 C, 88.4 to 329.0 W (405 samples); ON boot 40 to 50 C,
88.4 to 329.1 W (443 samples). Zero errors in every cell; the tenant's 160-token text is byte-identical
across all 100 runs and all five cells (`tenant_text_identical=True`); no idle run has a single ITL sample
above three times its own p50.

The intruder's prompt sizes as the tokenizer made them (recorded, not the pre-registered word counts):
the "4096-token" prime is **5120 to 5123 tokens** (the 64-word list tokenizes at 1.25 tokens per word;
the untimed calibration request read 5122); the demote intruder is 95 to 99 tokens (entry 64 tokens,
159.8 MB, the grid seed); the promote intruder is the day-15 pair's 86- and 89-token prompts (entry 64
tokens, hit of 64 with a 22- to 25-token suffix).

Verdict lines, verbatim (`box/stall-*/ev/*/receipt.json` `rule_line`; ms):

`STALL rule cell=stall-prime arm=prime n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.5 idle_p95=14.7 idle_p99=14.9 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.9 arm_p99=295.7 arm_max=315.9 stall_median=301.5 stall_min=285.6 stall_max=302.5 server_demote_ms=[] server_promote_ms=[] intruder_prompt_tokens=[5123, 5122, 5123, 5121, 5123, 5122, 5123, 5120, 5122, 5122] tenant_text_identical=True errors=0`

`STALL rule cell=stall-demote-off arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=15.0 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=87.6 arm_max=132.0 stall_median=117.5 stall_min=75.8 stall_max=118.6 server_demote_ms=[36.9, 42.1, 41.6, 41.8, 41.9, 42.6, 43.0, 41.6, 42.0] server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97] tenant_text_identical=True errors=0`

`STALL rule cell=stall-promote-off arm=promote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=15.2 arm_runs=10 arm_p50=13.4 arm_p95=14.7 arm_p99=16.6 arm_max=133.6 stall_median=85.0 stall_min=84.5 stall_max=120.3 server_demote_ms=[41.2, 41.7, 6.9, 6.2, 6.0, 6.2, 6.1, 6.2, 6.1, 6.1] server_promote_ms=[45.7, 46.1, 11.3, 10.6, 10.4, 10.5, 10.5, 10.6, 10.5, 10.5] intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`

`STALL rule cell=stall-demote-on arm=demote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=87.7 arm_max=207.6 stall_median=193.5 stall_min=75.7 stall_max=194.2 server_demote_ms=[112.8, 118.1, 117.7, 117.2, 117.9, 117.7, 118.4, 117.2, 117.6] server_promote_ms=[] intruder_prompt_tokens=[95, 99, 97, 98, 97, 99, 97, 97, 97, 97] tenant_text_identical=True errors=0`

`STALL rule cell=stall-promote-on arm=promote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.7 arm_p99=16.6 arm_max=211.2 stall_median=162.8 stall_min=162.2 stall_max=197.8 server_demote_ms=[116.7, 117.5, 82.9, 82.3, 82.1, 82.2, 82.2, 82.0, 82.4, 82.2] server_promote_ms=[122.7, 123.5, 88.8, 88.3, 88.0, 88.2, 88.2, 87.9, 88.4, 88.2] intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`

**Reading (the per-run ITL lists are in the receipts; "stretched tick" below means an ITL sample above
three times the run's own p50, a description of the receipts, not a rule).** The tenant's tick is 13.4 ms
and its idle p99 is 14.8 to 14.9 ms in every cell.

- **Prime.** Every one of the 10 runs shows exactly **five stretched ticks of 286 to 316 ms**, summing to
  1462 to 1481 ms against an intruder wall of 1499 to 1518 ms: the 5122-token prime took five
  `PREFILL_TICK_T = 1024` chunks (the tenant was unfinished, so no solo widening) and every chunk held the
  tenant's decode for the chunk's whole compute (about 290 ms per 1024 rows, 3.5k rows/s on this card
  with the plain program). The tenant paid the full prime, in five installments; the intruder's prompt
  never touched a cache (cache-off boot).
- **Demote OFF.** Two stretched ticks per run: the first is **89 ms without a demote** (run 2, the first
  insert of the boot, `server_demote_ms=[]`) and **126 to 132 ms with one** (the 64-token entry's D2H,
  36.9 to 43.0 ms by the server's own line, the day-15 first-touch shape of a fresh pinned region on
  every run because each entry is new), so the demote's share of the tenant's tick is the server's
  demote time; the second tick is 87 to 88 ms in every run including the no-demote run and is therefore
  not the demote (the intruder's own prefill-done work of the next tick; not attributed further here).
- **Demote ON.** The same two-tick shape; the first tick is **202 to 208 ms with a demote** (server
  demote 112.8 to 118.4 ms): the door adds about 76 ms to the tenant's tick per 160 MB entry on this
  card, the same figure as the day-15 pair's ON minus OFF steady-state demote (82 against 6.1) plus the
  first-touch shape, because here every entry is fresh.
- **Promote OFF.** One stretched tick per run: **134 ms on the first two runs** (first touch, server
  demote 41 and promote 46) and **98 to 99 ms steady** (server promote 10.4 to 11.3 plus the inline demote
  6.0 to 6.9). The copies are about 17 ms of that steady tick; the rest is the hit's admission restore
  (D2D), the 22- to 25-token suffix prime and the new entry's capture and insert, all inside the same
  tick.
- **Promote ON.** **211 ms on the first two runs, 176 to 177 ms steady** (server promote 87.9 to 88.8,
  inline demote 82.0 to 82.9): the door's promote-window cost on the tenant's tick is about 78 ms
  steady (176 against 98), matching the server's own promote line (88.2 against 10.5).

What this is and is not: one card class, one host, one artifact, `MEMRA_SERVE_SPEC=0`, 64-token entries
(160 MB; the issue's 3 GB entry is a 135k-token GLM shape and is not measured), N=5 per arm per order in
both orders inside one sitting; no number here is divided into a number from another box. The issue's
acceptance ("decode ITL p99 for peers unchanged while a demotion and promotion runs") is not met by any
arm today: the arm p99 is 87.6 to 295.7 ms against an idle p99 of 14.8 ms wherever the intruder's
work lands on more than one percent of the tenant's ticks (prime, demote), and the promote's single
stretched tick sits above p99 as the max (98 to 211 ms). These are the baselines the design note's cells
compare against.

## Task 2: the cancellation point in the prime loop

(filled after the gates; see below)

## Task 3: `OWNER-THREAD-OFFLOAD.md`

Written from the census and the door's path: the door's D2H and H2D are the one pair whose contract is
already the right shape for an off-tick move (fences are events, completion is an event per item,
publication is `ready_view` after the consumer fence; the copy sits on the owner stream only because the
constructor is handed it), and what Move 1 needs is a second per-device stream in `CudaTransfers`, the
waits turned into tick-top `poll`s, a `Demoting`/`promoting` entry state the LRU and admission refuse to
serve or evict, one more ledger term, two new fault cells and the stall cell repeated. The D2D capture and
restore (Move 2) have no contract today; the note states the producer/consumer ordering each needs, why
the recurrent-state copies stay on the owner stream at the boundary (the capture law), why a hit on a
`Capturing` entry must be a miss, and the delayed-copy-stream fault that proves the restore ordering under
the one-program law. Prime, trim and decode stay on the tick, with the reasons. Priced at about two
agent-days (Move 1) and four (Move 2, after Move 1); neither is started here.
