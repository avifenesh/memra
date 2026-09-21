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

**What was there.** The tick-top disconnect sweep (`worker.rs` "DISCONNECT ABORT", gap-scan F8) retires a
closed-channel session before any phase steps it, once per tick; `prefill_tick` primes one take per tick
(1024 tokens; 8192 for a sole fresh request; the whole prompt for the monolithic class) as ONE engine call,
and inside that call the sequential chunk walks stop at every internal chunk (`MEMRA_PRIME_CHUNK` or the
dynamic schedule) to read the chunk's logits back and stamp the odometer (`progress::note_prime_rows`), with
no check of the client. So a disconnected client's prompt ran to the end of the current take.

**What landed (commit `dfbf71ee1`, compile fix `47b03b901`).** `crates/memra-engine/src/progress.rs`: a
thread-local `PrimeCancelScope` guard installs a `Box<dyn Fn() -> bool>` "client gone?" predicate for the
duration of one prime call and restores the previous one on drop (the `?` paths included); a typed
`PrimeCancelled { chunk, rows_done, rows_total }` error; `prime_cancel_point(chunk, rows_done, rows_total)`,
which answers `Ok` with no scope installed, `Ok` while the predicate says the client is there, `Ok` at the
last chunk (a finished take is never cancelled; its logits return and the sweep retires the session), and
`Err(PrimeCancelled)` otherwise. `hybrid_forward.rs`: the three sequential walks ask it right after their
odometer stamp, once per completed chunk and before the next starts: the GEMM chunk loop
(`step35_prime_cache_batch` per chunk), the serial chunk walk (`prime_chunk`, the Qwen3.8 path) and the
single-engine hyper range walk. `worker.rs`: `prefill_tick` installs the scope around `prime_cache_overlaid`
with `{ let tx = s.tx.clone(); move || tx.is_closed() }` (the same predicate the sweep reads); both call
sites match the typed error through `prime_cancelled_abort`, which prints one receipt (`[prime] cancelled at
chunk K (R of T rows of this take primed; fed F, queued Q, prompt P, model M): client gone, nothing
published, cache released at retire`) and calls `abort_log` (now also printing `fed`): the session retires
as an aborted client, `retire_may_park` refuses the park (and a park needs `prefill_done` anyway), the
half-primed `Cache` returns to the pool at drop, no `Event::Error` goes to the closed channel, and the
capture sites (LCP split, grid seed, checkpoint) are never reached because they run only after an `Ok`
prime: nothing partial is published (the capture law). Unit tests: `prime_cancel_point_fires_only_under_an_
installed_scope_and_never_at_the_end`, the wiring gate `the_sequential_prime_walks_ask_the_cancellation_
point` (comment-stripped source, at least three live call sites), 4 passed. No new `MEMRA_*` read, no
flag, no new numeric program: with no scope installed the check is one thread-local read, and with a scope
the walk either continues exactly as before or stops between chunks.

**Not covered, stated.** The pipelined walks (the PP-2 split primes with a `next_slot` in flight, the ppN
wave walk) and `prime_cache_batch` (one call for several sessions) keep the tick-top sweep as their
cancellation point: returning mid-wave leaves another stage's work in flight against a cache the caller is
about to drop, a wider seam (a drain plus the tainted-cache contract) than one check per chunk; and one
member's disconnect cannot stop a wave that is priming its peers. The hyper walker route (`prime_service`)
already yields per chunk per tick under `MEMRA_PRIME_YIELD=1` and is otherwise the monolithic take.

**Gate (`tools/prime-cancel-gate.sh`, serving shape).** Two boots of the same binary (`MEMRA_SERVE_SPEC=0
MEMRA_PREFILL_TICK=8192 MEMRA_PRIME_CHUNK=256 MEMRA_PREFIX_CACHE_MB=2048`, so the whole prompt is one prime
call and the sweep cannot be what stops it): control (cold then warm request of a 6000-word prompt), fault
(a raw-socket streaming request for the same prompt closed after 300 ms, then the cold and warm requests).
On the target card (BOX3, binary `b44cea34…` built from `47b03b901`, under the collector,
`pro-single-day16/box/pcg/`, `--validate` rc=0), verbatim:

```
control: cold sha=f243df4517b99525 prompt_tokens=7508; warm sha=f243df4517b99525 cached_tokens=7488
server: [prime] cancelled at chunk 2 (768 of 7488 rows of this take primed; fed 0, queued 20, prompt 7508, model "gate"): client gone, nothing published, cache released at retire
server: [abort] client disconnected: model "gate", prompt 7508 (0 cached, 0 fed), 0 generated, billed to abort point, 0.34s
ok: stopped at chunk 2 after 768 of 7488 rows (within one chunk of the disconnect, before the take ended)
ok: session retired as a client abort (no park, cache released)
ok: nothing published (no '[prefix-cache] insert' before the next request)
ok: no 'prefill error' path taken
fault: cold sha=f243df4517b99525 prompt_tokens=7508 cached=0; warm sha=f243df4517b99525 cached_tokens=7488
ok: the next cold request restored nothing from the aborted prime (cached_tokens=0)
ok: cold digest unchanged versus the control boot
ok: warm digest unchanged versus the control boot
PRIME-CANCEL GATE: PASS (disconnect_ms=300 words=6000 cold=f243df4517b99525 warm=f243df4517b99525)
```

(The take is 7488 of 7508 rows because the grid seed of memra#602 stops the prime at the aligned boundary;
the cancel fired at the third completed 256-row chunk, 0.34 s after the request, against a prime that
would have run about 2.2 s.) The server's `[abort]` line carries a dash between `generated` and `billed` (pre-existing log text); this quote writes a comma there.

**One numeric program, proven on the changed binary on the target card (under `flock` on the canonical
lock, the hit gate does not take an inherited FD):** `tools/spec-on-cache-hit-gate.sh qwen`:
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok` clauses, spec-on boot with the sampled cells and the
spec-off twin; `pro-single-day16/box/hitgate/`); `qwen-a4-continuation-gate` on a 9296-token prompt:
`one call over 9296 tokens: logits_sha=5a28d463f1d8e148`, `9248 + 48 ok`, `9216 + 80 ok`, `9280 + 16 ok`
(the three grid-aligned splits; three unaligned splits SKIPPED by the gate's grid law),
`A4 CONTINUATION GATE: PASS` (`pro-single-day16/box/contgate/`). Two false starts are kept as the record:
the hit gate refused under the collector (it has no `--external-lock` arm; `hitgate-attempt1-collector-
refused/`) and refused its second boot because its stop matches the server by the name `memra-server`,
which my renamed binary `memra-server-pcg` escaped, leaving my own spec-on server on the port
(`hitgate-attempt2-renamed-binary/`; that server was mine and was stopped by pid; the rerun used the
canonical name).

**Local RTX 5090 Laptop GPU (`rtx5090-day16/`, the Qwen3.5-9B NVFP4 MTP artifact, the same tree's release
binary `binary.sha256`, each gate behind `flock -w 1800 /tmp/memra-5090.lock`; other lanes held the card and
the lock for part of the sitting, never signalled).** `tools/prime-cancel-gate.sh`: `PRIME-CANCEL GATE: PASS
(disconnect_ms=300 words=6000 cold=f243df4517b99525 warm=f243df4517b99525)`, cancel line `[prime] cancelled
at chunk 4 (1280 of 7488 rows of this take primed; fed 0, queued 20, prompt 7508, model "gate")`, abort at
0.32 s, 11 `ok` clauses (`pcg.log`). The 9B and the 27B produce the same 24-token continuation of this
word-list prompt at temperature 0 and the same 7508-token count (one tokenizer family), so the two gates'
digests coincide; each gate compares its own control against its own fault boot, nothing across models.
`qwen-a4-continuation-gate` on the same 9296-token prompt: `one call over 9296 tokens:
logits_sha=fd4ab9787e0a3823`, `9248 + 48 ok`, `9216 + 80 ok`, `9280 + 16 ok`, `A4 CONTINUATION GATE:
PASS` (`contgate.log`). The hit gate's first local attempt refused on the same renamed-binary port trap as
on the box (`hitgate-attempt1-renamed-binary.log`; my own server, stopped by pid); its rerun with the
canonical name is `hitgate.log`, quoted in the "Local hit gate" line of the scope section below.

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

## Scope and effort

Local hit gate (`rtx5090-day16/hitgate.log`, the 9B artifact, canonical binary name): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok`
clauses, the spec-on boot with the sampled cells and the spec-off twin).

Done: the memra#536 census as code reading with line numbers at `21307b636`; the pre-registered stall cell
on the target card class in five arms (prime, demote OFF and ON, promote OFF and ON) against an idle
control, N=5 per arm per order, both orders, validated and replayed, with the reading that the tenant's
tick absorbs the whole of each class (five 290 ms ticks for a 5122-token prime; a 42 ms D2H inside a 131 ms
tick OFF and a 118 ms D2H inside a 207 ms tick ON; a 98 ms promote tick OFF and 176 ms ON); the
cancellation point at the engine's internal chunk boundary with its typed outcome, its worker receipt and
its serving-shape gate, PASS on the target card and on the local 5090, with the hit gate and the
continuation gate green on the changed binary on both cards; the offload design note; the records. Not
done, stated: the pipelined walks and `prime_cache_batch` keep the tick-top sweep; no copy left the tick
(design only, awaiting the lead's ruling on Move 1); the stall cell's second 88 ms demote tick is
observed, not attributed; the issue's 3 GB GLM entry shape is not measured. Development pushes: `21307b636`,
`1646d421b`, `0ea1fd6b1`, `bf248692d`, `dfbf71ee1` (did not compile the server lib; my chain committed
before reading clippy), `47b03b901`, and the records tip, each announced `UNQUALIFIED DEVELOPMENT` by the
hook and logged in `.git/memra-gate-skips.log`; no qualification claimed. About 2.5 agent-hours against the
4-hour budget (the card was busy for two 120 s waits; the stall sitting itself took 11 minutes, the gates
about 4).
