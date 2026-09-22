# WP-A day 22: Move 2 slice 3, the receipt term for the two D2D classes and the `d2d-delay` fault

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `ff3f0f6f5`; `origin/main` `5df11152f` (integ36: the
lead's #634 fixes and lane C's day-24 gate accounting) merged `--no-ff` as `d2530e362` (clean; marker census OK). Every
push today in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an
engine-touching range otherwise); no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Task 1, pre-registration (this section is committed before any slice-3 code)

### What slice 3 fixes

After slices 1 and 2 a capture or restore item lands with `checksum = None`: the host-contract gate
(`Completion::require`) refuses it `Corrupt` by the slice-1 clause, and `capture_landed` / `restore_landed` prove only
that the completion EVENT fired. Nothing witnesses that the destination holds the source's bytes. The pre-registered
rule of day 19 chose (a), a device-side digest, unless its cost on the copy stream exceeds the copy's own time on the
target card (measured, N>=5, both orders); cell (v) below measures that and the reading is reported verbatim whatever
it says.

### The receipt, defined

- **The program.** `memra_tier::contracts::checksum` is SHA-256 over host bytes and has no device twin in this
  engine (no digest kernel exists in `crates/memra-engine/cu/`); a host SHA-256 over a D2H readback of 158 MB would put
  about 100 ms of hashing per class on the owner thread, the stall Move 2 exists to remove. So the receipt is a
  copy-stream REDUCTION with a CPU oracle, `memra_tier::conformance::d2d_receipt::receipt_digest`: the span's bytes as
  little-endian u64 words `w_j` (the last partial word zero-padded), four lanes `L_l = sum_j mix64(w_j + (j + 1) * C_l)`
  in wrapping u64 arithmetic (l = 0..4, `mix64` the splitmix64 finalizer, `C_l` four fixed odd constants), and the
  32-byte `Digest` is `le_bytes(L_l ^ mix64(n + C_l))` for l = 0..4 where `n` is the byte count. Wrapping integer sums
  are order-independent, so the device's block and atomic order cannot change the value and the CPU oracle is the same
  function over the same bytes. This is a receipt over KV BYTES, not a numeric program over tokens: no forward, no
  KV encoding, no tolerance. The device kernel `d2d_receipt_digest` (`cu/tier_receipt.cu`, its own fatbin) is bound to
  the oracle by a GPU cell (device digest of a readback equals `receipt_digest` of the same bytes, on two spans).
- **Where each digest is taken.** On the COPY stream, in this order, per item: the wait on the producer event, the
  SOURCE span's digest (behind the producer fence: it witnesses the bytes the producer finished writing), the
  `memcpy_dtod`, the DESTINATION's digest (after the copy), an asynchronous D2H of the eight lanes into a pinned
  receipt buffer (the engine's own `PinnedBacking`), then the completion event. `progress` reads the lanes only after
  the completion event is observed complete: `s.checksum = Some(destination digest)` and the item's expectation
  checksum is the source digest, so the existing gate clause `s.checksum != Some(e.checksum) => Corrupt` is the
  comparison, unchanged. A receipt-less item (`checksum = None`) stays refused by the same clause.
- **The engine's answer.** `CudaTransfers::d2d_receipt(&ticket)`: `NotReady` before the landing; after it the
  per-item source and destination digests and the gate's verdict (`Completion::require`, `device = true`, so for the
  restore class it is read after `install_consumer_wait`). The receipt LINE names both digests: one SHA-256 over the
  ordered per-item source digests and one over the ordered destination digests (the D2H line's `checksums_sha256`
  shape), plus `require=ok` or `require=<Error>`.
- **Typed failure.** A mismatch is `HostCaptureFailure::ReceiptMismatch` / `HostRestoreFailure::ReceiptMismatch`: the
  ticket has landed, so it retires (`retire(None)`) and acknowledges cleanly (nothing leaked), the capture's planes come
  back through their twins and drop with the shell (NOTHING published), the restore's destination cache drops and its
  source pin is released (NOTHING primed on), the tier latches off (`TIER DISABLED`) and the route latches to the tick
  program for the boot (`CAPTURE OFF-TICK DISABLED: D2D capture receipt mismatch ..` /
  `RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch ..`). The request that owned the restore re-admits and the
  OFF program serves it on a fresh cache. `Latched` (a lost observation) keeps its day-20 and day-21 shape.
- **The one-program law.** The receipt reads bytes and changes none: the copies, their order, their streams and the
  request's program are the day-21 tree's statement for statement. The identity gates (OFF against ON, `teeth=0`) and
  the hit gate's identity clause remain the proof; a digest of the KV bytes is evidence about the copy, never about
  the tokens.

### The `d2d-delay` fault door (the red arm)

`MEMRA_KV_HOST_FAULT=d2d-delay-capture` and `MEMRA_KV_HOST_FAULT=d2d-delay-restore` (two values of the existing fault
family; ONE boot arms exactly one D2D route, as the `contract-*` values arm one side; one-shot: the first submit of
that class takes it; their `docs/FLAGS.md` row lands in the same commit as the read). The engine's
`inject_d2d_early_reader(delay_ns)` arms the next D2D submit: (1) a spin kernel of `delay_ns` (200 ms;
`tier_delay_spin`, `globaltimer`) runs on the copy stream between the producer wait and the copy, so the copy cannot
have started when an early reader looks; (2) the DESTINATION digest is issued by an UNORDERED reader, the OWNER stream
at submit after the producer fence with no wait on the copy's event, exactly the read a publication into the index or
a first prime chunk issued before the completion event would make; an owner-stream event recorded after that read is
waited on by the copy stream before its completion event, so the lanes `progress` reads are the early reader's. The
source digest stays on the copy stream behind the producer fence. Expected under the fault: the destination digest
differs from the source digest (the destination is a fresh plane the delayed copy has not reached), `require` answers
`Corrupt`, the settle refuses typed, nothing is published or primed on, the tier and the route latch. A matching
receipt under the fault is a FAIL of the cell (the receipt did not catch the early reader), never a pass.

### Frozen schedule: `d2d_receipt_witnessed` (`memra-tier`, CPU bindings)

Over a `D2dReceiptFixture` (submit, poll, landed, `receipt` naming both digests, `require_receipt`, the caller's
`publish` that asks `require_receipt` first, retire, acknowledge, `latched`):

1. Before the event: no receipt (`NotReady`); a publish is refused `NotReady`.
2. Landed with a matching receipt: the receipt names both digests and they are equal; `require` is `Ok`; the
   publication succeeds exactly once; `retire(None)`, `acknowledge`.
3. Landed with NO witnessed checksum (a receipt-less item): `require` is `Corrupt`; the publish is refused; the
   caller latches; the ticket still retires and acknowledges (nothing leaked).
4. Landed with a MISMATCHING receipt (the early reader): the receipt names both digests and they differ; `require`
   is `Corrupt`; the publish is refused; the caller latches; the ticket retires and acknowledges.

Bindings: `day22_d2d_receipt_matching_publishes_once`, `day22_receipt_less_item_is_refused_and_latches`,
`day22_red_arm_early_reader_receipt_is_refused_and_latches`, plus `day22_receipt_digest_oracle_is_stable` (the CPU
oracle on fixed vectors: a one-byte flip, a one-byte extension and a transposition each change the digest). The
day-20 and day-21 schedules keep their receipt-less clause for fixtures without a witness and name this schedule for
the witnessed one.

### Cells (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; the collector; `executed-not-qualified`)

- The day-21 gate table (identity default and plain, OFF and ON; failure OFF and ON; the fault gate with the two new
  cells; twin OFF and ON; the hit gate OFF and ON; the unit cells `option_b_*`, `option_c_*`, the engine `d2d_*`
  cells including the two new ones). If identity, fault or hit is red in either arm the slice does not stand: `wip:`
  with the red receipt. The plain identity ON arm is also read for `D2D capture receipt .. require=ok` and
  `D2D restore receipt .. require=ok` lines (the green arm of the receipt, live).
- **Fault gate cell `d2d-capture`** (plain arm `MEMRA_SERVE_SPEC=0`, door ON, `MEMRA_KV_HOST_FAULT=d2d-delay-capture`;
  r1 P_A, r2 P_A): two completions served; door ON; exactly one `contracts door D2D capture receipt: .. require=Corrupt`
  line whose two named digests differ; exactly one `CAPTURE OFF-TICK DISABLED: D2D capture receipt mismatch` line and
  a `TIER DISABLED` line; zero `capture published off the tick`; zero `hit:` (nothing was published before r2); one
  `insert (seed)` AFTER the refusal (r2's capture ran on the tick); r1's text byte-equal to r2's (the tick program
  served both). A receipt that matched under the fault fails the cell.
- **Fault gate cell `d2d-restore`** (same arm, `MEMRA_KV_HOST_FAULT=d2d-delay-restore`; r1 P_A, r2 P_A): two
  completions; door ON; exactly one `D2D capture receipt: .. require=ok` and one `capture published off the tick`
  (r1's seed, green); exactly one `restore submitted off the tick`; exactly one `contracts door D2D restore receipt:
  .. require=Corrupt` whose digests differ; one `RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch`, a
  `TIER DISABLED`; zero `restore landed off the tick`; one `hit:` line AFTER the refusal (the tick program restored
  r2); r1's text byte-equal to r2's. A receipt that matched under the fault fails the cell.
- **Cell (v), the digest's price** (engine GPU cell `d2d_receipt_digest_price_against_the_copy`, one collector hold):
  on one 158 MB span (the 27B plain entry's size class, 32 planes at the identity gate's shape is about 1.9 MB of KV
  rows, so the cell uses the entry-size class the restore stall cell moved), event-timed on the copy stream, N=5 per
  order, both orders (copy then digest, digest then copy), medians of `memcpy_dtod` and of ONE digest kernel, plus
  the pair (source and destination digests) as `2 x` the digest median. Rule (day 19, unchanged): the receipt stays
  (a) unless the pair's median exceeds the copy's median; if it does, the reading is reported verbatim and the
  door review reads it, nothing is relaxed here today.

### The lead's note, received mid-day (revuto on #638, the take-ready pin leak)

The lead fixed a slice-2 leak on `lane/spill-integ37-20260922` (`0d021c10a`): `host_restore_take_ready`'s
fail-closed arm read `r.pin.take()` AFTER the `let`-else scrutinee `(r.cache.take(), r.pin.take())` had moved the
pin into the dropped tuple, so a ready restore with no cache left its source entry pinned for the boot. The rule:
decide the shape with borrows (`is_none()` / `is_some()`), THEN move. The lead's branch is merged into this lane
below. Audit of every take-and-match in the worker and the engine for the same shape (`rg 'match \(.*\.take\(\)|let
.*= \(.*\.take\(\)'`): `host_demote_settle_with` (`(pending.dead.take(), pending.contract.take())`) binds BOTH moved
values in its refutation arm and uses both, not the bug shape; the three `match (x.is_some(), y.take())` sites
(demote, capture, restore) decide the first with a borrow and bind the taken second in every arm, not the bug shape;
the only instance was the one the lead fixed. Slice 3 adds none: the engine's `progress` decides the receipt with
borrows (`match &e.receipt`, `let Some(lanes) = &receipt_lanes`), the capture's mismatch arm consumes `registered`
in a loop with no refutation, and the restore's mismatch arm `take`s the cache and the pin as separate statements.

The lead's second note (revuto round 2 on #638, `40964560a` on the same branch): the run loop's parked-only bounded
wait now covers a not-ready `Restoring` request (a request parked on its restore was spinning the owner thread as
#627's promote did), and `host_restore_probe_decision` returns `DropOrphan` for another request's ready restore only
past `RESTORE_READY_TICKS` (within the grace it is `Through`, the state left for its owner). Slice 3 parks a request on
NO new pending state and introduces NO new ready state: the receipt rides the existing `Capturing` and `Restoring`
states (the same tick-top polls, the same parks, the same grace), so neither the wait's guard nor the probe's grace
gains an arm today. The lead's branch tip `40964560a` is merged into this lane as `ac1ac91c5` (clean; marker census
OK), so the target-card gates below run on a tree that carries both fixes.

## Task 1, what landed (under `MEMRA_KV_HOST_CONTRACTS=1`, default OFF, decide-by 2026-10-05)

- `crates/memra-tier/src/conformance/d2d_receipt.rs`: the receipt program (`RECEIPT_LANES`, `mix64`,
  `receipt_lanes`, `receipt_digest_from_lanes`, `receipt_digest`), `ReceiptTerm`, the `D2dReceiptFixture` trait,
  the schedules `d2d_receipt_witnessed` (rules 1 and 2) and `d2d_receipt_refused` (rules 3 and 4). Bindings
  `tests/contracts/d2d_receipt_bindings.rs`: `day22_d2d_receipt_matching_publishes_once`,
  `day22_receipt_less_item_is_refused_and_latches`, `day22_red_arm_early_reader_receipt_is_refused_and_latches`
  (the matching schedule fails on the early-reader fixture, asserted), `day22_receipt_digest_oracle_is_stable`.
  The day-20 and day-21 receipt-less clauses renamed as such by name. `memra-tier` contracts `81 passed` (77 before).
- Engine (`crates/memra-engine/cu/tier_receipt.cu`, its own fatbin `MEMRA_TIER_RECEIPT_FATBIN`;
  `tier_transfer.rs`): `d2d_receipt_digest` (four wrapping u64 lanes, grid-stride over LE words, an aligned u64
  path and a byte path for tails and unaligned spans, warp-shuffle block reduction, one atomic per lane per block)
  and `tier_delay_spin` (`%globaltimer`); `ReceiptKernels` loaded by `new_with_copy_stream` (a module that will not
  load is a construction refusal); `ReceiptScratch` per D2D batch (64 bytes per item on the device, a pinned
  `PinnedBacking` twin, the receipt event), allocated BEFORE the batch's charge so a refusal submits nothing; in both
  `submit_d2d_capture` and `submit_d2d_restore`: the producer wait, the source digest, the copy, the destination
  digest, per item, then ONE D2H of the lanes and the receipt event (`seal_receipt`); `progress` lands a D2D item
  only once the receipt event is complete and fills `s.checksum` (destination) and the expectation (source) from the
  lanes; `d2d_receipt(&ticket)` (`NotReady` before the landing; the per-item `ReceiptTerm`s and the gate's
  verdict); `inject_d2d_early_reader(delay_ns)` (the one-shot fault: `tier_delay_spin` on the copy stream ahead of
  the copy, the destination digest on the OWNER stream at submit, an owner event the copy stream waits on before
  its completion event); `D2D_DELAY_FAULT_NS` = 200 ms. `ready_view` and `take_destination` keep refusing a D2D item
  BY DIRECTION (`Unsupported`), so the witnessed checksum opens no engine-side publication. Census
  `d2d_capture_rules_are_as_stated` and `d2d_restore_rules_are_as_stated` extended (the order wait, delay, source
  digest, copy, destination digest, early reader, event; the scratch before the charge; the fault RECORDS one owner
  event and installs no owner wait). GPU cells: the two day-20/21 cells now assert the receipt (checksum = the
  oracle, `d2d_receipt` verdict `Ok`, `ready_view` `Unsupported`; the restore's verdict `NotReady` before the
  install), plus `d2d_receipt_digest_matches_the_cpu_oracle` (three spans, a one-byte flip moves the lanes),
  `d2d_early_reader_fault_is_refused_by_the_receipt` (verdict `Corrupt`, the destination digest is the zeroed
  plane's, the ticket leaves cleanly, the planes hold the late copy; a second batch matches: one-shot) and
  `d2d_receipt_digest_price_against_the_copy` (cell (v)). `docs/KERNELS.md` rows for both kernels.
- Worker (`crates/memra-server/src/worker.rs`): `HostContractFault::{D2dDelayCapture, D2dDelayRestore}` from
  `d2d-delay-capture` / `d2d-delay-restore`; `take_fault` now filters to Move 1 faults (`is_move1`), so a D2D fault
  is never consumed by the D2H or H2D route; `take_d2d_fault(capture)` per class; the arming before each submit
  with one `capture fault armed (..)` / `restore fault armed (..)` line; `HostCaptureFailure::ReceiptMismatch`,
  `HostRestoreFailure::ReceiptMismatch`; `host_kv_planes_settle_capture` reads `d2d_receipt` after the landing,
  prints `[prefix-cache] contracts door D2D capture receipt: ticket issuer=.. seq=.. epochs=.. items=N bytes=B
  source_digests_sha256=.. destination_digests_sha256=.. require=ok|<Error>` and, on a refused verdict, retires and
  acknowledges, takes every fresh plane back through its twin and DROPS it, and returns `ReceiptMismatch` (the
  shell drops unpublished; `host_capture_latch` prints `CAPTURE OFF-TICK DISABLED: D2D capture receipt mismatch ..`
  and the tier latches); `host_kv_planes_settle_restore` reads `d2d_receipt` after the install (the gate requires
  the fence), prints the `D2D restore receipt` twin, retires and acknowledges, and on a refused verdict returns
  `ReceiptMismatch`: the machine DROPS the cache (the copy landed), hands the pin back for release, and
  `host_restore_latch_landed` prints `RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch ..; the destination
  cache dropped and the source pin released (the copy had landed); every later hit restores on the tick`, the tier
  latches; the parked request re-admits to the OFF program. `d2d_receipt_digests_hex` folds the ordered per-item
  digests (a receipt-less item as 32 zero bytes) in the D2H line's shape. `docs/FLAGS.md`: the fault row's two
  values and their description; the door row's day-22 sentence. No new flag beyond the two fault values; no new
  numeric program for the KV bytes (the receipt reads and changes none); `unsafe` only at the two documented kernel
  launches and the pinned scratch's zero-fill (the existing `PinnedBacking::alloc` contract); no external dependency.
- `tools/kv-host-contract-fault-gate.sh`: cells `d2d-capture` and `d2d-restore` exactly as pre-registered (plain
  arm, one boot each, r1 P_A then r2 P_A; `receipt_digests_differ` reads the one refused receipt line and requires
  its two digests to differ; `texts_equal` reads r1 and r2).

## Task 2, the target card: gates (BOX3, one RTX PRO 6000 Blackwell, 600 W; tree `ac1ac91c5`; receipts `pro-single-day22/box/`)

Built on the box from the shipped bundle (`build rc=0` in 3m 07s, `box/build.log`; binary sha256 in `box/binary.sha256`).
The collector cell `gates` (`tools/tier-battery.py --rig pro-single --external-lock`, `/tmp/memra-gpu.lock`, `LOCK.json`
owner `collector`) ran every gate in one lock hold; the hit gate under its own `flock` on the canonical lock; the unit
cells in a second collector hold. The lock was free at the start (the card idle at `0 MiB`); no retry was needed.
Every line verbatim, N=1 per gate cell, the card's regime in the collector's `command.gpu.csv` beside each cell;
`executed-not-qualified`.

| cell | arm | verdict line, verbatim |
|---|---|---|
| identity-default-off | door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-default-on | door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| failure-off | door OFF | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| failure-on | door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| contract-fault | ON by construction, 8 cells | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` |
| twin-off | door OFF | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` |
| twin-on | door ON | the identical line, `-> PASS` |
| hitgate-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |
| hitgate-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |

**The fault gate's two new cells, verbatim (`gates/contract-fault.log`; the six day-15/16 cells green as before).**

```
== cell d2d-capture: MEMRA_KV_HOST_FAULT=d2d-delay-capture (one-shot, the D2D capture receipt's red arm) ==
  ok: d2d-capture: two completions served
  ok: d2d-capture: door ON with the transfer engine
  ok: d2d-capture: the fault was armed on the first capture
capture receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..
  ok: d2d-capture: exactly one refused D2D capture receipt naming two different digests
  ok: d2d-capture: no capture receipt was accepted (the route latched on the first)
  ok: d2d-capture: the capture route latched typed on the mismatch
  ok: d2d-capture: the tier latched off
  ok: d2d-capture: nothing was published off the tick
  ok: d2d-capture: nothing was hit (no entry existed before r2)
  ok: d2d-capture: r2's capture ran on the tick after the latch (an insert follows the refusal)
  ok: d2d-capture: r1's text equals r2's (the tick program served both)
  ok: d2d-capture: no ticket leaked (no Capacity refusal, no leaked wording)
== cell d2d-restore: MEMRA_KV_HOST_FAULT=d2d-delay-restore (one-shot, the D2D restore receipt's red arm) ==
  ok: d2d-restore: two completions served
  ok: d2d-restore: door ON with the transfer engine
  ok: d2d-restore: r1's capture receipt was accepted (the green arm, live)
  ok: d2d-restore: r1's capture published off the tick
  ok: d2d-restore: the fault was armed on the first restore
  ok: d2d-restore: exactly one restore was submitted off the tick
restore receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..
  ok: d2d-restore: exactly one refused D2D restore receipt naming two different digests
  ok: d2d-restore: no restore receipt was accepted
  ok: d2d-restore: the restore route latched typed on the mismatch
  ok: d2d-restore: the tier latched off
  ok: d2d-restore: nothing landed off the tick (nothing primed on the refused cache)
  ok: d2d-restore: the tick program restored r2 after the refusal
  ok: d2d-restore: r1's text equals r2's
  ok: d2d-restore: no ticket leaked (no Capacity refusal, no leaked wording)
KV-HOST-CONTRACT-FAULT GATE: ALL GREEN
```

The server logs behind them (`contract-fault/d2d-capture-server.log`, `d2d-restore-server.log`), the typed lines in
order, digests cut to 16 hex here (full in the logs). Capture: `capture fault armed (MEMRA_KV_HOST_FAULT=d2d-delay-capture):
the copy is delayed and the destination digest is read early; the receipt must refuse`; `capture submitted off the tick
(seed): 64 tokens, 32 planes (158.8MB) on the contracts door's copy stream; ..`; `[prefix-cache] contracts door D2D
capture receipt: ticket issuer=2 seq=1 epochs=0/1/1 items=32 bytes=1900544 source_digests_sha256=d11e5c4b614f3a68..
destination_digests_sha256=1462aed093ef5fee.. require=Corrupt`; `[prefix-cache] CAPTURE OFF-TICK DISABLED: D2D capture
receipt mismatch (require=Corrupt, source_digests_sha256=d11e5c4b614f3a68.., destination_digests_sha256=1462aed093ef5fee..)
on ticket seq=1; retired and acknowledged, the fresh planes dropped, nothing published; every later capture runs on the
tick`; `[prefix-host] TIER DISABLED: D2D capture receipt mismatch (..)`; then r2's on-tick `[prefix-cache] insert (seed): 64
tokens, 158.8MB (..)`; no `hit:` line, no `capture published off the tick` line. Restore: r1's `D2D capture receipt: ..
seq=1 .. source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=d11e5c4b614f3a68.. require=ok` and `capture
published off the tick (seed): 64 tokens complete after 2 poll(s), 91.5ms from submission to completion, 91.5ms to
publication (tick-top poll)`; `restore fault armed (MEMRA_KV_HOST_FAULT=d2d-delay-restore): ..`; `restore submitted off
the tick: 64 tokens, 32 planes (158.8MB), ticket seq=2 on the contracts door's copy stream; recurrent state copied on the
owner stream; request parked`; `[prefix-cache] contracts door D2D restore receipt: ticket issuer=2 seq=2 epochs=0/1/1
items=32 bytes=1900544 source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..
require=Corrupt`; `[prefix-cache] RESTORE OFF-TICK DISABLED: D2D restore receipt mismatch (..) on ticket seq=2; retired and
acknowledged, nothing primed on the destination; the destination cache dropped and the source pin released (the copy had
landed); every later hit restores on the tick`; `[prefix-host] TIER DISABLED: ..`; then the tick program's `[prefix-cache]
hit: 64 of 89 prompt tokens from cache (model gate)`; no `restore landed off the tick` line.

**Readings.** (1) The early reader's destination digest `1462aed093ef5fee..` is the same in both cells: it is the fold of
32 zeroed planes of the same sizes (a fresh capture plane and a fresh session cache both start zeroed), and it differs from
the source fold `d11e5c4b614f3a68..`; the receipt caught the unordered reader in both classes. (2) The source fold is the
SAME string as the identity plain-ON arm's seq=1 capture and its two restores (`gates/identity-plain-on/`): the same
prompt's 64-token KV rows on this artifact, digested three ways (capture source behind the fence, capture destination after
the copy, restore source behind the fence) give one value, the receipt's consistency across classes read live. (3) A
receipt that matched under the fault would have failed both cells; neither did.

**The green arm, live (identity plain ON, `gates/identity-plain-on/`).** Two capture receipts and two restore receipts,
every one `require=ok` with equal digests: `D2D capture receipt: ticket issuer=2 seq=1 .. items=32 bytes=1900544
source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=d11e5c4b614f3a68.. require=ok`, `.. seq=3 ..
source_digests_sha256=9ebd9a233e35.. destination_digests_sha256=9ebd9a233e35.. require=ok`, `D2D restore receipt: .. seq=6
.. source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=d11e5c4b614f3a68.. require=ok` and `.. seq=7 ..`
(the same). Zero `require=Corrupt`, zero latch lines in any ON arm. The default (spec) ON arm carries zero receipt lines:
its publishes are spec-boundary (the tick program), as pre-registered; the hit gate's arms likewise (draft-bearing).

**Unit cells, first sitting (tree `ac1ac91c5`, `pro-single-day22/box-run1/unit/`), verbatim.** Server: `test result: ok. 8
passed; 0 failed; 0 ignored; 0 measured; 812 filtered out; finished in 0.28s` (`option_b_*`, `option_c_*`). Engine:
`d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`,
`d2d_receipt_digest_matches_the_cpu_oracle ... ok`,
`d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait ... ok`,
`d2d_early_reader_fault_is_refused_by_the_receipt ... FAILED` (`assertion `left == right` failed: the early reader saw
the fresh plane`: the destination digest was neither the zeroed plane's nor a stale value),
`d2d_receipt_digest_price_against_the_copy ... FAILED` (`called `Result::unwrap()` on an `Err` value:
DriverError(CUDA_ERROR_INVALID_HANDLE, "invalid resource handle")` at `elapsed_ms`); `test result: FAILED. 3 passed; 2
failed; 0 ignored; 0 measured; 554 filtered out; finished in 2.77s`. The three landed cells (the two classes with their
receipts, the oracle) passed; the fault gate's two cells had already passed on the same tree on the served path.

**Finding (recorded before the fix).** cudarc 0.19 tracks a `read` and a `write` event per slice and, when the context
"manages stream synchronization" (multi-stream mode, entered by `new_stream`, AND event tracking on), a use of a slice
on another stream first waits on the slice's last write and records its own use. The ENGINE runs with that tracking
DISABLED (`Engine::new`, `gpu.ctx.disable_event_tracking()`, the `MEMRA_EVT=1` escape hatch keeps it), so every engine
slice is untracked and the receipt's ordering in the server is exactly the explicit fences of this slice: that is why
the served fault gate's early reader was early and both cells refused. The unit fixture's plain `CudaContext::new(0)`
kept tracking ON, so cudarc made the fault's owner-stream reader wait on the copy-stream memcpy's write event: the
"early" reader read the LANDED copy (its digest was the oracle's), the receipt matched, and the cell's stale-digest
assertion failed. A second gap, real in the server too and closed in the same fix: the lanes' zero-fill was issued on
the copy stream by `alloc_zeros` while the fault's owner-stream reader added into the lanes with no fence between them
(it held on host timing: the memset was long done). The price cell failed on cudarc's default event flags
(`CU_EVENT_DISABLE_TIMING`). Fixes (`bb1a2b212`): the lanes are zeroed on the OWNER stream and fenced by a `zeroed`
event the copy stream waits on before its first digest (census: `wait < zeroed < delay`); the receipt fixture disables
event tracking exactly as the engine does (documented at the fixture); the price cell records `CU_EVENT_DEFAULT`
events. No served-path statement changed other than the lanes' stream and the fence. Under `MEMRA_EVT=1` the fault's
early reader would be ordered behind the copy by cudarc and the fault gate's two cells would FAIL as "a receipt that
matched under the fault": stated here, not hidden; the door's gates do not run under that hatch.

## Task 2, the target card, second sitting: every gate on the fixed tree (`bb1a2b212`, binary `2261af788b0a…`; receipts `pro-single-day22/box/`)

The fix touches a served-path statement (the lanes' stream and its fence), so the whole driver ran again on the fixed
tree (`build rc=0` in 43.97 s; the collector's `gates` cell waited one 120 s retry on a busy lock held by another lane,
never inspected, then ran in one hold). Every line verbatim, N=1 per gate cell, `executed-not-qualified`:

| cell | arm | verdict line, verbatim |
|---|---|---|
| identity-default-off | door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-default-on | door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-off | `MEMRA_SERVE_SPEC=0` | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity-plain-on | `MEMRA_SERVE_SPEC=0`, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (4 receipt lines `require=ok`, 0 `Corrupt`) |
| failure-off | door OFF | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| failure-on | door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| contract-fault | ON by construction, 8 cells | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`; the two D2D cells' refused lines again `capture receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..` and `restore receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..` |
| twin-off | door OFF | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` |
| twin-on | door ON | the identical line, `-> PASS` |
| hitgate-off | door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |
| hitgate-on | door ON | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` |
| unit-cell (`option_b_*`, `option_c_*`) | the copy-stream engine | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 812 filtered out; finished in 0.28s` |
| unit-cell (engine `d2d_*`) | the two D2D classes, the oracle, the early reader, cell (v) | `d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event ... ok`, `d2d_early_reader_fault_is_refused_by_the_receipt ... ok`, `d2d_receipt_digest_matches_the_cpu_oracle ... ok`, `d2d_receipt_digest_price_against_the_copy ... ok`, `d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait ... ok`; `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 554 filtered out; finished in 3.19s` |

**Reading.** Identity, fault and hit are green in both arms on the target card class, the pre-registered condition: the
slice STANDS under the door. Commits keep their `wip:` prefix (history is not rewritten); this record and
`OWNER-THREAD-OFFLOAD.md` are where the slice is promoted to "landed under the door".

### Cell (v), the receipt's price against the copy (engine cell, one collector hold, the same sitting), verbatim

`D2D-RECEIPT PRICE order=copy-first bytes=165675008 n_per_arm=5 copy_ms=[0.158496, 0.157376, 0.158208, 0.155296, 0.158112]
copy_median=0.158 digest_ms=[0.170272, 0.167584, 0.167424, 0.167456, 0.167712] digest_median=0.168 pair_median=0.335
pair_over_copy=2.12`

`D2D-RECEIPT PRICE order=digest-first bytes=165675008 n_per_arm=5 copy_ms=[0.148736, 0.157408, 0.156, 0.156896, 0.156128]
copy_median=0.156 digest_ms=[0.13568, 0.169696, 0.165632, 0.167456, 0.167552] digest_median=0.167 pair_median=0.335
pair_over_copy=2.15`

Event-timed on the copy stream over one 158 MiB span, N=5 per order, both orders, one warm pass first, the card's regime
in the collector's `command.gpu.csv` (the 600 W envelope). **Verdict by the pre-registered rule (day 19, restated in
Task 1 today): ONE digest costs 1.06x to 1.07x the copy's own time (0.168 against 0.158 ms; 0.167 against 0.156 ms) and
the PAIR the receipt needs (source and destination) costs 2.12x to 2.15x the copy (0.335 against 0.158 ms). The rule's
clause "unless the digest's cost on the copy stream exceeds the copy's own time" is MET on this card in both orders.**
By the day-19 text the consequence is that (b), the `Unwitnessed` arm, becomes the receipt and (a) stays a diagnostic;
today's brief ordered (a) as the deliverable and Task 1 pre-registered that this reading is reported verbatim and read by
the door review, so nothing is relaxed and nothing is decided here: the receipt as landed proves the bytes (the fault
gate's two red arms are the evidence that (b) could not give), and its price on the copy stream is stated as measured. What
the review should weigh, stated and not argued: the cost sits on the COPY stream, off the tick (the owner thread reads
2 KiB of pinned lanes at the settle); at 158 MB the pair is 0.34 ms against a 14 ms tick; the day-19 rule compared the
digest to the copy and did not weigh which stream carries it. A reading, not a verdict on the door.

## Task 3: records and checks

`STATE.md` rewritten (day 22). `OWNER-THREAD-OFFLOAD.md`: the day-22 section (pre-registration pointer, what landed, what
Move 2 still owes, cell (v)'s reading). `research/INDEX.md` row `spill-a-20260919/day22`. `docs/FLAGS.md`: the fault row's
two values, the door row's day-22 sentence, `MEMRA_TIER_RECEIPT_FATBIN` in the build-plumbing list. `docs/KERNELS.md`: the
two kernels' rows. Checks on the final tree: `cargo fmt --all -- --check` clean; clippy `-D warnings` clean on `memra-tier`
(`--all-targets`), `memra-engine` and `memra-server` (`--lib --tests`); `memra-tier` contracts `81 passed` (whole crate
green: 7, 62, 81, 18, 6, 32, 53, 4); `memra-engine` lib `tier_transfer` `6 passed; 6 ignored`; `memra-server` lib `806
passed; 14 ignored` (803 before: the two day-22 CPU tests and the lead's #638 test); `git diff --check` clean on every
source commit; `tools/check-flags.sh` (no uncovered runtime names); `tools/check-conflict-markers.sh` OK; `python3
tools/check-public-boundary.py check` `0 new`. Every push in `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged); no qualification claimed. The box worktree `/root/wt-a` is at `bb1a2b212` on `lane-a-day22`;
`/root/spill-receipts/a-day22/` and `a-day22-run1/` mirrored to `pro-single-day22/box/` and `box-run1/` (bins excluded);
the bundles removed on both ends; local `/tmp` scratch removed. Not run today: the local RTX 5090 door gates (owed with
the lock, as on days 18 to 21). #536 comment posted with this receipt; the issue stays open.

## Budget

About 4.6 agent-hours against 4: reading and the merge 0.4, the pre-registration 0.3, the tier schedule and bindings 0.3,
the engine kernel and receipt plumbing 0.9, the worker's receipt lines and mismatch arms with the fault values 0.6, the
fault gate's cells 0.2, the lead's two notes and the integ37 merge 0.2, the box (two builds, two full gate sittings, the
cudarc finding and the fix) 1.1, records and the comment 0.6. Blockers: none (one 120 s lock wait on the second sitting;
the 5090 was not touched). Open: Move 2's owed items 1 to 4 in `OWNER-THREAD-OFFLOAD.md`, the door review's reading of
cell (v), the 5090 door gates on this tree, the Move 1 owed items.
