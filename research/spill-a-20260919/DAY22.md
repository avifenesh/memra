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
