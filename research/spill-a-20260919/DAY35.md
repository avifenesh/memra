# WP-A day 35: design F settled by measurement; the demote's two owner-thread KV hashes pre-registered

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, base `05fbec3b2` (the lane tip the lead integrates as integ54; its
files are not touched here). Rig: the local RTX 5090 Laptop GPU only; BOX4 is lane B's and this day needs no box time.
No cross-card comparison. Every cell `executed-not-qualified`. Every engine push in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. The F decision cell (pre-registered before any arm is built or run)

**The question.** Design F (DAY33: the promote's staging fill as one host function on the copy stream, the KV batch
and the spans submitted in the probe's own tick) against the day-32 design H (the hash helper fills the staging while
the request parks; the poll that lands the fill submits the KV batch and the spans), both carrying design K (DAY34:
the promote's KV completion checksums on the hash helper). BOX4 read F flat without K (DAY34 section 7: 1b `+12.8` /
`+12.7` on the day-32 binary against `+13.0` / `+13.0` on the day-33 binary, `promote_in` 28.40 against 28.40). This
cell reads the 5090 with K on both arms.

**The arms.**

- `FK`: the lane tip `05fbec3b2`, built as it is.
- `HK`: `05fbec3b2` plus a revert of F's server commit `045ab57d4` on a local scratch branch
  (`scratch/spill-a-d35-hk`). Conflicts resolve by one rule: everything after F stays (the day-33 timeline
  `d7d54c7a9`, design K `100214477`, main's merged changes); only what `045ab57d4` changed goes back to its day-32
  form (the Filling phase and the helper's `Fill` job return beside `Hash` and `Sources`; the submission moves back
  from the probe to the poll that lands the fill; K's views go to the helper right after that submission, as K's
  code already does after any off-tick KV batch submission). F's engine and tier additions stay in `HK` unreached (no
  caller). The resolved patch is banked as `rtx5090-day35/hk-revert.patch` with its base before the cell runs. `HK`
  is a measurement arm, not a delivery: its bar is a clean build, clippy clean, and the server lib tests (F's day-33
  censuses reverted with it).
- `OFF`: the `FK` binary with the door off (neither F nor H is reached), so the cell reads DAY28 1b's `on_minus_off`
  for each ON arm against the same OFF boots.

**The cell.** The day-26 double-park cell's shape on the 5090 with a third arm: `stall_cell.py --mode promote --n 5`
byte for byte per boot, every receipt replayed (`STALL REPLAY`). Order o1 = `HK FK OFF` five times, order o2 = `OFF FK
HK` five times (o1 reversed): 30 boots, N=5 boots per arm per order. ONE bounded hold of `/tmp/memra-5090.lock` for the
whole cell (60 x 120 s attempts, never inside another lane's hold; then no compute app and at least 20000 MiB free,
bounded 15 x 60 s). The boot environment is the day-34 5090 A/B's: `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4
MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192`, plus `MEMRA_KV_HOST_CONTRACTS=1` on the ON arms; the
9B NVFP4 MTP artifact (its sha256 banked); each server's readiness bounded to 60 s. Both binaries are named
`memra-server` in their own directories. Raw logs, receipts and 250 ms card telemetry banked in `rtx5090-day35/`.

**The metrics, per order.**

- E2E: the promoting request's end to end, `wall_ms` of every promote-arm intruder run (the quantity DAY28 1b reads),
  median per arm over its five boots. Reported with 1b's `on_minus_off` for each ON arm against the order's OFF boots
  and DAY28's `<=+20.0` rule as a reading.
- PIN: B3's promote in-ms, the `[prefix-host] promote: .. in Y ms` line, steady (the second and later promote of each
  boot), median per arm.
- The pair noise of a metric in an order: the larger of the two ON arms' ranges (max minus min) of their five per-boot
  medians. The day-34 5090 A/B, the only prior reading of this cell's spread, gives per-boot e2e ranges of 2.70 to
  3.38 ms and PIN ranges of 0.3 to 0.9 ms.

**The decision, stated before running.** F is KEPT if and only if, in BOTH orders, `median(HK) - median(FK)` exceeds
the order's pair noise for E2E AND for PIN. Otherwise F is DELETED from the lane. The cell is complete only if all 30
boots come ready, every replay passes and every receipt reads `errors=0`; an incomplete cell is repeated whole once in
a new hold, and nothing is read from a partial one.

- KEEP: F stays as the lane's promote path; DAY35 records the 5090 margin beside BOX4's flat reading (a per-card win is
  enough; F is not worse on the target card).
- DELETE (door hygiene, no dead mechanism): `HK`'s server form becomes the lane's, and F leaves whole in the same
  change: `CudaTransfers::submit_h2d_spans_filled` and its host-function trampoline, the fill target and
  `enqueue_to_device_f32_after_fill` in `pinned_host.rs`, tier rule 6 (`H2dFillFixture`, its binding and red arm), the
  cells `h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies` and
  `h2d_fill_host_function_does_not_hold_the_owner_thread`, the fault gate's `filled_submission_after_refusal`
  assertion (back to its day-32 form), and the TESTING and FLAGS rows. Then the 5090 gate set on the resulting binary
  (identity x4, failure ON, fault default and plain, hit OFF and ON, the unit cells) before the push. The refutation
  goes to DAY35, the INDEX and OWNER-THREAD-OFFLOAD.

**Predictions.** FK as on day 34: the copy lands inside the probe's tick, K's hash over the write-combined leases takes
about 8.8 ms and lands at the second tick top, PIN about 16.5 ms. HK: the fill lands at the first tick top (about 8.4
ms), the KV batch and the spans go then, K's hash ends about 8.8 ms later, just past the second tick top, and the promote
publishes at the third, PIN about 25 ms. If that holds, F wins by about one tick (about 8 ms) on both metrics, well
past the day-34 spread, and F is kept. If HK's hash makes the second tick top, the arms read within a millisecond and F
is deleted.

## 2. The demote's two owner-thread KV hashes (design M; pre-registered, no code until F is settled)

**The census (file:line at `05fbec3b2`).**

| hash | site | when | on the 5090 (9B, write-combined leases), the day-34 A/B's 220 demotes |
|---|---|---|---|
| hash 1, the D2H item's completion checksum (the receipt) | `tier_transfer.rs:2369`, `CudaTransfers::progress` | the landing poll, over each landed KV lease | the ledger's `copy settle` median 8.53 ms, max 14.27 |
| the `flip-demote` fault (diagnostic, the one post-copy writer) | `worker.rs:13752`, `apply_flip_demote_fault` | after the settle, before the hand-off | none unless armed |
| hash 2, the bind's re-hash, compared with hash 1 | `worker.rs:9516`, `bind_tier_image` (`None => checksum(bytes)`) | the take-back, bind and publish | the ledger's `take-back bind and publish` median 8.60 ms, max 12.66 |

Both hashes are the same program (`checksum`, SHA-256 in the `valid-bytes` frame) over the same leases; hash 2 exists
to catch any change between the landing and the bind, which is the window `flip-demote` writes in by design. On BOX4
(cached leases) the same two segments read 1.54 and 1.57 ms. The owner's other demote segments (pre-submit, the
hashing polls) are not this item.

**Design M, two steps, one ordering rule.**

1. **M1, hash 1 on the helper (the D2H mirror of K).** Tier rule first, `conformance/d2h_deferred_checksum.rs`
   (additive, unversioned, `WIRE_VERSION` stays 1): a D2H batch whose completion checksums are DEFERRED lands an item
   only with its supplied digest (`producer_done` false until then); a view of a deferred item's destination is handed
   out only after that item's copy is observed complete (red arm: a binding that hands a view of a destination whose
   copy has not landed must fail the schedule); the destination stays engine-owned while a view is out (take-back,
   retire and recover are `Busy`); every view comes back exactly once; the supplied digest becomes the item's completion
   checksum, so it IS the D2H receipt. Engine: `defer_d2h_checksums(&ticket)` at submission (marks the batch), and
   `d2h_landed_views(&ticket)` (`NotReady` until every deferred item's copy landed, then one view per item) with the
   day-34 `supply` shape. Server: the demote's landing poll hands the views to the helper as one `Sources` job (the
   day-34 job kind), the next poll takes the reply and supplies it before the settle's completion step; fail closed as
   K (helper gone, a reply that does not describe the ticket, or the deadline latch the tier).
2. **M2, hash 2 on the helper (the day-28 pattern).** After the settle (its take-back included) and the flip, the KV leases MOVE
   to the helper inside the existing `Hashing` job beside the heap payloads (owned, no raw view); the helper hashes
   them with the same program and hands them back with their digests; the bind consumes each digest as a precomputed
   one (its byte count checked against the lease's) and compares it with the receipt exactly as today, including the
   `flip-demote` naming line.
3. **The ordering rule** (a census in the server, a schedule in the tier crate): hash 1 is taken over the landed bytes
   before any post-copy writer; the only post-copy writer (`flip-demote`) runs on the owner after hash 1 is supplied
   and before the hand-off; hash 2 is taken after the hand-off, so after every writer, and before the bind; nothing
   writes a lease between hash 2 and the bind (the helper owns it). The synchronous routes (`Block`, the handoff import,
   the on-tick route) keep both hashes on the owner as today.

No new `MEMRA_*` name, no new numeric program (bytes are hashed; no token is generated differently).

**Acceptance, stated before any M code.**

- (a) Verification semantics unchanged. The tier rule and its red arm; a native cell that supplies a wrong D2H digest
  and reads `Corrupt` with nothing published; a GPU server cell in which a test-only writer changes one lease byte after
  hash 1 is supplied and is refused at the bind (`checksum differs from its D2H contract receipt`, nothing published);
  the failure gate's `digest` cell on the door ON arm with `plane image checksum differs from its D2H receipt as
  injected` present and `VERIFY FAILED` caught; the fault gate's demote cells.
- (b) On the 5090: identity x4, failure ON, fault default and plain (every cell), hit OFF and ON with the day-24 census,
  the unit cells: ALL GREEN.
- (c) The owner's two hash segments on the 5090 over at least 20 steady demotes on M's boots: `copy settle` median at
  most 1.5 ms and max at most 3.0 ms, and `take-back bind and publish` median at most 1.5 ms and max at most 3.0 ms
  (the base: 8.53 and 8.60).
- (d) The cost: per order, the demote's `wall .. t0 to publication` median on M at most the base's plus 25 ms (three of
  the 5090's ticks: hash 1's round trip plus hash 2 inside the helper's job), and the demoting intruder's e2e median at
  most the base's plus 1.0 ms.

**The 5090 A/B (its cell).** The base binary (the lane after F is settled) against M, `stall_cell.py --mode demote --n
5` byte for byte, o1 = `base M` five times, o2 = `M base` five times, door ON, the section-1 boot environment, one
bounded hold. (c) is read on M's boots, (d) per order. A reading, not a clause: the tenant's `stall_median` in the
demote mode, where the owner's demote hold shows up.

**Predictions.** (c) both segments fall to about 0.5 ms (the non-hash owner work around each). (d) the demote wall
grows by about 10 to 20 ms: hash 1's helper pass (about 8.3 ms on write-combined leases) costs about one tick, and hash
2's 8.2 ms lengthens the helper's job; the tenant's demote-mode stall falls by roughly the 16 ms the owner stops holding.

## 3. Budget

0.5 agent-day. The 5090 is taken first by the lead's integ54 battery (about 30 minutes), then possibly by lane B; the
cell's hold waits behind both with bounded attempts. The arms are built under `systemd-run --user --scope -q -p
CPUQuota=1200% -p MemoryMax=20G`, niced, while another lane holds the card.

## 4. The F decision cell, as it ran (`rtx5090-day35/`)

- The arms as built: `HK` = `scratch/spill-a-d35-hk` `44abb96e0` (`a0f915d8a` plus `hk-revert.patch`: the fifteen
  conflict hunks resolved by section 1's rule; the Filling construction and the fill's submission gained the log-only
  timeline terms `fill handed` and `submitted`), binary `4465a68212350c3c..`; clippy `-D warnings` clean, server lib
  `887 passed`. `FK` = `a0f915d8a`, binary `07a0360ffbfaf240..`. Both `cargo build --release -p memra-server`, niced
  under the CPU quota while lane B held the card. The HK scratch commit's first try went in with hooks off; it was
  undone and recommitted with the pre-commit hook running before anything was built from it. The branch was local and
  is deleted after this record; the patch is the arm's source.
- The hold: taken on the 58th bounded attempt, after 57 busy ones behind lane B's day-36 O1 hold (`run.log`), taken 00:53:08Z, released 01:08:23Z; no
  compute app at the start or the end (`compute-apps.after.csv` empty). 30 boots, every one ready (`rc=0` 30 of 30),
  `STALL REPLAY: PASS` 30 of 30. The 9B artifact `52c9cceb190055e0..` (`run.log`). Card telemetry (`card-250ms.csv`,
  local-time stamps): 3659 samples, 71 to 88 C, 16.3 to 176.2 W.

**The decision, verbatim** (`reading-day35.log`):

- `DAY35 F CLAUSE order=o1 metric=e2e hk=86.75 fk=78.95 hk-minus-fk=+7.81 pair-noise=5.57 (hk range 5.57, fk range
  3.32) rule hk-minus-fk>pair-noise -> CLEARS`
- `DAY35 F CLAUSE order=o1 metric=pin hk=23.80 fk=16.20 hk-minus-fk=+7.60 pair-noise=0.90 (hk range 0.90, fk range
  0.50) rule hk-minus-fk>pair-noise -> CLEARS`
- `DAY35 F CLAUSE order=o2 metric=e2e hk=88.26 fk=80.82 hk-minus-fk=+7.44 pair-noise=5.03 (hk range 5.03, fk range
  3.84) rule hk-minus-fk>pair-noise -> CLEARS`
- `DAY35 F CLAUSE order=o2 metric=pin hk=24.10 fk=16.50 hk-minus-fk=+7.60 pair-noise=0.90 (hk range 0.80, fk range
  0.90) rule hk-minus-fk>pair-noise -> CLEARS`
- `DAY35 F DECISION -> KEEP`

**Readings** (not clauses): DAY28 1b against the same OFF boots, `order=o1 arm=hk on=86.8 off=64.7 on_minus_off=+22.1
rule <=+20.0 -> FAIL`, `order=o1 arm=fk on=78.9 off=64.7 on_minus_off=+14.3 .. -> PASS`, `order=o2 arm=hk .. +22.3 ..
FAIL`, `order=o2 arm=fk .. +14.9 .. PASS`. Polls: HK `polls [2, 3] (counts [1, 49])` and `[7, 43]`, FK `polls [2]
(counts [50])` in both orders. The helper's checksum time: HK 8.20 / 8.00 ms, FK 8.80 / 8.80 ms.

**What it says.** Section 1's prediction held: HK's fill lands at the first tick top, K's hash over the write-combined
leases then ends past the second, and the promote publishes at the third (`promote in` about 24 ms); FK's copy lands in
the probe's tick and the promote publishes at the second (about 16 ms). F is worth one tick per promote on the 5090 and
brings 1b under its bound there (HK reads +22.1 / +22.3, the day-32 form of BOX3's failure). BOX4 read F flat (DAY34
section 7). **F is kept**: it wins on the 5090 and is not worse on the target card. The "owed: a decision A/B" mark on
integ54 is closed by this cell.

## 5. Design M as built (recorded before any M cell runs)

In the pre-registered order: the tier rule (`ec40a3925`), the engine (`d0d2cdd0d`, plus `CudaPinnedLease::read_view`
in the server commit), the server (this section's commit).

- **M1 (hash 1), as registered.** The off-tick route's last submission step defers the ticket's D2H receipts
  (`host_d2h_checksums_defer`). The settle's new step 6a hands the landed destinations' views to the helper as a
  `Receipts` job once every KV copy landed (`d2h_landed_views`, `NotReady` before), takes the reply (`Poll`: hand the
  ticket back while it is out; `Block`: a bounded wait in the helper's deadline), supplies it, and re-reads the
  completion before the take-back and `require`. A helper gone, a reply that does not describe the ticket, or the
  deadline quarantine the source: the engine keeps the destinations (a leak, never a free under a live read). One
  line per landed demote: `demote receipts on the hash helper: ticket seq=S, N KV receipts (X MB in Y ms)`.
- **M2 (hash 2), one named departure.** The pre-registration said the KV leases MOVE to the helper (owned, no raw
  view). They cannot: `CudaPinnedLease` holds an `Rc` and a CUDA event and is not `Send`. As built, the day-34 pattern:
  after the settle and the `flip-demote` point the KV planes leave the image into a guard (`HostLeasesOnHelper`) that
  stays on the owner thread, and read views of their contract leases (`CudaPinnedLease::read_view`, an `unsafe fn`
  whose contract is the guard's) ride the existing `Hashing` job. The guard lands (the planes go back to the image)
  only when the helper's reply arrives; every other drop (a latch, the deadline, a helper gone, shutdown) leaks the
  planes. The bind consumes the helper's digest per lease as a precomputed one, its byte count checked, and keeps its
  own `checksum(bytes)` as the fallback; the receipt comparison and the `flip-demote` naming line are unchanged. One
  line per hand-off: `demote KV leases on the hash helper: ticket seq=S, N lease views (X MB) for the bind's
  re-hash`.
- **The fault gate's `hash-helper-gone` arm keys on the first HASH job.** It exited the helper on its first job of
  any kind; with M1 a demote's first job is its receipts job, which would move the arm from the `Hashing` phase (its
  documented red arm, the gate's `hash-helper-gone` cell) to the receipt step. The exit now sits in the hash path, so
  the cell reads what it read. The receipt step's own helper-gone and deadline arms have no fault-gate cell: named, owed.
- No new `MEMRA_*` name, no new numeric program. Clauses (a) to (d), the A/B cell and the predictions stand as
  section 2 wrote them.
- Checks: tier contracts `102 passed`; engine lib `548 passed`; server lib `886 passed`; clippy `-D warnings` on tier,
  engine and server, all targets, clean; the GPU-less `DOCS_RS=1 --target x86_64-unknown-linux-gnu` clippy pass
  `docsrs_rc=0`; `check-flags` clean; `cargo fmt --all -- --check` and `git diff --check` clean.

## 6. Design M on the 5090, as it ran (`rtx5090-day35/m/`): refuted as registered

- The hold: `hold taken (fd 9)` 03:47:19Z, released 04:03:25Z, after bounded waits behind lane B's day-36 O2 hold
  (`run.log`); no compute app at the start or the end. Binaries (`binaries.sha256`): base `07a0360ffbfaf240..` (the FK
  binary, the lane after F was settled), M `0ee7ba55af5ed567..` (`2562144dc`), the test binaries. 20 boots, every one
  ready, `STALL REPLAY: PASS` 20 of 20. Card telemetry (`card-250ms.csv`, local-time stamps): 3859 samples, 61 to 89
  C, 24.6 to 177.0 W.

**The clauses, verbatim** (`reading-day35m.log`, `run.log`):

- (c) `DAY35 M C copy-settle N=80 median=0.15 min=0.12 max=0.20 take-back N=80 median=0.07 min=0.06 max=0.08 rule
  N>=20 median<=1.5 max<=3.0 each -> PASS` (base: `copy-settle N=80 median=8.41 .. take-back N=80 median=8.43`).
- (d) `DAY35 M D order=o1 wall base=71.05 m=95.30 m-minus-base=+24.25 rule <=+25.0 | e2e base=124.29 m=107.93
  m-minus-base=-16.36 rule <=+1.0 -> PASS`; `DAY35 M D order=o2 wall base=69.55 m=96.05 m-minus-base=+26.50 rule
  <=+25.0 | e2e base=121.61 m=106.76 m-minus-base=-14.86 rule <=+1.0 -> FAIL`.
- (b) `gate identity-default-on rc=1 KV-HOST-SPILL IDENTITY GATE: 4 FAILURE(S) (teeth=0)`, `gate failure-on rc=1
  KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)`, `gate fault-default rc=1 KV-HOST-CONTRACT-FAULT GATE: 13 FAILURE(S)`
  (three failures each in `promote-postpublish`, `promote-reject`, `promote-readyview` and `promote-span-refusal`, one
  in `hash-helper-gone`); ALL GREEN: identity default OFF, plain OFF and ON, `fault-plain`, hit OFF and ON. Unit:
  engine `ok. 6 passed` (with `d2h_deferred_checksum_lands_with_the_supplied_digests`); door cells `FAILED. 17 passed; 1
  failed`, the new cell (below). **FAIL.**
- (a) The failure gate's `digest` cell printed `contracts door D2H receipt: Key plane image checksum differs from its
  D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote)`: the helper's re-hash saw the flipped byte, and r3 served
  the cold path with reference bytes. `FAIL: the promote caught it: VERIFY FAILED, loud and named`: no promote ran, so
  the verify arm had nothing to catch. The new GPU cell failed on its own fixture: `a Demoting entry has no source
  shell for its submitted ticket` (it called `host_demote_prefix_ref`, which leaves the shell to the sink; the fix, the
  sink `host_demote_prefix_entry` itself, is kept as a patch and was not run). **Not shown.**
- **M as registered fails (b) and (d); it is refuted.** Nothing in section 2 is relaxed.

**The cause, placed** (before any code change). M1 takes the receipts' reply at the NEXT tick-top poll after the
views go out, so every off-tick demote's copy phase gains one poll. Behind a long tick, that poll is late. In the
identity gate's default ON boot: `demote copy complete off the tick: ticket seq=3 complete after 2 poll(s), 300.8ms from
submission to completion` (day 34's same boot: `complete after 1 poll(s), 36.6ms`), with r2's admission and prime
(`[spec-k] .. prompt=102`, `[spec-acc] ctx=102`) logged between the submission and the D2H receipt. E_A reached its
`Hashing` phase late; it published only when `demote hashing settled synchronously by a second demote`, and the boot
has no promote line. A hit on a Demoting entry in its COPY phase does not park (day 29 parks the `Hashing` phase only),
so a request arriving in that longer window serves cold. Every failing check needs a promote after a demote (identity
r3, the digest cell's verify, the fault cells' "next promote"); the `hash-helper-gone` failure is r3's own demote
meeting the late entry through the `Block` wait. The spec-off run of the same fault gate (`fault-plain`, shorter ticks)
is ALL GREEN. The bytes held: `r3 ON == OFF byte identity (promoted restore == cold re-prime)` ok. Scope: the placement
is the mechanism plus this boot's log; no per-cell timeline was taken.

**What M bought** (readings): the owner's hold per steady demote `owner-held` median 18.28 to 1.34 ms; the demoting
intruder's e2e 15 to 16 ms lower in both orders; the tenant's `tenant-stall` median 44.61 / 42.64 to 39.38 / 39.48 ms.
The receipts took `receipts-ms N=90 median=8.10` on the helper.

**Next.** M's code leaves the lane tip in one revert commit (tier rule, engine calls, server), so no half-working
mechanism is integrated; git history and these receipts are the record. The half that adds no poll to the copy phase
(M2 alone: hash 2 inside the `Hashing` job, whose hits already park) is a separate design, pre-registered before any
code if it goes ahead.
