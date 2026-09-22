# Decision packet: `MEMRA_KV_HOST_CONTRACTS`, decide-by 2026-10-05 (draft for the owner)

Status: a DRAFT written by lane C on day 30 (2026-09-22) as the owner's input for the decide-by. It recommends
nothing. It becomes a `docs/decisions/` record only after the owner decides; until then it lives here. Every
number below was read from the receipt file named beside it (not from a summary), every verdict is quoted
verbatim, every cell is `executed-not-qualified` development evidence, and no number is compared across cards.
Cards: the target card is one RTX PRO 6000 Blackwell at its 600 W limit under the canonical collector
(`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`) with the Qwen3.8-27B NVFP4-Q5K MTP
artifact; the RTX 5090 rows are the local RTX 5090 Laptop GPU with the Qwen3.5-9B NVFP4 MTP artifact under
`flock /tmp/memra-5090.lock`. Paths are relative to `research/`; `A` = `spill-a-20260919`, `C` =
`spill-c-20260919`, `lead` = `spill-lead-20260919`. Rows marked "A's branch" are on
`origin/lane/spill-a-20260919` and not yet on `main` at the time of writing. Day 31 update: A day 24 is on `main`
since #643 (integ40); A day 25 (`483425d83`) is on the lead's integ41 branch, merged into this lane as `5cef58f09`;
the two RTX 5090 rows this packet listed as "not run" on day 30 were run on day 31 (`C/rtx5090-day31/`, `C/DAY31.md`)
and are filled in below. Day 34 update: A day 26 is on `main` since #647 (integ42); A day 27 (`8bfd3d103`) and ruling 39
are on the lead's integ43 ref (`d06a9dd8c`), merged into this lane as `b64647abb`; rows marked "integ43" are on that ref
and not yet on `main` at the time of writing. On day 34 section 2's account of the demote's tick cost was replaced by A's
code attribution, section 4 gained the rows of A days 26 and 27 and C day 33, and items 2, 3 and 7 and section 6 were
re-read against them (`C/DAY34.md`); every number added was re-read from its receipt file (appendix A). Day 35 update: the RTX 5090 class has its tenant-stall cell (`C/rtx5090-day35/`, `C/DAY35.md`; A day 16's five arms on
the 9B, six boots in one hold, two passes in opposite order); section 4's RTX 5090 table, item 7 and section 6 carry it, every
number read from `C/rtx5090-day35/reading.log` and the receipts it names; the tree `091a931c0` (main after #648 plus the integ43
ref at `2cb0f6c00`, PR #649). Day 36 update: option (a) and option 2a landed (A days 28 and 29, rulings 39 and 40), and
ruling 41 makes that code the door's serving path; they are on the lead's integ45 ref (`96d96e1c2`, battery tree
`1c540e050`), merged into this lane as `aeefd9233`, not yet on `main` at the time of writing. Item 2 carries their receipt,
section 2's account of what the demote's tick pays is re-read from A's day-28 finding 2, sections 3 and 4 gain the day-28 and
day-29 rows and integ45's RTX 5090 gates, and item 7 and section 6 were re-read against them (`C/DAY36.md`). Every number
added on day 36 was produced by a command over the receipt files, and the command is in appendix A beside it. Day 37
update: the RTX 5090 class has a demote-class tenant-stall cell on the option (a) tree (`C/rtx5090-day37/`,
`C/DAY37.md`; day 35's cell unchanged, run with two binaries, `091a931c0` and `8b889dcdf`, in one hold, interleaved in
both orders); section 4's RTX 5090 table gains its rows, and item 2's scope line, item 7 and section 6 were re-read
against it. Every figure added on day 37 comes from a command in appendix A.

## The one page

### 1. The question (section E of `C/HOSTPREFIX-DOOR.md`, stated there and not answered)

Whether the contract program becomes the naked default of the host tier on the target card class, stays a door
with a new date, or is deleted. Promotion would make every pageable D2H and every contract-routed H2D carry a
ticket, a per-item completion event, a completion checksum, a typed receipt with epochs and a `require`
verdict, and the fail-closed settle arms, with the OFF program and the `MEMRA_KV_HOST_CONTRACTS` read deleted;
Move 1's off-tick copies would be the only demote and promote path, and Move 2's off-tick capture and restore
the only D2D path. Deletion would remove the door, `host_tier_context` and its helpers, the FLAGS.md row, the
fault cells and the contract receipts, and keep the synchronous OFF program (one host-blocking synchronize per
plane at demote, an owner-stream `htod_u8_into` at promote, owner-stream D2D copies at capture and restore, no
receipt, no byte attestation, no typed unwind). "Whether that trade is a default, a longer door with a new
date, or a deletion is the owner's call at the review; nothing in this lane's records answers it."

### 2. What the door is today (tree `main` `ebe3fe17d`, integ39; A day 24 on A's branch)

Under `MEMRA_KV_HOST_CONTRACTS=1` with a host tier armed (`MEMRA_KV_HOST_MB > 0`):

- **Move 1, whole.** The D2H demote of every KV plane (trunk and MTP draft) is one `TransferEngine` batch on a
  second stream per device (`CudaTransfers::new_with_copy_stream`), submitted behind a producer fence with no
  owner wait; the entry is the worker's one `Demoting` entry, polled at the tick top and published after
  `require` against its receipt (A day 17). The H2D promote rides the same copy stream; the request PARKS on the
  requeue as the worker's one `Promoting` entry and re-admits to a device hit (A day 18); the owner-stream wait
  on the copy's event is installed at the SETTLE, not at submit (rule 3, A day 19), and the parked-only wait
  replaced the owner-thread spin (#627).
- **Move 2, slices 1 to 3.** The seed and LCP-split publishes' KV plane copies ride the copy stream as one
  capture batch (borrowed live source, owned registered destinations, event-ordered publication at a tick top;
  A day 20). A whole-entry device hit's row copies ride it as one restore batch (borrowed source under the
  device LRU's pin, borrowed destination, the request parked, rule 3's reader wait at the settle; A day 21).
  Both D2D classes carry a receipt: a copy-stream reduction over the source span behind the fence and over the
  destination after the copy, compared by `Completion::require`, with the `d2d-delay-*` red arms (A day 22).
- **The draft-bearing routes.** The restore of an MTP draft-bearing entry submits the draft K and V rows as two
  more items of the same batch under the same pin and receipt (`items=34` on the 27B; A day 23, on `main`). The
  spec-boundary CAPTURE of a draft-bearing entry routes the same way (A day 24, `d5708266f`, A's branch only).
- **The receipt term.** D2H: the engine's SHA-256 completion checksum per plane, checked against
  `bind_tier_image`'s bundle checksum, one `contracts door D2H receipt` line per demote; H2D: `require` against
  that receipt before `ready_view`, one `H2D receipt` line whose digest equals the D2H line's. D2D: four
  wrapping u64 lanes over LE words per item (`d2d_receipt_digest`, CPU oracle `receipt_digest`), source and
  destination digests named in the `D2D capture receipt` / `D2D restore receipt` lines.
- **Fail-closed arms, every one typed.** Boot refusals: any value other than `1`/`0`; `MEMRA_GLM5_TP_KV_HOST=1`
  (the arena, lease handoff pending); a checkpoint-directory model; a loaded vision tower. Refused by name:
  TP or latent (GLM) planes, a DFlash draft tail, a draft-bearing entry on a model with no MTP head, a handoff
  import, an unknown model. Route refusals hand the request to the tick program with their line (`demote
  refused (contracts door)`, `promote refused (contracts door)`, `capture refused (contracts door)`, `restore
  refused (contracts door)`). Latching arms (one `TIER DISABLED` line, every later copy of the boot on the tick
  program): an unobservable completion, a leaked ticket, fence or destination, a pending entry missing its shell
  or ticket, `CAPTURE OFF-TICK DISABLED` / `RESTORE OFF-TICK DISABLED` on a lost observation or a `ReceiptMismatch`
  (capture publishes nothing and drops its fresh planes; restore drops its cache and releases its pin; a
  `Latched` restore forgets the cache and keeps the pin, never a free under a running copy). A refused
  submission drains the owner stream and releases its fence or latches. A session leaving `active` settles a
  pending capture `Block` first (#634); the Block wait covers the receipt event (ruling 34).
- **The hash helper (A days 28 and 29, rulings 39 to 41; integ45).** After a demote's KV copy lands, the heap
  payloads of the image (the recurrent f32 planes, the boundary logits, the hidden row: `98 payloads (157.9MB)` per
  27B entry, `50 payloads (53.7MB)` per 9B entry) move to one long-lived helper thread per `HostTierContext`
  (`memra-host-hash`), and the entry sits in a `Hashing` phase of `PendingDemote` until the digests come back;
  `bind_tier_image` consumes the handed-in digests once the reply's ticket, payload count, byte counts and slots are
  checked (the helper runs the same `checksum` over the same bytes; the receipt term unchanged; `A/DAY28.md` section 2). A hit on a `Hashing` entry parks one tick (`hit parked on a Hashing entry`); the
  publish line carries the owner-thread ledger (`pre-submit`, `copy settle`, `hashing polls`, `take-back bind and
  publish`, `owner in-completion`) and ends `H hit(s) parked on the Hashing entry (R re-park(s))`. The
  `hash-helper-gone` and `hash-never-lands` cells (a 10 s deadline) latch the tier with the parked requests named, on
  the existing `MEMRA_KV_HOST_FAULT` row; no new `MEMRA_*` read.
- **What still runs on the tick under ON.** The recurrent f32 state (`conv_state`, `ssm_state`, about 157 MB of
  every 27B entry) is copied on the owner stream by both D2D classes (Move 2 owed item 1). On A's branch (A day 30,
  `A/DAY30.md`; the lead integrates) the demote's recurrent planes ride the KV ticket as D2H spans into a cached
  pinned staging set, so the demote's steady pre-submit reads 0.62 ms on the target card (day 29: 6.05, N=80 each),
  while the promote's H2D and both D2D recurrent copies stay on the owner stream. The by-reference
  demote routes (the admission reclaim flush, the pause sweep, the handoff) keep the day-16 synchronous program
  (Move 1 owed item 3). The H2D's settle-time wait is an owner-stream wait on the copy's event (rule 3), so the
  tick program orders behind the landing at the settle. What the demote's tick pays under ON since the helper, read
  from A's day-28 finding 2 (`A/DAY28.md`, "the demote's remaining owner-thread cost is the pre-submit segment") and
  counted from the ledger lines of the double-park cell's 10 ON boots (appendix A, command 1): the PRE-SUBMIT segment,
  `t0` to the KV submission's return plus the recurrent copies, in A's code order `reserve_image`, the KV submission
  (32 `alloc_host`, 32 `register_device`, the fence, `submit_batch`), then the 98 recurrent planes plus the logits and
  the hidden row (100 copies) copied one `clone_dtoh` and `synchronize` at a time into pageable `Vec`s
  (`host_entry_from_device`). It reads `median=6.07 min=5.85 max=6.77` (N=80) on the 4th to 11th demote of each boot and
  `median=42.19 min=37.39 max=43.91` (N=30) on the first three (day 28; day 29 `6.05` (5.92 to 6.61) and `42.27` (37.38
  to 44.13)): the first-touch step of the 32 pinned lease allocations. A's own figures ("about 6 ms at steady state",
  "40 to 45 ms" and "39 to 45 ms" "on the first two demotes of every boot", `A/DAY28.md` lines 263 to 265 and 362) read
  the 100 ledger lines carried in the promote runs' `server_log_lines`; each ON boot's first ledger line (ticket
  `seq=2`, pre-submit 37.39 to 39.28 on day 28 and 37.38 to 40.63 on day 29) is in no promote run's lines on 10 of 10
  boots both days (appendix A, command 1), so on the server logs the first-touch step covers three demotes per boot. Under
  `MEMRA_KV_HOST_VERIFY=1` (the identity and failure gates' diagnostic arm) the pre-submit segment reads `148.52` and
  `151.21` on the target card and `51.56` and `53.55` on the RTX 5090 (the identity default-ON boot's two ledger lines),
  because `t0` precedes the verify digest. The rest of the owner thread's share is small: `copy settle` `median=1.21`
  (1.15 to 1.27), `hashing polls` `median=0.01` (0.00 to 0.02), `take-back bind and publish` `median=1.29` (1.03 to
  1.53), N=110 each on day 28. The bundle hash itself runs on the helper, OFF the tick: `hashed in` `median=73.20`
  (73.00 to 73.50) ms per 157.9 MB (N=110, day 28; day 29 `73.10`, 72.90 to 73.50), `landed after` a median of 6
  tick-top polls. Move 1's two receipt hashes over the KV planes (`progress`'s completion checksum at the poll,
  `tier_transfer.rs:1710-1712`, and `bind_tier_image`'s per-KV-plane check, `9440-9451`; A day 27, file:line on
  `cb9fa5ef3`) stay on the owner thread, about 1.9 MB on the 27B's 64-token entry, about 0.9 ms each on the target host
  (about 1.8 ms together, A's code-read arithmetic, `A/DAY27.md`; inside `copy settle` and `take-back bind and publish`,
  not split by any line). Before the helper the tick paid `bind_tier_image`'s whole-image pass inside the tick
  (`demote_in-completion median=74.8` on the target card, A days 25 to 27, N_runs=100 each; 21 to 23 ms on the RTX
  5090; A day 27 section 1, integ43); C day 33's RTX 5090 measurement refuted the reading that the demote's hashes
  stream the write-combined destination (section 4, `DAY33 HASH-WC VERDICT`). On the hit gate's shape every
  spec-boundary capture settles synchronously at the session retire, not at a tick-top poll (A day 24 finding 2);
  the settle's own span on the owner thread is a fixed 0.4 ms (A day 25, `held_ms N=11 min=0.37 median=0.41
  max=0.44`, the copy already landed), stated here and not carried as a cost of the door.
  The publishes still on the tick by name: the fanout leader's and the pause sweep's snapshots, the
  `dspark-boundary` and `glm5-boundary` publishes, and every `OnTick` refusal.

### 3. Correctness evidence (both arms, both cards; verdict lines verbatim; N=1 per cell)

| Gate | Card | Tree | Door OFF | Door ON | Receipt path |
|---|---|---|---|---|---|
| Identity, default (spec) and plain | PRO 6000 | `913095404` (day 25) | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 `ok:` each) | same, 12 `ok:` each | `C/pro-single-day25/cells/identity-{default,plain}-{off,on}/gate.log` |
| Identity, default and plain, on main's tree after #627 | PRO 6000 | `91b0d4e08` (day 23) | `ALL GREEN (teeth=0)` x2 (12 `ok:`) | `ALL GREEN (teeth=0)` x2 (12 `ok:`) | `C/pro-single-day23-gates/cells/identity-*/gate.log` |
| Identity, default and plain | RTX 5090 | `91b0d4e08` (day 23) | `ALL GREEN (teeth=0)` x2 (12 `ok:`) | `ALL GREEN (teeth=0)` x2 (12 `ok:`) | `C/rtx5090-day23/identity-*/gate.log` |
| Failure, share-cap arm, default and plain | PRO 6000 | `91b0d4e08` (day 23) | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 `ok:`) | same, 15 `ok:` | `C/pro-single-day23-gates/cells/failure-{default,plain}-{off,on}/gate.log` |
| Failure, whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`) | PRO 6000 | `91b0d4e08` (day 23); `98170f182` (day 22) | `ALL GREEN` (14 `ok:`), server line `skip demote: entry 159.9MB > host budget 1MB` | `ALL GREEN` (14 `ok:`), the same refusal after the whole Move 1 contract ran | `C/pro-single-day23-gates/cells/failure-default-pct100-{off,on}/gate.log`; `C/pro-single-day22/cells/failure-default-pct100-on/` |
| Failure, share-cap arm | RTX 5090 | `91b0d4e08` (day 23) | `ALL GREEN` x2 (15 `ok:`) | `ALL GREEN` x2 (15 `ok:`) | `C/rtx5090-day23/failure-*/gate.log` |
| Failure, whole-budget arm (`MEMRA_KV_HOST_TENANT_PCT=100`), default and plain | RTX 5090 | `934a6da3a` (day 31; binary built at `5cef58f09`, `crates/` equal) | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x2 (14 `ok:`, 0 `FAIL:`), server line `[prefix-host] skip demote: entry 54.8MB > host budget 1MB` (default) and `... entry 54.6MB > host budget 1MB` (plain), twice per boot | `ALL GREEN` x2 (14 `ok:`), the same two refusal lines, each preceded in the boot by `demote submitted off the tick: 64 tokens, 54.8MB, ticket seq=3, 18 items ...` (plain: `54.6MB ... 16 items`), `contracts door D2H receipt ... require=ok` and `demote published off the tick: ticket seq=3 complete after 1 poll(s), 36.2ms from submission to completion (tick-top poll)` (plain seq=2: `24.9ms`), the refusal after the whole Move 1 contract ran, as on the target card | `C/rtx5090-day31/gates/failure-{default,plain}-pct100-{off,on}/{gate.log,verdict.txt,poolfull-demote-lines.txt}` |
| Contract fault, six Move 1 cells plus the ticket accounting clause | PRO 6000 | `913095404` (day 25) | n/a (ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (67 `ok:` default and plain); accounting lines `receipt seq=1 expected 1 + 0 capture ticket(s) submitted before it = 1`, `seq=2 expected 2 + 0 ... = 2` (default), `seq=3 expected 1 + 2 ... = 3`, `seq=4 expected 2 + 2 ... = 4` (plain) | `C/pro-single-day25/cells/fault-{default,plain}/gate.log` |
| Contract fault | RTX 5090 | `913095404` (day 25) | n/a | `ALL GREEN` (67 `ok:` default and plain), the same four accounting lines | `C/rtx5090-day25/fault-{default,plain}/gate.log` |
| Contract fault with the two D2D cells (`d2d-capture`, `d2d-restore` refusing by receipt) | PRO 6000 | `2b850b2b0` (A day 23) | n/a | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` | `A/pro-single-day23/box/gates/` (the fault log) |
| Hit gate, host tier ARMED in the ON arm, plain hits through the restore route | PRO 6000 | `b1e9c75b6` (day 27); `7349ef932` (day 29, under the collector's hold) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok:`) | `ALL GREEN (qwen)` (68 `ok:`), `ok: door arm: 7 route submission(s) across the two boots` | `C/pro-single-day27/cells/hit-{off,on}/gate.log`; `C/pro-single-day29/hitgate/{off,on}.gate.log` |
| Hit gate, ARMED, plain hits | RTX 5090 | `b1e9c75b6` (day 27) | `ALL GREEN (qwen)` (61 `ok:`) | `ALL GREEN (qwen)` (68 `ok:`), `7 route submission(s)` | `C/rtx5090-day27/hit-{off,on}/gate.log` |
| Hit gate, ARMED, the DRAFT-BEARING restore through the route (12 of 16 hits) | PRO 6000 | `2b850b2b0` (A day 23) | `ALL GREEN (qwen)` (61 `ok:`) | `ALL GREEN (qwen)` (68 `ok:`), `ok: door arm: 19 route submission(s) across the two boots` | `A/pro-single-day23/box/gates/hitgate-{off,on}.log` |
| Hit gate, ARMED, draft-bearing restore | RTX 5090 | `6a1909924` (integ39 battery) | `ALL GREEN (qwen)` (61 `ok:`) | `ALL GREEN (qwen)` (68 `ok:`), `19 route submission(s)` | `lead/integration-day12/integ39-hit-gate-5090/hit-{off,on}/gate.log` |
| Hit gate, ARMED, the spec-boundary CAPTURE and the draft-bearing restore both through the route | PRO 6000 | `185c57b4f` (A day 24, A's branch) | `ALL GREEN (qwen)` (61 `ok:`) | `ALL GREEN (qwen)` (68 `ok:`), `ok: door arm: 30 route submission(s) across the two boots`, 11 `capture submitted off the tick (spec-boundary)` | `A/pro-single-day24/box/gates/hitgate-{off,on}.log` on A's branch |
| Identity and failure, option (a) without 2a (the hash helper; a hit inside its window misses) | PRO 6000 | `f9284a711` (A day 28; code `45f824a75`) | `identity-default-off`, `identity-plain-off` `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 `ok:` each); `failure-off` `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 `ok:`) | **`identity-default-on` `KV-HOST-SPILL IDENTITY GATE: 4 FAILURE(S) (teeth=0)`** (8 `ok:`; `FAIL: E_A promoted back on the r3 probe (named log line)` and three more); `identity-plain-on` `ALL GREEN (teeth=0)` (12 `ok:`); **`failure-on` `KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)`** (14 `ok:`; `== cell 2: DIGEST MISMATCH`, `FAIL: the promote caught it: VERIFY FAILED, loud and named`) | `A/pro-single-day28/box/gates/{identity,failure}-*.log` |
| Contract fault, ten cells (the two hash cells `hash-helper-gone`, `hash-never-lands` added), option (a) without 2a | PRO 6000; RTX 5090 | `f9284a711` (A day 28) | n/a | PRO 6000 **`KV-HOST-CONTRACT-FAULT GATE: 23 FAILURE(S)`** (98 `ok:`; per cell `ok:`/`FAIL:` `presubmit` 9/1, `postpublish` 9/1, `promote-presubmit` 6/5, `promote-postpublish` 6/5, `promote-reject` 8/6, `promote-readyview` 6/5, `d2d-capture` 12/0, `d2d-restore` 14/0, `hash-helper-gone` 14/0, `hash-never-lands` 14/0); RTX 5090 default the same 23 and the same per-cell counts, plain `ALL GREEN` (121 `ok:`) | `A/pro-single-day28/box/gates/contract-fault.log`; `A/rtx5090-day28/fault-{default,plain}/gate.log` |
| Hit gate and twin gate, option (a) without 2a | PRO 6000; RTX 5090 (hit only) | `f9284a711` (A day 28) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok:`; the RTX 5090's first OFF cell `server died during boot` on the gate's own `flock -w 300` timeout and no verdict, its re-run `hit-off-rerun` 61 `ok:`); twin `PREFIX-NEWEST-TURN-FITS: ... -> PASS` | `ALL GREEN (qwen)` (68 `ok:`), `ok: door arm: 30 route submission(s) across the two boots`, both cards; twin `-> PASS` | `A/pro-single-day28/box/gates/{hitgate,twin}-{off,on}.log`; `A/rtx5090-day28/hit-{off,off-rerun,on}/gate.log` |
| Identity, failure, contract fault (ten cells), hit and twin gates, option (a) plus 2a | PRO 6000 | binary at `29a1cc366` (A day 29; code `867655368`; gate scripts moved to `259c75f62` and `06e290374`, no `crates/` change) | identity default and plain `ALL GREEN (teeth=0)` (12 `ok:` each); failure `ALL GREEN` (15 `ok:`); hit `ALL GREEN (qwen)` (61 `ok:`); twin `-> PASS` | identity default and plain `ALL GREEN (teeth=0)` (12 `ok:` each); failure `ALL GREEN` (15 `ok:`); `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 `ok:`, 0 `FAIL:`, every one of the ten cells 0 `FAIL:`); hit `ALL GREEN (qwen)` (68 `ok:`, `30 route submission(s)`); twin `-> PASS` | `A/pro-single-day29/box/gates/*.log` |
| Identity default, contract fault default and plain, hit gate, option (a) plus 2a | RTX 5090 | `29a1cc366` (A day 29; the fault cells re-run on script trees `259c75f62` and `06e290374`) | hit `ALL GREEN (qwen)` (61 `ok:`) | `identity-default-on` `ALL GREEN (teeth=0)` (12 `ok:`); `fault-default-rerun2` and `fault-plain-rerun2` `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 `ok:` each); hit `ALL GREEN (qwen)` (68 `ok:`, `30 route submission(s)`). The earlier fault cells: `fault-default` and `fault-plain` rc=1 with no verdict (the gate script's own abort at `== cell presubmit:`), `fault-default-rerun` `2 FAILURE(S)` (`presubmit` 10/1, `postpublish` 10/1, the gate's hand-off wait; A day 29 section 2), `fault-plain-rerun` `ALL GREEN` (123 `ok:`) | `A/rtx5090-day29/*/gate.log` |
| Contract fault default and plain (ten cells), hit gate | RTX 5090 | `1c540e050` (integ45 battery) | hit `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok:`) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 `ok:` each arm, 0 `FAIL:`, the ten cells incl. `hash-helper-gone` and `hash-never-lands` 14/0); hit `ALL GREEN (qwen)` (68 `ok:`), census `capture_submitted=12 ... restore_landed=13` and `2 ... 3`, `ok: door arm: 30 route submission(s) across the two boots` | `lead/integration-day12/integ45-{fault,hit}-gate-5090/*/gate.log` |
| Newest-turn-fits twin gate | PRO 6000 | `2b850b2b0` (A day 23) | `PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` | same `-> PASS` | `A/pro-single-day23/box/gates/twin-{off,on}.log` |
| Newest-turn-fits twin gate (the 27B) | RTX 5090 | `b74269af5` (day 26) | `... -> PASS` | `... -> PASS` | `C/rtx5090-day26/twin27-{off,on}/gate.log` |
| The retire-seam settle's cost (#634's Block settle at a session retire) | PRO 6000 | `8c96ca6b5` (day 25) | (OFF and ON in one cell) | `DAY25 RETIRE VERDICT: admissible=True ... -> HOLDS (R1, R2, R3 within 3.0 ms in both arms and both passes; the seam exercised in both passes)` | `C/pro-single-day25/retire/` |
| Review-round replays (PR #599, #605 findings) | PRO 6000 | `70038ed01`, `1b354be59` | `DAY15 REVIEW REPLAY: PASS` | `DAY16 REVIEW REPLAY: PASS` | `C/pro-single-day15-review/`, `C/pro-single-day16-review/` |

Every hit-gate "door ON" receipt before day 27 (A days 17 to 21 on the target card, 5090 days 17 and 18; C 5090
days 24 and 26) ran with the host tier UNARMED and covered the tick program in both arms; they are receipts of
the tick program, not of the route (ruling 33), and are not listed above.

### 4. Cost evidence (ms; per card, never across cards; stall = the tenant's worst ITL minus its p50, the
harness's rule; `stall_median` over 10 arm runs, N=5 per arm per order, both orders, unless stated)

Target card, one RTX PRO 6000 Blackwell at 600 W, the 27B, 64-token entries of 159.9 MB unless stated:

| Quantity | OFF | ON | N, order, regime | Receipt |
|---|---|---|---|---|
| Demote stall, the ON-TICK door (before Move 1) | `stall_median=117.5` | `stall_median=193.5` | A day 16, `1646d421b`; per boot N=5 per arm per order; three holds 32 to 58 C, 32.6 to 492.7 W | `A/pro-single-day16/box/stall-{off,on}/` (rule lines in `ev/`) |
| Promote stall, the on-tick door | `stall_median=85.0` | `stall_median=162.8` | as above | as above |
| Demote stall after Move 1's demote half | (day 16's 117.5, cross-sitting) | `stall_median=149.6` | A day 17, `fc46e230d`; 40 to 52 C, 53.0 to 331.3 W | `A/pro-single-day17/box/stall-on/` |
| Demote and promote stall after Move 1 whole (with the owner-thread spin) | (cross-sitting) | demote `stall_median=149.7`; promote `stall_median=81.9` | A day 18 run 2, `3df0cb2b3`; 43 to 55 C, 69.4 to 336.6 W | `A/pro-single-day18/box/stall-on/`, `box/replays.log` |
| The SAME-WINDOW OFF against ON pair on main's tree after #627 (the parked-only wait) | demote `stall_median=117.5`; promote `stall_median=85.2` | demote `stall_median=149.5`; promote `stall_median=81.9` | C day 23, `91b0d4e08`, one hold; 32 to 51 C, 33.1 to 362.7 W; reading `P3=on_at_off P5=demote_half_unchanged` | `C/pro-single-day23/stall/ev/{off,on}/{demote,promote}/receipt.json`, `stall/reading.log` |
| Move 1 decision cell (i), SAME WINDOW, door ON on the copy stream (X, integ38 tree) against door ON on the owner stream (Y, the day-16 tree rebuilt) | Y demote `193.4` / `193.1`, promote `162.9` / `162.5` | X demote `150.0` / `150.0`, promote `149.5` / `149.7` | C day 29, twenty boots interleaved XYXY../YXYX.., N=5 boots per arm per order; 33 to 51 C, 32.5 to 360.8 W; `y_minus_x=+43.4 unc=0.6 -> isolated`, `+43.0 unc=0.1 -> isolated` (demote); `+13.3 unc=0.4 -> isolated`, `+12.8 unc=0.2 -> isolated` (promote); `DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False admissible=True)` (`idle_p99_sitting=14.9`) | `C/pro-single-day29/stall/reading.log`, `stall/ev/{o1,o2}/bNN-{x,y}/` |
| The isolating cell, RESTORE class alone (a whole-entry zero-suffix hit, 5152 tokens, 309.9 MB, no prime of its own) | `stall_median=8.9` (pass 1), `9.0` (pass 2), IQR 0.1 | `stall_median=9.1`, `9.1`, IQR 0.1 | C day 28, `61dc1a52a`, one hold, two passes in opposite order; 33 to 60 C, 33.1 to 501.0 W; `restore pass1 on-off +0.2 (unc 0.2) isolated; ... restore pass2 on-off +0.2 (unc 0.1) isolated` | `C/pro-single-day28/box/stall/ev/pass{1,2}/exact-{off,on}/run/receipt.json`, `C/pro-single-day28/reading.log` |
| The isolating cell, CAPTURE class (a 5088-token seed, 308.0 MB, nothing evicting at 8192 MB) | `stall_median=283.7` / `283.7` (IQR 0.2) | `stall_median=284.4` / `284.4` (IQR 0.1) | as above; `capture pass1 on-off +0.7 (unc 0.2) isolated; ... capture pass2 on-off +0.7 (unc 0.2) isolated`; the cache-OFF prime arm read `301.5` (IQR 1.5) so day 28's own-share subtraction read `share=-17.8` / `-17.1` (wrong sign, unread) | as above, `pass{1,2}/{prime,capture-off,capture-on}/` |
| The capture arm's OWN share, re-read against a cache-ON boot whose 308 MB seed insert is refused by the typed oversize line before any copy (`MEMRA_PREFIX_CACHE_MB=128`; the prime still stops at the 5088 seed boundary) | base `refused-off` `stall_median=283.1` (pass 1), `283.2` (pass 2), IQR 0.1 / 0.2; `capture-off` `283.7` / `283.7`: `share=+0.6 unc=0.3 -> isolated`, `share=+0.5 unc=0.3 -> isolated` | `capture-on` `284.4` / `284.4`: `share=+1.2 unc=0.2 -> isolated`, `share=+1.2 unc=0.3 -> isolated`; `on_minus_off=+0.7 unc=0.3 -> isolated`, `+0.7 unc=0.2 -> isolated`; the `refused-on` control `on_minus_off=-0.1 unc=0.3 -> under_resolution` both passes | C day 30, `55b8da077`, eight boots in one hold, two passes in opposite order, N=5 per arm per order; 33 to 61 C, 32.0 to 501.4 W; `DAY30 CAPTURE-SHARE VERDICT: pass1 refused on-off -0.1 (unc 0.3) under_resolution; pass1 share off +0.6 (unc 0.3) isolated; pass1 share on +1.2 (unc 0.2) isolated; pass1 capture on-off +0.7 (unc 0.3) isolated; pass2 refused on-off -0.1 (unc 0.3) under_resolution; pass2 share off +0.5 (unc 0.3) isolated; pass2 share on +1.2 (unc 0.3) isolated; pass2 capture on-off +0.7 (unc 0.2) isolated; admissible=True` | `C/pro-single-day30/stall/ev/pass{1,2}/{refused-off,refused-on,capture-off,capture-on}/run/receipt.json`, `C/pro-single-day30/reading.log` |
| The D2D receipt's price, ONE 158 MiB span on the copy stream, event-timed (A's cell (v)) | copy `copy_median=0.158` (copy-first), `0.156` (digest-first) | digest `digest_median=0.168` / `0.167`; pair `pair_median=0.335` / `0.335`; `pair_over_copy=2.12` / `2.15` | A day 22, `bb1a2b212`, N=5 per order, both orders, one hold; 36 to 40 C, 33.5 to 95.8 W (a unit cell) | `A/pro-single-day22/box/unit-cell/command.log` |
| The same on A's day-23 tree | `0.159` / `0.158` | `0.168` / `0.169`; pair `0.336` / `0.337`; `pair_over_copy=2.11` / `2.13` | A day 23, `2b850b2b0` | `A/pro-single-day23/box/` (the unit-cell log) |
| The cached-destination pair (the door's copy-and-hash cost on the server's own lines, no tenant): demote pooled | `median 37.8 ms (N=10, min 6.1, max 43.1)` | `median 113.6 ms (N=10, min 82.1, max 118.4)` | A day 15, `for_device` cached, N=5 per arm per order, both orders, one hold; 36 to 51 C, 87.3 to 490.7 W; `WC PAIR REPLAY: PASS (12 checks)` | `A/pro-single-day15/wc-pair/` (replay `C/wc-pair.py`) |
| The cached pair: promote pooled; promote minus inline demote | `11.4 (N=10)`; `4.4 (N=10)` | `88.7 (N=10)`; `5.8 (N=10)` | as above | as above |
| The write-combined pair (the pre-`for_device` engine; superseded on this class by `docs/decisions/PINNED-DESTINATIONS.md`) | demote `37.8 (N=10)`; promote minus inline demote `4.5` | demote `169.2 (N=10)`; promote minus inline demote `33.2` | C day 16, `1b354be59`, one hold; 37 to 51 C, 33.0 to 491.6 W; `WC PAIR REPLAY: PASS (12 checks)` | `C/pro-single-day16/wc-pair2-retry3/` |
| One SHA-256 pass over 160 MiB of pinned host memory (the price of each on-tick hash of the Move 1 demote) | n/a (OFF hashes nothing) | `cached_ms=77.922 wc_ms=1698.063 heap_ms=77.990` (`cached_range=77.823..78.041`) | C day 18, `hash-micro --bytes 167772160 --n 5`, two orders; 36 to 38 C | `C/pro-single-day18/hashmicro/` |
| The arena against the pageable tier, door OFF in both arms (an input to the arena item, not the door's cost) | page-pinned `first_touch_page_o1=35.4 first_touch_page_o2=35.8 steady_demote_page=6.1 demote_page=38.0 promote_page=11.3 promote_excl_page=4.5` | arena `first_touch_arena_o1=0.0 first_touch_arena_o2=0.0 steady_demote_arena=6.2 demote_arena=6.2 promote_arena=6.6 promote_excl_arena=0.4` | C day 17, `N=5/arm/order pooled=10 orders=2`, 33 to 51 C, 32.5 to 492.5 W; `ARENA PAIR REPLAY: PASS (18 checks)` | `C/pro-single-day17/arena-pair/` |
| The double park AFTER the promoted-pin refusal (A day 26, ruling 36's proposal 1 landed, `e008bf502`; the day-25 pair's shape byte-for-byte): the tenant's stall; the request's e2e; the demote's on-tick share | stall `85.3` / `85.3`; e2e `115.3` / `115.4` (median over 50 runs per order); `demote_in median=6.2 promote_in median=10.7`; tenant top gaps `98.6` and `16.6` | stall `81.8` / `81.8` (IQR 0.0; `on_minus_off=-3.4 unc=0.1 -> isolated` both orders; `149.4` before the refusal, day 25); e2e `206.8` / `206.7` (`on_minus_off=+91.4 unc=1.2` / `unc=1.1 -> isolated`; `221.4` before); `demote_in-completion median=74.8` (`74.9` before: unchanged), `demote_in median=172.3` (`97.2` before: the completion poll waits a longer tick), `promote_completion median=19.6 promote_in median=26.1`, tenant top gaps `95.3` and `92.4`, `sum median=187.8` (`185.4` before); 100 of 100 ON runs `parked_per_run=[1] restore_submitted_per_run=[0] not_routed_per_run=[1]` | A day 26, twenty interleaved boots (ON OFF x5 / OFF ON x5), N=5 boots per arm per order, both orders, one hold; 32 to 52 C, 31.98 to 360.81 W (4672 samples at 250 ms); 20 of 20 replays PASS, `DAY26 DOUBLE-PARK ADMISSIBLE all_receipts=True`; verbatim `DAY26 CLAUSE 3 stall order=o1 on_stall_medians=[81.9, 81.8, 81.8, 81.8, 81.8] on_cell_median=81.8 IQR=0.0 off_cell_median=85.3 on_minus_off=-3.4 unc=0.1 -> isolated`, `DAY26 CLAUSE 2 e2e order=o1 on_median=206.8 off_median=115.3 on_minus_off=+91.4 unc=1.2 expected=+15.8 (day 25: +105.85 minus 90.1) |d-expected|=75.7 -> FAIL` (the clause's day-25 premise, re-derived by ruling 37: the token of a one-token request is emitted a tick after its prime, so the hash tick stays in the request's path; the code stays) | `A/pro-single-day26/box/double-park/ev/{o1,o2}/bNN-{on,off}/`, the lines regenerated on day 34 with `A/day26-reading.py pro-single-day26/box` and `A/day25-double-park-reading.py pro-single-day26/box/double-park/ev` (`A/DAY26.md`; on `main` since #647) |
| The same cell on the day-27 tree (`ad4f229e0`, the day-26 code): the baseline option (a) is to be measured against (A day 27 section 3) | stall `85.2` / `85.3`; e2e `115.4` / `115.5`; `demote_in median=6.2 promote_in median=10.7` | stall `81.9` / `81.8` (`on_minus_off=-3.4` / `-3.5 unc=0.1 -> isolated`); e2e `206.6` / `206.7` (`on_minus_off=+91.3 unc=0.9` / `1.0 -> isolated`); `demote_in-completion median=74.8`, `demote_in median=172.2`, gaps `95.3` and `92.4`, `sum median=187.8` | A day 27, the same shape, one hold, twenty boots; 33 to 51 C, 33.27 to 359.98 W (4671 samples); 20 of 20 replays PASS, `ADMISSIBLE all_receipts=True`; equal to day 26 within 0.1 ms | `A/pro-single-day27/box/double-park/ev/`, banked `box/reading-day25.log` and `box/reading-day26.log` (regenerated on day 34 with `A/day25-double-park-reading.py`, equal) (integ43) |
| The same cell on the day-28 tree (`f9284a711`, code `45f824a75`: option (a), the bundle hash on the helper thread; A day 28 sections 4 and 5) | stall `85.2` / `85.2`; e2e `115.4` / `115.4`; `demote_in median=6.2 promote_in median=10.6` | stall `81.8` / `81.9` (`on_minus_off=-3.4 unc=0.1 -> isolated`, both orders); e2e `132.3` / `132.2` (`on_minus_off=+16.9 unc=3.3` / `unc=1.1 -> isolated`); the ledger's `owner in-completion` `median=7.40` (A's N=100, clause 1c), `demote_in median=183.9` (t0 to publication, the helper's 73 ms inside it, off the tick), gaps `95.3` and `19.0`, `sum median=114.3` | A day 28, the same shape, one hold, twenty boots; 33 to 52 C, 31.69 to 359.64 W (4639 samples); 20 of 20 replays PASS, `ADMISSIBLE all_receipts=True`; `DAY28 VERDICT clauses_failed=0 -> ALL PASS` | `A/pro-single-day28/box/double-park/ev/`, `box/reading-day{25,26,28}.log` |
| The same cell on the day-29 tree (binary at `29a1cc366`, code `867655368`: option (a) plus 2a, the parked hit on a `Hashing` entry; A day 29) | stall `85.4` / `85.1`; e2e `115.5` / `115.3`; `demote_in median=6.2 promote_in median=10.7` | stall `81.7` / `81.8` (`on_minus_off=-3.6` / `-3.3 unc=0.1 -> isolated`); e2e `132.3` / `132.2` (`on_minus_off=+16.8 unc=1.8` / `+16.9 unc=2.7 -> isolated`); `owner in-completion` `median=7.39` (A's N=100), `demote_in median=183.9`, gaps `95.3` and `18.9`, `sum median=114.2`; 0 parked hits on the 110 ledger lines (appendix A command 1) | A day 29, the same shape, one hold, twenty boots; 33 to 50 C, 32.67 to 358.14 W (4639 samples); 20 of 20 replays PASS, `ADMISSIBLE all_receipts=True`; `DAY28 VERDICT clauses_failed=0 -> ALL PASS` | `A/pro-single-day29/box/double-park/ev/`, `box/reading-day{25,26,28}.log` |
| The demote's owner-thread ledger over the day-28 and day-29 cells' ON boots (every `demote digests landed off the tick` line of the server logs, 11 per boot, 10 boots; appendix A command 1) | n/a | day 28: `pre_submit` `median=6.07` (5.85 to 6.77, N=80, demotes 4 to 11) and `42.19` (37.39 to 43.91, N=30, the first three per boot); `copy_settle` `1.21`; `hashing_polls` `0.01`; `take_back_bind_publish` `1.29`; helper `hashed_in` `73.20` (73.00 to 73.50); `98 payloads (157.9MB)`. Day 29: `6.05` (5.92 to 6.61), `42.27` (37.38 to 44.13), `1.21`, `0.01`, `1.29`, `73.10` (72.90 to 73.50) | as the two rows above | as above, `server.log` per boot |
| The digest micro-cell on the target card's host: the bundle program (SHA-256, `memra_tier::contracts::checksum`) against the slice-3 four-lane digest (`memra_tier::conformance::receipt_digest`) over 160 MiB of heap, the memory kind of the 157 MB | n/a (OFF hashes nothing) | `sha_ms=77.589 lanes_ms=70.737 lanes_over_sha=0.912` (`sha_range=77.538..77.877 lanes_range=70.662..71.044 sha_gbps=2.162 lanes_gbps=2.372 sha_stable=true lanes_stable=true`) | A day 27, `A/day27-digest-micro/` (a detached project calling the two engine programs by path dependency), `--bytes 167772160 --n 5`, two orders interleaved call by call, pooled N=10 per program, one collector hold, the card idle (33 C, 33.50 W, 8 samples); A's pre-registered rule: (b) or (b') lowers a host's hash cost iff `lanes_over_sha < 1` with disjoint ranges, read TRUE here (9 percent; both programs compute-bound near 2.2 to 2.4 GB/s, so (b') would take about 7 ms off the 74.8, A's reading) | `A/pro-single-day27/box/digest-micro/ev/digest-micro.log` (integ43) |

RTX 5090 Laptop GPU, the 9B (this card's own figures):

| Quantity | OFF | ON | N, order, regime | Receipt |
|---|---|---|---|---|
| The D2D receipt's price, one 158 MiB span (cell (v)'s twin) | copy `copy_median=0.394` (copy-first), `0.395` (digest-first) | digest `digest_median=0.221` / `0.221`; pair `pair_median=0.442` / `0.441`; `pair_over_copy=1.12` / `1.12` | C day 28, `8803f4b6c`, N=5 per order, both orders, one sitting under `flock`; 58 C before and after, P8 before, no compute app, `power.limit [N/A]` | `C/rtx5090-day28/price/test.log`, `card.{before,after}.csv` |
| One SHA-256 pass over 160 MiB on this host | n/a | `cached_ms=37.339 wc_ms=1431.613 heap_ms=37.480` | C day 18, two orders; 58 to 59 C, 28.3 to 54.0 W | `C/rtx5090-day18/hashmicro/` |
| The door's demote and promote cost as a PAIR on this card (the day-16 `wc-cell` shape adapted to the 9B: cache 64 MB, host tier 8192 MB, the default spec boot, 64-token draft-bearing entries of 54.8 MB demoted as `18 items`; the server's own `demote: ... in Y ms` and `promote: ... in Y ms` lines; this class's destinations write-combined, printed inside the same hold by `tier-transfer-gate roundtrip`: `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4`, `driver_flags=6` at six sizes): demote pooled | `median 20.0 (N=10, min 4.8, max 24.1, IQR 18.0)`; per boot `[23.1, 23.8, 21.8, 5.1, 4.8, 5.3]` and `[18.2, 24.1, 22.6, 5.4, 6.5, 17.6]` (r2 to r7) | `median 57.3 (N=10, min 38.5, max 60.4, IQR 19.3)`; per boot `[57.8, 60.4, 58.9, 39.7, 39.5, 39.7]` and `[57.7, 58.8, 56.9, 39.4, 38.5, 38.7]`; the door's own `demote published off the tick ... from submission to completion` `median 26.5 (N=12, min 17.3, max 37.4, IQR 18.1)` | C day 31, tree `934a6da3a` (binary built at `5cef58f09`), four boots `o1-off, o1-on, o2-on, o2-off` in ONE collector hold on the `rtx5090` rig (35 s, 12:47:30Z to 12:48:06Z), N=5 per arm per order, both orders, pooled N=10; the collector's 250 ms CSV 192 samples 54 to 74 C, 9.5 to 169.9 W, the cell's 1 s CSV 36 samples 55 to 74 C, 27.7 to 169.3 W, `power.limit [N/A]`, P8 at 54 C before and P0 at 62 C after, no compute app before or after; `demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated` | `C/rtx5090-day31/pair/wc-pair/ev/{o1,o2}-{off,on}-server.log`, `C/rtx5090-day31/pair/reading.log`, `wc-pair-replay.log` (`WC PAIR REPLAY: PASS (12 checks)`), `pair/wc-pair/ev/pinned-default.log` |
| The same pair: promote pooled, and promote minus its inline demote | promote `median 15.6 (N=10, min 8.6, max 27.7, IQR 17.6)`, per boot `[27.6, 25.3, 8.6, 8.7, 8.8]` and `[27.7, 26.0, 9.5, 9.9, 21.3]` (r3 to r7); promote minus inline demote `median 3.5 (N=10, min 3.4, max 4.1, IQR 0.4)` | promote `median 21.1 (N=10, min 20.1, max 42.4, IQR 18.5)`, per boot `[42.4, 38.4, 21.5, 20.5, 20.1]` and `[40.2, 38.1, 20.6, 20.4, 20.1]`; promote minus inline demote `median -19.3 (N=10, min -37.4, max -15.4, IQR 7.4)` (negative on this tree because the inline demote is submitted inside the promote's window and published at a LATER tick top than the promote, so day 16's subtraction does not isolate the ON promote; the log order per hit is `demote submitted`, `promote published`, `D2H receipt`, `demote published`); the door's own `promote published off the tick ... from submission to completion` `median 14.9 (N=10, min 14.6, max 15.9, IQR 0.7)` | as above; `promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated`; every ON boot `parked=10` and `restore_submitted=5` for its 5 promotes (the day-29 double park, on this card, on draft-bearing entries: each hit parks for the promote and again for Move 2's restore, `restore landed ... 33.1ms to 33.4ms to re-admission`) | as above |
| The same pair, read as a whole (the verdict line of the pre-registered reading, verbatim) | | `DAY31 PAIR VERDICT: demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated; promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated; parked per boot [0, 10, 10, 0]; pinned=write-combined; admissible=True` | The pooled medians straddle a step the raw lists show in both arms: the first three demotes of every boot read 18 to 24 (OFF) and 57 to 60 (ON), the later ones 5 to 7 (OFF; o2-off's r7 17.6 excepted) and 38 to 40 (ON), the first-touch step of A day 17's pair on this card too; hence the IQRs of 18 to 20 and the `unc` of 26. Stated, not tuned: the cell reads what day 16's shape reads. | `C/rtx5090-day31/pair/reading.log` |
| The on-tick hash against write-combined memory on this card (section D item 6, the measurement side): the engine hash at the pair cell's entry size (54,800,000 B) and at 160 MiB over cached pinned, write-combined pinned and heap memory, plus the two-step route (memcpy of the write-combined buffer into cached pinned memory, then the hash of the copy), read against the day-31 ON demotes | n/a | `DAY33 HASH-WC VERDICT: 54.8MB cached 11.9 wc 473.8 heap 11.9 wc_copy 224.2 (N=10 each); 160MiB cached 36.6 wc 1450.8 heap 37.2 wc_copy 686.8 (N=10 each); day31 ON in median 48.3 max 60.4 (N=12) in_minus_completion median 21.6 (N=12); H1-single fits=False H1-twostep fits=False -> H1 refuted (no WC read route fits the ON demote's wall time; H2 or H3 stands, separated by the code census, not by this cell); pinned=write-combined; admissible=True` | C day 33, `hash-micro` (with its opt-in `--two-step` arm) in one collector hold, N=5 per kind per order, both orders, N=10 pooled, 56 to 58 C, 9.5 to 31.7 W, premise `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4` inside the hold; 41 ok, 0 FAIL. The write-combined rate is size-independent (`wc_gbps=0.116` at both sizes); the fastest WC single pass at the entry size (468.8 ms, N=20) is 7.8x the slowest ON demote `in` (60.4), the fastest two-step (222.0 ms) 3.7x; `two_hashes_cached_ms=23.723` sits beside the post-completion segment (`in` minus completion median 21.6, max 23.0). H2 against H3 is A day 27's code census (section 2): heap, H2's form, for about 98 percent of the hashed bytes | `C/rtx5090-day33/hashwc/reading.log`, `hashwc/ev/{micro,twostep}-{54m,160m}.log`, `hashwc/ev/pinned-default.log`, `C/DAY33.md` |
| The digest micro-cell on this host (the same two engine programs over 160 MiB of heap) | n/a | `sha_ms=37.111 lanes_ms=55.078 lanes_over_sha=1.484` (`sha_range=36.837..40.676 lanes_range=54.490..56.115 sha_gbps=4.521 lanes_gbps=3.046 sha_stable=true lanes_stable=true`) | A day 27, N=5 per order, two orders, pooled N=10, one collector hold on `/tmp/memra-5090.lock`, the card idle (56 C, 16.54 W, 5 samples); A's rule read FALSE here (the four-lane program 48 percent slower than SHA-NI on this host); the digests byte-identical to the target host's for both programs (`sha_digest=1c0d84cd...`, `lanes_digest=1bf19aae...`) | `A/rtx5090-day27/digest-micro/ev/digest-micro.log` (integ43) |
| The tenant-stall cell on this card, DEMOTE class (A day 16's shape on the 9B: `MEMRA_SERVE_SPEC=0`, cache 64 MB, host 8192 MB, the plain 64-token entry `54.6MB, 16 items`; the tenant 400 tokens at this card's 7.3 ms tick so the intruders land inside its window, A's intruder shapes byte-for-byte; the rule `stall = worst ITL minus p50`) | pass 1 `stall_median=63.7` (IQR 2.0), pass 2 `63.0` (IQR 2.3); `server_demote_ms` 17.1 to 26.9 | pass 1 `stall_median=67.0` (IQR 2.6), pass 2 `63.2` (IQR 1.5); `server_demote_ms` 54.6 to 70.2; `demote submitted off the tick` / `D2H receipt require=ok` / `demote published off the tick` in every demote run | C day 35, tree `091a931c0`, six boots in ONE collector hold on `/tmp/memra-5090.lock` (16:18Z to 16:29Z), two passes in opposite order (pass 1 prime, off, on; pass 2 on, off, prime), N=5 per arm per order inside every boot, N=10 pooled per pass; 58 to 89 C, 30.64 to 175.33 W (2603 samples at 250 ms), `power.limit [N/A]`; 10 of 10 `STALL REPLAY: PASS`; verbatim `demote pass1 off 63.7 on 67.0 on-off +3.3 (unc 3.3) under_resolution; ... demote pass2 off 63.0 on 63.2 on-off +0.3 (unc 2.7) under_resolution`. Pass 1 ran beside an unidentified co-tenant (card-wide memory 20.3 to 23.2 GB against 7.6 to 9.8 GB in the identical pass-2 boots; it left at 16:22:57Z); pass 2 is the clean pass | `C/rtx5090-day35/stall/ev/pass{1,2}/{off,on}/demote/receipt.json`, `C/rtx5090-day35/reading.log` |
| The same cell, PROMOTE class (a host hit of 64 with a 22- to 25-token suffix whose insert evicts the other entry; the promote-then-hit shape, so A day 26's refusal fires) | pass 1 `stall_median=47.7` (IQR 0.9), pass 2 `47.5` (IQR 2.4); `server_promote_ms` 8.2 to 10.3 steady (27.2 to 31.6 first touch), inline demote 4.6 to 6.6 steady | pass 1 `stall_median=49.3` (IQR 12.5: the first-touch pair and run 6 read 65.1 to 67.6), pass 2 `50.1` (IQR 1.3); `server_promote_ms` 24.7 to 26.2 steady (41.6 to 45.6 first touch), inline demote 78.3 to 83.1 steady (it spans the park); per run exactly one `promote submitted ... request parked`, one `H2D receipt require=ok`, one `promote published off the tick` (19.7 to 20.3 ms from submission to completion), one `restore not routed (contracts door)`, zero `restore submitted` | as above; verbatim `promote pass1 off 47.7 on 49.3 on-off +1.6 (unc 12.5) under_resolution; ... promote pass2 off 47.5 on 50.1 on-off +2.6 (unc 2.7) under_resolution` | `C/rtx5090-day35/stall/ev/pass{1,2}/{off,on}/promote/receipt.json`, `C/rtx5090-day35/reading.log` |
| The same cell, PRIME arm (the 5120- to 5123-token cold prime alone, a cache-off boot; the class's baseline, no door in it) | n/a (no door) | n/a | pass 2 (clean): `stall_median=278.4 stall_min=261.9 stall_max=285.3` (IQR 6.6), five stretched ticks per run of 220.9 to 292.6 ms summing 1270 to 1358 against intruder walls 1303 to 1391 (`PREFILL_TICK_T = 1024` chunks); pass 1 INADMISSIBLE: the memory admission refused 7 of 10 intruders beside the co-tenant (`[admit-oom] capacity reject: model="gate" ctx=5186 does not fit an IDLE box (available 1794MB), HTTP 400 context_length_exceeded`; the three admitted read 226.9 to 230.5); verbatim `prime pass1 10.5 (iqr 174.2) inadmissible; ... prime pass2 278.4 (iqr 6.6) admissible` | `C/rtx5090-day35/stall/ev/pass{1,2}/prime/prime/receipt.json`, `ev/pass1/prime/server.log` |
| The day-27 attribution on this card (the ON demote's `in - completion`), and where it lands in the tenant's ticks | `demote_in median=25.7` / `25.2` (N=9 per pass); two stretched ticks per demote, `top1_median=71.2 / 70.3`, `top2_median=41.3 / 41.9` (pass 1 / pass 2) | `demote_in median=67.4` / `62.9`, `completion median=44.0` / `41.0`, `in_minus_completion median=23.8` (21.7 to 26.2) / `21.6` (20.6 to 23.4), N=9 per pass; two stretched ticks, `top1_median=74.5 / 70.5`, `top2_median=70.9 / 69.4`: the door's share lands on the SECOND stretched tick (+29.6 / +27.5 over OFF's), the worst tick is the same in both arms, so the rule reads `+3.3 / +0.3` while the stretched-tick sum reads `146.9` against `112.8` (+34.1) and `140.0` against `112.3` (+27.7) | C day 35, the same receipts; `in - completion` about 1.8 to 2.0 of day 33's 11.9 ms heap pass at the entry size on this host (the segment also holds the two KV-plane receipt hashes, the publish and the insert; not split here). The tick placement is a post-hoc description (`C/day35-gaps-posthoc.py`, written after the run, labelled), the `in - completion` figures are the pre-registered reader's | `C/rtx5090-day35/reading.log` (`DAY35 ATTRIBUTION`), `C/rtx5090-day35/gaps-posthoc.log` |
| The demote's owner-thread ledger on this card, option (a) plus 2a (the identity gate's default-ON boot, `MEMRA_KV_HOST_VERIFY=1`, so the pre-submit segment includes the verify digest; no stall cell ran on this tree on this card) | n/a | ticket seq=3: `50 payloads (53.7MB) hashed in 12.9ms on the hash helper, landed after 6 poll(s)`, `pre-submit 51.56, copy settle 9.21 over 1 poll(s), hashing polls 0.01, take-back bind and publish 9.24; owner in-completion 60.80ms`, `1 hit(s) parked on the Hashing entry (5 re-park(s))`; seq=5: `13.0ms`, `pre-submit 53.55`, `copy settle 9.25`, `take-back bind and publish 9.11`, `owner in-completion 62.67ms`, 0 parked | N=2 ledger lines, one boot, A day 29, tree `29a1cc366` (appendix A command 1) | `A/rtx5090-day29/identity-default-on/ev/host-on-server.log` |
| The two-binary tenant-stall cell on this card, DEMOTE class, on the option (a) tree (day 35's cell byte-for-byte, run with two binaries: base `091a931c0`, option (a) `8b889dcdf`, A days 28 and 29 plus integ45's `c26255bc7`; the rule `stall = worst ITL minus p50`, and the pre-registered secondary quantity `top1_plus_top2`, the tenant's two largest gaps per run summed; blocks o1 = p2-opta against p1-base, o2 = p3-opta against p4-base) | base `stall` `61.6` (IQR 3.2), option (a) `63.8` (IQR 4.6); `top1_plus_top2` `110.7` (IQR 5.2) and `114.5` (IQR 8.3); option (a) minus base `+1.3 unc 4.1` / `+2.6 unc 5.5` (`stall`, o1 / o2) and `+2.5 unc 6.4` / `+4.5 unc 9.4` (`top1_plus_top2`), `under_resolution` | base `stall` `66.0` (IQR 3.7), option (a) `63.9` (IQR 3.3): option (a) minus base `-1.0 unc 4.2` / `-3.8 unc 4.7`, the DiD `-2.3 unc 5.8` / `-6.4 unc 7.3`, `under_resolution`; `top1_plus_top2` `143.7` (IQR 6.6) and `121.4` (IQR 5.0): option (a) minus base `-21.7 unc 5.2` / `-24.5 unc 8.5` `moved`, the DiD `-24.2 unc 8.2` / `-29.0 unc 12.7` `moved`; ON minus OFF within each binary: `stall` `+4.4 unc 4.9` (base) and `+0.1 unc 5.7` (option (a)), `under_resolution`; `top1_plus_top2` `+33.0 unc 8.4 isolated` and `+6.9 unc 9.7 under_resolution`; the server's `demote: ... in Y ms` pass medians 63.5 to 65.9 (base) and 93.3 to 95.7 (option (a), where it equals the ledger's `wall ... t0 to publication`) | C day 37, four programs p1-base, p2-opta, p3-opta, p4-base in ONE collector hold on `/tmp/memra-5090.lock` (19:47:19Z to 20:31:34Z), each day 35's six boots, N=5 per arm per order inside every boot, N=20 per program, N=40 per binary; 51 to 89 C, 8.74 to 174.32 W (10522 samples at 250 ms), `power.limit [N/A]`, card-wide memory 15 to 9753 MiB, no co-tenant; 40 of 40 `STALL REPLAY: PASS`, `DAY37 ADMISSIBLE: 40 of 40 receipts; all=True`. The difference is the whole `crates/` difference of the two trees, the route-contract commits (memra#504) included; the prime and OFF controls read `under_resolution` in both quantities (rows below) | `C/rtx5090-day37/stall/ev/p*/pass{1,2}/{off,on}/demote/receipt.json`, `C/rtx5090-day37/reading.log` |
| The same cell, PROMOTE class (day 35's promote-then-hit shape) | base `stall` `49.6` (IQR 5.8), option (a) `48.3` (IQR 2.4); `top1_plus_top2` `67.6` (IQR 7.0) and `65.7` (IQR 2.7); option (a) minus base `+0.5 unc 7.6` / `-1.9 unc 3.4` and `+0.9 unc 11.3` / `-2.4 unc 3.3`, `under_resolution` | base `stall` `52.1` (IQR 5.0), option (a) `50.9` (IQR 1.7): option (a) minus base `-1.4 unc 4.4` / `-1.3 unc 7.4`, the DiD `-1.8 unc 8.8` / `+0.7 unc 8.1`, `under_resolution`; `top1_plus_top2` `99.4` (IQR 8.4) and `76.9` (IQR 3.5): option (a) minus base `-23.1 unc 8.7` / `-21.4 unc 9.2` `moved`, the DiD `-24.0 unc 14.2` / `-19.1 unc 9.8` `moved`; ON minus OFF within each binary: `stall` `+2.5 unc 7.6` and `+2.6 unc 2.9`, `under_resolution`; `top1_plus_top2` `+31.8 unc 10.9` and `+11.1 unc 4.4`, `isolated` | as above | `C/rtx5090-day37/stall/ev/p*/pass{1,2}/{off,on}/promote/receipt.json`, `C/rtx5090-day37/reading.log` |
| The same cell, PRIME control (the cache-off boot, no door in it) | n/a (no door) | base `stall` `280.5` (IQR 33.9), option (a) `279.5` (IQR 13.3); `top1_plus_top2` `556.8` (IQR 56.7) and `558.2` (IQR 24.9); option (a) minus base `+27.9 unc 51.4` / `-6.4 unc 15.0` and `+45.6 unc 97.7` / `-11.6 unc 24.4`, `under_resolution`; all eight prime passes admissible; p1-base pass 1, the hold's first boot (the card from 51 C), read `229.0` against `269.7` to `294.2` in the other seven, the source of the base arm's IQR, cause not separated | as above | `C/rtx5090-day37/stall/ev/p*/pass{1,2}/prime/prime/receipt.json`, `C/rtx5090-day37/reading.log` |
| The demote's owner-thread ledger inside that cell, option (a) tree (every `demote digests landed off the tick` line of the demote arm's runs in the four option (a) ON boots; the base tree prints no ledger) | n/a | the boot's first three (N=12): `pre_submit=23.49 copy_settle=8.48 hashing_polls=0.00 take_back_publish=8.36 owner_in_completion=31.88 owner_held=40.29 hashed_in=12.9 wall=91.7`; the 4th onward (N=24): `pre_submit=25.06 copy_settle=8.30 hashing_polls=0.00 take_back_publish=8.28 owner_in_completion=33.48 owner_held=41.85 hashed_in=12.9 wall=95.3`; `payloads=[50]` (53.7 MB), `landed_polls_median=1`, `parked_hits_sum=0`; per ON boot 21 ledger lines and 0 `demote published off the tick` (base: 21 and 0 ledger lines), 0 `hit parked on a Hashing entry` and 0 `hash helper detached` in all eight ON boots | C day 37, as above (medians) | `C/rtx5090-day37/reading.log` (`DAY37 LEDGER`, `DAY37 LEDGER-BOOT`), `stall/ev/p{2,3}-opta/pass{1,2}/on/server.log` |

### 5. Open findings the review must weigh (each with its receipt)

1. **Cell (v)'s day-19 clause.** A's day-19 rule made the `Unwitnessed` arm the receipt "unless the digest's cost
   on the copy stream exceeds the copy's own time"; the PAIR the receipt needs reads `pair_over_copy=2.12` /
   `2.15` on the target card and `1.12` on the 5090, so by the clause's letter the pair exceeds the copy on both
   cards. A reported it verbatim and relaxed nothing; the lead's reading (integ38): the rule compared digest to
   copy without weighing the stream (0.34 ms against a 14 ms tick, off the tick). Receipts above.
2. **Cell (i)'s clause not met; the remainder attributed, and its whole-image pass now off the tick on the target card.** Move 1's pre-registered decision clause
   `stall_median(second stream) <= idle p99` read `clause_not_met` for both classes in the same window (150.0 and
   149.6 against 14.9): the copy stream took 43 ms off the demote's tenant stall and 13 ms off the promote's, and
   neither class is near idle. What remained in the demote's tick under ON (about 32 ms over OFF in day 23's
   same-window pair, 149.5 against 117.5) was attributed by no cell when this packet was drafted; it is attributed
   now by A's code census (A day 27 section 1, section 2 above): the tick pays `bind_tier_image`'s SHA-256 pass over
   the whole host image, `demote_in-completion median=74.8` per 159.8 MB entry on the target card (A days 25, 26 and
   27: `74.9`, `74.8`, `74.8`, N_runs=100 each), which OFF never computes; Move 1's own two receipt hashes are about
   1.8 ms over the 1.9 MB of KV planes, not 77.9 ms per pass. How the 74.8 ms pass shows as 32 ms in the demote arm's
   stall is the lead's reading (integ43, verbatim): "The 32 ms the cell (i) census could not attribute is this pass
   minus what OFF spends on the same tick"; no cell of this lane decomposes it further. The arm that would take the
   pass off the tick is option (a) (A day 27 section 2; ruling 39: "option (a) is approved for A day 28 as
   specified"), stated here as its acceptance gate and not as a prediction: (1) the day-26 double-park cell, one
   hold, both orders, N=5 boots per arm per order, `stall_median(ON) <= stall_median(OFF) + 2.0` on the
   promote-then-hit shape, the request's e2e `on_minus_off <= +20.0`, the demote's `in - completion` on the owner
   thread `<= 12.0` ms; (2) identity x4, failure x2, fault, twin and hit gates `ALL GREEN` in both arms on both
   cards; (3) two new fault cells, `hash-helper-gone` and `hash-never-lands`, each a typed line, nothing published,
   the tier latched; (4) a CPU unit cell proving the helper's digests equal `bind_tier_image`'s on-thread digests
   bitwise over a fixture image; (5) no flag, the door is the switch, no new `MEMRA_*` read (ruling 39 adds: the
   receipt term unchanged, `Hashing` as a `PendingDemote` state under the same fail-closed discipline, one
   long-lived helper per `HostTierContext` joined at shutdown and `host.disable`, the two fault values under the
   existing `MEMRA_KV_HOST_FAULT` row). The baseline those clauses move from is the day-27 row of section 4
   (`demote_in-completion median=74.8`; stall `81.9` / `81.8` against `85.2` / `85.3`; e2e `+91.3`).
   **The receipt (A days 28 and 29; day 36 update).** Option (a) landed on A day 28 and option 2a (the parked hit on a
   `Hashing` entry, ruling 40) on A day 29; ruling 41 on integ45, verbatim: "the day-28 and day-29 code is the door's
   serving path; Move 1 owed item 2 (the bundle hash off the tick) closes on these receipts; item 2a closes; the
   remaining owner-thread cost of a demote is the pre-submit segment (Move 2 owed item 1, A day 30 running)." Clause 1,
   verbatim from each sitting's `box/reading-day28.log`, day 28 first:
   `DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=81.8 off_cell_median=85.2 rule on<=off+2.0 -> PASS`,
   `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=132.3 off=115.4 on_minus_off=+16.9 rule <=+20.0 -> PASS`,
   `DAY28 CLAUSE 1a stall order=o2 N_boots_on=5 N_boots_off=5 on_cell_median=81.9 off_cell_median=85.2 rule on<=off+2.0 -> PASS`,
   `DAY28 CLAUSE 1b e2e order=o2 N_runs_on=50 N_runs_off=50 on=132.2 off=115.4 on_minus_off=+16.9 rule <=+20.0 -> PASS`,
   `DAY28 CLAUSE 1c owner in-completion N=100 median=7.40 min=7.17 max=45.01 runs_with_demote_without_ledger=0 rule <=12.0 -> PASS`,
   `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; day 29:
   `DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=81.7 off_cell_median=85.4 rule on<=off+2.0 -> PASS`,
   `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=132.3 off=115.5 on_minus_off=+16.8 rule <=+20.0 -> PASS`,
   `DAY28 CLAUSE 1a stall order=o2 N_boots_on=5 N_boots_off=5 on_cell_median=81.8 off_cell_median=85.1 rule on<=off+2.0 -> PASS`,
   `DAY28 CLAUSE 1b e2e order=o2 N_runs_on=50 N_runs_off=50 on=132.2 off=115.3 on_minus_off=+16.9 rule <=+20.0 -> PASS`,
   `DAY28 CLAUSE 1c owner in-completion N=100 median=7.39 min=7.20 max=45.18 runs_with_demote_without_ledger=0 rule <=12.0 -> PASS`,
   `DAY28 VERDICT clauses_failed=0 -> ALL PASS`. Clause 2 per gate per card (section 3's rows; A's words verbatim):
   on the day-28 tree the target card read **"Clause 2 FAILS on the target card**, in the default (spec) arm: three
   gates red, one cause" (`identity-default-on` `4 FAILURE(S) (teeth=0)`, `failure-on` `1 FAILURE(S)`, `contract-fault`
   `23 FAILURE(S)`; every other target gate `ALL GREEN` or `-> PASS`), and the RTX 5090 read `fault-default` `23
   FAILURE(S)` with `fault-plain` `ALL GREEN` (121 `ok:`) and the hit gate `ALL GREEN (qwen)` in both arms; the cause, A
   day 28's finding: "a hit that arrives inside the `Hashing` window is a miss". On the day-29 tree, A day 29:
   "**Clause 2 PASSES on both cards.**" Target card: identity x4 `ALL GREEN (teeth=0)` (12 `ok:` each), failure x2 `ALL
   GREEN` (15 `ok:` each), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 `ok:`, ten cells, 0 `FAIL:`), twin x2 `-> PASS`,
   hit `ALL GREEN (qwen)` 61 and 68 `ok:`; RTX 5090: `identity-default-on` `ALL GREEN (teeth=0)` (12), the fault gate
   `ALL GREEN` (123) in both arms on the `rerun2` cells, hit `ALL GREEN (qwen)` 61 and 68; integ45 re-ran the RTX 5090's
   fault gate in both arms (`ALL GREEN`, 123 `ok:` each) and the hit gate (61 and 68) on the battery tree `1c540e050`.
   Clauses 3 to 5 on day 28, A verbatim: "**Clause 3, the two fault cells.** **PASS**" (`hash-helper-gone` and
   `hash-never-lands` 14 `ok:` 0 `FAIL:` each), "**Clause 4, the bitwise digest unit cell.** **PASS**", "**Clause 5, no
   flag.** **PASS**". The parked-on-`Hashing` count (appendix A command 2, `grep -c 'hit parked on a Hashing entry'` over
   120 server logs: the day-29 target card's gate and double-park logs, every RTX 5090 day-29 cell, integ45's RTX 5090
   cells): 1 in each card's `identity-default-on` boot (the target card's ledger `1 hit(s) parked on the Hashing entry
   (35 re-park(s))`, the RTX 5090's `(5 re-park(s))`), 1 in the target card's `failure-on` `digest` cell, 2 in each of
   the four promote cells of every default-arm fault run on the day-29 code that reached them (the target card, the RTX
   5090's `fault-default-rerun` and `fault-default-rerun2`, integ45's `fault-default`), 0 in the other 101, the
   double-park cell's 20 boots and integ45's hit gate among them. What moved from the day-27 baseline, read on the promote-then-hit
   shape only: the owner thread's `in - completion` 74.8 to 7.40 (day 28) and 7.39 (day 29); the tenant's second gap
   92.4 to 19.0 and 18.9; the request's e2e `on_minus_off` +91.3 to +16.9 / +16.9 and +16.8 / +16.9; the tenant's stall
   81.9 / 81.8 to 81.8 / 81.9 and 81.7 / 81.8, against OFF 85.2 / 85.2 and 85.4 / 85.1. Scope: the double-park cell
   is the promote arm's shape; no demote-class tenant-stall cell (cell (i)'s demote class, 150.0 against 14.9) has run
   on the day-28 or day-29 tree on the target card, so what the target card's demote class's stall reads with the pass
   off the tick is measured by no line. On the RTX 5090 class, C day 37 ran day 35's demote-class cell on the option (a)
   tree (`8b889dcdf`) against `091a931c0`, two binaries in one hold, both orders (section 4): by the worst-tick rule
   option (a) minus base is `under_resolution` for the demote-on arm (`-1.0 unc 4.2`, `-3.8 unc 4.7`) and the demote DiD
   (`-2.3 unc 5.8`, `-6.4 unc 7.3`); by the pre-registered secondary quantity, the tenant's two largest gaps summed, the
   demote-on arm `moved` (`-21.7 unc 5.2`, `-24.5 unc 8.5`), its OFF control did not, and the demote DiD `moved`
   (`-24.2 unc 8.2`, `-29.0 unc 12.7`). That cell is the plain 64-token class on the 9B only.
   `C/rtx5090-day37/reading.log`, `C/pro-single-day29/stall/reading.log`, `C/pro-single-day23/stall/reading.log`, `A/DAY27.md` sections 1 to 3,
   `A/DAY28.md` sections 4 and 5, `A/DAY29.md` sections 4 to 6, `lead/INTEGRATION-DAY12.md` integ43 and integ45.
3. **The double park (day 29, not tuned).** On the integ38 tree the promote intruder's hit parks TWICE (the promote,
   then Move 2's restore off the tick: 10 `restore submitted off the tick` and 10 `promote submitted off the tick`
   per boot, re-admission `90.0` to `90.4` ms) and its inline demote publishes `after 1 poll(s)` in `22.3` to
   `58.7` ms (median 22.4; the day-23 tree read `97.0`), so the demote's two hashes land on one tick and the
   promote arm's tenant stall reads `149.6` where the trees of A day 18 and C day 23 read `81.9`. Which slice
   between `0713c1a79` and the integ38 tip moved the copy's landing is not determined.
   `C/pro-single-day29/stall/ev/o1/b01-x/promote/receipt.json` (`server_log_lines`), `C/DAY29.md`.
   A day 25 priced it on the target card (one hold, twenty interleaved boots, N=5 boots per arm per order, both
   orders, 20 of 20 replays PASS, 33 to 52 C), verbatim (`A/DAY25.md`, `A/pro-single-day25/box/double-park/`):
   `DAY25 DOUBLE-PARK stall order=o1 on_minus_off=+64.0 unc=0.1 -> isolated (on 149.4, off 85.4)`;
   `DAY25 DOUBLE-PARK e2e order=o1 on_minus_off=+105.8 unc=1.2 -> isolated (on 221.4, off 115.5)`;
   `DAY25 DOUBLE-PARK stall order=o2 on_minus_off=+64.2 unc=0.1 -> isolated (on 149.4, off 85.3)`;
   `DAY25 DOUBLE-PARK e2e order=o2 on_minus_off=+105.9 unc=1.1 -> isolated (on 221.4, off 115.4)`;
   `DAY25 DECOMPOSITION arm=on N_runs=100 parked_per_run=[2] restore_readmission median=90.1 range=89.7..91.5 restore_completion median=90.1 slack(readmission-completion) median=0.10 max=0.20 promote_completion median=19.6 promote_in median=26.1 demote_completion median=22.3 demote_in median=97.2 demote_in-completion median=74.9 idle_p50(tick)=13.46 residual median=+1.8 range=+1.1..+3.1 | tenant top gaps: largest median=162.8 second median=22.6 sum median=185.7`;
   `DAY25 DECOMPOSITION arm=off N_runs=100 parked_per_run=[0] promote_in median=10.8 demote_in median=6.2 | tenant top gaps: largest median=98.7 second median=16.6`.
   A's reading: the re-admission wait is one tick (13.46) plus the inline demote's two on-tick hashes (74.9) plus
   1.8 residual; the copy itself about 0.5 ms; by construction the restore route parks a whole-entry device hit whose
   planes the same request's promote just landed; the tenant's +64 is the demote's hashes on one tick, not the second
   park. A's proposal 1: in `host_restore_park_probe`, refuse the route by shape when `hpx.promoted_pin` names the hit
   entry (one typed line, the OFF device-hit copy on the tick, no flag, no new state). Lead ruling 36 (integ41):
   proposal 1 APPROVED for A day 26 as specified, with the refusal's typed line naming the pin and the entry so the
   gate can count it, and the hit gate's ON-arm route counts on both cards unmoved. Which slice moved the demote's
   landing (this lane's question) is not determined. Day 31 on the RTX 5090: every ON boot of the pair cell reads
   `parked=10, restore_submitted=5` for its 5 promotes, the same double park on draft-bearing entries (row above).
   A day 26 landed proposal 1 (`host_restore_promoted_this_admission`, one typed `restore not routed (contracts door)`
   line, no flag, no new state) and ran the same cell (section 4 row): 100 of 100 ON runs parked once with the typed
   refusal, the tenant's stall 149.4 to 81.8 (3.4 below OFF's 85.3, `isolated`), the request's e2e 221.4 to 206.8,
   `demote_in-completion` 74.8 unchanged; day 25's "the second park cost the tenant nothing" is refuted by that cell
   (the park had decided which tick the prime landed on). Ruling 37: the code stays. Which slice between `0713c1a79`
   and the integ38 tip moved the copy's landing is still not determined.
4. **A's day-24 retire-settle share, priced on A day 25 (Move 2 owed item 3, closed by ruling 36).** All 11
   spec-boundary captures of the hit gate's spec-on boot published `settled synchronously by a session retire`,
   `106.3` to `204.6` ms from submission to completion (`A/pro-single-day24/box/gates/hitgate-on/qwen-on-server.log`);
   day 24 read that as "the owner thread pays a host wait for the copy at the retire". A day 25 put a typed clause on
   the capture publish line (`; the settle held the owner thread H ms, entered A ms after submission`;
   `PendingCapture.settle_after_ms` / `settle_held_ms`, no new `MEMRA_*` read) and ran the hit gate ON on the target
   card on the same binary (`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 68 `ok:`, day 24's census counts exactly),
   verbatim (`A/DAY25.md`, `A/pro-single-day25/box/gates/`):
   `DAY25 RETIRE-SETTLE settled_by='settled synchronously by a session retire' N=11 whys=['spec-boundary'] toks=[64, 96, 128] | held_ms N=11 min=0.37 median=0.41 max=0.44 | entered_after_ms N=11 min=106.10 median=150.50 max=204.50 | completion_ms N=11 min=106.50 median=151.00 max=205.00 | share_held_over_completion N=11 min=0.00 median=0.00 max=0.00`;
   `DAY25 RETIRE-SETTLE settled_by='tick-top poll' N=3 whys=['seed'] toks=[64] | held_ms N=3 min=0.37 median=0.37 max=0.38 | entered_after_ms N=3 min=87.30 median=87.60 max=87.70 | completion_ms N=3 min=87.60 median=88.00 max=88.00 | share_held_over_completion N=3 min=0.00 median=0.00 max=0.00`.
   Day 24's "host wait for the 159 MB copy at the retire" is refuted by its own typed figure: the copy had landed
   (`share_held_over_completion` 0.00), and the 0.4 ms is the settle's fixed cost, the same at the retire and at the
   tick-top poll. A's proposal 2: no reordering earned, leave the seam. Lead ruling 36: proposal 2 ACCEPTED, the seam
   stays, 0.4 ms is the recorded price, Move 2 owed item 3 closes on this receipt. The C day-25 retire cell
   (`DAY25 RETIRE VERDICT ... -> HOLDS`, R1 to R3 within 3.0 ms) priced the SEED capture's retire settle on #634's
   `Block` arm; it stands beside A's, not in place of it.
5. **The capture share, read on day 30.** Day 28's subtraction arm was a cache-OFF boot whose prime is not split at
   the 5088 seed boundary (hence its 301.5 against 283.7). Against a cache-ON boot whose insert is refused before
   any copy the capture arm's own share is `+0.6` / `+0.5` ms door OFF and `+1.2` / `+1.2` ms door ON (`isolated`,
   unc 0.2 to 0.3), `on_minus_off` `+0.7` in both passes: the whole capture class is about a millisecond of the
   tenant's tick on this card at this entry size, the door's part about 0.6 ms of it, ON above OFF. The restore
   class's 9 ms in both arms is the allocation and the recurrent f32 copies that stay on the owner stream (owed
   item 1), not the rows. `C/pro-single-day30/reading.log`, `C/DAY30.md`.
6. **Neighbouring doors with their own dates.** `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-09-23; its own packet is
   `C/ADMIT-BY-MEMORY-DECISION-PACKET.md`, day 32; its part (b) demotes into the pinned host tier through
   `host_demote_prefix_ref(..., ContractD2h::OnTick)`, the day-16 synchronous tick program under both doors, not this
   door's copy-stream route; a no-op with the tier unarmed); `MEMRA_KV_PARK_COMPACT` (decide-by 2026-10-06 in the
   prose of its `docs/FLAGS.md` row since B's `72f89e233` and in `spill-b-20260919/KV-RESIDENCY-DESIGN.md`'s day-28
   addendum; the row's value column still reads `0 = OFF by design`; the reconciliation proposal is `C/DAY32.md` 2); `kv-tier-gate --kv-allocator vmm` (decide-by 2026-10-04, `docs/decisions/KV-PHYSICAL-RECLAIM.md`);
   the MoE slot cache door (decide-by 2026-10-04, `C/MOE-SLOT-CACHE-DOOR.md`). The arena path
   (`MEMRA_GLM5_TP_KV_HOST=1`) is refused with the door until the lease handoff is built (ruling 28).
7. **Still unbuilt or unmeasured (section D of the door table), after days 31, 33, A day 27 and A days 28 and 29.**
   Move 2 owed item 1, the recurrent f32 state on the contract (the demote's pre-submit segment, section 2: about 6 ms
   steady and 37 to 44 ms at first touch on the target card; A day 30 running, no receipt at the time of writing); the
   arena's lease handoff (item 1; refused with the door, ruling 28); the DFlash tail slice (item 2; no drafter artifact
   identity, no gate boots a drafter); verify digest v3 (item 3); the 9B entry's KV byte split (no `[prefix-host]` line
   prints the per-class split, and adding one is engine code; A day 27); a PRIME arm on the RTX 5090 class that the memory
   admission admits in every run (day 35's pass 1 admitted 3 of 10 beside a co-tenant, pass 2 admitted 10 of 10; a shorter
   prime or another `MEMRA_CTX` is a new pre-registration); and the split of the door's second stretched tick on that class
   (its `in - completion` of 21.6 to 23.8 ms against a second-tick delta of 27.5 to 29.6, day 35 Task 3, measured on
   `091a931c0`, a tree without option (a); day 37's cell on the option (a) tree reads the sum of the tenant's two
   largest gaps, `moved` by `-21.7` and `-24.5` against the base binary, and the owner thread's segments in the ledger,
   but not which of the two ticks moved); and a demote-class tenant-stall cell on the target card on the option (a)
   tree (none has run; item 2's scope). Option (a)
   and option 2a are no longer listed: they landed on A days 28 and 29 and ruling 41 made them the door's serving path
   (item 2). Resolved since day
   30 and no longer listed here: the tenant-stall cell on the RTX 5090 class (day 35, section 4: both classes
   `under_resolution` in both passes by the worst-tick rule, the door's shares on the second stretched tick); the
   demote-class tenant-stall cell on the option (a) code on the RTX 5090 class (day 37, section 4); the RTX 5090
   class's demote and promote PAIR and the
   whole-budget failure arm on the 5090 (items 5 and 9, day 31, sections 3 and 4); the promote-side census question
   and the day-16 write-combined contradiction (item 6: A's code census, `A/DAY27.md` section 1, puts the promote's
   hash 3 over the 1.9 MB of KV source leases inside `promote_completion` and receipts nothing of the recurrent state
   at promote, which fits the pair's 1.4 ms promote-share delta, and reads day 16's write-combined delta as the
   bundle pass plus the write-combined KV share; C day 33's measurement refuted the write-combined-stream reading on
   the 5090, `H1 refuted`, section 4).

### 6. The three outcomes the door hygiene rule allows, and what each would require (not recommended)

- **A naked default per card class.** On the target card class: delete the env read, `host_tier_context`'s
  door plumbing and the OFF program's demote, promote, capture and restore statements; the six fault cells and
  the D2D fault cells stay as the red arms of the check; the identity, failure, hit and twin gates keep their
  single arm. It requires a per-card decision (the per-hardware rule): the RTX 5090 class has one pair cell
  (day 31, N=10 per arm, one hold: demote `on_minus_off +37.3 unc 26.4 isolated`, promote `+5.4 unc 25.6
  under_resolution`) and one tenant-stall cell (day 35, two passes, N=10 per arm per pass, one hold: demote `+3.3 unc 3.3`
  and `+0.3 unc 2.7`, promote `+1.6 unc 12.5` and `+2.6 unc 2.7`, all `under_resolution` by the worst-tick rule, the
  door's share landing on a second stretched tick of about 70 ms (demote) and 40 ms (promote) per intruder), both on
  trees without option (a) (`934a6da3a`, `091a931c0`), and one two-binary tenant-stall cell on the option (a) tree
  (day 37, `8b889dcdf` against `091a931c0`, one hold, both orders, N=20 per side per block: option (a) minus base
  `under_resolution` by the worst-tick rule in every arm and both DiDs; by the sum of the two largest gaps the demote-on
  and promote-on arms and both DiDs `moved`, the OFF and prime controls `under_resolution`; the demote's owner thread
  held `owner_held=40.29` ms median over its first three demotes and `41.85` over the 4th on, N=12 and 24), and its destinations are
  write-combined; a default on either
  class would carry, on the demote's owner thread, the pre-submit segment (on the target card `median=6.07` on the
  4th to 11th demote of a boot and `42.19` on the first three, the first touch of the 32 pinned lease allocations;
  section 2), Move 1's two receipt hashes over the KV planes (about 0.9 ms each, about 1.8 ms together on the target
  host, A's code-read arithmetic), and a hit on a `Hashing` entry parked for one tick at a time until the digests land
  (1 line in the host-on boot of the identity gate's default-ON arm on each card, `35 re-park(s)` and `5 re-park(s)`; item 2); the
  bundle checksum runs on the helper thread since option (a) (`hashed in` `median=73.20` ms per 157.9 MB on the target
  card, 12.9 and 13.0 ms per 53.7 MB on the RTX 5090's two ledger lines, off the tick, and `hashed_in=12.9` median in
  both parts of day 37's 36 lines), and that code is on `main` since #652 (`0c86309bd`; appendix A, day 37 command 4); the arena would have to take the lease handoff or stay refused; the spec-boundary capture
  (A day 24) is on `main` since #643; the same-program law wanted the double-park finding (item 3) placed before
  the promote path is the only path, and A day 26's proposal 1 is on `main` since #647 with its cell (stall 81.8
  against 85.3, ruling 37). The hygiene rule's winner clause: once a default has served two weeks with its rollback
  seam unused, the seam is deleted.
- **A longer door with a new date and the missing gate named.** Requires the `docs/FLAGS.md` row to carry the new
  `decide-by:` and the row's reason ("pending its X row" is a date, not a state): the candidates the receipts
  name are Move 2 owed item 1 (the recurrent f32 state, the pre-submit segment; A day 30 running), a demote-class
  tenant-stall cell on the target card on the option (a) code (item 2's scope: none has run there; the RTX 5090
  class's ran on day 37), the double-park slice (which slice
  moved the demote's landing), and the 9B's KV byte split. Option (a)'s acceptance gate (A days 28 and 29, item 2,
  ruling 41), the 5090 pair cell (day 31), the 5090 tenant-stall cells (day 35, and day 37 on the option (a) tree)
  and the retire-settle price (A day 25, 0.4 ms) are no longer candidates. Every default-OFF door's date is 14 days after landing unless the row
  says why it needs longer.
- **Deletion, with the verdict and the receipt pointer moved to the removed-doors ledger.** Requires removing in
  one PR the env read, the boot wiring, `host_tier_context` and its helpers, the copy-stream constructor's callers,
  the `Demoting`, `Promoting`, `Capturing` and `Restoring` states and their tick-top polls, the D2D receipt kernel
  and its fatbin, the contract receipts, the fault values and their gate cells, the FLAGS.md and KERNELS.md rows,
  the door doc's status and the tests; the verdict and pointers to `C/HOSTPREFIX-DOOR.md`'s tables move to
  `docs/FLAGS.md`'s "Removed doors" ledger and the darklanes verdicts ledger. The sidecar route the door exercises
  (lane B's) is the lead's to keep or drop. What is lost with it is named in section 1: the offload, the
  attestation, the typed unwind.

## Appendix

### A. How every number above was traced

Each row names its receipt file. The stall rows were read from the harness rule lines inside `receipt.json`
(or the day's `reading.log`, itself generated from those receipts by the committed reading script; day 28's
lines were regenerated on day 30 with `C/day28-stall-reading.py` over `C/pro-single-day28/box/stall/ev` and
matched `C/pro-single-day28/reading.log`). The pair rows were regenerated with `C/wc-pair.py` and
`C/arena-pair.py` over their cell directories (`REPLAY: PASS` in each). Gate `ok:` counts were counted in the
named `gate.log` (or `hitgate-{off,on}.log`) with `grep -cE '(^|[^a-z])ok: '`. Regimes were computed from each
cell's own `command.gpu.csv` (250 ms samples) or, for the 5090 price cell, its `card.{before,after}.csv`.
Trees are the `tree=` lines of the cells' `CELL.txt` / `ev/` files as recorded in the day records. Day 34: the
A day 26 and day 27 double-park lines were regenerated from `ev/` with `A/day26-reading.py` and
`A/day25-double-park-reading.py` under the CPU quota (day 27's equal the banked `box/reading-day25.log` and
`box/reading-day26.log`; day 26 banks no reading log, so the regenerated lines are the receipt); the digest rows
were read from each host's `ev/digest-micro.log` rule line; the day-26 and day-27 regimes from each cell's
`command.gpu.csv` (4672 and 4671 samples), the digest cells' from theirs (8 and 5 samples); the day-33 row from
`C/rtx5090-day33/hashwc/reading.log`; the trees from `ev/CELL.txt`.

Day 36 (no card; every figure added on day 36 comes from one of these commands, each output banked under
`C/day36-cpu/` with the command as its first line or beside its output):

1. `python3 research/spill-c-20260919/day36-receipt-census.py research/spill-a-20260919
   research/spill-lead-20260919/integration-day12` (`receipt-census.log`): the owner-thread ledger over every
   `demote digests landed off the tick` line of A's day-28 and day-29 double-park ON boots (110 per day, 11 per boot,
   0 unparsed), the first three and the 4th to 11th per boot split, each boot's first ledger line checked against the
   promote runs' `server_log_lines` in `promote/receipt.json`; the regimes of the double-park, gates and unit-cell holds
   from each `command.gpu.csv`; the replay count in `double-park/ev/replays.log`; both cards' identity default-ON ledger
   and park lines; every gate log's last verdict line, `ok:` count (`(^|[^a-z])ok: `), `FAIL:` count, fault-gate cell
   list and per-cell `ok:`/`FAIL:`, and the hit gate's door census, for A day 28 and day 29 on both cards and for
   integ45's RTX 5090 cells; the trees from `box/tree.sha`, `battery.log`'s `start ... tree=` lines and integ45's
   `CELL.txt`.
2. `grep -c 'hit parked on a Hashing entry' research/spill-a-20260919/pro-single-day29/box/gates/*/*server.log
   research/spill-a-20260919/pro-single-day29/box/double-park/ev/o*/b*/server.log
   research/spill-a-20260919/rtx5090-day29/*/ev/*server.log
   research/spill-lead-20260919/integration-day12/integ45-*-gate-5090/*/ev/*server.log | sort` (`parked-census.log`,
   120 logs, 19 nonzero).
3. `grep -h '^DAY28 CLAUSE\|^DAY28 VERDICT\|^DAY28 REPORTED'` over each sitting's `box/reading-day28.log`, and `grep -H
   'DAY26 CLAUSE 2\|DAY26 CLAUSE 3\|DAY26 DECOMPOSITION\|DAY25 DOUBLE-PARK\|ADMISSIBLE'` over its
   `box/reading-day2{5,6}.log` (`clause-lines.log`): the clause lines of item 2 and the stall, e2e and decomposition
   figures of section 4's day-28 and day-29 rows. A's reading logs were read as banked, not regenerated.
4. `git merge-base --is-ancestor` of A's day-28 and day-29 code (`45f824a75`, `867655368`) against `934a6da3a`,
   `091a931c0`, `origin/main`, `1c540e050` and `aeefd9233` (`ancestry.log`): the "tree without option (a)" and "not on
   `main`" statements.

Day 37 (RTX 5090 class, one hold; every figure added on day 37 comes from one of these commands, each output banked
under `C/rtx5090-day37/` with the command as its first line):

1. `python3 research/spill-c-20260919/day37-stall-reading.py research/spill-c-20260919/rtx5090-day37/stall/ev`
   (`reading.log`): the per-binary per-arm stall and `top1_plus_top2` medians and IQRs (N=40), `ON - OFF` per binary,
   the per-block `d`, `unc` and outcome under the pre-registered rule (`C/DAY37.md` section 1), the DiDs, the per-pass
   lines, the ledger segments over every `demote digests landed off the tick` line of the option (a) ON boots, and the
   per-boot ledger and `demote published` counts.
2. `python3 research/spill-c-20260919/day37-regime.py research/spill-c-20260919/rtx5090-day37/stall/collector
   research/spill-c-20260919/rtx5090-day37/stall/ev` (`regime.log`): the hold's and each program's samples,
   temperature, power and memory from the collector's `command.gpu.csv`.
3. The boots' `build` lines, `git diff --stat 8b889dcdf a50922b27 -- crates/ | wc -l` (0) and the `--replay` of every
   receipt (40 PASS, 0 fail) (`provenance.log`).
4. `git merge-base --is-ancestor` of A's day-28 and day-29 code (`45f824a75`, `867655368`) against `origin/main` at
   `0c86309bd` (both ancestors; `ancestry.log`): the "on `main` since #652" statement.

A's own clause-1c figure (N=100) and the census's (N=110) differ by each boot's first ledger line, which is in no
promote run's lines (command 1, `outside_promote_runs=10` both days); both are quoted with their N.

### B. Numbers deliberately NOT carried into this packet

- The door table's "43 to 50 C, 88 to 329 W" for A day 16: this packet quotes the range computed over the three
  day-16 cells' own CSVs instead (32 to 58 C, 32.6 to 492.7 W, 1121 samples).
- Any `ok:` count, temperature or wattage not recomputed from a file today.
- The day-13 red finding's `5 FAILURE(S)` count and the day 13 to 17 failure gate's `1 FAILURE(S)`: resolved
  findings whose files were not reopened today; they are in `C/HOSTPREFIX-DOOR.md` section A with their paths.
- Cross-card ratios of any kind.
- The day-26 reader's lines on A's day-28 and day-29 sittings that grade a different pre-registration: `DAY26 CLAUSE 2
  ... expected=+15.8 (day 25: +105.85 minus 90.1) ... -> PASS` / `-> FAIL` and `DAY26 CLAUSE 3 ... against day 25's
  149.4 ... -> FINDING` (`box/reading-day26.log`; banked in `C/day36-cpu/clause-lines.log`). The acceptance gate of item 2
  is A day 27's, read by `A/day28-reading.py`; only its lines are carried as verdicts.

### C. Untraced

None at the time of writing: every number in sections 3 to 5 was read from the file named beside it. If a
later reader finds one that is not, it is to be struck rather than believed.
