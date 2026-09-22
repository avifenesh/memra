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
and are filled in below.

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
- **What still runs on the tick under ON.** The recurrent f32 state (`conv_state`, `ssm_state`, about 157 MB of
  every 27B entry) is copied on the owner stream by both D2D classes (Move 2 owed item 1). The by-reference
  demote routes (the admission reclaim flush, the pause sweep, the handoff) keep the day-16 synchronous program
  (Move 1 owed item 3). The H2D's settle-time wait is an owner-stream wait on the copy's event (rule 3), so the
  tick program orders behind the landing at the settle. The Move 1 demote's two host hashes stay on the owner
  thread inside the tick: `progress`'s completion checksum at the poll and `bind_tier_image`'s bundle checksum at
  publication (Move 1 owed item 2; the hash micro-cell below prices one pass). On the hit gate's shape every
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

RTX 5090 Laptop GPU, the 9B (this card's own figures):

| Quantity | OFF | ON | N, order, regime | Receipt |
|---|---|---|---|---|
| The D2D receipt's price, one 158 MiB span (cell (v)'s twin) | copy `copy_median=0.394` (copy-first), `0.395` (digest-first) | digest `digest_median=0.221` / `0.221`; pair `pair_median=0.442` / `0.441`; `pair_over_copy=1.12` / `1.12` | C day 28, `8803f4b6c`, N=5 per order, both orders, one sitting under `flock`; 58 C before and after, P8 before, no compute app, `power.limit [N/A]` | `C/rtx5090-day28/price/test.log`, `card.{before,after}.csv` |
| One SHA-256 pass over 160 MiB on this host | n/a | `cached_ms=37.339 wc_ms=1431.613 heap_ms=37.480` | C day 18, two orders; 58 to 59 C, 28.3 to 54.0 W | `C/rtx5090-day18/hashmicro/` |
| The door's demote and promote cost as a PAIR on this card (the day-16 `wc-cell` shape adapted to the 9B: cache 64 MB, host tier 8192 MB, the default spec boot, 64-token draft-bearing entries of 54.8 MB demoted as `18 items`; the server's own `demote: ... in Y ms` and `promote: ... in Y ms` lines; this class's destinations write-combined, printed inside the same hold by `tier-transfer-gate roundtrip`: `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4`, `driver_flags=6` at six sizes): demote pooled | `median 20.0 (N=10, min 4.8, max 24.1, IQR 18.0)`; per boot `[23.1, 23.8, 21.8, 5.1, 4.8, 5.3]` and `[18.2, 24.1, 22.6, 5.4, 6.5, 17.6]` (r2 to r7) | `median 57.3 (N=10, min 38.5, max 60.4, IQR 19.3)`; per boot `[57.8, 60.4, 58.9, 39.7, 39.5, 39.7]` and `[57.7, 58.8, 56.9, 39.4, 38.5, 38.7]`; the door's own `demote published off the tick ... from submission to completion` `median 26.5 (N=12, min 17.3, max 37.4, IQR 18.1)` | C day 31, tree `934a6da3a` (binary built at `5cef58f09`), four boots `o1-off, o1-on, o2-on, o2-off` in ONE collector hold on the `rtx5090` rig (35 s, 12:47:30Z to 12:48:06Z), N=5 per arm per order, both orders, pooled N=10; the collector's 250 ms CSV 192 samples 54 to 74 C, 9.5 to 169.9 W, the cell's 1 s CSV 36 samples 55 to 74 C, 27.7 to 169.3 W, `power.limit [N/A]`, P8 at 54 C before and P0 at 62 C after, no compute app before or after; `demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated` | `C/rtx5090-day31/pair/wc-pair/ev/{o1,o2}-{off,on}-server.log`, `C/rtx5090-day31/pair/reading.log`, `wc-pair-replay.log` (`WC PAIR REPLAY: PASS (12 checks)`), `pair/wc-pair/ev/pinned-default.log` |
| The same pair: promote pooled, and promote minus its inline demote | promote `median 15.6 (N=10, min 8.6, max 27.7, IQR 17.6)`, per boot `[27.6, 25.3, 8.6, 8.7, 8.8]` and `[27.7, 26.0, 9.5, 9.9, 21.3]` (r3 to r7); promote minus inline demote `median 3.5 (N=10, min 3.4, max 4.1, IQR 0.4)` | promote `median 21.1 (N=10, min 20.1, max 42.4, IQR 18.5)`, per boot `[42.4, 38.4, 21.5, 20.5, 20.1]` and `[40.2, 38.1, 20.6, 20.4, 20.1]`; promote minus inline demote `median -19.3 (N=10, min -37.4, max -15.4, IQR 7.4)` (negative on this tree because the inline demote is submitted inside the promote's window and published at a LATER tick top than the promote, so day 16's subtraction does not isolate the ON promote; the log order per hit is `demote submitted`, `promote published`, `D2H receipt`, `demote published`); the door's own `promote published off the tick ... from submission to completion` `median 14.9 (N=10, min 14.6, max 15.9, IQR 0.7)` | as above; `promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated`; every ON boot `parked=10` and `restore_submitted=5` for its 5 promotes (the day-29 double park, on this card, on draft-bearing entries: each hit parks for the promote and again for Move 2's restore, `restore landed ... 33.1ms to 33.4ms to re-admission`) | as above |
| The same pair, read as a whole (the verdict line of the pre-registered reading, verbatim) | | `DAY31 PAIR VERDICT: demote off 20.0 (N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated; promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc 25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated; parked per boot [0, 10, 10, 0]; pinned=write-combined; admissible=True` | The pooled medians straddle a step the raw lists show in both arms: the first three demotes of every boot read 18 to 24 (OFF) and 57 to 60 (ON), the later ones 5 to 7 (OFF; o2-off's r7 17.6 excepted) and 38 to 40 (ON), the first-touch step of A day 17's pair on this card too; hence the IQRs of 18 to 20 and the `unc` of 26. Stated, not tuned: the cell reads what day 16's shape reads. | `C/rtx5090-day31/pair/reading.log` |

### 5. Open findings the review must weigh (each with its receipt)

1. **Cell (v)'s day-19 clause.** A's day-19 rule made the `Unwitnessed` arm the receipt "unless the digest's cost
   on the copy stream exceeds the copy's own time"; the PAIR the receipt needs reads `pair_over_copy=2.12` /
   `2.15` on the target card and `1.12` on the 5090, so by the clause's letter the pair exceeds the copy on both
   cards. A reported it verbatim and relaxed nothing; the lead's reading (integ38): the rule compared digest to
   copy without weighing the stream (0.34 ms against a 14 ms tick, off the tick). Receipts above.
2. **Cell (i)'s clause not met.** Move 1's pre-registered decision clause `stall_median(second stream) <= idle p99`
   read `clause_not_met` for both classes in the same window (150.0 and 149.6 against 14.9): the copy stream took
   43 ms off the demote's tenant stall and 13 ms off the promote's, and neither class is near idle. What remains
   in the demote's tick under ON (about 32 ms over OFF in day 23's same-window pair, 149.5 against 117.5) is not
   attributed by any cell; the two on-tick hashes (77.9 ms per pass) are the census's candidate.
   `C/pro-single-day29/stall/reading.log`, `C/pro-single-day23/stall/reading.log`.
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
6. **Neighbouring doors with their own dates.** `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-09-23; its part (b) demotes
   into the pinned host tier through `host_demote_prefix_ref`, the door's route when ON); `MEMRA_KV_PARK_COMPACT`
   (decide-by 2026-10-06 per `spill-b-20260919/KV-RESIDENCY-DESIGN.md` day-28 addendum; the row still reads `0 =
   OFF by design`); `kv-tier-gate --kv-allocator vmm` (decide-by 2026-10-04, `docs/decisions/KV-PHYSICAL-RECLAIM.md`);
   the MoE slot cache door (decide-by 2026-10-04, `C/MOE-SLOT-CACHE-DOOR.md`). The arena path
   (`MEMRA_GLM5_TP_KV_HOST=1`) is refused with the door until the lease handoff is built (ruling 28).
7. **Still unbuilt or unmeasured (section D of the door table).** The arena's lease handoff; the DFlash tail slice
   (no drafter artifact identity, no gate boots a drafter); verify digest v3; the promote-side census question (the
   census puts a 77.9 ms pass at promote, the pair reads 5.8 against 4.4 on the target card; on the RTX 5090 the
   day-31 pair's ON demote reads `in` 38.5 to 60.4 against a 1431.6 ms micro-cell pass per 160 MiB over
   write-combined memory, so the two hashes over 54.8 MB cannot be reading that memory at that rate on this tree;
   which memory they read is not attributed by any cell). Run on day 31 and no longer missing: the RTX 5090
   class's demote and promote PAIR and the whole-budget failure arm on the 5090 (sections 3 and 4).

### 6. The three outcomes the door hygiene rule allows, and what each would require (not recommended)

- **A naked default per card class.** On the target card class: delete the env read, `host_tier_context`'s
  door plumbing and the OFF program's demote, promote, capture and restore statements; the six fault cells and
  the D2D fault cells stay as the red arms of the check; the identity, failure, hit and twin gates keep their
  single arm. It requires a per-card decision (the per-hardware rule): the RTX 5090 class has one pair cell
  (day 31, N=10 per arm, one hold: demote `on_minus_off +37.3 unc 26.4 isolated`, promote `+5.4 unc 25.6
  under_resolution`) and no tenant-stall cell, and its destinations are write-combined; the arena would have to
  take the lease handoff or stay refused; the spec-boundary capture (A day 24) is on `main` since #643; the
  same-program law wants the double-park finding (item 3) placed before the promote path is the only path (A day
  26's approved proposal 1 is the named change). The hygiene rule's winner clause: once a default has served two weeks
  with its rollback seam unused, the seam is deleted.
- **A longer door with a new date and the missing gate named.** Requires the `docs/FLAGS.md` row to carry the new
  `decide-by:` and the row's reason ("pending its X row" is a date, not a state): the candidates the receipts
  name are the 5090 pair cell, the attribution of the demote's remaining 32 ms (a hash-off-tick arm or a
  GPU-side digest, Move 1 owed item 2), the double-park slice, and the retire-settle price of the spec-boundary
  capture. Every default-OFF door's date is 14 days after landing unless the row says why it needs longer.
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
Trees are the `tree=` lines of the cells' `CELL.txt` / `ev/` files as recorded in the day records.

### B. Numbers deliberately NOT carried into this packet

- The door table's "43 to 50 C, 88 to 329 W" for A day 16: this packet quotes the range computed over the three
  day-16 cells' own CSVs instead (32 to 58 C, 32.6 to 492.7 W, 1121 samples).
- Any `ok:` count, temperature or wattage not recomputed from a file today.
- The day-13 red finding's `5 FAILURE(S)` count and the day 13 to 17 failure gate's `1 FAILURE(S)`: resolved
  findings whose files were not reopened today; they are in `C/HOSTPREFIX-DOOR.md` section A with their paths.
- Cross-card ratios of any kind.

### C. Untraced

None at the time of writing: every number in sections 3 to 5 was read from the file named beside it. If a
later reader finds one that is not, it is to be struck rather than believed.
