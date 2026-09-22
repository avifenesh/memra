# WP-A day 30: the recurrent f32 state rides the demote's copy stream (Move 2 owed item 1, the D2H half)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the target card (BOX3, one RTX PRO 6000 Blackwell, 96 GB,
600 W, `research/spill-lead-20260919/BOX-ACCESS.md`) and the local RTX 5090 rig where its lock allows; no cross-card
comparison. Every cell `executed-not-qualified`. Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development`
mode (logged; no qualification claimed).

## 0. Resync (the owner stopped the first day-30 run; this is the restart)

- `git fetch`; `origin/lane/spill-a-20260919` = `ebf94591d` at the restart: day 29 sealed (`1cce45646`) plus the
  stopped run's one act, a fast-forward onto the lead's local integ45 ref, pushed (the remote-tracking reflog). The
  worktree was clean at the same commit. The stopped run left no day-30 commit, branch, stash or file.
- BOX3, over the lead's ControlMaster socket only (`ssh -O check` first): `/root/wt-a` on the day-29 checkout, nothing
  newer; `/root/spill-receipts/` holds no `a-day30` directory (the `day30` directory there is lane C's, not touched);
  no `memra-server` or `cargo` process of mine; `/tmp/memra-gpu.lock` had no holder. The local RTX 5090 was idle.
  Nothing of the stopped run survived, so nothing was killed.
- Merged `origin/lane/spill-integ45-20260922` (fast-forward `ebf94591d..96d96e1c2`; no `HOSTPREFIX-DOOR.md` conflict
  arose) and pushed. `origin/main` then moved past `ca5a90e2a` to `189c91b15` (#595, research files only); merged it
  into the lane before this section's commit.

## 1. The read, before any design (file:line at `96d96e1c2`)

**Where the planes live.** A session's recurrent state is `memra_kv::RecurLayer { conv_state, ssm_state, ssm_state_alt }`
(`crates/memra-kv/src/lib.rs:1212`, fields 1213 to 1214), `Cache.recur` (`:2143`), device-resident and written by every
decode step. A published device entry owns its own device clones: `PrefixEntry.conv` / `.ssm`
(`crates/memra-server/src/worker.rs:7190`, `Vec<Option<CudaSlice<f32>>>`), taken at the boundary by
`prefix_snapshot` (`:17825`, the clones at `:17932` to `:17933`) and `prefix_capture_off_tick` (`:14363`, the clones at
`:14410` to `:14411`), both `engine.clone_dtod` on the OWNER stream (the capture rule's own words: the next decode
overwrites the live state, so a copy-stream read would race it). A host entry holds them as
`HostPrefixEntry.conv` / `.ssm` (`:8326`), each a `HostF32` (`:8262`): `Heap(Vec<f32>)` with the arena off (the
contracts door's case) or `Pinned(PinnedHostBuf)` from the startup arena.

**How the D2H demote moves them today.** `host_entry_from_device` (`:12023`): the KV planes go to the contract
(`ContractD2h::OffTick`, `host_kv_planes_submit_contract` `:10417`, one D2H batch on the copy stream), then every
conv and ssm plane goes down through `HostF32::down` (`:12165`, `:12179`), which with no arena lease is
`host_glm::read_f32` (`worker/host_glm.rs:13`): a synchronous `clone_dtoh` on the owner stream into a fresh pageable
`Vec`. That loop is the demote's `pre-submit` segment at steady state (DAY28 second finding). `last_logits` and
`last_h` are host `Vec`s already (`HostF32::from_slice`, a memcpy).

**How the H2D promote and both D2D classes move them.** Promote: `device_entry_from_host_parts` (`:13340`) uploads
each plane with `engine.htod` (`:13430`, `:13437`; `crates/memra-engine/src/lib.rs:13404`, `clone_htod` on the owner
stream from the heap `Vec`); the KV planes take the contract. D2D capture: the owner-stream `clone_dtod` above. D2D
restore: `prefix_restore_at` (`:18223`, `copy_into` at `:18247` to `:18248`) and `host_restore_submit` (`:15910`,
`copy_into` at `:15926` to `:15929`, on the owner stream, before the copy-stream KV batch);
`prefix_restore_validate` (`:18046`) checks the recurrent shapes.

**Bytes per entry**, counted over the receipt files (commands banked here, run from `research/spill-a-20260919`):

    grep -rhoE '[0-9]+ heap payloads \([0-9.]+MB\)' --include='*.log' . | sort | uniq -c
    grep -rhoE 'demote submitted off the tick: [0-9]+ tokens, [0-9.]+MB, ticket seq=[0-9]+, [0-9]+ items' \
      --include='*.log' . | sed -E 's/ticket seq=[0-9]+, //' | sort | uniq -c

The 27B (the target card's artifact): `98 heap payloads (157.9MB)` = 96 recurrent planes (48 layers, conv and ssm)
plus the logits and the hidden row, in a `64 tokens, 159.8MB` entry with `32 items` (34 with the MTP draft plane):
the recurrent state is about 98.8 percent of the demoted bytes at 64 tokens. The 9B (the 5090's artifact):
`50 heap payloads (53.7MB)` = 48 recurrent planes plus the two rows, in a `64 tokens, 54.6MB` entry with `16 items`
(18 with the draft).

**What the receipt term covers today.** A KV plane: the engine's `progress` checksums the landed pinned lease
(`tier_transfer.rs:1620`, `s.checksum = Some(checksum(item.host...bytes()?))`, hash 1, on the owner thread), the
bind checksums the same lease again (`bind_tier_image` `:9311`, hash 2) and refuses a difference (the `flip-demote`
fault is the one named exception). A recurrent plane: NO transfer receipt. It crosses by the synchronous
`clone_dtoh`, whose success is its only witness; its bundle checksum is the hash helper's digest of the heap `Vec`
(`Role::Recurrent`, `receipt None`, `precomputed` from `HostHashDigests`; day 28), with the byte count checked against
the payload.

**Every path that reads or writes the planes while a copy could be in flight.** The demote's source is the evicted
entry's OWN clones (`dead.conv` / `dead.ssm`). While the entry is `Demoting` it is in neither index (out of the device
LRU, unpublished on the host), so no restore, no spec session re-arm and no `spec_session_from_restored*`
(`crates/memra-engine/src/spec.rs:9215`, `:9275`, `:9396`; they take a restored `Cache` by value and write its live
`recur`, never an entry's clones) can reach them, and nothing writes them after the capture. The one reader is the
copy itself; the one writer of their storage afterwards is the retire: `dead` drops after the settle, and its
`CudaSlice`s free with `cuMemFreeAsync` on the owner stream. The engine context DISABLES cudarc's event tracking
(`crates/memra-engine/src/lib.rs:3759`), so that free is NOT ordered behind a copy-stream read: a copy-stream source
must stay owned by the transfer engine until its completion event is observed, exactly as the KV planes' registered
leases do. That is a requirement of the design below, not a tolerance.

## 2. Pre-registration (committed before any engine code moves)

**Surfaced conflict with the literal item.** "Replace the bundle checksum's heap share by the items' completion
checksums" read literally means the engine's `progress` hashes the landed span bytes, and `progress` runs on the owner
thread: that puts back about 73 ms per 157.9 MB on the target card's tick (the day-28 helper's own figure,
`hashed in 73.4ms`) and undoes ruling 39; on the 5090 with its `for_device` write-combined class it is worse (day 33,
verbatim: `DAY33 HASH-WC VERDICT: 54.8MB cached 11.9 wc 473.8 heap 11.9 wc_copy 224.2 (N=10 each); 160MiB cached 36.6
wc 1450.8 heap 37.2 wc_copy 686.8 (N=10 each); ...`, a size-independent 0.116 GB/s for a write-combined read). So a
span's completion carries the engine's part (its event observed complete, its `valid_bytes` equal to the submitted
bytes, under the batch's ticket), and its checksum is the hash helper's SHA-256 over the landed bytes, one digest that
is both the span's completion checksum and its bundle share. There is no second independent read of the span bytes in
this design. The strong form (a device four-lane digest of each source on the copy stream plus the CPU
`receipt_lanes` oracle over the landed bytes, `crates/memra-tier/src/conformance/d2d_receipt.rs:57`; about +71 ms on the
helper on the target card, +55 ms on the 5090 host, DAY27's digest micro-cell) is named and not built.

**The design (option R, a staging bounce).**

1. Item class. A typed f32 span, `D2hSpan { source: CudaSlice<f32>, destination: PinnedHostBuf }`, attaches to a live
   D2H batch's ticket right after `submit_batch` (`CudaTransfers::submit_d2h_spans`). The copy stream waits on a fresh
   owner-stream event, enqueues `cuMemcpyDtoHAsync` per span with no host wait, and records one event per span. The
   ticket is `producer_done` only when every KV item AND every span event is observed complete; `take_d2h_spans`
   before that refuses `NotReady`; `retire` refuses `Busy` while a span is untaken; a span taken back is marked written
   (the only way its bytes become readable) and returns its source. An unretired ticket's drop forgets its spans (a
   leak, never a free, while a copy may be in flight). New class, so a conformance rule in `memra-tier` first
   (`conformance/d2h_span.rs`, additive and unversioned beside the frozen schedules, `WIRE_VERSION` stays 1), with a
   CPU binding and a native GPU cell.
2. Host memory class: CACHED pinned (`PinnedHostBuf`, `cuMemHostAlloc` flags 0) on both cards, not `for_device`. The
   reason is day 33 above: the helper reads the landed bytes, and a write-combined read is 40x the cached one on the
   5090 (473.8 against 11.9 ms per 54.8 MB); on the target card `for_device` is already cached. This departs from the
   item's "pinned `for_device`" on the 5090 by name.
3. Allocation: fresh pinned memory per demote is refuted by arithmetic (157.9 MB at the BOX3 arena cell's 6.1 GiB/s
   single-allocation rate is about 24 ms, four times today's 6 ms). So a staging set is kept per `HostTierContext`
   (a pool of buffers keyed by length), allocated lazily at the first demote with no zero fill (unreadable until
   landed), handed to the hash helper with the job and returned with the reply. The resident form does not change:
   the helper copies each landed span into a fresh heap `Vec<f32>` (`HostF32::Heap`, the same bind program, the same
   SHA input) and hashes that `Vec`. The staging is not charged to the tier governor's pinned ledger (one image's
   recurrent bytes per context, 157.9 MB on the 27B): owed.
4. The worker. `host_entry_from_device`'s `OffTick` arm moves `dead.conv` / `dead.ssm` into spans after the ticket
   (placeholders `HostF32::Heap(vec![])` stand in the image); the settle takes the spans after the landing and before
   `retire`, puts each source back into `dead` (it frees at the shell's drop, after its event), and hands the landed
   staging with the logits and hidden rows to the helper. The completion lines gain the span count:
   `D2H receipt: ... items=128 (32 KV planes, 96 f32 spans)` on the 27B.
5. Receipt term, before and after. Before: KV = engine hash 1 + bind hash 2 compared; recurrent = none (a synchronous
   owner-stream copy's success) + the helper's bundle share. After: KV unchanged; recurrent = the span's completion
   under the ticket (event observed, `valid_bytes` equal to the submission, the ticket's quarantine on any error) + the
   helper's SHA over the landed bytes as both the span's checksum and the bundle share, its byte count checked against
   the payload at the bind.
6. Fail-closed arms. A span refused before any enqueue (no copy stream, a length or stream mismatch, an unknown,
   retired or already-spanned ticket): the spans come back, the sources return to `dead`, the KV ticket unwinds through
   `host_contract_abort` with a typed `tier D2H spans refused` line, the demote fails, nothing published. An enqueue or
   event error after the first span: the ticket is quarantined (`unknown`), the settle's `SourceQuarantined` arm
   latches the tier, the engine keeps the sources. The helper arms (`hash-helper-gone`, `hash-never-lands`) are
   unchanged; a helper that dies takes the staging with it (the next demote allocates a fresh set).
7. What stays on the owner stream: the D2D capture and restore recurrent copies (device bandwidth, about 0.1 ms, and
   the capture races the next decode; the D2D half is owed), the H2D promote's `engine.htod` of the recurrent state
   (the H2D half, owed), the DFlash draft tail (off the door's surface), the logits and hidden rows (host memcpy), and
   the `MEMRA_KV_HOST_VERIFY` digest.

**Acceptance, stated before running.**

- A1. Identity x4 (default and plain, OFF and ON), failure x2, the fault gate (every cell), twin, hit OFF/ON armed with
  the day-24 census, and the unit cells (the door's GPU cells, the engine's `d2d_` and new `d2h_span` cells, the CPU
  hash cells): ALL GREEN on the target card; on the 5090, identity default ON, fault default and plain, hit OFF/ON.
- A2. The day-26 double-park cell (twenty boots, both orders), the `pre-submit` segment of every demote after the
  third of its ON boot (the steady set; day 29's is 80 samples, median 6.05): the allocation alone, stated as a
  number: 1.0 ms (the 32 KV lease allocations, whose steady cost the day-17 arena pair put at about 0.1 ms, the 64
  registrations, the producer record and the 96 span enqueues and events). Pass: median at most 1.5 ms and max at
  most 3.0 ms over at least 80 samples.
- A3. The completion lines: the D2H receipt line's `items=` is the KV count plus the span count on every demote
  (27B: 32 + 96 = 128, 34 + 96 = 130 with the draft; 9B: 16 + 48 = 64, 18 + 48 = 66). The promote line is unchanged
  (its H2D half is owed).
- A4. DAY28 clauses 1a to 1c read by `day28-reading.py` on the same cell: ALL PASS.
- A5. No new numeric program (the identity gate's byte digests unchanged in kind; the resident form and the bundle
  program are the same) and no new `MEMRA_*` name (`tools/check-flags.sh`).
- Readings (not clauses): demote 1 of a boot adds the staging allocation (predicted at most 75 ms, from 37 to 41);
  demotes 2 and 3 discriminate DAY28's first-touch attribution: at most 3 ms means the day-28 first touch was the
  heap `Vec` pages, now on the helper; 30 to 45 ms means it is the pinned KV lease first touch; the helper's time grows
  by the staging copy (predicted at most 95 ms steady, from 73.4).

## 2a. Amendment to A3, committed before any run

Design item 4 and A3 put the span count into the D2H receipt line's `items=`. That line is parsed as the KV item
count by two readers: `tools/kv-host-contract-fault-gate.sh:314` (`d2h_receipt_items`, the first receipt's
`items=N`, which the partial-reject cell at `:336` to `:349` requires to equal the promote H2D batch's `1 of M items`;
the promote is the H2D half, owed and unchanged) and `research/spill-c-20260919/verify-day15.py:22` (`items=(\d+)
\((\d+) KV planes(, draft)?\)`). Adding the spans there would turn the fault gate's partial-reject cell red for a
reason that is not a defect. So, before any run:

- The D2H receipt line keeps `items=` as the KV item count and gains a suffix after `retired acknowledged`:
  `; 96 f32 spans landed under the ticket and taken back before the retire` (27B; 48 on the 9B). No `items=` in it.
- The demote copy-complete line (`demote copy complete off the tick: ... from submission to completion (...)`) gains
  the batch total after the mode: `items=130 (34 KV, 96 f32 spans);` on the 27B with the draft, `items=128 (32 KV, 96
  f32 spans);` without; 9B `items=66 (18 KV, 48 f32 spans);` and `items=64 (16 KV, 48 f32 spans);`.
- A3 as amended: on every contract demote of an ON arm, the copy-complete line's `items=` equals the KV count plus the
  span count (the pairs above), the receipt line's span suffix names the same span count, and the receipt line's
  `items=` is the KV count (32 or 34 on the 27B, 16 or 18 on the 9B). The promote line is unchanged.

## 3. What landed (the D2H half, in the stated order)

1. **Tier conformance** (`97a9e091f`): `crates/memra-tier/src/conformance/d2h_span.rs`, rules 1 to 5 of the span
   batch (refusal before enqueue is whole; one landing over items AND spans; the receipt term; back exactly once
   before retirement; fail closed), three schedules (`d2h_span_batch`, `d2h_span_enqueue_failure_quarantines`, the red
   arm `d2h_span_taken_on_the_items_landing_fails`) and a CPU binding (`tests/contracts/d2h_span_bindings.rs`,
   3 passed). Additive and unversioned; `WIRE_VERSION` stays 1.
2. **Engine seam** (`97a9e091f`): `PinnedHostBuf::new_unwritten` (cached pinned, no fill, unreadable until landed),
   `enqueue_from_device_f32` (`cuMemcpyDtoHAsync`, no host wait) and `mark_landed` (`crates/memra-engine/src/pinned_host.rs`);
   `D2hSpan`, `CudaTransfers::submit_d2h_spans` (`tier_transfer.rs:1644`: admission first, the owner-stream fence and the
   copy-stream wait, then one enqueue and one event per span; any error from the first enqueue on sets the ticket
   `unknown`) and `take_d2h_spans` (`:1733`); `progress` folds the span events into `producer_done`, `retire` is `Busy`
   while spans are untaken, an unretired entry's drop forgets its spans. The CPU census `d2h_span_rules_are_as_stated`
   (`:3338`) and the native ignored cell `d2h_span_batch_lands_with_its_ticket_on_the_copy_stream` (`:3397`: three spans
   of 3, 5 and 4 MiB, an invalid attach refused whole, a 300 ms copy-stream delay so the KV items land first and the
   take is refused `NotReady`, a second attach `Busy`, the bytes bitwise after the take, the injected second-span
   enqueue fault quarantining the ticket).
3. **Worker** (`fc637d26a`, `crates/memra-server/src/worker.rs`): the per-context staging pool
   (`HostTierContext.staging`, `staging_take` `:9214`, `staging_put` `:9223`, cleared at the latch `:8871`);
   `host_spans_submit` (`:10776`) moves every non-empty `dead.conv` / `dead.ssm` plane into a span after the KV
   ticket, and a refusal hands every span back, returns the sources to `dead`, returns the staging and unwinds the
   ticket through `host_contract_abort` (`tier D2H spans refused: <error> (<n> f32 spans handed back)`); the image
   carries empty heap placeholders for the spanned slots; the settle takes the spans after the landing and before the
   retire (`:10929`), puts each source back into `dead` (it frees with the shell, after its event) and returns the
   staging with the KV planes; the driver attaches each staging buffer to its placeholder payload (`:13041`), the helper
   copies it into the payload's heap `Vec` and hashes that (`:10043`, the same `host_hash_payload_digest` program and the
   same resident form), and the landed path returns the staging to the pool (`:13310`). The two lines as amended in
   section 2a. The CPU census `the_d2h_spans_ride_the_ticket_in_the_stated_order` (`:46098`) and two GPU door cells:
   `option_b_spans_ride_the_ticket_and_land_bitwise` (`:46444`) and `option_b_span_refusal_returns_every_plane_and_keeps_the_tier_on`
   (`:46560`, planes on a foreign stream refused `WrongOwner (4 f32 spans handed back)`, every plane whole, the tier
   on, the next owner-stream demote lands).
4. **CPU census** (the two above) plus the reader `day30-reading.py` and the sitting scripts (`a8d6b1df5`), all
   before any run.

Checks on the code: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and server
`Finished`; the `DOCS_RS=1 --target x86_64-unknown-linux-gnu` pass `Finished`; `check-flags: every runtime MEMRA_* name
resolves against 'docs/FLAGS.md' (no grandfather list)`; `check-conflict-markers: OK`; `git diff --check` clean; zero
em dashes. No new `MEMRA_*` name.

The reader against day 29's receipts (the baseline, same command): `DAY30 A2 pre-submit steady N=80 median=6.05
min=5.92 max=6.61 boots_on=10 demotes_per_boot=[11] ... -> FAIL`; demotes 1 to 3 of a boot `median=37.93`, `42.89`,
`42.27`; `helper hashed_in_ms N=110 median=73.1`; A3 `copy_complete_lines=0 ... -> FAIL` (the day-29 lines carry no
`items=` term).

## 4. The sitting (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day30/box/`)

Every number in sections 4 to 7 comes from a command over the receipt files, banked beside it:
`bash pro-single-day30/counts.sh <root>` (output `pro-single-day30/box/counts.log`, `rtx5090-day30/counts.log`; the
day-29 baseline by the same command, `pro-single-day30/box/baseline-day29-counts.log`), the day-30 reader
(`python3 day30-reading.py --double-park pro-single-day30/box/double-park/ev --spans 96 --kv 32,34`,
`box/reading-day30-doublepark.log`; `--spans 96 --kv 32,34 --a3-root pro-single-day30/box`, `box/reading-day30-a3.log`;
the same reader over day 29, `box/baseline-day29-reading-day30.log`), and the earlier readers unchanged
(`python3 day28-reading.py pro-single-day30/box/double-park/ev`, `python3 day25-double-park-reading.py
pro-single-day30/box/double-park/ev`, `python3 day26-reading.py pro-single-day30/box`, logs `box/reading-day28.log`,
`box/reading-day25.log`, `box/reading-day26.log`).

One binary for every cell, `0f54219982c5c79c3ee94d74b0d1d0274cd21ad21231c33a068f943ee5afdfea` (`bins/memra-server.sha256`),
built on `/root/wt-a` at `a8d6b1df5` (`tree.sha`, `unit/tree.sha`; the three day-30 code and script commits). Three
collector holds on `/tmp/memra-gpu.lock` and the hit gate's own `flock` in one sitting, 19:54:01Z to 20:27:01Z; zero lock
retries (`lock retries: 0 file(s)`); no compute app before or after (`compute apps before/after (data rows): 0/0`);
every `CELL.jsonl` closing row `executed-not-qualified False 0`. Telemetry at 250 ms (`command.gpu.csv`):

| cell | window (Z) | samples | temp | power | memory |
|---|---|---|---|---|---|
| double-park | 19:54:01 to 20:13:22 | 4,633 | 33 to 50 C | 33.0 to 361.9 W | 0 to 17,109 MiB |
| gates | 20:13:23 to 20:22:14 | 2,123 | 37 to 60 C | 87.6 to 506.9 W | 0 to 21,939 MiB |
| hit gate (own flock) | 20:22:15 to 20:23:45 | (no collector) | | | |
| unit-cell | 20:23:45 to 20:27:01 | 781 | 33 to 39 C | 31.8 to 97.7 W | 0 to 881 MiB |

The double-park cell is day 26's `pro-single-day26/double-park.sh` byte for byte (one hold, twenty boots, N=5 per arm
per order, both orders): `replays.log STALL REPLAY: PASS lines: 20`, `20 errors=0`, `DAY26 DOUBLE-PARK ADMISSIBLE
all_receipts=True`.

## 5. The reading, verbatim

**A2, the pre-submit census** (`box/reading-day30-doublepark.log`):

- `DAY30 A2 pre-submit steady N=80 median=0.62 min=0.58 max=1.10 boots_on=10 demotes_per_boot=[11] rule N>=80
  median<=1.5 max<=3.0 -> PASS`
- `DAY30 READING pre-submit demote 1 of its boot N=10 median=29.69 min=29.38 max=30.00`
- `DAY30 READING pre-submit demote 2 of its boot N=10 median=1.26 min=1.23 max=1.67`
- `DAY30 READING pre-submit demote 3 of its boot N=10 median=1.27 min=1.22 max=1.72`
- `DAY30 READING helper hashed_in_ms N=110 median=79.9 min=79.5 max=107.7; copy submission-to-completion N=110
  median=92.5`
- Day 29, the same reader (`box/baseline-day29-reading-day30.log`): `DAY30 A2 pre-submit steady N=80 median=6.05
  min=5.92 max=6.61 ... -> FAIL`; demotes 1 to 3 `median=37.93`, `42.89`, `42.27`; `helper hashed_in_ms N=110
  median=73.1`.
- By demote index of the boot (`counts.log`, readings): day 30 `helper: demote 1 N=10 median=106.20; demote 2 N=10
  median=105.45; demote 3 N=10 median=105.80; steady (4+) N=80 median=79.80`, `copy: demote 1 N=10 median=120.60; demote
  2 N=10 median=93.00; demote 3 N=10 median=92.15; steady (4+) N=80 median=92.35`, `wall: demote 1 N=10 median=241.20;
  demote 2 N=10 median=208.15; demote 3 N=10 median=207.60; steady (4+) N=80 median=179.00`. Day 29: `helper: ... steady
  (4+) N=80 median=73.10` (73.10 to 73.15 at every index), `copy: demote 1 N=10 median=39.40; demote 2 N=10
  median=135.30; demote 3 N=10 median=133.05; steady (4+) N=80 median=97.35`, `wall: ... steady (4+) N=80
  median=183.80`.

**A3, the completion lines** (`box/reading-day30-a3.log`, over every server log of the sitting):

- `DAY30 A3 logs=92 copy_complete_lines=132 receipts_paired=132 receipts_without_copy_line(on-tick)=0 bad=0 rule
  items==KV+spans, spans=96, KV in [32, 34], receipt items==KV and suffix==spans -> PASS`
- `counts.log`: `112 items=128 (32 KV, 96 f32 spans)`, `20 items=130 (34 KV, 96 f32 spans)`; the receipt lines
  `112 items=32 (16 KV planes)`, `20 items=34 (16 KV planes, draft)`; `132 ; 96 f32 spans landed under the ticket and
  taken back before the retire`. Submissions `112 ... 64 tokens, 159.8MB, 32 items`, `21 ... 64 tokens, 159.9MB, 34
  items`: the one submission without a copy-complete line is the injected receipt refusal
  (`gates/contract-fault/postpublish-server.log submitted=2 copy_complete=1`, `demote failed (tier D2H receipt refused:
  injected failure (MEMRA_KV_HOST_FAULT=contract-postpublish)); nothing demoted`). Span refusals in the sitting: `0`.

**A4, DAY28 clauses 1a to 1c** (`box/reading-day28.log`):

- `DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=76.9 off_cell_median=85.2 rule
  on<=off+2.0 -> PASS`; `order=o2 ... on_cell_median=76.9 off_cell_median=85.5 ... -> PASS`
- `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=127.5 off=115.5 on_minus_off=+12.0 rule <=+20.0 ->
  PASS`; `order=o2 ... on=127.2 off=115.6 on_minus_off=+11.7 ... -> PASS`
- `DAY28 CLAUSE 1c owner in-completion N=100 median=1.96 min=1.83 max=3.03 runs_with_demote_without_ledger=0 rule
  <=12.0 -> PASS`
- `DAY28 REPORTED wall in-completion median=86.7 (N=100); demote_completion median=92.5; helper hashed_in_ms
  median=79.8 min=79.5 max=106.6; payloads=[98] mb=[157.9]; hash_polls median=6 max=8; settle modes={'tick-top poll':
  100}; tenant top gaps: largest median=90.3 second median=19.0`
- `DAY28 VERDICT clauses_failed=0 -> ALL PASS`

**Day 25's reader** (`box/reading-day25.log`): `DAY25 DOUBLE-PARK stall order=o1 on_minus_off=-8.3 unc=0.2 ->
isolated (on 76.9, off 85.2)`; `stall order=o2 on_minus_off=-8.6 unc=0.1 -> isolated (on 76.9, off 85.5)`; `e2e
order=o1 on_minus_off=+12.0 unc=1.1 -> isolated (on 127.5, off 115.5)`; `e2e order=o2 on_minus_off=+11.7 unc=1.0 ->
isolated (on 127.2, off 115.6)`; `DECOMPOSITION arm=on N_runs=100 parked_per_run=[1] ... promote_completion
median=19.6 promote_in median=20.6 ... demote_in median=179.1 ... idle_p50(tick)=13.47 ... | tenant top gaps: largest
median=90.3 ...`; `arm=off N_runs=100 parked_per_run=[0] promote_in median=10.8 demote_in median=6.2 | tenant top gaps:
largest median=98.8 second median=16.7`.

**Day 26's reader** (its own day-26 clauses, not today's acceptance; `box/reading-day26.log`): `DAY26 CLAUSE 1 arm=on
N_runs=100 ... runs_with_parked_1_submitted_0_not_routed_1=100 -> PASS`; `DAY26 CLAUSE 2 e2e order=o1 on_median=127.5
off_median=115.5 on_minus_off=+12.0 unc=1.1 expected=+15.8 (day 25: +105.85 minus 90.1) |d-expected|=3.8 -> FAIL`;
`order=o2 ... on_minus_off=+11.7 unc=1.0 expected=+15.8 ... |d-expected|=4.1 -> FAIL`; `CLAUSE 3 ... -> FINDING`
both orders; clauses 4 and 5 `NO RECEIPT` / `NO VERDICT LINE` (the restore arm is not part of this cell and the reader
looks for the hit gate under a `cells/` layout this sitting does not use; the hit-gate verdicts are below).

## 6. Verdicts against the acceptance of section 2

**A1, every arm on the target card** (`counts.log`, verdict lines per gate log, verbatim):

- `identity-default-off`, `identity-default-on`, `identity-plain-off`, `identity-plain-on`: `KV-HOST-SPILL IDENTITY
  GATE: ALL GREEN (teeth=0)` (12 ok each).
- `failure-off`, `failure-on`: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok each).
- `contract-fault`: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL): `presubmit` 11, `postpublish` 11,
  `promote-presubmit` 11, `promote-postpublish` 11, `promote-readyview` 11, `promote-reject` 14, `d2d-capture` 12,
  `d2d-restore` 14, `hash-helper-gone` 14, `hash-never-lands` 14; both hash cells `hand-off ticket(s) ['3'], refusal
  ticket(s) ['3']` and `the tier latched off exactly once`.
- `twin-off`, `twin-on`: `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8
  cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0
  effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok ->
  PASS`.
- `hitgate-off`: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok; `armed=0 door_on=0`). `hitgate-on`:
  `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok), the census equal to day 24's: `armed=1 door_on=1
  capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0
  promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0` (spec-on), `capture_submitted=2
  capture_published=2 restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30 route submission(s) across the
  two boots (capture, restore, demote or promote off the tick)`.
- Unit cells on the box (`unit/`): server `option_b_`/`option_c_` `test result: ok. 10 passed; 0 failed` (day 29's 8
  plus `option_b_spans_ride_the_ticket_and_land_bitwise` and `option_b_span_refusal_returns_every_plane_and_keeps_the_tier_on`);
  engine `ok. 6 passed; 0 failed` (the five `d2d_` cells and `d2h_span_batch_lands_with_its_ticket_on_the_copy_stream`);
  CPU hash cells `ok. 11 passed; 0 failed` (with `the_d2h_spans_ride_the_ticket_in_the_stated_order` and
  `hash_helper_digests_equal_the_owner_thread_digests_bitwise`); tier contracts `ok. 3 passed; 0 failed`.
- The day-28 double-park cell: `DAY26 DOUBLE-PARK ADMISSIBLE all_receipts=True`, 20 of 20 replays `PASS`.

**A1 PASSES on the target card.**

**A2 PASSES.** Steady pre-submit **6.05 ms (N=80, 5.92 to 6.61) before, 0.62 ms (N=80, 0.58 to 1.10) after**, against
the predicted 1.0 ms and the rule `median<=1.5 max<=3.0`.

**A3 PASSES** on both cards (the 5090's in section 7).

**A4 PASSES**: 1a 76.9 against 85.2 and 76.9 against 85.5; 1b +12.0 and +11.7; 1c **1.96** (day 29: 7.39, max 45.18;
today max 3.03). `DAY28 VERDICT clauses_failed=0 -> ALL PASS`.

**A5 PASSES**: the identity gate's four arms green with `teeth=0` and the MEMRA_KV_HOST_VERIFY round trip matched in
the verify arms (the resident form and the bundle program are unchanged: the helper copies the landed staging into the
same heap `Vec` and hashes it with the same `host_hash_payload_digest`); `bash tools/check-flags.sh` clean, no new
`MEMRA_*` name.

**Readings.** Demote 1 of a boot 29.69 ms (predicted at most 75; day 29: 37.93): the staging allocation, 157.9 MB of
cached pinned memory in one lazy set per context. **Demotes 2 and 3 read 1.26 and 1.27 ms (day 29: 42.89 and 42.27):
inside the pre-registered "at most 3 ms" branch, so DAY28's first-touch cost was the heap `Vec` pages the synchronous
`clone_dtoh` wrote, not the pinned KV lease.** It moved with the pages to the helper: the helper reads 106.20, 105.45,
105.80 ms on demotes 1 to 3 and 79.80 ms at steady state (day 29: 73.1 at every index), so steady helper time grew by
6.7 ms for the staging copy (predicted at most 95). The copy's submission-to-completion is 92.35 ms steady (day 29:
97.35), one poll, tick-quantized: the copy now carries the 157.9 MB the owner thread used to pull, and still lands
inside one tenant tick. Wall t0 to publication steady 179.00 ms (day 29: 183.80).

## 7. Local RTX 5090 (`rtx5090-day30/`)

`battery-5090.sh` with the tree's release binary (`build.log` `rc=0` under the CPU quota, `binary.sha256`
`e9263537139c020dd8bb3ab6a0cc1f3c641acc106fd90313c4fe480fe760b66e`) and the 9B NVFP4 MTP artifact; each cell after a
bounded idle wait (15 x 120 s, no compute app and at least 20000 MiB free) and under the gate's own `flock
/tmp/memra-5090.lock`. Lane C's day-37 servers held the card for the first pass's first cell: `2026-09-22T20:24:48Z
identity-default-on NOT RUN: the card never freed in 15 waits` (each wait logged with nvidia-smi's own listing; the
holder was never inspected beyond it and never signalled). The card freed at wait 5 of the next cell. RTX 5090 Laptop
GPU; card before and after every cell 15 MiB used, 55 to 67 C, 9.9 to 43.0 W (`card.before.csv`, `card.after.csv`).
Verbatim (`battery.log`, each `gate.log`):

- `fault-default` (20:32:49Z to 20:34:18Z): `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL).
- `fault-plain` (20:34:18Z to 20:35:35Z): `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL).
- `hit-off` (20:35:35Z to 20:35:57Z): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok; `armed=0 door_on=0`).
- `hit-on` (20:35:57Z to 20:36:23Z): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok), census `armed=1 door_on=1
  capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0
  promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0` (spec-on), `capture_submitted=2
  capture_published=2 restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30 route submission(s) across the
  two boots`: equal to day 24's.
- `identity-default-on-rerun` (20:37:36Z to 20:37:51Z; the same gate, env and binary through the script's
  `identity-rerun` arm, added after the first pass; the tree under the gate script is `39378d6e4`, which moved no
  `crates/` or `tools/` file since `a8d6b1df5`): `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok); one hit
  parked on the `Hashing` entry (`... (ticket seq=3, 50 payloads, 53.7MB on the hash helper for 0.4ms) ...`), r3 and r4
  `prompt=102 cached=64 lcp=64`.
- A3 on this card (`rtx5090-day30/reading-day30-a3.log`, `--spans 48 --kv 16,18 --a3-root rtx5090-day30`): `DAY30 A3
  logs=35 copy_complete_lines=34 receipts_paired=34 receipts_without_copy_line(on-tick)=0 bad=0 rule items==KV+spans,
  spans=48, KV in [16, 18], receipt items==KV and suffix==spans -> PASS`; `16 items=64 (16 KV, 48 f32 spans)`, `18
  items=66 (18 KV, 48 f32 spans)`.
- The demote ledger lines per cell dir (`counts.log`, section `demote ledger lines per top-level cell dir`; day 29 by
  the same command, `baseline-day29-counts.log`), recorded as this card's gate traffic, not a cost cell:
  `fault-default: N=14 pre-submit median=3.77 min=0.44 max=18.62; helper median=30.35 ...; owner in-completion
  median=13.21`, `fault-plain: N=14 pre-submit median=2.17 ...; helper median=30.00 ...; owner in-completion
  median=10.57`, `identity-default-on-rerun: N=2 pre-submit median=43.91 min=37.53 max=50.28` (the verify arm, finding
  5); day 29 `fault-default-rerun2: N=14 pre-submit median=20.38 ...; helper median=12.90 ...; owner in-completion
  median=29.82`, `fault-plain-rerun2: N=14 pre-submit median=20.77 ...; helper median=12.90 ...; owner in-completion
  median=29.73`. The identity rerun's parked hit re-parked 13 times across the 30.7 ms hash (day 29: 5 across 12.9).

**A1 on the 5090 (identity default ON, fault default and plain, hit OFF/ON): ALL GREEN.** No number from this card is
compared to the target card's.

## 8. Findings

1. **The pre-submit segment is the allocation now.** 6.05 to 0.62 ms steady on the target card: the synchronous
   owner-stream `clone_dtoh` of 96 recurrent planes is off the owner thread. The owner thread's whole demote
   (`owner in-completion`) is 1.96 ms median, max 3.03.
2. **DAY28's first-touch attribution is settled**: the heap `Vec` pages (section 6, readings). The helper now pays that
   first touch (about +26 ms on demotes 1 to 3 of a boot) off the tick.
3. **The e2e cost of the double-park moved in the good direction**: +16.8 / +16.9 (day 29) to +12.0 / +11.7, and the
   stall ON against OFF from -3.6 / -3.3 to -8.3 / -8.6. Day 26's reader reads `|d-expected|=3.8 / 4.1 -> FAIL` against
   its day-26 expectation of +15.8; that is a day-26 clause, stated, not tuned, and not part of today's acceptance.
4. **A refused receipt frees the staging.** After the injected `contract-postpublish` refusal, the boot's next demote
   re-allocated its staging (`pre-submit 28.71`, `gates/contract-fault/postpublish-server.log`): the abort after the
   take drops the landed staging instead of returning it to the pool. Fail-closed and correct; the cost is one
   allocation after a refusal. Returning it on that path is a small owed item, not a defect.
5. **The verify arm keeps its owner-stream digest.** The identity gate's default ON arm (MEMRA_KV_HOST_VERIFY on)
   reads `pre-submit 140.78` and `111.58` on the target card (day 29: 148.52, 151.21) and `50.28`, `37.53` on the 5090
   (day 29: 51.56, 53.55): `host_roundtrip_digest` runs after the demote's `t0` and before the submission, reads the
   entry's device state with owner-stream copies and hashes it on the owner thread (section 2 item 7). It is a
   diagnostic arm, not the door's serving route, and it stays as it was.

## 9. What is owed, and integrability

Owed on Move 2 item 1 after today: **the H2D half** (the promote's recurrent `engine.htod` onto the copy stream under
the ticket), **the D2D half** (capture and restore recurrent copies), **the governor charge of the staging** (one
image's recurrent bytes per context, 157.9 MB on the 27B, outside the tier governor's pinned ledger today),
**the strong-form receipt** (the device four-lane digest of each span source plus the CPU oracle over the landed
bytes; about +71 ms on the target card's helper), **a span-refusal cell in the fault gate** (the refusal is covered by
the GPU unit cell `option_b_span_refusal_returns_every_plane_and_keeps_the_tier_on`, not by a serving-shape fault
cell), and returning the staging to the pool on the post-take abort (finding 4).

**Integrability**: the D2H half meets its pre-registered acceptance on the target card (A1 to A5, one sitting) and the
5090 half is ALL GREEN (identity default ON on the rerun). It is integrable as a complete D2H half; the H2D and D2D
halves remain owed and nothing here claims them. Every cell `executed-not-qualified`; no qualification claimed.

## 10. Checks, budget, cleanup

On the records commit: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and
server `Finished` (`rc=0`); the GPU-less `DOCS_RS=1 --target x86_64-unknown-linux-gnu` clippy pass `Finished`
(`rc=0`), both under the CPU quota (no Rust moved after `fc637d26a`); `check-flags: every runtime MEMRA_* name resolves
against 'docs/FLAGS.md' (no grandfather list)`; `check-conflict-markers: OK`; `git diff --check` clean;
`.gitattributes` (`*.log -whitespace`, `SUMMARY.txt -whitespace`) in `pro-single-day30/box/` and `rtx5090-day30/`; zero
em dashes in the day's own lines. Every count in sections 4 to 7 and in the door's rows comes from `counts.sh`, the
day-30 reader or the earlier readers named in section 4, run over the banked files. Every push in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode, logged to the gate-skips ledger; no qualification claimed.
Budget: about 2 agent-hours of 5 from the restart to the records commit. Box: `/root/wt-a` at `a8d6b1df5` on
`lane-a-day30`, clean; `/root/spill-receipts/a-day30/` mirrored to `pro-single-day30/box/` (the binary excluded, its
sha256 kept); the bundles that carried the tree removed on both ends; no server or process of mine left running;
nothing of other lanes touched. Local: the release build and the battery under the CPU quota; the 5090 held only
through the gates' own `flock`, after the bounded idle wait; the staged binary directory removed; no scratch of this
run left in `/tmp`.
