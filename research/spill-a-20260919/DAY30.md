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
