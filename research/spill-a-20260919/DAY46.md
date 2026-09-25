# WP-A day 46: OWED item 4 again, design S3 (S2's revision, DAY42 section 3)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. S2 (`7ce3f3243`) failed (c) on the target card and is reverted
(`16904e97d`; DAY42 section 3). Every cell `executed-not-qualified`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. Pre-registration (committed before any S3 code)

**What S2 taught** (DAY42 section 3). Its correctness held on the target card (every gate green, the span-flip cells
among them, the batched kernel bitwise) and its shape held: the receipt left the landing path (the copy settle moved
0.03 ms), (d) passed (PIN +0.20 / +0.10, e2e +0.33 / +0.30) and (e) passed (+0.027). Its price did not: (c)'s e2e rose
+3.16 / +3.06 ms against +1.0. The trace placed it: each `span_receipt_digests` launch ran a grid of up to 192 x 64
blocks (12288), and during the landed launches (1.05 and 1.9 ms, PCIe-bound reads of 157 MB of pinned host memory) the
owner stream ran **no kernel at all** (`owner busy in=0.00ms n=0 | before busy=1.84ms n=185`). A grid that fills every
SM with blocks waiting on host memory holds the owner's kernels until it drains.

**Design S3: S2, whole (sections 1 and 1a of DAY42), with the span digests' grid bounded.**

1. **At most one block per SM per launch.** `CudaTransfers` reads the card's SM count once at construction
   (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`). A launch of `span_receipt_digests` over `k` spans (k at most 64) runs a
   grid of `(B, k)` with `B = max(1, floor(SMs / k))`, capped by the largest span's need (`ceil(words / 2048)`), so a
   launch holds at most `max(SMs, k)` blocks of 256 threads, one per SM when `k <= SMs`: every SM keeps room for the
   owner's blocks. (188 SMs and 64 spans: B = 2, 128 blocks; 32 spans: B = 5, 160 blocks.)
2. **The landed digests (pinned host memory) take `B = 1`.** PCIe-bound reads need outstanding loads, not SMs: 64
   blocks per launch.
3. **Nothing else moves.** The program is the same wrapping four-lane sum per span (grid-stride, order-independent), so
   the batched kernel's bitwise cell (1 B to 3 MiB + 3, offsets 0 to 7, device and pinned host memory) holds with any
   grid; the take, the seal, the guard, the `Hashing` step's order, the promote's check, the red arms and the tier rule
   are S2's. The census `span_receipt_rules_are_as_stated` adds the grid rule (`B` from the SM count, `B = 1` for the
   landed launch).

**Acceptance, stated before any code** (S2's, whole; each card its own; the bounds are S's, unchanged):

- (a) Semantics: S2's (a), with the bitwise cell re-run at the bounded grids (the cell calls the launch helper at both
  `B` rules).
- (b) The gate set ALL GREEN (identity x4, failure OFF and ON, the fault gate default and plain with every cell, twin
  OFF and ON on the target card, hit OFF and ON) and the unit cells.
- (c) Demote price: S3 against G4, per order the steady demotes' wall median at most +8.0 ms and the demoting
  intruder's e2e median at most +1.0 ms (`day42-reading.py`, the arm directory `s2` holding S3's boots, the reader
  unchanged).
- (d) Promote price: S3 against G4, per order PIN at most +1.0 ms and e2e at most +1.0 ms.
- (e) The hump clause on S3 (two boots, the G'' control beside them, 16 demote runs each): median HUMP at most 0.15 ms,
  each boot's start temperature and SM clock recorded.
- Readings, no clause: `day42-trace-reading.py` on one traced boot per arm, and `owner-during-landed.py`: the owner's
  kernel time during each landed launch against the same-length window before it (S2 read 0.00 against 1.84 ms).

**Predictions.** The landed launches run about 3 to 6 ms (64 blocks over PCIe) beside the owner's kernels; the owner's
busy time inside them is at least half of the window before; (c)'s e2e within +0.5 ms; the wall within +2 ms (the
receipt lands before the helper's reply, about 85 ms); (d) PIN within +0.6 ms (the H2D destination digests take longer at
B = 2 on the landing path, about 0.3 ms); (e) flat.

**What each card decides.** Each card its own (a) to (e). The target card runs first (the 5090 needs the owner's reset);
the 5090 half runs when it is back.

**Sittings.** Target: `pro-single-s2/` unchanged in shape with S3's tip as the s2 arm (build.sh <S3 tip> b4816eda8), on
the box after its current sittings (item 15 and item 3's reading), the driver in full. 5090: `rtx5090-day42/`
(build.sh, card-run.sh, trace.sh) with S3's tip as the s2 arm.

**Budget.** 0.3 agent-day: the change and its censuses 0.1, the CPU cells 0.05, the target sitting (about 70 minutes of
card time) 0.15.

## 2. S3 as built (`e776b2843`) and its CPU cells

- Built: S2's code re-applied (the revert `16904e97d` reverted, code only) with `span_blocks` (the grid rule of section
  1, `SpanMemory::{Device, PinnedHost}`), `CudaTransfers.sm_count` (read at construction; a failed read is a
  construction refusal), the three launch sites passing their memory kind (the sources and the H2D destinations
  `Device`, the landed digests `PinnedHost`). The grid rule's census and values are their own unit test,
  `day46_span_digest_grids_leave_room_for_the_owner` (at 188 SMs: 64 spans take 2 blocks each, 32 take 5; at 82 SMs, 64
  take 1; pinned host always 1), rather than a clause inside `span_receipt_rules_are_as_stated`. The native kernel cell
  runs every base at its rule and device memory at one block per span too.
- CPU cells, green: engine lib `553 passed; 0 failed; 44 ignored`; server lib `912 passed; 0 failed; 24 ignored`; the
  tier crate; clippy `-D warnings` on the three crates; fmt; `git diff --check`; `tools/check-flags.sh`.
- The target sitting runs on the same box after item 15's (`pro-single-day42/run-all-2.sh`, S2's receipts moved to
  `a-s2-design-s2` first, S3's under `a-s2`), then item 3's 9950X-class reading again (the lead's order puts item 4
  first). The 5090 half waits for the card's reset.

## 3. S3 on the target card, as it ran (the same box; `pro-single-day42/run-all-2.sh`; receipts under `a-s2`, banked in `pro-single-s2/box-readings-s3/`)

- Build `s3 build rc=0` 05:18:53Z: s2 (S3's tip) `e67f46dfa020e3fd..`, g4 `5845c0e59f3db51b..`, gpp `7508d56dc8509ef3..`
  (both rebuilt in this run, their trees `b4816eda8` and `358749c9f` as before).
- **(c), verbatim** (`ab-demote rc=0` 05:46:17Z, 20 of 20 replays): `DAY42 S2 C order=o1 wall g4=101.40 s2=104.70
  s2-minus-g4=+3.30 rule <=+8.0 | e2e g4=176.02 s2=179.36 s2-minus-g4=+3.34 rule <=+1.0 -> FAIL`; `order=o2 wall
  g4=101.50 s2=104.75 .. +3.25 .. | e2e g4=176.20 s2=179.47 .. +3.27 .. -> FAIL`; **`DAY42 S2 DEMOTE -> FAIL`**.
- (d), verbatim (06:03:36Z): `DAY42 S2 D order=o1 .. -> PASS`, `order=o2 pin g4=13.50 s2=13.60 s2-minus-g4=+0.10 rule
  <=+1.0 | e2e g4=102.30 s2=102.53 s2-minus-g4=+0.23 rule <=+1.0 -> PASS`; `DAY42 S2 PROMOTE -> PASS`.
- (e), verbatim (06:08:43Z): `HUMP arm=xs2 boots=2 median-hump=+0.012 humps=False`, the control `HUMP arm=xgpp boots=2
  median-hump=+0.515 humps=True`: PASS.
- (a) and (b): every gate `.exit` 0 (identity x4, failure x2, the fault gate default and plain, twin x2, the hit gate x2),
  the unit cells `parallel=3/3 engine-serial-rc=0 door-rc=0 cpu-rc=0 engine-census-rc=0 tier-rc=0`.
- The trace reading: `TRACE steady N=6 copies_wall_ms=4.22 .. landed_wall_ms=3.15 landed_sum_ms=3.14 seal_delay_ms=7.46`
  (g4: `copies_wall_ms=3.30`). At one block per span the landed digests take as long as S2's full grid did.

**Where the price sits (a reading of both sittings' receipts, after the result).** The price is not the grid. Each
demote's copy settle and the intruder's seed capture settle land at the same tick top, the demote first; the capture's
settle takes its destination planes back (`take_plane`), and `take_plane` and `release_device` drain the WHOLE copy
stream (`synchronize_copy_stream`), which now holds the landed digests the seal just enqueued. The capture settle's
line reads `the settle held the owner thread 3.23ms .. 3.26ms` on S3's boots (S2's 3.09 to 3.15 in its trace boot) against
0.13 to 0.21 ms on g4's. S2's trace shows it directly: `cuStreamSynchronize dur=2.908` starting 0.066 ms after the landed
launch, then 31 more take-backs' syncs at its end; the owner stream ran no kernel during the landed launches because the
owner thread was waiting in that call, not because the grid filled the card. DAY46 section 1's premise ("a grid that
fills every SM ... holds the owner's kernels") was wrong; the S2 reading that suggested it is corrected here.

**Verdict, as registered: S3 FAILS (c) on the target card and is refuted.** It is reverted in one commit with these
receipts (the code returns to `b4816eda8`'s, G4 and T); its 5090 sitting is cancelled; the revision (the release paths'
copy-stream drain made precise, so a take-back waits for nothing but its own lease's work) is pre-registered in
`DAY48.md` before its code.
