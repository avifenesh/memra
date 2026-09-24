# WP-A day 36: the D2D half's restore price cell (DAY33 section 6)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, tip `50644beeb` (integ57, PR #711, its files not touched here).
Rig: the local RTX 5090 Laptop GPU; no target card is rented. Every cell `executed-not-qualified`. Every engine push in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode.

## 0. Resync

`git fetch`: the lane equals origin at `50644beeb`; `origin/main` `fa73d0e6c`; #711 is OPEN (in CI). Main is merged
before the first push after #711 lands, or before the day's last push.

## 1. Pre-registration (committed before any day-36 code)

**The cell is DAY33 section 6's, unchanged.** A log-only field, `recurrent copy H.HHms host, G.GGms owner stream`: the
host time around the restore's recurrent copy loop (`host_restore_submit`, step 1, the `copy_into` of every layer's
conv and ssm state and its `len` mirror, on the owner stream), and the owner-stream GPU time between two timing events
recorded on the owner stream around that loop. Read on the restore arm.

**The instrument, as it will be built.** The host time goes on the `restore submitted off the tick` line (`; recurrent
copy H.HHms host`). The GPU time is read without any host wait: the two events are kept with the pending restore, and
the `restore landed off the tick` line (at the request's re-admission, one tick or more later) reads their elapsed time
only if the end event is already complete (`; recurrent copy G.GGms owner stream`, else `; recurrent copy owner stream
pending`, counted). One named departure from DAY33's wording: the field splits over the two lines, because reading the
GPU time on the submitted line would need the owner thread to wait for the owner stream to drain, which is a change to
the thing being priced. An event that cannot be created prints `n/a` and never fails the restore. No new `MEMRA_*`
name, no dispatch change, no numeric program change.

**The run.** The day-26 restore arm (`pro-single-day26/restore-arm.sh`) byte for byte in its boot and harness:
`MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192
MEMRA_MAX_SESSIONS=4`, one door-ON boot, `stall_cell.py --mode restore --n 50` (100 timed restores), the receipt
replayed. On the 5090 with the 9B NVFP4 MTP artifact under one bounded hold of `/tmp/memra-5090.lock` (60 x 120 s,
never inside another lane's hold; no compute app and at least 20000 MiB free, bounded 15 x 60 s; a foreign compute app
is handled as on day 35: a bounded 60-minute wait outside the lock, and any amendment pre-registered before a boot).
250 ms card telemetry. Receipts in `rtx5090-day36/`.

**The reading.** Over every restore of the boot that carries the field: the host median and the owner-stream GPU median
(and min, max, N), the first restore of the boot also shown alone; the count of `pending` and `n/a`.

**The decision rule, DAY33's, verbatim:** "if on the target card the owner-stream GPU median is under 0.5 ms and the host
median under 0.5 ms per restore, the D2D half closes as not worth a door (the verdict and the receipt in
`OWNER-THREAD-OFFLOAD.md` and the verdicts ledger; no code). Otherwise a restore-recurrent design is pre-registered with
its own acceptance."

**What the 5090 reading can and cannot decide, stated now.** The rule is conditioned on the target card (RTX PRO 6000,
the 27B: 96 recurrent planes, 156.9 MB). The 5090 runs the 9B (48 planes, 52.7 MB) on a different memory system
and CPU. So the 5090 cell is a reading and decides neither branch. The verdict line reads `DAY36 PRICE READING (5090,
not the rule's card)`. Closing the half, or opening a design, needs either the target-card sitting of the same cell or
the lead's ruling that the 5090 reading decides. That choice is the lead's; this lane does not relax the clause.

**Predictions.** On the 9B: 24 recurrent layers (48 `copy_into`, 52.7 MB, the day-35 receipts' `conv 24 .. ssm 24`)
and 8 KV layers (8 `set_i32_one`). GPU: 52.7 MB read plus 52.7 MB written at about 0.8 to 0.9 TB/s effective on the
laptop card, about 0.12 ms. Host: 56 calls at a few microseconds each, about 0.2 to 0.4 ms.

**Budget.** 0.25 agent-day. The card is taken by bounded waits behind the other lanes.

## 2. The 5090 reading (`rtx5090-day36/`)

- The instrument `d4f53945f` (census `day36_the_restore_recurrent_copy_is_timed_without_a_wait`; server lib 894 passed,
  clippy and the `DOCS_RS=1` pass clean), binary `7d761ec1c7a00ac2..`; the cell and reader `0da5638f5`.
- One hold 06:38:56Z to 06:43:14Z, no compute app at the start or the end; one door-ON boot, 100 timed restores,
  `STALL REPLAY: PASS`. Card telemetry (`card-250ms.csv`): 1029 samples, 54 to 87 C, 9.1 to 174.1 W.
- Verbatim (`reading-day36.log`): `DAY36 READING submitted=100 landed=100 pending=0 n/a=0 replay_pass=True`;
  `recurrent-copy-host-ms N=100 median=0.130 min=0.110 max=0.200 first=0.120`; `recurrent-copy-owner-stream-ms N=100
  median=0.170 min=0.160 max=0.200 first=0.160`; `DAY36 PRICE READING (5090, not the rule's card) host median=0.130
  owner-stream median=0.170 (the rule's bound 0.5 each, read on the target card only)`.
- As registered, this decides neither branch. The prediction held for the GPU time (0.12 predicted, 0.17 read: about
  0.62 TB/s effective over 2 x 52.7 MB) and came in under it for the host (0.2 to 0.4 predicted, 0.13 read: about 2.3
  us per call over 56 calls).

## 3. The combined target-card sitting, pre-registered before it runs (the lead's ruling, option (1))

One rented RTX PRO 6000 Blackwell Workstation Edition (600 W), passed through the lead's acceptance, the 27B NVFP4 MTP
artifact staged by the lead (its sha256 banked), access only through the wrapper the lead sends, `/tmp/memra-gpu.lock`.
Nothing on this card is compared with any other card. Scripts: `pro-single-day36/` (committed with this section).

**Binaries, built on the box in this lane's own clone:** base `a0f915d8a` (F settled, the day-35 base; its code is
the code M' was measured against on the 5090), M' `55ae87616`, the tip `d4f53945f` (M', the day-36 instrument, main at
`fa73d0e6c`). The test binaries are built on the tip before any hold.

**The cells, in one sitting, the A/B and the price cell in one collector hold:**

1. **The price cell.** The day-26 restore arm (`pro-single-day26/restore-arm.sh`) byte for byte on the tip binary: one
   door-ON boot, `stall_cell.py --mode restore --n 50`, the receipt replayed. Read by `day36-reading.py --target`.
   **DAY33 section 6's rule, verbatim:** "if on the target card the owner-stream GPU median is under 0.5 ms and the host
   median under 0.5 ms per restore, the D2D half closes as not worth a door (the verdict and the receipt in
   `OWNER-THREAD-OFFLOAD.md` and the verdicts ledger; no code). Otherwise a restore-recurrent design is pre-registered
   with its own acceptance." If the rule says a design is owed, this lane stops at that design's pre-registration.
2. **M''s target-card A/B** (DAY35 section 7's cell with N as registered): `stall_cell.py --mode demote --n 5`, base
   against M', o1 = `base m` five times, o2 = `m base` five times, door ON, 20 boots. The card's boot environment is the
   PRO cells' (the day-17 and day-26 PRO demote environment): `MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192
   MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4` (the 5090's 64 MB prefix budget
   is the 9B's; a 27B entry is about 160 MB), each server's readiness bounded to 480 s (the PRO scripts' bound; the 27B
   loads slower than the 60 s the 5090 script allows). Read by `day35m-reading.py --m2`, section 7's (c) and (d)
   verbatim: (c) `take-back bind and publish` median at most 1.5 ms and max at most 3.0 ms over at least 20 steady
   demotes on M''s boots; (d) per order, the wall median on M' at most the base's plus 17.0 ms and the demoting
   intruder's e2e median at most the base's plus 1.0 ms.
3. **The gates on the tip binary** (M''s (a) and (b) on this card): identity x4, failure OFF and ON (the `digest` cell
   on the ON arm is (a)'s served-path cell), the fault gate default and plain (every cell), twin OFF and ON, hit OFF and
   ON with the day-24 census; the unit cells on the tip (the door's GPU cells including
   `option_b_off_tick_demote_hashes_ride_the_helper_and_a_changed_lease_is_refused`, the engine's D2D, D2H and H2D cells,
   the CPU censuses including `day35_` and `day36_`): ALL GREEN.

A compute app that is not memra's at a hold's start is handled as on day 35 (a bounded wait; any amendment
pre-registered before a boot). An incomplete cell is repeated whole once in a new hold; nothing is read from a partial
one. Order: build (outside any hold), the price cell and the A/B in one hold, the gates, the hit gate, the unit cells.

**Expected numbers, stated before the sitting.**

- The price cell: owner-stream GPU median 0.2 to 0.3 ms (2 x 156.9 MB at 1.1 to 1.5 TB/s effective); host median 0.25
  to 0.5 ms (112 calls: 96 recurrent `copy_into` and 16 `set_i32_one`, at this rig's 2.3 us per call and up to 1.8 times
  slower on a server CPU, BOX4's single-thread ratio on day 34). Predicted verdict: CLOSES, the host the closer half.
- M': (c) `take-back` about 0.5 ms (base about 1.6 ms, BOX4 day 34's `take-back` median 1.57 with hash 2 over about 2 MB
  of cached leases inside it): PASS. (d) the wall within about -1 to +2 ms of the base and the e2e within about -1.5 to
  +0.5 ms: PASS, the e2e the closer clause. On cached leases hash 2 is about 1 ms, so M''s card-level gain is small by
  construction; the 5090's 8 ms was write-combined read time.
- The gates green.

**Timing, for the rental.** The builds about 25 minutes (three servers and the test binaries), the price cell about 8
minutes, the A/B about 50 minutes (20 boots of the 27B), the gates about 20 minutes, the hit gate and the unit cells
about 15 minutes: about 2 hours from access.

## 3a. Amendment to section 3, before the sitting runs: M' carries the guard fix from review

Review on #711 (integ57) found an ordering bug in M''s guard: `host_demote_settle_hashing` landed the guard before the
reply's `seq` check, so a reply for another ticket could hand back, and the latch path then drop, leases whose views the
helper might still hold. The lead fixed it on the integration branch (`43d16d73f`: the guard lands only when `reply.seq
== seq`; otherwise it stays in `hashing` and its `Drop` leaks). The lead's ruling: M''s target-card cells run on a binary
that carries it. So, before any boot:

- **The M' arm, the price cell's binary and the gates' binary are one build: the lane tip after `origin/main` with #711
  is merged** (M' with the fix, the day-36 instrument). Its commit is written into the build receipt (`tree-tip.sha`).
- **The base arm is that same tip with M' removed**: `pro-single-day36/base-revert.patch`, the code of `55ae87616` and
  `43d16d73f` reversed and nothing else, applied on the box for the base build and taken back out (`build.sh` checks the
  tree returns to the tip). The two A/B arms then differ only in M'. Section 3's base `a0f915d8a` and M' `55ae87616`
  are replaced; the 5090 A/B (DAY35 section 8) stays as it ran.
- The cells, the rules (DAY33 section 6's verbatim; DAY35 section 7's (c) and (d) verbatim), N, the boot environment, the
  gate set and the expected numbers are unchanged.
- As prepared: `origin/main` `1d0cf13bc` (#711) merged as `e23383796`, no conflict (the fix `43d16d73f` in the lane;
  server lib 894 passed, clippy clean on tier, engine and server, `check-flags` and `check-conflict-markers` OK). The
  patch was cut there by reverting `43d16d73f` and `55ae87616` without a commit, crates only (2 files, 14 insertions, 486
  deletions; the one conflict, in the tests, resolved by dropping day 35's census and keeping day 36's), checked to
  compile (`cargo check -p memra-server`) with no M' symbol left and the day-36 instrument kept, and checked to apply to
  the tip (`git apply --check`). The build's argument is the commit that carries this section (its code equals
  `e23383796`'s).

## 4. The target-card sitting, as it ran (`pro-single-day36/box/`)

- BOX5: one RTX PRO 6000 Blackwell Workstation Edition (600 W), the lead's acceptance; the 27B artifact
  `1facf36c2db359dc..` (`model.sha256`). `build.sh 733075ed1` from 07:39Z, `rc=0`: the tip `dba61b98ee4290d9..` and the
  base (the tip plus `base-revert.patch`, `base-applied.stat`: `2 files changed, 14 insertions(+), 486 deletions(-)`,
  the tree back at the tip) `4e0277133e7d8052..`; iproute2 installed (`apt-iproute2.log`). `driver.sh`: price-and-ab
  `rc=0` (one collector hold, 07:44:56Z to 08:09:10Z), gates `rc=0` (08:20:57Z), hit OFF and ON `rc=0`, unit cells
  `rc=0` (08:22:21Z); every collector cell `executed-not-qualified False 0`; no lock retry; no compute app at the start
  or the end. Card telemetry (`m2/card-250ms.csv`): 5813 samples, 29 to 69 C, 15.8 to 600.8 W. Mirrored and checked
  file for file against the box's sha256 manifest (`box-manifest.sha256`, 506 of 506), binaries excluded (their hashes
  in `bins/*/memra-server.sha256`); the box scratch then removed.

**1. The price cell, DAY33 section 6's rule, verbatim** (`reading-day36-target.log`):

- `DAY36 READING submitted=100 landed=100 pending=0 n/a=0 replay_pass=True`
- `DAY36 READING recurrent-copy-host-ms N=100 median=0.190 min=0.180 max=0.200 first=0.180`
- `DAY36 READING recurrent-copy-owner-stream-ms N=100 median=0.290 min=0.280 max=0.300 first=0.290`
- `DAY36 PRICE VERDICT (target card) owner-stream median=0.290 host median=0.190 rule each < 0.5 ms per restore -> CLOSES`

**The D2D half of Move 2 owed item 1 closes as not worth a door.** The restore's recurrent copy costs 0.29 ms of
owner-stream GPU time and 0.19 ms of host time per restore on the target card (96 planes, 156.9 MB); the capture half
was refuted by construction on day 33. No code. The prediction held for the GPU time (0.2 to 0.3; about 1.08 TB/s
effective over 2 x 156.9 MB) and came in under it for the host (0.25 to 0.5 predicted: this box's CPU is a desktop part,
about 1.7 us per call over 112 calls).

**2. M' on the target card** (`reading-day35m2-target.log`, DAY35 section 7's (c) and (d) verbatim):

- (c) `DAY35 M2 C take-back N=80 median=0.11 min=0.10 max=0.11 rule N>=20 median<=1.5 max<=3.0 (copy-settle reading:
  copy-settle N=80 median=0.70 min=0.64 max=0.82) -> PASS` (base `take-back N=80 median=0.54`). **PASS.**
- (d) `DAY35 M2 D order=o1 wall base=114.70 m=114.30 m-minus-base=-0.40 rule <=+17.0 | e2e base=176.48 m=176.57
  m-minus-base=+0.09 rule <=+1.0 -> PASS`; `order=o2 wall base=114.70 m=114.20 m-minus-base=-0.50 .. e2e base=176.52
  m=176.54 m-minus-base=+0.01 .. -> PASS`. **PASS.**
- 20 boots, `STALL REPLAY: PASS` 20 of 20. Readings: the owner's hold per demote `owner-held` 2.67 to 2.27 ms; the
  helper's job 97.00 to 97.60 ms; the tenant stall flat (64.34 / 64.33 to 64.38 / 64.36 ms). As predicted, M''s gain on
  cached leases is small (hash 2 about 0.43 ms here) and the e2e sits at parity.

**3. The gates on the tip** (M''s (a) and (b) on this card): identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN
(teeth=0)` (12 ok each); failure OFF and ON `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok each), the ON arm's
`digest` cell with the bind's `contracts door D2H receipt` line `Key plane image checksum differs from its D2H receipt as
injected (MEMRA_KV_HOST_FAULT=flip-demote)` (the helper's re-hash saw the flipped byte), `VERIFY FAILED: promoted digest
.. != demote digest ..` and `ok: the promote caught it: VERIFY FAILED, loud and named`; the fault gate default and
plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (160 ok each); twin OFF and ON `PREFIX-NEWEST-TURN-FITS: .. -> PASS`;
hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 and 68 ok, the day-24 census `capture_submitted=12
capture_published=12 restore_submitted=13 restore_landed=13`). Unit cells: `ok. 18 passed` (the door's GPU cells,
`option_b_off_tick_demote_hashes_ride_the_helper_and_a_changed_lease_is_refused ... ok`), `ok. 10 passed` (the engine's
D2D, D2H span and H2D cells), `ok. 18 passed` (the CPU censuses, `day35_` and `day36_` among them), `ok. 6 passed`
(the engine censuses), `ok. 13 passed` (the tier rules). **ALL GREEN.**

**Verdict.** The D2D half closes (no door, no code). M' passes (a) to (d) on the 5090 (DAY35 section 8) and on the
target card. **Integrable: yes.**

**For the verdicts ledger** (`darklanes/agent-knowledge/gpu/verdicts-ledger.md`, a separate repository with its own
merge gate, so handed to the lead), the line as drafted:
`VERDICT:spill-restore-recurrent-d2d-not-a-door | scope: 27B NVFP4 MTP, one RTX PRO 6000 Blackwell WS, 100 restores,
2026-09-24 | the door restore's recurrent-state copy costs 0.29 ms owner-stream GPU and 0.19 ms host per restore (96
planes, 156.9 MB) against a pre-registered 0.5 ms bound each: the D2D half of the owner-thread offload closes with no
door; the capture half was refuted by construction (a write-after-read fence keeps its GPU time on the owner stream) |
keywords: tiered-kv, restore, recurrent, owner-thread, d2d | src: memra research/spill-a-20260919/DAY36.md section 4`

## 5. What is owed, checks, budget, cleanup

- Owed (Move 2 owed item 1 and beyond): **hash 1** (the D2H receipt) stays on the owner (the refuted M1; no off-thread
  form known that keeps the copy phase at one poll); **the fill on slower CPUs** (design F's copy misses the probe's
  tick where the fill outlasts it, BOX4 day 34); **the strong-form receipt**. The darklanes VERDICT line above. The D2D
  half is closed.
- Checks on the merged tree `e23383796`: `cargo fmt --all -- --check` clean; server lib 894 passed; clippy `-D warnings`
  on tier, engine and server, all targets, clean; the `DOCS_RS=1` pass `docsrs_rc=0` (on `d4f53945f`); `check-flags` and
  `check-conflict-markers: OK`; `git diff --check` clean; no em dashes in this lane's lines; no new `MEMRA_*` name.
- Budget: about 0.2 agent-day of work, plus the box's about 45 minutes from access to release.
- Cleanup: BOX5's `/root/wt-a` and `/root/spill-receipts/a-day36` removed after the checked mirror, no process of this
  lane on the box, `BOX5 RELEASED` sent; the 5090 carries no process of this lane; `/tmp/wt-a-d36` removed at close.
