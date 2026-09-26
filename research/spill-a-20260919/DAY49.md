# WP-A day 49: OWED items 7 and 8, the demote's first-touch attribution (one hypothesis, one cell)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on the tip the lead integrates as integ62 (`e34a6c598`). The lead's
order after integ62: items 7 to 14 in order. Items 7 and 8 share one hypothesis (the ledger's own status line), so they
share this pre-registration and one cell; each keeps its own reading and verdict. Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before any code)

**The two items.**

- Item 7 (C DAY39 section 7, ruling 43): lane C's b1 tree read `pre_submit=43.71` (first three demotes) and `42.44` (4th
  on) in C's stall cell, where day 28's double-park cell had shown a first-touch step (39 to 45 ms on the first two
  demotes, about 6 ms steady). On b1 the pre-submit held the recurrent f32 planes' D2H into fresh heap `Vec`s on the owner
  stream; since A day 30 those planes ride the ticket as spans into a reused pinned staging set, and the current tree's
  pre-submit reads about 1.2 ms on 64-token 27B entries (DAY48's boots) and 20.75 to 24.98 ms on 5122-token entries
  (DAY43's receipts).
- Item 8 (C DAY39 section 7): b2's helper read `hashed_in=107.3` / `104.8` ms against b1's 73.2 to 73.3. Since A day 30
  the helper copies each landed staging buffer into a fresh heap `Vec` before it hashes (`p.data =
  Arc::new(staged.as_f32_slice().to_vec())`); the current tree's helper reads 83.3 to 85.1 ms per 157.9 MB on the
  target card (DAY42 and DAY48).

**The hypothesis H** (the ledger's, tested, not assumed): the owner's (b1) and the helper's (b2 and today) copies into a
heap `Vec` take fresh pages on every demote where no host entry frees, and the page faults are the step. Where host
entries do free (a budget the LRU turns over), freed heap memory is reused and the step shrinks after the first demotes.

**The lines (log-only; no behavior, no numeric program, no flag changes).**

1. The helper's split: the reply carries the copy's time and bytes, the hash's time, and the helper thread's minor page
   faults across the copy (`getrusage(RUSAGE_THREAD)`, `ru_minflt` before and after); the owner's `demote digests landed`
   line gains `; helper split: copy X ms over B MB (minflt +M), hash Y ms`.
2. The pre-submit split: `host_kv_planes_submit_contract` times its pinned destinations (`alloc_host`, count and bytes),
   its registration, and the rest, with the owner thread's minor faults across the pinned allocations; the span attach
   (`host_spans_submit`) times itself; one line per demote, `[prefix-host] demote pre-submit split: ticket seq=S leases
   X ms (N pinned, B MB, minflt +M), register Y ms, spans Z ms, other W ms`.

**The cell** (one RTX PRO 6000 Blackwell, the 27B NVFP4 MTP artifact, the collector's hold; `pro-single-day49/`): the
attribution tree's one binary, `stall_cell.py --mode demote --n 5` (ten demotes per boot, nine steady), three arms:

- `nofree`: `MEMRA_KV_HOST_MB=8192` (ten 160 MB entries fit: no host entry frees in a boot).
- `free`: `MEMRA_KV_HOST_MB=480` (three entries fit: from the fourth demote on, every publication evicts one, freeing
  its heap payloads and its pinned leases).
- `long`: `--mode demote-long --n 5` at `MEMRA_KV_HOST_MB=8192` (5122-token entries, the 20 ms pre-submit), a reading.

The environment otherwise the S sittings' (`MEMRA_PREFIX_CACHE_MB=256`, 448 for `long`), door ON. Boots interleaved
`nofree free nofree free ..` five each, then `long` x3; each boot's start temperature and SM clock recorded. Reader
`day49-reading.py`: per arm, the steady demotes' (the 4th on for `free`, the 2nd on otherwise) medians of the helper's
copy ms, hash ms, copy minflt per MB, and the pre-submit split's segments.

**The rule, stated before any cell** (per the steady demotes; `pages` = copy bytes / 4096):

- Item 8, **H attributed** when the `nofree` arm's copy minflt median is at least 0.5 x pages, the `free` arm's is at most
  0.25 x pages, and the `nofree` copy median exceeds the `free` one by at least 5 ms. **H refuted for item 8** when the
  two arms' copy minflt medians are within a factor of 2 of each other; otherwise **not placed**, the split recorded.
- Item 7: the current pre-submit split is recorded for both 64-token arms and the long arm; H for item 7 is read through
  the same mechanism (b1's own program is not re-run: it no longer exists on this tree): **the b1 step is attributed to
  H** exactly when item 8's verdict is `attributed` (the same heap first-touch mechanism, the owner then and the helper
  now); otherwise item 7 closes as `not placed` with the current split as its record. The long arm's pre-submit is a
  reading: its segment shares name where the 20 ms sits.
- If H is attributed, the improvement that removes the first touch (a reused heap payload pool, or the staging handed
  to the entry instead of copied) is pre-registered as its own design before its code, with its own price clauses.

**Predictions.** `nofree`: about 38,500 faults per 157.9 MB copy (one per page), copy about 20 to 30 ms of the 84 ms;
`free`: far fewer after the 4th demote. The long arm's pre-submit sits mostly in its pinned lease allocations (64
leases, about 122 MB, fresh `cuMemHostAlloc` per demote).

**What each card decides.** The target card only (the attribution is the host's; the 5090's host is another class).

**Budget.** 0.4 agent-day: the lines and their census 0.15, the reader and the sitting 0.1, the card's share 0.15.

## 2. As built (`d77ccec61`), the CPU cells, and the sitting prepared

- The lines: `[prefix-host] demote pre-submit split: ticket seq=S leases X ms (N pinned, B MB, minflt +M), register Y
  ms, spans Z ms, other W ms (pre-submit T ms)` after each `demote submitted off the tick`, and `[prefix-host] demote
  helper split: ticket seq=S copy X ms over B MB (minflt +M), hash Y ms (helper Z ms)` after each `demote digests
  landed`. The helper's split is its own line rather than a suffix of the landed line (section 1 said a suffix): the
  landed line's text is pinned by earlier censuses, and a separate line changes nothing they read. `thread_minflt`
  reads `getrusage(RUSAGE_THREAD)`. The pre-submit split rides `PendingContractDemote` boxed (clippy's
  `large_enum_variant` on `HostImage` otherwise). Census `day49_the_split_lines_are_log_only`.
- CPU cells, green: server lib `921 passed; 0 failed; 25 ignored`; clippy `-D warnings`; fmt; `git diff --check`;
  `tools/check-flags.sh`.
- The reader `day49-reading.py` (checked on a synthetic fixture) and the sitting `pro-single-day49/` (`build.sh <tip>`,
  `driver.sh`, `cell.sh`: nofree and free interleaved five boots each, long x3, one collector hold, about 25 minutes of
  card time). Any host class with one RTX PRO 6000 Blackwell and the 27B artifact at
  `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`; the host class is recorded (`host-shape.txt`), and the reading is
  that class's.

## 3. The cell on the target card, as it ran (`pro-single-day49/box/`)

- Run by the lead on BOX10 (one RTX PRO 6000 Blackwell Workstation Edition; the host reads `AMD Ryzen 9 9950X3D2
  16-Core Processor`, 32 CPUs, 124 GB): `build.sh 03ec9b063` rc=0, the tip binary `ada9fe4eab1168aa..`
  (`binaries.sha256` and the lead's `box-binaries.sha256` agree); `driver.sh` rc=0; one collector hold on
  `/tmp/memra-gpu.lock` (`LOCK.json`: `inherited-flock-same-open-description`) from 10:42:30Z, `cell rc=0` at
  10:57:03Z; the card idle at the hold's start and at its end (`compute-apps.after.csv` empty). The model
  `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c2db359dc..`.
- Mirror: 91 files against the box's `box-mirror-manifest.sha256`, 91 OK, 0 mismatched (`lead-a49.out` and
  `lead-box-after.txt` are the lead's own driver output and the box's state after the run). Every boot's stall replay
  reads `STALL REPLAY: PASS` (16 of 16). Both readers re-run here on the mirrored receipts print the box's readings
  byte for byte.
- Thermal regime: boot starts 35 C (the first boot, the card idle at 180 MHz) then 55 to 71 C at 2820 to 2827 MHz; the
  hold's 250 ms telemetry 35 to 84 C over 3490 samples.
- Count, as read: a 64-token boot makes nine demotes, not the ten section 1 said (`--mode demote` seeds nothing, so the
  first intruder has no entry to demote); the long arm's seed makes ten. The reader's steady sets are its own rule's:
  nofree 8 per boot (N=40), free 6 per boot (N=30), long 9 per boot (N=27).

**The reading, verbatim** (`cell/reading-day49.log`):

```
DAY49 READING arm=nofree helper N=40 copy_ms=23.73 mb=156.9 minflt=38306 hash_ms=59.05 helper_ms=83.50 | pre-submit N=40 leases_ms=0.61 leases=32 lease_mb=1.9 lease_minflt=512 register_ms=0.15 spans_ms=0.16 other_ms=0.06 pre_ms=0.98
DAY49 READING arm=free helper N=30 copy_ms=7.64 mb=156.9 minflt=0 hash_ms=59.23 helper_ms=67.60 | pre-submit N=30 leases_ms=0.12 leases=32 lease_mb=1.9 lease_minflt=0 register_ms=0.14 spans_ms=0.16 other_ms=0.06 pre_ms=0.48
DAY49 READING arm=long helper N=27 copy_ms=24.03 mb=156.9 minflt=38306 hash_ms=59.21 helper_ms=139.80 | pre-submit N=27 leases_ms=19.26 leases=32 lease_mb=151.1 lease_minflt=36896 register_ms=0.17 spans_ms=0.17 other_ms=0.06 pre_ms=19.66
DAY49 ITEM8 pages=38306 nofree_minflt=38306 (rule >= 19153) free_minflt=0 (rule <= 9576) copy_ms nofree=23.73 free=7.64 diff=+16.09 (rule >= +5.0) -> H attributed
DAY49 ITEM7 -> b1 step attributed to H (the heap first touch) (the current pre-submit split above)
```

Per boot (steady medians, the demote's ledger line beside the split), every boot of an arm alike:

| arm | copy ms | copy minflt | helper ms | wall t0 to publication ms | landed after |
|---|---|---|---|---|---|
| nofree b01 to b05 | 23.56 to 24.09 | 38306.5 | 83.2 to 83.8 | 101.0 to 101.5 | 2 polls |
| free b01 to b05 | 7.60 to 7.67 | 0 | 67.3 to 67.7 | 88.4 to 88.8 | 1 poll |
| long b01 to b03 | 23.77 to 24.30 | 38306 | 139.4 to 140.1 | 361.7 to 361.8 | 12 polls |

**Verdict, as registered.** Item 8: **H attributed**. The helper's copy of the landed staging into a fresh heap `Vec`
takes one minor fault per 4 KiB page (38306 per 156.9 MB) on every demote where no host entry frees, and the faults are
16.09 ms of the helper's 83.50 ms. Where the LRU frees an entry, the freed heap memory is reused and the copy takes no
fault at all (the free arm from its fourth demote on; b01's third already read `minflt +768`). Item 7: **the b1 step is
attributed to H**, the same heap first touch, taken then on the owner thread and now on the helper.

**What else the cell read** (readings, no clause):

1. H's price where it lands. The first touch moves the demote's publication one tick-top poll later: the steady wall t0
   to publication reads 101.0 to 101.5 ms without frees and 88.4 to 88.8 ms with them (landed after 2 polls against 1),
   so about 12.7 ms of every demote's publication while the host tier fills. A request that meets the entry `Demoting`
   parks for that time.
2. The first demote of every boot holds the owner thread about 20 ms in its pre-submit (`pre-submit 20.02 ms` and
   `20.70 ms` in the b01 boots), and the split puts it in the `spans` segment (19.74 and 20.43 ms), where the span
   attach takes the staging set's buffers: the set is empty at boot, so its first demote allocates every buffer
   (`staging_take`, fresh pinned memory; placed from the code, the split does not time the takes apart). Once per
   context.
3. The long arm's pre-submit is 19.66 ms, 19.26 of it in its leases: 32 pinned KV destinations, 151.1 MB, 36896 minor
   faults, allocated fresh on the owner thread on every demote while no entry frees. The prediction had the place right
   and the shape wrong (it said 64 leases and 122 MB). That is 19 ms of the tenant's tick per long demote, a new owed
   item (OWED item 19, beside item 14, the same leases' frees).
4. The long arm's helper reads 139.80 ms against copy 24.03 plus hash 59.21: the rest, about 56 ms, is the bind's KV
   re-hash over the 151 MB of lease views (design M'), which the split does not time (it times the staged payloads).
5. The 64-token leases: nofree 0.61 ms and 512 faults per 1.9 MB, free 0.12 ms and none (the freed entries' pinned
   memory reused).

**What follows, as section 1 registered.** H is attributed and recurs on every demote while the host tier fills, so
the improvement that removes the first touch is pre-registered as its own design before its code (DAY51, OWED item
17).
