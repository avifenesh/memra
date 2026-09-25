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
