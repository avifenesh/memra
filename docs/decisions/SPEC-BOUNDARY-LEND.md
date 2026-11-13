# The speculative prime boundary lends the session's rows to its prefix entry (2026-11-12)

**Status:** adopted, memra#1221 part 2 (the plane copies and the pool growth of the boundary
publication). Receipts:
[`research/spec-boundary-lend-20261112/`](../../research/spec-boundary-lend-20261112/). Follows
[`SPEC-CAPTURE-HEADROOM-UNIFIED.md`](SPEC-CAPTURE-HEADROOM-UNIFIED.md) (part 1) and
[`research/spec-boundary-publish-20261110/`](../../research/spec-boundary-publish-20261110/)
(memra#1208).

## Question

A cold speculative request publishes its prompt's KV to the prefix cache once, after its first
burst. After part 1, on GB10 with Qwen3.8-27B and its MTP draft, that publication still allocated 34
planes and copied the session's rows into them: about 50 ms on the scheduler thread at a 131k
prompt and 110 ms at 260k, with the pool grown by 544 MB and 1,408 MB inside it. Can the entry
have the rows without the copies, while restores stay exact, eviction still frees real memory and a
session that lives on (parked, resumed, rewound, grown) stays correct?

## What the measurements showed

GB10, Qwen3.8-27B with its MTP draft, unfilled page cache, one cold request per cell on a fresh
server, one cell per lease, interleaved on one unit per context. Main's publisher with part 1's
trace probe (B), the publisher returning at its entry (Z, the floor of a publication that allocates
and copies nothing), and the copies moved to the engine's copy stream (O), N = 2 each:

| | 131k: B | O | Z | 260k: B | O | Z |
|---|---:|---:|---:|---:|---:|---:|
| boundary stall, ms | 63.2 | 52.3 | -2.9 | 122.5 | 124.4 | 6.5 |
| publisher on the scheduler thread, ms | 50.6 | 51.0 | 0.0 | 111.0 | 114.0 | 0.0 |
| pool growth inside the publisher, MB | 544 | 544 | 0 | 1,408 | 1,408 | 0 |

- The publisher is allocation-bound: 49 to 50 ms (131k) and 109 to 115 ms (260k) of it are
  `cuMemAllocAsync` calls, the few that grow the pool taking 2 to 33 ms each; the 34 device copies
  already run under them. Moving the copies to another stream (O) therefore changes nothing that
  matters, and part 1's arm H showed the growth does not overlap burst 1 either.
- Only a publication that allocates and copies nothing reaches the floor (Z).
- Main's entry adds 4.13 GB (131k) and 8.20 GB (260k) of pool used from the boundary to the end of
  the session, the rows held twice.

## Chosen

**The entry reads the session's rows in place, and the session's plane moves to the entry if the
session ends first.** Code: `memra_kv::KvPlane::lend`, `KvLend`, `KvLoan` (`plane.rs`), the lend
floor of `Cache::rollback`, `Cache::rewind_recurrent_from` (`lib.rs`) and the MTP scratch's
`set_len` and `set_plane_len` (`memra-engine/src/spec.rs`), and in `memra-server/src/worker.rs`
`PlaneBuf`, `SpecPlanes`, `spec_plane_lend` and the prefix cache's loan books
(`PrefixCache::reconcile_loans`).

- **The lend.** `KvPlane::lend(bytes)` returns a non-owning `CudaSlice<u8>` view of exactly the
  plane's first `bytes` (so the restore's exact-length check, the round-trip digests, the host tier
  and every other reader see the plane they saw), plus an `Arc` of the plane's loan. The MTP
  publisher (`spec-boundary`) lends every trunk K/V plane and the draft plane at `pos` rows. Nothing
  is allocated and nothing is copied. The publication decides once, before its first lend
  (`spec_publication_planes`): when any plane it would publish cannot lend (`KvPlane::can_lend`: a
  gate-only fully mapped VMM plane, a suspended plane, a range past the backed bytes) or is ordered
  on another stream than the engine's, the whole publication copies, as main did. The floor a loan
  enforces is the longest lend that lives: it falls when that lend goes.
- **The hand-off.** `KvPlane`'s `Drop` moves the whole plane (pooled operand, or on-demand VMM
  state) into its loan when a lend is alive, instead of freeing it; the loan frees it when its last
  lend goes. Every way a session's planes can go is covered by construction: retirement, a parked
  session's eviction, the grown cache of an affinity grow, the scratch a demotion drops, an OOM
  teardown. An on-demand VMM plane (the RTX PRO 6000 Workstation Edition default) releases every
  extent past the lent bytes when it is adopted, and again at a reconcile when its longest lend
  went. The adoption is reported to the prefix cache's queue (`AdoptionSink`), which the reconcile
  drains, so a reconcile walks the adopted books only.
- **Exactness.** The engine already holds a session's rows below its prime boundary append-only for
  the session's life (the doc of `SpecBoundaryCapture`); the copy relied on it for one burst, the
  lend relies on it for as long as the entry lives. It is now structural: every write that could
  start below a live lend refuses before it writes (`Cache::rollback`, `Cache::rewind_recurrent_from`,
  `MtpScratch::set_len` and `set_plane_len`, `KvPlane::into_pooled`, `pp::restore_cache_checkpoint`,
  which checks every layer before its first write, and `prefix_restore_validate`, whose
  destination must not lend), `KvPlane::release_beyond` keeps the lent bytes backed, and
  `TierTransfer::take_device` refuses a lent registry plane before it gives up ownership. A census
  pins every caller of those functions.
- **A checkpoint under the lend.** The boundary capture can sit above the session's turn
  checkpoint (the seed at a prompt end on the 32-token grid over a stable stop under it, or a
  miss-LCP split above the stable point). The speculative checkpoint rewind then checks the
  trunk's and the draft scratch's lends before it writes anything and refuses in place. The
  worker rewinds such a session into a fresh cache instead (`spec_lend_rewind`), on the affinity
  grow's own path: under the on-demand construction scope (an on-demand card keeps its planes
  on demand) and with its one out-of-memory reclaim retry, sized at the request's target
  capacity on the affinity path and at the session's own on the spec pool's grid path. The route
  is decided again on the retry: the reclaim evicts the unleased entries, those reading the
  session among them, so their lends go and the retry rewinds in place, as main did. Only a lend
  held by a leased entry, with the device still out of memory, sends the request cold (`[spec]
  rewind refused below a lent prefix row`). The fresh cache is the checkpoint's rows copied, the
  lent planes left to their entries. Before the attempt the worker prints `[spec] rewind below a
  lent prefix row: the session tries a fresh cache (SITE; model M)`, one literal for both sites
  (`LEND_REWIND_FRESH_LINE`; the cells count its prefix). Not measured on an RTX PRO 6000
  (on-demand planes).
- **Eviction semantics: charged bytes are real bytes.** An entry is charged the device bytes it
  holds. While the session holds the plane, the entry reads it and is charged nothing for it (the
  bytes are the session's, booked by admission); evicting it frees what it holds (the capture's
  recurrent planes) and nothing more, and the cache reports exactly that. When the session drops
  the plane, the plane is charged to one of the entries that read it at the next reconcile (the
  tick top, after retirement, before every insert, eviction and snapshot decision, and before
  admission's reclaim) and LRU and the budget treat it as any entry: the tick-top and retirement
  reconciles hold the cache to its budget (`reconcile_to_budget`: compaction by the rule below,
  then the oldest unleased entries), so an adoption never leaves it over budget past the tick
  that charged it; evicting the entry frees the plane. Two entries
  reading one plane (the miss-LCP split and the stable boundary of one cold burst) share one book
  and one charge. The book lists its readers; its holder is always one of them: when the holder
  leaves the cache, a reader that stays becomes the holder (with or without a charge), and a
  charge it carried moves with it, which frees nothing, so the total does not change. Every
  reconcile checks `total_bytes` against the entries' bytes (resynced when it breaks) and the
  charges against what the entries and adopted planes hold; a break prints `[prefix-loan]
  INVARIANT BROKEN`, counts in `memra_prefix_loan_invariant_breaks_total` (with the lent and
  adopted gauges and the adoption and compaction counters in /metrics) and fails a debug build.
  Host demotion (its byte budget counts the entry's logical bytes), the disk handoff and the
  exports keep the entry's
  logical bytes.
- **Admission counts each byte once.** Memory admission reads the device's free bytes from the
  device and is offered the cache's charged evictable bytes (`evictable_bytes`, exactly what
  `evict_all` frees), so a lent plane is the session's memory while the session holds it and the
  cache's charge after it drops, never both. A book charged to an unleased holder while another
  reader is leased is not offered (evicting the holder moves the charge to the leased reader),
  and the insert preflight counts it with the leased bytes (`leased_bytes`), so an insert never
  evicts its neighbours for room the lease keeps.
  Retirement reconciles in the same tick, and so does admission's reclaim before it reads the
  cache or sets the partial pass's target; a lender dropped in between is charged at the next
  reconcile, and until then the cache offers less than eviction would free, never more. The
  headroom ladders do not drop a parked session whose rows a surviving entry reads: dropping it
  would free none of its lent planes.
- **Over-hold and compaction.** An adopted pooled plane is the session's whole capacity, prompt +
  `max_tokens` + 8 rows. A bounded long-prompt request holds a few hundred rows past its entry; an
  open-output request on a card without memory-shaped admission (GB10: ctx 262,144) can hold many
  times its entry. An adopted plane is compacted, its readers' rows copied into exact planes of
  their own and the plane freed, exactly when that frees more than it copies
  (`loan_compaction_frees_more`: held - read > read), at three points: on unified memory at its
  adoption (the over-hold is the device's own memory there), whenever a reconcile leaves the cache
  over its budget (the tick top and retirement, before the LRU that follows), and on memory demand
  (an insert that does not fit, a partial eviction). The first failed allocation or copy stops
  compaction for the rest of the pass, and no compaction runs again until the cache freed at least
  the bytes that allocation needed (`compaction_may_retry`). On the served open-output cell (GB10,
  two cold 60k requests whose planes are the 262,144-row context): the first session's adopted
  planes compacted at its retirement (34 planes, 8,270 MB held, 1,892 MB copied, 0.3 to 0.4 ms on
  the scheduler thread; the second's at its own retirement), the cache ended at 4,303.9 MB charged
  == real against main's 4,286.3 MB, and MemAvailable followed main's within 0.25 GiB at its first,
  lowest and last readings, at the default budget and at 8 GB. The boundary stalls stayed under
  main's (4.3 and 0.0 ms against 26.8 and 14.7).
- **Scope.** The MTP publisher lends on a plane ordered on the engine's own stream (one device).
  A lend carries no cross-stream event, so a pipeline-parallel stage's plane copies. The DSpark and
  GLM-5 publishers keep copying (`SpecPlanes::Copy`): their park rewinds are not checked against a
  lend. PP host-bounce, SWA ring and latent caches already refuse publication. Not keyed on a
  device: it removes work everywhere.
- **Failure isolation.** A lent row is shared with a live session, so a failure in the session
  must not reach it. The rows `[0, pos)` an entry reads are written once, by the prime, before the
  publication, and never again (the floors above). A failed wave marks its caches tainted
  (`CacheTaintGuard`, `PpDecodeWave`, the stage merge in `hybrid_forward.rs`): a tainted cache is
  never resumed, and the wave wrote only rows at or past the cache's length, which a lend keeps at
  or past the lent rows. A session torn down after an error or an out-of-memory drops its planes,
  which move to the entries unchanged. A sticky device error affects every plane on the device,
  copied entries as much as lent ones.

## Rejected

- **Copying the planes (main).** At the head of part 1 the copy costs the stall it was measured to
  cost (59 to 63 ms at 131k, 122 ms at 260k on GB10) and holds the prompt's rows twice from the
  boundary to the session's end (4.13 and 8.20 GB of pool used).
- **The copies on a side stream** (arm O). The publisher is allocation-bound: the copies already
  ran under the allocation calls, so moving them to the copy stream took 11 ms off the stall at
  131k and added 2 ms at 260k, the pool still grew 544 and 1,408 MB, and the rows were still held
  twice. Every reader and every free of such an entry would also need a cross-stream join.
- **Reserving the entry's bytes earlier** (part 1's arms P and H): rejected there; the growth moves
  into burst 1 or before the first token.
- **Lazy publication** (read the session's rows in place, copy when the session ends). The same
  50 to 120 ms of allocation and copy moves to retirement on the scheduler thread, where the
  co-scheduled sessions and the next admission wait for it once per request, and a parked lender
  evicted for memory would have to allocate the copy at the worst moment. The hand-off needs no
  copy, and compaction keeps the copy for the case where it frees more than it costs.
- **Never compacting.** An adopted plane is the session's whole capacity: a 10k-token prompt on a
  GB10 open-output session (262,144 rows) would hold 26 times its entry until LRU evicted it.
- **Always compacting at adoption.** That is lazy publication's copy for every request; for a
  bounded long prompt the plane holds a few hundred rows more than the entry (558 rows, 18 MB, on a
  131,566-row session), which no copy pays for.
- **Charging a borrowed entry its logical bytes.** The bytes are the session's; counting them in the
  cache too would make the unified-memory insert budget (resident + live free - headroom) believe it
  holds memory it does not, admitting inserts under the headroom, and would evict borrowed entries
  to make room that their eviction cannot make.
- **Lending on the DSpark and GLM-5 routes.** Their parks rewind on their own paths, which this
  change does not check against a lend; they keep copying (`SpecPlanes::Copy`).

## Proof

- **Timing, same unit, interleaved** (B S S B B S, N = 3): at 131k (unit 2) the stall falls from
  59.1 to 7.9 ms and the publisher from 51.6 to 0.0 ms; at 260k (unit 1) from 118.1 to 5.5 ms and
  from 115.3 to 0.1 ms, every head cell under every main cell on both readings. No pool growth at
  the boundary; pool used at the boundary 4.17 and 8.23 GB lower. Rounds, TTFT, the total and
  every text unchanged (burst 1 at 260k: +7.2 ms, every head cell over every main cell, 0.3%,
  reported as measured; it runs before the publication). RTX 5090 at 28k: unchanged (one text,
  rounds 33.67 against 33.38 ms, TTFT 24.22 against 24.25 s over 4 cells each).
- **Exact restores.** The copying publication's 18 round-trip digests are lever 5's (main's) byte
  for byte, and every lending twin's digest equals its copy's (34 planes on the 27B geometry), so a
  lent entry restores what main's entry restores. Warm == cold pair cells on both the borrowed path
  (the lender parked when the hit reads it) and the adopted path (no parking: the lender dropped at
  retirement), greedy and sampled with the entry digests: every warm turn equal to its cold twin,
  head and main serving the same answers and publishing the same digests. An adopted entry's digest
  equals the borrowed entry's and main's for the same rows, and a 135k-token entry restores warm ==
  cold on GB10. The repo's four gates pass on the head: seeded invariance (54 rows), spec burst
  boundary (229 rows), prefix exact resend and prefix restore identity.
- **Writes below a lend fail closed**, each with a red arm that fails its GPU cell (rollback, recurrent
  rewind, `into_pooled`, the release clamp, the hand-off itself, the MTP scratch's setters), and four
  source censuses (the lending publisher, every caller of those functions, the KV crate's guards,
  the scratch's setters) that fail on main and pass at the head.
- **Eviction, charges, no leak.** The adoption cell: a borrowed entry is charged its own planes only;
  the lender dropped, the adopted capacity is charged exactly once; evicting frees it; an entry
  evicted while its lender lives frees nothing more, and the lender then frees its rows; pool used
  back at its baseline. The agentic replay (29 turns, c = 2, MTP route, temperature 0) on GB10 and
  the RTX 5090: every `[prefix-loan]` reconcile line reads charged == real (87 and 80 per boot), and
  after dropping every evictable pool the pool's used bytes are main's to the byte, with nothing
  charged. Same texts as main on every request main answered. Compaction: both sides of the rule on
  the GPU, compacted entries' digests equal to the copy's.
- **Review round 1** (README, "Review round 1"): the books' arithmetic on the CPU with two
  entries over one plane, the holder leaving before and after the adoption and a leased co-reader
  (red arms: the double count, the holder kept, the lease offered; each fails it); a GPU cell where
  one session publishes two entries over the same planes, charged == real at every step and the
  pool back at its baseline; the live-lend and VMM cells; censuses of the checkpoint restore, the
  rewind route, `take_device` and the publication's decision (red arms fail them); the replays and
  the gates rerun on the review head; the open-output compaction cell; the session gate's lend
  above the checkpoint.
- **CPU**: fmt, clippy `-D warnings` on all targets, the server, KV, engine and gguf suites, the
  coalescer contract and the portable suites, on a CPU build host.

## Limits and open items

- **Devices.** Timing cells on GB10 only; the RTX 5090 ran the no-regression cells, the agentic
  replay and the pair cells. No RTX PRO 6000 cell: its on-demand VMM planes lend and hand off (the
  KV crate's on-demand cell ran on GB10's VMM), not measured served.
- **Routes.** DSpark and GLM-5 still copy (their park rewinds are not checked against a lend). PP
  stage-owned caches copy (their planes are ordered on the stages' streams); no multi-card cell
  ran.
- **The floor guards the functions that lower a length**, `into_pooled` and `release_beyond`; the
  spec round loop's own length writes raise a length or restore the committed rows (at or past the
  boundary by construction, the invariant the copy relied on for one burst).
- **Over-hold.** On a discrete card an adopted plane is charged its capacity until the cache is
  over its budget, an insert needs room or it is evicted; bounded requests there hold a few
  hundred rows past their entry, which no compaction pays for. On unified memory a plane that
  holds more than twice what its readers read is compacted at its adoption.
- **Admission's arrival term** in the discrete-card partial reclaim still counts a speculative
  arrival's entry at its logical bytes, though at publication it needs only its recurrent planes:
  conservative, unchanged from main, left for its own item.
- **memra#1266**: main's MTP-route pooled-verify OOM at c = 2 on the RTX 5090 agentic replay. The
  head's boots do not reach it because they hold 2.3 GB less pool at that point; the uncovered
  transient is unchanged, so this change does not close it.
