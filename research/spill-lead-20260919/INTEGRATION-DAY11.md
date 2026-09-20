# Integration: day 11 (`lane/spill-integ8-20260921`, from `main` `30e9c7a38` = #576 merged)

## State at open (21:15Z)
Mac session finished (owner note). #573 (old session integ6) and #576 (integ7: C day 10, records, TESTING.md rewrite)
are on `main`. Other sessions on this rig: other Claude sessions (`wt-fix-379-suffix`, `wt-api-lockstep-router`,
`wt-mtp-*`) and codex PRs. Conflict resolution done on request for #566 (`docs/ROUTER.md`, union),
#557 (`research/INDEX.md`, main's spill-lead row) and #569 (`INDEX.md` + boundary allowlist, union); each pushed with
a comment; #566 pushed under its own `MEMRA_RELEASE_QUALIFICATION_MODE=development` (that branch retires the perf
skip), #557 with the logged `MEMRA_SKIP_PERF_CI=1`. None merged by this session.

## Lane B day 11 (`d1844b3c0`, pushed by the lead with the logged override; engine files in range are `main`'s plus
## the gate binary)
`--reclaim-cycles N` on `kv-tier-gate` (cli, `reclaim_contract::classify_cycles`, `active::Roundtrip`/`write_cycles`,
gate loop), tier suite `tests/reclaim/{main,day11}.rs`, `verify-day11.py`, TESTING.md line. Series verdicts verbatim:
```text
pro-single/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
rtx5090/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 residual_first=2097152 residual_last=2097152 g1_reclaim_qualified=false
pro-single/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
rtx5090/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
```
Per cycle on both cards: `free_after_restore == free_before` exactly, released 905,969,664 B, residual 2,097,152 B,
baseline drift 0, mapped-VA probe deltas 0, restored prefix bit-identical. PRO 600/600 W; the laptop 5090 reports
power limit `[N/A]` (max 175 W). Cross-box timing not compared. Lead code read: `classify_cycles` requires every cycle
`bounded_no_leak`, identical `free_before`, and residual exactly one granule for the one-time class; one cycle is
`unclassified`; growth in either the residual or the unreturned series is `growing-residual`; the per-cycle G1 line
and tightening (e) are untouched (`g1_reclaim_qualified` folds with AND). Refusals fail closed (count < 2, junk,
duplicate, no diagnostic, pooled, baseline case, second route), unit-tested.

## Lead rulings, day 11
6. **Lift tightening (e) for one class, under the series condition only.** The decision record pre-registered the
   lift: "lifting (e) for a specific class needs a lead ruling backed by evidence on both card classes". Both card
   classes now carry `one-time-driver-mapping-metadata` from a 5-cycle series with drift 0. Ruling:
   `g1_reclaim_qualified=true` with a nonzero residual is allowed only when all of: the run is a `--reclaim-cycles N`
   series with N >= 5; `residual_series_class = one-time-driver-mapping-metadata`; (a) to (c) hold in every cycle;
   the restored prefix is bit-identical in every cycle; baseline drift is 0. The label becomes
   `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, N cycles)`. A single roundtrip with a nonzero
   residual, a series shorter than 5, any other class, and any pooled run stay `not G1 PASS` / `false` /
   `not-applicable-pooled`. Criteria (a) to (d) are unchanged; (e) stays in force for every other shape. B day 12
   lands the gate line, `verify-day12.py`, the decision-record paragraph (evidence table, this ruling), the
   TESTING.md line, and reruns the 32k series on both cards to produce the label; the label exists only when the
   rerun prints it.
7. **8k cycles control.** The mapped-VA probe refused to re-reserve the original VMM address for the small 8k planes
   (K 5 granules, V 3) on both cards, deterministically; it freed the stray reservation and refused, as designed. No
   probe or CLI change. The 8k evidence stays the single-roundtrip `ACTIVE-8K G1 PASS` with residual 0 on both
   cards (day 10 PRO, day 9 5090). The limitation goes into the door doc as an open item; revisited only if the door
   is promoted at decide-by.
8. **Lane A local flake.** `crates/memra-tier/tests/storage/day4.rs` returns `Err(Busy)` at :143, :240, :309, :471
   on this rig only (3 of 53 on a quiet release rerun; 276/276 on the PRO). A day 10 task: reproduce under a CPU
   quota, decide race versus rig, fix or label; no engine change without a receipt. Dispatched after C day 11
   finishes (two-agent cap).

## Lane C day 11 (`f5c5f592e`, pushed by the lead with the logged override plus a boundary pin)
Installer catalog from the model plan: new `crates/memra-gguf/src/expert_banks.rs` (`expert_bank_catalog(plan, contract,
census)` selects the three bank requirements per MoE block by semantic id `LayerTensor::MoeExpert{Gate,Up,Down}Bank`,
trunk by position and MTP by depth, binds through `TensorContract::bind`; no tensor name spelled in the installer, no
architecture allowlist, no new env read). `native.rs` takes the contract from the model pack and the census from
`census_from_gguf`, refuses with `REFUSED: experts-via-tier expert catalog refused: <detail>` exit 2 (no MoE
projections, missing or ambiguous contract entry, contract `Missing`/`DuplicateCensusName`/`ShapeMismatch`, undeclared
or unconsumed scale planes, non-GGUF dialect, plan/model layer disagreement), prints
`[experts-via-tier] catalog blocks= banked= projections= catalog_sha256= records= records_sha256=`. Nits landed:
`expert_bank_cli` splits at `=` and matches keys exactly (look-alikes are usage errors), `SLOT_TAIL_PAD_BYTES` shared by
`gpu_slot_bytes` and `moe_cache::hard_slot_bytes` (10 literals, all 8, `slru-synthetic.json` re-pinned with identical
rows), `#[doc(hidden)] pub mod banked_residency` with no crate-root re-exports. Parity: `catalog_sha256`
`2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8` (41 blocks, 123 projections) in all three cells and
recomputed by `verify-day11.py` from memra's own `tensor-census.tsv` with the day-10 literal spelling; `installed
artifact_sha256=df27a780...7adf host_slots=16 max_expert_bytes=860160` byte-identical to day 10. Cells (BOX3, 600 W,
N=1, `executed-not-qualified`, `pro-single-day11/`):
```text
(a) run-gen default:            prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH   [expert-gpu-slru] slots=90705 allocated_bytes=78021538440 evictions=0
(b) run-spec gpu-bytes=6881344: === SELF-CONSISTENCY PASS ===   [expert-gpu-slru] slots=8 allocated_bytes=6881344 evictions=674569
(c) run-spec gpu-bytes=6021176: REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum (requested 6021176, minimum 6881344, ceiling 77968398729)   exit 2, collector refused
--validate: cells: 3, failed_commands: 0, refused_commands: 1, qualification: false
```
Push note: the pre-push public-boundary gate flagged the two gzipped 52 MB SLRU trace logs of cell (b)
(`provider_name_aws`, "first hit line 1810" of the compressed byte stream) while the checkout `check` saw nothing in
them; the lead first pinned both in the allowlist (`f5c5f592e`), CI's allowlist-drift step then called the pins stale.
Root cause (`tools/check-public-boundary.py`): the checkout scan prefilters candidates with `git grep --text -P` over
raw bytes and only then runs the text regex on `decode(errors="ignore")`; the commit and ref scans (the hook) skipped
the prefilter and scanned every blob's ignore-decoded text, where dropped undecodable bytes glue their neighbours into
a provider name that exists in no byte view. A first attempt (`errors="replace"`) was wrong the other way: it surfaced
four `final-logits.f32le.gz` receipts already on `main` (the provider name as raw bytes bounded by replacement characters). Fix in
integ8 (`tools/check-public-boundary.py` `raw_bytes_prefilter`): commit and ref scans ask the same raw-byte question
first (each rule source compiled as a bytes pattern; a source that will not compile keeps the blob a candidate), so
both halves judge one candidate set. Unit test `test_commit_scan_uses_the_checkout_scans_raw_byte_prefilter` (glued
bytes: text scan alone would report, prefilter refuses, `evaluate_content` None; gzip of clean text None; `\0` plus a
real needle still a finding; every shipped rule compiles as bytes). The two pins removed. Gate suite, `check`,
`verify-allowlist` and the CPU battery rerun on the final tree (`integration-day11/integ8-boundary/`).
Door doc item 3: first half landed, scale admission still pending.

## Lane A day 10 (`d9d424398`, pushed by the lead with the logged override; engine files in range are `main`'s)
Storage `Err(Busy)` flake classified **race in the test harness**, class (b) of ruling 8, with a strace line: closing a
lock descriptor does not release the `flock` while a sibling test's `Command::spawn` sits between
`clone3(CLONE_VM|CLONE_VFORK)` and `execve` (the child holds a copy of the fd table), so a fresh `try_lock` on
`.catalog-owner` or `.ownership-gc` reads `EAGAIN`, which the engine correctly reports as `Busy`
(`day10-flake/strace/attempt-1.strace`, thread `close(23)` at `.684756`, sibling `clone3` at `.684746`, `execve` at
`.684888`, `flock(...) = -1 EAGAIN` at `.684933`). Not an ordering bug in `object_store`/`catalog`/frozen conformance
(untouched); not rig-specific (the same `:471` failure sits in B's rented-5090 log; O_DIRECT on this tmpfs opens fine).
Fix, tests only: `tests/storage/mod.rs` process-wide `RwLock` fence, directories hold it shared, the four spawning tests
hold it exclusively; no sleep, no retry, no relaxed assertion. Before: 3 of 15 default-thread runs failed; after: 0 of
30 (5 default, 5 `--test-threads=1`, 20 stress), all under the CPU quota. CPU gate exit 0 (fmt, tier+kv 265 tests,
clippy tier/kv/engine, check-flags, diff-check). A's note for the lead, no change made: a production process that
spawns children while another thread re-locks a store file on a fresh descriptor would see the same transient `Busy`
(open item for the tier storage owner; memra-server spawns no children on that path today).

## Lane B day 12 (`1f66f008d`, pushed by the lead with the logged override; engine files in range are the gate binary)
Ruling 6 implemented: `reclaim_contract::series_verdict` (N >= 5, class `one-time-driver-mapping-metadata`, (a) to (c)
every cycle, restore bit-identical every cycle, drift 0; an exact `none` series is true without the classified label;
pooled `not-applicable-pooled`); `write_cycles` writes `series_min_cycles=5` and `series_label`; the status line prints
the label only from that verdict; the per-cycle line stays `false` for a nonzero residual. One tightening beyond the
day-11 AND, stated in B's DAY12.md: exact cycles over a drifting baseline stay `false`. Cells rerun on both cards with
the day-12 build, bytes identical to day 11 (residual 2,097,152 B each cycle, drift 0):
```text
one RTX PRO 6000 Blackwell: ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles) committed=32768 generated=128
RTX 5090 Laptop:            ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles) committed=32768 generated=128
RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 residual_first=2097152 residual_last=2097152 g1_reclaim_qualified=true
```
PRO continuation matches the frozen target-card bundle; the laptop has no frozen bundle (tokens match the rented-5090
bundle, logits and state differ), its continuation identity is in-process only and stated next to the label, not
folded into it. `verify-day12.py --require-complete` reproduces both labels. Tests: new `series_verdict` unit test (N=4
false, N=5 with one drifting cycle false, N=5 growing false, other classes false, differing restore false, N=5 exact
`none` true, pooled), `tests/reclaim/day12.rs` red arms on both cards' committed bytes, `test-day12.py` 11 arms; the
decision record carries the "Day 11 to 12" section (evidence table, ruling 6 text, the 8k open item). Gate status now:
**G1 holds at 8k (exact) and 32k (classified) on both card classes.** Still executed-not-qualified development
evidence; decide-by 2026-10-04 stands: promote the door to the naked default for the tiered materializer only with the
serving-shape gates run, otherwise delete it.

## Lanes
- B day 11 and day 12 sealed and pushed (`1f66f008d`); merged into integ8. B next: nothing until the serving-shape gates are scheduled.
- C day 11 sealed and pushed (`f5c5f592e`); merged into integ8.
- A day 10 sealed and pushed (`d9d424398`); merged into integ8. D sealed. E docs folded into #576. F idle.
