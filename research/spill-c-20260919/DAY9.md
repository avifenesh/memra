# Day nine — target-card banked residency

Repository `avifenesh/memra`, active lane `lane/spill-c-20260919`.
Target-class evidence: one RTX PRO 6000 Blackwell, 96 GB, configured/max 600 W.
Development correctness only; no performance/default, support-state, PP runtime,
or production qualification claim. No cross-card timing comparison.

## Wiring correction and compile gates

The inherited lane already contains the owner registry wiring; it is not still
the rejected 22-error patch. `MoeSlotCache` stores only an optional
`ExpertBankProxy` and `ExpertLeaseToken`; owner-only services/backing stay in
thread-local storage. Demand validates ThreadId; migration refuses WrongOwner.
Dispatch, admit and preload use the bank; speculative prefetch cannot bypass it;
frozen/parallel unsupported arms refuse. The native installer remains the existing
explicit default-OFF `--experts-via-tier` qualification door, not a new runtime flag.

**Engine Send+Sync, cache Send, pread Receiver !Sync.** Engine's Mutex makes the
cache's Send bound sufficient. Day-nine adds an ordinary-library-build compile
assertion at this boundary. The first check overconstrained the cache to Sync;
that failing log is retained. Corrected assertion passes Linux-target engine
lib/run-gen/run-spec strict clippy, including actual scoped PP-worker spawns.
Native strict release clippy also passes at `748903f73` with CUDA 13.2.
No unsafe Send/Sync impl, new dependency, new numerical program, or duplicate
bank integration was introduced.

Mac CPU battery: fmt, tier macOS and Linux-target all-target checks, **196 tier tests including doc tests**,
strict tier clippy, strict Linux-target engine clippy, flags, whitespace, frozen
trace replay and six verifier test methods all pass in
`raw/day9-cpu/20260920T160617Z/`. Tested source-file digests accompany the logs.
Linux-target engine checks use DOCS_RS
placeholders, not native execution. The initial full-file SLRU source hash failed
after adding the assertion; regenerated fixture changes only its source digest,
not any of its 2,013 frozen decisions. Initial failing checks are retained.

## Native cells

Frozen runtime source is recorded in `pro-single-day9/build/source.commit`
(`148e7f0e9`, integ5 merge), alongside build exit/log, binary hashes and artifact
hash. Subsequent assertion-only compilation does not overwrite those binaries.
All cells use prompt `[55,88,13]`, `MEMRA_NGEN=32`, `MEMRA_MOE_RESIDENT=0`.
Every cell is **N=1**: one execution per arm, `executed-not-qualified`
development evidence on the target card class. No median appears in this report.
The fresh same-card controls are the only comparison tapes.

| Cell | Verbatim gate outcome | GPU evictions | Host evictions | Physical reads |
|---|---|---:|---:|---:|
| Default native gen | MATCH | not instrumented | n/a | n/a |
| Default native spec | === SELF-CONSISTENCY PASS === | not instrumented | n/a | n/a |
| Default banked gen | MATCH | 0 | 30,704 | 30,720 |
| Default banked spec | === SELF-CONSISTENCY PASS === | 0 | 31,472 | 31,488 |
| 8 GiB banked gen, first completed and final set | MATCH | 12,091 | 22,061 | 22,077 |
| 8 GiB banked spec | === SELF-CONSISTENCY PASS === | 63,996 | 73,966 | 73,982 |
| 8 GiB native gen | MATCH | not instrumented | n/a | n/a |
| 8 GiB native spec | === SELF-CONSISTENCY PASS === | not instrumented | n/a | n/a |

Generation quote: `prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH`.
Default banks hold the working set: zero GPU evictions is expected, not pressure
qualification. 8 GiB is 9,986 GPU slots including eight-byte tail padding per
860,160-byte record; host bank remains 16 records, not 8 GiB.

**Bank budget per cell.** Host bank: 16 records of 860,160 bytes (13.1 MiB) under
the 256 MiB default `--expert-bank-host-bytes` ceiling in every banked cell. GPU
slot cache: default cells auto-size to 0.85 of free VRAM under the 0.80 hard
ceiling, 90,705 slots (72.7 GiB) in gen and 90,643 slots (72.6 GiB) in spec; the
8 GiB cells force `MEMRA_MOE_SLOTS=9986`, 8,589,637,648 bytes (8.00 GiB)
allocated (`allocated_bytes == slots * 860168` is verifier-checked). Native OFF
cells build the same slot cache without the bank and print no `[expert-gpu-slru]`
line, so their eviction count is not instrumented.

## Final seal and handoff

Repeated canonical-lock refusals are preserved, never called GPU failures.
Lead authorized a bounded 60-second retry cadence for up to 60 minutes and
pushing the completed four-cell pressure set together. `run-day9-pressure.py`
uses only the canonical collector for every attempt; names are never reused.
The final set acquired the lock on its second generation attempt, then completed
all four cells. No campaign remains running.

`verify-day9.py` replays **PASS**: eight required successful captures, raw and
250 ms telemetry hashes, 600 W envelope, canonical locks, exact commands, full
32-token tapes, all K1–8 rows, successful owner retirement, trace-derived read
and eviction counts, source/artifact/binary identities, and post-cell binary
hashes. Gen tapes agree across default/8GiB and ON/OFF, separately from the spec
class; all spec tapes and acceptance rows agree across those four arms. The
8GiB banked gen/spec re-read counts are **4,389 / 53,684**. The verifier never
compares against old-card tapes or interprets collector execution as qualification.
Six verifier test methods exercise malformed status, failed exit, timeout,
wrong command/power/cadence/lock, log hash drift, and rehashed missing tapes.

Runtime build/cells are explicitly at `148e7f0e9`; native strict clippy and the
new compile assertion are at `748903f73`. The only engine-source delta between
these revisions is that compile-time assertion, not wiring or numerical math.
The native build does not pretend to be a rebuild of a later receipt-only tip.
The complete pressure set and postcheck were pushed at `f157269e1`; earlier
cells were pushed individually. Hooks ran without bypasses.

`BUDGET-REFUSAL.md` proposes checked typed pre-allocation refusal instead of the
legacy eight-slot clamp. No flag behavior changed. There is no missing assigned
single-card correctness cell. Strict GPU-budget refusal remains a follow-up
proposal, and PP runtime migration/multi-card qualification remain outside this
single-card evidence (wrong-thread access still refuses, not RPC).

Day-nine effort is approximately **0.8 agent-hours**, under the 3.5-hour bound.
Prior cumulative hours are unavailable, so total consumption against the original
eight-day lane budget is not invented. The primary local lane and its remote
build remain active integration handoffs, not abandoned scratch. Raw receipts
are fully copied and committed; no temporary GPU process or campaign remains.

## Day-ten close (2026-09-20, from 16:10Z)

- Resync before acting: origin carried two commits from a parallel session
  (`f157269e1` complete pressure receipts, `066afe918` seal); fast-forwarded first.
- Merged the lead's local `lane/spill-integ5-20260920` (`5b3d0b08e`, integ5
  replayed onto main `f79b3e57`) as `a413937f2`, no conflicts. It brings the
  lead's `model_memory*`, `hybrid.rs`, `lib.rs`, `pp.rs` engine changes and no new
  `MEMRA_*` read.
- Independent local replay on the merged tree: `verify-day9.py` **PASS** (eight
  cells, verdicts identical to `pro-single-day9/VERDICT.json`), `test-day9.py`
  six red arms OK. An rsync dry-run of the remote `c-day9` tree against
  `pro-single-day9/` itemizes only mtime and permission flags: byte content is
  identical. The five refused attempts left 53-byte `REFUSED` console logs (kept)
  and empty directories, which git cannot represent.
- Push gate on the merged tree: `cargo fmt --all -- --check` clean;
  `cargo test -p memra-tier -p memra-kv --offline` 261 pass;
  `DOCS_RS=1 cargo clippy -p memra-engine --offline --all-targets -- -D warnings`
  clean. `git push origin HEAD` was **refused by the perf-ci pre-push gate at
  `a413937f2`** (engine files touched after the last perf-ci battery: the lead's
  files above). No `MEMRA_SKIP_PERF_CI`, no `--no-verify`; the lead pushes the
  branch from the shared `.git`.
- `MOE-SLOT-CACHE-DOOR.md`: what is landed, what is pending before the door can
  sit behind the tiered materializer, decide-by 2026-10-04. No code step was
  implemented: every candidate needs an `Engine` to test, awaits the budget
  decision, or is a transport.
- `BUDGET-REFUSAL.md`: three options (keep clamp plus door budget; lift clamp;
  typed plan-time refusal) with fail-closed behavior and test shape each, marked
  lead decision needed. No default changed.
- Repeatability cell: `run-day10-repeat.py` in remote tmux `c-day10-repeat`,
  receipts `/root/spill-receipts/c-day10/`, one N=1 8 GiB `spec-on` through the
  collector, 60 s waits, 60 min cap from 16:28Z. Lane B held the canonical lock
  at launch; every refusal is preserved. Outcome is appended below when the
  driver exits.
