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

Mac CPU battery: fmt, tier macOS and Linux-target all-target checks, tier tests,
strict tier clippy, strict Linux-target engine clippy, flags, whitespace all pass
in `raw/day9-cpu/20260920T155516Z/`. Linux-target engine checks use DOCS_RS
placeholders, not native execution. The initial full-file SLRU source hash failed
after adding the assertion; regenerated fixture changes only its source digest,
not any of its 2,013 frozen decisions. Initial failing checks are retained.

## Native cells so far

Frozen runtime source is recorded in `pro-single-day9/build/source.commit`
(`148e7f0e9`, integ5 merge), alongside build exit/log, binary hashes and artifact
hash. Subsequent assertion-only compilation does not overwrite those binaries.
All cells use prompt `[55,88,13]`, `MEMRA_NGEN=32`, `MEMRA_MOE_RESIDENT=0`.
The fresh same-card controls are the only comparison tapes.

| Cell | Verbatim gate outcome | GPU evictions | Host evictions | Physical reads |
|---|---|---:|---:|---:|
| Default native gen | MATCH | not instrumented | n/a | n/a |
| Default native spec | === SELF-CONSISTENCY PASS === | not instrumented | n/a | n/a |
| Default banked gen | MATCH | 0 | 30,704 | 30,720 |
| Default banked spec | === SELF-CONSISTENCY PASS === | 0 | 31,472 | 31,488 |
| 8 GiB banked gen, first completed | MATCH | 12,091 | 22,061 | 22,077 |

Generation quote: `prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH`.
Default banks hold the working set: zero GPU evictions is expected, not pressure
qualification. 8 GiB is 9,986 GPU slots including eight-byte tail padding per
860,160-byte record; host bank remains 16 records, not 8 GiB.

## In progress / handoff

Repeated canonical-lock refusals are preserved, never called GPU failures.
Lead authorized a bounded 60-second retry cadence for up to 60 minutes and
pushing the completed four-cell pressure set together. `run-day9-pressure.py`
uses only the canonical collector for every attempt; names are never reused.
`STATE.md` identifies the durable remote run. Final pressure/verifier results
will be appended after collection; absent cells remain pending.

`BUDGET-REFUSAL.md` proposes checked typed pre-allocation refusal instead of the
legacy eight-slot clamp. No flag behavior changed. Day-nine work so far is
approximately 0.6 agent-hours; prior cumulative hours are unavailable, so total
consumption against the eight-day lane budget is not invented.
