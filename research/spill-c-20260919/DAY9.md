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
