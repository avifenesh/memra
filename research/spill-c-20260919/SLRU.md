# Default SLRU CPU model — day 4

Source: memra integration `020d2047` (engine source remains unchanged on this
lane). Implementation: `crates/memra-tier/src/bank/slru.rs`; host binding:
`BankService::with_slru`. **CPU conformance, not native cache qualification.**

## Precisely ported policy

Read-only `crates/memra-engine/src/moe_cache.rs` references:

- 194–246: per-class probation/protected, 80% protected cap, demote protected
  LRU to probation MRU; eviction probation first then protected.
- 537–553: ascending classes, free slots reverse-initialized then popped, cap
  `max(1, floor(count * 0.8))`. Caller supplies the native planner's result;
  `size_class_plan` at 432–480 is NOT replaced with a new sizing heuristic.
- 713–737: hit ordering stays unchanged while that class has free slots.
- 774–827, 841–866: search **all eligible free classes first**, then smallest
  fitting class's LRU; prefetch skips keep IDs; first miss admits to probation.
- 1100–1127: pending demand installs wait then publishes without an extra hit
  promotion; repeated resident demand promotes; resident lookup alone does not.
- 1186–1237, 1248–1253: pending slots stay outside both queues and cannot be
  victims; duplicate prefetch is a no-op; retired abort returns a slot to free.

The policy uses full BankId, never local `(layer, projection, expert)` identity.
It does not implement the opt-in LFU frequency/decay branch, frozen CPU/GPU
assignment, worker-read admission, pointer-row invalidation, or CUDA operations.
Those mechanisms remain unchanged natively and block a wholesale replacement.
VecDeque makes this CPU model readable; it is O(n) on promotion, unlike the native
intrusive O(1) list. **Do not replace the tuned native hot path with this model.**

## Reproducible synthetic decision trace

`python3 research/spill-c-20260919/slru-trace.py --check` reproduces
`fixtures/slru-synthetic.json`: independent Python transcription of the above
source semantics, fixed RNG seed 20260919, three explicit classes, **2013
operations plus a separate 256-demand serialized trace**. The fixture includes native-source SHA256, every action, outcome,
selected slot, evicted original identity and all class orders. The Rust test
replays it and compares every decision AND queue order. The native source is
pinned, not executed: a changed source forces review/regeneration, not silent
continuation of an old oracle. No checkpoint/router trace or performance data.

Observed arm counts: aborted 24, admitted 350, capacity 730, hit 142, noop 482,
published 129, reserved 156. Trace covers pending exhaustion, prefetch keep,
full/free-class behavior, size classes, oversized refusal, demotion and eviction.
Unknown DMA completion is modeled by **not calling abort_retired**; this is not
a delayed CUDA-event experiment. The initial source-pin test failed because the
test used the contracts' domain-separated checksum rather than raw SHA256;
fixed the test to use sha2::Sha256, not the fixture/source.

## Actual BankedResidency binding and accounting boundary

`BankService::with_slru` installs the policy before requests; default constructors
retain the previous hotness-cache behavior. Class capacity must fit cache_bytes.
A caller-governed metadata lease is pinned for the policy lifetime; premature
release returns Busy. Estimate: per-slot three maximum-sized encoded keys plus
1024 B node allowance, plus 4096 B fixed allowance. This is conservative CPU
accounting, not allocator/RSS calibration or a fixed CUDA slot reservation.

The existing common governor still charges each exact backing and request/slot
metadata. Eviction drops **residency**, not old consumer ownership. The new test
holds an evicted lease, reloads the same BankId to another charged allocation,
and proves releasing the old generation cannot erase the new cache entry. A second
test drives the actual BankedResidency through all 256 serialized demands,
compares hits, slots and every queue order to the Python oracle, verifies source
bytes and the 352-byte cache cap, and drains all governor charges. After
all consumers retire, bytes return to the metadata baseline; after dropping the
service and explicitly releasing metadata, every resource is zero.

The host adapter applies decisions in logical order at **successful publication**.
It does not reserve native GPU slots during enqueue and optional tickets have no
native keep argument. Therefore arbitrary overlapping host ticket schedules are
NOT claimed decision-identical to CUDA submission schedules. Standalone policy
reserve/publish/abort provides the exact pending-state seam for the native owner.
No policy decision changes a numerical program, and the host path rejects device
and pinned requests. Unsupported sizes may remain uncached exact host output;
native slot exhaustion must keep its current explicit failure behavior.

## Native swap map — future rig work, not applied

- `moe_cache.rs:730,774,797,841,852,857,1186`: use the proved transitions with
  existing intrusive lists and fixed slots (or qualify a tuned equivalent),
  replacing BlockId with full BankId. Retain LFU/frozen modes or explicitly refuse
  unsupported policy selection; never silently substitute default SLRU.
- `moe_cache.rs:1100–1155,1197–1237`: bind transfer ticket to reserved slot;
  wait before publication; no policy promotion by prefetch or speculative hints.
- `hybrid_forward.rs:19826–20037`: cached demand and prefetch must use installed
  BankSource plus owner ReadyView; original q8/f32 compute choice stays fixed.
- `hybrid_forward.rs:14831–14965,21212–21550`: staged/grouped paths retain exact
  per-record layout and all scale planes; Q2_K remains staged f32 dequant.
- `lib.rs:6465–6482`: one existing cache/governor owner, not another global cache;
  source installation at model load before dispatch (BANK-SOURCE.md).

GPU/serving byte, token, logits, unknown-event, graph-pin and same-window latency
receipts are still missing. Full Hy3 conversion remains **NO-GO**.
