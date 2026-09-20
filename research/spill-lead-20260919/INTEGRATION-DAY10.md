# Integration — days 9–10 on the target card (`lane/spill-integ6-20260920`)

Lead: @agent-c07799 (old session, under the owner's "if something can merge, merge it"; the new-rig session was
told via HANDOVER-20260920 §LIVE COORDINATION). Base: `origin/main` `dbf88d46` (#568 merged). Lane tips merged:
A `b3dc864c`, B `7d213551`, C `90c7e68e`, D `a3aff2ae`, E `483f7d3d`. All merges clean.

## What this integrates (rented RTX PRO 6000 Blackwell — the target card class — 600/600 W, collector-locked,
## `executed-not-qualified`)
- **A day 8–9**: native conformance ALL PASS on the target card; six exact roundtrips 4 KiB–256 MiB; eight O_DIRECT
  storage cells byte-exact on a real block device (label: `block-device ext4 (virtio; NVMe ancestry provider-claimed,
  not proven)`); pread baseline N=1 (io_uring screening targets in `IO-BASELINE.md`, io_uring still deferred); and the
  **canonical v1.3 schedules bound natively**: `PASS v1.3 device_hand_back native CUDA`,
  `PASS v1.3 transfer_source_retirement native CUDA` — per-side retention graphs + destination charge past
  acknowledgement; frozen schedules unchanged (`V13-BINDING.md`).
- **B day 10**: BOX3 frozen baselines 8k/32k (`BOX3-BASELINES.json`); **`ACTIVE-8K G1 PASS` twice** on the target
  card (empty-plane swap, and native direct construction with 34 VMM planes); pooled control
  `not-applicable-pooled`; **32k**: bit-identical, one 2 MiB residual on this card too, mapped-VA release measured
  at 0 B → class stays `unclassified`, **not G1 PASS**. Allocator injection (`Cache` constructs VMM planes) landed
  behind the same door.
- **C day 9/10**: target-card baselines + banked experts (default and 8 GiB, gen + spec, ON/OFF) all `MATCH` /
  `SELF-CONSISTENCY PASS`; eviction counts identical to the 5090 run (deterministic SLRU trace); the existing
  `MoeSlotCache` owner-thread door audited (`Engine: Send + Sync` assertion); `BUDGET-REFUSAL.md` with three options
  for the lead decision on `MEMRA_MOE_SLOTS` (clamps to 8, cannot express refusal).
- **D day 10**: `pro-single` profile reviewed (schema enum fixed, 7 tests); **G2 scored** N=10/arm (5 AB + 5 BA),
  5 sizes × 2 directions, 200 visits ≥497 ms, 250 ms telemetry, 37–41 °C — pinned wins from 64 KiB up (1 MiB
  32.7 vs 17.5 GiB/s; 256 MiB ≈53 vs ≈37), 4 KiB within noise; `pp-transport-smoke PASS` (single-card loopback
  only); D archive `--validate` clean; bootstrap receipt: 30 exit-0 steps + one allowed prerequisite exit (not
  "all green").
- **E**: docs/ROUTER + INDEX alignment for v1.3.

## Battery (this tree, Mac, offline)
262 tier/kv tests · clippy `-D warnings` (engine, server, tier, kv, gguf all-targets, Linux target, `DOCS_RS=1`)
clean · fmt · flags census · publish census 12/12 · docs registry census · perf board current · collector Python
suite 85 passed · `git diff --check` clean.

## Gates
G1: 8k PASS on both card classes; 32k held by the unclassified one-granule residual (same on both cards; not VA
reservation). Suggested next probe: repeat demote/restore cycles in one process — a non-growing residual is
one-time driver metadata (non-leak). G2: first scored envelope exists (development evidence, one card class).
G0, G3–G7 unchanged.

## Review round on PR #573 (`55895c18`)
CI: every job pass. Automated review: six inline findings, all documentation drift against the merged tree, all
valid, fixed in the follow-up commit: `docs/TESTING.md` tier-transfer-gate section (canonical v1.3 is now bound
and passing; eleven-line verdict block; per-side pins), `--kv-allocator vmm` mechanism (direct construction;
containment is a call-site policy since `Cache::new_with_allocator` / `KvDev::alloc_vmm_u8` are public), the
mapped-VA probe receipt surface, the three probe-derived residual classes, the day-10 **zero-residual tightening
(e)** and the `verify-day10.py` pointer; `docs/decisions/KV-PHYSICAL-RECLAIM.md` scope (direct construction
landed; the surface the decide-by promotes or deletes) and criterion (e) recorded as an explicit tightening.
