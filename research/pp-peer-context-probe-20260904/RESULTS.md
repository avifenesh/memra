# PP peer qualification rerun

Date: 2026-09-04
Issue: memra#188
Source: `378b38910f37d38e09b284950757cc8b2281a39d`

## Verdict

The reported PP failure does **not** reproduce on the fresh non-production 4x RTX PRO 6000
Blackwell Server Edition host. Memra's fail-closed PP byte probes are correct and remain unchanged.
The earlier machines stay classified as host-specific native-peer failures: a successful capability
query or a different CUDA copy program cannot authorize native PP after Memra's exact production
primitive returns wrong bytes.

The rerun did find and fix a separate qualification defect. `tools/box-health.sh` treated a normal
P8 idle-link downshift to PCIe Gen1 as the same fault as a live workload stuck below the card's
maximum generation. It now records the idle state, drives its existing all-card peer-read ladder,
then immediately requires every active link to reach maximum generation and width. A below-max
reading outside P8 is still an immediate hard failure.

## Evidence

- CUDA driver 595.91.07, CUDA toolkit 13.0, four 96 GB Server Edition cards.
- A separate-process retained-primary-context matrix exercised all 12 ordered card pairs at 16 KiB,
  1 MiB, and 64 MiB. It compared producer-stream and consumer-stream forms of both
  `cuMemcpyPeerAsync` and peer-mapped `cuMemcpyDtoDAsync`: 144/144 cells preserved every byte.
- `pp-transport-smoke` on the production placement `MEMRA_PP_DEVICES=1,2,3` passed the fixed 16 KiB
  gate in both directions across both boundaries, all 16 production-slot probes at 1, 8, 16, and
  4096 tokens, and four alternating live boundary round trips. Every mismatch count was zero.
- The original box-health run reported four hard failures because all four idle P8 links read Gen1.
  During the probe workload every card trained to P0, Gen5 x16. The fixed gate reports four active
  Gen5 x16 passes, zero hard failures, and zero warnings on the same boot.
- The single-card rig takes the new P8 deferral path without inventing a fabric result. With no peer
  ladder to wake its link, active generation remains an explicit warning for workload telemetry.
- `tools/local-ci.sh --perf` exited 0: build, clippy, unit suites, 107-cell kernel check, correctness,
  c=64 serve stress, cache/spec gates, ignored GPU tests, and the available perf cell passed. The
  unrelated flags-census self-test warning observed on the base is tracked separately as memra#190.

## Scope

- No PP transport, ordering, probe, or fallback behavior changed.
- `peer-copy-direction-probe.cu` is a manual correctness diagnostic. It separates copy-engine
  direction and API form; it is not a benchmark and does not weaken any serving gate.
- Provider identity, machine identity, location, and fleet state remain in the private deployment
  repository rather than this public engine lane.
