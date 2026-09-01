# Ornith exact-width-16 source-verbatim pair-owner study

Date: 2026-08-25

Base source: `b40bd07c82fdbc5f5c200c8d3b0ab3310629c1f8`

Artifact: `Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF`,
`Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf`, SHA-256
`72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3`.

## Question and invariant

The exact-width B=16 tier moved the trunk onto healthy batched kernels, while
the MoE remained the dominant wall. This lane tested whether expert-owner
ordering could improve repeated NVFP4 row locality without reviving the cached
CSR-NVFP4 arithmetic, whose output depends on batch composition.

Every accepted candidate calls the existing `expert_dot_g_v` helper on the
original global NVFP4 row pointer for every routed pair. No decoded weight,
partial sum, or activation arithmetic is shared across pairs. The rows program
is the bit oracle.

## Correctness

The final owner-ordered per-pair form is bit-exact on both tested sm_120a card
classes. A realistic synthetic fixture uses Ornith's 256 experts, top-8 routing,
512-wide expert FFN, and 100 distinct experts across 128 pairs at B=16:

```text
B=2,4,8,9,12,16: bit_mismatch=0
B=1 composition_mismatch=0
ALL GREEN
```

On one rented RTX PRO 6000 Blackwell, the real artifact passed decode-batch
gate2 and gate3 at B=12 and B=16 for 32 steps: full logit-bit identity versus
isolated B=1, sampled/lean composition, and the penalty-dispatch tooth. These
are short pass/fail cells and remain valid despite the concurrent-tenant timing
exclusion below. Receipts: `raw/pro6000-card0/correctness/`.

The cached `moe_gate_up_silu8_dev_q8_csr_nvfp4` path was never re-admitted.

## Candidate ladder

### One owner warp, 32 live accumulators

The first pair owned an expert/output row. One warp walked feature groups
outside the pair list and retained gate/up accumulators for up to 16 pairs. It
was bit-exact but lost on a clean RTX PRO 6000 window:

```text
arm      B=8 tok/s   B=16 tok/s
base-a      670.6        504.4
cand-a      669.6        477.6
cand-b      673.4        478.2
base-b      672.9        520.9
```

B=16 center: 477.9 versus 512.7 tok/s (-6.8%). The Spot host was reclaimed
after this window, so its instance-store telemetry was lost; the result lines
above were captured before disconnect.

### Shared raw-row staging

Calling the original helper on a shared pointer is invalid because its streaming
loads require a global address. A shared-load arithmetic twin ran, but failed
the bit gate by thousands of elements. It never reached target timing.

### One owner warp, two live accumulators

Moving the pair loop outside the feature-group loop removed the accumulator
array but retained serial pair service:

```text
arm      B=8 tok/s   B=16 tok/s
base-a      668.0        523.4
cand-a      667.4        499.9
cand-b      678.6        497.3
base-b      not run      not run
```

The available clean B=16 comparisons are -4.5% and -5.0%. The Spot host was
reclaimed before the final arm.

### Owner-ordered ordinary pair warps

The final form builds one expert-major list of packed `(expert,pair)` indices,
then retains the rows geometry: one ordinary warp per pair/output row, the
original global-row dot helper, and the original per-pair output location.

On the clean local RTX 5090 Laptop GPU, five Nsight Systems timelines at the
realistic B=16 geometry measured:

```text
component                         median us
rows gate/up                         158.175
owner-ordered gate/up                160.383
owner-order setup                      5.792
owner total                          166.495  (+5.3% vs rows)
```

This is a valid local-hardware negative.

## Discarded shared-box target timing

Two B=8/B=16 process-level ABBAs, an all-width screen, and same-process CUDA
events were captured on card 0 of a two-card RTX PRO 6000 host. A separate
tenant was actively computing on card 1 throughout. Memra's measurement law
requires scored work on a multi-card box to serialize behind one owner because
the cards share the host and PCIe regime. Therefore every timed result in
`raw/pro6000-card0/perf-abba1/`, `perf-abba2/`, `curve-screen/`, and
`kernel-event-n5.txt` is retained as raw observation but **discarded as
performance evidence**.

The discarded process windows were visibly unstable: base B=16 ranged from
472.1 to 502.5 tok/s. The discarded event observations showed rows near 68.6 us
and owner near 75.2 us, but they do not establish a clean RTX PRO verdict.
The concurrent-tenant observation is recorded in
`raw/pro6000-card0/CONCURRENT-TENANT.txt`.

## Verdict

Do not merge the runtime flag, kernel, FFI wrapper, or dispatch arm. The
source-verbatim owner forms are correct, but the clean local form loses and the
only clean RTX PRO owner forms also lose; the final pair-ordered form still
lacks an isolated PRO timing gate. The shipped rows path remains authoritative.

The exact candidate is banked as `candidate-b40bd07c.patch`, bound to the base
SHA above. `run-pro6000-card0.sh` is the original target harness. Reconsider
the pair-ordered form only on an otherwise-idle PRO box under the canonical
`/tmp/memra-gpu.lock`; do not infer a target verdict from the discarded
shared-box window.
