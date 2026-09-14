# Device-resident narrow verification selection

2026-09-05. Default OFF (`MEMRA_DSV4_VERIFY_TOPK=legacy`).

Source inspection found that T<=8 verification still copied and CPU-sorted
every sparse layer/query's scores at <=4096 candidates. The earlier long
selector repair removed this boundary only above 4096. The short copy/sort
blocks whole-layer stream capture and serializes CPU/GPU work.

The new `device` arm uses native bitonic selection for this range. It shares
the original CUDA body but normalizes the integer sorting keys of both zero
signs: Rust's comparator treats -0 and +0 as equal before breaking ties by
index. Other callers retain the old raw-bit specialization; no score value is
rewritten. Non-NaN inputs, including infinities, have the same ordering as the
CPU witness. Large-history and wide-prefill dispatch remain unchanged.

`verify-topk-5090.log`: 45 CUDA/CPU oracle cells pass on the exclusively locked
non-serving 5090 Laptop GPU, with 1..4096 candidates, ties, signed zeros,
infinities, power-of-two edges, input and write guards. The old selector's
opposite signed-zero order is a positive distinction control.

`dsv4_c4_host_gate` now compares five composition arms for each case: device
residency/legacy selector, host residency/legacy selector, device/device,
host/device, then device/legacy again. It compares logits, sampled output and
rounds, every live cache class, suffix continuation, snapshots and dispatch
engagement. The complete two-card composed matrix now passes with binary
`0c7c2df525a399fbf9963178fa2819208a7b8b042efb0efa44c1b38014d8d355`.
Balanced sampled serving performance remains pending; no default was changed.

Full-layer/round graphs still need live position/compressor/cache metadata and
capture-safe copy targets. This change removes one measured-code obstruction;
it does not substitute small segment graphs for that requirement.

Primary sources checked today:
[NVIDIA graph capture/update contracts](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and [LMSYS V4 in-graph metadata](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/).
Upstream data-center GPU speedups are not inherited by the target SM120 pair.
