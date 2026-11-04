# The GLM TP-2 plain route samples on the device on sm_100a (2026-11-04)

**Chosen:** on sm_100a builds, a GLM TP-2 plain request (loaded TP sidecars, no pipeline stages,
no speculative route) keeps its full-vocabulary logits on the root head stream and draws its token
there with the DSV4 temperature/top-k/top-p kernels, one u32 back per draw
(`glm5_tp_sampler::enabled`). Greedy requests use the existing raw argmax kernels.
`MEMRA_GLM5_TP_DEVICE_SAMPLE` is deleted.

**Rejected:** the host sampler as the default for that route on sm_100a. It stays the program on
every other build, and on sm_100a for every request the device route refuses (non-neutral
penalties, min-p, constraints, full-row capture, a route that is not TP-2 plain).

**Measurement that settled it:** `research/glm5-tp2-gpu-sampler-20260908/RESULTS.md`, the 2x B200
pair, CUDA 13.1, one metered server binary for both arms, the launcher's TP-2 exec block. Five
served p32k OFF/ON pairs: 81.317 -> 91.250 wall tok/s (+12.22%), 11.894 -> 10.610 server ms/token.
The 160-token greedy SHA twins match in both windows. Build, fmt, clippy, the CPU suites and the
ignored GPU oracles (a 154,880-logit row, tied maxima, both greedy selectors, top-p membership,
seed and position determinism, boundary-row identity) pass. The owner's 2026-11-04 ruling makes
positive-receipt winners defaults keyed to the receipt's card, with the switch removed.

**What changes for a user:** sampled tokens on that route take the numeric class
`device-f64-exp-tree-cdf-v1` (f64 exponentials, position-keyed SplitMix64) instead of the host
sampler's f32, stateful draw. Sampled identity across the two classes was never claimed; greedy
ids are the same.

**Scope:** one card class, the B200 pair, keyed on the build as the other sm_100a defaults are. No
RTX 5090 or RTX PRO 6000 program changes.
