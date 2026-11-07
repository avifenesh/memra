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
ids are the same. Because the two classes differ, one request never crosses between them: a
sampler error on the first draw keeps the whole request on the host sampler, and an error after a
device draw ends the request with an error instead of finishing it on the host sampler.

**Scope:** one card class, the B200 pair, keyed on the build as the other sm_100a defaults are. No
RTX 5090 or RTX PRO 6000 program changes.

**Walk:** the receipt ran the launcher's exec block with the TP-2 expert split, the symmetric walk
and its graphs, and every accepted process logged TP graph engagement. The default is held to that
walk: `HybridModel::glm5_tp_device_sample_supported` applies the symmetric walk's own engagement
rule over every layer (symmetric glue loaded, `MEMRA_GLM5_TP_EXPERT_SPLIT=1`,
`MEMRA_TP_AR_1STAGE=1`, a real two-device pair). A TP-2 load that would take the root-orchestrated
walk keeps the host sampler, because no row measured the sampler there.

**Open at integration:** the B200 lane (fix4/b200-tp2, fe66a24a0d) deletes the symmetric walk and
rekeys the TP runtime lookup on the mixer sidecars, and its message says the symmetric walk was
never measured engaged. This receipt measured it engaged. If that deletion lands, the sampler has
no measured walk left: either it goes with the walk, or the integration records that its default
reaches the root-orchestrated walk with no receipt. That is the owner's call; no B200 row can be
measured now.
