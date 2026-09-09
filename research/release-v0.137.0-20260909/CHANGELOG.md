# v0.137.0 release notes

Owner-approved release membership, 2026-09-09. Build and publication receipts are recorded in GO-QUALIFICATION.md.

- GLM host-cache images retain model-owned latent, TP-rank, draft and boundary planes through demotion and promotion. The opt-in startup pinned arena reserves the full configured budget before readiness and recycles whole-image leases without request-time pin/free (#325). `MEMRA_GLM5_TP_KV_HOST` remains OFF by default.
- Validated DFlash retained-prefix restores use the suffix workspace charge while preserving other admission charges. Carrier failures refuse the request instead of silently running a cold prime (#377).
- DSV4 admitted full-token replay cadence and aligned dense exact-tree tails default ON with exact `0` rollback (#374); profiling and gate refinements are included (#376, #382). These supply no GLM performance claim.

- #379 adds the owned speculative-prime fairness walker, `MEMRA_PRIME_YIELD` OFF (unset/`0`), `1` enables bounded peer service; decide-by 2026-09-22. Receipts: `research/prefill-fairness-20260908/{RESULTS,COMPOSITION,CHUNKS,SEAM}.md`. GLM5 adapter qualification remains separate.
- #392 changes `MEMRA_DSV4_MOE_M1_SPLITK` to ON (unset = `graph`); explicit `0` restores sktail in a fresh process with fresh graphs, seam review 2026-09-22. Owner-accepted numeric drift and performance receipts: [Darklanes #520](https://github.com/avifenesh/darklanes/pull/520); actual unset/0 engagement: [Memra #392](https://github.com/avifenesh/memra/pull/392), namespace `splitk-default-on-b589df9-gate-r1`, binary `ae91d10f4404759b6bd0b0c102eaa074aec27aa5b86a8b5c0a772c96de5201ee`. This supplies no GLM performance claim.

- GLM TP-2 gains opt-in GPU sampling (#378, merge `ff0937dd09b808cd97dde174de1b6aedff63a0b8`, `MEMRA_GLM5_TP_DEVICE_SAMPLE` default OFF, decide-by 2026-09-22). Five p32k OFF/ON pairs across two windows measured median wall throughput 81.32 -> 91.25 tok/s and server decode 11.89 -> 10.61 ms/token; 160-token greedy/top-k-one twins were byte-identical. Unset/0 with a restart restores host sampling. [Sampler receipts](../glm5-tp2-gpu-sampler-20260908/RESULTS.md).

No published performance number changes. Publicity: skipped, maintenance release.

Tag ancestry: v0.136.0 points to `55d83a7cf3948d6d173d28e3e1fd18008fdeb9bf`, its release branch commit. That commit is not an ancestor of main; its squash on main is `72aa777c3763a21281fb9c1c55b2499f2c14d1ee`. v0.137.0 is tagged on the release PR squash on main so git describe resolves the actual release.
