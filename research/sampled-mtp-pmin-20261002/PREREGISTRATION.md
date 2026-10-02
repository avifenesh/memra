# Finite sampled MTP experiment, issue #673

Registered before native execution. One pinned Qwen3.5-9B NVFP4-MTP GGUF,
SHA256 52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de,
current runtime source 2873dd4ca37faa15cd4261e7b398926cced30106, one RTX 5090
Laptop. Use the binaries already built for #543; record their exact hashes.
The new collector changes no runtime source. Confirm the runtime input diff is empty.

Six statistical arms: plain sampling, default PMIN=0, PMIN=0.5 graph, PMIN=0.5 eager,
and forced-zero PMIN=1.1 with PMIN0 enabled on graph/eager. The 1.1 value is a
fault/edge instrument that guarantees confidence below threshold; it is not a
serving recommendation. Each arm uses a fresh process, 512 serial prefix-cache-off
requests, eight output tokens, and disjoint fixed seed ranges. Temperature 0.8,
top_k=20, top_p=0.95, min_p=0, neutral penalties. The fixed prompt asks for a sequence
of red/blue/green/yellow, and is frozen in the collector. All responses and token
IDs are retained, including early stops (pad the statistical vector with -1).

Compare each arm with plain sampling separately at all eight token positions.
Statistic: categorical total variation. Null calibration: 1999 shuffled-label
permutations, RNG seed 673. Familywise alpha 0.05 across 40 comparisons, Bonferroni
threshold 0.00125. A result means detected/no detected marginal difference at this
finite sample size, not proof of the full autoregressive law. The old 0.8->0.96
counterexample supplies a separate CPU power-control red arm which must reject.
No threshold or sample-count changes after seeing output.

Native probes use the same fixed prompt, seed 7, K=3 and 64 generated tokens on the
five MTP arms. Require actual graph_s/eager engagement; positive graph/eager token
identity; observed draft acceptance, rejection-residual and full-accept bonus;
a shorter positive-PMIN draft at the same control context; zero-draft rounds on
both forced-zero arms. Existing run-spec rejects zero total acceptance: only its
exact no-draft-accepted exit is admissible for the intentional zero-draft control,
and seeded self-consistency must still pass. All other errors fail. Existing
sample-check supplies device primitive checks and the unchanged exact rational
spec_stop suite supplies the algorithm-level CPU proof.

Record full request times and existing synchronized draft/verify phase clocks.
Confidence uses the raw head-row max and its 4-byte read, included in draft time;
this experiment does not invent a separate confidence-read timer. No performance
winner or default decision is made from sequential diagnostic arms.

Vendor HTTP cells: fresh default, positive 0.5, then rollback-to-default processes,
with cold/warm cache and four concurrent requests. Omit every sampling override,
including seed. Require valid response/usage, actual sampled MTP on cold/warm,
cache-grid accounting and observed concurrency admission. Different random outputs
are expected and are not compared for byte identity. They are bounded mechanics
receipts, not completed-answer quality scores.

One GPU cell, timeout 1200s, minimum free VRAM 20000MiB, loopback port 18121. Paired
arms remain together; no compilation inside the GPU job. All process cleanup
finishes before returning the lease. The collector uses a private CUDA cache.

The artifact has one MTP block. Multi-head chain graphs are not reachable and
remain unqualified here. No GLM, DFlash, Step, larger model, other topology or
serving configuration is inspected or qualified. The registered PMIN=0 serving
default remains unchanged. No issue is closed on this local subset.

## Instrument correction after attempt 1

The native CLI graph/eager/cutoff/accept/residual/bonus/zero-draft probes and all
three vendor HTTP cells passed. The statistical collector stopped after one
response per arm because API-key authentication defaults the response dialect to
OpenAI, which omits native token IDs. All six responses were retained. This is a
collector response-format error, not a native-model failure or a distribution
result. Device sample-check and six exact rational CPU tests also passed.

The statistical retry explicitly sets MEMRA_COMPAT=native while retaining auth.
It reruns only the six incomplete 512-request arms with identical prompts, seeds,
sampling parameters, sample counts and test thresholds. Already-passed CLI and
vendor cells are reused with their original raw receipts, not rerun. The retry's
result is explicitly scoped to statistical marginals. No output/quality sweep.

## Vendor-profile correction before the missing vendor cell

Post-listener traces revealed that the earlier no-override HTTP requests used
Memra's generic defaults (temperature 1, top_p 1, top_k 0), because the inline model
launch supplied no model-specific sampling metadata. Those records remain valid
generic-default controls; they are not vendor-profile qualification.

Use Qwen's thinking-mode general-task recipe from official README revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a: temperature 1, top_p 0.95, top_k 20,
min_p 0, presence_penalty 1.5, repetition_penalty 1. Install those defaults through a
hash-bound temporary MEMRA_MODEL_METADATA file, keeping the request free of
sampling and thinking overrides. Use the real chat-completions surface and the
artifact's default thinking template. The 64-token cap is a mechanics budget;
reasoning-only output at length is retained and is not a completed-answer claim.

Three fresh processes: default PMIN=0, experimental PMIN=0.5, rollback PMIN=0. Each has
one cold and one warm request plus four concurrent requests. Require loaded
metadata hash, post-listener sampled burst parameters matching the profile,
actual MTP and cache-grid accounting, and concurrency admission. A startup canary
or generic-default trace must fail the profile assertion. No successful CLI or
3072-request statistical cell is rerun. Declared vendor-only timeout 300 seconds,
minimum free VRAM 20000 MiB, port 18121. All original receipts and this correction stay
in the record. No serving default or runtime source is changed.
