# SparseSpec-L feasibility for memra

Date: 2026-08-11

Lane: `lane/cx-sparsespec`

Scope: CPU-only, read-only research; no implementation, build, benchmark, or GPU run.

## Verdict

**SparseSpec-L: DOOR-OPEN, narrowly, as a default-OFF sparse-proposer experiment.** Its sparse
KV view produces proposals only; the same target model then verifies those proposals against the
complete KV cache. That is on the lossless side of memra's boundary: changing the proposer may
change acceptance and speed, but the full target remains the authority for committed tokens [E1].
This is the same boundary on which memra admits reduced-expert drafting but rejects reduced-expert
verification (`research/spec-landscape-20260810/SURVEY.md:498-512`).

The door would close immediately if an implementation evicted the dense KV, used sparse attention
for verification, or committed a sparse-draft token without the ordinary full-target accept or
correction rule. Memra's greedy contract is token identity with plain decoding, and its sampled
contract uses target/draft rejection correction to preserve the target distribution
(`crates/memra-engine/src/spec.rs:1-9`,
`crates/memra-engine/src/spec.rs:7441-7505`, `docs/FLAGS.md:63`).

**Entropy K-controller: DOOR-CONDITIONAL as a separate default-OFF experiment.** It needs draft
logits, measured draft/verify costs, and accept/reject feedback, not sparse KV specifically [E1].
It can therefore be tested over memra's own-trim MTP proposer, but it must beat the current request
policy, confidence cut, and accepted-run controller rather than a straw-man globally fixed K
(`docs/FLAGS.md:54-57`, `crates/memra-engine/src/spec.rs:6028-6046`,
`crates/memra-engine/src/spec.rs:6388-6414`).

Neither verdict is a performance claim. SparseSpec-L's published implementation shape differs
from memra's cache and attention path, and this lane ran no build, GPU gate, or benchmark
([E1, implementation]; `crates/memra-kv/src/lib.rs:213-232`,
`crates/memra-engine/src/lib.rs:9819-9845`;
`research/sparsespec-20260811/PROGRESS.md:28-30`).

## External evidence quoted

The primary paper is arXiv:2607.27735v1, retrieved 2026-08-11. Later references to [E1] identify
the specific paper section; the short fragments below are the verbatim evidence anchors.

- **[E1] SparseSpec-L paper:** verification “retains the complete KV cache”; the “target model
  itself as both drafter and verifier”; evaluation was “using Hugging Face Transformers”; and the
  controller is “only moderately predictive”.
  [pipeline](https://arxiv.org/html/2607.27735v1#Sx3.SSx1),
  [recall index](https://arxiv.org/html/2607.27735v1#Sx3.SSx2),
  [controller](https://arxiv.org/html/2607.27735v1#Sx3.SSx3),
  [implementation](https://arxiv.org/html/2607.27735v1#Sx4.SSx1.SSS0.Px3),
  [limitations](https://arxiv.org/html/2607.27735v1#Sx7)
- **[E2] Vegas reference engine:** drafts are “verified in parallel against the full KV cache”;
  the “same weights draft and verify”; it is a “fork of vLLM”; and its integration includes
  “Accept/reject of drafted tokens”.
  [official repository](https://github.com/platformxlab/vegas#readme)
- **[E3] Earlier SparseSpec reference engine:** the repository calls itself a
  “proof-of-concept”, “not yet ready for production use”, with “full attention as target model”
  and “(rejection) sampling”.
  [official repository](https://github.com/sspec-project/SparseSpec#readme)

The paper supplies the algorithm and describes a Transformers prototype, but [E1] does not link a
SparseSpec-L engine. [E2] and [E3] are related sparse-to-full engines, not a drop-in implementation
of SparseSpec-L's exact controller or memra's custom kernels [E1/E2/E3].

## 1. Existing memra posture this extends

The house baseline remains the **own-trimmed MTP regime**: byte-verbatim NextN/MTP extraction from
the serving GGUF, a target-own-generation vocabulary ranking, a trimmed draft head, and adaptive
coverage repair. The existing landscape explicitly treats other proposers as extensions, not
replacements (`research/spec-landscape-20260810/SURVEY.md:14-24`).

The artifact rule is similarly explicit: extract the draft block from serving-GGUF bytes, then
trim and quantize the draft; do not reconstruct the block from a separate checkpoint
(`docs/DRAFT-REGIME.md:27-36`, `tools/extract_mtp_draft.py:2-7`). A standalone draft replaces the
embedded MTP head, while the serving model supplies token embeddings and the full target still
arbitrates every proposal (`crates/memra-engine/src/hybrid.rs:895-900`,
`crates/memra-engine/src/hybrid.rs:1493-1504`).

The current generic speculative entry point is not proposer-neutral: it asserts `k >= 1` and
requires `self.mtp`, while the trimmed draft's `d2t` mapping is normalized to target token ids
before verification (`crates/memra-engine/src/spec.rs:5452-5488`). SparseSpec-L therefore needs a
new proposer path or an explicit proposer abstraction; attaching a different GGUF draft head
cannot express it (`crates/memra-engine/src/spec.rs:5476-5488`).

The standing exactness gate runs the speculative stream against the plain target for every K from
1 through 8 and separately warns when a wrong proposer masks itself behind zero acceptance
(`crates/memra-engine/src/bin/run_spec.rs:1-8`,
`crates/memra-engine/src/bin/run_spec.rs:323-371`,
`crates/memra-engine/src/bin/run_spec.rs:421-438`). This gate remains mandatory; the full project
battery also requires kernel reference checks and run-gen argmax agreement
(`CONTRIBUTING.md:17-30`, `docs/TESTING.md:13-17`).

## 2. What SparseSpec-L actually does

| Question | Pinned answer | Evidence |
| --- | --- | --- |
| Separate draft model or head? | **Neither.** The target weights autoregressively draft under sparse attention, then the same weights verify under full attention. It is training-free self-speculation. | [E1, pipeline] |
| What is sparse? | A per-layer, per-head index into historical KV for the draft pass. The dense KV remains resident and authoritative. | [E1, recall index] |
| What becomes important? | For token position `i`, the method sums attention received from the most recent `W` verification queries, independently per layer and head; the reported construction fixes `W=16`. | [E1, recall index] |
| Which positions are recalled? | Each head keeps sink positions, a recent window, and the highest-scoring historical positions outside those two groups. | [E1, recall index] |
| When is the index refreshed? | After a full-context verification pass, using its materialized per-head attention statistics; no extra model forward is introduced for the scores. | [E1, pipeline/recall index] |
| How are drafts checked? | One parallel target pass uses the complete KV, accepts the valid prefix under standard speculative verification, and supplies the correction token. | [E1, pipeline] |
| Does it reduce authoritative KV capacity? | **No.** Recallability means the original dense K/V stays resident; the sparse index and gathered view are additional draft state. | [E1, recall index] |

The memory additions are therefore the per-layer/per-head importance state, index sets, and any
materialized gathered sparse K/V workspace; the dense target cache is not replaced [E1, recall
index]. The compute additions are score materialization during full verification, reduction and
top-k index rebuilding, sparse gather/attention, and sequential target-weight draft forwards;
the paper explicitly identifies non-fused sparse gather and dense non-FlashAttention verification
as prototype limitations [E1, implementation/limitations].

This differs from an MTP head in mechanism, not in the correctness role. Memra's MTP path runs a
cheap NextN proposer with its own scratch state and then performs a batched target verify
(`crates/memra-engine/src/spec.rs:1-9`). SparseSpec-L instead reuses the full target trunk for each
draft token but limits historical-attention reads; the complete target pass still decides what is
committed [E1, pipeline].

## 3. Decisive losslessness analysis

Let `q_sparse` be the token distribution produced by target weights over the sparse KV view and
`p_full` be the ordinary target distribution over the complete KV. SparseSpec-L is admissible only
when `q_sparse` is a proposer and `p_full` is the verifier and correction distribution [E1,
pipeline]. This matches memra's existing separation: one batched target forward evaluates the
pending token and drafts, and all returned columns are target logits
(`crates/memra-engine/src/spec.rs:2698-2703`,
`crates/memra-engine/src/spec.rs:7071-7082`).

This applies the kvcode/CacheBlend bar at the correct authority boundary. Kvcode had to recover
the authoritative target-cache bytes exactly before downstream output gates were meaningful,
whereas the survey keeps approximate CacheBlend reuse behind a default-OFF byte-identity door
(`research/kvcode-20260811/FEASIBILITY.md:9-32`,
`research/spec-landscape-20260810/SURVEY.md:683-686`). SparseSpec-L may approximate **only the
proposal distribution** because full dense KV is still used to compute `p_full`; making its sparse
view authoritative for target verification would fail that same bar [E1, pipeline].

For greedy decoding, accept only the longest prefix whose draft ids equal successive full-target
argmaxes, then emit the full target's token at the first unaccepted position. That is already
memra's host/device rule and yields the plain-target stream at any proposal quality
(`crates/memra-engine/src/spec.rs:7441-7502`). A deliberately bad or empty sparse index may reduce
acceptance, but it must not alter output tokens; that is the same zero-acceptance masking hazard
the current gate reports (`crates/memra-engine/src/bin/run_spec.rs:421-438`).

For temperature sampling, simple argmax-prefix comparison is insufficient. The future arm must
feed `q_sparse` and `p_full` into memra's existing modified rejection walk and correction
distribution; current code begins with the `p(x)/q(x)` accept rule, and the documented contract is
target-distribution equality plus seeded reproducibility
(`crates/memra-engine/src/spec.rs:7504-7505`, `docs/FLAGS.md:63`,
`crates/memra-engine/src/bin/run_spec.rs:357-371`). [E1] asserts preservation of the target output
distribution, while the related engines expose rejection-sampling integration [E2/E3]; no
SparseSpec-L-specific public engine was found here to substitute for a memra-side sampled-path
gate [E1/E2/E3].

The lossless proof does **not** require `q_sparse == p_full`; disagreement is what verification
corrects. It does require that verifier attention, logits, penalties, constraints, cache commit,
and correction sampling remain the ordinary full-target path
(`crates/memra-engine/src/spec.rs:2741-2779`,
`crates/memra-engine/src/spec.rs:7441-7505`). Sparse attention in the verifier would instead change
`p_full` and is **DOOR-CLOSED**, exactly like reduced-set expert verification
(`research/spec-landscape-20260810/SURVEY.md:503-512`).

## 4. Memra integration delta

| Surface | Current memra reality | Required default-OFF experiment boundary |
| --- | --- | --- |
| Proposer dispatch | `generate_spec` requires an MTP head (`crates/memra-engine/src/spec.rs:5452-5479`). | Add a sparse-self proposer mode without weakening or silently bypassing the MTP invariant for the house path [E1, pipeline]. |
| KV representation | Each full-attention layer owns contiguous token-major Q8_0-K and Q5_1-V byte planes (`crates/memra-kv/src/lib.rs:213-232`). | Keep those planes complete; add per-head indices and an indexed sparse read/gather used only by draft attention [E1, recall index]. |
| Decode attention API | Decode passes contiguous prefix views into `fa_decode_kvmod` (`crates/memra-engine/src/decode.rs:2758-2801`). | Add a separately gated indexed-attention path; related engines use page tables/custom attention, which do not map directly to this flat-byte API [E2/E3]. |
| Verification telemetry | `fa_decode_kvmod` takes Q/K/V and writes only attention output; its public signature has no score output (`crates/memra-engine/src/lib.rs:9819-9845`). | Add opt-in per-head importance output or an exact rematerialization path, while the naked verifier remains unchanged [E1, recall index]. |
| Verify/accept | `decode_step_t` returns every full-target logit column and the orchestrator accepts/corrects from them (`crates/memra-engine/src/spec.rs:2698-2779`, `crates/memra-engine/src/spec.rs:7441-7505`). | Reuse this authority; sparse state may choose proposals and K but never the target logits or committed distribution [E1, pipeline]. |
| Artifact path | Own-trim drafts are standalone GGUFs attached to the target; verifier exactness is independent of draft quality (`docs/DRAFT-REGIME.md:49-51`, `docs/DRAFT-REGIME.md:58-74`). | SparseSpec-L needs no draft GGUF, but it must not change the serving GGUF or displace the own-trim artifact baseline ([E1]; `research/spec-landscape-20260810/SURVEY.md:14-24`). |

The two public reference engines make the implementation class credible but not portable. Vegas
uses a vLLM/FlashAttention-3 fork, a custom per-step page table, verifier score output, and a
rejection sampler [E2]. Earlier SparseSpec uses a custom FlashInfer/vLLM-style stack and describes
itself as a non-production proof of concept [E3]. Memra instead consumes contiguous quantized byte
views through its own attention API (`crates/memra-engine/src/decode.rs:2758-2801`,
`crates/memra-engine/src/lib.rs:9819-9845`).

## 5. Hy3 claim: blocker removed, capacity problem retained

SparseSpec-L removes the **draft-head dependency in principle** because its target trunk is both
drafter and verifier; it does not require an MTP/EAGLE checkpoint [E1, pipeline]. That makes a
headless or shallow-head model eligible for a proposer experiment, but it does not establish that
the repeated sparse-attention target forwards are profitable on that model [E1, limitations].

For the existing Hy3 evidence, the public NextN=1 path accepted no chained second token and K>1
added no accepts in the recorded sweep (`research/hy3-spec-20260802/SUMMARY.md:63-69`). A
SparseSpec-L proposer is structurally capable of autoregressive multi-token drafting because it
re-runs target weights rather than recursively applying that head [E1, pipeline]; whether those
drafts pay for full expert/trunk execution is unmeasured and must not be inferred from the paper
[E1, limitations].

SparseSpec-L does **not** remove Hy3's KV-capacity blocker: the model assessment records the
resident case as KV-starved, and SparseSpec-L deliberately retains the complete dense cache for
verification (`research/model-192gb-20260806/ASSESSMENT.md:192-194`; [E1, recall index]). It
therefore removes one eligibility blocker only; it is not a cache-compression or spill result
(`research/model-192gb-20260806/ASSESSMENT.md:192-194`; [E1, recall index]).

## 6. Entropy controller as an independent idea

SparseSpec-L records each draft token's output entropy `H_i`; after verification, accepted and
rejected tokens update separate exponential-moving-average entropy centers. It maps a new `H_i`
to an estimated acceptance probability `p_i` by a two-class softmax over negative L1 distance to
those centers [E1, controller].

For a candidate depth `k`, it estimates accepted draft count as the sum of prefix-survival
probabilities, `sum_(m=1..k) product_(i=1..m) p_i`. It then chooses from a configured candidate set
the `k` maximizing `(1 + expected accepted drafts) / (k*C_d + C_v)`, where `C_d` and `C_v` are
measured draft and verification costs [E1, controller]. The useful transfer is the objective and
online feedback, not any paper-selected K or cost constant [E1, controller/limitations].

This controller is independently adoptable over own-trim MTP because that path already produces
draft logits and verification labels; the sparse-index mechanism is not an input to the formula
(`crates/memra-engine/src/spec.rs:6028-6037`,
`crates/memra-engine/src/spec.rs:7441-7505`; [E1, controller]). It should sit below an explicit
operator pin: `MEMRA_SPEC_K=<n>` currently pins every eligible request, including K=0, and disables
automatic demotion (`crates/memra-server/src/worker.rs:1598-1619`,
`crates/memra-server/src/worker.rs:1621-1646`,
`crates/memra-server/src/worker.rs:1655-1660`).

The correct baseline is not “fixed K everywhere.” With no operator pin, serving already chooses
K=0 for gated placement/concurrency, K=2 for a sufficiently resumed long prompt, and K=3
otherwise (`docs/FLAGS.md:54`, `crates/memra-server/src/worker.rs:1621-1652`). Within a round,
`MEMRA_SPEC_PMIN` can stop the chain on low top-1 confidence
(`crates/memra-engine/src/spec.rs:6028-6046`). Across rounds, the Qwen path has a default-off
accepted-run-plus-one controller with a recorded negative tuned-cell verdict, while Gemma enables
that policy by default (`crates/memra-engine/src/spec.rs:6388-6414`,
`crates/memra-engine/src/gemma_spec.rs:750-757`).

The paper itself calls entropy only a moderate, task-dependent signal [E1, limitations]. A memra
arm must therefore compare at least: current unpinned request policy; best fixed operator pin;
current p-min where applicable; accepted-run adapt; entropy-only; and entropy plus p-min. The
controller is **DOOR-CONDITIONAL** until it wins end-to-end under that matrix without changing
the exactness result (`docs/DRAFT-REGIME.md:32-36`,
`crates/memra-engine/src/spec.rs:6388-6414`).

## 7. Required future gate

No implementation is authorized by this lane. If the orchestrator opens one later, the first arm
must be named and scoped as **`SPARSESPEC-FULL-TARGET`**, default OFF, with these pass conditions
grounded in memra's existing exactness and baseline rules
(`CONTRIBUTING.md:17-30`, `research/spec-landscape-20260810/SURVEY.md:14-24`).

1. **Frozen authority.** Same target GGUF bytes, prompt token ids, template, cache format,
   penalties/sampling settings, and runtime commit for plain, own-trim, and sparse-self arms. The
   sparse index is proposal metadata only; full K/V remains the target authority
   ([E1, pipeline]; `crates/memra-kv/src/lib.rs:213-232`).
2. **Adversarial proposer test.** Empty, stale, and deliberately perturbed sparse indices may
   change drafted/accepted counts but must not change greedy target tokens. Zero acceptance is
   reported as a proposer failure, not accepted as proof of a useful path
   (`crates/memra-engine/src/bin/run_spec.rs:1-8`,
   `crates/memra-engine/src/bin/run_spec.rs:421-438`).
3. **Greedy byte identity.** Run the full `run-spec` K=1..8 battery and require every K plus the
   final self-consistency marker to pass against plain decoding
   (`crates/memra-engine/src/bin/run_spec.rs:323-371`, `docs/TESTING.md:13-17`). Variable controller
   depths need a trace showing every chosen K remains within the gated set [E1, controller].
4. **Sampled distribution path.** Preserve memra's `p_full/q_sparse` modified rejection rule and
   pass same-seed reproducibility; a greedy-only result does not authorize sampled serving
   (`docs/FLAGS.md:63`, `crates/memra-engine/src/bin/run_spec.rs:357-371`).
5. **Verifier non-interference.** Attention-stat collection OFF must leave the naked verifier
   byte-identical; ON must pass kernel-check, run-gen argmax, and run-spec before any score is
   considered (`CONTRIBUTING.md:17-30`). Sparse attention is forbidden in the target pass [E1,
   pipeline].
6. **House-baseline comparison.** On every target that supports it, compare against the current
   own-trim artifact under identical settings and decide by end-to-end tokens/s, not acceptance
   alone (`docs/DRAFT-REGIME.md:32-36`,
   `research/spec-landscape-20260810/SURVEY.md:14-24`). For Hy3, also retain plain decoding and the
   current exact K=1 arm; head independence does not waive the own-trim regime as the general
   proposer baseline (`research/hy3-spec-20260802/SUMMARY.md:54-69`,
   `research/spec-landscape-20260810/SURVEY.md:14-24`).

The entropy-only arm uses the same gate but leaves the proposer unchanged. An explicit
`MEMRA_SPEC_K` remains authoritative; automatic entropy selection is eligible only when the pin is
absent, matching the current operator-pin precedence
(`crates/memra-server/src/worker.rs:1621-1631`,
`crates/memra-server/src/worker.rs:1655-1660`).

## 8. Stop conditions and unknowns

Close the SparseSpec-L scored door if full-KV verification is skipped, if verifier score
instrumentation changes target logits, if rejected draft state leaks into committed cache, or if
any greedy K diverges from plain output. These are direct violations of the existing full-target
and K=1..8 contracts (`crates/memra-engine/src/spec.rs:2698-2779`,
`crates/memra-engine/src/bin/run_spec.rs:1-8`; [E1, pipeline]).

The following remain deliberately unknown rather than performance claims:

- whether indexed reads over memra's packed contiguous KV can beat its current dense attention
  path (`crates/memra-kv/src/lib.rs:213-232`,
  `crates/memra-engine/src/decode.rs:2776-2801`);
- whether verifier statistics can be exposed without perturbing the custom attention kernel's
  numeric or scheduling path (`crates/memra-engine/src/lib.rs:9819-9845`; [E1, limitations]);
- how the paper's per-head score/index state should be bounded across memra model families [E1,
  recall index];
- controller initialization, cold-start behavior, candidate-K set, and cost refresh cadence for
  memra [E1, controller/limitations];
- any useful Hy3 acceptance, memory, spill, or throughput result
  (`research/hy3-spec-20260802/SUMMARY.md:54-69`; [E1, limitations]); and
- whether a SparseSpec-L-specific production engine will appear; the currently public related
  references are vLLM/FlashAttention or custom FlashInfer research stacks [E2/E3].

This lane made no source changes, ran no formatter, build, GPU command, or benchmark, and makes no
memra performance claim. Its result is only the semantic and integration verdict above; the
standing project battery remains the authority for any future implementation
(`research/sparsespec-20260811/PROGRESS.md:28-30`, `CONTRIBUTING.md:17-30`,
`docs/TESTING.md:13-17`).
