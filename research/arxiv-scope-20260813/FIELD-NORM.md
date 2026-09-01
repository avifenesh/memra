# How much measurement the field actually publishes

Scope study of 10 arXiv papers in inference optimisation (speculative decoding, KV cache, MoE
inference, quantization), all submitted 2026-04 through 2026-08. Purpose: fix an empirical
reference for what published evaluation coverage looks like, so memra's internal bar is compared
against the field instead of against an invented standard.

N = 10. This is a **convenience sample** of recent work in one subfield, selected by topic search,
not a random sample of ML papers or even of all 2026 inference papers. Every number below is
traceable to an arXiv id. Raw extracted paper text for all 10 is committed beside this file in
`raw/` (one file per id).

Extraction confidence: 9 high, 1 medium (`2606.10493v1` — the paper itself is internally
inconsistent about which checkpoint produced which number), 0 low. **0 papers were excluded from
the medians.** Dropping the medium-confidence paper does not move either median (models 3,
hardware 1).

## 1. Headline numbers

**Models per paper:** median **3**, range **2 to 10**, mean 4.4.
- 0 of 10 evaluated exactly one model.
- 4 of 10 evaluated one or two models: `2608.10362v1` (2), `2608.05448v1` (2), `2607.09686v1` (2),
  `2606.10493v1` (2).
- 4 of 10 evaluated four or more: `2608.05303v1` (4), `2608.08721v1` (6), `2607.17733v1` (10),
  `2607.07964v2` (10).

**Hardware configurations per paper:** median **1**, range **1 to 2**.
- **8 of 10 used exactly one hardware configuration.**
- 2 of 10 used two: `2608.10362v1` (Jetson Orin Nano for the main evaluation, Jetson AGX Orin only
  for one supplementary draft-capacity analysis) and `2608.05303v1` (all reported latency/energy
  from a simulated 28nm ASIC; an RTX A6000 appears only in the artifact appendix for bf16 software
  runs).
- 0 of 10 used three or more.
- **0 of 10 published a headline claim measured on two different real accelerator classes.** In both
  two-hardware papers the second platform carries no primary result.

**Baselines per paper:** median **4**, range **1** (`2608.05448v1`, `2607.09686v1`) **to 18**
(`2607.07964v2`, most of whose baseline numbers are copied from the original papers rather than
re-run).

Hardware description quality is uneven inside the "one configuration" group:
- `2608.05448v1` names no GPU at all — the only description is "a consumer level GPU"; training
  hardware is never stated.
- `2608.08721v1` says "NVIDIA A100 GPUs" with no count and no 40/80 GB variant.
- `2604.23150v1` names 8x H100 for its 300-hour trace collection; the GPU type of the 64-GPU
  `DP8TP8->EP64` evaluation configuration is never named.
- `2608.05303v1` reports no silicon measurement at all — every latency and energy number comes from
  an instruction-level custom simulator.

With N=10, the 2nd through 9th ordered values form a distribution-free ~98% interval for the
population median: 2 to 10 models, 1 to 2 hardware configurations. The hardware finding is tight;
the model finding is not.

## 2. Distribution, one row per paper

Sorted by model count, then id.

| arXiv id | short name | topic | models | hw configs | baselines | spread published | negative result | artifact released |
|---|---|---|---|---|---|---|---|---|
| 2606.10493v1 | CPU-GPU hybrid local MoE | MoE serving | 2 | 1 | 5 | no | yes | no |
| 2607.09686v1 | MawForge | MoE expert materialization | 2 | 1 | 1 | no (N=5 stated) | yes | no |
| 2608.05448v1 | DBLAST | speculative decoding | 2 | 1 | 1 | no | yes | no |
| 2608.10362v1 | MemSpec | speculative decoding (edge) | 2 | 2 | 4 | no (N=3 stated) | yes | no |
| 2604.23150v1 | multi-node MoE placement | MoE serving | 3 | 1 | 2 | no | yes | no |
| 2608.08097v1 | OasisKV | KV cache | 3 | 1 | 5 | no | yes | no |
| 2608.05303v1 | EdgeXpert | MoE + spec, ASIC | 4 | 2 | 3 | no | yes | partial |
| 2608.08721v1 | LibraSpec | speculative decoding | 6 | 1 | 7 | no | yes | no |
| 2607.07964v2 | KronQ | quantization | 10 | 1 | 18 | no | yes | no |
| 2607.17733v1 | MXSens | quantization | 10 | 1 | 4 | no | yes | no |

Model counts are as extracted. Three papers admit a second defensible count: `2607.09686v1` is
2 base models run as 3 quantized profiles; `2606.10493v1` is 2 base families but 4 distinct named
checkpoints (and it swaps checkpoints between the performance and quality sections without
explanation); `2607.07964v2` is 10 or 11 depending on whether Gemma-3-12B and Gemma-3-12B-IT are
one model. Under the maximal reading the median rises to 3.5, the range to 2-11, and the count of
one-or-two-model papers falls from 4 to 2.

## 3. What the field does not do

**Spread: 0 of 10.** No paper in the sample publishes a central value together with a spread —
no standard deviation, confidence interval, min/max, or error bar on any performance or accuracy
table. Six papers state a repetition count or an averaging scope with no spread attached:
- `2608.10362v1`: "All results are averaged over three runs." Aggregates are geometric means.
- `2607.09686v1`: each cell "contains five valid repetitions, and the source artifacts report means
  and standard deviations for those cells" — the paper prints the means only. Both addenda are
  single runs.
- `2608.05448v1`: each value "averaged over all speculative iterations executed while generating
  1,000 responses".
- `2604.23150v1`: "averaged over 500 global batches".
- `2608.08097v1`: accuracy sampled as avg@8 / avg@k, and the prose argues results are "within
  run-to-run variance" without ever printing that variance.
- `2607.07964v2`: one avg@8 cell for one model; no seeds, no deviation anywhere else.

Four state nothing about repetition: `2608.08721v1`, `2608.05303v1` ("All evaluations perform
single-batch inference", a deterministic simulator), `2607.17733v1`, and `2606.10493v1` — which
says only "Each workload is executed multiple times ... we report the average", giving neither N
nor spread.

Thermal or clock regime: not stated in any paper the extraction checked for it. This negative was
explicitly verified only for `2608.10362v1`, where it matters most (a passively-cooled Jetson);
for the rest, treat it as not extracted rather than as a confirmed absence. Interleaving of arms
inside one measurement window is not described anywhere in the sample, and was not searched for
per-paper.

**Negative or against-hypothesis results: 10 of 10.** This is the sample's strongest evidence
habit, and it is not shallow:
- `2608.05303v1`: the motivating measurement contradicts the premise — speculative decoding
  "increasing the total energy consumption by 24.3% compared to autoregressive decoding", and their
  own coalescing mechanism ablated alone drops OLMoE GSM8K from 62.6% to 16.5%.
- `2607.09686v1`: the paper's central finding is that its own lever backfires — cache 35%→65%
  raises hit rate 86.73%→95.03% while decode collapses 13.8591→1.3033 tok/s.
- `2607.07964v2`: W4 average zero-shot below the GPTQ baseline on LLaMA-2-7B (65.9 vs 66.0) and
  LLaMA-3-70B (75.1 vs 75.4), plus an admission that much of the gain comes from an inherited
  correction rather than the novel term.

**Artifacts: 1 of 10 released anything; 0 of 10 released what produced the headline numbers.**
- `2608.05303v1` is the only release: "Publicly available?: Yes", Zenodo DOI
  10.5281/zenodo.21481269, a `run.py` entry point and pinned requirements. The authors state the
  artifact "runs in full precision (bf16)" while the paper's numbers are simulated at A8W4, and
  neither the RTL nor the instruction-level simulator that produced every reported latency and
  energy figure is in the release.
- 9 of 10 released nothing of their own. Eight make no availability statement at all.
  `2607.17733v1` has a Section 8 titled "Reproducibility Statement" that points only to in-paper
  setup text and pseudocode; the string "github" appears once in the whole paper, in an unrelated
  Qualcomm blog citation. `2604.23150v1` does not offer its 100k-request expert-activation trace
  dataset — the paper's central contribution — for download.
- 0 of 10 published per-run data. `2607.09686v1` cites its own run directories and runner scripts
  by relative path with no URL or DOI.

Venue quality does not explain these gaps. The sample includes `2608.10362v1` (LCTES 2026, DOI
10.1145/3814943.3816174), `2608.05303v1` (MICRO 2026), and `2606.10493v1` (arXiv comment states
OSDI '26 acceptance).

## 4. Where memra's evidence sits

memra's internal discipline, as written in `CLAUDE.md`, `ARCHITECTURE-H100.md`, and
`research/benchmarks.md`: interleaved A/B with a published floor of N>=3 pairs in both orders
(`research/benchmarks.md`) and N=5 as the practiced norm on the H100 lane ("every perf claim is
interleaved x5 on-box"), medians with min/max spread, arms alternated inside one measurement
window, cross-run and cross-day comparisons rejected as clock-drift-invalid, exactness gates
(`kernel-check`, `run-gen` argmax, `run-spec` K=1..8 self-consistency) passed on output identity
before any performance number counts, and raw per-run JSONL committed beside every summary row.

Against this sample:

- **Spread: above.** 0 of 10 papers publish a spread. The closest are `2608.10362v1` (N=3, means
  only) and `2607.09686v1` (N=5 per cell, deviations computed but not printed). memra publishes the
  median with min/max and the N, which exceeds every paper here.
- **Interleaving and thermal control: above.** No paper in the sample describes either.
- **Raw run data: above.** 0 of 10 published per-run data; 1 of 10 published code, and not the code
  that produced its numbers.
- **Output-identity gating before a perf number counts: above, with a caveat.** Several papers put
  quality tables next to performance tables, but none makes the performance number conditional on
  an exactness gate. The extraction did not search for such a gate by name, so read this as "not
  present in what was extracted" rather than a fully verified 0 of 10.
- **Model coverage: at or above.** memra's gate surfaces carry 12 distinct target GGUFs (11 in
  `tools/fast-gate/models.tsv` alone, across 15 model-bearing probe entries, plus the acceptance
  battery's cells), against a sample median of 3 and a sample maximum of 10. The comparison is not
  like-for-like: `2607.17733v1` and `2607.07964v2` quantize 10 checkpoints and score perplexity and
  zero-shot accuracy, while memra runs argmax and self-consistency gates plus throughput on 12
  GGUFs. Coverage in count is comparable to the top of the sample; coverage in evaluation breadth
  is different in kind.
- **Hardware breadth: above.** memra's per-hardware arm doctrine requires a mechanism to be measured
  on both the local 5090 and a PRO 6000 before it sets a default. No paper in this sample published
  a primary result on two real accelerator classes.

memra is below the sample on one axis worth naming: published task-quality benchmarking. Eight of
the ten papers score standard benchmark suites (GSM8K, MMLU, HumanEval, MT-Bench, WikiText-2
perplexity, AIME, GPQA-Diamond). memra's gates prove output identity against pinned goldens, which
is a stricter test of "did the kernel change the answer" and no test at all of "is the answer good".

## 5. Conclusion

Two of the three parts of the owner's claim hold, one does not.

**One hardware configuration is the norm — confirmed.** 8 of 10 used exactly one; the maximum in the
sample is two; 0 of 10 based a headline claim on two different real accelerator classes. A
single-platform result is what this subfield publishes, including at LCTES, MICRO, and OSDI.

**One or two models is not the norm — refuted by this sample.** 0 of 10 used a single model, the
median is 3, and 4 of 10 used four or more (up to 10 in the two quantization papers). Papers that
stay at two models tend to be the systems papers where the platform is the constraint
(`2608.10362v1` on 8 GB Jetson, `2607.09686v1` on a 24 GB MacBook, `2606.10493v1` on a 1.15 TB
single node) — the same shape as memra's own work. The coverage here is broader than expected, and
the honest report is the data, not the prior.

**The field's real weakness is not coverage; it is measurement rigor and released evidence.** 10 of
10 publish point numbers with no spread, 10 of 10 fail to release the artifact behind their headline
number, and 0 of 10 publish per-run data. memra's internal bar is above the sampled norm on every
one of those axes and at the top of the sample on model count, so it should not be treated as a
minimum to catch up to. It is already stricter than what this subfield publishes, and the honest
framing of any memra claim is a single-platform-plus-one-verification-rig result with N and spread
stated — which is more, not less, than the papers it sits next to.

## Limits of this sample

Ten papers, one subfield, four months, chosen by topic search rather than randomization. Single
extractor pass per paper with verbatim quotes recorded; no independent duplicate extraction, so
extraction error is unbounded on any single row even where quotes were verified against the raw
text in `raw/`. Model counts depend on a counting convention (base model vs quantized profile vs
named checkpoint) that changes the median by 0.5. Conference versions of these papers may carry
artifact evaluations that the arXiv version does not claim; `2606.10493v1` in particular states
OSDI '26 acceptance without an availability statement in this version.
