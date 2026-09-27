# Bounded prompt-prefix depth: scoped result

Verdict: A bounded request-prefix K=2 prose/K=4 code rule did not transfer as a universal win: Qwen's short and mid-length cells lost, while Gemma's code cells gained 5.9%, 8.0% and 3.8% at 256, 1K and 4K prompt tokens with six of six paired wins each; Gemma prose lost.

This comparison uses fixed K=3 as the owner-specified control. It is a
research result on six synthetic four-helper scenarios per format and length,
not a serving-default decision or proof against a freshly calibrated global
K=2 or K=4 control. No LLM or draft-head weights were trained or changed.

| Model and requested output | 256 tokens | 1,024 tokens | 4,096 tokens | 16,384 tokens |
|---|---:|---:|---:|---:|
| Qwen prose, K=2 vs K=3 | −5.58% | −6.45% | −4.32% | +1.59% |
| Qwen code, K=4 vs K=3 | −2.32% | −4.07% | −4.43% | +3.28% |
| Gemma prose, K=2 vs K=3 | −4.99% | −6.74% | −5.70% | −1.72% |
| Gemma code, K=4 vs K=3 | **+5.91%** | **+7.98%** | **+3.84%** | +1.15% |

The entries are pooled generated tokens / complete native request seconds
relative to fixed K=3, using the X=64 prefix arm. All three prefix budgets
selected the same K for every scored prompt, so X=128 and X=256 do not test
different depth choices on this corpus. Their rates and full per-cell tables
are in [RESULTS.md](RESULTS.md) and the independently replayed
[RESULTS.json](RESULTS.json).

Gemma code won 6/6 paired scenarios in each of the 256-, 1,024- and
4,096-token cells. Their pointwise 95% whole-scenario bootstrap intervals for
the pooled gain are [+4.37%, +7.57%], [+6.96%, +9.01%] and [+2.08%, +5.62%].
At 16,384 tokens it won 5/6, but the +1.15% pooled interval
[−2.00%, +4.32%] includes zero. Gemma prose lost about 5–7% at 256–4,096
tokens; Qwen prose lost about 4–6.5% there. Qwen code pooled rates were below
fixed K=3 at those lengths, while their pointwise intervals include zero.
Both Qwen 16,384-token intervals include zero. These are six paired synthetic
scenarios per cell, not a population-wide guarantee.

Every one of the 16 model/format/length cells retained six covered pairs.
There were zero matched loop exclusions. Both models selected an 8,192-token
output cap on disjoint qualification tasks after greedy fixed-depth identity,
sampled routing/schedule identity and real prose/code format checks. The
earlier complex-contract Qwen qualification returned no final code in four
requests at either an 8,192- or 12,288-token cap; the raw replay reproduces
that boundary. The simplified tasks make this comparison possible and
restrict its generality.

Time includes normal tokenization, prefix extraction, forecast, fresh-cache
prefill, sampled generation and detokenization; it excludes model load,
warmup and receipt writing. Generated-token counts include reasoning, while
final-answer format coverage is reported separately. The study used one
non-production RTX 5090, Qwen3.8-27B NVFP4+Q5_K with embedded MTP and Gemma 4
12B QAT Q4_0 with its Q8_0 assistant. The report pins model revisions,
weights, runtime archive and hardware. Each request starts a fresh native
cache: this is separate from the earlier continuing-session KV and online-D
learner studies and does not qualify HTTP or concurrent serving.

The hosted replay checked all 60 native runs and 304 routed prefixes, the
exact prompt tokens across paired arms, source and binary identities,
frozen orders, returned-token and time arithmetic, format coverage and loop
selection. Its JSON SHA-256 is
`a3d70de0619ea89166196979c6f0602d01376bef9563dbc7cd5301e94115c926`.
The private current-policy expanded archive scan found zero unapproved
matches; [ARCHIVE-BOUNDARY.md](ARCHIVE-BOUNDARY.md) names the exact archived
source exceptions. The task pod was deleted only after 5,716 remote receipt
files matched the local copy and a direct provider-list read confirmed
absence.

The first 64 user tokens sufficed to identify requested prose or code in this
instruction-first corpus. The positive Gemma code cells justify fresh
representative-code validation and same-shape fixed K=2/K=4/native controls
before attributing a gain specifically to *routing*. A cheap predictor of
reasoning-to-code transitions remains a separate research question. No
serving door or default is promoted here.
