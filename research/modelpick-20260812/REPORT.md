# Model pick — recommendation for owner, 2026-08-12

**Pick Qwen3.6-35B-A3B + Qwen3.6-27B, one per card.** Planning gross is **~$36/day**
on the pair, with **2–4 dev-days** because both architectures, GGUF paths, and model-specific
drafters are already supported. This is a deliberate exception to the few-provider heuristic:
at only 0.5%/1.0% assumed share—below today's smallest incumbent shares of 1.66%/2.75%—their
agentic pools still beat every clean <=3-provider gap found.

## Same-page evidence used to score

Fetched 2026-08-12 from OpenRouter's page-owned public JSON: weekly standard-variant prompt and
completion totals from the [rankings activity feed](https://openrouter.ai/api/frontend/v1/rankings/models?view=week);
provider count from `stats/endpoint`; latest completed-UTC-day token share, weighted effective
input/output price, and cache hit from `stats/effective-pricing?shape=v7`; Apps from
`stats/top-apps-for-model`; tool/structured error from the request-volume-weighted 2026-08-04..11
page series; uptime from the page's three-day mean. Pool/day = `(weekly prompt × effective input +
weekly completion × effective output) / 7`; it is gross provider billings, not profit.

| Model / visible agentic Apps | tok/wk; providers | effective $/M in / out; cache | leading incumbent: share; tool / structured error; uptime | pool/day |
|---|---:|---:|---:|---:|
| [Qwen3.6-35B-A3B](https://openrouter.ai/qwen/qwen3.6-35b-a3b) — Hermes, pi, OpenClaw | 122.64B; 9 | $0.125 / $1.065; 28.1% | Parasail: 37.7%; 0.98% / 0.35%; 99.83% | $3,655 |
| [Qwen3.6-27B](https://openrouter.ai/qwen/qwen3.6-27b) — Qwen Code, pi, OpenClaw | 26.55B; 9 | $0.285 / $2.816; 36.9% | Chutes: 37.2%; **4.70% / 11.13%**; **97.24%** | $1,806 |
| [Qwen3.5-122B-A10B](https://openrouter.ai/qwen/qwen3.5-122b-a10b) — OpenHands, Cline, LangChain | 14.67B; 5 | $0.308 / $2.364; **0%** | DeepInfra: 45.6%; 0.87% / n/a; 99.63% | $929 |
| [Step-3.7-Flash](https://openrouter.ai/stepfun/step-3.7-flash) — Hermes, Kilo, Cline | 1.243T; 3 | $0.055 / $1.149; 90.6% | StepFun: **99.73%**; DeepInfra has 0.21% and 23.56% tool error | $10.7K¹ |

¹ Owner's same-page daily capture is authoritative for the Step control; recombining the weekly
feed with today's weights gives $12.0K/day, showing the expected window mismatch rather than a
second estimate to optimize against.

## Ranked configurations

1. **(a) Qwen3.6-35B-A3B + Qwen3.6-27B — ~$36/day.** Assumptions: 0.5% × $3,655 +
   1.0% × $1,806. Q35 at 0.5% asks ~90 output tok/s, below memra's 187 tok/s 5090-development
   receipt (the PRO-pair gate is still required); Q27 at 1% asks ~33 tok/s versus 169 tok/s
   measured. Q8-class weight math is ~37.2 GB for
   35B (`35B × 1.0625 B`) and the Q27 Q8 receipt is 28.6 GB, so each has ample room on 96 GB
   for draft/KV/cache. The fit screen is Q8-class; today's tuned artifacts are Q35 IQ4_XS and
   Q27 NVFP4 and must be labeled honestly. Dev: **2–4 days** for pricing, prefix-cache and
   constraint/tool receipts, then listing gates. Risk: Q35's leader is healthy and both pages
   have nine providers; the intentionally sub-incumbent share assumptions are the uncertainty
   bound.

2. **(c) Keep Step + add Qwen3.6-27B — plan ~$30/day, not additive.** Use the observed 0.2%
   non-first-party Step share (~$21/day) and only 0.5% Q27 share in safely schedulable Step-idle
   windows (~$9/day). Step's measured placement is 49.64/59.27 GB; the resident Q27 process is
   ~16.3 GiB on card 0, so bytes fit. Dev: **3–5 days** for strict gateway arbitration and the
   correctness/cache campaign. Risk is decisive: active Q27 caused **+290% Step TTFT and -69%
   Step decode**; idle residency was neutral. If Step reaches ~0.3% and fills the pair, Q27 earns
   zero; StepFun's 99.73% moat caps upside.

3. **(b) Qwen3.5-122B-A10B + Qwen3.6-27B — ~$27/day base, ~$37/day upside.** One percent of
   each pool gives $9.3 + $18.1/day; Q122 reaches the upside only if its uniquely discounted
   cache-read offer wins 2%. Run Q122 IQ4_XS (60.2 GB) on one card and Q27 Q8 (28.6 GB) on the
   other; Q122 Q8 is 129.9 GB and would instead consume both cards. It is the same
   `qwen3_5_moe` kernel family but needs shape/artifact/listing qualification: **5–7 days**.
   Risks: the leading incumbent is
   healthy, the current page has zero cache hits, and the one-card listing is honestly int4 with
   quant-quality/filter risk.

The <=3-provider scan does not change the pick. Nemotron-3-Nano is the cleanest gap (30.62B/wk,
three providers, leading-provider uptime 93.17%) but its $252/day pool yields only **$5/day at
2% share** and costs a new `nemotron_h` arch. Ling-3.0-Flash has two providers, but Novita owns
98.82% with 0.52% tool error and 99.99% uptime. Inkling Small yields $7.6/day at 2% but is a
276B PP-2 multimodal/new-arch job. KAT Air/Pro have attractive one/two-provider pages, but no
public checkpoint/HF slug was found for those exact entries; the open 35B Dev checkpoint is a
different, zero-observed-pool listing. Gemma's visible top Apps do not show coding demand and its
large pools are crowded; GLM-Air's pool is only $203/day.
Vanilla Hy3 is $62.2K/day but Tencent owns 98.12%; a REAP derivative inherits **zero** of that
route volume and cannot be scored before the required five-arm report. These are watchlist or
research cards, not the next two listings.

Local fit/perf authority: [supported-model table](../../README.md#models-and-hardware),
[Q27/Step A/B receipt](../27bab-20260810/RESULTS.md), and
[96 GB Q122 byte receipts](../model-96gb-20260806/ASSESSMENT.md). Parameter/architecture checks
used the official [Qwen](https://huggingface.co/Qwen/Qwen3.5-122B-A10B),
[Nemotron](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), and
[Inkling](https://huggingface.co/thinkingmachines/Inkling-Small) repositories.
