# Native request-depth result

One RTX 5090, full-vocabulary Qwen and Gemma MTP, eight-turn native
prompt-checkpoint reuse, approximately 16K initial input, 512-token output
caps, temperature 0.7, top-k 20 and top-p 0.95. Output includes default
reasoning. Rates use complete native request time, including routing and
suffix prefill; this is not HTTP, task-quality or production qualification.

| Model | Global K | Prose/code/numeric/fallback K | Native tok/s | Fixed tok/s | Span tok/s | Prompt tok/s | Clean pairs |
|---|---:|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B NVFP4+Q5_K, embedded MTP | 3 | 3/2/3/3 | 104.800 | 106.225 | 106.222 | 105.377 | 6/6 |
| Gemma 4 12B QAT Q4_0, Q8_0 assistant | 3 | 3/3/4/3 | 179.042 | 185.064 | 183.802 | 184.995 | 6/6 |

All policies used the same calibration runs. A request-class change
needed support from at least two independent calibration conversations.
Each excluded set retains every raw arm and its independently reproduced loop flag.

Qwen3.8-27B NVFP4+Q5_K, embedded MTP:
- Prompt versus native: +0.550%; 4/6 paired wins.
- Prompt versus calibrated: -0.798%; 1/6 paired wins.
- Prompt versus context: -0.796%; 1/6 paired wins.
- Routed-call CPU time: 0.304 ms total over 48 requested turns.
- Performance-decision minimum met: true.

| Requested kind | Scored turns per arm | Fixed tok/s | Span tok/s | Prompt tok/s |
|---|---:|---:|---:|---:|
| prose | 18 | 83.988 | 83.989 | 81.268 |
| code | 12 | 116.913 | 116.899 | 116.175 |
| numeric | 12 | 145.997 | 146.005 | 143.309 |
| mixed | 6 | 114.466 | 114.460 | 111.641 |

The JSON report also groups nonterminal round tokens by the span rule's
prose/code/numeric label at round start. This diagnoses within-answer
mixing; it is not a semantic-quality score or a complete token census.

Gemma 4 12B QAT Q4_0, Q8_0 assistant:
- Prompt versus native: +3.325%; 6/6 paired wins.
- Prompt versus calibrated: -0.037%; 2/6 paired wins.
- Prompt versus context: +0.649%; 4/6 paired wins.
- Routed-call CPU time: 0.286 ms total over 48 requested turns.
- Performance-decision minimum met: true.

| Requested kind | Scored turns per arm | Fixed tok/s | Span tok/s | Prompt tok/s |
|---|---:|---:|---:|---:|
| prose | 18 | 145.999 | 145.480 | 145.473 |
| code | 12 | 228.956 | 226.631 | 229.146 |
| numeric | 12 | 227.037 | 225.450 | 230.156 |
| mixed | 6 | 223.010 | 214.757 | 218.552 |

The JSON report also groups nonterminal round tokens by the span rule's
prose/code/numeric label at round start. This diagnoses within-answer
mixing; it is not a semantic-quality score or a complete token census.

Greedy fixed-depth/target gates, prompt/native greedy identity, prompt/schedule
sampled identity, complete token/time accounting, per-request K engagement and
native checkpoint reuse are reconstructed from raw records.

Native binary recipe: `fcdea815d59a4dcff75e62042796237ce5a465af`.
Run orchestration: `92ee8e4454727153c667a67e03cf307706e325f5`.
Exact binary, source, input, timing and exclusion records accompany this report.
