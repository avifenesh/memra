# Bounded prompt forecasting versus fixed K=3

Independent native requests, unchanged full model/draft heads, one forecast per request.
Adaptive K is 2 for prose, 4 for code/numeric, and 3 for mixed/unknown.
The forecaster reads only the first X user-content tokenizer tokens.
Six scenarios were registered for each model, requested format and prompt length.
Fixed K=3 is the owner-specified control here; the study does not compare
against a newly calibrated global K=2 or K=4 policy or native adaptation.

Throughput is all returned model tokens divided by complete native request time,
including reasoning, normal tokenization, prediction, cache allocation, prefill and
generation. Model loading, warmup and receipt I/O are outside that clock.
Pointwise 95% intervals use paired scenario bootstrap; this is an exploratory matrix.

Measured native recipe: `bdf9f4305b1093c4ac35a8e387c689047c938c5e`.
Runtime archive SHA-256: `53cfab8e8a4c4a01362d54d9d96fd69100f99ab445b62ba81be1c096a17634bf`.
Staged harness SHA-256: `ad0b2dd147023eaff86d12b8f4059da307ba491f7c354302209333edf5a55631`.
Research GPU: NVIDIA GeForce RTX 5090, 32607 MiB reported VRAM, driver 595.91.07.

All three prefix budgets selected the same K on the scored instruction-first
prompts. Differences among their measured rates therefore do not represent
a different depth schedule.

## Qwen3.8-27B NVFP4+Q5_K with embedded MTP

Maximum returned tokens: 8,192, selected on a separate qualification task.

| Pinned artifact | Revision | SHA-256 |
|---|---|---|
| Qwen3.8-27B-NVFP4-Q5K-mtp.gguf | `0f82b27dbb264b731e7d20f576582c871ef1969c` | `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` |

| Requested output | Prompt tokens | Pairs | Fixed K=3 tok/s | X=64 tok/s (change) | X=128 tok/s (change) | X=256 tok/s (change) |
|---|---:|---:|---:|---:|---:|---:|
| prose | 256 | 6 | 166.01 | 156.75 (-5.58%) | 156.73 (-5.59%) | 156.76 (-5.57%) |
| prose | 1,024 | 6 | 163.28 | 152.74 (-6.45%) | 152.79 (-6.42%) | 152.76 (-6.44%) |
| prose | 4,096 | 6 | 146.61 | 140.28 (-4.32%) | 140.33 (-4.29%) | 140.33 (-4.29%) |
| prose | 16,384 | 6 | 97.72 | 99.27 (+1.59%) | 99.28 (+1.59%) | 99.29 (+1.61%) |
| code | 256 | 6 | 182.21 | 177.97 (-2.32%) | 177.98 (-2.32%) | 177.90 (-2.36%) |
| code | 1,024 | 6 | 163.32 | 156.67 (-4.07%) | 156.70 (-4.05%) | 156.73 (-4.03%) |
| code | 4,096 | 6 | 117.53 | 112.33 (-4.43%) | 112.36 (-4.40%) | 112.36 (-4.40%) |
| code | 16,384 | 6 | 51.85 | 53.55 (+3.28%) | 53.57 (+3.33%) | 53.58 (+3.35%) |

Prediction and final-format coverage are distinct from throughput:

| Requested output | Prompt tokens | All-arms format-covered pairs | X=64 forecast μs | X=128 forecast μs | X=256 forecast μs |
|---|---:|---:|---:|---:|---:|
| prose | 256 | 6 | 27.92 | 37.19 | 41.04 |
| prose | 1,024 | 6 | 23.74 | 31.11 | 39.67 |
| prose | 4,096 | 6 | 23.45 | 33.11 | 44.81 |
| prose | 16,384 | 6 | 26.99 | 37.08 | 44.00 |
| code | 256 | 6 | 25.56 | 40.59 | 55.92 |
| code | 1,024 | 6 | 26.78 | 38.44 | 45.74 |
| code | 4,096 | 6 | 28.09 | 44.48 | 59.84 |
| code | 16,384 | 6 | 22.63 | 38.05 | 48.30 |

Matched loop exclusions: 0. Primary rows retain capped outputs and format failures; the JSON contains the separate all-arms-format-covered subset, individual pair gains, output-length ratios, latency ratios, prediction/K counts and pointwise intervals.

## Gemma 4 12B QAT Q4_0 with Q8_0 assistant

Maximum returned tokens: 8,192, selected on a separate qualification task.

| Pinned artifact | Revision | SHA-256 |
|---|---|---|
| MTP/mtp-gemma-4-12B-it-Q8_0.gguf | `980b060c40a8539ac159e0501a3e0f66a6365af3` | `f58dff98ecf17079c364b1205954ba85763a00ad076c5595040db0a828eaec1b` |
| gemma-4-12b-it-qat-q4_0.gguf | `29d097773436b69ff9feafd636ab4cf873786537` | `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b` |

| Requested output | Prompt tokens | Pairs | Fixed K=3 tok/s | X=64 tok/s (change) | X=128 tok/s (change) | X=256 tok/s (change) |
|---|---:|---:|---:|---:|---:|---:|
| prose | 256 | 6 | 272.32 | 258.75 (-4.99%) | 258.78 (-4.97%) | 258.57 (-5.05%) |
| prose | 1,024 | 6 | 254.76 | 237.60 (-6.74%) | 237.55 (-6.75%) | 237.45 (-6.79%) |
| prose | 4,096 | 6 | 193.17 | 182.15 (-5.70%) | 182.39 (-5.58%) | 182.41 (-5.57%) |
| prose | 16,384 | 6 | 90.40 | 88.84 (-1.72%) | 88.87 (-1.69%) | 88.86 (-1.70%) |
| code | 256 | 6 | 357.54 | 378.69 (+5.91%) | 379.05 (+6.02%) | 378.84 (+5.96%) |
| code | 1,024 | 6 | 301.21 | 325.25 (+7.98%) | 325.31 (+8.00%) | 325.64 (+8.11%) |
| code | 4,096 | 6 | 179.72 | 186.63 (+3.84%) | 186.83 (+3.96%) | 186.91 (+4.00%) |
| code | 16,384 | 6 | 62.49 | 63.20 (+1.15%) | 63.23 (+1.20%) | 63.20 (+1.14%) |

Prediction and final-format coverage are distinct from throughput:

| Requested output | Prompt tokens | All-arms format-covered pairs | X=64 forecast μs | X=128 forecast μs | X=256 forecast μs |
|---|---:|---:|---:|---:|---:|
| prose | 256 | 6 | 14.47 | 19.93 | 21.67 |
| prose | 1,024 | 6 | 13.46 | 16.90 | 16.34 |
| prose | 4,096 | 6 | 16.47 | 20.34 | 17.30 |
| prose | 16,384 | 6 | 16.49 | 23.25 | 24.60 |
| code | 256 | 6 | 11.53 | 16.94 | 17.25 |
| code | 1,024 | 6 | 10.82 | 20.57 | 22.29 |
| code | 4,096 | 6 | 11.99 | 20.64 | 22.72 |
| code | 16,384 | 6 | 12.59 | 20.65 | 23.39 |

Matched loop exclusions: 0. Primary rows retain capped outputs and format failures; the JSON contains the separate all-arms-format-covered subset, individual pair gains, output-length ratios, latency ratios, prediction/K counts and pointwise intervals.

## Evidence boundary

This measures a native, single-GPU research driver with a fresh cache for each request.
It does not establish HTTP serving, concurrent-request performance, continuous-session KV reuse,
or correctness of each generated program. Python parsing is a format-coverage diagnostic.

Native/source/model identities and raw-audit status are in the accompanying JSON.
JSON SHA-256: `a3d70de0619ea89166196979c6f0602d01376bef9563dbc7cd5301e94115c926`.
