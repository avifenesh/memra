# Qwen code at fixed K=3: confidence cutoff grid

This is a native, independent-request comparison of draft stopping at a
fixed K=3 ceiling. C=0 avoids confidence computation; positive C settings
can shorten each draft round. No online confidence-threshold learner was
measured by this grid, and no served default changes.

Six frozen synthetic helper scenarios were paired across settings at each
prompt length. The metric is returned tokens divided by complete native
request seconds, including tokenization, prefill, drafting, verification
and detokenization. Model load, warmup and receipt writing are excluded.
Pointwise intervals resample six whole scenarios per length and are
exploratory; they do not adjust for choosing among cutoff settings.

| Code prompt tokens | Pairs | C=0 tok/s | C=0.15 tok/s (change, 95% interval) | C=0.30 tok/s (change, 95% interval) | C=0.30 with zero-draft tok/s (change, 95% interval) |
|---|---:|---:|---:|---:|---:|
| 256 | 6 | 180.83 | 185.01 (+2.31%, [+0.08%, +4.68%]) | 182.76 (+1.06%, [-5.05%, +6.51%]) | 178.38 (-1.36%, [-4.41%, +1.51%]) |
| 1,024 | 6 | 162.22 | 162.00 (-0.14%, [-2.54%, +5.19%]) | 158.86 (-2.07%, [-5.58%, +0.96%]) | 155.12 (-4.38%, [-8.97%, -1.04%]) |
| 4,096 | 6 | 116.86 | 115.17 (-1.44%, [-8.26%, +4.32%]) | 112.76 (-3.51%, [-11.02%, +5.20%]) | 113.04 (-3.26%, [-7.57%, +0.33%]) |
| 16,384 | 6 | 51.62 | 84.74 (+64.18%, [+8.34%, +104.74%]) | 54.27 (+5.13%, [-9.20%, +23.05%]) | 51.21 (-0.80%, [-14.48%, +14.35%]) |

Across all code lengths, the pooled rates below count each returned token
and request second once. Their scenario groups share topics across lengths,
so only the per-length rows carry bootstrap intervals.

| Setting | Pooled code tok/s | Change vs C=0 | Shortened code rounds / all code rounds | Format-covered code requests |
|---|---:|---:|---:|---:|
| C=0 | 100.64 | +0.00% | 0/2790 | 24/24 |
| C=0.15 | 107.49 | +6.81% | 297/4147 | 24/24 |
| C=0.30 | 97.01 | -3.61% | 302/2574 | 24/24 |
| C=0.30, zero-draft | 96.40 | -4.21% | 521/2757 | 24/24 |

Matched loop exclusions: 0.
The primary rows retain capped outputs and format misses. The JSON has
the separate all-arms-format-covered subset, paired changes, output-length
and latency ratios, round histograms, and whole-scenario intervals.

## Identity and scope

- Model: `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@0f82b27dbb264b731e7d20f576582c871ef1969c`, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`
  SHA-256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`; full `15,705,922,304` bytes.
- Drafter: Qwen embedded MTP with full target vocabulary.
- Research GPU: NVIDIA GeForce RTX 5090; one device.
- Measured binary SHA-256: `3b9e6c3e8f8bd50b4d7e6c7859da35ec70ce02b569f5fb84c4066b703c38fa71`.
- Patched runtime source SHA-256: `98a0a0118155663aa9abae29acdef845838b9367c6f6d17e87da7fe2fb1957a9`.
- Original sealed runtime SHA-256: `53cfab8e8a4c4a01362d54d9d96fd69100f99ab445b62ba81be1c096a17634bf`.
- Sampler: temperature 0.7, top-k 20, top-p 0.95; default thinking.
- Cache: fresh native cache for every request; no HTTP or concurrency claim.
- Correctness: target-only greedy oracle passed for every C setting,
  followed by matched driver tapes and seeded sampled reruns at the
  study's 0.7/20/0.95 decode shape.
- Format coverage checks fenced Python syntax, not functional correctness.
