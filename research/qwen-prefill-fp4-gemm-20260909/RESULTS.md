# FP4 activation ceiling: refused, 2026-09-09

The existing SM120 K64 FP4 program is 2.538x / 2.250x faster than the current repacked-weight W4A8 control at the two requested chunk-1024 GEMMs. Activation quantization is included. This is a synthetic kernel result, not a Qwen quality or serving result.

One binary, one process, six interleaved samples per arm, ten calls per sample, alternating forward/reverse arm order, nine untimed warmup calls per shape. CUDA events time device work including quantization and, for the SiLU rows, activation construction. RTX 5090 desktop, 170 SM, CUDA 13.0.88, sm_120a, automatic clocks, 33-39 C over the short cell. The entire cell lasted about three seconds; this is the requested first microbench, not sustained thermal qualification. Seed 20260909. Inputs are independent normal(0,1); weight nibbles are seeded random with finite E4M3 scale codes 40..64. These uncalibrated synthetic weight magnitudes explain the large absolute error units. No model weights or Qwen activations were used.

| K | N | SiLU/mul included | Arm | Median ms | GEMM input tok/s | Effective TOP/s | max abs vs W4A8 | max abs / max abs reference | relative L2 |
| ---: | ---: | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5120 | 17408 | False | Current W4A8, RP | 0.718235 | 1425717 | 254.15 | 0.000000 | 0.000000 | 0.000000 |
| 5120 | 17408 | False | Existing FP4, row scale + scale search | 0.282986 | 3618559 | 645.04 | 102.475979 | 0.087183 | 0.086135 |
| 5120 | 17408 | False | New row-local quantizer, fused when SiLU | 0.286523 | 3573882 | 637.07 | 121.819855 | 0.103640 | 0.095267 |
| 5120 | 17408 | True | Current W4A8, RP | 0.744304 | 1375782 | 245.24 | 0.000000 | 0.000000 | 0.000000 |
| 5120 | 17408 | True | Existing FP4, row scale + scale search | 0.314888 | 3251950 | 579.69 | 63.304703 | 0.088010 | 0.085875 |
| 5120 | 17408 | True | New row-local quantizer, fused when SiLU | 0.315466 | 3245996 | 578.62 | 65.469055 | 0.091019 | 0.090514 |
| 17408 | 5120 | False | Current W4A8, RP | 0.718030 | 1426124 | 254.22 | 0.000000 | 0.000000 | 0.000000 |
| 17408 | 5120 | False | Existing FP4, row scale + scale search | 0.319152 | 3208503 | 571.94 | 181.864441 | 0.080568 | 0.086044 |
| 17408 | 5120 | False | New row-local quantizer, fused when SiLU | 0.332861 | 3076361 | 548.39 | 202.374908 | 0.089655 | 0.095250 |
| 17408 | 5120 | True | Current W4A8, RP | 0.816386 | 1254309 | 223.59 | 0.000000 | 0.000000 | 0.000000 |
| 17408 | 5120 | True | Existing FP4, row scale + scale search | 0.406808 | 2517158 | 448.70 | 114.911930 | 0.089310 | 0.085896 |
| 17408 | 5120 | True | New row-local quantizer, fused when SiLU | 0.500571 | 2045663 | 364.66 | 126.258103 | 0.098128 | 0.090577 |

The new quantizer is not selected: it is slower and has higher synthetic error than the existing FP4 quantizer. At the down shape with SiLU/mul included, existing FP4 takes 0.406808 ms and the fused candidate 0.500571 ms. Merely removing the intermediate activation write did not pay for this quantizer's extra work. The fused kernel recomputes SiLU/mul in its row-amax and packing passes; a future fusion should retain/reuse those values and preserve the better scale selection. No runtime door was created for this losing probe.

Owner steering closes the entire FP4-activation arm. The 2.25-2.54x speed and about 8.6% relative L2 deviation are the FP4-activation ceiling, not a candidate for model gates. Prior refusal: `research/prefill-gemm-20260806/VERDICT.md` (native FP4 precision block) and `research/fp4-act-scoping-20260806/BRIEF.md` section 6 (W4A4 rescue failed the widened corpus, 4/10). Owner: "activation quant in serving path hurt correctness every try; not a perf lever". No margin/quality runs will spend GPU time re-proving that verdict. No FP4-activation door lands. The fused quantizer is also closed, with its negative receipt retained. The lane pivots to the register-only INT8 K16 roof and byte-identical W4A8 kernel engineering. Its relative L2 error against W4A8 is about 8.6% on these synthetic GEMMs. That is not a quality score or a margin receipt. Zero argmax flips, same-binary spec/plain K=1..8, cold/restored boundary identity, quality twin, sampled engagement, cold TTFT, decode and eight-turn cache gates are all still pending. No model or serving win is claimed.

Current W4A8 uses the production split-plane repack with RP=1 and default pipeline dispatch; the FP4 reader takes byte-equivalent interleaved GGUF blocks. Production integration must handle the existing RP layout explicitly and must route all affected prime row counts, including restored short suffixes, through the same row-local program. A prefill-only large-M switch with an incompatible short-tail numeric path is not acceptable.

No build cache was used. `RUSTC_WRAPPER=` and direct nvcc compile/link avoid the bundled-static-library sccache trap. `raw/symbols.txt` contains the new `fp4gemm_quant<false/true>` symbols, absent from the unchanged W4A8 object. SASS contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` and `IMMA.16816.S8.S8`. First build failure is retained: W4A8's object also references the separate f8f4 quantizer; adding that existing translation unit fixed the link. It is linked but not called by the benchmark.

Binary SHA256: `5e745e013ac6a6681275a9ca1d5413d66e29dd20665ba5c8e655f54f5d8f58c5`.
Engine base: `182819614874be818ff8bef0cfaf13cb3b8051e1`. All changes at this checkpoint are under this research directory. Issue #408 is active; no PR, hosted CI, merge, release, fleet change or deployment has occurred. The local rig ran no gates or builds.

The job held the canonical GPU lock, checked an empty compute list before launching, and released the lock when done. Post-job compute list was empty. Operational identity and lock/PID readback are in the private companion. Worktrees and scratch are retained because the owner requested a steering checkpoint before integration; this is not lane closure.

## INT8 pivot checkpoint

See [ROOF.md](ROOF.md) for the published 838 dense INT8 TOP/s peak, measured 515.116 TOP/s K16 roof, 48.92% profile attainment, and six byte-identical tile variants that all lose to current dispatch. FP8 was not built: INT8 is not near its measured roof. The lane remains in same-program INT8 research, before integration.
