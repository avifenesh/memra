# FP4/INT8 component cost, issue 439

Predeclared scope: 400 Qwen3.8-27B projection slots, grouped into ten classes by
actual GGUF shapes. The primary group is the 272 post-norm input slots. The full
A4 quality negative remains unchanged. No calibration or reduced-set dispatch
is implemented.

The model's expected SHA256 is
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
Census requires all 400 frozen scale records to name rank-two NVFP4 trunk weights,
ten uniform-shape classes, and 272 input slots. Each class uses its earliest-layer
real weight tensor and the frozen v3-amax multiplier. All shape/weight/scale pins
are written before timing.

Inputs are deterministic synthetic F32 values in [-2,2), with a fixed LCG seed.
Both arms use the same resident input and split-plane weight bytes. This is a
component throughput experiment, not a model forward or a quality test.

Measure the existing `memra_mmq_nvfp4_w4a8` and
`memra_mmq_nvfp4_calibrated_prefill` entrypoints directly. Each sample includes
activation quantization and GEMM; allocations, uploads and weight repacking are
outside the timed interval. Both arms get their own correctly sized scratch.
Clip statistics are disabled. CUDA events measure stream time; host wall time
is retained separately.

Row counts: 512, 2048, 4096. The primary aggregate is 4096, the engine's usual
long-prime cap. Per cell: 20 warm launches per arm, then ten interleaved rounds,
five AB and five BA, with 64 launches per timed sample. Aggregate using the sum
of each class's median time multiplied by its actual slot count. Report per-class
ratios, each order, ranges and the primary aggregate. Do not weight speed ratios
by MAC count or call the component aggregate a measured prefill-wall speedup.

Output controls require finite, nonzero, distinct arm outputs; each arm's full
output hash must remain stable across the timed repetitions. The invalid-scale
red arm must return 2902 before touching its null pointers. These controls prove
execution and fixture stability, not model accuracy or native qualification.

One exclusive GPU job, timeout 1200 seconds, minimum free VRAM 8192 MiB. Real
weights are read-only and loaded one class at a time; the full model is never
loaded on the GPU. Capture 250 ms clock/power/temperature telemetry throughout.
The result is local RTX 5090 Laptop evidence. Transfer to PRO 6000 requires its
own evidence. About 1.25x is the issue's engineering-value threshold, not a
measured result or an owner quality threshold.
