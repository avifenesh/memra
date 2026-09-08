# Native Spark FP8 correctness and long-context bring-up

Engine base: origin/main `8cfb182ba`; inherited Spark pack reapplied through `49a953e74`.
Validation target: nonproduction RTX 5090, sm_120a. All builds and gates ran remotely.
Model source: `XHToken/Spark-X2.5-4B-FP8` revision
`e186ce2d5a0442e1e42cad45e0c65907b84fc34c`.

The changes preserve fused-QKV FP8 codes and scale-grid row slices, add the missing
hd256 windowed prefill instantiations, and carry the checkpoint's declared dynamic
block128 FP8 activation program through decode and prefill. Unsupported native format
routes refuse rather than silently requantizing. Existing BF16 tensors stay preserved.
No new runtime environment flag or external inference kernel dependency is introduced.
The inherited rewrite-oracle diagnostic adds `MEMRA_REWRITE_ATOL` and
`MEMRA_REWRITE_RTOL` (both default 0.05); their FLAGS rows preserve the strict failed
qualification verdict. See GATES.md for the PR and merge requirements.

The model remains NativeReference while optimized numerical qualification is open.
The source-level checkpoint gate passes; component gates do not imply production
admission or a completed model/hardware serving qualification.
