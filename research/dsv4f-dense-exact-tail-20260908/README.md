# Dense M=1 exact-tree transport experiment

Base: `24555c1740f55c9108334e6f4a3c67d003f68661`.

The candidate copies the FP8 GEMV and FP32-dot M-row product/load bodies,
instantiates only M=1, and changes only reduction transport. Every thread
retains its eight-element chunks, sequential FMUL/FADD sequence, original FP8
LUT and unroll-by-two scheduling. One shared publication/barrier precedes
warp-0 `(p[l]+p[l+64])+(p[l+32]+p[l+96])`, followed by full-mask guarded
shuffle levels 16/8/4/2/1. This is the original 128-leaf tree, not warp-first
reassociation. The enclosing dsv4_gpu translation unit retains -fmad=false.

Gate-only host selector defaults OFF, decide-by 2026-09-22. M>1, grouped,
unsupported or misaligned calls retain control. Recursive M>32 tail chunks
are explicitly suppressed to avoid accidentally selecting the M=1 twin.
Selection is not device state: a captured graph freezes its kernel function.
Owners must drain streams and use separate candidate/control captured states.
Host enqueue counters include capture and are never labeled graph-execution
counts; standalone graphs require the actual candidate node name and changing
input outputs, plus repeated output identity. No serving flag is introduced.

`tools/dsv4-dense-exact-tail-gate.cu` includes the actual existing kernels.
Seven real FP8 shapes and the HC/compressor/router/head dot shapes are checked,
plus output-row and K boundaries, signed zeros, scale/code edges and
cancellation-sensitive inputs. Both F32 and BF16 dot weights are covered.
Each comparison checks finite raw-bit identity, prefix/suffix/tail guards,
immutable inputs/weights, two changing-input graph generations, repeatability,
actual node identity and selector independence of retained graphs. A separate
128-partial cancellation witness detects reassociation. M2/M33/grouped
fallbacks and raw refusal predicates are checked. Normal, memcheck and
synccheck runs on both GPUs are required, without suppressions.

Compile command (remote only):

```sh
nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off \
  -arch=sm_120a -lineinfo -Xptxas=-v tools/dsv4-dense-exact-tail-gate.cu \
  -lcublasLt -lcublas -ldl -o <owned-target>/component
```

No component or model speed is claimed at the source checkpoint. Emitted
barrier/register/spill inspection is required; source barrier counts are not
native wall savings. Existing selected-FP4 warp reduction was flat in its
older full-model program; this is a different dense consumer under full
replay, not revival of that expert door or evidence of a win.

Full-model integration is pending root coordination with the cadence owner.
The authorized comparison uses separate candidate/control graphs and equal
first-capture treatment, device sampler+diet, split-K OFF, cadence OFF and
GU N32 OFF. Full correctness/refusal/state/epoch/capture coverage precedes
20 sampled-envelope rows, five/block in candidate/control/control/candidate
order. A reproducible positive result around 1% qualifies; no 5% floor.
Ambiguous small results permit only one root-informed reverse confirmation.
Flat/negative evidence removes the experimental selector/twins in this lane.

No local rig builds, tests, lint or gates. Pushes use MEMRA_SKIP_PERF_CI=1;
hooks that execute local gates are disabled under the owner's instruction.
Raw hardware/controller/compile/numeric/model receipts live in private ops.
Keep the PR draft; root review is required before engine merge.
