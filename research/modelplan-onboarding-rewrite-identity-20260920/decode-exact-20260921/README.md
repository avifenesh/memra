# Decode-exact fallback at the prefill crossover (#580)

The native repair verdict is pending a fresh reviewed build. This record preserves
the failures that exposed the problem; no tolerance, format, or support state changes.

On one RTX PRO 6000 Blackwell, `retained-capture` with the pinned Qwen3-0.6B source
and `MEMRA_FAST=0` passes the four- and eight-token comparisons but fails at sixteen
tokens. Attempt005 reports max absolute error `0.39977264` under the fixed
`0.005` absolute/relative policy. Both argmaxes are token 2; that does not satisfy
the logit gate. No receipt is issued, and the remaining 24 caller cases do not run.

Attempt004 first refused late CUDA executable-library mappings. In attempt005,
`CUDA_FORCE_PRELOAD_LIBRARIES=1` was supplied at process creation. The observer
verified the three JIT/NVVM libraries before the checkpoint's first mmap, then at
the identity and candidate2 events. Their late-loading refusal disappeared; the
numerical failure remained. Every retained reference/candidate float file is
byte-identical between004 and005. These are separate startup identities, with
unchanged arithmetic, artifact and executable. The flag must be supplied before
CUDA initialization; it never exempts a later library change or refreshes a
retained origin. NVIDIA documents this startup mechanism in its
[CUDA environment guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/environment-variables.html).

The entire failed `lib.rs` is byte-identical to accepted main `21c5f932`;
`source-basis.json` records that comparison. This is a pre-existing dispatch defect
exposed by the new qualification, distinct from the Mac ARM reference issue #548.

`matmul_decode_exact` promises the tokenwise arithmetic at every row count. Its
non-fast quantized fallback delegates to ordinary `matmul`, whose prefill
mirror/MMQ/GEMM arms become eligible at sixteen rows before the FAST switch is
consulted. The repair keeps the existing `ExactScope` active throughout the
decode-exact call, including its fallbacks. The previous scope value is restored
on return or error. Ordinary prefill outside this API is unchanged.

The native capture retains the original four/eight/sixteen-token inputs and adds
fifteen- and seventeen-token comparisons. A fresh build must pass those fixed
comparisons, the unchanged default-fast controls, graph/prime capture, and the
later library/environment refusal cases. The scope restoration CPU test checks
error propagation, an already-active scope and early drop; it does not establish
numerical parity. Full macOS engine tests hit existing Linux-only libc APIs, so
the actual guard/test bodies are compiled independently and Linux-target lint is
run separately. No unrelated platform code is changed.

`red-evidence.json` seals the original source, executable, model and raw failed
outputs. `failed-004/` and `failed-005/` remain failures. The startup observer,
complete build and lease/finalization archives are also retained by the owner.
