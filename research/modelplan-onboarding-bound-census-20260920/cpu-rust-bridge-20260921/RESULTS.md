# Rust CPU bridge — bounded execution results

Exact `a055f272be593212436dd07788ae2f17f8410f56` passed the isolated Linux x86_64
component run with installed Rust 1.97.1 and GCC/OpenMP. The source snapshot's 913 files match
Git before and after execution. The actual companion and ABI2-only refusal fixture built with
warnings denied. Four separate processes passed with requested buffered/direct I/O and pipeline
off/on; each ran five ordinary Rust controls plus the exact ignored native bridge control.
The separate legacy-library admission/predictor refusal control also passed.

The positive control uses the actual Rust CPU module, exact extracted production HostBuf/HostExps
accessors, actual bound GGUF readers and the actual native companion. Two distinct experts and
two activation rows match the original memory-path output bits after source/model host stores are
dropped and the pathname replaced. Scoped descriptors have null weights and fd -1. The old-symbol
fixture would abort on numerical fallback; admission instead returns the expected scoped-extension
error, including through predictor projection.

The only host harness doubles are an unexercised CUDA-pinned type and the unexercised hybrid router
constructor, which panics if called. Positive Rust predictor/routing and prefetch submission are
not executed by this control. Native C++ prefetch lifetime/rollback/EIO evidence remains in the
separate `65dae022` companion bank. Requested direct mode can retain the existing buffered choice
for unaligned GGUF windows; this bridge receipt alone does not claim every projection used O_DIRECT.

The compiled companion SHA-256 is
`97d7f405515d0b62a8211a3ec1639f3e82c716e36812e30b23e5de4cf2e48c2c`.
The [raw bank](attempt-1-a055f272/evidence.json) retains compiler/test commands, versions, source
and test-executable hashes, private per-process settings and every test log. Cargo was offline,
source was read-only, target/temporary files were separate, CPU threads/jobs were bounded at four,
and CUDA visibility was empty. No installation or shared configuration changes were needed.
Harness elapsed times are not performance measurements.

This is component evidence and remains subject to exact-ref review. Scoped mirrors, composite
overlay/pruning/draft contracts, root activation, full-model/serving/GPU qualification, performance
choices and main integration remain outside this result.
