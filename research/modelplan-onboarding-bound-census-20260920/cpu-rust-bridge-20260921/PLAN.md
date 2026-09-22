# Rust scoped CPU descriptor bridge — execution pending

Bound expert windows now prepare retained reader handles for the optional scoped companion
entrypoints. Every scoped descriptor has a null weight pointer and fd -1; missing token/rows/
prefetch extension symbols refuse rather than falling back to ABI2. Raw and memory descriptors
retain their existing ABI2 calls and avoid allocating reader arrays. Buffered/direct selection
matches the existing alignment rule, and active scoped mirror requests still explicitly refuse.

Jobs and predictor tables retain reader handles; temporary descriptor arrays live through the C
call, and the companion retains before detached work. Predictor eligibility excludes pruned,
write-combined and CPU-unsupported encodings; other descriptor/reader/admission errors propagate
through table construction instead of disappearing as optional absence.

Local validation before this freeze: affected GGUF/CLI suites and Clippy pass; Linux-target
engine/server lib/bin/test Clippy passes with DOCS_RS documentation stubs. The new host harness
cross-typechecks the actual CPU module and exact extracted production host-storage types/methods.
Only unexercised CUDA-pinned type and hybrid-router construction have doubles. CUDA is not linked
or initialized by this harness; those omitted paths are not qualified.

The planned isolated Linux run builds the actual companion shared object, then runs the actual
Rust dispatch test: two different experts and two rows must match the original memory path bit
for bit after every source/model host store is dropped and its pathname replaced. A separate
ABI2-only symbol fixture aborts if numerical fallback is attempted; the scoped admission and
predictor paths must instead report errors. Prepared process-local environment uses installed
Rust 1.97.1, GCC/OpenMP, four threads/jobs maximum, offline Cargo, private source/target/receipts,
and no visible CUDA devices. This freeze does not claim those execution results yet.

Root activation, scoped mirror adoption, overlay/pruning and external draft contracts, end-to-end
model/serving correctness and performance qualification remain pending. No positive support state,
GPU use, source-independent native receipt reuse, or main merge is implied.
