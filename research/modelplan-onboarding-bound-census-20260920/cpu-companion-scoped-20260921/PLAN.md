# Scoped CPU companion candidate — validation pending

This candidate adds optional scoped token, rows and detached-prefetch entrypoints while preserving
the existing ABI v2 layouts, symbols and numerical routines. Prepared scoped readers feed the same
file-generation cache key, aligned read buffers, annex promotion, compute and accumulation paths.
They expose no raw descriptor; active scoped mirror-map requests explicitly refuse pending a bound
alternate-reader contract. This is not complete CPU feature parity or root activation.

Shared reader owners persist through call I/O drain or detached state completion. Thread-local
scratch releases scoped readers after the call drains. Queue publication inserts a complete owned vector as one deque batch with the strong exception
guarantee, so preparation failure cannot leave partially submitted jobs
referencing freed state. Prefetch preparation owns its annex claims and counters until handoff.

The committed tiny test includes the actual production translation unit. Planned bounded Linux
checks use g++ 13.3, OpenMP with four threads, no visible CUDA devices, a separate source snapshot,
target and receipt directory, and no shared configuration changes. Cases compare ABI v2 versus
scoped single-token and two-row output bits, cold reads, warm cache reuse, detached reader lifetime
and metadata failure ownership, in buffered/direct I/O and pipeline off/on processes. Synthetic
matrices are 128/512-wide Q8_0; this is not a model-scale benchmark or performance result.

The C++ candidate has not yet been compiled or executed at this freeze. It is being frozen to bind
that forthcoming build, not to claim a source/native gate. Existing `ed67b9a8` Rust/C reader controls
remain separate. Rust engine descriptor dispatch and mirror adoption are still pending; no engine
root starts using these new companion symbols in this commit. Further source and native CPU review
is required before adoption; GPU qualification and main integration remain coordinator-owned.


## Follow-up freeze before the second CPU attempt

The initial `5fc078c4` build never reached test execution: its transport snapshot omitted the
production IQ3S include. The original compiler log/result/source manifest are retained in
`attempt-1-5fc078c4/`. The next snapshot must include the complete quoted-include closure.

The test now deterministically blocks callbacks until after prefetch returns and caller owners
are released/unlinked, checks retained liveness, then unblocks and verifies original annex bytes
and final releases. Second-reader metadata failure proves partial preparation rollback before
any I/O is submitted. Demand/prefetch EIO controls check drain, nonpublication and successful
same-key retry. The queue uses one owned vector per deque entry, preserving FIFO and atomic
publication without a new allocation per individual I/O job. These new controls remain unrun
until the next frozen Linux build; no previous failure has been overwritten.

The complete `d71f4440` snapshot reached GCC source checking but stopped on
`-Werror=subobject-linkage`: `PrefetchClaims` was outside the anonymous namespace while its
`CacheKey` field was internal. The next minimal fix gives the guard the same internal linkage.
No test case executed on `d71f4440`; its exact source manifest and compiler diagnostic remain
in `attempt-1-d71f4440/`. The Rust bridge work is separate and not part of these CPU freezes.
