# GGUF disk views retain the parsed shard

GGUF shard mappings now have shared ownership. After metadata validation, bound disk access can
retain the exact parsed mapping, opened file and tensor range without another map or pathname
reopen. The raw extent factory is crate-private; the bound runtime still exposes only its opaque
view. Derived MLA and non-identity transform requests explicitly refuse a direct physical extent.
Ordinary raw-source disk policy and engine root activation are unchanged.

The single-file control proves pointer-identical tensor bytes, bound-runtime disk wrapping and
refusal of transformed/derived reads, then drops the loader/binding/source/file and replaces the
checkpoint pathname. The retained view and positioned reader still return the original bytes,
and out-of-range reads fail. The split-shard control proves that a tensor in shard 1 retains that
shard's file/mapping and survives the same drop/unlink/replacement sequence. Existing opened-source
identity tests continue to pass with shared mappings.

Full GGUF library tests: 358 passed, 2 ignored. The additional transformed/derived refusal
assertions pass in the targeted GGUF test. All-target GGUF Clippy with warnings denied, formatting
and whitespace checks pass. Raw intermediate failures, final tests, source hashes, the original
preserved map patch and its pre-intake CPU test log are retained losslessly.

This completes the earlier saved GGUF map work in the integration branch. Residency metadata,
SpillCtx policy/placement, scoped CPU expert access, other source/derived/draft paths and final
source/native gates still remain before universal activation. No GPU execution or native
checkpoint qualification is claimed. The newer transfer/VMM stack will be imported only through
the coordinator's next reviewed #542 reference.
