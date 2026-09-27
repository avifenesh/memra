# Aligned readers and Rust prefetch handoff — CPU results

Exact `af0cf8bcf6d6ceb0b1642a2e0887ed9ccba33135` passed all 14 commands in the
isolated Linux run: three warning-clean library builds, four aligned parity cases, four
prefetch handoff cases, aligned mirror refusal, legacy admission refusal and original parity.
Rust 1.97.1, GCC/OpenMP, four CPU threads/jobs maximum and empty CUDA visibility were retained.
All 914 source files match the frozen Git tree before and after execution.

Every aligned parity case uses six 69,632-byte windows at offsets divisible by 4096. In direct
mode, all six reader descriptors report direct=1, backed by the new F_GETFL/O_DIRECT check.
The existing cache reports six cold misses and exactly 417,792 inserted bytes. Token and row
outputs remain bit-identical to the same-program memory control. These are actual aligned Rust
reader executions, separately recorded from `a055f272`'s unaligned buffered-fallback controls.

The prefetch cases invoke the shared production Rust submission hook with controlled expert IDs.
The observer pauses actual native I/O before the real Rust reads; it does not replace their data.
After the call returns, argument/model/source owners are dropped and the pathname replaced. Six
weak contexts remain alive with submitted=6 and inflight=6. Unblock ends with final_live=0,
inflight=0 and all original annex bytes verified. This passes in buffered/direct mode with both
pipeline settings. Prediction/routing computation is still unexecuted (`router_executed=0`).

The active aligned mirror request refuses explicitly. An ABI2-only library still cannot receive a
scoped numerical fallback, and predictor admission errors propagate. Ordinary production exports
contain no observer hooks. Its library SHA-256 is unchanged from the previous bridge run:
`97d7f405515d0b62a8211a3ec1639f3e82c716e36812e30b23e5de4cf2e48c2c`.
The separately built observer library is
`15d5271f3fdebff290d9d5955fe17444ee4d879def8b5571ed488dc267b6e791`.

[Raw records](attempt-1-af0cf8bc/evidence.json) retain every build/case log, export listing,
source/compiler/test-binary identity and private process settings losslessly. Earlier narrower
receipts are unchanged. The host harness still replaces only the unused CUDA-pinned type and
unused router constructor; actual host-storage methods and CPU dispatch execute. No model/GPU/
serving or performance qualification, scoped mirror implementation, root activation, support
promotion or main merge is claimed. Exact-ref review remains a separate requirement.
