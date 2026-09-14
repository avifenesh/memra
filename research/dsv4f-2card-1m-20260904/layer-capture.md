# Whole-layer capture instrument, 2026-09-05

The complete model gate passes on the two RTX PRO 6000 Max-Q cards for both
device and active-host C4 residency. Binary:
`dd048ddc5845c6238e2c1235829c08c1c8190a94cfed025d953a5eb146435c0d`.
Every first speculative round captures and executes all 43 trunk layers once,
and matches the uncaptured logits-derived sampled stream and persistent state.
The census checks mHC, sparse attention and routed-expert kernel families in
each layer. Example C4 layers contain 102 kernels plus 49 memcpy nodes; HCA
examples contain 58 kernels plus 15 memcpy nodes. There is no warmup that would
mutate caches twice.

This is not persistent replay. The instrument drops each graph after its one
execution. Position/compressor scalars, logical cache offsets and copy targets
still need live bindings before reuse across rounds, and EP needs a multi-device
graph contract. Neither throughput nor a graph serving default is claimed.

## Event-tracking failure and repair

The initial local fixture allocated buffers with cudarc event tracking enabled.
Disabling tracking only around capture was too late: existing buffer pointer
guards still record their pre-existing events. The first capture executed, but
a subsequent capture failed with `CUDA_ERROR_INVALID_VALUE` while querying
capture status. The normal DSV4 loader already disables tracking before tensor
allocation. The instrument now requires that allocation contract, refuses the
tracking-on mode before running the body, and does not toggle it late.

`layer-capture-event-fix-5090.log` passes the forced refusal, exactly one GPU
execution, kernel census, deliberate body-error cleanup and subsequent capture.
The target complete-model capture gate then passed. Original failed fixture
receipts are retained; this was not a model or hardware failure.
