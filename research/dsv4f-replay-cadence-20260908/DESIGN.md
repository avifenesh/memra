# Three retained forward cadences

Base source: `24555c1740f55c9108334e6f4a3c67d003f68661`. One isolated owner,
branch `lane/dsv4f-replay-cadence-20260908`; issue #4 remains root-owned.
Current implementation is unmeasured. The original full-token replay remains
the exact oracle. Post-#358 main changes affect server admission/tests, not the
replay/compressor program. GU/dense candidates are separately owned and OFF for
this lane's first model comparison.

The first replay step captures three complete forward graphs per rank before
launching any: ordinary, C4-only emission, and C4+C128 emission. Host compressor
high-water/checkpoint metadata is restored after each capture pass. Live device
AR epochs are not advanced during capture. Actual live position selects the
retained graph: `(pos + 1) % 128 == 0`, otherwise `% 4 == 0`, otherwise ordinary.
There is no subsequent capture or node-parameter update loop.

Slots are `[ordinary/full forward, commit, C4 forward, C4+C128 forward]`.
All variants retain control upload/application, two live append copies per
compressor, selectors/attention bounds and both collectives through all 43
layers. The host omits inactive compressor emission/shift enqueue calls during
capture. Emitted bodies use the existing guarded kernels with identical active
geometry/arithmetic/reduction order. CUDA-IF is not reintroduced; no CUDA or FFI
implementation changes are needed.

The existing commit/head/device-sampling graph is shared by all three forward
variants. Both refusal words remain a real host barrier before commit. Paired
capture abort/drain/fail-stop, rollback/quarantine, stable model/cache lifetime,
64-bit uniform storage and per-block AR sequence checks remain mandatory.
Capacity 512..1024, replay input positions below 512, plain 256-prime/256-sampled
device-cache TP2/expert-ID EP, device sampler+diet ON, split-K OFF. This is a
request-local default-OFF diagnostic, not a serving route.

Expected forward kernel censuses are 2741 / 3140 / 3240 versus the original
3240. Every forward must still show 86 AR kernels, one embedding and 86 HC posts
per rank, with zero unsupported nodes. Device counters are separate for all four
slots. A 256-output row at input positions 256..511 must count
`[192, 256, 62, 2]` per rank, including the shared commit counter. The original
full-replay arm must count `[256, 256, 0, 0]`. The earlier 396.125 inactive-kernel
estimate applied to its 32-step window; this span's source count is 398.46875.
Neither count predicts wall savings.

The focused real-compressor component captures original full replay plus the
three variants, compares all pending/store bytes over 512 changing positions
plus nine decreasing/wrap positions on each GPU, checks nonzero seeded state,
token/position/ring-slot and full-u64 input freshness, and validates device
variant counters/censuses. It makes no sampling/model claim. The actual Rust
owner test extends partial-submit/capture cleanup to every slot and checks live
controls through all slots; the paired completion policy subprocess regression
remains required because owner storage dimensions changed.

The model gate compares full replay with cadence replay at every generated
step before scoring. It checks both-rank cache/hidden, logits, sampled tokens,
AR epochs, all-variant graph census and capture/device counters. Eight live
refusal cases cover ordinary/C4/C128 across both ranks, including nonzero prior
emissions and ring wrap, with no commit and retry/reset quarantine. Both scored
states are newly armed after correctness: A pays its first full-forward capture,
B pays all three first forward captures, and both pay their first commit capture
inside their first timed row. Prefix reset preserves stable allocations.

One same-load A5/B5/B5/A5, 20 eligible rows, measures the complete
sample_plus_forward_envelope without profiling. `--full-token-replay-cadence-baab`
is a separately scheduled bounded confirmation, never an automatic sweep.
Consistently positive small gains are retained; no arbitrary percentage floor
applies. A tiny/noisy signal can receive at most one reverse-order confirmation.
Flat/negative evidence removes only this cadence door and its dedicated support
in the same lane; the prior measured full replay remains available.

Remote-only build/components/model work uses the assigned development pair,
exact role/UUID checks, arch120a, two build jobs and the nonblocking shared fd9
GPU lock. Source checkpoint/draft PR and independent review precede full-model
execution. All scientific raw data lives in the companion private cadence lane.
