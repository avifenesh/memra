# Standalone full-model helper, review checkpoint

`dsv4_dense_exact_tail_gate <model-dir> <pinned-source.txt> <new-output-dir>`
uses merged full-token replay public APIs. A separate bin is discovered through
Cargo's existing src/bin auto-discovery. The only FFI declarations are the
already-reviewed dense selector and host enqueue counters. Existing Rust
replay gates, graphs, runtime and AR files are not modified.

The pinned program retains full replay, device sampler and small-kernel diet;
split-K is explicitly cleared. This source base contains neither cadence nor
GU N32 implementations. No future-main numeric/dispatch claim is made. The
model is boxed at a stable address and outlives all captured states. Gate
selection is host-thread-local and frozen into each captured kernel function.
The helper drains ranks and changes selection only before an uncaptured arm's
first execution, or for the separate eager correctness witness. Retained
graphs never read the selector and cannot change their dense semantics.

## Pre-timing checks

One common 256-token prefix is primed on control. For 256 changing samples,
eager OFF, replay OFF and replay ON must return identical next tokens, full
logit hashes and both-rank cache/hidden digests at every step. This checks the
first captured token too. All 72 AR epochs/rank advance by their exact wrapped
step count. Each graph has one capture per segment, 86 ARs/one embedding/86
HC posts per forward rank and no unsupported nodes.

Verbose DOT is read via the public dump API. Kernel ID records must contain
494 FP8 and 253 dot nodes per forward rank, plus two head dot nodes in rank-1
commit/sample. ON requires all those nodes to have candidate names and zero
control names; OFF requires the inverse. This includes the BF16 head if
admitted. The actual graph structure counts match across arms. Host enqueue
counters are checked separately: ON first capture [988,508], OFF [0,0], and
zero further host enqueues during retained replays. Device replay counters,
not host enqueue counters, establish replay execution.

Both captured arms then reset the same allocations to the immutable prefix,
replay the complete tape again, and reproduce final output/state/epochs and
unchanged DOT hashes. Twelve live refusal cells (both dense arms, two ranks,
positions 259/383/511 at layers 0/21/42) run after capture. Each requires both
refusal words as expected, unchanged cache/position, no commit-segment advance,
and refusal of ordinary retry and prefix-reset retry without new execution.

## Timing comparison

Qualification states are dropped. BOTH scored states start uncaptured and
allocate/restore outside timing. The first scored ON row and first scored OFF
row each perform their real first graph capture INSIDE measured time. No cold
capture is silently removed or compared against a previously captured control.
The helper asserts exactly one such row per arm.

Twenty rows: five/block, ON/OFF/OFF/ON. Initial carry draw is shared outside
timing; 256 forward/refusal/commit/head/sample/readback steps and the final
next draw are inside sample_plus_forward_envelope. Equal first-capture costs,
source inputs, samplers and graph-state ownership are explicit. Every row
checks eligibility, exact token/final-state/next-draw identity, capture/replay
counts, zero refusals and device epochs after timing. DOT is captured after
each arm's first and last scored rows, never inside timing. Both-arm mean and
pooled rates use all ten rows including first capture. No automatic verdict,
merge, claim transfer, or minimum 5% floor is implemented.

`--reverse` changes only block order to OFF/ON/ON/OFF and is not an automatic
second cell; it requires root's scheduling decision for an ambiguous signal.

Status: source checkpoint awaiting narrow independent helper review and
remote compilation. Model execution has NOT run and must wait for root's
review/source pin and GPU scheduling. No cadence/runtime integration overlap.
