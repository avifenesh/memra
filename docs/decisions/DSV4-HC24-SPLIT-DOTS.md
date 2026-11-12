# DSV4 HC24 split dots: S16 by default, 2026-09-09

## Decision

The device HC-24 pre-attention and pre-FFN sites (f32, N=24, K=16384) compute their dots as HC24
split dots with S=16 slices by default. `MEMRA_DSV4_HC_DOT_SPLIT` unset, `1` or `16` selects S16;
explicit `0` (or any other value) restores the sequential dots; `8` and `32` are explicit opt-in
classes. The owner
accepted the default on 2026-09-09. Contract and rollback: the `MEMRA_DSV4_HC_DOT_SPLIT` row in
[FLAGS.md](../FLAGS.md).

The kernels are `dsv4_hc_dot_split_partial_kernel<S>` and `dsv4_hc_dot_split_reduce_kernel<S>` in
`crates/memra-engine/cu/dsv4_dense_m1_exact_tail.cuh` ([KERNELS.md](../KERNELS.md)). Each of the
24*S blocks per token row sums a contiguous K slice in the exact-tail 128-leaf tree; one 32-thread
block per row then sums the partials in ascending slice order with explicit round-to-nearest f32
adds. There are no atomics and no fused multiply-add. Restarting accumulators and summing slices
changes association, so this is a numeric class of its own, not bit-identity with the exact-tail
dots, and each S is a distinct class. Since 2026-09-23 (memra #660) every row count takes the
split, so a verify, prefill or batched row carries the bits of the same row decoded alone.

## Rejected

- Sequential dots as the default: the S16 arm is faster (below) at unchanged task accuracy.
- S32 as the default: it was the fastest component, but its 1.407% advantage over S16 did not
  justify rebuilding and re-review, and only S16 has the model campaign receipts. S32 stays an
  explicit class.
- S8 as the default: it stays an explicit class.

## Measurement that settled it

- Speed, S16 against sequential: sampled pooled +2.701174% forward and +2.591532% reverse, 20
  eligible rows per order, first capture timed.
- Drift, all 128 DRIFT-R3 input hashes matching: 49 of 2048 top-1 changes (2.392578%); KL OFF to ON
  mean 0.005672511, max 0.421308907; ON to OFF mean 0.005758277, max 0.483190648; greedy 16/64
  identical; 2099 of 4096 matching tokens.
- Task accuracy: control 177/300, split-K plus HC 177/300, McNemar p = 1.00.
- Components: both-rank S8, S16 and S32 components with zero-error memcheck and synccheck, two
  fresh process observations per arm with external token, logit, cache and hidden equality.
  Source `d42196214`.

The component and campaign receipts are private. The text as it stood in the kernel inventory is
kept in [history/KERNELS.md](../history/KERNELS.md#hc24-split-dots-numeric-class-owner-accepted-default-on-s16).
Gate: `dsv4_hc_dot_split_gate --defaults`.
