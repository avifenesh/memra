# lane/b200-mla-depth-20260902: depth axis for the B200 MLA decode arm gate

Follow-on to lane/b200-mla-decode-20260902 (PR #83, merged 2d866205a), which shipped
`MEMRA_B200_MLA_DECODE_ARM` (default OFF) with a t_q-keyed split table and a gate that
measured one pool depth (n_slots=2048 / pool_rows=32768). This lane adds a kv-depth sweep to
the gate as a cheap kernel-level check. It does NOT add a depth key to the policy; see
section 2 for why that was designed, built, and then reverted inside the lane. Worktree
`/home/avifenesh/projects/wt-b200-mla-depth`, branch `lane/b200-mla-depth-20260902` off
`origin/main` 4e4262ce5. No GPU in this worktree; the 2x B200 pair belongs to the spawning
session, which runs the gate.

## 1. The depth receipt, and the correction that followed it (coordinator, 2026-09-02)

First message. Same 2x B200 SXM pair, END TO END: int11 830e238a0, real artifact, PP-2
resident, DFlash2 spec route + EAGER, vendor sampling, 256 tokens, one 256,756-token prompt
per arm, every other door identical (MATVEC_ARM, HC_FUSED_PRE, Q8_FUSE, KDA_FUSED_PROJ, W8,
PRIME_V2 arms 1+2):

| arm                          | TTFT     | decode after first token       |
|------------------------------|----------|--------------------------------|
| MEMRA_B200_MLA_DECODE_ARM=1  | 152.95 s | 15.58 tok/s (steady p50 19.5)  |
| arm OFF                      | 160.64 s | 32.08 tok/s                    |

Read at face value: the t-keyed arm halves decode at 256k context while winning at the
66-token prompt (part of the +5.6% five-door gain), and the kernel gate had only measured
pool_rows=32768. Which split would scale badly was unknown: absorb_q split=4 (t1, t4),
decompress_v split=4 (t1), attn_gathered split=2 (t1).

Correction, before this lane built anything against the box. The third bisect arm (ALL B200
doors off, spec route, same prompt) decoded at 15.42 tok/s with TTFT 153.1 s, the same slow
mode as the all-doors arm, while the all-doors-minus-MLA arm hit 32.08. The 512k and 1M
rungs on one posture showed the same two modes across reps (28.3 vs 10.7; 27.0 vs 6.2). So
the depth slowdown is BIMODAL on the spec route, independent of doors, and the MLA-arm
attribution is NOT established. Two plain-route arms (best vs off, no spec) are running; if
they are flat, the depth-key task is cancelled and the bimodality goes to the spec route
(drafter / KV at depth) instead. Raw receipts: darklanes
`research/glm5-b200-20260902/floor/raw/depthab/` (rsync pending at lane open).

What the kernels say about depth, so the gate's result can be read against it: `absorb_q`
and `decompress_v` take no depth input at all (per-token projections against 32 MB weight
matrices); `attn_gathered` walks the DSA top-k set, whose width the selector fixes, not the
pool. In isolation all three are depth-flat. The one way depth reaches them is the L2 state
they start from, and in serving that state is cold at every depth (the MoE expert stream
between two MLA layers evicts everything), which a warm-L2 microbench flatters for every arm
equally. The gate's scrub (section 3) removes that flattery.

## 2. The depth key: designed, built, reverted in-lane

Before the correction arrived this lane had built the full depth key: per-kernel kv-length
ceilings (`MLA_B200_*_KV_CEIL`, 32768 rows), a pure `mla_b200_arm_select(kernel, t_q, t_kv)`
composing the t_q table with the ceiling, a `t_kv` argument (`slot + t` at the forward's MLA
core) threaded through `mla_absorb_q` / `mla_decompress_v` / `mla_attn_gathered` and their
twelve callers (the forward plus four GPU test files), and the FLAGS/KERNELS rows for it.
All of it was reverted the moment the correction landed, on the flags law: a ceiling is a
behavior change, and the receipt that would have justified it attributes the loss to the
spec route, not the arm. Nothing of it is in this branch. If the plain-route arms DO show
the arm losing at depth, the design above is a one-commit rebuild (the shape is recorded
here so it is not re-derived), and the gate's per-(t, kv) table from the box is the number
that sets each ceiling.

The serving policy therefore stays exactly PR #83's: `mla_b200_arm_table_split(kernel, t_q)`,
t_q-keyed, no depth term. That is deliberate for the gate too: with no ceiling in the
policy, a split that genuinely scales badly at 256k FAILS the gate as a REGRESSION rather
than being hidden behind a ceiling, which is the cheap kernel-level check the correction
asked to keep.

## 3. Gate: kv sweep, cold-L2 scrub, regression on the t_q table at every (t, kv)

`mla-decode-arm-gate [device] [rounds=5] [max_kv=262144] [scrub=1]` now walks kv in
{2048, 32768, 131072, 262144} x t_q in {1,2,4,8} x the three kernels x split in {1,2,4,8}
(split=1 is the shipped launcher, never the twin at split=1). The pool is `kv` rows of
kv_rank floats (512 MB at 256k); the gathered set stays min(2048, kv) rows drawn at random
from it, as the DSA selector would. Before EVERY timed launch, at every depth, a scrub
streams 262144 x 512 floats (512 MB read + 512 MB written, above any current L2) through
the SMs via `Engine::rms_norm`, so each arm is timed from the cold-L2 state serving leaves
these kernels in; `scrub=0` reproduces PR #83's warm-L2 methodology for comparison.
Interleaved rounds, preallocated outputs, launch-to-completion brackets, as before.

Output: per (kv, t_q) the bit-identity verdict for split {2,4,8} of each kernel and the
per-split means; then a per-(t, kv) winner table (`best` = fastest split, `sel` = the
serving table's cell, `arm/shipped` = its ratio); one `REGRESSION ...` line per cell whose
table arm is slower than shipped by more than 5% (`MLA_B200_ARM_REGRESSION_MARGIN`); `note`
lines where the fastest split differs from the table. Exit 0 with `PASS` only when every
twin is bit-identical at every (t, kv) and no table cell regresses at any depth; `FAIL` +
exit 1 otherwise.

Box invocation:

```
MEMRA_CUDA_ARCH=100a cargo build -p memra-engine --bin mla-decode-arm-gate
flock <box lock> -c "NVIDIA_TF32_OVERRIDE=0 ./target/debug/mla-decode-arm-gate 0"
```

(`... 0 5 262144 0` for the warm-L2 twin; `... 0 5 32768` to stop at 32k on a small card.)
Memory at 256k: pool 512 MB + scrub pair 1 GB.

How to read it against section 1: (a) a split loses at depth for a kernel: the gate fails,
names the cell, and that is the receipt for a depth key (section 2's design); (b) every
split is depth-flat cold-L2: the kernels are cleared, the bimodality is the spec route's
(drafter / KV at depth), and the arm's t_q table stands; (c) the t_q=2 / t_q=8 cells (no
B200 number yet) show a winner: edit the t_q table, citing the run.

## 4. Build / lint receipts (this worktree, no GPU)

Recorded in the PR body: `cargo fmt --all -- --check`, `cargo clippy -p memra-engine
--all-targets -- -D warnings` at `MEMRA_CUDA_ARCH=120a` and `=100a`, `tools/check-flags.sh`,
`tools/docs-registry-census.sh`.

## 5. Open items

1. Run the gate on the pair (section 3); read it per the three outcomes above.
2. The plain-route arms (coordinator): flat means the depth-key task is cancelled for good
   and the bimodality is a spec-route lane; a real arm loss at depth means rebuild section
   2's design with the gate's numbers.
3. The end-to-end serving A/B from PR #83's open items (sampled twin with spec-engagement
   receipt, TTFT/TPOT/ITL) still gates any default flip.
