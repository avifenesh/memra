# WP-C day 81 (2026-09-26): OWED C11, I19, the door's prefetch two experts ahead, after the launch, before any code

Lead: "continue C11: register the next cut to the door's prefetch path (the remaining gen-only gap, 2 to 10 ms over 32
tokens by host; DAY72 placed it on the prefetch path's CPU work delaying the next launch)". I17 and I18 cut the path's
CPU time and read `flat`; the door still `loses` gen-only. Tree at start: `479cf7eda`.

## 0. What the readings leave, and why the next cut moves the work instead of shrinking it

`DAY72.md` section 3: the door's GPU work equals REF's and the GPU waits between kernels, because before each
expert's launch the loop first prefetches the next expert (`hybrid_forward.rs`), and the door's prefetch costs about
1.7 us more CPU than REF's (the owner's lease). Three cuts of that CPU time (I15, I17, I18) together took off less than
the wall resolves. I16 moved the whole prefetch after the launch and regressed: the next expert's copy lost its lead
and was exposed (`DAY75.md` section 3).

What neither tried: keep the copy's lead and still take the prefetch off the launch path, by prefetching one expert
further ahead. Today expert `j+1` is prefetched just before expert `j` launches (its copy leads by `j`'s compute). With
the prefetch two ahead and after the launch, expert `j+2` is prefetched right after expert `j`'s kernels are launched:
the CPU does the lease while the GPU runs `j`, and `j+2`'s copy is queued behind `j+1`'s, with `j+1`'s compute (and the
rest of `j`'s) as its lead, no less than today's. Only the first expert of each layer's list keeps today's order
(expert 1 prefetched before expert 0 launches), so one prefetch per layer per token stays before a launch where today
every one does.

## 1. Pre-registration: I19

**The change (door only; `hybrid_forward.rs`, the cached expert loop).** Under the door (the forward's prefetch
condition `e.expert_bank_prefetch()`): at `j = 0` the loop prefetches expert 1 before expert 0's kernels, as today; for
every `j` with an expert `j+2`, it prefetches expert `j+2` right after expert `j`'s accumulate, with `keep` naming
experts `j`'s and `j+1`'s blocks (so the reservation evicts neither the expert just launched nor the next one). The
legacy prefetch (REF, `MEMRA_MOE_PREFETCH=1` without the door) keeps today's order. The numeric program is unchanged
(a prefetch writes a block's bytes into a slot; dispatch decides which kernel reads which slot), so the tokens must
match every other arm's; the host demand sequence changes with the order (printed per arm, as in DAY75).

**Its risk, stated.** Two groups in flight instead of one: the in-flight bound (`BANKED_INFLIGHT`, 32 leases) holds six
of them; a group that does not fit is not prefetched and that expert is demanded at dispatch, a GPU miss the stage
lines would show.

**CPU gates before any card** (`day81-cpu/`): a source census that under the door the only prefetch before a launch
is expert 1's at `j = 0`, the two-ahead prefetch sits right after the accumulate in both cached branches, and the
legacy branch is unchanged; the `day50` census's count of the prefetch condition; the engine library; clippy and fmt;
a local RTX 5090 check that I19 exits 0 with `MATCH` and the tape of I18.

## 2. Pre-registration: the card cell `i19` (the 285K class, then a 9950X, then the RTX 5090; before its script)

DAY79's cell with I19 in I18's place: binaries `i18=c7294b912` (REF's arm and the door before I19) and `i19` (named in
section 2a); arms REF (`run-gen-i18`, `MEMRA_MOE_PREFETCH=1`), I18, I19, I19C (I19 with `--moe-dispatch-clock`),
order 1 (REF, I18, I19, I19C) x 5, order 2 reversed x 5, then the profiled pair (REF, I19, I19, REF), window rows kept,
reports by hash. Integrity as DAY75's (one host demand sequence within each door arm; I19's printed beside I18's, not
compared). Admissibility, then I19 against I18 (`improves`, `regresses`, `flat`, gen-only primary, the window beside
it), the door (I19, or I18 if I19 `regresses`) against REF. Part B, deciding nothing: I19's `gpu_idle` and
`h2d_exposed` against REF's, the stall this is meant to remove. `regresses` on either class reverts I19 with its
receipt; `flat` or `improves` stays.
