# WP-A day 66: the fanout split's scope (revuto on #731), a registered fix

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. It goes ahead of B1's code (DAY59 section 7), which waits in a
stash until this lands.

## 1. Pre-registration (committed before any code)

**The defect** (DAY59 section 8). The day-59 split's thread-local collects every `prefix_snapshot` and
`prefix_restore_at` call on the owner thread, but only the fanout takes it. A leftover from another caller since the
previous fanout is therefore added to the next fanout's snapshot line. DAY59's twin carried no leftover: its
receipts' counts are exactly the clean program's. Other shapes can carry one (a retire capture on the tick, a park
snapshot, a hit restore between two fanouts).

**The fix.** `prefix_copy_scoped(f)` discards whatever the thread-local holds, runs `f`, and returns `f`'s result
with the split `f` alone produced. The fanout's snapshot runs inside it. The restores' take already follows the
scoped snapshot with only the sibling restores between them, so it keeps its place. The wrong comment is corrected.
Nothing else changes: the calls, their order and the lines' wording stay as they are, and the split still decides
nothing.

**The cells (CPU).**

- `day66_a_stray_copy_before_a_fanout_never_reaches_its_split`: a stray timed call of each kind (0 to 3) before a
  scoped call must not appear in the scoped split, whose counts are exactly the scoped call's own. A second scoped call
  right after the first reads only its own calls.
- **The red arm.** A scratch patch in which `prefix_copy_scoped` does not discard: the cell must fail (the stray's
  count shows). The patch is grep-checked in its test binary (the marker is printed, so it cannot be compiled out) and
  the binary is found through cargo's JSON `executable` field.
- **The census.** `day59_the_fanout_copy_split_is_log_only` pins the fanout's snapshot inside `prefix_copy_scoped` and
  counts four `prefix_copy_split_take()`: the fn, the two in the scoped helper, and the restores' take.
- The server lib, clippy `-D warnings`, fmt.

**What each card decides.** No card: the change is log-only bookkeeping. DAY59's selection stands on its receipts
(section 8), so the twin is not rerun.

**Budget.** 0.05 agent-day.
