# The cell arms' sources and binaries

Each `ARM.sha` is the `arm.sha` that `tools/build1307.sh` wrote beside the arm's binary: the
sha256 of the engine files the arm was built from, with the `[diag1307]` lines applied.
`tools/arm_sha.py REV` prints the same list from a revision, so the code an arm's cells ran on can
be checked from the repo.

| arm | cells | sources | check | binary sha256 (each cell's `guard.log`) |
|---|---|---|---|---|
| B | sections 2 to 7 | main bbc6f83aea | `arm_sha.py bbc6f83aea` | `2ce138b751219fbd...` on both units |
| T | sections 2 to 7 | this branch's first commit as first pushed (4142526e8a, on bbc6f83aea) | below | `4824e4caf3ce6f3d...` on both units |
| D | section 5 | the same, with the deep dumps | below | `11551e36a0627045...` (unit B) |
| Bc | section 8 | main with landing 24 (63c351a2f4) | `arm_sha.py 63c351a2f4 --lib` | `53ffe02b4223eaf6...` (unit A) |
| Tc | section 8 | this branch at the fold's code commit ("fix(spec): the draft-state delta asks the engine ...") | `arm_sha.py THAT_COMMIT --lib` | `d7f00bff1a91002c...` (unit A) |

- B, T and D hold `spec.rs`, `dflash.rs`, `unified_memory.rs` and `dsv4_gpu.rs`; nothing else
  differed between main and the branch then. Bc and Tc also hold `lib.rs`, which the fold changed.
- **T and D after the rebase.** The branch was rebased onto landing 24 after sections 2 to 7 ran, so
  4142526e8a is not in main's history. Its `spec.rs`, `dflash.rs` and `unified_memory.rs` are the
  same blobs as in the rebased first commit ("fix(spec): on unified memory the pooled DFlash2
  verify width ..."), so `arm_sha.py` on that commit reproduces those three lines of `T.sha` and
  `D.sha` (`--deep` for D). The fourth, `dsv4_gpu.rs`, is bbc6f83aea's file plus this change's doc
  comment; landing 24 removed 88 other lines from it. That route is not loaded in any cell.
- **Bc and Tc are the landing code.** Their sources are main's and the branch's at the time of
  section 8; a later rebase that does not touch the five files keeps the hashes. After section 8
  one comment line of `unified_memory.rs` changed, so the head's own list differs from `Tc.sha` in
  that file's line and nowhere else.
- The first campaign (`unit-a/first/`) built B's and T's sources with an earlier diagnostic patch
  (no `extra_kib` and `col_kib` fields); its binaries are `0c91b738f736aef7...` (B) and
  `4d369df7572d8c85...` (T), and its `arm.sha` files were not kept.
