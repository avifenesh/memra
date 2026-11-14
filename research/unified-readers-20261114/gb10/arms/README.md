# The cell arms' sources and binaries

Each `ARM.sha` is the `arm.sha` that `tools/build1307.sh` wrote beside the arm's binary: the
sha256 of the engine files the arm was built from, with the `[diag1307]` lines applied.
`tools/arm_sha.py REV` prints the same list from a revision, so the commit an arm's cells ran on
can be checked from the repo.

| arm | sources | check | binary sha256 (each cell's `guard.log`) |
|---|---|---|---|
| B | main bbc6f83aea | `arm_sha.py bbc6f83aea` | `2ce138b751219fbd...` on both units |
| T | this branch's 4142526e8a | `arm_sha.py 4142526e8a` | `4824e4caf3ce6f3d...` on both units |
| D | 4142526e8a with the deep dumps | `arm_sha.py 4142526e8a --deep` | `11551e36a0627045...` (unit B) |

B, T and D hold `spec.rs`, `dflash.rs`, `unified_memory.rs` and `dsv4_gpu.rs`; nothing else differed
between main and the branch at 4142526e8a.

The first campaign (`unit-a/first/`) built the same sources with an earlier diagnostic patch (no
`extra_kib` and `col_kib` fields); its binaries are `0c91b738f736aef7...` (B) and
`4d369df7572d8c85...` (T), and its `arm.sha` files were not kept.
