# WP-C day 39 (2026-09-23): the target card's demote-class tenant-stall cell

Lane `lane/spill-c-20260919`. Start tip `18c900fc7`; `origin/lane/spill-integ47-20260923` (`160929a92`, A day 30
and C day 38) merged as `a20e4990d`; the packet status fix is `d036db237` (A day 30 landed on integ47, not on
`origin/main` `9717e8d57`; `day39-cpu/ancestry.log`). Every cell here is `executed-not-qualified` development
evidence. No number here is compared across cards or boxes; the RTX 5090 figures of days 35, 37 and 38 are context
only and never a denominator. No recommendation.

## 0. The binaries, recorded before any build

Each binary is built on the target card's box from the SHA below, in this lane's own box worktree, by
`day39-box-build.sh` (a detached checkout, a clean-tree check, `cargo build --release -p memra-server`, a copy of the
binary with its SHA-256 and the tree SHA it came from). The SHAs were fixed by the day-39 brief and are recorded here
before the first build starts.

| label | SHA | what the engine is |
|---|---|---|
| B0 | `091a931c0` | day 37's base tree: day 35's tree, main after #648 plus the integ43 ref, the pre-option-(a) engine |
| B1 | `9717e8d57` | `origin/main` at the day-39 fetch: option (a) (A day 28), option 2a (A day 29) and the bounded latch close (#652), without the D2H spans |
| B2 | `160929a92` | `origin/lane/spill-integ47-20260923` at the day-39 fetch: B1 plus A day 30's D2H spans (`97a9e091f`, `fc637d26a`); `git diff 9717e8d57 160929a92 -- crates` is those two commits' seven files and nothing else, and this lane's tip carries no crate diff against it |
