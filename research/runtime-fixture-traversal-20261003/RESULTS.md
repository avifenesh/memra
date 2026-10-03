# Runtime fixture traversal

The input planner omitted a compiled consumer when a runtime fixture path contained
parent traversal. A Rust reader returned `BEFORE`, then `AFTER` after changing only
`research/expected.md`. Its literal was `research/fixtures/../expected.md`. The
original planner returned a scoped plan with no packages and `server=false`.

The same omission occurred with `research/./alias/../expected.md`, where `alias`
pointed to `actual/nested`. The actual reader consumed `research/actual/expected.md`,
which differs from the lexical target. The repaired planner selects the server for
the proven literal and expands all checks for the symlink traversal.

Both readers were compiled with Rust 1.97.1 and executed before and after the fixture
edit. Executables stayed outside the fixture checkouts. `PROOF.json` retains their
source, compiler output, executable hashes, immutable fixture revisions, baseline
and candidate plans, planner source hashes and source commit IDs. The candidate
code receipt binds `19ab14c9de888b306ff3e71901a286d708b4e2e9`; later changes in this
lane add only this evidence and index entry. Private fixture Git bundles and the
exact execution helper are retained with the lane evidence.

The repair requires a known directory before removing a parent component. It checks
the original spelling, each traversed parent and the canonical target for symlinks.
Missing directories, patterned parents and root escapes expand validation. A path
that reaches the repository root covers its entire subtree. Ordinary literals,
safe format suffixes, deleted inputs and renamed inputs retain their consumers.

The 77 planner tests and 24 unchanged coverage-contract tests passed. All 101 ran
through the existing non-vacuity wrapper. The current source-tree input census,
14-package registry, Rust formatting and whitespace checks passed. This planner
edit selects full hosted CI; the CPU proof does not exempt any hosted component.

This is synthetic CPU input-classification evidence. It changes no native model
program or qualification requirement and grants no model, runtime or release
qualification. GPU selection remains in shadow mode.

Verdict: two executed reader controls reproduced the omission and passed after the
repair; ambiguous physical traversal expands validation.
