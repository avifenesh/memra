# Sparse CPU input preflight

Run this before the selected existing consumers in a sparse worktree:

```sh
python3 tools/sparse_input_preflight.py --ref HEAD
python3 tools/sparse_input_preflight.py --ref HEAD --check perf-board
```

`--root` selects another checkout; parent-traversal components (`..`) refuse so
normalization cannot erase an unsafe ancestor. Repeat `--check` to combine `perf-board`,
`support-records`, and `public-boundary-links`. The default selects all three.
Output is JSON. Exit 0 means that the modeled inputs match the pinned commit;
exit 1 reports omissions, unsafe materialization, or an unknown input contract.

The performance-board contract includes the hook and renderer source, board JSON,
generated documents, and mapped model cards. The support-record contract includes
the census, its copy fixture, metadata, source documents, model-pack sources,
required evidence, tracked optional sidecars, and complete tracked copy roots.
Unmodeled additions inside a copied root also refuse. The boundary-link contract
includes its reader sources, every tracked link, and the target closure derived
from Git blobs, including target parents and directory contents.
Its existing assertion enumerates the index, so the index link inventory must
match the pinned tree. Added, removed, changed or unmerged index links refuse
instead of silently narrowing coverage. Expected targets still come only from
the pinned Git tree.

The command compares regular blob bytes and Git's owner executable bit, exact link text,
and directory materialization. It anchors the root and every descendant component
with no-follow directory descriptors. Expected links resolve within the pinned
Git tree before filesystem inspection. Absolute targets, escapes, cycles, missing
Git targets, ambiguous copy types, unreadable inputs, and Git-reader failures
refuse. Changed reader source hashes require a new input audit and updated pins.
Git reads disable replacement objects and lazy blob fetching.

For missing inputs, `suggested_sparse_paths` lists the minimal parent directories
to consider adding with cone-mode `git sparse-checkout add`. Redundant descendants
are removed; missing ancestors do not suggest whole unrelated research trees.
Direct required leaves precede bulk copy contents in bounded diagnostics. Counts
and truncation flags describe omitted output. Root-level files need Git restore
if manually removed; cone additions already include them. The user chooses and
executes materialization. The command never checks out
files, changes permissions, regenerates facts, or suppresses an existing check.
It establishes input materialization only. Run every original consumer and gate
after it passes. It grants no model, runtime, serving, or native qualification.
Inputs can change after the command returns; this is a diagnostic, not an execution
sandbox. It covers these three contracts, not every command in the push hook.

Python 3.11 or newer, Git, and Linux descriptor/procfs support are required.
Diagnostics retain a total count and at most 200 problems and sparse suggestions.

## Integration coverage

```sh
python3 tools/test_sparse_input_preflight_integration.py
```

One owned Git repository uses exact original source, documents, receipt bytes,
and Git modes. It drives the original board and support consumers and the existing
boundary link assertion. Real sparse omissions must cause the original failures
and specific preflight diagnostics. Git then materializes the original inputs;
the consumers pass without regenerated facts or changed commits. The harness starts
with a tools-only cone checkout, requires board and receipt diagnostics to survive
the 200-entry bound, and feeds the actual suggested directories into Git.

The same repository exercises regular-byte and mode drift, valid owner-executable
0700/0744 modes, owner-execute removal refusal, FIFO/type refusal,
unreadability, physical parent links, pinned link targets and parent links,
cycles, finite repeated directory links, repeated/trailing separators, the real
Linux 40/41-link traversal boundary, escapes, absolute and untracked targets,
required versus optional sidecars,
unmodeled copied inputs, unknown readers, and Git failures. Coherent linter mutants
must fail the relevant witness assertion. Direct CLI checks cover clean success,
missing-board failure, and unknown-contract refusal. Fixture errors do not count
as mutant refusals. Scratch is retired before success is reported.

The CLI has no hook, planner, policy, or CI integration in this change. Its scoped
local check is the integration command above plus Python syntax and whitespace
checks. Existing CI and qualification gates retain their current selection.
