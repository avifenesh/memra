# Sparse CPU input preflight

`tools/sparse_input_preflight.py` checks, before you run the existing consumers in a sparse
worktree, that the inputs three CPU checks read are materialized exactly as the pinned commit has
them. It is for contributors who work in a cone-mode sparse checkout. It is a diagnostic: it changes
no file and grants no model, runtime, serving or native evidence.

```sh
python3 tools/sparse_input_preflight.py --ref HEAD
python3 tools/sparse_input_preflight.py --ref HEAD --check perf-board
```

`--check` repeats to combine `perf-board`, `support-records` and `public-boundary-links`; the default
is all three. `--root` selects another checkout and refuses `..` components. Output is JSON. Exit 0
means the modeled inputs match the pinned commit; exit 1 reports omissions, unsafe materialization or
an unknown input contract.

The three contracts:

- **perf-board**: the hook and renderer source, the board JSON, the generated documents and the
  mapped model cards.
- **support-records**: the census, its copy fixture, metadata, source documents, model-pack sources,
  required evidence, tracked optional sidecars and the complete tracked copy roots. An unmodeled
  addition inside a copied root refuses.
- **public-boundary-links**: the reader sources, every tracked link and the target closure derived
  from Git blobs. The index link inventory must match the pinned tree.

It compares regular blob bytes, the owner executable bit, exact link text and directory
materialization, walking every path component with no-follow directory descriptors. Absolute
targets, escapes, cycles, missing Git targets, ambiguous types, unreadable inputs and Git-reader
failures refuse. A changed reader source hash needs a new input audit and updated pins. Git reads
disable replacement objects and lazy blob fetching.

For missing inputs, `suggested_sparse_paths` lists the minimal parent directories to add with
`git sparse-checkout add`. Diagnostics keep a total count and at most 200 problems and suggestions.
You choose and run the materialization; then run every original consumer and gate, since inputs can
change after the command returns. It covers these three contracts, not every command in the push
hook. It needs Python 3.11 or newer, Git, and Linux descriptor and procfs support.

The CLI is not wired into the hook, the planner or CI. The thin public CI imports its link-resolution
helpers to pin merge-guard data inputs. Its integration test drives the real consumers through real
sparse omissions, materialization, mode and type drift, link edge cases and coherent mutants:

```sh
python3 tools/test_sparse_input_preflight_integration.py
```

The earlier text of this page is kept in
[docs/history/TESTING.md](../docs/history/TESTING.md#sparse-cpu-input-preflight).
