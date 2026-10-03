# Mandatory guards with no requested edges

The coverage selector returned no-change before inspecting the catalog when
`required_edges=[]`. This omitted explicitly requested mandatory guards and their
control dependencies, including stale or unavailable sources. An independently
executed requested guard exited 42 while its plan still selected nothing. This
did not certify a pass: no-change result admission correctly refused execution.

The selector now retains no-change only when there are neither requested edges nor
mandatory tests. Explicit mandatory requests proceed through the normal catalog,
scope, source and control checks. Iterable catalogs are retained while testing for
the request. No-request optional catalogs remain no-change and non-passing.

The executable proof binds baseline `79ac1f8cdbbd0dc4056c1f65e4e382758da1466b`
and clean code `b140ae856e4c1e8d3a32f13ad7a5fc47fec517ed`. A real positive guard
exited 0 and printed its assertion witness; the requested failure/red program
exited 42. Five actual old/new CLI cases cover mandatory bundles, the requested
failure, stale source, missing control and optional-only no-change. A genuine
zero-requested-edge receipt checks the selected guard and red assertions and passes.
A failed requested guard and no-change admission refuse. `PROOF.json` retains the
programs, source hashes, exits/stdout, manifests, selection output and admission
verdicts. Raw diagnostics remain privately retained; public reasons and hashes
preserve their identity.

All 121 composed planner/coverage controls passed through the existing non-vacuity wrapper,
as did the 16 registered tool-choice receipt controls and 14-package registry.
Formatting and whitespace checks passed. The typed-context comparator remains
unchanged. The owning CI floor rises to 121; full hosted CI gates merge.

This is CPU selection/admission evidence. Source pins, controls, native identities,
every selected assertion, typed execution counts and zero skips remain required.
No native program, model/default, compiler default, tolerance or required native
gate changed. Qualification stays false and GPU selection remains in shadow mode.

Verdict: mandatory guards and dependencies remain requested with zero affected
edges; optional no-request catalogs remain non-passing no-change.

Before the first push, this lane was composed on frozen compiled-input parent
`b160a7312a062ded5f67ea5784672aacad9d7aed` in PR #934. Both documentation/index
entries and all typed and graph controls remain present. The actual combined suite
ran 121 tests with floor 121, plus the 16 tool-choice receiver controls and
14-package registry. The executable proof was repeated on the clean composed
source `cc217cfb472142e880fb0eb002caa1c257d75c67`. The coverage program and its
tests match code `b140ae856e`; no native inputs or execution changed.
