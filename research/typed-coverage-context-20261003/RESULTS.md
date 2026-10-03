# Typed coverage contexts

Python equality treated `request.n=1` and `request.n=true` as the same coverage
context. An independently executed CPU fixture admits the integer and rejects the
boolean. The original coverage selector nevertheless returned `scoped`, and result
admission certified `passed` with changed execution or per-result contexts.
An omitted key also matched an explicitly required null scope value.

The repaired comparison preserves JSON value types recursively. Scope keys must
exist. Boolean, integer and float values cannot substitute for one another; nested
arrays and objects receive the same check. Object key order remains irrelevant.
Finite floating-point values preserve their exact representation, including the
signed-zero distinction in the existing contract digest. Unknown non-JSON data
cannot establish matching context.

The executable proof binds baseline `5c8abf9e763c0cc9ec93d324089d3d9aaa80fb5d`
and clean candidate code `65ffcd3eab363833cb2be93961c71b1588c62ef8`. It runs the
actual original and candidate CLI entry points, using one hash-bound intake fixture.
Changed scope, missing-null-key and changed-result controls returned success before
the repair; the repaired selector expands or result admission refuses. The changed
execution-context API control similarly passed before and refuses after. A matching
typed context still passes. `PROOF.json` retains fixture code, source/helper hashes,
each manifest, CLI output/exit codes and API verdicts. Raw diagnostics remain with
the execution record; the public file includes their hashes and final reasons.

All 109 CPU planner/coverage controls passed through the non-vacuity wrapper: the
77 planner tests and 32 coverage tests. The 14-package registry, formatting and
whitespace checks passed. The existing source/control/regression/native-identity,
edge-assertion, typed-counter and zero-skip checks remain in place. The owning CI
floor rises to 109; full hosted CI still gates merge.

This is synthetic CPU assertion-coverage evidence. No model, native program,
compiler default, artifact, qualification tolerance or required native gate changed.
Returned qualification remains false and GPU selection stays in shadow mode.

Verdict: four mismatched or missing context controls refuse coverage after the
repair, while the matching typed control remains admitted.

After composing with tool-choice merge `45b6cba68e81e810840e4f38a95d0518d9dff894`,
the four program/doc/floor files still match the tested code commit. All 109
planner/coverage controls and the newly registered 16 tool-choice receiver controls
passed. The stricter current global validator admitted both preserved tool-choice receipt
bundles (`native-v3` and `native-v4`), each with 23 independently asserted edges
and mandatory controls, keeping qualification false. `COMPOSED-REPLAY.json` binds
the replay source and receipt hashes. This replay used CPU only and did not repeat
native execution.
