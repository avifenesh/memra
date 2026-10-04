# Thin public CPU CI

Question: can owner changes run affected compile, lint and source-bound merge validation while external, daily and publication events retain complete CPU validation?

Budget: issue #1002, $0, local CPU 120 minutes, no rentals. This CI tooling change needs no GPU work.

The router and planner execute from an immutable GitHub event-base checkout, separate from the candidate checkout. Isolated Python excludes candidate cwd, PYTHONPATH and candidate sitecustomize. The entry checks base code/data bytes and Git owner-executable modes before imports and reads an explicit candidate repo and source head. Missing base entry selects a workflow-literal FULL bootstrap without candidate imports. Author and stable repository lineage also guard the workflow mode directly.

The complete CPU workflow's booleans, workspace and available-contract defaults are literals. Candidate full-plan/change-class outputs cannot suppress full jobs. Full completion and conclusive full admission use isolated stdlib checks of actual job results and source head. Thin contracts and results use the isolated base entry. Routing, policy or helper edits expand FULL until the reviewed base is updated. No pull_request_target, secret inheritance or privilege increase is introduced.

At `76e0ed5615bc94ca8a200052e5808109c30e7f8c`, the existing real fourteen-package Git/event witness passed in 408.819 seconds from a fresh depth-one, no-local clone. It asserts the historical 2ab commit object is absent and no alternates are present. Three frozen original caller files and their manifest are verified by literal manifest SHA, raw file SHA-256 and original Git modes before running the actual old route/full script bodies. A coherent manifest replacement refuses. The witness retains hostile producer/import, missing-base/source/root, restored owner docs, source/include/native, guard and caller-once controls; SFT executes once. See [scoped proof](SCOPED-PROOF.json), [22 source/fixture pins](SOURCE-PINS.json), [fixture manifest](../../tools/fixtures/public_ci_legacy_2ab8/manifest.json) and [raw output](raw/integration.log).

Production trust helpers, workflows and standing test support retain the same 17 pins as the repaired source. Existing 25 policy and 18 classifier arms passed at their recorded earlier source and were not repeated for this fixture-only change. The current inventory, 10 workflow mappings, action pins and clean-source checks passed.

All 74 legacy non-selection full command bodies remain exact. Mandatory Python callers, native compile/ABI/SASS and dense-control admission stay intact. Checker code and consumed data retain exact byte/mode/membership binding. Native math, emitted programs, compiler/build defaults, model artifacts/defaults, tolerances and GPU shadow behavior are unchanged.

The current full witness job took 409.803 seconds wall and 459.183 CPU seconds, with 5.6G peak memory and zero swap. No overall speed comparison is claimed.

The prior source's 494.577s local integration remains [historical](historical-de584/SCOPED-PROOF.json). Hosted de584 gates failed at 110.008s because a historical Git object path was unavailable in its source checkout. The first fresh local fixture setup failed at 85.612s because an identical adapter did not need a commit. Both failed cohorts are retained; exact adapter bytes were preserved and the unnecessary commit removed. The new witness has no historical Git-object dependency. Earlier 241/25/classifier/caller results retain their original sources; no unchanged 25/241/native compiler replay is claimed.

Current hosted full validation and current security review remain required before merge. PR-modified workflow YAML and code under test still require source review. The visible conclusive check is not installed as a GitHub-required rule. No native, model, runtime or serving qualification is granted.

Reproduction: `python3 tools/public_ci.py check-inventory`; `python3 tools/test_public_ci_integration.py -v`.

Publicity: skipped: maintenance release.
