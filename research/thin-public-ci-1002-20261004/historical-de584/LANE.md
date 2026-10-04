# Thin public CPU CI

Question: can owner changes run affected compile, lint and source-bound merge validation while external, daily and publication events retain complete CPU validation?

Budget: issue #1002, $0, local CPU 120 minutes, no rentals. This CI tooling change needs no GPU work.

The router and planner execute from an immutable GitHub event-base checkout, separate from the candidate checkout. Isolated Python excludes candidate cwd, PYTHONPATH and candidate sitecustomize. The entry checks base code/data bytes and Git owner-executable modes before imports and reads an explicit candidate repo and source head. Missing base entry selects a workflow-literal FULL bootstrap without candidate imports. Author and stable repository lineage also guard the workflow mode directly.

The complete CPU workflow's booleans, workspace and available-contract defaults are literals. Candidate full-plan/change-class outputs cannot suppress full jobs. Full completion and conclusive full admission use isolated stdlib checks of actual job results and source head. Thin contracts and results use the isolated base entry. Routing, policy or helper edits expand FULL until the reviewed base is updated. No pull_request_target, secret inheritance or privilege increase is introduced.

At `73bd47a54d84df96dfadc1f2781f172bb0068d70`, the existing real fourteen-package Git/event witness passed in 494.577 seconds. It executes hostile router/full-plan code against the published vulnerable caller, then proves base-owned/literal full selection despite hostile producer, PYTHONPATH/sitecustomize/json inputs. It includes missing-base bootstrap, stale trusted bytes, explicit candidate binding, restored owner docs routing and all standing source/include/native, guard omission and caller-once controls. SFT executes once. Existing 25 policy tests and 18 classifier arms passed with zero skips. Inventory, 10 workflow mappings, action pins and clean immutable source checks passed. See [scoped proof](SCOPED-PROOF.json), [source pins](SOURCE-PINS.json) and [raw output](raw/integration.log).

All 74 legacy non-selection full command bodies remain exact. Mandatory Python callers, native compile/ABI/SASS and dense-control admission stay intact. Checker code and consumed data retain exact byte/mode/membership binding. Native math, emitted programs, compiler/build defaults, model artifacts/defaults, tolerances and GPU shadow behavior are unchanged.

The whole local proof took 504.659 seconds wall and 473.161 CPU seconds, with 2.4G peak memory and zero swap. No overall CI speed comparison is claimed. The prior strict reader cost has its own source-bound result and is not a timing at the repaired source.

The prior full hosted run at `2ab8eec8ddd5b27ecaaf22f55ddf728151be560c` passed, then security review held its PR-owned router boundary. [That proof](historical-2ab8/SCOPED-PROOF.json) retains its original source and timings. Failed heredoc and empty-diff fixture repair cohorts are recorded separately. The original 241 cohort had 240 passes and one concurrency-expectation failure; corrected 25 passed at its recorded source. No uniform 241 or native compiler replay at this source is claimed.

Current hosted full validation and current security review remain required before merge. PR-modified workflow YAML and code under test still require source review. The visible conclusive check is not installed as a GitHub-required rule. No native, model, runtime or serving qualification is granted.

Reproduction: `python3 tools/public_ci.py check-inventory`; `python3 tools/test_public_ci_integration.py -v`.

Publicity: skipped: maintenance release.
