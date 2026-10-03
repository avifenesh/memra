# Expert-tier self-test CI admission

Expert-tier self-tests refuse optimized Python and CI executes real assertion and recipe controls; ordinary plan generation and native obligations remain unchanged.

The original builder printed PASS under optimized Python after an incorrect
`pruned_experts` result was planted. Normal Python rejected it. That historical
command retains its `9efd4579` context; the builder bytes match the `a409e51b`
base. The self-test now refuses optimization before creating its fixture, through
CLI flags, environment configuration and direct imports. Ordinary plan generation
still works under `-O` and `-OO` and matches normal output byte for byte.

At `3001d3b01c868277612704e660fc5db0b4d9af96`, the actual 13-control suite passed
with zero skips and expected failures. Execution observed all 26 assertion sites
and nine recipe cases. Nine independently mutated source copies produced real
assertion failures: removed optimization guard, empty self-test, missing assertion,
omitted recipe, lowered control floor, admitted skips or expected failures, and
either missing CI caller. All nine unmutated controls passed first. Workflow
census passed for all nine workflow files.

The whole builder module AST matches the base after removing only the new
self-test guard. Existing assertions, recipes, ordinary CLI generation, planner,
support readers, registry and skip census remain unchanged. A builder edit still
selects all nine CI jobs and full native obligations. The CI gates job runs the
actual self-test and strict admission runner unconditionally.

`PROOF.json` binds source and helper hashes, observations, plan witnesses and raw
outputs. `cpu-controls.log` retains passing and failing output, with private source
and fixture prefixes normalized. Both original and public log hashes are recorded.
Publication changes only evidence files and preserves the tested source hashes.

Reproduce the real self-test and admission controls:

```sh
python3 tools/build_expert_tier_plan.py --self-test
python3 tools/run_expert_tier_contract.py
```

This is CPU self-test admission, with no native model, runtime, serving or emitted
native-plan qualification. Native math, compiler/build defaults, model
artifacts/defaults, tolerances and required native-gate coverage remain unchanged.
GPU selection remains shadow-only.

publicity: skipped: maintenance release
