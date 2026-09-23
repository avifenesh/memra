# Compiler-private preflight authority repair

Finding `TC541-IDENTITY-01` is reproduced on frozen
`343b3bec0dc145209f7e1d9486f303cf9718134a`: a safe external `TensorSource` implementation could
return an ordinary config/plan tuple and bypass config semantics and source-factor preflight.
The retained exact probe demonstrates both unsupported `hidden_act=relu` and independently
mismatched plan acceptance on that revision. Its assertion deliberately fails there.

The hook now returns compiler-private `BoundProgramRef`, whose private field borrows the complete
`BoundTensorSource` bundle. Only the actual in-crate `BoundRuntimeSource` implementation constructs
it from its own bundle. There is no public constructor, re-export, tuple conversion, or independently
supplied config/plan. External source implementations cannot name the return type to override the
hook, including by forwarding another bound source's authority. The public source trait remains
implementable: external sources use the default None and full ordinary preflight. The same exact
old probe fails with E0053 on the repaired source; the forwarding probe fails with E0603 because
the authority type is private.

The genuine bound path still clones its own sealed config/plan, preserving selected components,
normalized Step factors, and the same opened-source bundle used for its materialization and
composite identity. Independent sources cannot inject a swapped source through that private hook.

Validation:

- Executed frozen-old external probe: both bypasses reproduced, expected failing assertion.
  Identical external source against repair: override rejected at compilation (E0053).
- Separate external forwarding probe: private-type refusal (E0603).
- Three external integration tests: unsupported config rejected, stale/mismatched plan ignored
  in favor of canonical compilation, and source-backed unexpected RoPE factors still rejected.
- Two compile-fail doctests cover tuple injection and cross-source forwarding. CI explicitly runs
  these and the external integration controls.
- `cargo test --locked -p memra-gguf -p memra-cli`: 358 GGUF passed, 2 ignored; 13 CLI,
  1 Step integration, 7 inspector, 3 external preflight, and 2 compile-fail doctests passed.
- Fresh separately named `541-preflight-seal-clippy-final-pass.log.gz` records successful GGUF/CLI
  all-targets Clippy with warnings denied. The older composition lint failure and its successful
  repair remain unchanged in the original bank; no log is relabeled.
- Existing actual-source runtime identity/snapshot harness: 18 passed.
- Linux-target engine/server lib/bin/test Clippy passes with warnings denied under `DOCS_RS=1`.
  This is documentation-stub type/source evidence, not CUDA execution.
- Formatting and stage whitespace check passed. All raw attempts, expected failures, intermediate
  fixture initialization error and final passes are preserved losslessly with hashes. Temporary
  reproduction checkout and project directories were removed.

This minimal repair does not activate a root, change placement, import the newer upstream transfer
stack, run native gates, promote support, or merge main. Both original finder and independent
reviewer must rereview this immutable repair before the composed source boundary regains GO.
