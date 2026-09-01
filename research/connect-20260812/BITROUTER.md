# BitRouter registry submission

State: **OPEN — Q27 + Q35-A3B**.

- PR: <https://github.com/bitrouter/bitrouter/pull/814>
- Fork branch: `avifenesh/bitrouter:feat/tiyuvta-provider`
- Current fork commit: `6e4729e237562e58bc98009639f9b1c5154106f8`
  (`fix(registry): apply introductory tiyuvta pricing`)
- Initial fork commit: `1f4d1b2 feat(registry): add tiyuvta provider`
- API base: `https://api.tiyuvta.ai/v1`
- Models and current list prices:
  - `qwen/qwen3.6-27b`: `$0.28/M` input, `$0.07/M` cached input, `$2.69/M` output.
  - `qwen/qwen3.6-35b-a3b`: `$0.12/M` input, `$0.03/M` cached input, `$1.03/M` output.
- Cached input is separately metered at 25% of each model's ordinary input rate.
- Manifest copy: `bitrouter/registry/providers/tiyuvta.yaml`

The Q35 amendment followed live evidence at the same base: readiness, the OpenAI catalog, and the
schema-2.4 Provider Monitor feed all list the exact id. Its 21-check public gate finished with zero
failures at `2026-08-12T12:14:18.185401Z`; the raw summary SHA-256 is
`77f3d70ed792503f71777a5e5aa0b4d235ab927f3f5fa4a13643756b8a3aa2de`.

## Validation receipts

- `cargo test -p dist-helper`: 33 passed.
- `cargo run -p dist-helper -- registry validate`: valid, 52 canonical models and 51 providers.
- `cargo run -p dist-helper -- registry build`: passed.
- `cargo run -p dist-helper -- check`: passed.
- The current helper has no `registry docs` subcommand; `check` is the available generated-state
  check.

The same four local gates were rerun after the pricing amendment. At the final live recheck, PR #814
was open and mergeable at exact head `6e4729e23756`, with no check rollup attached to that fork
head. GitHub had marked the prior latest-head CI, PR, package, and release workflow runs
`action_required` with no jobs, so a maintainer must approve fork workflows before they run. On the
prior Q27 head, formatting,
clippy on all three operating systems, dist, docs,
doctests, feature isolation, MSRV, repository hygiene, macOS tests, and Windows tests had passed;
Ubuntu tests were still running. The `validate title` job failed inside
`ytanikin/pr-conventional-commits` while trying to write a label with `Resource not accessible by
integration`; the title itself is conventional (`feat(registry): add tiyuvta provider`). This is
an upstream fork-token workflow failure, not a manifest validation failure.

The only push was to the BitRouter fork branch required for the upstream PR. No memra origin was
pushed.

## Pair and pricing amendment receipt

- Closed: exact public id and Ontario/Canada inference location confirmed.
- Closed: authenticated 21-check protocol/accounting gate passed from the public origin.
- Closed: manifest amended, validated, built, checked, and pushed only to the BitRouter fork.
- Closed: the fork manifest now matches the public input/cached/output price triples and both
  `is_ready=true` records.
