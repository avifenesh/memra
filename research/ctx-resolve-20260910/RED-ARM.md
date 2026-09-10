# Red arm: the five context tests, run against the defect they exist to catch (2026-09-10)

A check that has never failed is not a check. Each test added with the `MEMRA_CTX` resolution fix
was re-run against an injected reintroduction of the behaviour it forbids, on the dev box, at the
same tree as the PR. All five fail red, and each failure prints the pre-fix number.

Green baseline first, same tree, same box:
`cargo test -p memra-server --lib -- ctx` 3 passed, and
`cargo test -p memra-server --lib -- declared_context_tests` 3 passed
(`a_ctx_bounded_request_is_not_gated_because_context_is_its_only_limit` is a pre-existing test the
`ctx` filter also matches, so the five new ones are 2 + 3).

## Injection 1: the pre-fix literal in the resolver

`worker::resolve_ctx`, `None` arm: `Ok(model_ctx)` becomes `Ok(8192)`. Script: `red-arm.sh`.
Log: `red-arm-1-worker-fallback.log`.

```
test worker::tests::an_unbounded_request_on_an_unset_ctx_is_capped_by_the_checkpoint_not_by_8192 ... FAILED
test worker::tests::an_unset_ctx_resolves_to_the_models_own_declared_context_never_a_constant ... FAILED
  left: Ok(8192)     right: Ok(1048576)
  left: 608192       right: 1048576
```

Same injection against the dsv4 module (`red-arm-2-dsv4-fallback.log`):

```
test dsv4_serve::declared_context_tests::the_dsv4_route_serves_the_window_the_checkpoint_declares ... FAILED
  left: Ok(8192)     right: Ok(1048576)
```

## Injection 2: a substituted default for an undeclared field

`declared_context_from_config`: `.unwrap_or(0)` becomes `.unwrap_or(8192)`, i.e. a config with no
usable `max_position_embeddings` silently acquires one.

```
test dsv4_serve::declared_context_tests::an_undeclared_or_unusable_field_reads_as_zero_and_then_refuses ... FAILED
  left: 8192         right: 0
```

## Injection 3: a swallowed config read

`dsv4_declared_context`: the `read_to_string` error becomes `unwrap_or_else(|_| "{}")`, i.e. a
missing `config.json` reads as an empty config instead of refusing.

```
test dsv4_serve::declared_context_tests::the_config_file_itself_is_the_source_and_a_broken_one_refuses ... FAILED
  panicked at 'no config.json: 0'
```

## Coverage, stated honestly

Injection 1 is the defect as it actually shipped. Injections 2 and 3 are the two ways the same
class could return through the new code path (substitute a default for an undeclared field, or
swallow the read that answers the question), and they exist because the two refusal tests are not
sensitive to injection 1 on their own. Every tree was restored to the green state after each run,
verified by diff for the last one.

Non-vacuity: the input sets are the four declared windows we actually serve or plan to
(1048576 / 262144 / 32768 / 4096), the five unusable spellings of the variable, and five malformed
`max_position_embeddings` shapes, plus a real directory with a missing, a malformed and a valid
`config.json`. Caller and trigger: `cargo test -p memra-server --lib` on every CI run of this
repository, no flag or fixture required to reach them.

Ran on the shared dev box (vast, ssh5:38312), CPU only, no GPU lock taken, `pgrep -af cargo`
checked clean first. Nothing ran on the rig.
