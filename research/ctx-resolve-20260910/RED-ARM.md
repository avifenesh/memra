# Red arm: eight context tests against five ways the window can misresolve (2026-09-10)

A check that has never failed is not a check. Every test added with the `MEMRA_CTX` resolution fix
was re-run against an injected reintroduction of the behaviour it forbids, one injection at a time,
on the dev box, at the tree in this PR. Full transcript: `red-arm-sweep.log`. Script:
`red-arm.sh` (it injects, runs, restores, and re-runs green at the end).

## The three cases the gate must tell apart

Silent resolution is this codebase's recurring failure shape, so the resolver produces three
DISTINGUISHABLE outcomes and the tests assert they cannot blur into each other:

| the checkpoint's `config.json` | outcome | why not a default |
|---|---|---|
| declares a window the engine can carry (`max_position_embeddings: 1048576`) | SERVED at that window | the checkpoint is the authority; an explicit `MEMRA_CTX` overrides it |
| declares NOTHING (field absent, or `null`) | REFUSES, message says `undeclared` | this is real ambiguity: no number is derivable, so the operator must pin one |
| declares something the engine CANNOT honour | REFUSES, message says `ceiling` and names the declaration | narrowing to what fits, silently, IS the defect |

The third case has two flavours, both refusing: a present but unusable declaration (string, float,
negative, `0`, array) refuses naming the value, and a declaration above `ENGINE_MAX_CTX`
(`u32::MAX`, the width the mainline `ModelConfig.context_length` can carry) refuses rather than
clamping. `the_three_cases_produce_three_distinguishable_outcomes` asserts the three refusal texts
are pairwise different and that each names its own case, so a gate cannot report one situation as
another.

`ENGINE_MAX_CTX` is a REPRESENTATION bound, stated as such in the code: it makes no claim about
VRAM, KV budget, or what a given box can prime.

## The five injections and what each turned red

```
=== INJECTION 1: the pre-fix literal, unset resolves to 8192
  an_unset_ctx_resolves_to_the_models_own_declared_context_never_a_constant     FAILED
  an_unbounded_request_on_an_unset_ctx_is_capped_by_the_checkpoint_not_by_8192  FAILED
  the_dsv4_route_serves_the_window_the_checkpoint_declares                      FAILED
  a_checkpoint_that_declares_nothing_refuses_as_undeclared                      FAILED
  a_declaration_beyond_the_engine_ceiling_refuses_instead_of_narrowing          FAILED
  the_three_cases_produce_three_distinguishable_outcomes                        FAILED

=== INJECTION 2: an undeclared window silently acquires a default
  a_checkpoint_that_declares_nothing_refuses_as_undeclared                      FAILED
  the_three_cases_produce_three_distinguishable_outcomes                        FAILED

=== INJECTION 3: a beyond-ceiling declaration is clamped instead of refused
  a_declaration_beyond_the_engine_ceiling_refuses_instead_of_narrowing          FAILED
  the_three_cases_produce_three_distinguishable_outcomes                        FAILED

=== INJECTION 4: a present but unusable declaration is treated as absent
  a_present_but_unusable_declaration_refuses_and_names_the_value                FAILED
  the_config_file_itself_is_the_source_and_a_broken_one_refuses                 FAILED

=== INJECTION 5: the config read is swallowed
  the_config_file_itself_is_the_source_and_a_broken_one_refuses                 FAILED

=== RESTORED, green re-run
  3 passed (worker `ctx` filter, one of which is a pre-existing test the filter matches)
  6 passed (dsv4 `declared_context_tests`)
```

Injection 3 is the one that matters most for the sibling lane's finding that six of nine merged
default-ON DSV4F doors are inert because unset resolves to `Ok(false)`: clamping a beyond-ceiling
declaration to the ceiling is exactly that shape, a resolution that succeeds while quietly meaning
something narrower than the config says. The test asserts `!= Ok(ENGINE_MAX_CTX)` for that input,
so "make it fit" cannot pass as a fix.

## Non-vacuity

Input sets: the four declared windows in play (1048576 / 262144 / 32768 / 4096), the ceiling and
both sides of it (`ENGINE_MAX_CTX`, `+1`), five unusable spellings of `MEMRA_CTX`
(`""`, `"abc"`, `"-1"`, `"1048576tokens"`, `"0"`), five malformed `max_position_embeddings`
shapes (string, negative, zero, float, array), two absent shapes (missing key, `null`), and a real
directory with a missing, a malformed, an unusable-declaration and a valid `config.json`.

Caller and trigger: `cargo test -p memra-server --lib`, which the `server-tests` CI job runs on
every push to this repository. No flag, fixture or hardware is required to reach any of them.

Ran on the shared dev box (vast, ssh5:38312), CPU only, no GPU lock taken, `pgrep -af cargo`
checked clean first. Nothing ran on the rig.
