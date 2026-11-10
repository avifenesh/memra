## Independent review (clean context), on 387141ae14 (the fix, the gate and the session-gate check)

The reviewer read the diff against `release/v0.141.0` and the code around it; no builds, no device. Findings as reported, with my disposition after each.

**1. Bug (prefix reuse regression): a DSpark request that ends on a stop string is no longer parked.** The DSpark park gate needs `sess.pos() == s.fed.len()`. After the fix `dspark_burst_post` pushes into `fed` only through the completing token, while the engine still commits the whole round (only EOS and the request budget cut the commit short). With the reuse pool and the prefix cache on, a turn that ends on a stop string inside a round logs `dspark-park REFUSED ... invariant failure` and the session is freed. Before the fix the whole burst was public, so `pos == fed` and the session parked, rewound to its prime boundary. Suggested: let a recurrent session whose rewind target is within `fed` park when `pos > fed`.
- Disposition: confirmed in the code and fixed in 081a5d7101. `dspark_park_shape_ok(pos, fed, recurrent)` lets a recurrent trunk with rows past `fed` into the park arm, where it parks rewound to a prime boundary that `dspark_park_rewind` already requires to lie inside `fed`; every other trunk keeps the strict equality. The same shape followed a stop id on main (a stop id is never in `fed`), so that case parks now too. Unit test `dspark_park_takes_a_recurrent_session_that_stopped_inside_a_round`; device row: the DSpark boot with the prefix cache on (see the PR body).

**2. Risk (scope gap): in pooled cohorts the response still closes late.** `step_spec_pooled` and `step_dspark_pooled` run `*_burst_post` and `finish` only after the whole cohort call returns, so a request that stopped early gets its usage and `[DONE]` when the slowest member's burst ends. EOS behaves the same way today. Counts are correct there. The gate sends one request at a time, so it never runs a pooled cohort.
- Disposition: agreed, not fixed here. The counts are fixed on the pooled paths (same helper, same clamp) and the stopped member's burst ends at its round, but its `Done` waits for the cohort call to return, as an EOS-ended member's does on main. Finishing a member inside the cohort call is a change to the cohort driver, not to the stop check; filed as #1209.

**3. Nit: the `[spec-stop]` receipt is missing on some paths** (the GLM5 hook, gemma, and the burst-cadence arms of DSpark and GLM5 with `MEMRA_SPEC_FIRST_TOKEN_EAGER=0`), so a gate boot with the eager door off would fail every stop row on its receipt count.
- Disposition: fixed in 081a5d7101: every arm that latches a stop string prints the receipt (`spec_stop_receipt`, `spec_stop_receipt_burst`).

**4. Nit: two comments are now stale** (the DSpark and GLM5 eager-door comments said the loop never reads the hook and both arms produce the same `burst` vector).
- Disposition: fixed in 081a5d7101.

**5. Nit (gate): the `turn2` check does not prove the parked session was restored.** `want_cached` passes on any prefix hit.
- Disposition: partly taken. On this model the MTP session pool does not park a stop-ended (or any) turn whose checkpoint is off the 32-token prime grid (`spec-park SKIPPED (grid law)` in the boot logs), and the second turn restores from the prefix cache at the 1248-token boundary, the same path a non-stop turn takes; its text, finish and count equal plain decode's. The gate now records every row's park and reuse lines and fails on a `-park REFUSED` line (the shape of finding 1). The engine invariant itself (committed list equal to the cache rows after a burst the hook stopped) is session-gate's new check.

**6. Nit (gate): the budget row can be skipped silently** when only one stop word was picked.
- Disposition: fixed in 081a5d7101: the budget count comes from the stop string the row uses, and the judge fails a boot that lacks any row kind.

**7. Risk, minor (gate): the close timing includes the stop string's own tokens** (the surface holds back text that could start a stop string), so a correct multi-token span on GB10 could approach the limit.
- Disposition: taken before the first speculative judge: a stop-ended row's limit adds twice the time plain decode's own row took from its last visible chunk to its end, which is that hold-back in plain decode.

Checked and reported correct by the reviewer: the emission order against `advance_sample_emit` (stop id before decode, EOS counted with no text, count then send then check, a stop on the last budget token is a stop); stop strings across slices and bursts; send failure (no latch, the new debug asserts cannot fire); `token_events == public_len` and `tokens_emitted == generated.len()`; the MTP engine on a false verdict (the admission-yield exit, the committed / cache rows / pending invariants unchanged); DSpark and GLM5 burst state at the verdict (the state a reached target leaves, the pooled `live` honours it, GLM5's anchor flush); no missed hook callers; gemma's count-only clamp; the source-shape tests; the gate's log regexes; that the gate fails the unfixed binary on counts, receipts and close; and session-gate's `since` offset.
