# Independent review (memra#1209)

One clean-context review of ede1b8119f..dcaa2a1524 (Opus, read-only), before the device runs on
the final head. Verdict as given: "No serious correctness bug found. Every path traced keeps the
early Done's tokens, stop reason and spec counts equal to what `spec_burst_post` /
`dspark_burst_post` and `finish` later compute."

What it confirmed (summary): the hook's inputs are final when it fires and identical to the post
step's; the hook fires at most once, on an Ok outcome, at the gate pass after the stopping round;
the two RefCell borrows never overlap; a step error after an early Done is not sent; a follow-up
request is admitted only at the next tick top, after the session is parked or freed; the verdict
refactor keeps `spec_burst_post` and `dspark_burst_post` as they were.

**1. Risk (gate): the cohort judge's hold-back slack applied to every member 0, `max_tokens` and
EOS rows included,** so an end hook that fired a round or two late on those rows would pass.
- Disposition: fixed in ae4b6a3455. The allowance is by end kind: none at `max_tokens`, one of the
  request's rounds on EOS (the EOS token carries no text), the hold-back allowance only for rows
  that ended on a stop string.

**2. Risk: a pooled member closed early with nothing emitted (a stop id as its first token) could
pass `step_oom_parkable` after a cohort OOM fault and be replayed for a client that already has
`[DONE]`.**
- Disposition: fixed in ae4b6a3455: the OOM park arm skips a member whose `early_done` is set; the
  source-shape test of the park arm names the guard (f47c2092a4).

**3. Test gap: nothing proved that the early receipt equals what the post step computes, or that
`finish` sends nothing after an early Done and catches a mismatch; the red arm mutated the new
function rather than reproducing the old late close.**
- Disposition: taken. `finish` publishes through `publish_terminal`, tested by
  `finish_sends_nothing_after_an_early_done_and_checks_it` (nothing after an early Done, a
  mismatch reported and never sent); `pooled_member_closes_at_its_round_with_the_post_steps_receipt`
  now compares each early receipt with the post step's computation from the same burst. The red
  arm on CPU stays a mutation (main has no hook to test); the old late close itself is the device
  red arm: main's cohort boots on both devices (README).

**4. Nit: the early path skipped `finish`'s `tokens_emitted == generated.len()` check.**
- Disposition: fixed in ae4b6a3455: `pooled_early_done` takes the round hook's sent count and
  closes early only when it equals the kept tokens; otherwise `finish` closes as before.

**5. Nit (gate): member 0's log chunk includes its companions' lines; the settle line is matched
by completion tokens.**
- Disposition: kept, stated. It can only cause a false failure (demotion, park lines), and the
  companions run to 192 tokens, past every member 0's count.

**6. Nit: the token vector was cloned twice on every close.**
- Disposition: fixed: `DoneReceipt::send` takes the receipt by value (one copy in `finish`, one
  more only on the early path, where the receipt is kept for the comparison).

**7. Nit: the stop-string debug assertion no longer fired on a budget end (MaxNew is decided in
the verdict now).**
- Disposition: fixed: it asserts no completed stop string unless the stop is a stop string, stop
  id or EOS, on both post steps.

**8. Nit (gate): `--cohort 5` labelled its four-companion throughput cohorts "of 5".**
- Disposition: fixed: the throughput cohort is at most the companions there are, and is labelled so.

**9. Known limitation: context-full ends and the DSpark burst-cadence arm still close after the
whole cohort.**
- Disposition: stated in the decision record, the README and the PR.
