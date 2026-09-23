# Direct-admission fixture boundary

The frozen `fa0b36899` selected campaign passed inspection, Eager/Graph capture,
and all eight retained GraphSession drift cases. Its first worker case measured
and reinstalled a valid eager receipt in the actual worker test executable, then
failed with `ADMISSION_USAGE_MISSING` before its positive caller or drift witness.
The phase stayed failed, its selected index was quarantined, and the other sixteen
cases, including the T16 fallback, stayed unrun. The compressed stdout/stderr here
retain the original bytes and hashes. This does not revise the earlier #585 result.

The fixture uses real `prepare_request` and `admit` to create its session. That
inner entry does not publish events. Production's outer worker loop publishes
`PromptUsage` after `admit` returns, using the returned `Session.n_prompt` and
`Session.n_cached`. The old probe demanded this outer-loop event without entering
that loop. The repair first checks those real returned counts and an empty event
channel, then invokes the same production publisher using the returned session.
The original exact-one-usage-event assertion remains before the actual worker
caller. The subsequent positive, mutation, refusal, restored-environment, and
same-origin checks are unchanged.

The existing production send is extracted into a private helper; its production
location, count values, event count and ignored send-error behavior are unchanged.
Publication stays outside `admit`. This does not qualify full scheduler accounting
or HTTP usage behavior. No numerical program or worker dispatch is changed.

`../run-worker-publisher-tests.py` runs the actual extracted publisher/channel and
its two Rust CPU tests: exact single delivery with prompt/cached pairs 4/0, 37/19,
and 0/0, plus dropped-receiver behavior. Removing the send from an isolated copy
makes the delivery test fail. Only unrelated event payload types are stand-ins;
this is a CPU publisher control, not a native admission or caller pass. The strict
Linux-target server all-targets clippy check, formatting and diff checks pass.

The native green result remains pending independent source review, fresh
exact-source build and all required selected cases. The original failed run and
its sixteen unrun cases remain unchanged.
