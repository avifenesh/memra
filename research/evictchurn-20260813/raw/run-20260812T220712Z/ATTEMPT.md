# Non-scored harness attempt

This attempt stopped at the round-robin harness gate and is retained so the failure is not
erased. The exact client-side failure was:

```text
RuntimeError: stream completed without a visible text token
```

Eight of 80 completions ended at EOS without visible UTF-8. The worker completed all 80
requests and published their cache counters, but the imported prefixmoney exactness client
treated those valid terminal-only streams as failed requests. No server/runtime change was
made. The harness was narrowed to retain a terminal choice event as an EOS-only completion,
record its empty-byte SHA-256, and label that row's timing basis `terminal_eos_event`. The same
workload was then rerun from fresh servers in `run-20260812T221018Z`; only that complete run is
scored.

This directory is incomplete by construction: the fail-closed runner stopped before the hot-set
and sequential-scan phases and therefore did not write its final thermal summary or manifest.
