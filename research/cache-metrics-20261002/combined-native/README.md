# Combined rig native evidence

Cell `1d051ea29b31` ran the actual C904 background/image integration base plus A's
rig-only metrics work. It passed every group: cache accounting, plain/MTP lifecycle,
forced queue, two active-KV observations, a worker-panic control after HTTP 200,
plain/MTP identity, both native/OpenAI JSON format cache gates, the short background
callback composition check, and Qwen3.6 MoE c4 (#777). The exact source patch, binary,
artifact and environment identities remain in their receipts. No DSv4 native run
or support promotion is claimed.

The background cell uses event-driven test-only callbacks from the real server and
InMemoryJobStore. It proves one queued HTTP acknowledgement separately from one
terminal generation/accounting callback, stable repeated reads, cancellation with
partial output, and no duplicate billing or successful completion histogram entry
for the cancelled job. It does not repeat C's long-deadline campaign.

B's subsequent review found two issues: the HTTP lane label needed authenticated
lane resolution, and short-prime routing needed a GDN capability guard. Their
fixes and affected native rechecks are recorded in `../review-fixes/` and
`../review-native/`. The pre-fix combined cell is retained under its original
identity, not relabelled as execution of the later binary. Its cache/capacity/fault,
format, background and c4 assertions are unaffected by those two bounded fixes.
