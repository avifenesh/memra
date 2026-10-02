# Rig-only shared metrics preparation

This source excludes the prepared DSv4 emitter, allocation, cache and settlement
integrations. Those changes are preserved on the separate deferred branch and
are not part of this PR. #522 remains open.

Build eb617bcebf04 passes 27 selected metrics/HTTP tests, Clippy, seven Prometheus
parser tests and three cache usage tests. Source, patch and binary identities are
recorded beside the logs. The broad preceding CPU run passed 664 engine, 375
model/config and 84 KV tests; the server suite passed 1,035 tests before the latest
bounded HTTP status/cancellation regressions. GPU-only ignored tests are not
claimed as executed.

Final native evidence will be recorded on the reconciled background/image tree.
The earlier local native before/after evidence remains separately identified by
its source and binary; it is not relabelled as a run of this source.
