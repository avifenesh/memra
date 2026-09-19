# Day-4 shared-file handoff

No new MEMRA environment name, runtime flag, kernel or default. Existing build controls
remain documented by DAY3-SHARED-FRAGMENTS.md; no decide-by door was introduced.

TESTING.md pointer should now name minimum integrated source `020d2047` plus D day-4
bootstrap/collector, not `914229ae`. A/B/C runners are now present. `--validate` accepts
both byte receipts and telemetry; `--first-hour` is correctness-only; see RIG-DAY1.md.

Public module `peer::test_support` avoids adding a lead-owned Cargo feature. B can consume
`FakePeerCapacity<G>` with its own shared governor; PEER-SEAM.md documents the seam.

Initial push refused because publish.yml omitted memra-tier. Lead repaired it on integration
`200a3c66`; D independently inspected and merged that one-line topological publish-order
repair. The following lane push passed hooks. No shared file was edited directly by D.
