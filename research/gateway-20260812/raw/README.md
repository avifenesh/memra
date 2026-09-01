# Gateway receipt index

All timestamps are UTC. Raw request and server records contain no prompt or
response bodies; completion equality is represented by SHA-256 classes.

- `sources/`: current official OpenRouter schema/application snapshots and
  captured public tiyuvta.ai state.
- `local/`: source tests, GPU correctness gates, manifest verification,
  independent ledger/output replay, billing totals, and credential scan.
- `remote-setup/`: eu-west host, runtime, binary, artifact, and build provenance.
- `preflight/`: final corrected-runner pair preflight, PASS.
- `soak/`: final 7,200-second pair soak, PASS, including both request ledgers,
  probe JSONL files, unfiltered server logs, five-second thermal/VM samples, and
  nested off-host manifests.
- `preflight-attempts/20260812T071607Z`: stopped before GPU work because the
  cx-q35bug process still owned the serving port.
- `preflight-attempts/20260812T080240Z`: exposed and retained the probe's invalid
  assumption that a valid early EOS had to consume the full output maximum.
- `preflight-attempts/20260812T080532Z`: exact-commit guard stopped the run before
  GPU work after an incorrect expected commit was supplied.
- `preflight-attempts/20260812T080548Z`: exposed and retained the probe's invalid
  rejection of the two frozen Week-1 c4 batched-prime output classes.
- `preflight-attempts/20260812T081016Z`: pair protocol preflight passed, but its
  manifest exposed a lexical thermal min/max reduction; excluded before soak.

Each remote directory carries a SHA-256 manifest or attempt manifest. The local
`remote-receipt-validation.log` verifies every top-level and nested manifest;
`remote-evidence-analysis.log` independently replays the scored assertions; and
`staged-receipt-validation.log` verifies every manifest reference directly from
the staged Git blobs.
