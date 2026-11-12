# tools

Scripts and small binaries for building, gating, measuring and releasing memra. It is for
contributors looking for the right tool. There is no per-tool index: each tool documents itself in
its header comment, and most print it with `--help` or on a usage error.

Where to look instead:

- Which gate to run for a change, its command and pass line: [docs/TESTING.md](../docs/TESTING.md).
- The CPU contracts and fixtures (`test_*`, `run_*_contract.py`, the censuses):
  [docs/TESTING.md, CPU contracts](../docs/TESTING.md#cpu-contracts-and-fixtures).
- The dev-loop gate: [fast-gate/README.md](fast-gate/README.md).
- Hosted workflows and GPU CI (`gpu-ci.py`, `trusted_public_ci.py`): [docs/CI.md](../docs/CI.md).
- Release tools (`release-battery.sh`, `release_qualification.py`, `qualify-release.py`,
  `cell_evidence.py`, `daily-regression.py`): [docs/RELEASE-QUALIFICATION.md](../docs/RELEASE-QUALIFICATION.md)
  and [docs/RELEASING.md](../docs/RELEASING.md).
- The pre-push hook: `hooks/pre-push`, enabled with `git config core.hooksPath tools/hooks`.

The earlier index, with its notes on `box-health.sh`, `peer-read-probe.cu`,
`peer-copy-direction-probe.cu` and the host-tier spill gates, is kept in
[docs/history/TESTING.md](../docs/history/TESTING.md#tools-index).
