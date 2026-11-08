# RTX 5090 decode A/B behind #1102's accepted sm_120a perf flag (2026-11-08)

The cell tool's perf A/B for #1102 times qwen3.8-27b decode, 3 base runs against 3 head runs. It
flagged that program in two of four records: -3.07 / +2.63 / -3.01 / +1.62%. On the RTX 5090 the
change cannot reach decode, because the v2/v3 F8F4 GEMMs are off on discrete cards. This
measurement decided the flag.

Binaries are the cell tool's own: A = run-gen of main `259ef0d310`, B = run-gen of #1102
`fccd0958f2`. main and #1102 had the same crates as the landed `e787fac9dc` and its parent, apart
from #1102's change. The command is the tool's perf command (`MEMRA_NGEN=128` on its fixed prompt)
in the tool's build environment. The rig build lock and the 5090 lease were held for the whole run
(`quiet-5090b.sh`).

Fixed before the first run, in `decode-ab.py`:
- **Drop rule:** a run is dropped when the 1-minute load average just before it is above 24, the
  rig's thread count.
- **Stopping rule:** A B pairs continue until both arms have 10 kept runs.
- **Decision:** ACCEPT when the B median is at least the A median, or when |delta| < 1.5% with a
  two-sided Mann-Whitney p >= 0.05.

The first attempt was INCONCLUSIVE. Another project's Go tests held the rig at load 45-66, and only
1 + 1 runs were kept. This attempt (`decode-ab.json`, 22 runs) dropped one run (load 28.4) and kept
11 + 10:

| arm | median | mean |
|---|---|---|
| A, main | 43.02 tok/s | 43.07 tok/s |
| B, #1102 | 42.97 tok/s | 42.46 tok/s |

The difference is -0.12%, Mann-Whitney p 0.34: ACCEPT. The landed sm_120a record
(`research/release-qualification/cells/419037a145cc-rtx5090-battery-perf-serving/`) carries this
as its accepted-flag reason (`cells-accept.py`, `record-5090.sh`). Its own A/B was -1.29% and was
not flagged.
