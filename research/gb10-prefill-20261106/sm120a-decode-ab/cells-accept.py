"""The sm_120a landing record for #1102's head with the perf flag accepted on a measured reason.

The caller already holds the 5090 lease and the rig build lock. The trees are prebuilt, so the tool's
build step is a no-op; its flock points at a private lock file so it does not wait on the lock the
caller holds. The reason cites the quiet decode A/B (decode-ab.json) and the code reach.
usage: cells-accept.py REPO HEAD BASE WORK OUT DECODE_AB_JSON
"""
import json
import sys
from pathlib import Path

repo, head, base, work, out, ab = sys.argv[1:7]
sys.path.insert(0, str(Path(repo) / "tools"))
import cell_evidence as ce  # noqa: E402

s = json.loads(Path(ab).read_text())["summary"]
# 09:28: main moved under #1099 (f68bf32282), so a record on this head would be void before it
# lands. The A/B above is kept; the record runs later on #1102's final head.
SUPERSEDED = {"2d9f6ed3ff2192b16731286921aeee4b7fb002ea"}
if ce.resolve(Path(repo), head) in SUPERSEDED:
    print("head superseded by main moving; record skipped, decode A/B kept")
    sys.exit(0)
reason = (f"not a regression: quiet interleaved decode A/B of the same binaries on this card, qwen3.8-27b, "
          f"{s['kept']['A']} + {s['kept']['B']} runs kept (drop rule: 1-min load > 24 before a run), "
          f"base median {s['median_A']:.2f} tok/s, head {s['median_B']:.2f} tok/s ({s['delta_pct']:+.2f}%, "
          f"Mann-Whitney p {s['mann_whitney_p']}); the change does not reach decode on this card (the v2/v3 F8F4 "
          f"GEMMs are off on discrete cards); earlier records of the same program: -3.07 / +2.63 / -3.01 / +1.62%")
ce.BUILD_LOCK = Path.home() / ".cache/memra-gemm-v3/no-op-build.lock"
record = ce.produce(Path(repo), head, "rtx5090", ["battery", "perf", "serving"], Path(out), base=base,
                    work=Path(work), lock=False, accept_regression=reason)
print("status", record["status"], record["failures"])
print("accepted_regression", record["kinds"]["perf"].get("accepted_regression"))
