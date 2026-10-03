set -euo pipefail
for family in qwen gemma; do
  python3 research/mtp-calibrated-depth-20260920/analyze.py \
    --family "$family" --ledger "$RUNNER_TEMP/mtp-cold/$family-selected-sets.json" \
    --receipts "$RUNNER_TEMP/mtp-cold" --output "$RUNNER_TEMP/cold-$family.json"
  python3 research/mtp-continuing-session-20260921/analyze.py \
    --family "$family" --ledger "$RUNNER_TEMP/mtp-warm/experiment/$family-selected-sets.json" \
    --receipts "$RUNNER_TEMP/mtp-warm/experiment" --output "$RUNNER_TEMP/warm-$family.json"
done
python3 - <<'PY'
import json, os
from pathlib import Path
temp = Path(os.environ["RUNNER_TEMP"])
for mode, lane in [
    ("cold", Path("research/mtp-calibrated-depth-20260920")),
    ("warm", Path("research/mtp-continuing-session-20260921")),
]:
    for family in ["qwen", "gemma"]:
        expected = json.loads((lane / f"{family}-analysis.json").read_text())
        actual = json.loads((temp / f"{mode}-{family}.json").read_text())
        assert actual == expected, (mode, family, "report changed")
print("All four reports reproduced exactly.")
PY

