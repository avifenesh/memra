set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import subprocess
import sys

lane = Path("research/mtp-context-depth-20260921")
tools = lane / "publication"
for folder, program in [
    ("prior-receipts", "reproduce_development.py"),
    ("receipts", "reproduce.py"),
    ("latest-receipts", "reproduce_latest.py"),
]:
    receipts = lane / folder
    pin = (receipts / "manifest.sha256").read_text().split()[0]
    subprocess.run([
        sys.executable, str(tools / program), "--receipts", str(receipts),
        "--manifest-sha256", pin,
    ], check=True)
    subprocess.run([
        sys.executable, str(tools / "verify_boundary.py"),
        "--receipts", str(receipts),
        "--review", str(receipts / "boundary-review.json"),
        "--prefix", receipts.as_posix(), "--check",
    ], check=True)
PY
