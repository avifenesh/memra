"""Recompute both model reports from the portable archive bundle before release."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from analyze import analyze
from unpack_receipts import unpack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    receipts = a.root / "receipts"
    public = receipts / "public"
    with tempfile.TemporaryDirectory(prefix="public-reproduction-", dir=a.root / "tmp") as directory:
        destination = Path(directory) / "unpacked"
        count = unpack(public, destination)
        if count != 3:
            raise ValueError("The portable bundle is missing a model or common archive")
        reports = {}
        for family in ["qwen", "gemma"]:
            expected = json.loads((receipts / f"experiment/{family}-offline-audit.json").read_text())
            actual = analyze(
                destination / f"experiment/{family}-selected-sets.json",
                destination / "experiment", family,
            )
            if json.loads(json.dumps(actual)) != expected:
                raise ValueError(f"{family}: archive reproduction changed the report")
            reports[family] = {
                "runs": actual["audited_runs"], "turns": actual["audited_turns"],
                "complete_planned_matrix": actual["complete_planned_matrix"],
                "verdict": "exact-report-match",
            }
    result = {
        "archives": count, "reports": reports,
        "analyzer_sha256": hashlib.sha256(Path(__file__).with_name("analyze.py").read_bytes()).hexdigest(),
        "reader_sha256": hashlib.sha256(Path(__file__).with_name("unpack_receipts.py").read_bytes()).hexdigest(),
    }
    data = (json.dumps(result, indent=2) + "\n").encode()
    (public / "REPRODUCTION.json").write_bytes(data)
    (receipts / "public-reproduction-audit.json").write_bytes(data)
    for name in ["bundle.json", "manifest.json"]:
        path = public / name
        manifest = json.loads(path.read_text())
        manifest["reproduction"] = {"file": "REPRODUCTION.json", "sha256": hashlib.sha256(data).hexdigest()}
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
