"""Use completed receipts to prove that forged calibration choices are rejected."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from analyze import analyze, audit_selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tmp", required=True, type=Path)
    a = p.parse_args()
    results = []
    for family in ["qwen", "gemma"]:
        winner, _, _ = audit_selection(a.receipts, family)
        for mutation in ["wrong-winner", "forged-calibration-record"]:
            with tempfile.TemporaryDirectory(prefix="selection-red-", dir=a.tmp) as directory:
                root = Path(directory)
                for cycle in [0, 1]:
                    name = f"{family}-calibration-{cycle}"
                    (root / name).symlink_to((a.receipts / name).resolve(), target_is_directory=True)
                shutil.copyfile(a.receipts / "workloads.lock.json", root / "workloads.lock.json")
                selection = json.loads((a.receipts / f"{family}-selection.json").read_text())
                if mutation == "wrong-winner":
                    selection["selected_k"] = 1 if winner != 1 else 2
                    expected = "selected K was not the calibration winner"
                else:
                    selection["calibration"][0]["records"][0]["tokens"] += 1
                    expected = "selection changed its calibration records"
                (root / f"{family}-selection.json").write_text(json.dumps(selection))
                try:
                    audit_selection(root, family)
                except AssertionError as exc:
                    if expected not in str(exc):
                        raise AssertionError(f"Red arm failed for an unintended reason: {exc}") from exc
                else:
                    raise AssertionError(f"Accepted {family} {mutation}")
            results.append({"family": family, "mutation": mutation, "verdict": "rejected"})
        for mutation in ["duplicate-evaluation-seed", "forged-throughput"]:
            with tempfile.TemporaryDirectory(prefix="evaluation-red-", dir=a.tmp) as directory:
                root = Path(directory)
                groups = json.loads((a.receipts / f"{family}-selected-sets.json").read_text())
                first_folder = groups[0]["receipt_dir"]
                for item in a.receipts.iterdir():
                    if item.name != first_folder:
                        (root / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
                overlay = root / first_folder
                overlay.mkdir()
                for item in (a.receipts / first_folder).iterdir():
                    if item.name != "runs.json":
                        (overlay / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
                records = json.loads((a.receipts / first_folder / "runs.json").read_text())
                if mutation == "duplicate-evaluation-seed":
                    groups[1]["seed"] = groups[0]["seed"]
                    expected = "duplicate or wrong evaluation seed"
                else:
                    groups[0]["records"][0]["e2e_tok_s"] *= 2
                    records[0]["e2e_tok_s"] *= 2
                    expected = "per-run throughput differs from token/time totals"
                (overlay / "runs.json").write_text(json.dumps(records))
                ledger = root / "red-ledger.json"
                ledger.write_text(json.dumps(groups))
                try:
                    analyze(ledger, root, family)
                except AssertionError as exc:
                    if expected not in str(exc):
                        raise AssertionError(f"Red arm failed for an unintended reason: {exc}") from exc
                else:
                    raise AssertionError(f"Accepted {family} {mutation}")
            results.append({"family": family, "mutation": mutation, "verdict": "rejected"})
    result = {
        "positive_calibration_audits": 2,
        "red_arms": results,
        "analyzer_sha256": hashlib.sha256(Path(__file__).with_name("analyze.py").read_bytes()).hexdigest(),
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
