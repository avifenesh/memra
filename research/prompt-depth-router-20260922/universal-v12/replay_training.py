"""Replay fresh prose K/D/C labels from sealed training-only bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v11"))
import training_replay as prior_replay
import measurement_rows_fresh
import pilot_depth
import seal_training


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def replay(archive, manifest_path, rows_out):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or manifest["scope"]
        != "fresh code-prose-math training-only native data"
        or manifest["model_sha256"] != seal_training.MODEL_SHA
        or manifest["binary_sha256"] != seal_training.BINARY_SHA
        or manifest["training_workloads_sha256"] != TRAIN_SHA
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or sha(archive) != manifest["archive_sha256"]
    ):
        raise ValueError("sealed prose training manifest differs")
    allowed = (
        "inputs/phase-training/",
        "source/universal-v12/",
        "source/joint-v9/",
        "source/joint-v11/",
        "source/joint-v4/",
        "source/private_ops/",
        "diagnostic/pilot-results/",
        "diagnostic/pilot-models/",
        "diagnostic/judge-preflight/",
        "native/training-mixed-results/",
    )
    if any(
        not name.startswith(allowed)
        and name not in {
            "native/rental.json",
            "native/cuda-accept.json",
            "native/run-meta.json",
            "native/pilot-result.json",
            "inputs/parent-custody.json",
            "inputs/provider-precheck.json",
        }
        for name in manifest["members"]
    ):
        raise ValueError("sealed prose training archive has unrelated files")
    with tempfile.TemporaryDirectory(
        prefix="mtp-v12-mixed-replay-",
    ) as folder:
        root = Path(folder)
        members = prior_replay.extract(archive, manifest, root)
        for group, names in {
            "universal-v12": (
                "collect_mixed.py",
                "measurement_rows_fresh.py",
                "seal_training.py",
                "replay_training.py",
                "pilot_depth.py",
            ),
            "joint-v9": (
                "collect.py", "eval.py", "measurement_rows.py",
            ),
            "joint-v11": (
                "collect.py", "measurement_rows.py",
                "training_replay.py",
            ),
            "joint-v4": (
                "train_depth.py", "train_confidence.py",
            ),
        }.items():
            for name in names:
                if sha(root / "source" / group / name) != (
                    sha(BASE / group / name)
                ):
                    raise ValueError("sealed mixed training source differs")
        workloads = root / "inputs/phase-training"
        seal_training.expected_workloads(workloads)
        if sha(root / "native/pilot-result.json") != (
            manifest["pilot_sha256"]
        ):
            raise ValueError("fixed depth pilot archive member changed")
        pilot_depth.replay(
            root / "diagnostic/pilot-results",
            root / "native/pilot-result.json",
            workloads, root / "native/run-meta.json",
            root / "diagnostic/pilot-models",
        )
        judge = root / "diagnostic/judge-preflight"
        if sha(judge / "manifest.json") != (
            manifest["judge_preflight_sha256"]
        ):
            raise ValueError("sealed independent judge preflight changed")
        judge_receipt = json.loads(
            (judge / "manifest.json").read_text()
        )
        if (
            judge_receipt["status"]
            != "independent-judge-template-and-order-qualified"
            or len(judge_receipt["receipts"]) != 2
            or judge_receipt["packet_renderer_sha256"]
            != sha(
                root / "source/universal-v12/prose_packets.py"
            )
            or {
                item["order"]: item["choice"]
                for item in judge_receipt["receipts"]
            }.get(0) not in ("A++", "A+")
            or {
                item["order"]: item["choice"]
                for item in judge_receipt["receipts"]
            }.get(1) not in ("B++", "B+")
            or any(
                sha(judge / f"response-{item['order']}.json")
                != item["raw_response_sha256"]
                for item in judge_receipt["receipts"]
            )
        ):
            raise ValueError("sealed independent judge access differs")
        if sha(root / "inputs/parent-custody.json") != (
            manifest["parent_custody_sha256"]
        ):
            raise ValueError("sealed older training parent custody changed")
        if sha(root / "inputs/provider-precheck.json") != (
            manifest["provider_precheck_sha256"]
        ):
            raise ValueError("sealed rental precheck changed")
        parents = json.loads(
            (root / "inputs/parent-custody.json").read_text()
        )
        metadata = json.loads(
            (root / "native/run-meta.json").read_text()
        )
        judge_file = metadata["judge_source_file"]
        access_file = metadata["judge_preflight_source_file"]
        if (
            Path(judge_file).name != judge_file
            or Path(access_file).name != access_file
            or not judge_file.endswith(".py")
            or not access_file.endswith(".py")
        ):
            raise ValueError("sealed private judge path differs")
        if (
            metadata["model_sha256"] != seal_training.MODEL_SHA
            or metadata["binary_sha256"] != seal_training.BINARY_SHA
            or metadata["training_workloads_sha256"] != TRAIN_SHA
            or metadata["source_full_manifest_sha256"] != FULL_SHA
            or metadata["customer_capture"] is not False
            or metadata["cuda_allocated"] is not True
            or metadata["rental_sha256"]
            != sha(root / "native/rental.json")
            or metadata["cuda_accept_sha256"]
            != sha(root / "native/cuda-accept.json")
            or metadata["parent_custody_sha256"]
            != sha(root / "inputs/parent-custody.json")
            or metadata["provider_precheck_sha256"]
            != sha(root / "inputs/provider-precheck.json")
            or metadata["v11_training_archive_sha256"]
            != parents["v11_archive_sha256"]
            or metadata["ops_source_sha256"]
            != sha(root / "source/private_ops/run_meta_v12.py")
            or metadata["judge_source_sha256"]
            != sha(root / "source/private_ops" / judge_file)
            or metadata["judge_preflight_source_sha256"]
            != sha(root / "source/private_ops" / access_file)
        ):
            raise ValueError("sealed research host metadata differs")
        for name, expected in metadata["source_files_sha256"].items():
            if (
                not name.startswith(
                    (
                        "universal-v12/", "joint-v9/",
                        "joint-v11/", "joint-v4/",
                    )
                )
                or sha(root / "source" / name) != expected
            ):
                raise ValueError("sealed native source changed after run metadata")
        rows_manifest = measurement_rows_fresh.build(
            root / "native/training-mixed-results", workloads,
            rows_out,
        )
        classes = json.loads(
            (rows_out / "token-classes.json").read_text()
        )
    return {
        "schema": 1,
        "status": "fresh-mixed-training-native-K-D-C-replay-match",
        "archive_sha256": manifest["archive_sha256"],
        "training_workloads_sha256": TRAIN_SHA,
        "source_full_manifest_sha256": FULL_SHA,
        "members": members,
        "sessions": 336,
        "row_counts": {
            kind: item["count"]
            for kind, item in rows_manifest["rows"].items()
        },
        "token_byte_classes": len(classes),
        "excluded_looped": rows_manifest["excluded_looped"],
        "reference_tok_s": rows_manifest["reference_tok_s"],
        "training_rows_manifest_sha256": sha(
            rows_out / "manifest.json"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.archive.resolve(), args.manifest.resolve(),
        args.rows_out.resolve(),
    )
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "status": result["status"],
        "row_counts": result["row_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
