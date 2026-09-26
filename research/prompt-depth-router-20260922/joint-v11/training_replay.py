"""Replay v11 randomized K/D/C native training receipts from sealed bytes."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"
DOMAINS = ("ifeval", "gsm8k")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def extract(archive, manifest, root):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in manifest["members"]
                or name in seen
                or not member.isfile()
                or logical.is_absolute()
                or ".." in logical.parts
                or str(logical) != name
                or member.size != manifest["members"][name]["bytes"]
            ):
                raise ValueError(f"unsafe or changed training member: {name}")
            seen.add(name)
            path = root.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            with source.extractfile(member) as input_file, path.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    value.update(chunk)
                    output.write(chunk)
            if value.hexdigest() != manifest["members"][name]["sha256"]:
                raise ValueError(f"training member hash differs: {name}")
    if seen != set(manifest["members"]):
        raise ValueError("sealed training data omits members")
    return len(seen)


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or manifest["model_sha256"] != MODEL_SHA
        or manifest["binary_sha256"] != BINARY_SHA
        or manifest["workloads_sha256"] != WORKLOAD_SHA
    ):
        raise ValueError("sealed training manifest differs")
    with tempfile.TemporaryDirectory(prefix="mtp-v11-training-replay-") as temp:
        root = Path(temp)
        members = extract(archive, manifest, root)
        inputs = root / "inputs/workloads"
        native = root / "native/training-results"
        if sha(inputs / "manifest.json") != WORKLOAD_SHA:
            raise ValueError("v11 training prompt split differs")
        tasks = json.loads((inputs / "manifest.json").read_text())
        expected_inputs = {"manifest.json"} | {
            entry["file"]
            for phase in ("qualification", "training")
            for domain in ("ifeval", "gsm8k")
            for entry in tasks["groups"][phase][domain]
        }
        actual_inputs = {
            path.relative_to(inputs).as_posix()
            for path in inputs.rglob("*") if path.is_file()
        }
        if (
            set(tasks["groups"]) != {"qualification", "training"}
            or set(tasks["splits"]) != {"qualification", "training"}
            or tasks["source_full_manifest_sha256"]
            != "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"
            or tasks["validation_projection_sha256"]
            != "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
            or actual_inputs != expected_inputs
        ):
            raise ValueError("training archive exposes a reserved split")
        metadata = json.loads((root / "native/run-meta.json").read_text())
        pilot_path = root / "diagnostic/pilot-result.json"
        pilot = json.loads(pilot_path.read_text())
        behavior_path = root / "diagnostic/behavior-v10.json"
        behavior = json.loads(behavior_path.read_text())
        source_path = root / "inputs/source-manifest.json"
        source = json.loads(source_path.read_text())
        rental_path = root / "inputs/rental.json"
        rental = json.loads(rental_path.read_text())
        if (
            metadata["model_sha256"] != MODEL_SHA
            or metadata["binary_sha256"] != BINARY_SHA
            or metadata["customer_capture"] is not False
            or metadata["workloads_sha256"] != WORKLOAD_SHA
            or metadata["full_workloads_sha256"]
            != tasks["source_full_manifest_sha256"]
            or metadata["validation_workloads_sha256"]
            != tasks["validation_projection_sha256"]
            or metadata["source_manifest_sha256"] != sha(source_path)
            or metadata["rental_sha256"] != sha(rental_path)
            or metadata["ops_source_sha256"]
            != sha(root / "source/private_ops/training_runmeta.py")
            or metadata["finalizer_source_sha256"]
            != sha(root / "source/private_ops/finalize_v11.py")
            or metadata["v10_projection_source_sha256"]
            != sha(root / "source/private_ops/prepare_v10_projection.py")
            or metadata["pilot_result_sha256"] != sha(pilot_path)
            or metadata["behavior_v10_sha256"] != sha(behavior_path)
            or behavior["archive_sha256"] != metadata["v10_archive_sha256"]
            or metadata["v9_collect_sha256"]
            != sha(root / "source/joint-v9/collect.py")
            or metadata["instance_id"] != rental["instance_id"]
            or metadata["provider"] != rental["provider"]
        ):
            raise ValueError("training host or model metadata differs")
        if source["schema"] != 1:
            raise ValueError("training source manifest schema differs")
        for name, expected in source["files"].items():
            if sha(root / "source/joint-v11" / name) != expected:
                raise ValueError(f"training source changed: {name}")
        for name, expected in metadata["v9_source_sha256"].items():
            if sha(root / "source/joint-v9" / name) != expected:
                raise ValueError(f"v9 parent source changed: {name}")
        for name, expected in metadata["v4_source_sha256"].items():
            if sha(root / "source/joint-v4" / name) != expected:
                raise ValueError(f"v4 feature source changed: {name}")
        sys.path.insert(0, str(root / "source/joint-v9"))
        import collect as v9_collect
        collect_path = root / "source/joint-v11/collect.py"
        spec = importlib.util.spec_from_file_location(
            "v11_training_collect", collect_path
        )
        v11_collect = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v11_collect)
        pilot_root = root / "diagnostic/pilot-results"
        k_name = "training-32-random-k"
        d_name = "training-33-k20-explore-d"
        k_entry = tasks["groups"]["qualification"]["ifeval"][0]
        d_entry = tasks["groups"]["qualification"]["gsm8k"][0]
        k_observed = v11_collect.verify_random(
            pilot_root / k_name, k_entry, 32
        )
        d_observed = v9_collect.verify_session(
            pilot_root / d_name, d_entry, "explore-d", 20, "training"
        )
        if (
            pilot["status"] != "random-K-and-D-C-pilot-qualified"
            or pilot["training_use"] is not False
            or pilot["graded_instruction_inputs"] != 136
            or pilot["c_offer_labels"] <= 0
            or pilot["d_round_labels"] <= 0
            or pilot["v10_fixed_rows"]["k"] <= 0
            or pilot["v10_fixed_rows"]["d"] <= 0
            or pilot["v10_fixed_rows"]["c"] != 0
            or pilot["code_parent_status"]
            != "pinned-code-training-extracted"
            or pilot["transfer_parent_status"]
            != "pinned-transfer-grader-inputs-extracted"
            or pilot["sessions"] != [k_name, d_name]
            or pilot["k_actions"] != {
                str(k): v for k, v in k_observed["k_actions"].items()
            }
            or pilot["d_exposure"] != d_observed["d_exposure"]
            or json.loads(
                (pilot_root / f"{k_name}.result.json").read_text()
            ) != canonical(k_observed)
            or json.loads(
                (pilot_root / f"{d_name}.result.json").read_text()
            ) != canonical(d_observed)
            or pilot["native_kv_later_turns"] != [7, 7]
        ):
            raise ValueError("v11 disjoint K/D pilot replay differs")
        for name, entry, arm, cap, k, extra in (
            (
                k_name, k_entry, "fixed:3", 3, 20,
                "draft-schedule=" + ",".join(map(
                    str, v11_collect.draft_schedule(32)
                )),
            ),
            (
                d_name, d_entry, "explore-d", 4, 20,
                f"explore-seed={20926000 + 33 * 31 + 20}",
            ),
        ):
            command = json.loads(
                (pilot_root / f"{name}.command.json").read_text()
            )
            exit_row = json.loads(
                (pilot_root / f"{name}.exit.json").read_text()
            )
            argv = command["argv"]
            log = (pilot_root / f"{name}.stderr.log").read_text()
            if (
                command["model_sha256"] != MODEL_SHA
                or command["binary_sha256"] != BINARY_SHA
                or command["workload_sha256"] != entry["sha256"]
                or exit_row["returncode"] != 0
                or argv[2] != "embedded"
                or Path(argv[3]).name != entry["file"]
                or Path(argv[4]).name != name
                or argv[5] != arm
                or argv[6] != str(entry["seed"])
                or argv[7:10] != ["4096", "65536", "1.0"]
                or argv[10:13] != [
                    f"cap={cap}", "sampler-top-k=20",
                    f"draft-top-k={k}",
                ]
                or argv[13:] != [extra]
                or "full_vocab=248320 draft_vocab=248320 mtp=embedded"
                not in log
                or "target_top_k=20" not in log
            ):
                raise ValueError(f"v11 pilot command or engagement differs: {name}")

        observed = set()
        for domain_index, domain in enumerate(DOMAINS):
            entries = tasks["groups"]["training"][domain]
            if len(entries) != 16:
                raise ValueError("v11 training topic count differs")
            for index, entry in enumerate(entries):
                if sha(inputs / entry["file"]) != entry["sha256"]:
                    raise ValueError("training conversation text differs")
                global_index = domain_index * 16 + index
                for k in (3, 10, 20):
                    for label in ("fixed-d3", "explore-d"):
                        name = f"training-{global_index}-k{k}-{label}"
                        row = json.loads((native / f"{name}.result.json").read_text())
                        actual = v9_collect.verify_session(
                            native / name, entry, label, k, "training"
                        )
                        if row != actual:
                            raise ValueError(f"training native receipt differs: {name}")
                        command = json.loads(
                            (native / f"{name}.command.json").read_text()
                        )
                        exit_row = json.loads(
                            (native / f"{name}.exit.json").read_text()
                        )
                        if (
                            command["model_sha256"] != MODEL_SHA
                            or command["binary_sha256"] != BINARY_SHA
                            or command["workload_sha256"] != entry["sha256"]
                            or exit_row["returncode"] != 0
                        ):
                            raise ValueError(f"training command differs: {name}")
                        log = (native / f"{name}.stderr.log").read_text()
                        if (
                            "full_vocab=248320 draft_vocab=248320 mtp=embedded"
                            not in log
                            or "target_top_k=20" not in log
                            or f"draft_top_k={k}" not in log
                        ):
                            raise ValueError(f"training MTP engagement differs: {name}")
                        observed.add(name)
                name = f"training-{global_index}-random-k"
                row = json.loads((native / f"{name}.result.json").read_text())
                actual = v11_collect.verify_random(
                    native / name, entry, global_index
                )
                command = json.loads(
                    (native / f"{name}.command.json").read_text()
                )
                exit_row = json.loads(
                    (native / f"{name}.exit.json").read_text()
                )
                if (
                    row != canonical(actual)
                    or command["model_sha256"] != MODEL_SHA
                    or command["binary_sha256"] != BINARY_SHA
                    or command["workload_sha256"] != entry["sha256"]
                    or command["draft_schedule"]
                    != v11_collect.draft_schedule(global_index)
                    or command["target_top_k"] != 20
                    or command["assignment"] != "randomized-turn"
                    or exit_row["returncode"] != 0
                ):
                    raise ValueError(f"random K native receipt differs: {name}")
                log = (native / f"{name}.stderr.log").read_text()
                if (
                    "full_vocab=248320 draft_vocab=248320 mtp=embedded"
                    not in log or "target_top_k=20" not in log
                ):
                    raise ValueError(f"random K engagement differs: {name}")
                observed.add(name)
        if len(observed) != 224:
            raise ValueError("v11 training arm inventory differs")
    return {
        "status": "training-native-random-K-D-C-KV-replay-match",
        "sessions": len(observed),
        "members": members,
        "archive_sha256": manifest["archive_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.archive, args.manifest)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
