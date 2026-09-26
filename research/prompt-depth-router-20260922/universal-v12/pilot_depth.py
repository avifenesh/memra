"""Pilot fixed D and same-K learned-C/D no-op identity before training."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v9"))
import eval as v9_eval
import collect as v9_collect


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
DOMAINS = ("code", "prose", "math")
V9_SHA = "a914e20a4f823acbdf189202806415785d60507fa0f775ab588b10083d3c034b"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def model_files(archive, manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["archive_sha256"] != V9_SHA
        or sha(archive) != V9_SHA
    ):
        raise ValueError("fixed-K C/D pilot parent archive changed")
    wanted = {
        f"training/cd-models/augmented/topk{k}/{kind}-history.tsv":
        out / f"topk{k}/{kind}-history.tsv"
        for k in (3, 10, 20)
        for kind in ("depth", "confidence")
    }
    if out.exists():
        if any(
            sha(path) != manifest["members"][name]["sha256"]
            for name, path in wanted.items()
        ):
            raise ValueError("resumed C/D pilot model bytes changed")
    else:
        out.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            seen = set()
            for member in source:
                if member.name not in wanted:
                    continue
                if member.name in seen or not member.isfile():
                    raise ValueError("C/D pilot model archive member differs")
                seen.add(member.name)
                expected = manifest["members"][member.name]
                if member.size != expected["bytes"]:
                    raise ValueError("C/D pilot model byte count differs")
                path = wanted[member.name]
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, path.open("xb") as output:
                    while chunk := inp.read(1024 * 1024):
                        output.write(chunk)
                if sha(path) != expected["sha256"]:
                    raise ValueError("C/D pilot model SHA differs")
            if seen != set(wanted):
                raise ValueError("C/D pilot model archive is incomplete")
    return {
        name: manifest["members"][name]["sha256"]
        for name in sorted(wanted)
    }


def record_result(out, row):
    path = out / f"{row['name']}.result.json"
    if path.exists():
        if json.loads(path.read_text()) != canonical(row):
            raise ValueError("resumed native pilot result changed")
    else:
        save(path, row)


def pilot(args):
    if (
        sha(args.model) != v9_eval.MODEL_SHA256
        or sha(args.binary) != v9_eval.BINARY_SHA256
        or sha(args.workloads / "manifest.json") != TRAIN_SHA
    ):
        raise ValueError("fixed depth pilot source changed")
    meta = json.loads(args.run_meta.read_text())
    gpu = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ], text=True,
    ).strip().splitlines()
    if (
        meta["model_sha256"] != v9_eval.MODEL_SHA256
        or meta["binary_sha256"] != v9_eval.BINARY_SHA256
        or meta["training_workloads_sha256"] != TRAIN_SHA
        or meta["customer_capture"] is not False
        or len(gpu) != 1
        or gpu[0] != meta["gpu_uuid"]
    ):
        raise ValueError("fixed depth pilot moved research GPU")
    manifest = json.loads(
        (args.workloads / "manifest.json").read_text()
    )
    if set(manifest["groups"]) != {
        "qualification", "training",
    }:
        raise ValueError("fixed depth pilot saw reserved prompts")
    weights = model_files(
        args.v9_archive, args.v9_manifest,
        args.model_out,
    )
    args.out.mkdir(exist_ok=True)
    sessions = []
    for domain in DOMAINS:
        entry = manifest["groups"]["qualification"][domain][0]
        for depth in (1, 2):
            spec = {
                "label": f"fixed-k20-d{depth}-c0",
                "k": 20, "cap": depth,
                "arm": f"fixed:{depth}", "extra": [],
            }
            native_args = SimpleNamespace(
                binary=args.binary, model=args.model,
                workloads=args.workloads, out=args.out,
                phase=f"pilot-{domain}",
            )
            row = v9_eval.run_one(native_args, entry, 0, spec)
            rounds = v9_collect.table(
                args.out / row["name"] / "rounds.tsv"
            )
            actual_depths = [
                int(item["draft_depth"])
                for item in rounds
                if item["eligible_for_learning"] == "true"
            ]
            if (
                not actual_depths
                or max(actual_depths) > depth
                or (depth == 1 and set(actual_depths) != {1})
                or (depth == 2 and 2 not in actual_depths)
                or row["cached_later_turns"] != 7
                or set(row["k_actions"]) != {20}
                or row["c_decisions"] != 0
                or row["c_stops"] != 0
            ):
                raise ValueError("native fixed depth did not engage")
            record_result(args.out, row)
            sessions.append({
                "domain": domain, "depth": depth,
                "session": row["name"],
                "eligible_rounds": len(actual_depths),
                "actual_depths": {
                    str(d): actual_depths.count(d)
                    for d in sorted(set(actual_depths))
                },
                "loop_hits": row["loops"],
            })
    code = manifest["groups"]["qualification"]["code"][0]
    checks = []
    native_args = SimpleNamespace(
        binary=args.binary, model=args.model,
        workloads=args.workloads, out=args.out,
        phase="pilot-code",
    )
    for k in (3, 10, 20):
        fixed = v9_eval.run_one(native_args, code, 0, {
            "label": f"fixed-k{k}-d3-c0",
            "k": k, "cap": 3, "arm": "fixed:3",
            "extra": [],
        })
        record_result(args.out, fixed)
        extra = [
            f"depth-model={args.model_out}/topk{k}/depth-history.tsv",
            f"confidence-model={args.model_out}/topk{k}/confidence-history.tsv",
        ]
        noop = v9_eval.run_one(native_args, code, 0, {
            "label": f"cd-noop-k{k}",
            "k": k, "cap": 4, "arm": "noop-cd",
            "extra": extra,
        })
        record_result(args.out, noop)
        for turn in range(1, 9):
            ids = (
                args.out / fixed["name"]
                / f"turn-{turn}.output.ids"
            ).read_bytes()
            observed = (
                args.out / noop["name"]
                / f"turn-{turn}.output.ids"
            ).read_bytes()
            if observed != ids:
                raise ValueError("model-running C/D no-op differs at fixed K")
        if (
            fixed["cached_later_turns"] != 7
            or noop["cached_later_turns"] != 7
            or set(fixed["k_actions"]) != {k}
            or set(noop["k_actions"]) != {k}
            or noop["cd_model_s"] <= 0
        ):
            raise ValueError("same-K C/D no-op did not engage")
        checks.append({
            "k": k,
            "fixed_session": fixed["name"],
            "noop_session": noop["name"],
            "noop_cd_model_seconds": noop["cd_model_s"],
        })
    return {
        "schema": 1,
        "status": "fixed-D1-D2-and-same-K-CD-noops-qualified",
        "model_sha256": v9_eval.MODEL_SHA256,
        "binary_sha256": v9_eval.BINARY_SHA256,
        "training_workloads_sha256": TRAIN_SHA,
        "run_meta_sha256": sha(args.run_meta),
        "gpu_uuid": meta["gpu_uuid"],
        "sessions": sessions,
        "cd_noop_checks": checks,
        "pilot_model_archive_sha256": V9_SHA,
        "pilot_model_member_sha256": weights,
    }


def replay(root, receipt_path, workloads, run_meta, model_dir):
    receipt = json.loads(receipt_path.read_text())
    meta = json.loads(run_meta.read_text())
    manifest = json.loads(
        (workloads / "manifest.json").read_text()
    )
    if (
        receipt["status"]
        != "fixed-D1-D2-and-same-K-CD-noops-qualified"
        or receipt["model_sha256"] != v9_eval.MODEL_SHA256
        or receipt["binary_sha256"] != v9_eval.BINARY_SHA256
        or receipt["training_workloads_sha256"] != TRAIN_SHA
        or receipt["run_meta_sha256"] != sha(run_meta)
        or receipt["gpu_uuid"] != meta["gpu_uuid"]
        or len(receipt["sessions"]) != 6
        or len(receipt["cd_noop_checks"]) != 3
        or receipt["pilot_model_archive_sha256"] != V9_SHA
    ):
        raise ValueError("sealed fixed-depth pilot receipt differs")
    for item in receipt["sessions"]:
        domain = item["domain"]
        depth = item["depth"]
        if domain not in DOMAINS or depth not in (1, 2):
            raise ValueError("sealed pilot domain or depth differs")
        name = f"pilot-{domain}-0-fixed-k20-d{depth}-c0"
        if item["session"] != name:
            raise ValueError("sealed fixed-depth pilot name differs")
        spec = {
            "label": f"fixed-k20-d{depth}-c0",
            "arm": f"fixed:{depth}", "k": 20,
        }
        entry = manifest["groups"]["qualification"][domain][0]
        observed = v9_eval.verify(root / name, entry, spec)
        recorded = json.loads(
            (root / f"{name}.result.json").read_text()
        )
        depths = [
            int(row["draft_depth"])
            for row in v9_collect.table(root / name / "rounds.tsv")
            if row["eligible_for_learning"] == "true"
        ]
        if (
            canonical(observed) != recorded
            or not depths
            or max(depths) > depth
            or (depth == 1 and set(depths) != {1})
            or (depth == 2 and 2 not in depths)
            or item["eligible_rounds"] != len(depths)
            or item["actual_depths"] != {
                str(d): depths.count(d) for d in sorted(set(depths))
            }
            or item["loop_hits"] != observed["loops"]
        ):
            raise ValueError("sealed D1/D2 native pilot changed")
    for item in receipt["cd_noop_checks"]:
        k = item["k"]
        if k not in (3, 10, 20):
            raise ValueError("sealed C/D no-op K differs")
        fixed_name = f"pilot-code-0-fixed-k{k}-d3-c0"
        noop_name = f"pilot-code-0-cd-noop-k{k}"
        if (
            item["fixed_session"] != fixed_name
            or item["noop_session"] != noop_name
        ):
            raise ValueError("sealed C/D no-op session name differs")
        for kind in ("depth", "confidence"):
            member = (
                f"training/cd-models/augmented/"
                f"topk{k}/{kind}-history.tsv"
            )
            path = model_dir / f"topk{k}/{kind}-history.tsv"
            if sha(path) != receipt["pilot_model_member_sha256"][member]:
                raise ValueError("sealed C/D no-op model weights differ")
        entry = manifest["groups"]["qualification"]["code"][0]
        fixed = v9_eval.verify(root / fixed_name, entry, {
            "label": f"fixed-k{k}-d3-c0",
            "arm": "fixed:3", "k": k,
        })
        noop = v9_eval.verify(root / noop_name, entry, {
            "label": f"cd-noop-k{k}",
            "arm": "noop-cd", "k": k,
        })
        if (
            json.loads((root / f"{fixed_name}.result.json").read_text())
            != canonical(fixed)
            or json.loads((root / f"{noop_name}.result.json").read_text())
            != canonical(noop)
            or noop["cd_model_s"] <= 0
            or item["noop_cd_model_seconds"] != noop["cd_model_s"]
        ):
            raise ValueError("sealed C/D no-op native receipt changed")
        for turn in range(1, 9):
            expected = (
                root / fixed_name / f"turn-{turn}.output.ids"
            ).read_bytes()
            observed = (
                root / noop_name / f"turn-{turn}.output.ids"
            ).read_bytes()
            if expected != observed:
                raise ValueError("sealed C/D no-op sampled output changed")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "binary", "model", "workloads", "run-meta", "out",
        "receipt", "v9-archive", "v9-manifest", "model-out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "binary", "model", "workloads", "run_meta", "out",
        "receipt", "v9_archive", "v9_manifest", "model_out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = pilot(args)
    save(args.receipt, result)
    print(json.dumps({
        "status": result["status"],
        "sessions": len(result["sessions"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
