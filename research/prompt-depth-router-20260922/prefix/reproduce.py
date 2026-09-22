"""Replay native audits and compiled prefix decisions before publishing results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from audit import audit_run, rows, same_tapes, save, sha
from report import report


def reproduce(root, runtime_archive, expected_runtime_sha, out):
    source = json.loads((root / "source.json").read_text())
    if sha(runtime_archive) != expected_runtime_sha or source["runtime_source_sha256"] != expected_runtime_sha:
        raise ValueError("runtime archive has no matching external identity")
    workloads = json.loads((root / "workloads/manifest.json").read_text())
    if workloads.get("schema") == 2:
        if sha(root / "workloads/workloads_simple.py") != workloads["generator_sha256"]:
            raise ValueError("versioned workload generator differs from frozen inputs")
        if source["harness_files"]["prefix/workloads.py"] != workloads["helper_sha256"]:
            raise ValueError("versioned workload helper differs from measured source")
    state = json.loads((root / "native/status.json").read_text())
    freeze = json.loads((root / "native/FREEZE.json").read_text())
    identity = json.loads((root / "native/identity.json").read_text())
    if (freeze["pipeline_sha256"] != source["harness_files"]["prefix/pipeline.py"]
            or identity["runner_sha256"] != source["harness_files"]["prefix/run.py"]
            or identity["source"] != source):
        raise ValueError("executed pipeline, runner or source identity changed")
    if state["status"] != "completed":
        raise ValueError("the matrix is not complete")
    with tempfile.TemporaryDirectory(prefix="prefix-replay-") as temporary:
        temp = Path(temporary)
        repo = temp / "repo"
        repo.mkdir()
        with tarfile.open(runtime_archive) as archive:
            archive.extractall(repo, filter="data")
        compiled = repo / "crates/memra-engine/src/bin/prefix_study_io"
        for local, name in ((Path(__file__).parent.parent / "router.rs", "router.rs"),
                            (Path(__file__).with_name("prefix_policy.rs"), "prefix_policy.rs")):
            if sha(local) != sha(compiled / name):
                raise ValueError("replay classifier differs from the actual compiled native source")
        base = repo / "research/mtp-context-depth-20260921"
        sys.path.insert(0, str(base))
        from audit_context import audit_run as context_auditor
        from loop_audit import loop_candidate
        from run_study import metrics_from_log
        binary = temp / "prefix-replay"
        subprocess.run(["rustc", "--edition=2024", "-O", str(Path(__file__).with_name("replay.rs")),
                        "-o", str(binary)], check=True, capture_output=True)
        replayed, prefixes = [], 0
        paired_prompts = {}
        for relative in state["completed"]:
            run = root / "native" / relative
            saved = json.loads(run.with_name(run.name + ".audit.json").read_text())
            family, phase = saved["family"], saved["phase"]
            inputs = workloads["families"][family]
            entry = (inputs["qualification"] if phase == "qualification"
                     else inputs["scenarios"][str(int(saved["label"].split("-")[0]))])
            expected_seed = entry["seed"]
            if phase == "qualification" and saved["label"].startswith("sampled-"):
                expected_seed += 10
            elif phase == "qualification" and saved["label"].startswith("coverage-"):
                expected_seed += 20
            command = json.loads(run.with_name(run.name + ".command.json").read_text())
            exit_row = json.loads(run.with_name(run.name + ".exit.json").read_text())
            if (saved["seed"] != expected_seed or command["seed"] != expected_seed
                    or command["binary_sha256"] != source["binaries"][f"{family}-prefix-study"]
                    or command["workload_sha256"] != entry["sha256"]
                    or command["arm"] != saved["arm"]
                    or command["max_new"] != saved["max_new"]
                    or command["gate"] != saved["gate"]
                    or exit_row["returncode"] != 0 or exit_row["contamination"]):
                raise ValueError("command, binary, seed or successful-exit binding changed")
            expected_args = [saved["arm"], str(expected_seed), str(saved["max_new"]),
                             "32768", "0" if saved["gate"] else "0.7"]
            if command["argv"][5:10] != expected_args:
                raise ValueError("executed request settings differ from the declared policy")
            if command["argv"][10:] != (["gate"] if saved["gate"] else []):
                raise ValueError("correctness path flag or extra execution settings changed")
            if sha(root / "workloads" / entry["file"]) != entry["sha256"]:
                raise ValueError("workload bytes changed")
            if phase == "scored":
                scenario = int(saved["label"].split("-")[0])
                for turn in range(1, 9):
                    key = (family, scenario, turn)
                    prompt_sha = sha(run / f"turn-{turn}.prompt.ids")
                    previous = paired_prompts.setdefault(key, prompt_sha)
                    if previous != prompt_sha:
                        raise ValueError("paired policies received different prompt tokens")
            metrics = metrics_from_log(run.with_name(run.name + ".log").read_text())[saved["arm"]]
            audited = audit_run(run, entry, saved["arm"], saved["seed"],
                                metrics, context_auditor, loop_candidate)
            for key, value in audited.items():
                if saved[key] != value:
                    raise ValueError("retained audit differs from independent raw replay: " + relative)
            for route in rows(run / "routing.tsv"):
                if route["source"] != "prefix":
                    continue
                data = (run / f'turn-{route["turn"]}.prefix.bin').read_bytes()
                result = subprocess.check_output(
                    [str(binary), route["budget"], route["tokens_read"]], input=data,
                ).decode().strip().split("\t")
                expected = [route[key] for key in ("kind", "k", "tokens_read", "decoded_bytes", "inspected_bytes")]
                if result != expected:
                    raise ValueError("compiled forecaster replay disagrees with native prediction")
                prefixes += 1
            replayed.append(relative)
        for family in ("qwen", "gemma"):
            directory = root / "native" / family / "qualification"
            for k in (2, 4):
                same_tapes(directory / "greedy-k3", directory / f"greedy-k{k}")
            same_tapes(directory / "sampled-prefix128", directory / "sampled-replay128")
        result = report(root)
        result["independent_replay"] = {
            "runs": len(replayed), "prefixes": prefixes,
            "raw_token_and_time_audits": "pass", "compiled_forecaster_replay": "pass",
            "reproducer_sha256": sha(Path(__file__)),
        }
        save(out, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "runtime_archive", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--expected-runtime-sha", required=True)
    args = parser.parse_args()
    result = reproduce(args.root, args.runtime_archive, args.expected_runtime_sha, args.out)
    print(json.dumps(result["independent_replay"]))
