"""Remote CPU qualification and raw timing for the dependency-free router."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    status = {"status": "running", "commands": commands}

    def command(name, argv, stdin=None):
        started = time.perf_counter_ns()
        with (out / f"{name}.stdout").open("wb") as stdout:
            with (out / f"{name}.stderr").open("wb") as stderr:
                result = subprocess.run(
                    argv, cwd=HERE, input=stdin, stdout=stdout, stderr=stderr,
                    timeout=180, check=False,
                )
        commands.append({
            "name": name, "argv": [str(v) for v in argv],
            "returncode": result.returncode,
            "wall_ns": time.perf_counter_ns() - started,
        })
        if result.returncode:
            print((out / f"{name}.stdout").read_text(errors="replace"))
            print((out / f"{name}.stderr").read_text(errors="replace"))
            raise RuntimeError(f"{name} failed with exit {result.returncode}")
        return (out / f"{name}.stdout").read_text()

    try:
        metadata = {
            "schema": 1,
            "scope": "CPU classification and depth selection; no model decoding",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip(),
            "source_hashes": {
                name: sha(HERE / name)
                for name in ("router.rs", "main.rs", "request_routing.rs", "cases.tsv", "run_cpu.py")
            },
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_model": next(
                (line.split(":", 1)[1].strip()
                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")),
                "unreported",
            ),
            "profile": [2, 4, 4, 4],
            "ceiling": 8,
            "batches": 5,
            "iterations_per_group_per_batch": 20000,
            "input_byte_limit": 262144,
            "timer": "std::time::Instant; timer-inclusive, in-process complete routing call",
        }
        metadata["rustc"] = command("rustc-version", ["rustc", "-Vv"])
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        command("format", ["rustfmt", "--edition", "2024", "--check", "main.rs", "router.rs"])
        compiler = ["rustc", "--edition=2024", "-D", "warnings", "-C", "opt-level=3"]
        command("build-tests", [*compiler, "--test", "main.rs", "-o", str(out / "router-tests")])
        command("tests", [str(out / "router-tests"), "--nocapture"])
        command("build", [*compiler, "main.rs", "-o", str(out / "prompt-depth-router")])
        binary = out / "prompt-depth-router"
        metadata["binary_sha256"] = sha(binary)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        # Exercise actual stdin/profile parsing and JSON output as a separate
        # integration check. This is not the in-process latency measurement.
        startup = []
        examples = [
            ("Explain this code.", "prose", 2),
            ("Write a Python function.", "code", 4),
            ("Calculate 7 * 8.", "numeric", 4),
            ("Write code and explain it.", "mixed", 4),
            ("Continue.", "unknown", 4),
        ]
        for repeat in range(4):
            for index, (prompt, kind, k) in enumerate(examples):
                value = json.loads(command(
                    f"cli-{repeat}-{index}",
                    [str(binary), "classify", "2,4,4,4", "8"], prompt.encode(),
                ))
                if value != {"kind": kind, "k": k}:
                    raise RuntimeError(f"CLI case {index} returned {value}")
                startup.append(commands[-1]["wall_ns"])
        oversized = json.loads(command(
            "cli-oversized", [str(binary), "classify", "2,4,4,4", "8"],
            ("Explain " + "x" * 262144).encode(),
        ))
        if oversized != {"kind": "unknown", "k": 4}:
            raise RuntimeError("oversized CLI input did not fall back")
        (out / "process-startup.json").write_text(json.dumps({
            "scope": "process launch plus stdin, routing, stdout, and Python orchestration",
            "samples_ns": startup,
        }, indent=2) + "\n")

        all_rows = []
        for batch in range(5):
            text = command(
                f"bench-{batch}",
                [str(binary), "bench", str(HERE / "cases.tsv"), "2,4,4,4", "8", "20000"],
            )
            rows = [json.loads(line) for line in text.splitlines()]
            if len(rows) != 7 or any(len(r["samples_ns"]) != 20000 for r in rows):
                raise RuntimeError("incomplete timing batch")
            for row in rows:
                all_rows.append({"batch": batch, **row})
                print(json.dumps({k: v for k, v in row.items() if k != "samples_ns"}))
        (out / "timings.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_rows)
        )
        status["status"] = "passed"
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
