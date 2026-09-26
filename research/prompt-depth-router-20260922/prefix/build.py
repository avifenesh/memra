"""Build prefix-policy tests and native programs on hosted CPU CI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from prepare import build

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from archive_io import write_archive  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    commands = []
    status = {"status": "building", "commands": commands}

    def command(name, argv, cwd=source, env=None):
        started = time.monotonic()
        with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.stderr").open("w") as stderr:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=2700)
        commands.append({"name": name, "argv": list(map(str, argv)),
                         "returncode": result.returncode, "seconds": time.monotonic() - started})
        if result.returncode:
            print((out / f"{name}.stderr").read_text()[-16000:])
            print((out / f"{name}.stdout").read_text()[-8000:])
            raise RuntimeError(name + " failed")

    try:
        receipt = build(args.source_archive, source)
        inputs = {
            str(path.relative_to(source)): {"bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(source.rglob("*")) if path.is_file()
        }
        write_archive(out / "runtime-source.tar.gz", {name: source / name for name in inputs})
        harness = {
            str(path.relative_to(HERE)): path
            for path in sorted(HERE.rglob("*")) if path.is_file() and path.suffix in {".rs", ".py", ".md"}
        }
        write_archive(out / "harness-source.tar.gz", harness)
        metadata = {
            "source_recipe_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
            "prepared_source": receipt, "source_files": inputs,
            "runtime_source_sha256": sha(out / "runtime-source.tar.gz"),
            "harness_source_sha256": sha(out / "harness-source.tar.gz"),
            "harness_files": {name: sha(path) for name, path in harness.items()},
            "scope": "CPU compilation and prefix-bound tests; not GPU qualification",
        }
        environment = {**os.environ, "MEMRA_NVCC": "/usr/local/cuda-13.1/bin/nvcc",
                       "MEMRA_CUDA_ARCH": "120a", "CARGO_BUILD_JOBS": "2"}
        command("rustc", ["rustc", "-Vv"])
        command("nvcc", [environment["MEMRA_NVCC"], "--version"])
        bins = ["--bin", "qwen-prefix-study", "--bin", "gemma-prefix-study"]
        command("build", ["cargo", "build", "--locked", "--release", "-p", "memra-engine", *bins], env=environment)
        command("clippy", ["cargo", "clippy", "--locked", "--release", "-p", "memra-engine",
                          *bins, "--", "-D", "warnings"], env=environment)
        binary_dir = out / "binaries"
        binary_dir.mkdir()
        for name in ["qwen-prefix-study", "gemma-prefix-study"]:
            shutil.copyfile(source / "target/release" / name, binary_dir / name)
            (binary_dir / name).chmod(0o755)
        for name, expected in inputs.items():
            if sha(source / name) != expected["sha256"]:
                raise RuntimeError("build changed a bound source file: " + name)
        metadata["binaries"] = {path.name: sha(path) for path in sorted(binary_dir.iterdir())}
        (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status["status"] = "built-not-gpu-qualified"
        shutil.rmtree(source)
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
