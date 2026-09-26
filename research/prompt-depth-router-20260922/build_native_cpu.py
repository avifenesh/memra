"""Build the pinned native measurement programs on CPU CI before renting a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare_native import prepare
from archive_io import write_archive

HERE = Path(__file__).resolve().parent


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
    receipt = prepare(args.source_archive, source)
    commands = []
    status = {"status": "building", "commands": commands}

    def command(name, argv, cwd=source, env=None):
        started = time.monotonic()
        with (out / f"{name}.stdout").open("w") as stdout:
            with (out / f"{name}.stderr").open("w") as stderr:
                result = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout,
                                        stderr=stderr, timeout=2700)
        commands.append({"name": name, "argv": list(map(str, argv)),
                         "returncode": result.returncode,
                         "seconds": time.monotonic() - started})
        if result.returncode:
            print((out / f"{name}.stderr").read_text()[-16000:])
            raise RuntimeError(f"{name} failed")

    try:
        inputs = {
            str(p.relative_to(source)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(source.rglob("*")) if p.is_file()
        }
        source_tar = out / "runtime-source.tar.gz"
        write_archive(source_tar, {name: source / name for name in inputs})
        harness_files = {
            p.name: sha(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in (".rs", ".py", ".tsv", ".md")
        }
        harness_tar = out / "harness-source.tar.gz"
        write_archive(harness_tar, {name: HERE / name for name in harness_files})
        metadata = {
            "source_recipe_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
            "parent": receipt,
            "runtime_source_sha256": sha(source_tar),
            "source_files": inputs,
            "harness_source_sha256": sha(harness_tar),
            "harness_files": harness_files,
            "compiler_scope": "CPU build; no GPU execution or qualification",
        }
        (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        env = os.environ.copy()
        env.update(MEMRA_NVCC="/usr/local/cuda-13.1/bin/nvcc",
                   MEMRA_CUDA_ARCH="120a", CARGO_BUILD_JOBS="2")
        command("rustc-version", ["rustc", "-Vv"])
        command("nvcc-version", [env["MEMRA_NVCC"], "--version"])
        bins = ["--bin", "mtp-depth-study", "--bin", "gemma-depth-study"]
        command("build", ["cargo", "build", "--locked", "--release", "-p", "memra-engine", *bins], env=env)
        command("clippy", ["cargo", "clippy", "--locked", "--release", "-p", "memra-engine",
                          *bins, "--", "-D", "warnings"], env=env)
        binary_dir = out / "binaries"
        binary_dir.mkdir()
        for name in ["mtp-depth-study", "gemma-depth-study"]:
            shutil.copyfile(source / "target/release" / name, binary_dir / name)
            (binary_dir / name).chmod(0o755)
        command("build-router", ["rustc", "--edition=2024", "-D", "warnings", "-C", "opt-level=3",
                                str(HERE / "main.rs"), "-o", str(binary_dir / "prompt-depth-router")])
        for name, record in inputs.items():
            if sha(source / name) != record["sha256"]:
                raise RuntimeError("build modified a bound source input: " + name)
        metadata["binaries"] = {p.name: sha(p) for p in sorted(binary_dir.iterdir())}
        (out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status["status"] = "built-not-gpu-qualified"
        # The exact source archive and executables are retained; compiler scratch
        # is not an evidence artifact.
        shutil.rmtree(source)
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
