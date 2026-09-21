#!/usr/bin/env python3
"""CPU tests of the actual generic walker and scheduler, without CUDA stubs.

This compiles only those two std-only source modules. It is not a server build,
adapter qualification, or GPU performance test. Cargo tests exercise the same
modules as part of their owning crates on a supported build host.
"""

from pathlib import Path
import json
import subprocess
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="memra-prime-fairness-") as owned:
        scratch = Path(owned)
        harness = scratch / "tests.rs"
        modules = {
            "prime_walker": root / "crates/memra-engine/src/prime_walker.rs",
            "prime_fairness": root / "crates/memra-server/src/prime_fairness.rs",
        }
        harness.write_text(
            "extern crate self as memra_engine;\n"
            + "".join(
                f"#[path = {json.dumps(str(path), ensure_ascii=False)}]\npub mod {name};\n"
                for name, path in modules.items()
            )
        )
        binary = scratch / "tests"
        subprocess.run(
            ["rustc", "--edition", "2024", "--test", str(harness), "-o", str(binary)],
            check=True,
        )
        return subprocess.run([str(binary)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
