"""Verify the exact research binary's sampled C source ordering."""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
SOURCE_SHA = "7765982aacad20867b406029b945cdec9f60e5694e0e9489731e3ffc8ce24d96"
SOURCE_MANIFEST_SHA = "d74059bfd75c0e5c9a5c03001d28b6cf8a623077fd6f9767bcfcd2180c891826"
SOURCE_FILE = "crates/memra-engine/src/spec.rs"
BRANCHES = (
    ("sampled-chain-graph", 12890, 12917),
    ("sampled-single-head-graph", 13116, 13144),
    ("sampled-eager", 13316, 13367),
)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def source_lines(path):
    with tarfile.open(path, "r:gz") as source:
        text = source.extractfile(SOURCE_FILE).read().decode()
    return text.splitlines()


def source_order(lines):
    observed = []
    for name, first, last in BRANCHES:
        area = lines[first - 1:last]
        append = [
            first + index for index, line in enumerate(area)
            if "draft.push(d);" in line
        ]
        learned = [
            first + index for index, line in enumerate(area)
            if "learner.confidence_stop(j," in line
        ]
        stop = [
            first + index for index, line in enumerate(area)
            if "if learned_stop ||" in line
        ]
        if (
            len(append) != 1
            or len(learned) != 1
            or len(stop) != 1
            or not append[0] < learned[0] < stop[0]
        ):
            raise ValueError(f"sampled C source order differs: {name}")
        observed.append({
            "branch": name,
            "append_line": append[0],
            "learned_stop_line": learned[0],
            "break_decision_line": stop[0],
        })
    guard = "\n".join(lines[12066:12075])
    if (
        "if sampled && pmin0 && p_min > 0.0" not in guard
        or "sampled PMIN0 needs a pre-draw decision" not in guard
    ):
        raise ValueError("sampled PMIN0 source guard differs")
    return observed


def two_token_example():
    q = {"A": Decimal("0.8"), "B": Decimal("0.2")}
    p = dict(q)
    discarded = q["B"]
    biased = {
        "A": q["A"] + discarded * p["A"],
        "B": discarded * p["B"],
    }
    retained = dict(q)
    if biased != {"A": Decimal("0.96"), "B": Decimal("0.04")}:
        raise ValueError("two-token discard example changed")
    if retained != p:
        raise ValueError("retain-before-stop example changed")
    return {
        "target": {token: str(value) for token, value in p.items()},
        "discard_before_verify": {
            token: str(value) for token, value in biased.items()
        },
        "retain_then_verify": {
            token: str(value) for token, value in retained.items()
        },
    }


def verify(binary, source_archive, source_manifest):
    metadata = json.loads(source_manifest.read_text())
    if (
        sha(binary) != BINARY_SHA
        or sha(source_archive) != SOURCE_SHA
        or sha(source_manifest) != SOURCE_MANIFEST_SHA
        or metadata["binary_sha256"] != BINARY_SHA
        or metadata["source_archive_sha256"] != SOURCE_SHA
        or metadata["model_sha256"] != MODEL_SHA
    ):
        raise ValueError("sampled C proof does not match binary/source pair")
    observed = source_order(source_lines(source_archive))
    return {
        "schema": 1,
        "status": "exact-source-retains-sampled-pick-before-C-stop",
        "binary_sha256": BINARY_SHA,
        "source_archive_sha256": SOURCE_SHA,
        "model_sha256": MODEL_SHA,
        "source_file": SOURCE_FILE,
        "sampled_branches": observed,
        "positive_sampled_pmin0_refused": True,
        "two_token_counterexample": two_token_example(),
        "limit": "Source-order proof only; full sampled Qwen distribution and serving gates remain separate.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.binary, args.source, args.manifest)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "status": result["status"],
        "sampled_branches": len(result["sampled_branches"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
