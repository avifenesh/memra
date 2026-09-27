"""Fit and evaluate a shared output-format tree on verified archived records."""

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

from model import (
    FEATURES, KINDS, Policy, State, export_rust, fit_tree, format_map,
    future_kind, predict, simple_rule,
)

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
sys.path.insert(0, str(LANE))
from reproduce_native import unpack  # noqa: E402

PIN = "2c6558fbc158ff936ea1d47c54c66659fd884982ea6afd7222a41cb8a842352d"
HORIZON = 8
K_FOR_KIND = (2, 4, 4, 3)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_tsv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compressed_rows(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as packed:
            with io.TextIOWrapper(packed, encoding="utf8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def collect(root):
    workloads = json.loads((root / "workloads/manifest.json").read_text())
    all_rows, packets, audits, turns = [], [], [], []
    for family, offset in (("qwen", 20310000), ("gemma", 20320000)):
        for split, pools, arms, seed_offset in (
            ("calibration", ["cal-a", "cal-b", "cal-c"], ["k2", "k3", "k4"], 1000),
            ("heldout", [f"held-{c}" for c in "abcdef"], ["calibrated"], 2000),
        ):
            for index, pool in enumerate(pools):
                group = f"{family}/{pool}"
                for arm in arms:
                    run = root / "native" / family / split / f"{index:02}-{pool}" / f"{offset + seed_offset + index}-{arm}"
                    pieces = {int(r["id"]): bytes.fromhex(r["hex"]) for r in read_tsv(run / "token-bytes.tsv")}
                    spans = read_tsv(run / "spans.tsv")
                    requested = workloads["families"][family][pool]["predicted_kinds"]
                    for turn in range(1, 9):
                        ids = [int(x) for x in (run / f"turn-{turn}.output.ids").read_text().split()]
                        tokens = [pieces[token] for token in ids]
                        data = b"".join(tokens)
                        labels = format_map(data)
                        offsets = [0]
                        for token in tokens:
                            offsets.append(offsets[-1] + len(token))
                        hint = KINDS.index(requested[turn - 1]) if requested[turn - 1] in KINDS else 3
                        these = [r for r in spans if int(r["turn"]) == turn and r["eligible"] == "true"]
                        if not these or any(int(r["k"]) != (int(arm[1:]) if arm.startswith("k") else 3) for r in these):
                            raise ValueError("forecast input is not the registered fixed-depth control")
                        if split == "calibration":
                            starts = {position: None for position in range(0, max(0, len(ids) - HORIZON + 1), 8)}
                        else:
                            starts = {int(r["output_start"]): r for r in these}
                            if len(starts) != len(these):
                                raise ValueError("duplicate eligible round start")
                        state = State(hint)
                        for position in range(len(ids) + 1):
                            if position in starts:
                                start = offsets[position]
                                end = offsets[min(len(ids), position + HORIZON)]
                                label = future_kind(data, labels, start, end) if position + HORIZON <= len(ids) else 3
                                previous = future_kind(data, labels, offsets[max(0, position - HORIZON)], start)
                                span = starts[position]
                                row = {
                                    "id": f"{group}/{arm}/{turn}/{position}",
                                    "group": group, "family": family, "split": split, "arm": arm,
                                    "turn": turn, "position": position, "label": label,
                                    "previous_label": previous, "prompt": hint,
                                    "features": state.features(),
                                    "transition": position > 0 and previous < 3 and label < 3 and previous != label,
                                    "terminal_window": position + HORIZON > len(ids),
                                }
                                if span is not None:
                                    row["persistence"] = KINDS.index(span["context_before"])
                                all_rows.append(row)
                                # Deterministic low-rate samples, with future text kept only
                                # in the separate human-inspection record.
                                sample = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
                                if sample % 97 == 0:
                                    prefix = data[:start]
                                    packets.append({
                                        "id": row["id"], "hint": hint,
                                        "base": prefix[:-16].hex(), "append": prefix[-16:].hex(),
                                        "features": row["features"],
                                    })
                                    audits.append({
                                        "id": row["id"], "label": KINDS[label],
                                        "prefix_hex": data[max(0, start - 128):start].hex(),
                                        "future_hex": data[start:end].hex(),
                                    })
                            if position < len(ids):
                                state.observe(tokens[position])
                        turns.append({"group": group, "arm": arm, "turn": turn, "tokens": len(ids)})
    train = {row["group"].split("/", 1)[1] for row in all_rows if row["split"] == "calibration"}
    held = {row["group"].split("/", 1)[1] for row in all_rows if row["split"] == "heldout"}
    if train & held or len(train) != 3 or len(held) != 6:
        raise ValueError("conversation split isolation changed")
    if not packets:
        raise ValueError("no runtime agreement samples")
    return all_rows, packets, audits, turns


def metrics(rows, guesses):
    known = [(row, guessed) for row, guessed in zip(rows, guesses) if row["label"] < 3]
    confusion = [[0] * 4 for _ in range(3)]
    for row, guessed in known:
        confusion[row["label"]][guessed] += 1
    classes = {}
    for kind in range(3):
        tp = confusion[kind][kind]
        support = sum(confusion[kind])
        predicted = sum(confusion[actual][kind] for actual in range(3))
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        classes[KINDS[kind]] = {
            "support": support, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        }
    supported = [row["f1"] for row in classes.values() if row["support"]]
    return {
        "rows": len(rows), "labelled_rows": len(known),
        "ambiguous_rows": len(rows) - len(known),
        "accuracy": sum(confusion[k][k] for k in range(3)) / len(known) if known else None,
        "coverage": sum(g < 3 for _, g in known) / len(known) if known else None,
        "macro_f1": statistics.mean(supported) if supported else None,
        "mapping_agreement": sum(K_FOR_KIND[r["label"]] == K_FOR_KIND[g] for r, g in known) / len(known) if known else None,
        "classes": classes, "confusion": confusion,
    }


def evaluate(rows, tree, transfer):
    result, predictions = {}, []
    for family in ("qwen", "gemma"):
        held = [r for r in rows if r["split"] == "heldout" and r["family"] == family]
        guesses = {
            "prompt": [r["prompt"] for r in held],
            "persistence": [r["persistence"] for r in held],
            "simple_rule": [simple_rule(r["features"]) for r in held],
            "shared_tree": [predict(tree, r["features"]) for r in held],
            "transfer_tree": [predict(transfer[family], r["features"]) for r in held],
        }
        result[family] = {}
        for name, values in guesses.items():
            subsets = {
                "all": list(range(len(held))),
                "transitions": [i for i, r in enumerate(held) if r["transition"]],
                "starts": [i for i, r in enumerate(held) if r["position"] == 0],
            }
            result[family][name] = {
                subset: metrics([held[i] for i in indices], [values[i] for i in indices])
                for subset, indices in subsets.items()
            }
            policies = {}
            counts, switches = collections.Counter(), 0
            for row, guess in zip(held, values):
                key = row["group"], row["turn"]
                policy = policies.setdefault(key, Policy())
                before = policy.k
                selected = policy.select(guess)
                switches += before != selected
                counts[selected] += 1
                predictions.append({
                    "id": row["id"], "model": name, "label": row["label"],
                    "prediction": guess, "shadow_k": selected,
                })
            result[family][name]["shadow"] = {"k_counts": dict(counts), "switches": switches}
            result[family][name]["per_conversation"] = {
                group: metrics([r for r in held if r["group"] == group],
                               [g for r, g in zip(held, values) if r["group"] == group])
                for group in sorted({r["group"] for r in held})
            }
    return result, predictions


def native_check(out, tree, packets):
    model_file = out / "tree.rs"
    model_file.write_text(export_rust(tree))
    environment = {**os.environ, "FORECAST_TREE_RS": str(model_file.resolve())}
    binary = out / "forecast-runtime"
    commands = [
        ["rustc", "--edition=2024", "-C", "opt-level=3", str(HERE / "runtime.rs"), "-o", str(binary)],
        ["rustc", "--edition=2024", "--test", str(HERE / "runtime.rs"), "-o", str(out / "runtime-tests")],
        [str(out / "runtime-tests")],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        (out / f"native-{index}.stdout").write_text(result.stdout)
        (out / f"native-{index}.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"native command {index} failed; see retained stdout/stderr")
    packet_file = out / "runtime-inputs.tsv"
    packet_file.write_text("".join(f'{r["hint"]}\t{r["base"]}\t{r["append"]}\n' for r in packets))
    output = subprocess.check_output([str(binary), "check", str(packet_file)], text=True)
    (out / "runtime-check.tsv").write_text(output)
    actual = output.splitlines()
    if len(actual) != len(packets):
        raise ValueError("runtime lost agreement cases")
    for line, row in zip(actual, packets):
        features, guessed = line.split("\t")
        if tuple(map(int, features.split(","))) != tuple(row["features"]) or int(guessed) != predict(tree, row["features"]):
            raise ValueError("Python/Rust feature or prediction mismatch at " + row["id"])
    with (out / "latency.tsv").open("w") as stream:
        subprocess.run([str(binary), "bench", str(packet_file), "100000"], stdout=stream, check=True)
    samples = sorted(int(r["elapsed_ns"]) for r in read_tsv(out / "latency.tsv"))
    if len(samples) != 100000:
        raise ValueError("latency samples are incomplete")
    return {
        "calls": len(samples), "agreement_cases": len(packets), "rust_tests": 2,
        "median_us": statistics.median(samples) / 1000,
        "p95_us": samples[math.ceil(0.95 * (len(samples) - 1))] / 1000,
        "p99_us": samples[math.ceil(0.99 * (len(samples) - 1))] / 1000,
        "max_us": samples[-1] / 1000, "binary_sha256": sha(binary),
        "tree_source_sha256": sha(model_file), "latency_sha256": sha(out / "latency.tsv"),
        "p99_target_met": samples[math.ceil(0.99 * (len(samples) - 1))] < 5000,
    }


def percent(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def markdown(report):
    cpu = report["runtime"]
    lines = [
        "# Upcoming-format prediction: archived-data pilot", "",
        "One shared shallow tree; no LLM or draft-head training. Features use only",
        "the prompt prediction and committed prefix available before a decision.",
        "The target is the machine-defined format of the next eight returned tokens.",
        "These are reused synthetic conversations, not a new generation-speed result.", "",
        "| Model | Predictor | Labelled rows | Macro F1 | Coverage | Transition rows | Transition macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family, methods in report["evaluation"].items():
        for name, cells in methods.items():
            all_rows, changes = cells["all"], cells["transitions"]
            lines.append(
                f"| {family} | {name} | {all_rows['labelled_rows']} | {percent(all_rows['macro_f1'])} | "
                f"{percent(all_rows['coverage'])} | {changes['labelled_rows']} | {percent(changes['macro_f1'])} |"
            )
    lines += [
        "", "Transfer fits use only the other model's calibration conversations.",
        "The deployed candidate is the single pooled tree. Ambiguous future windows",
        "are outside the labelled denominator; abstentions on labelled windows count",
        "against recall. Per-class, per-conversation and request-start results are in",
        "`RESULTS.json`. The included label samples allow inspection of the structural",
        "annotation; these scores are not independent human semantic judgments.", "",
        f"The shared tree contains {report['tree_nodes']} nodes and has maximum depth four.",
        f"Training used {report['training_rows']} known-label prefix examples across six calibration conversations.",
        f"Shared and transfer fitting took {report['fit_seconds']:.3f} CPU wall seconds in total.", "",
        f"Python/Rust agreement passed {cpu['agreement_cases']} sampled prefixes and {cpu['rust_tests']} Rust tests.",
        f"Incremental observation, feature extraction, tree prediction and policy selection:",
        f"{cpu['median_us']:.3f} us median / {cpu['p95_us']:.3f} us p95 / {cpu['p99_us']:.3f} us p99",
        f"over {cpu['calls']:,} calls; maximum {cpu['max_us']:.3f} us.",
        "This includes state-copy and timer overhead, with 1,000 untimed warmup calls.",
        f"CPU: {report['host']['cpu']}; {report['host']['rustc']}.", "",
        "The shadow policy starts at K=3 and maps confident prose to 2, code/numeric",
        "to 4 and abstention to 3, with two confirmations and one-step changes.",
        "Shadow decisions do not establish throughput: changing K changes subsequent",
        "generation. A native paired comparison remains necessary.", "",
        f"Input manifest: `{PIN}`.",
        f"Source commit: `{report['source_commit']}`.",
        f"Shared model SHA-256: `{report['tree_sha256']}`.",
        "Raw features/labels, predictions, sample text windows, models, runtime",
        "agreement, timings, binary and source hashes accompany this report.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="forecast-records-") as temporary:
        root = Path(temporary)
        unpack(LANE / "native-receipts", PIN, root)
        rows, packets, audits, turns = collect(root)
    training = [row for row in rows if row["split"] == "calibration"]
    started = time.perf_counter()
    tree = fit_tree(training)
    transfer = {
        family: fit_tree([row for row in training if row["family"] != family])
        for family in ("qwen", "gemma")
    }
    fit_seconds = time.perf_counter() - started
    write_json(out / "tree.json", tree)
    for family, value in transfer.items():
        write_json(out / f"transfer-to-{family}.json", value)
    evaluation, predictions = evaluate(rows, tree, transfer)
    compressed_rows(out / "features-and-labels.jsonl.gz", rows)
    compressed_rows(out / "predictions.jsonl.gz", predictions)
    compressed_rows(out / "label-audit.jsonl.gz", audits)
    write_json(out / "runtime-inputs.json", packets)
    runtime = native_check(out, tree, packets)
    cpu_info = Path("/proc/cpuinfo").read_text()
    cpu = next((line.split(":", 1)[1].strip() for line in cpu_info.splitlines() if line.startswith("model name")), "unknown")
    report = {
        "schema": 1, "input_manifest_sha256": PIN, "horizon_tokens": HORIZON,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_files": {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in {".py", ".rs", ".md"}},
        "host": {"cpu": cpu, "platform": platform.platform(),
                 "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        "training_rows": sum(row["label"] < 3 for row in training),
        "tree_nodes": len(tree["nodes"]), "tree_sha256": sha(out / "tree.json"),
        "fit_seconds": fit_seconds, "turns": turns, "runtime": runtime, "evaluation": evaluation,
    }
    write_json(out / "RESULTS.json", report)
    (out / "RESULTS.md").write_text(markdown(report))
    inventory = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.iterdir())}
    write_json(out / "manifest.json", {"source_commit": report["source_commit"], "files": inventory})
    print((out / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
