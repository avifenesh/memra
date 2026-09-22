"""Validate native raw records and reproduce the request-routing comparison."""
import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile

from fit_profile import fit_profile

PARENT_SHA = "943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8"
MODELS = {
    "qwen": ("Qwen3.8-27B NVFP4+Q5_K, embedded MTP", 7),
    "gemma": ("Gemma 4 12B QAT Q4_0, Q8_0 assistant", 5),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


@contextlib.contextmanager
def auditors(parent_archive):
    if digest(parent_archive) != PARENT_SHA:
        raise ValueError("unrecognized parent archive; no code loaded")
    with tempfile.TemporaryDirectory(prefix="prompt-depth-auditors-") as temporary:
        modules = {}
        with tarfile.open(parent_archive) as archive:
            for name in ("audit_reuse", "audit_context", "loop_audit"):
                member = archive.getmember(f"research/mtp-context-depth-20260921/{name}.py")
                if not member.isfile() or member.size > 100000:
                    raise ValueError("invalid pinned auditor member")
                path = Path(temporary) / f"{name}.py"
                path.write_bytes(archive.extractfile(member).read())
                spec = importlib.util.spec_from_file_location("prompt_" + name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[name] = module
        yield modules


def verify_request_predictions(root, harness_archive, expected_harness_sha, source, workloads):
    if digest(harness_archive) != expected_harness_sha or source["harness_source_sha256"] != expected_harness_sha:
        raise ValueError("unrecognized routing source; no classifier compiled")
    with tempfile.TemporaryDirectory(prefix="prompt-depth-classifier-") as directory:
        destination = Path(directory)
        with tarfile.open(harness_archive) as archive:
            for name in ("main.rs", "router.rs"):
                member = archive.getmember(name)
                if not member.isfile() or member.size > 100000:
                    raise ValueError("unsafe classifier source")
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != source["harness_files"][name]:
                    raise ValueError("classifier source identity changed")
                (destination / name).write_bytes(data)
        binary = destination / "router"
        subprocess.run(
            ["rustc", "--edition=2024", "-O", str(destination / "main.rs"), "-o", str(binary)],
            check=True, capture_output=True, text=True,
        )
        for pools in workloads["families"].values():
            for entry in pools.values():
                path = root / "workloads" / entry["file"]
                if digest(path) != entry["sha256"]:
                    raise ValueError("frozen workload bytes changed")
                instructions = [s.strip() for s in path.read_text().split("\n---TURN---\n")]
                if len(instructions) != 8:
                    raise ValueError("workload turn count changed")
                got = [
                    json.loads(subprocess.check_output(
                        [str(binary), "classify", "1,1,1,1", "8"], input=user.encode()
                    ))["kind"] for user in instructions
                ]
                if got != entry["predicted_kinds"]:
                    raise ValueError("pre-generation classes differ from the exact classifier")


def inspect(root, parent_archive, harness_archive, expected_harness_sha):
    native = root / "native"
    freeze = json.loads((native / "FREEZE.json").read_text())
    state = json.loads((native / "status.json").read_text())
    if state["status"] != "completed":
        raise ValueError("native sequence is incomplete or failed")
    expected_orders = []
    for index in range(6):
        order = ["native", "calibrated", "context", "routed"]
        offset = index // 2
        order = order[offset:] + order[:offset]
        expected_orders.append(order[::-1] if index % 2 else order)
    expected_seeds = {
        "qwen": {"calibration": [20311000 + i for i in range(3)],
                 "heldout": [20312000 + i for i in range(6)], "qualification": 20313000},
        "gemma": {"calibration": [20321000 + i for i in range(3)],
                  "heldout": [20322000 + i for i in range(6)], "qualification": 20323000},
    }
    if (freeze["calibration_pools"] != ["cal-a", "cal-b", "cal-c"]
            or freeze["heldout_pools"] != [f"held-{c}" for c in "abcdef"]
            or freeze["orders"] != expected_orders
            or freeze["seeds"] != expected_seeds
            or freeze["max_new"] != 512 or freeze["ctx"] != 49152
            or freeze["temperature"] != 0.7):
        raise ValueError("registered native comparison changed")
    if (len(state["completed_groups"]) != 24
            or len({g["receipt_dir"] for g in state["completed_groups"]}) != 24):
        raise ValueError("native completion coverage changed")
    workloads = json.loads((root / "workloads/manifest.json").read_text())
    if digest(root / "workloads/manifest.json") != freeze["workload_manifest_sha256"]:
        raise ValueError("workload freeze changed")
    source = json.loads((root / "source.json").read_text())
    if source["source_recipe_commit"] != freeze["recipe_commit"]:
        raise ValueError("native source recipe changed")
    verify_request_predictions(root, harness_archive, expected_harness_sha, source, workloads)
    output = {"models": {}, "source": source, "freeze": freeze}
    gpu_identity = set()
    with auditors(parent_archive) as audit:
        for family, (title, cap) in MODELS.items():
            pools = workloads["families"][family]
            records = {}

            def group(phase, index, pool, seed, expected_arms, temperature, max_new):
                directory = native / family / phase / f"{index:02}-{pool}"
                summary = json.loads(directory.with_name(directory.name + "-audit.json").read_text())
                exit_receipt = json.loads(directory.with_name(directory.name + "-exit.json").read_text())
                identity = json.loads((directory / "identity.json").read_text())
                if exit_receipt["returncode"] != 0 or summary["seed"] != seed:
                    raise ValueError("run failure or seed mismatch")
                if [r["arm"] for r in summary["records"]] != expected_arms:
                    raise ValueError("missing, extra, or reordered policy")
                if identity["source_recipe_commit"] != freeze["recipe_commit"]:
                    raise ValueError("binary recipe differs between arms")
                if identity["runtime_binding_sha256"] != freeze["runtime_binding_sha256"]:
                    raise ValueError("prepared runtime binding differs between arms")
                if identity["workload_sha256"] != pools[pool]["sha256"]:
                    raise ValueError("runner consumed another workload")
                binary = "mtp-depth-study" if family == "qwen" else "gemma-depth-study"
                if identity["binary_sha256"] != source["binaries"][binary]:
                    raise ValueError("executed binary does not match the bound artifact")
                gpu_identity.add(identity["gpu"])
                found_loops = []
                for record in summary["records"]:
                    arm = record["arm"]
                    run = directory / f"{seed}-{arm}"
                    command = json.loads((directory / f"{seed}-{arm}.command.json").read_text())
                    runtime_arm = identity["request_depths"].get(
                        arm, f'fixed:{identity["fixed_depths"][arm]}'
                        if arm in identity["fixed_depths"] else arm
                    )
                    if command[5] != runtime_arm:
                        raise ValueError("executed policy differs from its frozen mapping")
                    if (int(command[6]) != seed or int(command[7]) != max_new
                            or int(command[8]) != 49152 or float(command[9]) != temperature):
                        raise ValueError("request shape differs from the registered comparison")
                    if temperature == 0 and "gate" not in command:
                        raise ValueError("greedy reference gate was not requested")
                    turns = read_rows(run / "turns.tsv")
                    if len(turns) != 8:
                        raise ValueError("incomplete native conversation")
                    total_tokens = total_seconds = 0
                    for index, row in enumerate(turns, 1):
                        if int(row["turn"]) != index:
                            raise ValueError("turn order changed")
                        ids = [int(v) for v in (run / f"turn-{index}.output.ids").read_text().split()]
                        seconds = float(row["elapsed_s"])
                        if (len(ids) != int(row["output_tokens"]) or not ids
                                or not math.isfinite(seconds) or seconds <= 0):
                            raise ValueError("invalid complete-request denominator or token count")
                        if digest(run / f"turn-{index}.user.txt") != pools[pool]["instruction_sha256"][index - 1]:
                            raise ValueError("measured instruction differs from its freeze")
                        flag = audit["loop_audit"].loop_candidate(ids)
                        if flag:
                            found_loops.append({"arm": arm, "turn": index, **flag})
                        total_tokens += len(ids)
                        total_seconds += seconds
                    if (record["tokens"] != total_tokens
                            or not math.isclose(record["elapsed_s"], total_seconds, abs_tol=1e-6)
                            or not math.isclose(record["e2e_tok_s"], total_tokens / total_seconds, rel_tol=1e-7)):
                        raise ValueError("summary does not reproduce from complete-request records")
                    if sum(int(t["drafted"]) for t in turns) <= 0:
                        raise ValueError("speculative path did not engage")
                    reuse = audit["audit_reuse"].audit_reuse(run)
                    if reuse != record["reuse"]:
                        raise ValueError("native reuse receipt differs from token-prefix reconstruction")
                    priors = directory / "context-priors.tsv" if arm == "context" else None
                    if priors and (digest(priors) != digest(native / f"{family}-priors.tsv")
                                   or identity["context_priors_sha256"] != digest(priors)):
                        raise ValueError("span control used a different calibration")
                    span_audit = audit["audit_context"].audit_run(run, arm, cap, priors)
                    if span_audit != record["context_audit"]:
                        raise ValueError("span observations differ from token-byte reconstruction")
                    routing = read_rows(run / "routing.tsv")
                    if len(routing) != 8:
                        raise ValueError("routing receipt coverage changed")
                    for turn, route in enumerate(routing, 1):
                        if int(route["turn"]) != turn:
                            raise ValueError("routing sequence changed")
                        if runtime_arm.startswith("prompt:"):
                            if route["source"] != "prompt":
                                raise ValueError("prompt classifier was not invoked")
                        elif runtime_arm.startswith("schedule:"):
                            schedule = [int(k) for k in runtime_arm.split(":", 1)[1].split(",")]
                            if route["source"] != "schedule" or int(route["k"]) != schedule[turn - 1]:
                                raise ValueError("fixed replay schedule was not applied")
                        elif route["source"] != "control" or route["k"] != "-" or int(route["routing_ns"]) != 0:
                            raise ValueError("a control unexpectedly used prompt routing")
                    records[(phase, pool, arm)] = {
                        "record": record, "turns": turns, "routing": routing,
                        "run": run, "command": command, "spans": read_rows(run / "spans.tsv"),
                    }
                if found_loops != summary["loop_flags"] or bool(found_loops) != summary["excluded"]:
                    raise ValueError("matched-set exclusion does not reproduce")
                return summary

            qualification = group(
                "qualification", 0, "qualification", freeze["seeds"][family]["qualification"],
                ["native", *[f"k{k}" for k in range(1, cap + 1)]], 0, 128,
            )
            for record in qualification["records"][1:]:
                for turn in range(1, 9):
                    for tape in ("prompt", "output"):
                        base = native / qualification["receipt_dir"]
                        left = base / f'{qualification["seed"]}-native/turn-{turn}.{tape}.ids'
                        right = base / f'{qualification["seed"]}-{record["arm"]}/turn-{turn}.{tape}.ids'
                        if left.read_bytes() != right.read_bytes():
                            raise ValueError("fixed-depth greedy identity failed")
            calibration = []
            for index, pool in enumerate(freeze["calibration_pools"]):
                order = [f"k{k}" for k in range(1, cap + 1)]
                order = order[index:] + order[:index]
                if index % 2:
                    order.reverse()
                calibration.append(group(
                    "calibration", index, pool, freeze["seeds"][family]["calibration"][index],
                    order, 0.7, 512,
                ))
            clean = [g for g in calibration if not g["excluded"]]
            if len(clean) < 2:
                raise ValueError("insufficient independent calibration")
            rates = {
                k: sum(next(r for r in g["records"] if r["arm"] == f"k{k}")["tokens"] for g in clean)
                / sum(next(r for r in g["records"] if r["arm"] == f"k{k}")["elapsed_s"] for g in clean)
                for k in range(1, cap + 1)
            }
            global_k = min(rates, key=lambda k: (-rates[k], k))
            trials = {}
            span_totals = collections.defaultdict(lambda: [0, 0, 0])
            for kind in ("prose", "code", "numeric"):
                trials[kind] = {}
                for k in range(1, cap + 1):
                    values = []
                    for g in clean:
                        cell = records[("calibration", g["pool"], f"k{k}")]
                        predicted = pools[g["pool"]]["predicted_kinds"]
                        chosen = [r for r in cell["turns"] if predicted[int(r["turn"]) - 1] == kind]
                        values.append((sum(int(r["output_tokens"]) for r in chosen),
                                       sum(float(r["elapsed_s"]) for r in chosen)))
                    trials[kind][k] = values
            profile, fitting = fit_profile(global_k, trials, [g["seed"] for g in clean])
            recorded_fit = json.loads((native / f"{family}-FIT.json").read_text())
            if recorded_fit["global_k"] != global_k or recorded_fit["profile"] != profile:
                raise ValueError("profile does not reproduce from matched request timings")
            if recorded_fit["prompt_fit"] != json.loads(json.dumps(fitting)):
                raise ValueError("profile support/cost evidence changed")
            for g in clean:
                for k in range(1, cap + 1):
                    observations = records[("calibration", g["pool"], f"k{k}")]["record"]["context_audit"]["observations"]
                    for label, values in observations.items():
                        kind, observed = label.split(":")
                        if int(observed) != k:
                            raise ValueError("fixed calibration changed its depth")
                        for name in (kind, "all"):
                            for j, value in enumerate(values):
                                span_totals[(name, k)][j] += value
            expected_prior = f"global_k\t{global_k}\ncontext\tk\trounds\ttokens\telapsed_ns\n"
            for kind in ("all", "prose", "code", "numeric"):
                for k in range(1, cap + 1):
                    n, tokens, ns = span_totals[kind, k]
                    expected_prior += f"{kind}\t{k}\t{n}\t{tokens}\t{ns}\n"
            if (native / f"{family}-priors.tsv").read_text() != expected_prior:
                raise ValueError("span and request policies did not use the same calibration budget")

            for index, order, temperature in [(1, ["native", "routed"], 0), (2, ["routed", "replay"], 0.7)]:
                q = group("qualification", index, "qualification",
                          freeze["seeds"][family]["qualification"] + index, order, temperature, 128)
                base = native / q["receipt_dir"]
                for turn in range(1, 9):
                    for tape in ("prompt", "output"):
                        paths = [base / f'{q["seed"]}-{arm}/turn-{turn}.{tape}.ids' for arm in order]
                        if paths[0].read_bytes() != paths[1].read_bytes():
                            raise ValueError("prompt routing failed its token-identity comparison")

            heldout = [
                group("heldout", i, pool, freeze["seeds"][family]["heldout"][i],
                      freeze["orders"][i], 0.7, 512)
                for i, pool in enumerate(freeze["heldout_pools"])
            ]
            if json.loads((native / f"{family}-HELDOUT.json").read_text()) != heldout:
                raise ValueError("held-out summary omitted or changed recorded sets")
            clean_held = [g for g in heldout if not g["excluded"]]
            arms = ("native", "calibrated", "context", "routed")
            pooled = {
                arm: sum(next(r for r in g["records"] if r["arm"] == arm)["tokens"] for g in clean_held)
                / sum(next(r for r in g["records"] if r["arm"] == arm)["elapsed_s"] for g in clean_held)
                for arm in arms
            } if clean_held else {}
            pairs = {}
            for control in arms[:-1]:
                gains = []
                for g in clean_held:
                    r = {r["arm"]: r for r in g["records"]}
                    gains.append(100 * (r["routed"]["e2e_tok_s"] / r[control]["e2e_tok_s"] - 1))
                pairs[control] = {
                    "paired_percent": gains, "wins": sum(v > 0 for v in gains),
                    "pooled_percent": 100 * (pooled["routed"] / pooled[control] - 1) if pooled else None,
                }
            routing_ns = 0
            kind_counts = collections.Counter()
            per_kind = {}
            observed_spans = collections.defaultdict(collections.Counter)
            for kind in ("prose", "code", "numeric", "mixed", "unknown"):
                per_kind[kind] = {}
                for arm in arms:
                    tokens = seconds = turns = 0
                    for g in clean_held:
                        cell = records[("heldout", g["pool"], arm)]
                        predicted = pools[g["pool"]]["predicted_kinds"]
                        for row in cell["turns"]:
                            if predicted[int(row["turn"]) - 1] == kind:
                                tokens += int(row["output_tokens"])
                                seconds += float(row["elapsed_s"])
                                turns += 1
                    per_kind[kind][arm] = {
                        "turns": turns, "tokens": tokens, "seconds": seconds,
                        "tok_s": tokens / seconds if seconds else None,
                    }
            for g in heldout:
                cell = records[("heldout", g["pool"], "routed")]
                for turn, route in enumerate(cell["routing"], 1):
                    expected_kind = pools[g["pool"]]["predicted_kinds"][turn - 1]
                    expected_k = profile.get(expected_kind, global_k)
                    if route["source"] != "prompt" or route["kind"] != expected_kind or int(route["k"]) != expected_k:
                        raise ValueError("served request K differs from frozen classification/profile")
                    if any(int(s["k"]) != expected_k for s in cell["spans"]
                           if int(s["turn"]) == turn and s["eligible"] == "true"):
                        raise ValueError("full-round K changed inside a routed request")
                    routing_ns += int(route["routing_ns"])
                    kind_counts[expected_kind] += 1
                    if not g["excluded"]:
                        for span in cell["spans"]:
                            if int(span["turn"]) == turn and span["eligible"] == "true":
                                observed_spans[expected_kind][span["context_before"]] += (
                                    int(span["output_end"]) - int(span["output_start"])
                                )
            output["models"][family] = {
                "model": title, "global_k": global_k, "profile": profile,
                "calibration_sets": len(calibration), "clean_calibration_sets": len(clean),
                "heldout_sets": len(heldout), "clean_heldout_sets": len(clean_held),
                "performance_verdict_eligible": len(clean_held) >= 5,
                "pooled_e2e_tok_s": pooled, "comparisons": pairs,
                "routing_cpu_ns": routing_ns, "request_kind_counts": dict(kind_counts),
                "per_request_kind": per_kind,
                "observed_full_round_tokens": {k: dict(v) for k, v in observed_spans.items()},
                "excluded_heldout": [{"pool": g["pool"], "flags": g["loop_flags"]}
                                     for g in heldout if g["excluded"]],
            }
    if len(gpu_identity) != 1:
        raise ValueError("hardware identity changed during the comparison")
    output["gpu_identity"] = next(iter(gpu_identity))
    return output


def markdown(result):
    lines = [
        "# Native request-depth result", "",
        "One RTX 5090, full-vocabulary Qwen and Gemma MTP, eight-turn native",
        "prompt-checkpoint reuse, approximately 16K initial input, 512-token output",
        "caps, temperature 0.7, top-k 20 and top-p 0.95. Output includes default",
        "reasoning. Rates use complete native request time, including routing and",
        "suffix prefill; this is not HTTP, task-quality or production qualification.", "",
        "| Model | Global K | Prose/code/numeric/fallback K | Native tok/s | Fixed tok/s | Span tok/s | Prompt tok/s | Clean pairs |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for m in result["models"].values():
        p = m["profile"]
        values = "/".join(str(p[k]) for k in ("prose", "code", "numeric", "fallback"))
        r = m["pooled_e2e_tok_s"]
        rates = [f'{r[k]:.3f}' if r else "not scored" for k in ("native", "calibrated", "context", "routed")]
        lines.append(f"| {m['model']} | {m['global_k']} | {values} | {' | '.join(rates)} | {m['clean_heldout_sets']}/{m['heldout_sets']} |")
    lines += ["", "All policies used the same calibration runs. A request-class change",
              "needed support from at least two independent calibration conversations.",
              "Each excluded set retains every raw arm and its independently reproduced loop flag.", ""]
    for m in result["models"].values():
        lines.append(m["model"] + ":")
        for control, values in m["comparisons"].items():
            if values["pooled_percent"] is not None:
                lines.append(f"- Prompt versus {control}: {values['pooled_percent']:+.3f}%; "
                             f"{values['wins']}/{m['clean_heldout_sets']} paired wins.")
        lines.append(f"- Routed-call CPU time: {m['routing_cpu_ns']/1e6:.3f} ms total over 48 requested turns.")
        lines.append(f"- Performance-decision minimum met: {str(m['performance_verdict_eligible']).lower()}.")
        lines.append("")
        lines += [
            "| Requested kind | Scored turns per arm | Fixed tok/s | Span tok/s | Prompt tok/s |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, cells in m["per_request_kind"].items():
            if not cells["routed"]["turns"]:
                continue
            rates = [f'{cells[k]["tok_s"]:.3f}' for k in ("calibrated", "context", "routed")]
            lines.append(f"| {kind} | {cells['routed']['turns']} | {' | '.join(rates)} |")
        lines += [
            "",
            "The JSON report also groups nonterminal round tokens by the span rule's",
            "prose/code/numeric label at round start. This diagnoses within-answer",
            "mixing; it is not a semantic-quality score or a complete token census.",
            "",
        ]
    lines += [
        "Greedy fixed-depth/target gates, prompt/native greedy identity, prompt/schedule",
        "sampled identity, complete token/time accounting, per-request K engagement and",
        "native checkpoint reuse are reconstructed from raw records.",
        "",
        f"Native binary recipe: `{result['source']['source_recipe_commit']}`.",
        f"Run orchestration: `{result['source']['harness_commit']}`.",
        "Exact binary, source, input, timing and exclusion records accompany this report.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parent-archive", type=Path, required=True)
    parser.add_argument("--harness-archive", type=Path, required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = inspect(args.records, args.parent_archive, args.harness_archive, args.harness_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    text = markdown(result)
    (args.out / "RESULTS.md").write_text(text)
    if args.check and args.check.read_text() != text:
        raise SystemExit("native report differs from raw evidence")
    print(text)
