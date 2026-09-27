#!/usr/bin/env python3
"""Offline replay of day-11 reclaim-cycle receipts on two cards.

NOT CUDA execution, NOT a timing comparison between the cards, NOT serving qualification.
Each cell is `--case active --kv-allocator vmm --reclaim-diagnostic --reclaim-cycles 5` in one
process on one cache; the replay checks collector journals, build/source/binary identity, the
frozen continuation surfaces, every cycle's roundtrip arithmetic and probe accounting, and the
series class (lead ruling, day 11). A nonzero residual never becomes G1 PASS here.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURFACES = ("prefix-state.tsv", "final-state.tsv", "tokens.u32le", "logits.tsv",
            "final-logits.f32le", "prompt.u32le", "plan.debug")
CELLS = ("cycles-32768", "cycles-8192")
CYCLES = 5
RIGS = {
    "pro-single": {"raw": "pro-single-day11", "lock": "/tmp/memra-gpu.lock",
                   "card": "one RTX PRO 6000 Blackwell 96 GB",
                   "power": {"power.limit": "600.00 W", "power.max_limit": "600.00 W"},
                   "frozen": "BOX3-BASELINES.json"},
    "rtx5090": {"raw": "rtx5090-day11", "lock": "/tmp/memra-5090.lock",
                "card": "RTX 5090 Laptop GPU (local development rig; no frozen bundle of its own)",
                "power": None, "frozen": None},
}
RENTED = HERE / "rented-5090-20260919"
NONE, ONE_TIME, GROWING, UNCLASSIFIED = ("none", "one-time-driver-mapping-metadata",
                                         "growing-residual", "unclassified")


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"verify-{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


day9 = load("day9")
day10 = load("day10")
require = day9.require
fields = day9.fields


def data(path):
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def ints(mapping):
    return {k: int(v) for k, v in mapping.items() if v.lstrip("-").isdigit()}


# ---- pure series rule (mirrors kv_tier_gate/reclaim_contract.rs::classify_cycles) ----

def observe(before, demoted, restored, released, granule):
    gain = demoted - before
    reacquired = demoted - restored
    residual = released - gain
    exact = released > 0 and gain == released and reacquired == gain
    bounded = (released > 0 and granule > 0 and gain > 0 and gain >= max(0, released - granule)
               and restored == before and reacquired == gain)
    return exact, bounded, residual


def classify_cycles(cycles, granule):
    """cycles: list of dicts free_before/free_after_demote/free_after_restore/released."""
    if granule == 0:
        return "not-applicable-pooled"
    if len(cycles) < 2:
        return UNCLASSIFIED
    obs = [observe(c["free_before"], c["free_after_demote"], c["free_after_restore"], c["released"], granule)
           for c in cycles]
    residuals = [o[2] for o in obs]
    steady = all(o[1] for o in obs) and all(c["free_before"] == cycles[0]["free_before"] for c in cycles)
    if steady and all(r == 0 for r in residuals):
        return NONE
    if steady and all(r == granule for r in residuals):
        return ONE_TIME
    unreturned = [cycles[0]["free_before"] - c["free_after_restore"] for c in cycles]

    def grows(series):
        return all(b >= a for a, b in zip(series, series[1:])) and series[-1] > series[0]
    if grows(residuals) or grows(unreturned):
        return GROWING
    return UNCLASSIFIED


# ---- receipt replay ----

def refused(folder, rig):
    """A cell the gate refused (exit 2, `REFUSED:` last line) before any token ran against a
    suspended cache. Recorded verbatim; it is a non-result for that control, never a verdict."""
    spec = RIGS[rig]
    attempts = sorted(p for p in folder.glob("collector-*") if p.is_dir())
    require(len(attempts) == 1, "refused cell with more than one executed attempt")
    c = attempts[0]
    require(json.loads((c / "lock.json").read_text()) ==
            {"rig": rig, "lock": spec["lock"], "acquired": True}, "wrong rig/lock")
    cap = json.loads((c / "command.capture.json").read_text())
    require(cap["status"] == "refused" and cap["exit_code"] == 2 and not cap["timed_out"], "not a refusal")
    quote = cap["failure_quote"]
    log = (c / "command.log").read_text().splitlines()
    require(quote.startswith("REFUSED: ") and log[-1] == quote, "refusal quote is not the last console line")
    require((folder / "receipt/REFUSED.txt").read_text().strip() == quote, "REFUSED.txt differs from console")
    require(not (folder / "receipt/ACTIVE.txt").exists() and not (folder / "receipt/reclaim-cycles.txt").exists()
            and not any(line.startswith("reclaim-cycle ") for line in log), "refused cell produced a cycle")
    for phase in ("before", "after"):
        require(len((c / f"command.{phase}.log").read_text().splitlines()) == 1, "co-tenant snapshot")
    limits = cap["gpu_power_limits"]
    envelope = {k: limits[0][k] for k in ("power.limit", "power.max_limit")}
    if spec["power"] is not None:
        require(envelope == spec["power"], "power envelope")
    identity = fields(folder / "receipt/identity.txt")
    require(identity["binary_sha256"] == (folder / "binary.sha256").read_text().split()[0], "binary hash")
    context = int(identity["context"])
    return {"card": spec["card"], "power_envelope": envelope,
            "source": (folder / "source.commit").read_text().strip(), "binary_sha256": identity["binary_sha256"],
            "artifact_sha256": identity["artifact_sha256"], "status": "refused", "console": quote,
            "prompt_committed_before_refusal": max((int(l.split("=")[1]) for l in log if l.startswith("baseline prompt committed=")), default=0),
            "verdict": f"ACTIVE-{context // 1024}K reclaim-cycles control not executed: {quote}"}


def collector(folder, rig):
    """Journal, lock, telemetry, power envelope, co-tenancy, command shape. Returns (identity, envelope)."""
    spec = RIGS[rig]
    require((folder / "collector.exit").read_text().strip() == "0", "collector failed")
    attempt = int((folder / "successful-attempt").read_text())
    c = folder / f"collector-{attempt}"
    require(json.loads((c / "lock.json").read_text()) ==
            {"rig": rig, "lock": spec["lock"], "acquired": True}, "wrong rig/lock")
    cap = json.loads((c / "command.capture.json").read_text())
    require(cap["exit_code"] == 0 and not cap["timed_out"] and cap["qualification"] is False
            and cap["status"] == "executed-not-qualified", "unsuccessful/promoted capture")
    limits = cap["gpu_power_limits"]
    require(len(limits) == 1 and limits[0]["device"] == "0", "power envelope census")
    envelope = {k: limits[0][k] for k in ("power.limit", "power.max_limit")}
    if spec["power"] is not None:
        require(envelope == spec["power"], "power envelope")
    cells = [json.loads(x) for x in (c / "CELL.jsonl").read_text().splitlines()]
    require([x["event"] for x in cells] == ["start", "end"], "incomplete journal")
    require(cells[-1]["exit_code"] == 0 and cells[-1]["qualification"] is False, "bad CELL end")
    require(cells[-1]["capture"]["sha256"] == sha(c / "command.capture.json"), "capture hash")
    require(cap["raw_log"]["sha256"] == sha(c / "command.log"), "raw log hash")
    require(cap["gpu_telemetry"]["interval_ms"] == 250, "telemetry interval")
    require(cap["gpu_telemetry"]["raw_csv"]["sha256"] == sha(c / "command.gpu.csv"), "telemetry hash")
    samples = list(csv.DictReader((c / "command.gpu.csv").open()))
    require(samples, "empty telemetry")
    for sample in samples:
        sample = {k.strip(): v.strip() for k, v in sample.items()}
        require(sample["power.limit [W]"] == envelope["power.limit"]
                and sample["power.max_limit [W]"] == envelope["power.max_limit"], "power changed inside cell")
    for phase in ("before", "after"):
        path = c / f"command.{phase}.log"
        require(len(path.read_text().splitlines()) == 1, "co-tenant snapshot")
        require(sha(path) == cap["compute_apps"][phase]["raw_log"]["sha256"], "snapshot hash")
    log = (c / "command.log").read_text()
    require(not any(x in log for x in ("VMM cleanup", "VMM VA cleanup", "CUDA_ERROR_")),
            "native cleanup/runtime error")
    identity = fields(folder / "receipt/identity.txt")
    require(identity["binary_sha256"] == (folder / "binary.sha256").read_text().split()[0], "binary hash")
    source = (folder / "source.commit").read_text().strip()
    build = folder.parent / "build"
    require((build / "source.txt").read_text().strip() == source
            and (build / "binary.sha256").read_text().split()[0] == identity["binary_sha256"]
            and (build / "exit").read_text().strip() == "0", "no matching successful native build")
    command = cap["command"]
    for key, value in (("--case", "active"), ("--kv-allocator", "vmm"), ("--tiers", "host"),
                       ("--reclaim-cycles", str(CYCLES)), ("--context", identity["context"])):
        require(command[command.index(key) + 1] == value, f"command {key}")
    require("--reclaim-diagnostic" in command and "--same-program" in command, "wrong program/diagnostic")
    require(folder.name == f"cycles-{identity['context']}", "cell label/context mismatch")
    # Console lines: one per cycle plus the series line, in order, no stray G1 verdict.
    cycle_lines = [line for line in log.splitlines() if line.startswith("reclaim-cycle ")]
    require([line.split()[1].rstrip(":") for line in cycle_lines] ==
            [f"{k}/{CYCLES}" for k in range(1, CYCLES + 1)], "console cycle lines")
    series_lines = [line for line in log.splitlines() if line.startswith("RECLAIM-CYCLES: ")]
    require(len(series_lines) == 1 and "G1 PASS" not in log.replace("not G1 PASS", ""), "console series line")
    return identity, envelope, series_lines[0]


def continuation(folder, identity, rig):
    """Frozen-bundle comparison. PRO: must match BOX3 bundle. 5090 laptop: same artifact/plan/prompt
    as the rented RTX 5090 bundle; decoded surfaces compared and REPORTED, never assumed."""
    r = folder / "receipt"
    context = identity["context"]
    done = fields(r / "ACTIVE.txt")
    require(done["committed"] == context and done["active_engaged"] == "true", "unfinished context")
    require((r / "ACTIVE.txt").read_text().splitlines()[0].endswith("not G1 PASS"), "status line claims G1")
    for key, name in (("prefix_state_manifest_sha256", "prefix-state.tsv"),
                      ("final_state_manifest_sha256", "final-state.tsv"),
                      ("tokens_sha256", "tokens.u32le"), ("logit_rows_sha256", "logits.tsv")):
        require(done[key] == decoded_sha(r / name), f"completion hash: {key}")
    require(identity["plan_debug_sha256"] == decoded_sha(r / "plan.debug"), "plan hash")
    require(identity["prompt_sha256"] == decoded_sha(r / "prompt.u32le"), "prompt hash")
    require(len(data(r / "tokens.u32le")) == 128 * 4, "continuation token count")
    require(len(data(r / "prompt.u32le")) == (int(context) - 128) * 4, "prompt count")
    rows = list(csv.DictReader((r / "logits.tsv").open(), delimiter="\t"))
    require([int(row["committed"]) for row in rows] == list(range(int(context) - 128, int(context) + 1)),
            "missing decision/final logit rows")
    require(rows[-1]["logits_f32le_sha256"] == decoded_sha(r / "final-logits.f32le"), "final logit hash")
    # In-process identity (the binary enforces it per cycle; the replay checks the bytes).
    require(data(r / "restored-prefix-state.tsv") == data(r / "prefix-state.tsv"), "restored prefix differs")
    construction = fields(r / "allocation-construction.txt")
    require(construction["allocator"] == "Vmm" and construction["construction"] == "direct"
            and construction["empty_plane_swap"] == "false" and construction["position"] == "0"
            and int(construction["vmm_planes"]) >= 32 and construction["pooled_planes"] == "0",
            "direct VMM construction not engaged")
    if RIGS[rig]["frozen"]:
        frozen = json.loads((HERE / RIGS[rig]["frozen"]).read_text())["baselines"][context]
        for key in ("artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "context"):
            require(identity[key] == frozen["identity"][key], f"wrong baseline identity: {key}")
        for name in SURFACES:
            require(decoded_sha(r / name) == frozen["surfaces_sha256"][name], f"baseline mismatch: {name}")
        return "matches the frozen target-card bundle (BOX3-BASELINES.json)"
    rented = RENTED / f"day6-baseline-{context}/receipt"
    rented_identity = fields(rented / "identity.txt")
    for key in ("artifact_sha256", "plan_debug_sha256", "prompt_sha256", "program", "context"):
        require(identity[key] == rented_identity[key], f"rented-bundle identity mismatch: {key}")
    differing = [name for name in SURFACES if decoded_sha(r / name) != decoded_sha(rented / name)]
    if not differing:
        return "matches the rented RTX 5090 frozen bundle (day6)"
    return ("decoded surfaces differ from the rented RTX 5090 bundle on " + ", ".join(differing)
            + "; this laptop card has no frozen bundle, continuation identity is in-process only")


def cycle(receipt, k, prefix_bytes):
    """One cycle directory: day-10 roundtrip receipt set, re-verified, plus in-process identity."""
    d = receipt / f"cycle-{k}"
    flags = fields(d / "active-reclaim.txt")
    m = ints(flags)
    require(m["vmm_granularity_bytes"] > 0, "cycle is not VMM")
    day9.verify_plane_census(d, m)
    day10.check_probe(d, m, flags)
    require(m["demote_count"] == m["reload_count"] == 32, "engagement")
    require(m["device_charged_after_demote"] == 0 and
            m["pinned_charged_after_demote"] == m["logical_d2h_bytes"] > 0, "accounting")
    gain = m["free_after_demote_bytes"] - m["free_before_bytes"]
    reacquired = m["free_after_demote_bytes"] - m["free_after_restore_bytes"]
    require(m["reclaimed_bytes"] == gain and m["reacquired_bytes"] == reacquired, "reported deltas")
    exact, bounded, residual = observe(m["free_before_bytes"], m["free_after_demote_bytes"],
                                       m["free_after_restore_bytes"], m["vmm_released_chunk_bytes"],
                                       m["vmm_granularity_bytes"])
    require(m["residual_bytes"] == residual, "residual math")
    require(flags["vmm_fixed_va_restored"] == "true", "fixed VA absent")
    require(flags["reclaim_observed"] == str(bounded).lower(), "bounded flag mismatch")
    require(flags["reclaim_exact_equal"] == str(exact).lower(), "exact flag mismatch")
    require(flags["g1_reclaim_qualified"] == str(bounded and residual == 0).lower(), "unsupported G1 promotion")
    if residual == 0:
        require(flags["residual_class"] == "none", "zero residual class")
    else:
        require(flags["residual_class"] != "none", "nonzero residual labelled none")
    restored = data(d / "restored-prefix-state.tsv")
    require(restored == prefix_bytes, f"cycle {k} restored prefix differs")
    return {"cycle": k, "free_before_bytes": m["free_before_bytes"],
            "free_after_demote_bytes": m["free_after_demote_bytes"],
            "free_after_restore_bytes": m["free_after_restore_bytes"],
            "vmm_released_chunk_bytes": m["vmm_released_chunk_bytes"],
            "vmm_granularity_bytes": m["vmm_granularity_bytes"],
            "reclaimed_bytes": gain, "reacquired_bytes": reacquired, "residual_bytes": residual,
            "restore_residual_bytes": m["free_before_bytes"] - m["free_after_restore_bytes"],
            "reclaim_observed": bounded, "reclaim_exact_equal": exact,
            "g1_reclaim_qualified": flags["g1_reclaim_qualified"], "residual_class": flags["residual_class"],
            "restored_prefix_state_manifest_sha256": hashlib.sha256(restored).hexdigest()}


def series(receipt, rows):
    """reclaim-cycles.tsv/.txt agree with the cycle directories and with the pure rule."""
    table = list(csv.DictReader((receipt / "reclaim-cycles.tsv").read_text().splitlines(), delimiter="\t"))
    require(len(table) == len(rows) == CYCLES, "series row count")
    first = rows[0]["free_before_bytes"]
    for row, expected in zip(table, rows):
        for key in ("free_before_bytes", "free_after_demote_bytes", "free_after_restore_bytes",
                    "vmm_released_chunk_bytes", "reclaimed_bytes", "reacquired_bytes", "residual_bytes",
                    "restore_residual_bytes", "g1_reclaim_qualified", "residual_class",
                    "restored_prefix_state_manifest_sha256"):
            require(str(row[key]) == str(expected[key]), f"series row {row['cycle']}: {key}")
        require(int(row["cycle"]) == expected["cycle"], "series cycle index")
        require(int(row["free_before_drift_bytes"]) == expected["free_before_bytes"] - first, "drift column")
        require(row["reclaim_observed"] == str(expected["reclaim_observed"]).lower()
                and row["reclaim_exact_equal"] == str(expected["reclaim_exact_equal"]).lower(), "series flags")
    summary = fields(receipt / "reclaim-cycles.txt")
    granule = rows[0]["vmm_granularity_bytes"]
    require({r["vmm_granularity_bytes"] for r in rows} == {granule}, "granule changed between cycles")
    cycles = [{"free_before": r["free_before_bytes"], "free_after_demote": r["free_after_demote_bytes"],
               "free_after_restore": r["free_after_restore_bytes"], "released": r["vmm_released_chunk_bytes"]}
              for r in rows]
    klass = classify_cycles(cycles, granule)
    residuals = [r["residual_bytes"] for r in rows]
    require(int(summary["cycles"]) == CYCLES and int(summary["vmm_granularity_bytes"]) == granule, "series header")
    require(summary["residual_series_class"] == klass, "series class does not follow the pure rule")
    require(summary["residual_series_bytes"] == ",".join(map(str, residuals)), "series bytes")
    require(int(summary["residual_first_cycle_bytes"]) == residuals[0]
            and int(summary["residual_last_cycle_bytes"]) == residuals[-1], "series ends")
    require(int(summary["free_before_drift_last_bytes"]) == rows[-1]["free_before_bytes"] - first, "series drift")
    require(summary["all_cycles_reclaim_observed"] == str(all(r["reclaim_observed"] for r in rows)).lower(),
            "series bounded flag")
    require(summary["all_cycles_restored_bit_identical"] == "true", "series identity flag")
    qualified = all(r["g1_reclaim_qualified"] == "true" for r in rows)
    require(summary["g1_reclaim_qualified"] == str(qualified).lower(), "series G1 flag")
    require(not (qualified and any(residuals)), "nonzero residual promoted")
    return klass, qualified


def verdict(context, rows, klass, qualified, baseline_status):
    tag = f"ACTIVE-{context // 1024}K"
    residuals = [r["residual_bytes"] for r in rows]
    bounded = all(r["reclaim_observed"] for r in rows)
    require(qualified == (bounded and klass == NONE and not any(residuals)), "verdict inputs disagree")
    if qualified and baseline_status.startswith("matches"):
        return f"{tag} G1 PASS across {CYCLES} cycles, residual 0 B each cycle, class none"
    if qualified:
        return (f"{tag} physical reclaim/restore bit-identical in-process across {CYCLES} cycles, "
                f"residual 0 B each cycle, class none, no frozen bundle for this card, not G1 PASS")
    kind = "physical reclaim/restore bit-identical" if bounded else \
        "copy/restore bit-identical, reclaim criteria (a)-(c) failed in at least one cycle"
    if len(set(residuals)) == 1:
        shape = f"residual {residuals[0]} B each cycle"
    else:
        shape = f"residual series {','.join(map(str, residuals))} B"
    return f"{tag} {kind} across {CYCLES} cycles, {shape}, class {klass}, not G1 PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", action="store_true", help="write day11-raw-manifest.json from the raw bytes")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    result = {}
    manifest = {}
    for rig, spec in RIGS.items():
        raw = HERE / spec["raw"]
        if not raw.exists():
            continue
        manifest.update({f"{spec['raw']}/{p.relative_to(raw)}": sha(p) for p in sorted(raw.rglob("*")) if p.is_file()})
        for folder in sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("cycles-")):
            label = folder.name
            if not (folder / "collector.exit").exists():
                continue
            if (folder / "collector.exit").read_text().strip() == "2":
                result[f"{rig}/{label}"] = refused(folder, rig)
                continue
            identity, envelope, console = collector(folder, rig)
            context = int(identity["context"])
            baseline_status = continuation(folder, identity, rig)
            receipt = folder / "receipt"
            prefix_bytes = data(receipt / "prefix-state.tsv")
            rows = [cycle(receipt, k, prefix_bytes) for k in range(1, CYCLES + 1)]
            klass, qualified = series(receipt, rows)
            result[f"{rig}/{label}"] = {
                "card": spec["card"], "power_envelope": envelope, "source": (folder / "source.commit").read_text().strip(),
                "binary_sha256": identity["binary_sha256"], "artifact_sha256": identity["artifact_sha256"],
                "baseline": baseline_status, "console": console, "cycles": rows,
                "residual_series_class": klass,
                "verdict": verdict(context, rows, klass, qualified, baseline_status)}
    if args.require_complete:
        expected = {f"{rig}/{label}" for rig in RIGS for label in CELLS}
        require(expected <= set(result), f"pending cells: {sorted(expected - set(result))}")
        for rig, spec in RIGS.items():
            build = HERE / spec["raw"] / "build"
            for name in ("exit", "clippy.exit"):
                require((build / name).read_text().strip() == "0", f"{rig} native {name} failed")
    mp = HERE / "day11-raw-manifest.json"
    if args.seal:
        mp.write_text(json.dumps(manifest, indent=2) + "\n")
    elif mp.exists():
        require(json.loads(mp.read_text()) == manifest, "raw archive membership/hash mismatch")
    print(json.dumps({"checked": result,
                      "pending": sorted({f"{rig}/{label}" for rig in RIGS for label in CELLS} - set(result)),
                      "cuda_executed_by_replay": False, "cross_card_timing_compared": False}, indent=2))


if __name__ == "__main__":
    main()
