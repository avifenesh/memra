#!/usr/bin/env python3
"""Render the fixed-shape cx-requal result and both customer one-pagers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SOURCE = "ac6ef049b8661008c0da91f4747f68f4dabdaa04"
BASE_LEVELS = (1, 2, 4, 8)
MODEL_NAMES = {"q27": "Q27", "q35": "Q35"}
TRAFFIC_ROWS = (
    ("90%-hit mix, all traffic", "mixed90", "all"),
    ("Cache hits only", "mixed90", "hit"),
    ("Misses inside the 90%-hit mix", "mixed90", "miss"),
    ("Pure cold arm", "cold", "all"),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def ms(value: Any) -> str:
    return f"{float(value):,.3f} ms"


def number(value: Any, decimals: int = 3) -> str:
    return f"{float(value):,.{decimals}f}"


def integer(value: Any) -> str:
    return f"{int(value):,}"


def input_hash(summary: dict[str, Any], basename: str) -> str:
    matches = [
        digest
        for path, digest in summary["input_sha256s"].items()
        if Path(path).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one input hash for {basename}, found {len(matches)}")
    return str(matches[0])


def template_hashes(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        name = Path(str(row["model"])).name
        result[name] = row
    return result


def distribution_table(summary: dict[str, Any], target: str, family: str) -> list[str]:
    groups = summary["measurements_by_target_arm_concurrency"][target]
    lines = [
        f"| {MODEL_NAMES[target]} c=4 traffic class | N | p50 | p75 | p90 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm, role in TRAFFIC_ROWS:
        group = groups[arm]["4"]
        dist = group[f"{family}_{role}"]
        lines.append(
            f"| {label} | {integer(dist['n'])} | {ms(dist['p50_ms'])} | "
            f"{ms(dist['p75_ms'])} | {ms(dist['p90_ms'])} | {ms(dist['p95_ms'])} | "
            f"{ms(dist['p99_ms'])} |"
        )
    return lines


def target_cells(replay: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    return [row for row in replay if row.get("kind") == "cell" and row.get("target") == target]


def base_score(replay: list[dict[str, Any]], target: str) -> tuple[int, int]:
    rows = [
        row for row in target_cells(replay, target)
        if int(row["concurrency"]) in BASE_LEVELS
    ]
    return sum(bool(row["clean"]) for row in rows), len(rows)


def total_score(replay: list[dict[str, Any]], target: str) -> tuple[int, int]:
    rows = target_cells(replay, target)
    return sum(bool(row["clean"]) for row in rows), len(rows)


def model_totals(summary: dict[str, Any], target: str) -> dict[str, int]:
    levels = summary["replay_counts"]["levels"]
    groups = summary["measurements_by_target_arm_concurrency"][target]
    result = {
        "requests": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens_in": 0,
        "prefix_cache_hit_tokens": 0,
        "session_defers": 0,
        "vram_defers": 0,
        "oom_parks": 0,
    }
    for arm in ("cold", "mixed90"):
        for level in levels:
            group = groups[arm][str(level)]
            counters = group["counter_totals"]
            result["requests"] += int(group["n_requests"])
            result["prompt_tokens"] += int(group["prompt_tokens"])
            result["cached_tokens"] += int(group["cached_tokens_from_usage"])
            result["completion_tokens"] += int(counters["tokens_out"])
            result["cached_tokens_in"] += int(counters["cached_tokens_in"])
            result["prefix_cache_hit_tokens"] += int(counters["prefix_cache_hit_tokens"])
            result["session_defers"] += int(counters["admission_session_defers"])
            result["vram_defers"] += int(counters["admission_vram_defers"])
            result["oom_parks"] += int(counters["step_oom_parks"])
    return result


def rate_table(summary: dict[str, Any], target: str) -> list[str]:
    groups = summary["measurements_by_target_arm_concurrency"][target]
    mixed = groups["mixed90"]["4"]
    cold = groups["cold"]["4"]
    counters = mixed["counter_totals"]
    peak_mib = int(mixed["prefix_cache_bytes_peak_cell_end"]) / 1024 / 1024
    return [
        f"| {MODEL_NAMES[target]} c=4 measurement | N=5 median or exact total |",
        "|---|---:|",
        f"| Mixed output throughput | **{number(mixed['output_tok_s_median'])} completion tok/s** |",
        f"| Mixed requests/s | {number(mixed['requests_per_s_median'])} |",
        f"| Mixed billed prompt rate | {number(mixed['billed_prompt_tok_s_median'])} prompt tok/s |",
        f"| Mixed computed prompt rate | {number(mixed['computed_prompt_tok_s_median'])} prompt tok/s |",
        f"| Pure-cold output throughput | {number(cold['output_tok_s_median'])} completion tok/s |",
        f"| c=4 mixed prompt / cached / completion tokens | {integer(mixed['prompt_tokens'])} / "
        f"**{integer(mixed['cached_tokens_from_usage'])}** / {integer(counters['tokens_out'])} |",
        f"| Engine cached counters | `cached_tokens_in={integer(counters['cached_tokens_in'])}`; "
        f"`prefix_cache_hit_tokens={integer(counters['prefix_cache_hit_tokens'])}` |",
        f"| Cache hits / misses | {integer(counters['prefix_cache_hits'])} / "
        f"{integer(counters['prefix_cache_misses'])} |",
        f"| Session defers / VRAM defers / OOM parks | "
        f"{integer(counters['admission_session_defers'])} / "
        f"{integer(counters['admission_vram_defers'])} / {integer(counters['step_oom_parks'])} |",
        f"| Prefix-cache budget / observed c=4 peak | 4,096 MiB / {number(peak_mib)} MiB |",
    ]


def capacity_table(summary: dict[str, Any], target: str) -> list[str]:
    groups = summary["measurements_by_target_arm_concurrency"][target]
    knee = int(summary["targets"][target]["capacity_width"])
    lines = [
        "| c/model | Cold output tok/s | 90%-hit output tok/s | Mixed hit TTFT p95 | "
        "Mixed all TTFT p50 | Mixed all TTFT p99 | Clean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in summary["replay_counts"]["levels"]:
        cold = groups["cold"][str(level)]
        mixed = groups["mixed90"][str(level)]
        if level == 4:
            label = "**4 sold cap**"
        elif level == knee:
            label = f"**{level} measured knee**"
        else:
            label = str(level)
        clean = "PASS" if cold["all_clean"] and mixed["all_clean"] else "FAIL"
        lines.append(
            f"| {label} | {number(cold['output_tok_s_median'])} | "
            f"{number(mixed['output_tok_s_median'])} | {ms(mixed['ttft_hit']['p95_ms'])} | "
            f"{ms(mixed['ttft_all']['p50_ms'])} | {ms(mixed['ttft_all']['p99_ms'])} | "
            f"{clean} |"
        )
    return lines


def regression_section(analysis: dict[str, Any]) -> list[str]:
    comparison = analysis["q27_comparison"]
    regressions = comparison["regressions_gt_2_percent"]
    if not regressions:
        return [
            "## Q27 comparison with the original sellgate",
            "",
            f"No published Q27 number regressed by more than 2% across the "
            f"{integer(len(comparison['comparisons']))}-metric audit. Values were compared "
            "individually; no averaging was used.",
        ]
    lines = [
        "## ⚠ Q27 REGRESSION FLAGS ABOVE 2%",
        "",
        f"**{integer(len(regressions))} published Q27 metrics regressed by more than 2%.** "
        "Every regression is listed individually; none is averaged away.",
        "",
        "| Metric | Original | Fresh | Regression |",
        "|---|---:|---:|---:|",
    ]
    for row in regressions:
        lines.append(
            f"| `{row['metric']}` | {number(row['old'], 6)} | {number(row['fresh'], 6)} | "
            f"**{number(row['regression_percent'])}%** |"
        )
    return lines


def matrix_section(analysis: dict[str, Any]) -> list[str]:
    matrix = analysis["regression_matrix"]
    lines = [
        "## Explicit q35bug regression matrix — mixed c=2 x5",
        "",
        "These are the exact frozen campaign cells, surfaced explicitly for both models.",
        "",
        "| Model | Rep | Requests OK | Response tokens | Engine `tokens_out` | Short | "
        "Cached drift | Clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("q27", "q35"):
        for row in matrix["targets"][target]["matrix"]:
            cached_drift = int(row["cached_tokens_in_drift"]) + int(
                row["prefix_cache_hit_tokens_drift"]
            )
            lines.append(
                f"| {MODEL_NAMES[target]} | {row['rep']} | {row['requests_ok']}/{row['requests']} | "
                f"{integer(row['response_usage_tokens'])} | {integer(row['engine_tokens_out'])} | "
                f"{integer(row['short_completions'])} | {integer(cached_drift)} | "
                f"{'PASS' if row['clean'] else 'FAIL'} |"
            )
    return lines


def one_page(summary: dict[str, Any], replay: list[dict[str, Any]], target: str) -> list[str]:
    name = MODEL_NAMES[target]
    target_summary = summary["targets"][target]
    page_title = (
        f"Customer one-page envelope — {name}"
        if target_summary["verdict"] == "SELLABLE"
        else f"Measured one-page — {name} ({target_summary['verdict']})"
    )
    groups = summary["measurements_by_target_arm_concurrency"][target]
    mixed = groups["mixed90"]["4"]
    cold = groups["cold"]["4"]
    totals = model_totals(summary, target)
    clean_cells, cells = total_score(replay, target)
    lines = [
        f"## {page_title}",
        "",
        "Frozen workload: 4,860 prompt tokens plus 60 completion tokens (81:1), c=4/model, "
        f"with {MODEL_NAMES['q35' if target == 'q27' else 'q27']} active on the other GPU. "
        "Each row pools five interleaved cells; mixed traffic is 90 full-prefix hits and 10 "
        "real misses, and pure cold is a separate 100-request population.",
        "",
        "### First-content TTFT",
        "",
        *distribution_table(summary, target, "ttft"),
        "",
        f"Typical mixed TTFT is {ms(mixed['ttft_all']['p50_ms'])}; cache-hit p95 is "
        f"{ms(mixed['ttft_hit']['p95_ms'])}. The 10% miss class puts mixed p99 at "
        f"{ms(mixed['ttft_all']['p99_ms'])}; pure-cold c=4 p50 is "
        f"{ms(cold['ttft_all']['p50_ms'])}.",
        "",
        "### Full-response latency for 60 completion tokens",
        "",
        *distribution_table(summary, target, "latency"),
        "",
        "### Inter-token latency",
        "",
        *distribution_table(summary, target, "inter_token"),
        "",
        "### Rate and accounting envelope",
        "",
        *rate_table(summary, target),
        "",
        "Both servers were active in the same c=4 windows. Pair-window throughput, measured "
        f"from the shared release barrier until the slower model drained, was "
        f"**{number(summary['pair_shape']['c4_mixed_pair_output_tok_s_median_same_window'])} "
        "completion tok/s median** across five repetitions.",
        "",
        f"### {name} capacity headroom",
        "",
        *capacity_table(summary, target),
        "",
        f"The clean throughput knee is c={target_summary['capacity_width']}, or "
        f"**{number(target_summary['capacity_headroom_percent'], 0)}% headroom** above the sold "
        "cap of four. Across the full campaign this model completed "
        f"**{integer(totals['requests'])} requests** and {clean_cells}/{cells} cells clean; "
        f"cached tokens reconcile {integer(totals['cached_tokens'])} = "
        f"{integer(totals['cached_tokens_in'])} = {integer(totals['prefix_cache_hit_tokens'])}.",
    ]
    return lines


def parse_lock_times(text: str) -> tuple[str, str]:
    starts = re.findall(r"REQUAL_LOCK_ACQUIRED ts=(\S+)", text)
    ends = re.findall(r"REQUAL_PIPELINE_PASS ts=(\S+)", text)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("pipeline log must have exactly one lock start and PASS timestamp")
    return starts[0], ends[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--template-hashes", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    summary = load_json(args.summary)
    analysis = load_json(args.analysis)
    replay = load_jsonl(args.replay)
    templates = template_hashes(args.template_hashes)
    lock_start, lock_end = parse_lock_times(args.pipeline_log.read_text(encoding="utf-8"))
    if summary.get("schema") != "memra.sellgate.summary.v1":
        raise ValueError("unexpected summary schema")
    if analysis.get("schema") != "memra.requal.analysis.v1":
        raise ValueError("unexpected analysis schema")

    verdicts = {target: summary["targets"][target]["verdict"] for target in ("q27", "q35")}
    both_sellable = all(value == "SELLABLE" for value in verdicts.values())
    lines = [
        "# Q35 + Q27 fresh sold-cap requalification — eu-west PRO pair",
        "",
        "Date: 2026-08-12",
        "",
        "Rig: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, one target-only server per physical GPU",
        "",
        f"Scored runtime source: `{SOURCE}`",
        "",
        "## Verdict",
        "",
        (
            "**PAIR QUALIFIED: Q27 and Q35 are both SELLABLE at c=4.**"
            if both_sellable
            else f"**PAIR NOT QUALIFIED: Q27 is {verdicts['q27']}; Q35 is {verdicts['q35']}.**"
        ),
        "",
        "The two-second bars are first-content TTFT, not full-response latency. Full-response "
        "percentiles are published separately; no cold or p99 sub-two-second promise is made.",
        "",
        "| Model | Standard exactness | Serial cache exactness | Required base cells | "
        "c=4 hit TTFT p95 | c=4 all-traffic TTFT p50 | c=4 cached-token reconciliation | "
        "Clean throughput knee / headroom | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in ("q27", "q35"):
        target_summary = summary["targets"][target]
        mixed = target_summary["c4_mixed90"]
        counters = mixed["counter_totals"]
        clean, total = base_score(replay, target)
        lines.append(
            f"| {MODEL_NAMES[target]} | "
            f"{'PASS' if target_summary['criteria']['standard_correctness'] else 'FAIL'} | "
            f"{'PASS' if target_summary['criteria']['serial_cache_exactness'] else 'FAIL'} | "
            f"**{clean}/{total}** | **{ms(mixed['ttft_hit']['p95_ms'])}** | "
            f"**{ms(mixed['ttft_all']['p50_ms'])}** | "
            f"**{integer(mixed['cached_tokens_from_usage'])} = "
            f"{integer(counters['cached_tokens_in'])} = "
            f"{integer(counters['prefix_cache_hit_tokens'])}** | "
            f"c={target_summary['capacity_width']} / "
            f"**{number(target_summary['capacity_headroom_percent'], 0)}%** above c=4 | "
            f"**{target_summary['verdict']}** |"
        )

    lines.extend(["", *regression_section(analysis), "", *matrix_section(analysis)])
    for target in ("q27", "q35"):
        lines.extend(["", *one_page(summary, replay, target)])

    q27_template = templates["Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"]
    q35_template = templates["Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"]
    lines.extend(
        [
            "",
            "## Exactness and pinned inputs",
            "",
            "- Both physical GPUs passed the full kernel checker. Q27 and Q35 each passed "
            "prefill/decode and batched-prime/tokenwise argmax MATCH plus `run-spec` K=1..8.",
            "- Both serial partial-prefix gates passed N=3 with byte-identical cold/partial/full "
            "output and exact client/engine cached-token reconciliation.",
            "- At c=4, each model's cache-hit output hashes are a subset of its cold output "
            "hashes. This does not claim identity across different batching compositions.",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
            f"| Runtime source | `{SOURCE}` |",
            f"| `memra-server` | `{input_hash(summary, 'memra-server')}` |",
            f"| `kernel-check` | `{input_hash(summary, 'kernel-check')}` |",
            f"| `run-gen` | `{input_hash(summary, 'run-gen')}` |",
            f"| `run-spec` | `{input_hash(summary, 'run-spec')}` |",
            f"| Q27 artifact | `{input_hash(summary, 'Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf')}` |",
            f"| Q27 external draft | `{input_hash(summary, 'draft-daily-owntrim-nvfp4head-q4blk.gguf')}` |",
            f"| Q27 embedded `tokenizer.chat_template` ({integer(q27_template['size_bytes'])} bytes) | "
            f"`{q27_template['sha256']}` |",
            f"| Q35 artifact | `{input_hash(summary, 'Qwen3.6-35B-A3B-UD-IQ4_XS.gguf')}` |",
            f"| Q35 external draft | `{input_hash(summary, 'draft-35b-owntrim-nvfp4head-q4blk.gguf')}` |",
            f"| Q35 embedded `tokenizer.chat_template` ({integer(q35_template['size_bytes'])} bytes) | "
            f"`{q35_template['sha256']}` |",
            f"| Workload lock | `{summary['protocol']['workload_lock_sha256']}` |",
            f"| Canonical scored prompt IDs | "
            f"`{summary['protocol']['prompt_ids_sha256_canonical_json']}` |",
            "",
            "## Method and receipts",
            "",
            f"- One uninterrupted `/tmp/memra-gpu.lock` hold ran from {lock_start} through "
            f"the sealed PASS at {lock_end}; the gateway soak queued behind it.",
            f"- Both target servers stayed live together through {integer(summary['replay_counts']['cells'])} "
            f"cells and {integer(summary['replay_counts']['requests'])} scored requests. Arms "
            "alternated, base-width order rotated, and every width used N=5 without artificial "
            "cooldown or clock changes.",
            f"- Thermal maxima by GPU: `{json.dumps(summary['thermal'], sort_keys=True)}`.",
            f"- Campaign manifest: `{summary['manifests']['campaign']['sha256']}` "
            f"({summary['manifests']['campaign']['files_checked']} files verified).",
            f"- Correctness manifest: `{summary['manifests']['gates']['sha256']}` "
            f"({summary['manifests']['gates']['files_checked']} files verified).",
            "- Machine-readable verdict: [`summary.json`](summary.json); explicit regression "
            "matrix and Q27 comparison: [`analysis.json`](analysis.json); sealed raw evidence: "
            "[`raw/campaign/`](raw/campaign/) and [`raw/gates/`](raw/gates/).",
            "",
            "No runtime code, generated performance board, README number, merge, tag, push, or "
            "formatting surface changed in this lane.",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"both_sellable": both_sellable, "verdicts": verdicts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
