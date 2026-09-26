"""Render the code-only v2 held-out C comparison from replayed records."""

import argparse
import json
from pathlib import Path


def change(row):
    interval = row["paired_bootstrap_95_percent"]
    if interval is None:
        return f'{row["pooled_gain_percent"]:+.2f}%'
    return (
        f'{row["pooled_gain_percent"]:+.2f}% '
        f'[{interval[0]:+.2f}%, {interval[1]:+.2f}%]'
    )


def render(result):
    if result["status"] == "no-positive-development-c":
        return (
            "# Qwen code-only v2 held-out result\n\n"
            "The tuple-utility code qualifier passed, but no positive fixed-C "
            "candidate was selected from the development grid. No held-out "
            "throughput run was started.\n"
        )
    if result["status"] != "measured-heldout-code-v2":
        raise ValueError("code-only v2 has no completed result")
    lines = [
        "# Qwen code-only v2 held-out result",
        "",
        "The first all-format held-out qualifier stopped at 4K prose after",
        "8,192 reasoning tokens with no final answer. That failure is retained",
        "separately. This version qualifies requested **code** only on a new",
        "tuple-utility task, then compares the selected development C against",
        "fixed K=3/C=0 and fixed K=2/C=0 on six untouched task families.",
        "",
        f"Selected development arm: `{result['selected_c']}` "
        f"(pmin={result['selected_pmin']}, zero-draft={'on' if result['selected_pmin0'] else 'off'}).",
        f"Code qualification: {result['qualification_code_covered']}/4 covered; "
        f"prose qualification diagnostic: {result['qualification_prose_covered']}/4 covered.",
        "",
        "| Code prompt tokens | Pairs | All arms code-covered | K=3/C=0 tok/s | Selected tok/s | Change vs K=3/C=0 (95% interval) | K=2/C=0 tok/s | Change vs K=2/C=0 (95% interval) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length in (256, 1024, 4096, 16384):
        row = result["code_by_prompt_tokens"][str(length)]
        if not row["pairs"]:
            lines.append(f"| {length:,} | 0 | 0 | — | — | — | — | — |")
            continue
        arms, comparisons = row["arms"], row["selected_vs"]
        lines.append(
            f"| {length:,} | {row['pairs']} | "
            f"{result['all_arms_format_covered_code'][str(length)]['pairs']} | "
            f"{arms['k3off']['tokens_per_second']:.2f} | "
            f"{arms['selected']['tokens_per_second']:.2f} | "
            f"{change(comparisons['k3off'])} | "
            f"{arms['k2off']['tokens_per_second']:.2f} | "
            f"{change(comparisons['k2off'])} |"
        )
    pooled = result["all_code"]
    if pooled["pairs"]:
        arms, comparisons = pooled["arms"], pooled["selected_vs"]
        lines += [
            "",
            "Pooled across code lengths (the same six topics recur at each length,",
            "so no pooled bootstrap interval is attached):",
            "",
            "| Pairs | K=3/C=0 tok/s | Selected tok/s | Change vs K=3/C=0 | K=2/C=0 tok/s | Change vs K=2/C=0 | Selected confidence-shortened rounds |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {pooled['pairs']} | {arms['k3off']['tokens_per_second']:.2f} | "
            f"{arms['selected']['tokens_per_second']:.2f} | "
            f"{change(comparisons['k3off'])} | "
            f"{arms['k2off']['tokens_per_second']:.2f} | "
            f"{change(comparisons['k2off'])} | "
            f"{arms['selected']['confidence_shortened_rounds']}/{arms['selected']['rounds']} |",
        ]
    lines += [
        "",
        f"Matched loop exclusions: {len(result['matched_loop_exclusions'])}.",
        "The primary code rows retain capped outputs and format misses; the",
        "machine-readable report includes the all-arms-format-covered subset,",
        "paired gains, output-length ratios, latency ratios, and prose diagnostics.",
        "Intervals are pointwise whole-scenario resamples of this synthetic",
        "corpus and do not adjust for selecting C on the earlier grid.",
        "",
        "Both versions use the same pinned Qwen NVFP4+Q5_K artifact, full embedded",
        "MTP head, sampled 0.7/20/0.95 decode, one non-production RTX 5090, and",
        "fresh native caches. This is a fixed-cutoff test, not live C learning,",
        "warm-session KV, HTTP/concurrency qualification, or functional code",
        "correctness.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(render(json.loads(args.report.read_text())))


if __name__ == "__main__":
    main()
