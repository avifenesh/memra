"""Render the independently replayed Qwen fixed-confidence comparison."""

import argparse
import json
from pathlib import Path


LABELS = {
    "c015": "C=0.15",
    "c030": "C=0.30",
    "c030zero": "C=0.30, zero-draft",
}


def result_cell(row):
    interval = row["paired_bootstrap_95_percent"]
    if interval is None:
        return f'{row["tokens_per_second"]:.2f} ({row["pooled_gain_percent"]:+.2f}%)'
    return (
        f'{row["tokens_per_second"]:.2f} ({row["pooled_gain_percent"]:+.2f}%, '
        f'[{interval[0]:+.2f}%, {interval[1]:+.2f}%])'
    )


def render(report, identity):
    if report["status"] != "measured-fixed-cutoff-only":
        raise ValueError("fixed confidence grid is not a scored result")
    source = identity["source"]
    artifact = identity["artifacts"]["qwen"][0]
    lines = [
        "# Qwen code at fixed K=3: confidence cutoff grid",
        "",
        "This is a native, independent-request comparison of draft stopping at a",
        "fixed K=3 ceiling. C=0 avoids confidence computation; positive C settings",
        "can shorten each draft round. No online confidence-threshold learner was",
        "measured by this grid, and no served default changes.",
        "",
        "Six frozen synthetic helper scenarios were paired across settings at each",
        "prompt length. The metric is returned tokens divided by complete native",
        "request seconds, including tokenization, prefill, drafting, verification",
        "and detokenization. Model load, warmup and receipt writing are excluded.",
        "Pointwise intervals resample six whole scenarios per length and are",
        "exploratory; they do not adjust for choosing among cutoff settings.",
        "",
        "| Code prompt tokens | Pairs | C=0 tok/s | C=0.15 tok/s (change, 95% interval) | C=0.30 tok/s (change, 95% interval) | C=0.30 with zero-draft tok/s (change, 95% interval) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for length in (256, 1024, 4096, 16384):
        row = report["code_by_prompt_tokens"][str(length)]
        if not row["pairs"]:
            lines.append(f"| {length:,} | 0 | — | — | — | — |")
            continue
        arms = row["arms"]
        lines.append(
            f"| {length:,} | {row['pairs']} | {arms['off']['tokens_per_second']:.2f} | "
            + " | ".join(result_cell(arms[label]) for label in LABELS) + " |"
        )
    lines += [
        "",
        "Across all code lengths, the pooled rates below count each returned token",
        "and request second once. Their scenario groups share topics across lengths,",
        "so only the per-length rows carry bootstrap intervals.",
        "",
        "| Setting | Pooled code tok/s | Change vs C=0 | Shortened code rounds / all code rounds | Format-covered code requests |",
        "|---|---:|---:|---:|---:|",
    ]
    pooled = report["all_code"]["arms"]
    for label in ("off", *LABELS):
        row = pooled[label]
        lines.append(
            f"| {LABELS.get(label, 'C=0')} | {row['tokens_per_second']:.2f} | "
            f"{row['pooled_gain_percent']:+.2f}% | "
            f"{row['confidence_shortened_rounds']}/{row['rounds']} | "
            f"{row['format_covered']}/{report['all_code']['pairs']} |"
        )
    lines += [
        "",
        f"Matched loop exclusions: {len(report['matched_loop_exclusions'])}.",
        "The primary rows retain capped outputs and format misses. The JSON has",
        "the separate all-arms-format-covered subset, paired changes, output-length",
        "and latency ratios, round histograms, and whole-scenario intervals.",
        "",
        "## Identity and scope",
        "",
        f"- Model: `{artifact['repo']}@{artifact['revision']}`, `{artifact['file']}`",
        f"  SHA-256 `{artifact['sha256']}`; full `{identity['artifacts']['qwen'][0]['bytes']:,}` bytes.",
        "- Drafter: Qwen embedded MTP with full target vocabulary.",
        f"- Research GPU: {identity['gpu'].splitlines()[1].split(',')[0].strip()}; one device.",
        f"- Measured binary SHA-256: `{source['binaries']['qwen-prefix-study']}`.",
        f"- Patched runtime source SHA-256: `{source['runtime_source_sha256']}`.",
        f"- Original sealed runtime SHA-256: `{source['base_runtime_source_sha256']}`.",
        "- Sampler: temperature 0.7, top-k 20, top-p 0.95; default thinking.",
        "- Cache: fresh native cache for every request; no HTTP or concurrency claim.",
        "- Correctness: target-only greedy oracle passed for every C setting,",
        "  followed by matched driver tapes and seeded sampled reruns at the",
        "  study's 0.7/20/0.95 decode shape.",
        "- Format coverage checks fenced Python syntax, not functional correctness.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    identity = json.loads(args.identity.read_text())
    args.out.write_text(render(report, identity))


if __name__ == "__main__":
    main()
