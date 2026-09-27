"""Render independently replayed prefix-study results, with no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

NAMES = {
    "qwen": "Qwen3.8-27B NVFP4+Q5_K with embedded MTP",
    "gemma": "Gemma 4 12B QAT Q4_0 with Q8_0 assistant",
}


def number(value):
    return f"{value:.2f}"


def render(source, out):
    result = json.loads(source.read_text())
    if result.get("independent_replay", {}).get("raw_token_and_time_audits") != "pass":
        raise ValueError("rendering requires the independently replayed report")
    out.mkdir(parents=True, exist_ok=True)
    identity = result["identity"]
    runtime = identity["source"]
    gpu = [part.strip() for part in identity["gpu"].splitlines()[1].split(",")]
    lines = [
        "# Bounded prompt forecasting versus fixed K=3",
        "",
        "Independent native requests, unchanged full model/draft heads, one forecast per request.",
        "Adaptive K is 2 for prose, 4 for code/numeric, and 3 for mixed/unknown.",
        "The forecaster reads only the first X user-content tokenizer tokens.",
        "Six scenarios were registered for each model, requested format and prompt length.",
        "Fixed K=3 is the owner-specified control here; the study does not compare",
        "against a newly calibrated global K=2 or K=4 policy or native adaptation.",
        "",
        "Throughput is all returned model tokens divided by complete native request time,",
        "including reasoning, normal tokenization, prediction, cache allocation, prefill and",
        "generation. Model loading, warmup and receipt I/O are outside that clock.",
        "Pointwise 95% intervals use paired scenario bootstrap; this is an exploratory matrix.",
        "",
        f"Measured native recipe: `{runtime['source_recipe_commit']}`.",
        f"Runtime archive SHA-256: `{runtime['runtime_source_sha256']}`.",
        f"Staged harness SHA-256: `{runtime['harness_source_sha256']}`.",
        f"Research GPU: {gpu[0]}, {gpu[2]} reported VRAM, driver {gpu[3]}.",
        "",
    ]
    if result["independent_replay"]["same_depth_choices_across_budgets"]:
        lines += [
            "All three prefix budgets selected the same K on the scored instruction-first",
            "prompts. Differences among their measured rates therefore do not represent",
            "a different depth schedule.",
            "",
        ]
    for family, model in result["models"].items():
        lines += [
            "## " + NAMES[family],
            "",
            f"Maximum returned tokens: {model['selected_max_new']:,}, selected on a separate qualification task.",
            "",
            "| Pinned artifact | Revision | SHA-256 |",
            "|---|---|---|",
        ]
        for artifact in identity["artifacts"][family]:
            lines.append(
                f"| {artifact['file']} | `{artifact['revision']}` | `{artifact['sha256']}` |"
            )
        lines += [
            "",
            "| Requested output | Prompt tokens | Pairs | Fixed K=3 tok/s | X=64 tok/s (change) | X=128 tok/s (change) | X=256 tok/s (change) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for kind in ("prose", "code"):
            for length in (256, 1024, 4096, 16384):
                cell = model["cells"][f"{kind}-{length}"]["requested_format"]
                if not cell["pairs"]:
                    lines.append(f"| {kind} | {length:,} | 0 | — | — | — | — |")
                    continue
                arms = cell["arms"]
                values = [number(arms["fixed3"]["tokens_per_second"])]
                for budget in (64, 128, 256):
                    row = arms[f"prefix{budget}"]
                    values.append(f'{number(row["tokens_per_second"])} ({row["pooled_gain_percent"]:+.2f}%)')
                lines.append(f"| {kind} | {length:,} | {cell['pairs']} | " + " | ".join(values) + " |")
        lines += [
            "",
            "Prediction and final-format coverage are distinct from throughput:",
            "",
            "| Requested output | Prompt tokens | All-arms format-covered pairs | X=64 forecast μs | X=128 forecast μs | X=256 forecast μs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for kind in ("prose", "code"):
            for length in (256, 1024, 4096, 16384):
                cell = model["cells"][f"{kind}-{length}"]
                arms = cell["requested_format"]["arms"]
                timing = [number(arms[f"prefix{x}"]["routing_us"]["median"]) if arms else "—"
                          for x in (64, 128, 256)]
                lines.append(
                    f"| {kind} | {length:,} | {cell['all_arms_format_covered']['pairs']} | "
                    + " | ".join(timing) + " |")
        lines += [
            "",
            f"Matched loop exclusions: {len(model['loop_exclusions'])}. "
            "Primary rows retain capped outputs and format failures; the JSON contains the separate "
            "all-arms-format-covered subset, individual pair gains, output-length ratios, latency "
            "ratios, prediction/K counts and pointwise intervals.",
            "",
        ]
    lines += [
        "## Evidence boundary",
        "",
        "This measures a native, single-GPU research driver with a fresh cache for each request.",
        "It does not establish HTTP serving, concurrent-request performance, continuous-session KV reuse,",
        "or correctness of each generated program. Python parsing is a format-coverage diagnostic.",
        "",
        "Native/source/model identities and raw-audit status are in the accompanying JSON.",
        "JSON SHA-256: `" + hashlib.sha256(source.read_bytes()).hexdigest() + "`.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def plot(source, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = json.loads(source.read_text())
    colors = ("#0072B2", "#D55E00", "#009E73")
    lengths = (256, 1024, 4096, 16384)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for row, family in enumerate(("qwen", "gemma")):
        for column, kind in enumerate(("prose", "code")):
            axis = axes[row, column]
            for budget, color, factor in zip((64, 128, 256), colors, (.96, 1, 1.04)):
                points = []
                for length in lengths:
                    cell = result["models"][family]["cells"][f"{kind}-{length}"]["requested_format"]
                    if cell["pairs"]:
                        value = cell["arms"][f"prefix{budget}"]
                        points.append((length, value["pooled_gain_percent"],
                                       value["paired_bootstrap_95_percent"]))
                if not points:
                    continue
                x = [length * factor for length, _, _ in points]
                y = [value for _, value, _ in points]
                lower = [max(0, value - interval[0]) if interval else 0
                         for _, value, interval in points]
                upper = [max(0, interval[1] - value) if interval else 0
                         for _, value, interval in points]
                axis.errorbar(x, y, yerr=[lower, upper], fmt="o-", markersize=4,
                              linewidth=1.2, capsize=3, label=f"First {budget} tokens", color=color)
            axis.axhline(0, color="#555555", linewidth=.8, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks(lengths, ["256", "1,024", "4,096", "16,384"])
            axis.grid(axis="y", alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(f"{'Qwen3.8-27B' if family == 'qwen' else 'Gemma 4 12B'} · requested {kind}")
            axis.set_ylabel("Throughput change versus fixed K=3 (%)")
            if row == 1:
                axis.set_xlabel("User prompt length (tokenizer tokens)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Does a bounded prompt forecast improve speculative depth?", fontsize=14)
    figure.savefig(out / "prefix-depth-comparison.svg")
    figure.savefig(out / "prefix-depth-comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    render(args.source, args.out)
    if args.plot:
        plot(args.source, args.out)
