"""Render the sealed v3 Qwen code results without selecting a new winner."""

import argparse
import json
from pathlib import Path


LABELS = {
    "learn-c3": "Live learned C, K=3",
    "monitor-c3": "Monitor control, K=3/C=0",
    "fixed-c3": "Calibrated fixed C, K=3",
    "fixed:3": "Fixed K=3/C=0",
    "fixed:2": "Fixed K=2/C=0",
}


def render(result, quality, costs, oracle):
    passed = {
        arm: sum(row["pass"] for row in quality["rows"] if row["arm"] == arm)
        for arm in LABELS
    }
    totals = {
        arm: sum(row["arm"] == arm for row in quality["rows"])
        for arm in LABELS
    }
    if any(total != 48 for total in totals.values()):
        raise ValueError("one-case functional coverage is incomplete")
    metrics = result["metrics"]
    bands = result["conversation_bootstrap_95_percent_gain_vs_k3c0"]
    learned = metrics["learn-c3"]
    paired = result["learned_vs_comparator_bootstrap_95_percent"]
    trajectories = result["learned_confidence_trajectories"]
    nonzero_applied = sum(
        row["applied_before"] != "0.000000000,0.000000000"
        for row in trajectories
    )
    probes = sum(row["decision"]["probe"] for row in trajectories)
    moves = sum(
        row["applied_before"] != row["applied_after"] for row in trajectories
    )
    lines = [
        "# Qwen K=3 live learned confidence: held-out native result",
        "",
        "The earlier C=0/0.15/0.30 grid used preset cutoffs. This run tested a live "
        "cutoff derived from verified accepted prefixes and measured K-dependent cost.",
        "",
        "## Complete-request comparison",
        "",
        "| Arm | tok/s | Gain vs K=3/C=0 (95% conversation bootstrap) | Output ratio | Time ratio | Fenced code | One-case function probes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, label in LABELS.items():
        row = metrics[arm]
        low, high = bands[arm]
        lines.append(
            f"| {label} | {row['tokens_per_s']:.2f} | "
            f"{row['gain_vs_k3c0_percent']:+.2f}% [{low:+.2f}, {high:+.2f}] | "
            f"{row['output_token_ratio_vs_k3c0']:.3f} | "
            f"{row['complete_time_ratio_vs_k3c0']:.3f} | "
            f"{row['format_covered']}/{row['turns']} | "
            f"{passed[arm]}/{totals[arm]} |"
        )
    lines.extend([
        "",
        f"Live learned C versus the calibrated fixed C: "
        f"{100*(learned['tokens_per_s']/metrics['fixed-c3']['tokens_per_s']-1):+.2f}%, "
        f"conversation bootstrap "
        f"[{paired['fixed-c3'][0]:+.2f}%, {paired['fixed-c3'][1]:+.2f}%]. "
        f"Versus the equal-budget monitor: "
        f"{100*(learned['tokens_per_s']/metrics['monitor-c3']['tokens_per_s']-1):+.2f}%, "
        f"[{paired['monitor-c3'][0]:+.2f}%, {paired['monitor-c3'][1]:+.2f}%].",
        "",
        f"The learner applied a nonzero cutoff on {nonzero_applied}/{len(trajectories)} turns, "
        f"changed its cutoff after {moves} turns and made {probes} full-offer probes. "
        f"Controller and trace work took {learned['controller_wall_s']:.2f}s "
        f"within {learned['complete_request_s']:.2f}s of complete request time.",
        "",
        f"Live learned C offered K=1/2/3 on "
        f"{learned['offered_draft_histogram']['1']}/"
        f"{learned['offered_draft_histogram']['2']}/"
        f"{learned['offered_draft_histogram']['3']} native rounds. "
        f"Mean actually offered draft tokens per round were "
        f"{learned['offered_drafts_per_round']:.3f} live, "
        f"{metrics['fixed-c3']['offered_drafts_per_round']:.3f} fixed C and "
        f"{metrics['fixed:3']['offered_drafts_per_round']:.3f} K=3/C=0. "
        "The fixed-C path did not retain per-round chosen-confidence traces. "
        f"{learned['accepted_per_drafted']:.3f} drafts were accepted. "
        "Draft acceptance is a diagnostic, not the throughput or code-quality score.",
        "",
        "## Requested reference-length cells",
        "",
        "| Padding characters | Live C tok/s | K=3/C=0 tok/s | Live gain | Fixed C tok/s | K=2/C=0 tok/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    lengths = result["requested_reference_padding_cells"]
    for padding in (256, 1024, 4096, 16384):
        cell = str(padding)
        if any(cell not in lengths[arm] for arm in (
            "learn-c3", "fixed:3", "fixed-c3", "fixed:2"
        )):
            continue
        live_rate = lengths["learn-c3"][cell]["tokens_per_s"]
        base_rate = lengths["fixed:3"][cell]["tokens_per_s"]
        fixed_rate = lengths["fixed-c3"][cell]["tokens_per_s"]
        k2_rate = lengths["fixed:2"][cell]["tokens_per_s"]
        lines.append(
            f"| {padding:,} | {live_rate:.2f} | {base_rate:.2f} | "
            f"{100*(live_rate/base_rate-1):+.2f}% | "
            f"{fixed_rate:.2f} | {k2_rate:.2f} |"
        )
    lines.extend([
        "",
        "These cells are grouped by frozen requested reference padding, "
        "with realized prompt-token counts retained in `RESULTS.json`. "
        "They contain fewer whole conversations than the pooled result.",
        "",
        "## Calibration bound",
        "",
        f"The three disjoint calibration conversations provided {oracle['rounds']:,} "
        f"eligible full K=3 offers. Measured eligible round costs for K=1/2/3 were "
        f"{costs['round_ns']['1']/1e6:.3f}/"
        f"{costs['round_ns']['2']/1e6:.3f}/"
        f"{costs['round_ns']['3']/1e6:.3f} ms. The optimistic hindsight "
        f"cost oracle was {oracle['oracle_gain_percent']:+.2f}% above K=3/C=0 "
        f"[{oracle['conversation_bootstrap_95_percent']['oracle_gain_percent'][0]:+.2f}, "
        f"{oracle['conversation_bootstrap_95_percent']['oracle_gain_percent'][1]:+.2f}] "
        f"and is not a live result. Calibration selected fixed "
        f"C=({oracle['best_calibrated_fixed']['cutoffs'][0]:.6f}, "
        f"{oracle['best_calibrated_fixed']['cutoffs'][1]:.6f}); its modeled "
        "gain is in-sample and may be optimistic.",
        "",
        "## Scope",
        "",
        f"Included conversations: {result['included_conversations']}; matched whole-conversation "
        f"loop exclusions: {result['excluded_conversations']}. "
        "All timed arms used sampled temperature 0.7, top-k 20, top-p 0.95, "
        "the full embedded Qwen MTP head, one non-production RTX 5090 and "
        "eight-turn native KV continuation. The function probes cover one "
        "valid-domain case per request across all 48 held-out requests per arm, "
        "including any performance-loop exclusions; they are not general "
        "code-correctness proof. Conversation bootstrap intervals with at most "
        "six synthetic conversations are exploratory. "
        "Greedy identity and controlled small-vocabulary sampling checks do not "
        "qualify a served positive-C policy; Memra #673 remains the sampled "
        "serving gate.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    for name in ("results", "quality", "costs", "oracle", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(render(*(
        json.loads(getattr(args, name).read_text())
        for name in ("results", "quality", "costs", "oracle")
    )))


if __name__ == "__main__":
    main()
