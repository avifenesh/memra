"""Frozen online-C policy replay over independently measured Qwen requests.

The controller sees only the result of its chosen arm. The requests are
independent, so choosing one measured arm cannot change a later prompt.
This is a counterfactual replay, not a live continuous-KV runtime result.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import statistics

from report_fixed import report as audit_grid


ARMS = ("off", "c015", "c030", "c030zero")
CALIBRATION = (0, 1)
EVALUATION = (2, 3, 4, 5)
PROBE_EVERY = 6
ACCEPT_LOW = 0.65
ACCEPT_HIGH = 0.82
KEEP_GAIN = 0.02


def rate(row):
    return row["output_tokens"] / row["elapsed_s"]


def pool(rows):
    tokens = sum(row["output_tokens"] for row in rows)
    seconds = sum(row["elapsed_s"] for row in rows)
    return {
        "requests": len(rows),
        "tokens": tokens,
        "native_request_seconds": seconds,
        "native_tokens_per_second": tokens / seconds,
        "format_covered": sum(row["format"]["requested_format_covered"] for row in rows),
    }


def groups(root):
    result = {}
    for scenario in (*CALIBRATION, *EVALUATION):
        saved = {
            label: json.loads(
                (root / "qwen/scored" / f"{scenario:02}-{label}.audit.json").read_text()
            )["requests"]
            for label in ARMS
        }
        for turn in range(8):
            rows = {label: saved[label][turn] for label in ARMS}
            if rows["off"]["kind"] != "code":
                continue
            length = rows["off"]["length_target"]
            if any(row["kind"] != "code" or row["length_target"] != length for row in rows.values()):
                raise ValueError("offline policy saw mismatched requests")
            result[scenario, turn + 1] = rows
    return result


def calibrate(pairs):
    chosen = [
        rows for (scenario, _), rows in pairs.items()
        if scenario in CALIBRATION and not any(row["loop"] for row in rows.values())
    ]
    if len(chosen) < 4:
        raise ValueError("insufficient independent calibration requests")
    baseline = {}
    for length in (256, 1024, 4096, 16384):
        records = [row["off"] for row in chosen if row["off"]["length_target"] == length]
        if not records:
            raise ValueError("calibration did not cover every prompt length")
        baseline[length] = pool(records)["native_tokens_per_second"]
    scores = {
        label: [rate(row[label]) / baseline[row[label]["length_target"]] for row in chosen]
        for label in ARMS
    }
    means = {label: statistics.mean(values) for label, values in scores.items()}
    incumbent = max(ARMS, key=lambda label: (means[label], -ARMS.index(label)))
    return baseline, scores, incumbent


class Controller:
    def __init__(self, baseline, scores, incumbent):
        self.baseline = baseline
        self.scores = {name: list(values) for name, values in scores.items()}
        self.incumbent = incumbent
        self.recent = deque(maxlen=4)
        self.requests = 0
        self.probes = 0

    def choose(self):
        current = ARMS.index(self.incumbent)
        if (self.requests + 1) % PROBE_EVERY:
            return self.incumbent, False
        alpha = (
            sum(accepted for accepted, _ in self.recent)
            / sum(drafted for _, drafted in self.recent)
            if self.recent and sum(drafted for _, drafted in self.recent) else None
        )
        if alpha is not None and alpha < ACCEPT_LOW:
            direction = 1
        elif alpha is not None and alpha > ACCEPT_HIGH:
            direction = -1
        else:
            direction = 1 if self.probes % 2 == 0 else -1
        candidate = current + direction
        if not 0 <= candidate < len(ARMS):
            candidate = current - direction
        self.probes += 1
        if not 0 <= candidate < len(ARMS):
            return self.incumbent, False
        return ARMS[candidate], True

    def observe(self, label, probe, row):
        before = self.incumbent
        self.requests += 1
        alpha = row["accepted"] / row["drafted"] if row["drafted"] else None
        if row["loop"]:
            return {"incumbent_before": before, "incumbent_after": before,
                    "probe": probe, "accepted_fraction": alpha,
                    "reward": None, "adopted": False, "loop": True}
        if row["drafted"]:
            self.recent.append((row["accepted"], row["drafted"]))
        reward = rate(row) / self.baseline[row["length_target"]]
        comparator = statistics.mean(self.scores[before][-8:])
        self.scores[label].append(reward)
        adopted = probe and reward > comparator * (1 + KEEP_GAIN)
        if adopted:
            self.incumbent = label
        return {
            "incumbent_before": before,
            "incumbent_after": self.incumbent,
            "probe": probe,
            "accepted_fraction": alpha,
            "reward": reward,
            "comparator": comparator,
            "adopted": adopted,
            "loop": False,
        }


def replay(root):
    audited = audit_grid(root)
    if audited["status"] != "measured-fixed-cutoff-only":
        raise ValueError("fixed C grid has not been audited")
    pairs = groups(root)
    baseline, scores, incumbent = calibrate(pairs)
    initial = incumbent
    controller = Controller(baseline, scores, incumbent)
    traces, selected, off, calibrated = [], [], [], []
    by_scenario = {
        scenario: {"selected": [], "off": [], "calibrated": []} for scenario in EVALUATION
    }
    excluded = []
    for (scenario, turn), rows in pairs.items():
        if scenario not in EVALUATION:
            continue
        label, probe = controller.choose()
        # Only the chosen native outcome reaches the learner.
        observation = controller.observe(label, probe, rows[label])
        remove = any(row["loop"] for row in rows.values())
        traces.append({
            "scenario": scenario, "turn": turn,
            "length_target": rows[label]["length_target"],
            "chosen": label, **observation,
            "matched_exclusion": remove,
        })
        if remove:
            excluded.append({"scenario": scenario, "turn": turn})
            continue
        selected.append(rows[label])
        off.append(rows["off"])
        calibrated.append(rows[initial])
        by_scenario[scenario]["selected"].append(rows[label])
        by_scenario[scenario]["off"].append(rows["off"])
        by_scenario[scenario]["calibrated"].append(rows[initial])
    if not selected:
        raise ValueError("no matched code requests survived the frozen loop screen")
    summary = {
        "selected": pool(selected),
        "off": pool(off),
        "calibrated_fixed": pool(calibrated),
    }
    for control in ("off", "calibrated_fixed"):
        summary[f"selected_vs_{control}_percent"] = 100 * (
            summary["selected"]["native_tokens_per_second"]
            / summary[control]["native_tokens_per_second"] - 1
        )
    scenarios = {}
    for scenario, rows in by_scenario.items():
        if not rows["selected"]:
            continue
        rates = {name: pool(values)["native_tokens_per_second"] for name, values in rows.items()}
        scenarios[str(scenario)] = {
            "rates": rates,
            "selected_vs_off_percent": 100 * (rates["selected"] / rates["off"] - 1),
            "selected_vs_calibrated_percent": 100 * (
                rates["selected"] / rates["calibrated"] - 1
            ),
        }
    return {
        "status": "offline-adaptive-replay",
        "scope": "independent native requests; offline full-information calibration is excluded from evaluation; replayed state persists across held-out requests; no live cutoff switching, CPU policy cost or continuous-KV claim",
        "policy": {
            "arms": ARMS,
            "calibration_scenarios": CALIBRATION,
            "evaluation_scenarios": EVALUATION,
            "probe_every": PROBE_EVERY,
            "acceptance_low": ACCEPT_LOW,
            "acceptance_high": ACCEPT_HIGH,
            "keep_gain": KEEP_GAIN,
            "reward": "chosen-arm output tokens / native request seconds, divided by calibration C=0 rate for this prompt length",
            "policy_cpu_time_included": False,
        },
        "initial_calibrated_arm": initial,
        "calibration_reference_rates": {str(length): value for length, value in baseline.items()},
        "calibration_scores": {
            label: statistics.mean(values) for label, values in scores.items()
        },
        "evaluation": summary,
        "by_scenario": scenarios,
        "matched_loop_exclusions": excluded,
        "trace": traces,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "initial_calibrated_arm": result["initial_calibrated_arm"],
        "selected_vs_off_percent": result["evaluation"]["selected_vs_off_percent"],
    }))


if __name__ == "__main__":
    main()
