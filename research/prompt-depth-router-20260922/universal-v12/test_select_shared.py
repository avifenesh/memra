"""Guard against a field-specific winner being called one shared policy."""

import json
from pathlib import Path
import tempfile
import unittest

import select_shared


FIXED = sorted(select_shared.REQUIRED_FIXED)
LEARNED = ("joint-shared", "joint-code-only")
NOOPS = {
    "joint-shared": "joint-noop-shared",
    "joint-code-only": "joint-noop-code-only",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixture(root, prose_shared=105.0, judge="f" * 64):
    arms = root / "validation-arms.json"
    specs = [
        {"label": label, "role": "fixed"}
        for label in FIXED
    ]
    for label, policy in zip(LEARNED, ("a" * 64, "b" * 64)):
        specs.extend((
            {
                "label": label, "role": "learned",
                "noop_label": NOOPS[label],
                "policy_sha256": policy,
                "selectable": True,
                "arm": "joint-ckd",
            },
            {
                "label": NOOPS[label], "role": "noop",
                "policy_sha256": policy,
            },
        ))
    write(arms, {
        "schema": 1, "phase": "validation",
        "source_manifest_sha256": "c" * 64,
        "model_manifest_sha256": "d" * 64,
        "arms": specs,
    })
    domains = {}
    for domain in select_shared.DOMAINS:
        rates = {label: 100.0 for label in FIXED}
        rates["fixed-k20-d2-c0"] = 102.0
        rates["joint-shared"] = (
            prose_shared if domain == "prose" else 105.0
        )
        rates["joint-code-only"] = (
            99.0 if domain == "prose" else 110.0
        )
        rates.update({label: 100.0 for label in NOOPS.values()})
        comparisons = {}
        for label in LEARNED:
            comparisons[label] = {
                "vs_own_noop": {
                    "status": "paired",
                    "delta_percent": rates[label] - 100.0,
                },
                "vs_each_eligible_fixed": {
                    fixed: {
                        "status": "paired",
                        "delta_percent":
                        100 * (rates[label] / rates[fixed] - 1),
                    }
                    for fixed in FIXED
                },
            }
        domains[domain] = {
            "arms": {
                label: {
                    "tokens": rate, "seconds": 1.0,
                    "tok_s": rate, "quality_eligible": True,
                }
                for label, rate in rates.items()
            },
            "eligible_fixed": FIXED,
            "comparisons": comparisons,
            "byte_identical_noops": list(NOOPS.values()),
        }
    domains["prose"]["judge_receipt_sha256"] = judge
    score = root / "validation-score.json"
    write(score, {
        "schema": 1, "phase": "validation",
        "arms_sha256": select_shared.sha(arms),
        "source_manifest_sha256": select_shared.VALIDATION_SHA,
        "model_manifest_sha256": "d" * 64,
        "quality_sha256": "e" * 64,
        "judge_config_sha256": "f" * 64,
        "domains": domains,
    })
    return score, arms


class SharedSelectionTest(unittest.TestCase):
    def test_prose_regression_excludes_faster_pooled_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            score, arms = fixture(Path(folder))
            selection, final = select_shared.choose(score, arms)
            self.assertEqual(selection["status"], "selected")
            self.assertEqual(
                selection["selected_policy"]["label"], "joint-shared",
            )
            self.assertEqual(
                selection["global_fixed"], "fixed-k20-d2-c0",
            )
            self.assertEqual(
                {row["label"] for row in final},
                {"joint-shared", "joint-noop-shared",
                 "fixed-k20-d2-c0", "fixed-k20-d3-c0"},
            )
            self.assertNotIn("chosen_by_domain", selection)

    def test_no_global_candidate_keeps_final_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            score, arms = fixture(Path(folder), prose_shared=99.0)
            selection, final = select_shared.choose(score, arms)
            self.assertEqual(selection["status"], "global-no-go")
            self.assertIsNone(selection["selected_policy"])
            self.assertEqual(final, [])

    def test_prose_judge_and_arm_lineage_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            score, arms = fixture(Path(folder), judge="")
            with self.assertRaises(ValueError):
                select_shared.choose(score, arms)
            score, arms = fixture(Path(folder))
            payload = json.loads(arms.read_text())
            payload["source_manifest_sha256"] = "0" * 64
            write(arms, payload)
            with self.assertRaises(ValueError):
                select_shared.choose(score, arms)


if __name__ == "__main__":
    unittest.main()
