"""Check that reversed-order disagreement cannot hide a prose loss."""

import json
from pathlib import Path
import tempfile
import unittest

import score_prose


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class ProseOrderTest(unittest.TestCase):
    def test_disagreement_counts_as_tie_and_checks_model_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packets = root / "packets"
            judged = root / "judged"
            packets.mkdir()
            judged.mkdir()
            config = root / "config.json"
            write(config, {
                "model_id": "pinned-test-judge",
                "region": "test-region",
                "template_sha256": score_prose.TEMPLATE_SHA,
                "input_usd_per_million_budget": 1,
                "output_usd_per_million_budget": 1,
                "total_usd_cap": 1,
                "spend_basis": "live_global_standard_quote",
            })
            write(judged / "pricing.json", {
                "model_id": "pinned-test-judge",
                "region": "test-region",
                "global_standard": {
                    kind: {"usd_per_million": 1}
                    for kind in ("input", "output")
                },
            })
            write(judged / "profile.json", {
                "model_id": "pinned-test-judge",
                "status": "ACTIVE",
                "foundation_model_arns": [
                    "synthetic-model-identity",
                ],
            })
            source = []
            answers = []
            for turn in range(1, 9):
                for order in (0, 1):
                    source.append({
                        "conversation": 0, "turn": turn,
                        "candidate": "learned", "control": "fixed",
                        "order": order, "response_a": (
                            "learned" if order == 0 else "fixed"
                        ),
                        "prompt_sha256": f"{turn:02d}{order}" + "a" * 61,
                    })
                    answers.append({
                        "packet_index": len(answers),
                        "prompt_sha256": source[-1]["prompt_sha256"],
                        "model_id": "pinned-test-judge",
                        "provider_response_id": f"r-{turn}-{order}",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "choice": "A+" if order == 0 else "A+",
                    })
            packet_path = packets / "packets.jsonl"
            result_path = judged / "results.jsonl"
            packet_path.write_text("".join(
                json.dumps(item) + "\n" for item in source
            ))
            result_path.write_text("".join(
                json.dumps(item) + "\n" for item in answers
            ))
            write(packets / "manifest.json", {
                "schema": 1, "phase": "validation",
                "template_sha256": score_prose.TEMPLATE_SHA,
                "packets_sha256": score_prose.sha(packet_path),
                "packet_count": len(source),
                "pairs": [["learned", "fixed"]],
                "arms_sha256": "b" * 64,
                "workloads_sha256": "c" * 64,
            })
            write(judged / "manifest.json", {
                "schema": 1,
                "packets_sha256": score_prose.sha(packet_path),
                "config_sha256": score_prose.sha(config),
                "results_sha256": score_prose.sha(result_path),
                "model_id": "pinned-test-judge",
                "pricing_sha256": score_prose.sha(
                    judged / "pricing.json"
                ),
                "profile_sha256": score_prose.sha(
                    judged / "profile.json"
                ),
                "prior_judge_manifest_sha256": None,
                "cumulative_usage": {
                    "input_tokens": 1600,
                    "output_tokens": 320,
                },
                "cumulative_quoted_spend_usd": 0.00192,
            })
            result = score_prose.score(packets, judged, config)
            comparison = result["comparisons"]["learned::vs::fixed"]
            self.assertEqual(comparison["wins"], 0)
            self.assertEqual(comparison["ties"], 8)
            self.assertEqual(comparison["reverse_order_disagreements"], 8)
            self.assertEqual(comparison["point_win_fraction"], 0.5)
            answers[0]["model_id"] = "unmatched-judge"
            result_path.write_text("".join(
                json.dumps(item) + "\n" for item in answers
            ))
            write(judged / "manifest.json", {
                "schema": 1,
                "packets_sha256": score_prose.sha(packet_path),
                "config_sha256": score_prose.sha(config),
                "results_sha256": score_prose.sha(result_path),
                "model_id": "pinned-test-judge",
                "pricing_sha256": score_prose.sha(
                    judged / "pricing.json"
                ),
                "profile_sha256": score_prose.sha(
                    judged / "profile.json"
                ),
                "prior_judge_manifest_sha256": None,
                "cumulative_usage": {
                    "input_tokens": 1600,
                    "output_tokens": 320,
                },
                "cumulative_quoted_spend_usd": 0.00192,
            })
            with self.assertRaises(ValueError):
                score_prose.score(packets, judged, config)


if __name__ == "__main__":
    unittest.main()
