"""Catch a phase or judge pin drifting between unrun pipeline stages."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
TRAIN = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
VALIDATION = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
FULL = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
TEMPLATE = "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"


def constants(name):
    tree = ast.parse((ROOT / name).read_text())
    result = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            result[node.targets[0].id] = node.value.value
    return result


class PhaseContractTest(unittest.TestCase):
    def test_training_validation_final_pins_agree(self):
        for name in (
            "eval.py", "collect_prose.py",
            "measurement_rows_prose.py", "fit_shared.py",
            "arms_shared.py", "seal_training.py",
            "replay_training.py", "pilot_depth.py",
            "supervise_v12.py",
        ):
            with self.subTest(name=name):
                self.assertEqual(constants(name)["TRAIN_SHA"], TRAIN)
        for name in (
            "eval.py", "quality_tasks.py", "prose_packets.py",
            "select_shared.py", "seal_evaluation.py",
            "supervise_v12.py",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    constants(name)["VALIDATION_SHA"], VALIDATION,
                )
        for name in (
            "eval.py", "collect_prose.py",
            "measurement_rows_prose.py", "fit_shared.py",
            "arms_shared.py", "quality_tasks.py",
            "prose_packets.py", "score_final.py",
            "seal_training.py", "replay_training.py",
            "seal_evaluation.py", "supervise_v12.py",
        ):
            with self.subTest(name=name):
                self.assertEqual(constants(name)["FULL_SHA"], FULL)

    def test_prose_judge_template_pin_agrees(self):
        for name in (
            "prose_packets.py", "score_prose.py",
            "judge_bedrock.py", "supervise_v12.py",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    constants(name)["TEMPLATE_SHA"], TEMPLATE,
                )


if __name__ == "__main__":
    unittest.main()
