"""Exercise the cutoff controller's bidirectional and cost decisions."""

import unittest

from offline_adaptive import ARMS, Controller, calibrate


def row(tokens_per_second, accepted=3, drafted=4, length=256, loop=False):
    return {
        "output_tokens": 100,
        "elapsed_s": 100 / tokens_per_second,
        "accepted": accepted,
        "drafted": drafted,
        "length_target": length,
        "loop": loop,
        "format": {"requested_format_covered": True},
    }


class ConfidenceControllerTest(unittest.TestCase):
    def test_low_acceptance_probes_higher_c_and_adopts_only_on_wall_gain(self):
        scores = {arm: [1.0] * 8 for arm in ARMS}
        learner = Controller({256: 100.0}, scores, "c015")
        learner.requests = 5
        learner.recent.extend([(1, 2)] * 4)
        chosen, probe = learner.choose()
        self.assertEqual((chosen, probe), ("c030", True))
        decision = learner.observe(chosen, probe, row(110, accepted=1, drafted=2))
        self.assertTrue(decision["adopted"])
        self.assertEqual(learner.incumbent, "c030")

        learner.requests = 11
        learner.recent.clear()
        learner.recent.extend([(9, 10)] * 4)
        chosen, probe = learner.choose()
        self.assertEqual((chosen, probe), ("c015", True))
        decision = learner.observe(chosen, probe, row(95, accepted=9, drafted=10))
        self.assertFalse(decision["adopted"])
        self.assertEqual(learner.incumbent, "c030")

    def test_chosen_loop_does_not_train_the_controller(self):
        scores = {arm: [1.0] * 8 for arm in ARMS}
        learner = Controller({256: 100.0}, scores, "off")
        before = {arm: list(values) for arm, values in learner.scores.items()}
        decision = learner.observe("off", False, row(100, loop=True))
        self.assertTrue(decision["loop"])
        self.assertEqual(learner.scores, before)
        self.assertEqual(len(learner.recent), 0)

    def test_initial_cutoff_uses_disjoint_full_arm_calibration(self):
        pairs = {}
        for scenario in (0, 1):
            for turn, length in enumerate((256, 1024, 4096, 16384), 1):
                base = 100 / (1 + turn)
                pairs[scenario, turn] = {
                    "off": row(base, length=length),
                    "c015": row(base * 1.05, length=length),
                    "c030": row(base * 0.98, length=length),
                    "c030zero": row(base * 0.97, length=length),
                }
        baseline, scores, initial = calibrate(pairs)
        self.assertEqual(initial, "c015")
        self.assertEqual(set(baseline), {256, 1024, 4096, 16384})
        self.assertGreater(sum(scores["c015"]), sum(scores["off"]))


if __name__ == "__main__":
    unittest.main()
