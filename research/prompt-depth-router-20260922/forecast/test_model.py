import unittest

from model import Policy, State, fit_tree, format_map, future_kind, predict


class ForecastTests(unittest.TestCase):
    def test_future_targets_do_not_enter_prefix_features(self):
        prefix = b"Here is the implementation:\n"
        inputs = []
        labels = []
        for suffix in (b"```python\nreturn count + 1\nreturn count + 2\n```\n",
                       b"The implementation should preserve the original order."):
            state = State(1)
            state.observe(prefix)
            inputs.append(state.features())
            data = prefix + suffix
            labels.append(future_kind(data, format_map(data), len(prefix), len(data)))
        self.assertEqual(inputs[0], inputs[1])
        self.assertNotEqual(labels[0], labels[1])

    def test_fence_and_numeric_targets_are_separate_from_prompt_hint(self):
        data = b"Explanation.\n```python\nreturn total + count\n```\nMore prose.\n"
        start = data.index(b"return")
        self.assertEqual(future_kind(data, format_map(data), start, start + 20), 1)
        numbers = b"10 + 20 = 30; 40 + 50 = 90"
        self.assertEqual(future_kind(numbers, format_map(numbers), 0, len(numbers)), 2)
        text = b"```text\nthis is a quoted passage, not program syntax\n```\n"
        start = text.index(b"this")
        self.assertEqual(future_kind(text, format_map(text), start, start + 30), 3)

    def test_training_refuses_held_out_rows(self):
        row = {"features": (0,) * 16, "label": 0, "group": "held-a", "split": "heldout"}
        with self.assertRaises(ValueError):
            fit_tree([row])

    def test_tree_learns_a_threshold_and_abstains_on_impure_leaf(self):
        rows = [
            {"features": (value,) + (0,) * 15, "label": int(value >= 2),
             "group": "cal-a", "split": "calibration"}
            for value in range(4) for _ in range(80)
        ]
        tree = fit_tree(rows)
        self.assertEqual(predict(tree, (0,) * 16), 0)
        self.assertEqual(predict(tree, (3,) + (0,) * 15), 1)
        for row in rows:
            row["features"] = (0,) * 16
        self.assertEqual(predict(fit_tree(rows), (0,) * 16), 3)

    def test_observation_and_policy_are_incremental(self):
        text = b"A" * 200 + b"\n```python\nreturn 123\n```\n"
        whole = State(1)
        whole.observe(text)
        partial = State(1)
        for byte in text:
            partial.observe(bytes([byte]))
        self.assertEqual(whole.features(), partial.features())
        policy = Policy()
        self.assertEqual([policy.select(k) for k in (0, 0, 1, 1, 2, 3, 3)],
                         [3, 2, 2, 3, 4, 4, 3])


if __name__ == "__main__":
    unittest.main()
