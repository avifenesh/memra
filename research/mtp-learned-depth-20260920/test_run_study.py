import tempfile
import unittest
from pathlib import Path

import run_study as study


class ReceiptTests(unittest.TestCase):
    def test_partition_preserves_original_schedule_and_seed_offsets(self):
        from collections import Counter
        whole = study.select_cycles()
        parts = study.select_cycles(start=0, stop=3) + study.select_cycles(start=3, stop=8)
        self.assertEqual(parts, whole)
        self.assertEqual([i for i, _ in whole], list(range(8)))
        for position in range(4):
            self.assertEqual(set(Counter(order[position] for _, order in whole).values()), {2})
        transitions = Counter((order[i], order[i+1]) for _, order in whole for i in range(3))
        self.assertEqual(len(transitions), 12)
        self.assertEqual(set(transitions.values()), {2})
        for start, stop in [(-1, 8), (0, 9), (4, 4), (5, 2)]:
            with self.assertRaises(ValueError):
                study.select_cycles(start=start, stop=stop)
        with self.assertRaises(ValueError):
            study.select_cycles(gate=True, start=1)

    def test_incomplete_metrics_are_not_a_success(self):
        for text in ['no metrics', 'PERF_METRICS_START\n{}']:
            with self.assertRaises(ValueError):
                study.metrics_from_log(text)
        result = study.metrics_from_log(
            'diagnostic\nPERF_METRICS_START\n'
            '{"scenarios":{"native":{"tokens":10}}}\nPERF_METRICS_END\n'
        )
        self.assertEqual(result['native']['tokens'], 10)

    def test_pooled_denominator_and_losing_pairs_are_preserved(self):
        rows = []
        for seed, learned_tokens, seconds in [(1, 5, 1), (2, 1100, 100)]:
            for arm, tokens in [('native', 10 * seconds), ('learned', learned_tokens)]:
                rows.append(dict(seed=seed, arm=arm, tokens=tokens,
                                 elapsed_s=seconds, e2e_tok_s=tokens / seconds))
        summary = study.summarize(rows)
        self.assertEqual(summary['pooled_e2e_tok_s']['native'], 10)
        self.assertAlmostEqual(summary['pooled_e2e_tok_s']['learned'], 1105 / 101)
        self.assertEqual(summary['wins'], 1)
        self.assertEqual(summary['pairs'], 2)
        self.assertEqual(summary['worst_pair_percent'], -50)

    def test_late_turn_divergence_and_empty_output_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ['native', 'measured']:
                path = root / f'1-{arm}'
                path.mkdir()
                for turn in range(1, 9):
                    (path / f'turn-{turn}.output.ids').write_text('12\n24\n')
                    (path / f'turn-{turn}.prompt.ids').write_text('1\n2\n3\n')
            study.audit_control_ids(root, 1, ['measured'])
            (root / '1-measured/turn-7.output.ids').write_text('12\n25\n')
            with self.assertRaisesRegex(ValueError, 'turn=7'):
                study.audit_control_ids(root, 1, ['measured'])

    def test_prompt_divergence_fails_even_if_output_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ['native', 'measured']:
                path = root / f'1-{arm}'
                path.mkdir()
                (path / 'turn-1.prompt.ids').write_text('1\n2\n')
                (path / 'turn-1.output.ids').write_text('12\n24\n')
            (root / '1-measured/turn-1.prompt.ids').write_text('1\n3\n')
            with self.assertRaisesRegex(ValueError, 'prompt identity'):
                study.audit_control_ids(root, 1, ['measured'], turns=1)

    def test_second_conversation_is_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ['native', 'measured']:
                path = root / f'1-{arm}'
                path.mkdir()
                for turn in range(1, 17):
                    (path / f'turn-{turn}.output.ids').write_text('12\n24\n')
                    (path / f'turn-{turn}.prompt.ids').write_text('1\n2\n3\n')
            study.audit_control_ids(root, 1, ['measured'], turns=16)
            (root / '1-measured/turn-15.output.ids').write_text('12\n25\n')
            with self.assertRaisesRegex(ValueError, 'turn=15'):
                study.audit_control_ids(root, 1, ['measured'], turns=16)
            for arm in ['native', 'measured']:
                (root / f'1-{arm}/turn-1.output.ids').write_text('')
            with self.assertRaisesRegex(ValueError, 'empty native'):
                study.audit_control_ids(root, 1, ['measured'])


if __name__ == '__main__':
    unittest.main()
