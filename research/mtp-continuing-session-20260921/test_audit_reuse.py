import csv
import tempfile
import unittest
from pathlib import Path

from audit_reuse import audit_reuse


class ReuseAuditTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for turn in range(1, 9):
            prompt = list(range(100 * turn + 20))
            (root / f'turn-{turn}.prompt.ids').write_text('\n'.join(map(str, prompt)))
            (root / f'turn-{turn}.output.ids').write_text('1\n2\n')
            cached = (turn - 1) * 100
            rows.append(dict(turn=turn, prompt_tokens=len(prompt), output_tokens=2,
                             cached_tokens=cached, new_input_tokens=len(prompt) - cached,
                             checkpoint_tokens=turn * 100, resumed='true' if cached else 'false'))
        return rows

    def save(self, root, rows):
        with (root / 'turns.tsv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_prefix_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save(root, self.fixture(root))
            result = audit_reuse(root)
            self.assertEqual(result['cached_tokens'], 2800)
            self.assertEqual(result['new_input_tokens'], 960)

    def test_reject_cold_fallback_even_when_totals_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            rows[3].update(cached_tokens=0, new_input_tokens=420, resumed='false')
            self.save(root, rows)
            with self.assertRaisesRegex(ValueError, 'did not reuse'):
                audit_reuse(root)

    def test_reject_false_hit_on_changed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save(root, self.fixture(root))
            path = root / 'turn-3.prompt.ids'
            ids = path.read_text().splitlines()
            ids[70] = '9999'
            path.write_text('\n'.join(ids))
            with self.assertRaisesRegex(ValueError, 'did not reuse'):
                audit_reuse(root)

    def test_reject_partial_rebuild_disguised_as_a_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.fixture(root)
            rows[3].update(cached_tokens=1, new_input_tokens=419)
            self.save(root, rows)
            with self.assertRaisesRegex(ValueError, 'did not reuse'):
                audit_reuse(root)


if __name__ == '__main__':
    unittest.main()
