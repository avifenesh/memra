"""Data-edge controls for the existing support census; no support promotion."""

import json
from pathlib import Path
import shutil
import unittest

import support_record_inputs as data
import validation_plan as vp


ROOT = Path(__file__).resolve().parent.parent
GATE = 'research/support-fixture/gates.txt'
LOCK = 'research/support-fixture/artifact.lock'
TINY = 'research/support-fixture/tiny-gate.tsv'


def metadata(path=GATE):
    return ('[[record]]\nid="fixture"\n[record.gates]\nConfig="passed"\n'
            '[record.evidence]\nConfig=[' + json.dumps(path) + ']\n')


class SupportRecordDataInputs(unittest.TestCase):
    def setUp(self):
        import test_validation_plan as fixtures
        self.fixture = fixtures.ValidationPlanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        for name in vp.TOOL_CONTRACTS['support-records']['inputs']:
            if name != data.RECORDS:
                self.fixture.put(name, (ROOT / name).read_text())
        self.fixture.put(data.RECORDS, metadata())
        self.fixture.put(GATE, 'Config=passed\n')
        self.base = self.fixture.commit()

    def plan(self, path):
        return vp.make_plan([path], vp.Tree(self.repo, 'HEAD'), vp.Tree(self.repo, 'HEAD'))

    def test_current_record_data_has_seven_required_and_fourteen_potential_paths(self):
        resolved = data.resolve(vp.LocalTree(ROOT))
        self.assertEqual(len(resolved['required']), 7)
        self.assertEqual(len(resolved['optional']), 14)
        self.assertIn('research/modelplan-onboarding-hy3-20260830/tiny/artifact.lock', resolved['optional'])

    def test_required_gate_and_both_potential_sidecars_select_the_contract(self):
        for path in (GATE, LOCK, TINY):
            with self.subTest(path=path):
                plan = self.plan(path)
                self.assertEqual(plan['mode'], 'scoped')
                self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])
                self.assertFalse(any(plan['jobs'].values()))
                self.assertEqual(plan['cpu_contracts'][0]['cpu'],
                                 ['tools/unittest-floor.sh', 'tools', 'test_check_support_states.py', '21'])
                self.assertEqual(plan['cpu_contracts'][0]['native'], [])
                self.assertFalse(plan['native']['qualification'])

    def test_creation_and_deletion_of_each_sidecar_keep_selection(self):
        before = self.base
        for path in (LOCK, TINY):
            for present in (True, False):
                with self.subTest(path=path, present=present):
                    if present:
                        self.fixture.put(path, 'family=fixture\n')
                    else:
                        (self.repo / path).unlink()
                    after = self.fixture.commit()
                    plan = vp.event_plan(self.repo, 'push', '', before, after)
                    before = after
                    self.assertEqual(plan['changed'], [path])
                    self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])

    def test_neither_optional_alternative_is_required_for_execution_selection(self):
        self.assertFalse((self.repo / LOCK).exists())
        self.assertFalse((self.repo / TINY).exists())
        self.assertEqual(vp.cpu_contract_names(self.repo, 'support-records'), ['support-records'])
        self.fixture.put(LOCK, 'family=fixture\n')
        self.assertEqual(vp.cpu_contract_names(self.repo, 'support-records'), ['support-records'])
        self.fixture.put(TINY, 'family\tfixture\n')
        self.assertEqual(vp.cpu_contract_names(self.repo, 'support-records'), ['support-records'])

    def test_missing_required_evidence_retains_named_refusal(self):
        (self.repo / GATE).unlink()
        with self.assertRaisesRegex(vp.Refused, 'selected contract input is missing: support-records: ' + GATE):
            vp.cpu_contract_names(self.repo, 'support-records')

    def test_before_and_after_record_references_both_keep_reader_edges(self):
        new = 'research/new-support-fixture/gates.txt'
        self.fixture.put(data.RECORDS, metadata(new))
        self.fixture.put(new, 'Config=passed\n')
        after = self.fixture.commit()
        for path in (GATE, LOCK, TINY, new, 'research/new-support-fixture/artifact.lock',
                     'research/new-support-fixture/tiny-gate.tsv'):
            with self.subTest(path=path):
                plan = vp.make_plan([path], vp.Tree(self.repo, self.base), vp.Tree(self.repo, after))
                self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])

    def test_changed_or_partial_reader_expands_checks(self):
        self.fixture.put('tools/check-support-states.py', '# changed reader\n')
        self.fixture.commit()
        self.assertEqual(self.plan(GATE)['mode'], 'full')
        (self.repo / 'tools/check-support-states.py').unlink()
        self.fixture.commit()
        self.assertEqual(self.plan(GATE)['mode'], 'full')

    def test_malformed_or_unknown_metadata_expands_checks(self):
        for bad in ('record = "not a list"\n', '[[record]]\ngates=true\nevidence={}\n',
                    '[[record]]\n[record.gates]\nUnknown="passed"\n[record.evidence]\nUnknown=[]\n',
                    '[[record]]\n[record.gates]\nConfig="passed"\n[record.evidence]\nConfig="not a list"\n',
                    'not valid toml [', metadata().replace('passed', 'unknown'),
                    metadata().replace('passed', 'pending')):
            with self.subTest(bad=bad):
                self.fixture.put(data.RECORDS, bad)
                self.fixture.commit()
                self.assertEqual(self.plan(GATE)['mode'], 'full')

    def test_noncanonical_or_escaping_evidence_expands_checks(self):
        for path in ('../outside/gates.txt', '/outside/gates.txt', 'research/a/../gates.txt',
                     'research//gates.txt', './research/gates.txt', 'research\\gates.txt',
                     'research/with\nnewline/gates.txt'):
            with self.subTest(path=path):
                self.fixture.put(data.RECORDS, metadata(path))
                self.fixture.commit()
                self.assertEqual(self.plan(GATE)['mode'], 'full')

    def test_only_exact_known_ci_token_excludes_filesystem_reads(self):
        self.fixture.put(data.RECORDS, metadata('ci:verify-tiny'))
        self.fixture.commit()
        self.assertEqual(data.resolve(vp.LocalTree(self.repo)), {'required': [], 'optional': []})
        self.fixture.put(data.RECORDS, metadata('ci:unmodelled'))
        self.fixture.commit()
        self.assertEqual(self.plan(GATE)['mode'], 'full')

    def test_required_and_optional_symlink_reads_expand_checks(self):
        for path in (GATE, LOCK, TINY):
            with self.subTest(path=path):
                target = self.repo / path
                if target.exists():
                    target.unlink()
                target.symlink_to('another-file')
                self.fixture.commit()
                self.assertEqual(self.plan(path)['mode'], 'full')
                target.unlink()
                if path == GATE:
                    self.fixture.put(path, 'Config=passed\n')
                self.fixture.commit()

    def test_evidence_parent_directory_symlink_expands_checks(self):
        shutil.rmtree(self.repo / 'research/support-fixture')
        (self.repo / 'research/support-fixture').symlink_to('other-fixture', target_is_directory=True)
        self.fixture.commit()
        self.assertEqual(self.plan(LOCK)['mode'], 'full')

    def test_derived_required_missing_after_deletion_still_selects_contract(self):
        (self.repo / GATE).unlink()
        after = self.fixture.commit()
        plan = vp.event_plan(self.repo, 'push', '', self.base, after)
        self.assertEqual([c['id'] for c in plan['cpu_contracts']], ['support-records'])

    def test_new_record_evidence_derives_exact_paths_without_wildcards(self):
        self.fixture.put(data.RECORDS, metadata() + '\n' + metadata('research/second/gates.txt'))
        self.fixture.commit()
        resolved = data.resolve(vp.LocalTree(self.repo))
        self.assertEqual(resolved['required'], sorted([GATE, 'research/second/gates.txt']))
        self.assertEqual(len(resolved['optional']), 4)
        self.assertFalse(any('*' in p for p in resolved['required'] + resolved['optional']))


if __name__ == '__main__':
    unittest.main()
