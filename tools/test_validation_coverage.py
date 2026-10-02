#!/usr/bin/env python3
import hashlib
from pathlib import Path
import tempfile
import unittest

import validation_coverage as vc


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memra-coverage-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'harness.py').write_text('assert actual == expected\n')
        self.digest = hashlib.sha256((self.root / 'harness.py').read_bytes()).hexdigest()
        self.context = {'model': 'fixture', 'hardware': 'cpu'}

    def case(self, name, edges, cost=1, **kwargs):
        return dict(id=name, covers=edges, cost=cost, inputs={'harness.py': self.digest},
                    scope=self.context, **kwargs)

    def plan(self, edges, tests):
        return vc.select(edges, tests, self.context, self.root)

    def test_one_composite_test_replaces_five_redundant_boots(self):
        edges = ['cache', 'usage', 'isolation', 'metrics', 'concurrency']
        cases = [self.case(edge, [edge]) for edge in edges]
        cases.append(self.case('composite-cache-meter', edges, cost=2))
        p = self.plan(edges, cases)
        self.assertEqual(p['selected'], ['composite-cache-meter'])
        self.assertEqual(p['uncovered'], [])
        self.assertFalse(p['qualification'])

    def test_mandatory_regression_cannot_be_optimized_away(self):
        p = self.plan(['a', 'b'], [self.case('broad', ['a', 'b']),
                                   self.case('past-failure', ['a'], mandatory=True)])
        self.assertEqual(p['selected'], ['broad', 'past-failure'])

    def test_controls_are_selected_with_their_positive(self):
        p = self.plan(['a'], [self.case('positive', ['a'], controls=['red']), self.case('red', [])])
        self.assertEqual(p['selected'], ['positive', 'red'])

    def test_uncovered_edge_refuses_reduced_battery(self):
        p = self.plan(['a', 'b'], [self.case('one', ['a'])])
        self.assertEqual(p['decision'], 'expand')
        self.assertEqual(p['uncovered'], ['b'])

    def test_different_model_or_hardware_cannot_cover_an_edge(self):
        case = self.case('other', ['a'])
        case['scope'] = {'model': 'other-model', 'hardware': 'cpu'}
        self.assertEqual(self.plan(['a'], [case])['decision'], 'expand')

    def test_native_identity_values_must_be_present_and_nonempty(self):
        keys = ('model', 'artifact', 'hardware', 'numeric_program')
        for value in (None, '', ' ', False, 0, [], {}):
            for absent in (False, True):
                with self.subTest(value=value, absent=absent):
                    case = self.case('native', ['a'], kind='gpu')
                    case['scope'] = dict.fromkeys(keys, value)
                    context = {} if absent else dict(case['scope'])
                    self.assertEqual(vc.select(['a'], [case], context, self.root)['decision'], 'expand')

    def test_changed_harness_invalidates_coverage_claim(self):
        (self.root / 'harness.py').write_text('pass\n')
        p = self.plan(['a'], [self.case('stale', ['a'])])
        self.assertEqual(p['decision'], 'expand')
        self.assertIn('coverage input changed', str(p['ineligible']))

    def test_missing_control_refuses(self):
        with self.assertRaisesRegex(ValueError, 'unknown control'):
            self.plan(['a'], [self.case('positive', ['a'], controls=['missing'])])

    def test_control_cycle_refuses(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.plan(['a'], [self.case('one', ['a'], controls=['two']), self.case('two', [], controls=['one'])])

    def test_changed_control_invalidates_positive(self):
        bad = self.case('red', []); bad['inputs'] = {'missing.py': self.digest}
        p = self.plan(['a'], [self.case('positive', ['a'], controls=['red']), bad])
        self.assertEqual(p['decision'], 'expand')

    def test_unavailable_mandatory_test_refuses_even_with_broad_cover(self):
        bad = self.case('regression', ['a'], mandatory=True); bad['scope'] = {'model': 'other'}
        p = self.plan(['a'], [bad, self.case('broad', ['a'])])
        self.assertEqual(p['decision'], 'expand')

    def test_cost_and_control_cost_influence_selection(self):
        p = self.plan(['a', 'b'], [self.case('expensive', ['a', 'b'], cost=10),
                                   self.case('a', ['a']), self.case('b', ['b'])])
        self.assertEqual(p['selected'], ['a', 'b'])

    def test_ties_are_deterministic(self):
        for cases in ([self.case('z', ['a']), self.case('a', ['a'])],
                      [self.case('a', ['a']), self.case('z', ['a'])]):
            self.assertEqual(self.plan(['a'], cases)['selected'], ['a'])

    def test_invalid_cost_and_duplicate_test_refuse(self):
        for cost in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):self.plan(['a'], [self.case('bad', ['a'], cost)])
        with self.assertRaises(ValueError):self.plan(['a'], [self.case('same', ['a']), self.case('same', ['a'])])

    def test_pass_requires_every_edge_assertion_not_just_exit_success(self):
        p = self.plan(['a', 'b'], [self.case('composite', ['a', 'b'])])
        result = {'composite': dict(contract_id=p['contract_id'], status='passed', context=self.context, executed=2, skipped=0, edges={'a': 'passed'})}
        with self.assertRaisesRegex(ValueError, 'explicit passing assertion'):
            vc.validate_results(p, result, self.context, self.root)
        result['composite']['edges']['b'] = 'passed'
        self.assertEqual(vc.validate_results(p, result, self.context, self.root)['status'], 'passed')

    def test_empty_skipped_wrong_context_and_failed_runs_refuse(self):
        p = self.plan(['a'], [self.case('one', ['a'])])
        for field, value in [('executed', 0), ('skipped', 1), ('status', 'failed'), ('context', {})]:
            result = dict(contract_id=p['contract_id'], status='passed', context=self.context, executed=1, skipped=0, edges={'a': 'passed'})
            result[field] = value
            with self.assertRaises(ValueError):vc.validate_results(p, {'one': result}, self.context, self.root)

    def test_selected_red_control_must_pass(self):
        p = self.plan(['a'], [self.case('one', ['a'], controls=['red']), self.case('red', [])])
        good = dict(contract_id=p['contract_id'], status='passed', context=self.context, executed=1, skipped=0, edges={'a': 'passed'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            vc.validate_results(p, {'one': good}, self.context, self.root)
        bad = dict(good, status='failed')
        with self.assertRaisesRegex(ValueError, 'failed'):
            vc.validate_results(p, {'one': good, 'red': bad}, self.context, self.root)

    def test_no_edges_is_not_a_pass_receipt(self):
        p = self.plan([], [])
        self.assertEqual(p['decision'], 'no-change')
        with self.assertRaises(ValueError):vc.validate_results(p, {}, self.context, self.root)

    def test_native_coverage_needs_all_identity_axes(self):
        test = self.case('native', ['a'], kind='gpu')
        self.assertEqual(self.plan(['a'], [test])['decision'], 'expand')

    def test_boolean_counter_is_not_execution_evidence(self):
        p = self.plan(['a'], [self.case('one', ['a'])])
        r = dict(contract_id=p['contract_id'], status='passed', context=self.context, executed=True, skipped=False, edges={'a': 'passed'})
        with self.assertRaises(ValueError):vc.validate_results(p, {'one': r}, self.context, self.root)

    def test_new_harness_contract_rejects_old_pass(self):
        old = self.plan(['a'], [self.case('one', ['a'])])
        result = dict(contract_id=old['contract_id'], status='passed', context=self.context,
                      executed=1, skipped=0, edges={'a': 'passed'})
        (self.root / 'harness.py').write_text('assert changed == expected\n')
        self.digest = hashlib.sha256((self.root / 'harness.py').read_bytes()).hexdigest()
        new = self.plan(['a'], [self.case('one', ['a'])])
        self.assertNotEqual(old['contract_id'], new['contract_id'])
        with self.assertRaisesRegex(ValueError, 'different source/coverage contract'):
            vc.validate_results(new, {'one': result}, self.context, self.root)

    def test_source_change_after_selection_refuses_admission(self):
        p = self.plan(['a'], [self.case('one', ['a'])])
        result = dict(contract_id=p['contract_id'], status='passed', context=self.context,
                      executed=1, skipped=0, edges={'a': 'passed'})
        (self.root / 'harness.py').write_text('changed after selection\n')
        with self.assertRaisesRegex(ValueError, 'changed before result admission'):
            vc.validate_results(p, {'one': result}, self.context, self.root)

    def test_red_controls_own_assertions_cannot_hide_behind_aggregate_pass(self):
        p = self.plan(['a'], [self.case('one', ['a'], controls=['red']), self.case('red', ['red-control'])])
        def result(edges):
            return dict(contract_id=p['contract_id'], status='passed', context=self.context,
                        executed=1, skipped=0, edges=edges)
        with self.assertRaisesRegex(ValueError, 'red/red-control'):
            vc.validate_results(p, {'one': result({'a': 'passed'}),
                                   'red': result({'red-control': 'failed'})}, self.context, self.root)

    def test_tampered_plan_cannot_remove_a_required_edge(self):
        p = self.plan(['a'], [self.case('one', ['a'])])
        p['contract']['required_edges'] = []
        with self.assertRaisesRegex(ValueError, 'changed source/coverage contract'):
            vc.validate_results(p, {}, self.context, self.root)


if __name__ == '__main__':
    unittest.main()
