#!/usr/bin/env python3
import hashlib
from pathlib import Path
import tempfile
import subprocess
import sys
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

    def test_typed_intake_fixture_cannot_be_covered_by_a_boolean_request(self):
        # These inputs engage different real fixture behavior, despite True == 1.
        def admit(request):
            return type(request['n']) is int and 1 <= request['n'] <= 4
        declared = dict(self.context, request={'n': 1})
        actual = dict(self.context, request={'n': True})
        self.assertTrue(admit(declared['request']))
        self.assertFalse(admit(actual['request']))
        case = self.case('intake', ['a']); case['scope'] = declared
        self.assertEqual(vc.select(['a'], [case], actual, self.root)['decision'], 'expand')

    def test_scope_distinguishes_nested_json_numeric_and_boolean_types(self):
        for declared, actual in ((1, True), (False, 0), (1, 1.0),
                                 ({'n': 1}, {'n': True}),
                                 ([1, {'strict': False}], [True, {'strict': 0}])):
            with self.subTest(declared=declared, actual=actual):
                case = self.case('typed', ['a']); case['scope'] = {'request': declared}
                context = dict(self.context, request=actual)
                self.assertEqual(vc.select(['a'], [case], context, self.root)['decision'], 'expand')

    def test_scope_null_requires_presence_but_matching_null_remains_valid(self):
        case = self.case('nullable', ['a']); case['scope'] = {'request': None}
        self.assertEqual(self.plan(['a'], [case])['decision'], 'expand')
        present = dict(self.context, request=None)
        self.assertEqual(vc.select(['a'], [case], present, self.root)['decision'], 'scoped')

    def test_changed_execution_and_receipt_contexts_cannot_alias_json_types(self):
        for declared, actual in (({'n': 1}, {'n': True}),
                                 ({'flags': [False]}, {'flags': [0]}),
                                 ({'temperature': 1.0}, {'temperature': 1})):
            with self.subTest(declared=declared, actual=actual):
                context = dict(self.context, request=declared)
                changed = dict(self.context, request=actual)
                case = self.case('one', ['a']); case['scope'] = context
                plan = vc.select(['a'], [case], context, self.root)
                result = {'one': dict(contract_id=plan['contract_id'], status='passed',
                                      context=changed, executed=1, skipped=0, edges={'a': 'passed'})}
                with self.assertRaisesRegex(ValueError, 'context changed'):
                    vc.validate_results(plan, result, changed, self.root)
                with self.assertRaisesRegex(ValueError, 'mismatched test result'):
                    vc.validate_results(plan, result, context, self.root)

    def test_unavailable_null_scoped_mandatory_control_cannot_use_broad_cover(self):
        control = self.case('red', [], mandatory=True)
        control['scope'] = {'strict': None}
        plan = self.plan(['a'], [self.case('broad', ['a']), control])
        self.assertEqual(plan['decision'], 'expand')
        self.assertIn('mandatory', plan['reason'])

    def test_matching_reordered_json_objects_and_null_results_are_admitted(self):
        declared = dict(self.context, request={'n': 1, 'values': [None, False, 1.0]})
        reordered = {'request': {'values': [None, False, 1.0], 'n': 1},
                     'hardware': 'cpu', 'model': 'fixture'}
        case = self.case('one', ['a']); case['scope'] = declared
        plan = vc.select(['a'], [case], reordered, self.root)
        self.assertEqual(plan['decision'], 'scoped')
        result = {'one': dict(contract_id=plan['contract_id'], status='passed', context=declared,
                              executed=1, skipped=0, edges={'a': 'passed'})}
        self.assertEqual(vc.validate_results(plan, result, declared, self.root)['status'], 'passed')

    def test_native_identity_axes_do_not_hide_a_typed_request_mismatch(self):
        context = dict(self.context, artifact='fixture-bytes', numeric_program='fixed', request={'n': True})
        case = self.case('native', ['a'], kind='gpu')
        case['scope'] = dict(context, request={'n': 1})
        self.assertEqual(vc.select(['a'], [case], context, self.root)['decision'], 'expand')

    def test_signed_zero_cannot_rebind_a_different_json_contract(self):
        declared = dict(self.context, temperature=0.0)
        changed = dict(self.context, temperature=-0.0)
        self.assertNotEqual(vc.contract_digest(declared), vc.contract_digest(changed))
        case = self.case('one', ['a']); case['scope'] = declared
        self.assertEqual(vc.select(['a'], [case], changed, self.root)['decision'], 'expand')

    def test_empty_edges_keep_the_explicit_mandatory_control_bundle(self):
        plan = self.plan([], [self.case('guard', ['guard'], mandatory=True, controls=['red']),
                              self.case('red', ['control-red'])])
        self.assertEqual(plan['decision'], 'scoped')
        self.assertEqual(plan['selected'], ['guard', 'red'])
        self.assertEqual(plan['contract']['required_edges'], [])
        self.assertFalse(plan['qualification'])

    def test_empty_edges_without_mandatory_requests_remain_nonpassing_no_change(self):
        for cases in ([], [self.case('optional', ['guard'])]):
            with self.subTest(cases=cases):
                plan = self.plan([], cases)
                self.assertEqual(plan['decision'], 'no-change')
                self.assertEqual(plan['selected'], [])
                with self.assertRaises(ValueError):
                    vc.validate_results(plan, {}, self.context, self.root)

    def test_empty_edges_still_expand_unavailable_mandatory_sources_and_scope(self):
        for cause in ('missing', 'stale', 'scope'):
            with self.subTest(cause=cause):
                case = self.case('guard', ['guard'], mandatory=True)
                if cause == 'missing': case['inputs'] = {'missing.py': self.digest}
                if cause == 'stale': case['inputs'] = {'harness.py': '0' * 64}
                if cause == 'scope': case['scope'] = {'hardware': 'other'}
                plan = self.plan([], [case])
                self.assertEqual(plan['decision'], 'expand')
                self.assertIn('mandatory', plan['reason'])

    def test_empty_edges_validate_required_control_existence_and_source(self):
        guard = self.case('guard', ['guard'], mandatory=True, controls=['red'])
        with self.assertRaisesRegex(ValueError, 'unknown control'):
            self.plan([], [guard])
        red = self.case('red', ['control-red']); red['inputs'] = {'missing.py': self.digest}
        self.assertEqual(self.plan([], [guard, red])['decision'], 'expand')

    def test_zero_requested_edges_admit_a_genuine_mandatory_execution_receipt(self):
        source = self.root / 'guard.py'
        source.write_text('assert 2 + 2 == 4\nprint("guard assertion passed")\n')
        guard = self.case('guard', ['guard'], mandatory=True)
        guard['inputs'] = {'guard.py': hashlib.sha256(source.read_bytes()).hexdigest()}
        plan = self.plan([], [guard])
        run = subprocess.run([sys.executable, str(source)], capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout, 'guard assertion passed\n')
        results = {'guard': dict(contract_id=plan['contract_id'], status='passed', context=self.context,
                                 executed=1, skipped=0, edges={'guard': 'passed'})}
        admitted = vc.validate_results(plan, results, self.context, self.root)
        self.assertEqual(admitted['status'], 'passed')
        self.assertEqual(admitted['edges'], 0)  # Requested edges only; the guard assertion was checked.
        self.assertFalse(admitted['qualification'])

    def test_empty_edges_reject_failed_skipped_missing_or_unasserted_guard_results(self):
        plan = self.plan([], [self.case('guard', ['guard'], mandatory=True)])
        good = dict(contract_id=plan['contract_id'], status='passed', context=self.context,
                    executed=1, skipped=0, edges={'guard': 'passed'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            vc.validate_results(plan, {}, self.context, self.root)
        for field, value in (('status', 'failed'), ('skipped', 1), ('edges', {})):
            with self.subTest(field=field):
                result = {**good, field: value}
                with self.assertRaises(ValueError):
                    vc.validate_results(plan, {'guard': result}, self.context, self.root)

    def test_empty_edges_preserve_generator_catalogs_and_recursive_controls(self):
        cases = [self.case('optional', []),
                 self.case('guard', ['guard'], mandatory=True, controls=['red']),
                 self.case('red', ['control-red'], controls=['nested']),
                 self.case('nested', ['nested-control'])]
        plan = self.plan([], iter(cases))
        self.assertEqual(plan['selected'], ['guard', 'nested', 'red'])
        self.assertEqual(plan['decision'], 'scoped')

    def test_mandatory_guard_with_no_named_edges_still_needs_execution(self):
        plan = self.plan([], [self.case('guard', [], mandatory=True)])
        self.assertEqual(plan['selected'], ['guard'])
        self.assertEqual(plan['decision'], 'scoped')
        with self.assertRaisesRegex(ValueError, 'missing'):
            vc.validate_results(plan, {}, self.context, self.root)


if __name__ == '__main__':
    unittest.main()
