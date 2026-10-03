#!/usr/bin/env python3
"""Admit all original agreement-summary predicates and identified CPU controls."""
import json
from pathlib import Path
import sys
import unittest

REQUIRED_METHODS = ('test_ordinary_selftest_cli', 'test_original_eight_predicates_ten_entries_and_real_calls', 'test_each_wrong_predicate_fails_and_restores', 'test_missing_duplicate_and_reordered_traffic_iterations_refuse', 'test_projection_fixture_and_original_assertions_preserved', 'test_noop_selftest_or_csv_refuses', 'test_missing_or_replaced_predicate_refuses', 'test_optimization_refuses_before_fixture', 'test_mandatory_ci_caller', 'test_removed_masked_swallowed_caller_refuses', 'test_ordinary_summary_and_csv_bytes_unchanged_under_optimization', 'test_summary_change_retains_full_native_selection', 'test_exact_control_inventory', 'test_missing_replaced_unrelated_duplicate_discovery_refuses', 'test_missing_duplicate_or_no_success_execution_refuses', 'test_skipped_failed_expected_failure_and_unexpected_success_refuse')
MODULE = "test_agreement_summary_contract"
GROUPS = {(MODULE, "AgreementSummaryTests"): REQUIRED_METHODS}
REQUIRED_IDS = frozenset(f"{MODULE}.AgreementSummaryTests.{name}" for name in REQUIRED_METHODS)
ORIGINAL_IDS = frozenset({f"{MODULE}.AgreementSummaryTests.test_original_eight_predicates_ten_entries_and_real_calls"})


def control_id(case):
    key = (type(case).__module__, type(case).__name__)
    if key not in GROUPS or type(case) is not getattr(sys.modules.get(key[0]), key[1], None):
        raise ValueError('unrelated control type')
    identity = f'{key[0]}.{key[1]}.{case._testMethodName}'
    if case.id() != identity:
        raise ValueError('control identity does not match actual method')
    return identity


def discovered_ids(suite):
    identities, active = [], set()
    def visit(item):
        if isinstance(item, unittest.TestSuite):
            if id(item) in active:
                raise ValueError('cyclic control suite')
            active.add(id(item))
            for child in item:
                visit(child)
            active.remove(id(item))
        else:
            identities.append(control_id(item))
    visit(suite)
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate discovered control identity')
    if set(identities) != REQUIRED_IDS:
        raise ValueError('missing or replaced required control identity')
    return identities


def run(suite, *, stream=sys.stderr):
    try:
        discovered = discovered_ids(suite)
    except ValueError as error:
        print('agreement-summary contract: FAIL: discovery: ' + str(error), file=stream)
        return 1
    executed, succeeded = [], []
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            executed.append(control_id(test))
            super().startTest(test)
        def addSuccess(self, test):
            succeeded.append(control_id(test))
            super().addSuccess(test)
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=1, resultclass=Result).run(suite)
    except (TypeError, ValueError, AttributeError) as error:
        print('agreement-summary contract: FAIL: malformed framework result: ' + str(error), file=stream)
        return 1
    if (type(result.testsRun) is not int or len(executed) != len(set(executed))
            or set(executed) != set(discovered) or result.testsRun != len(discovered)):
        print('agreement-summary contract: FAIL: executed identities do not match discovery', file=stream)
        return 1
    fields = ('failures', 'errors', 'skipped', 'expectedFailures', 'unexpectedSuccesses')
    if any(type(getattr(result, name, None)) is not list for name in fields):
        print('agreement-summary contract: FAIL: malformed result evidence', file=stream)
        return 1
    if (len(succeeded) != len(set(succeeded)) or set(succeeded) != set(discovered)
            or not result.wasSuccessful() or any(getattr(result, name) for name in fields)):
        print(f'agreement-summary contract: FAIL: executed={result.testsRun} skipped={len(result.skipped)} '
              f'expected_failures={len(result.expectedFailures)}', file=stream)
        return 1
    original = [identity for identity in executed if identity in ORIGINAL_IDS]
    print(f'agreement-summary contract: PASS: original={len(original)} executed={result.testsRun} '
          f'discovered={len(discovered)} unique={len(set(executed))} skipped=0 expected_failures=0', file=stream)
    print('agreement-summary contract identities: ' + json.dumps({'discovered': discovered,
          'executed': executed, 'succeeded': succeeded, 'original': original}, sort_keys=True), file=stream)
    return 0


def main():
    if not __debug__:
        raise RuntimeError('agreement-summary admission controls require enabled assertions')
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent), pattern='test_agreement_summary_contract.py')
    return run(suite)


if __name__ == '__main__':
    sys.exit(main())
