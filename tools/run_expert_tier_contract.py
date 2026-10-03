#!/usr/bin/env python3
"""Run the expert-tier admission controls with a non-vacuity and zero-skip gate."""

from pathlib import Path
import json
import sys
import unittest


# This inventory is independent of discovery. Deleting or replacing a test must
# not silently shrink the obligations that the runner admits.
REQUIRED_METHODS = (
    'test_ordinary_selftest_cli',
    'test_real_selftest_executes_nine_cases_and_26_assertions',
    'test_cli_optimization_refuses',
    'test_environment_optimization_refuses',
    'test_import_optimization_refuses_before_fixture',
    'test_planted_incorrect_result_fails',
    'test_noop_selftest_refuses_observation',
    'test_removed_assertion_refuses_observation',
    'test_ci_wiring',
    'test_removed_or_masked_ci_caller_refuses',
    'test_ordinary_plan_generation_unchanged_under_optimization',
    'test_builder_change_retains_full_selection',
    'test_runner_refuses_empty_short_skipped_failed_and_expected_failures',
    'test_runner_refuses_missing_replaced_duplicate_and_unexecuted_controls',
)
MODULE = 'test_expert_tier_plan_contract'
REQUIRED_IDS = frozenset(f'{MODULE}.ExpertTierContractTests.{name}' for name in REQUIRED_METHODS)
MINIMUM = len(REQUIRED_IDS)


def discovered_ids(suite):
    module = sys.modules.get(MODULE)
    canonical = getattr(module, 'ExpertTierContractTests', None)
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
            if canonical is None or type(item) is not canonical:
                raise ValueError('unrelated control type')
            identity = f'{MODULE}.ExpertTierContractTests.{item._testMethodName}'
            if item.id() != identity:
                raise ValueError('control identity does not match its actual method')
            identities.append(identity)

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
        print('expert-tier contract: FAIL: discovery: ' + str(error), file=stream)
        return 1
    executed = []

    class Result(unittest.TextTestResult):
        def startTest(self, test):
            executed.append(test.id())
            super().startTest(test)

    result = unittest.TextTestRunner(stream=stream, verbosity=1, resultclass=Result).run(suite)
    if (len(executed) != len(set(executed)) or set(executed) != set(discovered)
            or result.testsRun != len(discovered)):
        print('expert-tier contract: FAIL: executed control identities do not match discovery', file=stream)
        return 1
    if not result.wasSuccessful() or result.testsRun < MINIMUM or result.skipped or result.expectedFailures:
        print(f'expert-tier contract: FAIL: executed={result.testsRun} floor={MINIMUM} '
              f'skipped={len(result.skipped)} expected_failures={len(result.expectedFailures)}', file=stream)
        return 1
    print(f'expert-tier contract: PASS: executed={result.testsRun} floor={MINIMUM} '
          f'discovered={len(discovered)} unique={len(set(executed))} '
          'skipped=0 expected_failures=0', file=stream)
    print('expert-tier contract identities: ' + json.dumps({
        'discovered': discovered, 'executed': executed}, sort_keys=True), file=stream)
    return 0


def main():
    if not __debug__:
        raise RuntimeError('expert-tier admission controls require enabled assertions')
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent), pattern='test_expert_tier_plan_contract.py')
    return run(suite)


if __name__ == '__main__':
    sys.exit(main())
