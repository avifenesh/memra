#!/usr/bin/env python3
"""Run the expert-tier admission controls with a non-vacuity and zero-skip gate."""

from pathlib import Path
import sys
import unittest


MINIMUM = 13


def run(suite, *, stream=sys.stderr):
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not result.wasSuccessful() or result.testsRun < MINIMUM or result.skipped or result.expectedFailures:
        print(f'expert-tier contract: FAIL: executed={result.testsRun} floor={MINIMUM} '
              f'skipped={len(result.skipped)} expected_failures={len(result.expectedFailures)}', file=stream)
        return 1
    print(f'expert-tier contract: PASS: executed={result.testsRun} floor={MINIMUM} '
          'skipped=0 expected_failures=0', file=stream)
    return 0


def main():
    if not __debug__:
        raise RuntimeError('expert-tier admission controls require enabled assertions')
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent), pattern='test_expert_tier_plan_contract.py')
    return run(suite)


if __name__ == '__main__':
    sys.exit(main())
