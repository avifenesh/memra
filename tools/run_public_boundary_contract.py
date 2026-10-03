#!/usr/bin/env python3
"""Run every real boundary regression with a non-vacuity and zero-skip gate."""

from pathlib import Path
import sys
import unittest


MINIMUM = 60


def run(suite, *, stream=sys.stderr):
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if not result.wasSuccessful() or result.testsRun < MINIMUM or result.skipped:
        print(f'public-boundary contract: FAIL: executed={result.testsRun} '
              f'floor={MINIMUM} skipped={len(result.skipped)}', file=stream)
        return 1
    print(f'public-boundary contract: PASS: executed={result.testsRun} '
          f'floor={MINIMUM} skipped=0', file=stream)
    return 0


def main():
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parent), pattern='test_public_boundary.py')
    return run(suite)


if __name__ == '__main__':
    sys.exit(main())
