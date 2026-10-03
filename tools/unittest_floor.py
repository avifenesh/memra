#!/usr/bin/env python3
"""Run unittest discovery and admit only complete, successful execution."""
import re
import os
import sys
import unittest


FIELDS = {'discovered', 'run', 'passed', 'skipped', 'failures', 'errors',
          'expected_failures', 'unexpected_successes'}


def admit(evidence, minimum):
    if not isinstance(evidence, dict) or set(evidence) != FIELDS:
        return False
    if any(type(value) is not int or value < 0 for value in evidence.values()):
        return False
    if type(minimum) is not int or minimum < 0:
        return False
    return (evidence['run'] > 0 and evidence['run'] >= minimum
            and evidence['discovered'] == evidence['run'] == evidence['passed']
            and not any(evidence[name] for name in FIELDS - {'discovered', 'run', 'passed'}))


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_tests = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successful_tests += 1


class Runner(unittest.TextTestRunner):
    resultclass = Result

    def run(self, test):
        discovered = test.countTestCases()
        result = super().run(test)
        result.discovered = discovered
        return result


def execution_evidence(result):
    names = {'skipped': 'skipped', 'failures': 'failures', 'errors': 'errors',
             'expected_failures': 'expectedFailures', 'unexpected_successes': 'unexpectedSuccesses'}
    evidence = {'discovered': result.discovered, 'run': result.testsRun,
                'passed': result.successful_tests}
    for name, attribute in names.items():
        value = getattr(result, attribute)
        if type(value) is not list:
            raise ValueError('malformed unittest outcome collection: ' + name)
        evidence[name] = len(value)
    return evidence


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        print('usage: unittest-floor.sh <start-dir> <pattern> <min-tests> [unittest args]', file=sys.stderr)
        return 2
    start, pattern, value, *extra = argv
    if not re.fullmatch(r'[0-9]+', value):
        print('unittest-floor: <min-tests> must be a non-negative integer', file=sys.stderr)
        return 2
    try:
        minimum = int(value)
    except ValueError:
        print('unittest-floor: invalid integer floor', file=sys.stderr)
        return 2
    if not __debug__ or sys.flags.optimize:
        print('unittest-floor: FAIL: assertions must be enabled before test imports', file=sys.stderr)
        return 1
    # Preserve the import root of the original `python3 -m unittest` invocation.
    sys.path.insert(0, os.getcwd())
    program = unittest.main(module=None, argv=['unittest-floor', 'discover', '-s', start,
                                             '-p', pattern, *extra],
                            testRunner=Runner, exit=False)
    try:
        evidence = execution_evidence(program.result)
    except (AttributeError, TypeError, ValueError) as error:
        print('unittest-floor: FAIL: malformed execution evidence: ' + str(error), file=sys.stderr)
        return 1
    if not admit(evidence, minimum):
        print(f'unittest-floor: FAIL: required execution rejected for {start} ({pattern}); '
              f'floor={minimum}; ' + ' '.join(f'{name}={evidence[name]}' for name in sorted(evidence)),
              file=sys.stderr)
        return 1
    print(f"unittest-floor: OK: ran {evidence['run']} tests (floor {minimum}) for {start} ({pattern}); "
          'skipped=0 expected_failures=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
