"""Runs only inside the component input view. Writes a non-vacuous unittest receipt."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, "/source/tools")
contract = json.loads(Path("/out/contract.json").read_text())
suite = unittest.defaultTestLoader.loadTestsFromNames(contract["modules"])


def names(tests):
    for test in tests:
        if isinstance(test, unittest.TestSuite):
            yield from names(test)
        else:
            yield test.id()


roster = sorted(names(suite))
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {"tests": roster, "run": result.testsRun, "skipped": len(result.skipped),
          "failures": len(result.failures), "errors": len(result.errors),
          "expected_failures": len(result.expectedFailures),
          "unexpected_successes": len(result.unexpectedSuccesses)}
record["pass"] = (result.wasSuccessful() and result.testsRun >= contract["min_tests"]
                  and result.testsRun == len(roster) == len(set(roster))
                  and not result.skipped and not result.expectedFailures)
Path("/out/result.json").write_text(json.dumps(record, sort_keys=True) + "\n")
sys.exit(0 if record["pass"] else 1)
