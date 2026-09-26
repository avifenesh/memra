"""Do not count a sandbox failure as a failed model answer."""

from types import SimpleNamespace
import unittest
from unittest import mock

import quality_tasks


class SandboxReceiptTest(unittest.TestCase):
    def test_bubblewrap_failure_is_an_operational_error(self):
        task = {
            "visible_test_count": 1,
            "test_setup_code": "",
            "test_list": [
                "assert f() == 1",
                "assert f() == 1",
                "assert f() == 1",
            ],
        }
        answer = "```python\ndef f():\n    return 1\n```"
        with mock.patch(
            "quality_tasks.subprocess.run",
            return_value=SimpleNamespace(
                returncode=1,
                stderr="bwrap: namespace unavailable",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "sandbox failed"):
                quality_tasks.code_grade(answer, task)


if __name__ == "__main__":
    unittest.main()
