"""Keep the independent prose judge off native request CPU cores."""

import io
from pathlib import Path
import unittest
from unittest import mock

import supervise_v12


class CpuIsolationTest(unittest.TestCase):
    def test_stage_launches_only_on_declared_cores(self):
        with mock.patch.object(
            supervise_v12.subprocess, "run",
        ) as launched:
            launched.return_value.returncode = 0
            supervise_v12.stage(
                io.StringIO(), Path("/synthetic/source"),
                "judge.py", "--help",
                cpus=[11, 13], low_priority=True,
            )
        command = launched.call_args.args[0]
        self.assertEqual(
            command[:6],
            ["nice", "-n", "10", "taskset", "-c", "11,13"],
        )
        self.assertEqual(command[-2:], [
            "/synthetic/source/judge.py", "--help",
        ])

    def test_judge_uses_disjoint_reserved_cores(self):
        with mock.patch.object(
            supervise_v12.os, "sched_getaffinity",
            return_value={3, 5, 7, 9, 11, 13},
        ):
            native, judge = supervise_v12.cpu_slices()
        self.assertEqual(native, [3, 5, 7, 9])
        self.assertEqual(judge, [11, 13])
        self.assertFalse(set(native) & set(judge))

    def test_too_few_cores_refuses_overlap(self):
        with mock.patch.object(
            supervise_v12.os, "sched_getaffinity",
            return_value={0, 1, 2},
        ):
            with self.assertRaises(ValueError):
                supervise_v12.cpu_slices()


if __name__ == "__main__":
    unittest.main()
