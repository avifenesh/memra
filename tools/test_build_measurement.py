"""Guard the CPU-only measurement's separation from native qualification jobs."""
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import measure_build_reuse as measurement


class MeasurementBoundaryTests(unittest.TestCase):
    def test_measurement_cannot_dispatch_gpu_or_publish_qualification(self):
        source = (Path(__file__).resolve().parents[1] / ".github/workflows/gpu-ci.yml").read_text()
        for name, next_name in (("build", "build-feedback-measurement"),
                                ("gpu", "qualification-result")):
            job = source.split("\n  " + name + ":\n", 1)[1].split("\n  " + next_name + ":\n", 1)[0]
            self.assertIn("!inputs.build_reuse_only", job.split("    steps:", 1)[0])
        report = source.split("\n  qualification-result:\n", 1)[1]
        self.assertIn("always() && !inputs.build_reuse_only", report.split("    steps:", 1)[0])
        job = source.split("\n  build-feedback-measurement:\n", 1)[1].split("\n  gpu:\n", 1)[0]
        self.assertIn("if: ${{ inputs.build_reuse_only }}", job)
        self.assertIn("runs-on: ubuntu-24.04", job)
        self.assertIn("tools/measure_build_reuse.py", job)
        self.assertNotIn("tools/gpu-ci.py capture", job)
        self.assertNotIn("checks: write", job)

    def test_measurement_refuses_existing_or_source_outputs_before_compiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            existing = root / "owned-by-someone-else"
            existing.mkdir()
            (existing / "keep").write_text("preserve")
            for out in (repo / "output", existing):
                with patch.object(measurement.subprocess, "run") as execute:
                    with self.assertRaisesRegex(ValueError, "new and outside"):
                        measurement.measure(repo, out, Path("unused"), Path("unused"), 2)
                    execute.assert_not_called()
            self.assertEqual((existing / "keep").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
