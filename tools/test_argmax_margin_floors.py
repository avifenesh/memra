#!/usr/bin/env python3
"""CPU census: the release battery's argmax cells against the floors table.

The battery passes --require-floor, so a roster model without its floor rows refuses a
release. These tests fail when the tree is in that state, and pin that the census itself
fails for a new roster model, a half-calibrated device and a floor row for other prompt bytes.
"""
import hashlib
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_argmax_floors", REPO / "tools/check-argmax-floors.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)

SHA = {"board-2048": "a" * 64}
CELLS = [("own/m", "M", "board-2048", "eager"), ("own/m", "M", "board-2048", "batch")]


def row(model, decode, device, sha=SHA["board-2048"]):
    return "\t".join((model, "board-2048", sha, decode, device, "1.0e0", "2000", "7", "1.0e-1",
                      "5.0e-1", "8.0e-1", "-", "research/x/probe.log"))


class FloorsCensusTest(unittest.TestCase):
    def test_the_tree_has_a_floor_row_for_every_battery_cell_on_every_covered_device(self):
        r = subprocess.run([sys.executable, str(REPO / "tools/check-argmax-floors.py")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("every floor row present", r.stdout)
        self.assertIn("maintained classes gb10, rtx5090, rtxpro6000 each calibrated", r.stdout)

    def test_the_battery_lists_two_decode_programs_per_roster_model_on_the_gate_prompt(self):
        cells = census.battery_cells(REPO)
        roster = [line.split("\t") for line in (REPO / "tools/release-roster.tsv").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        self.assertEqual(len(cells), 2 * len(roster))
        for _, rid, path, *_ in roster:
            model = pathlib.Path(path).name.removesuffix(".gguf")
            for decode in ("eager", "batch"):
                self.assertIn((rid, model, "board-2048", decode), cells)

    def test_complete_table_passes(self):
        rows = census.floor_rows("# comment\n" + "\n".join(
            row("M", decode, device) for decode in ("eager", "batch") for device in ("card A", "card B")))
        self.assertEqual(census.missing_rows(CELLS, rows, SHA), [])

    def test_a_new_roster_model_without_rows_fails_on_every_covered_device(self):
        rows = census.floor_rows("\n".join(
            row("M", decode, device) for decode in ("eager", "batch") for device in ("card A", "card B")))
        cells = CELLS + [("own/new", "NEW", "board-2048", "eager"), ("own/new", "NEW", "board-2048", "batch")]
        missing = census.missing_rows(cells, rows, SHA)
        self.assertEqual(len(missing), 4)
        self.assertEqual({(m[0], m[2], m[4]) for m in missing},
                         {(d, "NEW", a) for d in ("card A", "card B") for a in ("eager", "batch")})

    def test_a_half_calibrated_device_fails(self):
        rows = census.floor_rows("\n".join((row("M", "eager", "card A"), row("M", "batch", "card A"),
                                            row("M", "eager", "card B"))))
        self.assertEqual(census.missing_rows(CELLS, rows, SHA),
                         [("card B", "own/m", "M", "board-2048", "batch")])

    def test_a_row_for_other_prompt_bytes_does_not_count(self):
        rows = census.floor_rows("\n".join((row("M", "eager", "card A"),
                                            row("M", "batch", "card A", sha="b" * 64))))
        self.assertEqual(census.missing_rows(CELLS, rows, SHA),
                         [("card A", "own/m", "M", "board-2048", "batch")])

    def test_a_maintained_class_with_no_calibrated_device_fails(self):
        ci = census.change_impact(REPO)
        rules, arch = ci.tree_rules(REPO)
        maintained = ci.maintained_classes(rules, arch)
        self.assertEqual(maintained, ["gb10", "rtx5090", "rtxpro6000"])
        class_of = lambda name: ci.device_class_of(rules, name)
        rows = census.floor_rows("\n".join((
            row("M", "eager", "NVIDIA GeForce RTX 5090 Laptop GPU"),
            row("M", "eager", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"))))
        self.assertEqual(census.uncalibrated_classes(maintained, rows, class_of), ["gb10"])
        rows.add(tuple(row("M", "eager", "NVIDIA GB10").split("\t")[:5]))
        self.assertEqual(census.uncalibrated_classes(maintained, rows, class_of), [])
        # A class that is not maintained is never demanded, and an unknown name covers nothing.
        only_h100 = census.floor_rows(row("M", "eager", "NVIDIA H100 80GB HBM3") + "\n" + row("M", "eager", "Some Other GPU"))
        self.assertEqual(census.uncalibrated_classes(maintained, only_h100, class_of), maintained)

    def test_device_names_map_to_maintained_unmaintained_or_unknown(self):
        ci = census.change_impact(REPO)
        rules, arch = ci.tree_rules(REPO)
        for name, status in (
            ("NVIDIA GeForce RTX 5090 Laptop GPU", "maintained rtx5090"),
            ("NVIDIA GeForce RTX 5090", "maintained rtx5090"),  # a desktop card: same class, its own floor rows
            ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "maintained rtxpro6000"),
            ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "maintained rtxpro6000"),
            ("NVIDIA GB10", "maintained gb10"),
            ("NVIDIA H100 80GB HBM3", "unmaintained h100"),
            ("NVIDIA B200", "unmaintained b200"),
            ("Some Other GPU", "unknown"),  # never "unmaintained" by default
            ("unknown", "unknown"),
        ):
            with self.subTest(name=name):
                self.assertEqual(ci.device_status(rules, arch, name), status)
                cli = subprocess.run([sys.executable, str(REPO / "tools/change_impact.py"), "--repo", str(REPO),
                                      "--device-status", name], capture_output=True, text=True)
                self.assertEqual((cli.returncode, cli.stdout.strip()), (0, status))

    def test_an_empty_table_covers_no_device_and_a_malformed_one_is_refused(self):
        self.assertEqual(census.missing_rows(CELLS, census.floor_rows("# only comments\n"), SHA), [])
        with self.assertRaises(ValueError):
            census.floor_rows("M\tboard-2048\n")

    def test_the_census_reads_a_roster_through_the_battery(self):
        with tempfile.TemporaryDirectory(prefix="memra-floors-test-") as tmp:
            roster = pathlib.Path(tmp) / "roster.tsv"
            roster.write_text("# a comment\nown\tlab/new-model\t/models/New-Model-Q8.gguf\tnote\n")
            self.assertEqual(census.battery_cells(REPO, roster),
                             [("lab/new-model", "New-Model-Q8", "board-2048", "eager"),
                              ("lab/new-model", "New-Model-Q8", "board-2048", "batch")])
            roster.write_text("guest\tlab/x\t/models/X.gguf\n")
            with self.assertRaises(subprocess.CalledProcessError):
                census.battery_cells(REPO, roster)

    def test_the_prompt_digest_is_the_file_the_gate_reads(self):
        prompt = REPO / "research/e2e/prompts/board-2048.txt"
        digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
        table = (REPO / "tools/argmax-margin-floors.tsv").read_text()
        for line in table.splitlines():
            if line.strip() and not line.startswith("#") and line.split("\t")[1] == "board-2048":
                self.assertEqual(line.split("\t")[2], digest)


if __name__ == "__main__":
    unittest.main()
