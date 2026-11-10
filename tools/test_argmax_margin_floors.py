#!/usr/bin/env python3
"""CPU census: the release battery's argmax cells against the floors table.

The battery passes --require-floor, so a roster model without its floor rows refuses a
release. These tests fail when the tree is in that state, and pin that the census itself
fails for a new roster model, a half-calibrated device and a floor row for other prompt bytes.
"""
import hashlib
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_argmax_floors", REPO / "tools/check-argmax-floors.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)

SHA = {"board-2048": "a" * 64}
# A receipt's table: 20 rows in the probe's printed form, the last 12 a battery window.
RECEIPT_ROWS = [f"{p:<8} {7:<13} {'1.000000000e0':<16} {7:<12} {'1.000000000e0':<16} {'1.0e-1':<16} yes"
                for p in range(2028, 2048)]
RECEIPT = ("[moe] resident-experts decision (dev0): experts 1.00GB + trunk 1.00GB vs free 9.00GB "
           "(expert budget 6.00GB) -> RESIDENT\npos header\n"
           + "\n".join(r + "   " for r in RECEIPT_ROWS) + "\nPRIME-GATE: class=MATCH\n")
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
        self.assertIn("every battery floor row has main's stored rows for the 12-row window, from its receipt",
                      r.stdout)

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

    def test_a_receipt_must_be_a_banked_file_of_the_tree(self):
        receipts = {}
        census.floor_rows("\n".join((row("M", "eager", "card A"), row("M", "batch", "card A"))), receipts)
        with tempfile.TemporaryDirectory(prefix="memra-floors-test-") as tmp:
            repo = pathlib.Path(tmp)
            (repo / "research/x").mkdir(parents=True)
            (repo / "research/x/probe.log").write_text("table\n")
            self.assertEqual(census.bad_receipts(receipts, repo), [])
            for path in ("/home/user/.cache/gate.log", "research/../x/probe.log", "research/y/probe.log",
                         "logs/probe.log"):
                with self.subTest(path=path):
                    key = next(iter(receipts))
                    self.assertEqual(census.bad_receipts({key: path}, repo), [(key, path)])

    def test_the_tree_receipts_are_banked(self):
        receipts = {}
        census.floor_rows((REPO / "tools/argmax-margin-floors.tsv").read_text(), receipts)
        self.assertEqual(census.bad_receipts(receipts, REPO), [])
    # memra#1204: main's stored rows per battery floor row.

    def stored(self, model="M", decode="eager", device="card A", receipt="research/x/probe.log",
               rows=None, window=12, sha=SHA["board-2048"], model_sha="c" * 64, experts="resident"):
        rows = RECEIPT_ROWS[-window:] if rows is None else rows
        text = ("# a comment\n" + "".join(f"{k}\t{v}\n" for k, v in (
            ("model", model), ("model_sha256", model_sha), ("prompt", "board-2048"), ("prompt_sha256", sha),
            ("decode", decode), ("device", device), ("window", window), ("experts", experts),
            ("commit", "031f146690"),
            ("receipt", receipt))) + "rows\n" + "".join(r + "\n" for r in rows))
        return (pathlib.Path(f"{model}-{decode}-{device}.rows"), *census.stored_table(text))

    def problems(self, tables, devices=("card A",)):
        receipts = {}
        rows = census.floor_rows("\n".join(row("M", d, dev) for d in ("eager", "batch") for dev in devices), receipts)
        return census.stored_problems(CELLS, rows, receipts, SHA, tables, lambda p: RECEIPT)

    def test_every_battery_floor_row_needs_its_stored_rows_from_its_receipt(self):
        both = [self.stored(decode="eager"), self.stored(decode="batch")]
        self.assertEqual(self.problems(both), [])
        # One decode program without its table, and a device with none.
        self.assertEqual(self.problems(both[:1]), ["MISSING stored rows: device='card A' model=M prompt=board-2048 "
                                                   "decode=batch window=12"])
        self.assertEqual(len(self.problems(both, devices=("card A", "card B"))), 2)
        # Another window does not count for the battery's.
        self.assertIn("MISSING stored rows", self.problems([both[0], self.stored(decode="batch", window=2)])[0])

    def test_a_stored_table_from_another_receipt_or_with_other_rows_is_stale(self):
        eager = self.stored(decode="eager")
        other = self.stored(decode="batch", receipt="research/y/probe.log")
        self.assertEqual(self.problems([eager, other]),
                         ["STALE stored rows M-batch-card A.rows: receipt research/y/probe.log is not the floor "
                          "row's research/x/probe.log"])
        moved = RECEIPT_ROWS[-12:-1] + [RECEIPT_ROWS[-1].replace("1.000000000e0", "1.100000000e0")]
        self.assertEqual(self.problems([eager, self.stored(decode="batch", rows=moved)]),
                         ["STALE stored rows M-batch-card A.rows: not the last 12 rows of research/x/probe.log"])
        self.assertEqual(self.problems([eager, self.stored(decode="batch", experts="none")]),
                         ["STALE stored rows M-batch-card A.rows: experts none but the receipt ran resident"])
        self.assertEqual(self.problems([eager, self.stored(decode="batch"), self.stored(decode="batch")]),
                         ["TWO stored tables for one artifact: device='card A' model=M decode=batch"])

    def test_a_malformed_stored_table_is_refused(self):
        for bad in ("model\tM\n", "rows\n" + RECEIPT_ROWS[0] + "\n",
                    "".join(f"{k}\tv\n" for k in census.STORED_FIELDS) + "rows\n"):
            with self.subTest(bad=bad[:30]):
                with self.assertRaises(ValueError):
                    census.stored_table(bad)
        with self.assertRaises(ValueError):
            self.stored(model_sha="not-a-sha")
        with self.assertRaises(ValueError):
            self.stored(rows=RECEIPT_ROWS[-11:])

    def test_the_tree_stored_rows_are_well_formed(self):
        # Every tracked table parses with the census's reader (a 64-hex model digest, as many rows
        # as its window). The engine's own reader runs on them in tools/test_argmax_margin_gate.py,
        # which has the built probe.
        tables = census.stored_tables(REPO / "tools/argmax-margin-rows")
        self.assertTrue(tables)
        for path, header, rows in tables:
            self.assertEqual(int(header["window"]), len(rows), path)

    def test_the_battery_window_is_the_gates_default_window(self):
        gate = (REPO / "tools/argmax-margin-gate.sh").read_text()
        default = re.search(r"^WINDOW=(\d+)\s", gate, re.M)
        self.assertIsNotNone(default)
        self.assertEqual(int(default.group(1)), census.BATTERY_WINDOW)

    def test_the_prompt_digest_is_the_file_the_gate_reads(self):
        prompt = REPO / "research/e2e/prompts/board-2048.txt"
        digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
        table = (REPO / "tools/argmax-margin-floors.tsv").read_text()
        for line in table.splitlines():
            if line.strip() and not line.startswith("#") and line.split("\t")[1] == "board-2048":
                self.assertEqual(line.split("\t")[2], digest)


if __name__ == "__main__":
    unittest.main()
