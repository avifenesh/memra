#!/usr/bin/env python3
"""CPU-only adversarial tests of the day-12 series verdict (lead ruling 6) and verdict wording."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).with_name("verify-day12.py")
spec = importlib.util.spec_from_file_location("verify", path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

G = 2097152
# Day-11 32k series bytes, identical on both card classes except the baseline (PRO shown).
PRO_32K = dict(free_before=85313847296, free_after_demote=86217719808, free_after_restore=85313847296,
               released=905969664)
LAPTOP_32K = dict(free_before=8784248832, free_after_demote=9688121344, free_after_restore=8784248832,
                  released=905969664)
PRO_8K = dict(free_before=86098182144, free_after_demote=86299508736, free_after_restore=86098182144,
              released=201326592)
LABEL = "ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)"
OK = [True] * 5


def row(c, k):
    exact, bounded, residual = verify.observe(c["free_before"], c["free_after_demote"],
                                              c["free_after_restore"], c["released"], G)
    return {"cycle": k, "free_before_bytes": c["free_before"], "free_after_demote_bytes": c["free_after_demote"],
            "free_after_restore_bytes": c["free_after_restore"], "vmm_released_chunk_bytes": c["released"],
            "vmm_granularity_bytes": G, "reclaimed_bytes": c["free_after_demote"] - c["free_before"],
            "reacquired_bytes": c["free_after_demote"] - c["free_after_restore"], "residual_bytes": residual,
            "restore_residual_bytes": c["free_before"] - c["free_after_restore"], "reclaim_observed": bounded,
            "reclaim_exact_equal": exact, "g1_reclaim_qualified": str(bounded and residual == 0).lower(),
            "residual_class": "none" if residual == 0 else "unclassified",
            "restored_prefix_bit_identical": True,
            "restored_prefix_state_manifest_sha256": "0" * 64}


class SeriesVerdictTests(unittest.TestCase):
    def test_both_card_classes_five_cycle_series_carry_the_label(self):
        for series in (PRO_32K, LAPTOP_32K):
            with self.subTest(series=series):
                self.assertEqual(verify.series_verdict([series] * 5, G, OK, 32768), (verify.ONE_TIME, "true", LABEL))
        self.assertEqual(verify.series_verdict([PRO_32K] * 6, G, [True] * 6, 32768)[2],
                         "ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 6 cycles)")
        self.assertEqual(verify.series_verdict([PRO_32K] * 5, G, OK, 16384)[2],
                         "ACTIVE-16K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)")

    def test_shorter_than_five_stays_false(self):
        for n in range(1, 5):
            with self.subTest(n=n):
                klass, qualified, label = verify.series_verdict([PRO_32K] * n, G, [True] * n, 32768)
                self.assertEqual((qualified, label), ("false", None))

    def test_one_drifting_cycle_stays_false(self):
        drifted = dict(PRO_32K, free_before=PRO_32K["free_before"] + G, free_after_demote=PRO_32K["free_after_demote"] + G,
                       free_after_restore=PRO_32K["free_after_restore"] + G)
        for k in range(5):
            series = [dict(PRO_32K) for _ in range(5)]
            series[k] = drifted
            with self.subTest(k=k):
                klass, qualified, label = verify.series_verdict(series, G, OK, 32768)
                self.assertNotEqual(klass, verify.ONE_TIME)
                self.assertEqual((qualified, label), ("false", None))

    def test_growing_and_other_classes_stay_false(self):
        growing = [dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - k * G) for k in range(5)]
        self.assertEqual(verify.series_verdict(growing, G, OK, 32768), (verify.GROWING, "false", None))
        two = dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - G)
        never_back = dict(PRO_32K, free_after_restore=PRO_32K["free_after_restore"] - G)
        for shape in ([two] * 5, [never_back] * 5, [PRO_32K, PRO_8K, PRO_32K, PRO_32K, PRO_32K]):
            with self.subTest(shape=shape):
                klass, qualified, label = verify.series_verdict(shape, G, OK, 32768)
                self.assertNotEqual(klass, verify.ONE_TIME)
                self.assertEqual((qualified, label), ("false", None))

    def test_exact_none_is_true_without_a_label_and_pooled_is_not_applicable(self):
        self.assertEqual(verify.series_verdict([PRO_8K] * 5, G, OK, 8192), (verify.NONE, "true", None))
        self.assertEqual(verify.series_verdict([PRO_32K] * 5, 0, OK, 32768),
                         (verify.NOT_APPLICABLE_POOLED, verify.NOT_APPLICABLE_POOLED, None))
        self.assertEqual(verify.series_verdict([PRO_8K] * 5, 0, OK, 8192)[1], verify.NOT_APPLICABLE_POOLED)
        exact_drifted = dict(PRO_8K, free_before=PRO_8K["free_before"] - G, free_after_demote=PRO_8K["free_after_demote"] - G,
                             free_after_restore=PRO_8K["free_after_restore"] - G)
        self.assertEqual(verify.series_verdict([PRO_8K, PRO_8K, exact_drifted, exact_drifted, exact_drifted], G, OK, 8192)[1:],
                         ("false", None))

    def test_restore_identity_is_required_in_every_cycle(self):
        for k in range(5):
            identical = list(OK)
            identical[k] = False
            with self.subTest(k=k):
                self.assertEqual(verify.series_verdict([PRO_32K] * 5, G, identical, 32768)[1:], ("false", None))
        self.assertEqual(verify.series_verdict([PRO_32K] * 5, G, [True] * 4, 32768)[1:], ("false", None))

    def test_one_byte_anywhere_removes_the_label(self):
        for k in range(5):
            for key in ("free_before", "free_after_demote", "free_after_restore", "released"):
                series = [dict(PRO_32K) for _ in range(5)]
                series[k][key] += 1
                with self.subTest(cycle=k, key=key):
                    self.assertEqual(verify.series_verdict(series, G, OK, 32768)[1:], ("false", None))


class VerdictTests(unittest.TestCase):
    def rows(self, cycles):
        return [row(c, k + 1) for k, c in enumerate(cycles)]

    def test_label_is_returned_verbatim_and_only_with_the_ruling_six_verdict(self):
        rows = self.rows([PRO_32K] * 5)
        self.assertEqual(verify.verdict(32768, rows, verify.ONE_TIME, "true", LABEL, "matches"), LABEL)
        self.assertEqual(verify.verdict(32768, rows, verify.ONE_TIME, "true", LABEL,
                                        "decoded surfaces differ from the rented RTX 5090 bundle"), LABEL)
        self.assertNotIn("\u2014", LABEL)
        for bad in ((verify.ONE_TIME, "false", LABEL), (verify.NONE, "true", LABEL),
                    (verify.ONE_TIME, "true", LABEL.replace("5 cycles", "4 cycles"))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify.verdict(32768, rows, *bad, "matches")

    def test_without_a_label_the_day11_wording_is_unchanged(self):
        rows = self.rows([PRO_32K] * 5)
        self.assertEqual(verify.verdict(32768, rows, verify.ONE_TIME, "false", None, "matches"),
                         "ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, "
                         "residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS")
        eight = self.rows([PRO_8K] * 5)
        self.assertEqual(verify.verdict(8192, eight, verify.NONE, "true", None, "matches the frozen target-card bundle"),
                         "ACTIVE-8K G1 PASS across 5 cycles, residual 0 B each cycle, class none")
        self.assertIn("no frozen bundle for this card, not G1 PASS",
                      verify.verdict(8192, eight, verify.NONE, "true", None, "decoded surfaces differ"))
        growing = [dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - k * G) for k in range(5)]
        text = verify.verdict(32768, self.rows(growing), verify.GROWING, "false", None, "matches")
        self.assertIn("class growing-residual, not G1 PASS", text)
        with self.assertRaises(ValueError):
            verify.verdict(32768, rows, verify.ONE_TIME, "true", None, "matches")

    def test_console_label_must_be_the_final_status_line_exactly_once(self):
        tail = "RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 g1_reclaim_qualified=true\n"
        good = tail + LABEL + " committed=32768 generated=128\n"
        self.assertEqual(verify.console_label(good, 32768, LABEL), LABEL)
        for bad in (tail + "ACTIVE_COPY_RESTORE_CAPTURED; reclaim qualification incomplete; see metrics; "
                    "continuation comparison pending; not G1 PASS committed=32768 generated=128\n",
                    LABEL + "\n" + good, good + "ACTIVE-32K G1 PASS committed=32768 generated=128\n",
                    tail + LABEL.replace("5 cycles", "4 cycles") + " committed=32768 generated=128\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify.console_label(bad, 32768, LABEL)
        plain = tail.replace("true", "false") + ("ACTIVE_COPY_RESTORE_CAPTURED; reclaim qualification incomplete; "
                                                 "see metrics; continuation comparison pending; not G1 PASS "
                                                 "committed=32768 generated=128\n")
        self.assertIsNone(verify.console_label(plain, 32768, None))
        with self.assertRaises(ValueError):
            verify.console_label(good, 32768, None)

    def test_series_summary_red_arms(self):
        rows = self.rows([PRO_32K] * 5)
        header = ["cycle", "free_before_bytes", "free_after_demote_bytes", "free_after_restore_bytes",
                  "vmm_released_chunk_bytes", "reclaimed_bytes", "reacquired_bytes", "residual_bytes",
                  "restore_residual_bytes", "free_before_drift_bytes", "reclaim_observed", "reclaim_exact_equal",
                  "residual_class", "g1_reclaim_qualified", "restored_prefix_state_manifest_sha256"]

        def write(folder, klass="one-time-driver-mapping-metadata", g1="true", label=LABEL, minimum=5,
                  identical="true", mutate=None):
            lines = ["\t".join(header)]
            for r in rows:
                values = dict(r, free_before_drift_bytes=0, reclaim_observed=str(r["reclaim_observed"]).lower(),
                              reclaim_exact_equal=str(r["reclaim_exact_equal"]).lower())
                if mutate:
                    mutate(values)
                lines.append("\t".join(str(values[k]) for k in header))
            (folder / "reclaim-cycles.tsv").write_text("\n".join(lines) + "\n")
            (folder / "reclaim-cycles.txt").write_text(
                f"cycles=5\nvmm_granularity_bytes={G}\nresidual_series_class={klass}\n"
                f"residual_series_bytes={','.join([str(G)] * 5)}\nresidual_first_cycle_bytes={G}\n"
                f"residual_last_cycle_bytes={G}\nfree_before_drift_last_bytes=0\nall_cycles_reclaim_observed=true\n"
                f"all_cycles_restored_bit_identical={identical}\ng1_reclaim_qualified={g1}\n"
                f"series_min_cycles={minimum}\nseries_label={label}\n")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write(folder)
            self.assertEqual(verify.series(folder, rows, 32768), (verify.ONE_TIME, "true", LABEL))
            for bad in (dict(klass="none"), dict(klass="growing-residual"), dict(g1="false"),
                        dict(label="not-printed"), dict(label=LABEL.replace("5 cycles", "4 cycles")),
                        dict(minimum=4), dict(identical="false")):
                write(folder, **bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    verify.series(folder, rows, 32768)

            def shift(values):
                values["residual_bytes"] = int(values["residual_bytes"]) + 1
            write(folder, mutate=shift)
            with self.assertRaises(ValueError):
                verify.series(folder, rows, 32768)
            # A restore that differs in one cycle: the receipt must say false and not-printed.
            broken = [dict(r) for r in rows]
            broken[2]["restored_prefix_bit_identical"] = False
            write(folder, identical="false", g1="false", label="not-printed")
            self.assertEqual(verify.series(folder, broken, 32768), (verify.ONE_TIME, "false", None))
            write(folder)
            with self.assertRaises(ValueError):
                verify.series(folder, broken, 32768)


if __name__ == "__main__":
    unittest.main()
