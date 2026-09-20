#!/usr/bin/env python3
"""CPU-only adversarial tests of the day-11 series rule and verdict wording; no GPU claims."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name("verify-day11.py")
spec = importlib.util.spec_from_file_location("verify", path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

G = 2097152
PRO_32K = dict(free_before=84743421952, free_after_demote=85647294464, free_after_restore=84743421952,
               released=905969664)
PRO_8K = dict(free_before=86098182144, free_after_demote=86299508736, free_after_restore=86098182144,
              released=201326592)


def row(c, k, first=None):
    exact, bounded, residual = verify.observe(c["free_before"], c["free_after_demote"],
                                              c["free_after_restore"], c["released"], G)
    return {"cycle": k, "free_before_bytes": c["free_before"], "free_after_demote_bytes": c["free_after_demote"],
            "free_after_restore_bytes": c["free_after_restore"], "vmm_released_chunk_bytes": c["released"],
            "vmm_granularity_bytes": G, "reclaimed_bytes": c["free_after_demote"] - c["free_before"],
            "reacquired_bytes": c["free_after_demote"] - c["free_after_restore"], "residual_bytes": residual,
            "restore_residual_bytes": c["free_before"] - c["free_after_restore"], "reclaim_observed": bounded,
            "reclaim_exact_equal": exact, "g1_reclaim_qualified": str(bounded and residual == 0).lower(),
            "residual_class": "none" if residual == 0 else "unclassified",
            "restored_prefix_state_manifest_sha256": "0" * 64}


class SeriesRuleTests(unittest.TestCase):
    def test_target_card_receipts_as_series(self):
        self.assertEqual(verify.classify_cycles([PRO_32K] * 5, G), verify.ONE_TIME)
        self.assertEqual(verify.classify_cycles([PRO_8K] * 5, G), verify.NONE)
        self.assertEqual(verify.classify_cycles([PRO_32K], G), verify.UNCLASSIFIED)
        self.assertEqual(verify.classify_cycles([PRO_32K] * 2, 0), "not-applicable-pooled")

    def test_growing_and_leaking(self):
        growing = [dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - k * G) for k in range(5)]
        self.assertEqual(verify.classify_cycles(growing, G), verify.GROWING)
        leaking = [dict(free_before=PRO_8K["free_before"] - k * G, free_after_demote=PRO_8K["free_after_demote"] - k * G,
                        free_after_restore=PRO_8K["free_before"] - (k + 1) * G, released=PRO_8K["released"])
                   for k in range(4)]
        self.assertEqual(verify.classify_cycles(leaking, G), verify.GROWING)
        self.assertEqual(verify.classify_cycles([PRO_8K, PRO_32K], G), verify.GROWING)

    def test_every_other_shape_is_unclassified(self):
        two = dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - G)
        never_back = dict(PRO_32K, free_after_restore=PRO_32K["free_after_restore"] - G)
        drifted = {k: v + G for k, v in PRO_32K.items() if k != "released"} | {"released": PRO_32K["released"]}
        for series in ([PRO_32K, PRO_8K], [PRO_32K, PRO_8K, PRO_32K], [two] * 3, [never_back] * 3,
                       [PRO_32K, drifted]):
            with self.subTest(series=series):
                self.assertEqual(verify.classify_cycles(series, G), verify.UNCLASSIFIED)

    def test_mutating_one_cycle_moves_the_one_time_class(self):
        for k in range(5):
            for key in ("free_before", "free_after_demote", "free_after_restore", "released"):
                series = [dict(PRO_32K) for _ in range(5)]
                series[k][key] += 1
                with self.subTest(cycle=k, key=key):
                    self.assertNotEqual(verify.classify_cycles(series, G), verify.ONE_TIME)


class VerdictTests(unittest.TestCase):
    def rows(self, cycles):
        return [row(c, k + 1) for k, c in enumerate(cycles)]

    def test_one_time_metadata_is_recorded_but_never_g1_pass(self):
        rows = self.rows([PRO_32K] * 5)
        text = verify.verdict(32768, rows, verify.ONE_TIME, False, "matches the frozen target-card bundle")
        self.assertEqual(text, "ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, "
                               "residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS")
        self.assertNotIn("—", text)
        with self.assertRaises(ValueError):
            verify.verdict(32768, rows, verify.ONE_TIME, True, "matches the frozen target-card bundle")

    def test_zero_residual_needs_a_frozen_bundle_for_g1(self):
        rows = self.rows([PRO_8K] * 5)
        self.assertEqual(verify.verdict(8192, rows, verify.NONE, True, "matches the frozen target-card bundle"),
                         "ACTIVE-8K G1 PASS across 5 cycles, residual 0 B each cycle, class none")
        text = verify.verdict(8192, rows, verify.NONE, True, "decoded surfaces differ from the rented RTX 5090 bundle")
        self.assertIn("no frozen bundle for this card, not G1 PASS", text)
        with self.assertRaises(ValueError):
            verify.verdict(8192, rows, verify.NONE, False, "matches the frozen target-card bundle")

    def test_growing_series_names_its_shape(self):
        growing = [dict(PRO_32K, free_after_demote=PRO_32K["free_after_demote"] - k * G) for k in range(5)]
        text = verify.verdict(32768, self.rows(growing), verify.GROWING, False, "matches")
        self.assertIn("residual series 2097152,4194304,6291456,8388608,10485760 B, class growing-residual, not G1 PASS", text)
        self.assertIn("reclaim criteria (a)-(c) failed", text)

    def test_series_summary_red_arms(self):
        import tempfile
        rows = self.rows([PRO_32K] * 5)
        header = ["cycle", "free_before_bytes", "free_after_demote_bytes", "free_after_restore_bytes",
                  "vmm_released_chunk_bytes", "reclaimed_bytes", "reacquired_bytes", "residual_bytes",
                  "restore_residual_bytes", "free_before_drift_bytes", "reclaim_observed", "reclaim_exact_equal",
                  "residual_class", "g1_reclaim_qualified", "restored_prefix_state_manifest_sha256"]

        def write(folder, klass="one-time-driver-mapping-metadata", g1="false", mutate=None):
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
                f"all_cycles_restored_bit_identical=true\ng1_reclaim_qualified={g1}\n")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write(folder)
            self.assertEqual(verify.series(folder, rows), (verify.ONE_TIME, False))
            for bad in (dict(klass="none"), dict(klass="growing-residual"), dict(g1="true")):
                write(folder, **bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    verify.series(folder, rows)

            def shift(values):
                values["residual_bytes"] = int(values["residual_bytes"]) + 1
            write(folder, mutate=shift)
            with self.assertRaises(ValueError):
                verify.series(folder, rows)


if __name__ == "__main__":
    unittest.main()
