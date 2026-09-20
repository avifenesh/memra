#!/usr/bin/env python3
"""CPU-only adversarial tests of day-10 verdict arithmetic; no GPU claims."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name("verify-day10.py")
spec = importlib.util.spec_from_file_location("verify", path)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class VerdictTests(unittest.TestCase):
    def fixture(self):
        metrics = dict(demote_count=32, reload_count=32, device_charged_after_demote=0,
                       pinned_charged_after_demote=25, logical_d2h_bytes=25,
                       free_before_bytes=100, free_after_demote_bytes=120,
                       free_after_restore_bytes=100, reclaimed_bytes=20, reacquired_bytes=20,
                       vmm_granularity_bytes=2, vmm_released_chunk_bytes=20, residual_bytes=0)
        flags = dict(vmm_fixed_va_restored="true", reclaim_observed="true",
                     reclaim_exact_equal="true", g1_reclaim_qualified="true", residual_class="none")
        return metrics, flags

    def test_exact(self):
        self.assertEqual(verify.verdict(*self.fixture(), 8192), "ACTIVE-8K G1 PASS")

    def test_arithmetic_and_engagement_red_arms(self):
        for key in ("demote_count", "reload_count", "device_charged_after_demote",
                    "pinned_charged_after_demote", "free_before_bytes", "free_after_demote_bytes",
                    "free_after_restore_bytes", "reclaimed_bytes", "reacquired_bytes",
                    "vmm_released_chunk_bytes", "residual_bytes"):
            with self.subTest(key=key):
                m, f = self.fixture()
                m[key] += 1
                with self.assertRaises(ValueError):
                    verify.verdict(m, f, 8192)

    def test_nonzero_class_on_one_card_cannot_pass(self):
        m, f = self.fixture()
        m.update(free_after_demote_bytes=118, reclaimed_bytes=18, reacquired_bytes=18, residual_bytes=2)
        f.update(reclaim_exact_equal="false", g1_reclaim_qualified="false")
        for klass in ("unclassified", "va-reservation-page-table"):
            f["residual_class"] = klass
            self.assertIn("not G1 PASS", verify.verdict(m, f, 32768))
            f["g1_reclaim_qualified"] = "true"
            with self.assertRaises(ValueError):
                verify.verdict(m, f, 32768)
            f["g1_reclaim_qualified"] = "false"

    def test_pooled_cannot_claim_fixed_va_or_g1(self):
        m, f = self.fixture()
        m["vmm_granularity_bytes"] = 0
        for key in ("vmm_fixed_va_restored", "g1_reclaim_qualified", "residual_class"):
            f[key] = "not-applicable-pooled"
        self.assertIn("not-applicable-pooled", verify.verdict(m, f, 8192))
        for key in ("vmm_fixed_va_restored", "g1_reclaim_qualified", "residual_class"):
            bad = dict(f, **{key: "true"})
            with self.assertRaises(ValueError):
                verify.verdict(m, bad, 8192)


if __name__ == "__main__":
    unittest.main()
