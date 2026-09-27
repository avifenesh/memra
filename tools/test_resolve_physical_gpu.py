#!/usr/bin/env python3
"""CPU command-routing tests for resolve-physical-gpu.py (memra#264 follow-up).

No GPU touches this file. Every case supplies a plain device list (the exact shape
`nvidia-smi --query-gpu=index,uuid,memory.free` prints) and asserts what logical device 0
resolves to under a given CUDA_VISIBLE_DEVICES. This is the seam the issue asked for: a
CPU command-routing reproduction that can plant an NVML stand-in and assert the resolved
card, without claiming any native OOM or numerical result.
"""

import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "resolve_physical_gpu", Path(__file__).with_name("resolve-physical-gpu.py")
)
_resolve_physical_gpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resolve_physical_gpu)

GpuResolutionError = _resolve_physical_gpu.GpuResolutionError
parse_nvidia_smi_query = _resolve_physical_gpu.parse_nvidia_smi_query
resolve_physical_device = _resolve_physical_gpu.resolve_physical_device


TWO_CARDS = [
    {"index": "0", "uuid": "GPU-aaaa0000-0000-0000-0000-000000000000", "free_mib": 97000},
    {"index": "1", "uuid": "GPU-bbbb1111-1111-1111-1111-111111111111", "free_mib": 1000},
]


class ParseNvidiaSmiQueryTests(unittest.TestCase):
    def test_parses_index_uuid_free(self):
        text = "0, GPU-aaaa, 97000\n1, GPU-bbbb, 1000\n"
        parsed = parse_nvidia_smi_query(text)
        self.assertEqual(parsed, [
            {"index": "0", "uuid": "GPU-aaaa", "free_mib": 97000},
            {"index": "1", "uuid": "GPU-bbbb", "free_mib": 1000},
        ])

    def test_ignores_blank_lines(self):
        self.assertEqual(parse_nvidia_smi_query("\n0, GPU-aaaa, 97000\n\n"),
                          [{"index": "0", "uuid": "GPU-aaaa", "free_mib": 97000}])

    def test_refuses_malformed_line(self):
        with self.assertRaises(GpuResolutionError):
            parse_nvidia_smi_query("0, GPU-aaaa\n")

    def test_refuses_non_numeric_free(self):
        with self.assertRaises(GpuResolutionError):
            parse_nvidia_smi_query("0, GPU-aaaa, N/A\n")


class ResolvePhysicalDeviceTests(unittest.TestCase):
    def test_unset_selector_uses_natural_order(self):
        # No CUDA_VISIBLE_DEVICES: logical 0 is nvidia-smi's own first device, unchanged
        # from the pre-fix `-i 0` behavior on a single-card host.
        device = resolve_physical_device(TWO_CARDS, None)
        self.assertEqual(device["index"], "0")
        self.assertEqual(device["free_mib"], 97000)

    def test_empty_string_selector_means_no_device_visible(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "")

    def test_ordinal_selector_maps_logical_to_named_physical(self):
        # This is the exact reproduction from the issue: CUDA_VISIBLE_DEVICES selects
        # physical card 1 (1000 MiB free) while physical card 0 (97000 MiB free) sits idle
        # and unselected. A hardcoded `-i 0` would report 97000; this must report 1000.
        device = resolve_physical_device(TWO_CARDS, "1")
        self.assertEqual(device["index"], "1")
        self.assertEqual(device["uuid"], "GPU-bbbb1111-1111-1111-1111-111111111111")
        self.assertEqual(device["free_mib"], 1000)

    def test_ordinal_selector_reordered_visible_set(self):
        # "1,0" puts physical 1 at logical 0 and physical 0 at logical 1, the reordering
        # case the issue names separately from a plain single-selector list.
        device = resolve_physical_device(TWO_CARDS, "1,0")
        self.assertEqual(device["index"], "1")
        self.assertEqual(device["free_mib"], 1000)

    def test_uuid_selector_resolves_directly(self):
        device = resolve_physical_device(
            TWO_CARDS, "GPU-bbbb1111-1111-1111-1111-111111111111"
        )
        self.assertEqual(device["index"], "1")
        self.assertEqual(device["free_mib"], 1000)

    def test_uuid_selector_reordered_visible_set(self):
        cvd = "GPU-bbbb1111-1111-1111-1111-111111111111,GPU-aaaa0000-0000-0000-0000-000000000000"
        device = resolve_physical_device(TWO_CARDS, cvd)
        self.assertEqual(device["index"], "1")

    def test_asymmetric_free_memory_selected_card_starved(self):
        # Selected physical card (1) is starved while the unselected card (0) is wide open.
        # A busy-unselected-card false refusal is exactly the failure mode named in the
        # issue: the resolved card's own number, not the idle neighbor's, must come back.
        device = resolve_physical_device(TWO_CARDS, "1")
        self.assertLess(device["free_mib"], 2048)

    def test_asymmetric_free_memory_selected_card_open(self):
        # Inverse: selected physical card (0) is wide open while the unselected card (1) is
        # starved. An empty-unselected-card false pass is the other failure mode named in
        # the issue; the resolved card's own number must come back, not the busy neighbor's.
        device = resolve_physical_device(TWO_CARDS, "0")
        self.assertGreater(device["free_mib"], 90000)

    def test_mixed_ordinal_and_uuid_selectors_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "0,GPU-bbbb1111-1111-1111-1111-111111111111")

    def test_unknown_ordinal_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "7")

    def test_unknown_uuid_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "GPU-does-not-exist")

    def test_repeated_selector_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "0,0")

    def test_logical_index_out_of_range_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device(TWO_CARDS, "0", logical_index=1)

    def test_no_devices_at_all_refused(self):
        with self.assertRaises(GpuResolutionError):
            resolve_physical_device([], None)


if __name__ == "__main__":
    unittest.main()
