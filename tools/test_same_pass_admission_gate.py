#!/usr/bin/env python3
"""tools/same-pass-admission-gate.py's log reading, no GPU (memra#1255).

`pair_shape()` decides V1 (the pair met in one admission pass on the DFlash route) and V2 (it was
not admitted twice on one reading); `charge_of()` reads a request's charge from the server's lines.
The unfixed tree's pass reads placed and overlapping; the fixed tree's reads placed through its
settle wait and not overlapping; a pair the worker took in two passes, or priced plain, reads not
placed; the keeper's own lines never count.
Run: python3 tools/test_same_pass_admission_gate.py
"""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "same_pass_admission_gate", Path(__file__).resolve().parent / "same-pass-admission-gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
MB = gate.MB
PAIR = {5772, 5741}


def spec_k(prompt, active):
    return (f'[spec-k] model="gate" tenant="" K=0 source=eligibility-fallback prompt={prompt} cached=0 '
            f'lcp=0 active={active} wave=2 placement=single-or-non-pp2')


PRIME = "[prime-yield] count=1 remaining=1 chunks=1 wall_ms=2270.200 max_chunk_ms=2270.200"
KEEPER_ACC = "[dspark-acc] ctx=180 burst=20/31 cum=61/106=0.575"
SETTLE = ('[admission] settle wait: model="gate" ctx=5901 effective free 1650MB after 816MB booked for '
          '1 admitted session(s) with no route state yet < cost 814MB + reserve 1576MB; requeued for the '
          'next pass [queue 1]')
OOM = ('[admit-oom] step OOM parked session back to queue (model gate, retry 1/3, 2 active) [dev0 free 54MB '
       'pool-res 20960MB pool-used 19660MB]: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")')
COST_SPEC = ('[admission] request cost: model="gate" ctx=5901 path=spec = 29696 B/token x ctx + 470MB '
             'prefill-workspace + 0MB fixed = 645MB')
COST_PLAIN = ('[admission] request cost: model="gate" ctx=5901 path=plain = 29696 B/token x ctx + 477MB '
              'prefill-workspace + 0MB fixed = 652MB')


class PairShape(unittest.TestCase):
    def test_unfixed_pass_admits_both_on_one_reading(self):
        lines = [KEEPER_ACC, COST_SPEC, spec_k(5772, 2), COST_SPEC, spec_k(5741, 3), PRIME, OOM]
        s = gate.pair_shape(lines, PAIR)
        self.assertTrue(s["placed"])
        self.assertTrue(s["both_admitted_in_one_pass"])
        self.assertEqual(s["oom_lines"], 1)

    def test_fixed_pass_settles_the_second(self):
        lines = [COST_SPEC, spec_k(5772, 2), SETTLE, KEEPER_ACC, PRIME, COST_SPEC, spec_k(5741, 3), PRIME]
        s = gate.pair_shape(lines, PAIR)
        self.assertTrue(s["placed"])
        self.assertFalse(s["both_admitted_in_one_pass"])
        self.assertEqual(s["settle_waits"], 1)
        self.assertEqual(s["oom_lines"], 0)

    def test_two_passes_without_a_settle_wait_are_not_placed(self):
        lines = [COST_SPEC, spec_k(5772, 2), KEEPER_ACC, COST_SPEC, spec_k(5741, 3), PRIME]
        s = gate.pair_shape(lines, PAIR)
        self.assertFalse(s["placed"])
        self.assertFalse(s["both_admitted_in_one_pass"])

    def test_keeper_admissions_never_count(self):
        lines = [COST_SPEC, spec_k(5772, 2), COST_SPEC, spec_k(180, 2), KEEPER_ACC, COST_SPEC, spec_k(5741, 2)]
        s = gate.pair_shape(lines, PAIR)
        self.assertFalse(s["both_admitted_in_one_pass"])
        self.assertFalse(s["placed"])

    def test_a_late_settle_wait_does_not_place_the_pair(self):
        lines = [COST_SPEC, spec_k(5772, 2), PRIME, SETTLE, COST_SPEC, spec_k(5741, 2), PRIME]
        self.assertFalse(gate.pair_shape(lines, PAIR)["placed"])

    def test_a_pair_priced_plain_is_not_placed(self):
        # The first GPU run's shape: beside the keeper the pair was a projected wave of 3 and the
        # default spec band priced it plain; the two admissions were also a pass apart, with no
        # prime line in the window at all.
        lines = [COST_PLAIN, spec_k(5741, 2)] + [KEEPER_ACC] * 3 + [COST_PLAIN, spec_k(5772, 2)]
        s = gate.pair_shape(lines, PAIR)
        self.assertEqual(s["pair_routes"], ["plain", "plain"])
        self.assertFalse(s["placed"])
        self.assertFalse(s["both_admitted_in_one_pass"])

    def test_a_decode_burst_between_the_admissions_is_a_tick(self):
        lines = [COST_SPEC, spec_k(5772, 2), KEEPER_ACC, COST_SPEC, spec_k(5741, 2)]
        self.assertFalse(gate.pair_shape(lines, PAIR)["both_admitted_in_one_pass"])


class Reserve(unittest.TestCase):
    def test_the_applied_reserve_is_read_from_a_defer_or_settle_line(self):
        defer = ('[admit-oom] VRAM defer: 2 active, effective free 2112MB (driver + 1652MB pool-cached) '
                 '< cost 815MB + reserve 1653MB \u2014 queueing (FIFO) [pool res 22750MB used 21098MB]')
        self.assertEqual(gate.RE_RESERVE.search(defer).group(1), "1653")
        self.assertEqual(gate.RE_RESERVE.search(SETTLE).group(1), "1576")
        self.assertIsNone(gate.RE_RESERVE.search(COST_SPEC))


class ChargeOf(unittest.TestCase):
    def test_spec_charge_is_cost_plus_plane(self):
        lines = [
            '[admission] per-session draft-state charge: model="gate" +169MB (DFlash draft plane at this cap)',
            '[admission] request cost: model="gate" ctx=5901 path=spec = 29696 B/token x ctx + 470MB '
            'prefill-workspace + 0MB fixed = 645MB',
        ]
        self.assertEqual(gate.charge_of(lines), (645 + 169) * MB)

    def test_a_plain_charge_is_refused(self):
        lines = ['[admission] request cost: model="gate" ctx=5901 path=plain = 29696 B/token x ctx + 573MB '
                 'prefill-workspace + 0MB fixed = 748MB']
        self.assertIsNone(gate.charge_of(lines))

    def test_no_cost_line(self):
        self.assertIsNone(gate.charge_of([PRIME]))


if __name__ == "__main__":
    unittest.main()
