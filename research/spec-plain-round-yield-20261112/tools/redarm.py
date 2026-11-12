#!/usr/bin/env python3
"""Write one red arm of memra#1244's tests from the green worker.rs. usage: redarm.py ARM GREEN OUT
parent     the parent's behaviour under the new names: the hook ignores the round yield, the tick
           never turns it on, a host-sampled session is no rider candidate. The hook test and the
           wiring test fail.
nohook     only the hook ignores the round yield: the hook test and the wiring test fail.
nodue      spec_round_yield_due never says yes: the naming test fails.
nowrite    the tick computes the verdict and writes false: the wiring test fails.
noride     a host-sampled session is no rider candidate: the wiring test fails.
nolane     round_yield_weighs ignores the lane: the lane test fails.
nolanewire the tick weighs every live plain session without the helper: the wiring test fails.
noheld     plain_prime_runs ignores phase (b)'s hold: the prime test fails.
nooutlook  the tick passes no hold to plain_prime_runs: the wiring test fails.
nowindows  phase (b)'s time windows never count as closed: the prime test fails."""
import pathlib
import sys

arm, green, out = sys.argv[1], pathlib.Path(sys.argv[2]).read_text(), pathlib.Path(sys.argv[3])


def swap(s, a, b):
    assert s.count(a) == 1, (arm, s.count(a), a[:60])
    return s.replace(a, b)


def nohook(s):
    return swap(s, "        let keep = spec_admit_keep(waiting, self.round_yield, self.entry);\n",
                "        let keep = spec_admit_keep(waiting, false, self.entry);\n")


def nowrite(s):
    return swap(s, "                active[i].spec_round_yield = round_yield.is_some();\n",
                "                active[i].spec_round_yield = round_yield.is_some() && false;\n")


def noride(s):
    return swap(s, "        && s.choice.is_none()\n        && lm.model.plain_rider_capable()\n",
                "        && s.choice.is_none()\n        && devsample_meta(s).is_some()\n        && lm.model.plain_rider_capable()\n")


def nodue(s):
    return swap(s, "    if !admit_yield_on {\n        return None;\n    }\n    plain.iter()",
                "    if admit_yield_on || !admit_yield_on {\n        return None;\n    }\n    plain.iter()")


def nolane(s):
    return swap(s, "    live && !speculative && lane == crate::lanes::Lane::Interactive\n",
                "    let _ = lane;\n    live && !speculative\n")


def nolanewire(s):
    return swap(s, "                    if !round_yield_weighs(!finished.contains(&i), speculative, s.lane) {\n",
                "                    if finished.contains(&i) || speculative {\n")


def noheld(s):
    return swap(s, "        && queued > 0\n        && !held_back\n", "        && queued > 0\n        && (held_back || !held_back)\n")


def nooutlook(s):
    return swap(s, "plain_prime_runs(s.prefill_done, s.prefill_queue.len(), chunk, held_back);\n",
                "plain_prime_runs(s.prefill_done, s.prefill_queue.len(), chunk, held_back && false);\n")


def nowindows(s):
    return swap(s, "    shortest_phase_a.is_some_and(|phase_a| window < phase_a)\n",
                "    let _ = (window, shortest_phase_a);\n    false\n")


arms = {"parent": lambda s: noride(nowrite(nohook(s))), "nohook": nohook, "nodue": nodue, "nowrite": nowrite,
        "noride": noride, "nolane": nolane, "nolanewire": nolanewire, "noheld": noheld, "nooutlook": nooutlook, "nowindows": nowindows}
out.write_text(arms[arm](green))
