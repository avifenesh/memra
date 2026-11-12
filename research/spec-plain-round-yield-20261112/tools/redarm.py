#!/usr/bin/env python3
"""Write one red arm of memra#1244's tests from the green worker.rs. usage: redarm.py ARM GREEN OUT
parent   the parent's behaviour under the new names: the hook ignores the round yield, the tick
         never turns it on, a host-sampled session is no rider candidate. The hook test and the
         wiring test fail.
nohook   only the hook ignores the round yield: the hook test and the wiring test fail.
nodue    spec_round_yield_due never says yes: the naming test fails.
nowrite  the tick computes the verdict and writes false: the wiring test fails.
noride   a host-sampled session is no rider candidate: the wiring test fails."""
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
    return swap(s, "                active[i].spec_round_yield = round_yield;\n",
                "                active[i].spec_round_yield = round_yield && false;\n")


def noride(s):
    return swap(s, "        && s.choice.is_none()\n        && lm.model.plain_rider_capable()\n",
                "        && s.choice.is_none()\n        && devsample_meta(s).is_some()\n        && lm.model.plain_rider_capable()\n")


def nodue(s):
    return swap(s, "    admit_yield_on\n        && plain.iter().any(|&(model, tick)| {",
                "    admit_yield_on\n        && false\n        && plain.iter().any(|&(model, tick)| {")


arms = {"parent": lambda s: noride(nowrite(nohook(s))), "nohook": nohook, "nodue": nodue, "nowrite": nowrite, "noride": noride}
out.write_text(arms[arm](green))
