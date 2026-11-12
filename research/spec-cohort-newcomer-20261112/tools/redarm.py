#!/usr/bin/env python3
"""Write one red arm of memra#1227's tests. usage: redarm.py ARM GREEN BASE OUT"""
import pathlib
import sys

arm, green, base, out = sys.argv[1], pathlib.Path(sys.argv[2]).read_text(), pathlib.Path(sys.argv[3]).read_text(), pathlib.Path(sys.argv[4])


def swap(s, a, b):
    assert s.count(a) == 1, (arm, s.count(a), a[:60])
    return s.replace(a, b)


def between(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


if arm == "main":
    fa, fb = "/// The two passes of phase (a) (memra#1227)", "/// A session that may join this tick's pooled MTP cohort"
    ta, tb = "    /// memra#1227: the sessions that step before the pooled cohorts", "    /// A burst ends at the round its public stream ends"
    s = swap(base, fb, between(green, fa, fb) + fb)
    s = swap(s, tb, between(green, ta, tb) + tb)
elif arm == "oldorder":
    s = swap(green, "    order.iter().copied().filter(|&i| cold_serial(i)).collect()\n",
             "    let _ = (order, cold_serial);\n    Vec::new()\n")
elif arm == "nojoin":
    s = swap(green, "        SpecPass::Cohorts => !stepped_first || pooled,\n",
             "        SpecPass::Cohorts => {\n            let _ = pooled;\n            !stepped_first\n        }\n")
elif arm == "swapped":
    s = swap(green, "            for pass in [SpecPass::ColdSerial, SpecPass::Cohorts] {\n",
             "            for pass in [SpecPass::Cohorts, SpecPass::ColdSerial] {\n")
elif arm == "nopeer":
    s = swap(green, "                    && s.spec_pool_peer\n                    && lm.model.spec_pool_capable()\n                    && !spec_pool_candidate(s, lm)\n",
             "                    && !spec_pool_candidate(s, lm)\n")
elif arm == "noquantum":
    a = "                    for &i in &spec_order {\n                        let owes = memra_engine::prime_walker::prime_yield_enabled()\n"
    b = "                        active[i].prime_service.peer_quantum = owes;\n                    }\n"
    i = green.index(a)
    j = green.index(b, i) + len(b)
    assert green.count(a) == 1
    s = green[:i] + green[j:]
else:
    raise SystemExit(f"unknown arm {arm}")
out.write_text(s)
