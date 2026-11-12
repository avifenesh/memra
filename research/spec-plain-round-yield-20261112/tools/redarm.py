#!/usr/bin/env python3
"""Write one red arm of memra#1244's tests from the green worker.rs. usage: redarm.py ARM GREEN OUT
parent       the parent's behaviour under the new names: the hook ignores the round yield, the tick
             never turns it on, a host-sampled session is no rider candidate. The hook test and
             the wiring test fail.
nohook       only the hook ignores the round yield: the hook test and the wiring test fail.
nodue        spec_round_yield_due never says yes: the naming test fails.
nowrite      the tick computes the verdict and writes false: the wiring test fails.
noride       a host-sampled session is no rider candidate: the wiring test fails.
noridelane   a host-sampled session of any lane rides: the wiring test fails.
nolane       round_yield_weighs ignores the lane: the lane test fails.
nolanewire   the tick weighs every live plain session without the helper: the wiring test fails.
noheld       plain_prime_runs ignores phase (b)'s hold: the prime test fails.
nooutlook    the tick passes no hold to plain_prime_runs: the wiring test fails.
nowindows    phase (b)'s time windows never count as closed: the prime test fails.
nochoice     plain_prime_runs ignores a waiting choice group and the walker: the prime test fails.
carriedA     the riding rule's held-back models take the rider candidates too (round-2 review
             red arm A): the carried test fails.
carriedB     no session is a cohort candidate in the riding rule (round-2 review red arm B):
             the wiring test fails.
carriedC     the riding rule's held-back models take dark decoders too: the carried test fails.
nomtp        the yield is decided with no MTP burst, or beside a hookless one: the naming test
             fails.
nocadence    dark work runs on every yielded tick: the cadence test fails.
nocadencewire phase (c) and (d) ignore the dark cadence: the wiring test fails."""
import pathlib
import sys

arm, green, out = sys.argv[1], pathlib.Path(sys.argv[2]).read_text(), pathlib.Path(sys.argv[3])


def swap(s, a, b):
    assert s.count(a) == 1, (arm, s.count(a), a[:60])
    return s.replace(a, b)


RIDE = "        && (devsample_meta(s).is_some() || s.lane == crate::lanes::Lane::Interactive)\n"


def nohook(s):
    return swap(s, "        let keep = spec_admit_keep(waiting, self.round_yield, self.entry);\n",
                "        let keep = spec_admit_keep(waiting, false, self.entry);\n")


def nowrite(s):
    return swap(s, "                active[i].spec_round_yield = round_yield.is_some();\n",
                "                active[i].spec_round_yield = round_yield.is_some() && false;\n")


def noride(s):
    return swap(s, RIDE, "        && devsample_meta(s).is_some()\n")


def noridelane(s):
    return swap(s, RIDE, "        && (devsample_meta(s).is_some() || true)\n")


def nodue(s):
    return swap(s, "    if !admit_yield_on || !mtp_bursts || hookless_bursts {\n        return None;\n    }\n",
                "    if admit_yield_on || !admit_yield_on {\n        return None;\n    }\n")


def nomtp(s):
    return swap(s, "    if !admit_yield_on || !mtp_bursts || hookless_bursts {\n",
                "    if !admit_yield_on {\n        let _ = (mtp_bursts, hookless_bursts);\n")


def nolane(s):
    return swap(s, "    live && !speculative && lane == crate::lanes::Lane::Interactive\n",
                "    let _ = lane;\n    live && !speculative\n")


def nolanewire(s):
    return swap(s, "                    if !round_yield_weighs(!finished.contains(&i), speculative, s.lane) {\n",
                "                    if finished.contains(&i) || speculative {\n")


def noheld(s):
    return swap(s, "        && queued > 0\n        && !held_back\n", "        && queued > 0\n        && (held_back || !held_back)\n")


def nooutlook(s):
    return swap(s, "                        chunk,\n                        held_back,\n                        choice_waiting(s),\n",
                "                        chunk,\n                        held_back && false,\n                        choice_waiting(s),\n")


def nowindows(s):
    return swap(s, "    shortest_phase_a.is_some_and(|phase_a| window < phase_a)\n",
                "    let _ = (window, shortest_phase_a);\n    false\n")


def nochoice(s):
    return swap(s, "        && !choice_waits\n        && !walks\n", "        && (choice_waits || !choice_waits)\n        && (walks || !walks)\n")


def carried_a(s):
    return swap(s, "        if f.plain_decoding && !f.rider && f.interactive {\n",
                "        if f.plain_decoding && (f.rider || !f.rider) && f.interactive {\n")


def carried_b(s):
    return swap(s, "        cohort: spec_pool_candidate(s, lm),\n", "        cohort: spec_pool_candidate(s, lm) && false,\n")


def carried_c(s):
    return swap(s, "        if f.plain_decoding && !f.rider && f.interactive {\n",
                "        if f.plain_decoding && !f.rider && (f.interactive || !f.interactive) {\n")


def nocadence(s):
    return swap(s, "        if self.yielded >= rounds.unwrap_or(1).max(1) {\n",
                "        if self.yielded >= rounds.unwrap_or(1).max(1) || true {\n")


def nocadencewire(s):
    s = swap(s, "                    || !(dark_turn || active[i].lane == crate::lanes::Lane::Interactive)\n", "")
    s = swap(s, "                        && (dark_turn || active[i].lane == crate::lanes::Lane::Interactive)\n", "")
    return swap(s, "            let adaptive_cap = if dark_turn {\n", "            let adaptive_cap = if dark_turn || true {\n")


arms = {"parent": lambda s: noride(nowrite(nohook(s))), "nohook": nohook, "nodue": nodue, "nowrite": nowrite,
        "noride": noride, "noridelane": noridelane, "nolane": nolane, "nolanewire": nolanewire, "noheld": noheld,
        "nooutlook": nooutlook, "nowindows": nowindows, "nochoice": nochoice, "carriedA": carried_a,
        "carriedB": carried_b, "carriedC": carried_c, "nomtp": nomtp, "nocadence": nocadence,
        "nocadencewire": nocadencewire}
out.write_text(arms[arm](green))
