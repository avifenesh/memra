#!/usr/bin/env python3
"""mutants.py NAME CTX_BOOT_RS: apply one source mutation to a copy of ctx_boot.rs (memra#1316).
Each is a red arm of the CPU tests: dd_checks.sh applies it, requires a rebuild and requires the
tests to fail. `mutants.py list` prints the names.

  budget-zero           the on-device red arm (arm_b.py): the retry budget at zero
  budget-ge             a wait that ends exactly on the budget is no longer made
  bound-sleeps-only     the budget counts the sleeps, not elapsed time (the first version)
  retry-discrete        the policy retries on a discrete device too
  any-error             the policy retries any error, not only out of memory
  no-floor              the MemAvailable floor is not applied
  no-reread             the host reading is taken once, before the loop
  wiring-unified-always create_context's wiring treats every device as unified
  wiring-any-error      create_context's wiring takes every driver error for out of memory
  wiring-no-floor       create_context's wiring runs the policy with a zero floor
  attr-error-is-unified an unreadable INTEGRATED attribute reads as unified
  line-names-the-error  the retry line carries the phrase the serving gates scan for
"""
import re
import sys

MUTANTS = {}


def mutant(name):
    def reg(fn):
        MUTANTS[name] = fn
        return fn
    return reg


def sub(s, old, new):
    if s.count(old) != 1:
        sys.exit(f"mutants: anchor found {s.count(old)} times, want 1: {old!r}")
    return s.replace(old, new)


@mutant("budget-zero")
def _(s):
    head, sep, tail = s.partition("pub const RECLAIM_RETRY: ReclaimRetry = ReclaimRetry {")
    tail, n = re.subn(r"budget: [^\n]*,\n", "budget: Duration::ZERO,\n", tail, count=1)
    if not sep or n != 1:
        sys.exit("mutants: budget line not found")
    return head + sep + tail


@mutant("budget-ge")
def _(s):
    return sub(s, "if elapsed + backoff > policy.budget {", "if elapsed + backoff >= policy.budget {")


@mutant("bound-sleeps-only")
def _(s):
    s = sub(s, "    let mut tries = 1u32;\n", "    let mut tries = 1u32;\n    let mut slept = Duration::ZERO;\n")
    s = sub(s, "        let elapsed = now().saturating_sub(started);\n        if elapsed + backoff",
            "        let elapsed = slept;\n        if elapsed + backoff")
    return sub(s, "        wait(backoff);\n", "        wait(backoff);\n        slept += backoff;\n")


@mutant("retry-discrete")
def _(s):
    return sub(s, "if !unified || !is_oom(&err) {", "if !is_oom(&err) {")


@mutant("any-error")
def _(s):
    s = sub(s, "if !unified || !is_oom(&err) {", "if !unified {")
    return sub(s, "Err(e) if is_oom(&e) => err = e,\n            Err(e) => return Err(e),", "Err(e) => err = e,")


@mutant("no-floor")
def _(s):
    return sub(s, "reading.filter(|h| h.available >= policy.min_available)", "reading")


@mutant("no-reread")
def _(s):
    return sub(s, "    loop {\n        let reading = host();", "    let reading = host();\n    loop {")


@mutant("wiring-unified-always")
def _(s):
    return sub(s, "        attribute_says_integrated(integrated),\n", "        true || attribute_says_integrated(integrated),\n")


@mutant("wiring-any-error")
def _(s):
    return sub(s, "        create,\n        is_driver_oom,\n", "        create,\n        |e| true || is_driver_oom(e),\n")


@mutant("wiring-no-floor")
def _(s):
    return sub(s, "    retry_reclaim_race(\n        &RECLAIM_RETRY,\n",
               "    retry_reclaim_race(\n        &ReclaimRetry {\n            min_available: 0,\n            ..RECLAIM_RETRY\n        },\n")


@mutant("attr-error-is-unified")
def _(s):
    return sub(s, "    attr.is_ok_and(|v| v != 0)\n", "    attr.map_or(true, |v| v != 0)\n")


@mutant("line-names-the-error")
def _(s):
    return sub(s, "context creation returned driver code {OOM_CODE} (no \\\n             memory) on unified memory",
               "context creation answered out of memory (code {OOM_CODE}) on \\\n             unified memory")


def main():
    if sys.argv[1] == "list":
        print(" ".join(MUTANTS))
        return
    name, path = sys.argv[1], sys.argv[2]
    if name not in MUTANTS:
        sys.exit(f"mutants: unknown mutation {name}")
    with open(path) as f:
        s = f.read()
    with open(path, "w") as f:
        f.write(MUTANTS[name](s))
    print(f"mutants: {name} applied")


if __name__ == "__main__":
    main()
