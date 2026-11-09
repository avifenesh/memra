#!/usr/bin/env python3
"""Build the multithreaded-leader red control: the previous runner with this change's tests.

before = production half of PREV darklane.rs (1d1b3d8463, the first fix) + test module of
FINAL darklane.rs, minus the pure-logic test, which calls functions only this change adds.
Usage: gen-before.py PREV_FILE FINAL_FILE OUT_FILE
"""
import hashlib
import sys

prev = open(sys.argv[1]).read()
final = open(sys.argv[2]).read()
out = sys.argv[3]
MARK = "#[cfg(test)]\nmod tests {"
assert prev.count(MARK) == 1 and final.count(MARK) == 1
tests = final[final.index(MARK):]

start = tests.index("    /// The /proc reading, case by case.")
end = tests.index("    /// Set in the job's environment: this test binary then runs as the job leader below.")
tests = tests[:start] + tests[end:]
for name in ("classify_leader_stop", "ThreadStop", "thread_stop(", "LeaderStop"):
    assert name not in tests, name

before = prev[:prev.index(MARK)] + tests
open(out, "w").write(before)
for name, text in (("prev", prev), ("final", final), ("before", before)):
    print(name, hashlib.sha256(text.encode()).hexdigest())
