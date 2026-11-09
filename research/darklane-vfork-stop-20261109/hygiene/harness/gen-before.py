#!/usr/bin/env python3
"""Build the test-hygiene round's two red controls (the continued-job round's, with this round's tests): earlier runners with this change's tests.

Each is the production half of an earlier darklane.rs plus the test module of FINAL darklane.rs,
minus the pure-logic test, whose table this change moves:
before          = e47a3bbf04 (the scan round): no stop is directed at a one-thread process.
leader-excluded = fff211cb6d (the late-member round): it is, at members other than the leader.
Usage: gen-before.py SCAN_FILE LATE_FILE FINAL_FILE BEFORE_OUT LEADER_EXCLUDED_OUT
"""
import hashlib
import sys

scan, late, final = (open(path).read() for path in sys.argv[1:4])
MARK = "#[cfg(test)]\nmod tests {"
assert all(text.count(MARK) == 1 for text in (scan, late, final))
tests = final[final.index(MARK):]

start = tests.index("    /// The /proc reading, case by case.")
end = tests.index("    /// A process that renamed itself to bytes that are not UTF-8")
tests = tests[:start] + tests[end:]
for name in ("classify_process_stop", "ThreadStop", "thread_stop(", "ProcessStop", "scanned_member",
             "process_pgrp"):
    assert name not in tests, name

before = scan[:scan.index(MARK)] + tests
excluded = late[:late.index(MARK)] + tests
open(sys.argv[4], "w").write(before)
open(sys.argv[5], "w").write(excluded)
for name, text in (("scan", scan), ("late", late), ("final", final), ("before", before),
                   ("leader-excluded", excluded)):
    print(name, hashlib.sha256(text.encode()).hexdigest())
