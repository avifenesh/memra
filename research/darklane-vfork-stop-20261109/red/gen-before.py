#!/usr/bin/env python3
"""Build the red control: the base commit's runner with this change's tests.

before = production half of BASE darklane.rs + test module of FINAL darklane.rs, minus the
pure-logic test (it calls functions only the fix adds) and with the test-local proc_state
parser of BASE (FINAL's calls the fix's proc_stat_state).
Usage: gen-before.py BASE_FILE FINAL_FILE OUT_FILE
"""
import hashlib
import sys

base = open(sys.argv[1]).read()
final = open(sys.argv[2]).read()
out = sys.argv[3]
MARK = "#[cfg(test)]\nmod tests {"
assert base.count(MARK) == 1 and final.count(MARK) == 1
tests = final[final.index(MARK):]

old_proc_state = base[base.index("    fn proc_state(pid: u32)"):]
old_proc_state = old_proc_state[:old_proc_state.index("\n    }\n") + len("\n    }\n")]
new_proc_state = tests[tests.index("    fn proc_state(pid: u32)"):]
new_proc_state = new_proc_state[:new_proc_state.index("\n    }\n") + len("\n    }\n")]
tests = tests.replace(new_proc_state, old_proc_state, 1)

start = tests.index("    /// The /proc reading, case by case.")
end = tests.index("    /// Set in the job's environment: this test binary then runs as the job leader below.")
tests = tests[:start] + tests[end:]
assert "classify_leader_threads" not in tests and "proc_stat_state" not in tests

before = base[:base.index(MARK)] + tests
open(out, "w").write(before)
for name, text in (("base", base), ("final", final), ("before", before)):
    print(name, hashlib.sha256(text.encode()).hexdigest())
