#!/usr/bin/env python3
"""Build the late-member round's two red controls.

before      = production half of PREV darklane.rs (e47a3bbf04, the scan round) + test module of
              FINAL darklane.rs, minus the pure-logic test, whose table this change moves.
status-text = FINAL with one line changed back: a thread's status is read as text, as it was
              before the scan round. It isolates the byte parse from the getpgid scan.
Usage: gen-before.py PREV_FILE FINAL_FILE BEFORE_OUT STATUS_TEXT_OUT
"""
import hashlib
import sys

prev = open(sys.argv[1]).read()
final = open(sys.argv[2]).read()
MARK = "#[cfg(test)]\nmod tests {"
assert prev.count(MARK) == 1 and final.count(MARK) == 1
tests = final[final.index(MARK):]

start = tests.index("    /// The /proc reading, case by case.")
end = tests.index("    /// A process that renamed itself to bytes that are not UTF-8")
tests = tests[:start] + tests[end:]
for name in ("classify_process_stop", "ThreadStop", "thread_stop(", "Unreached", "scanned_member",
             "process_pgrp"):
    assert name not in tests, name
before = prev[:prev.index(MARK)] + tests
open(sys.argv[3], "w").write(before)

BYTES = """        match std::fs::read(task.path().join("status")) {
            Ok(status) => threads.push((tid, thread_stop(&status).ok_or_else(unparsable)?)),
"""
TEXT = """        match std::fs::read_to_string(task.path().join("status")) {
            Ok(status) => threads.push((
                tid,
                thread_stop(status.as_bytes()).ok_or_else(unparsable)?,
            )),
"""
assert final.count(BYTES) == 1
status_text = final.replace(BYTES, TEXT)
open(sys.argv[4], "w").write(status_text)

for name, text in (("prev", prev), ("final", final), ("before", before), ("status-text", status_text)):
    print(name, hashlib.sha256(text.encode()).hexdigest())
