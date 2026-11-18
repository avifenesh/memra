#!/usr/bin/env python3
"""Induced delays for the two constraint compiler tests of crates/memra-server/src/constrained.rs (memra#1335).
Each arm edits the TEST code only, in place, and applies to main's file and to the fixed file alike.

  late-start    each test's compile closure sleeps 300 ms before it sends its start signal:
                a worker thread that the host schedules late.
  late-dequeue  each test's compile factory sleeps 300 ms on its first call, which the supervisor
                thread makes before it reads its queue: a supervisor that wakes late, after the
                first job was submitted.

Added for the review fold:

  wrong-deadline       PRODUCTION code of constrained.rs: the supervisor waits
                       CONSTRAINT_COMPILE_TIMEOUT for a worker instead of the job's own deadline.
  worker-late-dequeue  FILE is crates/memra-server/src/worker.rs: the compile factory of
                       slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress
                       sleeps 300 ms on its first call (late-dequeue for the third test).

usage: mutate.py ARM FILE"""
import sys
from pathlib import Path

arm, path = sys.argv[1], Path(sys.argv[2])
text = path.read_text()
SLEEP = "std::thread::sleep(Duration::from_millis(300));"


def once(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)


if arm == "late-start":
    old = "let _ = started_tx.send(());"
    assert text.count(old) == 2, text.count(old)
    text = text.replace(old, f"{SLEEP}\n                    {old}")
elif arm == "late-dequeue":
    first = ("static LATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);\n"
             "            if LATE.swap(false, std::sync::atomic::Ordering::SeqCst) {\n"
             f"                {SLEEP}\n"
             "            }\n")
    # the sibling test's factory
    once("        let compiler = ConstraintCompiler::spawn_for_test(result_tx, move || {\n",
         "        let compiler = ConstraintCompiler::spawn_for_test(result_tx, move || {\n            " + first)
    # the issue's test's factory
    once("            move || {\n                let calls = Arc::clone(&calls);\n                let release = Arc::clone(&release);\n",
         "            move || {\n            " + first
         + "                let calls = Arc::clone(&calls);\n                let release = Arc::clone(&release);\n")
elif arm == "wrong-deadline":
    once("                    let wait = job.deadline.saturating_duration_since(Instant::now());\n",
         "                    let wait = CONSTRAINT_COMPILE_TIMEOUT;\n")
elif arm == "worker-late-dequeue":
    once("            crate::constrained::ConstraintCompiler::spawn_for_test(result_tx, move || {\n"
         "                let started_tx = started_tx.clone();\n"
         "                let release_rx = std::sync::Arc::clone(&release_rx);\n",
         "            crate::constrained::ConstraintCompiler::spawn_for_test(result_tx, move || {\n"
         "                static LATE: std::sync::atomic::AtomicBool =\n"
         "                    std::sync::atomic::AtomicBool::new(true);\n"
         "                if LATE.swap(false, std::sync::atomic::Ordering::SeqCst) {\n"
         "                    std::thread::sleep(std::time::Duration::from_millis(300));\n"
         "                }\n"
         "                let started_tx = started_tx.clone();\n"
         "                let release_rx = std::sync::Arc::clone(&release_rx);\n")
else:
    raise SystemExit(f"unknown arm {arm}")
path.write_text(text)
print(f"applied {arm}")
