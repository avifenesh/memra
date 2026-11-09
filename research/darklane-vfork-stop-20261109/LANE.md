# Background stop acknowledgement and vfork

Verdict: a stop-mode yield that lands while the job's leader waits in vfork is never acknowledged. The runner stays `running`, never publishes `yielded` and never sends SIGCONT, and the job stays frozen until shutdown. This explains the `darklane` stop-cycle flake ([original](tests/original-flake.txt): same job and symptom, reproduced here on 3.2% of stops) and matches #939's unexplained failure. The runner now reads the stop per thread from /proc, for the leader when waitid has nothing to report and for every other member of the job's process group, and directs it at each thread of a multithreaded process on which no thread has taken it. Before: the deterministic tests fail. After: they pass, and a 300-cycle stress on a constantly spawning dash job acknowledged every stop. The multithreaded cases came from review: [Multithreaded leaders](#multithreaded-leaders-review-round) and [Group members](#group-members-re-review).

## Mechanism

dash 0.5.12 (`/bin/sh` on Ubuntu 26.04 and 24.04) runs every external command through `vfork` ([strace](probe-clean/strace-dash.txt)); bash forks ([strace](probe-clean/strace-bash.txt)). glibc's `posix_spawn` and CPython 3.14's `subprocess.run` clone with `CLONE_VM|CLONE_VFORK` ([strace](probe-first/strace-python-subprocess.txt), [strace](probe-first/strace-python-posix-spawn.txt)).

A vfork parent waits in the kernel, in state D, until its child execs or exits. A group SIGSTOP posts the stop to both. The child, which has not exec'd yet, stops at its next return to user space. The parent cannot act on its own pending stop while it waits, and the child cannot exec until SIGCONT. The leader never stops, so `waitid(WSTOPPED)` never reports it.

Since #951 (704084a40f) the runner publishes `yielded` only after waitid reports the leader stopped. In this state it waits forever: the job stays frozen (leader D, child T), `/metrics` says `running`, and the next valley never resumes the job. Only shutdown recovers it (CONT, TERM, group KILL). Before #951 the runner published `yielded` on signal submission and resumed normally; only the test's wait for state T failed, which is #939's observation (runner yielded after 2 ms, shell not T within 30 s, job `while :; do sleep 0.01; done`). This lane has no /proc state from #939's run; the job and the mechanism are the same.

Exposure: `MEMRA_BG_JOB` runs through `sh -c`, and dash stays the leader. A shell job spawns every external command through vfork. dash 0.5.12 does not exec even a single-command job (`sh -c 'sleep 0.001'` vforks `sleep`, [strace](probe-clean/strace-dash-single.txt)), so a yield in the first moments after launch hits the same window; after that dash waits in `wait4` and stops normally. A job started with `exec` makes the program the leader, and a program that spawns with `posix_spawn` or CPython `subprocess` has the window on every spawn. A stopped thread of a multithreaded leader does not help: the group stop completes only when every thread stops. When the thread waiting in vfork is the leader's main thread, the stop can be lost on the other threads too, which then keep running ([review round](#multithreaded-leaders-review-round)). The same holds for a multithreaded program that dash runs without exec, as another member of the group, while the shell itself stops at once ([re-review](#group-members-re-review)).

## Proof

[`vfork_stop_probe.py`](vfork_stop_probe.py) mirrors the runner: `SHELL -c CMD` in its own process group, `kill(-pgid, SIGSTOP)` after a random 1 to 15 ms, the runner's waitid acknowledgement polled for 0.5 s, then SIGCONT. A miss records every group member from /proc and `kcmp(KCMP_VM)`.

x86_64, Ubuntu 26.04.1, kernel 7.0.0-38-generic, dash 0.5.12-12ubuntu3, bash 5.3-2ubuntu1, glibc 2.43, 16 pinned CPUs at nice 10:

| run | job | stops | never acknowledged |
|---|---|---|---|
| [clean](probe-clean/) | dash `while :; do sleep 0.01; done` | 1000 | 32 (3.2%) |
| [clean](probe-clean/) | dash `while :; do /bin/true; done` | 1000 | 265 (26.5%) |
| [clean](probe-clean/) | bash, both jobs | 2000 | 0 |
| [clean](probe-clean/hold-dash-sleep.jsonl), hold 3 s | dash sleep loop | 44 | 3 |
| [first](probe-first/), stops at 3 misses, hold 3 s | dash sleep loop / true loop | 18 / 5 | 3 / 3 |
| [first](probe-first/), a cargo build started on the same CPUs during it | bash, both jobs | 6000 | 0 |

All 306 misses have one shape. One of them, raw (`probe-clean/hold-dash-sleep.jsonl`, abridged to the fields that carry the proof):

```
leader  sh  state D  wchan kernel_clone  syscall 58 (vfork)  ShdPnd 0000000000050000 (SIGCHLD, SIGSTOP)
child   sh  state T  wchan do_signal_stop  flags 0x400040 (PF_FORKNOEXEC: not exec'd)  exe /usr/bin/dash  kcmp VM vs leader: same mm
waitid at 0.5 s: nothing; at 3.5 s: nothing, same states
re-stop right after SIGCONT: not acknowledged within 0.5 s
```

The wait for the vfork child is inlined into `kernel_clone` on this kernel, hence the wchan. The re-stop missed in 9 of 9 held cases: the child had not run between the CONT and the STOP.

## Fix

`child_stop_acknowledged` keeps its contract (the stop took effect, exits stay with `try_wait`). When waitid has nothing to report, the leader's threads are read from `/proc/<pid>/task/*/status` (`State`, `SigPnd`, `ShdPnd`); since the re-review every other member of the group is read the same way (see [Group members](#group-members-re-review)). The stop has taken effect when every live thread is stopped (T, or t under a tracer), or blocked uninterruptibly (D, or I) and bound to stop: some thread is stopped, which means a group stop is under way and marked every thread for it; or SIGSTOP is in that thread's own pending set (`SigPnd`); or the leader has one live thread and SIGSTOP is still in the shared pending set (`ShdPnd`). Such a thread handles the stop before it next returns to user space, so no leader thread runs job code again before SIGCONT. The first version of this fix accepted a stop still in `ShdPnd` for any number of blocked threads; that holds for one live thread only (review round below). A running leader inside an ordinary vfork is D with nothing pending and is not acknowledged. VRAM residency is unchanged: a vfork-parked leader keeps its memory like a stopped one. Resume is unchanged: SIGCONT lets the child exec and clears the parent's pending stop.

Rejected:

- cgroup v2 freezer (`cgroup.freeze`). It is the kernel's own answer: the vfork waiter counts as frozen, and it would also catch members that left the process group. It needs a cgroup the server can write. Under an SSH login on the 26.04 host the server's cgroup is a root-owned session scope (`not-writable`, checked 2026-11-09); containers usually mount cgroupfs read-only; a system service needs `Delegate=yes`. SIGSTOP would remain the fallback on those launches, so the SIGSTOP path has to be correct by itself.
- Acknowledging on signal submission, as before #951. It brings back #951's defect: a zombie leader read as yielded.
- CONT and STOP again after a timeout. The re-stop right after SIGCONT missed 9 of 9 times; a delay long enough to let the child exec lets job code run during traffic, and the outcome stays probabilistic.

Not covered, unchanged from before: processes that left the process group (the group's signals do not reach them either), and a member stopped by a tracer that suppresses the stop.

## Tests

- `stop_acknowledges_a_leader_parked_in_vfork_on_a_child_stopped_before_exec` is deterministic. The test binary itself is the job leader: a thread calls `posix_spawn` with an open file action on a FIFO, so the vfork child blocks before exec until the test opens the write end. The stop lands with the spawning thread in D and the other threads stopping. The test asserts the acknowledgement, the shape (threads D and T, the vfork child T, waitid reporting nothing), the resume, and exit 0 after the FIFO opens.
- `stop_resume_cycles_on_a_constantly_spawning_shell_job` (ignored, stress): 300 stop and resume cycles on dash `while :; do /bin/true; done`.
- `leader_thread_states_classify_the_stop`: the /proc reading case by case, including the parked dash leader from the probe and a running shell inside an ordinary vfork.
- `repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown` and `stop_mode_full_cycle_launch_yield_resume_shutdown` asserted state T after the yield. Their job runs `sleep` through dash, so they now accept the parked shape: every thread T, or T/D threads with a stopped child that has not exec'd. Since the review round they wait for it (30 s guard), as do the deterministic and stress tests: at ba03102882 the acknowledgement read the leader's threads only, and the vfork child could still be runnable with its stop pending at that moment. Since 82b9b52534 that child is a group member and the acknowledgement waits for it to stop, so these waits are redundant but harmless.

Red control: [`red/gen-before.py`](red/gen-before.py) builds the base runner with this change's tests: the production half of the base `darklane.rs` (sha256 `b7a0b7db4a24937ddc88d9e969f6869ccd83d0df2305b4d870e120585f2ec170`, unchanged from 65fb297a59 to fc8d6ef689) and the test module of the fixed file, minus the pure-logic test, which calls functions only the fix adds. [`red/red.sh`](red/red.sh) runs the `darklane::` module, stress included, on that file and then on the fixed one. Results, all on the host above:

| run | tree (darklane.rs sha256) | deterministic test | stress | darklane module |
|---|---|---|---|---|
| [red](red/before.txt) | base runner `76ae1cbd2515154a5cff56c7f7ea32618676c6b99ded88afc040266287ae5d7f` | FAIL at the 30 s guard, state `running`, yields 0 | FAIL on stop 6, leader D, yields 5 | 15 passed, 2 failed |
| [red](red/final.txt) | fix `f7a7f6d4f6d30e4c0619b79140bb8b83a48f12560e6472fb194c73aed422bcec` | PASS | PASS, 300 of 300, 20 parked in vfork | 18 passed |
| [first](tests/) | base runner, tests before formatting | FAIL at the 30 s guard, state `running`, yields 0 | FAIL on stop 1, leader D | 15 passed, 1 failed, 1 ignored |
| [first](tests/) | fix, before formatting | PASS (0.14 s) | PASS, 300 of 300, 122 parked in vfork | 17 passed, 1 ignored |

After the fix, rebased on fc8d6ef689 ([verify/](verify/summary.txt)): 20 runs of `darklane::` with the stress test, 18 of 18 passed each time, 6,000 stops acknowledged, 2,268 of them with the leader parked in vfork (91 to 136 per run). The complete memra-server suite passed twice: 1,118 passed, 0 failed, 11 ignored in the library, 7 integration tests. Release all-target Clippy with warnings denied is clean ([clippy.log](verify/clippy.log), dependency build lines dropped; [clippy2.log](verify/clippy2.log), re-lint after touching `darklane.rs`), and so are `cargo fmt --all -- --check` and `git diff --check`. The scripts are in [`harness/`](harness/); they wrote `proof/` and `proof2/`, committed here as `probe-first/` and `probe-clean/`.

## Multithreaded leaders (review round)

The review of the first version of this fix (1d1b3d8463, [PR comment](https://github.com/avifenesh/memra/pull/1130#issuecomment-6742393509)) found two gaps, both on a leader with more than one thread, and a race in three test asserts. The kernel queues a group SIGSTOP once, in the shared pending set, and flags one thread to take it: the main thread first, if it can take signals (`complete_signal()` in kernel/signal.c). Only that thread acts on it.

- No thread takes the stop. When the flagged main thread is the one waiting in vfork, the other threads are never told. They keep running and never start the group stop, the leader reads D and S, and the first version read that as running, forever. In the default setup a side effect saves it: the stopped vfork child's CLD_STOPPED SIGCHLD goes to another thread (glibc's `posix_spawn` blocks every signal in the spawning thread), which then takes the SIGSTOP too. A SIGCHLD handler installed with SA_NOCLDSTOP, or other threads that block SIGCHLD, remove that path. dash leaders are single-threaded, so this needs a multithreaded program in the group: as the leader when the job starts it with `exec`, and as another member when it does not, which the first review missed ([re-review](#group-members-re-review)).
- A blocked thread that was not flagged runs on. The first version acknowledged every thread blocked with SIGSTOP in the shared set. That holds for one live thread, which is the thread the kernel flagged. With more, a blocked thread that was not flagged returns to user space when its wait ends and runs job code during the yield, while the runner reports `yielded`.

The review's probe, [`multithread/probe/`](multithread/probe/): a C leader ([`leader.c`](multithread/probe/leader.c)) whose main thread posix_spawns a child that blocks opening a FIFO before exec, a second thread that bumps a counter every 1 ms, and a stdlib Python driver ([`driver.py`](multithread/probe/driver.py)) that sends the group SIGSTOP and polls the first version's acknowledgement, ported line for line, every 25 ms for 3 s. Five runs each, on the second host named under [Scope](#scope) ([summary](multithread/probe/repeat-summary.txt), raw JSONL in [`repeat/`](multithread/probe/repeat/)):

| leader (main thread parked in vfork unless noted) | first version acknowledges | threads at 3 s | second thread's counter at the stop, at 3 s |
|---|---|---|---|
| second thread sleeps; SIGCHLD handler with SA_NOCLDSTOP | never | main D, second S | 42-52, 2864-2884 |
| second thread sleeps and blocks SIGCHLD | never | main D, second S | 45-53, 2840-2887 |
| second thread first waits in vfork on a child that leaves the group and execs after 1.5 s; SA_NOCLDSTOP | at 7-12 ms | main D, second S | 0, 1458-1502 |
| first row, plus SIGSTOP to the second thread by tgkill | at 9-12 ms | main D, second T | frozen |
| third row, plus SIGSTOP to the second thread by tgkill | at 7-12 ms | main D, second T | 0, 0 |
| single thread (the dash shape) | at 10-12 ms | main D | none |
| a spawner thread parked, the main thread joining (the first deterministic test's shape) | at 12-46 ms | T, T, D | frozen |

After the tgkill, one group SIGCONT resumed the second thread with no SIGSTOP left pending in the shared set or in either thread's own set, and the leader exited 0 once the FIFO opened ([`resume-after-tgkill.txt`](multithread/probe/resume-after-tgkill.txt)).

Fix: the reading is per thread, from `/proc/<pid>/task/*/status`, and a blocked thread counts only when it is bound to stop (the [Fix](#fix) section's three conditions). When no thread is stopped, more than one is live and some live thread lacks SIGSTOP in its own set, the runner sends SIGSTOP to each live thread with tgkill and reads again. A sleeping thread wakes, takes its stop and starts the group stop, which marks every thread, the parked one included: the next tick reads a stopped thread. A blocked thread keeps the stop in its own set and takes it before it next returns to user space. tgkill checks the thread group, so a stale tid cannot reach another process; ESRCH, a thread that exited after the listing, is skipped. SIGCONT flushes every stop signal from the shared set and from each thread's own set, so resume is unchanged. A single-threaded leader is never sent a thread-directed stop.

Tests:

- `stop_reaches_the_other_thread_of_a_leader_whose_main_thread_waits_in_vfork`: the probe's `never` shape. The test binary is the job leader again, but the leader half runs from an `.init_array` constructor before libtest starts, so the process's main thread is the one that posix_spawns; the second thread inherits a blocked SIGCHLD from it and sleeps in a loop. The test asserts the acknowledgement, the shape (main thread D, second thread T, the vfork child T, waitid reporting nothing), the resume, and exit 0 after the FIFO opens.
- `stop_holds_the_other_blocked_thread_of_a_leader_whose_main_thread_waits_in_vfork`: the probe's third row, made deterministic. The second thread waits in `clone(CLONE_VM|CLONE_VFORK)` on a child that leaves the job's process group and blocks opening a second FIFO before exec. After the acknowledgement the test opens that FIFO, which ends the second thread's wait. The second thread must stop before it runs job code (creating a file), and run it after the resume; the leader then exits 0.
- `leader_threads_classify_the_stop` replaces the case table: the readings above, before and after the stop is directed at each thread.
- The asserts on the shape after the yield, in the repeated-cycle, deterministic and stress tests, now wait for it under the 30 s guard, as `stop_mode_full_cycle_launch_yield_resume_shutdown` already did. At ba03102882 the acknowledgement read the leader's threads only, and the vfork child could still be runnable with its stop pending then; since the re-review it is a group member and the acknowledgement waits for it (see [Group members](#group-members-re-review)). The review's starvation probe ([`starve.py`](multithread/probe/starve.py), leader and child at nice 19 on one CPU beside two nice-0 busy loops, the tests' 5 ms tick) found the child in R at 9 of 30 acknowledgements, and T 200 ms later in all 9 ([summary](multithread/probe/starve-summary.txt), [raw](multithread/probe/starve.jsonl)).

Red control: [`multithread/harness/gen-before.py`](multithread/harness/gen-before.py) builds the previous runner with this round's tests: the production half of 1d1b3d8463's `darklane.rs` (sha256 `f7a7f6d4f6d30e4c0619b79140bb8b83a48f12560e6472fb194c73aed422bcec`, the first round's fix) and the test module of the fixed file, minus the pure-logic test, which calls functions only this round adds. [`multithread/harness/red.sh`](multithread/harness/red.sh) runs the `darklane::` module, stress included, on that file and then on the fixed one. The crates/ digest is the sha256 of `sha256sum` over every file under `crates/`, sorted in the en_US.UTF-8 locale; the fix arm's digest is that of the committed tree. On x86_64 Ubuntu 26.04.1, kernel 7.0.0-38-generic, glibc 2.43, dash 0.5.12-12ubuntu3, 16 pinned CPUs at nice 10:

| run | tree | main thread parked, second sleeping | second thread blocked too | darklane module |
|---|---|---|---|---|
| [red](multithread/red/before.txt) | previous runner: darklane.rs `a37ec0ae1734446b767a858ebdcb539e5823a59648238a6fe02d981bd67e67ec`, crates/ `059390a0396f0a0a063bbde648bbb0a5c403d68430630ec33e549ad8f017e97d` | FAIL at the 30 s guard, state `running`, threads D and S, yields 0 | FAIL, job code ran after the yield was acknowledged: state `yielded`, threads D and S | 17 passed, 2 failed |
| [red](multithread/red/final.txt) | fix: darklane.rs `c7b1c52557e1b29fb8c953186a920655f1206bee5db9eb8a2b3d4ecbd4ada000`, crates/ `0b81281a048a7e21e01ba88d8c3df150c3a53729a9858f3370d5cf8ca99496be` | PASS | PASS | 20 passed |

Verification of the fix, same host ([`multithread/verify/`](multithread/verify/summary.txt), [`multithread/harness/verify.sh`](multithread/harness/verify.sh)): 20 runs of `darklane::` with the stress test, 20 of 20 passed each time, 6,000 stops acknowledged, 301 of them with the leader parked in vfork (12 to 23 per run). That share follows the box's load, not the code: on the first round's production code it read 20 in the red control and 91 to 136 in the verification. The complete memra-server suite passed twice: 1,120 passed, 0 failed, 11 ignored in the library, 7 integration tests. `cargo fmt --all -- --check` and release all-target Clippy with warnings denied are clean, and `darklane.rs` had the same sha256 at the end. `verify.sh` grepped `parked in vfork:` for its summary, which also matches the stress test's new wait lines; [`multithread/harness/assemble.py`](multithread/harness/assemble.py) rebuilt `summary.txt` from the raw logs, keeping the result line `verify.sh` wrote for each run and the stress count from its log. It and [`stage_probe.py`](multithread/harness/stage_probe.py) copied the receipts and the review's probe here with the work directory and host redacted.

## Group members (re-review)

The re-review of ba03102882 ([comment](https://github.com/avifenesh/memra/pull/1130#issuecomment-6743656100)) found the same lost stop one process down; the code before this lane had it too. `launch` runs `sh -c CMD` and dash does not exec even a single command, so `MEMRA_BG_JOB=/path/to/prog` makes the program a member of the job's process group that is not its leader. The acknowledgement read the leader only. When the program's main thread waits in `posix_spawn` and its other threads cannot take the CLD_STOPPED SIGCHLD, the kernel flags only the program's main thread and the other threads run on. dash, waiting in `wait4`, stops at once, so waitid reports the leader stopped and the runner published `yielded` while the program kept running.

The re-review's probe, [`members/probe/`](members/probe/): a Rust harness that includes ba03102882's acknowledgement code verbatim ([`extract.py`](members/probe/extract.py) copies it out of the file with sha256 `c7b1c525...`) and drives it like `runner_loop`: `sh -c CMD` in its own process group, the group SIGSTOP 300 ms after launch, the acknowledgement every 25 ms for up to 3 s, the program's counter at the yield and 2 s later, then SIGCONT and the FIFO released. Programs: the review round's [`leader.c`](multithread/probe/leader.c) and [`busyleader.c`](members/probe/busyleader.c), whose main thread blocks SIGCHLD, starts eight busy threads that inherit the mask, and posix_spawns the FIFO child. Three runs each, two for `busyleader`, on the second host named under [Scope](#scope) ([summary](members/probe/summary.txt), [busyleader summary](members/probe/busy-summary.txt), raw [`results.jsonl`](members/probe/results.jsonl) and [`busy.jsonl`](members/probe/busy.jsonl)):

| job (`sh -c ...`) | leader | `yielded` at | program's other threads at the yield, 2 s later | counter at the yield, 2 s later |
|---|---|---|---|---|
| `busyleader` | sh | 25 ms | 8 R, 8 R | 1025-1041, 7215-7275 |
| `leader main-spawn-maskchld` | sh | 25 ms | S, S | 303-310, 2129-2186 |
| `leader main-spawn-nocldstop` | sh | 25 ms | S, S | 313-316, 2170-2190 |
| `leader two-block-nocldstop` | sh | 25 ms | D, S | 0, 766-785 |
| `leader main-spawn` (default SIGCHLD) | sh | 25 ms | T, T | frozen |
| each of the above with `exec` in front | the program | 25-50 ms | T, T (two-block: D, then T) | frozen |

Fix: `child_stop_acknowledged` reads every member of the job's process group, found by scanning /proc for the processes whose process group (`/proc/<pid>/stat`) is the job's. Each member gets the leader's per-thread reading (`process_stop`), including the per-thread tgkill when no thread of it has taken the stop. A member with no live thread (exiting, or a zombie not yet reaped) cannot run job code and counts as stopped, and so does a member that is gone by the time it is read. The leader keeps its rules: waitid first, and its exit is left to `try_wait`. The members are read after the leader's waitid stop report too, because in this shape waitid on the leader reports first. Every member is read on every tick, so each member whose stop is unclaimed gets it directed in the same tick. Before the tgkill to a member other than the leader, its process group is read again, so the stop stays inside the job's group; pids are allocated cyclically, so a member that exited since the scan is not replaced within a tick in any case. A /proc scan rather than a walk of `/proc/<pid>/task/*/children`: the scan also finds members whose parent exited, which are reparented away from the leader but stay in the group and get the group's SIGSTOP, and it does not need `CONFIG_PROC_CHILDREN`. Its cost is one read of `/proc/<pid>/stat` per process on the host per tick, and only while a stop is pending. The stated limit stays: a process that left the group gets neither the group's signals nor this check.

One consequence: the dash shape's vfork child is a member too, so the acknowledgement now waits until it reads stopped. The review round's third finding (the child can still be runnable with its stop pending when the leader reads parked) is now held by the runner, not only by the tests' waits.

Test: `stop_reaches_the_other_thread_of_a_group_member_whose_main_thread_waits_in_vfork` runs the review round's main-thread leader (the `.init_array` half, its second thread sleeping with SIGCHLD blocked) through `sh -c` without exec; a trailing `exit $?` keeps any shell from exec'ing it. Right at the acknowledgement, with no wait, it asserts the shell T, the program's main thread D and its other thread T, and the program's vfork child T; then the resume, and exit 0 once the FIFO opens. The pure-logic test is `process_threads_classify_the_stop`: a process with no live thread reads `Gone`, and it covers the process-group parse.

Red control: [`members/harness/gen-before.py`](members/harness/gen-before.py) builds the previous runner with this round's tests: the production half of ba03102882's `darklane.rs` (sha256 `c7b1c52557e1b29fb8c953186a920655f1206bee5db9eb8a2b3d4ecbd4ada000`, the review round's fix) and the test module of the fixed file, minus the pure-logic test. [`members/harness/red.sh`](members/harness/red.sh) runs the `darklane::` module, stress included, on that file and then on the fixed one; the crates/ digest is computed as in the review round, and the fix arm's is that of the committed tree. Same host and pinning as the review round:

| run | tree | program as a group member, its main thread parked | darklane module |
|---|---|---|---|
| [red](members/red/before.txt) | previous runner: darklane.rs `60f3a80ec94d25dfd90fdc6fc8de71875419f30505ab7a589e53b6c0b5acf255`, crates/ `510f810186c1674505e1aa717b482d3955dbf02eac29f5a62330a790a1021f46` | FAIL at the acknowledgement: state `yielded`, the shell T, the program's threads D and S | 19 passed, 1 failed |
| [red](members/red/final.txt) | fix: darklane.rs `189b26b905637752340eac0fc573fc756f5bec18daae84b7be8656cced9d1e03`, crates/ `4dcbdce1fcaf4257e61285abae761e7ab9ecd9237b29fda38467c9fd503af560` | PASS | 21 passed |

Verification of the fix, same host ([`members/verify/`](members/verify/summary.txt), [`members/harness/verify.sh`](members/harness/verify.sh)): 20 runs of `darklane::` with the stress test, 21 of 21 passed each time, 6,000 stops acknowledged, 309 of them with the leader parked in vfork (8 to 23 per run). The complete memra-server suite passed twice: 1,121 passed, 0 failed, 11 ignored in the library, 7 integration tests. `cargo fmt --all -- --check` and release all-target Clippy with warnings denied are clean, and `darklane.rs` had the same sha256 at the end. [`members/harness/assemble.py`](members/harness/assemble.py) and [`stage_members_probe.py`](members/harness/stage_members_probe.py) copied the receipts and the re-review's probe here with the work directory and host redacted.

## Group scan (landing review)

A review of the landing PR ([comment](https://github.com/avifenesh/memra/pull/1150#discussion_r4517915674)) found that any process on the host could fail the job. The group scan read `/proc/<pid>/stat` of every listed process as text and passed up every error but "gone". The kernel prints a process's comm in that line as it is, so one process anywhere on the host whose name is not UTF-8 made the read an error; so does another user's entry on a /proc mounted with `hidepid`. `child_stop_acknowledged` returned the error and the runner marked the job `failed`, which is terminal: never resumed, never relaunched. This is new with the group-member fix; the leader-only code read no other process.

Fix: the scan no longer reads anything of a process outside the job. It asks the kernel for each listed process's process group (`getpgid`), which needs no read access to the entry and has no line to parse. A refused answer passes the process over (the kernel itself refuses getpgid to no one; a security module can), and any other error still fails the stop, because a scan that may have missed a member must not acknowledge. The re-check before a directed stop uses the same call. The thread readings (`/proc/<pid>/task/<tid>/status`) are taken only from the leader and from known members, and their `Name:` line carries the comm as bytes too, so they are now parsed as bytes; the kernel escapes a newline there, so a comm cannot forge a `State:` line. Their errors stay errors: a process of the job that cannot be read fails the job instead of counting as stopped. This replaces the per-process `/proc/<pid>/stat` read described under "Group members"; the scan's cost per tick is one getpgid per listed process, and only while a stop is pending.

Tests, both through the runner: `stop_is_acknowledged_beside_a_process_outside_the_job_whose_comm_is_not_utf8` keeps a process outside the job's group alive whose comm is `\xff\xfedl` (a shell that wrote its own `/proc/$$/comm`; the test checks that its stat line is not UTF-8) and requires the job to yield and resume beside it. `stop_reads_a_group_member_whose_comm_is_not_utf8` gives that name to the job's own program, a member of the group, and asserts right at the acknowledgement that the member is stopped. The pure-logic test covers the byte parse (a comm that is not UTF-8 and spells a `State:` field) and the scan's rule for each answer: member, other group, gone, refused, any other error.

Red control: [`scan/harness/gen-before.py`](scan/harness/gen-before.py) builds the previous runner with this round's tests: the production half of 8795e8d8c5's `darklane.rs` (sha256 `9717b88a6c12605d4e7e21864976a207849655cde638623e55c6d5e28e09c39a`) and the test module of the fixed file, minus the pure-logic test. [`scan/harness/red.sh`](scan/harness/red.sh) runs the `darklane::` module, stress included, on that file and then on the fixed one. Same host and pinning as the earlier rounds:

| run | tree | job beside a process whose comm is not UTF-8 | darklane module |
|---|---|---|---|
| [red](scan/red/before.txt) | previous runner: darklane.rs `d759f6babf4deb7c0018fdaf01c4a2df8c3845b962335742510acf72d6b45ff7`, crates/ `fc4cc7d8362f7032750b5fe8549a65394c8307d2ba7d757536915e2292459f1f` | FAIL: state `failed` in both new tests, and in six other stop tests that ran while those processes were alive | 14 passed, 8 failed |
| [red](scan/red/final.txt) | fix: darklane.rs `2314b608c4b5818fa70fbe7989e166f2f59768c6ef87810fa3167c73b3bca38e`, crates/ `98b081e342286aaa1952c855117d883cc261ff0ee0cc8e2e89755c4a1d91ec67` | PASS | 23 passed |

The six other failures are the finding's reach, not noise: the tests share one host, so the two byte-named processes failed every job whose stop was read while they lived. Which tests overlap them varies with scheduling.

Verification of the fix, same host ([`scan/verify/`](scan/verify/summary.txt), [`scan/harness/verify.sh`](scan/harness/verify.sh)): 20 runs of `darklane::` with the stress test, 23 of 23 passed each time, 6,000 stops acknowledged, 329 of them with the leader parked in vfork (11 to 21 per run). The complete memra-server suite passed twice: 1,142 passed, 0 failed, 11 ignored in the library, 7 integration tests. `cargo fmt --all -- --check` and release all-target Clippy with warnings denied are clean, and `darklane.rs` had the same sha256 at the end. [`scan/harness/assemble.py`](scan/harness/assemble.py) copied the receipts in with the same redactions as the earlier rounds.

The stated limits stay: a process that left the group gets neither the group's signals nor this check, and a member the server cannot see in /proc (a `hidepid` mount that hides it) is outside the scan.

## Late members, and the review of the scan round

Two reviews are answered here: an earlier review comment on this PR ([late members](https://github.com/avifenesh/memra/pull/1130#discussion_r4516293834)) and the clean-context review of the scan round ([comment](https://github.com/avifenesh/memra/pull/1130#issuecomment-6753223068)).

Late members. A process the group's stop never reached was read as "not yet" on every tick when it had one thread: the stop was pending nowhere on it, nothing would ever make it stop, and the runner only directed a stop at processes with more than one live thread. The job then never yielded and never resumed, the same end state as the first finding of this lane. A forked child cannot be such a process, because a group signal that arrives during fork is delivered to the child as well. The ways in are `setpgid` into the job's group after the stop was sent (a process of the job that had left the group, or a child a member moves in) and a SIGCONT from outside the job to a member that had stopped. Fix: a member other than the leader whose one live thread has not stopped and carries SIGSTOP in neither pending set (`Unreached`) gets the stop directed at it, like the threads of an unclaimed multithreaded process. A thread that has just taken the stop off its queue reads the same for an instant; the extra stop it then gets is flushed by SIGCONT with every other stop signal. The leader is left out: it was in the group when the stop was sent, and directing a stop at it on that reading would stop a running job whenever the reading is taken with no stop pending (the first draft did, and `repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown` caught it in the status-text arm of this round's first run: [`late/first-draft/`](late/first-draft/status-text.txt), darklane.rs `d00acb1ae76c10f59665fa8243578a1de98b2912b86382e50923377821fe7f73`).

Review of the scan round, all six points taken:

- A refused `getpgid` was passed over for every process, so a refusal for all of them (a filter on the server's own `getpgid`) read as "no members". The scan now asks for the leader first, the server's own unreaped child, and fails the stop when that answer is refused.
- The gap that stays is named in the module header and the `MEMRA_BG_YIELD_MODE` row: a process whose process group a security module refuses to tell the server is passed over, so a member confined that way is not waited for. SELinux and Smack have that hook.
- "No other process on the host can fail it" said more than the code does. An error listing /proc, or reading a known member, still fails the job, on purpose: the sentence now reads "cannot fail the job by its name or its /proc permissions".
- The byte-named member test used a shell loop that stops by itself, so it passed whether or not the member was read, and its red arm failed on the stat scan, not on the status parse. It is replaced by `stop_reaches_the_other_thread_of_a_group_member_whose_comm_is_not_utf8`: the vfork program of the group-member round with its comm set to `\xff\xfedl` (`prctl`), whose other thread runs on unless the runner reads the member and directs the stop. The job line ends in `exit $?`, so no shell execs it.
- The `status-text` arm isolates the byte parse: the fixed file with one line changed back, a thread's `status` read as text.
- The `crates/` digest sorts paths in the host's locale, en_US.UTF-8, in every round; the harness now says so.

New test for late members: `stop_is_directed_at_a_member_the_group_stop_never_reached` builds the shape without the race. Two `sleep` processes share a group, the stop is sent to the leader alone, and the member stays asleep with no stop pending. The same reading of the leader directs nothing at it; then the acknowledgement must direct the stop at the member and come only once the member reads stopped.

Red controls ([`late/harness/gen-before.py`](late/harness/gen-before.py), [`late/harness/red.sh`](late/harness/red.sh)), same host and pinning as the earlier rounds. `before` is the production half of e47a3bbf04's `darklane.rs` (sha256 `2314b608c4b5818fa70fbe7989e166f2f59768c6ef87810fa3167c73b3bca38e`) with this round's tests, minus the pure-logic test:

| run | tree | result | darklane module |
|---|---|---|---|
| [before](late/red/before.txt) | previous runner: darklane.rs `fe7c161ac42de6110ada59d30e6ed344aa332520916f3904c59d48da229a0718`, crates/ `593f4b59b43338306f85e55a105f49106a5567bfeb9c9348dce733b7832f380d` | FAIL: the member the stop never reached is not acknowledged within the 30 s guard | 22 passed, 1 failed |
| [status-text](late/red/status-text.txt) | fix with a thread's status read as text: darklane.rs `118ef756f8aed6f1309048f22467caa79f4d9109c8117f1f3b91a6ee3d8fde11`, crates/ `c0862ef667d6c298fffba520467520ec7638d6d14d68f321192c15c4370c3045` | FAIL: the byte-named member's job ends in state `failed` | 23 passed, 1 failed |
| [final](late/red/final.txt) | fix: darklane.rs `1b1332e9250758035532950ebd846295acd1c5e6864c83ebdb9d0ac8773261db`, crates/ `72ac4a50add90f4e368a3c91240ccfd123f03c30783abef87147f17f8262542f` | PASS | 24 passed |

Verification of the fix, same host ([`late/verify/`](late/verify/summary.txt), [`late/harness/verify.sh`](late/harness/verify.sh)): 20 runs of `darklane::` with the stress test, 24 of 24 passed each time, 6,000 stops acknowledged, 2,392 of them with the leader parked in vfork (15 to 151 per run; the share follows the box's load, which was higher for part of this run). The complete memra-server suite passed twice: 1,143 passed, 0 failed, 11 ignored in the library, 7 integration tests. `cargo fmt --all -- --check` and release all-target Clippy with warnings denied are clean, and `darklane.rs` had the same sha256 at the end.

Limits after this round: a process that left the group gets neither the group's signals nor this check; a member the server cannot see in /proc (a `hidepid` mount that hides it) or whose process group a security module refuses is outside the scan; and a leader continued from outside the job with its stop still unacknowledged is read again, not stopped again.

## Scope

Platforms: the first round's probe and every round's tests ran on one x86_64 Ubuntu 26.04.1 host. A second x86_64 26.04 host has the same dash, kernel and glibc; both reviews' probes ran there, the first round's probe did not. DGX Spark: Ubuntu 24.04.5, dash 0.5.12-6ubuntu5, aarch64, kernel 7.0.0-1019-nvidia, glibc 2.39; read, not run. arm64 has no vfork syscall; glibc implements vfork with clone and `CLONE_VFORK`, the kernel's wait is the same, and so are the /proc formats.

No GPU and no model are used. This CPU lifecycle fix qualifies no runtime, model or serving binary. Hostnames and work-directory paths in the raw files are redacted (`<host>`, `$WORK`). Trailing spaces and final blank lines in the `.log` and `.txt` files and both reviews' probe files are stripped; nothing else is edited, apart from the review round's rebuilt `multithread/verify/summary.txt` described above.
