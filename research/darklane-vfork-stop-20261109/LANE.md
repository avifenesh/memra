# Background stop acknowledgement and vfork

Verdict: a stop-mode yield that lands while the job's leader waits in vfork is never acknowledged. The runner stays `running`, never publishes `yielded` and never sends SIGCONT, and the job stays frozen until shutdown. This explains the `darklane` stop-cycle flake ([original](tests/original-flake.txt): same job and symptom, reproduced here on 3.2% of stops) and matches #939's unexplained failure. The runner now reads the stop per thread from /proc when waitid has nothing to report. Before: the new deterministic test fails at its 30 s guard. After: it passes, and a 300-cycle stress on a constantly spawning dash job acknowledged every stop.

## Mechanism

dash 0.5.12 (`/bin/sh` on Ubuntu 26.04 and 24.04) runs every external command through `vfork` ([strace](probe-clean/strace-dash.txt)); bash forks ([strace](probe-clean/strace-bash.txt)). glibc's `posix_spawn` and CPython 3.14's `subprocess.run` clone with `CLONE_VM|CLONE_VFORK` ([strace](probe-first/strace-python-subprocess.txt), [strace](probe-first/strace-python-posix-spawn.txt)).

A vfork parent waits in the kernel, in state D, until its child execs or exits. A group SIGSTOP posts the stop to both. The child, which has not exec'd yet, stops at its next return to user space. The parent cannot act on its own pending stop while it waits, and the child cannot exec until SIGCONT. The leader never stops, so `waitid(WSTOPPED)` never reports it.

Since #951 (704084a40f) the runner publishes `yielded` only after waitid reports the leader stopped. In this state it waits forever: the job stays frozen (leader D, child T), `/metrics` says `running`, and the next valley never resumes the job. Only shutdown recovers it (CONT, TERM, group KILL). Before #951 the runner published `yielded` on signal submission and resumed normally; only the test's wait for state T failed, which is #939's observation (runner yielded after 2 ms, shell not T within 30 s, job `while :; do sleep 0.01; done`). This lane has no /proc state from #939's run; the job and the mechanism are the same.

Exposure: `MEMRA_BG_JOB` runs through `sh -c`, and dash stays the leader. A shell job spawns every external command through vfork. dash 0.5.12 does not exec even a single-command job (`sh -c 'sleep 0.001'` vforks `sleep`, [strace](probe-clean/strace-dash-single.txt)), so a yield in the first moments after launch hits the same window; after that dash waits in `wait4` and stops normally. A job started with `exec` makes the program the leader, and a program that spawns with `posix_spawn` or CPython `subprocess` has the window on every spawn. A stopped thread of a multithreaded leader does not help: the group stop completes only when every thread stops.

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

`child_stop_acknowledged` keeps its contract (the stop took effect, exits stay with `try_wait`). When waitid has nothing to report, `leader_stop_took_effect` reads `/proc/<pid>/task/*/stat`. The stop has taken effect when every live thread is stopped (T, or t under a tracer), or blocked uninterruptibly (D, or I) with the stop still in force: some thread is stopped, which means a group stop is under way, or SIGSTOP is still in the leader's shared pending set (`ShdPnd`). A blocked thread handles the pending stop before it next returns to user space, so no leader thread runs job code again before SIGCONT. A running leader inside an ordinary vfork is D with nothing pending and is not acknowledged. VRAM residency is unchanged: a vfork-parked leader keeps its memory like a stopped one. Resume is unchanged: SIGCONT lets the child exec and clears the parent's pending stop.

Rejected:

- cgroup v2 freezer (`cgroup.freeze`). It is the kernel's own answer: the vfork waiter counts as frozen, and it would also catch members that left the process group. It needs a cgroup the server can write. Under an SSH login on the 26.04 host the server's cgroup is a root-owned session scope (`not-writable`, checked 2026-11-09); containers usually mount cgroupfs read-only; a system service needs `Delegate=yes`. SIGSTOP would remain the fallback on those launches, so the SIGSTOP path has to be correct by itself.
- Acknowledging on signal submission, as before #951. It brings back #951's defect: a zombie leader read as yielded.
- CONT and STOP again after a timeout. The re-stop right after SIGCONT missed 9 of 9 times; a delay long enough to let the child exec lets job code run during traffic, and the outcome stays probabilistic.

Not covered, unchanged from before: processes that left the process group, and a leader stopped by a tracer that suppresses the stop.

## Tests

- `stop_acknowledges_a_leader_parked_in_vfork_on_a_child_stopped_before_exec` is deterministic. The test binary itself is the job leader: a thread calls `posix_spawn` with an open file action on a FIFO, so the vfork child blocks before exec until the test opens the write end. The stop lands with the spawning thread in D and the other threads stopping. The test asserts the acknowledgement, the shape (threads D and T, the vfork child T, waitid reporting nothing), the resume, and exit 0 after the FIFO opens.
- `stop_resume_cycles_on_a_constantly_spawning_shell_job` (ignored, stress): 300 stop and resume cycles on dash `while :; do /bin/true; done`.
- `leader_thread_states_classify_the_stop`: the /proc reading case by case, including the parked dash leader from the probe and a running shell inside an ordinary vfork.
- `repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown` and `stop_mode_full_cycle_launch_yield_resume_shutdown` asserted state T after the yield. Their job runs `sleep` through dash, so they now accept the parked shape: every thread T, or T/D threads with a stopped child that has not exec'd.

Red control: [`red/gen-before.py`](red/gen-before.py) builds the base runner with this change's tests: the production half of the base `darklane.rs` (sha256 `b7a0b7db4a24937ddc88d9e969f6869ccd83d0df2305b4d870e120585f2ec170`, unchanged from 65fb297a59 to fc8d6ef689) and the test module of the fixed file, minus the pure-logic test, which calls functions only the fix adds. [`red/red.sh`](red/red.sh) runs the `darklane::` module, stress included, on that file and then on the fixed one. Results, all on the host above:

| run | tree (darklane.rs sha256) | deterministic test | stress | darklane module |
|---|---|---|---|---|
| [red](red/before.txt) | base runner `76ae1cbd2515154a5cff56c7f7ea32618676c6b99ded88afc040266287ae5d7f` | FAIL at the 30 s guard, state `running`, yields 0 | FAIL on stop 6, leader D, yields 5 | 15 passed, 2 failed |
| [red](red/final.txt) | fix `f7a7f6d4f6d30e4c0619b79140bb8b83a48f12560e6472fb194c73aed422bcec` | PASS | PASS, 300 of 300, 20 parked in vfork | 18 passed |
| [first](tests/) | base runner, tests before formatting | FAIL at the 30 s guard, state `running`, yields 0 | FAIL on stop 1, leader D | 15 passed, 1 failed, 1 ignored |
| [first](tests/) | fix, before formatting | PASS (0.14 s) | PASS, 300 of 300, 122 parked in vfork | 17 passed, 1 ignored |

After the fix, rebased on fc8d6ef689 ([verify/](verify/summary.txt)): 20 runs of `darklane::` with the stress test, 18 of 18 passed each time, 6,000 stops acknowledged, 2,268 of them with the leader parked in vfork (91 to 136 per run). The complete memra-server suite passed twice: 1,118 passed, 0 failed, 11 ignored in the library, 7 integration tests. Release all-target Clippy with warnings denied is clean ([clippy.log](verify/clippy.log), dependency build lines dropped; [clippy2.log](verify/clippy2.log), re-lint after touching `darklane.rs`), and so are `cargo fmt --all -- --check` and `git diff --check`. The scripts are in [`harness/`](harness/); they wrote `proof/` and `proof2/`, committed here as `probe-first/` and `probe-clean/`.

Platforms: the probe ran on one x86_64 Ubuntu 26.04.1 host. A second x86_64 26.04 host has the same dash and kernel (read, not run). DGX Spark: Ubuntu 24.04.5, dash 0.5.12-6ubuntu5, aarch64, kernel 7.0.0-1019-nvidia, glibc 2.39; read, not run. arm64 has no vfork syscall; glibc implements vfork with clone and `CLONE_VFORK`, the kernel's wait is the same, and so are the /proc formats.

No GPU and no model are used. This CPU lifecycle fix qualifies no runtime, model or serving binary. Hostnames and work-directory paths in the raw files are redacted (`<host>`, `$WORK`). Trailing spaces and final blank lines in the `.log` and `.txt` files are stripped; nothing else is edited.
