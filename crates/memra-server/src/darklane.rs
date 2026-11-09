//! Dead-darklane background jobs — valley detection + a yield-first job runner
//! (lane/darklane-training, 2026-08-07).
//!
//! THE THESIS (owner, standing): idle serve capacity carries owner research/training jobs,
//! yielding instantly to paying traffic. This module is the ENGINE half only: when the box
//! is idle, what runs, and how it gets out of the way. Scheduling policy and economics
//! (which jobs, what they're worth, when a valley is "worth" filling) belong to the product
//! repo — the seam is `MEMRA_BG_JOB` + the checkpoint protocol below.
//!
//! VALLEY DETECTION reuses worker truth that already exists instead of inventing a new
//! sensor: the scheduler flips `health` to `PHASE_IDLE` exactly when `active.is_empty() &&
//! queue.is_empty()` (worker.rs loop top) and `set_phase` stamps the beat on entry — so
//! `phase == IDLE` + `beat_age_ms` IS the idle duration, to the millisecond, with zero new
//! hot-path cost. `PENDING_ADMITS` closes the HTTP→worker handoff gap (a request the handler
//! has submitted but the worker hasn't popped yet is traffic, not idleness). A dedicated serve
//! route (the DSv4 thread) is not the scheduler, so the snapshot's `idle_for_ms` folds every
//! registered route in: the process is idle only while the scheduler AND every route are idle
//! with nothing queued, and for as long as the most recently idled of them (memra#500).
//!
//! THE LANE CLASS: below EVERY serving lane. Harvest is still a *request* class the engine
//! admits and schedules; a background job is not a request at all — it runs only while the
//! engine has NOTHING (no interactive, no judge, no harvest, no queue) and yields on the
//! first sign of any of them. Asymmetric hysteresis on purpose: yield fires on the busy
//! EDGE (any activity, no debounce — paying traffic never waits for a threshold), resume
//! waits for a full `MEMRA_VALLEY_S` of quiet (a between-requests gap in an active
//! conversation is not a valley).
//!
//! YIELD MECHANISM v1 — simplest honest first: the job is a CHILD PROCESS in its own
//! process group. The runner submits SIGSTOP/SIGCONT on poll ticks (`MEMRA_BG_POLL_MS`,
//! default 25 ms) and publishes yielded once a later tick observes the stop take effect on
//! every member of the job's process group: the leader (`sh -c`) and every process the job
//! started that stayed in the group, which usually includes the job's program, since dash does
//! not exec even a single command. A member's stop took effect when waitid reports it stopped
//! (the leader only), or /proc shows every thread of it stopped, or blocked in the kernel and
//! bound to stop before it next runs: some thread is stopped (the group stop is under way),
//! SIGSTOP is in the thread's own pending set, or, for a process with one live thread only,
//! SIGSTOP is still in the shared pending set. A member other than the leader with no live
//! thread (exiting, or a zombie not yet reaped) counts as stopped. The leader never does: a
//! leader that exited or is gone is never published as yielded, and try_wait reaps it with its
//! real exit status (#951). The /proc form exists for vfork: a thread waiting in vfork on a child that the same stop
//! caught before exec stays in D until SIGCONT, so waitid never reports the stop (dash spawns
//! every external command that way; `research/darklane-vfork-stop-20261109/`). The kernel flags
//! one thread per process to take a group stop, the main thread first; when that thread waits
//! in vfork, the other threads are never told. So when no thread of a multithreaded member has
//! stopped, the runner directs SIGSTOP at each of its live threads (tgkill) and reads again.
//! Members are found by process group in /proc; a process that left the group is outside the
//! group's signals and outside this check. OS scheduling and signal delivery control latency.
//! Historical timing receipts live in `research/darktrain-20260807/`. Two consequences the
//! operator must know:
//!   * a SIGSTOPPED process KEEPS its memory — VRAM included, and so does a leader parked
//!     in vfork. The VRAM budget below is therefore carved out for the LIFE of the job, not
//!     per-valley.
//!   * SIGSTOP is only cheap for jobs whose working set can sit cold. GPU-resident training
//!     that can't sit on stopped VRAM uses checkpoint mode instead.
//!
//! CHECKPOINT PROTOCOL v1 (`MEMRA_BG_YIELD_MODE=checkpoint`) — the seam for training-class
//! jobs. The "checkpoint callback" is process-level: SIGUSR1 to the job's group means
//! "checkpoint NOW and exit"; the job writes its state to disk and exits **75**
//! (EX_TEMPFAIL, the sysexits convention for "transient — retry me"). The runner relaunches
//! the SAME command in the next valley and the job resumes from its own checkpoint file.
//! Exit 0 = complete (never relaunched); any other exit = failed (never relaunched, loud).
//! A job that outlives `MEMRA_BG_CKPT_GRACE_MS` after SIGUSR1 is SIGKILLed — the yield
//! bound holds even against a wedged job, and the on-disk checkpoint is whatever it last
//! wrote (at-least-once semantics: a training step may repeat, never be lost). Toy proof:
//! `tools/bg-ckpt-counter.py`. An in-process trainer API can replace this seam later
//! without changing the valley/scheduler half.
//!
//! GPU MEMORY DISCIPLINE: the job gets a VRAM budget (`MEMRA_BG_VRAM_MB`, default 0 =
//! CPU-only) from the serve headroom and is REFUSED at launch if it doesn't fit:
//! launch requires `min free across visible GPUs >= budget + serve headroom`, where the
//! headroom term composes with the existing `MEMRA_MOE_RESIDENT_HEADROOM_GB` pattern
//! (default 2.0 GB — the same class of reserve the resident planner keeps beside weights).
//! Fail-closed: a budget > 0 with no readable `nvidia-smi` is a refusal, not a shrug. The
//! runner enforces fit at launch; staying inside the budget at runtime is the job's
//! contract (documented in FLAGS.md) — v1 has no cgroup-style VRAM enforcement.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

use crate::health::SharedHealth;
use crate::worker::PENDING_ADMITS;

// ---------------------------------------------------------------------------
// Valley detection
// ---------------------------------------------------------------------------

/// `MEMRA_VALLEY_S` (default 2.0): how long the worker must be COMPLETELY idle (no active
/// sessions, no queued admissions, no pending HTTP handoffs) before the box is "in a
/// valley". Read once — the threshold must not move under a running process.
pub fn valley_threshold_s() -> f64 {
    static T: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("MEMRA_VALLEY_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0)
    })
}

/// The valley signal — a read-only view over worker truth (health phase + beat age +
/// the pending-admit gauge). Cheap enough to evaluate per /metrics call AND per runner
/// poll: two atomic loads and a subtraction.
#[derive(Clone)]
pub struct ValleySignal {
    health: SharedHealth,
}

impl ValleySignal {
    pub fn new(health: SharedHealth) -> Self {
        Self { health }
    }

    /// Seconds the worker has been completely idle; 0.0 the instant there is ANY work
    /// (active/queued sessions => phase != IDLE; a busy or backlogged serve route => no
    /// `idle_for_ms`; submitted-not-yet-popped requests => PENDING_ADMITS > 0; loading/dead
    /// phases are not idleness either).
    pub fn idle_seconds(&self) -> f64 {
        let s = self.health.snapshot();
        match s.idle_for_ms {
            Some(ms) if PENDING_ADMITS.load(Ordering::Acquire) == 0 => ms as f64 / 1000.0,
            _ => 0.0,
        }
    }

    /// The resume-side signal: a full threshold of quiet.
    pub fn in_valley(&self) -> bool {
        self.idle_seconds() >= valley_threshold_s()
    }

    /// The yield-side signal: ANY activity, no debounce. Deliberately not `!in_valley()` —
    /// the asymmetry (instant yield, debounced resume) is the whole point.
    pub fn busy_now(&self) -> bool {
        self.idle_seconds() == 0.0
    }
}

// ---------------------------------------------------------------------------
// Background job runner
// ---------------------------------------------------------------------------

/// Job lifecycle states, published at /metrics ("bg" block). u8-backed so the /metrics
/// read is a bare atomic load.
pub const BG_WAITING: u8 = 0; // configured, waiting for the first/next valley to launch
pub const BG_RUNNING: u8 = 1; // child running in a valley
pub const BG_YIELDED: u8 = 2; // SIGSTOPped for serve traffic (stop mode)
pub const BG_PREEMPTED: u8 = 3; // checkpointed out (ckpt mode), relaunches next valley
pub const BG_DONE: u8 = 4; // exit 0 — complete, never relaunched
pub const BG_FAILED: u8 = 5; // non-0/75 exit — never relaunched, loud
pub const BG_REFUSED: u8 = 6; // VRAM budget did not fit at launch (retries next valley)

pub fn bg_state_str(s: u8) -> &'static str {
    match s {
        BG_WAITING => "waiting_valley",
        BG_RUNNING => "running",
        BG_YIELDED => "yielded",
        BG_PREEMPTED => "preempted",
        BG_DONE => "done",
        BG_FAILED => "failed",
        BG_REFUSED => "refused_vram",
        _ => "unknown",
    }
}

/// Shared observable state — the /metrics "bg" block reads ONLY this (atomics, never the
/// child handle). Counters are cumulative for the process life.
#[derive(Default)]
pub struct BgJobState {
    pub state: AtomicU8,
    pub launches: AtomicU64,
    pub yields: AtomicU64,
    pub resumes: AtomicU64,
    /// checkpoint-mode preemptions (SIGUSR1 sent); `ckpt_kills` counts the subset that
    /// blew the grace window and were SIGKILLed (dirty preempt — at-least-once resume).
    pub preempts: AtomicU64,
    pub ckpt_kills: AtomicU64,
    /// wall micros from busy-edge observation to the yield signal having been SENT (stop
    /// mode: SIGSTOP returned; ckpt mode: SIGUSR1 returned). This measures signal
    /// submission, not the later observed stop acknowledgement. Last observed value.
    pub last_yield_signal_us: AtomicU64,
    pub job_pid: AtomicU32, // 0 = no live child
    pub vram_budget_mb: AtomicU64,
}

impl BgJobState {
    pub fn to_json(&self, yield_mode: &str) -> serde_json::Value {
        let pid = self.job_pid.load(Ordering::Acquire);
        serde_json::json!({
            "state": bg_state_str(self.state.load(Ordering::Acquire)),
            "yield_mode": yield_mode,
            "launches": self.launches.load(Ordering::Relaxed),
            "yields": self.yields.load(Ordering::Relaxed),
            "resumes": self.resumes.load(Ordering::Relaxed),
            "preempts": self.preempts.load(Ordering::Relaxed),
            "ckpt_kills": self.ckpt_kills.load(Ordering::Relaxed),
            "last_yield_signal_us": self.last_yield_signal_us.load(Ordering::Relaxed),
            "job_pid": if pid == 0 { serde_json::Value::Null } else { pid.into() },
            "vram_budget_mb": self.vram_budget_mb.load(Ordering::Relaxed),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YieldMode {
    /// SIGSTOP/SIGCONT the job's process group. Memory (VRAM included) stays resident
    /// while stopped — budget accordingly.
    Stop,
    /// SIGUSR1 = "checkpoint and exit 75"; relaunch next valley. For jobs whose stopped
    /// working set must not squat on VRAM.
    Checkpoint,
}

impl YieldMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            YieldMode::Stop => "stop",
            YieldMode::Checkpoint => "checkpoint",
        }
    }
}

pub struct BgConfig {
    /// The job command, run via `sh -c` in its own process group.
    pub cmd: String,
    pub poll_ms: u64,
    pub yield_mode: YieldMode,
    pub ckpt_grace_ms: u64,
    /// VRAM the job may use (0 = CPU-only, no GPU probe at launch).
    pub vram_budget_mb: u64,
    /// Serve headroom (MB) that must REMAIN free above the job's budget at launch —
    /// the MEMRA_MOE_RESIDENT_HEADROOM_GB composition.
    pub headroom_mb: u64,
}

impl BgConfig {
    /// None when MEMRA_BG_JOB is unset — the runner (and its /metrics block) simply does
    /// not exist; zero cost on every deployment that doesn't ask for it.
    pub fn from_env() -> Option<Self> {
        let cmd = std::env::var("MEMRA_BG_JOB")
            .ok()
            .filter(|c| !c.trim().is_empty())?;
        let u = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let mode = match std::env::var("MEMRA_BG_YIELD_MODE").as_deref() {
            Ok("checkpoint") => YieldMode::Checkpoint,
            Ok("stop") | Err(_) => YieldMode::Stop,
            Ok(other) => {
                eprintln!(
                    "[darklane] WARN: bad MEMRA_BG_YIELD_MODE {other:?} \
                           (stop|checkpoint); using stop"
                );
                YieldMode::Stop
            }
        };
        let headroom_gb: f64 = std::env::var("MEMRA_MOE_RESIDENT_HEADROOM_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.0);
        Some(Self {
            cmd,
            poll_ms: u("MEMRA_BG_POLL_MS", 25).max(1),
            yield_mode: mode,
            ckpt_grace_ms: u("MEMRA_BG_CKPT_GRACE_MS", 5000),
            vram_budget_mb: u("MEMRA_BG_VRAM_MB", 0),
            headroom_mb: (headroom_gb * 1024.0) as u64,
        })
    }
}

/// Min free VRAM (MB) across visible GPUs, via nvidia-smi. Min, not sum: v1 does not know
/// which card the job lands on, and on a PP-2 pair BOTH cards carry serve shards — the
/// budget must fit the tightest one. None (no tool / parse failure) = fail closed.
fn nvidia_min_free_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse::<u64>().ok())
        .collect::<Option<Vec<u64>>>()?
        .into_iter()
        .min()
}

/// Handle main() keeps: flip `stop`, then `join()` — the loop polls the flag every
/// `poll_ms` and exits after cleaning up the child (see shutdown() below).
pub struct BgHandle {
    pub state: Arc<BgJobState>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl BgHandle {
    /// Called after axum::serve returns (drain complete). MUST run before process exit in
    /// the graceful path: a SIGSTOPped orphan would stay frozen forever. The ungraceful
    /// path (SIGKILL on the server) is covered by PR_SET_PDEATHSIG=SIGKILL on the child.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Signal a whole process group. pgid == child pid (process_group(0) at spawn).
fn signal_group(pgid: u32, sig: i32) -> std::io::Result<()> {
    if unsafe { libc::kill(-(pgid as i32), sig) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn kill_group(pgid: u32, sig: i32) {
    let _ = signal_group(pgid, sig);
}

/// A reaped leader does not imply that its owned process group is empty. Keep the
/// group's identity until remaining members have exited or received SIGKILL.
fn terminate_owned_group(c: &mut std::process::Child, deadline: std::time::Instant) {
    let pgid = c.id();
    if std::time::Instant::now() < deadline {
        kill_group(pgid, libc::SIGCONT);
        kill_group(pgid, libc::SIGTERM);
    }
    loop {
        let child_alive = matches!(c.try_wait(), Ok(None));
        let group_gone = matches!(
            signal_group(pgid, 0),
            Err(err) if err.raw_os_error() == Some(libc::ESRCH)
        );
        if group_gone && !child_alive {
            return;
        }
        if std::time::Instant::now() >= deadline {
            kill_group(pgid, libc::SIGKILL);
            if child_alive {
                let _ = c.kill();
            }
            let _ = c.wait();
            return;
        }
        std::thread::sleep(
            std::time::Duration::from_millis(10)
                .min(deadline.saturating_duration_since(std::time::Instant::now())),
        );
    }
}

fn terminate_owned_group_with_grace(c: &mut std::process::Child) {
    terminate_owned_group(
        c,
        std::time::Instant::now() + std::time::Duration::from_secs(2),
    );
}

/// Observe a stop without reaping an exit that Child::try_wait must classify.
/// Sending SIGSTOP successfully does not prove the child remains alive or stopped.
///
/// The stop has taken effect once it has on every member of the job's process group: the
/// leader, and every process the job started that stayed in the group. `sh -c` is the leader
/// and dash does not exec even a single command, so the job's program is usually one of the
/// other members. waitid reports the leader's stop only once EVERY thread of the leader has
/// stopped, and a thread waiting in vfork cannot stop until its child execs or exits. The same
/// group SIGSTOP stops that child before exec, so the waiting thread stays in D until SIGCONT and
/// waitid never reports the stop. dash spawns every external command with vfork; glibc's
/// posix_spawn (and through it CPython's subprocess) clones with CLONE_VFORK too
/// (`research/darklane-vfork-stop-20261109/`). So the leader's stop is read from /proc when
/// waitid has nothing to report, every other member's stop is read from /proc even when waitid
/// reports the leader stopped, and each reading directs the stop at each thread of a process on
/// which no thread has taken it: see [`process_stop`]. A member with no live thread (an exiting
/// or unreaped process) cannot run job code and counts as stopped; the leader's exit is left to
/// try_wait. Called only after a group SIGSTOP was sent successfully.
fn child_stop_acknowledged(pid: u32) -> std::io::Result<bool> {
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WSTOPPED | libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let leader = if info.si_code == libc::CLD_STOPPED {
        true
    } else if unsafe { info.si_pid() } != 0 {
        return Ok(false); // An exit: try_wait reaps and classifies it on the next tick.
    } else {
        process_stop(pid, pid)? == ProcessStop::TookEffect
    };
    // Every member is read, so each one whose stop no thread has taken gets it this tick.
    let mut members = true;
    for member in group_members(pid)? {
        members &= matches!(
            process_stop(member, pid)?,
            ProcessStop::TookEffect | ProcessStop::Gone
        );
    }
    Ok(leader && members)
}

/// A /proc read that failed because the process or thread is gone.
fn proc_gone(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::NotFound || err.raw_os_error() == Some(libc::ESRCH)
}

/// The process group of a `/proc/<pid>/stat` line: the third field after the LAST ')' (comm
/// may itself contain spaces and parentheses).
fn proc_stat_pgrp(stat: &str) -> Option<u32> {
    stat[stat.rfind(')')? + 1..]
        .split_whitespace()
        .nth(2)?
        .parse()
        .ok()
}

/// The process group of `pid`, or None once it is gone.
fn process_pgrp(pid: u32) -> std::io::Result<Option<u32>> {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => proc_stat_pgrp(&stat)
            .map(Some)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "unparsable stat")),
        Err(err) if proc_gone(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Every process in process group `pgid` except its leader, from a scan of /proc. A process
/// that left the group is not found, and the group's signals do not reach it either.
fn group_members(pgid: u32) -> std::io::Result<Vec<u32>> {
    let mut members = Vec::new();
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        let Some(pid) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
            continue; // Not a process directory.
        };
        if pid != pgid && process_pgrp(pid)? == Some(pgid) {
            members.push(pid);
        }
    }
    Ok(members)
}

/// What one `/proc/<pid>/task/<tid>/status` says about a submitted stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThreadStop {
    /// The `State:` letter.
    state: char,
    /// SIGSTOP in the thread's own pending set (`SigPnd`), where a thread-directed stop lands.
    own: bool,
    /// SIGSTOP in the process's shared pending set (`ShdPnd`), where the group stop lands.
    shared: bool,
}

/// Parse a `/proc/.../status` text into a [`ThreadStop`]; None when a field is missing.
fn thread_stop(status: &str) -> Option<ThreadStop> {
    let field = |name: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .map(str::trim)
    };
    let sigstop = |name: &str| {
        let mask = u64::from_str_radix(field(name)?, 16).ok()?;
        Some(mask & (1u64 << (libc::SIGSTOP - 1)) != 0)
    };
    Some(ThreadStop {
        state: field("State:")?.chars().next()?,
        own: sigstop("SigPnd:")?,
        shared: sigstop("ShdPnd:")?,
    })
}

/// What one process's threads say about a submitted group stop.
#[derive(Debug, PartialEq, Eq)]
enum ProcessStop {
    /// Every live thread is stopped, or blocked in the kernel and bound to stop before it next
    /// returns to user space.
    TookEffect,
    /// Not yet, and a later tick reads again: a thread can still run while a group stop is
    /// under way, the process has one live thread, or every live thread already carries the stop
    /// in its own pending set.
    NotYet,
    /// No thread has taken the stop, more than one is live, and some live thread does not carry
    /// it in its own pending set. The stop must be directed at each live thread.
    Unclaimed,
    /// No live thread: the process is exiting, or is gone.
    Gone,
}

fn classify_process_stop(threads: &[ThreadStop]) -> ProcessStop {
    let live: Vec<ThreadStop> = threads
        .iter()
        .copied()
        .filter(|t| !matches!(t.state, 'Z' | 'X'))
        .collect();
    if live.is_empty() {
        return ProcessStop::Gone;
    }
    let stopped = live.iter().any(|t| matches!(t.state, 'T' | 't'));
    let single = live.len() == 1;
    let bound_to_stop = |t: &ThreadStop| match t.state {
        'T' | 't' => true,
        // D, and I (TASK_IDLE = uninterruptible without load accounting).
        'D' | 'I' => stopped || t.own || (single && t.shared),
        _ => false,
    };
    if live.iter().all(bound_to_stop) {
        ProcessStop::TookEffect
    } else if !stopped && !single && live.iter().any(|t| !t.own) {
        ProcessStop::Unclaimed
    } else {
        ProcessStop::NotYet
    }
}

/// A process's threads (tid and reading), or None once the process is gone.
fn process_threads(pid: u32) -> std::io::Result<Option<Vec<(libc::pid_t, ThreadStop)>>> {
    let tasks = match std::fs::read_dir(format!("/proc/{pid}/task")) {
        Ok(tasks) => tasks,
        Err(err) if proc_gone(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    let unparsable =
        || std::io::Error::new(std::io::ErrorKind::InvalidData, "unparsable task status");
    let mut threads = Vec::new();
    for task in tasks {
        let task = match task {
            Ok(task) => task,
            Err(err) if proc_gone(&err) => return Ok(None),
            Err(err) => return Err(err),
        };
        let tid = task
            .file_name()
            .to_str()
            .and_then(|tid| tid.parse().ok())
            .ok_or_else(unparsable)?;
        match std::fs::read_to_string(task.path().join("status")) {
            Ok(status) => threads.push((tid, thread_stop(&status).ok_or_else(unparsable)?)),
            Err(err) if proc_gone(&err) => {} // The thread exited after the listing.
            Err(err) => return Err(err),
        }
    }
    Ok(Some(threads))
}

/// The stop has taken effect on a process of the job's group when every live thread is
/// stopped, or blocked in the kernel (D, or I) and bound to stop before it next returns to user
/// space, so that none of its threads runs job code again before SIGCONT. A blocked thread is
/// bound to stop when:
///   * some thread is stopped: the group stop is under way and marked every thread for it;
///   * SIGSTOP is in the thread's own pending set: the kernel flagged that thread to take it;
///   * the thread is the process's only live thread and SIGSTOP is in the shared pending set.
///
/// The last condition holds for one live thread only. The kernel queues a group SIGSTOP once
/// per process and flags one thread to take it, the main thread first, and tells no other. With
/// more threads the shared set does not say which thread was flagged, and an unflagged thread
/// whose wait ends returns to user space and runs. When the flagged thread is the one waiting in
/// vfork, no thread takes the stop at all: the others run on and never start the group stop.
/// So when no thread is stopped and more than one is live, the stop is directed at each live
/// thread (tgkill), which flags each one, and read again. A flagged sleeping or running thread
/// starts the group stop, and a flagged blocked one carries the stop in its own pending set.
/// tgkill checks the thread group, so a stale tid cannot reach another process, and SIGCONT
/// flushes thread-directed stops with the group's. A member other than the leader is checked to
/// be still in group `pgid` right before: pids are allocated cyclically, so one that exited since
/// the scan is not handed to another process within a tick, and the check keeps the stop inside
/// the group regardless. Called only after a group SIGSTOP was sent successfully.
fn process_stop(pid: u32, pgid: u32) -> std::io::Result<ProcessStop> {
    let reading = |threads: &[(libc::pid_t, ThreadStop)]| {
        classify_process_stop(&threads.iter().map(|&(_, t)| t).collect::<Vec<_>>())
    };
    let Some(threads) = process_threads(pid)? else {
        return Ok(ProcessStop::Gone);
    };
    let first = reading(&threads);
    if first != ProcessStop::Unclaimed {
        return Ok(first);
    }
    if pid != pgid && process_pgrp(pid)? != Some(pgid) {
        return Ok(ProcessStop::Gone); // It exited, or left the group, since the scan.
    }
    for &(tid, t) in &threads {
        if matches!(t.state, 'Z' | 'X') {
            continue;
        }
        let rc = unsafe { libc::syscall(libc::SYS_tgkill, pid as libc::pid_t, tid, libc::SIGSTOP) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            // ESRCH: the thread exited after the listing.
            if err.raw_os_error() != Some(libc::ESRCH) {
                return Err(err);
            }
        }
    }
    Ok(match process_threads(pid)? {
        None => ProcessStop::Gone,
        Some(threads) => match reading(&threads) {
            // Every live thread now carries the stop, unless it already took it and stopped.
            ProcessStop::Unclaimed => ProcessStop::NotYet,
            again => again,
        },
    })
}

/// Spawn the supervisor thread. `in_valley` / `busy_now` are injected (prod: ValleySignal;
/// tests: atomics) so the whole state machine is pinned GPU-free. `vram_probe` likewise
/// (prod: nvidia-smi; tests: a constant).
pub fn spawn_runner(
    cfg: BgConfig,
    in_valley: Arc<dyn Fn() -> bool + Send + Sync>,
    busy_now: Arc<dyn Fn() -> bool + Send + Sync>,
    vram_probe: Arc<dyn Fn() -> Option<u64> + Send + Sync>,
) -> BgHandle {
    let state = Arc::new(BgJobState::default());
    state
        .vram_budget_mb
        .store(cfg.vram_budget_mb, Ordering::Relaxed);
    let stop = Arc::new(AtomicBool::new(false));
    let (st, sp) = (state.clone(), stop.clone());
    eprintln!(
        "[darklane] bg runner armed: cmd={:?} mode={} poll={}ms vram_budget={}MB \
               (+{}MB serve headroom must stay free)",
        cfg.cmd,
        cfg.yield_mode.as_str(),
        cfg.poll_ms,
        cfg.vram_budget_mb,
        cfg.headroom_mb
    );
    let thread = std::thread::Builder::new()
        .name("memra-bg-runner".into())
        .spawn(move || runner_loop(cfg, in_valley, busy_now, vram_probe, st, sp))
        .expect("spawn bg runner thread");
    BgHandle {
        state,
        stop,
        thread: Some(thread),
    }
}

/// Prod wiring: valley signal from worker health.
pub fn spawn_from_env(health: SharedHealth) -> Option<BgHandle> {
    let cfg = BgConfig::from_env()?;
    let v = ValleySignal::new(health);
    let v2 = v.clone();
    Some(spawn_runner(
        cfg,
        Arc::new(move || v.in_valley()),
        Arc::new(move || v2.busy_now()),
        Arc::new(nvidia_min_free_mb),
    ))
}

fn runner_loop(
    cfg: BgConfig,
    in_valley: Arc<dyn Fn() -> bool + Send + Sync>,
    busy_now: Arc<dyn Fn() -> bool + Send + Sync>,
    vram_probe: Arc<dyn Fn() -> Option<u64> + Send + Sync>,
    st: Arc<BgJobState>,
    stop: Arc<AtomicBool>,
) {
    let mut child: Option<std::process::Child> = None;
    let mut stop_requested = false;
    let poll = std::time::Duration::from_millis(cfg.poll_ms);
    // one loud line per refusal EPISODE, not per poll (a tight card would log 40/s).
    let mut refusal_logged = false;

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let s = st.state.load(Ordering::Acquire);
        match s {
            BG_WAITING | BG_PREEMPTED | BG_REFUSED => {
                if in_valley() {
                    // GPU memory discipline: fits-or-refused, re-checked every launch
                    // (headroom moves as sessions park/retire — a refusal is not forever).
                    if cfg.vram_budget_mb > 0 {
                        let free = vram_probe();
                        let need = cfg.vram_budget_mb + cfg.headroom_mb;
                        let fits = free.is_some_and(|f| f >= need);
                        if !fits {
                            if !refusal_logged {
                                eprintln!(
                                    "[darklane] REFUSED: vram budget {}MB + serve \
                                           headroom {}MB > min free {:?}MB (fail-closed \
                                           when unreadable); will retry next valley",
                                    cfg.vram_budget_mb, cfg.headroom_mb, free
                                );
                                refusal_logged = true;
                            }
                            st.state.store(BG_REFUSED, Ordering::Release);
                            std::thread::sleep(poll);
                            continue;
                        }
                    }
                    refusal_logged = false;
                    match launch(&cfg.cmd, cfg.vram_budget_mb) {
                        Ok(c) => {
                            st.job_pid.store(c.id(), Ordering::Release);
                            st.launches.fetch_add(1, Ordering::Relaxed);
                            st.state.store(BG_RUNNING, Ordering::Release);
                            eprintln!(
                                "[darklane] job launched (pid {}, {})",
                                c.id(),
                                if s == BG_PREEMPTED {
                                    "resume from checkpoint"
                                } else {
                                    "fresh"
                                }
                            );
                            child = Some(c);
                        }
                        Err(err) => {
                            eprintln!("[darklane] job spawn FAILED: {err}");
                            st.state.store(BG_FAILED, Ordering::Release);
                        }
                    }
                }
            }
            BG_RUNNING => {
                let c = child.as_mut().expect("running state implies child");
                // exit first: a finished job must not be signaled.
                match c.try_wait() {
                    Ok(Some(status)) => {
                        terminate_owned_group_with_grace(c);
                        handle_exit(status, &st, /*preempting=*/ false);
                        st.job_pid.store(0, Ordering::Release);
                        child = None;
                        stop_requested = false;
                    }
                    Ok(None) => {
                        if stop_requested {
                            match child_stop_acknowledged(c.id()) {
                                Ok(true) => {
                                    st.yields.fetch_add(1, Ordering::Relaxed);
                                    st.state.store(BG_YIELDED, Ordering::Release);
                                    stop_requested = false;
                                }
                                Ok(false) => {} // Exit is reaped by try_wait on the next tick.
                                Err(err) => {
                                    eprintln!("[darklane] stop acknowledgement failed: {err}");
                                    st.state.store(BG_FAILED, Ordering::Release);
                                }
                            }
                        } else if busy_now() {
                            let t0 = std::time::Instant::now();
                            let pgid = c.id();
                            match cfg.yield_mode {
                                YieldMode::Stop => {
                                    match signal_group(pgid, libc::SIGSTOP) {
                                        Ok(()) => {
                                            st.last_yield_signal_us.store(
                                                t0.elapsed().as_micros() as u64,
                                                Ordering::Relaxed,
                                            );
                                            stop_requested = true;
                                        }
                                        Err(err) => {
                                            // An exit racing the signal retains its real status.
                                            match c.try_wait() {
                                                Ok(Some(status)) => {
                                                    terminate_owned_group_with_grace(c);
                                                    handle_exit(status, &st, false);
                                                    st.job_pid.store(0, Ordering::Release);
                                                    child = None;
                                                }
                                                _ => {
                                                    eprintln!(
                                                        "[darklane] stop signal failed: {err}"
                                                    );
                                                    st.state.store(BG_FAILED, Ordering::Release);
                                                }
                                            }
                                        }
                                    }
                                }
                                YieldMode::Checkpoint => {
                                    kill_group(pgid, libc::SIGUSR1);
                                    st.last_yield_signal_us
                                        .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
                                    st.preempts.fetch_add(1, Ordering::Relaxed);
                                    preempt_wait(c, &st, cfg.ckpt_grace_ms);
                                    st.job_pid.store(0, Ordering::Release);
                                    child = None;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("[darklane] try_wait failed: {err}; treating job as failed");
                        terminate_owned_group_with_grace(c);
                        st.state.store(BG_FAILED, Ordering::Release);
                        st.job_pid.store(0, Ordering::Release);
                        child = None;
                    }
                }
            }
            BG_YIELDED => {
                // A stopped child can still be killed. Reap it before attempting resume.
                let c = child.as_mut().expect("yielded state implies child");
                match c.try_wait() {
                    Ok(Some(status)) => {
                        terminate_owned_group_with_grace(c);
                        handle_exit(status, &st, false);
                        st.job_pid.store(0, Ordering::Release);
                        child = None;
                    }
                    Ok(None) if in_valley() => match signal_group(c.id(), libc::SIGCONT) {
                        Ok(()) => {
                            st.resumes.fetch_add(1, Ordering::Relaxed);
                            st.state.store(BG_RUNNING, Ordering::Release);
                        }
                        Err(err) => {
                            eprintln!("[darklane] resume signal failed: {err}");
                            st.state.store(BG_FAILED, Ordering::Release);
                        }
                    },
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("[darklane] yielded child wait failed: {err}");
                        st.state.store(BG_FAILED, Ordering::Release);
                    }
                }
            }
            _ => break, // DONE / FAILED — terminal; the thread's work is over.
        }
        std::thread::sleep(poll);
    }

    // Shutdown cleanup: never leave a stopped orphan. CONT first (a stopped process
    // cannot act on TERM), then TERM (checkpoint-class jobs get their handler), brief
    // grace, then KILL the group.
    if let Some(mut c) = child.take() {
        terminate_owned_group_with_grace(&mut c);
        eprintln!("[darklane] owned process group terminated at shutdown");
        st.job_pid.store(0, Ordering::Release);
    }
}

/// Spawn the job: own process group (signals hit the whole tree, never the server) +
/// PDEATHSIG=SIGKILL (a SIGKILLed server cannot run cleanup; the kernel reaps the job —
/// KILL not TERM because a job its runner had STOPPED would never act on TERM).
/// The budget is exported so the job can size itself (`MEMRA_BG_VRAM_MB`).
fn launch(cmd: &str, vram_budget_mb: u64) -> std::io::Result<std::process::Child> {
    use std::os::unix::process::CommandExt;
    let mut c = std::process::Command::new("sh");
    c.arg("-c")
        .arg(cmd)
        .env("MEMRA_BG_VRAM_MB", vram_budget_mb.to_string())
        .process_group(0);
    unsafe {
        c.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
    c.spawn()
}

/// Classify a job exit. Exit 75 (EX_TEMPFAIL) = "checkpointed, resume me" — the job may
/// use it voluntarily too (self-preempting at a step boundary), not only under SIGUSR1.
fn handle_exit(status: std::process::ExitStatus, st: &BgJobState, preempting: bool) {
    match status.code() {
        Some(0) => {
            eprintln!("[darklane] job COMPLETE (exit 0)");
            st.state.store(BG_DONE, Ordering::Release);
        }
        Some(75) => {
            eprintln!("[darklane] job checkpointed (exit 75) — relaunches next valley");
            st.state.store(BG_PREEMPTED, Ordering::Release);
        }
        code => {
            if preempting {
                // died under SIGUSR1 without the protocol exit — treat the on-disk
                // checkpoint as authoritative and relaunch (at-least-once).
                eprintln!(
                    "[darklane] job exited {code:?} during preemption — \
                           relaunching from last checkpoint next valley"
                );
                st.state.store(BG_PREEMPTED, Ordering::Release);
            } else {
                eprintln!("[darklane] job FAILED ({status}) — not relaunching");
                st.state.store(BG_FAILED, Ordering::Release);
            }
        }
    }
}

/// Checkpoint-mode preemption wait: the job has `grace_ms` after SIGUSR1 to write its
/// checkpoint and exit; past that it is SIGKILLed (the yield bound holds even against a
/// wedged job — VRAM is freed by process death either way).
fn preempt_wait(c: &mut std::process::Child, st: &BgJobState, grace_ms: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(grace_ms);
    loop {
        match c.try_wait() {
            Ok(Some(status)) => {
                terminate_owned_group(c, deadline);
                handle_exit(status, st, /*preempting=*/ true);
                return;
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            _ => {
                kill_group(c.id(), libc::SIGKILL);
                let _ = c.wait();
                st.ckpt_kills.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[darklane] job blew the {grace_ms}ms checkpoint grace — \
                           SIGKILLed (dirty preempt; resumes from last on-disk state)"
                );
                st.state.store(BG_PREEMPTED, Ordering::Release);
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the whole state machine, GPU-free (injected valley/busy/vram signals)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::PHASE_IDLE;
    use crate::health::WorkerHealth;

    /// /proc/<pid>/stat field 3 — 'T' is stopped, 'R'/'S' running/sleeping, gone = None.
    fn proc_state(pid: u32) -> Option<char> {
        proc_stat_fields(std::path::Path::new(&format!("/proc/{pid}/stat"))).map(|f| f.0)
    }

    /// `/proc/.../stat` fields after comm: (state, ppid, flags).
    fn proc_stat_fields(path: &std::path::Path) -> Option<(char, u32, i64)> {
        let s = std::fs::read_to_string(path).ok()?;
        let rest: Vec<&str> = s[s.rfind(')')? + 1..].split_whitespace().collect();
        Some((
            rest.first()?.chars().next()?,
            rest.get(1)?.parse().ok()?,
            rest.get(6)?.parse().ok()?,
        ))
    }

    /// Every thread's tid and state, from `/proc/<pid>/task/*/stat`.
    fn leader_tids(pid: u32) -> Vec<(u32, char)> {
        let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
            return Vec::new();
        };
        tasks
            .filter_map(|t| {
                let t = t.ok()?;
                let tid = t.file_name().to_str()?.parse().ok()?;
                Some((tid, proc_stat_fields(&t.path().join("stat"))?.0))
            })
            .collect()
    }

    /// Every thread's state, from `/proc/<pid>/task/*/stat`.
    fn leader_thread_states(pid: u32) -> Vec<char> {
        leader_tids(pid).into_iter().map(|t| t.1).collect()
    }

    /// Children of `pid` that were forked and have not exec'd (PF_FORKNOEXEC), with state.
    fn unexeced_children(pid: u32) -> Vec<(u32, char)> {
        let Ok(procs) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        procs
            .filter_map(|p| {
                let child: u32 = p.ok()?.file_name().to_str()?.parse().ok()?;
                let (state, ppid, flags) =
                    proc_stat_fields(std::path::Path::new(&format!("/proc/{child}/stat")))?;
                (ppid == pid && flags & libc::PF_FORKNOEXEC as i64 != 0).then_some((child, state))
            })
            .collect()
    }

    /// The shapes a group SIGSTOP leaves the leader in: every thread stopped, or a thread
    /// blocked in vfork (D) on a child that the same stop caught before exec.
    fn leader_stopped_or_vfork_parked(pid: u32) -> bool {
        let threads = leader_thread_states(pid);
        if threads.is_empty() {
            return false;
        }
        threads.iter().all(|s| *s == 'T')
            || (threads.iter().all(|s| matches!(s, 'T' | 'D'))
                && unexeced_children(pid).iter().any(|&(_, s)| s == 'T'))
    }

    /// waitid's own view: has the whole leader (every thread) reported stopped?
    fn leader_reports_stopped(pid: u32) -> bool {
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WSTOPPED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        assert_eq!(rc, 0, "waitid: {}", std::io::Error::last_os_error());
        info.si_code == libc::CLD_STOPPED
    }

    /// WP-A day 57 (`research/spill-a-20260919/DAY57.md`, OWED item 24): wait for an
    /// acknowledgement under a 30 s hang guard. The stop-mode cycle's claim is the wiring (the
    /// runner acts on each signal and the job reaches the state it names); a wall bound on how soon
    /// the scheduler runs the runner thread failed under starvation (1 of 100 beside sixteen
    /// burners at 3 s). A runner that never acts never acknowledges and fails at the guard. The
    /// latency each wait read is printed.
    fn wait_acknowledged<F: Fn() -> bool>(what: &str, f: F) {
        let t0 = std::time::Instant::now();
        let guard = std::time::Duration::from_secs(30);
        while t0.elapsed() < guard {
            if f() {
                println!("{what}: acknowledged after {}ms", t0.elapsed().as_millis());
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("no acknowledgement within the 30 s guard: {what}");
    }

    fn wait_for<F: Fn() -> bool>(what: &str, ms: u64, f: F) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ms);
        while std::time::Instant::now() < deadline {
            if f() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("timed out ({ms}ms) waiting for: {what}");
    }

    struct Sig {
        valley: Arc<AtomicBool>,
        busy: Arc<AtomicBool>,
    }
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn sigs() -> (
        Sig,
        Arc<dyn Fn() -> bool + Send + Sync>,
        Arc<dyn Fn() -> bool + Send + Sync>,
    ) {
        let valley = Arc::new(AtomicBool::new(false));
        let busy = Arc::new(AtomicBool::new(false));
        let (v, b) = (valley.clone(), busy.clone());
        (
            Sig { valley, busy },
            Arc::new(move || v.load(Ordering::Acquire)),
            Arc::new(move || b.load(Ordering::Acquire)),
        )
    }

    fn cfg(cmd: &str, mode: YieldMode) -> BgConfig {
        BgConfig {
            cmd: cmd.into(),
            poll_ms: 5,
            yield_mode: mode,
            ckpt_grace_ms: 2000,
            vram_budget_mb: 0,
            headroom_mb: 2048,
        }
    }

    // A failed acknowledgement must still shut down the test's owned process group.
    struct RunnerGuard(Option<BgHandle>);

    impl RunnerGuard {
        fn shutdown(mut self) {
            self.0.take().unwrap().shutdown();
        }
    }

    impl Drop for RunnerGuard {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                if std::thread::panicking() {
                    let st = &h.state;
                    let pid = st.job_pid.load(Ordering::Acquire);
                    let (threads, children) = if pid == 0 {
                        (Vec::new(), Vec::new())
                    } else {
                        (leader_thread_states(pid), unexeced_children(pid))
                    };
                    eprintln!(
                        "owned runner assertion failed: state={} pid={pid} child_state={:?} \
                         threads={threads:?} unexeced_children={children:?} yields={} resumes={}",
                        bg_state_str(st.state.load(Ordering::Acquire)),
                        (pid != 0).then(|| proc_state(pid)).flatten(),
                        st.yields.load(Ordering::Relaxed),
                        st.resumes.load(Ordering::Relaxed)
                    );
                }
                h.shutdown();
            }
        }
    }

    struct TestDir(std::path::PathBuf);

    fn make_fifo(path: &std::path::Path) {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { libc::mkfifo(c.as_ptr(), 0o600) },
            0,
            "mkfifo: {}",
            std::io::Error::last_os_error()
        );
    }

    /// Open a FIFO's write end without blocking: true once a reader waits in its open.
    fn open_fifo_writer(path: &std::path::Path) -> bool {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .is_ok()
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn owned_pid_record(path: &std::path::Path) -> Option<u32> {
        let record = std::fs::read_to_string(path).ok()?;
        record.ends_with('\n').then_some(())?;
        record.trim().parse().ok()
    }

    // Keep cleanup bound to the exact owned descendant even after it is orphaned.
    struct OwnedDescendant(std::os::fd::OwnedFd);

    impl OwnedDescendant {
        fn open(pid: u32) -> Self {
            use std::os::fd::FromRawFd;
            let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
            assert!(fd >= 0, "pidfd_open: {}", std::io::Error::last_os_error());
            Self(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd as i32) })
        }

        fn alive(&self) -> bool {
            use std::os::fd::AsRawFd;
            let mut poll = libc::pollfd {
                fd: self.0.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&mut poll, 1, 0) };
            assert!(rc >= 0, "pidfd poll: {}", std::io::Error::last_os_error());
            rc == 0
        }
    }

    impl Drop for OwnedDescendant {
        fn drop(&mut self) {
            use std::os::fd::AsRawFd;
            for signal in [libc::SIGCONT, libc::SIGKILL] {
                unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        self.0.as_raw_fd(),
                        signal,
                        std::ptr::null::<libc::siginfo_t>(),
                        0,
                    );
                }
            }
        }
    }

    struct OwnedChild(std::process::Child);

    impl Drop for OwnedChild {
        fn drop(&mut self) {
            kill_group(self.0.id(), libc::SIGKILL);
            let _ = self.0.wait();
        }
    }

    #[test]
    fn expired_checkpoint_deadline_does_not_grant_descendant_term_grace() {
        let path = std::env::temp_dir().join(format!(
            "darklane-expired-checkpoint-{}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let script = dir.0.join("descendant.sh");
        println!("owned checkpoint receipt dir={}", dir.0.display());
        let ready = dir.0.join("ready");
        let pid_file = dir.0.join("pid");
        let release = dir.0.join("release");
        let term = dir.0.join("term-after-deadline");
        std::fs::write(
            &script,
            format!(
                "trap 'echo TERM > \"{}\"; exit 0' TERM; echo ready > '{}'; while :; do sleep 0.01; done",
                term.display(),
                ready.display()
            ),
        )
        .unwrap();
        let cmd = format!(
            "trap '' HUP; sh '{}' & echo $! > '{}'; while test ! -f '{}'; do sleep 0.005; done; exit 75",
            script.display(),
            pid_file.display(),
            release.display()
        );
        let mut child = OwnedChild(launch(&cmd, 0).unwrap());
        wait_acknowledged("owned checkpoint descendant ready", || {
            ready.exists() && owned_pid_record(&pid_file).is_some()
        });
        let pid = owned_pid_record(&pid_file).unwrap();
        assert_eq!(unsafe { libc::getpgid(pid as i32) }, child.0.id() as i32);
        let owned = OwnedDescendant::open(pid);
        assert!(owned.alive());
        std::fs::write(release, []).unwrap();
        wait_acknowledged("checkpoint leader exited without reaping", || {
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let rc = unsafe {
                libc::waitid(
                    libc::P_PID,
                    child.0.id(),
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            assert_eq!(rc, 0);
            if info.si_code == libc::CLD_EXITED {
                assert_eq!(unsafe { info.si_status() }, 75);
                true
            } else {
                false
            }
        });
        let state = BgJobState::default();
        // The original checkpoint deadline is expired before cleanup begins. A TERM
        // handler writing this marker proves that cleanup granted a new grace period.
        preempt_wait(&mut child.0, &state, 0);
        assert_eq!(state.state.load(Ordering::Acquire), BG_PREEMPTED);
        assert!(
            !term.exists(),
            "an expired checkpoint deadline must not grant a fresh TERM grace"
        );
        wait_acknowledged("owned checkpoint descendant terminated", || !owned.alive());
        assert_eq!(child.0.wait().unwrap().code(), Some(75));
    }

    fn descendant_cleanup_case(action: &str) {
        let path = std::env::temp_dir().join(format!(
            "darklane-descendant-{}-{action}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let ready = dir.0.join("descendant");
        let release = dir.0.join("release");
        let tail = match action {
            "normal" => format!(
                "while test ! -f '{}'; do sleep 0.005; done; exit 0",
                release.display()
            ),
            "checkpoint" => "trap 'exit 75' USR1; wait".into(),
            "shutdown" => "wait".into(),
            _ => panic!("unknown owned-group test action"),
        };
        // The descendant inherits ignored HUP/TERM/USR1. The leader restores TERM,
        // so shutdown must not mistake its prompt exit for an empty group.
        let cmd = format!(
            "trap '' HUP TERM USR1; sleep 600 & echo $! > '{}'; trap - TERM; {tail}",
            ready.display()
        );
        let mode = if action == "checkpoint" {
            YieldMode::Checkpoint
        } else {
            YieldMode::Stop
        };
        let (sig, v, b) = sigs();
        let mut config = cfg(&cmd, mode);
        if action == "checkpoint" {
            config.ckpt_grace_ms = 100;
        }
        let h = spawn_runner(config, v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("owned descendant ready", || {
            st.state.load(Ordering::Acquire) == BG_RUNNING && owned_pid_record(&ready).is_some()
        });
        let pid = st.job_pid.load(Ordering::Acquire);
        let descendant = owned_pid_record(&ready).unwrap();
        assert_eq!(unsafe { libc::getpgid(descendant as i32) }, pid as i32);
        let owned = OwnedDescendant::open(descendant);
        assert!(owned.alive());
        sig.valley.store(false, Ordering::Release);
        match action {
            "normal" => {
                std::fs::write(release, []).unwrap();
                wait_acknowledged("normal group completion", || {
                    st.state.load(Ordering::Acquire) == BG_DONE
                        && st.job_pid.load(Ordering::Acquire) == 0
                });
                guard.shutdown();
            }
            "checkpoint" => {
                sig.busy.store(true, Ordering::Release);
                wait_acknowledged("checkpoint group completion", || {
                    st.state.load(Ordering::Acquire) == BG_PREEMPTED
                        && st.job_pid.load(Ordering::Acquire) == 0
                });
                assert_eq!(st.preempts.load(Ordering::Relaxed), 1);
                assert_eq!(st.ckpt_kills.load(Ordering::Relaxed), 0);
                guard.shutdown();
            }
            "shutdown" => guard.shutdown(),
            _ => unreachable!(),
        }
        wait_acknowledged("TERM-ignoring owned descendant terminated", || {
            !owned.alive()
        });
        assert_eq!(st.job_pid.load(Ordering::Acquire), 0);
        assert_eq!(st.yields.load(Ordering::Relaxed), 0);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn normal_leader_exit_terminates_term_ignoring_descendant() {
        descendant_cleanup_case("normal");
    }

    #[test]
    fn checkpoint_leader_exit_terminates_term_ignoring_descendant() {
        descendant_cleanup_case("checkpoint");
    }

    #[test]
    fn shutdown_terminates_descendant_after_leader_exits() {
        descendant_cleanup_case("shutdown");
    }

    #[test]
    fn killing_only_yielded_leader_terminates_remaining_owned_group() {
        let path =
            std::env::temp_dir().join(format!("darklane-leader-only-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let ready = dir.0.join("descendant");
        let cmd = format!(
            "trap '' HUP; sleep 600 & echo $! > '{}'; wait",
            ready.display()
        );
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("owned descendant ready", || {
            st.state.load(Ordering::Acquire) == BG_RUNNING && owned_pid_record(&ready).is_some()
        });
        let pid = st.job_pid.load(Ordering::Acquire);
        let descendant = owned_pid_record(&ready).unwrap();
        assert_eq!(unsafe { libc::getpgid(descendant as i32) }, pid as i32);
        let owned = OwnedDescendant::open(descendant);
        assert!(owned.alive());
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("owned group stopped", || {
            st.state.load(Ordering::Acquire) == BG_YIELDED
                && proc_state(pid) == Some('T')
                && proc_state(descendant) == Some('T')
        });
        // Deliberately kill only the leader, preserving the review's HUP-ignoring child.
        assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGKILL) }, 0);
        wait_acknowledged("killed leader classified", || {
            st.state.load(Ordering::Acquire) == BG_FAILED && st.job_pid.load(Ordering::Acquire) == 0
        });
        wait_acknowledged("remaining owned descendant terminated", || !owned.alive());
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 0);
        guard.shutdown();
    }

    fn exit_between_wait_and_stop(code: i32, expected: u8) {
        let path = std::env::temp_dir().join(format!(
            "darklane-exit-window-{}-{code}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let ready = dir.0.join("ready");
        let release = dir.0.join("release");
        let cmd = format!(
            "echo $$ > '{}'; while test ! -f '{}'; do sleep 0.005; done; exit {code}",
            ready.display(),
            release.display()
        );
        let (sig, valley, _) = sigs();
        let trigger = sig.busy.clone();
        let observed_exit = Arc::new(AtomicBool::new(false));
        let observed = observed_exit.clone();
        let busy = Arc::new(move || {
            if !trigger.swap(false, Ordering::AcqRel) {
                return false;
            }
            // Called after the runner's try_wait returned None. Let this exact child
            // exit, then observe it without reaping before the runner sends SIGSTOP.
            let pid: u32 = std::fs::read_to_string(&ready)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            std::fs::write(&release, []).unwrap();
            wait_acknowledged("owned child exited in stop window", || {
                let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                let rc = unsafe {
                    libc::waitid(
                        libc::P_PID,
                        pid,
                        &mut info,
                        libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                    )
                };
                assert_eq!(rc, 0, "waitid: {}", std::io::Error::last_os_error());
                if info.si_code == libc::CLD_EXITED {
                    assert_eq!(unsafe { info.si_status() }, code);
                    true
                } else {
                    false
                }
            });
            observed.store(true, Ordering::Release);
            true
        });
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), valley, busy, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("owned child ready", || {
            st.state.load(Ordering::Acquire) == BG_RUNNING && dir.0.join("ready").exists()
        });
        let pid = st.job_pid.load(Ordering::Acquire);
        assert_eq!(unsafe { libc::getpgid(pid as i32) }, pid as i32);
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("real exit classified and reaped", || {
            st.state.load(Ordering::Acquire) == expected && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert!(observed_exit.load(Ordering::Acquire));
        assert_eq!(st.yields.load(Ordering::Relaxed), 0);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 0);
        assert!(proc_state(pid).is_none());
        guard.shutdown();
    }

    #[test]
    fn normal_exit_between_wait_and_stop_is_done_without_yield() {
        exit_between_wait_and_stop(0, BG_DONE);
    }

    #[test]
    fn failed_exit_between_wait_and_stop_is_failed_without_yield() {
        exit_between_wait_and_stop(7, BG_FAILED);
    }

    #[test]
    fn killed_stopped_child_is_reaped_without_resume() {
        let (sig, v, b) = sigs();
        let h = spawn_runner(
            cfg("while :; do :; done", YieldMode::Stop),
            v,
            b,
            Arc::new(|| None),
        );
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        assert_eq!(
            signal_group(pid, -1).unwrap_err().raw_os_error(),
            Some(libc::EINVAL),
            "signal failures must be visible; this targets only the owned live group"
        );
        assert_eq!(st.yields.load(Ordering::Relaxed), 0);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 0);
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("acknowledged stop", || {
            st.state.load(Ordering::Acquire) == BG_YIELDED
        });
        assert_eq!(proc_state(pid), Some('T'));
        signal_group(pid, libc::SIGKILL).unwrap();
        wait_acknowledged("killed child reaped", || {
            st.state.load(Ordering::Acquire) == BG_FAILED && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 0);
        assert!(proc_state(pid).is_none());
        guard.shutdown();
    }

    #[test]
    fn repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown() {
        let (sig, v, b) = sigs();
        let h = spawn_runner(
            cfg("while :; do sleep 0.01; done", YieldMode::Stop),
            v,
            b,
            Arc::new(|| None),
        );
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        for cycle in 1..=2 {
            sig.valley.store(false, Ordering::Release);
            sig.busy.store(true, Ordering::Release);
            wait_acknowledged("fresh stop acknowledged", || {
                st.state.load(Ordering::Acquire) == BG_YIELDED
                    && st.yields.load(Ordering::Relaxed) == cycle
            });
            // dash spawns `sleep` with vfork: the stop can park the shell in D on a child
            // it caught before exec (research/darklane-vfork-stop-20261109/). That child is a
            // member of the group, and the runner waits for it to stop before it publishes the
            // yield, so this wait is redundant; it stays as a harmless guard.
            wait_acknowledged("yield to T, or to D parked in vfork", || {
                leader_stopped_or_vfork_parked(pid)
            });
            if cycle == 1 {
                sig.busy.store(false, Ordering::Release);
                sig.valley.store(true, Ordering::Release);
                wait_acknowledged("first resume", || {
                    st.resumes.load(Ordering::Relaxed) == 1
                        && matches!(proc_state(pid), Some('R' | 'S'))
                });
                assert!(
                    !child_stop_acknowledged(pid).unwrap(),
                    "the first stop must not acknowledge a resumed child"
                );
            }
        }
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        guard.shutdown();
        assert!(
            proc_state(pid).is_none(),
            "stopped shutdown must reap the child"
        );
        assert_eq!(st.job_pid.load(Ordering::Acquire), 0);
    }

    /// The /proc reading, case by case. A reading is (state, SIGSTOP in the thread's own
    /// pending set, SIGSTOP in the shared set). The single-threaded parked shell (D with SIGSTOP
    /// shared) and a running shell inside an ordinary vfork (D, nothing pending) differ only in
    /// the pending set; the second must never acknowledge. With more than one live thread a
    /// shared stop proves nothing: a blocked thread counts once a thread is stopped or the stop
    /// is in its own set, and a process with no stopped thread and a thread without its own stop
    /// gets the stop directed at each thread. A process with no live thread is gone. Group
    /// members are found by the process group in `/proc/<pid>/stat`.
    #[test]
    fn process_threads_classify_the_stop() {
        use ProcessStop::*;
        let t = |state, own, shared| ThreadStop { state, own, shared };
        for (threads, want) in [
            (vec![t('T', false, false)], TookEffect),
            (vec![t('t', false, false)], TookEffect),
            (vec![t('Z', false, false), t('T', false, false)], TookEffect),
            // A spawner thread parked in vfork while the others stop.
            (
                vec![
                    t('T', false, false),
                    t('T', false, false),
                    t('D', false, false),
                ],
                TookEffect,
            ),
            // The parked dash leader, and a running one inside an ordinary vfork.
            (vec![t('D', false, true)], TookEffect),
            (vec![t('D', false, false)], NotYet),
            (vec![t('Z', false, false), t('D', false, true)], TookEffect),
            (vec![t('R', false, true)], NotYet),
            // The main thread parked in vfork, the other thread sleeping or blocked elsewhere:
            // the kernel flagged the main thread only.
            (vec![t('D', false, true), t('S', false, true)], Unclaimed),
            (vec![t('D', false, true), t('D', false, true)], Unclaimed),
            (vec![t('D', false, true), t('I', false, true)], Unclaimed),
            (vec![t('D', true, true), t('D', false, true)], Unclaimed),
            // After the stop was directed at each thread.
            (vec![t('D', true, true), t('D', true, true)], TookEffect),
            (vec![t('D', true, true), t('S', true, true)], NotYet),
            (vec![t('D', true, true), t('T', true, true)], TookEffect),
            // A group stop under way: wait for it.
            (vec![t('S', false, false), t('T', false, false)], NotYet),
            // An exiting or unreaped process.
            (vec![t('Z', false, false)], Gone),
            (vec![], Gone),
        ] {
            assert_eq!(classify_process_stop(&threads), want, "{threads:?}");
        }
        // The probe's parked dash leader: SIGCHLD and SIGSTOP shared.
        let parked = "Name:\tsh\nState:\tD (disk sleep)\nSigPnd:\t0000000000000000\n\
                      ShdPnd:\t0000000000050000\n";
        assert_eq!(thread_stop(parked), Some(t('D', false, true)));
        let directed = "State:\tD (disk sleep)\nSigPnd:\t0000000000040000\n\
                        ShdPnd:\t0000000000050000\n";
        assert_eq!(thread_stop(directed), Some(t('D', true, true)));
        let resumed = "State:\tS (sleeping)\nSigPnd:\t0000000000000000\n\
                       ShdPnd:\t0000000000010000\n";
        assert_eq!(thread_stop(resumed), Some(t('S', false, false)));
        assert_eq!(thread_stop("State:\tD (disk sleep)\nSigPnd:\t0\n"), None);
        assert_eq!(thread_stop("SigPnd:\t0\nShdPnd:\t0\n"), None);
        assert_eq!(
            proc_stat_pgrp("4242 (a ) (b) T 4200 4100 4100 0"),
            Some(4100)
        );
        assert_eq!(proc_stat_pgrp("4242 (sh) S 1 4242 4242 0"), Some(4242));
        assert_eq!(proc_stat_pgrp("4242 (sh) S 1"), None);
    }

    /// Set in the job's environment: this test binary then runs as the job leader below.
    const VFORK_LEADER_FIFO: &str = "DARKLANE_TEST_VFORK_FIFO";

    /// The job-leader half of the next test. glibc and musl posix_spawn clone with
    /// CLONE_VM | CLONE_VFORK and run file actions in the child before exec, so the
    /// spawning thread waits in vfork while the child blocks opening the FIFO for reading.
    /// The other threads (the test harness's and this one, joining) are ordinary sleepers.
    fn vfork_leader(fifo: std::ffi::OsString) {
        use std::os::unix::ffi::OsStrExt;
        let fifo = std::ffi::CString::new(fifo.as_bytes()).unwrap();
        let spawner = std::thread::spawn(move || unsafe {
            let mut actions: libc::posix_spawn_file_actions_t = std::mem::zeroed();
            assert_eq!(libc::posix_spawn_file_actions_init(&mut actions), 0);
            assert_eq!(
                libc::posix_spawn_file_actions_addopen(
                    &mut actions,
                    0,
                    fifo.as_ptr(),
                    libc::O_RDONLY,
                    0
                ),
                0
            );
            let argv = [
                c"sh".as_ptr().cast_mut(),
                c"-c".as_ptr().cast_mut(),
                c"exit 0".as_ptr().cast_mut(),
                std::ptr::null_mut(),
            ];
            let envp = [std::ptr::null_mut()];
            let mut pid = 0;
            let rc = libc::posix_spawn(
                &mut pid,
                c"/bin/sh".as_ptr(),
                &actions,
                std::ptr::null(),
                argv.as_ptr(),
                envp.as_ptr(),
            );
            assert_eq!(rc, 0, "posix_spawn");
            libc::posix_spawn_file_actions_destroy(&mut actions);
            let mut status = 0;
            assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
            assert!(
                libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
                "spawned child status {status}"
            );
        });
        spawner.join().unwrap();
    }

    /// A group SIGSTOP that lands while a leader thread waits in vfork stops the vfork child
    /// before exec, and the waiting thread cannot stop until that child execs or exits: it
    /// stays in D until SIGCONT, the leader's group stop never completes, and waitid never
    /// reports it stopped. The runner must still acknowledge the yield and resume the job
    /// (research/darklane-vfork-stop-20261109/: dash spawns every external command this way).
    #[test]
    fn stop_acknowledges_a_leader_parked_in_vfork_on_a_child_stopped_before_exec() {
        if let Some(fifo) = std::env::var_os(VFORK_LEADER_FIFO) {
            return vfork_leader(fifo);
        }
        let path = std::env::temp_dir().join(format!("darklane-vfork-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let fifo = dir.0.join("fifo");
        make_fifo(&fifo);
        let cmd = format!(
            "{VFORK_LEADER_FIFO}='{}' exec '{}' --exact \
             darklane::tests::stop_acknowledges_a_leader_parked_in_vfork_on_a_child_stopped_before_exec \
             --nocapture > '{}' 2>&1",
            fifo.display(),
            std::env::current_exe().unwrap().display(),
            dir.0.join("leader.log").display()
        );
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        wait_acknowledged(
            "spawning thread waits in vfork on a child opening the FIFO",
            || {
                leader_thread_states(pid).contains(&'D')
                    && unexeced_children(pid).iter().any(|&(_, s)| s == 'S')
            },
        );
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("stop acknowledged with a thread parked in vfork", || {
            st.state.load(Ordering::Acquire) == BG_YIELDED && st.yields.load(Ordering::Relaxed) == 1
        });
        // The vfork child is a member of the group, and the runner waits for it to stop before it
        // publishes the yield, so this wait is redundant; it stays as a harmless guard.
        wait_acknowledged("threads D and T, the vfork child stopped", || {
            let threads = leader_thread_states(pid);
            threads.contains(&'D') && threads.contains(&'T') && leader_stopped_or_vfork_parked(pid)
        });
        assert!(
            !leader_reports_stopped(pid),
            "the leader's group stop cannot complete while a thread waits in vfork"
        );
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("resumed, the vfork child back in its open", || {
            st.resumes.load(Ordering::Relaxed) == 1
                && unexeced_children(pid).iter().all(|&(_, s)| s != 'T')
        });
        // A writer releases the child's open: it execs, posix_spawn returns, the leader exits 0.
        wait_acknowledged("FIFO writer opened", || open_fifo_writer(&fifo));
        wait_acknowledged("leader completed", || {
            st.state.load(Ordering::Acquire) == BG_DONE && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        guard.shutdown();
    }

    /// Set in the job's environment to a directory: the test binary then runs
    /// [`main_thread_leader`] before libtest starts.
    const MAIN_LEADER_DIR: &std::ffi::CStr = c"DARKLANE_TEST_MAIN_LEADER_DIR";
    /// `sleeper` or `blocked`: what the leader's second thread does.
    const MAIN_LEADER_CASE: &std::ffi::CStr = c"DARKLANE_TEST_MAIN_LEADER_CASE";

    // Every process of this test binary runs this before main. It returns at once unless the
    // job's environment names a main-thread leader.
    #[used]
    #[unsafe(link_section = ".init_array")]
    static MAIN_THREAD_LEADER: extern "C" fn() = main_thread_leader;

    /// The job-leader half of the two main-thread tests. It runs before libtest, so the thread
    /// waiting in vfork is the leader's main thread, the one the kernel flags first to take a
    /// group stop. The main thread blocks SIGCHLD and starts a second thread, which inherits
    /// the mask (as helper threads that block signals do): the CLD_STOPPED notice of the
    /// stopped vfork child then cannot wake it to take the stop by accident. `sleeper`: the
    /// second thread sleeps in a loop. `blocked`: it first waits in vfork itself, on a child
    /// that leaves the job's process group (the group stop does not reach it) and blocks opening
    /// `held` before exec; when that wait ends it creates `ran`, then sleeps in a loop. The main
    /// thread posix_spawns a child that blocks opening `fifo` before exec, waits for it and
    /// exits 0. Raw libc only: the Rust runtime is not set up yet.
    extern "C" fn main_thread_leader() {
        use std::ffi::{CStr, CString};
        let dir = unsafe { libc::getenv(MAIN_LEADER_DIR.as_ptr()) };
        if dir.is_null() {
            return;
        }
        let dir = unsafe { CStr::from_ptr(dir) }.to_bytes();
        let path = |name: &str| CString::new([dir, &b"/"[..], name.as_bytes()].concat()).unwrap();
        let case = unsafe { libc::getenv(MAIN_LEADER_CASE.as_ptr()) };
        let blocked = !case.is_null() && unsafe { CStr::from_ptr(case) } == c"blocked";
        let second: &'static SecondThread = Box::leak(Box::new(SecondThread {
            held: blocked.then(|| path("held")),
            ran: path("ran"),
        }));
        let fifo = path("fifo");
        unsafe {
            let mut chld: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut chld);
            libc::sigaddset(&mut chld, libc::SIGCHLD);
            libc::pthread_sigmask(libc::SIG_BLOCK, &chld, std::ptr::null_mut());
            let mut thread: libc::pthread_t = std::mem::zeroed();
            let arg = second as *const SecondThread as *mut libc::c_void;
            if libc::pthread_create(&mut thread, std::ptr::null(), second_thread, arg) != 0 {
                libc::_exit(3);
            }
            let mut actions: libc::posix_spawn_file_actions_t = std::mem::zeroed();
            libc::posix_spawn_file_actions_init(&mut actions);
            libc::posix_spawn_file_actions_addopen(
                &mut actions,
                0,
                fifo.as_ptr(),
                libc::O_RDONLY,
                0,
            );
            let argv = [c"true".as_ptr().cast_mut(), std::ptr::null_mut()];
            let envp = [std::ptr::null_mut()];
            let mut pid = 0;
            let rc = libc::posix_spawn(
                &mut pid,
                c"/bin/true".as_ptr(),
                &actions,
                std::ptr::null(),
                argv.as_ptr(),
                envp.as_ptr(),
            );
            let mut status = 0;
            let ok = rc == 0
                && libc::waitpid(pid, &mut status, 0) == pid
                && libc::WIFEXITED(status)
                && libc::WEXITSTATUS(status) == 0;
            libc::_exit(if ok { 0 } else { 4 });
        }
    }

    struct SecondThread {
        held: Option<std::ffi::CString>,
        ran: std::ffi::CString,
    }

    extern "C" fn second_thread(arg: *mut libc::c_void) -> *mut libc::c_void {
        let second = unsafe { &*(arg as *const SecondThread) };
        if let Some(held) = &second.held {
            // clone, not vfork: the child runs on its own stack until it execs.
            let mut stack = vec![0u8; 256 << 10];
            let top = ((stack.as_mut_ptr() as usize + stack.len()) & !15) as *mut libc::c_void;
            let flags = libc::CLONE_VM | libc::CLONE_VFORK | libc::SIGCHLD;
            let arg = held.as_ptr().cast_mut().cast();
            if unsafe { libc::clone(held_child, top, flags, arg) } < 0 {
                unsafe { libc::_exit(5) };
            }
            // Job code: it must not run while the job is yielded.
            let flags = libc::O_CREAT | libc::O_WRONLY | libc::O_CLOEXEC;
            unsafe { libc::close(libc::open(second.ran.as_ptr(), flags, 0o600)) };
        }
        let tick = libc::timespec {
            tv_sec: 0,
            tv_nsec: 1_000_000,
        };
        loop {
            unsafe { libc::nanosleep(&tick, std::ptr::null_mut()) };
        }
    }

    extern "C" fn held_child(held: *mut libc::c_void) -> libc::c_int {
        unsafe {
            libc::setpgid(0, 0);
            libc::open(held.cast_const().cast(), libc::O_RDONLY);
            let argv = [c"true".as_ptr(), std::ptr::null()];
            libc::execv(c"/bin/true".as_ptr(), argv.as_ptr());
            libc::_exit(127)
        }
    }

    /// The job command for [`main_thread_leader`], with its FIFOs made in `dir`. With `exec` the
    /// program replaces the shell as the group's leader; without it the shell stays the leader
    /// and the program is another member of the group (the trailing `exit` keeps any shell
    /// from exec'ing it).
    fn main_thread_leader_job(case: &str, dir: &TestDir, exec: bool) -> String {
        make_fifo(&dir.0.join("fifo"));
        make_fifo(&dir.0.join("held"));
        format!(
            "{}={case} {}='{}' {}'{}' > '{}' 2>&1{}",
            MAIN_LEADER_CASE.to_str().unwrap(),
            MAIN_LEADER_DIR.to_str().unwrap(),
            dir.0.display(),
            if exec { "exec " } else { "" },
            std::env::current_exe().unwrap().display(),
            dir.0.join("leader.log").display(),
            if exec { "" } else { "; exit $?" }
        )
    }

    /// The children of `pid`.
    fn children_of(pid: u32) -> Vec<u32> {
        let Ok(procs) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        procs
            .filter_map(|p| {
                let child: u32 = p.ok()?.file_name().to_str()?.parse().ok()?;
                let (_, ppid, _) =
                    proc_stat_fields(std::path::Path::new(&format!("/proc/{child}/stat")))?;
                (ppid == pid).then_some(child)
            })
            .collect()
    }

    /// The state of the leader's main thread and of its other thread.
    fn main_and_other(pid: u32) -> Option<(char, char)> {
        let threads = leader_tids(pid);
        let [a, b] = threads[..] else {
            return None;
        };
        if a.0 == pid {
            Some((a.1, b.1))
        } else {
            Some((b.1, a.1))
        }
    }

    /// The kernel flags one thread to take a group SIGSTOP, the main thread first, and tells
    /// no other. When the main thread waits in vfork it cannot take the stop, so no thread ever
    /// stops: the other thread runs on and waitid never reports the stop. The runner must direct
    /// the stop at each thread, acknowledge the yield once the other thread stops, and resume
    /// the job (research/darklane-vfork-stop-20261109/, review round).
    #[test]
    fn stop_reaches_the_other_thread_of_a_leader_whose_main_thread_waits_in_vfork() {
        let path = std::env::temp_dir().join(format!("darklane-main-vfork-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let cmd = main_thread_leader_job("sleeper", &dir, true);
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        wait_acknowledged(
            "main thread waits in vfork on a child opening the FIFO",
            || {
                matches!(main_and_other(pid), Some(('D', 'R' | 'S')))
                    && unexeced_children(pid).iter().any(|&(_, s)| s == 'S')
            },
        );
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged(
            "stop acknowledged with the main thread parked in vfork",
            || {
                st.state.load(Ordering::Acquire) == BG_YIELDED
                    && st.yields.load(Ordering::Relaxed) == 1
            },
        );
        wait_acknowledged(
            "main thread D, the other thread and the vfork child stopped",
            || main_and_other(pid) == Some(('D', 'T')) && leader_stopped_or_vfork_parked(pid),
        );
        assert!(
            !leader_reports_stopped(pid),
            "the leader's group stop cannot complete while its main thread waits in vfork"
        );
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged(
            "resumed, the other thread and the vfork child running",
            || {
                st.resumes.load(Ordering::Relaxed) == 1
                    && matches!(main_and_other(pid), Some(('D', 'R' | 'S')))
                    && unexeced_children(pid).iter().all(|&(_, s)| s != 'T')
            },
        );
        wait_acknowledged("FIFO writer opened", || {
            open_fifo_writer(&dir.0.join("fifo"))
        });
        wait_acknowledged("leader completed", || {
            st.state.load(Ordering::Acquire) == BG_DONE && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        guard.shutdown();
    }

    /// With more than one live thread, a stop still in the shared pending set does not say which
    /// thread the kernel flagged to take it. Here the main thread waits in vfork and the other
    /// thread waits in vfork too, on a child outside the job's process group: both are blocked,
    /// the stop is pending, and only the main thread is flagged. When the other thread's wait
    /// ends it must stop before it runs job code, which would create `ran`.
    #[test]
    fn stop_holds_the_other_blocked_thread_of_a_leader_whose_main_thread_waits_in_vfork() {
        let path =
            std::env::temp_dir().join(format!("darklane-main-blocked-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let ran = dir.0.join("ran");
        let cmd = main_thread_leader_job("blocked", &dir, true);
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        wait_acknowledged(
            "both threads wait in vfork on children opening FIFOs",
            || {
                let children = unexeced_children(pid);
                main_and_other(pid) == Some(('D', 'D'))
                    && children.len() == 2
                    && children.iter().all(|&(_, s)| s == 'S')
            },
        );
        let outside: Vec<u32> = unexeced_children(pid)
            .into_iter()
            .map(|(child, _)| child)
            .filter(|&child| unsafe { libc::getpgid(child as i32) } != pid as i32)
            .collect();
        assert_eq!(outside.len(), 1, "one vfork child left the job's group");
        let _held = OwnedDescendant::open(outside[0]);
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("stop acknowledged with both threads blocked", || {
            st.state.load(Ordering::Acquire) == BG_YIELDED && st.yields.load(Ordering::Relaxed) == 1
        });
        // End the other thread's wait: its child opens `held` and execs.
        wait_acknowledged("held FIFO writer opened", || {
            open_fifo_writer(&dir.0.join("held"))
        });
        wait_acknowledged("the other thread stops, or runs job code", || {
            ran.exists() || main_and_other(pid) == Some(('D', 'T'))
        });
        assert!(
            !ran.exists(),
            "job code ran after the yield was acknowledged: threads {:?}",
            leader_tids(pid)
        );
        assert_eq!(st.state.load(Ordering::Acquire), BG_YIELDED);
        assert!(
            !leader_reports_stopped(pid),
            "the leader's group stop cannot complete while its main thread waits in vfork"
        );
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("resumed, the other thread runs job code", || {
            st.resumes.load(Ordering::Relaxed) == 1 && ran.exists()
        });
        wait_acknowledged("FIFO writer opened", || {
            open_fifo_writer(&dir.0.join("fifo"))
        });
        wait_acknowledged("leader completed", || {
            st.state.load(Ordering::Acquire) == BG_DONE && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        guard.shutdown();
    }

    /// `sh -c` is the job's leader, and dash does not exec even a single command, so the job's
    /// program is usually another member of the process group. Here the program's main thread
    /// waits in vfork and its other thread blocks SIGCHLD. The group stop stops the shell at once,
    /// so waitid reports the leader stopped, while the program's other thread is never told and
    /// runs on. The runner must publish the yield only once every member is stopped: the shape is
    /// asserted right at the acknowledgement, with no wait.
    #[test]
    fn stop_reaches_the_other_thread_of_a_group_member_whose_main_thread_waits_in_vfork() {
        let path =
            std::env::temp_dir().join(format!("darklane-member-vfork-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let dir = TestDir(path);
        let cmd = main_thread_leader_job("sleeper", &dir, false);
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&cmd, YieldMode::Stop), v, b, Arc::new(|| None));
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        // The program is the shell's only child.
        let program = || match children_of(pid)[..] {
            [program] => Some(program),
            _ => None,
        };
        wait_acknowledged(
            "the program's main thread waits in vfork on a child opening the FIFO",
            || {
                program().is_some_and(|p| {
                    matches!(main_and_other(p), Some(('D', 'R' | 'S')))
                        && unexeced_children(p).iter().any(|&(_, s)| s == 'S')
                })
            },
        );
        let prog = program().unwrap();
        assert_eq!(unsafe { libc::getpgid(prog as i32) }, pid as i32);
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged(
            "stop acknowledged with the program's main thread parked in vfork",
            || {
                st.state.load(Ordering::Acquire) == BG_YIELDED
                    && st.yields.load(Ordering::Relaxed) == 1
            },
        );
        assert_eq!(proc_state(pid), Some('T'), "the shell stops at once");
        assert_eq!(
            main_and_other(prog),
            Some(('D', 'T')),
            "the program's other thread was running at the yield: threads {:?}",
            leader_tids(prog)
        );
        let children = unexeced_children(prog);
        assert!(
            !children.is_empty() && children.iter().all(|&(_, s)| s == 'T'),
            "the vfork child was running at the yield: {children:?}"
        );
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged(
            "resumed, the program's other thread and the vfork child running",
            || {
                st.resumes.load(Ordering::Relaxed) == 1
                    && matches!(main_and_other(prog), Some(('D', 'R' | 'S')))
                    && unexeced_children(prog).iter().all(|&(_, s)| s != 'T')
            },
        );
        wait_acknowledged("FIFO writer opened", || {
            open_fifo_writer(&dir.0.join("fifo"))
        });
        wait_acknowledged("job completed", || {
            st.state.load(Ordering::Acquire) == BG_DONE && st.job_pid.load(Ordering::Acquire) == 0
        });
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        guard.shutdown();
    }

    /// The flake's own shape, at volume: dash vforks `/bin/true` constantly, so many stops
    /// land in the vfork window. Every stop must be acknowledged and resumed.
    #[test]
    #[ignore = "stress: 300 stop/resume cycles on a constantly spawning shell job; run with --ignored"]
    fn stop_resume_cycles_on_a_constantly_spawning_shell_job() {
        let (sig, v, b) = sigs();
        let h = spawn_runner(
            cfg("while :; do /bin/true; done", YieldMode::Stop),
            v,
            b,
            Arc::new(|| None),
        );
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        let mut parked = 0;
        for cycle in 1..=300 {
            sig.valley.store(false, Ordering::Release);
            sig.busy.store(true, Ordering::Release);
            wait_acknowledged(&format!("stop {cycle} acknowledged"), || {
                st.state.load(Ordering::Acquire) == BG_YIELDED
                    && st.yields.load(Ordering::Relaxed) == cycle
            });
            // The vfork child is a group member the runner waits for: a redundant, harmless wait.
            wait_acknowledged(
                &format!("stop {cycle}: yield to T, or to D parked in vfork"),
                || leader_stopped_or_vfork_parked(pid),
            );
            if leader_thread_states(pid).contains(&'D') {
                parked += 1;
            }
            sig.busy.store(false, Ordering::Release);
            sig.valley.store(true, Ordering::Release);
            wait_acknowledged(&format!("resume {cycle}"), || {
                st.resumes.load(Ordering::Relaxed) == cycle
                    && st.state.load(Ordering::Acquire) == BG_RUNNING
            });
        }
        println!("stops acknowledged with the leader parked in vfork: {parked} of 300");
        guard.shutdown();
    }

    #[test]
    fn valley_signal_reads_worker_truth() {
        let h = WorkerHealth::new();
        let v = ValleySignal::new(h.clone());
        // LOADING is not idleness.
        assert_eq!(v.idle_seconds(), 0.0);
        assert!(v.busy_now());
        // IDLE: age accrues from the phase stamp. Retry-tolerant: PENDING_ADMITS is
        // process-global and handler tests running in parallel bump it transiently.
        h.set_phase(PHASE_IDLE);
        std::thread::sleep(std::time::Duration::from_millis(30));
        // (>= 0.02 implies !busy_now at that instant — no separate racy assert.)
        wait_for("idle age accrues", 2000, || v.idle_seconds() >= 0.02);
        // a pending admit is traffic even while the phase is still IDLE (the handoff gap).
        PENDING_ADMITS.fetch_add(1, Ordering::Release);
        assert_eq!(v.idle_seconds(), 0.0);
        assert!(v.busy_now());
        PENDING_ADMITS.fetch_sub(1, Ordering::Release);
        // BUSY: zero again.
        h.beat_busy();
        assert_eq!(v.idle_seconds(), 0.0);
    }

    /// memra#500: a DSv4 request runs on its own thread, so the scheduler sits IDLE through
    /// it. Before routes published, the valley read that as quiet and resumed a background job
    /// on top of a live request.
    #[test]
    fn a_busy_serve_route_is_not_a_valley() {
        let h = WorkerHealth::new();
        let v = ValleySignal::new(h.clone());
        h.set_phase(PHASE_IDLE);
        let load = crate::route_telemetry::RouteLoad::new("valley-route", 1);
        let r = h.register_route("valley-route", load.clone());
        r.set_idle();
        std::thread::sleep(std::time::Duration::from_millis(30));
        wait_for("idle age accrues", 2000, || v.idle_seconds() >= 0.02);
        r.begin_request();
        assert_eq!(v.idle_seconds(), 0.0, "the route is serving");
        r.set_idle();
        // idle again but with a request reserved on the route's queue: traffic
        let ticket = load.try_reserve(0, 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(v.idle_seconds(), 0.0, "a queued route request is traffic");
        drop(ticket);
        wait_for("idle age accrues again", 2000, || v.idle_seconds() >= 0.02);
    }

    #[test]
    fn stop_mode_full_cycle_launch_yield_resume_shutdown() {
        let (sig, v, b) = sigs();
        let h = spawn_runner(
            cfg("while :; do sleep 0.01; done", YieldMode::Stop),
            v,
            b,
            Arc::new(|| None),
        );
        let st = h.state.clone();
        let guard = RunnerGuard(Some(h));
        // no valley -> no launch.
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.state.load(Ordering::Acquire), BG_WAITING);
        // valley -> launch.
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("launch", || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        assert!(pid > 0);
        wait_acknowledged("job running", || matches!(proc_state(pid), Some('R' | 'S')));
        // busy edge -> SIGSTOP. The wiring claim is the yield itself plus the counters;
        // wall-clock tightness is poll_ms config, not an OS promise — a 500ms bound
        // starved out under a co-running perf battery (2026-08-30, local-ci full load:
        // the runner thread didn't get scheduled for >500ms), and the 3s bound that replaced it
        // starved out too (WP-A day 57: 1 of 100 beside sixteen burners). The runner acknowledges
        // the yield (its state), then the job reads stopped.
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_acknowledged("the runner yields", || {
            st.state.load(Ordering::Acquire) == BG_YIELDED
        });
        wait_acknowledged("yield to T, or to D parked in vfork", || {
            leader_stopped_or_vfork_parked(pid)
        });
        assert_eq!(st.state.load(Ordering::Acquire), BG_YIELDED);
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        // back to valley -> SIGCONT.
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_acknowledged("the runner resumes", || {
            st.resumes.load(Ordering::Relaxed) == 1
        });
        wait_acknowledged("resume", || matches!(proc_state(pid), Some('R' | 'S')));
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        // shutdown never leaves an orphan (stopped or otherwise).
        guard.shutdown();
        wait_acknowledged("job reaped", || proc_state(pid).is_none_or(|s| s == 'Z'));
    }

    #[test]
    fn checkpoint_mode_preempts_and_resumes_counter() {
        // The toy proof in miniature: a counter that checkpoints to disk on SIGUSR1 and
        // exits 75; a relaunch resumes FROM the file. This is the deliverable-4 seam.
        let dir = std::env::temp_dir().join(format!("darklane-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("counter");
        let job = format!(
            "n=$(cat {f} 2>/dev/null || echo 0); trap 'echo $n > {f}; exit 75' USR1; \
             while :; do n=$((n+1)); echo $n > {f}.live; sleep 0.005; done",
            f = f.display()
        );
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&job, YieldMode::Checkpoint), v, b, Arc::new(|| None));
        let st = h.state.clone();
        sig.valley.store(true, Ordering::Release);
        wait_for("launch", 1000, || {
            st.state.load(Ordering::Acquire) == BG_RUNNING
        });
        // let it make progress, then preempt.
        wait_for("progress", 2000, || {
            std::fs::read_to_string(f.with_extension("live"))
                .is_ok_and(|s| s.trim().parse::<u64>().unwrap_or(0) >= 3)
        });
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_for("preempted", 3000, || {
            st.state.load(Ordering::Acquire) == BG_PREEMPTED
        });
        let ck1: u64 = std::fs::read_to_string(&f).unwrap().trim().parse().unwrap();
        assert!(ck1 >= 3, "checkpoint wrote the counter (got {ck1})");
        assert_eq!(st.preempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            st.ckpt_kills.load(Ordering::Relaxed),
            0,
            "clean ckpt, no kill"
        );
        // next valley -> relaunch resumes FROM the checkpoint (not from zero).
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_for("relaunch", 1000, || {
            st.launches.load(Ordering::Relaxed) == 2
        });
        wait_for("resumed past ck1", 2000, || {
            std::fs::read_to_string(f.with_extension("live"))
                .is_ok_and(|s| s.trim().parse::<u64>().unwrap_or(0) > ck1)
        });
        h.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn job_completion_and_failure_are_terminal() {
        // exit 0 = done, never relaunched.
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg("true", YieldMode::Stop), v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("done", 2000, || st.state.load(Ordering::Acquire) == BG_DONE);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.launches.load(Ordering::Relaxed), 1, "done is terminal");
        h.shutdown();
        // nonzero exit = failed, never relaunched.
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg("exit 3", YieldMode::Stop), v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("failed", 2000, || {
            st.state.load(Ordering::Acquire) == BG_FAILED
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.launches.load(Ordering::Relaxed), 1, "failed is terminal");
        h.shutdown();
    }

    #[test]
    fn vram_budget_refused_when_it_does_not_fit_and_fail_closed() {
        // budget 1000MB + headroom 2048MB > free 2500MB -> refused.
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1000;
        let h = spawn_runner(c, v, b, Arc::new(|| Some(2500)));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("refused", 1000, || {
            st.state.load(Ordering::Acquire) == BG_REFUSED
        });
        assert_eq!(st.launches.load(Ordering::Relaxed), 0);
        h.shutdown();
        // unreadable probe -> fail closed (still refused).
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1;
        let h = spawn_runner(c, v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("refused (fail closed)", 1000, || {
            st.state.load(Ordering::Acquire) == BG_REFUSED
        });
        h.shutdown();
        // fits -> launches (and completes).
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1000;
        let h = spawn_runner(c, v, b, Arc::new(|| Some(4000)));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("launched when fits", 2000, || {
            st.state.load(Ordering::Acquire) == BG_DONE
        });
        h.shutdown();
    }
}
