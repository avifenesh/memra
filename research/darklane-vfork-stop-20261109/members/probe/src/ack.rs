/// Observe a stop without reaping an exit that Child::try_wait must classify.
/// Sending SIGSTOP successfully does not prove the child remains alive or stopped.
///
/// waitid reports a stop only once EVERY thread of the leader has stopped, and a thread
/// waiting in vfork cannot stop until its child execs or exits. The same group SIGSTOP stops
/// that child before exec, so the waiting thread stays in D until SIGCONT and waitid never
/// reports the stop. dash spawns every external command with vfork; glibc's posix_spawn
/// (and through it CPython's subprocess) clones with CLONE_VFORK too
/// (`research/darklane-vfork-stop-20261109/`). When waitid has nothing to report, the stop is
/// read from /proc instead, and directed at each leader thread when no thread has taken it:
/// see [`leader_stop_took_effect`]. Called only after a group SIGSTOP was sent successfully.
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
    if info.si_code == libc::CLD_STOPPED {
        return Ok(true);
    }
    if unsafe { info.si_pid() } != 0 {
        return Ok(false); // An exit: try_wait reaps and classifies it on the next tick.
    }
    leader_stop_took_effect(pid)
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

/// What the leader's threads say about a submitted group stop.
#[derive(Debug, PartialEq, Eq)]
enum LeaderStop {
    /// Every live thread is stopped, or blocked in the kernel and bound to stop before it next
    /// returns to user space.
    TookEffect,
    /// Not yet, and a later tick reads again: a thread can still run (or none is alive) while
    /// a group stop is under way, the leader has one live thread, or every live thread already
    /// carries the stop in its own pending set.
    NotYet,
    /// No thread has taken the stop, more than one is live, and some live thread does not carry
    /// it in its own pending set. The stop must be directed at each live thread.
    Unclaimed,
}

fn classify_leader_stop(threads: &[ThreadStop]) -> LeaderStop {
    let live: Vec<ThreadStop> = threads
        .iter()
        .copied()
        .filter(|t| !matches!(t.state, 'Z' | 'X'))
        .collect();
    let stopped = live.iter().any(|t| matches!(t.state, 'T' | 't'));
    let single = live.len() == 1;
    let bound_to_stop = |t: &ThreadStop| match t.state {
        'T' | 't' => true,
        // D, and I (TASK_IDLE = uninterruptible without load accounting).
        'D' | 'I' => stopped || t.own || (single && t.shared),
        _ => false,
    };
    if !live.is_empty() && live.iter().all(bound_to_stop) {
        LeaderStop::TookEffect
    } else if !stopped && !single && live.iter().any(|t| !t.own) {
        LeaderStop::Unclaimed
    } else {
        LeaderStop::NotYet
    }
}

/// The leader's threads (tid and reading), or None once the leader is gone.
fn leader_threads(pid: u32) -> std::io::Result<Option<Vec<(libc::pid_t, ThreadStop)>>> {
    let gone = |err: &std::io::Error| {
        err.kind() == std::io::ErrorKind::NotFound || err.raw_os_error() == Some(libc::ESRCH)
    };
    let tasks = match std::fs::read_dir(format!("/proc/{pid}/task")) {
        Ok(tasks) => tasks,
        Err(err) if gone(&err) => return Ok(None),
        Err(err) => return Err(err),
    };
    let unparsable =
        || std::io::Error::new(std::io::ErrorKind::InvalidData, "unparsable task status");
    let mut threads = Vec::new();
    for task in tasks {
        let task = task?;
        let tid = task
            .file_name()
            .to_str()
            .and_then(|tid| tid.parse().ok())
            .ok_or_else(unparsable)?;
        match std::fs::read_to_string(task.path().join("status")) {
            Ok(status) => threads.push((tid, thread_stop(&status).ok_or_else(unparsable)?)),
            Err(err) if gone(&err) => {} // The thread exited after the listing.
            Err(err) => return Err(err),
        }
    }
    Ok(Some(threads))
}

/// The stop has taken effect on the leader when every live thread is stopped, or blocked in the
/// kernel (D, or I) and bound to stop before it next returns to user space, so that no leader
/// thread runs job code again before SIGCONT. A blocked thread is bound to stop when:
///   * some thread is stopped: the group stop is under way and marked every thread for it;
///   * SIGSTOP is in the thread's own pending set: the kernel flagged that thread to take it;
///   * the thread is the leader's only live thread and SIGSTOP is in the shared pending set.
///
/// The last condition holds for one live thread only. The kernel queues a group SIGSTOP once
/// and flags one thread to take it, the main thread first, and tells no other. With more
/// threads the shared set does not say which thread was flagged, and an unflagged thread whose
/// wait ends returns to user space and runs. When the flagged thread is the one waiting in
/// vfork, no thread takes the stop at all: the others run on and never start the group stop.
/// So when no thread is stopped and more than one is live, the stop is directed at each live
/// thread (tgkill), which flags each one, and read again. A flagged sleeping thread starts the
/// group stop, and a flagged blocked one carries the stop in its own pending set. tgkill checks
/// the thread group, so a stale tid cannot reach another process, and SIGCONT flushes
/// thread-directed stops with the group's. Called only after a group SIGSTOP was sent
/// successfully.
fn leader_stop_took_effect(pid: u32) -> std::io::Result<bool> {
    let reading = |threads: &[(libc::pid_t, ThreadStop)]| {
        classify_leader_stop(&threads.iter().map(|&(_, t)| t).collect::<Vec<_>>())
    };
    let Some(threads) = leader_threads(pid)? else {
        return Ok(false);
    };
    match reading(&threads) {
        LeaderStop::TookEffect => return Ok(true),
        LeaderStop::NotYet => return Ok(false),
        LeaderStop::Unclaimed => {}
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
    Ok(leader_threads(pid)?.is_some_and(|threads| reading(&threads) == LeaderStop::TookEffect))
}
