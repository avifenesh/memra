// Probe for the PR #1130 re-review: runs ba03102882's acknowledgement code (copied verbatim
// into ack.rs by extract.py) against job shapes, the way runner_loop drives it: `sh -c CMD` in
// its own process group, group SIGSTOP, then child_stop_acknowledged every 25 ms.
#![allow(dead_code)]

use std::io::Write;
use std::os::unix::process::CommandExt;
use std::time::{Duration, Instant};

include!("ack.rs");

fn read_ctr(path: &std::path::Path) -> u64 {
    std::fs::read(path)
        .ok()
        .and_then(|b| b.get(..8).map(|s| u64::from_ne_bytes(s.try_into().unwrap())))
        .unwrap_or(0)
}

fn stat_fields(pid: u32) -> Option<(String, char, u32, i32)> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let comm = s[s.find('(')? + 1..s.rfind(')')?].to_string();
    let rest: Vec<&str> = s[s.rfind(')')? + 1..].split_whitespace().collect();
    Some((comm, rest[0].chars().next()?, rest[1].parse().ok()?, rest[2].parse().ok()?))
}

fn procs() -> Vec<u32> {
    std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect()
}

/// Every thread: tid, State letter, SigPnd, ShdPnd.
fn threads(pid: u32) -> Vec<String> {
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return vec![];
    };
    let mut out = vec![];
    for t in tasks.flatten() {
        let Ok(s) = std::fs::read_to_string(t.path().join("status")) else { continue };
        let f = |n: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(n))
                .map(|v| v.trim().to_string())
                .unwrap_or_default()
        };
        out.push(format!(
            "{}:{}:own={}:shd={}",
            t.file_name().to_str().unwrap(),
            f("State:").chars().next().unwrap_or('?'),
            f("SigPnd:"),
            f("ShdPnd:")
        ));
    }
    out.sort();
    out
}

fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let leader = &args[1];
    let base = std::path::PathBuf::from(&args[2]);
    let reps: usize = args[3].parse().unwrap();
    let only: Option<&str> = args.get(4).map(|s| s.as_str());
    let cases = [
        ("exec-main-spawn-maskchld", "exec {L} main-spawn-maskchld {F} {C}"),
        ("exec-main-spawn-nocldstop", "exec {L} main-spawn-nocldstop {F} {C}"),
        ("exec-two-block-nocldstop", "exec {L} two-block-nocldstop {F} {C}"),
        ("exec-main-spawn-default", "exec {L} main-spawn {F} {C}"),
        ("exec-thread-spawn", "exec {L} thread-spawn {F} {C}"),
        ("exec-single", "exec {L} single {F} {C}"),
        ("sh-main-spawn-maskchld", "{L} main-spawn-maskchld {F} {C}"),
        ("sh-main-spawn-nocldstop", "{L} main-spawn-nocldstop {F} {C}"),
        ("sh-two-block-nocldstop", "{L} two-block-nocldstop {F} {C}"),
        ("sh-main-spawn-default", "{L} main-spawn {F} {C}"),
        ("exec-busy8-maskchld", "exec {B} {F} {C}"),
        ("sh-busy8-maskchld", "{B} {F} {C}"),
    ];
    let busy = std::path::Path::new(leader).with_file_name("busyleader");
    let mut out = std::io::stdout();
    for (case, tmpl) in cases {
        if only.is_some_and(|o| !case.starts_with(o)) {
            continue;
        }
        for rep in 0..reps {
            let dir = base.join(format!("{case}-{rep}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let fifo = dir.join("fifo");
            let ctr = dir.join("ctr");
            let cfifo = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(cfifo.as_ptr(), 0o600) }, 0);
            let cmd = tmpl
                .replace("{L}", leader)
                .replace("{B}", busy.to_str().unwrap())
                .replace("{F}", fifo.to_str().unwrap())
                .replace("{C}", ctr.to_str().unwrap());
            let mut c = std::process::Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .process_group(0)
                .spawn()
                .unwrap();
            let pid = c.id();
            sleep_ms(300);
            let leader_comm = stat_fields(pid).map(|f| f.0).unwrap_or_default();
            let cpid = procs()
                .into_iter()
                .find(|&p| stat_fields(p).is_some_and(|f| (f.0 == "mtleader" || f.0 == "busyleader") && f.3 == pid as i32))
                .unwrap_or(0);
            let before = threads(cpid);
            let ctr_stop = read_ctr(&ctr);
            let t0 = Instant::now();
            assert_eq!(unsafe { libc::kill(-(pid as i32), libc::SIGSTOP) }, 0);
            let mut ack_ms: Option<u128> = None;
            let mut err = String::new();
            while t0.elapsed() < Duration::from_millis(3000) {
                sleep_ms(25);
                match child_stop_acknowledged(pid) {
                    Ok(true) => {
                        ack_ms = Some(t0.elapsed().as_millis());
                        break;
                    }
                    Ok(false) => {}
                    Err(e) => {
                        err = e.to_string();
                        break;
                    }
                }
            }
            let ctr_ack = read_ctr(&ctr);
            let at_ack = threads(cpid);
            sleep_ms(2000);
            let ctr_late = read_ctr(&ctr);
            let late = threads(cpid);
            let leader_late = threads(pid);
            unsafe { libc::kill(-(pid as i32), libc::SIGCONT) };
            sleep_ms(200);
            let ctr_resumed = read_ctr(&ctr);
            let resumed = threads(cpid);
            // Release the FIFO child, then wait for the job to finish.
            let mut released = false;
            for _ in 0..100 {
                let fd = unsafe { libc::open(cfifo.as_ptr(), libc::O_WRONLY | libc::O_NONBLOCK) };
                if fd >= 0 {
                    unsafe { libc::close(fd) };
                    released = true;
                    break;
                }
                sleep_ms(10);
            }
            let mut exit = String::from("timeout");
            let t1 = Instant::now();
            while t1.elapsed() < Duration::from_millis(2000) {
                if let Ok(Some(st)) = c.try_wait() {
                    exit = format!("{st}");
                    break;
                }
                sleep_ms(10);
            }
            // Children of the C program outside the group (two-block's sleeper).
            for p in procs() {
                if cpid != 0 && stat_fields(p).is_some_and(|f| f.2 == cpid) {
                    unsafe { libc::kill(p as i32, libc::SIGKILL) };
                }
            }
            unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
            let _ = c.wait();
            writeln!(
                out,
                "{{\"case\":\"{case}\",\"rep\":{rep},\"leader_comm\":\"{leader_comm}\",\"leader_pid\":{pid},\"cprog_pid\":{cpid},\
                 \"ack_ms\":{},\"err\":\"{err}\",\"ctr_stop\":{ctr_stop},\"ctr_ack\":{ctr_ack},\"ctr_late\":{ctr_late},\
                 \"ctr_resumed\":{ctr_resumed},\"released\":{released},\"exit\":\"{exit}\",\
                 \"cprog_before\":{before:?},\"cprog_at_ack\":{at_ack:?},\"cprog_late\":{late:?},\"leader_late\":{leader_late:?},\"cprog_resumed\":{resumed:?}}}",
                ack_ms.map(|v| v.to_string()).unwrap_or("null".into()),
            )
            .unwrap();
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
