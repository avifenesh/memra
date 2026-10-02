#!/usr/bin/env python3
"""Exact-source, standard-library-only Coalescer contracts and decisive mutants."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path('crates/memra-server/src/dsv4_serve.rs'))
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
s = a.source.read_text()
begin = 'struct Coalescer<S, A = bool> {'
end = '/// What a row asks of a B-row step besides its device argmax.'
assert s.count(begin) == s.count(end) == 1
core = s[s.index(begin):s.index(end)]
for marker in ('if serial && g.in_flight > 0 {', 'self.window.max(g.last_run / 10)', 't0.max(g.published)'):
    assert core.count(marker) == 1, marker
binding = {'source': str(a.source), 'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(), 'core_sha256': hashlib.sha256(core.encode()).hexdigest(), 'begin_marker': begin, 'end_marker': end, 'cpu_affinity': min(os.sched_getaffinity(0)), 'production_edits': False}
(a.out/'binding.json').write_text(json.dumps(binding, indent=2))
common = r'''
#[derive(Clone, Copy)] enum LanePhase { Coalesce }
static DEPOSITS: real_std::sync::Mutex<Option<real_std::sync::mpsc::Sender<()>>> = real_std::sync::Mutex::new(None);
fn lane_phase(_: LanePhase) {
    if let Some(tx) = DEPOSITS.lock().unwrap().as_ref() { let _ = tx.send(()); }
}
'''
real_tests = r'''
#[cfg(test)] mod contracts {
    use super::*;
    use real_std::sync::{Arc, Mutex, mpsc};
    use real_std::time::Duration;
    fn exercise(groups: usize) {
        let core = Arc::new(Coalescer::<u32>::with_window(2, groups, Duration::from_secs(10)));
        for _ in 0..4 { core.join(); }
        let (deposit_tx, deposit_rx) = mpsc::channel();
        *DEPOSITS.lock().unwrap() = Some(deposit_tx);
        let (start_tx, start_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let widths = Arc::new(Mutex::new(Vec::new()));
        let spawn = |lane: u32| {
            let (core, release_rx, start_tx, widths) = (core.clone(), release_rx.clone(), start_tx.clone(), widths.clone());
            real_std::thread::spawn(move || {
                let mut state = 0;
                let result = core.step(lane, false, &mut state, &mut |toks, _, states| {
                    widths.lock().unwrap().push(toks.len());
                    start_tx.send(()).unwrap();
                    release_rx.lock().unwrap().recv_timeout(Duration::from_secs(5)).expect("callback release deadline");
                    for state in states { **state += 1; }
                    Ok(toks.iter().map(|&tok| RowOut { tok: tok + 7, logits: None }).collect())
                });
                (result, state)
            })
        };
        let mut handles = vec![spawn(0), spawn(1)];
        for _ in 0..2 { deposit_rx.recv_timeout(Duration::from_secs(3)).expect("first deposits"); }
        start_rx.recv_timeout(Duration::from_secs(3)).expect("first full batch");
        handles.push(spawn(2)); handles.push(spawn(3));
        for _ in 0..2 { deposit_rx.recv_timeout(Duration::from_secs(3)).expect("pending deposits"); }
        // The deposit hook runs while the core lock is held. Reacquiring it observes
        // the depositor's completed decision, not a scheduler wake-up guess.
        let snapshot = { let g = core.lock(); (g.members, g.in_flight, g.waiting.len()) };
        release_tx.send(()).unwrap(); release_tx.send(()).unwrap();
        for (lane, handle) in handles.into_iter().enumerate() {
            let (result, count) = handle.join().unwrap();
            assert_eq!(result, Ok(RowOut { tok: lane as u32 + 7, logits: None }));
            assert_eq!(count, 1, "each row runs exactly once");
        }
        assert_eq!(*widths.lock().unwrap(), vec![2, 2]);
        { let g = core.lock(); assert_eq!(g.in_flight, 0); assert!(g.waiting.is_empty() && g.done.is_empty()); }
        for _ in 0..4 { core.leave(); }
        *DEPOSITS.lock().unwrap() = None;
        println!("groups={groups} snapshot={snapshot:?} widths=[2,2] rows=4 once=true");
        assert_eq!(snapshot, if groups == 1 { (4, 2, 2) } else { (4, 4, 0) });
    }
    #[test] fn single_workspace_retains_pending_members() { exercise(1); }
    #[test] fn multiple_workspace_control() { exercise(2); }
}
'''
virtual_std = r'''
mod std {
    pub use crate::real_std::*;
    pub mod time {
        pub use crate::real_std::time::Duration;
        static NOW: crate::real_std::sync::atomic::AtomicU64 = crate::real_std::sync::atomic::AtomicU64::new(0);
        use crate::real_std::sync::atomic::Ordering::SeqCst;
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)] pub struct Instant(u64);
        impl Instant {
            pub fn now() -> Self { Self(NOW.load(SeqCst)) }
            pub fn elapsed(self) -> Duration { Duration::from_nanos(NOW.load(SeqCst).saturating_sub(self.0)) }
        }
        pub fn set(n: u64) { NOW.store(n, SeqCst); }
        pub fn advance(d: Duration) { NOW.fetch_add(d.as_nanos().try_into().unwrap(), SeqCst); }
        pub fn nanos() -> u64 { NOW.load(SeqCst) }
    }
    pub mod sync {
        pub use crate::real_std::sync::*;
        pub static WAITS: Mutex<Vec<super::time::Duration>> = Mutex::new(Vec::new());
        pub struct Condvar(crate::real_std::sync::Condvar);
        impl Condvar {
            pub fn new() -> Self { Self(crate::real_std::sync::Condvar::new()) }
            pub fn notify_all(&self) { self.0.notify_all(); }
            pub fn wait<'a,T>(&self, g: MutexGuard<'a,T>) -> LockResult<MutexGuard<'a,T>> { self.0.wait(g) }
            pub fn wait_timeout<'a,T>(&self, g: MutexGuard<'a,T>, d: super::time::Duration) -> LockResult<(MutexGuard<'a,T>, WaitTimeoutResult)> {
                WAITS.lock().unwrap().push(d);
                super::time::advance(d);
                self.0.wait_timeout(g, super::time::Duration::ZERO)
            }
        }
    }
}
'''
virtual_tests = r'''
#[cfg(test)] mod contracts {
    use super::*;
    use real_std::sync::{Arc, mpsc};
    use real_std::time::Duration;
    fn reset() { std::time::set(0); std::sync::WAITS.lock().unwrap().clear(); *DEPOSITS.lock().unwrap() = None; }
    fn step(core: &Coalescer<u32>) -> (u32, u64) {
        let mut state = 0;
        let mut at = 0;
        let result = core.step(9, false, &mut state, &mut |toks, _, states| {
            at = std::time::nanos();
            assert_eq!(toks, &[9]);
            for s in states { **s += 1; }
            Ok(vec![RowOut { tok: 16, logits: None }])
        });
        assert_eq!(result, Ok(RowOut { tok: 16, logits: None }));
        assert_eq!(state, 1);
        (state, at)
    }
    #[test] fn serial_adaptive_window() {
        reset();
        let core = Coalescer::<u32>::with_window(4, 1, Duration::from_micros(500));
        for _ in 0..4 { core.join(); }
        core.lock().last_run = Duration::from_secs(2);
        assert_eq!(step(&core).1, 200_000_000);
        assert_eq!(*std::sync::WAITS.lock().unwrap(), vec![Duration::from_millis(200)]);
        println!("serial: wait=200ms, one row once");
    }
    #[test] fn multiple_workspace_uses_base_window() {
        reset();
        let core = Coalescer::<u32>::with_window(4, 2, Duration::from_micros(500));
        for _ in 0..4 { core.join(); }
        core.lock().last_run = Duration::from_secs(2);
        assert_eq!(step(&core).1, 500_000);
        assert_eq!(*std::sync::WAITS.lock().unwrap(), vec![Duration::from_micros(500)]);
        println!("multiple workspaces: wait=500us, one row once");
    }
    #[test] fn publication_restarts_the_wait_origin() {
        reset();
        let core = Arc::new(Coalescer::<u32>::with_window(4, 1, Duration::from_micros(500)));
        for _ in 0..4 { core.join(); }
        core.lock().in_flight = 2;
        let (tx, rx) = mpsc::channel(); *DEPOSITS.lock().unwrap() = Some(tx);
        let worker = core.clone();
        let handle = real_std::thread::spawn(move || step(&worker));
        rx.recv_timeout(Duration::from_secs(3)).expect("deposit deadline");
        { let mut g = core.lock(); assert_eq!(g.waiting.len(), 1); std::time::set(200_000_000); g.in_flight = 0; g.published = std::time::Instant::now(); g.last_run = Duration::from_secs(2); }
        core.cv.notify_all();
        let (_, at) = handle.join().unwrap();
        *DEPOSITS.lock().unwrap() = None;
        assert_eq!(at, 400_000_000);
        assert_eq!(*std::sync::WAITS.lock().unwrap(), vec![Duration::from_millis(200)]);
        println!("publication at 200ms: callback at 400ms, one row once");
    }
}
'''
variants = [
    ('real-green', core, '', real_tests, True),
    ('virtual-green', core, virtual_std, virtual_tests, True),
    ('red-inflight', core.replace('if serial && g.in_flight > 0 {', 'if false && g.in_flight > 0 {'), '', real_tests, False),
    ('red-adaptive', core.replace('self.window.max(g.last_run / 10)', 'self.window'), virtual_std, virtual_tests, False),
    ('red-publication', core.replace('t0.max(g.published)', 't0'), virtual_std, virtual_tests, False),
]
expected_failures = {
    'real-green': set(),
    'virtual-green': set(),
    'red-inflight': {'single_workspace_retains_pending_members'},
    'red-adaptive': {'serial_adaptive_window', 'publication_restarts_the_wait_origin'},
    'red-publication': {'publication_restarts_the_wait_origin'},
}
def pin():
    os.sched_setaffinity(0, {binding['cpu_affinity']})
results = []
for name, implementation, shim, tests, expected in variants:
    rs = a.out / (name + '.rs')
    rs.write_text('#![allow(dead_code)]\nextern crate std as real_std;\nmod contract {\n' + shim + common + implementation + tests + '\n}\n')
    binary = a.out / name
    with (a.out / (name + '-build.log')).open('w') as log:
        build = subprocess.run(['rustc', '--edition=2024', '--test', '-C', 'codegen-units=1', str(rs), '-o', str(binary)], stdout=log, stderr=subprocess.STDOUT, timeout=60, preexec_fn=pin)
    assert build.returncode == 0, f'{name}: compilation failed'
    with (a.out / (name + '.log')).open('w') as log:
        run = subprocess.run([str(binary), '--test-threads=1', '--nocapture'], stdout=log, stderr=subprocess.STDOUT, timeout=15, preexec_fn=pin)
    assert run.returncode == (0 if expected else 101), f'{name}: unexpected verdict {run.returncode}'
    output = (a.out / (name + '.log')).read_text()
    counts = re.findall(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', output)
    assert len(counts) == 1, f'{name}: missing or ambiguous test summary'
    passed, failed, ignored, measured, filtered = map(int, counts[0])
    total = 2 if not shim else 3
    assert (passed + failed, failed, ignored, measured, filtered) == (total, len(expected_failures[name]), 0, 0, 0), f'{name}: wrong executed/failed/skip count'
    failed_names = set(re.findall(r'^    contract::contracts::(\w+)$', output, re.M))
    assert failed_names == expected_failures[name], f'{name}: unrelated failure {failed_names}'
    results.append({'case': name, 'exit_code': run.returncode, 'expected_pass': expected, 'source_sha256': hashlib.sha256(rs.read_bytes()).hexdigest()})
    print(json.dumps(results[-1]), flush=True)
(a.out/'results.json').write_text(json.dumps({'binding': binding, 'results': results, 'pass': True}, indent=2))
