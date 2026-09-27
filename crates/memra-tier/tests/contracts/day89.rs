//! Day 89 (I23, `research/spill-c-20260919/DAY89.md`): a seeded trace against one `tier::Governor` (reserves,
//! queue work, expiries, marks, pins, releases foreign, double, pinned and unretired among them, cancels and dirty
//! bytes, over five tenants and every priority) and against a bare `LeaseIssuer` (the owner proxy's kind). Every
//! outcome, local lease number and `used()` goes into a transcript whose SHA-256 and line count were recorded at the
//! pre-change tree (crates equal to `0155bc69f`) before any I23 line; I23 must reproduce it exactly.
//! `day89_record` (ignored) prints the digest and writes the transcript under the test target's temporary dir.
use memra_tier::contracts::*;
use memra_tier::tier::{Governor, QueueOutcome};
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const GOVERNOR_SHA256: &str = "eb44719384934a6866922535bee3a19316650d7cfa981dfe4a8abf435b5611b7";
const GOVERNOR_LINES: usize = 9025;
const ISSUER_SHA256: &str = "ab87c2501ed9ad0699d712752cd6c34eeb9b354d58a4b74a229be3f7735702ac";
const ISSUER_LINES: usize = 3000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick(&mut self, n: usize) -> usize {
        self.below(n as u64) as usize
    }
}

const PRIORITIES: [Priority; 5] = [
    Priority::MandatoryActive,
    Priority::AdmittedRestore,
    Priority::Demand,
    Priority::OptionalPrefetch,
    Priority::Backup,
];
const STATES: [ChargeState; 5] = [
    ChargeState::Reserved,
    ChargeState::InUse,
    ChargeState::Quarantined,
    ChargeState::Retired,
    ChargeState::Released,
];

fn budget(pageable: u64, staging: u64, inflight: u64) -> TierBudget {
    let mut b = TierBudget::zero(1);
    b.pageable = pageable;
    b.staging = staging;
    b.inflight = inflight;
    b
}

fn used(g: &Governor) -> String {
    let u = g.used();
    format!("p{} s{} i{}", u.pageable, u.staging, u.inflight)
}

fn outcome<T>(r: &Result<T>, show: impl Fn(&T) -> String) -> String {
    match r {
        Ok(v) => format!("ok {}", show(v)),
        Err(e) => format!("err {e:?}"),
    }
}

/// The governor's transcript for `seed`: one line per step.
fn governor_transcript(seed: u64, steps: usize) -> String {
    let now = Arc::new(AtomicU64::new(1_000));
    let clock = {
        let now = now.clone();
        Arc::new(move || now.load(Ordering::Relaxed))
    };
    let mut g = Governor::new(
        budget(1_000, 120, 8),
        budget(150, 0, 0),
        3,
        500,
        clock.clone(),
    )
    .unwrap();
    let mut foreign =
        Governor::new(budget(1_000, 120, 8), TierBudget::zero(1), 3, 0, clock).unwrap();
    let tenants: Vec<Digest> = (1..=5u8).map(|t| [t; 32]).collect();
    let mut rng = Rng(seed);
    let mut live: Vec<ChargedLease> = Vec::new();
    let mut released: Vec<ChargedLease> = Vec::new();
    let mut foreign_live: Vec<ChargedLease> = Vec::new();
    let mut pins: Vec<LeasePin> = Vec::new();
    let mut queued: Vec<u64> = Vec::new();
    let mut out = String::new();
    for step in 0..steps {
        let request = |rng: &mut Rng| BudgetRequest {
            bytes: budget(rng.below(260), rng.below(40), rng.below(3)),
            priority: PRIORITIES[rng.pick(PRIORITIES.len())],
            deadline: Deadline(if rng.below(10) == 0 {
                now.load(Ordering::Relaxed)
            } else {
                now.load(Ordering::Relaxed) + 1 + rng.below(300)
            }),
            tenant: tenants[rng.pick(tenants.len())],
        };
        let op = rng.below(14);
        let line = match op {
            0..=2 => {
                let r = g.reserve(&request(&mut rng));
                let line = outcome(&r, |l| format!("lease {}", l.id().1));
                if let Ok(l) = r {
                    live.push(l);
                }
                format!("reserve {line}")
            }
            3 => {
                let r = g.enqueue(request(&mut rng));
                let line = outcome(&r, |id| format!("queued {id}"));
                if let Ok(id) = r {
                    queued.push(id);
                }
                format!("enqueue {line}")
            }
            4 => {
                let r = g.dispatch();
                let line = match &r {
                    Ok(Some(QueueOutcome::Admitted(id, l))) => {
                        format!("admitted {id} lease {}", l.id().1)
                    }
                    Ok(Some(QueueOutcome::Expired(id))) => format!("expired {id}"),
                    Ok(None) => "none".to_owned(),
                    Err(e) => format!("err {e:?}"),
                };
                match r {
                    Ok(Some(QueueOutcome::Admitted(id, l))) => {
                        queued.retain(|q| *q != id);
                        live.push(l);
                    }
                    Ok(Some(QueueOutcome::Expired(id))) => queued.retain(|q| *q != id),
                    _ => {}
                }
                format!("dispatch {line}")
            }
            5 => {
                let id = if !queued.is_empty() && rng.below(4) != 0 {
                    queued[rng.pick(queued.len())]
                } else {
                    rng.below(64)
                };
                let r = g.cancel_queued(id);
                if r.is_ok() {
                    queued.retain(|q| *q != id);
                }
                format!("cancel {id} {}", outcome(&r, |_| String::new()))
            }
            6 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let s = STATES[rng.pick(STATES.len())];
                let r = g.mark(&live[i], s);
                format!(
                    "mark {} {s:?} {}",
                    live[i].id().1,
                    outcome(&r, |_| String::new())
                )
            }
            7 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let r = live[i].pin();
                let line = format!("pin {} {}", live[i].id().1, outcome(&r, |_| String::new()));
                if let Ok(p) = r {
                    pins.push(p);
                }
                line
            }
            8 if !pins.is_empty() => {
                let i = rng.pick(pins.len());
                drop(pins.swap_remove(i));
                format!("unpin {i}")
            }
            9 | 10 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let r = g.release(&live[i]);
                let line = format!(
                    "release {} {}",
                    live[i].id().1,
                    outcome(&r, |_| String::new())
                );
                if r.is_ok() {
                    released.push(live.swap_remove(i));
                }
                line
            }
            11 if !released.is_empty() => {
                let i = rng.pick(released.len());
                let r = g.release(&released[i]);
                let m = g.mark(&released[i], ChargeState::Retired);
                let p = released[i].pin();
                format!(
                    "double {} {} mark {} pin {}",
                    released[i].id().1,
                    outcome(&r, |_| String::new()),
                    outcome(&m, |_| String::new()),
                    outcome(&p, |_| String::new())
                )
            }
            12 => {
                if foreign_live.is_empty() || rng.below(3) == 0 {
                    let r = foreign.reserve(&request(&mut rng));
                    let line = outcome(&r, |l| format!("lease {}", l.id().1));
                    if let Ok(l) = r {
                        foreign_live.push(l);
                    }
                    format!("foreign-reserve {line}")
                } else {
                    let i = rng.pick(foreign_live.len());
                    let r = g.release(&foreign_live[i]);
                    let m = g.mark(&foreign_live[i], ChargeState::Retired);
                    format!(
                        "foreign-release {} mark {}",
                        outcome(&r, |_| String::new()),
                        outcome(&m, |_| String::new())
                    )
                }
            }
            _ => {
                let t = rng.pick(tenants.len());
                let bytes = rng.below(300);
                let r = g.set_dirty(tenants[t], bytes);
                format!("dirty {t} {bytes} {}", outcome(&r, |_| String::new()))
            }
        };
        now.fetch_add(rng.below(25), Ordering::Relaxed);
        writeln!(out, "{step} {line} | {}", used(&g)).unwrap();
    }
    // The drain: every pin dropped, every live charge retired and released, the queue cancelled.
    pins.clear();
    for l in &live {
        let m = g.mark(l, ChargeState::Retired);
        let r = g.release(l);
        writeln!(
            out,
            "drain {} mark {} release {} | {}",
            l.id().1,
            outcome(&m, |_| String::new()),
            outcome(&r, |_| String::new()),
            used(&g)
        )
        .unwrap();
    }
    for id in queued {
        writeln!(
            out,
            "drain-cancel {id} {}",
            outcome(&g.cancel_queued(id), |_| String::new())
        )
        .unwrap();
    }
    writeln!(out, "end | {}", used(&g)).unwrap();
    out
}

/// A bare issuer's transcript (the owner proxy's kind: zero budgets, no governor).
fn issuer_transcript(seed: u64, steps: usize) -> String {
    let mut issuer = LeaseIssuer::default();
    let mut other = LeaseIssuer::default();
    let alien = other.issue(TierBudget::zero(1)).unwrap();
    let mut rng = Rng(seed);
    let mut live: Vec<ChargedLease> = Vec::new();
    let mut released: Vec<ChargedLease> = Vec::new();
    let mut pins: Vec<LeasePin> = Vec::new();
    let mut out = String::new();
    for step in 0..steps {
        let op = rng.below(8);
        let line = match op {
            0 | 1 => {
                let r = issuer.issue(budget(rng.below(50), 0, 0));
                let line = outcome(&r, |l| format!("lease {}", l.id().1));
                if let Ok(l) = r {
                    live.push(l);
                }
                format!("issue {line}")
            }
            2 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let s = STATES[rng.pick(STATES.len())];
                format!(
                    "mark {} {s:?} {}",
                    live[i].id().1,
                    outcome(&issuer.mark(&live[i], s), |_| String::new())
                )
            }
            3 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let r = live[i].pin();
                let line = format!("pin {} {}", live[i].id().1, outcome(&r, |_| String::new()));
                if let Ok(p) = r {
                    pins.push(p);
                }
                line
            }
            4 if !pins.is_empty() => {
                let i = rng.pick(pins.len());
                drop(pins.swap_remove(i));
                format!("unpin {i}")
            }
            5 if !live.is_empty() => {
                let i = rng.pick(live.len());
                let r = issuer.release(&live[i]);
                let line = format!(
                    "release {} {} state {}",
                    live[i].id().1,
                    outcome(&r, |_| String::new()),
                    outcome(&live[i].state(), |s| format!("{s:?}"))
                );
                if r.is_ok() {
                    released.push(live.swap_remove(i));
                }
                line
            }
            6 if !released.is_empty() => {
                let i = rng.pick(released.len());
                format!(
                    "double {} release {} mark {} state {}",
                    released[i].id().1,
                    outcome(&issuer.release(&released[i]), |_| String::new()),
                    outcome(&issuer.mark(&released[i], ChargeState::Retired), |_| {
                        String::new()
                    }),
                    outcome(&released[i].state(), |s| format!("{s:?}"))
                )
            }
            _ => format!(
                "alien release {} mark {}",
                outcome(&issuer.release(&alien), |_| String::new()),
                outcome(&issuer.mark(&alien, ChargeState::Retired), |_| String::new(
                ))
            ),
        };
        writeln!(out, "{step} {line}").unwrap();
    }
    out
}

fn digest(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn transcripts() -> [(&'static str, String); 2] {
    let mut g = String::new();
    for seed in [0x0089_0001u64, 0x0089_0002, 0x0089_0003] {
        g.push_str(&governor_transcript(seed, 3_000));
    }
    [
        ("governor", g),
        ("issuer", issuer_transcript(0x0089_00aa, 3_000)),
    ]
}

#[test]
fn day89_governor_and_issuer_match_the_pre_change_fixture() {
    let [(_, g), (_, i)] = transcripts();
    assert_eq!(
        (digest(&g).as_str(), g.lines().count()),
        (GOVERNOR_SHA256, GOVERNOR_LINES)
    );
    assert_eq!(
        (digest(&i).as_str(), i.lines().count()),
        (ISSUER_SHA256, ISSUER_LINES)
    );
}

#[test]
#[ignore = "records the fixture; run at the pre-change tree only"]
fn day89_record() {
    for (name, t) in transcripts() {
        let path =
            std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("day89-{name}.txt"));
        std::fs::write(&path, &t).unwrap();
        println!(
            "DAY89 FIXTURE {name} sha256={} lines={} path={}",
            digest(&t),
            t.lines().count(),
            path.display()
        );
    }
}
