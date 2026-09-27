//! Day 93 (I26, `research/spill-c-20260919/DAY93.md`): a seeded trace through the dispatch adapter over a bank with
//! fewer host slots than records (hits, misses, evictions, evicted leases held by open demands, fills), singles and
//! grouped demands at both priorities, demands held open to the ticket limit and refused past it. After every
//! operation the transcript records what a caller can observe: each outcome and refusal, the leased records and their
//! first bytes, the SLRU orders and every record's slot, the host cache's records and bytes, the owned leases, the
//! reader's calls and each id's heat; `used()` only when no demand is open. By design (DAY93 section 1) it records
//! neither the ticket inventory nor `used()` while a demand is open. Its SHA-256 and line count were recorded at I25,
//! before any I26 line; I26 must reproduce it exactly. `day93_record` (ignored) prints the digest and writes it.
use super::day85::{bank_governed, first, spread, state};
use super::*;
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;

const TRANSCRIPT_SHA256: &str = "c9a09b37b595b054d5f5683e9f48ba4b66b623b09c5bc44d78f5dc8dcd9d10bc";
const TRANSCRIPT_LINES: usize = 6249;

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
}

enum Open {
    One(ExpertDemand),
    Many(ExpertDemands),
}

fn heat(b: &Banks, ids: &[BankId]) -> Vec<u64> {
    ids.iter()
        .map(|id| b.heat().0.get(id).copied().unwrap_or(0))
        .collect()
}

fn transcript(priority: Priority, seed: u64, steps: usize) -> String {
    let (_, ids) = spread();
    let (positioned, req, g) = bank_governed(5, priority);
    let map = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    let mut d = SlruExpertDispatch::new(positioned, map, req, epochs()).unwrap();
    let mut rng = Rng(seed);
    let mut open: Vec<Open> = Vec::new();
    let mut out = String::new();
    writeln!(
        out,
        "# day93 {priority:?} seed {seed:#x}: observable outcomes only; no ticket inventory, no used() while a demand is open"
    )
    .unwrap();
    for step in 0..steps {
        let hold = rng.below(3) == 0;
        let line = match rng.below(20) {
            0..=9 => {
                let start = 3 * rng.below(ids.len() as u64 / 3) as usize;
                let blocks: Vec<_> = ids[start..start + 3]
                    .iter()
                    .map(|id| (dispatch_id(&id.record).unwrap(), 16))
                    .collect();
                match d.demand_many(&blocks) {
                    Ok(demands) => {
                        let seen: Vec<_> = demands
                            .leases
                            .iter()
                            .map(|l| (dispatch_id(&l.id().record).unwrap(), first(l)))
                            .collect();
                        let fin = if hold {
                            open.push(Open::Many(demands));
                            "held".to_owned()
                        } else {
                            format!("finish {:?}", d.finish_many(&demands))
                        };
                        format!("group {start} ok {seen:?} {fin}")
                    }
                    Err(e) => format!("group {start} err {e:?}"),
                }
            }
            10..=14 => {
                let n = rng.below(ids.len() as u64) as usize;
                let local = dispatch_id(&ids[n].record).unwrap();
                let bytes = if rng.below(15) == 0 { 8 } else { 16 };
                match d.demand(local, bytes) {
                    Ok(demand) => {
                        let seen = (
                            dispatch_id(&demand.lease.id().record).unwrap(),
                            first(&demand.lease),
                        );
                        let fin = if hold {
                            open.push(Open::One(demand));
                            "held".to_owned()
                        } else {
                            format!("finish {:?}", d.finish(&demand))
                        };
                        format!("single {n} {bytes} ok {seen:?} {fin}")
                    }
                    Err(e) => format!("single {n} {bytes} err {e:?}"),
                }
            }
            15..=16 if !open.is_empty() => {
                let i = rng.below(open.len() as u64) as usize;
                let r = match open.swap_remove(i) {
                    Open::One(demand) => d.finish(&demand),
                    Open::Many(demands) => d.finish_many(&demands),
                };
                format!("finish-open {i} {r:?}")
            }
            _ => {
                let n = rng.below(ids.len() as u64) as usize;
                let (entries, _) = spread();
                let s = &entries[n].1.layout.segments[0];
                let bytes: Vec<u8> = (s.offset..s.offset + s.storage_bytes)
                    .map(|p| byte(s.tensor.as_ref().unwrap(), p))
                    .collect();
                let digest = checksum(&bytes);
                let local = dispatch_id(&ids[n].record).unwrap();
                format!(
                    "fill {n} {:?}",
                    d.admit_filled(local, HostBytes::Heap(bytes), digest)
                )
            }
        };
        writeln!(
            out,
            "{step} {line} | {} heat={:?}",
            state(d.bank(), &ids),
            heat(d.bank(), &ids)
        )
        .unwrap();
        if open.is_empty() {
            writeln!(out, "  used {:?}", g.borrow().used).unwrap();
        }
    }
    for o in open.drain(..) {
        let r = match o {
            Open::One(demand) => d.finish(&demand),
            Open::Many(demands) => d.finish_many(&demands),
        };
        writeln!(out, "drain {r:?} | {}", state(d.bank(), &ids)).unwrap();
    }
    writeln!(out, "end used {:?}", g.borrow().used).unwrap();
    out
}

fn transcripts() -> String {
    let mut all = String::new();
    for (priority, seed) in [
        (Priority::Demand, 0x0093_0001u64),
        (Priority::Demand, 0x0093_0002),
        (Priority::OptionalPrefetch, 0x0093_0003),
        (Priority::OptionalPrefetch, 0x0093_0004),
    ] {
        all.push_str(&transcript(priority, seed, 1_500));
    }
    all
}

fn digest(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn day93_adapter_matches_the_i25_fixture() {
    let t = transcripts();
    assert_eq!(
        (digest(&t).as_str(), t.lines().count()),
        (TRANSCRIPT_SHA256, TRANSCRIPT_LINES)
    );
}

#[test]
#[ignore = "records the fixture; run at I25 only"]
fn day93_record() {
    let t = transcripts();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("day93-adapter.txt");
    std::fs::write(&path, &t).unwrap();
    println!(
        "DAY93 FIXTURE adapter sha256={} lines={} path={}",
        digest(&t),
        t.lines().count(),
        path.display()
    );
}
