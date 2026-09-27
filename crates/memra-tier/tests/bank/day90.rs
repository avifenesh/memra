//! Day 90 (I24, `research/spill-c-20260919/DAY90.md`): a seeded trace through the owner proxy over a six-record bank
//! with three host slots (so fills evict and `collect_evicted` has candidates): single and grouped demands, finishes,
//! failed finishes and their retries, byte reads (single, by index, each), a foreign owner's finishes, residency reads
//! and closes. After every bank call the bank's state (cached records, owned leases, open ticket sequences, the
//! governor's charge) joins the transcript; its SHA-256 and line count were recorded at I23 before any I24 line, and
//! I24 must reproduce it exactly. `day90_record` (ignored) prints the digest and writes the transcript.
use super::day64::group_bank;
use super::*;
use sha2::{Digest as _, Sha256};
use std::cell::Cell;
use std::fmt::Write as _;

const TRANSCRIPT_SHA256: &str = "dc8b0bdc2d664e050cf5e9a7117b98afc490c2b026f36d703e07426b639a6bdc";
const TRANSCRIPT_LINES: usize = 11996;

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

/// The bank behind the proxy, logging its state after every call and failing the next finish on request.
struct Probe {
    bank: SlruExpertDispatch<Heat, Reader>,
    g: Rc<RefCell<Governor>>,
    fail: Rc<Cell<bool>>,
    log: Rc<RefCell<String>>,
}
impl Probe {
    fn state(&self, call: &str) {
        let b = self.bank.bank();
        let mut tickets: Vec<u64> = b.tickets().iter().map(|t| t.sequence).collect();
        tickets.sort_unstable();
        let u = &self.g.borrow().used;
        writeln!(
            self.log.borrow_mut(),
            "  {call}: cached={} owned={} tickets={tickets:?} used=p{} s{} i{}",
            b.cached_records(),
            b.owned_leases(),
            u.pageable,
            u.staging,
            u.inflight
        )
        .unwrap();
    }
}
impl ExpertDispatchBank for Probe {
    fn validate(&self, local: ExpertDispatchId, bytes: usize) -> Result<()> {
        self.bank.validate(local, bytes)
    }
    fn demand(&mut self, local: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
        let r = self.bank.demand(local, bytes);
        self.state("demand");
        r
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        if self.fail.take() {
            self.state("finish-injected");
            return Err(Error::NotReady);
        }
        let r = self.bank.finish(demand);
        self.state("finish");
        r
    }
    fn host_resident(&self, local: ExpertDispatchId) -> Result<bool> {
        self.bank.host_resident(local)
    }
    fn demand_many(&mut self, blocks: &[(ExpertDispatchId, usize)]) -> Result<ExpertDemands> {
        let r = self.bank.demand_many(blocks);
        self.state("demand_many");
        r
    }
    fn finish_many(&mut self, demands: ExpertDemands) -> Result<()> {
        if self.fail.take() {
            self.state("finish_many-injected");
            return Err(Error::NotReady);
        }
        let r = self.bank.finish_many(demands);
        self.state("finish_many");
        r
    }
}

type Probed = (
    Probe,
    Vec<ExpertDispatchId>,
    Rc<Cell<bool>>,
    Rc<RefCell<String>>,
);

fn probed() -> Probed {
    let g = gov();
    let mut entries = Vec::new();
    let mut map = BTreeMap::new();
    let mut locals = Vec::new();
    for n in [9u32, 21, 47, 60, 83, 99] {
        let mut l = layout(u64::from(n), 4, 16);
        l.segments.truncate(1);
        l.requirements.truncate(1);
        let id = bank_id(n, &l);
        let local = dispatch_id(&id.record).unwrap();
        map.insert(local, id.clone());
        locals.push(local);
        entries.push((id, Some(record(l))));
    }
    let bank = BankService::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        BankLimits {
            cache_bytes: 48,
            batch_bytes: 64,
            items: MAX_GROUP,
            tickets: 4,
        },
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(3).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 3)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let fail = Rc::new(Cell::new(false));
    let log = Rc::new(RefCell::new(String::new()));
    (
        Probe {
            bank: SlruExpertDispatch::new(bank, map, req, epochs()).unwrap(),
            g,
            fail: fail.clone(),
            log: log.clone(),
        },
        locals,
        fail,
        log,
    )
}

fn outcome<T>(r: &Result<T>, show: impl Fn(&T) -> String) -> String {
    match r {
        Ok(v) => format!("ok {}", show(v)),
        Err(e) => format!("err {e:?}"),
    }
}
fn unit(_: &()) -> String {
    String::new()
}

fn transcript(seed: u64, steps: usize) -> String {
    let (probe, locals, fail, log) = probed();
    let mut owner = ExpertBankOwner::register(Box::new(probe), 3).unwrap();
    let proxy = owner.proxy();
    let (other_bank, other_blocks) = group_bank();
    let mut other = ExpertBankOwner::register(Box::new(other_bank), 1).unwrap();
    let foreign = other.proxy();
    let foreign_single = foreign
        .demand(other_blocks[0].0, other_blocks[0].1)
        .unwrap();
    let mut rng = Rng(seed);
    let mut singles: Vec<ExpertLeaseToken> = Vec::new();
    let mut groups: Vec<ExpertGroupToken> = Vec::new();
    let mut done_singles: Vec<ExpertLeaseToken> = Vec::new();
    let mut done_groups: Vec<ExpertGroupToken> = Vec::new();
    for step in 0..steps {
        let op = rng.below(13);
        let line = match op {
            0 | 1 => {
                let local = if rng.below(12) == 0 {
                    (2, 0, 5)
                } else {
                    locals[rng.pick(locals.len())]
                };
                let bytes = if rng.below(12) == 0 { 8 } else { 16 };
                let r = proxy.demand(local, bytes);
                let line = format!(
                    "demand {local:?} {bytes} {}",
                    outcome(&r, |_| String::new())
                );
                if let Ok(t) = r {
                    singles.push(t);
                }
                line
            }
            2 | 3 => {
                let over = usize::from(rng.below(10) == 0);
                let n = 1 + rng.pick(MAX_GROUP + over);
                let mut blocks = Vec::new();
                while blocks.len() < n {
                    let local = locals[rng.pick(locals.len())];
                    if rng.below(15) == 0 || !blocks.iter().any(|&(l, _)| l == local) {
                        blocks.push((local, 16));
                    }
                }
                let r = proxy.demand_many(&blocks);
                let ids: Vec<u16> = blocks.iter().map(|b| b.0.2).collect();
                let line = format!("demand_many {ids:?} {}", outcome(&r, |_| String::new()));
                if let Ok(t) = r {
                    groups.push(t);
                }
                line
            }
            4 if !singles.is_empty() => {
                let i = rng.pick(singles.len());
                let injected = rng.below(4) == 0;
                fail.set(injected);
                let r = proxy.finish(&singles[i]);
                fail.set(false);
                let line = format!("finish {i} injected={injected} {}", outcome(&r, unit));
                if r.is_ok() {
                    done_singles.push(singles.swap_remove(i));
                }
                line
            }
            5 if !groups.is_empty() => {
                let i = rng.pick(groups.len());
                let injected = rng.below(4) == 0;
                fail.set(injected);
                let r = proxy.finish_group(&groups[i]);
                fail.set(false);
                let line = format!("finish_group {i} injected={injected} {}", outcome(&r, unit));
                if r.is_ok() {
                    done_groups.push(groups.swap_remove(i));
                }
                line
            }
            6 if !singles.is_empty() => {
                let i = rng.pick(singles.len());
                let r = proxy.with_bytes(&singles[i], |b| {
                    b.iter().map(|&x| u64::from(x)).sum::<u64>()
                });
                format!("with_bytes {i} {}", outcome(&r, |s| s.to_string()))
            }
            7 if !groups.is_empty() => {
                let i = rng.pick(groups.len());
                let index = rng.pick(MAX_GROUP + 1);
                let r = proxy.with_bytes_at(&groups[i], index, |b| b.len());
                format!(
                    "with_bytes_at {i} {index} {}",
                    outcome(&r, |n| n.to_string())
                )
            }
            8 if !groups.is_empty() => {
                let i = rng.pick(groups.len());
                let mut seen = Vec::new();
                let r = proxy.with_bytes_each(&groups[i], |k, b| {
                    seen.push((k, b.len()));
                    Ok::<(), ()>(())
                });
                format!(
                    "with_bytes_each {i} {} {seen:?}",
                    outcome(&r, |v| format!("{v:?}"))
                )
            }
            9 if !done_singles.is_empty() || !done_groups.is_empty() => {
                if !done_singles.is_empty() && (done_groups.is_empty() || rng.below(2) == 0) {
                    let i = rng.pick(done_singles.len());
                    let f = proxy.finish(&done_singles[i]);
                    let b = proxy.with_bytes(&done_singles[i], |b| b.len());
                    format!(
                        "stale-single {} bytes {}",
                        outcome(&f, unit),
                        outcome(&b, |n| n.to_string())
                    )
                } else {
                    let i = rng.pick(done_groups.len());
                    let f = proxy.finish_group(&done_groups[i]);
                    let b = proxy.with_bytes_at(&done_groups[i], 0, |b| b.len());
                    format!(
                        "stale-group {} bytes {}",
                        outcome(&f, unit),
                        outcome(&b, |n| n.to_string())
                    )
                }
            }
            10 => {
                let mine = singles.first().map(|t| outcome(&foreign.finish(t), unit));
                let theirs = outcome(&proxy.finish(&foreign_single), unit);
                let group = groups
                    .first()
                    .map(|t| outcome(&foreign.finish_group(t), unit));
                format!("foreign mine={mine:?} theirs={theirs} group={group:?}")
            }
            11 => {
                let local = locals[rng.pick(locals.len())];
                let one = proxy.host_resident(local);
                let many = proxy.host_resident_many(&locals[..MAX_GROUP]);
                format!(
                    "resident {local:?} {} many {}",
                    outcome(&one, |b| b.to_string()),
                    outcome(&many, |m| format!("{m:?}"))
                )
            }
            12 if !singles.is_empty() || !groups.is_empty() || rng.below(50) == 0 => {
                format!("close {}", outcome(&owner.close(), unit))
            }
            _ => "idle".to_owned(),
        };
        if line.starts_with("close ok") {
            // A closed owner's registry entry is gone; the trace ends here with its state.
            writeln!(log.borrow_mut(), "{step} {line}").unwrap();
            break;
        }
        writeln!(log.borrow_mut(), "{step} {line}").unwrap();
    }
    for t in singles.drain(..) {
        let r = proxy.finish(&t);
        writeln!(log.borrow_mut(), "drain finish {}", outcome(&r, unit)).unwrap();
    }
    for t in groups.drain(..) {
        let r = proxy.finish_group(&t);
        writeln!(log.borrow_mut(), "drain finish_group {}", outcome(&r, unit)).unwrap();
    }
    let closed = owner.close();
    writeln!(log.borrow_mut(), "end close {}", outcome(&closed, unit)).unwrap();
    foreign.finish(&foreign_single).unwrap();
    other.close().unwrap();
    log.borrow().clone()
}

fn transcripts() -> String {
    let mut all = String::new();
    for seed in [0x0090_0001u64, 0x0090_0002, 0x0090_0003, 0x0090_0004] {
        writeln!(all, "seed {seed:#x}").unwrap();
        all.push_str(&transcript(seed, 2_500));
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
fn day90_proxy_and_bank_match_the_i23_fixture() {
    let t = transcripts();
    assert_eq!(
        (digest(&t).as_str(), t.lines().count()),
        (TRANSCRIPT_SHA256, TRANSCRIPT_LINES)
    );
}

#[test]
#[ignore = "records the fixture; run at I23 only"]
fn day90_record() {
    let t = transcripts();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("day90-proxy.txt");
    std::fs::write(&path, &t).unwrap();
    println!(
        "DAY90 FIXTURE proxy sha256={} lines={} path={}",
        digest(&t),
        t.lines().count(),
        path.display()
    );
}
