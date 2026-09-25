//! DAY67 (`research/spill-c-20260919/DAY67.md`): a fixed CPU and memory probe for `run-gen --cpu-probe`, log only. It
//! runs on the calling thread after every timed phase and reads how fast that thread's core and memory are in this
//! boot: a register-only compute chain, then dependent loads over an L1-, an L2- and a DRAM-sized cyclic permutation,
//! then the compute chain again. It allocates its arrays itself and frees them before returning.
use std::hint::black_box;
use std::time::Instant;

/// One probe's readings: nanoseconds per step or per load, and the CPU the thread read before and after.
#[derive(Clone, Copy, Debug)]
pub struct CpuProbe {
    pub cpu: i32,
    pub cpu_after: i32,
    pub compute_ns: f64,
    pub l1_ns: f64,
    pub l2_ns: f64,
    pub dram_ns: f64,
    pub compute2_ns: f64,
}

/// The sizes of one probe: compute steps, and per level its bytes and its dependent loads.
#[derive(Clone, Copy, Debug)]
pub struct ProbeSizes {
    pub compute_steps: u64,
    pub l1: (usize, u64),
    pub l2: (usize, u64),
    pub dram: (usize, u64),
}

/// DAY67's registered sizes: 2^26 compute steps; 2^24 loads over 4 KiB, 2^22 over 512 KiB, 2^21 over 256 MiB.
pub const DAY67_SIZES: ProbeSizes = ProbeSizes {
    compute_steps: 1 << 26,
    l1: (4 << 10, 1 << 24),
    l2: (512 << 10, 1 << 22),
    dram: (256 << 20, 1 << 21),
};

fn compute(steps: u64) -> f64 {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let started = Instant::now();
    for _ in 0..steps {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    black_box(x);
    started.elapsed().as_nanos() as f64 / steps as f64
}

/// DAY71: `compute`'s register-only chain without its clock, for `counted_chain`: `steps` xorshift steps, three
/// dependent shift-and-xor pairs each (`compute` itself is kept as DAY67 and DAY68 ran it).
fn chain(steps: u64) -> u64 {
    let mut x: u64 = black_box(0x9E37_79B9_7F4A_7C15);
    for _ in 0..steps {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    x
}

/// A random single-cycle permutation of `n` slots (Sattolo's algorithm, a fixed seed), so a chase visits every slot.
fn cycle(n: usize) -> Vec<u32> {
    let mut next: Vec<u32> = (0..n as u32).collect();
    let mut s: u64 = 0xD1B5_4A32_D192_ED03;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % i as u64) as usize;
        next.swap(i, j);
    }
    next
}

fn chase((bytes, loads): (usize, u64)) -> f64 {
    let next = cycle((bytes / std::mem::size_of::<u32>()).max(2));
    // The array was written in full by `cycle`, so its pages are mapped before the timed loads.
    let mut at = 0u32;
    let started = Instant::now();
    for _ in 0..loads {
        at = next[at as usize];
    }
    black_box(at);
    started.elapsed().as_nanos() as f64 / loads as f64
}

fn current_cpu() -> i32 {
    // SAFETY: `sched_getcpu` takes no argument and reads the calling thread's CPU.
    unsafe { libc::sched_getcpu() }
}

/// Run the probe at `sizes` on the calling thread.
pub fn run(sizes: ProbeSizes) -> CpuProbe {
    let cpu = current_cpu();
    let compute_ns = compute(sizes.compute_steps);
    let l1_ns = chase(sizes.l1);
    let l2_ns = chase(sizes.l2);
    let dram_ns = chase(sizes.dram);
    let compute2_ns = compute(sizes.compute_steps);
    CpuProbe {
        cpu,
        cpu_after: current_cpu(),
        compute_ns,
        l1_ns,
        l2_ns,
        dram_ns,
        compute2_ns,
    }
}

/// DAY68 (`research/spill-c-20260919/DAY68.md`): the phase probe's steps, a 2^20-step compute chain (about 1 ms at
/// 5.7 GHz).
pub const PHASE_STEPS: u64 = 1 << 20;

/// DAY68: one phase point's reading: the CPU the thread read, and nanoseconds per step of a `PHASE_STEPS` chain.
/// DAY71 (`research/spill-c-20260919/DAY71.md`): with `counters`, the core's own counters around the same chain are
/// appended (`counters_fields`).
pub fn phase_line(phase: &str, counters: bool) -> String {
    let cpu = current_cpu();
    if !counters {
        let ns = compute(PHASE_STEPS);
        return format!("phase={phase} cpu={cpu} compute_ns={ns:.3}");
    }
    match counted_chain(PHASE_STEPS) {
        Some(c) => format!(
            "phase={phase} cpu={cpu} compute_ns={:.3} {}",
            c.wall_ns as f64 / PHASE_STEPS as f64,
            c.fields(current_cpu())
        ),
        None => {
            let ns = compute(PHASE_STEPS);
            format!("phase={phase} cpu={cpu} compute_ns={ns:.3} counters=unavailable")
        }
    }
}

/// DAY71: one counted chain's deltas: monotonic-clock nanoseconds, the thread's TSC, and its CPU's MPERF and APERF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counted {
    pub wall_ns: u64,
    pub tsc: u64,
    pub mperf: u64,
    pub aperf: u64,
}

impl Counted {
    /// The fields DAY71 section 1 registers, after the phase line's `compute_ns`.
    pub fn fields(&self, cpu_after: i32) -> String {
        format!(
            "cpu_after={cpu_after} wall_ns={} tsc={} mperf={} aperf={}",
            self.wall_ns, self.tsc, self.mperf, self.aperf
        )
    }
}

/// DAY71: whether this CPU reports `RDPRU` with MPERF (ECX 0) and APERF (ECX 1) readable: CPUID Fn8000_0008 EBX bit 4,
/// and its EDX bits 23:16 (the largest ECX `RDPRU` reads) at least 1.
pub fn counters_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x8000_0000).eax < 0x8000_0008 {
            return false;
        }
        let leaf = __cpuid(0x8000_0008);
        (leaf.ebx >> 4) & 1 == 1 && ((leaf.edx >> 16) & 0xff) >= 1
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `RDPRU` with ECX `which`: 0 reads MPERF, 1 reads APERF, of the CPU the thread runs on.
///
/// # Safety
/// The CPU must report `RDPRU` with `which` readable (`counters_available`), and the OS must allow it at user level
/// (Linux does unless the thread set `PR_SET_TSC` to `PR_TSC_SIGSEGV`); the cell checks that with
/// `--cpu-probe-counters-check` in a separate process first.
#[cfg(target_arch = "x86_64")]
unsafe fn rdpru(which: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: the caller's contract above; the instruction reads two registers and the flags, and touches no memory.
    unsafe {
        std::arch::asm!(
            ".byte 0x0f, 0x01, 0xfd",
            in("ecx") which,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack)
        );
    }
    (u64::from(hi) << 32) | u64::from(lo)
}

/// DAY71: the counters' four readings at one instant, in a fixed order (clock, TSC, MPERF, APERF).
#[cfg(target_arch = "x86_64")]
fn read_counters() -> (Instant, u64, u64, u64) {
    let now = Instant::now();
    // SAFETY: `_rdtsc` reads the time-stamp counter; every x86_64 CPU has it.
    let tsc = unsafe { std::arch::x86_64::_rdtsc() };
    // SAFETY: only called after `counters_available` (see `counted_chain`).
    let (mperf, aperf) = unsafe { (rdpru(0), rdpru(1)) };
    (now, tsc, mperf, aperf)
}

/// DAY71: a `steps`-step compute chain bracketed by the counters, or `None` where the CPU has no `RDPRU`.
pub fn counted_chain(steps: u64) -> Option<Counted> {
    #[cfg(target_arch = "x86_64")]
    {
        if !counters_available() {
            return None;
        }
        let (t0, tsc0, m0, a0) = read_counters();
        // `chain` starts from a `black_box` seed and its result goes through `black_box` before the second reading,
        // so the chain runs between the two readings.
        black_box(chain(steps));
        let (t1, tsc1, m1, a1) = read_counters();
        Some(Counted {
            wall_ns: t1.duration_since(t0).as_nanos() as u64,
            tsc: tsc1.wrapping_sub(tsc0),
            mperf: m1.wrapping_sub(m0),
            aperf: a1.wrapping_sub(a0),
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = steps;
        None
    }
}

/// DAY71: `--cpu-probe-counters-check`'s line: one counted chain, or `counters=unavailable`.
pub fn counters_check_line() -> String {
    let cpu = current_cpu();
    match counted_chain(PHASE_STEPS) {
        Some(c) => format!(
            "counters-check cpu={cpu} rdpru=ok {}",
            c.fields(current_cpu())
        ),
        None => format!("counters-check cpu={cpu} counters=unavailable"),
    }
}

impl CpuProbe {
    /// The `[cpu-probe]` line's fields in their registered order.
    pub fn line(&self) -> String {
        format!(
            "cpu={} cpu_after={} compute_ns={:.3} l1_ns={:.3} l2_ns={:.3} dram_ns={:.3} compute2_ns={:.3}",
            self.cpu,
            self.cpu_after,
            self.compute_ns,
            self.l1_ns,
            self.l2_ns,
            self.dram_ns,
            self.compute2_ns
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permutation is one cycle through every slot, and a probe at reduced sizes reads finite positive values.
    #[test]
    fn the_probe_chases_one_cycle_and_reads_positive_values() {
        let next = cycle(1000);
        let (mut at, mut seen) = (0u32, 0usize);
        loop {
            at = next[at as usize];
            seen += 1;
            if at == 0 {
                break;
            }
        }
        assert_eq!(seen, 1000);
        let p = run(ProbeSizes {
            compute_steps: 1 << 12,
            l1: (4 << 10, 1 << 12),
            l2: (64 << 10, 1 << 12),
            dram: (1 << 20, 1 << 12),
        });
        for v in [p.compute_ns, p.l1_ns, p.l2_ns, p.dram_ns, p.compute2_ns] {
            assert!(v.is_finite() && v > 0.0, "{v}");
        }
        assert!(p.cpu >= 0 && p.cpu_after >= 0);
        assert!(p.line().starts_with("cpu="));
        let phase = phase_line("gate", false);
        assert!(
            phase.starts_with("phase=gate cpu=") && phase.contains(" compute_ns="),
            "{phase}"
        );
    }

    /// DAY71: with counters asked for, the phase line keeps DAY68's leading fields and appends either the registered
    /// counter fields (a CPU with `RDPRU`) or `counters=unavailable`; the check line says which.
    #[test]
    fn the_counted_phase_line_keeps_the_phase_fields_and_names_its_counters() {
        let phase = phase_line("gate", true);
        assert!(
            phase.starts_with("phase=gate cpu=") && phase.contains(" compute_ns="),
            "{phase}"
        );
        let check = counters_check_line();
        if counters_available() {
            for field in [" cpu_after=", " wall_ns=", " tsc=", " mperf=", " aperf="] {
                assert!(phase.contains(field), "{phase}");
            }
            assert!(check.contains(" rdpru=ok "), "{check}");
            let c = counted_chain(1 << 12).expect("counters");
            assert!(c.wall_ns > 0 && c.tsc > 0 && c.aperf > 0, "{c:?}");
        } else {
            assert!(phase.ends_with(" counters=unavailable"), "{phase}");
            assert!(check.ends_with(" counters=unavailable"), "{check}");
            assert!(counted_chain(1 << 12).is_none());
        }
        let c = Counted {
            wall_ns: 1_528_888,
            tsc: 6_571_234,
            mperf: 6_571_000,
            aperf: 8_745_000,
        };
        assert_eq!(
            c.fields(4),
            "cpu_after=4 wall_ns=1528888 tsc=6571234 mperf=6571000 aperf=8745000"
        );
    }

    /// The registered sizes on this host, printed (log only, by hand).
    #[test]
    #[ignore = "DAY67: the full-size probe, run by hand"]
    fn the_registered_probe_on_this_host() {
        let started = Instant::now();
        let p = run(DAY67_SIZES);
        println!(
            "[cpu-probe] {} wall_ms={}",
            p.line(),
            started.elapsed().as_millis()
        );
    }
}
