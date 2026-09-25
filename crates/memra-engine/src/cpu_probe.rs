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
