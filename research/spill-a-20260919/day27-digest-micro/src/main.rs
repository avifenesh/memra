//! WP-A day 27 digest micro-cell (`research/spill-a-20260919/DAY27.md`, task 2 option (b)).
//!
//! Times, over one heap buffer of `--bytes` bytes, the two digest programs the host tier owns today:
//! `memra_tier::contracts::checksum` (SHA-256 with the `valid-bytes` domain frame: the D2H receipt
//! and `bind_tier_image`'s bundle checksum) and `memra_tier::conformance::receipt_digest` (the
//! four-lane CPU oracle of the Move 2 slice-3 D2D receipt kernel). Heap is the memory kind of the
//! bytes that cost the demote tick (the recurrent f32 planes land in `Vec<f32>` under the door, see
//! DAY27 section 1); C's day-18 `hash-micro` read heap equal to cached pinned within 0.4 percent on
//! both hosts. `--n` passes per program per order, two orders (sha first, then lanes first),
//! interleaved call by call; one rule line. No CUDA, no env read, no engine code.
use memra_tier::conformance::receipt_digest;
use memra_tier::contracts::checksum;
use std::time::Instant;

fn fill(dst: &mut [u8]) {
    // A fixed LCG stream: deterministic, not compressible, not zero.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for chunk in dst.chunks_mut(8) {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let b = s.to_le_bytes();
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn range(v: &[f64]) -> (f64, f64) {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes: usize = 167_772_160;
    let mut n: usize = 5;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bytes" => bytes = args.next().ok_or("--bytes needs a value")?.parse()?,
            "--n" => n = args.next().ok_or("--n needs a value")?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let host = cpu_model();
    let t = Instant::now();
    let mut buf = vec![0u8; bytes];
    let alloc_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    fill(&mut buf);
    let fill_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "digest-micro host=\"{host}\" bytes={bytes} n_per_order={n} alloc_ms={alloc_ms:.3} fill_ms={fill_ms:.3} programs=sha256(contracts::checksum),lanes(conformance::receipt_digest) memory=heap"
    );
    // Warm both programs once (page-in of the buffer, instruction cache); not counted.
    let sha_ref = checksum(&buf);
    let lanes_ref = receipt_digest(&buf);
    let mut sha = [Vec::new(), Vec::new()];
    let mut lanes = [Vec::new(), Vec::new()];
    let mut sha_stable = true;
    let mut lanes_stable = true;
    for order in 0..2 {
        for pass in 0..n {
            for step in 0..2 {
                let sha_turn = (step == 0) == (order == 0);
                if sha_turn {
                    let t = Instant::now();
                    let d = std::hint::black_box(checksum(&buf));
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    sha_stable &= d == sha_ref;
                    sha[order].push(ms);
                    println!("pass order={} n={} program=sha256 ms={ms:.3}", order + 1, pass + 1);
                } else {
                    let t = Instant::now();
                    let d = std::hint::black_box(receipt_digest(&buf));
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    lanes_stable &= d == lanes_ref;
                    lanes[order].push(ms);
                    println!("pass order={} n={} program=lanes ms={ms:.3}", order + 1, pass + 1);
                }
            }
        }
    }
    let sha_all: Vec<f64> = sha.concat();
    let lanes_all: Vec<f64> = lanes.concat();
    let sha_ms = median(&sha_all);
    let lanes_ms = median(&lanes_all);
    let (slo, shi) = range(&sha_all);
    let (llo, lhi) = range(&lanes_all);
    let gbps = |ms: f64| bytes as f64 / 1e9 / (ms / 1e3);
    println!(
        "DIGEST-MICRO rule host=\"{host}\" bytes={bytes} n_per_order={n} pooled={} orders=2 memory=heap \
         sha_ms={sha_ms:.3} lanes_ms={lanes_ms:.3} sha_range={slo:.3}..{shi:.3} lanes_range={llo:.3}..{lhi:.3} \
         sha_o1={:.3} sha_o2={:.3} lanes_o1={:.3} lanes_o2={:.3} sha_gbps={:.3} lanes_gbps={:.3} \
         lanes_over_sha={:.3} sha_stable={sha_stable} lanes_stable={lanes_stable} sha_digest={} lanes_digest={}",
        2 * n,
        median(&sha[0]),
        median(&sha[1]),
        median(&lanes[0]),
        median(&lanes[1]),
        gbps(sha_ms),
        gbps(lanes_ms),
        lanes_ms / sha_ms,
        hex(&sha_ref),
        hex(&lanes_ref),
    );
    Ok(())
}
