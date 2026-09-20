//! Copy plumbing only: N=1 is NOT a scored G2 sweep or a performance/default claim.
//! Mac: rustc --edition=2024 this-file.rs -o h2d-probe; h2d-probe --dry-run
//! Linux GPU invocation MUST be wrapped by tools/tier-battery.py (300s maximum).
use std::error::Error;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SIZES: [usize; 10] = [
    4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824,
];

#[derive(Debug)]
struct Options {
    dry: bool,
    sizes: Vec<usize>,
    directions: Vec<&'static str>,
    reverse: bool,
    copies: u32,
}
impl Options {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut out = Self {
            dry: false,
            sizes: SIZES.to_vec(),
            directions: vec!["h2d", "d2h"],
            reverse: false,
            copies: 1,
        };
        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--dry-run" => out.dry = true,
                "--bytes" => {
                    let n = args.next().ok_or("--bytes needs an integer")?.parse()?;
                    if !SIZES.contains(&n) {
                        return Err("--bytes must be one of the registered 4KiB..1GiB sizes".into());
                    }
                    out.sizes = vec![n];
                }
                "--direction" => {
                    out.directions = match args.next().as_deref() {
                        Some("h2d") => vec!["h2d"],
                        Some("d2h") => vec!["d2h"],
                        Some("both") => vec!["h2d", "d2h"],
                        _ => return Err("--direction must be h2d|d2h|both".into()),
                    };
                }
                "--order" => match args.next().as_deref() {
                    Some("ab") => out.reverse = false,
                    Some("ba") => out.reverse = true,
                    _ => return Err("--order must be ab|ba".into()),
                },
                // Outer balancing/calibration belongs to the G2 runner. Copies within
                // one visit do not increase its independent observation count (N=1).
                "--repeats" => {
                    if args.next().as_deref() != Some("1") {
                        return Err("probe requires --repeats 1".into());
                    }
                }
                "--copies" => {
                    out.copies = args.next().ok_or("--copies needs an integer")?.parse()?;
                    if !(1..=100_000).contains(&out.copies) {
                        return Err("--copies must be in 1..100000".into());
                    }
                }
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        Ok(out)
    }
}

fn unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

fn pattern(i: usize, seed: u8) -> u8 {
    ((i.wrapping_mul(17).wrapping_add(usize::from(seed))) % 251) as u8
}

fn verify(bytes: &[u8], seed: u8) -> Result<()> {
    if let Some(i) = bytes
        .iter()
        .enumerate()
        .position(|(i, b)| *b != pattern(i, seed))
    {
        return Err(format!("byte mismatch at {i}").into());
    }
    Ok(())
}

#[derive(Default)]
struct Sample {
    unix_start_ns: u128,
    unix_end_ns: u128,
    mono_start_ns: u128,
    mono_end_ns: u128,
    wall_ns: Option<u128>,
    event_ms: Option<f64>,
    setup_ns: Option<u128>,
    verify_ns: Option<u128>,
    expected_sha256: Option<String>,
    actual_sha256: Option<String>,
    power_before: Option<String>,
    power_after: Option<String>,
}

impl Sample {
    fn print(&self, bytes: usize, direction: &str, pinned: bool, opts: &Options) {
        let dry = opts.dry;
        let copies = opts.copies;
        let order = if opts.reverse { "ba" } else { "ab" };
        let num = |x: Option<u128>| x.map_or("null".into(), |x| x.to_string());
        let string = |x: &Option<String>| x.as_ref().map_or("null".into(), |x| format!("{x:?}"));
        println!(
            "{{\"schema_version\":1,\"record\":\"sample\",\"evidence_class\":{:?},\"bytes\":{bytes},\"direction\":{direction:?},\"arm\":{:?},\"pinned_flags\":{},\"n\":1,\"copies\":{copies},\"order\":{order:?},\"event_timing\":\"sum-per-operation-owner-stream\",\"completed_bytes\":{},\"verified_bytes\":{},\"unix_start_ns\":{},\"unix_end_ns\":{},\"mono_start_ns\":{},\"mono_end_ns\":{},\"wall_ns\":{},\"event_ms\":{},\"setup_ns\":{},\"verify_ns\":{},\"expected_sha256\":{},\"actual_sha256\":{},\"power_before\":{},\"power_after\":{},\"identity\":{}}}",
            if dry {
                "dry-run-no-cuda"
            } else {
                "n1-plumbing-not-qualified"
            },
            if pinned {
                "pinned-cacheable"
            } else {
                "pageable"
            },
            if pinned { "0" } else { "null" },
            if dry {
                0
            } else {
                bytes as u64 * u64::from(copies)
            },
            if dry { 0 } else { bytes },
            self.unix_start_ns,
            self.unix_end_ns,
            self.mono_start_ns,
            self.mono_end_ns,
            num(self.wall_ns),
            self.event_ms.map_or("null".into(), |x| x.to_string()),
            num(self.setup_ns),
            num(self.verify_ns),
            string(&self.expected_sha256),
            string(&self.actual_sha256),
            self.power_before.as_deref().unwrap_or("null"),
            self.power_after.as_deref().unwrap_or("null"),
            if dry { "null" } else { "true" },
        );
    }
}

#[cfg(target_os = "linux")]
mod native {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, result, sys};
    use memra_engine::PinnedHostBuf;
    use sha2::{Digest, Sha256};
    use std::process::Command;
    use std::sync::Arc;

    pub struct Probe {
        // Context outlives stream and allocations. One owner thread/stream throughout.
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        selector: String,
    }
    impl Probe {
        pub fn new() -> Result<Self> {
            let available = std::fs::read_to_string("/proc/meminfo")?
                .lines()
                .find_map(|line| line.strip_prefix("MemAvailable:").map(str::trim))
                .and_then(|s| s.split_whitespace().next())
                .ok_or("MemAvailable missing")?
                .parse::<u64>()?
                * 1024;
            if available < 4 << 30 {
                return Err("requires at least 4GiB available host memory".into());
            }
            let ctx = CudaContext::new(0)?;
            if result::mem_get_info()?.0 < 2 << 30 {
                return Err("requires at least 2GiB free VRAM".into());
            }
            let uuid = ctx.uuid()?;
            let hex: String = uuid
                .bytes
                .iter()
                .map(|x| format!("{:02x}", *x as u8))
                .collect();
            let selector = format!(
                "GPU-{}-{}-{}-{}-{}",
                &hex[..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..]
            );
            let stream = ctx.new_stream()?;
            Ok(Self {
                ctx,
                stream,
                selector,
            })
        }
        fn power(&self) -> Result<String> {
            // Match CUDA's actual device via UUID even when ordinal visibility is remapped.
            // The selector never enters a receipt; publish only the sampler's integer index.
            let out = Command::new("nvidia-smi")
                .args([
                    "-i",
                    &self.selector,
                    "--query-gpu=index,power.limit,power.max_limit",
                    "--format=csv,noheader",
                ])
                .output()?;
            if !out.status.success() {
                return Err("nvidia-smi power query failed".into());
            }
            let text = String::from_utf8(out.stdout)?;
            let fields: Vec<_> = text.trim().split(',').map(str::trim).collect();
            if fields.len() != 3 || fields[0].parse::<usize>().is_err() {
                return Err("malformed power query".into());
            }
            for s in &fields[1..] {
                let watts: f64 = s.strip_suffix(" W").ok_or("unknown power limit")?.parse()?;
                if !watts.is_finite() || watts <= 0.0 {
                    return Err("invalid power limit".into());
                }
            }
            Ok(format!(
                "{{\"device\":{:?},\"power.limit\":{:?},\"power.max_limit\":{:?}}}",
                fields[0], fields[1], fields[2]
            ))
        }
        fn copy(&self, dev: &mut CudaSlice<u8>, host: &mut [u8], direction: &str) -> Result<()> {
            let copy = if direction == "h2d" {
                self.stream.memcpy_htod(host, dev)
            } else {
                self.stream.memcpy_dtoh(dev, host)
            };
            // Fence even on enqueue failure before host/device buffers may be recycled.
            let fence = self.stream.synchronize();
            copy?;
            fence?;
            Ok(())
        }
        fn visit(
            &self,
            dev: &mut CudaSlice<u8>,
            host: &mut [u8],
            direction: &str,
            seed: u8,
            clock: &Instant,
            copies: u32,
        ) -> Result<Sample> {
            let setup = Instant::now();
            for (i, b) in host.iter_mut().enumerate() {
                *b = pattern(i, seed);
            }
            let expected = format!("{:x}", Sha256::digest(&*host));
            if direction == "d2h" {
                self.copy(dev, host, "h2d")?;
                host.fill(255);
            } else {
                self.stream.memset_zeros(dev)?;
                self.stream.synchronize()?;
            }
            let setup_ns = setup.elapsed().as_nanos();
            let power_before = self.power()?;
            let unix_start_ns = unix_ns();
            let mono_start_ns = clock.elapsed().as_nanos();
            let wall = Instant::now();
            // One interval per completed copy, not an interval spanning the loop.
            // copy() fences on the owner stream, so event time includes the host
            // submission/fence gap before recording the end (not DMA-only time).
            let mut event_ms = 0.0_f64;
            for _ in 0..copies {
                let start = self
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                self.copy(dev, host, direction)?;
                let end = self
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                end.synchronize()?;
                let operation_ms = start.elapsed_ms(&end)?;
                if !operation_ms.is_finite() || operation_ms < 0.0 {
                    return Err("invalid event timing".into());
                }
                event_ms += f64::from(operation_ms);
            }
            let wall_ns = wall.elapsed().as_nanos();
            let mono_end_ns = clock.elapsed().as_nanos();
            let unix_end_ns = unix_ns();
            let power_after = self.power()?;
            if power_before != power_after {
                return Err("power envelope changed within visit".into());
            }
            let verification = Instant::now();
            if direction == "h2d" {
                host.fill(255);
                self.copy(dev, host, "d2h")?;
            }
            verify(host, seed)?;
            let actual = format!("{:x}", Sha256::digest(&*host));
            if expected != actual {
                return Err("SHA256 mismatch".into());
            }
            Ok(Sample {
                unix_start_ns,
                unix_end_ns,
                mono_start_ns,
                mono_end_ns,
                wall_ns: Some(wall_ns),
                event_ms: Some(event_ms),
                setup_ns: Some(setup_ns),
                verify_ns: Some(verification.elapsed().as_nanos()),
                expected_sha256: Some(expected),
                actual_sha256: Some(actual),
                power_before: Some(power_before),
                power_after: Some(power_after),
            })
        }
        pub fn size(&self, bytes: usize, opts: &Options, clock: &Instant) -> Result<()> {
            let allocation = Instant::now();
            let mut pageable = vec![0u8; bytes];
            self.ctx.bind_to_thread()?;
            // Reuse the engine's cacheable pinned allocation (flags=0), not cudarc's
            // write-combined allocation, whose CPU readback is a different experiment.
            let mut pinned = PinnedHostBuf::new(bytes)?;
            let mut dev = self.stream.alloc_zeros::<u8>(bytes)?;
            self.stream.synchronize()?;
            println!(
                "{{\"schema_version\":1,\"record\":\"allocation\",\"bytes\":{bytes},\"pageable_bytes\":{bytes},\"pinned_bytes\":{bytes},\"device_bytes\":{bytes},\"allocation_ns\":{}}}",
                allocation.elapsed().as_nanos()
            );
            // Two complete distinct-pattern controls per arm/direction. Controls remain
            // raw rows but are NOT observations; timed N=1 rows follow the warm controls.
            for &direction in &opts.directions {
                let arms = if opts.reverse {
                    [true, false]
                } else {
                    [false, true]
                };
                for is_pinned in arms {
                    let host = if is_pinned {
                        pinned.as_mut_slice()
                    } else {
                        &mut pageable
                    };
                    for seed in [3, 71] {
                        let control = self.visit(&mut dev, host, direction, seed, clock, 1)?;
                        println!(
                            "{{\"schema_version\":1,\"record\":\"control\",\"bytes\":{bytes},\"direction\":{direction:?},\"pinned\":{is_pinned},\"seed\":{seed},\"identity\":true,\"sha256\":{:?}}}",
                            control.actual_sha256.unwrap()
                        );
                    }
                    self.visit(&mut dev, host, direction, 113, clock, opts.copies)?
                        .print(bytes, direction, is_pinned, opts);
                }
            }
            Ok(())
        }
    }
}

fn run() -> Result<()> {
    let opts = Options::parse(std::env::args().skip(1))?;
    let clock = Instant::now();
    // Comparator red control is CPU-only and must reject a corrupted byte.
    let mut red: Vec<_> = (0..4096).map(|i| pattern(i, 3)).collect();
    verify(&red, 3)?;
    red[2048] ^= 1;
    if verify(&red, 3).is_ok() {
        return Err("comparator red control accepted corruption".into());
    }
    if opts.dry {
        for &bytes in &opts.sizes {
            for &direction in &opts.directions {
                for pinned in if opts.reverse {
                    [true, false]
                } else {
                    [false, true]
                } {
                    let now = unix_ns();
                    Sample {
                        unix_start_ns: now,
                        unix_end_ns: now,
                        mono_start_ns: clock.elapsed().as_nanos(),
                        mono_end_ns: clock.elapsed().as_nanos(),
                        ..Sample::default()
                    }
                    .print(bytes, direction, pinned, &opts);
                }
            }
        }
    } else {
        #[cfg(target_os = "linux")]
        {
            let probe = native::Probe::new()?;
            for &bytes in &opts.sizes {
                probe.size(bytes, &opts, &clock)?;
            }
        }
        #[cfg(not(target_os = "linux"))]
        return Err("CUDA execution requires Linux; use --dry-run for schema plumbing".into());
    }
    println!(
        "{{\"schema_version\":1,\"record\":\"RESULT\",\"status\":{:?},\"samples\":{},\"n_per_size_direction_arm\":1,\"comparator_red_rejected\":true,\"qualified\":false}}",
        if opts.dry {
            "dry-run-no-cuda"
        } else {
            "n1-plumbing-not-qualified"
        },
        opts.sizes.len() * opts.directions.len() * 2
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_fail_closed() {
        for args in [
            vec!["--bytes", "0"],
            vec!["--copies", "0"],
            vec!["--copies", "100001"],
            vec!["--copies", "-1"],
            vec!["--copies", "nope"],
            vec!["--copies"],
            vec!["--repeats", "2"],
            vec!["--direction", "bad"],
            vec!["--surprise"],
        ] {
            assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
        }
        assert!(
            Options::parse(
                ["--dry-run", "--bytes", "4096", "--order", "ba"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .unwrap()
            .reverse
        );
    }
    #[test]
    fn copies_bounds_and_default() {
        assert_eq!(Options::parse(std::iter::empty()).unwrap().copies, 1);
        for n in [1, 2, 100_000] {
            let opts = Options::parse(["--copies".to_owned(), n.to_string()].into_iter()).unwrap();
            assert_eq!(opts.copies, n);
        }
    }
    #[test]
    fn full_byte_comparator() {
        for seed in [3, 71, 113] {
            let mut bytes: Vec<_> = (0..4096).map(|i| pattern(i, seed)).collect();
            verify(&bytes, seed).unwrap();
            for i in [0, 2048, 4095] {
                bytes[i] ^= 1;
                assert!(verify(&bytes, seed).is_err());
                bytes[i] ^= 1;
            }
        }
    }
}
