//! vmm-call-cost: what each CUDA VMM driver call costs on this card, and whether it waits for
//! queued device work (WP-B day 37 stage 0, `research/spill-b-20260919/DAY37.md` 1.5).
//!
//! WHY. The on-demand KV planes of `MEMRA_KV_ALLOCATOR=vmm` map granules as a session's rows
//! grow and release them at retire and park. Where those calls may run (inline on the CUDA
//! owner thread, on a helper thread, or only at an idle tick) depends on two facts no receipt
//! holds: the host wall of each call, and whether a call returns only after work already queued
//! on the device drains. This probe measures both, per extent size, in three regimes:
//!
//!   * `idle`: nothing queued.
//!   * `busy`: at least `--queue-ms` of large device memsets queued on the owner stream on an
//!     unrelated buffer before the call. `queued_ms_remaining` is the time from the call's
//!     start until the queue's tail event completes: an op that waits for the queue returns
//!     at about that time; one that does not returns long before it.
//!   * `cross`: the same queue, the call made from a helper thread while the owner thread times
//!     a small memset plus its event on a second, idle stream (`owner_us`). If the helper's call
//!     holds a context lock across a device wait, the owner's launch waits with it.
//!
//! Ops per repetition, in order: `reserve` (cuMemAddressReserve), `create` (cuMemCreate),
//! `map` (cuMemMap), `access` (cuMemSetAccess), `zero` (the cuMemsetD8Async enqueue of the new
//! range), `unmap` (cuMemUnmap), `release` (cuMemRelease), `free` (cuMemAddressFree).
//!
//! Output: one `VMM-CALL-COST op=.. extent=.. regime=.. N=.. median_us=.. p95_us=.. max_us=..
//! queued_ms_remaining=..` line per cell (and `owner_us=` / `owner_base_us=` on `cross`), then
//! the pre-registered branch readings (`VMM-CALL-COST BLOCKS`, `GROW-PLACEMENT`,
//! `RELEASE-PLACEMENT`). Diagnostic only: no flag, no default, nothing reads its output but the
//! lane's record.
//!
//! Usage: `vmm-call-cost [--device N] [--reps N] [--queue-ms N] [--extents 1,4,16,64]`.
use cudarc::driver::{CudaContext, CudaStream, sys};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const OPS: [&str; 8] = [
    "reserve", "create", "map", "access", "zero", "unmap", "release", "free",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Regime {
    Idle,
    Busy,
    Cross,
}
impl Regime {
    fn name(self) -> &'static str {
        match self {
            Regime::Idle => "idle",
            Regime::Busy => "busy",
            Regime::Cross => "cross",
        }
    }
}

struct Args {
    device: usize,
    reps: usize,
    queue_ms: f64,
    extents: Vec<usize>,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        device: 0,
        reps: 20,
        queue_ms: 60.0,
        extents: vec![1, 4, 16, 64],
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = argv
            .get(i + 1)
            .ok_or_else(|| format!("{} needs a value", argv[i]))?;
        match argv[i].as_str() {
            "--device" => a.device = v.parse()?,
            "--reps" => a.reps = v.parse()?,
            "--queue-ms" => a.queue_ms = v.parse()?,
            "--extents" => {
                a.extents = v
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument {other}").into()),
        }
        i += 2;
    }
    if a.reps == 0 || a.extents.is_empty() || a.extents.contains(&0) {
        return Err("REFUSED: reps and extents must be positive".into());
    }
    Ok(a)
}

fn prop(device: i32) -> sys::CUmemAllocationProp {
    sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType(0),
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}

/// One VMM op to run, on whichever thread the regime names.
#[derive(Clone, Copy)]
enum Op {
    Reserve {
        bytes: usize,
        align: usize,
    },
    Create {
        bytes: usize,
    },
    Map {
        va: u64,
        bytes: usize,
        handle: u64,
    },
    Access {
        va: u64,
        bytes: usize,
    },
    Zero {
        va: u64,
        bytes: usize,
        stream: usize,
    },
    Unmap {
        va: u64,
        bytes: usize,
    },
    Release {
        handle: u64,
    },
    Free {
        va: u64,
        bytes: usize,
    },
}

/// Runs one op and returns (wall, produced value: VA for reserve, handle for create).
fn run_op(ctx: &Arc<CudaContext>, device: i32, op: Op) -> Res<(Duration, u64)> {
    ctx.bind_to_thread()?;
    let p = prop(device);
    let t0 = Instant::now();
    let out = unsafe {
        // SAFETY: every op is called on addresses and handles this probe reserved or created
        // itself, in the order reserve, create, map, access, zero, unmap, release, free; the
        // zero enqueue targets a range that is mapped and accessible at the call.
        match op {
            Op::Reserve { bytes, align } => {
                let mut va = 0;
                sys::cuMemAddressReserve(&mut va, bytes, align, 0, 0).result()?;
                va
            }
            Op::Create { bytes } => {
                let mut h = 0;
                sys::cuMemCreate(&mut h, bytes, &p, 0).result()?;
                h
            }
            Op::Map { va, bytes, handle } => {
                sys::cuMemMap(va, bytes, 0, handle, 0).result()?;
                0
            }
            Op::Access { va, bytes } => {
                let desc = sys::CUmemAccessDesc {
                    location: p.location,
                    flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                };
                sys::cuMemSetAccess(va, bytes, &desc, 1).result()?;
                0
            }
            Op::Zero { va, bytes, stream } => {
                sys::cuMemsetD8Async(va, 0, bytes, stream as sys::CUstream).result()?;
                0
            }
            Op::Unmap { va, bytes } => {
                sys::cuMemUnmap(va, bytes).result()?;
                0
            }
            Op::Release { handle } => {
                sys::cuMemRelease(handle).result()?;
                0
            }
            Op::Free { va, bytes } => {
                sys::cuMemAddressFree(va, bytes).result()?;
                0
            }
        }
    };
    Ok((t0.elapsed(), out))
}

/// The busy queue: `count` memsets of `buf_bytes` on `stream`, then the tail event.
struct Queue {
    buf: u64,
    buf_bytes: usize,
    count: usize,
}
impl Queue {
    fn launch(&self, stream: &CudaStream) -> Res<cudarc::driver::CudaEvent> {
        for _ in 0..self.count {
            // SAFETY: `buf` is this probe's own pooled allocation of `buf_bytes`.
            unsafe {
                sys::cuMemsetD8Async(self.buf, 0x5a, self.buf_bytes, stream.cu_stream()).result()?
            };
        }
        Ok(stream.record_event(None)?)
    }
}

/// Spin until the event completes; returns the instant it was first seen complete.
fn drained_at(ev: &cudarc::driver::CudaEvent) -> Instant {
    while !ev.is_complete() {
        std::hint::spin_loop();
    }
    Instant::now()
}

#[derive(Default)]
struct Cell {
    op_us: Vec<f64>,
    remaining_ms: Vec<f64>,
    owner_us: Vec<f64>,
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

/// Time a small memset plus its event on the idle second stream, from the calling thread.
fn owner_probe(side: &CudaStream, small: u64) -> Res<f64> {
    let t0 = Instant::now();
    // SAFETY: `small` is this probe's own 4 KiB allocation.
    unsafe { sys::cuMemsetD8Async(small, 0, 4096, side.cu_stream()).result()? };
    let ev = side.record_event(None)?;
    ev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6)
}

fn main() -> Res<()> {
    let a = parse_args()?;
    let ctx = CudaContext::new(a.device)?;
    let owner = ctx.new_stream()?;
    let side = ctx.new_stream()?;
    let device = cudarc::driver::result::device::get(a.device as i32)?;
    let mut supported = 0;
    // SAFETY: valid device and output pointer.
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut supported,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            device,
        )
        .result()?
    };
    if supported != 1 {
        return Err("REFUSED: device does not support CUDA VMM".into());
    }
    let p = prop(device);
    let mut gran = 0usize;
    // SAFETY: complete properties and a valid output pointer.
    unsafe {
        sys::cuMemGetAllocationGranularity(
            &mut gran,
            &p,
            sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        .result()?
    };
    // The busy queue's buffer and the owner probe's small buffer: pooled, this probe's own.
    let buf_bytes: usize = 512 << 20;
    let qbuf = owner.alloc_zeros::<u8>(buf_bytes)?;
    let sbuf = side.alloc_zeros::<u8>(4096)?;
    let (qptr, _qg) = cudarc::driver::DevicePtr::device_ptr(&qbuf, &owner);
    let (sptr, _sg) = cudarc::driver::DevicePtr::device_ptr(&sbuf, &side);
    owner.synchronize()?;
    // Calibrate one memset of the queue buffer, then size the queue for >= queue_ms.
    let mut one = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        let ev = Queue {
            buf: qptr,
            buf_bytes,
            count: 1,
        }
        .launch(&owner)?;
        ev.synchronize()?;
        one.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let one_ms = pct(&one, 0.5).max(0.01);
    let count = ((a.queue_ms / one_ms).ceil() as usize).clamp(1, 20_000);
    let queue = Queue {
        buf: qptr,
        buf_bytes,
        count,
    };
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: valid output pointers, context bound.
    unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total).result()? };
    let name = cudarc::driver::result::device::get_name(device).unwrap_or_default();
    println!(
        "[vmm-call-cost] device {} {name:?} granularity={gran} free={free} total={total} \
         queue: {count} x {buf_bytes} B memsets (one {one_ms:.3} ms, target {} ms) reps={} extents={:?}",
        a.device, a.queue_ms, a.reps, a.extents
    );
    // The helper thread for `cross`: runs one op per request on its own bound context.
    let (req_tx, req_rx) = mpsc::channel::<Op>();
    let (rep_tx, rep_rx) = mpsc::channel::<Result<(Duration, u64), String>>();
    let hctx = ctx.clone();
    let helper = std::thread::spawn(move || {
        while let Ok(op) = req_rx.recv() {
            let r = run_op(&hctx, device, op).map_err(|e| e.to_string());
            if rep_tx.send(r).is_err() {
                break;
            }
        }
    });
    // Owner launch base (no helper call in flight), for the cross regime's comparison.
    let mut base = Vec::new();
    for _ in 0..a.reps.max(20) {
        base.push(owner_probe(&side, sptr)?);
    }
    let owner_base_med = pct(&base, 0.5);
    let mut cells: std::collections::BTreeMap<(usize, &'static str, &'static str), Cell> =
        Default::default();
    for &n in &a.extents {
        let bytes = n * gran;
        for regime in [Regime::Idle, Regime::Busy, Regime::Cross] {
            for _ in 0..a.reps {
                let mut va = 0u64;
                let mut handle = 0u64;
                for (i, name) in OPS.iter().enumerate() {
                    let op = match i {
                        0 => Op::Reserve { bytes, align: gran },
                        1 => Op::Create { bytes },
                        2 => Op::Map { va, bytes, handle },
                        3 => Op::Access { va, bytes },
                        4 => Op::Zero {
                            va,
                            bytes,
                            stream: owner.cu_stream() as usize,
                        },
                        5 => Op::Unmap { va, bytes },
                        6 => Op::Release { handle },
                        _ => Op::Free { va, bytes },
                    };
                    let cell = cells.entry((n, regime.name(), name)).or_default();
                    // The zero enqueue must land before the unmap in every regime.
                    if i == 5 {
                        owner.synchronize()?;
                    }
                    let (wall, out, remaining) = match regime {
                        Regime::Idle => {
                            let (w, o) = run_op(&ctx, device, op)?;
                            (w, o, 0.0)
                        }
                        Regime::Busy => {
                            let tail = queue.launch(&owner)?;
                            let t0 = Instant::now();
                            let (w, o) = run_op(&ctx, device, op)?;
                            let done = drained_at(&tail);
                            (w, o, done.duration_since(t0).as_secs_f64() * 1e3)
                        }
                        Regime::Cross => {
                            let tail = queue.launch(&owner)?;
                            let t0 = Instant::now();
                            req_tx.send(op)?;
                            // The owner's small launch while the helper's call is in flight.
                            let owner_us = owner_probe(&side, sptr)?;
                            let (w, o) = rep_rx
                                .recv()?
                                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                            let done = drained_at(&tail);
                            cell.owner_us.push(owner_us);
                            (w, o, done.duration_since(t0).as_secs_f64() * 1e3)
                        }
                    };
                    match i {
                        0 => va = out,
                        1 => handle = out,
                        _ => {}
                    }
                    cell.op_us.push(wall.as_secs_f64() * 1e6);
                    cell.remaining_ms.push(remaining);
                }
                owner.synchronize()?;
            }
        }
    }
    drop(req_tx);
    let _ = helper.join();
    for ((n, regime, op), c) in &cells {
        let mut line = format!(
            "VMM-CALL-COST op={op} extent={n} regime={regime} N={} median_us={:.1} p95_us={:.1} \
             max_us={:.1} queued_ms_remaining={:.2}",
            c.op_us.len(),
            pct(&c.op_us, 0.5),
            pct(&c.op_us, 0.95),
            pct(&c.op_us, 1.0),
            pct(&c.remaining_ms, 0.5),
        );
        if *regime == "cross" {
            line.push_str(&format!(
                " owner_us={:.1} owner_base_us={owner_base_med:.1}",
                pct(&c.owner_us, 0.5)
            ));
        }
        println!("{line}");
    }
    // Branch readings, DAY37 1.5, verbatim rules.
    let med = |n: usize, r: &str, op: &str| cells.get(&(n, r, op)).map(|c| pct(&c.op_us, 0.5));
    let rem = |n: usize, r: &str, op: &str| {
        cells
            .get(&(n, r, op))
            .map(|c| pct(&c.remaining_ms, 0.5) * 1e3)
    };
    let own = |n: usize, op: &str| cells.get(&(n, "cross", op)).map(|c| pct(&c.owner_us, 0.5));
    let mut blocks_queue: std::collections::BTreeMap<&str, bool> = Default::default();
    let mut blocks_owner: std::collections::BTreeMap<&str, bool> = Default::default();
    for op in OPS {
        let mut bq = false;
        let mut bo = false;
        for &n in &a.extents {
            let (Some(idle), Some(busy), Some(rem_us)) =
                (med(n, "idle", op), med(n, "busy", op), rem(n, "busy", op))
            else {
                continue;
            };
            let q = busy >= 0.5 * rem_us && busy >= 10.0 * idle;
            let o = own(n, op).is_some_and(|o| o >= 10.0 * owner_base_med);
            println!(
                "VMM-CALL-COST BLOCKS op={op} extent={n} busy_median_us={busy:.1} \
                 remaining_at_call_us={rem_us:.1} idle_median_us={idle:.1} behind_queue={} \
                 owner_from_helper={}",
                if q { "yes" } else { "no" },
                if o { "yes" } else { "no" }
            );
            bq |= q;
            bo |= o;
        }
        blocks_queue.insert(op, bq);
        blocks_owner.insert(op, bo);
    }
    let one = a.extents.iter().copied().min().unwrap_or(1);
    let grow_p95: f64 = ["create", "map", "access"]
        .iter()
        .map(|op| {
            cells
                .get(&(one, "busy", op))
                .map(|c| pct(&c.op_us, 0.95))
                .unwrap_or(f64::NAN)
        })
        .sum();
    let grow_blocks = ["create", "map", "access"]
        .iter()
        .any(|op| blocks_queue.get(op).copied().unwrap_or(false));
    println!(
        "GROW-PLACEMENT extent={one} busy_p95_sum_us={grow_p95:.1} blocks_behind_queue={} rule \
         p95<=100 -> {}",
        if grow_blocks { "yes" } else { "no" },
        if grow_p95 <= 100.0 {
            "inline"
        } else {
            "helper"
        }
    );
    let rel_q = blocks_queue["unmap"] || blocks_queue["release"];
    let rel_o = blocks_owner["unmap"] || blocks_owner["release"];
    println!(
        "RELEASE-PLACEMENT unmap_or_release_blocks_behind_queue={} blocks_owner_from_helper={} -> {}",
        if rel_q { "yes" } else { "no" },
        if rel_o { "yes" } else { "no" },
        match (rel_q, rel_o) {
            (false, _) => "owner-tick",
            (true, false) => "helper",
            (true, true) => "idle-tick",
        }
    );
    Ok(())
}
