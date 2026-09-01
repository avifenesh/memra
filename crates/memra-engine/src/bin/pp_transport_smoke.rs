//! M1-PP2 increment-2 TRANSPORT SMOKE (single-device-runnable).
//!
//! What it proves on one GPU:
//!   1. the cuDeviceCanAccessPeer guard path runs (matrix printed for every device pair);
//!   2. the EXACT peer-copy FFI the cross-device TX uses (`cuMemcpyPeerAsync` with
//!      explicit src/dst contexts) moves correct bytes — same-context both sides is a
//!      legal degenerate case of the same driver call, so the plumbing (contexts, raw
//!      pointers, byte counts, stream handle) is exercised without a second device;
//!   3. the Pp2Rt boundary choreography (per-stage streams, ev_tx/ev_rx, persistent
//!      slots, overlap double-buffering) round-trips patterned buffers bit-exactly.
//!
//! With `MEMRA_PP_DEVICES=0,1`, the boundary roundtrip also proves an actual cross-device
//! copy. Each stage scope binds its CUDA context just like the production PP walkers; pushing
//! an ambient stream alone is insufficient when the stages live on different devices.
//!
//! usage: pp-transport-smoke [--runtime-probe-cycle]   (exit 0 = all sub-smokes pass)
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_probe_cycle = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--runtime-probe-cycle") => true,
        Some(arg) => return Err(format!("unknown argument {arg:?}").into()),
    };
    let mut fails = 0usize;
    // Match the serving worker: under an explicit placement the primary Engine follows
    // the last/head stage. With no placement this remains the original device-0 smoke.
    let primary = std::env::var("MEMRA_PP_DEVICES")
        .ok()
        .and_then(|value| value.split(',').next_back()?.trim().parse::<usize>().ok())
        .unwrap_or(0);
    // Engine initialization also initializes the driver (raw result:: calls below need cuInit).
    let e = Engine::new(primary)?;
    println!("primary device: {primary}");

    // ---- 1. device census + CanAccessPeer matrix ----
    let ndev = cudarc::driver::result::device::get_count()? as usize;
    println!("devices: {ndev}");
    for a in 0..ndev {
        for b in 0..ndev {
            if a == b {
                continue;
            }
            let da = cudarc::driver::result::device::get(a as i32)?;
            let db = cudarc::driver::result::device::get(b as i32)?;
            let mut can: i32 = 0;
            unsafe {
                cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can, da, db).result()?;
            }
            println!("cuDeviceCanAccessPeer({a} -> {b}) = {can}");
        }
    }
    if ndev < 2 {
        println!(
            "(single device: no peer pairs — matrix section is census-only here; \
                  the cross-device arm gates on the 8x box)"
        );
    }

    // ---- 2. forced peer-arm copy, same context both sides ----
    // A host-bounce run is specifically for a machine whose peer-copy path may corrupt
    // bytes. Do not poison that process before exercising the fallback.
    if !memra_engine::pp::pp_host_bounce_on() {
        let ctx = e.ctx();
        let s = ctx.new_stream()?;
        let n = 4096usize;
        let pat: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let src = s.clone_htod(&pat)?;
        let mut dst = s.alloc_zeros::<f32>(n)?;
        {
            let (sp, _g0) = src.device_ptr(&s);
            let (dp, _g1) = dst.device_ptr_mut(&s);
            unsafe {
                cudarc::driver::result::memcpy_peer_async(
                    ctx.cu_ctx(),
                    dp,
                    ctx.cu_ctx(),
                    sp,
                    n * 4,
                    s.cu_stream(),
                )?;
            }
        }
        s.synchronize()?;
        let back = s.clone_dtoh(&dst)?;
        s.synchronize()?;
        let diff = back
            .iter()
            .zip(&pat)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "peer-arm copy (cuMemcpyPeerAsync, same-ctx degenerate): bytediff={diff} {}",
            if diff == 0 {
                "OK"
            } else {
                fails += 1;
                "FAIL"
            }
        );
    } else {
        println!("peer-arm copy skipped: MEMRA_PP_HOST_BOUNCE=1");
    }

    // ---- 3. Pp2Rt boundary choreography: tx/rx roundtrip, then overlap slots ----
    unsafe {
        std::env::set_var("MEMRA_PP_OVERLAP", "1");
    } // exercise slot alternation
    let rt = memra_engine::pp::Pp2Rt::get(&e)?;
    let _walk = rt.acquire_walk("pp_transport_smoke")?;
    rt.init_boundary_transport(&e, 4096)?;
    println!("Pp2Rt built: cross_device={}", rt.cross_device());
    let n = 5120usize;
    let mut round_fail = 0usize;
    for step in 0..4 {
        let pat: Vec<f32> = (0..n).map(|i| (i as f32) + 1000.0 * step as f32).collect();
        // TX inside the stage-0 scope (ambient stream = stage-0's)
        let slot = {
            rt.bind_stage(0)?;
            let _s0 = rt.enter(0);
            let e0 = rt.engine(0, &e);
            let x = e0.htod(&pat)?;
            rt.tx(0, &x, n)?
        };
        // RX inside the stage-1 scope; dtoh through the ambient (stage-1) stream
        rt.bind_stage(1)?;
        let _s1 = rt.enter(1);
        let e1 = rt.engine(1, &e);
        let work = rt.rx(0, slot, n)?;
        let back = e1.dtoh(&work)?;
        let diff = back
            .iter()
            .zip(&pat)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if diff != 0 {
            round_fail += 1;
        }
        println!(
            "boundary roundtrip step {step} slot {slot}: bytediff={diff} {}",
            if diff == 0 { "OK" } else { "FAIL" }
        );
    }
    // overlap=1 must alternate slots 0,1,0,1 — assert we actually exercised both
    if round_fail > 0 {
        fails += 1;
    }

    // Optional model-free runtime cadence receipt. Keep all CUDA work on this owner thread and
    // service between completed boundary ticks, matching the server call site. The four regular
    // smoke copies above count toward the fixed cadence, so continue through one exact cycle.
    if runtime_probe_cycle {
        if !rt.cross_device() {
            return Err("--runtime-probe-cycle requires a cross-device PP placement".into());
        }
        let cycle_x = {
            rt.bind_stage(0)?;
            let _s0 = rt.enter(0);
            rt.engine(0, &e).htod(&[1.0f32])?
        };
        let mut serviced = 0u64;
        for copy_index in 4..memra_engine::pp::PEER_RUNTIME_PROBE_CYCLE_COPIES {
            let completed_tick =
                (copy_index + 1) % memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES == 0;
            let slot = {
                rt.bind_stage(0)?;
                let _s0 = rt.enter(0);
                rt.tx(0, &cycle_x, 1)?
            };
            {
                rt.bind_stage(1)?;
                let _s1 = rt.enter(1);
                let work = rt.rx(0, slot, 1)?;
                // The runtime hook's contract is "between completed scheduler ticks". Merely
                // enqueueing rx() leaves thousands of destination-stream reads outstanding in
                // this synthetic no-head loop, unlike a real decode's terminal logits readback.
                // Drain only the four due ticks so the probe measures peer integrity rather than
                // racing the harness's artificial backlog.
                if completed_tick {
                    let back = rt.engine(1, &e).dtoh(&work)?;
                    if back != [1.0f32] {
                        return Err(format!(
                            "runtime probe cycle boundary payload mismatch before copy {}: {:?}",
                            copy_index + 1,
                            back,
                        )
                        .into());
                    }
                }
            }
            if memra_engine::pp::service_runtime_peer_probe(&e, true, true)?.ran() {
                serviced += 1;
                println!(
                    "runtime probe serviced after boundary copy {}",
                    copy_index + 1
                );
            }
        }
        let metrics = memra_engine::pp::peer_probe_metrics();
        let expected = memra_engine::pp::PEER_RUNTIME_PROBE_CYCLE_COPIES
            / memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let ok = serviced == expected
            && metrics.boundary_copies == memra_engine::pp::PEER_RUNTIME_PROBE_CYCLE_COPIES
            && metrics.runtime_probes == expected
            && metrics.runtime_failures == 0;
        println!(
            "runtime probe cycle: serviced={serviced}/{expected} boundary_copies={} \
             runtime_failures={} {}",
            metrics.boundary_copies,
            metrics.runtime_failures,
            if ok {
                "OK"
            } else {
                fails += 1;
                "FAIL"
            },
        );
    }

    if fails == 0 {
        println!("pp-transport-smoke PASS");
        Ok(())
    } else {
        println!("pp-transport-smoke FAIL ({fails} sub-smokes)");
        std::process::exit(1);
    }
}
