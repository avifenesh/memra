//! WP-A day 37 (DAY37.md section 3): which concurrent action holds another thread's CUDA calls while a long
//! kernel runs on one stream of a shared (primary) context. Thread H (the holder) launches a 300 ms
//! `%globaltimer` spin on its own stream and then, 20 ms later, performs one action X. Thread V (the victim),
//! 40 ms after the spin's launch, times one call Y on its own stream. Every (X, Y) pair runs N times; a
//! `serial` column runs X and Y on one thread (the cells' serial form). One line per run:
//! `PROBE x=<X> y=<Y> mode=<two-thread|serial> run=<k> y_ms=<ms> x_ms=<ms>`.
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg, result, sys};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const SPIN: &str = r#"
extern "C" __global__ void spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
extern "C" __global__ void touch(float* p) { p[0] += 1.0f; }
"#;

const XS: &[&str] = &[
    "none",
    "free-host",
    "malloc-host",
    "free-async",
    "stream-sync-own",
    "event-sync-own",
    "ctx-sync",
];
const YS: &[&str] = &["alloc-zeros", "malloc-host", "htod-pageable", "launch", "event-query"];

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct Kit {
    ctx: Arc<CudaContext>,
    spin: cudarc::driver::CudaFunction,
    touch: cudarc::driver::CudaFunction,
}

fn launch_spin(kit: &Kit, s: &Arc<CudaStream>, ns: u64) {
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let mut b = s.launch_builder(&kit.spin);
    b.arg(&ns);
    unsafe { b.launch(cfg) }.unwrap();
}

/// The holder's action X on its own stream `s` (the spin already queued there).
fn do_x(kit: &Kit, s: &Arc<CudaStream>, x: &str, pre: &mut Option<*mut std::ffi::c_void>) -> f64 {
    let t = Instant::now();
    match x {
        "none" => {}
        "free-host" => {
            let p = pre.take().unwrap();
            unsafe { result::free_host(p) }.unwrap();
        }
        "malloc-host" => {
            let p = unsafe { result::malloc_host(4 << 20, 0) }.unwrap();
            *pre = Some(p);
        }
        "free-async" => {
            let d = s.alloc_zeros::<f32>(1 << 20).unwrap();
            drop(d);
        }
        "stream-sync-own" => s.synchronize().unwrap(),
        "event-sync-own" => s.record_event(None).unwrap().synchronize().unwrap(),
        "ctx-sync" => kit.ctx.synchronize().unwrap(),
        _ => unreachable!(),
    }
    ms(t.elapsed())
}

/// The victim's call Y on its own stream `s`.
fn do_y(kit: &Kit, s: &Arc<CudaStream>, y: &str, pageable: &[f32]) -> f64 {
    let t = Instant::now();
    match y {
        "alloc-zeros" => {
            let d = s.alloc_zeros::<f32>(1 << 20).unwrap();
            std::mem::forget(d); // never freed inside the probe's timing (a free is an X, not a Y)
        }
        "malloc-host" => {
            let p = unsafe { result::malloc_host(4 << 20, 0) }.unwrap();
            let _ = p; // leaked on purpose: a free would be an X
        }
        "htod-pageable" => {
            let d = s.clone_htod(pageable).unwrap();
            std::mem::forget(d);
        }
        "launch" => {
            let mut d = s.alloc_zeros::<f32>(1).unwrap();
            s.synchronize().unwrap();
            let t2 = Instant::now();
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
            let mut b = s.launch_builder(&kit.touch);
            b.arg(&mut d);
            unsafe { b.launch(cfg) }.unwrap();
            let out = ms(t2.elapsed());
            std::mem::forget(d);
            return out;
        }
        "event-query" => {
            let e = s.record_event(None).unwrap();
            let _ = e.is_complete();
        }
        _ => unreachable!(),
    }
    ms(t.elapsed())
}

/// DAY37 section 4's cross-context extension: a second, created context on the same card, driven by raw
/// driver calls (cudarc 0.19 exposes only the primary context as a safe type).
struct RawCtx {
    ctx: sys::CUcontext,
    spin: sys::CUfunction,
    stream: sys::CUstream,
}
unsafe impl Send for RawCtx {}
unsafe impl Sync for RawCtx {}

fn raw_ctx(ptx_src: &str) -> RawCtx {
    unsafe {
        let mut dev: sys::CUdevice = 0;
        sys::cuDeviceGet(&mut dev, 0).result().unwrap();
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        sys::cuCtxCreate_v4(&mut ctx, std::ptr::null_mut(), 0, dev).result().unwrap();
        sys::cuCtxSetCurrent(ctx).result().unwrap();
        let src = std::ffi::CString::new(ptx_src).unwrap();
        let m = result::module::load_data(src.as_ptr() as *const _).unwrap();
        let f = result::module::get_function(m, std::ffi::CString::new("spin").unwrap()).unwrap();
        let s = result::stream::create(result::stream::StreamKind::NonBlocking).unwrap();
        RawCtx { ctx, spin: f, stream: s }
    }
}

fn raw_spin(r: &RawCtx, ns: u64) {
    let mut ns = ns;
    let mut params = [&mut ns as *mut u64 as *mut std::ffi::c_void];
    unsafe { result::launch_kernel(r.spin, (1, 1, 1), (1, 1, 1), 0, r.stream, &mut params) }.unwrap();
}

fn raw_y(r: &RawCtx, y: &str, pageable: &[f32]) -> f64 {
    let t = Instant::now();
    unsafe {
        match y {
            "alloc-zeros" => {
                let _ = result::malloc_async(r.stream, 4 << 20).unwrap();
            }
            "malloc-host" => {
                let _ = result::malloc_host(4 << 20, 0).unwrap();
            }
            "htod-pageable" => {
                let d = result::malloc_async(r.stream, std::mem::size_of_val(pageable)).unwrap();
                result::memcpy_htod_async(d, pageable, r.stream).unwrap();
            }
            "event-query" => {
                let e = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).unwrap();
                result::event::record(e, r.stream).unwrap();
                let _ = result::event::query(e);
            }
            _ => unreachable!(),
        }
    }
    ms(t.elapsed())
}

fn cross(kit: &Arc<Kit>, ptx_src: &str, n: usize, pageable: &Arc<Vec<f32>>) {
    let raw = Arc::new(raw_ctx(ptx_src));
    let ys = ["alloc-zeros", "malloc-host", "htod-pageable", "event-query"];
    for (holder, victim) in [("created", "primary"), ("primary", "created"), ("primary", "primary")] {
        for x in ["none", "free-host"] {
            for y in ys {
                for run in 1..=n {
                    let barrier = Arc::new(Barrier::new(2));
                    let (k1, r1, b1) = (kit.clone(), raw.clone(), barrier.clone());
                    let h = std::thread::spawn(move || {
                        let pre = if holder == "created" {
                            unsafe { sys::cuCtxSetCurrent(r1.ctx) }.result().unwrap();
                            let p = if x == "free-host" { Some(unsafe { result::malloc_host(4 << 20, 0) }.unwrap()) } else { None };
                            unsafe { sys::cuStreamSynchronize(r1.stream) }.result().unwrap();
                            b1.wait();
                            raw_spin(&r1, 300_000_000);
                            p
                        } else {
                            k1.ctx.bind_to_thread().unwrap();
                            let s = k1.ctx.new_stream().unwrap();
                            let p = if x == "free-host" { Some(unsafe { result::malloc_host(4 << 20, 0) }.unwrap()) } else { None };
                            s.synchronize().unwrap();
                            b1.wait();
                            launch_spin(&k1, &s, 300_000_000);
                            std::mem::forget(s);
                            p
                        };
                        std::thread::sleep(Duration::from_millis(20));
                        let t = Instant::now();
                        if let Some(p) = pre {
                            unsafe { result::free_host(p) }.unwrap();
                        }
                        ms(t.elapsed())
                    });
                    let (k2, r2, b2, pg) = (kit.clone(), raw.clone(), barrier.clone(), pageable.clone());
                    let v = std::thread::spawn(move || {
                        if victim == "created" {
                            unsafe { sys::cuCtxSetCurrent(r2.ctx) }.result().unwrap();
                            b2.wait();
                            std::thread::sleep(Duration::from_millis(40));
                            raw_y(&r2, y, &pg)
                        } else {
                            k2.ctx.bind_to_thread().unwrap();
                            let s = k2.ctx.new_stream().unwrap();
                            s.synchronize().unwrap();
                            b2.wait();
                            std::thread::sleep(Duration::from_millis(40));
                            let out = do_y(&k2, &s, y, &pg);
                            std::mem::forget(s);
                            out
                        }
                    });
                    let x_ms = h.join().unwrap();
                    let y_ms = v.join().unwrap();
                    println!("CROSS holder={holder} victim={victim} x={x} y={y} run={run} y_ms={y_ms:.2} x_ms={x_ms:.2}");
                    unsafe { sys::cuCtxSetCurrent(raw.ctx) }.result().unwrap();
                    unsafe { sys::cuCtxSynchronize() }.result().unwrap();
                    kit.ctx.synchronize().unwrap();
                }
            }
        }
    }
}

fn main() {
    let n: usize = std::env::args().nth(1).map(|a| a.parse().unwrap()).unwrap_or(3);
    let tracking = std::env::args().nth(2).map(|a| a == "tracking").unwrap_or(true);
    let ctx = CudaContext::new(0).unwrap();
    if !tracking {
        unsafe { ctx.disable_event_tracking() };
    }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        SPIN,
        cudarc::nvrtc::CompileOptions { arch: Some("compute_90"), ..Default::default() },
    )
    .unwrap();
    let ptx_src = ptx.to_src();
    let m = ctx.load_module(ptx).unwrap();
    let kit = Arc::new(Kit { ctx: ctx.clone(), spin: m.load_function("spin").unwrap(), touch: m.load_function("touch").unwrap() });
    let pageable: Arc<Vec<f32>> = Arc::new((0..(1usize << 20)).map(|i| i as f32).collect());
    println!("PROBE header n={n} event_tracking={}", ctx.is_event_tracking());
    if std::env::args().nth(3).as_deref() == Some("cross") {
        cross(&kit, &ptx_src, n, &pageable);
        return;
    }
    for x in XS {
        for y in YS {
            for run in 1..=n {
                // Two threads.
                let barrier = Arc::new(Barrier::new(2));
                let (k1, b1, xs) = (kit.clone(), barrier.clone(), x.to_string());
                let holder = std::thread::spawn(move || {
                    k1.ctx.bind_to_thread().unwrap();
                    let s = k1.ctx.new_stream().unwrap();
                    let mut pre = if xs == "free-host" { Some(unsafe { result::malloc_host(4 << 20, 0) }.unwrap()) } else { None };
                    s.synchronize().unwrap();
                    b1.wait();
                    launch_spin(&k1, &s, 300_000_000);
                    std::thread::sleep(Duration::from_millis(20));
                    let x_ms = do_x(&k1, &s, &xs, &mut pre);
                    s.synchronize().unwrap();
                    if let Some(p) = pre.take() { unsafe { result::free_host(p) }.unwrap(); }
                    x_ms
                });
                let (k2, b2, ys, pg) = (kit.clone(), barrier.clone(), y.to_string(), pageable.clone());
                let victim = std::thread::spawn(move || {
                    k2.ctx.bind_to_thread().unwrap();
                    let s = k2.ctx.new_stream().unwrap();
                    s.synchronize().unwrap();
                    b2.wait();
                    std::thread::sleep(Duration::from_millis(40));
                    let y_ms = do_y(&k2, &s, &ys, &pg);
                    s.synchronize().unwrap();
                    y_ms
                });
                let x_ms = holder.join().unwrap();
                let y_ms = victim.join().unwrap();
                println!("PROBE x={x} y={y} mode=two-thread run={run} y_ms={y_ms:.2} x_ms={x_ms:.2}");
                kit.ctx.synchronize().unwrap();
                // Serial: one thread, the spin on stream A, X on A, then Y on stream B.
                kit.ctx.bind_to_thread().unwrap();
                let a = kit.ctx.new_stream().unwrap();
                let bstream = kit.ctx.new_stream().unwrap();
                let mut pre = if *x == "free-host" { Some(unsafe { result::malloc_host(4 << 20, 0) }.unwrap()) } else { None };
                a.synchronize().unwrap();
                launch_spin(&kit, &a, 300_000_000);
                std::thread::sleep(Duration::from_millis(20));
                let x_ms = do_x(&kit, &a, x, &mut pre);
                let y_ms = do_y(&kit, &bstream, y, &pageable);
                kit.ctx.synchronize().unwrap();
                if let Some(p) = pre.take() { unsafe { result::free_host(p) }.unwrap(); }
                let _ = sys::CUresult::CUDA_SUCCESS;
                println!("PROBE x={x} y={y} mode=serial run={run} y_ms={y_ms:.2} x_ms={x_ms:.2}");
            }
        }
    }
}
