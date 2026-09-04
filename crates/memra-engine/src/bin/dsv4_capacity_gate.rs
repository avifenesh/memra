//! DSV4 two-card native-context allocation gate.
//!
//! Loads the exact artifact at 1M model context with bundled DSpark, then allocates
//! one full-capacity compact trunk state, DSpark session state, and width-32 prefill
//! transaction workspace. This is a reachability gate, not serving or performance.

use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi;
use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::FixtureSpec;
use std::path::Path;

fn gate_stream_topk(gpu: &Dsv4Gpu) {
    let stage = &gpu.stages[0];
    stage
        .gpu
        .ctx
        .bind_to_thread()
        .expect("bind stream-topk gate context");
    let stream = stage.gpu.stream();
    let s = 3usize;
    let topk = 512usize;
    let win = 128usize;
    let idx_stride = win + topk;
    for nb in [4_103usize, 16_397, 250_003] {
        let scores: Vec<f32> = (0..s * nb)
            .map(|z| {
                let row = z / nb;
                let col = z % nb;
                ((col.wrapping_mul(2_654_435_761usize) + row * 97) % 1_009) as f32
            })
            .collect();
        let work_stride = nb.div_ceil(4096) * 512;
        let mut score_d = stream
            .alloc_zeros::<f32>(scores.len())
            .expect("stream-topk score alloc");
        stream
            .memcpy_htod(&scores, &mut score_d)
            .expect("stream-topk score upload");
        let mut idx_d = stream
            .alloc_zeros::<i32>(s * idx_stride)
            .expect("stream-topk idx alloc");
        let mut work_a = stream
            .alloc_zeros::<u64>(s * work_stride)
            .expect("stream-topk work-a alloc");
        let mut work_b = stream
            .alloc_zeros::<u64>(s * work_stride)
            .expect("stream-topk work-b alloc");
        let rc = unsafe {
            dsv4_ffi::memra_dsv4_topk_idx_stream_m(
                score_d.device_ptr(&stream).0 as *const f32,
                s as i32,
                nb as i32,
                topk as i32,
                win as i32,
                idx_d.device_ptr_mut(&stream).0 as *mut i32,
                idx_stride as i32,
                work_a.device_ptr_mut(&stream).0 as *mut u64,
                work_b.device_ptr_mut(&stream).0 as *mut u64,
                work_stride as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, 0, "stream-topk nb={nb} rc={rc}");
        let mut got = vec![0i32; s * idx_stride];
        stream
            .memcpy_dtoh(&idx_d, &mut got)
            .expect("stream-topk idx download");
        stream.synchronize().expect("stream-topk synchronize");
        for row in 0..s {
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by(|&a, &b| {
                scores[row * nb + b]
                    .partial_cmp(&scores[row * nb + a])
                    .expect("finite synthetic score")
                    .then(a.cmp(&b))
            });
            let expected: Vec<i32> = order[..topk].iter().map(|&j| (j + win) as i32).collect();
            assert_eq!(
                &got[row * idx_stride + win..row * idx_stride + win + topk],
                expected,
                "stream-topk mismatch nb={nb} row={row}"
            );
        }
        println!("[dsv4-capacity-gate] stream-topk nb={nb} rows={s} topk={topk} EXACT");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4_capacity_gate <model-dir> <fixtures.json> [dev0,dev1]");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let fixture = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|raw| {
            raw.split(',')
                .map(|part| part.parse().expect("device index"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1]);
    let max_seq = 1_048_576usize;
    let chunk = 32usize;
    let gpu = Dsv4Gpu::load(model_dir, &devices, fixture.variant, max_seq)
        .expect("load 1M DSV4 GPU model");
    assert!(
        gpu.dspark.is_some(),
        "capacity gate requires bundled DSpark"
    );
    let post_load = gpu.vram_report().expect("post-load VRAM report");
    let state = gpu
        .alloc_decode_state_for_transient(max_seq, chunk)
        .expect("allocate full 1M compact trunk state");
    let dstate = gpu
        .dspark_alloc_state()
        .expect("allocate DSpark session state");
    let vstate = gpu
        .alloc_prefill_state_for(max_seq, chunk)
        .expect("allocate width-32 1M prefill transaction state");
    let post_alloc = gpu.vram_report().expect("post-allocation VRAM report");
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "[dsv4-capacity-gate] placement split_at={} post_load={} post_alloc={} cache_gib=[{:.3},{:.3}] verify_gib=[{:.3},{:.3}]",
        gpu.split_at,
        post_load
            .iter()
            .map(|(dev, free, total, _)| format!(
                "dev{dev}:{:.3}/{:.3}GiB",
                gib(total - free),
                gib(*total)
            ))
            .collect::<Vec<_>>()
            .join(","),
        post_alloc
            .iter()
            .map(|(dev, free, total, _)| format!(
                "dev{dev}:{:.3}/{:.3}GiB",
                gib(total - free),
                gib(*total)
            ))
            .collect::<Vec<_>>()
            .join(","),
        gib(state.cache_bytes[0]),
        gib(state.cache_bytes[1]),
        gib(vstate.bytes[0]),
        gib(vstate.bytes[1]),
    );
    gate_stream_topk(&gpu);
    std::hint::black_box((&state, &dstate, &vstate));
    println!("[dsv4-capacity-gate] PASS 1M compact state + DSpark + chunk32 workspace");
}
