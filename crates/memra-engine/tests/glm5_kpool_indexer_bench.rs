//! STAGE BENCH for the glm5_next DSA k-pool indexer — pool-key build, score, selection, and the
//! gathered attend, timed separately at production shapes.
//!
//! NOT A GATE and NOT A CORRECTNESS TEST. It launches the four indexer kernels on synthetic
//! device buffers — no model, no checkpoint — because what it measures is kernel time as a
//! function of SHAPE, and the shapes are what the product claim rests on (`index_topk` 2048,
//! `index_kpool` 4, 32 indexer heads, head_dim 128, hidden 4096, contexts up to 1M).
//!
//! ## RIG LAW — THE NUMBERS THIS PRINTS ON A LAPTOP 5090 ARE INVALID.
//!
//! The development rig's 5090 is a MOBILE part that thermally throttles to roughly half its clock
//! under sustained load, and house law reserves it for lock-serialized CORRECTNESS work only. Any
//! timing it produces is noise dressed as a measurement and MUST NOT be quoted, banked in a lane
//! receipt, or compared against another box's numbers. Run this on a proper serving-class box:
//!
//! ```text
//! cargo test -p memra-engine --release --test glm5_kpool_indexer_bench -- --ignored --nocapture
//! ```
//!
//! `--release` is not optional: a debug-profile host loop mismeasures the launch overhead that
//! dominates the small shapes. Every reported number is the MEDIAN of `TRIALS` timed repetitions
//! after `WARMUP` untimed ones. Each repetition is host wall time around ONE launch plus a
//! `stream.synchronize()`, so it includes launch overhead and one sync — coarser than CUDA events,
//! and deliberately so: it is the number a caller actually pays per stage.
//!
//! `free_device_mb` reads `nvidia-smi`, which reports the CUDA async pool's RETAINED memory as
//! used. On a small card a late rung of the ladder can therefore SKIP even though the allocator
//! would have recycled enough — run the ladder in a fresh process if a rung skips unexpectedly.
//!
//! ## WHAT IT SEPARATES, AND WHY THE POOL-KEY STAGE IS TWO NUMBERS
//!
//! `build_cold` rebuilds every pool key in the cache — what a cold prime pays, and what EVERY
//! call paid before the pool-key plane became resident. `build_step` builds only the pools one
//! decode step completes (at `pool` 4, either zero or one pool), which is what a resident plane
//! actually costs per step. Reporting only the cold number would hide the entire residency win;
//! reporting only the step number would hide what a prime still pays. Both are printed.
//!
//! `select` is the radix selection. It is the stage the shipped kernel changed
//! (`O(select_k * n_pools / threads)` -> `O(8 * n_pools / threads)`), so it is the one to watch
//! against context length: it should now grow LINEARLY in `n_pools` and be FLAT in `select_k`.
//!
//! `score` is likewise TWO numbers, for the same reason the pool-key stage is. `score_ref` is the
//! retained block-per-(query, pool) kernel — what every call paid before the tiled scorer landed,
//! and the arithmetic the tiled one is gated bit-identical to
//! (`glm5_kpool_indexer_gpu::gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel`).
//! `score` is the shipped tiled kernel. Reporting only the new number would leave the win
//! unpriced and would hide a regression that made the fast path fall back; reporting only the old
//! one would measure a kernel nothing calls. Both are printed, in the SAME run on the SAME box,
//! so the ratio is a measurement rather than a comparison across two boxes' clocks. `score_ref`
//! is the expensive stage of this bench (~1.3 s per launch at 1M/512 on a serving-class box,
//! x14 repetitions), and it is why the ladder takes minutes rather than seconds.
//!
//! ## MACHINE-READABLE OUTPUT
//!
//! One line per (context, t_q) shape, prefixed `KPOOL_BENCH`, `key=value` fields, space separated:
//!
//! ```text
//! KPOOL_BENCH ctx=65536 t_q=1 pools=16384 select_k=512 heads=32 d=128 kv_rank=512 \
//!   build_cold_ms=... build_step_ms=... score_ref_ms=... score_ms=... select_ms=... \
//!   attend_ms=... warmup=3 trials=11
//! ```
//!
//! A shape whose buffers do not fit the visible device prints `KPOOL_BENCH_SKIP ctx=... need_mb=...
//! free_mb=...` and moves on, so the ladder runs on an 80 GB box and a 24 GB box alike.

use memra_engine::Engine;

/// Production geometry: glm5_next's indexer, one MLA layer.
const HEADS: usize = 32;
const D: usize = 128;
const INDEX_TOPK: usize = 2048;
const POOL: usize = 4;
/// The MLA latent row the gathered attend walks: kv_lora_rank 512, NoPE so d_rope 0.
const KV_RANK: usize = 512;
const D_ROPE: usize = 0;
/// MLA query heads (not indexer heads).
const N_HEAD: usize = 8;

/// Context ladder. 1M is the product claim; the ladder stops where the device stops.
const CONTEXTS: [usize; 5] = [4_096, 16_384, 65_536, 262_144, 1_048_576];
/// Query counts: 1 is decode, 512 is a prefill chunk.
const T_QS: [usize; 2] = [1, 512];

const WARMUP: usize = 3;
const TRIALS: usize = 11;

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    v[v.len() / 2]
}

/// Deterministic filler — the bench measures shape, not values, but constant buffers would let a
/// future kernel take a data-dependent shortcut and flatter itself.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as f32 / (1u64 << 31) as f32 - 0.5
        })
        .collect()
}

/// Median wall time of `f`, in milliseconds, after `WARMUP` untimed runs. The stream is
/// synchronized inside the timed region, so what is measured is kernel completion, not launch.
fn timed(e: &Engine, mut f: impl FnMut()) -> f32 {
    for _ in 0..WARMUP {
        f();
    }
    e.stream().synchronize().expect("warmup sync");
    let mut ms = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let t0 = std::time::Instant::now();
        f();
        e.stream().synchronize().expect("trial sync");
        ms.push(t0.elapsed().as_secs_f32() * 1e3);
    }
    median(ms)
}

#[test]
#[ignore = "bench: needs a CUDA device, and its numbers are INVALID on the throttling dev rig — \
            run on a serving-class box with --release"]
fn bench_kpool_indexer_stages() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let select_k_full = INDEX_TOPK / POOL;
    println!(
        "# glm5_next DSA k-pool indexer, stage bench. heads={HEADS} d={D} pool={POOL} \
         index_topk={INDEX_TOPK} kv_rank={KV_RANK} n_head={N_HEAD} warmup={WARMUP} trials={TRIALS}"
    );
    println!("# RIG LAW: timings from the laptop 5090 are invalid — serving-class box only.");

    for &ctx in &CONTEXTS {
        let n_pools = ctx / POOL;
        let select_k = select_k_full.min(n_pools);
        let state_width = 2 * D;
        let latent_width = KV_RANK + D_ROPE;

        for &t_q in &T_QS {
            if t_q > ctx {
                continue;
            }
            let width = select_k * POOL + POOL - 1;
            // f32 elements, then bytes: state plane + pool keys + latent + q/score/idx scratch.
            let elems = ctx * state_width          // indexer state rows
                + n_pools * D                      // resident pool keys
                + ctx * latent_width               // MLA latent plane
                + t_q * HEADS * D                  // indexer queries
                + t_q * HEADS                      // head weights
                + t_q * n_pools                    // score row
                + t_q * N_HEAD * KV_RANK * 2       // q_lat + o_lat
                + POOL * D; // ape
            let need_mb = (elems * 4 + t_q * width * 4) / (1024 * 1024);
            let free_mb = free_device_mb();
            // Leave a quarter of free memory as headroom for the allocator's own churn.
            if need_mb + need_mb / 4 > free_mb {
                println!(
                    "KPOOL_BENCH_SKIP ctx={ctx} t_q={t_q} need_mb={need_mb} free_mb={free_mb}"
                );
                continue;
            }

            let state = e.htod(&noise(ctx * state_width, 0x51A7)).expect("state");
            let ape = e.htod(&noise(POOL * D, 0x1234)).expect("ape");
            let mut pool_keys = e.uninit(n_pools * D).expect("pool keys");
            let q = e.htod(&noise(t_q * HEADS * D, 0x9F13)).expect("q");
            let hw = e.htod(&noise(t_q * HEADS, 0x2B41)).expect("head weights");
            let mut score = e.uninit(t_q * n_pools).expect("score");
            let mut idx = e.uninit_i32(t_q * width).expect("index list");
            let latent = e.htod(&noise(ctx * latent_width, 0x77C1)).expect("latent");
            let q_lat = e
                .htod(&noise(t_q * N_HEAD * KV_RANK, 0x3311))
                .expect("q_lat");
            let q_pe = e.uninit((t_q * N_HEAD * D_ROPE).max(1)).expect("q_pe");
            let mut o_lat = e.uninit(t_q * N_HEAD * KV_RANK).expect("o_lat");
            let qk_scale = (D as f32).powf(-0.5);
            let head_scale = (HEADS as f32).powf(-0.5);
            let attn_scale = (KV_RANK as f32).powf(-0.5);

            // COLD build: every pool key in the cache, the pre-residency per-call cost.
            let build_cold = timed(&e, || {
                e.mla_kpool_pool_keys(&state, &ape, &mut pool_keys, 0, n_pools, POOL, D, 0)
                    .expect("cold pool-key build");
            });
            // STEP build: only the pools this call's `t_q` tokens completed — the resident cost.
            let step_begin = n_pools.saturating_sub(t_q.div_ceil(POOL));
            let build_step = timed(&e, || {
                e.mla_kpool_pool_keys(
                    &state,
                    &ape,
                    &mut pool_keys,
                    step_begin,
                    n_pools,
                    POOL,
                    D,
                    0,
                )
                .expect("incremental pool-key build");
            });
            // The pre-tiling scorer, timed in the same run so the ratio is one box's clock.
            let score_ref_ms = timed(&e, || {
                e.mla_kpool_score_ref(
                    &q,
                    &pool_keys,
                    &hw,
                    &mut score,
                    t_q,
                    HEADS,
                    D,
                    n_pools,
                    POOL,
                    ctx - t_q,
                    qk_scale,
                    head_scale,
                )
                .expect("reference score");
            });
            let score_ms = timed(&e, || {
                e.mla_kpool_score(
                    &q,
                    &pool_keys,
                    &hw,
                    &mut score,
                    t_q,
                    HEADS,
                    D,
                    n_pools,
                    POOL,
                    ctx - t_q,
                    qk_scale,
                    head_scale,
                )
                .expect("score");
            });
            let select_ms = timed(&e, || {
                e.mla_kpool_select(
                    &score,
                    &mut idx,
                    t_q,
                    n_pools,
                    POOL,
                    select_k,
                    width,
                    ctx - t_q,
                    true,
                )
                .expect("radix selection");
            });
            let attend_ms = timed(&e, || {
                e.mla_attn_gathered(
                    &q_lat, &q_pe, &latent, &idx, &mut o_lat, N_HEAD, KV_RANK, D_ROPE, t_q, width,
                    attn_scale,
                )
                .expect("gathered attend");
            });

            println!(
                "KPOOL_BENCH ctx={ctx} t_q={t_q} pools={n_pools} select_k={select_k} \
                 heads={HEADS} d={D} kv_rank={KV_RANK} width={width} \
                 build_cold_ms={build_cold:.4} build_step_ms={build_step:.4} \
                 score_ref_ms={score_ref_ms:.4} score_ms={score_ms:.4} \
                 select_ms={select_ms:.4} attend_ms={attend_ms:.4} \
                 warmup={WARMUP} trials={TRIALS}"
            );
        }
    }
}

/// Free VRAM in MiB, so the ladder can stop where the device stops instead of OOM-ing the run.
fn free_device_mb() -> usize {
    match std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
    {
        Ok(out) => String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or(0),
        // No nvidia-smi: assume plenty and let a real OOM speak for itself rather than skipping
        // every shape silently (a bench that reports nothing looks like a bench that ran).
        Err(_) => usize::MAX / 2,
    }
}
