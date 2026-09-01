//! Increment 4 gate: the MLA CUDA forward against the `memra_engine::mla` CPU f32 oracle.
//!
//! Three gates, all on device:
//!   1. PREFILL parity  — the absorbed-form GPU core vs `mla_attend_absorbed`/`mla_attend_naive`
//!      at full `MlaDims::GLM52` (rope 64) and `MlaDims::GLM5_NEXT` (NoPE, 64 heads x 256).
//!   2. DECODE parity   — prefill T rows into the latent cache, then t=1 steps; every step is
//!      compared against the oracle's FULL-SEQUENCE recompute at that position.
//!   3. BLOCK forward   — `HybridModel::forward` on the glm-dsa micro fixture actually runs.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing:
//!   flock /tmp/memra-5090.lock cargo test -p memra-engine --test mla_gpu_forward -- --ignored
//!
//! Every test below takes `gpu_guard()` first: these cases allocate enough device memory
//! that two of them running concurrently (cargo's default thread-per-test) fail on
//! allocation rather than on arithmetic — a gate that reports arithmetic failure for a
//! scheduling reason is worse than no gate. The guard makes the invocation above correct
//! without `--test-threads=1`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use memra_gguf::micro_gguf::write_glm_dsa_micro;

// BEFORE-RECEIPT gate: the glm-dsa micro fixture must run a real prefill through
// `Mixer::Mla`. Before increment 4 this panics in `mla_forward_unimplemented()`.

/// Process-wide GPU serialization for the cases in this file. `flock` serializes across
/// PROCESSES; this serializes the threads inside one test binary.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
#[allow(clippy::unusual_byte_groupings)] // allow: mnemonic grouping of a pinned seed/magic constant
fn gpu_glm_dsa_micro_block_forward_runs() {
    let _gpu = gpu_guard();
    let p = std::env::temp_dir().join(format!("memra-mla-fwd-gpu-{}.gguf", std::process::id()));
    write_glm_dsa_micro(&p, 0x4_F0_2026).unwrap();
    let g = GgufFile::open(&p).unwrap();
    let e = Engine::new(0).expect("CUDA device 0");
    let model = HybridModel::load(&e, &g).expect("glm-dsa micro fixture loads");
    std::fs::remove_file(&p).ok();

    let tokens: Vec<u32> = vec![1, 5, 9, 13, 17, 21];
    let logits = model.forward(&e, &tokens).expect("glm-dsa micro prefill");
    let n_vocab = model.cfg.n_vocab as usize;
    assert_eq!(logits.len(), tokens.len() * n_vocab, "logits shape");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "MLA forward produced non-finite logits"
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-4,
        "logits degenerate (flat) — a projection is dead"
    );
}

// ---------------------------------------------------------------- parity harness

use memra_engine::mla::{
    MlaDims, MlaInputs, mla_attend_absorbed, mla_attend_naive, rope_interleaved,
};
use memra_gguf::micro_gguf::Rng;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn maxabs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

/// Random MLA inputs at the oracle's own unit-ish scale (weights ~ 1/sqrt(rank)), so scores and
/// decompressed values stay O(1) and an absolute f32 tolerance is meaningful.
struct Case {
    q_nope: Vec<f32>,
    q_pe: Vec<f32>,
    c_kv: Vec<f32>,
    k_pe: Vec<f32>,
    w_uk: Vec<f32>,
    w_uv: Vec<f32>,
}

impl Case {
    fn new(d: &MlaDims, t_q: usize, t_kv: usize, seed: u64) -> Self {
        let mut rng = Rng(seed | 1);
        let ws = 1.0 / (d.kv_rank as f32).sqrt();
        Case {
            q_nope: rng.fill(t_q * d.n_head * d.d_nope, 1.0),
            q_pe: rng.fill(t_q * d.n_head * d.d_rope, 1.0),
            c_kv: rng.fill(t_kv * d.kv_rank, 1.0),
            k_pe: rng.fill(t_kv * d.d_rope, 1.0),
            w_uk: rng.fill(d.n_head * d.d_nope * d.kv_rank, ws),
            w_uv: rng.fill(d.n_head * d.d_v * d.kv_rank, ws),
        }
    }

    fn inputs<'a>(&'a self, t_q: usize, t_kv: usize) -> MlaInputs<'a> {
        MlaInputs {
            q_nope: &self.q_nope,
            q_pe: &self.q_pe,
            c_kv: &self.c_kv,
            k_pe: &self.k_pe,
            w_uk: &self.w_uk,
            w_uv: &self.w_uv,
            t_q,
            t_kv,
        }
    }

    /// Latent cache rows in the DEVICE layout: `[c_kv | k_pe]` per token, `kv_rank + d_rope` wide.
    /// This is `StatePlan::LatentKvCache { width }` and `MlaGeom::latent_dim`.
    fn cache_rows(&self, d: &MlaDims, t_kv: usize) -> Vec<f32> {
        let mut rows = Vec::with_capacity(t_kv * (d.kv_rank + d.d_rope));
        for t in 0..t_kv {
            rows.extend_from_slice(&self.c_kv[t * d.kv_rank..(t + 1) * d.kv_rank]);
            rows.extend_from_slice(&self.k_pe[t * d.d_rope..(t + 1) * d.d_rope]);
        }
        rows
    }

    /// `attn_k_b` in the CHECKPOINT layout, ne {d_nope, kv_rank, n_head}: element (h, l, p) at
    /// `h*kv_rank*d_nope + l*d_nope + p`. mla.rs stores `w_uk` as [h][p][l], so this transposes.
    fn wk_b(&self, d: &MlaDims) -> Vec<f32> {
        let (nh, dn, r) = (d.n_head, d.d_nope, d.kv_rank);
        let mut w = vec![0.0f32; nh * r * dn];
        for h in 0..nh {
            for l in 0..r {
                for p in 0..dn {
                    w[h * r * dn + l * dn + p] = self.w_uk[h * dn * r + p * r + l];
                }
            }
        }
        w
    }
}

/// Run the three core kernels (absorb -> absorbed attention -> decompress) and compare the
/// per-head attention output against BOTH oracle forms.
fn gpu_core_parity(e: &Engine, d: &MlaDims, t_q: usize, t_kv: usize, seed: u64, tol: f32) {
    let (nh, dn, dr, dv, r) = (d.n_head, d.d_nope, d.d_rope, d.d_v, d.kv_rank);
    let c = Case::new(d, t_q, t_kv, seed);
    let x = c.inputs(t_q, t_kv);
    let want_absorbed = mla_attend_absorbed(d, &x);
    let want_naive = mla_attend_naive(d, &x);

    let q_nope = e.htod(&c.q_nope).unwrap();
    // NoPE: the rope planes are empty; allocate one element so no slice is null-backed.
    let q_pe = e
        .htod(if dr == 0 { &[0.0f32][..] } else { &c.q_pe })
        .unwrap();
    let cache = e.htod(&c.cache_rows(d, t_kv)).unwrap();
    let wk_b = e.htod(&c.wk_b(d)).unwrap();
    // attn_v_b ne {kv_rank, d_v, n_head} is ALREADY the mla.rs [h][j][l] order — no transpose.
    let wv_b = e.htod(&c.w_uv).unwrap();

    let mut q_lat = e.uninit(t_q * nh * r).unwrap();
    e.mla_absorb_q(&q_nope, &wk_b, &mut q_lat, t_q, nh, dn, r)
        .unwrap();
    let mut o_lat = e.uninit(t_q * nh * r).unwrap();
    e.mla_attn_absorbed(
        &q_lat,
        &q_pe,
        &cache,
        &mut o_lat,
        nh,
        r,
        dr,
        t_q,
        t_kv,
        d.scale(),
    )
    .unwrap();
    let mut out = e.uninit(t_q * nh * dv).unwrap();
    e.mla_decompress_v(&o_lat, &wv_b, &mut out, t_q, nh, dv, r)
        .unwrap();
    let got = e.dtoh(&out).unwrap();

    assert!(
        got.iter().all(|v| v.is_finite()),
        "GPU MLA core produced non-finite values (dims {d:?} t_q {t_q} t_kv {t_kv})"
    );
    let scale = maxabs(&want_absorbed).max(1.0);
    for (name, want) in [("absorbed", &want_absorbed), ("naive", &want_naive)] {
        let md = maxdiff(&got, want);
        assert!(
            md <= tol * scale,
            "GPU MLA core vs CPU {name} oracle: maxdiff {md:.3e} (scale {scale:.3e}, \
             rel {:.3e}, tol {tol:.1e}) dims {d:?} t_q {t_q} t_kv {t_kv} seed {seed}",
            md / scale
        );
    }
    assert!(
        maxabs(&got) > 1e-6,
        "GPU MLA core output is degenerate (all ~zero)"
    );
}

/// GLM-5.2 geometry (rope 64) — the glm-dsa door.
fn glm52() -> MlaDims {
    MlaDims::GLM52
}
/// glm5_next / GLM-5.3-Flash geometry (NoPE: rope 0, 64 heads x 256, v 256, rank 512).
fn glm5_next() -> MlaDims {
    MlaDims::GLM5_NEXT
}

/// GATE 1 — PREFILL parity. Causal prefill (t_q == t_kv) and the chunked shape (queries a
/// suffix of a populated cache), at both production geometries plus shrunk shapes that make a
/// layout bug obvious before the 64-head cases run.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_prefill_parity_vs_cpu_oracle() {
    let _gpu = gpu_guard();
    let e = Engine::new(0).expect("CUDA device 0");

    // shrunk shapes first: both rope and NoPE, several seeds.
    let small = [
        MlaDims {
            n_head: 4,
            d_nope: 24,
            d_rope: 8,
            d_v: 32,
            kv_rank: 64,
        },
        MlaDims {
            n_head: 3,
            d_nope: 12,
            d_rope: 4,
            d_v: 20,
            kv_rank: 48,
        },
        MlaDims {
            n_head: 4,
            d_nope: 32,
            d_rope: 0,
            d_v: 32,
            kv_rank: 64,
        },
    ];
    for (i, d) in small.iter().enumerate() {
        for seed in [7u64, 1234, 0xB1E55ED] {
            gpu_core_parity(&e, d, 6, 6, seed + i as u64, 1e-5); // pure prefill
            gpu_core_parity(&e, d, 3, 11, seed + 5 + i as u64, 1e-5); // chunked
        }
    }

    // Production geometries. Wider accumulations (576/512-dot, rank-512 decompress) and the
    // kernel's tiled online softmax reorder the sum vs the oracle's single pass ⇒ looser bound.
    for d in [glm52(), glm5_next()] {
        gpu_core_parity(&e, &d, 4, 4, 20260827, 1e-4);
        gpu_core_parity(&e, &d, 3, 9, 20260828, 1e-4);
        gpu_core_parity(&e, &d, 17, 17, 20260829, 1e-4); // spans several softmax tiles
    }
}

/// GATE 2 — STEPWISE DECODE parity through the real latent cache. Rows are appended with the
/// append kernel exactly as the forward arm does (prefill block, then one row per step), and
/// EVERY step is compared against the oracle's FULL-SEQUENCE recompute at that position — the
/// bar the task sets, and the one that catches a cache that drifts as it grows.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_decode_stepwise_parity_vs_cpu_oracle() {
    let _gpu = gpu_guard();
    let e = Engine::new(0).expect("CUDA device 0");
    for (d, tol) in [
        (
            MlaDims {
                n_head: 4,
                d_nope: 24,
                d_rope: 8,
                d_v: 32,
                kv_rank: 64,
            },
            1e-5,
        ),
        (
            MlaDims {
                n_head: 4,
                d_nope: 32,
                d_rope: 0,
                d_v: 32,
                kv_rank: 64,
            },
            1e-5,
        ),
        (glm52(), 1e-4),
        (glm5_next(), 1e-4),
    ] {
        let (nh, dn, dr, dv, r) = (d.n_head, d.d_nope, d.d_rope, d.d_v, d.kv_rank);
        let width = r + dr;
        let prefill = 5usize;
        let steps = 4usize;
        let total = prefill + steps;
        // ONE case holds the whole context; each decode step reads its own query rows out of it,
        // so the oracle recompute and the GPU cache see identical tensors by construction.
        let c = Case::new(&d, total, total, 0x5_7E9 + total as u64);
        let rows = c.cache_rows(&d, total);

        let mut cache = e.zeros(total * width).unwrap();
        let wk_b = e.htod(&c.wk_b(&d)).unwrap();
        let wv_b = e.htod(&c.w_uv).unwrap();

        // append helper: rows [lo, hi) of the c_kv / k_pe planes into the cache at slot lo
        let append = |lo: usize, hi: usize, cache: &mut _| {
            let n = hi - lo;
            let c_kv = e.htod(&c.c_kv[lo * r..hi * r]).unwrap();
            let k_pe = e
                .htod(if dr == 0 {
                    &[0.0f32][..]
                } else {
                    &c.k_pe[lo * dr..hi * dr]
                })
                .unwrap();
            e.mla_append_latent(cache, &c_kv, &k_pe, lo, n, r, dr)
                .unwrap();
        };
        append(0, prefill, &mut cache);
        // The appended plane must equal the host-built rows bit for bit — an append bug would
        // otherwise surface only as a soft maxdiff in the attention output.
        let got_rows = e.dtoh(&cache).unwrap();
        assert_eq!(
            &got_rows[..prefill * width],
            &rows[..prefill * width],
            "append_latent did not reproduce the [c_kv | k_pe] row layout (dims {d:?})"
        );

        for step in 0..steps {
            let pos = prefill + step;
            append(pos, pos + 1, &mut cache);
            let t_kv = pos + 1;

            // GPU: one query (the row just appended) against the whole cache.
            let q_nope = e
                .htod(&c.q_nope[pos * nh * dn..(pos + 1) * nh * dn])
                .unwrap();
            let q_pe = e
                .htod(if dr == 0 {
                    &[0.0f32][..]
                } else {
                    &c.q_pe[pos * nh * dr..(pos + 1) * nh * dr]
                })
                .unwrap();
            let mut q_lat = e.uninit(nh * r).unwrap();
            e.mla_absorb_q(&q_nope, &wk_b, &mut q_lat, 1, nh, dn, r)
                .unwrap();
            let mut o_lat = e.uninit(nh * r).unwrap();
            e.mla_attn_absorbed(
                &q_lat,
                &q_pe,
                &cache,
                &mut o_lat,
                nh,
                r,
                dr,
                1,
                t_kv,
                d.scale(),
            )
            .unwrap();
            let mut out = e.uninit(nh * dv).unwrap();
            e.mla_decompress_v(&o_lat, &wv_b, &mut out, 1, nh, dv, r)
                .unwrap();
            let got = e.dtoh(&out).unwrap();

            // Oracle: FULL-SEQUENCE recompute over 0..t_kv, taking the LAST position.
            let want = mla_attend_absorbed(
                &d,
                &MlaInputs {
                    q_nope: &c.q_nope[pos * nh * dn..(pos + 1) * nh * dn],
                    q_pe: &c.q_pe[pos * nh * dr..(pos + 1) * nh * dr],
                    c_kv: &c.c_kv[..t_kv * r],
                    k_pe: &c.k_pe[..t_kv * dr],
                    w_uk: &c.w_uk,
                    w_uv: &c.w_uv,
                    t_q: 1,
                    t_kv,
                },
            );
            let scale = maxabs(&want).max(1.0);
            let md = maxdiff(&got, &want);
            assert!(
                md <= tol * scale,
                "decode step {step} (pos {pos}, t_kv {t_kv}): maxdiff {md:.3e} \
                 (scale {scale:.3e}, rel {:.3e}, tol {tol:.1e}) dims {d:?}",
                md / scale
            );
        }
    }
}

/// GATE 3 — the interleaved ("NORM") rope kernel against `mla.rs::rope_interleaved`, which is
/// the rotation the GLM-5.2 projection chain applies. Pinned separately from the attention core
/// so a rope regression cannot hide inside an attention maxdiff.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_rope_interleaved_matches_cpu() {
    let _gpu = gpu_guard();
    let e = Engine::new(0).expect("CUDA device 0");
    const BASE: f32 = 8_000_000.0; // GLM-5.2 rope_theta
    for &d_rope in &[8usize, 64] {
        let (n_pos, n_vec) = (7usize, 5usize);
        let mut rng = Rng(0x0DE_2026);
        let host = rng.fill(n_pos * n_vec * d_rope, 1.0);
        let positions: Vec<i32> = (0..n_pos as i32).map(|p| p * 13).collect();

        let mut want = host.clone();
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for p in 0..n_pos {
            for v in 0..n_vec {
                let off = (p * n_vec + v) * d_rope;
                rope_interleaved(
                    &mut want[off..off + d_rope],
                    d_rope,
                    positions[p] as f32,
                    BASE,
                );
            }
        }

        let mut x = e.htod(&host).unwrap();
        let pos_d = e.htod_i32(&positions).unwrap();
        e.mla_rope_interleaved(&mut x, &pos_d, n_pos, n_vec, d_rope, BASE)
            .unwrap();
        let got = e.dtoh(&x).unwrap();
        let md = maxdiff(&got, &want);
        assert!(
            md <= 1e-4 * maxabs(&want).max(1.0),
            "interleaved rope d_rope {d_rope}: maxdiff {md:.3e}"
        );
    }

    // NoPE contract: d_rope == 0 is a NO-OP, never a zero-extent launch.
    let host = vec![1.0f32, -2.0, 3.0];
    let mut x = e.htod(&host).unwrap();
    let pos_d = e.htod_i32(&[0i32]).unwrap();
    e.mla_rope_interleaved(&mut x, &pos_d, 1, 1, 0, BASE)
        .unwrap();
    assert_eq!(e.dtoh(&x).unwrap(), host, "NoPE rope must not modify data");
}

/// GATE 4 — the CACHED arm (`mla_attn_cached`): the latent-cache plumbing itself.
///
/// Gates 1-3 drive the kernels directly, so none of them touches slot bookkeeping, the
/// `len`/`len_d` lock-step, or the plane borrow in the forward arm. This one primes the
/// micro fixture through the real `Cache` (prime = slot 0, t = P) and then takes one decode
/// step (slot = P, t = 1), and requires the result to match the STATELESS `forward` over the
/// same token sequence — the two arms share the core, so any disagreement is cache plumbing.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
#[allow(clippy::unusual_byte_groupings)] // allow: mnemonic grouping of a pinned seed/magic constant
fn gpu_mla_cached_prime_decode_matches_stateless_forward() {
    let _gpu = gpu_guard();
    let p = std::env::temp_dir().join(format!("memra-mla-cached-{}.gguf", std::process::id()));
    write_glm_dsa_micro(&p, 0xC_ACE_0802).unwrap();
    let g = GgufFile::open(&p).unwrap();
    let e = Engine::new(0).expect("CUDA device 0");
    let model = HybridModel::load(&e, &g).expect("glm-dsa micro fixture loads");
    std::fs::remove_file(&p).ok();

    // `prime_cache` asserts T >= PRIME_MIN_T (16) — the prime path is gated by its callers,
    // so the primed prefix must clear that floor. n_vocab is 128 in the micro fixture.
    let tokens: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % 128).collect();
    let n_vocab = model.cfg.n_vocab as usize;

    // Reference: stateless full-sequence prefill, last row.
    let full = model.forward(&e, &tokens).expect("stateless MLA prefill");
    let want = &full[(tokens.len() - 1) * n_vocab..];

    // Cached: prime tokens[..n-1], then decode the last token.
    let mut cache = memra_kv::Cache::new(&e, &model.cfg, 64).expect("latent cache allocates");
    for (il, layer) in cache.latent.iter().enumerate() {
        assert!(
            layer.is_some(),
            "layer {il}: StatePlan::LatentKvCache did not allocate a latent plane"
        );
    }
    let (_prime_logits, _, _) = model
        .prime_cache(&e, &tokens[..tokens.len() - 1], &mut cache, 0)
        .expect("MLA prime through the latent cache");
    // Only the TRUNK layers are walked by a prime: the fixture's last block is the MTP head
    // (block_count = n_trunk + nextn), which owns a latent plane of its own that the trunk
    // walk never touches. Asserting over every plane would fail on that one by construction.
    let observed: Vec<usize> = cache
        .latent
        .iter()
        .map(|l| l.as_ref().unwrap().len)
        .collect();
    for il in 0..model.layers.len() {
        assert_eq!(
            observed[il],
            tokens.len() - 1,
            "trunk layer {il}: latent plane length after prime (all planes: {observed:?})"
        );
    }
    for il in model.layers.len()..observed.len() {
        assert_eq!(
            observed[il], 0,
            "MTP block {il}: the trunk prime must not append to the MTP latent plane \
             (all planes: {observed:?})"
        );
    }
    let (got, _) = model
        .decode_step_h(&e, tokens[tokens.len() - 1], &mut cache)
        .expect("MLA T=1 decode through the latent cache");

    assert_eq!(got.len(), n_vocab);
    assert!(got.iter().all(|v| v.is_finite()));
    // BAR: the glm_dsa pack's own `CheckpointParityGate` (max_abs 0.005, max_rel 0.005,
    // require_argmax) — the house threshold for a WHOLE-STACK comparison, not the 1e-4
    // kernel-level bound gates 1-3 use.
    //
    // Why the kernel bound is the wrong one here: the two arms are not the same floating-point
    // computation. Stateless runs every projection as one T=20 GEMM; cached runs T=19 then
    // T=1, and a cuBLAS GEMM's reduction order depends on M. That divergence enters at wq_a /
    // wkv_a / wo and every FFN+MoE GEMM, and compounds over the stack — it is prefill-vs-decode
    // drift, not MLA error, which is exactly what `bin/t2probe` exists to measure. The MLA core
    // itself is held to 1e-4 against the CPU oracle by gates 1-2, including through the real
    // append path, so this gate's job is that the CACHE PLUMBING agrees, at the bar the pack
    // declares for the model.
    let scale = maxabs(want).max(1.0);
    let md = maxdiff(&got, want);
    assert!(
        md <= 0.005 && md <= 0.005 * scale,
        "cached prime+decode vs stateless forward: maxdiff {md:.3e} (scale {scale:.3e}, \
         rel {:.3e}) exceeds the glm_dsa checkpoint-parity bar 5e-3",
        md / scale
    );
    // argmax is the shape the serving path actually consumes.
    let am = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                if x > b.1 { (i, x) } else { b }
            })
            .0
    };
    assert_eq!(am(&got), am(want), "cached vs stateless argmax disagree");
}
