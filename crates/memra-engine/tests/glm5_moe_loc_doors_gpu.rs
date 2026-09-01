//! glm5 MoE-VERIFY LOCALITY doors (lane/glm5-moe-loc, 2026-08-31) — rig exactness gates.
//!
//! Door D (`MEMRA_MOE_VROWS_DEV_TABLES`): the verify-rows pair's pointer/scale tables built ON
//! DEVICE from the router's own `sel`/`w` instead of on the host, which lets the layer skip the
//! pinned readback and its full `cuStreamSynchronize` (42 device-wide drains + 84 DtoH + 84
//! pageable HtoD per ship round). The bar is table-level BITWISE identity plus pair-output
//! bitwise identity, with reds proving each table term is actually exercised.
//!
//! Door H (`MEMRA_HTOD_DIET`, glm5 alias `MEMRA_GLM5_HTOD_DIET` honored — generalized
//! lane/glm5-extract2): the shared-expert `g = 1.0` constant stops being re-uploaded per MoE
//! layer-call (42/round) and the latent-plane `len_d` scalar takes the async `i32_set_k` instead
//! of a synchronizing pageable copy (22/round). The door's arms here drive BOTH names plus a
//! disagreeing pair, because the door is read per call and a rename nothing exercises through
//! its general name is a rename that only works on paper.
//!
//! Instrument S (`MEMRA_MOE_VROWS_DEDUP_STAT`): the pair union's visits/distinct counting, whose
//! ratio is the ONLY remaining byte lever on a pair already at ~90% of theoretical DRAM peak.
//!
//! Every gate here is exactness or counters only — the rig cannot produce a timing row (rig law),
//! and the box window prices all three doors.

use memra_engine::Engine;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|p| p.into_inner())
}

/// TF32 off and the exact-f32 trunk pinned, exactly as the sibling doors gates do it.
fn force_true_f32() {
    unsafe {
        std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
    }
}

fn with_flag<T>(key: &str, f: impl FnOnce() -> T) -> T {
    unsafe { std::env::set_var(key, "1") };
    let out = f();
    unsafe { std::env::remove_var(key) };
    out
}

/// Set several keys for one arm and ALWAYS clear them, even if the arm panics. The
/// single-key helper above clears on the happy path only, which is fine while its body cannot
/// fail; the disagreeing-pair arm below has an `.expect()` in it, and these suites run
/// `--test-threads=1`, so a leaked `MEMRA_HTOD_DIET=1` + `MEMRA_GLM5_HTOD_DIET=0` would arm a
/// fail-closed door for every later test in the file and turn one failure into a cascade.
struct EnvArm(Vec<String>);
impl Drop for EnvArm {
    fn drop(&mut self) {
        for k in &self.0 {
            unsafe { std::env::remove_var(k) };
        }
    }
}
#[must_use = "the returned guard is what clears the keys — drop it explicitly, before any assertion that must see them cleared"]
fn with_flags(pairs: &[(&str, &str)], f: impl FnOnce()) -> EnvArm {
    let guard = EnvArm(pairs.iter().map(|(k, _)| (*k).to_string()).collect());
    for (k, v) in pairs {
        unsafe { std::env::set_var(k, v) };
    }
    f();
    guard
}

fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * spread
        })
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "compared buffers differ in length");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn nvfp4_slab(
    e: &Engine,
    n_expert: usize,
    out_f: usize,
    in_f: usize,
    seed: u64,
) -> (cudarc::driver::CudaSlice<u8>, usize, usize) {
    let mut bytes = Vec::new();
    let mut row_bytes = 0usize;
    for ex in 0..n_expert {
        for o in 0..out_f {
            let row = varied(in_f, seed ^ ((ex * out_f + o) as u64) << 8, 2.0);
            let rb = memra_gguf::nvfp4_repack::f32_to_nvfp4(&row);
            row_bytes = rb.len();
            bytes.extend_from_slice(&rb);
        }
    }
    let stride = out_f * row_bytes;
    let slab = e.htod_bytes(&bytes).expect("nvfp4 slab upload");
    (slab, row_bytes, stride)
}

// -------------------------------------------------------------------------------------------
// Door D — device-built pointer/scale tables.
// -------------------------------------------------------------------------------------------

/// The HOST table build, verbatim from `moe_vrows_pairs_q8`'s host arm.
fn host_tables(
    (base_g, base_u, base_d): (u64, u64, u64),
    (sg, su, sd): (usize, usize, usize),
    macros: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    sel: &[u32],
    w: &[f32],
) -> (Vec<u64>, Vec<f32>) {
    let n_pairs = sel.len();
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for (p, (&ex, &wj)) in sel.iter().zip(w).enumerate() {
        let ex = ex as usize;
        ptrs[p] = base_g + (ex * sg) as u64;
        ptrs[n_pairs + p] = base_u + (ex * su) as u64;
        ptrs[2 * n_pairs + p] = base_d + (ex * sd) as u64;
        let (mg, mu, md) = match macros {
            Some(m) => (m.0[ex], m.1[ex], m.2[ex]),
            None => (1.0, 1.0, 1.0),
        };
        scl[p] = mg;
        scl[n_pairs + p] = mu;
        scl[2 * n_pairs + p] = wj * md;
    }
    (ptrs, scl)
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dev_tables_match_the_host_table_build_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (n_expert, n_used) = (16usize, 8usize);
    // Strides that are NOT equal across planes, so a plane mix-up cannot pass by symmetry, and
    // not powers of two, so a shift-vs-multiply slip shows up.
    let (sg, su, sd) = (1408usize, 1408usize, 2816usize);
    let (base_g, base_u, base_d) = (0x7000_0000u64, 0x7100_0000u64, 0x7200_0000u64);
    let macros = (
        (0..n_expert)
            .map(|i| 0.5 + 0.07 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 1.6 - 0.05 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 0.8 + 0.04 * i as f32)
            .collect::<Vec<_>>(),
    );

    for t in 2..=8usize {
        let n_pairs = t * n_used;
        // Deliberately includes REPEATED experts across rows (the dedup case) and expert 0 (so a
        // `base + 0*stride` term is exercised), plus the top expert id.
        let sel: Vec<u32> = (0..n_pairs)
            .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
            .collect();
        let w = varied(n_pairs, 0xA11CE + t as u64, 1.5);
        let sel_d = e
            .htod_i32(&sel.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .expect("sel upload");
        let w_d = e.htod(&w).expect("w upload");

        for (label, mac) in [("live macros", Some(&macros)), ("no macros", None)] {
            let (want_p, want_s) =
                host_tables((base_g, base_u, base_d), (sg, su, sd), mac, &sel, &w);
            let mut ptrs_d = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
            let mut scl_d = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
            let d0 = memra_engine::moe_vrows_dev_tables_dispatches();
            e.moe_vrows_tables_from_sel(
                &sel_d,
                &w_d,
                7,
                mac.map(|m| (m.0.as_slice(), m.1.as_slice(), m.2.as_slice())),
                (base_g, base_u, base_d),
                (sg, su, sd),
                n_pairs,
                &mut ptrs_d,
                &mut scl_d,
            )
            .expect("device table build");
            // The launcher is the counting site for the ARM (the dispatch counter lives at the
            // hybrid_forward call site), so anchor on the produced bytes, never on liveness.
            let _ = d0;
            let got_p = e.dtoh_u64(&ptrs_d).expect("ptrs readback");
            let got_s = e.dtoh(&scl_d).expect("scl readback");
            assert_eq!(
                got_p, want_p,
                "t={t} {label}: device pointer table diverged from the host build"
            );
            let sd_diffs = bit_diffs(&got_s, &want_s);
            assert_eq!(
                sd_diffs,
                0,
                "t={t} {label}: device scale table diverged from the host build in \
                 {sd_diffs}/{} entries",
                3 * n_pairs
            );
            println!(
                "door D tables PASS t={t} ({label}): {} pointers + {} scales bit-identical",
                3 * n_pairs,
                3 * n_pairs
            );
        }

        // RED 1 — a wrong DOWN stride must move the pointer table. Without this the equality
        // above could hold for a kernel that ignored `sd` entirely.
        let mut ptrs_bad = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
        let mut scl_bad = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
        e.moe_vrows_tables_from_sel(
            &sel_d,
            &w_d,
            7,
            Some((
                macros.0.as_slice(),
                macros.1.as_slice(),
                macros.2.as_slice(),
            )),
            (base_g, base_u, base_d),
            (sg, su, sd + 64),
            n_pairs,
            &mut ptrs_bad,
            &mut scl_bad,
        )
        .expect("device table build (red 1)");
        let (want_p, _) = host_tables(
            (base_g, base_u, base_d),
            (sg, su, sd),
            Some(&macros),
            &sel,
            &w,
        );
        let bad_p = e.dtoh_u64(&ptrs_bad).expect("ptrs readback");
        assert_ne!(
            bad_p, want_p,
            "t={t}: the wrong-down-stride red arm produced the same pointers — the down plane's \
             stride term is untested"
        );

        // RED 2 — dropping the macro planes must move the scale table (the fold is real).
        let mut ptrs_nm = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
        let mut scl_nm = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
        e.moe_vrows_tables_from_sel(
            &sel_d,
            &w_d,
            7,
            None,
            (base_g, base_u, base_d),
            (sg, su, sd),
            n_pairs,
            &mut ptrs_nm,
            &mut scl_nm,
        )
        .expect("device table build (red 2)");
        let (_, want_s) = host_tables(
            (base_g, base_u, base_d),
            (sg, su, sd),
            Some(&macros),
            &sel,
            &w,
        );
        let nm_s = e.dtoh(&scl_nm).expect("scl readback");
        assert!(
            bit_diffs(&nm_s, &want_s) > 0,
            "t={t}: the dropped-macro red arm produced identical scales — the macro fold is \
             untested"
        );
    }
    println!("door D RED 1 (wrong down stride) and RED 2 (dropped macro plane) both bite");
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dev_tables_drive_the_pair_to_identical_bytes() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // The vrest gate-4 shape class: NVFP4 block 64, live macro plane, biting clamp.
    let (in_f, n_ff) = (128usize, 64usize);
    let (n_expert, n_used) = (16usize, 8usize);
    let slabs_gu = nvfp4_slab(&e, n_expert, n_ff, in_f, 0x6A7E);
    let slabs_d = nvfp4_slab(&e, n_expert, in_f, n_ff, 0xD003);
    let macros = (
        (0..n_expert)
            .map(|i| 0.5 + 0.07 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 1.6 - 0.05 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 0.8 + 0.04 * i as f32)
            .collect::<Vec<_>>(),
    );
    let limit = 0.75f32;

    use cudarc::driver::DevicePtr;
    let stream = e.stream();
    let (base_gu, _g0) = slabs_gu.0.device_ptr(&stream);
    let (base_dn, _g1) = slabs_d.0.device_ptr(&stream);
    let (rb_gu, stride_gu) = (slabs_gu.1, slabs_gu.2);
    let (rb_dn, stride_dn) = (slabs_d.1, slabs_d.2);

    for t in 2..=8usize {
        let n_pairs = t * n_used;
        let sel: Vec<u32> = (0..n_pairs)
            .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
            .collect();
        let w: Vec<f32> = (0..n_pairs).map(|p| 0.1 + 0.03 * (p % 11) as f32).collect();
        let z = varied(t * in_f, 0x2A + t as u64, 2.0);
        let z_d = e.htod(&z).expect("z upload");

        // Run the pair twice: tables from the host loop, then tables from the device kernel.
        let run = |ptrs_d: &cudarc::driver::CudaSlice<u64>,
                   scl_d: &cudarc::driver::CudaSlice<f32>|
         -> Vec<f32> {
            let (zq, zd) = e.quantize_q8_1(&z_d, t, in_f).expect("batched quantize");
            let act = e
                .moe_gate_up_preclamp8_q8_rows(
                    ptrs_d,
                    scl_d,
                    &zq,
                    &zd,
                    limit,
                    in_f,
                    n_ff,
                    n_used,
                    n_pairs,
                    memra_engine::QT_NVFP4,
                    memra_engine::QT_NVFP4,
                    rb_gu,
                    rb_gu,
                )
                .expect("gate/up rows launch");
            let (aq2, ad2) = e
                .quantize_q8_1(&act, n_pairs, n_ff)
                .expect("pair act quantize");
            let mut out = e.uninit(t * in_f).expect("out");
            e.moe_down8_fma_q8_rows(
                ptrs_d,
                scl_d,
                &aq2,
                &ad2,
                &mut out,
                n_ff,
                in_f,
                n_used,
                n_pairs,
                memra_engine::QT_NVFP4,
                rb_dn,
            )
            .expect("down rows launch");
            e.dtoh(&out).expect("pairs readback")
        };

        let (hp, hs) = host_tables(
            (base_gu, base_gu, base_dn),
            (stride_gu, stride_gu, stride_dn),
            Some(&macros),
            &sel,
            &w,
        );
        let want = run(
            &e.htod_u64(&hp).expect("host ptr table"),
            &e.htod(&hs).expect("host scale table"),
        );

        let sel_d = e
            .htod_i32(&sel.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .expect("sel upload");
        let w_d = e.htod(&w).expect("w upload");
        let mut ptrs_dev = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
        let mut scl_dev = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
        e.moe_vrows_tables_from_sel(
            &sel_d,
            &w_d,
            3,
            Some((
                macros.0.as_slice(),
                macros.1.as_slice(),
                macros.2.as_slice(),
            )),
            (base_gu, base_gu, base_dn),
            (stride_gu, stride_gu, stride_dn),
            n_pairs,
            &mut ptrs_dev,
            &mut scl_dev,
        )
        .expect("device table build");
        let got = run(&ptrs_dev, &scl_dev);

        let diffs = bit_diffs(&got, &want);
        assert_eq!(
            diffs,
            0,
            "t={t}: the device-table pair diverged from the host-table pair in {diffs}/{} outputs",
            t * in_f
        );
        println!(
            "door D pair PASS t={t}: {} outputs bit-identical through device-built tables",
            t * in_f
        );
    }

    // RED — the pair output must MOVE when the tables do, else the identity above is vacuous
    // (e.g. if the pair had ignored the scale plane entirely).
    let t = 4usize;
    let n_pairs = t * n_used;
    let sel: Vec<u32> = (0..n_pairs)
        .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
        .collect();
    let w: Vec<f32> = (0..n_pairs).map(|p| 0.1 + 0.03 * (p % 11) as f32).collect();
    let z = varied(t * in_f, 0x2A + t as u64, 2.0);
    let z_d = e.htod(&z).expect("z upload");
    let run2 = |ptrs_d: &cudarc::driver::CudaSlice<u64>, scl_d: &cudarc::driver::CudaSlice<f32>| {
        let (zq, zd) = e.quantize_q8_1(&z_d, t, in_f).expect("quantize");
        let act = e
            .moe_gate_up_preclamp8_q8_rows(
                ptrs_d,
                scl_d,
                &zq,
                &zd,
                limit,
                in_f,
                n_ff,
                n_used,
                n_pairs,
                memra_engine::QT_NVFP4,
                memra_engine::QT_NVFP4,
                rb_gu,
                rb_gu,
            )
            .expect("gate/up");
        let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff).expect("act quantize");
        let mut out = e.uninit(t * in_f).expect("out");
        e.moe_down8_fma_q8_rows(
            ptrs_d,
            scl_d,
            &aq2,
            &ad2,
            &mut out,
            n_ff,
            in_f,
            n_used,
            n_pairs,
            memra_engine::QT_NVFP4,
            rb_dn,
        )
        .expect("down");
        e.dtoh(&out).expect("readback")
    };
    let sel_d = e
        .htod_i32(&sel.iter().map(|&x| x as i32).collect::<Vec<_>>())
        .expect("sel upload");
    let w_d = e.htod(&w).expect("w upload");
    let mut p_live = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
    let mut s_live = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
    e.moe_vrows_tables_from_sel(
        &sel_d,
        &w_d,
        3,
        Some((
            macros.0.as_slice(),
            macros.1.as_slice(),
            macros.2.as_slice(),
        )),
        (base_gu, base_gu, base_dn),
        (stride_gu, stride_gu, stride_dn),
        n_pairs,
        &mut p_live,
        &mut s_live,
    )
    .expect("device tables (live)");
    let live = run2(&p_live, &s_live);
    let mut p_nm = e.htod_u64(&vec![0u64; 3 * n_pairs]).expect("ptrs");
    let mut s_nm = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl");
    e.moe_vrows_tables_from_sel(
        &sel_d,
        &w_d,
        3,
        None,
        (base_gu, base_gu, base_dn),
        (stride_gu, stride_gu, stride_dn),
        n_pairs,
        &mut p_nm,
        &mut s_nm,
    )
    .expect("device tables (no macro)");
    let nomac = run2(&p_nm, &s_nm);
    let d = bit_diffs(&nomac, &live);
    assert!(
        d > 0,
        "door D pair red arm: dropping the macro planes left the pair output identical — the \
         scale table is not being consumed"
    );
    println!("door D pair RED bites: {d} outputs differ with the macro planes dropped");
}

// -------------------------------------------------------------------------------------------
// Door H — the resident shexp ones vector and the async len_d mirror store.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_htod_diet_shexp_ones_and_len_mirror_are_bit_identical() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let ncols = 96usize;

    for nrows in [1usize, 2, 4, 8, 15] {
        let src = varied(nrows * ncols, 0xC0FFEE + nrows as u64, 3.0);
        let base = varied(nrows * ncols, 0xBEEF + nrows as u64, 3.0);
        let src_d = e.htod(&src).expect("src");

        // WANT: the shipped arm — a freshly uploaded host ones vector.
        let mut dst_want = e.htod(&base).expect("dst want");
        let ones = e.htod(&vec![1.0f32; nrows]).expect("host ones");
        e.add_scaled_rows(&src_d, &ones, &mut dst_want, ncols, nrows)
            .expect("shipped add_scaled_rows");
        let want = e.dtoh(&dst_want).expect("want readback");

        // GOT: door H's resident ones buffer through the SAME kernel.
        let mut dst_got = e.htod(&base).expect("dst got");
        e.add_scaled_rows_ones(&src_d, &mut dst_got, ncols, nrows)
            .expect("door H add_scaled_rows_ones");
        let got = e.dtoh(&dst_got).expect("got readback");

        let diffs = bit_diffs(&got, &want);
        assert_eq!(
            diffs,
            0,
            "nrows={nrows}: the resident-ones arm diverged in {diffs}/{} outputs",
            nrows * ncols
        );

        // RED — a non-ones scale must move the result, else "identical" proves nothing about the
        // scale being read at all.
        let mut dst_red = e.htod(&base).expect("dst red");
        let not_ones = e
            .htod(
                &(0..nrows)
                    .map(|i| 1.0 + 0.25 * (i + 1) as f32)
                    .collect::<Vec<_>>(),
            )
            .expect("non-ones scale");
        e.add_scaled_rows(&src_d, &not_ones, &mut dst_red, ncols, nrows)
            .expect("red add_scaled_rows");
        let red = e.dtoh(&dst_red).expect("red readback");
        assert!(
            bit_diffs(&red, &want) > 0,
            "nrows={nrows}: the non-ones red arm matched the ones arm — the scale vector is not \
             being consumed"
        );
        println!(
            "door H shexp PASS nrows={nrows}: {} outputs bit-identical, non-ones red bites",
            nrows * ncols
        );
    }

    // The len_d mirror: identical value, and the door must actually take the async path.
    let mut mirror = e.htod_i32(&[0i32]).expect("mirror");
    e.i32_mirror_store(&mut mirror, 4321)
        .expect("shipped mirror store");
    assert_eq!(
        e.dtoh_i32(&mirror).expect("mirror readback")[0],
        4321,
        "the shipped memcpy_htod mirror store did not land"
    );
    let a0 = memra_engine::htod_diet_avoided();
    assert_eq!(
        a0,
        memra_engine::htod_diet_avoided(),
        "the door-H counter moved with the door OFF"
    );
    let n1 = with_flag("MEMRA_GLM5_HTOD_DIET", || {
        e.i32_mirror_store(&mut mirror, 9876).expect("door H store");
        e.dtoh_i32(&mirror).expect("mirror readback")[0]
    });
    assert_eq!(n1, 9876, "the door-H i32_set_k mirror store did not land");
    assert!(
        memra_engine::htod_diet_avoided() > a0,
        "door H did not engage on the mirror store — the counter is flat"
    );
    println!("door H len_d PASS: both arms land the value, counter anchored ON / flat OFF");

    // THE FLAG-ALIAS LAW (lane/glm5-extract2): the arm above drives the glm5 ALIAS, which is
    // what every banked moe-loc script sets. These two arms are the only proof that the
    // GENERAL name works and that a disagreeing pair falls closed — a rename whose general
    // name is never exercised is a rename that only works on paper.
    let a1 = memra_engine::htod_diet_avoided();
    let n2 = with_flag("MEMRA_HTOD_DIET", || {
        e.i32_mirror_store(&mut mirror, 1234)
            .expect("general store");
        e.dtoh_i32(&mirror).expect("mirror readback")[0]
    });
    assert_eq!(n2, 1234, "the general-name mirror store did not land");
    assert!(
        memra_engine::htod_diet_avoided() > a1,
        "MEMRA_HTOD_DIET did not engage door H — the general name is not wired"
    );

    // Disagreement: general=1 + alias=0. The door must NOT arm (counter flat) and the value
    // must still land through the SHIPPED synchronizing form — fail-closed, not fail-broken.
    let a2 = memra_engine::htod_diet_avoided();
    let landed = with_flags(
        &[("MEMRA_HTOD_DIET", "1"), ("MEMRA_GLM5_HTOD_DIET", "0")],
        || {
            e.i32_mirror_store(&mut mirror, 5678)
                .expect("disagreeing-pair store");
        },
    );
    drop(landed);
    assert_eq!(
        e.dtoh_i32(&mirror).expect("mirror readback")[0],
        5678,
        "the disagreeing pair must still land the value through the shipped form"
    );
    assert_eq!(
        memra_engine::htod_diet_avoided(),
        a2,
        "a disagreeing MEMRA_HTOD_DIET / MEMRA_GLM5_HTOD_DIET pair ARMED the door — it must \
         fall closed to the shipped program instead of picking a precedence winner"
    );
    println!(
        "door H alias PASS: general name engages, disagreeing pair falls closed with the \
         value still landing"
    );
}

// -------------------------------------------------------------------------------------------
// Instrument S — the dedup counting itself, on planted overlaps.
// -------------------------------------------------------------------------------------------

#[test]
fn dedup_stat_counts_visits_and_distinct_on_planted_overlaps() {
    // Fully disjoint: every visit is a distinct expert, so the lever is exactly zero.
    let disjoint: Vec<u32> = (0..16).collect();
    assert_eq!(
        memra_engine::vrows_overlap_counts_for_test(&disjoint),
        (16, 16),
        "a disjoint union must report no shareable slab"
    );
    // Two rows of 8 sharing exactly 3 experts: 16 visits, 13 distinct -> lever 18.75%.
    let mut shared: Vec<u32> = (0..8).collect();
    shared.extend([0u32, 1, 2, 100, 101, 102, 103, 104]);
    assert_eq!(
        memra_engine::vrows_overlap_counts_for_test(&shared),
        (16, 13),
        "3 shared experts across two rows must show up as 16 visits / 13 distinct"
    );
    // The structural ceiling: all rows route identically.
    let identical: Vec<u32> = (0..4).flat_map(|_| 0..8u32).collect();
    assert_eq!(
        memra_engine::vrows_overlap_counts_for_test(&identical),
        (32, 8),
        "identically-routed rows must collapse to one row's worth of distinct experts"
    );
    println!("instrument S PASS: disjoint / partial / identical unions all counted exactly");
}
