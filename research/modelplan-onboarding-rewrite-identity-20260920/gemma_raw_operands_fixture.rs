// CPU recording stand-ins. Production method bodies are inserted by the runner.
#![allow(dead_code, unused_variables)]
use std::cell::RefCell;
type CudaSlice<T> = Vec<T>;
type R<T> = Result<T, Box<dyn std::error::Error>>;
type Pair = (CudaSlice<f32>, CudaSlice<f32>);
type Triple = (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>);
mod memra_gguf {
    pub mod execution_manifest {
        pub enum RewriteSurface {
            DecodeEager,
        }
    }
}
struct Sampler {
    greedy: bool,
    penalty: usize,
}
impl Sampler {
    fn is_greedy(&self) -> bool {
        self.greedy
    }
    fn penalty_last_n(&self) -> usize {
        self.penalty
    }
}
mod model {
    pub struct GpuTensor {
        pub id: usize,
        pub class: char,
    }
}
mod hybrid {
    use super::model::GpuTensor;
    pub struct FullAttnLayer {
        pub wq: GpuTensor,
        pub wk: GpuTensor,
        pub wv: GpuTensor,
    }
}
struct Norm(Vec<f32>);
impl Norm {
    fn float_data(&self) -> &Vec<f32> {
        &self.0
    }
}
struct Layer {
    attn_norm: Norm,
}
struct Config {
    n_embd: usize,
    rms_eps: f32,
}
struct Aux(Vec<f32>);
impl Aux {
    fn ones(&self, _: &Engine) -> &Vec<f32> {
        &self.0
    }
}
struct HybridModel {
    cfg: Config,
    layers: Vec<Layer>,
    gemma4_aux: Option<Aux>,
    swa: bool,
    gemma: bool,
}
struct Cache;
struct Engine {
    fast: bool,
    fuse: bool,
    attention: bool,
    exact: RefCell<bool>,
    calls: RefCell<Vec<String>>,
    expected: Vec<f32>,
}
thread_local! {static RAW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };}
struct Exact<'a>(&'a Engine, bool);
impl Drop for Exact<'_> {
    fn drop(&mut self) {
        self.0.exact.replace(self.1);
    }
}
impl Engine {
    fn stage_a_raw_needed() -> bool {
        RAW.get()
    }
    fn exact_scope(&self, on: bool) -> Exact<'_> {
        Exact(self, self.exact.replace(on))
    }
    fn uses_q8_1_fast(&self, w: &model::GpuTensor) -> bool {
        self.fast && w.class != 'f'
    }
    fn zeros(&self, n: usize) -> R<Vec<f32>> {
        Ok(vec![0.; n])
    }
    fn uninit(&self, n: usize) -> R<Vec<f32>> {
        self.zeros(n)
    }
    fn rms_norm_decode(
        &self,
        x: &Vec<f32>,
        w: &Vec<f32>,
        h: &mut Vec<f32>,
        n: usize,
        t: usize,
        eps: f32,
    ) -> R<()> {
        if x.len() != n * t || h.len() != n * t || w.len() != n {
            return Err("bad raw operand geometry".into());
        }
        assert_eq!(eps, 0.00001);
        self.calls.borrow_mut().push("norm".into());
        // Deliberately not a numerical RMS oracle: record the precise residual/norm operands.
        for i in 0..n * t {
            h[i] = x[i] * w[i % n];
        }
        assert_eq!(
            h, &self.expected,
            "wrong residual or layer norm (including range head)"
        );
        Ok(())
    }
    fn matmul_pre(
        &self,
        w: &model::GpuTensor,
        q: &Vec<i8>,
        d: &Vec<f32>,
        h: &Vec<f32>,
        t: usize,
    ) -> R<Vec<f32>> {
        assert_eq!(q, &vec![127; 4 * t]);
        assert_eq!(d, &vec![99.; t]);
        if self.uses_q8_1_fast(w) {
            if self.attention {
                assert!(
                    h.is_empty(),
                    "fast sibling must not gain a raw prefill operand"
                );
            }
        } else {
            assert_eq!(
                h, &self.expected,
                "slow projection lost its real F32 activation"
            );
        }
        self.calls.borrow_mut().push(format!("single{}", w.id));
        Ok(vec![w.id as f32; t])
    }
    fn matmul(&self, w: &model::GpuTensor, h: &Vec<f32>, t: usize) -> R<Vec<f32>> {
        assert_eq!(h, &self.expected);
        self.calls.borrow_mut().push(format!("plain{}", w.id));
        Ok(vec![w.id as f32; t])
    }
    fn quantize_q8_1(&self, _: &Vec<f32>, t: usize, _: usize) -> R<(Vec<i8>, Vec<f32>)> {
        Ok((vec![127; 4 * t], vec![99.; t]))
    }
    fn mmq_act_begin(&self) {}
    fn clone_dtod(&self, x: &Vec<f32>) -> R<Vec<f32>> {
        Ok(x.clone())
    }
    fn pair(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        t: usize,
        class: char,
    ) -> R<Option<Pair>> {
        assert!(
            self.uses_q8_1_fast(a) && self.uses_q8_1_fast(b),
            "fused path bypassed FAST/weight eligibility"
        );
        if self.fuse && a.class == class && b.class == class {
            self.calls.borrow_mut().push("fused2".into());
            Ok(Some((vec![a.id as f32; t], vec![b.id as f32; t])))
        } else {
            Ok(None)
        }
    }
    fn triple(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        c: &model::GpuTensor,
        t: usize,
    ) -> R<Option<Triple>> {
        assert!(
            [a, b, c].iter().all(|w| self.uses_q8_1_fast(w)),
            "fused3 bypassed FAST/weight eligibility"
        );
        if self.fuse && [a, b, c].iter().all(|w| w.class == 'q') {
            self.calls.borrow_mut().push("fused3".into());
            Ok(Some((
                vec![a.id as f32; t],
                vec![b.id as f32; t],
                vec![c.id as f32; t],
            )))
        } else {
            Ok(None)
        }
    }
    fn matmul_q4_fused2(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        _: &Vec<i8>,
        _: &Vec<f32>,
    ) -> R<Option<Pair>> {
        self.pair(a, b, 1, 'q')
    }
    fn matmul_nvfp4_fused2(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        _: &Vec<i8>,
        _: &Vec<f32>,
        t: usize,
    ) -> R<Option<Pair>> {
        self.pair(a, b, t, 'n')
    }
    fn matmul_q4_fused3(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        c: &model::GpuTensor,
        _: &Vec<i8>,
        _: &Vec<f32>,
    ) -> R<Option<Triple>> {
        self.triple(a, b, c, 1)
    }
    fn matmul_q4_fused2_batched(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        _: &Vec<i8>,
        _: &Vec<f32>,
        t: usize,
    ) -> R<Option<Pair>> {
        if !(2..=8).contains(&t) {
            return Ok(None);
        }
        self.pair(a, b, t, 'q')
    }
    fn matmul_q4_fused3_batched(
        &self,
        a: &model::GpuTensor,
        b: &model::GpuTensor,
        c: &model::GpuTensor,
        _: &Vec<i8>,
        _: &Vec<f32>,
        t: usize,
    ) -> R<Option<Triple>> {
        if !(2..=8).contains(&t) {
            return Ok(None);
        }
        self.triple(a, b, c, t)
    }
}
impl HybridModel {
    fn uses_gemma_program(&self) -> bool {
        self.gemma
    }
    fn uses_sliding_gated_moe_program(&self) -> bool {
        false
    }
    fn protect_rewrite_execution(&self) -> R<()> {
        Ok(())
    }
    fn require_rewrite(&self, _: memra_gguf::execution_manifest::RewriteSurface) -> R<()> {
        Ok(())
    }
    fn gemma4_geom(&self, _: usize) -> (usize, usize, usize, f32, f32, bool) {
        (4, 1, 1, 1., 1., self.swa)
    }
    // ACTUAL_PRODUCTION_METHODS
}
fn setup(
    swa: bool,
    fast: bool,
    fuse: bool,
    t: usize,
    classes: [char; 3],
) -> (HybridModel, Engine, hybrid::FullAttnLayer, Vec<f32>) {
    let m = HybridModel {
        cfg: Config {
            n_embd: 4,
            rms_eps: 0.00001,
        },
        layers: vec![
            Layer {
                attn_norm: Norm(vec![2.; 4]),
            },
            Layer {
                attn_norm: Norm(vec![3.; 4]),
            },
        ],
        gemma4_aux: Some(Aux(vec![1.; 4])),
        swa,
        gemma: true,
    };
    let x: Vec<f32> = (0..4 * t).map(|i| i as f32 + 0.125).collect();
    let e = Engine {
        fast,
        fuse,
        attention: true,
        exact: RefCell::new(false),
        calls: RefCell::new(vec![]),
        expected: x.iter().map(|v| v * 3.).collect(),
    };
    let [q, k, v] = classes;
    let fa = hybrid::FullAttnLayer {
        wq: model::GpuTensor { id: 0, class: q },
        wk: model::GpuTensor { id: 1, class: k },
        wv: model::GpuTensor { id: 2, class: v },
    };
    (m, e, fa, x)
}
fn exercise(swa: bool, fast: bool, fuse: bool, classes: [char; 3]) {
    for t in [1, 4, 15, 16, 17] {
        for door in 0..3 {
            if door == 0 && t != 1 {
                continue;
            }
            let (m, e, fa, x) = setup(swa, fast, fuse, t, classes);
            let q = vec![127; 4 * t];
            let d = vec![99.; t];
            let p = vec![0; t];
            let mut c = Cache;
            let result = match door {
                0 => m.gemma4_decode_attn(&e, &fa, 1, &x, &q, &d, &p, &mut c),
                1 => m.gemma4_verify_attn(&e, &fa, 1, &x, &q, &d, &p, t, &mut c),
                _ => m.gemma4_verify_attn_stream(&e, &fa, 1, &x, &q, &d, &p, t, &mut c, 0, &[]),
            }
            .unwrap();
            if !swa {
                assert_eq!(result.1, result.2);
                assert!(!e.calls.borrow().contains(&"single2".to_string()));
            }
            let raw_needed =
                !fast || classes[0] == 'f' || classes[1] == 'f' || (swa && classes[2] == 'f');
            assert_eq!(
                e.calls.borrow().iter().filter(|s| *s == "norm").count(),
                usize::from(raw_needed)
            );
            if !fast {
                assert!(!e.calls.borrow().iter().any(|s| s.starts_with("fused")));
            }
        }
    }
}
#[test]
fn slow_q4_residual_and_layer_norm_at_all_boundaries() {
    for swa in [false, true] {
        exercise(swa, false, true, ['q'; 3]);
    }
}
#[test]
fn all_fast_keeps_empty_fallback_and_existing_fusions() {
    for swa in [false, true] {
        for fuse in [false, true] {
            exercise(swa, true, fuse, ['q'; 3]);
        }
    }
}
#[test]
fn mixed_projections_do_not_change_fast_sibling_operands() {
    for classes in [['f', 'q', 'q'], ['q', 'f', 'q'], ['n', 'n', 'f']] {
        exercise(true, true, true, classes);
    }
}
#[test]
fn global_unused_v_is_not_an_operand() {
    exercise(false, true, true, ['q', 'q', 'f']);
}
#[test]
fn bad_residual_geometry_refuses_before_projection() {
    let (m, e, fa, _) = setup(true, false, true, 4, ['q'; 3]);
    assert!(
        m.gemma4_attention_raw(&e, &fa, 1, &vec![1.; 15], 4)
            .is_err()
    );
    assert!(e.calls.borrow().is_empty());
}
#[test]
fn dense_fusions_obey_fast_at_t1_and_verify_boundaries() {
    for fast in [false, true] {
        for t in [1, 4, 15, 16, 17] {
            let (m, mut e, fa, _) = setup(true, fast, true, t, ['q'; 3]);
            e.attention = false;
            m.dense(&e, &fa.wq, &fa.wk, e.expected.clone(), t).unwrap();
            assert_eq!(
                e.calls.borrow().iter().any(|s| s.starts_with("fused")),
                fast && t <= 8
            );
        }
    }
}
#[test]
fn f32_verify_crossover_is_scoped_and_fast_is_unchanged() {
    for raw in [false, true] {
        for prior in [false, true] {
            for t in [1, 4, 15, 16, 17] {
                for fail in [false, true] {
                    RAW.set(raw);
                    let (m, e, _, _) = setup(true, !raw, true, t, ['q'; 3]);
                    e.exact.replace(prior);
                    for run in [
                        HybridModel::scope_gemma4_verify_trunk,
                        HybridModel::scope_gemma4_verify_t_am_stream,
                    ] {
                        let got = run(&m, &e, &vec![0; t], t, fail);
                        if fail {
                            assert!(got.is_err());
                        } else {
                            assert_eq!(got.unwrap(), prior || (raw && t >= 16));
                        }
                        assert_eq!(*e.exact.borrow(), prior, "verify scope leaked across calls");
                    }
                }
            }
        }
    }
}
#[test]
fn f32_public_generators_keep_eager_and_direct_dc_refuses_before_writes() {
    for raw in [false, true] {
        RAW.set(raw);
        let (m, _, _, _) = setup(true, !raw, true, 1, ['q'; 3]);
        for greedy in [false, true] {
            for penalty in [0, 1] {
                let sampler = Sampler { greedy, penalty };
                assert_eq!(m.route_generate(&sampler), !raw);
                assert_eq!(
                    m.route_generate_with(&sampler),
                    !raw && greedy && penalty == 0
                );
            }
        }
        let mut writes = 0;
        let result = m.dc_entry(&mut writes);
        assert_eq!(result.is_err(), raw);
        assert_eq!(writes, usize::from(!raw));
        if raw {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("no qualified F32 program")
            );
        }
    }
}

#[test]
fn gemma_f32_cannot_fall_through_to_generic_device_counter_route() {
    for raw in [false, true] {
        for gemma in [false, true] {
            for enabled in [false, true] {
                RAW.set(raw);
                let (mut m, _, _, _) = setup(true, !raw, true, 1, ['q'; 3]);
                m.gemma = gemma;
                let sampler = Sampler {
                    greedy: true,
                    penalty: 0,
                };
                assert_eq!(
                    m.route_qwen_generate(&sampler, enabled, 4, 4),
                    enabled && !gemma
                );
                assert_eq!(
                    m.route_qwen_generate_with(&sampler, enabled, 4, 4),
                    enabled && !gemma
                );
            }
        }
    }
}
