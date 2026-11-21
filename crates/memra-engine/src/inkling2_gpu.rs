//! Inkling 2 Mini on one GPU: the native route of the typed plan
//! (`memra_gguf::model_plan::inkling2`) over the checkpoint's own safetensors, with the
//! weights resident in the bytes the checkpoint stores, a carried cache per slot and the kernels
//! of `cu/inkling2_gpu.cu` (and `cu/inkling2_actq.cu` for the declared expert program).
//!
//! The program is ours for this card and the plan's: 48 cache slots over 24 weight sets, a KV
//! cache per full-attention slot, a KV ring per sliding slot, the last `kernel - 1` inputs of
//! four short convolutions per slot. Nothing here keys on an architecture name: every choice
//! reads the plan (layer class, window, extent, log scale, loop slots, the per-layer expert mode).
//!
//! Numeric classes, per physical layer, from `Inkling2Numerics::experts[layer].activation`:
//!
//! - `Declared` (the family's default): the checkpoint's required activation quantizer, NVFP4
//!   W4A4 or MXFP8 W8A8, on the block-scaled tensor-core GEMMs of `cu/inkling2_actq.cu`.
//! - `Float` (BRING-UP, never the default and never a support claim): the stored expert bytes
//!   on f32 inputs, exact dequantization. It exists because it has an independent oracle.
//!
//! Outside the experts the activations are f32 on the device in both classes; the BF16 rounding
//! points of a serving class are a later, measured decision.
//!
//! A call is a chunk of rows (a prefill chunk or one decode token). Every kernel computes a row
//! from that row alone, so the bits of a row do not depend on the chunking.

use core::ffi::c_void;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_gguf::model_plan::{
    ActivationPlan, AttentionInputs, AttentionPlan, BeyondExtent, ClipScale, ExpertActivationMode,
    ExpertWeightFormat, Inkling2Plan, LayerPlan, LogitsTransform, LoopMessageTarget, MlpPlan,
    ModelPlan, MoeMlpPlan, NormKind, NormPlan, RelativeBiasAttentionPlan, ResidualTopology,
    RouterPlan, ShortConvTaps, WeightTransform,
};
use memra_gguf::model_plan::{BlockAttentionPlan, ValueNorm};
use memra_gguf::safetensors::StModel;
use memra_reference::inkling2::ops::decimal;

/// The class a route runs, as the plan names it.
pub fn numeric_class(plan: &ModelPlan) -> Result<String, String> {
    let program = plan.inkling2.as_ref().ok_or("not an inkling2 plan")?;
    Ok(format!(
        "inkling2-gpu-f32-stream/{:?}",
        program.numerics.class
    ))
}

mod ffi {
    use core::ffi::c_void;
    unsafe extern "C" {
        pub fn memra_ink2g_tile_words() -> i32;
        pub fn memra_ink2g_gemm(
            x: *const c_void,
            ldx: i64,
            w: *const c_void,
            scales: *const c_void,
            wstride: i64,
            sstride: i64,
            amax: *const f32,
            denom: f32,
            fmt: i32,
            round_bf16: i32,
            tiles: *const u32,
            ntiles: i32,
            xdiv: i32,
            single: i32,
            y: *mut c_void,
            ldy: i64,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_embed(
            table: *const c_void,
            ids: *const u32,
            norm_w: *const f32,
            eps: f64,
            out: *mut f32,
            r: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_norm(
            x: *const f32,
            norm_w: *const f32,
            eps: f64,
            y: *mut f32,
            r: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_mix(
            states: *const *const f32,
            ns: i32,
            queries: *const f32,
            nq: i32,
            eps: f64,
            out: *mut f32,
            out_stride: i64,
            r: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_controller(
            x: *const f32,
            msg: *const f32,
            down: *const c_void,
            forget: *const c_void,
            gain: *const c_void,
            eps: f64,
            out: *mut f32,
            r: i32,
            h: i32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_heads(
            mode: i32,
            x: *const f32,
            scale: *const f32,
            thr: f64,
            eps: f64,
            pos0: i64,
            log_floor: i32,
            alpha: f64,
            dst: *mut f32,
            cap: i64,
            r: i32,
            heads: i32,
            width: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_conv(
            mode: i32,
            x: *const f32,
            state_in: *const f32,
            state_out: *mut f32,
            taps: *const f32,
            dynamic: *const f32,
            groups: i32,
            add: *const f32,
            out: *mut f32,
            r: i32,
            c: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_attn(
            q: *const f32,
            rel: *const f32,
            table: *const f32,
            sink: *const f32,
            kc: *const f32,
            vc: *const f32,
            cap: i64,
            pos0: i64,
            window: i32,
            extent: i32,
            mask_beyond: i32,
            out: *mut f32,
            r: i32,
            heads: i32,
            kv_heads: i32,
            dim: i32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_gate(y: *mut f32, g: *const f32, n: i64, stream: *mut c_void) -> i32;
        pub fn memra_ink2g_route(
            logits: *const f32,
            bias: *const f32,
            scale: f64,
            top_k: i32,
            tie_high: i32,
            ids: *mut u32,
            wts: *mut f32,
            margin: *mut f32,
            r: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_swiglu(
            gu: *const f32,
            wts: *const f32,
            gate_cap: f64,
            up_cap: f64,
            act: *mut f32,
            p: i32,
            f: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_sum(
            dn: *const f32,
            wts: *const f32,
            out: *mut f32,
            r: i32,
            top_k: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_sum_bf16(
            dn: *const c_void,
            wts: *const f32,
            out: *mut f32,
            r: i32,
            top_k: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_to_bf16(
            x: *const f32,
            y: *mut c_void,
            rows: i32,
            h: i32,
            rep: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_head(
            logits: *mut f32,
            cap: f64,
            best: *mut u32,
            r: i32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_tiles(
            ids: *const u32,
            tiles: *mut u32,
            pairs: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_groups(
            ids: *const u32,
            groups: *mut u32,
            pairs: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2g_copy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: i64,
            stream: *mut c_void,
        ) -> i32;

        // cu/inkling2_actq.cu (the declared expert program)
        pub fn memra_ink2_actq_built() -> i32;
        pub fn memra_ink2_actq(
            kind: i32,
            situ: i32,
            x: *const c_void,
            gamma: *const f32,
            rows: *const u32,
            nrows: i32,
            values: *mut c_void,
            scales: *mut c_void,
            amax: *mut f32,
            y: *mut c_void,
            k: i32,
            gate_cap: f32,
            up_cap: f32,
            mode: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_ink2_expert_gemm(
            kind: i32,
            nb: i32,
            act: *const c_void,
            act_scales: *const c_void,
            act_amax: *const f32,
            w: *const c_void,
            w_scales: *const c_void,
            w_amax: *const f32,
            denominator: f32,
            groups: *const u32,
            ngroups: i32,
            out: *mut c_void,
            acc: *mut f32,
            n: i32,
            k: i32,
            wpb: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

const FMT_BF16: i32 = 0;
const FMT_NVFP4: i32 = 1;
const FMT_MXFP8: i32 = 2;
/// Words of one act-quant group record at `nb = 1` (`memra_ink2_group_words(1)`).
const GROUP_WORDS: usize = 10;

fn ck(name: &str, rc: i32) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        r if r >= 10000 => Err(format!("{name}: CUDA error {}", r - 10000)),
        r => Err(format!("{name}: contract refused ({r})")),
    }
}

/// Wrong programs the checkpoint gate must tell apart from the right one (fault injection of the
/// gate, not runtime options).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedArms {
    /// Every quantized expert weight multiplies as bf16(its dequantized value), as a BF16 loader
    /// holds it (float class only).
    pub dequant_bf16: bool,
    /// The short convolutions forget their history between calls.
    pub conv_state_dropped: bool,
    /// The router keeps the higher expert among equal keys.
    pub ties_high: bool,
    /// A forward that fails part way commits the conv histories it wrote (the program before the
    /// route's commit rule). Needs `Route::fail_after_slot`.
    pub commit_on_error: bool,
}

/// The device, its stream and what the route did on it.
pub struct Dev {
    st: Arc<CudaStream>,
    /// Kernel launches since the last reset.
    pub launches: Cell<u64>,
    /// Device-to-device copies since the last reset.
    pub copies: Cell<u64>,
    /// Host round trips (a download the route itself needs) since the last reset.
    pub round_trips: Cell<u64>,
    /// Bytes allocated through this handle.
    pub allocated: Cell<usize>,
}

fn dptr<T>(s: &CudaSlice<T>, st: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr(st);
    p
}

fn dptr_mut<T>(s: &mut CudaSlice<T>, st: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr_mut(st);
    p
}

impl Dev {
    pub fn new(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("cuda context: {e:?}"))?;
        Ok(Self::on(ctx.default_stream()))
    }

    /// The driver's free and total bytes (`cuMemGetInfo`), the only headroom figure that counts
    /// the CUDA context and fragmentation (LAW:measure-driver-free-not-nvidia-smi).
    pub fn driver_free(&self) -> Result<(usize, usize), String> {
        self.st
            .context()
            .mem_get_info()
            .map_err(|e| format!("mem_get_info: {e:?}"))
    }

    /// A handle on an existing stream (the kernel-check binary's engine).
    pub fn on(st: Arc<CudaStream>) -> Self {
        Self {
            st,
            launches: Cell::new(0),
            copies: Cell::new(0),
            round_trips: Cell::new(0),
            allocated: Cell::new(0),
        }
    }

    fn bind(&self) -> Result<(), String> {
        self.st
            .context()
            .bind_to_thread()
            .map_err(|e| format!("bind: {e:?}"))
    }

    fn sp(&self) -> *mut c_void {
        self.st.cu_stream() as *mut c_void
    }

    pub fn up<T: cudarc::driver::DeviceRepr>(&self, v: &[T]) -> Result<CudaSlice<T>, String> {
        self.allocated
            .set(self.allocated.get() + std::mem::size_of_val(v));
        self.st.clone_htod(v).map_err(|e| format!("htod: {e:?}"))
    }

    pub fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>, String> {
        self.allocated
            .set(self.allocated.get() + n * std::mem::size_of::<T>());
        self.st
            .alloc_zeros::<T>(n.max(1))
            .map_err(|e| format!("alloc of {n}: {e:?}"))
    }

    /// The first `n` elements at element offset `at`.
    pub fn down<T: cudarc::driver::DeviceRepr + Default + Clone>(
        &self,
        v: &CudaSlice<T>,
        at: usize,
        n: usize,
    ) -> Result<Vec<T>, String> {
        let view = v
            .try_slice(at..at + n)
            .ok_or_else(|| format!("download of {n} at {at} past a buffer of {}", v.len()))?;
        let out = self
            .st
            .clone_dtoh(&view)
            .map_err(|e| format!("dtoh: {e:?}"))?;
        self.sync()?;
        Ok(out)
    }

    fn put<T: cudarc::driver::DeviceRepr>(
        &self,
        host: &[T],
        dst: &mut CudaSlice<T>,
    ) -> Result<(), String> {
        let mut view = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| format!("upload of {} into a buffer of {}", host.len(), 0))?;
        self.st
            .memcpy_htod(host, &mut view)
            .map_err(|e| format!("htod: {e:?}"))
    }

    pub fn sync(&self) -> Result<(), String> {
        self.st.synchronize().map_err(|e| format!("sync: {e:?}"))
    }

    pub fn reset_counters(&self) {
        self.launches.set(0);
        self.copies.set(0);
        self.round_trips.set(0);
    }

    fn launched(&self, name: &str, rc: i32) -> Result<(), String> {
        self.launches.set(self.launches.get() + 1);
        ck(name, rc)
    }

    /// f32 rows `src` to `dst` (`n` elements).
    fn copy(&self, dst: &mut CudaSlice<f32>, src: &CudaSlice<f32>, n: usize) -> Result<(), String> {
        self.copies.set(self.copies.get() + 1);
        let st = &self.st;
        ck("memra_ink2g_copy", unsafe {
            ffi::memra_ink2g_copy(
                dptr_mut(dst, st) as *mut c_void,
                dptr(src, st) as *const c_void,
                (n * 4) as i64,
                self.sp(),
            )
        })
    }
}

// ---- the checkpoint ----------------------------------------------------------------------------

/// The checkpoint directory: the shard index plus the pack's auxiliary tensor files.
pub struct Ckpt {
    pub st: StModel,
    /// The pack's tensor contract bound against the census of every file (shards and auxiliary
    /// files) before any upload: missing, unexpected, ambiguous, wrong-shape and wrong-quant
    /// tensors are refused here, and every name the loader reads must be one it binds.
    pub binding: memra_gguf::checkpoint_binding::CheckpointBinding,
    /// The plan the contract was compiled for; `Route::load` refuses another.
    plan: ModelPlan,
    /// Every header the census was built from (shards and auxiliary files).
    headers: BTreeMap<String, memra_gguf::safetensors::StInfo>,
    pack: &'static memra_gguf::model_packs::ModelPack,
    /// `(prefix, model)` of each auxiliary file the pack names (the drafter's `mtp.safetensors`).
    pub auxiliary: Vec<(&'static str, StModel)>,
    pub bytes_read: Cell<usize>,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Ckpt {
    /// Opens the shard index and every auxiliary tensor file `pack` declares. A declared file
    /// that is missing is an error: the contract owns its tensors.
    pub fn open(
        dir: &std::path::Path,
        pack: &'static memra_gguf::model_packs::ModelPack,
        cfg: &memra_gguf::config::ModelConfig,
        plan: &ModelPlan,
    ) -> Result<Self, String> {
        let st = StModel::open(dir).map_err(|e| format!("open {}: {e}", dir.display()))?;
        let mut headers: BTreeMap<String, memra_gguf::safetensors::StInfo> = BTreeMap::new();
        for name in st.names() {
            let info = st.info(name).ok_or_else(|| format!("{name}: no header"))?;
            headers.insert(name.clone(), info.clone());
        }
        let mut auxiliary = Vec::new();
        for aux in pack.auxiliary_files {
            let path = dir.join(aux.file);
            let model =
                StModel::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            for name in model.names() {
                let info = model
                    .info(name)
                    .ok_or_else(|| format!("{name}: no header"))?;
                let full = format!("{}{name}", aux.prefix);
                if headers.insert(full.clone(), info.clone()).is_some() {
                    return Err(format!(
                        "tensor {full} of {} collides with an indexed tensor",
                        aux.file
                    ));
                }
            }
            auxiliary.push((aux.prefix, model));
        }
        let census = memra_gguf::source::census_from_safetensors_headers(&headers)?;
        let binding = memra_gguf::checkpoint_binding::bind_census(Some(pack), cfg, plan, &census)
            .map_err(|e| format!("tensor contract: {e}"))?;
        Ok(Self {
            st,
            binding,
            plan: plan.clone(),
            headers,
            pack,
            auxiliary,
            bytes_read: Cell::new(0),
        })
    }

    /// The binding's red arms: censuses the contract must refuse (one extra tensor, one missing,
    /// one shape changed, one dtype substituted) and names the loader must not read. Lines to
    /// print and the number of arms that were not refused.
    pub fn contract_red_arms(
        &self,
        cfg: &memra_gguf::config::ModelConfig,
    ) -> Result<(Vec<String>, usize), String> {
        use memra_gguf::checkpoint_binding::bind_census;
        use memra_gguf::source::census_from_safetensors_headers;
        let mut lines = Vec::new();
        let mut fails = 0;
        let mut arm = |name: &str, headers: BTreeMap<String, memra_gguf::safetensors::StInfo>| {
            let verdict = census_from_safetensors_headers(&headers).and_then(|census| {
                bind_census(Some(self.pack), cfg, &self.plan, &census).map_err(|e| e.to_string())
            });
            match verdict {
                Ok(_) => {
                    fails += 1;
                    lines.push(format!("contract red arm {name}: ACCEPTED FAIL"));
                }
                Err(e) => lines.push(format!(
                    "contract red arm {name}: refused ({}) OK",
                    e.chars().take(160).collect::<String>()
                )),
            }
        };
        let (first, info) = self
            .headers
            .iter()
            .find(|(n, i)| n.starts_with("model.llm.layers.0.") && i.dtype == "BF16")
            .map(|(n, i)| (n.clone(), i.clone()))
            .ok_or("no BF16 tensor in layer 0")?;
        let mut extra = self.headers.clone();
        extra.insert("model.llm.layers.0.extra_proj.weight".into(), info.clone());
        arm("extra tensor", extra);
        let mut missing = self.headers.clone();
        missing.remove(&first);
        arm(&format!("missing {first}"), missing);
        let mut shaped = self.headers.clone();
        if let Some(i) = shaped.get_mut(&first) {
            i.shape[0] += 1;
        }
        arm(&format!("shape of {first}"), shaped);
        let mut typed = self.headers.clone();
        if let Some(i) = typed.get_mut(&first) {
            i.dtype = "F16".into();
        }
        arm(&format!("dtype of {first}"), typed);
        for name in [
            "model.llm.layers.0.extra_proj.weight",
            "model.llm.embed_tokens.weight",
        ] {
            match self.contracted(name) {
                Ok(()) => {
                    fails += 1;
                    lines.push(format!("loader read of {name}: ACCEPTED FAIL"));
                }
                Err(_) => lines.push(format!("loader read of {name}: refused OK")),
            }
        }
        match self.contracted(&first) {
            Ok(()) => lines.push(format!("loader read of {first}: bound OK")),
            Err(e) => {
                fails += 1;
                lines.push(format!("loader read of {first}: {e} FAIL"));
            }
        }
        Ok((lines, fails))
    }

    /// `name` is a tensor the bound contract owns, or a quantization plane of one.
    fn contracted(&self, name: &str) -> Result<(), String> {
        let bound = |n: &str| {
            self.binding
                .bound
                .tensors
                .get(&memra_gguf::model_packs::inkling2::semantic_tensor_id(n))
                .is_some_and(|t| t.checkpoint_names.iter().any(|c| c == n))
        };
        if bound(name) {
            return Ok(());
        }
        for plane in ["_scale", "_amax", "_outer_scale_denominator"] {
            if name.strip_suffix(plane).is_some_and(bound) {
                return Ok(());
            }
        }
        Err(format!(
            "{name} is not a tensor the bound inkling2 contract owns: the loader reads only what the contract binds"
        ))
    }

    fn shaped(
        &self,
        name: &str,
        dtypes: &[&str],
        shape: &[usize],
    ) -> Result<(&str, &[u8]), String> {
        self.contracted(name)?;
        let (info, bytes) = self
            .st
            .raw(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        // The bound contract has checked the storage and shape of every tensor; the loader's own
        // expectation is an assertion that the two agree.
        debug_assert!(
            dtypes.contains(&info.dtype.as_str())
                && info
                    .shape
                    .iter()
                    .map(|v| *v as usize)
                    .eq(shape.iter().copied()),
            "{name}: {} {:?} in the checkpoint, the loader expects {dtypes:?} {shape:?}",
            info.dtype,
            info.shape
        );
        self.bytes_read.set(self.bytes_read.get() + bytes.len());
        Ok((info.dtype.as_str(), bytes))
    }

    /// A BF16 or F32 tensor as its exact f32 values.
    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (dtype, bytes) = self.shaped(name, &["BF16", "F32"], shape)?;
        Ok(if dtype == "BF16" {
            bf16_to_f32(bytes)
        } else {
            le_f32(bytes)
        })
    }

    fn bf16(&self, dev: &Dev, name: &str, shape: &[usize]) -> Result<CudaSlice<u8>, String> {
        dev.up(self.shaped(name, &["BF16"], shape)?.1)
    }

    fn f32_dev(&self, dev: &Dev, name: &str, shape: &[usize]) -> Result<CudaSlice<f32>, String> {
        dev.up(&self.f32s(name, shape)?)
    }
}

/// NVFP4's per-expert amax and outer-scale denominator must be finite and positive: the weight's
/// global scale is amax / denominator, and anything else is not a weight.
fn check_nvfp4_globals(name: &str, amax: &[f32], denominator: f32) -> Result<(), String> {
    if let Some((e, a)) = amax
        .iter()
        .enumerate()
        .find(|(_, a)| !a.is_finite() || **a <= 0.0)
    {
        return Err(format!(
            "{name}_amax[{e}] = {a}: an NVFP4 amax is finite and positive"
        ));
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(format!(
            "{name}_outer_scale_denominator = {denominator}: finite and positive"
        ));
    }
    Ok(())
}

/// An E8M0 (ue8m0) scale byte of 0xFF is NaN; a scale grid holding one is refused.
fn check_ue8m0(name: &str, scales: &[u8]) -> Result<(), String> {
    match scales.iter().position(|s| *s == 0xFF) {
        Some(at) => Err(format!("{name}_scale byte {at} is 0xFF, the E8M0 NaN code")),
        None => Ok(()),
    }
}

/// One projection of a layer's stacked experts, in the stored bytes.
struct Bank {
    fmt: i32,
    data: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    /// Per-expert amax (NVFP4; one unused element for MXFP8).
    amax: CudaSlice<f32>,
    denom: f32,
    n: usize,
    k: usize,
}

impl Bank {
    fn load(
        ck: &Ckpt,
        dev: &Dev,
        name: &str,
        format: ExpertWeightFormat,
        experts: usize,
        n: usize,
        k: usize,
    ) -> Result<Self, String> {
        let nan = |bytes: &[u8]| bytes.iter().any(|b| b & 0x7f == 0x7f);
        match format {
            ExpertWeightFormat::Nvfp4Amax => {
                if !k.is_multiple_of(16) {
                    return Err(format!("{name}: NVFP4 rows are whole blocks of 16"));
                }
                let codes = ck.shaped(name, &["U8"], &[experts, n, k / 2])?.1;
                let scales = ck
                    .shaped(
                        &format!("{name}_scale"),
                        &["F8_E4M3"],
                        &[experts, n, k / 16],
                    )?
                    .1;
                if nan(scales) {
                    return Err(format!("{name}_scale holds an e4m3 NaN code"));
                }
                let amax = ck.f32s(&format!("{name}_amax"), &[experts])?;
                let denom = ck.f32s(&format!("{name}_outer_scale_denominator"), &[1])?[0];
                check_nvfp4_globals(name, &amax, denom)?;
                Ok(Self {
                    fmt: FMT_NVFP4,
                    data: dev.up(codes)?,
                    scales: dev.up(scales)?,
                    amax: dev.up(&amax)?,
                    denom,
                    n,
                    k,
                })
            }
            ExpertWeightFormat::Mxfp8Row32 => {
                if !k.is_multiple_of(32) {
                    return Err(format!("{name}: MXFP8 rows are whole blocks of 32"));
                }
                let values = ck.shaped(name, &["F8_E4M3"], &[experts, n, k])?.1;
                if nan(values) {
                    return Err(format!("{name} holds an e4m3 NaN code"));
                }
                let scales = ck
                    .shaped(&format!("{name}_scale"), &["U8"], &[experts, n, k / 32])?
                    .1;
                check_ue8m0(name, scales)?;
                Ok(Self {
                    fmt: FMT_MXFP8,
                    data: dev.up(values)?,
                    scales: dev.up(scales)?,
                    amax: dev.up(&[1.0f32])?,
                    denom: 1.0,
                    n,
                    k,
                })
            }
        }
    }

    fn strides(&self) -> (i64, i64) {
        match self.fmt {
            FMT_NVFP4 => ((self.n * self.k / 2) as i64, (self.n * self.k / 16) as i64),
            _ => ((self.n * self.k) as i64, (self.n * self.k / 32) as i64),
        }
    }
}

struct ConvW {
    /// Static: `[channels, kernel]`, oldest input first. Dynamic: `[kernel, channels]` by lag.
    taps: CudaSlice<f32>,
    /// Dynamic taps: the BF16 projection `[groups * kernel, hidden]` and its group count.
    dynamic: Option<(CudaSlice<u8>, usize)>,
    kernel: usize,
    channels: usize,
}

struct CtrlW {
    down: CudaSlice<u8>,
    forget: CudaSlice<u8>,
    gain: CudaSlice<u8>,
}

/// One physical layer's weights on the device.
struct LayerW {
    /// One norm (shared) or three (q, k, v).
    norms: Vec<CudaSlice<f32>>,
    /// The attention mix queries and their count (none on the first layer).
    attn_queries: Option<(CudaSlice<f32>, usize)>,
    wq: CudaSlice<u8>,
    wr: CudaSlice<u8>,
    sink: CudaSlice<u8>,
    gate: CudaSlice<u8>,
    wk: CudaSlice<u8>,
    wv: CudaSlice<u8>,
    wo: CudaSlice<u8>,
    q_scale: Option<CudaSlice<f32>>,
    r_scale: Option<CudaSlice<f32>>,
    table: CudaSlice<f32>,
    kconv: ConvW,
    vconv: ConvW,
    oconv: ConvW,
    mconv: ConvW,
    mlp_query: CudaSlice<f32>,
    mlp_norm: CudaSlice<f32>,
    router_w: CudaSlice<u8>,
    router_b: Option<CudaSlice<f32>>,
    w13: Bank,
    w2: Bank,
    /// Controllers in `[q, k, v, mlp]` order and the layer's four message queries `[4, hidden]`.
    ctrl: Option<(Vec<CtrlW>, CudaSlice<f32>)>,
}

struct SlotState {
    kc: CudaSlice<f32>,
    vc: CudaSlice<f32>,
    /// Rows the cache holds: the context for a full slot, a ring for a sliding one.
    cap: usize,
    /// K, V, attention output, MLP output: two buffers each. `Route::parity` names the committed
    /// one; a call writes the other and the route swaps only when the whole call succeeded.
    conv: [[CudaSlice<f32>; 2]; 4],
}

/// What a caller can read out of a call, per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tap {
    /// The stream after attention (the input of the MLP depth mix), `[rows, hidden]`.
    ResidAttn,
    /// The slot's output, `[rows, hidden]`.
    ResidFfn,
    /// The MoE input (after the mix, the controller and the norm), `[rows, hidden]`.
    MoeIn,
    /// The MoE output before its short convolution, `[rows, hidden]`.
    MoeOut,
    /// The router's own weights in rank order, `[rows, top_k]`.
    RouteWeights,
    /// The router's margin: key of the last pick minus key of the first one left out, `[rows]`.
    RouteMargin,
    /// Attention probe (only for a slot `Observer::attn_probe` names): the clipped and scaled
    /// queries `[rows, heads * dim]`, the relative-position inputs `[rows, heads * rank]`, the
    /// sink logits `[rows, heads]`, the bias table `[rank, extent]`, the slot's whole key and
    /// value caches `[cap, kv_heads * dim]` and the attention output before the gate.
    AttnQ,
    AttnRel,
    AttnSink,
    AttnTable,
    AttnKeys,
    AttnValues,
    AttnOut,
}

/// A caller's view into a call.
pub trait Observer {
    /// Which float taps to download.
    fn wants(&self, tap: Tap) -> bool;
    fn floats(&mut self, slot: u32, tap: Tap, data: &[f32]);
    /// Download the attention probe taps of this slot.
    fn attn_probe(&self, _slot: u32) -> bool {
        false
    }
    /// The router's own picks in rank order, `[rows, top_k]`.
    fn picks(&mut self, slot: u32, ids: &[u32]);
    /// Picks and weights to run instead of the router's (`[rows, top_k]` each), or `None`.
    fn forced(&mut self, slot: u32, rows: usize) -> Option<(Vec<u32>, Vec<f32>)>;
    /// An MoE input to run instead of the computed one (`[rows, hidden]`), or `None`.
    fn forced_moe_in(&mut self, _slot: u32, _rows: usize) -> Option<Vec<f32>> {
        None
    }
}

/// Which rows of a call get logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// No logits (a prefill chunk before the last).
    None,
    /// The last row: argmax only.
    LastArgmax,
    /// Every row: logits and argmax.
    All,
}

pub struct CallOut {
    /// Argmax per row that got a head.
    pub argmax: Vec<u32>,
    /// `[rows, kept vocabulary]` for `Head::All`.
    pub logits: Vec<f32>,
}

struct Work {
    h: CudaSlice<f32>,
    prelude: CudaSlice<f32>,
    bank_h: CudaSlice<f32>,
    /// One block buffer per slot that starts a block.
    blocks: BTreeMap<u32, CudaSlice<f32>>,
    xin: CudaSlice<f32>,
    xc: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    msg: CudaSlice<f32>,
    qb: CudaSlice<f32>,
    rb: CudaSlice<f32>,
    sinkb: CudaSlice<f32>,
    gateb: CudaSlice<f32>,
    kp: CudaSlice<f32>,
    vp: CudaSlice<f32>,
    kcv: CudaSlice<f32>,
    vcv: CudaSlice<f32>,
    dynk: CudaSlice<f32>,
    dynv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    mixm: CudaSlice<f32>,
    mn: CudaSlice<f32>,
    rl: CudaSlice<f32>,
    ids: CudaSlice<u32>,
    wts: CudaSlice<f32>,
    margin: CudaSlice<f32>,
    tiles: CudaSlice<u32>,
    gu: CudaSlice<f32>,
    act: CudaSlice<f32>,
    dn: CudaSlice<f32>,
    moe: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    best: CudaSlice<u32>,
    tok: CudaSlice<u32>,
    /// the declared expert program
    dq: DeclaredWork,
}

/// The route's memory at load: what it allocated by class, and what the driver reported.
#[derive(Debug, Clone, Default)]
pub struct MemoryAccount {
    pub device_total: usize,
    /// Driver free before the route allocated anything (the CUDA context is already counted).
    pub driver_free_at_open: usize,
    pub weights: usize,
    pub context: usize,
    /// Full-attention K and V bytes per context position, over every full slot.
    pub per_position: usize,
    pub kv_full: usize,
    pub kv_rings: usize,
    /// Both buffers of every conv history.
    pub conv: usize,
    /// Activations of one chunk and the head.
    pub work: usize,
    /// The declared expert program's buffers.
    pub declared_work: usize,
    pub driver_free_before_kv: usize,
    pub driver_free_after_load: usize,
    /// Positions the driver's free memory before the full caches holds at this chunk.
    pub max_context: usize,
}

impl MemoryAccount {
    pub fn describe(&self) -> String {
        let g = |b: usize| b as f64 / (1u64 << 30) as f64;
        format!(
            "memory: device {:.3} GiB, driver free at open {:.3}, weights {:.3}, full KV {:.3} ({} positions x {} B), rings {:.3}, conv {:.3}, work {:.3}, declared work {:.3}, driver free before KV {:.3} and after load {:.3}, max context {}",
            g(self.device_total),
            g(self.driver_free_at_open),
            g(self.weights),
            g(self.kv_full),
            self.context,
            self.per_position,
            g(self.kv_rings),
            g(self.conv),
            g(self.work),
            g(self.declared_work),
            g(self.driver_free_before_kv),
            g(self.driver_free_after_load),
            self.max_context
        )
    }
}

/// Rows of logits one head launch holds.
const HEAD_ROWS: usize = 32;

pub struct Route {
    pub dev: Dev,
    plan: ModelPlan,
    layers: Vec<LayerW>,
    slots: Vec<SlotState>,
    embed: CudaSlice<u8>,
    embed_norm: CudaSlice<f32>,
    unembed: CudaSlice<u8>,
    out_norm: CudaSlice<f32>,
    out_query: CudaSlice<f32>,
    w: Work,
    /// Rows one call may hold.
    pub chunk: usize,
    /// Positions the full-attention caches hold.
    pub context: usize,
    pub position: u64,
    /// Which conv-history buffer of every slot is committed (`SlotState::conv`).
    parity: usize,
    red: RedArms,
    /// Fault injection of the atomicity gate: the next forward fails after this slot ran.
    pub fail_after_slot: Option<u32>,
    softcap: f64,
    keep: usize,
    hidden: usize,
    /// Bytes of resident weights.
    pub weight_bytes: usize,
    /// Where the card's memory went at load.
    pub account: MemoryAccount,
}

fn program(plan: &ModelPlan) -> Result<&Inkling2Plan, String> {
    plan.inkling2
        .as_ref()
        .ok_or_else(|| "the inkling2 GPU route needs an inkling2 plan".to_string())
}

fn attention_of(layer: &LayerPlan) -> Result<&RelativeBiasAttentionPlan, String> {
    match &layer.attention {
        AttentionPlan::RelativeBias(a) => Ok(a),
        _ => Err(format!("slot {}: not relative-bias attention", layer.index)),
    }
}

fn residual_of(layer: &LayerPlan) -> Result<&BlockAttentionPlan, String> {
    match &layer.residual {
        ResidualTopology::BlockAttention(r) => Ok(r),
        _ => Err(format!(
            "slot {}: not a block attention residual",
            layer.index
        )),
    }
}

fn moe_of(layer: &LayerPlan) -> Result<&MoeMlpPlan, String> {
    match &layer.mlp {
        MlpPlan::Moe(m) => Ok(m),
        _ => Err(format!("slot {}: not an MoE layer", layer.index)),
    }
}

fn rms(norm: &NormPlan) -> Result<f64, String> {
    if norm.kind != NormKind::Rms || norm.weight_transform != WeightTransform::Identity {
        return Err("a norm other than plain RMS".into());
    }
    Ok(decimal(norm.epsilon))
}

/// Refuse every plan shape the route does not implement, before any upload.
fn check_plan(plan: &ModelPlan) -> Result<(), String> {
    let program = program(plan)?;
    if plan.embedding_scale != 1.0 || program.vision_executed || program.output_queries != 1 {
        return Err(
            "a scaled embedding, an executed vision tower or several output queries".into(),
        );
    }
    rms(&program.embedding_norm)?;
    rms(&plan.output_norm)?;
    let lp = &program.loop_plan;
    if lp.slots.len() != plan.layers.len() || lp.execution_order.len() != plan.layers.len() {
        return Err("one loop slot and one visit per plan layer".into());
    }
    if lp.targets
        != [
            LoopMessageTarget::Query,
            LoopMessageTarget::Key,
            LoopMessageTarget::Value,
            LoopMessageTarget::Mlp,
        ]
    {
        return Err("a loop message order other than q, k, v, mlp".into());
    }
    if lp.gate_rank == 0 || lp.gate_rank > 128 {
        return Err("a controller rank outside 1..=128".into());
    }
    for layer in &plan.layers {
        let at = layer.index;
        let a = attention_of(layer)?;
        let r = residual_of(layer)?;
        let m = moe_of(layer)?;
        rms(&layer.pre_attention_norm)?;
        rms(&layer.pre_mlp_norm)?;
        if a.value_norm != ValueNorm::WeightlessRms || a.key_clip.scale != ClipScale::None {
            return Err(format!(
                "slot {at}: a value norm or key clip the route does not run"
            ));
        }
        if !a.query_heads.is_multiple_of(a.kv_heads) || a.head_dim > 256 || a.relative.rank > 256 {
            return Err(format!("slot {at}: head geometry outside the kernels"));
        }
        if a.relative.extent == 0 || a.relative.extent > 1024 {
            return Err(format!("slot {at}: a bias table wider than 1024"));
        }
        if a.key_conv != a.value_conv
            || a.output_conv.taps != ShortConvTaps::Static
            || program.mlp_output_conv.taps != ShortConvTaps::Static
        {
            return Err(format!(
                "slot {at}: a short convolution layout the route does not run"
            ));
        }
        for conv in [&a.key_conv, &a.output_conv, &program.mlp_output_conv] {
            if conv.kernel < 2 || conv.kernel > 8 {
                return Err(format!(
                    "slot {at}: a short convolution of {} taps",
                    conv.kernel
                ));
            }
        }
        if !r.normalize_values {
            return Err(format!("slot {at}: a depth mix over unnormalized states"));
        }
        match (a.inputs, r.attention_queries) {
            (_, 0) | (AttentionInputs::SharedNorm, 1) | (AttentionInputs::SeparateQkvNorms, 3) => {}
            _ => {
                return Err(format!(
                    "slot {at}: residual queries do not match the inputs"
                ));
            }
        }
        if m.shared.is_some() {
            return Err(format!("slot {at}: a shared expert"));
        }
        if !matches!(
            m.router,
            RouterPlan::Sigmoid {
                normalize_selected: true,
                ..
            }
        ) || !matches!(m.activation, ActivationPlan::SwiGluSoftcapped { .. })
        {
            return Err(format!(
                "slot {at}: a router or activation the route does not run"
            ));
        }
        if m.experts_per_token == 0 || m.experts_per_token > 64 || m.expert_count > 1024 {
            return Err(format!("slot {at}: expert counts outside the kernels"));
        }
    }
    for transform in &plan.logits {
        if !matches!(
            transform,
            LogitsTransform::Softcap(_) | LogitsTransform::Truncate(_)
        ) {
            return Err("a logits transform other than softcap and truncation".into());
        }
    }
    Ok(())
}

const LLM: &str = "model.llm.";

impl Route {
    /// Uploads the text weights as stored and allocates the state of every slot.
    /// `context`: positions a request may reach; `chunk`: rows one call may hold.
    pub fn load(
        dev: Dev,
        ck: &Ckpt,
        plan: &ModelPlan,
        context: usize,
        chunk: usize,
    ) -> Result<Self, String> {
        check_plan(plan)?;
        if ck.plan != *plan {
            return Err("the checkpoint's tensor contract was bound for another plan".into());
        }
        dev.bind()?;
        let (free_at_open, device_total) = dev.driver_free()?;
        let program = program(plan)?;
        let lp = &program.loop_plan;
        let hidden = plan.hidden_size as usize;
        let vocab = plan.vocab_size as usize;
        let physical = program.numerics.experts.len();
        let declared = program
            .numerics
            .experts
            .iter()
            .any(|e| e.activation == ExpertActivationMode::Declared);
        if declared && unsafe { ffi::memra_ink2_actq_built() } != 1 {
            return Err(
                "the plan runs the declared act-quant experts and this build has no block-scaled target"
                    .into(),
            );
        }
        if declared
            && (!program.numerics.gate_up_interleaved
                || !program.numerics.routed_weight_before_down)
        {
            return Err(
                "the declared expert program needs interleaved gate/up and the routing weight before down"
                    .into(),
            );
        }
        if !program.numerics.gate_up_interleaved {
            return Err("gate/up rows that are not interleaved".into());
        }
        let mut softcap = 0.0;
        let mut keep = vocab;
        for transform in &plan.logits {
            match transform {
                LogitsTransform::Softcap(cap) => softcap = decimal(*cap),
                LogitsTransform::Truncate(rows) => keep = keep.min(*rows as usize),
                _ => {}
            }
        }
        let core = (lp.core_end - lp.core_start) as usize;
        let cross = if core > 0 {
            Some((
                ck.f32s(
                    &format!("{LLM}middle_cycle_cross_pass_queries"),
                    &[2 * core, hidden],
                )?,
                ck.f32s(
                    &format!("{LLM}middle_cycle_cross_pass_kv_queries"),
                    &[2 * core, hidden],
                )?,
            ))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(physical);
        for p in 0..physical as u32 {
            let slots: Vec<&memra_gguf::model_plan::LoopSlotPlan> =
                lp.slots.iter().filter(|s| s.physical_layer == p).collect();
            let first = slots
                .first()
                .ok_or_else(|| format!("physical layer {p} has no slot"))?;
            let layer = &plan.layers[first.slot as usize];
            let a = attention_of(layer)?;
            let r = residual_of(layer)?;
            let m = moe_of(layer)?;
            let numerics = &program.numerics.experts[p as usize];
            let (heads, kvh, dim, rank, extent) = (
                a.query_heads as usize,
                a.kv_heads as usize,
                a.head_dim as usize,
                a.relative.rank as usize,
                a.relative.extent as usize,
            );
            let (qw, kvw) = (heads * dim, kvh * dim);
            let prefix = format!("{LLM}layers.{p}.");
            let norms = match a.inputs {
                AttentionInputs::SharedNorm => {
                    vec![ck.f32_dev(&dev, &format!("{prefix}attn_norm.weight"), &[hidden])?]
                }
                AttentionInputs::SeparateQkvNorms => vec![
                    ck.f32_dev(&dev, &format!("{prefix}q_input_norm.weight"), &[hidden])?,
                    ck.f32_dev(&dev, &format!("{prefix}k_input_norm.weight"), &[hidden])?,
                    ck.f32_dev(&dev, &format!("{prefix}v_input_norm.weight"), &[hidden])?,
                ],
            };
            let attn_queries = match r.attention_queries {
                0 => None,
                count => {
                    let index = p
                        .checked_sub(1)
                        .ok_or("the first layer has no attention mix")?;
                    let name = format!("{LLM}attn_res.attn_queries.{index}");
                    let count = count as usize;
                    let values = if count == 1 {
                        ck.f32s(&name, &[hidden])?
                    } else {
                        ck.f32s(&name, &[count, hidden])?
                    };
                    Some((dev.up(&values)?, count))
                }
            };
            let conv = |name: &str,
                        plan_conv: &memra_gguf::model_plan::ShortConvPlan|
             -> Result<ConvW, String> {
                let (channels, kernel) = (plan_conv.channels as usize, plan_conv.kernel as usize);
                Ok(match plan_conv.taps {
                    ShortConvTaps::Static => ConvW {
                        taps: ck.f32_dev(
                            &dev,
                            &format!("{prefix}{name}.weight"),
                            &[channels, 1, kernel],
                        )?,
                        dynamic: None,
                        kernel,
                        channels,
                    },
                    ShortConvTaps::StaticPlusDynamic { groups } => ConvW {
                        taps: ck.f32_dev(
                            &dev,
                            &format!("{prefix}{name}.static_w"),
                            &[kernel, channels],
                        )?,
                        dynamic: Some((
                            ck.bf16(
                                &dev,
                                &format!("{prefix}{name}.weight_proj"),
                                &[kernel * groups as usize, hidden],
                            )?,
                            groups as usize,
                        )),
                        kernel,
                        channels,
                    },
                })
            };
            let scale = |name: &str,
                         kind: ClipScale,
                         width: usize|
             -> Result<Option<CudaSlice<f32>>, String> {
                Ok(match kind {
                    ClipScale::PerHeadChannel => Some(ck.f32_dev(
                        &dev,
                        &format!("{prefix}attn.{name}.scale"),
                        &[heads, width],
                    )?),
                    ClipScale::None => None,
                })
            };
            let experts = m.expert_count as usize;
            let width = m.expert_intermediate_size as usize;
            let selection_bias = matches!(
                m.router,
                RouterPlan::Sigmoid {
                    selection_bias: true,
                    ..
                }
            );
            let ctrl = if slots.iter().any(|s| s.message) {
                let (query_mlp, key_value) =
                    cross.as_ref().ok_or("a message slot outside a loop")?;
                let index = p
                    .checked_sub(lp.core_start)
                    .filter(|i| (*i as usize) < core)
                    .ok_or("a message slot is a core layer")? as usize;
                let queries = memra_reference::inkling2::ops::message_queries(
                    query_mlp, key_value, index, hidden,
                );
                let rank = lp.gate_rank as usize;
                let mut ctrls = Vec::with_capacity(4);
                for target in ["attn", "k", "v", "mlp"] {
                    let cp = format!("{prefix}controllers.{target}.");
                    ctrls.push(CtrlW {
                        down: ck.bf16(&dev, &format!("{cp}down_weight"), &[rank, 2 * hidden])?,
                        forget: ck.bf16(&dev, &format!("{cp}forget_weight"), &[hidden, rank])?,
                        gain: ck.bf16(&dev, &format!("{cp}gain_weight"), &[hidden, rank])?,
                    });
                }
                Some((ctrls, dev.up(&queries)?))
            } else {
                None
            };
            layers.push(LayerW {
                norms,
                attn_queries,
                wq: ck.bf16(&dev, &format!("{prefix}attn.wq_du.weight"), &[qw, hidden])?,
                wr: ck.bf16(
                    &dev,
                    &format!("{prefix}attn.wr_du.weight"),
                    &[heads * rank, hidden],
                )?,
                sink: ck.bf16(
                    &dev,
                    &format!("{prefix}attn.attn_sink_proj.weight"),
                    &[heads, hidden],
                )?,
                gate: ck.bf16(
                    &dev,
                    &format!("{prefix}attn.gate_attn.weight"),
                    &[qw, hidden],
                )?,
                wk: ck.bf16(&dev, &format!("{prefix}attn.wk_dv.weight"), &[kvw, hidden])?,
                wv: ck.bf16(&dev, &format!("{prefix}attn.wv_dv.weight"), &[kvw, hidden])?,
                wo: ck.bf16(&dev, &format!("{prefix}attn.wo_ud.weight"), &[hidden, qw])?,
                q_scale: scale("q_l2_clip", a.query_clip.scale, dim)?,
                r_scale: scale("rel_l2_clip", a.relative.clip.scale, rank)?,
                table: ck.f32_dev(
                    &dev,
                    &format!("{prefix}attn.rel_logits_proj.proj"),
                    &[rank, extent],
                )?,
                kconv: conv("attn.k_sconv", &a.key_conv)?,
                vconv: conv("attn.v_sconv", &a.value_conv)?,
                oconv: conv("attn_sconv", &a.output_conv)?,
                mconv: conv("mlp_sconv", &program.mlp_output_conv)?,
                mlp_query: ck.f32_dev(
                    &dev,
                    &format!("{LLM}attn_res.mlp_queries.{p}"),
                    &[hidden],
                )?,
                mlp_norm: ck.f32_dev(&dev, &format!("{prefix}mlp_norm.weight"), &[hidden])?,
                router_w: ck.bf16(
                    &dev,
                    &format!("{prefix}mlp.gate.weight"),
                    &[experts, hidden],
                )?,
                router_b: if selection_bias {
                    Some(ck.f32_dev(&dev, &format!("{prefix}mlp.gate.bias"), &[experts])?)
                } else {
                    None
                },
                w13: Bank::load(
                    ck,
                    &dev,
                    &format!("{prefix}mlp.experts.w13_weight"),
                    numerics.weight,
                    experts,
                    2 * width,
                    hidden,
                )?,
                w2: Bank::load(
                    ck,
                    &dev,
                    &format!("{prefix}mlp.experts.w2_weight"),
                    numerics.weight,
                    experts,
                    hidden,
                    width,
                )?,
                ctrl,
            });
        }
        let embed = ck.bf16(&dev, &format!("{LLM}embed.weight"), &[vocab, hidden])?;
        let embed_norm = ck.f32_dev(&dev, &format!("{LLM}embed_norm.weight"), &[hidden])?;
        let unembed = ck.bf16(&dev, &format!("{LLM}unembed.weight"), &[vocab, hidden])?;
        let out_norm = ck.f32_dev(&dev, &format!("{LLM}norm.weight"), &[hidden])?;
        let out_query = ck.f32_dev(&dev, &format!("{LLM}attn_res.output_query"), &[hidden])?;
        let weight_bytes = dev.allocated.get();
        let mut account = MemoryAccount {
            device_total,
            driver_free_at_open: free_at_open,
            weights: weight_bytes,
            context,
            ..MemoryAccount::default()
        };

        // state (the full-attention caches, the only part that grows with the context, are
        // allocated last, after the admission check)
        let mut full_kv: Vec<(usize, usize)> = Vec::new();
        let mut slots = Vec::with_capacity(plan.layers.len());
        let (
            mut max_q,
            mut max_kv,
            mut max_r,
            mut max_heads,
            mut max_e,
            mut max_k,
            mut max_f,
            mut max_dyn,
        ) = (
            0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 1usize,
        );
        let mut blocks = BTreeMap::new();
        for layer in &plan.layers {
            let a = attention_of(layer)?;
            let m = moe_of(layer)?;
            let kvw = (a.kv_heads * a.head_dim) as usize;
            let cap = match a.window {
                // the window of a call's oldest row plus the rows of the call
                Some(window) => (window as usize - 1 + chunk).next_power_of_two(),
                None => context,
            };
            let history = a.key_conv.kernel as usize - 1;
            let kv_rows = if a.window.is_some() {
                account.kv_rings += 2 * cap * kvw * 4;
                cap
            } else {
                full_kv.push((slots.len(), cap * kvw));
                account.per_position += 2 * kvw * 4;
                1
            };
            let before = dev.allocated.get();
            slots.push(SlotState {
                kc: dev.zeros(kv_rows * kvw)?,
                vc: dev.zeros(kv_rows * kvw)?,
                cap,
                conv: [
                    [dev.zeros(history * kvw)?, dev.zeros(history * kvw)?],
                    [dev.zeros(history * kvw)?, dev.zeros(history * kvw)?],
                    [
                        dev.zeros((a.output_conv.kernel as usize - 1) * hidden)?,
                        dev.zeros((a.output_conv.kernel as usize - 1) * hidden)?,
                    ],
                    [
                        dev.zeros((program.mlp_output_conv.kernel as usize - 1) * hidden)?,
                        dev.zeros((program.mlp_output_conv.kernel as usize - 1) * hidden)?,
                    ],
                ],
            });
            account.conv += dev.allocated.get() - before - 2 * kv_rows * kvw * 4;
            max_q = max_q.max((a.query_heads * a.head_dim) as usize);
            max_kv = max_kv.max(kvw);
            max_r = max_r.max((a.query_heads * a.relative.rank) as usize);
            max_heads = max_heads.max(a.query_heads as usize);
            max_e = max_e.max(m.expert_count as usize);
            max_k = max_k.max(m.experts_per_token as usize);
            max_f = max_f.max(m.expert_intermediate_size as usize);
            if let ShortConvTaps::StaticPlusDynamic { groups } = a.key_conv.taps {
                max_dyn = max_dyn.max((groups * a.key_conv.kernel) as usize);
            }
            if residual_of(layer)?.starts_block {
                blocks.insert(layer.index, dev.zeros(chunk * hidden)?);
            }
        }
        let pairs = chunk * max_k;
        let tile_words = unsafe { ffi::memra_ink2g_tile_words() } as usize;
        let rh = chunk * hidden;
        let before_work = dev.allocated.get();
        let dq = DeclaredWork::alloc(&dev, if declared { pairs } else { 0 }, hidden, max_f)?;
        account.declared_work = dev.allocated.get() - before_work;
        let w = Work {
            h: dev.zeros(rh)?,
            prelude: dev.zeros(rh)?,
            bank_h: dev.zeros(rh)?,
            blocks,
            xin: dev.zeros(3 * rh)?,
            xc: dev.zeros(3 * rh)?,
            xn: dev.zeros(3 * rh)?,
            msg: dev.zeros(4 * rh)?,
            qb: dev.zeros(chunk * max_q)?,
            rb: dev.zeros(chunk * max_r)?,
            sinkb: dev.zeros(chunk * max_heads)?,
            gateb: dev.zeros(chunk * max_q)?,
            kp: dev.zeros(chunk * max_kv)?,
            vp: dev.zeros(chunk * max_kv)?,
            kcv: dev.zeros(chunk * max_kv)?,
            vcv: dev.zeros(chunk * max_kv)?,
            dynk: dev.zeros(chunk * max_dyn)?,
            dynv: dev.zeros(chunk * max_dyn)?,
            att: dev.zeros(chunk * max_q)?,
            proj: dev.zeros(rh)?,
            mixm: dev.zeros(rh)?,
            mn: dev.zeros(rh)?,
            rl: dev.zeros(chunk * max_e)?,
            ids: dev.zeros(pairs)?,
            wts: dev.zeros(pairs)?,
            margin: dev.zeros(chunk)?,
            tiles: dev.zeros(pairs * tile_words)?,
            gu: dev.zeros(pairs * 2 * max_f)?,
            act: dev.zeros(pairs * max_f)?,
            dn: dev.zeros(pairs * hidden)?,
            moe: dev.zeros(rh)?,
            logits: dev.zeros(HEAD_ROWS * keep)?,
            best: dev.zeros(HEAD_ROWS)?,
            tok: dev.zeros(chunk)?,
            dq,
        };
        account.work = dev.allocated.get() - before_work - account.declared_work;
        // ---- admission: the full-attention caches against what the driver has free now
        dev.sync()?;
        let (free_before_kv, _) = dev.driver_free()?;
        account.driver_free_before_kv = free_before_kv;
        account.max_context = free_before_kv / account.per_position.max(1);
        let kv_bytes = context * account.per_position;
        if kv_bytes > free_before_kv {
            return Err(format!(
                "context {context} needs {kv_bytes} bytes of full-attention cache and the driver has {free_before_kv} free; this card holds at most {} positions at chunk {chunk}. {}",
                account.max_context,
                account.describe()
            ));
        }
        for (slot, elems) in full_kv {
            slots[slot].kc = dev.zeros(elems)?;
            slots[slot].vc = dev.zeros(elems)?;
        }
        account.kv_full = kv_bytes;
        dev.sync()?;
        account.driver_free_after_load = dev.driver_free()?.0;
        Ok(Self {
            dev,
            plan: plan.clone(),
            layers,
            slots,
            embed,
            embed_norm,
            unembed,
            out_norm,
            out_query,
            w,
            chunk,
            context,
            position: 0,
            parity: 0,
            red: RedArms::default(),
            fail_after_slot: None,
            softcap,
            keep,
            hidden,
            weight_bytes,
            account,
        })
    }

    /// Kept vocabulary rows (the padded ids are never computed).
    pub fn vocab(&self) -> usize {
        self.keep
    }

    /// Forget the request: position 0, empty caches, zero convolution history.
    pub fn reset(&mut self) -> Result<(), String> {
        self.position = 0;
        self.parity = 0;
        for slot in &mut self.slots {
            for pair in &mut slot.conv {
                for conv in pair {
                    let zeros = vec![0f32; conv.len()];
                    self.dev.put(&zeros, conv)?;
                }
            }
        }
        Ok(())
    }

    /// The red arms of a gate run. Each arm is a wrong program of one class; asking for it on a
    /// route whose plan runs none of that class is refused, so a receipt never names an arm that
    /// did not run.
    pub fn set_red(&mut self, red: RedArms) -> Result<(), String> {
        let program = program(&self.plan)?;
        let float_layers = program
            .numerics
            .experts
            .iter()
            .filter(|e| e.activation == ExpertActivationMode::Float)
            .count();
        if red.dequant_bf16 && float_layers == 0 {
            return Err(
                "the dequant-bf16 red arm changes the float-class experts and this plan runs none"
                    .into(),
            );
        }
        self.red = red;
        Ok(())
    }

    pub fn red(&self) -> RedArms {
        self.red
    }

    #[allow(clippy::too_many_arguments)] // allow: one GEMM over one stored weight
    fn gemm_bf16(
        dev: &Dev,
        x: &CudaSlice<f32>,
        x_at: usize,
        w: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let st = &dev.st;
        dev.launched("memra_ink2g_gemm", unsafe {
            ffi::memra_ink2g_gemm(
                (dptr(x, st) + (x_at * 4) as u64) as *const c_void,
                k as i64,
                dptr(w, st) as *const c_void,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                1.0,
                FMT_BF16,
                0,
                std::ptr::null(),
                0,
                1,
                i32::from(m == 1),
                dptr_mut(y, st) as *mut c_void,
                n as i64,
                m as i32,
                n as i32,
                k as i32,
                dev.sp(),
            )
        })
    }

    #[allow(clippy::too_many_arguments)] // allow: one GEMM over one stacked bank
    fn gemm_bank(
        dev: &Dev,
        x: &CudaSlice<f32>,
        bank: &Bank,
        tiles: &CudaSlice<u32>,
        ntiles: usize,
        xdiv: usize,
        single: bool,
        y: &mut CudaSlice<f32>,
        round_bf16: bool,
    ) -> Result<(), String> {
        let st = &dev.st;
        let (ws, ss) = bank.strides();
        dev.launched("memra_ink2g_gemm", unsafe {
            ffi::memra_ink2g_gemm(
                dptr(x, st) as *const c_void,
                bank.k as i64,
                dptr(&bank.data, st) as *const c_void,
                dptr(&bank.scales, st) as *const c_void,
                ws,
                ss,
                dptr(&bank.amax, st) as *const f32,
                bank.denom,
                bank.fmt,
                i32::from(round_bf16),
                dptr(tiles, st) as *const u32,
                ntiles as i32,
                xdiv as i32,
                i32::from(single),
                dptr_mut(y, st) as *mut c_void,
                bank.n as i64,
                0,
                bank.n as i32,
                bank.k as i32,
                dev.sp(),
            )
        })
    }

    /// Depth mix of `states` (device pointers of `[rows, hidden]` buffers) with `nq` queries.
    #[allow(clippy::too_many_arguments)] // allow: the states, the queries and the output of one mix
    fn mix(
        dev: &Dev,
        states: &[u64],
        queries: &CudaSlice<f32>,
        nq: usize,
        eps: f64,
        out: &mut CudaSlice<f32>,
        out_stride: usize,
        rows: usize,
        hidden: usize,
    ) -> Result<(), String> {
        let st = &dev.st;
        let ptrs: Vec<*const f32> = states.iter().map(|p| *p as *const f32).collect();
        // the kernel's out is __restrict__: no state may live in it
        let lo = dptr(out, st);
        let hi = lo + (out.len() * 4) as u64;
        if states.iter().any(|p| *p >= lo && *p < hi) {
            return Err("the depth mix reads a state from its own output".into());
        }
        dev.launched("memra_ink2g_mix", unsafe {
            ffi::memra_ink2g_mix(
                ptrs.as_ptr(),
                ptrs.len() as i32,
                dptr(queries, st) as *const f32,
                nq as i32,
                eps,
                dptr_mut(out, st) as *mut f32,
                out_stride as i64,
                rows as i32,
                hidden as i32,
                dev.sp(),
            )
        })
    }

    /// One chunk of rows through every slot. `tokens.len() <= self.chunk`.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        head: Head,
        mut obs: Option<&mut (dyn Observer + '_)>,
    ) -> Result<CallOut, String> {
        let rows = tokens.len();
        if rows == 0 || rows > self.chunk {
            return Err(format!(
                "a call of {rows} rows; the route holds 1..={}",
                self.chunk
            ));
        }
        let pos0 = self.position;
        if pos0 as usize + rows > self.context {
            return Err(format!(
                "position {} is past the route's context of {}",
                pos0 as usize + rows,
                self.context
            ));
        }
        let vocab = self.plan.vocab_size;
        if let Some(bad) = tokens.iter().find(|t| **t >= vocab) {
            return Err(format!("token {bad} outside the vocabulary of {vocab}"));
        }
        let Self {
            dev,
            plan,
            layers,
            slots,
            embed,
            embed_norm,
            unembed,
            out_norm,
            out_query,
            w,
            chunk,
            position,
            parity,
            red,
            fail_after_slot,
            softcap,
            keep,
            hidden,
            ..
        } = self;
        let (hidden, chunk, red, softcap, keep) = (*hidden, *chunk, *red, *softcap, *keep);
        dev.bind()?;
        let st = dev.st.clone();
        let program = program(plan)?;
        let lp = &program.loop_plan;
        let rh = chunk * hidden;
        dev.put(tokens, &mut w.tok)?;
        dev.launched("memra_ink2g_embed", unsafe {
            ffi::memra_ink2g_embed(
                dptr(embed, &st) as *const c_void,
                dptr(&w.tok, &st) as *const u32,
                dptr(embed_norm, &st) as *const f32,
                rms(&program.embedding_norm)?,
                dptr_mut(&mut w.h, &st) as *mut f32,
                rows as i32,
                hidden as i32,
                dev.sp(),
            )
        })?;

        let mut blocks: Vec<u32> = Vec::new();
        let mut prelude_blocks: Option<Vec<u32>> = None;
        let mut bank: Option<Vec<u32>> = None;
        let mut pass = None;
        for &slot in &lp.execution_order {
            let info = lp.slots[slot as usize];
            if info.pass != pass {
                match info.pass {
                    Some(0) => {
                        dev.copy(&mut w.prelude, &w.h, rows * hidden)?;
                        prelude_blocks = Some(blocks.clone());
                    }
                    Some(_) => {
                        let list = prelude_blocks
                            .as_ref()
                            .ok_or("a loop pass follows the first pass")?;
                        let after_first = blocks
                            .get(list.len() + 1..)
                            .ok_or("the first core layer starts a residual block")?;
                        bank = Some(after_first.to_vec());
                        dev.copy(&mut w.bank_h, &w.h, rows * hidden)?;
                        dev.copy(&mut w.h, &w.prelude, rows * hidden)?;
                        blocks = list.clone();
                    }
                    None => {}
                }
                pass = info.pass;
            }
            let layer = &plan.layers[slot as usize];
            let lw = &layers[info.physical_layer as usize];
            let a = attention_of(layer)?;
            let r = residual_of(layer)?;
            let m = moe_of(layer)?;
            let mix_eps = decimal(r.epsilon);
            let has_msg = info.message;
            if has_msg {
                let (_, queries) = lw
                    .ctrl
                    .as_ref()
                    .ok_or("a message slot without controllers")?;
                let list = bank
                    .as_ref()
                    .ok_or("a slot reads a message before any pass ran")?;
                let mut states: Vec<u64> = list.iter().map(|b| dptr(&w.blocks[b], &st)).collect();
                states.push(dptr(&w.bank_h, &st));
                Self::mix(
                    dev, &states, queries, 4, mix_eps, &mut w.msg, rh, rows, hidden,
                )?;
            }
            let (heads, kvh, dim, rank, extent) = (
                a.query_heads as usize,
                a.kv_heads as usize,
                a.head_dim as usize,
                a.relative.rank as usize,
                a.relative.extent as usize,
            );
            let (qw, kvw) = (heads * dim, kvh * dim);
            let state_ptrs = |w: &Work, blocks: &[u32]| -> Vec<u64> {
                let mut s: Vec<u64> = blocks.iter().map(|b| dptr(&w.blocks[b], &st)).collect();
                s.push(dptr(&w.h, &st));
                s
            };

            // ---- attention inputs: mix, controllers, norms. src[i] is a device pointer.
            let h_ptr = dptr(&w.h, &st);
            let mut src = [h_ptr; 3];
            if let Some((queries, count)) = &lw.attn_queries {
                let states = state_ptrs(w, &blocks);
                Self::mix(
                    dev, &states, queries, *count, mix_eps, &mut w.xin, rh, rows, hidden,
                )?;
                let base = dptr(&w.xin, &st);
                for (i, s) in src.iter_mut().enumerate() {
                    *s = base + ((if *count == 3 { i } else { 0 }) * rh * 4) as u64;
                }
            }
            if has_msg {
                let (ctrls, _) = lw
                    .ctrl
                    .as_ref()
                    .ok_or("a message slot without controllers")?;
                let xc = dptr_mut(&mut w.xc, &st);
                let msg = dptr(&w.msg, &st);
                for i in 0..3 {
                    let out = xc + (i * rh * 4) as u64;
                    dev.launched("memra_ink2g_controller", unsafe {
                        ffi::memra_ink2g_controller(
                            src[i] as *const f32,
                            (msg + (i * rh * 4) as u64) as *const f32,
                            dptr(&ctrls[i].down, &st) as *const c_void,
                            dptr(&ctrls[i].forget, &st) as *const c_void,
                            dptr(&ctrls[i].gain, &st) as *const c_void,
                            decimal(lp.controller_epsilon),
                            out as *mut f32,
                            rows as i32,
                            hidden as i32,
                            lp.gate_rank as i32,
                            dev.sp(),
                        )
                    })?;
                    src[i] = out;
                }
            }
            let attn_eps = rms(&layer.pre_attention_norm)?;
            let xn = dptr_mut(&mut w.xn, &st);
            let mut normed = [xn; 3];
            for i in 0..3 {
                let norm_index = if lw.norms.len() == 3 { i } else { 0 };
                // one launch where an earlier input had the same source and the same norm
                if let Some(j) = (0..i).find(|j| {
                    src[*j] == src[i] && (if lw.norms.len() == 3 { *j } else { 0 }) == norm_index
                }) {
                    normed[i] = normed[j];
                    continue;
                }
                normed[i] = xn + (i * rh * 4) as u64;
                dev.launched("memra_ink2g_norm", unsafe {
                    ffi::memra_ink2g_norm(
                        src[i] as *const f32,
                        dptr(&lw.norms[norm_index], &st) as *const f32,
                        attn_eps,
                        normed[i] as *mut f32,
                        rows as i32,
                        hidden as i32,
                        dev.sp(),
                    )
                })?;
            }
            let [xq, xk, xv] = normed;

            // ---- projections
            let proj = |dev: &Dev,
                        x: u64,
                        wt: &CudaSlice<u8>,
                        y: &mut CudaSlice<f32>,
                        n: usize|
             -> Result<(), String> {
                dev.launched("memra_ink2g_gemm", unsafe {
                    ffi::memra_ink2g_gemm(
                        x as *const c_void,
                        hidden as i64,
                        dptr(wt, &st) as *const c_void,
                        std::ptr::null(),
                        0,
                        0,
                        std::ptr::null(),
                        1.0,
                        FMT_BF16,
                        0,
                        std::ptr::null(),
                        0,
                        1,
                        i32::from(rows == 1),
                        dptr_mut(y, &st) as *mut c_void,
                        n as i64,
                        rows as i32,
                        n as i32,
                        hidden as i32,
                        dev.sp(),
                    )
                })
            };
            proj(dev, xq, &lw.wq, &mut w.qb, qw)?;
            proj(dev, xq, &lw.wr, &mut w.rb, heads * rank)?;
            proj(dev, xq, &lw.sink, &mut w.sinkb, heads)?;
            proj(dev, xq, &lw.gate, &mut w.gateb, qw)?;
            proj(dev, xk, &lw.wk, &mut w.kp, kvw)?;
            proj(dev, xv, &lw.wv, &mut w.vp, kvw)?;
            if let Some((wp, groups)) = &lw.kconv.dynamic {
                proj(dev, xk, wp, &mut w.dynk, groups * lw.kconv.kernel)?;
            }
            if let Some((wp, groups)) = &lw.vconv.dynamic {
                proj(dev, xv, wp, &mut w.dynv, groups * lw.vconv.kernel)?;
            }

            // ---- clip / tau epilogues on Q and R
            let (log_floor, alpha) = match a.log_scale {
                Some(scale) => (scale.floor as i32, decimal(scale.alpha)),
                None => (0, 0.0),
            };
            let heads_k = |dev: &Dev,
                           mode: i32,
                           x: u64,
                           scale: Option<&CudaSlice<f32>>,
                           thr: f64,
                           eps: f64,
                           tau: bool,
                           dst: u64,
                           cap: usize,
                           nheads: usize,
                           width: usize|
             -> Result<(), String> {
                dev.launched("memra_ink2g_heads", unsafe {
                    ffi::memra_ink2g_heads(
                        mode,
                        x as *const f32,
                        scale.map_or(std::ptr::null(), |s| dptr(s, &st) as *const f32),
                        thr,
                        eps,
                        pos0 as i64,
                        if tau { log_floor } else { 0 },
                        alpha,
                        dst as *mut f32,
                        cap as i64,
                        rows as i32,
                        nheads as i32,
                        width as i32,
                        dev.sp(),
                    )
                })
            };
            let qb = dptr_mut(&mut w.qb, &st);
            heads_k(
                dev,
                0,
                qb,
                lw.q_scale.as_ref(),
                decimal(a.query_clip.threshold),
                decimal(a.query_clip.epsilon),
                true,
                qb,
                0,
                heads,
                dim,
            )?;
            let rb = dptr_mut(&mut w.rb, &st);
            heads_k(
                dev,
                0,
                rb,
                lw.r_scale.as_ref(),
                decimal(a.relative.clip.threshold),
                decimal(a.relative.clip.epsilon),
                true,
                rb,
                0,
                heads,
                rank,
            )?;

            // ---- K and V: short convolution with the slot's history, clip / norm into the cache
            let cur = *parity;
            let ss = &mut slots[slot as usize];
            let conv = |dev: &Dev,
                        cw: &ConvW,
                        x: u64,
                        state: &mut [CudaSlice<f32>; 2],
                        dynamic: u64,
                        add: u64,
                        out: u64|
             -> Result<(), String> {
                // read the committed history, write the next one into the other buffer
                let state_in = dptr(&state[cur], &st);
                let state_out = if red.conv_state_dropped {
                    0
                } else {
                    dptr_mut(&mut state[1 - cur], &st)
                };
                dev.launched("memra_ink2g_conv", unsafe {
                    ffi::memra_ink2g_conv(
                        i32::from(cw.dynamic.is_some()),
                        x as *const f32,
                        state_in as *const f32,
                        state_out as *mut f32,
                        dptr(&cw.taps, &st) as *const f32,
                        dynamic as *const f32,
                        cw.dynamic.as_ref().map_or(0, |d| d.1 as i32),
                        add as *const f32,
                        out as *mut f32,
                        rows as i32,
                        cw.channels as i32,
                        cw.kernel as i32,
                        dev.sp(),
                    )
                })
            };
            let kcv = dptr_mut(&mut w.kcv, &st);
            let vcv = dptr_mut(&mut w.vcv, &st);
            conv(
                dev,
                &lw.kconv,
                dptr(&w.kp, &st),
                &mut ss.conv[0],
                dptr(&w.dynk, &st),
                0,
                kcv,
            )?;
            conv(
                dev,
                &lw.vconv,
                dptr(&w.vp, &st),
                &mut ss.conv[1],
                dptr(&w.dynv, &st),
                0,
                vcv,
            )?;
            let kc = dptr_mut(&mut ss.kc, &st);
            let vc = dptr_mut(&mut ss.vc, &st);
            heads_k(
                dev,
                0,
                kcv,
                None,
                decimal(a.key_clip.threshold),
                decimal(a.key_clip.epsilon),
                false,
                kc,
                ss.cap,
                kvh,
                dim,
            )?;
            heads_k(
                dev,
                1,
                vcv,
                None,
                0.0,
                decimal(a.value_norm_epsilon),
                false,
                vc,
                ss.cap,
                kvh,
                dim,
            )?;

            // ---- attention, output gate, output projection, short convolution, residual
            dev.launched("memra_ink2g_attn", unsafe {
                ffi::memra_ink2g_attn(
                    qb as *const f32,
                    rb as *const f32,
                    dptr(&lw.table, &st) as *const f32,
                    dptr(&w.sinkb, &st) as *const f32,
                    kc as *const f32,
                    vc as *const f32,
                    ss.cap as i64,
                    pos0 as i64,
                    a.window.map_or(0, |v| v as i32),
                    extent as i32,
                    i32::from(a.relative.beyond_extent == BeyondExtent::Masked),
                    dptr_mut(&mut w.att, &st) as *mut f32,
                    rows as i32,
                    heads as i32,
                    kvh as i32,
                    dim as i32,
                    rank as i32,
                    dev.sp(),
                )
            })?;
            if let Some(o) = obs.as_deref_mut()
                && o.attn_probe(slot)
            {
                o.floats(slot, Tap::AttnQ, &dev.down(&w.qb, 0, rows * qw)?);
                o.floats(
                    slot,
                    Tap::AttnRel,
                    &dev.down(&w.rb, 0, rows * heads * rank)?,
                );
                o.floats(slot, Tap::AttnSink, &dev.down(&w.sinkb, 0, rows * heads)?);
                o.floats(
                    slot,
                    Tap::AttnTable,
                    &dev.down(&lw.table, 0, rank * extent)?,
                );
                o.floats(slot, Tap::AttnKeys, &dev.down(&ss.kc, 0, ss.cap * kvw)?);
                o.floats(slot, Tap::AttnValues, &dev.down(&ss.vc, 0, ss.cap * kvw)?);
                o.floats(slot, Tap::AttnOut, &dev.down(&w.att, 0, rows * qw)?);
            }
            dev.launched("memra_ink2g_gate", unsafe {
                ffi::memra_ink2g_gate(
                    dptr_mut(&mut w.att, &st) as *mut f32,
                    dptr(&w.gateb, &st) as *const f32,
                    (rows * qw) as i64,
                    dev.sp(),
                )
            })?;
            dev.launched("memra_ink2g_gemm", unsafe {
                ffi::memra_ink2g_gemm(
                    dptr(&w.att, &st) as *const c_void,
                    qw as i64,
                    dptr(&lw.wo, &st) as *const c_void,
                    std::ptr::null(),
                    0,
                    0,
                    std::ptr::null(),
                    1.0,
                    FMT_BF16,
                    0,
                    std::ptr::null(),
                    0,
                    1,
                    i32::from(rows == 1),
                    dptr_mut(&mut w.proj, &st) as *mut c_void,
                    hidden as i64,
                    rows as i32,
                    hidden as i32,
                    qw as i32,
                    dev.sp(),
                )
            })?;
            let h_mut = dptr_mut(&mut w.h, &st);
            if r.starts_block {
                // the incoming stream becomes a block; the layer starts a new stream
                let Work {
                    blocks: pool, h, ..
                } = &mut *w;
                let buffer = pool
                    .get_mut(&slot)
                    .ok_or("a block-starting slot without a block buffer")?;
                dev.copy(buffer, h, rows * hidden)?;
                blocks.push(slot);
                conv(
                    dev,
                    &lw.oconv,
                    dptr(&w.proj, &st),
                    &mut ss.conv[2],
                    0,
                    0,
                    h_mut,
                )?;
            } else {
                conv(
                    dev,
                    &lw.oconv,
                    dptr(&w.proj, &st),
                    &mut ss.conv[2],
                    0,
                    h_mut,
                    h_mut,
                )?;
            }
            if let Some(o) = obs.as_deref_mut()
                && o.wants(Tap::ResidAttn)
            {
                o.floats(slot, Tap::ResidAttn, &dev.down(&w.h, 0, rows * hidden)?);
            }

            // ---- MLP: mix, controller, norm, routed experts, short convolution, residual
            let states = state_ptrs(w, &blocks);
            Self::mix(
                dev,
                &states,
                &lw.mlp_query,
                1,
                mix_eps,
                &mut w.mixm,
                rh,
                rows,
                hidden,
            )?;
            if has_msg {
                let (ctrls, _) = lw
                    .ctrl
                    .as_ref()
                    .ok_or("a message slot without controllers")?;
                let mixm = dptr_mut(&mut w.mixm, &st);
                dev.launched("memra_ink2g_controller", unsafe {
                    ffi::memra_ink2g_controller(
                        mixm as *const f32,
                        (dptr(&w.msg, &st) + (3 * rh * 4) as u64) as *const f32,
                        dptr(&ctrls[3].down, &st) as *const c_void,
                        dptr(&ctrls[3].forget, &st) as *const c_void,
                        dptr(&ctrls[3].gain, &st) as *const c_void,
                        decimal(lp.controller_epsilon),
                        mixm as *mut f32,
                        rows as i32,
                        hidden as i32,
                        lp.gate_rank as i32,
                        dev.sp(),
                    )
                })?;
            }
            dev.launched("memra_ink2g_norm", unsafe {
                ffi::memra_ink2g_norm(
                    dptr(&w.mixm, &st) as *const f32,
                    dptr(&lw.mlp_norm, &st) as *const f32,
                    rms(&layer.pre_mlp_norm)?,
                    dptr_mut(&mut w.mn, &st) as *mut f32,
                    rows as i32,
                    hidden as i32,
                    dev.sp(),
                )
            })?;
            if let Some(o) = obs.as_deref_mut() {
                if let Some(forced) = o.forced_moe_in(slot, rows) {
                    if forced.len() != rows * hidden {
                        return Err("a forced MoE input of the wrong size".into());
                    }
                    dev.put(&forced, &mut w.mn)?;
                }
                if o.wants(Tap::MoeIn) {
                    o.floats(slot, Tap::MoeIn, &dev.down(&w.mn, 0, rows * hidden)?);
                }
            }
            let mode = program.numerics.experts[info.physical_layer as usize].activation;
            Self::experts(
                dev, w, lw, m, program, mode, red, rows, hidden, slot, &mut obs,
            )?;
            if let Some(o) = obs.as_deref_mut()
                && o.wants(Tap::MoeOut)
            {
                o.floats(slot, Tap::MoeOut, &dev.down(&w.moe, 0, rows * hidden)?);
            }
            conv(
                dev,
                &lw.mconv,
                dptr(&w.moe, &st),
                &mut ss.conv[3],
                0,
                h_mut,
                h_mut,
            )?;
            if let Some(o) = obs.as_deref_mut()
                && o.wants(Tap::ResidFfn)
            {
                o.floats(slot, Tap::ResidFfn, &dev.down(&w.h, 0, rows * hidden)?);
            }
            if *fail_after_slot == Some(slot) {
                *fail_after_slot = None;
                if red.commit_on_error {
                    // the red arm: the histories this call wrote become the committed ones
                    *parity = 1 - *parity;
                }
                return Err(format!("injected failure after slot {slot}"));
            }
        }

        // ---- head: output mix, norm, unembed over the kept rows, softcap
        let last = plan.layers.last().ok_or("an empty plan")?;
        let mix_eps = decimal(residual_of(last)?.epsilon);
        let mut states: Vec<u64> = blocks.iter().map(|b| dptr(&w.blocks[b], &st)).collect();
        states.push(dptr(&w.h, &st));
        Self::mix(
            dev,
            &states,
            out_query,
            1,
            mix_eps,
            &mut w.mixm,
            rh,
            rows,
            hidden,
        )?;
        dev.launched("memra_ink2g_norm", unsafe {
            ffi::memra_ink2g_norm(
                dptr(&w.mixm, &st) as *const f32,
                dptr(out_norm, &st) as *const f32,
                rms(&plan.output_norm)?,
                dptr_mut(&mut w.mn, &st) as *mut f32,
                rows as i32,
                hidden as i32,
                dev.sp(),
            )
        })?;
        let mut out = CallOut {
            argmax: Vec::new(),
            logits: Vec::new(),
        };
        let first = match head {
            Head::None => rows,
            Head::LastArgmax => rows - 1,
            Head::All => 0,
        };
        let mut at = first;
        while at < rows {
            let n = HEAD_ROWS.min(rows - at);
            Self::gemm_bf16(
                dev,
                &w.mn,
                at * hidden,
                unembed,
                &mut w.logits,
                n,
                keep,
                hidden,
            )?;
            dev.launches.set(dev.launches.get() + 1); // softcap and argmax are two launches
            dev.launched("memra_ink2g_head", unsafe {
                ffi::memra_ink2g_head(
                    dptr_mut(&mut w.logits, &st) as *mut f32,
                    softcap,
                    dptr_mut(&mut w.best, &st) as *mut u32,
                    n as i32,
                    keep as i32,
                    dev.sp(),
                )
            })?;
            out.argmax.extend(dev.down(&w.best, 0, n)?);
            if head == Head::All {
                out.logits.extend(dev.down(&w.logits, 0, n * keep)?);
            }
            at += n;
        }
        // the call succeeded: commit the position and the conv histories it wrote
        *position += rows as u64;
        *parity = 1 - *parity;
        Ok(out)
    }

    /// The routed experts of `rows` rows: `w.mn` in, `w.moe` out.
    #[allow(clippy::too_many_arguments)] // allow: the layer, its plan and the call's rows
    fn experts(
        dev: &Dev,
        w: &mut Work,
        lw: &LayerW,
        m: &MoeMlpPlan,
        program: &Inkling2Plan,
        mode: ExpertActivationMode,
        red: RedArms,
        rows: usize,
        hidden: usize,
        slot: u32,
        obs: &mut Option<&mut (dyn Observer + '_)>,
    ) -> Result<(), String> {
        let st = dev.st.clone();
        let experts = m.expert_count as usize;
        let top_k = m.experts_per_token as usize;
        let width = m.expert_intermediate_size as usize;
        let pairs = rows * top_k;
        let RouterPlan::Sigmoid { scaling_factor, .. } = m.router else {
            return Err("a router other than normalized sigmoid".into());
        };
        let ActivationPlan::SwiGluSoftcapped { gate_cap, up_cap } = m.activation else {
            return Err("an activation other than softcapped SwiGLU".into());
        };
        Self::gemm_bf16(
            dev,
            &w.mn,
            0,
            &lw.router_w,
            &mut w.rl,
            rows,
            experts,
            hidden,
        )?;
        dev.launched("memra_ink2g_route", unsafe {
            ffi::memra_ink2g_route(
                dptr(&w.rl, &st) as *const f32,
                lw.router_b
                    .as_ref()
                    .map_or(std::ptr::null(), |b| dptr(b, &st) as *const f32),
                decimal(scaling_factor),
                top_k as i32,
                i32::from(red.ties_high),
                dptr_mut(&mut w.ids, &st) as *mut u32,
                dptr_mut(&mut w.wts, &st) as *mut f32,
                dptr_mut(&mut w.margin, &st) as *mut f32,
                rows as i32,
                experts as i32,
                dev.sp(),
            )
        })?;
        // The picks leave the device only when a caller reads them or the call holds several rows
        // (the expert tiles of a chunk are grouped on the host).
        let mut host_ids: Option<Vec<u32>> = None;
        if let Some(o) = obs.as_deref_mut() {
            let ids = dev.down(&w.ids, 0, pairs)?;
            o.picks(slot, &ids);
            if o.wants(Tap::RouteWeights) {
                o.floats(slot, Tap::RouteWeights, &dev.down(&w.wts, 0, pairs)?);
            }
            if o.wants(Tap::RouteMargin) {
                o.floats(slot, Tap::RouteMargin, &dev.down(&w.margin, 0, rows)?);
            }
            host_ids = Some(ids);
            if let Some((ids, wts)) = o.forced(slot, rows) {
                if ids.len() != pairs
                    || wts.len() != pairs
                    || ids.iter().any(|e| *e as usize >= experts)
                {
                    return Err("forced picks of the wrong shape".into());
                }
                dev.put(&ids, &mut w.ids)?;
                dev.put(&wts, &mut w.wts)?;
                host_ids = Some(ids);
            }
        }
        if rows > 1 && host_ids.is_none() {
            dev.round_trips.set(dev.round_trips.get() + 1);
            host_ids = Some(dev.down(&w.ids, 0, pairs)?);
        }
        match mode {
            ExpertActivationMode::Float => {
                // BRING-UP: the stored expert bytes on f32 inputs.
                let ntiles = match (&host_ids, rows) {
                    (Some(ids), r) if r > 1 => {
                        let words = unsafe { ffi::memra_ink2g_tile_words() } as usize;
                        let (table, n) = tiles_table(ids, words);
                        dev.put(&table, &mut w.tiles)?;
                        n
                    }
                    _ => {
                        dev.launched("memra_ink2g_tiles", unsafe {
                            ffi::memra_ink2g_tiles(
                                dptr(&w.ids, &st) as *const u32,
                                dptr_mut(&mut w.tiles, &st) as *mut u32,
                                pairs as i32,
                                dev.sp(),
                            )
                        })?;
                        pairs
                    }
                };
                let before = program.numerics.routed_weight_before_down;
                Self::gemm_bank(
                    dev,
                    &w.mn,
                    &lw.w13,
                    &w.tiles,
                    ntiles,
                    top_k,
                    rows == 1,
                    &mut w.gu,
                    red.dequant_bf16,
                )?;
                dev.launched("memra_ink2g_swiglu", unsafe {
                    ffi::memra_ink2g_swiglu(
                        dptr(&w.gu, &st) as *const f32,
                        if before {
                            dptr(&w.wts, &st) as *const f32
                        } else {
                            std::ptr::null()
                        },
                        decimal(gate_cap),
                        decimal(up_cap),
                        dptr_mut(&mut w.act, &st) as *mut f32,
                        pairs as i32,
                        width as i32,
                        dev.sp(),
                    )
                })?;
                Self::gemm_bank(
                    dev,
                    &w.act,
                    &lw.w2,
                    &w.tiles,
                    ntiles,
                    1,
                    rows == 1,
                    &mut w.dn,
                    red.dequant_bf16,
                )?;
                dev.launched("memra_ink2g_sum", unsafe {
                    ffi::memra_ink2g_sum(
                        dptr(&w.dn, &st) as *const f32,
                        if before {
                            std::ptr::null()
                        } else {
                            dptr(&w.wts, &st) as *const f32
                        },
                        dptr_mut(&mut w.moe, &st) as *mut f32,
                        rows as i32,
                        top_k as i32,
                        hidden as i32,
                        dev.sp(),
                    )
                })
            }
            ExpertActivationMode::Declared => declared_experts(
                dev,
                &mut w.dq,
                &w.mn,
                &w.ids,
                &w.wts,
                host_ids.as_deref(),
                &lw.w13,
                &lw.w2,
                (rows, top_k, hidden, width),
                (gate_cap, up_cap),
                DeclaredRed::default(),
                &mut w.moe,
            ),
        }
    }

    /// A request in chunks of at most `self.chunk` rows.
    pub fn call(
        &mut self,
        tokens: &[u32],
        head: Head,
        mut obs: Option<&mut (dyn Observer + '_)>,
    ) -> Result<CallOut, String> {
        let mut out = CallOut {
            argmax: Vec::new(),
            logits: Vec::new(),
        };
        let chunks: Vec<&[u32]> = tokens.chunks(self.chunk).collect();
        let count = chunks.len();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let last = i + 1 == count;
            if head == Head::LastArgmax && !last {
                self.forward(chunk, Head::None, obs.as_deref_mut())?;
                continue;
            }
            let part = self.forward(chunk, head, obs.as_deref_mut())?;
            out.argmax.extend(part.argmax);
            out.logits.extend(part.logits);
        }
        Ok(out)
    }
}

/// Buffers of the declared expert program for up to `pairs` routed pairs.
struct DeclaredWork {
    xb16: CudaSlice<u8>,
    groups: CudaSlice<u32>,
    aq_v: CudaSlice<u8>,
    aq_s: CudaSlice<u8>,
    aq_a: CudaSlice<f32>,
    gu16: CudaSlice<u8>,
    dn16: CudaSlice<u8>,
    /// Red-arm buffers (the gate only): the pair -> token row list and unit routing weights.
    grows: Option<CudaSlice<u32>>,
    ones: Option<CudaSlice<f32>>,
}

/// Wrong declared programs the gate must tell apart from the right one (fault injection).
#[derive(Debug, Clone, Copy, Default)]
struct DeclaredRed {
    /// The routing weight multiplies the bf16 down rows instead of the SiTU input of the down
    /// projection's quantizer.
    weight_after_down: bool,
    /// The token row is quantized once, at the token's index, and the GEMM reads the pair's
    /// index (the route's program before memra#1397's fix).
    token_rows: bool,
}

impl DeclaredWork {
    /// The buffers the red arms read.
    fn with_red(mut self, dev: &Dev, pairs: usize, top_k: usize) -> Result<Self, String> {
        let rows: Vec<u32> = (0..pairs).map(|p| (p / top_k) as u32).collect();
        self.grows = Some(dev.up(&rows)?);
        self.ones = Some(dev.up(&vec![1f32; pairs])?);
        Ok(self)
    }

    /// `pairs == 0`: a route with no declared layer holds placeholders.
    fn alloc(dev: &Dev, pairs: usize, hidden: usize, max_f: usize) -> Result<Self, String> {
        let p = pairs.max(1);
        let widest = hidden.max(max_f);
        Ok(Self {
            xb16: dev.zeros(p * hidden * 2)?,
            groups: dev.zeros(p * GROUP_WORDS)?,
            aq_v: dev.zeros(p * widest)?,
            aq_s: dev.zeros(p * widest / 16)?,
            aq_a: dev.zeros(p)?,
            gu16: dev.zeros(p * 2 * max_f * 2)?,
            dn16: dev.zeros(p * hidden * 2)?,
            grows: None,
            ones: None,
        })
    }
}

/// The declared expert program of one layer (memra_reference::inkling2_actquant::expert_block_*):
/// the MoE input `mn` rounded to bf16, quantized once per routed pair, the gate/up GEMM, SiTU
/// times the routing weight and its quantization, the down GEMM, and the bf16 rows summed in rank
/// order into `out`. `ids` / `wts` are the router's picks on the device; `host_ids` (a call of
/// several rows) groups the pairs by expert on the host, else every pair is its own group.
/// `shape` is (rows, top_k, hidden, width).
#[allow(clippy::too_many_arguments)]
fn declared_experts(
    dev: &Dev,
    dq: &mut DeclaredWork,
    mn: &CudaSlice<f32>,
    ids: &CudaSlice<u32>,
    wts: &CudaSlice<f32>,
    host_ids: Option<&[u32]>,
    w13: &Bank,
    w2: &Bank,
    shape: (usize, usize, usize, usize),
    caps: (f32, f32),
    red: DeclaredRed,
    out: &mut CudaSlice<f32>,
) -> Result<(), String> {
    let st = dev.st.clone();
    let (rows, top_k, hidden, width) = shape;
    let pairs = rows * top_k;
    let red_rows = if red.token_rows {
        dptr(
            dq.grows
                .as_ref()
                .ok_or("the token-rows red arm needs its buffers")?,
            &st,
        )
    } else {
        0
    };
    let gamma_in = if red.weight_after_down {
        dptr(
            dq.ones
                .as_ref()
                .ok_or("the weight-after-down red arm needs its buffers")?,
            &st,
        )
    } else {
        dptr(wts, &st)
    };
    if w13.fmt != w2.fmt || w13.k != hidden || w13.n != 2 * width || w2.k != width || w2.n != hidden
    {
        return Err("declared experts: bank shapes disagree with the layer".into());
    }
    let kind = if w13.fmt == FMT_NVFP4 { 0 } else { 1 };
    dev.launched("memra_ink2g_to_bf16", unsafe {
        ffi::memra_ink2g_to_bf16(
            dptr(mn, &st) as *const f32,
            dptr_mut(&mut dq.xb16, &st) as *mut c_void,
            rows as i32,
            hidden as i32,
            if red.token_rows { 1 } else { top_k as i32 },
            dev.sp(),
        )
    })?;
    let ngroups = match host_ids {
        Some(ids) if rows > 1 => {
            let (table, n) = tiles_table(ids, GROUP_WORDS);
            dev.put(&table, &mut dq.groups)?;
            n
        }
        _ => {
            dev.launched("memra_ink2g_groups", unsafe {
                ffi::memra_ink2g_groups(
                    dptr(ids, &st) as *const u32,
                    dptr_mut(&mut dq.groups, &st) as *mut u32,
                    pairs as i32,
                    dev.sp(),
                )
            })?;
            pairs
        }
    };
    // One block per routed pair: the input holds a row per pair (the token row repeated, or the
    // pair's gate/up row). The quantizer writes the row it reads.
    let quantize = |dq: &mut DeclaredWork, situ: bool, k: usize| -> Result<(), String> {
        let (x, gamma, sel) = if situ {
            (dptr(&dq.gu16, &st), gamma_in, 0)
        } else {
            (dptr(&dq.xb16, &st), 0, red_rows)
        };
        dev.launched("memra_ink2_actq", unsafe {
            ffi::memra_ink2_actq(
                kind,
                i32::from(situ),
                x as *const c_void,
                gamma as *const f32,
                sel as *const u32,
                pairs as i32,
                dptr_mut(&mut dq.aq_v, &st) as *mut c_void,
                dptr_mut(&mut dq.aq_s, &st) as *mut c_void,
                dptr_mut(&mut dq.aq_a, &st) as *mut f32,
                std::ptr::null_mut(),
                k as i32,
                caps.0,
                caps.1,
                0,
                dev.sp(),
            )
        })
    };
    let gemm = |dq: &mut DeclaredWork, bank: &Bank, to_gu: bool| -> Result<(), String> {
        let out = if to_gu {
            dptr_mut(&mut dq.gu16, &st)
        } else {
            dptr_mut(&mut dq.dn16, &st)
        };
        dev.launched("memra_ink2_expert_gemm", unsafe {
            ffi::memra_ink2_expert_gemm(
                kind,
                1,
                dptr(&dq.aq_v, &st) as *const c_void,
                dptr(&dq.aq_s, &st) as *const c_void,
                dptr(&dq.aq_a, &st) as *const f32,
                dptr(&bank.data, &st) as *const c_void,
                dptr(&bank.scales, &st) as *const c_void,
                dptr(&bank.amax, &st) as *const f32,
                bank.denom,
                dptr(&dq.groups, &st) as *const u32,
                ngroups as i32,
                out as *mut c_void,
                std::ptr::null_mut(),
                bank.n as i32,
                bank.k as i32,
                0,
                dev.sp(),
            )
        })
    };
    quantize(dq, false, hidden)?;
    gemm(dq, w13, true)?;
    quantize(dq, true, width)?;
    gemm(dq, w2, false)?;
    dev.launched("memra_ink2g_sum_bf16", unsafe {
        ffi::memra_ink2g_sum_bf16(
            dptr(&dq.dn16, &st) as *const c_void,
            if red.weight_after_down {
                dptr(wts, &st) as *const f32
            } else {
                std::ptr::null()
            },
            dptr_mut(out, &st) as *mut f32,
            rows as i32,
            top_k as i32,
            hidden as i32,
            dev.sp(),
        )
    })
}

// ---- kernel gate ---------------------------------------------------------------------------------
// Every kernel of cu/inkling2_gpu.cu against the f64 reference operation of the same role
// (memra_reference::inkling2::ops) on seeded data, plus the properties the route leans on: a row's
// bits do not depend on the rows beside it, state carried across calls equals one call, and each
// red arm is a different function.

/// Expert tiles of a chunk: `{expert, count, pair[..]}` records of `words` u32, pairs grouped by expert.
fn tiles_table(ids: &[u32], words: usize) -> (Vec<u32>, usize) {
    let mut by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (pair, expert) in ids.iter().enumerate() {
        by.entry(*expert).or_default().push(pair as u32);
    }
    let mut table = Vec::new();
    let mut n = 0;
    for (expert, list) in by {
        for chunk in list.chunks(words - 2) {
            table.push(expert);
            table.push(chunk.len() as u32);
            table.extend_from_slice(chunk);
            table.extend(std::iter::repeat_n(0u32, words - 2 - chunk.len()));
            n += 1;
        }
    }
    (table, n)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self, scale: f32) -> f32 {
        ((0..6).map(|_| self.unit()).sum::<f32>() - 3.0) * scale * 1.4
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.gauss(scale)).collect()
    }
    /// Values a BF16 tensor can hold.
    fn bf16(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| memra_reference::inkling2_actquant::bf16(self.gauss(scale)))
            .collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(n);
        v
    }
}

fn bf16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn wide(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| f64::from(*v)).collect()
}

/// Largest `|got - want| / (|want| + floor)`.
fn worst_rel(got: &[f32], want: &[f64], floor: f64) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / (w.abs() + floor))
        .fold(0.0, f64::max)
}

fn bits_differ(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// The gate's cells: lines to print and the number that failed.
pub fn ink2g_gate(dev: &Dev) -> Result<(Vec<String>, usize), String> {
    use memra_reference::inkling2::ops;
    dev.bind()?;
    let st = dev.st.clone();
    let mut lines = Vec::new();
    let mut fails = 0usize;
    let mut verdict = |ok: bool, line: String| {
        if !ok {
            fails += 1;
        }
        lines.push(format!("{line} {}", if ok { "OK" } else { "FAIL" }));
    };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    // a result that rounds an f64 value once to f32
    const ROUND: f64 = 1.2e-7;

    // ---- BF16 GEMM
    {
        let (m, n, k) = (70usize, 200usize, 320usize);
        let wv = rng.bf16(n * k, 0.05);
        let x = rng.vec(m * k, 1.0);
        let (wd, xd) = (dev.up(&bf16_bytes(&wv))?, dev.up(&x)?);
        let mut y = dev.zeros::<f32>(m * n)?;
        Route::gemm_bf16(dev, &xd, 0, &wd, &mut y, m, n, k)?;
        let got = dev.down(&y, 0, m * n)?;
        let mut ratio = 0f64;
        for r in 0..m {
            let xr = wide(&x[r * k..(r + 1) * k]);
            for o in 0..n {
                let (mut exact, mut mag) = (0f64, 0f64);
                for i in 0..k {
                    let p = f64::from(wv[o * k + i]) * xr[i];
                    exact += p;
                    mag += p.abs();
                }
                ratio = ratio.max((f64::from(got[r * n + o]) - exact).abs() / mag.max(1e-300));
            }
        }
        let mut alone = 0usize;
        for r in [0usize, 13, 63, 64, 69] {
            let mut one = dev.zeros::<f32>(n)?;
            Route::gemm_bf16(dev, &xd, r * k, &wd, &mut one, 1, n, k)?;
            alone += bits_differ(&dev.down(&one, 0, n)?, &got[r * n..(r + 1) * n]);
        }
        let bound = k as f64 * 2f64.powi(-23);
        verdict(
            ratio <= bound && alone == 0,
            format!(
                "ink2g-gemm-bf16 [m={m} n={n} k={k}] max|err|/sum|p|={ratio:.3e} (bound {bound:.3e}); rows alone (decode GEMV) vs in the call (tile GEMM) bit-differ={alone}"
            ),
        );
    }

    // ---- expert banks as stored: NVFP4 and MXFP8, expert tiles, the dequant-bf16 red arm
    for (name, fmt) in [
        ("ink2g-gemm-nvfp4", FMT_NVFP4),
        ("ink2g-gemm-mxfp8", FMT_MXFP8),
    ] {
        let (experts, n, k, rows, top_k) = (5usize, 130usize, 96usize, 30usize, 5usize);
        let pairs = rows * top_k;
        let (data, scales, amax, denom, bank_ref) = if fmt == FMT_NVFP4 {
            let codes = rng.bytes(experts * n * k / 2);
            let mut scales = rng.bytes(experts * n * k / 16);
            scales.iter_mut().for_each(|s| *s = 0x20 + (*s % 0x50));
            let amax: Vec<f32> = (0..experts).map(|_| 0.5 + rng.unit()).collect();
            let bank = ops::ExpertBank::Nvfp4 {
                codes: codes.clone(),
                scales: scales.clone(),
                amax: amax.clone(),
                denominator: 2688.0,
                rows: n,
                inputs: k,
            };
            (codes, scales, amax, 2688.0f32, bank)
        } else {
            let mut values = rng.bytes(experts * n * k);
            values.iter_mut().for_each(|v| {
                if *v & 0x7f == 0x7f {
                    *v &= 0x80;
                }
            });
            let mut scales = rng.bytes(experts * n * k / 32);
            scales.iter_mut().for_each(|s| *s = 110 + (*s % 14));
            let bank = ops::ExpertBank::Mxfp8 {
                values: values.clone(),
                exponents: scales.clone(),
                rows: n,
                inputs: k,
            };
            (values, scales, vec![1.0f32], 1.0f32, bank)
        };
        let bank = Bank {
            fmt,
            data: dev.up(&data)?,
            scales: dev.up(&scales)?,
            amax: dev.up(&amax)?,
            denom,
            n,
            k,
        };
        let x = rng.vec(rows * k, 1.0);
        let ids: Vec<u32> = (0..pairs)
            .map(|_| (rng.next() % experts as u64) as u32)
            .collect();
        let words = unsafe { ffi::memra_ink2g_tile_words() } as usize;
        let (table, ntiles) = tiles_table(&ids, words);
        let (xd, td) = (dev.up(&x)?, dev.up(&table)?);
        let mut y = dev.zeros::<f32>(pairs * n)?;
        Route::gemm_bank(dev, &xd, &bank, &td, ntiles, top_k, false, &mut y, false)?;
        let got = dev.down(&y, 0, pairs * n)?;
        Route::gemm_bank(dev, &xd, &bank, &td, ntiles, top_k, false, &mut y, true)?;
        let red = dev.down(&y, 0, pairs * n)?;
        let weights: Vec<Vec<f64>> = (0..experts).map(|e| bank_ref.expert(e)).collect();
        let (mut ratio, mut red_ratio) = (0f64, 0f64);
        for p in 0..pairs {
            let xr = wide(&x[(p / top_k) * k..(p / top_k + 1) * k]);
            let we = &weights[ids[p] as usize];
            for o in 0..n {
                let (mut exact, mut mag, mut rexact) = (0f64, 0f64, 0f64);
                for i in 0..k {
                    let wv = we[o * k + i];
                    exact += wv * xr[i];
                    mag += (wv * xr[i]).abs();
                    rexact +=
                        f64::from(memra_reference::inkling2_actquant::bf16(wv as f32)) * xr[i];
                }
                ratio = ratio.max((f64::from(got[p * n + o]) - exact).abs() / mag.max(1e-300));
                red_ratio =
                    red_ratio.max((f64::from(red[p * n + o]) - rexact).abs() / mag.max(1e-300));
            }
        }
        // a decode step's tiles are built on the device: one pair a tile
        let mut tiles_dev = dev.zeros::<u32>(top_k * words)?;
        let ids_dev = dev.up(&ids[..top_k])?;
        dev.launched("memra_ink2g_tiles", unsafe {
            ffi::memra_ink2g_tiles(
                dptr(&ids_dev, &st) as *const u32,
                dptr_mut(&mut tiles_dev, &st) as *mut u32,
                top_k as i32,
                dev.sp(),
            )
        })?;
        let mut one = dev.zeros::<f32>(top_k * n)?;
        Route::gemm_bank(
            dev, &xd, &bank, &tiles_dev, top_k, top_k, true, &mut one, false,
        )?;
        let mut one_red = dev.zeros::<f32>(top_k * n)?;
        Route::gemm_bank(
            dev,
            &xd,
            &bank,
            &tiles_dev,
            top_k,
            top_k,
            true,
            &mut one_red,
            true,
        )?;
        let alone_red = bits_differ(&dev.down(&one_red, 0, top_k * n)?, &red[..top_k * n]);
        let alone = bits_differ(&dev.down(&one, 0, top_k * n)?, &got[..top_k * n]);
        let bound = (k + 2) as f64 * 2f64.powi(-23);
        let red_moves = bits_differ(&got, &red);
        // MXFP8 weights are exact in bf16: the red arm is the same function there
        let red_ok = if fmt == FMT_NVFP4 {
            red_moves > 0
        } else {
            true
        };
        verdict(
            ratio <= bound && red_ratio <= bound && alone == 0 && alone_red == 0 && red_ok,
            format!(
                "{name} [experts={experts} n={n} k={k} pairs={pairs} tiles={ntiles}] max|err|/sum|p|={ratio:.3e} (bound {bound:.3e}); decode GEMV on device tiles vs the tile GEMM bit-differ={alone} (red arm {alone_red}); red arm dequant-bf16: vs its own reference {red_ratio:.3e}, elements moved {red_moves}/{}",
                pairs * n
            ),
        );
    }

    // ---- embedding and norm
    {
        let (vocab, h, rows) = (50usize, 2304usize, 9usize);
        let table = rng.bf16(vocab * h, 0.02);
        let nw = rng.vec(h, 1.0);
        let ids: Vec<u32> = (0..rows)
            .map(|_| (rng.next() % vocab as u64) as u32)
            .collect();
        let (td, nd, idd) = (dev.up(&bf16_bytes(&table))?, dev.up(&nw)?, dev.up(&ids)?);
        let mut out = dev.zeros::<f32>(rows * h)?;
        dev.launched("memra_ink2g_embed", unsafe {
            ffi::memra_ink2g_embed(
                dptr(&td, &st) as *const c_void,
                dptr(&idd, &st) as *const u32,
                dptr(&nd, &st) as *const f32,
                1e-6,
                dptr_mut(&mut out, &st) as *mut f32,
                rows as i32,
                h as i32,
                dev.sp(),
            )
        })?;
        let got = dev.down(&out, 0, rows * h)?;
        let mut want = Vec::new();
        for id in &ids {
            let row = wide(&table[*id as usize * h..(*id as usize + 1) * h]);
            want.extend(ops::rms_norm(&row, Some(&nw), 1e-6));
        }
        let e1 = worst_rel(&got, &want, 1e-30);
        let x = rng.vec(rows * h, 3.0);
        let xd = dev.up(&x)?;
        dev.launched("memra_ink2g_norm", unsafe {
            ffi::memra_ink2g_norm(
                dptr(&xd, &st) as *const f32,
                dptr(&nd, &st) as *const f32,
                1e-6,
                dptr_mut(&mut out, &st) as *mut f32,
                rows as i32,
                h as i32,
                dev.sp(),
            )
        })?;
        let got = dev.down(&out, 0, rows * h)?;
        let want: Vec<f64> = x
            .chunks_exact(h)
            .flat_map(|row| ops::rms_norm(&wide(row), Some(&nw), 1e-6))
            .collect();
        let e2 = worst_rel(&got, &want, 1e-30);
        verdict(
            e1 <= ROUND && e2 <= ROUND,
            format!(
                "ink2g-embed-norm [rows={rows} hidden={h}] embed max rel={e1:.3e}, norm max rel={e2:.3e} (one f32 rounding {ROUND:.1e})"
            ),
        );
    }

    // ---- depth mix
    {
        let (h, rows, ns, nq) = (2304usize, 7usize, 4usize, 3usize);
        let states: Vec<Vec<f32>> = (0..ns).map(|s| rng.vec(rows * h, 1.0 + s as f32)).collect();
        let queries = rng.vec(nq * h, 2.0);
        let sd: Vec<CudaSlice<f32>> = states.iter().map(|s| dev.up(s)).collect::<Result<_, _>>()?;
        let qd = dev.up(&queries)?;
        let mut out = dev.zeros::<f32>(nq * rows * h)?;
        let ptrs: Vec<u64> = sd.iter().map(|s| dptr(s, &st)).collect();
        Route::mix(dev, &ptrs, &qd, nq, 1e-6, &mut out, rows * h, rows, h)?;
        let got = dev.down(&out, 0, nq * rows * h)?;
        let mut worst = 0f64;
        for r in 0..rows {
            let wides: Vec<Vec<f64>> = states
                .iter()
                .map(|s| wide(&s[r * h..(r + 1) * h]))
                .collect();
            let refs: Vec<&[f64]> = wides.iter().map(Vec::as_slice).collect();
            for (q, want) in ops::depth_mix(&queries, &refs, 1e-6).iter().enumerate() {
                let at = (q * rows + r) * h;
                worst = worst.max(worst_rel(&got[at..at + h], want, 1e-9));
            }
        }
        verdict(
            worst <= 4.0 * ROUND,
            format!("ink2g-mix [rows={rows} states={ns} queries={nq}] max rel={worst:.3e}"),
        );
    }

    // ---- loop controller
    {
        let (h, rows, rank) = (2304usize, 5usize, 64usize);
        let (x, msg) = (rng.vec(rows * h, 2.0), rng.vec(rows * h, 0.7));
        let (down, forget, gain) = (
            rng.bf16(rank * 2 * h, 0.02),
            rng.bf16(h * rank, 0.1),
            rng.bf16(h * rank, 0.1),
        );
        let (xd, md) = (dev.up(&x)?, dev.up(&msg)?);
        let (dd, fd, gd) = (
            dev.up(&bf16_bytes(&down))?,
            dev.up(&bf16_bytes(&forget))?,
            dev.up(&bf16_bytes(&gain))?,
        );
        let mut out = dev.zeros::<f32>(rows * h)?;
        dev.launched("memra_ink2g_controller", unsafe {
            ffi::memra_ink2g_controller(
                dptr(&xd, &st) as *const f32,
                dptr(&md, &st) as *const f32,
                dptr(&dd, &st) as *const c_void,
                dptr(&fd, &st) as *const c_void,
                dptr(&gd, &st) as *const c_void,
                1e-6,
                dptr_mut(&mut out, &st) as *mut f32,
                rows as i32,
                h as i32,
                rank as i32,
                dev.sp(),
            )
        })?;
        let got = dev.down(&out, 0, rows * h)?;
        let want: Vec<f64> = (0..rows)
            .flat_map(|r| {
                ops::controller(
                    &wide(&x[r * h..(r + 1) * h]),
                    &wide(&msg[r * h..(r + 1) * h]),
                    &down,
                    &forget,
                    &gain,
                    1e-6,
                )
            })
            .collect();
        let worst = worst_rel(&got, &want, 1e-6);
        verdict(
            worst <= 4.0 * ROUND,
            format!("ink2g-controller [rows={rows} rank={rank}] max rel={worst:.3e}"),
        );
    }

    // ---- clip / tau / value norm, and the ring destination
    {
        let (rows, heads, width, cap, pos0, floor) =
            (6usize, 24usize, 128usize, 8usize, 21u64, 24u32);
        let x = rng.vec(rows * heads * width, 3.0);
        let scale = rng.vec(heads * width, 1.0);
        let (xd, sd) = (dev.up(&x)?, dev.up(&scale)?);
        let mut out = dev.zeros::<f32>(rows * heads * width)?;
        let mut ring = dev.zeros::<f32>(cap * heads * width)?;
        let launch = |mode: i32,
                      scaled: bool,
                      thr: f64,
                      log_floor: i32,
                      dst: u64,
                      cap: usize|
         -> Result<(), String> {
            dev.launched("memra_ink2g_heads", unsafe {
                ffi::memra_ink2g_heads(
                    mode,
                    dptr(&xd, &st) as *const f32,
                    if scaled {
                        dptr(&sd, &st) as *const f32
                    } else {
                        std::ptr::null()
                    },
                    thr,
                    1e-6,
                    pos0 as i64,
                    log_floor,
                    0.1,
                    dst as *mut f32,
                    cap as i64,
                    rows as i32,
                    heads as i32,
                    width as i32,
                    dev.sp(),
                )
            })
        };
        let outp = dptr_mut(&mut out, &st);
        launch(0, true, 22.627, floor as i32, outp, 0)?;
        let got = dev.down(&out, 0, rows * heads * width)?;
        let mut want = Vec::new();
        for r in 0..rows {
            let mut row = wide(&x[r * heads * width..(r + 1) * heads * width]);
            let tau = ops::log_position_scale(pos0 + r as u64, floor, 0.1);
            ops::clip_heads(&mut row, width, 22.627, 1e-6, Some(&scale), tau);
            want.extend(row);
        }
        let e_clip = worst_rel(&got, &want, 1e-30);
        // value norm into a ring of 8 rows: row r lands at (pos0 + r) % cap
        let ringp = dptr_mut(&mut ring, &st);
        launch(1, false, 0.0, 0, ringp, cap)?;
        let got = dev.down(&ring, 0, cap * heads * width)?;
        let mut e_norm = 0f64;
        for r in 0..rows {
            let at = ((pos0 as usize + r) % cap) * heads * width;
            let want: Vec<f64> = x[r * heads * width..(r + 1) * heads * width]
                .chunks_exact(width)
                .flat_map(|head| ops::rms_norm(&wide(head), None, 1e-6))
                .collect();
            e_norm = e_norm.max(worst_rel(&got[at..at + heads * width], &want, 1e-30));
        }
        verdict(
            e_clip <= 2.0 * ROUND && e_norm <= ROUND,
            format!(
                "ink2g-heads [rows={rows} heads={heads} width={width}] clip+scale+tau max rel={e_clip:.3e}, value norm into a ring max rel={e_norm:.3e}"
            ),
        );
    }

    // ---- short convolutions: carried state equals one call, the red arm does not
    for (name, dynamic) in [("ink2g-conv-static", false), ("ink2g-conv-dynamic", true)] {
        let (total, c, k, groups) = (11usize, 1024usize, 4usize, 8usize);
        let x = rng.vec(total * c, 1.0);
        let add = rng.vec(total * c, 1.0);
        let taps = rng.bf16(c * k, 0.3);
        let dyn_taps = rng.vec(total * groups * k, 0.3);
        let mut history = ops::ConvHistory::default();
        let mut want = Vec::new();
        for r in 0..total {
            let row = wide(&x[r * c..(r + 1) * c]);
            let out = if dynamic {
                ops::short_conv_dynamic(
                    &row,
                    &mut history,
                    &taps,
                    &wide(&dyn_taps[r * groups * k..(r + 1) * groups * k]),
                    k,
                    groups,
                )
            } else {
                ops::short_conv_static(&row, &mut history, &taps, k)
            };
            want.extend(
                out.iter()
                    .zip(&add[r * c..(r + 1) * c])
                    .map(|(o, a)| f64::from(*a) + o),
            );
        }
        let (xd, ad, td, dd) = (
            dev.up(&x)?,
            dev.up(&add)?,
            dev.up(&taps)?,
            dev.up(&dyn_taps)?,
        );
        let run = |splits: &[usize], carry: bool| -> Result<Vec<f32>, String> {
            // the committed history and the one a call writes, swapped after every call
            let mut states = [
                dev.zeros::<f32>((k - 1) * c)?,
                dev.zeros::<f32>((k - 1) * c)?,
            ];
            let mut cur = 0usize;
            let mut out = dev.zeros::<f32>(total * c)?;
            let mut at = 0usize;
            for rows in splits {
                dev.launched("memra_ink2g_conv", unsafe {
                    ffi::memra_ink2g_conv(
                        i32::from(dynamic),
                        (dptr(&xd, &st) + (at * c * 4) as u64) as *const f32,
                        dptr(&states[cur], &st) as *const f32,
                        if carry {
                            dptr_mut(&mut states[1 - cur], &st) as *mut f32
                        } else {
                            std::ptr::null_mut()
                        },
                        dptr(&td, &st) as *const f32,
                        (dptr(&dd, &st) + (at * groups * k * 4) as u64) as *const f32,
                        groups as i32,
                        (dptr(&ad, &st) + (at * c * 4) as u64) as *const f32,
                        (dptr_mut(&mut out, &st) + (at * c * 4) as u64) as *mut f32,
                        *rows as i32,
                        c as i32,
                        k as i32,
                        dev.sp(),
                    )
                })?;
                cur = 1 - cur;
                at += rows;
            }
            dev.down(&out, 0, total * c)
        };
        let whole = run(&[total], true)?;
        let split = run(&[5, 1, 1, 3, 1], true)?;
        let dropped = run(&[5, 1, 1, 3, 1], false)?;
        let e = worst_rel(&whole, &want, 1e-6);
        let carried = bits_differ(&whole, &split);
        let red = bits_differ(&whole, &dropped);
        verdict(
            e <= 4.0 * ROUND && carried == 0 && red > 0,
            format!(
                "{name} [rows={total} channels={c} kernel={k}] max rel={e:.3e}; calls of 5,1,1,3,1 vs one call bit-differ={carried}; red arm conv-state moved {red}/{}",
                total * c
            ),
        );
    }

    // ---- relative-bias attention with the per-query sink
    // The long cell crosses several INK2G_ATTN_TILE key tiles.
    for (name, window, extent, mask, cap, total, heads, kvh) in [
        (
            "ink2g-attn-sliding",
            8usize,
            8usize,
            true,
            16usize,
            40usize,
            24usize,
            8usize,
        ),
        ("ink2g-attn-full", 0, 12, false, 64, 40, 24, 8),
        ("ink2g-attn-full-masked", 0, 12, true, 64, 40, 24, 8),
        ("ink2g-attn-full-long", 0, 12, false, 2600, 2600, 2, 1),
    ] {
        let (dim, rank) = (128usize, 16usize);
        let kvw = kvh * dim;
        let q = rng.vec(total * heads * dim, 1.5);
        let rel = rng.vec(total * heads * rank, 1.0);
        let table = rng.bf16(rank * extent, 0.5);
        let sink = rng.vec(total * heads, 2.0);
        let keys = rng.vec(total * kvw, 1.5);
        let values = rng.vec(total * kvw, 1.0);
        let shape = ops::AttentionShape {
            query_heads: heads,
            kv_heads: kvh,
            head_dim: dim,
            window: (window > 0).then_some(window),
            extent,
            mask_beyond_extent: mask,
        };
        let positions: Vec<u64> = (0..total as u64).collect();
        let mut want = Vec::new();
        for r in 0..total {
            let bias = ops::relative_bias(
                &wide(&rel[r * heads * rank..(r + 1) * heads * rank]),
                &table,
                rank,
                extent,
            );
            want.extend(ops::attend(
                &shape,
                &wide(&q[r * heads * dim..(r + 1) * heads * dim]),
                &bias,
                &wide(&sink[r * heads..(r + 1) * heads]),
                r as u64,
                ops::CachedRows {
                    positions: &positions[..=r],
                    keys: &wide(&keys[..(r + 1) * kvw]),
                    values: &wide(&values[..(r + 1) * kvw]),
                },
            ));
        }
        let (qd, rd, td, sd) = (dev.up(&q)?, dev.up(&rel)?, dev.up(&table)?, dev.up(&sink)?);
        let run = |splits: &[usize]| -> Result<Vec<f32>, String> {
            let mut kc = dev.zeros::<f32>(cap * kvw)?;
            let mut vc = dev.zeros::<f32>(cap * kvw)?;
            let mut out = dev.zeros::<f32>(total * heads * dim)?;
            let mut at = 0usize;
            for rows in splits {
                // the rows of the call enter the cache before any row attends
                for r in at..at + rows {
                    let slot = r % cap;
                    let mut kv = kc
                        .try_slice_mut(slot * kvw..(slot + 1) * kvw)
                        .ok_or("ring row")?;
                    dev.st
                        .memcpy_htod(&keys[r * kvw..(r + 1) * kvw], &mut kv)
                        .map_err(|e| format!("{e:?}"))?;
                    let mut vv = vc
                        .try_slice_mut(slot * kvw..(slot + 1) * kvw)
                        .ok_or("ring row")?;
                    dev.st
                        .memcpy_htod(&values[r * kvw..(r + 1) * kvw], &mut vv)
                        .map_err(|e| format!("{e:?}"))?;
                }
                dev.launched("memra_ink2g_attn", unsafe {
                    ffi::memra_ink2g_attn(
                        (dptr(&qd, &st) + (at * heads * dim * 4) as u64) as *const f32,
                        (dptr(&rd, &st) + (at * heads * rank * 4) as u64) as *const f32,
                        dptr(&td, &st) as *const f32,
                        (dptr(&sd, &st) + (at * heads * 4) as u64) as *const f32,
                        dptr(&kc, &st) as *const f32,
                        dptr(&vc, &st) as *const f32,
                        cap as i64,
                        at as i64,
                        window as i32,
                        extent as i32,
                        i32::from(mask),
                        (dptr_mut(&mut out, &st) + (at * heads * dim * 4) as u64) as *mut f32,
                        *rows as i32,
                        heads as i32,
                        kvh as i32,
                        dim as i32,
                        rank as i32,
                        dev.sp(),
                    )
                })?;
                dev.sync()?;
                at += rows;
            }
            dev.down(&out, 0, total * heads * dim)
        };
        let chunked = run(&vec![8; total / 8])?;
        let ones = run(&vec![1; total])?;
        // against the scale of a value row: an output is a convex mix of values of order 1
        let e = worst_rel(&chunked, &want, 1.0);
        let alone = bits_differ(&chunked, &ones);
        verdict(
            e <= ROUND && alone == 0,
            format!(
                "{name} [positions={total} window={window} extent={extent} mask_beyond={mask} cache rows={cap}] max |err| / (|ref| + 1)={e:.3e}; calls of 8 vs calls of 1 bit-differ={alone}"
            ),
        );
    }

    // ---- router: rank order, ties to the lower expert, the ties-high red arm
    {
        let (rows, experts, top_k) = (9usize, 128usize, 8usize);
        let mut logits = rng.vec(rows * experts, 2.0);
        // rows 0..3 hold exact ties across the selection edge
        for r in 0..3 {
            for i in 0..experts {
                logits[r * experts + i] = [0.5f32, -0.25, 0.5, 1.0][(i + r) % 4];
            }
        }
        // a zero selection bias: the constructed ties stay ties
        let bias = vec![0f32; experts];
        let (ld, bd) = (dev.up(&logits)?, dev.up(&bias)?);
        let mut ids = dev.zeros::<u32>(rows * top_k)?;
        let mut wts = dev.zeros::<f32>(rows * top_k)?;
        let mut margin = dev.zeros::<f32>(rows)?;
        let mut run = |tie_high: i32| -> Result<(Vec<u32>, Vec<f32>), String> {
            dev.launched("memra_ink2g_route", unsafe {
                ffi::memra_ink2g_route(
                    dptr(&ld, &st) as *const f32,
                    dptr(&bd, &st) as *const f32,
                    8.0,
                    top_k as i32,
                    tie_high,
                    dptr_mut(&mut ids, &st) as *mut u32,
                    dptr_mut(&mut wts, &st) as *mut f32,
                    dptr_mut(&mut margin, &st) as *mut f32,
                    rows as i32,
                    experts as i32,
                    dev.sp(),
                )
            })?;
            Ok((
                dev.down(&ids, 0, rows * top_k)?,
                dev.down(&wts, 0, rows * top_k)?,
            ))
        };
        let (got_ids, got_wts) = run(0)?;
        let (red_ids, _) = run(1)?;
        let (mut id_bad, mut w_err) = (0usize, 0f64);
        for r in 0..rows {
            let (sel, weights) = ops::route(
                &wide(&logits[r * experts..(r + 1) * experts]),
                &{
                    // the router's weight is the identity here: logits are given directly
                    let mut eye = vec![0f32; experts * experts];
                    for i in 0..experts {
                        eye[i * experts + i] = 1.0;
                    }
                    eye
                },
                Some(&bias),
                top_k,
                8.0,
            );
            for j in 0..top_k {
                if got_ids[r * top_k + j] as usize != sel[j] {
                    id_bad += 1;
                }
            }
            w_err = w_err.max(worst_rel(
                &got_wts[r * top_k..(r + 1) * top_k],
                &weights,
                1e-30,
            ));
        }
        let red_rows = (0..rows)
            .filter(|r| got_ids[r * top_k..(r + 1) * top_k] != red_ids[r * top_k..(r + 1) * top_k])
            .count();
        let untied_same = (3..rows)
            .all(|r| got_ids[r * top_k..(r + 1) * top_k] == red_ids[r * top_k..(r + 1) * top_k]);
        verdict(
            id_bad == 0 && w_err <= ROUND && red_rows == 3 && untied_same,
            format!(
                "ink2g-route [rows={rows} experts={experts} top_k={top_k}] picks in rank order differ={id_bad}, weights max rel={w_err:.3e}; red arm ties-high moved {red_rows}/3 tied rows and no untied row"
            ),
        );
    }

    // ---- softcapped SwiGLU, the routed sum, softcap and argmax
    {
        let (pairs, f, h, top_k) = (16usize, 1152usize, 2304usize, 8usize);
        let gu = rng.vec(pairs * 2 * f, 6.0);
        let wts = rng.vec(pairs, 1.0);
        let (gd, wd) = (dev.up(&gu)?, dev.up(&wts)?);
        let mut act = dev.zeros::<f32>(pairs * f)?;
        dev.launched("memra_ink2g_swiglu", unsafe {
            ffi::memra_ink2g_swiglu(
                dptr(&gd, &st) as *const f32,
                dptr(&wd, &st) as *const f32,
                4.0,
                25.0,
                dptr_mut(&mut act, &st) as *mut f32,
                pairs as i32,
                f as i32,
                dev.sp(),
            )
        })?;
        let got = dev.down(&act, 0, pairs * f)?;
        let want: Vec<f64> = (0..pairs * f)
            .map(|i| {
                ops::softcapped_swiglu(f64::from(gu[2 * i]), f64::from(gu[2 * i + 1]), 4.0, 25.0)
                    * f64::from(wts[i / f])
            })
            .collect();
        let e_act = worst_rel(&got, &want, 1e-30);
        let dn = rng.vec(pairs * h, 1.0);
        let dd = dev.up(&dn)?;
        let rows = pairs / top_k;
        let mut out = dev.zeros::<f32>(rows * h)?;
        dev.launched("memra_ink2g_sum", unsafe {
            ffi::memra_ink2g_sum(
                dptr(&dd, &st) as *const f32,
                std::ptr::null(),
                dptr_mut(&mut out, &st) as *mut f32,
                rows as i32,
                top_k as i32,
                h as i32,
                dev.sp(),
            )
        })?;
        let got = dev.down(&out, 0, rows * h)?;
        let want: Vec<f64> = (0..rows * h)
            .map(|i| {
                (0..top_k)
                    .map(|s| f64::from(dn[((i / h) * top_k + s) * h + i % h]))
                    .sum()
            })
            .collect();
        let e_sum = worst_rel(&got, &want, 1e-6);
        // softcap, then the first of equal maxima
        let n = 5000usize;
        let mut logits = rng.vec(2 * n, 20.0);
        logits[777] = 500.0;
        logits[4000] = 500.0;
        logits[n + 3] = 400.0;
        let mut ld = dev.up(&logits)?;
        let mut best = dev.zeros::<u32>(2)?;
        dev.launched("memra_ink2g_head", unsafe {
            ffi::memra_ink2g_head(
                dptr_mut(&mut ld, &st) as *mut f32,
                30.0,
                dptr_mut(&mut best, &st) as *mut u32,
                2,
                n as i32,
                dev.sp(),
            )
        })?;
        let capped = dev.down(&ld, 0, 2 * n)?;
        let want: Vec<f64> = logits
            .iter()
            .map(|v| ops::softcap(f64::from(*v), 30.0))
            .collect();
        let e_cap = worst_rel(&capped, &want, 1e-30);
        let best = dev.down(&best, 0, 2)?;
        verdict(
            e_act <= 2.0 * ROUND && e_sum <= 2.0 * ROUND && e_cap <= ROUND && best == [777, 3],
            format!(
                "ink2g-swiglu-sum-head SwiGLU max rel={e_act:.3e}, routed sum max rel={e_sum:.3e}, softcap max rel={e_cap:.3e}, argmax {best:?} (first of equal maxima: [777, 3])"
            ),
        );
    }

    let mut skipped: Vec<String> = Vec::new();
    // ---- the declared expert program end to end (declared_experts) against the reference block
    // (memra_reference::inkling2_actquant::expert_block_*), as one call of several rows (host
    // groups) and as one call per row (device groups): the two must be bit-identical, since each
    // routed pair owns its quantized row.
    if unsafe { ffi::memra_ink2_actq_built() } == 1 {
        use memra_reference::inkling2_actquant as aq;
        for (name, fmt) in [
            ("ink2g-moe-declared-nvfp4", FMT_NVFP4),
            ("ink2g-moe-declared-mxfp8", FMT_MXFP8),
        ] {
            let (experts, hidden, width, rows, top_k) =
                (6usize, 256usize, 128usize, 9usize, 4usize);
            let pairs = rows * top_k;
            let caps = (4.0f32, 25.0f32);
            // (data, scales, amax) of a bank [experts, n, k] in the stored layout
            let mut make = |n: usize, k: usize| -> (Vec<u8>, Vec<u8>, Vec<f32>) {
                if fmt == FMT_NVFP4 {
                    let codes = rng.bytes(experts * n * k / 2);
                    let mut scales = rng.bytes(experts * n * k / 16);
                    scales.iter_mut().for_each(|s| *s = 0x20 + (*s % 0x50));
                    let amax = (0..experts).map(|_| 0.5 + rng.unit()).collect();
                    (codes, scales, amax)
                } else {
                    let mut values = rng.bytes(experts * n * k);
                    values.iter_mut().for_each(|v| {
                        if *v & 0x7f == 0x7f {
                            *v &= 0x80;
                        }
                    });
                    let mut scales = rng.bytes(experts * n * k / 32);
                    scales.iter_mut().for_each(|s| *s = 116 + (*s % 6));
                    (values, scales, vec![1.0f32])
                }
            };
            let (d13, s13, a13) = make(2 * width, hidden);
            let (d2, s2, a2) = make(hidden, width);
            let denom = if fmt == FMT_NVFP4 { 2688.0f32 } else { 1.0 };
            let bank = |data: &[u8], scales: &[u8], amax: &[f32], n: usize, k: usize| {
                Ok::<Bank, String>(Bank {
                    fmt,
                    data: dev.up(data)?,
                    scales: dev.up(scales)?,
                    amax: dev.up(amax)?,
                    denom,
                    n,
                    k,
                })
            };
            let w13 = bank(&d13, &s13, &a13, 2 * width, hidden)?;
            let w2 = bank(&d2, &s2, &a2, hidden, width)?;
            let x: Vec<f32> = rng.vec(rows * hidden, 1.0);
            // top_k distinct experts per row, in rank order
            let mut ids = Vec::with_capacity(pairs);
            for _ in 0..rows {
                let mut picked: Vec<u32> = Vec::new();
                while picked.len() < top_k {
                    let e = (rng.next() % experts as u64) as u32;
                    if !picked.contains(&e) {
                        picked.push(e);
                    }
                }
                ids.extend(picked);
            }
            let wts: Vec<f32> = (0..pairs).map(|_| 0.25 + rng.unit()).collect();
            let mut dq =
                DeclaredWork::alloc(dev, pairs, hidden, width)?.with_red(dev, pairs, top_k)?;
            let (xd, idd, wd) = (dev.up(&x)?, dev.up(&ids)?, dev.up(&wts)?);
            let mut out = dev.zeros::<f32>(rows * hidden)?;
            declared_experts(
                dev,
                &mut dq,
                &xd,
                &idd,
                &wd,
                Some(&ids),
                &w13,
                &w2,
                (rows, top_k, hidden, width),
                caps,
                DeclaredRed::default(),
                &mut out,
            )?;
            let got = dev.down(&out, 0, rows * hidden)?;
            // the red arms, each from fresh buffers
            let mut red_moved = Vec::new();
            for red in [
                DeclaredRed {
                    weight_after_down: true,
                    ..DeclaredRed::default()
                },
                DeclaredRed {
                    token_rows: true,
                    ..DeclaredRed::default()
                },
            ] {
                let mut rq =
                    DeclaredWork::alloc(dev, pairs, hidden, width)?.with_red(dev, pairs, top_k)?;
                let mut ro = dev.zeros::<f32>(rows * hidden)?;
                declared_experts(
                    dev,
                    &mut rq,
                    &xd,
                    &idd,
                    &wd,
                    Some(&ids),
                    &w13,
                    &w2,
                    (rows, top_k, hidden, width),
                    caps,
                    red,
                    &mut ro,
                )?;
                red_moved.push(bits_differ(&dev.down(&ro, 0, rows * hidden)?, &got));
            }
            let mut alone = 0usize;
            for r in 0..rows {
                let (xr, ir, wr) = (
                    dev.up(&x[r * hidden..(r + 1) * hidden])?,
                    dev.up(&ids[r * top_k..(r + 1) * top_k])?,
                    dev.up(&wts[r * top_k..(r + 1) * top_k])?,
                );
                let mut one = dev.zeros::<f32>(hidden)?;
                declared_experts(
                    dev,
                    &mut dq,
                    &xr,
                    &ir,
                    &wr,
                    None,
                    &w13,
                    &w2,
                    (1, top_k, hidden, width),
                    caps,
                    DeclaredRed::default(),
                    &mut one,
                )?;
                alone += bits_differ(
                    &dev.down(&one, 0, hidden)?,
                    &got[r * hidden..(r + 1) * hidden],
                );
            }
            let (mut worst, mut equal) = (0f64, 0usize);
            for r in 0..rows {
                let row: Vec<f32> = x[r * hidden..(r + 1) * hidden]
                    .iter()
                    .map(|v| aq::bf16(*v))
                    .collect();
                let sel: Vec<usize> = ids[r * top_k..(r + 1) * top_k]
                    .iter()
                    .map(|e| *e as usize)
                    .collect();
                let gamma = &wts[r * top_k..(r + 1) * top_k];
                let want = if fmt == FMT_NVFP4 {
                    let up = aq::Nvfp4Experts {
                        codes: &d13,
                        scales: &s13,
                        amax: &a13,
                        denominator: denom,
                        n: 2 * width,
                        k: hidden,
                    };
                    let down = aq::Nvfp4Experts {
                        codes: &d2,
                        scales: &s2,
                        amax: &a2,
                        denominator: denom,
                        n: hidden,
                        k: width,
                    };
                    aq::expert_block_w4a4(&row, &sel, gamma, &up, &down, caps).out
                } else {
                    let up = aq::Mxfp8Experts {
                        values: &d13,
                        exponents: &s13,
                        n: 2 * width,
                        k: hidden,
                    };
                    let down = aq::Mxfp8Experts {
                        values: &d2,
                        exponents: &s2,
                        n: hidden,
                        k: width,
                    };
                    aq::expert_block_w8a8(&row, &sel, gamma, &up, &down, caps).out
                };
                let g = &got[r * hidden..(r + 1) * hidden];
                let (mut num, mut den) = (0f64, 0f64);
                for (a, b) in g.iter().zip(&want) {
                    num += (f64::from(*a) - f64::from(*b)).powi(2);
                    den += f64::from(*b).powi(2);
                    equal += usize::from(a.to_bits() == b.to_bits());
                }
                worst = worst.max((num / den.max(1e-300)).sqrt());
            }
            // Bit for bit: every bf16 output equals the reference block's. Each red arm must
            // move outputs.
            let total = rows * hidden;
            verdict(
                equal == total && alone == 0 && red_moved.iter().all(|m| *m > 0),
                format!(
                    "{name} [rows={rows} top_k={top_k} hidden={hidden} width={width}] vs the reference block: bf16 outputs equal {equal}/{total} (worst row rel L2 {worst:.3e}); one call per row vs one call bit-differ={alone}; red arms moved: weight-after-down {}/{total}, token-rows {}/{total}",
                    red_moved[0], red_moved[1]
                ),
            );
        }
    } else {
        skipped.push(
            "ink2g-moe-declared-nvfp4 / -mxfp8 SKIP: this build has no block-scaled act-quant target"
                .to_string(),
        );
    }
    // ---- load-time refusals of the expert scale planes
    {
        let refused = [
            check_ue8m0("mxfp8", &[127, 0xFF, 120]).is_err(),
            check_nvfp4_globals("nvfp4", &[1.0, f32::NAN], 2688.0).is_err(),
            check_nvfp4_globals("nvfp4", &[1.0, f32::INFINITY], 2688.0).is_err(),
            check_nvfp4_globals("nvfp4", &[1.0, 0.0], 2688.0).is_err(),
            check_nvfp4_globals("nvfp4", &[1.0, -2.0], 2688.0).is_err(),
            check_nvfp4_globals("nvfp4", &[1.0, 2.0], 0.0).is_err(),
        ];
        let accepted = check_ue8m0("mxfp8", &[0, 127, 254]).is_ok()
            && check_nvfp4_globals("nvfp4", &[0.5, 3.0], 2688.0).is_ok();
        verdict(
            refused.iter().all(|r| *r) && accepted,
            format!(
                "ink2g-load-refusals E8M0 0xFF, NVFP4 amax NaN / inf / 0 / negative, denominator 0: refused {}/{}; valid planes accepted {accepted}",
                refused.iter().filter(|r| **r).count(),
                refused.len()
            ),
        );
    }
    dev.sync()?;
    lines.extend(skipped);
    Ok((lines, fails))
}
