//! Storage classes of the Gated DeltaNet recurrent state (research/gdn-state-fp16-20261107).
//!
//! A GDN layer's state is always allocated fp32 (`RecurLayer::ssm_state` and its ping-pong
//! twin). The prime, every capture (prefix entries, checkpoints, `CacheSnapshot`) and every
//! restore run the fp32 class, so the prime is the fp32 program bit for bit whatever the model's
//! decode class is. A model whose decode class is fp16 (`HybridModel::gdn_decode_fp16`) rounds
//! a layer's state to fp16 once, when its session starts decoding (`Engine::gdn_ensure_decode`),
//! and from then on its decode and verify scans load and store fp16 in the first half of the
//! state buffer, rounding after every token: half the state bytes per step, and a T-row verify
//! still equals T decode steps bit for bit. A prime that resumes on a decoded session widens the
//! state back to fp32 first (`gdn_ensure_prime`, exact).
//!
//! The two exactness laws hold by construction:
//!
//! - decode == verify: both run the fp16 sequential scan with per-token rounding from the same
//!   rounded state;
//! - restore == unrestored: the prime and every capture are fp32, a restored plane is fp32, and
//!   the decode-start rounding is a pure function of the fp32 state, applied on both paths. The
//!   conversions are exact on a state that is already fp16-representable, so a decoded state may
//!   leave and re-enter the session (a snapshot, a rollback, a multi-turn prime) any number of
//!   times without a value changing.

use crate::Engine;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use memra_gguf::model_plan::{
    AttentionPlan, GdnGateActivation, MlpPlan, ModelPlan, ResidualTopology, StatePlan,
};
use memra_kv::RecurLayer;

pub use memra_kv::gdn_state_words;

/// The value width every GDN state kernel serves (`gdn_scan_s128*`): d_state = 128.
const GDN_D: usize = 128;

impl Engine {
    /// Refuse a GDN state buffer too short for one d_state=128 state of `n_head` value heads in
    /// the class `st16` names (the resident buffers are fp32-sized, so an fp16 state uses the
    /// first half of them).
    pub(crate) fn gdn_state_len_check(
        &self,
        what: &str,
        len: usize,
        n_head: usize,
        st16: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let want = gdn_state_words(GDN_D * GDN_D * n_head, st16);
        if len < want {
            return Err(format!(
                "{what}: GDN state buffer of {len} f32 words; the {} class needs {want} for \
                 {n_head} heads",
                if st16 { "fp16" } else { "fp32" }
            )
            .into());
        }
        Ok(())
    }

    /// Refuse a verify snapshot ring shorter than its `t - 1` slots in the class `st16` names
    /// (slot r starts at element r*H*128*128, so at word r * gdn_state_words(..)).
    pub(crate) fn gdn_snap_len_check(
        &self,
        what: &str,
        len: usize,
        n_head: usize,
        t: usize,
        st16: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let want = t.saturating_sub(1) * gdn_state_words(GDN_D * GDN_D * n_head, st16);
        if len < want {
            return Err(format!(
                "{what}: snapshot ring of {len} f32 words; {} slots of the {} class need {want}",
                t.saturating_sub(1),
                if st16 { "fp16" } else { "fp32" }
            )
            .into());
        }
        Ok(())
    }

    fn gdn_state_convert(
        &self,
        kernel: &str,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n_elems: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if src.len() < n_elems || dst.len() < n_elems {
            return Err(format!(
                "{kernel}: buffers of {} / {} f32 words, the conversion needs {n_elems} each",
                src.len(),
                dst.len()
            )
            .into());
        }
        let f = self.func(kernel);
        let n = n_elems as u64;
        let cfg = LaunchConfig::for_num_elems(u32::try_from(n_elems)?);
        let s = self.gpu.stream();
        let mut b = s.launch_builder(&f);
        b.arg(src).arg(dst).arg(&n);
        unsafe {
            b.launch(cfg)?;
        }
        Ok(())
    }

    /// `dst` = the fp32 state `src` of `n_elems` elements rounded to fp16 (nearest even), in the
    /// fp16 layout (element i at byte 2*i, the first n/2 words). Distinct buffers.
    pub fn gdn_state_to_fp16_into(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n_elems: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.gdn_state_convert("gdn_state_to_fp16", src, dst, n_elems)
    }

    /// `dst` = the fp16 state packed in the first half of `src`, widened to fp32 (exact).
    pub fn gdn_state_to_f32_into(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n_elems: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.gdn_state_convert("gdn_state_to_f32", src, dst, n_elems)
    }

    /// A class change launches kernels, so it must never land inside a CUDA graph capture (a
    /// captured conversion would re-run on every replay). Graph paths convert eagerly first.
    fn gdn_refuse_capturing(&self, what: &str) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::sys::CUstreamCaptureStatus as S;
        if self.gpu.stream().capture_status()? != S::CU_STREAM_CAPTURE_STATUS_NONE {
            return Err(format!(
                "{what}: GDN state class change inside a CUDA graph capture; the caller must \
                 bring the cache to its class before capturing"
            )
            .into());
        }
        Ok(())
    }

    /// Decode-phase entry of one GDN layer: an fp16 decode class rounds the fp32 state to fp16
    /// once (the decode-start rounding) and every later call is a no-op. Returns the `st16` the
    /// layer's decode and verify scans run. The state's buffer pointers do not change (the
    /// rounded state is copied back), so captured graphs and pointer tables stay valid.
    pub fn gdn_ensure_decode(
        &self,
        rl: &mut RecurLayer,
        fp16_class: bool,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if !fp16_class {
            if rl.ssm_fp16 {
                return Err("GDN layer holds an fp16 state but its model decodes fp32".into());
            }
            return Ok(false);
        }
        if rl.ssm_fp16 {
            return Ok(true);
        }
        self.gdn_refuse_capturing("gdn_ensure_decode")?;
        let n = rl.ssm_state.len();
        let RecurLayer {
            ssm_state,
            ssm_state_alt,
            ..
        } = rl;
        self.gdn_state_to_fp16_into(ssm_state, ssm_state_alt, n)?;
        self.copy_into(ssm_state, 0, ssm_state_alt, gdn_state_words(n, true))?;
        rl.ssm_fp16 = true;
        Ok(true)
    }

    /// Prime-phase entry of one GDN layer: a layer that has been decoding in fp16 widens back to
    /// fp32 (exact), the class every prime kernel runs. A no-op on an fp32 layer.
    pub fn gdn_ensure_prime(&self, rl: &mut RecurLayer) -> Result<(), Box<dyn std::error::Error>> {
        if !rl.ssm_fp16 {
            return Ok(());
        }
        self.gdn_refuse_capturing("gdn_ensure_prime")?;
        let n = rl.ssm_state.len();
        let RecurLayer {
            ssm_state,
            ssm_state_alt,
            ..
        } = rl;
        self.gdn_state_to_f32_into(ssm_state, ssm_state_alt, n)?;
        self.copy_into(ssm_state, 0, ssm_state_alt, n)?;
        rl.ssm_fp16 = false;
        Ok(())
    }

    /// `gdn_ensure_decode` over every recurrent layer of a cache (graph and pointer-table paths
    /// call it eagerly before capture or replay). Returns the class of the layers.
    pub fn gdn_ensure_decode_cache(
        &self,
        cache: &mut crate::cache::Cache,
        fp16_class: bool,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        for rl in cache.recur.iter_mut().flatten() {
            self.gdn_ensure_decode(rl, fp16_class)?;
        }
        Ok(fp16_class)
    }

    /// `gdn_ensure_prime` over every recurrent layer of a cache.
    pub fn gdn_ensure_prime_cache(
        &self,
        cache: &mut crate::cache::Cache,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for rl in cache.recur.iter_mut().flatten() {
            self.gdn_ensure_prime(rl)?;
        }
        Ok(())
    }
}

/// fp32 to IEEE binary16 bits, round to nearest even, as the device's `__float2half_rn`: values
/// at or above 65520 in magnitude become infinity, the range below 2^-14 is subnormal (multiples
/// of 2^-24), magnitudes at or below 2^-25 become zero, and every NaN becomes 0x7fff.
pub fn f32_to_fp16_bits(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let u = b & 0x7fff_ffff;
    if u >= 0x7f80_0000 {
        return if u == 0x7f80_0000 {
            sign | 0x7c00
        } else {
            0x7fff
        };
    }
    let (r, rem) = if u > 0x477f_efff {
        (sign | 0x7bff, 0x8000_0000u32)
    } else if u >= 0x3880_0000 {
        (sign | ((u - 0x3800_0000) >> 13), u << 19)
    } else if u < 0x3300_0001 {
        (sign, u)
    } else {
        let shift = 0x7e - (u >> 23);
        let m = (u & 0x7f_ffff) | 0x80_0000;
        (sign | (m >> shift), m << (32 - shift))
    };
    if rem > 0x8000_0000 || (rem == 0x8000_0000 && r & 1 != 0) {
        r + 1
    } else {
        r
    }
}

/// IEEE binary16 bits widened to fp32 (exact).
pub fn fp16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let v = match e {
        0 => m as f32 / 16_777_216.0,
        31 if m == 0 => f32::INFINITY,
        31 => return f32::NAN,
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    f32::from_bits(v.to_bits() | sign)
}

/// Round fp32 values to fp16 (`f32_to_fp16_bits`) and pack two per f32 word, element i at byte
/// 2*i: the GDN fp16 state layout on the host.
pub fn gdn_state_pack_fp16(vals: &[f32]) -> Vec<f32> {
    vals.chunks(2)
        .map(|p| {
            let lo = f32_to_fp16_bits(p[0]);
            let hi = p.get(1).map_or(0, |&x| f32_to_fp16_bits(x));
            f32::from_bits(lo | (hi << 16))
        })
        .collect()
}

/// Inverse of `gdn_state_pack_fp16`: every packed fp16 widened to fp32 (exact).
pub fn gdn_state_unpack_fp16(words: &[f32]) -> Vec<f32> {
    words
        .iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [fp16_bits_to_f32(b & 0xffff), fp16_bits_to_f32(b >> 16)]
        })
        .collect()
}

/// Whether a plan's recurrent layers are all the Qwen3.5-family GDN program the fp16 decode
/// class serves: every recurrent layer a Gated DeltaNet with d_state 128 and the SiLU z-gate,
/// the serial residual stream (no hyper-connection trunk), and at least one such layer.
pub fn plan_gdn_decode_eligible(plan: &ModelPlan) -> bool {
    let mut gdn = 0usize;
    for l in &plan.layers {
        if !matches!(l.residual, ResidualTopology::Serial) {
            return false;
        }
        match (&l.attention, &l.state) {
            (AttentionPlan::GatedDeltaNet(g), StatePlan::Recurrent { .. }) => {
                if g.key_head_dim as usize != GDN_D
                    || g.value_head_dim as usize != GDN_D
                    || g.gate_activation != GdnGateActivation::Silu
                {
                    return false;
                }
                gdn += 1;
            }
            (_, StatePlan::Recurrent { .. }) => return false,
            _ => {}
        }
    }
    gdn > 0
}

/// The shape the fp16 decode class was measured on (Qwen3.8-27B: a dense hybrid of 48 GDN layers
/// with 48 value heads of 128, research/gdn-state-fp16-20261107). A device default applies
/// only to this shape; any other eligible plan keeps fp32 unless `MEMRA_GDN_STATE_FP16=1` forces it.
pub fn plan_gdn_decode_measured(plan: &ModelPlan) -> bool {
    plan_gdn_decode_eligible(plan)
        && plan
            .layers
            .iter()
            .all(|l| matches!(l.mlp, MlpPlan::Dense(_)))
        && plan
            .layers
            .iter()
            .filter_map(|l| match &l.attention {
                AttentionPlan::GatedDeltaNet(g) => Some(g.value_heads),
                _ => None,
            })
            .fold((0usize, true), |(n, ok), heads| (n + 1, ok && heads == 48))
            == (48, true)
}

/// The decode-class decision for one model: `MEMRA_GDN_STATE_FP16=1|0` forces it (only on an
/// eligible plan; a force on any other plan is ignored with a warning by the caller), unset
/// follows the device default (fp16 on GB10 for the measured shape, fp32 everywhere else).
pub fn gdn_decode_fp16_on(env: Option<&str>, eligible: bool, measured: bool, gb10: bool) -> bool {
    if !eligible {
        return false;
    }
    match env {
        Some("1") => true,
        Some("0") => false,
        _ => gb10 && measured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_class_is_forced_or_follows_the_measured_gb10_default() {
        // unset: fp16 only on GB10 for the measured shape
        assert!(gdn_decode_fp16_on(None, true, true, true));
        assert!(!gdn_decode_fp16_on(None, true, true, false));
        assert!(!gdn_decode_fp16_on(None, true, false, true));
        // a force decides on any eligible plan, either way
        assert!(gdn_decode_fp16_on(Some("1"), true, false, false));
        assert!(!gdn_decode_fp16_on(Some("0"), true, true, true));
        // an ineligible plan is fp32 whatever the env says
        assert!(!gdn_decode_fp16_on(Some("1"), false, false, true));
        // anything but 0/1 is unset
        assert!(gdn_decode_fp16_on(Some("yes"), true, true, true));
    }

    #[test]
    fn host_pack_rounds_to_nearest_even_and_unpack_is_exact() {
        let p = |e: i32| 2f32.powi(e);
        let vals = [1.0f32, -2.5, 1.0 + p(-11), 1.0 + 3.0 * p(-11), 0.0];
        let packed = gdn_state_pack_fp16(&vals);
        assert_eq!(packed.len(), 3);
        let back = gdn_state_unpack_fp16(&packed);
        // exactly representable values survive; halfway cases round to even
        assert_eq!(back[0], 1.0);
        assert_eq!(back[1], -2.5);
        assert_eq!(back[2], 1.0);
        assert_eq!(back[3], 1.0 + p(-9));
        assert_eq!(back[4], 0.0);
        // a second rounding of a rounded state changes nothing
        assert_eq!(
            gdn_state_unpack_fp16(&gdn_state_pack_fp16(&back[..5])),
            back
        );
        // range: 65504 is the largest finite value and 65520 ties to infinity; subnormals are
        // the multiples of 2^-24 and 2^-25 ties to zero
        let edge = [
            65_519.0f32,
            65_520.0,
            -1e6,
            3.0 * p(-24),
            p(-25),
            1.5 * p(-25),
            p(-14) - p(-26),
            -p(-20),
        ];
        let back = gdn_state_unpack_fp16(&gdn_state_pack_fp16(&edge));
        assert_eq!(back[0], 65_504.0);
        assert_eq!(back[1], f32::INFINITY);
        assert_eq!(back[2], f32::NEG_INFINITY);
        assert_eq!(back[3], 3.0 * p(-24));
        assert_eq!(back[4], 0.0);
        assert_eq!(back[5], p(-24));
        assert_eq!(back[6], p(-14));
        assert_eq!(back[7], -p(-20));
        assert_eq!(f32_to_fp16_bits(f32::NAN), 0x7fff);
        // every finite fp16 value widens and rounds back to its own bits
        for h in (0u32..0x7c00).chain(0x8000..0xfc00) {
            assert_eq!(f32_to_fp16_bits(fp16_bits_to_f32(h)), h, "bits {h:#06x}");
        }
        assert_eq!(gdn_state_words(128 * 128 * 48, true), 128 * 128 * 24);
        assert_eq!(gdn_state_words(128 * 128 * 48, false), 128 * 128 * 48);
    }
}
