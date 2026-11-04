//! Per-model numeric profile (owner decision 2026-11-03, findings S15:F1 and S05:F0).
//!
//! A few kernel defaults are keyed per model: the rms_norm block, the FA vec crossover, the
//! windowed and hd512 split widths, the gemma split ladder seam, the fused q4_0 row mapping,
//! the v4 decode capacity guard, the spec-serving stream-k key and the step37 serving door
//! defaults. Each one changes the launch shape, and with it the FP order or the kernel
//! family, so it belongs to the model whose qualification measured it.
//!
//! Each loaded `HybridModel` owns its profile (`HybridModel::numeric`, built once at load
//! by [`NumericProfile::for_model`]). Every model entry point installs it on the calling
//! thread for the length of the call ([`NumericProfile::enter`]), and the launchers read the
//! innermost installed profile ([`NumericProfile::current`]). Outside every model scope they
//! read [`NumericProfile::BASE`], the program of a model that keys nothing (the qwen35,
//! qwen3 and llama class, every non-hybrid family, kernel-check, unit tests). Two models in
//! one process therefore each run their own program, in any load or call order.
//!
//! The scope is per thread. A worker thread a model spawns for its own launches enters the
//! spawner's profile first (`let profile = NumericProfile::current();` before the spawn,
//! `let _numeric = profile.enter();` inside the closure).
//!
//! Env overrides keep their meaning: `MEMRA_RMS_BLOCK`, `MEMRA_FA_VEC_MIN`,
//! `MEMRA_FA_V4_MAX`, `MEMRA_FA_SPW`, `MEMRA_FA_SP512`, `MEMRA_Q40_MR`, `MEMRA_MMQ_SK` and the
//! step37 door flags win over the profile for every model, as they did over the old statics.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::OnceLock;

/// One model's per-model numeric defaults. `Copy`, so a scope installs it by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericProfile {
    /// v4 decode threshold (`fa_v4_at`): v4 below it, v3 at and above. 0 turns the v4 decode
    /// family off: the gqa > 8 capacity guard (fa_v4_smem's per-warp Q arrays hold gqa <= 8,
    /// research/122b-bringup-20260806/). `fa_v4_at` also checks that capacity from each
    /// launch's own head counts (`crate::FA_V4_GQA_MAX`), so it holds outside a model scope too.
    pub fa_v4_max: usize,
    /// FA vec-lane t_kv floor (`fa_vec_min_tkv`).
    pub fa_vec_min: usize,
    /// Windowed-attention split width.
    pub fa_spw: usize,
    /// hd512 (gemma global layers) split width.
    pub fa_sp512: usize,
    /// Fused t=1 q4_0 pair/triple row mapping: true = mr1 (one row per warp).
    pub fused_mr1: bool,
    /// rms_norm block size (`rms_block`). The per-thread partial-sum split changes with
    /// blockDim, so the block is a numeric class, battery-arbitrated per model.
    pub rms_block: u32,
    /// gemma4 split ladder seam (`fa_split_keys` with `MEMRA_FA_SP16=1`).
    pub fa_sp_gemma: bool,
    /// Stream-k MMQ entry (-1 = env/default, 0 = force tiling, 1 = admit the deterministic
    /// form selector). Keyed under spec intent (`MEMRA_DRAFT` set, `MEMRA_MMQ_SK` unset).
    pub mmq_sk_force: i8,
    /// The step37 serving door defaults (`step37_door`): ON for the SlidingGatedMoe program.
    pub step37_serving: bool,
}

impl NumericProfile {
    /// What a model that keys nothing runs, and what every launch outside a model scope reads.
    pub const BASE: Self = Self {
        fa_v4_max: usize::MAX,
        fa_vec_min: crate::FA_VEC_MIN_TKV,
        fa_spw: 32,
        fa_sp512: 16,
        fused_mr1: false,
        rms_block: 256,
        fa_sp_gemma: false,
        mmq_sk_force: -1,
        step37_serving: false,
    };

    /// The profile one model's load keys. `gqa` is `n_head / n_head_kv` (0 when `n_head_kv`
    /// is 0), `gemma_moe` is `Some(real_moe)` for the gemma batch program, `glm5_next_b200` is
    /// a glm5_next model on an sm_100a build, `sliding_gated_moe` is the SlidingGatedMoe
    /// (step37) program, `spec_mmq_n_embd` is `Some(n_embd)` under spec intent. Pure: the
    /// loader passes what it already computed.
    pub fn for_model(
        gqa: u32,
        gemma_moe: Option<bool>,
        glm5_next_b200: bool,
        sliding_gated_moe: bool,
        spec_mmq_n_embd: Option<u32>,
    ) -> Self {
        let mut p = Self::BASE;
        // FA v4 GQA capacity guard: fa_v4_smem's per-warp Q arrays hold gqa <= 8 (see the
        // loader's comment and research/122b-bringup-20260806/).
        if gqa > 8 {
            p.fa_v4_max = 0;
        }
        if let Some(real_moe) = gemma_moe {
            // gemma4 fa-vec crossover: vec-always measured fastest, 119.9 (96) / 130.0 (48) /
            // 133.2 (1) tok/s tg128-regime (sweep 2026-07-10).
            p.fa_vec_min = 1;
            // windowed split: MoE 26B = 32 (grid-limited t=1 under the raw-e4m3 sV ceiling),
            // dense 31B = 64 (37.13/37.12 vs 36.87/36.86 at 1.7k, N=2; sweeps 2026-07-12).
            p.fa_spw = if real_moe { 32 } else { 64 };
            // hd512 global split: 26B = 16 (2026-07-11 N=2), 31B = 32 (36.86/36.93 vs
            // 36.73/36.73 at 1.7k, 2026-07-12).
            p.fa_sp512 = if real_moe { 16 } else { 32 };
            // fused t=1 pair/triple mr1: dense +1.1% short / +0.6% depth on the 31B; the MoE
            // 26B regresses -1.2% (its shared-expert fused2 shapes lose to the finer grid).
            p.fused_mr1 = !real_moe;
            // rms_norm block 1024: single-row 2816-col norms are one-block latency-bound at
            // 256 threads (7us/launch measured).
            p.rms_block = 1024;
            // gemma4 fa split ladder seam (d1736 sweep; see fa_split_keys).
            p.fa_sp_gemma = true;
        }
        // GLM-5.3-Flash on sm_100a: rms_norm block 1024 (B200 posture flip 2026-09-06, +2.59%
        // alone and inside every all4 receipt).
        if glm5_next_b200 {
            p.rms_block = 1024;
        }
        // Spec-serving stream-k key: big dense forces tiling, MoE/small defer to the
        // fail-closed TILE form (2026-07-27; the timing selector was removed 2026-08-14).
        if let Some(n_embd) = spec_mmq_n_embd {
            p.mmq_sk_force = if n_embd >= 3500 { 0 } else { -1 };
        }
        p.step37_serving = sliding_gated_moe;
        p
    }

    /// The profile installed on this thread: the innermost entered model's, else [`Self::BASE`].
    pub fn current() -> Self {
        CURRENT.get()
    }

    /// Install this profile on the current thread until the returned scope drops; the drop
    /// restores whatever was installed before, so scopes nest.
    pub fn enter(self) -> NumericScope {
        NumericScope {
            prev: CURRENT.replace(self),
            _thread: PhantomData,
        }
    }
}

thread_local! {
    static CURRENT: Cell<NumericProfile> = const { Cell::new(NumericProfile::BASE) };
}

/// An installed [`NumericProfile`]. Not `Send`: it restores the thread it was entered on.
#[must_use = "the profile is installed only while the scope lives"]
pub struct NumericScope {
    prev: NumericProfile,
    _thread: PhantomData<*const ()>,
}

impl Drop for NumericScope {
    fn drop(&mut self) {
        CURRENT.set(self.prev);
    }
}

/// A numeric env override, parsed once per process. A value that does not parse is no
/// override, so the profile's default applies, as the old latched readers did.
pub(crate) fn env_override<T: std::str::FromStr + Copy + Send + Sync>(
    cell: &'static OnceLock<Option<T>>,
    name: &str,
) -> Option<T> {
    *cell.get_or_init(|| std::env::var(name).ok().and_then(|v| v.parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::NumericProfile;

    fn gemma31() -> NumericProfile {
        NumericProfile::for_model(2, Some(false), false, false, None)
    }

    #[test]
    fn qwen_class_keys_nothing() {
        assert_eq!(
            NumericProfile::for_model(4, None, false, false, None),
            NumericProfile::BASE
        );
        // Small/MoE under spec intent keeps the -1 key.
        assert_eq!(
            NumericProfile::for_model(4, None, false, false, Some(2048)),
            NumericProfile::BASE
        );
    }

    #[test]
    fn per_model_keys_are_the_loader_values() {
        let g31 = gemma31();
        assert_eq!(
            (
                g31.fa_vec_min,
                g31.fa_spw,
                g31.fa_sp512,
                g31.fused_mr1,
                g31.rms_block,
                g31.fa_sp_gemma
            ),
            (1, 64, 32, true, 1024, true)
        );
        let g26 = NumericProfile::for_model(2, Some(true), false, false, None);
        assert_eq!((g26.fa_spw, g26.fa_sp512, g26.fused_mr1), (32, 16, false));
        assert_eq!(
            NumericProfile::for_model(16, None, false, false, None).fa_v4_max,
            0
        );
        assert_eq!(
            NumericProfile::for_model(1, None, true, false, None).rms_block,
            1024
        );
        assert_eq!(
            NumericProfile::for_model(4, None, false, false, Some(5120)).mmq_sk_force,
            0
        );
        assert!(NumericProfile::for_model(4, None, false, true, None).step37_serving);
    }

    #[test]
    fn scopes_nest_and_restore() {
        let gemma = gemma31();
        let gqa16 = NumericProfile::for_model(16, None, false, false, None);
        assert_eq!(NumericProfile::current(), NumericProfile::BASE);
        {
            let _outer = gemma.enter();
            assert_eq!(NumericProfile::current(), gemma);
            {
                let _inner = gqa16.enter();
                assert_eq!(NumericProfile::current(), gqa16);
            }
            assert_eq!(NumericProfile::current(), gemma);
        }
        assert_eq!(NumericProfile::current(), NumericProfile::BASE);
    }

    #[test]
    fn a_scope_is_per_thread_and_a_spawned_worker_inherits_by_entering() {
        let gemma = gemma31();
        let _scope = gemma.enter();
        let profile = NumericProfile::current();
        let (bare, entered) = std::thread::scope(|s| {
            let bare = s.spawn(NumericProfile::current).join().unwrap();
            let entered = s
                .spawn(move || {
                    let _numeric = profile.enter();
                    NumericProfile::current()
                })
                .join()
                .unwrap();
            (bare, entered)
        });
        assert_eq!(bare, NumericProfile::BASE);
        assert_eq!(entered, gemma);
    }
}
