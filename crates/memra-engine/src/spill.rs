//! SPILLING-PLAN: full tiered spilling (VRAM ↔ pinned-host ↔ mmap-disk).
//!
//! Today memra has the VRAM↔pinned-host leg (the `MoeSlotCache` GPU slot cache + the pinned
//! `HostExps` host store). This module adds the **third tier**: a `HostBuf::Mmap` arm (model.rs)
//! so cold experts are demand-faulted from the GGUF file on disk instead of held in RAM, plus the
//! runtime memory probe (`MemBudget`) that decides — per expert, at load — which tier each block
//! lives in. Never hardcode: VRAM is queried via `cuMemGetInfo`, host RAM via `/proc/meminfo`.
//!
//! THE GATE (SPILLING-PLAN §8): spilling is a memory-PLACEMENT change, never a numerics change. A
//! `Mmap` expert and a `Pinned` expert feed `qmatvec_view` byte-for-byte identical GGUF bytes — the
//! `Pinned`/`Paged` stores copied FROM exactly those on-disk bytes — so argmax is unchanged.
//!
//! The disk tier is gated behind `MEMRA_SPILL_DISK`. Unset (default) = the current all-host
//! behavior, byte-identical: `HostExps::tiers` stays `None` and every expert slices the single
//! pinned/paged backing store. The daily models (9B/27B) fit 24 GB and NEVER trigger spill.

use crate::Engine;
use crate::model::HostBuf;
use memra_gguf::{
    bound_disk::ExpertDiskView,
    source::{GgufSource, TensorSource},
};
use std::sync::Arc;

const DEFAULT_PINNED_FRAC: f64 = 0.60;

fn parse_pinned_frac(raw: Option<&str>) -> Result<f64, &'static str> {
    let frac = raw
        .unwrap_or("0.60")
        .parse::<f64>()
        .map_err(|_| "expected a number")?;
    if frac.is_finite() && frac > 0.0 && frac <= 1.0 {
        Ok(frac)
    } else {
        Err("expected a finite fraction greater than 0 and at most 1")
    }
}

fn configured_pinned_frac() -> f64 {
    static PINNED_FRAC: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *PINNED_FRAC.get_or_init(|| {
        let raw = std::env::var("MEMRA_SPILL_PINNED_FRAC").ok();
        match parse_pinned_frac(raw.as_deref()) {
            Ok(frac) => frac,
            Err(reason) => {
                crate::spill_pread::note_config_fallback();
                eprintln!(
                    "[spill] invalid MEMRA_SPILL_PINNED_FRAC={:?} ({reason}); using {DEFAULT_PINNED_FRAC}",
                    raw.as_deref().unwrap_or("")
                );
                DEFAULT_PINNED_FRAC
            }
        }
    })
}

/// Runtime free-memory budget (SPILLING-PLAN §2). Both numbers are QUERIED at load, never
/// hardcoded — free host RAM "varies with other LLM servers", so the split between pinned (Tier 1)
/// and disk (Tier 2) must be decided against the live machine state.
#[derive(Clone, Copy, Debug)]
pub struct MemBudget {
    /// Free VRAM in bytes, from `cuMemGetInfo` (authoritative; accounts for other GPU processes).
    pub free_vram: usize,
    /// Bytes of host RAM safe to pin: `/proc/meminfo MemAvailable` × `pinned_frac` (default 0.60).
    /// Capped so `cudaHostAlloc` can neither OOM nor evict the page cache the Tier-2 mmap depends on.
    pub free_pinnable_ram: usize,
}

impl MemBudget {
    pub fn probe(e: &Engine) -> Result<Self, Box<dyn std::error::Error>> {
        let (free_vram, _total) = e.ctx().mem_get_info()?; // same call moe_cache.rs:77 uses
        let avail = read_meminfo_kb("MemAvailable")? * 1024; // MemAvailable (NOT MemFree)
        let frac = configured_pinned_frac();
        Ok(MemBudget {
            free_vram,
            free_pinnable_ram: ((avail as f64 * frac) as usize).min(avail),
        })
    }
}

/// Parse one `/proc/meminfo` field (a value in kB) by key, e.g. "MemAvailable".
fn read_meminfo_kb(key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    for line in s.lines() {
        // line form: "MemAvailable:   12345678 kB"
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start_matches(':').trim();
            let kb: usize = rest
                .split_whitespace()
                .next()
                .ok_or("malformed /proc/meminfo line")?
                .parse()?;
            return Ok(kb);
        }
    }
    Err(format!("/proc/meminfo: key {key} not found").into())
}

/// Is the disk tier (Tier 2) enabled? Gated behind `MEMRA_SPILL_DISK`. Default (unset) = off =>
/// the unchanged all-host path (`HostExps::tiers` stays `None`). Set to anything to force-on.
#[inline]
pub fn disk_tier_enabled() -> bool {
    std::env::var("MEMRA_SPILL_DISK").is_ok()
}

/// One load's shared pinned-RAM budget. Sources own the parsed shard mappings; each placement
/// requests its exact tensor window through the source instead of holding whole-shard handles.
pub struct SpillCtx {
    pub pinned_remaining: usize,
    pub n_pinned: usize,
    pub n_mmap: usize,
    pub mmap_bytes: usize,
}

impl SpillCtx {
    pub fn open(
        g: &memra_gguf::GgufFile,
        budget: &MemBudget,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_source(&GgufSource(g), budget)
    }
    pub(crate) fn from_source(
        src: &dyn TensorSource,
        budget: &MemBudget,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if src.gguf_tensor_metadata()?.is_none() {
            return Err("GGUF spill context requires a GGUF source".into());
        }
        Ok(Self {
            pinned_remaining: budget.free_pinnable_ram,
            n_pinned: 0,
            n_mmap: 0,
            mmap_bytes: 0,
        })
    }
    pub(crate) fn tensor_view(
        &self,
        src: &dyn TensorSource,
        name: &str,
    ) -> Result<ExpertDiskView, Box<dyn std::error::Error>> {
        let view = src
            .try_find_gguf_disk(name)?
            .ok_or_else(|| format!("missing GGUF spill extent {name}"))?;
        // Same configured random/normal advice as the historical whole-shard spill maps.
        // The owning source retains the parsed map; no alternate pathname or mapping is opened.
        view.advise_expert_access();
        Ok(view)
    }
}

/// Pin while the same running budget covers the expert; otherwise retain the exact disk window.
/// Placement never widens a tensor window and both tiers copy/read the same encoded bytes.
pub fn place_expert(
    ctx: &mut SpillCtx,
    e: &Engine,
    view: ExpertDiskView,
) -> Result<HostBuf, Box<dyn std::error::Error>> {
    let len = view.len();
    if ctx.pinned_remaining >= len {
        let mut p = unsafe { e.ctx().alloc_pinned::<u8>(len)? };
        p.as_mut_slice()?.copy_from_slice(view.bytes());
        let base = p.as_ptr()?;
        ctx.pinned_remaining -= len;
        ctx.n_pinned += 1;
        Ok(HostBuf::Pinned {
            slice: Arc::new(p),
            base,
            len,
        })
    } else {
        let host = HostBuf::from_disk(view)?;
        ctx.n_mmap += 1;
        ctx.mmap_bytes += len;
        Ok(host)
    }
}

/// SPILLING-PLAN §3/§5: a single spillable weight block over the same `{Pinned, Mmap}` substrate.
/// Lifted from the `HostExps` fields so dense weights (dense-70B case) can reuse the disk tier
/// without the 256-expert stacking. Carried for the requested generalization; the MoE path uses
/// `HostExps` directly (which now embeds the same tier machinery via `HostBuf`).
pub struct SpillBlock {
    pub host: HostBuf,
    pub qtype: i32,
    pub in_f: usize,
    pub out_f: usize,
    pub row_bytes: usize,
}

impl SpillBlock {
    /// The H2D DMA source for this block — resolves the tier (`Pinned` fast / `Mmap` demand-fault).
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.host.as_bytes()
    }
}

/// SPILLING-PLAN §3: the requested `Tiered` generalization. Structurally it is the existing
/// `HostExps` (Tier 1/2 host backing, per-block) composed with the existing `MoeSlotCache`
/// (Tier 0 GPU residency). Both seams are already present and unchanged; this names the composition.
/// The MoE hot loop drives the two seams directly (`expert_bytes()` + `with_moe_cache`), so this is
/// a documentation/structural alias, not a new hot path.
pub struct Tiered {
    pub host: crate::model::HostExps, // Tier 1/2 (Pinned hot / Mmap cold), per-expert
    pub slots: crate::moe_cache::MoeSlotCache, // Tier 0 GPU residency (existing slot cache)
}

#[cfg(all(test, unix))]
mod tests {
    use super::{
        DEFAULT_PINNED_FRAC, MemBudget, SpillCtx, configured_pinned_frac, parse_pinned_frac,
    };
    use crate::spill_pread::config_fallbacks;
    use memra_gguf::{GgufFile, source::GgufSource};

    #[test]
    fn pinned_frac_accepts_only_finite_values_in_range() {
        assert_eq!(parse_pinned_frac(None), Ok(DEFAULT_PINNED_FRAC));
        assert_eq!(parse_pinned_frac(Some("0.25")), Ok(0.25));
        assert_eq!(parse_pinned_frac(Some("1")), Ok(1.0));
        for invalid in ["0", "-0.1", "1.6", "NaN", "inf", "not-a-number"] {
            assert!(
                parse_pinned_frac(Some(invalid)).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn invalid_pinned_frac_is_counted_and_uses_safe_default() {
        const CHILD: &str = "MEMRA_INVALID_PINNED_FRAC_TEST_CHILD";
        const TEST: &str = "spill::tests::invalid_pinned_frac_is_counted_and_uses_safe_default";
        if std::env::var_os(CHILD).is_some() {
            assert_eq!(config_fallbacks(), 0);
            assert_eq!(configured_pinned_frac(), DEFAULT_PINNED_FRAC);
            assert_eq!(config_fallbacks(), 1);
            return;
        }

        for raw in ["1.6", "not-a-number"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg(TEST)
                .arg("--exact")
                .arg("--nocapture")
                .env(CHILD, "1")
                .env("MEMRA_SPILL_PINNED_FRAC", raw)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "invalid-config child failed for {raw:?}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(
                stderr.contains(&format!("invalid MEMRA_SPILL_PINNED_FRAC={raw:?}"))
                    && stderr.contains("using 0.6"),
                "invalid-config warning missing from child stderr:\n{stderr}"
            );
        }
    }

    #[test]
    fn spill_ctx_keeps_parsed_gguf_inode_after_path_replacement() {
        let path =
            std::env::temp_dir().join(format!("memra-spill-inode-{}.gguf", std::process::id()));
        memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
        let original = std::fs::read(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let expected = gguf
            .tensor_data(gguf.find("token_embd.weight").unwrap())
            .to_vec();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, vec![0xA5u8; original.len()]).unwrap();
        let budget = MemBudget {
            free_vram: 0,
            free_pinnable_ram: 0,
        };
        let spill = SpillCtx::open(&gguf, &budget).unwrap();
        let view = spill
            .tensor_view(&GgufSource(&gguf), "token_embd.weight")
            .unwrap();
        drop(gguf);
        assert_eq!(view.bytes(), expected);
        assert!(view.subrange(0..expected.len() + 1).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), vec![0xA5u8; original.len()]);

        std::fs::remove_file(path).ok();
    }
}
