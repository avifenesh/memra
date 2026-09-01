//! memra-kv — the dual KV/recurrent cache, extracted (Phase D, ARCHITECTURE-H100.md §5).
//!
//! Moved VERBATIM from memra-engine/src/cache.rs behind the `KvDev` seam: the cache only
//! ever needed 7 device ops (alloc/copy/set), so the trait is that surface and nothing
//! more. The append/dequant KERNELS stay in the engine fatbins — this crate owns the
//! structure, sizing math, and the KV format policy (env-selected, shared by the engine's
//! fatbin router and every cache consumer). memra-engine re-exports this as `cache` so
//! call sites are unchanged.

// ---------------- KV format policy (env-selected; moved from memra-engine) ----------------

/// Env-selected KV cache formats (MEMRA_KV_K / MEMRA_KV_V). The engine's flash-fatbin router
/// and the cache sizing below MUST agree — both read this one function.
pub fn kv_cache_formats() -> (&'static str, &'static str) {
    static F: std::sync::OnceLock<(&'static str, &'static str)> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let k = match std::env::var("MEMRA_KV_K").as_deref() {
            Ok("fp8") => "fp8",
            Ok("q8_0") | Ok("") | Err(_) => "q8_0",
            Ok(o) => panic!("MEMRA_KV_K={o} unsupported (q8_0 | fp8)"),
        };
        let v = match std::env::var("MEMRA_KV_V").as_deref() {
            Ok("q4_0") => "q4_0",
            Ok("fp8") => "fp8",
            Ok("q5_1") | Ok("") | Err(_) => "q5_1",
            Ok(o) => panic!("MEMRA_KV_V={o} unsupported (q5_1 | q4_0 | fp8)"),
        };
        if (k, v) != ("q8_0", "q5_1") {
            eprintln!("[memra] KV cache format: K={k} V={v} (non-default — new numeric config)");
        }
        (k, v)
    })
}

/// Per-32-element block bytes for the selected (K, V) formats.
pub fn kv_blk_bytes() -> (usize, usize) {
    let (k, v) = kv_cache_formats();
    let kb = match k {
        "fp8" => 32,
        _ => 34,
    };
    let vb = match v {
        "q4_0" => 18,
        "fp8" => 32,
        _ => 24,
    };
    (kb, vb)
}

/// Exact allocation geometry for one rank of a tensor-parallel KV sidecar.
///
/// The context-linear coefficient and fixed allocation bytes are shared by the CUDA allocator
/// and serving admission. Keeping both consumers on this function prevents a new KV format or
/// rank width from making the pre-admit estimate disagree with the buffers allocated at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpKvRankAllocationShape {
    pub kv_dim_k: usize,
    pub kv_dim_v: usize,
    pub k_token_bytes: usize,
    pub v_token_bytes: usize,
    pub fixed_bytes: usize,
}

impl TpKvRankAllocationShape {
    pub fn bytes_per_token(self) -> usize {
        self.k_token_bytes.saturating_add(self.v_token_bytes)
    }

    pub fn allocation_bytes(self, capacity: usize) -> usize {
        self.bytes_per_token()
            .saturating_mul(capacity)
            .saturating_add(self.fixed_bytes)
    }
}

pub fn tp_kv_rank_allocation_shape(
    kv_dim_k: usize,
    kv_dim_v: usize,
    ranks: usize,
) -> Result<TpKvRankAllocationShape, String> {
    if ranks == 0 || kv_dim_k == 0 || kv_dim_v == 0 {
        return Err(format!(
            "TP KV dimensions and rank count must be nonzero: k={kv_dim_k} v={kv_dim_v} \
             ranks={ranks}"
        ));
    }
    if kv_dim_k % ranks != 0 || kv_dim_v % ranks != 0 {
        return Err(format!(
            "TP KV dimensions k={kv_dim_k} v={kv_dim_v} are not divisible by TP={ranks}"
        ));
    }
    let local_k = kv_dim_k / ranks;
    let local_v = kv_dim_v / ranks;
    if local_k % 32 != 0 || local_v % 32 != 0 {
        return Err(format!(
            "TP KV local dimensions k={local_k} v={local_v} must be 32-aligned"
        ));
    }
    let (k_block_bytes, v_block_bytes) = kv_blk_bytes();
    let k_token_bytes = (local_k / 32)
        .checked_mul(k_block_bytes)
        .ok_or("TP KV K token-byte overflow")?;
    let v_token_bytes = (local_v / 32)
        .checked_mul(v_block_bytes)
        .ok_or("TP KV V token-byte overflow")?;
    Ok(TpKvRankAllocationShape {
        kv_dim_k: local_k,
        kv_dim_v: local_v,
        k_token_bytes,
        v_token_bytes,
        // Two CUDA byte planes retain their existing 8-byte tail pads, plus one i32 length
        // mirror. Allocator alignment remains visible through the device pool high-water.
        fixed_bytes: 8 + 8 + std::mem::size_of::<i32>(),
    })
}

/// FP8-GLOBALS switch (MEMRA_GEMMA_GKV, default ON): gemma global (hd512) layers keep
/// their KV in e4m3 — the dequant-latency arc (HANDOVER). Windowed layers stay q8_0/q5_1.
pub fn gkv_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_GEMMA_GKV")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// FP8-WINDOWED switch (MEMRA_GEMMA_WKV; serving-mode default): SPEC serving (MEMRA_DRAFT
/// set) -> OFF, plain -> ON — the acceptance-vs-depth record lives on the engine-side
/// history of `Engine::wkv_on` (git). Explicit env always wins.
pub fn wkv_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_GEMMA_WKV")
            .map(|v| v != "0")
            .unwrap_or_else(|_| std::env::var("MEMRA_DRAFT").is_err())
    })
}

/// Per-model FP8-KV door (-1 = unset → env/default off; 0 = off; 1 = on). Set at qwen
/// model load: the 2026-07-12 arc closed per-model — 9B +0.7-4% scaling with depth,
/// 27B flat (weight-bound), 35B −2% (fp8 format-gates its v3 dp4a lane off). Explicit
/// MEMRA_KV_FP8 wins. 9B adoption attempt REVERTED by measurement 2026-07-29 (−1% at 12k
/// on the then-current build) — loaders currently store 0.
pub static KV_FP8_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// QWEN FP8-KV switch (MEMRA_KV_FP8 explicit; else the per-model KV_FP8_FORCE door set
/// at model load; else OFF). Non-gemma full-attn layers hold e4m3 K/V via the kf8vf8
/// module.
pub fn kv_fp8_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    if let Some(v) = *ENV.get_or_init(|| std::env::var("MEMRA_KV_FP8").ok().map(|v| v == "1")) {
        return v;
    }
    matches!(KV_FP8_FORCE.load(std::sync::atomic::Ordering::Relaxed), 1)
}

/// Step35 SWA-ring experiment (default OFF). The first cut is deliberately architecture-scoped:
/// Gemma4's row-0-addressed window kernels cannot consume a rebased ring view.
pub fn swa_ring_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SWA_RING").as_deref() == Ok("1"))
}

/// With the SWA-ring door open, `prime_chunk_tokens` caps every legal chunk at this bound. The
/// ring carries one whole maximum-size prime chunk in addition to the reader's window.
pub const PRIME_CHUNK_MAX_TOKENS: usize = 4096;
const SWA_VIEW_ALIGNMENT_ROWS: usize = 32;

/// Physical rows required by the Step35 SWA reader contract. Prime starts at
/// `(base_len - (window - 1)) & !31`, so at most 31 masked rows precede the live window.
pub fn swa_ring_rows(window: usize, max_ctx: usize) -> usize {
    max_ctx.min(window + PRIME_CHUNK_MAX_TOKENS + (SWA_VIEW_ALIGNMENT_ROWS - 1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvRing {
    rows: usize,
    window: usize,
    base: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRingAppend {
    Contiguous {
        write_row: usize,
    },
    Rebase {
        src_row: usize,
        keep_rows: usize,
        new_base: usize,
        write_row: usize,
    },
}

impl KvRing {
    pub fn new(rows: usize, window: usize) -> Self {
        assert!(window > 0 && rows > 0, "invalid SWA ring geometry");
        Self {
            rows,
            window,
            base: 0,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn base(&self) -> usize {
        self.base
    }
    pub fn window(&self) -> usize {
        self.window
    }

    /// Plan a contiguous physical append. When the tail would wrap, retain the caller's exact
    /// aligned read prefix at row zero; the following read remains one contiguous CUDA view.
    pub fn append_plan(
        &self,
        len: usize,
        retain_from: usize,
        append_rows: usize,
    ) -> Result<KvRingAppend, String> {
        if len < self.base || retain_from < self.base || retain_from > len {
            return Err(format!(
                "SWA ring lapped required rows (base {}, retain {retain_from}, len {len})",
                self.base
            ));
        }
        let used = len - self.base;
        if used > self.rows {
            return Err(format!(
                "SWA ring state exceeds capacity ({used} > {})",
                self.rows
            ));
        }
        if used.saturating_add(append_rows) <= self.rows {
            return Ok(KvRingAppend::Contiguous {
                write_row: used % self.rows,
            });
        }

        let keep_rows = len - retain_from;
        if keep_rows.saturating_add(append_rows) > self.rows {
            return Err(format!(
                "SWA ring append does not fit (keep {keep_rows} + append {append_rows} > {})",
                self.rows
            ));
        }
        Ok(KvRingAppend::Rebase {
            src_row: retain_from - self.base,
            keep_rows,
            new_base: retain_from,
            write_row: keep_rows,
        })
    }

    pub fn apply_rebase(&mut self, new_base: usize) {
        debug_assert!(new_base >= self.base);
        self.base = new_base;
    }

    pub fn physical_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        if start < self.base || end < start || end - self.base > self.rows {
            return Err(format!(
                "SWA ring view [{start},{end}) is outside resident [{},{})",
                self.base,
                self.base + self.rows
            ));
        }
        let start_row = (start - self.base) % self.rows;
        let len = end - start;
        debug_assert!(
            start_row + len <= self.rows,
            "ring view must be contiguous after rebase"
        );
        Ok(start_row..start_row + len)
    }

    /// A rewind is usable only when the next aligned Step35 window view is still resident.
    pub fn can_rewind_to(&self, len: usize) -> bool {
        let raw = len.saturating_sub(self.window - 1);
        let view_start = raw & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
        view_start >= self.base
    }
}

// ---------------- the device seam ----------------

/// The 7 device ops the cache needs — nothing more. Implemented by the engine (and by
/// any future backend); all ops are stream-ordered on the implementor's worker stream.
pub trait KvDev {
    fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>>;
    fn uninit(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>>;
    fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>>;
    fn htod_i32(&self, v: &[i32]) -> Result<CudaSlice<i32>, Box<dyn std::error::Error>>;
    fn clone_dtod(
        &self,
        src: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>>;
    fn copy_into(
        &self,
        dst: &mut CudaSlice<f32>,
        off: usize,
        src: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn set_i32_one(&self, d: &mut CudaSlice<i32>, v: i32)
    -> Result<(), Box<dyn std::error::Error>>;
}

use cudarc::driver::CudaSlice;
use memra_gguf::config::{LayerKind, ModelConfig};
use memra_gguf::model_plan::{ModelPlan, ResidualTopology, StatePlan};

/// Per-full-attn-layer growing KV cache, resident on GPU. QUANTIZED (KVQUANT-PLAN §B):
/// K stored q8_0 (34 B/32 elem), V stored q5_1 (24 B/32 elem). Per-token byte layout keeps the
/// [token, kv_head, dim] element order so a 32-block never straddles a head (assert head_dim%32==0).
/// Element-within-token index = kv_head*head_dim + d; block = idx/32; lane = idx%32.
pub struct KvLayer {
    pub k: CudaSlice<u8>,   // q8_0 packed, capacity max_ctx*k_tok_bytes
    pub v: CudaSlice<u8>,   // q5_1 packed, capacity max_ctx*v_tok_bytes
    pub kv_dim_k: usize,    // head_dim_k * n_head_kv  (K elements per token)
    pub kv_dim_v: usize,    // head_dim_v * n_head_kv  (V elements per token)
    pub k_tok_bytes: usize, // (kv_dim_k/32)*34
    pub v_tok_bytes: usize, // (kv_dim_v/32)*24
    pub len: usize,
    /// Step35 SWA physical-row state. `len` remains absolute; `None` keeps the original flat
    /// `[0, max_ctx)` addressing contract.
    pub ring: Option<KvRing>,
    /// Device-resident mirror of `len` (CUDA-GRAPH-PLAN Phase 2). Holds the KV write SLOT for the
    /// append-dc kernel (old len, before this step's append); after `inc_seqlen` it holds the new
    /// len == t_kv for fa_decode_dc. Kept in lock-step with the host `len`. i32[1].
    pub len_d: CudaSlice<i32>,
}

impl KvLayer {
    pub fn physical_rows(
        &self,
        start: usize,
        end: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        match &self.ring {
            Some(ring) => ring.physical_range(start, end),
            None => Ok(start..end),
        }
    }
}

/// Per-linear-attn-layer fixed recurrent state.
/// conv_state and ssm_state are BOTH kept RESIDENT on GPU — the conv ring assemble + roll runs
/// on-device (conv_assemble_and_roll), so there is no per-step dtoh/htod for either.
pub struct RecurLayer {
    pub conv_state: CudaSlice<f32>, // GPU [conv_dim, d_conv-1] (channel c, tap j at c*pad + j)
    pub ssm_state: CudaSlice<f32>,  // GPU [d_state, d_state, num_v] transposed M[col][i]
    /// PERSISTENT second SSM-state buffer for the gdn-scan double buffer (DECODE DETERMINISM FIX).
    /// gdn_scan needs DISTINCT in/out state buffers. The old eager path allocated a fresh
    /// `state_scratch` via `e.uninit` every step and swapped its pointer into `ssm_state`; that
    /// per-step alloc/free churned the stream-ordered async pool, and the freed prior `ssm_state`
    /// block was recycled by the next step's scratch while a kernel referencing the swapped-in state
    /// was still in flight — a use-after-reuse that produced RUN-TO-RUN nondeterministic decode
    /// (two identical prompt primes diverged). We instead PING-PONG between two STABLE resident
    /// buffers (no per-step alloc/free, no pool churn): step writes into the spare, then swaps the
    /// two owned buffers in place. Stable pointers, identical math. Sized like `ssm_state`.
    pub ssm_state_alt: CudaSlice<f32>,
}

pub struct ResidentTpKvCacheRank {
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    len_d: CudaSlice<i32>,
    /// Physical row of LOGICAL row 0 after the last ring rebase (graph increment A: the
    /// windowed device-counter fa derives its view as {lstart = max(0, len - window);
    /// physical = lstart - base}). Host-written at rebase (rare) and at cache init; None
    /// until the graph door first arms it.
    base_d: Option<CudaSlice<i32>>,
}

impl ResidentTpKvCacheRank {
    pub fn new(k: CudaSlice<u8>, v: CudaSlice<u8>, len_d: CudaSlice<i32>) -> Self {
        Self {
            k,
            v,
            len_d,
            base_d: None,
        }
    }

    pub fn base_d(&self) -> Option<&CudaSlice<i32>> {
        self.base_d.as_ref()
    }

    pub fn base_d_mut(&mut self) -> Option<&mut CudaSlice<i32>> {
        self.base_d.as_mut()
    }

    pub fn arm_base_d(&mut self, buf: CudaSlice<i32>) {
        self.base_d = Some(buf);
    }

    pub fn k(&self) -> &CudaSlice<u8> {
        &self.k
    }

    pub fn v(&self) -> &CudaSlice<u8> {
        &self.v
    }

    pub fn len_d(&self) -> &CudaSlice<i32> {
        &self.len_d
    }

    pub fn k_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.k
    }

    pub fn v_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.v
    }

    pub fn planes_mut(&mut self) -> (&mut CudaSlice<u8>, &mut CudaSlice<u8>) {
        (&mut self.k, &mut self.v)
    }

    /// Split-borrow for the dcw append: both planes mutably plus the device counters shared.
    #[allow(clippy::type_complexity)]
    pub fn planes_and_counters_mut(
        &mut self,
    ) -> (
        &mut CudaSlice<u8>,
        &mut CudaSlice<u8>,
        &CudaSlice<i32>,
        Option<&CudaSlice<i32>>,
    ) {
        (&mut self.k, &mut self.v, &self.len_d, self.base_d.as_ref())
    }

    pub fn len_d_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.len_d
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpKvTransaction {
    generation: u64,
    base_len: usize,
}

impl TpKvTransaction {
    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn base_len(self) -> usize {
        self.base_len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpKvAppendPlan {
    transaction: TpKvTransaction,
    target: usize,
    write_row: usize,
    ring_append: Option<KvRingAppend>,
}

impl TpKvAppendPlan {
    pub fn target(self) -> usize {
        self.target
    }

    pub fn write_row(self) -> usize {
        self.write_row
    }

    pub fn ring_append(self) -> Option<KvRingAppend> {
        self.ring_append
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TpKvGrowPlan {
    rows: usize,
    source_row: usize,
    copy_rows: usize,
    target_base: usize,
    k_bytes: usize,
    v_bytes: usize,
    source_capacity: usize,
    target_capacity: usize,
    ring_window: Option<usize>,
    target_physical_rows: usize,
    kv_dim_k: usize,
    kv_dim_v: usize,
    k_tok_bytes: usize,
    v_tok_bytes: usize,
    ranks: usize,
    next_generation: u64,
}

impl TpKvGrowPlan {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn source_row(&self) -> usize {
        self.source_row
    }

    pub fn copy_rows(&self) -> usize {
        self.copy_rows
    }

    pub fn k_bytes(&self) -> usize {
        self.k_bytes
    }

    pub fn v_bytes(&self) -> usize {
        self.v_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TpKvTransactionState {
    committed_len: usize,
    staged_len: usize,
    next_generation: u64,
    active: Option<TpKvTransaction>,
}

impl TpKvTransactionState {
    fn new() -> Self {
        Self {
            committed_len: 0,
            staged_len: 0,
            next_generation: 1,
            active: None,
        }
    }

    fn begin(&mut self) -> Result<TpKvTransaction, String> {
        if let Some(active) = self.active {
            return Err(format!(
                "TP KV transaction generation {} is already active at base {}",
                active.generation, active.base_len
            ));
        }
        if self.staged_len != self.committed_len {
            return Err(format!(
                "TP KV cache is half-committed: staged {} != committed {}",
                self.staged_len, self.committed_len
            ));
        }
        let transaction = TpKvTransaction {
            generation: self.next_generation,
            base_len: self.committed_len,
        };
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or("TP KV transaction generation overflow")?;
        self.active = Some(transaction);
        Ok(transaction)
    }

    fn validate(&self, transaction: TpKvTransaction) -> Result<(), String> {
        if self.active != Some(transaction) {
            return Err(format!(
                "stale TP KV transaction generation {} at base {}",
                transaction.generation, transaction.base_len
            ));
        }
        if transaction.base_len != self.committed_len {
            return Err(format!(
                "TP KV transaction base {} != committed length {}",
                transaction.base_len, self.committed_len
            ));
        }
        Ok(())
    }

    fn append_target(
        &self,
        transaction: TpKvTransaction,
        rows: usize,
        capacity: usize,
    ) -> Result<usize, String> {
        self.validate(transaction)?;
        if rows == 0 {
            return Err("TP KV append must contain at least one row".into());
        }
        let target = self
            .staged_len
            .checked_add(rows)
            .ok_or("TP KV staged length overflow")?;
        if target > capacity {
            return Err(format!(
                "TP KV append exceeds capacity: {target} > {capacity}"
            ));
        }
        Ok(target)
    }

    fn publish_append(
        &mut self,
        transaction: TpKvTransaction,
        target: usize,
    ) -> Result<(), String> {
        self.validate(transaction)?;
        if target <= self.staged_len {
            return Err(format!(
                "TP KV append target {target} must exceed staged length {}",
                self.staged_len
            ));
        }
        self.staged_len = target;
        Ok(())
    }

    fn commit_target(
        &self,
        transaction: TpKvTransaction,
        accepted_rows: usize,
    ) -> Result<usize, String> {
        self.validate(transaction)?;
        let staged_rows = self
            .staged_len
            .checked_sub(transaction.base_len)
            .ok_or("TP KV staged length precedes its transaction base")?;
        if accepted_rows > staged_rows {
            return Err(format!(
                "TP KV commit accepts {accepted_rows} rows from a {staged_rows}-row transaction"
            ));
        }
        transaction
            .base_len
            .checked_add(accepted_rows)
            .ok_or_else(|| "TP KV committed length overflow".to_string())
    }

    fn publish_finalize(
        &mut self,
        transaction: TpKvTransaction,
        target: usize,
    ) -> Result<(), String> {
        self.validate(transaction)?;
        if target < transaction.base_len || target > self.staged_len {
            return Err(format!(
                "TP KV finalize target {target} outside transaction range {}..={}",
                transaction.base_len, self.staged_len
            ));
        }
        self.committed_len = target;
        self.staged_len = target;
        self.active = None;
        Ok(())
    }

    fn rewind(&mut self, target: usize, capacity: usize) -> Result<(), String> {
        if target > capacity {
            return Err(format!(
                "TP KV rewind target {target} exceeds capacity {capacity}"
            ));
        }
        self.committed_len = target;
        self.staged_len = target;
        self.active = None;
        Ok(())
    }
}

pub struct ResidentTpKvCache {
    ranks: Vec<ResidentTpKvCacheRank>,
    kv_dim_k: usize,
    kv_dim_v: usize,
    k_tok_bytes: usize,
    v_tok_bytes: usize,
    capacity: usize,
    ring: Option<KvRing>,
    state: TpKvTransactionState,
}

impl ResidentTpKvCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ranks: Vec<ResidentTpKvCacheRank>,
        kv_dim_k: usize,
        kv_dim_v: usize,
        k_tok_bytes: usize,
        v_tok_bytes: usize,
        capacity: usize,
    ) -> Self {
        Self::new_inner(
            ranks,
            kv_dim_k,
            kv_dim_v,
            k_tok_bytes,
            v_tok_bytes,
            capacity,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_swa(
        ranks: Vec<ResidentTpKvCacheRank>,
        kv_dim_k: usize,
        kv_dim_v: usize,
        k_tok_bytes: usize,
        v_tok_bytes: usize,
        capacity: usize,
        window: usize,
    ) -> Self {
        Self::new_inner(
            ranks,
            kv_dim_k,
            kv_dim_v,
            k_tok_bytes,
            v_tok_bytes,
            capacity,
            Some(KvRing::new(swa_ring_rows(window, capacity), window)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        ranks: Vec<ResidentTpKvCacheRank>,
        kv_dim_k: usize,
        kv_dim_v: usize,
        k_tok_bytes: usize,
        v_tok_bytes: usize,
        capacity: usize,
        ring: Option<KvRing>,
    ) -> Self {
        Self {
            ranks,
            kv_dim_k,
            kv_dim_v,
            k_tok_bytes,
            v_tok_bytes,
            capacity,
            ring,
            state: TpKvTransactionState::new(),
        }
    }

    pub fn begin_transaction(&mut self) -> Result<TpKvTransaction, String> {
        self.state.begin()
    }

    pub fn committed_len(&self) -> usize {
        self.state.committed_len
    }

    pub fn staged_len(&self) -> usize {
        self.state.staged_len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn physical_capacity(&self) -> usize {
        self.ring
            .as_ref()
            .map(KvRing::rows)
            .unwrap_or(self.capacity)
    }

    pub fn ring_window(&self) -> Option<usize> {
        self.ring.as_ref().map(KvRing::window)
    }

    pub fn ring_base(&self) -> Option<usize> {
        self.ring.as_ref().map(KvRing::base)
    }

    pub fn physical_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        match &self.ring {
            Some(ring) => ring.physical_range(start, end),
            None => {
                if end < start || end > self.capacity {
                    return Err(format!(
                        "TP KV linear view [{start},{end}) exceeds capacity {}",
                        self.capacity
                    ));
                }
                Ok(start..end)
            }
        }
    }

    pub fn can_rewind_to(&self, target: usize) -> bool {
        target <= self.capacity
            && self
                .ring
                .as_ref()
                .is_none_or(|ring| ring.can_rewind_to(target))
    }

    pub fn kv_dim_k(&self) -> usize {
        self.kv_dim_k
    }

    pub fn kv_dim_v(&self) -> usize {
        self.kv_dim_v
    }

    pub fn k_tok_bytes(&self) -> usize {
        self.k_tok_bytes
    }

    pub fn v_tok_bytes(&self) -> usize {
        self.v_tok_bytes
    }

    pub fn ranks_len(&self) -> usize {
        self.ranks.len()
    }

    pub fn rank(&self, rank: usize) -> Option<&ResidentTpKvCacheRank> {
        self.ranks.get(rank)
    }

    pub fn rank_mut(&mut self, rank: usize) -> Option<&mut ResidentTpKvCacheRank> {
        self.ranks.get_mut(rank)
    }

    pub fn ranks(&self) -> &[ResidentTpKvCacheRank] {
        &self.ranks
    }

    pub fn ranks_mut(&mut self) -> &mut [ResidentTpKvCacheRank] {
        &mut self.ranks
    }

    pub fn prepare_grow(
        &self,
        target_capacity: usize,
        rows: usize,
    ) -> Result<TpKvGrowPlan, String> {
        if let Some(active) = self.state.active {
            return Err(format!(
                "TP KV grow refuses active transaction generation {} at base {}",
                active.generation, active.base_len
            ));
        }
        if self.state.staged_len != self.state.committed_len {
            return Err(format!(
                "TP KV grow requires quiescent state, got committed/staged={}/{}",
                self.state.committed_len, self.state.staged_len
            ));
        }
        if target_capacity <= self.capacity {
            return Err(format!(
                "TP KV grow target capacity {target_capacity} must exceed source capacity {}",
                self.capacity
            ));
        }
        if target_capacity > i32::MAX as usize {
            return Err(format!(
                "TP KV grow target capacity {target_capacity} exceeds i32 device mirrors"
            ));
        }
        if rows > self.state.committed_len {
            return Err(format!(
                "TP KV grow rows {rows} exceed committed length {}",
                self.state.committed_len
            ));
        }
        let (source_row, copy_rows, target_base, ring_window, target_physical_rows) =
            match &self.ring {
                Some(ring) => {
                    let raw = rows.saturating_sub(ring.window().saturating_sub(1));
                    let target_base = raw & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
                    let physical = ring.physical_range(target_base, rows)?;
                    (
                        physical.start,
                        physical.len(),
                        target_base,
                        Some(ring.window()),
                        swa_ring_rows(ring.window(), target_capacity),
                    )
                }
                None => (0, rows, 0, None, target_capacity),
            };
        let k_bytes = copy_rows
            .checked_mul(self.k_tok_bytes)
            .ok_or("TP KV grow K byte extent overflow")?;
        let v_bytes = copy_rows
            .checked_mul(self.v_tok_bytes)
            .ok_or("TP KV grow V byte extent overflow")?;
        Ok(TpKvGrowPlan {
            rows,
            source_row,
            copy_rows,
            target_base,
            k_bytes,
            v_bytes,
            source_capacity: self.capacity,
            target_capacity,
            ring_window,
            target_physical_rows,
            kv_dim_k: self.kv_dim_k,
            kv_dim_v: self.kv_dim_v,
            k_tok_bytes: self.k_tok_bytes,
            v_tok_bytes: self.v_tok_bytes,
            ranks: self.ranks.len(),
            next_generation: self.state.next_generation,
        })
    }

    pub fn publish_grow(&mut self, plan: TpKvGrowPlan) -> Result<(), String> {
        if self.state != TpKvTransactionState::new() {
            return Err(format!(
                "TP KV grow target must be fresh, got committed/staged={}/{} active={}",
                self.state.committed_len,
                self.state.staged_len,
                self.state.active.is_some()
            ));
        }
        if self.capacity != plan.target_capacity
            || self.capacity <= plan.source_capacity
            || self.kv_dim_k != plan.kv_dim_k
            || self.kv_dim_v != plan.kv_dim_v
            || self.k_tok_bytes != plan.k_tok_bytes
            || self.v_tok_bytes != plan.v_tok_bytes
            || self.ranks.len() != plan.ranks
            || self.ring.as_ref().map(KvRing::window) != plan.ring_window
            || self.physical_capacity() != plan.target_physical_rows
        {
            return Err("TP KV grow target layout does not match its source plan".into());
        }
        if plan.rows > self.capacity {
            return Err(format!(
                "TP KV grow rows {} exceed target capacity {}",
                plan.rows, self.capacity
            ));
        }
        if let Some(ring) = self.ring.as_mut() {
            let mut target_ring = *ring;
            target_ring.apply_rebase(plan.target_base);
            if !target_ring.can_rewind_to(plan.rows) {
                return Err(format!(
                    "TP KV grow target ring base {} cannot expose committed length {}",
                    target_ring.base(),
                    plan.rows
                ));
            }
            *ring = target_ring;
        }
        self.state.committed_len = plan.rows;
        self.state.staged_len = plan.rows;
        self.state.next_generation = plan.next_generation;
        self.state.active = None;
        Ok(())
    }

    pub fn prepare_append(
        &self,
        transaction: TpKvTransaction,
        rows: usize,
    ) -> Result<TpKvAppendPlan, String> {
        let target = self.state.append_target(transaction, rows, self.capacity)?;
        let ring_append = self
            .ring
            .as_ref()
            .map(|ring| {
                let staged_retain =
                    target.saturating_sub(ring.window()) & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
                let rollback_retain = transaction
                    .base_len
                    .saturating_sub(ring.window().saturating_sub(1))
                    & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
                ring.append_plan(
                    self.state.staged_len,
                    staged_retain.min(rollback_retain),
                    rows,
                )
            })
            .transpose()?;
        let write_row = match ring_append {
            Some(KvRingAppend::Contiguous { write_row })
            | Some(KvRingAppend::Rebase { write_row, .. }) => write_row,
            None => self.state.staged_len,
        };
        Ok(TpKvAppendPlan {
            transaction,
            target,
            write_row,
            ring_append,
        })
    }

    /// Read-only peek at the NEXT append's ring plan: (write_row, would_rebase). The dcw
    /// (device-counter) append path uses it to route rebase tokens through the full host
    /// path — the in-kernel row (len - base) is only valid for contiguous appends.
    pub fn peek_append_ring(&self, rows: usize) -> Result<(usize, bool), String> {
        let target = self.state.staged_len + rows;
        let plan = self
            .ring
            .as_ref()
            .map(|ring| {
                let staged_retain =
                    target.saturating_sub(ring.window()) & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
                let rollback_retain = self
                    .state
                    .staged_len
                    .saturating_sub(ring.window().saturating_sub(1))
                    & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
                ring.append_plan(
                    self.state.staged_len,
                    staged_retain.min(rollback_retain),
                    rows,
                )
            })
            .transpose()?;
        Ok(match plan {
            Some(KvRingAppend::Contiguous { write_row }) => (write_row, false),
            Some(KvRingAppend::Rebase { write_row, .. }) => (write_row, true),
            None => (self.state.staged_len, false),
        })
    }

    pub fn publish_append_rebase(&mut self, plan: TpKvAppendPlan) -> Result<(), String> {
        self.state.validate(plan.transaction)?;
        match (self.ring.as_mut(), plan.ring_append) {
            (
                Some(ring),
                Some(KvRingAppend::Rebase {
                    new_base,
                    keep_rows,
                    ..
                }),
            ) => {
                if keep_rows > ring.rows() {
                    return Err(format!(
                        "TP KV ring rebase keeps {keep_rows} rows in {} physical rows",
                        ring.rows()
                    ));
                }
                let mut target_ring = *ring;
                target_ring.apply_rebase(new_base);
                if !target_ring.can_rewind_to(plan.transaction.base_len) {
                    return Err(format!(
                        "TP KV ring rebase to {new_base} laps transaction base {}",
                        plan.transaction.base_len
                    ));
                }
                *ring = target_ring;
                Ok(())
            }
            (Some(_), Some(KvRingAppend::Contiguous { .. })) | (None, None) => Ok(()),
            _ => Err("TP KV append plan does not match cache ring layout".into()),
        }
    }

    pub fn publish_append_plan(&mut self, plan: TpKvAppendPlan) -> Result<(), String> {
        if let Some(KvRingAppend::Rebase { new_base, .. }) = plan.ring_append {
            if self.ring.as_ref().map(KvRing::base) != Some(new_base) {
                return Err(format!(
                    "TP KV append rebase {new_base} was not published before its state"
                ));
            }
        }
        self.state.publish_append(plan.transaction, plan.target)
    }

    pub fn publish_hydration(
        &mut self,
        logical_len: usize,
        resident_start: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.state != TpKvTransactionState::new() {
            return Err("TP KV hydration target must be fresh".into());
        }
        if resident_start > logical_len || logical_len > self.capacity {
            return Err(format!(
                "TP KV hydration range [{resident_start},{logical_len}) exceeds capacity {}",
                self.capacity
            )
            .into());
        }
        match self.ring.as_mut() {
            Some(ring) => {
                let rows = logical_len - resident_start;
                if rows > ring.rows() {
                    return Err(format!(
                        "TP KV hydration requires {rows} rows in a {}-row ring",
                        ring.rows()
                    )
                    .into());
                }
                let mut hydrated_ring = *ring;
                hydrated_ring.apply_rebase(resident_start);
                if !hydrated_ring.can_rewind_to(logical_len) {
                    return Err(format!(
                        "TP KV hydration base {resident_start} cannot expose logical length \
                         {logical_len}"
                    )
                    .into());
                }
                *ring = hydrated_ring;
            }
            None if resident_start != 0 => {
                return Err("linear TP KV hydration must start at absolute row zero".into());
            }
            None => {}
        }
        self.rewind_to(logical_len)
    }

    pub fn append_target(
        &self,
        transaction: TpKvTransaction,
        rows: usize,
    ) -> Result<usize, String> {
        self.state.append_target(transaction, rows, self.capacity)
    }

    pub fn publish_append(
        &mut self,
        transaction: TpKvTransaction,
        target: usize,
    ) -> Result<(), String> {
        self.state.publish_append(transaction, target)
    }

    pub fn commit_target(
        &self,
        transaction: TpKvTransaction,
        accepted_rows: usize,
    ) -> Result<usize, String> {
        self.state.commit_target(transaction, accepted_rows)
    }

    pub fn validate_transaction(&self, transaction: TpKvTransaction) -> Result<(), String> {
        self.state.validate(transaction)
    }

    pub fn publish_finalize(
        &mut self,
        transaction: TpKvTransaction,
        target: usize,
    ) -> Result<(), String> {
        if !self.can_rewind_to(target) {
            return Err(format!(
                "TP KV finalize target {target} is outside the resident cache window/capacity"
            ));
        }
        self.state.publish_finalize(transaction, target)
    }

    pub fn rewind_to(&mut self, target: usize) -> Result<(), Box<dyn std::error::Error>> {
        if !self.can_rewind_to(target) {
            return Err(format!(
                "TP KV rewind target {target} is outside the resident cache window/capacity"
            )
            .into());
        }
        let target_i32 =
            i32::try_from(target).map_err(|_| "TP KV length exceeds i32 device mirror")?;
        for rank in &mut self.ranks {
            let stream = rank.len_d.stream().clone();
            stream.memcpy_htod(&[target_i32], &mut rank.len_d)?;
        }
        self.state.rewind(target, self.capacity)?;
        Ok(())
    }
}

pub struct Cache {
    pub kv: Vec<Option<KvLayer>>,
    pub recur: Vec<Option<RecurLayer>>,
    /// Optional per-layer tensor-parallel KV planes. The ordinary owning-stage cache remains
    /// allocated as the rollback oracle until the distributed serving path is fully qualified.
    pub tp_kv: Vec<Option<ResidentTpKvCache>>,
    pub pos: usize,
    pub max_ctx: usize,
    /// BATCHED-TICK increment 2 component 3 (lean logits, 2026-08-01): device-side park of
    /// this session's LAST logits row. Device-sampled rows in the batched serving tick skip
    /// the [n_vocab] logits D2H entirely; the tick instead dtod-copies the row here (device
    /// bandwidth, ~µs) so the ONE consumer that truly needs the final row — the KV-reuse
    /// pool's park-at-retire (an empty-suffix resume samples from parked last_logits) —
    /// can D2H it once at retire. Lazily allocated on the first lean tick; None on every
    /// non-lean path (zero cost). Travels with the Cache into the reuse pool.
    pub last_logits_dev: Option<CudaSlice<f32>>,
    /// DFlash tap sink (dflash lane, 2026-07-13): when armed, the gemma4 verify/prime
    /// trunks copy the residual stream AFTER each tapped layer into `buf` rows
    /// ([t, n_taps*hidden] row-major — the drafter fc input layout). None on every
    /// non-dflash path (zero cost).
    pub dflash_taps: Option<DflashTapSink>,
}

/// The context-linear K/V layout for one full-attention layer. This is the single sizing source
/// used by both `Cache::new_inner` and `cache_bytes_per_token`: admission must never reimplement
/// Gemma's per-layer geometry or the active KV-format doors independently from the allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FullAttentionClass {
    Ordinary,
    GemmaGlobal,
    GemmaWindowed,
}

fn full_attention_class(plan: &ModelPlan, il: u32) -> FullAttentionClass {
    let layer = plan
        .layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .find(|layer| layer.index == il)
        .unwrap_or_else(|| panic!("ModelPlan has no layer {il}"));
    if !matches!(layer.residual, ResidualTopology::Gemma { .. }) {
        return FullAttentionClass::Ordinary;
    }
    match layer.state {
        StatePlan::SlidingKvCache { .. } => FullAttentionClass::GemmaWindowed,
        StatePlan::KvCache { .. } => FullAttentionClass::GemmaGlobal,
        _ => panic!("Gemma layer {il} does not declare a KV-cache state"),
    }
}

fn full_attention_kv_layout(
    cfg: &ModelConfig,
    plan: &ModelPlan,
    il: u32,
) -> (usize, usize, usize, usize) {
    debug_assert_eq!(cfg.layer_kind(il), LayerKind::FullAttention);
    let class = full_attention_class(plan, il);
    let n_head_kv = cfg.n_head_kv as usize;
    let (kv_dim_k, kv_dim_v) = match class {
        FullAttentionClass::GemmaGlobal | FullAttentionClass::GemmaWindowed => {
            let g = cfg
                .gemma4
                .as_ref()
                .expect("Gemma ModelPlan layer requires Gemma cache geometry");
            let hd = match class {
                FullAttentionClass::GemmaWindowed => g.key_length_swa,
                FullAttentionClass::GemmaGlobal => g.key_length_global,
                FullAttentionClass::Ordinary => unreachable!(),
            } as usize;
            // E4B ships a SCALAR head_count_kv (per-layer vec empty; scalar = 2 in
            // the gguf, landing in cfg.n_head_kv): kv_dim = hd * 2 for BOTH kinds —
            // swa 2x256 = 512, global 2x512 = 1024. The old fallback used
            // key_length_global (512) for both, which HALVED the global layers' K/V
            // (the attn writes wk.out_features = 1024 rows): every E4B global layer
            // stored/attended half its K/V and the batched append read row strides
            // wrong — THE cross-mode maxdiff-30 root (2026-07-12 bisect, il=5 slot-1
            // byte forensics). 26B/31B keep the per-layer vec.
            let d = match g.head_count_kv.get(il as usize) {
                Some(n) => hd * *n as usize,
                None => hd * n_head_kv,
            };
            (d, d)
        }
        FullAttentionClass::Ordinary => (
            cfg.head_dim_k as usize * n_head_kv,
            cfg.head_dim_v as usize * n_head_kv,
        ),
    };
    assert!(
        kv_dim_k % 32 == 0 && kv_dim_v % 32 == 0,
        "KVQUANT requires per-layer kv_dim_k%32==0 && kv_dim_v%32==0 \
         (layer {il}: k={kv_dim_k} v={kv_dim_v})"
    );
    let (kbb, vbb) = kv_blk_bytes();
    let g4_global_fp8 = gkv_on() && class == FullAttentionClass::GemmaGlobal;
    let g4_windowed_fp8 = wkv_on() && class == FullAttentionClass::GemmaWindowed;
    let qwen_fp8 = kv_fp8_on() && class == FullAttentionClass::Ordinary;
    let (kbb_l, vbb_l) = if g4_global_fp8 || g4_windowed_fp8 || qwen_fp8 {
        (32, 32)
    } else {
        (kbb, vbb)
    };
    (kv_dim_k, kv_dim_v, kbb_l, vbb_l)
}

fn kv_plane_allocation_bytes(rows: usize, token_bytes: usize) -> usize {
    rows * token_bytes + 8
}

/// Context-linear bytes allocated by one trunk cache token.
///
/// Fixed allocations (the 8-byte plane tail pads, `len_d`, recurrent state, and optional lazy
/// buffers) are deliberately excluded. Admission adds their measured high-water residual as a
/// request-independent activation term; multiplying this coefficient by the request's own
/// `ctx_cap` exactly mirrors the context-scaled allocations in `Cache::new_inner`.
pub fn cache_bytes_per_token(cfg: &ModelConfig) -> usize {
    cache_bytes_per_token_for_layers(cfg, 0, cfg.n_layer as usize)
}

/// Context-linear cache bytes per token owned by layers in `[lo, hi)`. PP admission uses the
/// same layer ranges as `Cache::new_ppn`, so each device is charged for exactly the cache planes
/// it allocates rather than for the aggregate model geometry.
pub fn cache_bytes_per_token_for_layers(cfg: &ModelConfig, lo: usize, hi: usize) -> usize {
    let plan = ModelPlan::compile(cfg).expect("cache sizing requires a compilable ModelPlan");
    cache_bytes_per_token_for_plan(cfg, &plan, lo, hi)
}

pub fn cache_bytes_per_token_for_plan(
    cfg: &ModelConfig,
    plan: &ModelPlan,
    lo: usize,
    hi: usize,
) -> usize {
    assert!(
        lo <= hi && hi <= cfg.n_layer as usize,
        "cache layer range out of bounds"
    );
    let shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
    (lo as u32..hi as u32)
        .filter(|&il| cfg.layer_kind(il) == LayerKind::FullAttention)
        .filter(|&il| shared == 0 || il < cfg.n_layer - shared)
        .map(|il| {
            let (kv_dim_k, kv_dim_v, kbb, vbb) = full_attention_kv_layout(cfg, plan, il);
            (kv_dim_k / 32) * kbb + (kv_dim_v / 32) * vbb
        })
        .sum()
}

/// Portion of [`cache_bytes_per_token`] whose physical row count is capped by the Step35 SWA
/// ring. Zero with the flag off and for every non-Step35 architecture.
pub fn cache_ring_bytes_per_token(cfg: &ModelConfig) -> usize {
    cache_ring_bytes_per_token_for_layers(cfg, 0, cfg.n_layer as usize)
}

/// Ring-capped portion of [`cache_bytes_per_token_for_layers`] for `[lo, hi)`.
pub fn cache_ring_bytes_per_token_for_layers(cfg: &ModelConfig, lo: usize, hi: usize) -> usize {
    assert!(
        lo <= hi && hi <= cfg.n_layer as usize,
        "cache layer range out of bounds"
    );
    let Ok(plan) = memra_gguf::model_plan::ModelPlan::compile(cfg) else {
        return 0;
    };
    cache_ring_bytes_per_token_for_plan(cfg, &plan, lo, hi)
}

pub fn cache_ring_bytes_per_token_for_plan(
    cfg: &ModelConfig,
    plan: &ModelPlan,
    lo: usize,
    hi: usize,
) -> usize {
    let total = plan.layers.len() + plan.mtp_blocks.len();
    assert!(
        lo <= hi && hi <= total,
        "cache plan layer range out of bounds"
    );
    if !swa_ring_on() {
        return 0;
    }
    let shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
    plan.layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .filter(|layer| (lo..hi).contains(&(layer.index as usize)))
        .filter(|layer| {
            matches!(
                layer.state,
                memra_gguf::model_plan::StatePlan::SlidingKvCache { .. }
            )
        })
        .filter(|layer| shared == 0 || layer.index < cfg.n_layer - shared)
        .map(|layer| {
            let (kv_dim_k, kv_dim_v, kbb, vbb) = full_attention_kv_layout(cfg, plan, layer.index);
            (kv_dim_k / 32) * kbb + (kv_dim_v / 32) * vbb
        })
        .sum()
}

/// Physical row cap shared by the Step35 SWA trunk and MTP scratch; zero when no ring is active.
pub fn cache_ring_row_cap(cfg: &ModelConfig) -> usize {
    let Ok(plan) = memra_gguf::model_plan::ModelPlan::compile(cfg) else {
        return 0;
    };
    cache_ring_row_cap_for_plan(&plan)
}

pub fn cache_ring_row_cap_for_plan(plan: &memra_gguf::model_plan::ModelPlan) -> usize {
    if !swa_ring_on() {
        return 0;
    }
    plan.layers
        .iter()
        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
        .filter_map(|layer| match layer.state {
            memra_gguf::model_plan::StatePlan::SlidingKvCache { window, .. } => {
                Some(window as usize)
            }
            _ => None,
        })
        .map(|window| swa_ring_rows(window, usize::MAX))
        .max()
        .unwrap_or(0)
}

/// See [`Cache::dflash_taps`]. Armed per forward by the dflash round (t = that forward's
/// row count); the trunk writes tap slot s of row r at buf[r*n_taps*hidden + s*hidden ..].
pub struct DflashTapSink {
    pub layer_ids: Vec<usize>,
    pub buf: CudaSlice<f32>,
    pub hidden: usize,
    pub t: usize,
    /// Row offset for writers that walk the buffer in windows (the qwen chunked prime):
    /// tap rows land at [base..base+t_chunk). Whole-buffer writers leave it 0.
    pub base: usize,
}

/// Snapshot of the dual cache taken BEFORE a spec-decode draft+verify round (MTP-PLAN §C/§D.4).
/// - Full-attn KV: only the per-layer `len` is recorded; rollback truncates (append-only,
///   position-addressed — no copy). C.1.
/// - Linear-attn conv/ssm: real device-to-device COPIES of the recurrent state, because those
///   buffers are mutated IN PLACE by the verify pass and have no position index to truncate. C.2.
///   (CudaSlice::clone is an Arc refcount, NOT a buffer copy — so we alloc fresh + memcpy_dtod.)
pub struct CacheSnapshot {
    pub kv_len: Vec<Option<usize>>, // per layer (Some for full-attn layers)
    pub tp_kv_len: Vec<Option<usize>>, // per layer (Some for TP full-attn layers)
    pub conv: Vec<Option<CudaSlice<f32>>>, // per layer (Some for linear-attn layers, D2D copy)
    pub ssm: Vec<Option<CudaSlice<f32>>>,
    pub pos: usize,
}

impl Cache {
    /// Allocate GPU-resident caches sized by arch + max context.
    pub fn new(
        e: &impl KvDev,
        cfg: &ModelConfig,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(&|_| e, cfg, None, max_ctx)
    }

    pub fn new_planned(
        e: &impl KvDev,
        cfg: &ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(&|_| e, cfg, Some(plan), max_ctx)
    }

    /// M1-PP2 increment 2 (stage-owned KV): layers [0, split) allocate through `dev0`,
    /// layers [split, n) through `dev1` — each pipeline stage's cache lives on the
    /// device that runs the stage. With dev0 == dev1 this is byte-for-byte `new`
    /// (the single-device plumbing gate). Sizing math is IDENTICAL either way.
    pub fn new_pp2(
        dev0: &dyn KvDev,
        dev1: &dyn KvDev,
        split: usize,
        cfg: &ModelConfig,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_inner(
            &|il| if il < split { dev0 } else { dev1 },
            cfg,
            None,
            max_ctx,
        )
    }

    /// M2 N-stage twin of `new_pp2`: `fence` is the stage map from `memra_engine::pp::
    /// pp_cuts` ([0, c1, .., n_trunk]); layer il allocates through the engine of the
    /// stage that runs it. Layers at/beyond the fence end (MTP/NextN blocks) allocate
    /// through the LAST stage. Sizing math is IDENTICAL to `new` — only the allocating
    /// device varies.
    pub fn new_ppn<'a>(
        devs: &[&'a dyn KvDev],
        fence: &[usize],
        cfg: &ModelConfig,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(
            devs.len() + 1,
            fence.len(),
            "ppn cache: devs vs fence mismatch"
        );
        let pick = |il: usize| -> &dyn KvDev {
            let s = match fence[1..fence.len() - 1].binary_search(&il) {
                Ok(k) => k + 1,
                Err(k) => k,
            };
            devs[s.min(devs.len() - 1)]
        };
        Self::new_inner(&pick, cfg, None, max_ctx)
    }

    pub fn new_ppn_planned<'a>(
        devs: &[&'a dyn KvDev],
        fence: &[usize],
        cfg: &ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(
            devs.len() + 1,
            fence.len(),
            "ppn cache: devs vs fence mismatch"
        );
        let pick = |il: usize| -> &dyn KvDev {
            let stage = match fence[1..fence.len() - 1].binary_search(&il) {
                Ok(index) => index + 1,
                Err(index) => index,
            };
            devs[stage.min(devs.len() - 1)]
        };
        Self::new_inner(&pick, cfg, Some(plan), max_ctx)
    }

    /// Shared allocation walk: `pick(il)` supplies the device that OWNS layer il's
    /// cache state (always the same device outside the pp2 door).
    fn new_inner<'a>(
        pick: &dyn Fn(usize) -> &'a dyn KvDev,
        cfg: &ModelConfig,
        plan: Option<&memra_gguf::model_plan::ModelPlan>,
        max_ctx: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let fallback_plan = if plan.is_none() {
            Some(ModelPlan::compile(cfg)?)
        } else {
            None
        };
        let plan = plan
            .or(fallback_plan.as_ref())
            .expect("cache allocation requires a ModelPlan");
        let n = cfg.n_layer as usize;
        let mut kv = Vec::with_capacity(n);
        let mut recur = Vec::with_capacity(n);
        let head_dim_k = cfg.head_dim_k as usize;
        let head_dim_v = cfg.head_dim_v as usize;
        assert!(
            head_dim_k % 32 == 0 && head_dim_v % 32 == 0,
            "KVQUANT requires head_dim_k%32==0 && head_dim_v%32==0 (got k={head_dim_k} v={head_dim_v})"
        );
        for il in 0..cfg.n_layer {
            // stage-owned allocation (pp2): the device that runs this layer allocates it.
            let e = pick(il as usize);
            let layer = plan
                .layers
                .iter()
                .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
                .find(|layer| layer.index == il)
                .ok_or_else(|| format!("cache ModelPlan has no layer {il}"))?;
            // E4B KV-SHARING: the trailing shared_kv_layers have no k/v of their own — they
            // attend an earlier layer's cache (hybrid_forward resolves the target). No KvLayer
            // here: any accidental use is a loud unwrap at bring-up, and rewind/len loops
            // (iter_mut().flatten()) skip None naturally.
            let g4_shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
            if g4_shared > 0 && il >= cfg.n_layer - g4_shared {
                kv.push(None);
                recur.push(None);
                continue;
            }
            match layer.state {
                StatePlan::KvCache { .. } | StatePlan::SlidingKvCache { .. } => {
                    // Gemma per-layer geometry and every KV-format door are resolved by the same
                    // helper admission uses for its analytic byte coefficient.
                    let (kv_dim_k, kv_dim_v, kbb_l, vbb_l) =
                        full_attention_kv_layout(cfg, plan, il);
                    let k_tok_bytes = (kv_dim_k / 32) * kbb_l;
                    let v_tok_bytes = (kv_dim_v / 32) * vbb_l;
                    let planned_window = plan
                        .layers
                        .iter()
                        .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
                        .find(|layer| layer.index == il)
                        .and_then(|layer| match layer.state {
                            StatePlan::SlidingKvCache { window, .. } => Some(window),
                            _ => None,
                        });
                    let ring = if swa_ring_on() {
                        planned_window.map(|window| {
                            let window = window as usize;
                            KvRing::new(swa_ring_rows(window, max_ctx), window)
                        })
                    } else {
                        None
                    };
                    let alloc_rows = ring.as_ref().map(KvRing::rows).unwrap_or(max_ctx);
                    kv.push(Some(KvLayer {
                        // +8B tail pad: the v4 stage's aligned funnelshift window reads up to
                        // 4B past the final block (PR #3's finding, adopted pad-style — the
                        // expert-dot precedent; zero hot-loop branches, values discarded).
                        k: e.alloc_u8(kv_plane_allocation_bytes(alloc_rows, k_tok_bytes))?,
                        v: e.alloc_u8(kv_plane_allocation_bytes(alloc_rows, v_tok_bytes))?,
                        kv_dim_k,
                        kv_dim_v,
                        k_tok_bytes,
                        v_tok_bytes,
                        len: 0,
                        ring,
                        len_d: e.htod_i32(&[0])?,
                    }));
                    recur.push(None);
                }
                StatePlan::Recurrent {
                    conv_width,
                    conv_kernel,
                    state_width,
                } => {
                    kv.push(None);
                    recur.push(Some(RecurLayer {
                        conv_state: e.zeros(
                            conv_width as usize * (conv_kernel as usize).saturating_sub(1),
                        )?,
                        ssm_state: e.zeros(state_width as usize)?,
                        ssm_state_alt: e.zeros(state_width as usize)?,
                    }));
                }
                ref state => {
                    return Err(format!(
                        "native cache allocator has no implementation for layer {il} state {state:?}"
                    )
                    .into());
                }
            }
        }
        Ok(Cache {
            kv,
            recur,
            tp_kv: (0..n).map(|_| None).collect(),
            pos: 0,
            max_ctx,
            dflash_taps: None,
            last_logits_dev: None,
        })
    }

    pub fn has_swa_ring(&self) -> bool {
        self.kv.iter().flatten().any(|layer| layer.ring.is_some())
            || self
                .tp_kv
                .iter()
                .flatten()
                .any(|layer| layer.ring_window().is_some())
    }

    pub fn can_rollback(&self, snap: &CacheSnapshot, accept_len: usize) -> bool {
        let local = self
            .kv
            .iter()
            .zip(&snap.kv_len)
            .all(|(layer, saved)| match (layer, saved) {
                (Some(layer), Some(saved)) => layer
                    .ring
                    .as_ref()
                    .is_none_or(|ring| ring.can_rewind_to(saved + accept_len)),
                _ => true,
            });
        let tensor = self
            .tp_kv
            .iter()
            .zip(&snap.tp_kv_len)
            .all(|(layer, saved)| match (layer, saved) {
                (Some(layer), Some(saved)) => saved
                    .checked_add(accept_len)
                    .is_some_and(|target| layer.can_rewind_to(target)),
                _ => true,
            });
        local && tensor
    }

    /// Snapshot the dual cache before a spec-decode draft+verify round (MTP-PLAN §C/§D.4).
    /// Records each full-attn `len` (cheap) and makes a REAL device copy of each linear-attn
    /// conv_state/ssm_state (a fresh alloc + memcpy_dtod — NOT an Arc clone).
    pub fn snapshot(&self, e: &impl KvDev) -> Result<CacheSnapshot, Box<dyn std::error::Error>> {
        let n = self.kv.len();
        let mut kv_len = Vec::with_capacity(n);
        let mut tp_kv_len = Vec::with_capacity(n);
        let mut conv = Vec::with_capacity(n);
        let mut ssm = Vec::with_capacity(n);
        for il in 0..n {
            match &self.kv[il] {
                Some(kvl) => kv_len.push(Some(kvl.len)),
                None => kv_len.push(None),
            }
            tp_kv_len.push(
                self.tp_kv[il]
                    .as_ref()
                    .map(ResidentTpKvCache::committed_len),
            );
            match &self.recur[il] {
                Some(rl) => {
                    conv.push(Some(e.clone_dtod(&rl.conv_state)?));
                    ssm.push(Some(e.clone_dtod(&rl.ssm_state)?));
                }
                None => {
                    conv.push(None);
                    ssm.push(None);
                }
            }
        }
        Ok(CacheSnapshot {
            kv_len,
            tp_kv_len,
            conv,
            ssm,
            pos: self.pos,
        })
    }

    /// PERSISTENT-BUFFER snapshot (spec-decode hot loop): refresh `snap` IN PLACE — same values as
    /// `snapshot()` but the conv/ssm device buffers are reused across rounds (D2D copy-into, ZERO
    /// allocations vs 2 fresh clones per linear layer per round). `snap` must come from a prior
    /// `snapshot()` of THIS cache (same layer shapes).
    pub fn snapshot_into(
        &self,
        e: &impl KvDev,
        snap: &mut CacheSnapshot,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.kv.len();
        for il in 0..n {
            snap.kv_len[il] = self.kv[il].as_ref().map(|kvl| kvl.len);
            snap.tp_kv_len[il] = self.tp_kv[il]
                .as_ref()
                .map(ResidentTpKvCache::committed_len);
            if let Some(rl) = &self.recur[il] {
                let dc = snap.conv[il]
                    .as_mut()
                    .expect("snapshot_into: shape mismatch (conv)");
                let ds = snap.ssm[il]
                    .as_mut()
                    .expect("snapshot_into: shape mismatch (ssm)");
                let (cn, sn) = (rl.conv_state.len(), rl.ssm_state.len());
                e.copy_into(dc, 0, &rl.conv_state, cn)?;
                e.copy_into(ds, 0, &rl.ssm_state, sn)?;
            }
        }
        snap.pos = self.pos;
        Ok(())
    }

    /// Roll the cache back to exactly `snap.pos + accept_len` committed tokens (MTP-PLAN §C).
    /// - Full-attn KV (C.1): set len = snapshot_len + accept_len (truncate, no copy).
    /// - Linear-attn (C.2): RESTORE the snapshot conv/ssm (real D2D copy back into the resident
    ///   buffers). The caller must then REPLAY the `accept_len` committed tokens through the full
    ///   T=1 decode path to rebuild the recurrent state for those positions. We restore (not
    ///   replay here) because replay needs the model; this only resets state to the pre-round value.
    /// `cache.pos` is set to `snap.pos` so the caller's replay advances it back to the commit point.
    pub fn rollback(
        &mut self,
        e: &impl KvDev,
        snap: &CacheSnapshot,
        accept_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.can_rollback(snap, accept_len) {
            return Err(
                "SWA ring rewind checkpoint has been lapped; full re-prime required".into(),
            );
        }
        for il in 0..self.kv.len() {
            if let (Some(kvl), Some(saved)) = (self.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + accept_len;
                // keep the device mirror in lock-step (CUDA-GRAPH-PLAN Phase 2). Set IN PLACE
                // (stable pointer): a fresh htod_i32 would reallocate len_d, but its old pointer is
                // baked into the captured decode graph's append/inc/fa_decode kernels — replacing it
                // strands the graph on a freed buffer (stale-pointer hazard). memcpy_htod in place.
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            if let (Some(kvl), Some(saved)) = (self.tp_kv[il].as_mut(), snap.tp_kv_len[il]) {
                kvl.rewind_to(saved + accept_len)?;
            }
            if let Some(rl) = self.recur[il].as_mut() {
                if let Some(c) = &snap.conv[il] {
                    e.copy_into(&mut rl.conv_state, 0, c, c.len())?;
                }
                if let Some(s) = &snap.ssm[il] {
                    e.copy_into(&mut rl.ssm_state, 0, s, s.len())?;
                }
            }
        }
        self.pos = snap.pos;
        Ok(())
    }
}

#[cfg(test)]
mod tp_transaction_tests {
    use super::{
        Cache, KvRingAppend, ResidentTpKvCache, TpKvTransactionState, tp_kv_rank_allocation_shape,
    };

    fn empty_tp_cache(capacity: usize) -> ResidentTpKvCache {
        ResidentTpKvCache::new(Vec::new(), 128, 128, 136, 96, capacity)
    }

    #[test]
    fn step_tp8_rank_allocation_matches_the_official_kv_geometry() {
        let shape = tp_kv_rank_allocation_shape(8 * 128, 8 * 128, 8).unwrap();
        assert_eq!((shape.kv_dim_k, shape.kv_dim_v), (128, 128));
        assert_eq!((shape.k_token_bytes, shape.v_token_bytes), (136, 96));
        assert_eq!(shape.bytes_per_token(), 232);
        assert_eq!(shape.fixed_bytes, 20);
        assert_eq!(shape.allocation_bytes(262_144), 232 * 262_144 + 20);
    }

    #[test]
    fn tp_rank_allocation_refuses_non_divisible_and_non_block_aligned_shards() {
        assert!(tp_kv_rank_allocation_shape(1024, 1024, 3).is_err());
        assert!(tp_kv_rank_allocation_shape(1024, 1024, 64).is_err());
        assert!(tp_kv_rank_allocation_shape(0, 1024, 8).is_err());
    }

    #[test]
    fn partial_commit_publishes_only_the_accepted_prefix() {
        let mut state = TpKvTransactionState::new();
        let transaction = state.begin().unwrap();
        let staged = state.append_target(transaction, 3, 8).unwrap();
        state.publish_append(transaction, staged).unwrap();
        assert_eq!(state.committed_len, 0);
        assert_eq!(state.staged_len, 3);

        let committed = state.commit_target(transaction, 2).unwrap();
        state.publish_finalize(transaction, committed).unwrap();
        assert_eq!(state.committed_len, 2);
        assert_eq!(state.staged_len, 2);
        assert!(state.active.is_none());
        assert!(state.validate(transaction).is_err());
    }

    #[test]
    fn rollback_restores_the_committed_boundary() {
        let mut state = TpKvTransactionState::new();
        let first = state.begin().unwrap();
        let staged = state.append_target(first, 1, 8).unwrap();
        state.publish_append(first, staged).unwrap();
        let committed = state.commit_target(first, 1).unwrap();
        state.publish_finalize(first, committed).unwrap();

        let speculative = state.begin().unwrap();
        let staged = state.append_target(speculative, 2, 8).unwrap();
        state.publish_append(speculative, staged).unwrap();
        assert_eq!(state.committed_len, 1);
        assert_eq!(state.staged_len, 3);
        state
            .publish_finalize(speculative, speculative.base_len)
            .unwrap();
        assert_eq!(state.committed_len, 1);
        assert_eq!(state.staged_len, 1);
        assert!(state.validate(speculative).is_err());
    }

    #[test]
    fn rejects_nested_stale_and_out_of_range_actions() {
        let mut state = TpKvTransactionState::new();
        let transaction = state.begin().unwrap();
        assert!(state.begin().is_err());
        assert!(state.append_target(transaction, 0, 2).is_err());
        assert!(state.append_target(transaction, 3, 2).is_err());
        let staged = state.append_target(transaction, 2, 2).unwrap();
        state.publish_append(transaction, staged).unwrap();
        assert!(state.commit_target(transaction, 3).is_err());
        state.publish_finalize(transaction, 0).unwrap();
        assert!(state.publish_append(transaction, 1).is_err());
    }

    #[test]
    fn rewind_resets_visibility_and_invalidates_an_active_transaction() {
        let mut state = TpKvTransactionState::new();
        let transaction = state.begin().unwrap();
        let staged = state.append_target(transaction, 3, 8).unwrap();
        state.publish_append(transaction, staged).unwrap();
        state.rewind(1, 8).unwrap();
        assert_eq!(state.committed_len, 1);
        assert_eq!(state.staged_len, 1);
        assert!(state.active.is_none());
        assert!(state.validate(transaction).is_err());
        assert!(state.rewind(9, 8).is_err());
    }

    #[test]
    fn grow_preserves_generation_and_publishes_only_the_checkpoint_prefix() {
        let mut source = empty_tp_cache(8);
        let first = source.begin_transaction().unwrap();
        let staged = source.append_target(first, 5).unwrap();
        source.publish_append(first, staged).unwrap();
        let committed = source.commit_target(first, 5).unwrap();
        source.publish_finalize(first, committed).unwrap();

        let rolled_back = source.begin_transaction().unwrap();
        source
            .publish_finalize(rolled_back, rolled_back.base_len())
            .unwrap();
        let plan = source.prepare_grow(16, 3).unwrap();
        assert_eq!(plan.rows(), 3);
        assert_eq!(plan.k_bytes(), 3 * 136);
        assert_eq!(plan.v_bytes(), 3 * 96);

        let mut target = empty_tp_cache(16);
        target.publish_grow(plan).unwrap();
        assert_eq!(target.committed_len(), 3);
        assert_eq!(target.staged_len(), 3);
        assert_eq!(target.capacity(), 16);
        let next = target.begin_transaction().unwrap();
        assert_eq!(next.generation(), rolled_back.generation() + 1);
        assert_eq!(next.base_len(), 3);
    }

    #[test]
    fn grow_refuses_active_source_and_invalid_target_state_or_layout() {
        let mut active = empty_tp_cache(8);
        active.begin_transaction().unwrap();
        assert!(active.prepare_grow(16, 0).is_err());

        let mut source = empty_tp_cache(8);
        source.rewind_to(5).unwrap();
        assert!(source.prepare_grow(8, 5).is_err());
        assert!(source.prepare_grow(16, 6).is_err());
        let plan = source.prepare_grow(16, 4).unwrap();

        let mut wrong_layout = ResidentTpKvCache::new(Vec::new(), 128, 128, 144, 96, 16);
        assert!(wrong_layout.publish_grow(plan).is_err());

        let plan = source.prepare_grow(16, 4).unwrap();
        let mut dirty_target = empty_tp_cache(16);
        dirty_target.rewind_to(1).unwrap();
        assert!(dirty_target.publish_grow(plan).is_err());
    }

    #[test]
    fn swa_transaction_rebase_preserves_the_rollback_window() {
        let mut cache = ResidentTpKvCache::new_swa(Vec::new(), 128, 128, 136, 96, 10_000, 32);
        assert_eq!(cache.physical_capacity(), 32 + 4096 + 31);
        cache.publish_hydration(8250, 4096).unwrap();
        assert_eq!(cache.ring_base(), Some(4096));

        let transaction = cache.begin_transaction().unwrap();
        let plan = cache.prepare_append(transaction, 10).unwrap();
        assert_eq!(plan.target(), 8260);
        assert_eq!(plan.write_row(), 58);
        assert_eq!(
            plan.ring_append(),
            Some(KvRingAppend::Rebase {
                src_row: 4096,
                keep_rows: 58,
                new_base: 8192,
                write_row: 58,
            })
        );
        cache.publish_append_rebase(plan).unwrap();
        cache.publish_append_plan(plan).unwrap();
        assert_eq!(cache.ring_base(), Some(8192));
        assert_eq!(cache.physical_range(8228, 8260).unwrap(), 36..68);

        let rollback = cache.commit_target(transaction, 0).unwrap();
        cache.publish_finalize(transaction, rollback).unwrap();
        assert_eq!((cache.committed_len(), cache.staged_len()), (8250, 8250));
        assert!(cache.rewind_to(8200).is_err());
    }

    #[test]
    fn swa_grow_normalizes_only_the_live_prefix_and_preserves_generation() {
        let mut source = ResidentTpKvCache::new_swa(Vec::new(), 128, 128, 136, 96, 10_000, 32);
        source.publish_hydration(8250, 4096).unwrap();
        let transaction = source.begin_transaction().unwrap();
        source
            .publish_finalize(transaction, transaction.base_len())
            .unwrap();

        let plan = source.prepare_grow(20_000, 8250).unwrap();
        assert_eq!(plan.source_row(), 4096);
        assert_eq!(plan.copy_rows(), 58);
        assert_eq!(plan.k_bytes(), 58 * 136);
        assert_eq!(plan.v_bytes(), 58 * 96);

        let mut target = ResidentTpKvCache::new_swa(Vec::new(), 128, 128, 136, 96, 20_000, 32);
        target.publish_grow(plan).unwrap();
        assert_eq!(target.ring_base(), Some(8192));
        assert_eq!((target.committed_len(), target.staged_len()), (8250, 8250));
        assert_eq!(target.physical_range(8192, 8250).unwrap(), 0..58);
        let next = target.begin_transaction().unwrap();
        assert_eq!(next.generation(), transaction.generation() + 1);
    }

    #[test]
    fn cache_reports_a_materialized_distributed_swa_ring() {
        let mut cache = Cache {
            kv: Vec::new(),
            recur: Vec::new(),
            tp_kv: vec![None],
            pos: 0,
            max_ctx: 10_000,
            dflash_taps: None,
            last_logits_dev: None,
        };
        assert!(!cache.has_swa_ring());
        cache.tp_kv[0] = Some(ResidentTpKvCache::new_swa(
            Vec::new(),
            128,
            128,
            136,
            96,
            10_000,
            512,
        ));
        assert!(cache.has_swa_ring());
    }
}

#[cfg(test)]
mod swa_ring_tests {
    use super::{KvRing, KvRingAppend, kv_plane_allocation_bytes, swa_ring_rows};

    #[test]
    fn allocation_rows_cover_window_max_prime_and_alignment_slack() {
        assert_eq!(swa_ring_rows(512, 262_144), 512 + 4096 + 31);
        assert_eq!(swa_ring_rows(512, 4096), 4096);
        assert_eq!(
            kv_plane_allocation_bytes(4639, 1088),
            4639 * 1088 + 8,
            "the Step35 session plane allocates ring rows plus the existing tail pad",
        );
    }

    #[test]
    fn ring_matches_flat_bytes_before_wrap() {
        let ring = KvRing::new(swa_ring_rows(512, 262_144), 512);
        let flat: Vec<u32> = (0..1024).collect();
        let mut physical = vec![u32::MAX; ring.rows()];
        let KvRingAppend::Contiguous { write_row } = ring.append_plan(0, 0, flat.len()).unwrap()
        else {
            panic!("first append unexpectedly wrapped")
        };
        physical[write_row..write_row + flat.len()].copy_from_slice(&flat);
        let view = ring.physical_range(0, flat.len()).unwrap();
        assert_eq!(&physical[view], flat.as_slice());
    }

    #[test]
    fn wrap_rebases_the_exact_aligned_prime_view() {
        let mut ring = KvRing::new(swa_ring_rows(512, 262_144), 512);
        let flat: Vec<u32> = (0..8192).collect();
        let mut physical = vec![u32::MAX; ring.rows()];
        let KvRingAppend::Contiguous { write_row } = ring.append_plan(0, 0, 4096).unwrap() else {
            panic!("first prime chunk unexpectedly wrapped")
        };
        physical[write_row..write_row + 4096].copy_from_slice(&flat[..4096]);

        let off = (4096usize - (512 - 1)) & !31usize;
        let KvRingAppend::Rebase {
            src_row,
            keep_rows,
            new_base,
            write_row,
        } = ring.append_plan(4096, off, 4096).unwrap()
        else {
            panic!("second prime chunk did not wrap")
        };
        let retained = physical[src_row..src_row + keep_rows].to_vec();
        physical[..keep_rows].copy_from_slice(&retained);
        ring.apply_rebase(new_base);
        physical[write_row..write_row + 4096].copy_from_slice(&flat[4096..8192]);

        let view = ring.physical_range(off, 8192).unwrap();
        assert_eq!(&physical[view], &flat[off..8192]);
        assert_eq!(ring.base(), off);
    }

    #[test]
    fn rewind_declines_once_the_required_window_was_lapped() {
        let mut ring = KvRing::new(swa_ring_rows(512, 262_144), 512);
        let KvRingAppend::Rebase { new_base, .. } = ring.append_plan(4096, 3584, 4096).unwrap()
        else {
            panic!("expected wrap")
        };
        ring.apply_rebase(new_base);
        assert!(ring.can_rewind_to(4095));
        assert!(!ring.can_rewind_to(4094));
        assert!(!ring.can_rewind_to(0));
    }
}
