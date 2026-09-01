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

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
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
    if !local_k.is_multiple_of(32) || !local_v.is_multiple_of(32) {
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

/// Step35 SWA ring. Default OFF unless the loader arms the step37 serving default (owner flip
/// 2026-08-27: the ring frees 16.4 GB on card0 at the natural 262144 context with identical
/// throughput and ids, and the W8 doors OOM there without it). Architecture-scoped by its call
/// sites: Gemma4's row-0-addressed window kernels cannot consume a rebased ring view, which is
/// why the default arms per loaded family rather than globally. `MEMRA_SWA_RING=1` forces ON,
/// `=0` is the kill switch either way.
static SWA_RING_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_swa_ring_default(on: bool) {
    SWA_RING_DEFAULT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn swa_ring_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    match *ENV.get_or_init(|| match std::env::var("MEMRA_SWA_RING").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }) {
        Some(forced) => forced,
        None => SWA_RING_DEFAULT.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// With the SWA-ring door open, `prime_chunk_tokens` caps every legal chunk at this bound. The
/// ring carries one whole maximum-size prime chunk in addition to the reader's window.
pub const PRIME_CHUNK_MAX_TOKENS: usize = 4096;
const SWA_VIEW_ALIGNMENT_ROWS: usize = 32;

/// Physical rows required by the Step35 SWA reader contract. Prime starts at
/// `(base_len - (window - 1)) & !31`, so at most 31 masked rows precede the live window.
pub fn swa_ring_rows(window: usize, max_ctx: usize) -> usize {
    // window + max prime chunk + REWIND HEADROOM + alignment. append_plan requires
    // keep_rows + append_rows <= rows, and keep_rows is now window + SWA_REWIND_SLACK_ROWS, so the
    // headroom has to be in `rows` or a full-size prime chunk stops fitting. Costs
    // SWA_REWIND_SLACK_ROWS rows per ring-backed plane.
    max_ctx.min(
        window + PRIME_CHUNK_MAX_TOKENS + SWA_REWIND_SLACK_ROWS + (SWA_VIEW_ALIGNMENT_ROWS - 1),
    )
}

/// Rows a ring-backed plane keeps BELOW the aligned window start so a backward rewind stays legal.
///
/// This is HEADROOM THE RING IS SIZED FOR, not slack scavenged from it. The original geometry
/// (window + prime chunk + 31) left exactly one alignment block spare once a full prime chunk had
/// to fit, and one block only covers a rewind shallower than 32 rows. Clamping a deeper request up
/// to `base` instead makes the append legal while leaving the attention window pointing below rows
/// the ring no longer holds — which produced all-NaN head logits and seed hiddens at pos 8661
/// rather than an error. A ring that cannot serve the rewinds its own callers perform is
/// undersized; the fix is to size it, not to keep redistributing 32 rows.
pub const SWA_REWIND_SLACK_ROWS: usize = 512;

/// The retain a ring-backed append must request: the aligned window start for `first_row`, minus
/// the rewind slack, but NEVER below what the ring still holds.
///
/// The clamp is the part that took three attempts to find. Asking for slack unconditionally makes
/// the REWIND legal and then breaks the very next APPEND: after a rewind, `first_row` moves back
/// while `base` does not, so the ideal retain falls under `base` and append_plan refuses it
/// ("SWA ring lapped required rows (base 4128, retain 4096, len 4669)"). Rows below `base` are
/// gone and, being older than the window, are not needed — so clamping up to `base` is both legal
/// and correct. Slack is an optimisation the ring grants when it can, never a demand.
pub fn swa_retain_from(first_row: usize, window: usize, base: usize) -> usize {
    let ideal = first_row
        .saturating_sub(window.saturating_sub(1))
        .saturating_sub(SWA_REWIND_SLACK_ROWS)
        & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
    ideal.max(base)
}

/// WORKING-SET rows the DSA indexer TAIL RING books by default.
///
/// IT IS NOT A BOUND ON ANYTHING, and that is the correction (lane/glm53-ring-sizing,
/// 2026-08-28). The ring first shipped sized as `prime_chunk_bound + 1024`, i.e. against the
/// largest `t` a CHUNKED prefill could hand one call. glm5_next, the one architecture this ring
/// exists for, is eager-only (`ResidualTopology::HyperConnections` refuses every batched and
/// speculative entry point) and primes MONOLITHICALLY: `prime_cache_hyper` never calls
/// `prime_chunk_ranges`, so its per-call `t` is the whole prompt and its only ceiling is the
/// admission limit itself. A ring that bounds `t` is therefore a ring of `max_ctx` rows, which is
/// the flat plane, which is no ring at all. The bench box measured what the wrong bound cost:
/// 4630 usable prompt tokens inside a configured `MEMRA_CTX=8192`, against 7300 with the ring off
/// on the same binary (research/glm53-flash-bringup-20260827/rebaseline-and-surface-20260828,
/// receipts 13 and 14).
///
/// So `t` was removed from the requirement instead of the constant being raised.
/// `mla_kpool_indices` DRAINS the ring inside the call: it appends what fits, builds the pool keys
/// that frees, and continues, so a call of any `t` is served by a ring of any size. The only
/// correctness floor left is ONE POOL, enforced by the engine because the state plan does not
/// carry `pool`; everything above it is working set, and the ring can never again be the reason a
/// prompt is refused.
///
/// 5120 is chosen, not derived. It is what the flag already books, so every banked memory number
/// stays exactly true (1M: 13.5 GiB to 1.56 GiB over 12 MLA layers), it is one nominal 4096-token
/// prime chunk plus slack so a CHUNKED architecture drains in exactly one iteration and pays zero
/// extra launches, and at 5 MiB per layer it is noise against the 1 GiB per layer the ring
/// deletes. A monolithic 1M prime drains in about 205 iterations per MLA layer, two kernel
/// launches each, against a prefill of a million tokens.
pub const INDEX_RING_WORKING_ROWS: usize = 5120;

/// Physical rows of the DSA k-pool indexer state plane when it is a TAIL RING, or `None` to keep
/// the flat `max_ctx`-row plane. PURE: the env read is [`index_ring_rows`].
///
/// `explicit` is a parsed `MEMRA_DSA_INDEX_RING`: `Some(0)` disables the ring, `Some(n)` pins the
/// row budget (gates use a tiny one to reach the wrap in a micro fixture), `None` books the
/// working-set default. There is deliberately NO per-call `t` input any more: see
/// [`INDEX_RING_WORKING_ROWS`]. A ring that is not SHORTER than the flat plane is pointless, so
/// `rows >= max_ctx` collapses to `None` and the short-context sessions that dominate the test
/// suite keep byte-for-byte their old allocation.
pub fn index_ring_rows_for(explicit: Option<usize>, max_ctx: usize) -> Option<usize> {
    let rows = match explicit {
        Some(0) => return None,
        Some(n) => n,
        None => INDEX_RING_WORKING_ROWS,
    };
    (rows > 0 && rows < max_ctx).then_some(rows)
}

/// PHYSICAL rows the SHIPPED DEFAULT derivation books for `max_ctx`: no `MEMRA_DSA_INDEX_RING`
/// override. Pure, so the sizing gate can assert on it without racing another test's environment.
/// This is the one function the sizing gate calls.
pub fn index_ring_default_rows(max_ctx: usize) -> Option<usize> {
    index_ring_rows_for(None, max_ctx)
}

/// Rows of `remaining` the indexer may append to the tail ring BEFORE the pool-key build has to
/// drain it, or `None` when the rows this call still owes an unbuilt pool are already lapped.
///
/// `ring` is the EFFECTIVE ring (a multiple of `pool`; `0` is the flat plane), `pools_ready` the
/// pools whose keys are already resident, `cur` the absolute row the next append lands on, and
/// `remaining` the rows of this call not yet appended.
///
/// THE WHOLE SAFETY ARGUMENT, in one window. The plane has exactly one writer
/// (`Engine::mla_index_append`) and exactly one reader (`Engine::mla_kpool_pool_keys`), and a row
/// is read exactly once, by the pool-key build of the pool it belongs to. So the rows that must be
/// live at any instant are `[pools_ready * pool, cur + take)`: everything below has already been
/// read and is dead, everything above is not written yet. `take` is whatever is left of the ring
/// after the carry-over `live = cur - pools_ready * pool`, and the caller drains and comes back.
///
/// PROGRESS is guaranteed by `ring >= pool`, which the engine enforces separately, because a build
/// leaves `live = cur mod pool < pool` behind. The one input that can still make this `None` is a
/// `pools_ready` that sits further than `ring` below `cur`: a rewind that reduced the cache without
/// clamping `index_pools_ready`, or a pool-key plane reallocation. Those rows are genuinely gone
/// and no amount of draining brings them back, so it refuses.
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
pub fn index_ring_take(
    ring: usize,
    pool: usize,
    pools_ready: usize,
    cur: usize,
    remaining: usize,
) -> Option<usize> {
    if ring == 0 {
        return Some(remaining);
    }
    debug_assert!(
        ring % pool == 0,
        "the effective ring is a whole number of pools"
    );
    let live = cur.checked_sub(pools_ready.saturating_mul(pool))?;
    (ring > live).then(|| (ring - live).min(remaining))
}

/// DSA k-pool indexer TAIL RING sizing (`MEMRA_DSA_INDEX_RING`, default ON, see docs/FLAGS.md).
/// Unparseable values are treated as unset. `MEMRA_PRIME_CHUNK` is NO LONGER READ HERE: the ring
/// is drained inside the call, so no prefill chunk discipline can size it or break it.
pub fn index_ring_rows(max_ctx: usize) -> Option<usize> {
    let explicit = std::env::var("MEMRA_DSA_INDEX_RING")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    index_ring_rows_for(explicit, max_ctx)
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

    /// Plan a checkpoint restore into a FRESH ring (base 0): the aligned live window ending at
    /// absolute `len`, as (new_base, source physical rows). `len` is the ABSOLUTE stream length
    /// the checkpoint recorded — for a lapped ring it exceeds the physical row count, so a
    /// restore that copies `len` rows from row zero is an out-of-bounds device slice (the
    /// 2026-08-29 warm-turn-at-40k GPU-worker panic). Refuses when this ring no longer holds
    /// the window (checkpoint lapped: the caller must full re-prime).
    pub fn restore_plan(&self, len: usize) -> Result<(usize, std::ops::Range<usize>), String> {
        let raw = len.saturating_sub(self.window.saturating_sub(1));
        let new_base = raw & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
        let physical = self.physical_range(new_base, len)?;
        Ok((new_base, physical))
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
    /// D2D copy with an offset on BOTH sides. `copy_into` always reads the source from 0, which
    /// cannot express "copy this window OUT of a tail ring" — the shape the latent-plane
    /// snapshot/restore needs (lane/glm5-prefix-latent, 2026-08-30).
    fn copy_range_into(
        &self,
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
        src: &CudaSlice<f32>,
        src_off: usize,
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
    /// Device-resident mirror of `ring.base()` (physical row of logical row 0) for the WINDOWED
    /// device-counter draft arm (`append_kv_quantized_dcw` / `fa_decode_dcw`): the kernels derive
    /// the SWA view as {lstart = max(0, len - window); physical = row - base} entirely from
    /// device state, so a captured draft chain replays with zero per-token node updates. Armed
    /// only on ring-backed draft-scratch planes (step35); `None` keeps the plain `_dc` contract
    /// (base 0) and costs nothing. The ONE writer is the rebase arm of `prepare_kv_append`
    /// (rebases are host-side, outside any captured region); rewinds move `len`/`len_d` only,
    /// never `base`, so no other site touches it. i32[1].
    pub base_d: Option<CudaSlice<i32>>,
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

/// Per-MLA-layer latent KV plane (DESIGN.md §3.2). ONE row per token, `width` elements wide,
/// where `width` == `StatePlan::LatentKvCache { width }` == kv_lora_rank + rope_head_dim:
///   row = [ rmsnorm(c_kv) : kv_rank | rope(k_pe) : d_rope ]
/// There is NO V plane — V is the FIRST `kv_rank` elements of the SAME row, and every query
/// head streams that one row (MQA). NoPE models (glm5_next, rope_head_dim 0) have width ==
/// kv_rank and no k_pe tail.
///
/// f32, UNQUANTIZED, deliberately: increment 4 is the correctness arm and its gate is maxdiff
/// against the `memra_engine::mla` f32 oracle, whose `c_kv` is f32. DESIGN.md §3.2's eventual
/// q8_0 latent row (576 = 18 blocks, V view boundary 512 = 16 blocks, both on a 32-element
/// boundary) is a later increment; quantizing here would fork the plane from the oracle it is
/// gated against. The `% 32 == 0` KVQUANT constraint therefore does NOT apply to this plane.
pub struct LatentKvLayer {
    /// [max_ctx * width] f32, row-major by token.
    pub rows: CudaSlice<f32>,
    pub width: usize,
    pub len: usize,
    /// Device mirror of `len`, kept in lock-step exactly like `KvLayer::len_d`.
    pub len_d: CudaSlice<i32>,
    /// DSA k-pool indexer state, [max_ctx * index_width] f32 row-major by token:
    ///   row = [ k_norm(wk(x)) : index_head_dim | index_kpool_compress_gate(x) : index_head_dim ]
    /// `None` when the layer declares `index_width == 0` (no k-pool indexer). The reference's
    /// `past_key_values.update_indexer` carries the same two channels; its third (a per-token
    /// validity flag) is DELIBERATELY absent — this cache is single-sequence and unpadded, so
    /// every row below `len` is valid and pooling starts at token 0, which is exactly the scope
    /// `memra_reference::kpool_allowed_tokens` documents for itself. A batched/padded arm needs
    /// that channel back and its own gate.
    ///
    /// `len` above is authoritative for BOTH planes: they are appended in the same call and
    /// must never carry independent lengths.
    ///
    /// MEMORY — the TAIL RING, and it SHIPPED (`index_ring_rows`, `MEMRA_DSA_INDEX_RING`).
    /// Flat, this plane is `2 * index_head_dim` = 256 f32 = 1 KiB per token per layer, i.e.
    /// **12 GiB (12.88 GB)** over glm5_next's 12 MLA layers at 1M — larger than the latent
    /// plane's share of the same budget is comfortable with. Two reductions were considered:
    ///   * **f16/bf16 rows — DECLINED.** The rows feed the pool-key softmax, whose output feeds
    ///     the ReLU score, whose ties the selection order depends on. Halving the mantissa moves
    ///     scores, and moved scores move which pools win a tie — the one thing the gates forbid.
    ///     It would need its own selection-parity gate at serving scale before it could ship, and
    ///     it buys 6 GiB where the option below buys 11.94.
    ///   * **A tail ring — the real answer, and the one implemented.** With `index_pool_keys`
    ///     resident (below), a row of this plane is read exactly once: by the pool-key build of
    ///     the pool it belongs to. Every row under `index_pools_ready * pool` is therefore
    ///     PROVABLY DEAD — this cache has exactly ONE in-call reader of the plane
    ///     (`Engine::mla_kpool_pool_keys`) and ONE writer (`Engine::mla_index_append`), and
    ///     `CacheSnapshot` does not carry latent planes at all, so nothing else can observe a
    ///     lapped row. (`snapshot_plane`, lane/glm5-prefix-latent, is a second reader BETWEEN
    ///     calls, and it reads only the LIVE tail window `[index_pools_ready * pool, len)` —
    ///     the liveness argument is unchanged.) The plane only has to hold the incomplete tail
    ///     plus whatever slice of the current call is in flight, so a ring of `R` rows with `R`
    ///     a multiple of `pool` (which keeps each pool contiguous mod `R`) replaces 12 GiB with
    ///     60 MiB, EXACTLY: same rows, same kernel, different addresses, zero numeric cost,
    ///     gated by `gpu_kpool_tail_ring_wraps_and_matches_the_flat_plane`.
    ///     Net before: 12 GiB here + 1.5 GiB of pool keys. Net after: 1.56 GiB, an 8.7x cut,
    ///     because a pool key is `index_head_dim` f32 per `pool` tokens = 32 f32/token against 256.
    ///
    /// `R` DOES NOT BOUND THE PER-CALL `t`, and getting that wrong is what shipped a regression
    /// (lane/glm53-ring-sizing, 2026-08-28). The first cut sized `R` against the largest `t` a
    /// CHUNKED prefill could hand one call; glm5_next primes MONOLITHICALLY, so its `t` is the
    /// whole prompt and the guard refused every prompt past `R`: 4630 usable tokens inside a
    /// configured 8192. `mla_kpool_indices` now DRAINS the ring inside the call, appending what
    /// fits and building the pool keys that frees, so `R` is a working-set choice
    /// ([`INDEX_RING_WORKING_ROWS`]) with a floor of one pool and nothing else.
    ///
    /// The indexer's `pool` is the one input the state plan does NOT carry, the same gap that
    /// makes `index_pool_keys` a lazy allocation below. So `pool` is not used to size the ring:
    /// the allocator books `index_ring_rows` PHYSICAL rows and the engine rounds that DOWN to a
    /// multiple of `pool` on first use, so the effective ring is always `>= rows - pool + 1`.
    pub index_rows: Option<CudaSlice<f32>>,
    pub index_width: usize,
    /// PHYSICAL rows of `index_rows` when it is a tail ring; `None` when the plane is flat
    /// (`max_ctx` rows, absolute row addressing). The EFFECTIVE ring is this rounded down to a
    /// multiple of the indexer's `pool`, computed by the engine — see the field doc above.
    pub index_ring_rows: Option<usize>,
    /// RESIDENT DSA pool-key plane, `[max_ctx / pool * index_head_dim]` f32 row-major by pool.
    ///
    /// LAYOUT: `index_pool_keys[p * d + c]` is channel `c` of the collapsed key of pool `p`, i.e.
    /// of cache rows `[p * pool, (p + 1) * pool)`. `d` is `index_width / 2` (the indexer's head
    /// dim); `pool` comes from the layer's `MlaIndexerGeom` and is NOT in the state plan, which is
    /// why this buffer is allocated on FIRST USE by the engine rather than by the allocator below.
    ///
    /// INVALIDATION RULE, and it is the whole point: a pool's key is a function of exactly its own
    /// `pool` rows of `index_rows` plus the layer's constant `kpool_ape`. `index_rows` is
    /// APPEND-ONLY — a row is written once, when its token is appended, and never rewritten — so
    /// once a pool's LAST row lands the key is FINAL and is never recomputed. `index_pools_ready`
    /// is how many leading pools hold such final keys; each call builds only
    /// `[index_pools_ready, len / pool)` and then advances it. The incomplete tail is NOT a pool
    /// and has no key: rows `[len / pool * pool, len)` reach the query through the selection
    /// kernel's `always_tail` append, recomputed every call.
    ///
    /// The rule therefore has exactly ONE trigger: if `len` ever DECREASES (a rewind that
    /// overwrites already-pooled rows), `index_pools_ready` must be clamped to `len / pool` by the
    /// same code that shortens `len`. Use `truncate_index_pool_keys` for that. Today `len` is
    /// written in two places (`HybridModel::mla_attn_cached`, and `restore_plane` on a FRESH
    /// layer) and only ever grows — `Cache::rollback` does not touch the latent planes at all —
    /// so no caller needs the clamp yet; `mla_kpool_indices` asserts the invariant on every call
    /// so a future rewind that forgets it fails loudly instead of selecting against stale keys,
    /// and `snapshot_plane`/`validate_restore` assert it at both prefix-cache seams.
    pub index_pool_keys: Option<CudaSlice<f32>>,
    /// Pools `[0, index_pools_ready)` of `index_pool_keys` hold FINAL keys.
    pub index_pools_ready: usize,
    /// RESIDENT copy of the indexer's `pool` (tokens per k-pool), the one geometry input the
    /// state plan does NOT carry (the same gap that makes `index_pool_keys` a lazy allocation).
    /// `0` until the engine's first indexer call writes it (`mla_attn_cached`, which refuses a
    /// nonzero value that disagrees with the loaded geometry rather than overwriting it). The
    /// latent-plane snapshot/restore path (lane/glm5-prefix-latent, 2026-08-30) reads it to
    /// address the tail ring and size the restored key plane; it refuses to capture a plane
    /// whose pool is still unknown.
    pub index_pool: usize,
}

impl LatentKvLayer {
    /// Shorten the resident pool-key plane to what `len` still justifies. Call from any path that
    /// REDUCES `len`; pools at or above `len / pool` may have been built over rows the rewind is
    /// about to overwrite, so their keys are no longer final.
    pub fn truncate_index_pool_keys(&mut self, pool: usize) {
        if pool == 0 {
            return;
        }
        self.index_pools_ready = self.index_pools_ready.min(self.len / pool);
    }
}

/// Physical row of absolute row `abs` in an indexer state plane. `ring_rows == 0` is the flat
/// plane (absolute addressing); otherwise the EFFECTIVE ring is `ring_rows` rounded down to a
/// whole number of pools, exactly the engine's own rounding (`mla_kpool_indices`), because a
/// ring that is not a multiple of `pool` would split a pool across the wrap.
pub fn index_plane_physical_row(ring_rows: usize, pool: usize, abs: usize) -> usize {
    if ring_rows == 0 {
        return abs;
    }
    debug_assert!(pool > 0, "index plane addressing requires a known pool");
    let effective = ring_rows / pool * pool;
    debug_assert!(effective > 0, "the effective ring holds at least one pool");
    abs % effective
}

/// One MLA/DSA layer's captured latent-plane state: everything `mla_attn_cached` +
/// `mla_kpool_indices` need to continue as if the destination session had primed the prefix
/// itself (lane/glm5-prefix-latent, 2026-08-30; design in
/// research/glm5-prefix-latent-20260830/DESIGN.md).
///
/// The three asymmetries against an ordinary `PrefixPlane`, and how each is carried:
///   * `rows` is deliberately UNQUANTIZED f32 (the maxdiff oracle depends on the f32 plane), so
///     the copy is f32-for-f32 — no quantization program is introduced at the snapshot seam.
///   * `index_rows` is a TAIL RING whose rows below `index_pools_ready * pool` are OVERWRITTEN
///     by design, so "the index plane" is not copyable and not rebuildable: the snapshot carries
///     the DERIVED keys (final by the append-only invariant, bit-identical to a rebuild) plus
///     the `len % pool` still-live tail rows (`index_tail`, at most `pool - 1` rows).
///   * `index_pool_keys` / `index_pools_ready` carry the append-only finality invariant, so the
///     capture asserts `index_pools_ready == len / pool` (every call boundary leaves the drain
///     there) and the restore re-establishes both, keeping the engine's residency tripwire and
///     `index_ring_take` arithmetic blind to the fact that a restore happened.
pub struct LatentPlaneSnapshot {
    /// Rows `[0..len)` of the latent plane, `len * width` f32.
    pub rows: CudaSlice<f32>,
    pub width: usize,
    pub len: usize,
    /// `0` = the layer has no indexer state plane (and every `index_*` field below is empty).
    pub index_width: usize,
    /// The indexer's pool size at capture (`LatentKvLayer::index_pool`); `0` iff no indexer.
    pub index_pool: usize,
    /// The live tail-ring rows `[index_pools_ready * pool, len)`, `(len % pool) * index_width`
    /// f32; `None` when the boundary is pool-aligned.
    pub index_tail: Option<CudaSlice<f32>>,
    /// The FINAL pool keys `[0..index_pools_ready * d)`, d = `index_width / 2`; `None` when no
    /// pool has completed.
    pub index_pool_keys: Option<CudaSlice<f32>>,
    pub index_pools_ready: usize,
}

impl LatentPlaneSnapshot {
    /// Device bytes this snapshot holds, for the prefix cache's byte ledger. The defective
    /// pre-lane entry cost ZERO bytes per token; this is the honest bill.
    pub fn bytes(&self) -> usize {
        let tail = self.index_tail.as_ref().map_or(0, CudaSlice::len);
        let keys = self.index_pool_keys.as_ref().map_or(0, CudaSlice::len);
        (self.rows.len() + tail + keys) * std::mem::size_of::<f32>()
    }
}

/// The generation-destroyed slice of one latent layer's BOUNDARY state, captured EAGERLY at a
/// spec session's prompt boundary (lane/glm5-prefix-latent2, 2026-09-01) so a DEFERRED prefix
/// publication can be completed later against the live plane:
///   * the latent `rows` and the FINAL pool keys are append-only BELOW the boundary for the
///     session's lifetime (the glm5 verify rollback truncates back to the accepted length,
///     never below the prime boundary), so `snapshot_plane_at` slices them from the LIVE
///     layer at publish time — no eager copy of the big planes;
///   * the incomplete tail-ring rows are read-once and OVERWRITTEN by the very next pool
///     build, so they travel HERE or the boundary is unrecoverable by publish time (the KDA
///     conv/ssm half of the same problem rides the sibling `CacheSnapshot`).
pub struct LatentTailCapture {
    /// Boundary length (== capture pos) — the row count the deferred publisher slices.
    pub len: usize,
    /// Latent width at capture; the publish-time slice validates it against the live layer.
    pub width: usize,
    pub index_width: usize,
    /// The indexer's pool size at capture (`0` iff no indexer plane).
    pub index_pool: usize,
    /// `len / pool` at the boundary (the capture asserts the drain invariant, same as
    /// `snapshot_plane`).
    pub index_pools_ready: usize,
    /// The live tail-ring rows `[pools_ready * pool, len)` at the boundary,
    /// `(len % pool) * index_width` f32; `None` when the boundary is pool-aligned (or the
    /// layer has no indexer).
    pub index_tail: Option<CudaSlice<f32>>,
}

impl LatentTailCapture {
    /// Device bytes held eagerly (the tail only — the big planes are sliced at publish).
    pub fn bytes(&self) -> usize {
        self.index_tail.as_ref().map_or(0, CudaSlice::len) * std::mem::size_of::<f32>()
    }
}

impl LatentKvLayer {
    /// Deep-copy this layer's latent-plane state OUT of a live session cache. Stream-ordered on
    /// the implementor's worker stream, like every other prefix-capture copy. Errors instead of
    /// capturing anything a restore could not make whole:
    ///   * `len == 0` (the caller records an unexecuted layer as absent instead),
    ///   * an indexer plane whose `pool` was never resolved,
    ///   * `index_pools_ready != len / pool` — a capture off a drained call boundary would
    ///     publish keys that are behind or ahead of their rows (the finality invariant).
    pub fn snapshot_plane(
        &self,
        e: &impl KvDev,
    ) -> Result<LatentPlaneSnapshot, Box<dyn std::error::Error>> {
        let (len, width) = (self.len, self.width);
        if len == 0 {
            return Err("latent snapshot at len 0 (record the layer as absent instead)".into());
        }
        if self.rows.len() < len * width {
            return Err(format!(
                "latent plane holds {} f32 but len {len} x width {width} requires {}",
                self.rows.len(),
                len * width,
            )
            .into());
        }
        let mut rows = e.uninit(len * width)?;
        e.copy_range_into(&mut rows, 0, &self.rows, 0, len * width)?;
        if self.index_width == 0 {
            return Ok(LatentPlaneSnapshot {
                rows,
                width,
                len,
                index_width: 0,
                index_pool: 0,
                index_tail: None,
                index_pool_keys: None,
                index_pools_ready: 0,
            });
        }
        let pool = self.index_pool;
        if pool == 0 {
            return Err(format!(
                "latent snapshot: index plane (width {}) has an unresolved pool — no indexer \
                 call ran against this layer, so its derived state cannot be validated",
                self.index_width,
            )
            .into());
        }
        let d = self.index_width / 2;
        let pools_ready = self.index_pools_ready;
        if pools_ready != len / pool {
            return Err(format!(
                "latent snapshot: index_pools_ready {pools_ready} != len/pool {} (len {len}, \
                 pool {pool}); a capture must sit at a drained call boundary or its keys \
                 violate the append-only finality invariant",
                len / pool,
            )
            .into());
        }
        let index_pool_keys = if pools_ready > 0 {
            let src = self
                .index_pool_keys
                .as_ref()
                .ok_or("latent snapshot: pools are ready but the resident key plane is gone")?;
            if src.len() < pools_ready * d {
                return Err(format!(
                    "latent snapshot: resident key plane holds {} f32 but {pools_ready} pools \
                     x d {d} require {}",
                    src.len(),
                    pools_ready * d,
                )
                .into());
            }
            let mut keys = e.uninit(pools_ready * d)?;
            e.copy_range_into(&mut keys, 0, src, 0, pools_ready * d)?;
            Some(keys)
        } else {
            None
        };
        let tail_rows = len - pools_ready * pool;
        let index_tail = if tail_rows > 0 {
            let src = self
                .index_rows
                .as_ref()
                .ok_or("latent snapshot: index_width > 0 but the state plane is gone")?;
            let ring = self.index_ring_rows.unwrap_or(0);
            // The tail starts pool-aligned and is shorter than one pool, and the effective ring
            // is a whole number of pools, so the window is contiguous in ring and flat layouts.
            let phys = index_plane_physical_row(ring, pool, pools_ready * pool);
            let want = (phys + tail_rows) * self.index_width;
            if src.len() < want {
                return Err(format!(
                    "latent snapshot: index plane holds {} f32 but the live tail window \
                     requires {want}",
                    src.len(),
                )
                .into());
            }
            let mut tail = e.uninit(tail_rows * self.index_width)?;
            e.copy_range_into(
                &mut tail,
                0,
                src,
                phys * self.index_width,
                tail_rows * self.index_width,
            )?;
            Some(tail)
        } else {
            None
        };
        Ok(LatentPlaneSnapshot {
            rows,
            width,
            len,
            index_width: self.index_width,
            index_pool: pool,
            index_tail,
            index_pool_keys,
            index_pools_ready: pools_ready,
        })
    }

    /// EAGER half of the deferred boundary capture (doc on [`LatentTailCapture`]): copy out
    /// only what generation will destroy — the incomplete tail-ring rows — plus the boundary
    /// metadata the publish-time slice validates against. Same preconditions as
    /// `snapshot_plane` (len > 0, resolved pool, the pools-ready drain invariant); the big
    /// planes are NOT copied here.
    pub fn snapshot_tail(
        &self,
        e: &impl KvDev,
    ) -> Result<LatentTailCapture, Box<dyn std::error::Error>> {
        let (len, width) = (self.len, self.width);
        if len == 0 {
            return Err("latent tail capture at len 0 (record the layer as absent instead)".into());
        }
        if self.index_width == 0 {
            return Ok(LatentTailCapture {
                len,
                width,
                index_width: 0,
                index_pool: 0,
                index_pools_ready: 0,
                index_tail: None,
            });
        }
        let pool = self.index_pool;
        if pool == 0 {
            return Err(format!(
                "latent tail capture: index plane (width {}) has an unresolved pool — no \
                 indexer call ran against this layer, so its derived state cannot be validated",
                self.index_width,
            )
            .into());
        }
        let pools_ready = self.index_pools_ready;
        if pools_ready != len / pool {
            return Err(format!(
                "latent tail capture: index_pools_ready {pools_ready} != len/pool {} (len \
                 {len}, pool {pool}); a capture must sit at a drained call boundary",
                len / pool,
            )
            .into());
        }
        let tail_rows = len - pools_ready * pool;
        let index_tail = if tail_rows > 0 {
            let src = self
                .index_rows
                .as_ref()
                .ok_or("latent tail capture: index_width > 0 but the state plane is gone")?;
            let ring = self.index_ring_rows.unwrap_or(0);
            let phys = index_plane_physical_row(ring, pool, pools_ready * pool);
            let want = (phys + tail_rows) * self.index_width;
            if src.len() < want {
                return Err(format!(
                    "latent tail capture: index plane holds {} f32 but the live tail window \
                     requires {want}",
                    src.len(),
                )
                .into());
            }
            let mut tail = e.uninit(tail_rows * self.index_width)?;
            e.copy_range_into(
                &mut tail,
                0,
                src,
                phys * self.index_width,
                tail_rows * self.index_width,
            )?;
            Some(tail)
        } else {
            None
        };
        Ok(LatentTailCapture {
            len,
            width,
            index_width: self.index_width,
            index_pool: pool,
            index_pools_ready: pools_ready,
            index_tail,
        })
    }

    /// DEFERRED half of the boundary capture: complete a [`LatentPlaneSnapshot`] at the
    /// captured boundary by slicing the append-only planes (`rows` `[0..cap.len)`, FINAL pool
    /// keys `[0..cap.index_pools_ready * d)`) from the LIVE layer and moving the eagerly
    /// captured tail in. Every disagreement between the capture and the live layer refuses —
    /// a publication is an optimization and must never publish planes it cannot prove are the
    /// boundary's (the append-only-below-boundary invariant is what makes the slice legal:
    /// the glm5 verify rollback truncates to the accepted length, never below the prime
    /// boundary, and pool keys are final the instant their last row lands).
    pub fn snapshot_plane_at(
        &self,
        e: &impl KvDev,
        cap: LatentTailCapture,
    ) -> Result<LatentPlaneSnapshot, Box<dyn std::error::Error>> {
        let (len, width) = (cap.len, cap.width);
        if len == 0 {
            return Err("latent boundary publish at len 0".into());
        }
        if width != self.width {
            return Err(format!(
                "latent boundary publish: captured width {width} != live width {}",
                self.width,
            )
            .into());
        }
        if self.len < len {
            return Err(format!(
                "latent boundary publish: live len {} < boundary {len} — the plane was \
                 truncated below the capture boundary",
                self.len,
            )
            .into());
        }
        if self.rows.len() < len * width {
            return Err(format!(
                "latent boundary publish: live plane holds {} f32 but boundary {len} x width \
                 {width} requires {}",
                self.rows.len(),
                len * width,
            )
            .into());
        }
        let mut rows = e.uninit(len * width)?;
        e.copy_range_into(&mut rows, 0, &self.rows, 0, len * width)?;
        if cap.index_width != self.index_width {
            return Err(format!(
                "latent boundary publish: captured index_width {} != live {}",
                cap.index_width, self.index_width,
            )
            .into());
        }
        if cap.index_width == 0 {
            return Ok(LatentPlaneSnapshot {
                rows,
                width,
                len,
                index_width: 0,
                index_pool: 0,
                index_tail: None,
                index_pool_keys: None,
                index_pools_ready: 0,
            });
        }
        if cap.index_pool != self.index_pool {
            return Err(format!(
                "latent boundary publish: captured pool {} != live pool {}",
                cap.index_pool, self.index_pool,
            )
            .into());
        }
        let d = cap.index_width / 2;
        let pools_ready = cap.index_pools_ready;
        if self.index_pools_ready < pools_ready {
            return Err(format!(
                "latent boundary publish: live index_pools_ready {} < boundary {pools_ready} \
                 — the key plane was clamped below the capture boundary",
                self.index_pools_ready,
            )
            .into());
        }
        let index_pool_keys = if pools_ready > 0 {
            let src = self
                .index_pool_keys
                .as_ref()
                .ok_or("latent boundary publish: pools are ready but the key plane is gone")?;
            if src.len() < pools_ready * d {
                return Err(format!(
                    "latent boundary publish: key plane holds {} f32 but {pools_ready} pools \
                     x d {d} require {}",
                    src.len(),
                    pools_ready * d,
                )
                .into());
            }
            let mut keys = e.uninit(pools_ready * d)?;
            e.copy_range_into(&mut keys, 0, src, 0, pools_ready * d)?;
            Some(keys)
        } else {
            None
        };
        Ok(LatentPlaneSnapshot {
            rows,
            width,
            len,
            index_width: cap.index_width,
            index_pool: cap.index_pool,
            index_tail: cap.index_tail,
            index_pool_keys,
            index_pools_ready: pools_ready,
        })
    }

    /// Device-independent half of the restore preflight: every shape/identity/bounds check, no
    /// copies, so the caller can validate EVERY layer before the first byte moves (a malformed
    /// entry must never leave a half-restored cache for a fallback to consume).
    pub fn validate_restore(
        &self,
        snap: &LatentPlaneSnapshot,
        max_ctx: usize,
    ) -> Result<(), String> {
        if self.len != 0 {
            return Err("restore destination latent plane is not fresh".into());
        }
        if self.width != snap.width {
            return Err(format!(
                "snapshot width {} != destination width {}",
                snap.width, self.width,
            ));
        }
        if snap.len == 0 || snap.len > max_ctx {
            return Err(format!("snapshot len {} outside [1,{max_ctx}]", snap.len));
        }
        if snap.rows.len() < snap.len * snap.width {
            return Err(format!(
                "snapshot rows plane holds {} f32 but len {} x width {} requires {} \
                 (truncated capture)",
                snap.rows.len(),
                snap.len,
                snap.width,
                snap.len * snap.width,
            ));
        }
        if self.rows.len() < snap.len * self.width {
            return Err(format!(
                "destination latent plane holds {} f32 but the restore requires {}",
                self.rows.len(),
                snap.len * self.width,
            ));
        }
        if self.index_width != snap.index_width {
            return Err(format!(
                "snapshot index width {} != destination {}",
                snap.index_width, self.index_width,
            ));
        }
        if snap.index_width == 0 {
            return Ok(());
        }
        let pool = snap.index_pool;
        if pool == 0 {
            return Err("snapshot carries an index plane with an unresolved pool".into());
        }
        if self.index_pool != 0 && self.index_pool != pool {
            return Err(format!(
                "snapshot pool {pool} != destination resident pool {}",
                self.index_pool,
            ));
        }
        let d = snap.index_width / 2;
        if snap.index_pools_ready != snap.len / pool {
            return Err(format!(
                "snapshot index_pools_ready {} != len/pool {} (len {}, pool {pool}): the \
                 append-only finality invariant does not hold, so its keys are stale",
                snap.index_pools_ready,
                snap.len / pool,
                snap.len,
            ));
        }
        match (&snap.index_pool_keys, snap.index_pools_ready) {
            (Some(keys), ready @ 1..) => {
                if keys.len() < ready * d {
                    return Err(format!(
                        "snapshot key plane holds {} f32 but {ready} pools x d {d} require {}",
                        keys.len(),
                        ready * d,
                    ));
                }
            }
            (None, 0) => {}
            (Some(_), 0) => return Err("snapshot carries keys for zero ready pools".into()),
            (None, ready) => {
                return Err(format!(
                    "snapshot claims {ready} ready pools but carries no keys"
                ));
            }
        }
        let tail_rows = snap.len - snap.index_pools_ready * pool;
        match (&snap.index_tail, tail_rows) {
            (Some(tail), rows @ 1..) => {
                if tail.len() < rows * snap.index_width {
                    return Err(format!(
                        "snapshot tail holds {} f32 but {rows} rows x index width {} require {}",
                        tail.len(),
                        snap.index_width,
                        rows * snap.index_width,
                    ));
                }
            }
            (None, 0) => {}
            (Some(_), 0) => return Err("snapshot carries a tail at a pool-aligned boundary".into()),
            (None, rows) => {
                return Err(format!(
                    "snapshot owes {rows} live tail rows but carries none"
                ));
            }
        }
        if self.index_rows.is_none() {
            return Err("destination declares an index plane but allocated none".into());
        }
        if tail_rows > 0 {
            let ring = self.index_ring_rows.unwrap_or(0);
            let phys = index_plane_physical_row(ring, pool, snap.index_pools_ready * pool);
            let want = (phys + tail_rows) * self.index_width;
            let have = self.index_rows.as_ref().map_or(0, CudaSlice::len);
            if have < want {
                return Err(format!(
                    "destination index plane holds {have} f32 but the tail window requires \
                     {want}",
                ));
            }
        }
        Ok(())
    }

    /// Deep-copy a snapshot INTO this freshly allocated layer: latent rows at `[0..len)`,
    /// `len` + device mirror, and (for indexer-bearing layers) the resident key plane sized to
    /// the SESSION's capacity — exactly the `capacity_tokens / pool * d` sizing
    /// `mla_kpool_indices` books, so the next call keeps it resident instead of reallocating
    /// (a reallocation resets `index_pools_ready` and, under the ring, the rows to rebuild the
    /// keys from are gone) — plus `index_pools_ready` and the live tail rows at their physical
    /// ring (or flat) addresses. Validation runs first; a shape error moves no bytes.
    pub fn restore_plane(
        &mut self,
        e: &impl KvDev,
        snap: &LatentPlaneSnapshot,
        max_ctx: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.validate_restore(snap, max_ctx)?;
        e.copy_range_into(&mut self.rows, 0, &snap.rows, 0, snap.len * snap.width)?;
        if snap.index_width > 0 {
            let pool = snap.index_pool;
            let d = snap.index_width / 2;
            // `zeros`, not `uninit`: unbuilt key slots must not carry garbage a diagnostic
            // D2H could mistake for state. The engine only ever reads `[0..pools_ready * d)`.
            let mut keys = e.zeros(((max_ctx / pool) * d).max(1))?;
            if let Some(src) = &snap.index_pool_keys {
                e.copy_range_into(&mut keys, 0, src, 0, snap.index_pools_ready * d)?;
            }
            self.index_pool_keys = Some(keys);
            self.index_pools_ready = snap.index_pools_ready;
            self.index_pool = pool;
            if let Some(tail) = &snap.index_tail {
                let tail_rows = snap.len - snap.index_pools_ready * pool;
                let ring = self.index_ring_rows.unwrap_or(0);
                let phys = index_plane_physical_row(ring, pool, snap.index_pools_ready * pool);
                let dst = self
                    .index_rows
                    .as_mut()
                    .ok_or("destination index plane vanished after validation")?;
                e.copy_range_into(
                    dst,
                    phys * self.index_width,
                    tail,
                    0,
                    tail_rows * self.index_width,
                )?;
            }
        }
        self.len = snap.len;
        let len_i32 = i32::try_from(snap.len).map_err(|_| "latent length exceeds i32 mirror")?;
        e.set_i32_one(&mut self.len_d, len_i32)?;
        Ok(())
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
    /// Per-layer MLA latent KV plane (`StatePlan::LatentKvCache`). `None` on every non-MLA
    /// layer, so `iter().flatten()` loops skip them the way they skip `kv`/`recur` holes.
    pub latent: Vec<Option<LatentKvLayer>>,
    /// Optional per-layer tensor-parallel KV planes. The ordinary owning-stage cache remains
    /// allocated as the rollback oracle until the distributed serving path is fully qualified.
    pub tp_kv: Vec<Option<ResidentTpKvCache>>,
    /// glm5 TP (`MEMRA_GLM5_TP`) per-layer, per-rank KDA state planes: `[rank 0 (root),
    /// rank 1, ...]` shard-geometry conv ring + ssm ping-pong, lazily hydrated by the
    /// engine's TP walk on first touch (the kpool-plane precedent). The canonical
    /// `recur[il]` planes stay allocated untouched (full-width; never read by the TP walk).
    /// `None` everywhere the seam is off. The prefix-cache snapshot seams REFUSE while any
    /// slot is live (per-rank planes are not carried by CacheSnapshot); the SPEC
    /// verify/rollback seam is WIRED for these planes since lane/glm5-composition
    /// (admitted behind MEMRA_GLM5_SPEC_TP, default OFF) — the snapshot refusal is now a
    /// live runtime guard, never dead code.
    pub glm5_tp_recur: Vec<Option<Vec<RecurLayer>>>,
    /// glm5 TP PEER replicas of the MLA latent+indexer plane (replicated deterministic
    /// compute: every rank appends identical bytes in the same calls), one per peer rank
    /// (`[i]` = rank `i + 1`). The canonical `latent[il]` IS the root replica. Lazily
    /// hydrated like the field above.
    pub glm5_tp_latent_peer: Vec<Option<Vec<LatentKvLayer>>>,
    pub pos: usize,
    pub max_ctx: usize,
    /// A failed multi-stage wave may have advanced only a prefix of layers/rows. Such state is
    /// not a legal rollback point and must never be retried or returned to a reuse pool.
    pub tainted: bool,
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
    /// HC-contract tap sink (glm5 DFlash2 draft source, 2026-08-30): when armed, the
    /// HyperConnections prime/verify walks write the STREAM-MEAN (`hc_contract`) of each
    /// tapped layer's completed output into HOST rows — see [`HcTapSink`]. Host-resident by
    /// design: under a ppN split the tapped layers span stage devices, and the drafter
    /// consumes the rows on the head engine; a host sink makes the seam placement-invariant
    /// (the probe's capture seam was host-side too). None on every non-dflash2 path
    /// (zero cost: one Option check per layer).
    pub hc_taps: Option<HcTapSink>,
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
    let full_attn: usize = (lo as u32..hi as u32)
        .filter(|&il| cfg.layer_kind(il) == LayerKind::FullAttention)
        .filter(|&il| shared == 0 || il < cfg.n_layer - shared)
        .map(|il| {
            let (kv_dim_k, kv_dim_v, kbb, vbb) = full_attention_kv_layout(cfg, plan, il);
            (kv_dim_k / 32) * kbb + (kv_dim_v / 32) * vbb
        })
        .sum();
    full_attn + latent_kv_bytes_per_token_for_plan(cfg, plan, lo, hi)
}

/// Context-linear bytes per token owned by `StatePlan::LatentKvCache` layers in `[lo, hi)`,
/// mirroring `Cache::new_inner`'s latent arm plus the engine's lazy resident pool-key plane
/// (lane/glm5-gpf-workspace, 2026-08-30).
///
/// UNTIL THIS TERM EXISTED, glm5_next's admission coefficient was literally 0 B/token: the
/// per-token sum above matches `LayerKind::FullAttention` KV planes only, its 34 KDA layers are
/// `Recurrent` (correctly 0/token), and its 11 MLA layers are `LatentKvCache` — unmatched. The
/// 262k 2-card cell (`research/glm53-flash-bringup-20260827/262k-2card-20260830/`) banked the
/// resulting receipt line (`request cost: ... = 0 B/token x ctx + 155MB fixed`): admission
/// admitted prompts the device could never serve and the failure surface was a mid-stream
/// engine OOM. The prefix-latent lane named the same accounting hole.
///
/// Terms, each anchored on the allocation it mirrors:
///   * latent rows: `width` f32 per token per layer (`Cache::new_inner`,
///     `rows: e.zeros(max_ctx * width)` — eager, ctx-scaled).
///   * resident k-pool keys: `index_head_dim` f32 per POOL of tokens per layer
///     (`mla_kpool_indices`, lazy `capacity_pools * d` — ctx-scaled). `pool` is not in the
///     state plan; it comes from `cfg.glm5` (`index_kpool`). A latent plan without that config
///     charges pool = 1, which only ever over-reserves.
///   * the flat indexer state plane: `index_width` f32 per token per layer, charged ONLY when
///     the tail ring is explicitly disabled (`MEMRA_DSA_INDEX_RING=0` -> flat `max_ctx` rows).
///     With the ring on (default), the plane is a fixed working set
///     ([`INDEX_RING_WORKING_ROWS`]) and belongs to admission's fixed-residual class. (At
///     `max_ctx` below the ring rows the allocator also books a flat plane; that plane is
///     smaller than the ring's fixed bytes, so leaving it to the residual class only
///     under-counts a bounded, small amount.)
///
/// Every family whose plan compiles no `LatentKvCache` layer gets 0 from this function —
/// their coefficient is byte-identical to the pre-lane behavior.
pub fn latent_kv_bytes_per_token_for_plan(
    cfg: &ModelConfig,
    plan: &ModelPlan,
    lo: usize,
    hi: usize,
) -> usize {
    let ring_disabled = std::env::var("MEMRA_DSA_INDEX_RING")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        == Some(0);
    plan.layers
        .iter()
        .filter(|layer| (lo..hi).contains(&(layer.index as usize)))
        .map(|layer| match layer.state {
            StatePlan::LatentKvCache { width, index_width } => {
                let latent = width as usize * std::mem::size_of::<f32>();
                let index_width = index_width as usize;
                let pool = cfg
                    .glm5
                    .as_ref()
                    .map(|g| g.index_kpool as usize)
                    .filter(|&p| p > 0)
                    .unwrap_or(1);
                // One pool key of `index_head_dim = index_width / 2` f32 per `pool` tokens.
                let pool_keys = if index_width > 0 {
                    (index_width / 2) * std::mem::size_of::<f32>() / pool
                } else {
                    0
                };
                let flat_plane = if index_width > 0 && ring_disabled {
                    index_width * std::mem::size_of::<f32>()
                } else {
                    0
                };
                latent + pool_keys + flat_plane
            }
            _ => 0,
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

/// See [`Cache::hc_taps`]. Armed per walk by the glm5 DFlash2 draft source; the hc trunk
/// writes the CONTRACTED (stream-mean) completed output of tapped layer `layer_ids[s]` for
/// walk row r at `rows[(base + r) * n_taps * hidden + s * hidden ..][..hidden]` — the
/// drafter fc's input layout, measured by the dflash2 probe's capture seam
/// (research/glm53-flash-bringup-20260827/dflash2-probe-20260829/: stream-mean of the
/// completed layer output == the SGLang glm5_next hc_contract aux-hidden definition).
pub struct HcTapSink {
    /// Plan layer indices whose COMPLETED output is tapped, in drafter fc slot order.
    pub layer_ids: Vec<usize>,
    /// Host rows, `[t, n_taps * hidden]` row-major.
    pub rows: Vec<f32>,
    pub hidden: usize,
    /// Total rows the sink covers.
    pub t: usize,
    /// Row offset of the CURRENT walk's row 0 (chunked primes set it per chunk; the verify
    /// walk leaves it 0).
    pub base: usize,
    /// ABSOLUTE position of sink row 0 (lane/glm5-prefix-latent2, 2026-09-01): a SUFFIX
    /// prime over a restored cache writes at `cache.pos`-derived bases starting at the
    /// restored boundary, while its sink covers only the suffix rows — the writer lands
    /// row r of a walk at sink row `base - origin + r`. Fresh-prompt sinks leave it 0
    /// (byte-identical indexing to before the field existed).
    pub origin: usize,
    /// DEVICE STAGING (lane/glm5-loop-port, 2026-08-30): one optional `[t * hidden]` buffer
    /// per tap slot, allocated lazily by the walk ON THE WRITING engine's device (under a
    /// ppN split each tapped layer belongs to exactly one stage, so a slot's buffer lives
    /// where its layer runs). When `device_stage` is set the trunk walk D2D-copies the
    /// contracted rows here instead of blocking on a mid-walk DtoH — the five in-walk host
    /// syncs the 3way window priced into the fixed round cost (map row #17) — and the
    /// round drains every slot into `rows` at its ONE post-walk sync point.
    pub dev: Vec<Option<CudaSlice<f32>>>,
    /// Arm device staging. Verify-round sinks set it; PRIME sinks stay host-staged BY
    /// DESIGN — a `[prompt, hidden]` per-slot device transient at 16k-prompt depth is
    /// ~1.3 GiB of VRAM the prime must not hold, and the prime's per-chunk DtoH amortizes
    /// over >= 256 rows (DFlash2 TTFT is near-constant already, 3way cell 4).
    pub device_stage: bool,
}

impl HcTapSink {
    pub fn new(layer_ids: Vec<usize>, hidden: usize, t: usize) -> Self {
        let n_taps = layer_ids.len();
        Self {
            layer_ids,
            rows: vec![0.0; t * n_taps * hidden],
            hidden,
            t,
            base: 0,
            origin: 0,
            dev: (0..n_taps).map(|_| None).collect(),
            device_stage: false,
        }
    }

    /// Suffix-prime sink (doc on [`Self::origin`]): covers `t` rows whose first row sits at
    /// absolute position `origin` — the restored-boundary continuation shape.
    pub fn new_at(layer_ids: Vec<usize>, hidden: usize, t: usize, origin: usize) -> Self {
        Self {
            origin,
            ..Self::new(layer_ids, hidden, t)
        }
    }

    /// Device-staged sink (doc on [`Self::device_stage`]): the walk stages tap rows on
    /// device and the consumer drains them post-walk in one sync.
    pub fn new_device_staged(layer_ids: Vec<usize>, hidden: usize, t: usize) -> Self {
        Self {
            device_stage: true,
            ..Self::new(layer_ids, hidden, t)
        }
    }
}

/// Snapshot of the dual cache taken BEFORE a spec-decode draft+verify round (MTP-PLAN §C/§D.4).
/// - Full-attn KV: only the per-layer `len` is recorded; rollback truncates (append-only,
///   position-addressed — no copy). C.1.
/// - Linear-attn conv/ssm: real device-to-device COPIES of the recurrent state, because those
///   buffers are mutated IN PLACE by the verify pass and have no position index to truncate. C.2.
///   (We alloc fresh + memcpy_dtod. NOTE, corrected memra-next#23: the parenthetical here used
///   to justify that with "CudaSlice::clone is an Arc refcount, NOT a buffer copy", which is
///   false in the LOCKED cudarc 0.19.8 — `Clone` is `try_clone().unwrap()` = alloc + D2D copy. The explicit
///   copy is still the right call here, for two reasons that are NOT aliasing: it is fallible
///   rather than panicking, and it places the copy on the calling engine's current stream instead
///   of the source slice's. Genuine aliasing needs an `Arc<CudaSlice<T>>`.)
///
/// IT COVERS TWO OF THE CACHE'S FOUR STATE PLANES, AND THAT IS A KNOWN HOLE
/// (lane/prefix-restore-toolcall, 2026-08-28). `Cache` also has `tp_kv` (recorded here as
/// `tp_kv_len`) and `latent`, and NOTHING in this struct or in `Cache::rollback` mentions
/// `latent`. A `StatePlan::LatentKvCache` layer keeps its FULL-ATTENTION history there, so
/// rolling back a latent-bearing cache moves `pos` while every MLA layer keeps its longer
/// `len`: the next tokens append past the boundary and attend stale rows. The identical
/// two-plane assumption in the server's `PrefixEntry` is what made a glm5_next prefix-cache
/// hit restore an EMPTY attention history while reporting `cached_tokens: N of N`, and it
/// fabricated instead of failing (research/prefix-restore-toolcall-20260828/).
///
/// Today nothing reaches it: `maybe_plain_checkpoint` refuses to arm on a latent-bearing
/// cache, and the spec rewind cannot fire because every latent model is EAGER-ONLY with no
/// drafter. IT BECOMES LIVE THE MOMENT A LATENT MODEL GETS A SPEC ARM. Growing latent
/// awareness here is not a symmetric addition: the rows are unquantized f32, `index_rows` is
/// a tail ring rather than a flat addressable plane, and `index_pool_keys` /
/// `index_pools_ready` carry an append-only finality invariant (`truncate_index_pool_keys`
/// exists precisely because a `len` that moves backwards invalidates them).
pub struct CacheSnapshot {
    pub kv_len: Vec<Option<usize>>, // per layer (Some for full-attn layers)
    pub tp_kv_len: Vec<Option<usize>>, // per layer (Some for TP full-attn layers)
    pub conv: Vec<Option<CudaSlice<f32>>>, // per layer (Some for linear-attn layers, D2D copy)
    pub ssm: Vec<Option<CudaSlice<f32>>>,
    pub pos: usize,
}

impl Cache {
    pub fn ensure_usable(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.tainted {
            return Err(format!(
                "{path}: cache was tainted by a failed pipeline wave and cannot be reused"
            )
            .into());
        }
        Ok(())
    }

    pub fn mark_tainted(&mut self) {
        self.tainted = true;
        self.last_logits_dev = None;
        self.dflash_taps = None;
    }

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
    pub fn new_ppn(
        devs: &[&dyn KvDev],
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

    pub fn new_ppn_planned(
        devs: &[&dyn KvDev],
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
        let mut latent = Vec::with_capacity(n);
        let head_dim_k = cfg.head_dim_k as usize;
        let head_dim_v = cfg.head_dim_v as usize;
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
                latent.push(None);
                continue;
            }
            match layer.state {
                StatePlan::KvCache { .. } | StatePlan::SlidingKvCache { .. } => {
                    // KVQUANT block constraint. Scoped to the QUANTIZED planes: it was a
                    // function-wide assert, which made any model whose cfg head dims are not
                    // 32-multiples unallocatable even when no layer owns a quantized plane —
                    // glm-dsa's latent row (kv_lora + rope) is exactly that shape.
                    assert!(
                        head_dim_k.is_multiple_of(32) && head_dim_v.is_multiple_of(32),
                        "KVQUANT requires head_dim_k%32==0 && head_dim_v%32==0 \
                         (layer {il}: k={head_dim_k} v={head_dim_v})"
                    );
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
                        base_d: None,
                    }));
                    recur.push(None);
                    latent.push(None);
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
                    latent.push(None);
                }
                StatePlan::LatentKvCache { width, index_width } => {
                    // ONE f32 row per token for the whole layer (MQA): no per-head planes, no
                    // V plane. `width` is the plan's own number, not re-derived here — the
                    // engine's MLA arm asserts it against the loaded `MlaGeom`.
                    let width = width as usize;
                    assert!(
                        width > 0,
                        "layer {il}: LatentKvCache width must be positive"
                    );
                    kv.push(None);
                    recur.push(None);
                    let index_width = index_width as usize;
                    // TAIL RING: the indexer plane is read exactly once per row, by its own
                    // pool's key build, so it only has to hold the incomplete tail plus one
                    // call's tokens. `None` keeps the flat `max_ctx`-row plane.
                    let index_ring = if index_width == 0 {
                        None
                    } else {
                        index_ring_rows(max_ctx)
                    };
                    let index_rows = match index_width {
                        0 => None,
                        w => Some(e.zeros(index_ring.unwrap_or(max_ctx) * w)?),
                    };
                    latent.push(Some(LatentKvLayer {
                        rows: e.zeros(max_ctx * width)?,
                        width,
                        len: 0,
                        len_d: e.htod_i32(&[0])?,
                        index_rows,
                        index_width,
                        index_ring_rows: index_ring,
                        // Sized from the indexer's `pool`, which the state plan does not carry;
                        // the engine allocates it the first time the layer selects.
                        index_pool_keys: None,
                        index_pools_ready: 0,
                        index_pool: 0,
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
            latent,
            tp_kv: (0..n).map(|_| None).collect(),
            glm5_tp_recur: (0..n).map(|_| None).collect(),
            glm5_tp_latent_peer: (0..n).map(|_| None).collect(),
            pos: 0,
            max_ctx,
            tainted: false,
            dflash_taps: None,
            hc_taps: None,
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
        // glm5 TP-2 state is per-rank and lives outside CacheSnapshot; a snapshot taken over
        // live TP planes would silently drop the peer's half. Spec (the only snapshot
        // consumer for this family) is co-refused with the TP door — hold that closed here.
        if self.glm5_tp_recur.iter().any(Option::is_some)
            || self.glm5_tp_latent_peer.iter().any(Option::is_some)
        {
            return Err(
                "cache snapshot is unwired for glm5 TP rank state (MEMRA_GLM5_TP): \
                        per-rank planes are not carried by CacheSnapshot"
                    .into(),
            );
        }
        self.ensure_usable("cache snapshot")?;
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
        if self.glm5_tp_recur.iter().any(Option::is_some)
            || self.glm5_tp_latent_peer.iter().any(Option::is_some)
        {
            return Err("cache snapshot_into is unwired for glm5 TP rank state \
                        (MEMRA_GLM5_TP): per-rank planes are not carried by CacheSnapshot"
                .into());
        }
        self.ensure_usable("cache snapshot refresh")?;
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
    ///   `cache.pos` is set to `snap.pos` so the caller's replay advances it back to the commit point.
    pub fn rollback(
        &mut self,
        e: &impl KvDev,
        snap: &CacheSnapshot,
        accept_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.ensure_usable("cache rollback")?;
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
        Cache, INDEX_RING_WORKING_ROWS, KvRingAppend, ResidentTpKvCache, TpKvTransactionState,
        index_ring_default_rows, index_ring_rows_for, index_ring_take, tp_kv_rank_allocation_shape,
    };

    /// glm5_next's declared k-pool width, from `crates/memra-gguf/src/model_packs/glm5_next/mod.rs`
    /// (`KpoolPlan { pool: 4, .. }`). The ONE architecture `MEMRA_DSA_INDEX_RING` exists for.
    const GLM5_NEXT_POOL: usize = 4;
    /// Packed indexer row: `2 * index_head_dim` (128) f32 = 1 KiB per token per MLA layer.
    const GLM5_NEXT_STATE_ROW_BYTES: usize = 2 * 128 * 4;
    /// What the tail ring costs per MLA layer, at EVERY configured context. 5 MiB against the
    /// 1 GiB per layer a flat plane costs at 1M.
    const RING_BYTES_PER_LAYER: usize = INDEX_RING_WORKING_ROWS * GLM5_NEXT_STATE_ROW_BYTES;

    /// THE SIZING GATE (lane/glm53-ring-sizing, 2026-08-28).
    ///
    /// The regression this exists for, measured on the bench box three arms one env flag apart
    /// on the SAME binary (research/glm53-flash-bringup-20260827/rebaseline-and-surface-20260828,
    /// receipts 13 and 14): at `MEMRA_CTX=8192` the ring ON served at most 4630 prompt tokens,
    /// `MEMRA_DSA_INDEX_RING=0` served 7300, and the pre-ring binary served 7312. USABLE CONTEXT
    /// WAS A FRACTION OF CONFIGURED CONTEXT because the ring was sized against a chunked-prefill
    /// bound, and glm5_next primes MONOLITHICALLY (`prime_cache_hyper`, no `prime_chunk_ranges`),
    /// so its per-call `t` is the whole prompt.
    ///
    /// So the gate asserts the RATIO, never one number: for every configured context, a single
    /// monolithic prime of the WHOLE context must be admitted by the ring the shipped default
    /// derivation books for it. It runs the shipped admission rule (`index_ring_take`) in the
    /// shipped drain shape, so it fails exactly when the engine fails.
    ///
    /// And it asserts the ring is STILL A RING, at every one of those contexts: a "fix" that
    /// grows the plane back to `max_ctx` rows passes the acceptance half and is a silent revert
    /// of the 11.94 GiB this flag exists to free.
    #[test]
    fn the_derived_ring_serves_a_monolithic_prime_of_the_whole_configured_context() {
        // Two decades of context, and this model's NATIVE 1,048,576. A sizing that works at 8192
        // and breaks at 262144 is not a sizing.
        for max_ctx in [8192usize, 262_144, 1 << 20] {
            let rows = index_ring_default_rows(max_ctx).unwrap_or_else(|| {
                panic!("the ring must engage at max_ctx {max_ctx}: it is where the saving is")
            });
            // The engine rounds the booked rows DOWN to a multiple of `pool`, because the state
            // plan does not carry `pool` and the allocator cannot book a pool-aligned budget.
            let ring = rows / GLM5_NEXT_POOL * GLM5_NEXT_POOL;

            // MONOLITHIC PRIME: one call, nothing resident, `t` = the whole configured context.
            let mut cur = 0usize;
            let mut pools_ready = 0usize;
            let mut steps = 0usize;
            while cur < max_ctx {
                let take = index_ring_take(ring, GLM5_NEXT_POOL, pools_ready, cur, max_ctx - cur)
                    .unwrap_or_else(|| {
                        panic!(
                            "MEMRA_CTX={max_ctx}: the {ring}-row ring refused a monolithic prime \
                         after {cur} of {max_ctx} tokens ({}% of the configured context). \
                         USABLE CONTEXT MUST BE AT LEAST CONFIGURED CONTEXT. This is the \
                         4630-of-8192 regression, in arithmetic.",
                            cur * 100 / max_ctx
                        )
                    });
                assert!(
                    take > 0,
                    "MEMRA_CTX={max_ctx}: the drain made no progress at row {cur}: a zero take \
                     is an infinite loop in the engine, not a refusal"
                );
                cur += take;
                pools_ready = cur / GLM5_NEXT_POOL;
                steps += 1;
                assert!(
                    steps <= max_ctx,
                    "MEMRA_CTX={max_ctx}: the drain did not terminate"
                );
            }
            assert_eq!(cur, max_ctx, "the whole prompt must be appended");

            // STILL A RING, and the property is that the plane DOES NOT GROW WITH CONTEXT.
            // Per MLA layer, and glm5_next has 12 of them. A sizing "fix" that bought
            // acceptance by scaling the ring toward `max_ctx` is a silent revert of the
            // 11.94 GiB, and it passes the acceptance half above, so this is the half that
            // catches it. At 1M the flat plane is 1 GiB per layer and the ring is 5 MiB.
            let ring_bytes = rows * GLM5_NEXT_STATE_ROW_BYTES;
            let flat_bytes = max_ctx * GLM5_NEXT_STATE_ROW_BYTES;
            assert_eq!(
                ring_bytes, RING_BYTES_PER_LAYER,
                "MEMRA_CTX={max_ctx}: the ring books {rows} rows, not the context-independent \
                 {INDEX_RING_WORKING_ROWS}. A plane that tracks max_ctx is the flat plane \
                 wearing a modulus"
            );
            assert!(
                rows < max_ctx,
                "MEMRA_CTX={max_ctx}: a ring of {rows} rows is not shorter than the flat plane \
                 it replaces, so it would not engage at all"
            );
            // An ABSOLUTE cap, so that raising the working-set constant to buy acceptance fails
            // here too rather than moving `RING_BYTES_PER_LAYER` along with it. 16 MiB per layer
            // is 3x the shipped ring and still 64x under the flat plane at 1M.
            assert!(
                ring_bytes <= 16 << 20,
                "MEMRA_CTX={max_ctx}: {} MiB per MLA layer, over glm5_next's 12 of them. The ring \
                 exists to delete 11.94 GiB; a working set this large is not paying for itself",
                ring_bytes >> 20
            );
            println!(
                "MEMRA_CTX={max_ctx}: ring {rows} rows (effective {ring}), monolithic prime of \
                 {max_ctx} tokens admitted in {steps} drain step(s); plane {} MiB/layer vs flat \
                 {} MiB/layer",
                ring_bytes >> 20,
                flat_bytes >> 20
            );
        }
    }

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
    fn index_ring_sizing_is_pure_and_carries_no_per_call_t() {
        // Default derivation: the working-set constant, engaged only when it is actually SHORTER
        // than the flat plane it replaces.
        let rows = INDEX_RING_WORKING_ROWS;
        assert_eq!(index_ring_rows_for(None, 1 << 20), Some(rows));
        assert_eq!(index_ring_default_rows(1 << 20), Some(rows));
        // 4k context: the ring would be LONGER than the flat plane, so it does not engage and
        // the saving at that context is honestly zero.
        assert_eq!(index_ring_rows_for(None, 4096), None);
        assert_eq!(index_ring_rows_for(None, rows), None);
        assert_eq!(index_ring_rows_for(None, rows + 1), Some(rows));

        // THE CORRECTION (lane/glm53-ring-sizing). The derivation reads no prefill chunk bound at
        // all now, so the SAME rows are booked at every context above the collapse point, and no
        // value of any other flag can move them. Under the old rule an assumed 4096-token chunk
        // sized the ring and a monolithic prime blew straight through it.
        for max_ctx in [8192usize, 262_144, 1 << 20] {
            assert_eq!(
                index_ring_rows_for(None, max_ctx),
                Some(INDEX_RING_WORKING_ROWS),
                "the derived ring must not vary with the configured context"
            );
        }

        // The knob: 0 is the rollback seam, n pins the row budget (how the wraparound gate
        // reaches a wrap in a micro fixture).
        assert_eq!(index_ring_rows_for(Some(0), 1 << 20), None);
        assert_eq!(index_ring_rows_for(Some(16), 64), Some(16));
        assert_eq!(index_ring_rows_for(Some(64), 64), None);
    }

    /// The admission rule itself, over the shapes the engine actually presents it.
    #[test]
    fn index_ring_take_drains_instead_of_bounding_the_call() {
        const POOL: usize = GLM5_NEXT_POOL;
        // A flat plane takes the whole call in one bite, whatever else is true.
        assert_eq!(index_ring_take(0, POOL, 0, 0, 1 << 20), Some(1 << 20));
        // Fresh monolithic prime over a ring 16 times shorter than the call: it takes the ring,
        // never more, and never refuses.
        assert_eq!(index_ring_take(64, POOL, 0, 0, 1024), Some(64));
        // Steady state after a build: the carry-over is under one pool, so the next bite is at
        // least `ring - pool + 1` and progress is guaranteed.
        for cur in 0..64usize {
            let ready = cur / POOL;
            let take = index_ring_take(64, POOL, ready, cur, 1024).expect("never lapses");
            assert!(
                (64 - POOL + 1..=64).contains(&take),
                "cur {cur}: take {take} outside the guaranteed progress band"
            );
        }
        // A call SHORTER than what fits is taken whole, so a decode step is one iteration.
        assert_eq!(index_ring_take(64, POOL, 4, 16, 1), Some(1));
        // The one surviving lapse: resident pool keys further than the ring behind the append.
        // A rewind that did not clamp `index_pools_ready`, or a pool-key reallocation.
        assert_eq!(index_ring_take(16, POOL, 0, 64, 1), None);
        assert_eq!(index_ring_take(16, POOL, 0, 16, 1), None);
        assert_eq!(index_ring_take(16, POOL, 0, 15, 1), Some(1));
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
        assert_eq!(cache.physical_capacity(), 32 + 4096 + 512 + 31);
        // 8250 -> 8762: the extra alignment block moved the wrap point, and at 8250 this append is
        // now Contiguous — the test would keep passing while no longer exercising the rebase
        // it is named for. Every offset below moves by the same 32 rows; intent unchanged.
        cache.publish_hydration(8762, 4096).unwrap();
        assert_eq!(cache.ring_base(), Some(4096));

        let transaction = cache.begin_transaction().unwrap();
        let plan = cache.prepare_append(transaction, 10).unwrap();
        assert_eq!(plan.target(), 8772);
        assert_eq!(plan.write_row(), 58);
        assert_eq!(
            plan.ring_append(),
            Some(KvRingAppend::Rebase {
                src_row: 4608,
                keep_rows: 58,
                new_base: 8704,
                write_row: 58,
            })
        );
        cache.publish_append_rebase(plan).unwrap();
        cache.publish_append_plan(plan).unwrap();
        assert_eq!(cache.ring_base(), Some(8704));
        assert_eq!(cache.physical_range(8740, 8772).unwrap(), 36..68);

        let rollback = cache.commit_target(transaction, 0).unwrap();
        cache.publish_finalize(transaction, rollback).unwrap();
        assert_eq!((cache.committed_len(), cache.staged_len()), (8762, 8762));
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
            latent: Vec::new(),
            tp_kv: vec![None],
            glm5_tp_recur: vec![None],
            glm5_tp_latent_peer: vec![None],
            pos: 0,
            max_ctx: 10_000,
            tainted: false,
            dflash_taps: None,
            hc_taps: None,
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
    use super::{
        KvRing, KvRingAppend, PRIME_CHUNK_MAX_TOKENS, SWA_REWIND_SLACK_ROWS,
        SWA_VIEW_ALIGNMENT_ROWS, kv_plane_allocation_bytes, swa_retain_from, swa_ring_rows,
    };

    #[test]
    fn allocation_rows_cover_window_max_prime_and_alignment_slack() {
        assert_eq!(swa_ring_rows(512, 262_144), 512 + 4096 + 512 + 31);
        assert_eq!(swa_ring_rows(512, 4096), 4096);
        assert_eq!(
            kv_plane_allocation_bytes(5151, 1088),
            5151 * 1088 + 8,
            "the Step35 session plane allocates ring rows plus the existing tail pad",
        );
    }

    /// REGRESSION, the SWA-ring MTP lap (2026-08-28) — BOTH steps, which took three attempts to
    /// separate on hardware.
    ///
    /// Step 1, the REWIND. A rebase that retains exactly the window parks `base` at the newest
    /// legal value, so the next backward rewind — even by one token — floors an alignment block
    /// under it and is refused:
    ///   rewind_to=4638 window=512 base=4128 rows=4639 needed_view_start=4096 < base
    ///
    /// Step 2, the RE-APPEND, which a slack-only fix broke. After a legal rewind `first_row` moves
    /// back while `base` does not, so an unclamped ideal retain falls under `base` and the append
    /// itself is refused: "SWA ring lapped required rows (base 4128, retain 4096, len 4669)".
    /// Slack is something the ring GRANTS when it can, never something a caller may demand.
    #[test]
    fn retain_grants_rewind_slack_but_never_asks_below_base() {
        const WINDOW: usize = 512;
        let rows = swa_ring_rows(WINDOW, 262_144);
        let len = rows;
        let aligned = |pos: usize| (pos - (WINDOW - 1)) & !(SWA_VIEW_ALIGNMENT_ROWS - 1);

        // step 1 — from base 0 the retain sits below the aligned window start, so a rewind of up
        // to a full alignment block survives the rebase.
        let retain = swa_retain_from(len, WINDOW, 0);
        assert!(retain <= aligned(len) - SWA_REWIND_SLACK_ROWS);
        let mut ring = KvRing::new(rows, WINDOW);
        ring.apply_rebase(retain);
        assert!(
            ring.can_rewind_to(len - 1),
            "a one-token rewind must survive the rebase"
        );
        assert!(ring.can_rewind_to(len - SWA_REWIND_SLACK_ROWS));

        // ...and a full prime chunk still fits at that retention, which is why the ring grew.
        assert!(len - retain + PRIME_CHUNK_MAX_TOKENS <= rows);

        // the headroom is REAL, not clamped away: every rewind within it is legal from a base
        // the ring was actually sized to keep. This is what the 32-row version could not do —
        // it clamped instead, leaving the window pointing below resident rows (all-NaN logits).
        for depth in [1usize, 32, 256, SWA_REWIND_SLACK_ROWS] {
            assert!(
                ring.can_rewind_to(len - depth),
                "a {depth}-row rewind must be resident, not clamped away",
            );
        }

        // step 2 — the property that actually keeps this safe is NOT `retain >= base`, it is that
        // the attention WINDOW is fully resident: window_start >= base. The clamp to `base` is
        // correct exactly while that holds, and v3's NaN came from clamping with only 32 rows of
        // headroom, where a deeper rewind clamped into a window that ran below resident rows.
        // With the ring sized for SWA_REWIND_SLACK_ROWS, every rewind inside the headroom keeps a
        // complete window — so the clamp is safe by construction rather than by luck.
        let base = ring.base();
        for depth in [1usize, 32, 256, SWA_REWIND_SLACK_ROWS] {
            let window_start = (len - depth - (WINDOW - 1)) & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
            assert!(
                window_start >= base,
                "after a {depth}-row rewind the window starts at {window_start}, below base \
                 {base} — clamping here would serve rows the ring no longer holds (the pos-8661 \
                 all-NaN case)",
            );
            assert!(swa_retain_from(len - depth, WINDOW, base) >= base);
        }

        // and one row past the headroom the window DOES run below base — the case that must stay
        // refused rather than clamped, which is what can_rewind_to enforces.
        let past = len - (SWA_REWIND_SLACK_ROWS + WINDOW);
        let past_start = (past - (WINDOW - 1)) & !(SWA_VIEW_ALIGNMENT_ROWS - 1);
        assert!(
            past_start < base,
            "beyond the headroom the window must fall below base"
        );
        assert!(
            !ring.can_rewind_to(past),
            "and can_rewind_to must refuse it"
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

    /// The 2026-08-29 warm-turn-at-40k panic: a checkpoint on a LAPPED ring records an absolute
    /// `len` far past the physical rows, and a flat `len`-row restore is an out-of-bounds device
    /// slice. The plan must hand back only the aligned live window plus the base to rebase a
    /// fresh target to — and refuse once the source ring no longer holds that window.
    #[test]
    fn restore_plan_copies_the_window_not_the_absolute_length() {
        let mut ring = KvRing::new(swa_ring_rows(512, 262_144), 512);
        // Before any wrap: the plan is exactly the flat prefix.
        let (base, phys) = ring.restore_plan(400).unwrap();
        assert_eq!((base, phys), (0, 0..400));

        // Lap the ring far past its physical capacity (a 40k-token session), the way a real
        // prime does: 4096-row chunks, rebasing whenever the tail would wrap.
        let mut live = 0usize;
        while live < 40_960 {
            let retain = swa_retain_from(live, 512, ring.base());
            if let KvRingAppend::Rebase { new_base, .. } =
                ring.append_plan(live, retain, 4096).unwrap()
            {
                ring.apply_rebase(new_base);
            }
            live += 4096;
        }
        assert!(ring.base() > 0, "a 40k walk must have lapped the ring");
        let (base, phys) = ring.restore_plan(live).unwrap();
        assert_eq!(base, (live - (512 - 1)) & !31usize);
        assert!(
            base >= ring.base(),
            "the plan must stay above the ring floor"
        );
        assert_eq!(phys.len(), live - base);
        assert!(
            phys.end <= ring.rows(),
            "the copy must fit the physical buffer ({} rows), got {:?}",
            ring.rows(),
            phys
        );

        // A checkpoint from before the rebase is gone: refuse, never slice.
        assert!(ring.restore_plan(400).is_err());
    }
}
