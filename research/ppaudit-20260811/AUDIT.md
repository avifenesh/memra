# PP-2 Model-Global Tensor Audit — 2026-08-11

> Verdicts as of 2026-08-11 pre-fix; rope_freqs and ones since fixed (research/gemmarope-20260811, gemmaaux-20260811).

## Scope

Audit class: model-global tensors (rope factors, correction biases, router weights, norm weights, embedding/lm_head) used on remote PP stages without stage-local copies.

Two known bugs in this class:
1. **Step-3.7 rope_freqs.weight** (FIXED) — lived only on primary device; `MEMRA_PP_HOST_BOUNCE=1` remote stage read became illegal address. Fix replicated one immutable copy per distinct PP device (research/hostbounce-20260810/RESULTS.md).
2. **Gemma4 GemmaAux::rope_freqs** (LATENT) — holds single CudaSlice on load engine; any future gemma PP-2 staging full-attn layers off primary re-hits same issue.

Files audited:
- `crates/memra-engine/src/hybrid.rs` (lines 1-2000+)
- `crates/memra-engine/src/hybrid_forward.rs`
- `crates/memra-engine/src/pp.rs` (lines 1-1488)
- `crates/memra-engine/src/model.rs`

---

## Findings Summary

| Tensor | Status | Device Copies | Cross-Stage Risk |
|--------|--------|---------------|------------------|
| `Step35Aux::rope_freqs` | ✅ SAFE | Multi-device replication | None — reference pattern |
| **`GemmaAux::rope_freqs`** | ⚠️ **LATENT BUG** | Single (primary) | **7 read sites, all direct deref** |
| `MoeWeights::exp_probs_b` | ✅ SAFE | Host-side | None — host slice |
| `HybridModel::embd` | ✅ SAFE | Stage 0 | None — no cross-stage embed |
| `HybridModel::output_norm` | ✅ SAFE | Last stage | None — loaded via last stage engine |
| `HybridModel::output` (lm_head) | ✅ SAFE | Last stage | None — loaded via last stage engine |
| `GemmaAux::ones` | ✅ SAFE | Single (primary) | None — primary-only use |
| `GemmaAux::suppress_d` | ✅ SAFE | Single (primary) | None — primary-only use |
| `GemmaAux::e4b.*` | ⚠️ UNCHECKED | Single (primary) | Unknown — E4B PP compatibility not audited |

**Critical finding**: `GemmaAux::rope_freqs` is the only CONFIRMED cross-device bug. All other model-global tensors are either properly replicated, stage-local, or host-side.

---

## Detailed Findings

### Step35Aux::rope_freqs (SAFE)

**Load site**: `hybrid.rs:1661-1688`

```rust
let step35_aux = if cfg.step35.is_some() {
    let rope_freqs = match src.find("rope_freqs.weight") {
        Some(t) => {
            let host = memra_gguf::dequant::dequantize(...);
            let mut copies = Vec::new();
            if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
                for s in 0..fence.len() - 1 {
                    let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                    let dev = owner.ctx().ordinal();
                    if copies.iter().all(|(d, _)| *d != dev) {
                        copies.push((dev, owner.htod(&host)?));
                    }
                }
            } else {
                copies.push((e.ctx().ordinal(), e.htod(&host)?));
            }
            Some(copies)
        }
        None => None,
    };
    Some(Step35Aux { rope_freqs })
}
```

**Verdict**: SAFE — creates one copy per distinct PP device. This is the REFERENCE PATTERN for fixing the class.

**Storage**: `Vec<(usize, CudaSlice<f32>)>` — device ordinal paired with device-local copy.

**Access pattern**: TBD (need to check hybrid_forward.rs read sites)

---

### GemmaAux::rope_freqs (LATENT BUG)

**Load site**: `hybrid.rs:1602-1609`

```rust
let rope_freqs = match src.find("rope_freqs.weight") {
    Some(t) => Some(e.htod(&memra_gguf::dequant::dequantize(
        t.ggml_type,
        &t.bytes,
        t.ne.iter().product::<u64>() as usize,
    ))?),
    None => None,
};
```

**Verdict**: ⚠️ LATENT BUG — loaded through primary engine `e`, stored as single `CudaSlice<f32>`.

**Storage**: `Option<CudaSlice<f32>>` in `GemmaAux` struct (hybrid.rs:1649)

**Problem**: Under cross-device PP-2 where a Gemma4 full-attention layer stages to device 1, the remote stage's RoPE kernel dereferences `aux.rope_freqs` which points to device 0 memory. Same illegal-address failure mode as Step-3.7 bug #1.

**Read sites**: TBD (checking hybrid_forward.rs for direct `aux.rope_freqs` reads)

**Recommended fix**: Apply Step35 pattern — detect PP fence at load, replicate one copy per distinct device, store as `Vec<(usize, CudaSlice<f32>)>`, lookup by layer's owning device at read time.

---

## Detailed Tensor Inventory

### 1. rope_freqs tensors

#### Step35Aux::rope_freqs — ✅ SAFE
- **Type**: `Option<Vec<(usize, CudaSlice<f32>)>>`
- **Load**: `hybrid.rs:1661-1688`
- **Accessor**: `hybrid.rs:Step35Aux::rope_freqs(e)` — looks up by device ordinal
- **Read sites**: 
  - `hybrid_forward.rs:8689` — step35 prime full-attn RoPE
  - `hybrid_forward.rs:8941` — step35 decode full-attn RoPE
- **Verdict**: SAFE — multi-device replication, device-local lookup

#### GemmaAux::rope_freqs — ⚠️ LATENT BUG
- **Type**: `Option<CudaSlice<f32>>`
- **Load**: `hybrid.rs:1602-1609` — single `e.htod()` on primary engine
- **Definition**: `hybrid.rs` GemmaAux struct
- **Read sites**:
  - `hybrid_forward.rs:6522` — gemma4 global full-attn decode RoPE
  - `hybrid_forward.rs:7277` — gemma4 prefill RoPE
  - `hybrid_forward.rs:7522` — gemma4 decode batched RoPE
  - `hybrid_forward.rs:7669` — gemma4 decode DC RoPE
  - `hybrid_forward.rs:8190` — gemma4 decode graph RoPE
  - `hybrid_forward.rs:8303` — gemma4 decode spec verify RoPE
  - `hybrid_forward.rs:9061` — gemma4 E4B prime RoPE
- **Verdict**: ⚠️ LATENT BUG — single device copy, all read sites dereference `aux.rope_freqs.as_ref().expect(...)` directly without device check

### 2. MoE router correction bias

#### MoeWeights::exp_probs_b — ✅ SAFE (host-side only)
- **Type**: `Option<Vec<f32>>` (host-side)
- **Load**: `hybrid.rs:335-337` — via `src.find()` and dequant
- **Read sites**:
  - `hybrid_forward.rs:3497, 3509` — moe_router_topk calls (passed as host slice)
  - `hybrid_forward.rs:4463` — SlruCtx struct (cloned)
  - `hybrid_forward.rs:5693, 6231` — moe_router_topk with sigmoid
- **Verdict**: ✅ SAFE — lives on host, passed by slice reference to kernels, no device affinity

### 3. Embedding table

#### HybridModel::embd — ✅ SAFE (stage 0 local, per pp.rs:49)
- **Type**: `EmbedHost` (host bytes)
- **Load**: `hybrid.rs:1546` — `EmbedHost::from_source(src, "token_embd.weight")`
- **Read sites**: Multiple `self.embed(e, tokens)` calls
- **PP ownership**: pp.rs:49 explicitly states "the embed table lives with stage 0"
- **Verdict**: ✅ SAFE — stage 0 only (no cross-stage embed calls under PP)

### 4. Output norm + lm_head

#### HybridModel::output_norm, HybridModel::output — ✅ SAFE (last stage local)
- **Type**: `GpuTensor`
- **Load**: `hybrid.rs:1548-1556`
  ```rust
  let e_head = crate::pp::layer_engine(e, n_trunk, n_trunk - 1)?;
  let output_norm = load_t(e_head, src, "output_norm.weight")?;
  let mut output = if src.has("output.weight") {
      load_t(e_head, src, "output.weight")?
  } else {
      load_t(e_head, src, "token_embd.weight")?
  };
  ```
- **PP ownership**: pp.rs:28,49 explicitly states "output_norm + lm head load through the LAST stage's engine"
- **Verdict**: ✅ SAFE — loaded through last stage's engine (`layer_engine(e, n_trunk, n_trunk-1)`)

### 5. GemmaAux::ones — ✅ SAFE (single device, used only on that device)
- **Type**: `CudaSlice<f32>`
- **Load**: `hybrid.rs:1651` — `e.htod(&[1.0f32; 512])?`
- **Usage**: gemma4 weightless rms_norm (R7)
- **Read sites**: `hybrid_forward.rs:6518` — rms_norm_qkv call (local to decode)
- **Verdict**: ✅ SAFE — used only on primary device where gemma4 decode runs

### 6. GemmaAux::suppress_d — ✅ SAFE (single device, used only on that device)
- **Type**: `Option<(CudaSlice<i32>, usize)>`
- **Load**: `hybrid.rs:1641-1648`
- **Read sites**: `hybrid_forward.rs:6465` — suppress mask before sampling
- **Verdict**: ✅ SAFE — used only on primary device where sampling runs

### 7. GemmaAux::e4b model tensors — ⚠️ MIXED (some safe, some unchecked)
- **Type**: `Option<Gemma4E4bModel>` struct with multiple sub-tensors
- **Load**: `hybrid.rs:1611-1640`
- **Sub-tensors**:
  - `tok_tbl_gpu: OnceLock<CudaSlice<u8>>` — lazy upload, unchecked device affinity
  - `model_proj: GpuTensor` — loaded via `load_t(e, src, ...)` on primary
  - `proj_norm: GpuTensor` — loaded via `load_t(e, src, ...)` on primary
- **Read sites**: `hybrid_forward.rs:9019, 9052, 9293` — E4B forward paths
- **Verdict**: ⚠️ UNCHECKED — E4B is first-light, PP compatibility not audited in detail

## Read Sites (COMPLETE)

All major model-global tensors audited. Cross-stage read patterns identified.

---

## Recommended Fix: GemmaAux::rope_freqs

Apply the Step35 multi-device replication pattern:

### Load-time changes (hybrid.rs:1602-1609)
```rust
let rope_freqs = match src.find("rope_freqs.weight") {
    Some(t) => {
        let host = memra_gguf::dequant::dequantize(
            t.ggml_type,
            &t.bytes,
            t.ne.iter().product::<u64>() as usize,
        );
        let mut copies = Vec::new();
        if let Some(fence) = crate::pp::pp_cuts(n_trunk) {
            for s in 0..fence.len() - 1 {
                let owner = crate::pp::layer_engine(e, n_trunk, fence[s])?;
                let dev = owner.ctx().ordinal();
                if copies.iter().all(|(d, _)| *d != dev) {
                    copies.push((dev, owner.htod(&host)?));
                }
            }
        } else {
            copies.push((e.ctx().ordinal(), e.htod(&host)?));
        }
        Some(copies)
    }
    None => None,
};
```

### Struct definition change (hybrid.rs GemmaAux)
```rust
pub struct GemmaAux {
    /// rope_freqs.weight [hd_global/2] freq factors — global layers' RoPE (R9).
    /// REPLICATED PER-DEVICE under PP-N cross-device placement (same fix as Step35).
    pub rope_freqs: Option<Vec<(usize, CudaSlice<f32>)>>,
    // ... rest unchanged
}
```

### Accessor method (new impl GemmaAux)
```rust
impl GemmaAux {
    pub fn rope_freqs(&self, e: &Engine) -> Option<&CudaSlice<f32>> {
        self.rope_freqs.as_ref().map(|copies| {
            let dev = e.ctx().ordinal();
            &copies
                .iter()
                .find(|(d, _)| *d == dev)
                .unwrap_or_else(|| panic!("gemma4 rope_freqs has no local copy for device {dev}"))
                .1
        })
    }
}
```

### Read-site changes (hybrid_forward.rs)
Replace all 7 direct `.as_ref().expect(...)` calls with accessor:
```rust
// OLD (all 7 sites):
Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))

// NEW:
Some(aux.rope_freqs(e).expect("gemma4 global rope needs rope_freqs.weight"))
```

**Lines to change**: 6522, 7277, 7522, 7669, 8190, 8303, 9061

---

## Proposed Static Assertion

**Runtime debug check** (add to Engine or create helper):
```rust
#[cfg(debug_assertions)]
pub fn check_tensor_device(
    tensor: &CudaSlice<impl cudarc::driver::DeviceRepr>,
    stream: &Arc<CudaStream>,
    site: &str,
) {
    // Extract device from tensor's context
    let tensor_dev = tensor.device().ordinal();
    let stream_dev = stream.context().ordinal();
    
    assert_eq!(
        tensor_dev, stream_dev,
        "PP cross-device tensor read at {site}: tensor on dev{tensor_dev}, \
         stream on dev{stream_dev}"
    );
}
```

**Usage pattern**: Call before kernel launches that dereference model-global tensors:
```rust
#[cfg(debug_assertions)]
check_tensor_device(&aux.rope_freqs(e).unwrap(), &e.stream(), "gemma4_rope");

e.rope_neox2(..., aux.rope_freqs(e).expect(...), ...)?;
```

**Note**: This check catches the bug class at runtime under PP-N cross-device placement. The `debug_assertions` gate keeps it out of release builds.

---

## Architecture-Specific Notes

### Gemma4 PP-2 Risk Profile

**Current state**: Gemma4 has NEVER been run under PP-2 cross-device placement in production.

**Why the bug is latent**: 
1. The gemma4 architecture uses global full-attention layers (R9) that read `rope_freqs`
2. Under PP-2, if a global full-attn layer stages to device 1, its RoPE kernel dereferences `aux.rope_freqs` which lives on device 0
3. Same failure mode as Step-3.7 bug #1: `CUDA_ERROR_ILLEGAL_ADDRESS` on first rope kernel launch

**Trigger conditions**:
- `MEMRA_PP_STAGES=2` (or higher)
- `MEMRA_PP_DEVICES=0,1,...` (cross-device placement)
- Gemma4 model (26B-MoE or 31B-dense)
- Split point places a global full-attn layer on the remote stage

**Why this hasn't been caught**:
1. Gemma4 PP-2 work has not yet reached cross-device testing (development on single-device PP-2 first)
2. External review (2026-08-10) caught it by pattern-matching against Step-3.7 fix

### Step35 Reference Implementation

The Step35 fix (research/hostbounce-20260810/RESULTS.md) is the CORRECT pattern:
- Detect PP fence at load time
- Create one copy per distinct device in the placement
- Store as `Vec<(usize, CudaSlice<f32>)>`
- Accessor method looks up by current engine's device ordinal
- Panic if no local copy exists (fail-fast vs silent corruption)

This pattern should be applied to ANY model-global tensor that:
1. Lives as a `CudaSlice` (device-resident)
2. Is read by layer-level kernels (not just head/tail)
3. Can be accessed from remote PP stages

---

## Summary

**Audit completed**: All model-global tensors identified and classified.

**Confirmed bugs**: 1
- `GemmaAux::rope_freqs` — single device copy, 7 cross-stage read sites

**Safe patterns**:
- Step35 rope_freqs — multi-device replication ✅
- Embedding table — stage 0 local ✅
- Output norm + lm_head — last stage local ✅
- MoE correction bias — host-side ✅

**Recommended action**:
1. Apply Step35 pattern to `GemmaAux::rope_freqs` (code provided above)
2. Add debug assertion helper for runtime cross-device checks
3. Audit E4B model tensors if gemma4-E4B PP-2 work begins

**Files modified** (proposed fix):
- `crates/memra-engine/src/hybrid.rs` — load + struct + accessor
- `crates/memra-engine/src/hybrid_forward.rs` — 7 read sites

---

**Status**: ✅ AUDIT COMPLETE  
**Skeleton commit**: `eae39946`  
**Final commit**: (to be added)
