# MoESD target-efficiency instrumentation rider — design

Date: 2026-08-11

Lane: read-only design study (no code edits, no GPU runs)

Status: SKELETON — will be refined in committed increments

## Executive summary

This document specifies the instrumentation harness to measure whether **batch-amortized verify** justifies reopening the PP-2 speculative decode verdict. The closed PP-2 receipts (research/specpp2-20260810) show verify = 95% of round time at c=1, K=1 loses 18.8%. MoESD (arXiv 2505.19645, NeurIPS 2025) proposes that at HIGH batch, MoE verify columns collapse in marginal cost because expert unions saturate — delivering up to 2.29x on Qwen2-57B-A14B. 

The question this harness answers: at Step-3.7 PP-2 serving shape and c=8..32, does T_T(B,gamma) show near-free verify columns, or does the closed K=0 verdict remain correct at all widths?

If target efficiency shows genuine amortization AND projected acceptance*gamma > serial throughput, the trained-head (DSpark) lane gains a second venue. Otherwise, the next dollar goes to DSpark acceptance training alone.

## (a) Measurement matrix

**Model**: Step-3.7-Flash IQ4_XS + external MTP Q8_0 (the serving shape from research/newboxgates-20260811)

**Rig**: box1 (2x RTX PRO 6000 Blackwell Server, the Vast verification box)

**Configuration**: PP-2 serve shape (MEMRA_PP_STAGES=2, MEMRA_PP_DEVICES=0,1, MEMRA_CTX=262144, MEMRA_MOE_GROUPED=1, MEMRA_PREFILL_TICK=2048)

**Matrix dimensions**:
- B in {1, 2, 4, 8, 16, 24, 32} — batch sizes from solo to high-concurrency serving
- gamma in {1, 2, 3, 4, 6, 8} — verify column counts (gamma=1 is plain decode baseline)

**Per cell, N=5 interleaved repetitions** (mirroring research/newboxgates-20260811/RESULTS.md protocol):
1. Wall ms/step — median end-to-end latency across N=5
2. Expert-union size per layer — how many DISTINCT experts are activated across all B*gamma tokens in that forward pass
3. Effective tok/s if all gamma columns were accepted — (B * gamma) / (ms/step / 1000)
4. Realistic tok/s under measured acceptance — effective_toks * acceptance_rate (using Step-3.7 K=1..8 per-position acceptance from research/specpp2-20260810/RESULTS.md §K-sweep)

**Expert-union counting**: The router selects K experts per token (Step-3.7 has topk routing). For a batch of B sequences with gamma tokens each, the union is the DISTINCT expert IDs selected across all B*gamma routing decisions at that layer. Instrument via: during decode_batch_layers (decode_batch.rs), accumulate the router's `sel` arrays per layer (the `moe_fwd_topk` output contains expert indices per token); emit per-layer union cardinality to JSONL.

**Thermal regime**: Continuous NVML sampling at 500ms intervals, same as newboxgates protocol.

## (b) Decision rule (stated BEFORE measurement)

The MoESD paper's break-even is where T_T(B,1) / T_T(B,gamma) justifies the acceptance cost. Propose:

**X = 1.5** (target efficiency threshold): gamma-column verify must be ≤ 1/1.5 = 67% the cost of gamma serial steps

**Y = 8** (minimum batch size): amortization must hold at serving-relevant concurrency (c=8 is where plain batching already wins 1.73x per research/newboxgates-20260811)

**Decision**: IF target_efficiency(B=Y, gamma=4) > X AND projected_toks_realistic(B=Y, gamma=4) > measured_plain_toks(B=Y), THEN batch-verify amortization is live and DSpark training gains PP-2 as a second venue. OTHERWISE, K=0 stays correct and the next dollar goes to DSpark acceptance-only.

**Rationale**: gamma=4 corresponds to K=3 speculative depth (the measured profitable K on single-card, research/spec-landscape-20260810/SURVEY.md). B=8 is the crossover where plain batching dominates on PP-2. If amortization fails there, it won't save spec at any serving load.

## (c) Harness shape

**Standalone bin** (like research/optipipe-20260810 pattern, NOT run-spec extension).

**Why standalone**: This measures T_T(B,gamma) as a serving-load simulation, not a correctness gate. run-spec is a self-consistency oracle; extending it would conflate measurement with gating. serve-layer measurement would require server changes and N=5 interleaving discipline across real traffic.

**Harness structure**:
- Bin: `crates/memra-tools/src/bin/moesd-gate.rs` (new)
- Shell driver: `research/moesd-harness-20260811/run-box1.sh` (interleaves N=5, holds flock, emits JSONL)
- Input: same model paths as newboxgates (Step-3.7 IQ4_XS + MTP Q8_0)
- Output: per-cell JSONL rows with fields: `{run, B, gamma, ms_step, layers: [{layer_id, union_size, router_k}], effective_toks, realistic_toks, thermal_max_C, thermal_max_W}`

**Exact counters to emit**:
1. Wall ms/step — time the decode_batch_layers call (or equivalent multi-sequence forward)
2. Per-layer expert-union size — count distinct expert IDs in the concatenated `sel` arrays for that layer's B*gamma tokens
3. Router K — Step-3.7's topk parameter (static per model, but emit for reproducibility)
4. Effective tok/s — (B * gamma) / (ms_step / 1000)
5. Realistic tok/s — effective * acceptance_proxy (where acceptance_proxy uses the K-sweep per-position acceptance from specpp2-20260810 for gamma-1 positions)

**Lock discipline**: Hold `/tmp/memra-gpu.lock` for entire run (exclusive), release at 0 MiB, verify no competing process before/after (same as newboxgates).

**N=5 interleaving**: Forward/reverse alternating order by repetition (run 1: B ascending gamma ascending, run 2: B descending gamma descending, ...). Exclude one warmup per boot.

## (d) Cost estimate

**Total cells**: 7 B values * 6 gamma values = 42 cells

**N=5 repetitions**: 42 * 5 = 210 measurement points

**Per-point estimate**: ~1-5 seconds per forward pass (based on newboxgates decode cells: c=1 98.3 tok/s ≈ 10ms/tok, c=8 173.6 tok/s ≈ 5.8ms/tok; B*gamma in-flight tokens range 1..256)

**GPU-hours**: Pessimistic (5s/point * 210 points) = 1050s ≈ 0.3 hours. Realistic (accounting for boot/setup overhead, thermal settling): **~1 GPU-hour on box1**.

**Rig**: box1 or Vast 47297516 (both are 2x RTX PRO 6000 pairs; box1 preferred per standing verification-rig policy).

## (e) What it must NOT do

1. **No serving default changes**: This is measurement-only. No runtime flags, no admission policy changes, no default-on paths.

2. **No verdict reopening**: The closed PP-2 spec verdicts (research/specpp2-20260810, research/specmech-20260810) remain closed. This harness measures WHETHER a future mechanism (trained heads + batched verify scheduler) could change the answer. It does not re-litigate the existing K=0 policy.

3. **No runs without flock**: Every invocation must hold `/tmp/memra-gpu.lock` exclusively. Thermal/clock drift from concurrent compute invalidates comparisons (per evidence discipline in CLAUDE.md).

4. **No live serving traffic**: Standalone bin only. Do not instrument the production server or run this during soak.

5. **No expert-routing changes**: Measure the EXISTING router's selections. Do not mask, prune, or override expert IDs. The union size is an OBSERVATION, not a knob.

## Implementation notes

### Instrumentation hooks: CHOSEN PATH = spec.rs verify tier

**Selected approach**: Instrument the EXISTING `decode_step_t` verify path (crates/memra-engine/src/spec.rs) rather than decode_batch.rs. Rationale:
1. `decode_step_t` already implements T-column batched verify (T=K+1, K up to 8 per MEMRA_SPEC_CAPMAX)
2. The verify path already runs MoE layers with the router's device-side topk, emitting `sel_d: CudaSlice<i32>` (shape `[t, n_used]`) per layer
3. No new launch structure needed — only add a D2H readback of `sel_d` per layer, count distinct IDs host-side, emit to telemetry

**Router output format** (lib.rs:2609-2622, hybrid_forward.rs:4833):
- `e.moe_router_topk(logits, t, n_expert, n_used)` returns `(sel_d: CudaSlice<i32>, w_d: CudaSlice<f32>)`
- `sel_d` shape: `[t, n_used]` — for t tokens, each has n_used selected expert IDs (i32)
- This is the DEVICE allocation; for instrumentation, issue a D2H copy after each MoE layer's routing

**Per-layer expert-union collection**:
- After the router call in the MoE FFN path (e.g., hybrid_forward.rs:4833 or the verify trunk's equivalent), read back `sel_d` to host: `let sel_h = e.dtoh(&sel_d)?;`
- Collect distinct expert IDs: `use std::collections::HashSet; let union: HashSet<i32> = sel_h.iter().copied().collect(); let union_size = union.len();`
- Accumulate per-layer union sizes in a per-step vector, emit to JSONL at step completion

**Exact hook point** (spec.rs verify path):
- The verify trunk runs through `HybridModel::decode_layers_t` or similar (must audit spec.rs for the exact call site)
- Each MoE FFN layer's router produces `sel_d` — hook AFTER that router call, BEFORE the expert matmuls
- Add a conditional telemetry branch: `if MEMRA_MOESD_TELEM=1 { let sel_h = e.dtoh(&sel_d)?; record_union(il, sel_h, t, n_used); }`

**Exact instrumentation points** (file:line citations):

1. **crates/memra-engine/src/spec.rs:2330** — `decode_step_t_h_emb_dev` is the device-logits verify entry point
   - This calls the trunk's layer loop (likely through HybridModel methods)
   - Each MoE FFN layer produces `sel_d` from `e.moe_router_topk(logits, t, n_expert, n_used)` (hybrid_forward.rs:4833)
   
2. **Add telemetry struct to spec.rs** (after SpecTelemetry at line 279):
   ```rust
   #[derive(Clone, Debug)]
   pub struct MoESDTelemetry {
       pub layers: Vec<MoESDLayerUnion>,
   }
   
   #[derive(Clone, Debug)]
   pub struct MoESDLayerUnion {
       pub id: usize,           // absolute layer index
       pub union: usize,        // distinct expert count
       pub n_expert: usize,     // total bank size (288)
       pub n_used: usize,       // topk parameter (8)
   }
   ```

3. **Hook into MoE FFN path** (hybrid_forward.rs or via spec.rs wrapper):
   - After `let (sel_d, w_d) = e.moe_router_topk(logits, t, n_expert, n_used)?;` (hybrid_forward.rs:4833)
   - Add conditional telemetry branch:
   ```rust
   if let Some(telem) = moesd_telemetry.as_mut() {
       let sel_h: Vec<i32> = e.dtoh(&sel_d)?;
       let union: HashSet<i32> = sel_h.iter().copied().collect();
       telem.layers.push(MoESDLayerUnion {
           id: il,
           union: union.len(),
           n_expert,
           n_used,
       });
   }
   ```
   - Threading: pass `&mut Option<MoESDTelemetry>` through decode_step_t_h_emb_dev → layer loop → MoE FFN

4. **crates/memra-tools/src/bin/moesd-gate.rs** — NEW standalone bin:
   ```rust
   // Parse --B=X --gamma=Y --run=N from args
   // Load Step-3.7 IQ4_XS + MTP Q8_0 (same paths as newboxgates)
   // Set MEMRA_MOESD_TELEM=1 env internally (or pass flag)
   // Create B caches, prime with B unique prompts (or repeat same prompt)
   // Call model.decode_step_t_h_emb_dev with t=gamma, collect MoESDTelemetry
   // Time the forward pass, compute metrics, emit JSONL line to stdout
   ```

**Cost per measurement point**:
- One D2H copy per MoE layer per verify step: Step-3.7 has ~60 layers (half are MoE FFN), so ~30 D2H copies of `[t, n_used]` i32 arrays
- At t=8 (gamma=8), n_used=8: 8*8*4 = 256 bytes per layer → 30 layers * 256 bytes ≈ 7.7 KB per step
- D2H bandwidth cost negligible vs the ~10-50ms verify step itself (PCIe gen4 x16 = ~30 GB/s, 7.7KB = 0.26 μs)
- Total overhead: ~30 * 0.26μs ≈ 8μs per step (0.02% of a 50ms step)

### Acceptance proxy and realistic throughput projection

Step-3.7 K-sweep per-position acceptance (research/specpp2-20260810/RESULTS.md §K-sweep):
- K=1: pos[0]=0.737 → tokens/round 1.74
- K=2: pos[0]=0.655, pos[1]=0.388 → tokens/round 2.04
- K=3: pos[0]=0.676, pos[1]=0.384, pos[2]=0.048 → tokens/round 2.11

**Projected acceptance per gamma** (for realistic tok/s):
- gamma=1 (plain decode): acceptance=1.0 (baseline)
- gamma=2 (K=1): acceptance = 0.737 (pos 0 only)
- gamma=3 (K=2): acceptance = (0.655 + 0.388) / 2 = 0.5215 → effective 1.56 accepted per round
- gamma=4 (K=3): acceptance = (0.676 + 0.384 + 0.048) / 3 = 0.369 → effective 1.48 accepted per round
- gamma=6 (K=5): pessimistic, use K=3 tail = 0.048 for pos 3-5, so (0.676 + 0.384 + 0.048 + 0.048 + 0.048) / 5 = 0.241
- gamma=8 (K=7): pessimistic tail = 0.048 for pos 3-7, so ≈ 0.177 average acceptance

**Formula**: realistic_toks = (B * sum_over_gamma(acceptance[pos])) / (ms_step / 1000)

Where acceptance[pos] comes from the K-sweep table, with tail positions beyond K=3 assumed at 0.048 (the measured K=3 pos-2 value, pessimistic floor).

**Target efficiency** (MoESD paper metric):
- target_eff(B, gamma) = T_T(B,1) / T_T(B,gamma)
- If target_eff > 1.5, then gamma columns cost < 67% of gamma serial steps → amortization is real

### JSONL schema (finalized)

Per measurement point (one cell in the B x gamma matrix, one repetition):

```jsonl
{"run":1,"B":8,"gamma":4,"ms_step":35.42,"layers":[{"id":0,"union":96,"n_expert":288,"n_used":8},{"id":1,"union":102,"n_expert":288,"n_used":8},...],
 "effective_toks":903.6,"accepted_toks":333.3,"realistic_toks":226.0,"target_eff":1.38,
 "thermal_max_C":[56,57],"thermal_max_W":[511,558],"thermal_avg_C":[54,55],"config":"PP2"}
```

**Field definitions**:
- `run`: repetition number (1..5)
- `B`: batch size (sequences in flight)
- `gamma`: verify columns (tokens per sequence in this forward)
- `ms_step`: wall time for the batched verify forward (ms)
- `layers`: array of per-layer MoE routing observations:
  - `id`: absolute layer index (0-based)
  - `union`: count of DISTINCT expert IDs selected across all B*gamma tokens at this layer
  - `n_expert`: total expert bank size (Step-3.7 = 288 per layer)
  - `n_used`: router's topk parameter (Step-3.7 = 8 experts per token)
- `effective_toks`: (B * gamma) / (ms_step / 1000) — throughput if all columns were accepted
- `accepted_toks`: sum over gamma positions of B * acceptance[pos] — expected accepted tokens per round under measured acceptance
- `realistic_toks`: accepted_toks / (ms_step / 1000) — projected throughput under measured acceptance
- `target_eff`: ms_step(B, gamma=1) / ms_step(B, gamma) — the MoESD metric (> 1.0 means amortization; > 1.5 is decision threshold)
- `thermal_max_C` / `thermal_max_W`: peak sampled GPU temps/powers during this measurement
- `thermal_avg_C`: average sampled temps (for steady-state verification)
- `config`: serving configuration tag (always "PP2" for this study)

**Summary row** (emitted once per (B, gamma) cell after N=5 reps):

```jsonl
{"summary":true,"B":8,"gamma":4,"N":5,"ms_step_median":35.42,"ms_step_range":[35.1,35.9],
 "union_median":[{"id":0,"union":96,"range":[94,98]},...],"target_eff_median":1.38,
 "realistic_toks_median":226.0,"verdict":"amortized"}
```

Where `verdict` is:
- `"amortized"` if target_eff_median > 1.5
- `"marginal"` if 1.2 < target_eff_median <= 1.5
- `"serial"` if target_eff_median <= 1.2 (no amortization, verify columns cost near-full steps)

### Shell harness structure (run-box1.sh)

**Lock protocol** (mirrors research/newboxgates-20260811/run-box1.sh and research/p0iso-20260810/run-box1.sh):
```bash
#!/usr/bin/env bash
set -euo pipefail

# Acquire exclusive GPU lock
exec 200>/tmp/memra-gpu.lock
flock -x 200 || { echo "lock failed"; exit 1; }

# Verify no competing processes
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > before.txt
[ -s before.txt ] && { echo "GPU busy"; exit 1; }

# N=5 interleaved repetitions
for run in 1 2 3 4 5; do
  # Alternating order: odd runs ascending, even runs descending
  if [ $((run % 2)) -eq 1 ]; then
    B_ORDER="1 2 4 8 16 24 32"
    GAMMA_ORDER="1 2 3 4 6 8"
  else
    B_ORDER="32 24 16 8 4 2 1"
    GAMMA_ORDER="8 6 4 3 2 1"
  fi
  
  for B in $B_ORDER; do
    for gamma in $GAMMA_ORDER; do
      # One warmup on first cell of first run only
      if [ $run -eq 1 ] && [ "$B" = "1" ] && [ "$gamma" = "1" ]; then
        ./moesd-gate --warmup --B=$B --gamma=$gamma
      fi
      
      # Measurement with NVML sampling
      nvidia-smi dmon -s pucvmet -c 999999 -d 500 > thermal_${run}_${B}_${gamma}.csv &
      DMON_PID=$!
      
      ./moesd-gate --B=$B --gamma=$gamma --run=$run >> raw_${run}.jsonl
      
      kill $DMON_PID
      wait $DMON_PID 2>/dev/null || true
    done
  done
done

# Emit summary rows per (B, gamma) cell
python3 compute_summary.py raw_*.jsonl > RESULTS.jsonl

# Verify clean release
nvidia-smi --query-compute-apps=pid --format=csv,noheader > after.txt
[ -s after.txt ] && { echo "WARNING: GPU not released"; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | grep -q "^0$" || echo "WARNING: VRAM not at 0 MiB"

# Release lock
flock -u 200
```

**moesd-gate bin interface**:
```
./moesd-gate --B=<batch_size> --gamma=<verify_cols> [--run=<N>] [--warmup]
  --B: batch size (1..32)
  --gamma: verify columns (1..8)
  --run: repetition number (1..5), for JSONL output
  --warmup: skip JSONL emission, used for first-cell thermal settling
  
Environment (matches newboxgates):
  MEMRA_PP_STAGES=2
  MEMRA_PP_DEVICES=0,1
  MEMRA_CTX=262144
  MEMRA_MOE_GROUPED=1
  MEMRA_PREFILL_TICK=2048
  MEMRA_MOESD_TELEM=1  (enables expert-union telemetry in verify path)
  
Output: one JSONL line per invocation to stdout
```

**compute_summary.py**: reads raw_*.jsonl (N=5 reps * 42 cells = 210 lines), groups by (B, gamma), emits median + range + verdict per cell.

---

## Final decision logic

**Inputs** (from N=5 measurements at B=8, gamma=4):
1. `ms_step_median(B=8, gamma=4)` — median verify time for 8 sequences * 4 columns
2. `ms_step_median(B=8, gamma=1)` — median plain decode time for 8 sequences * 1 token (baseline)
3. `target_eff = ms_step(8,1) / ms_step(8,4)` — amortization ratio
4. `realistic_toks_median(8,4)` — projected throughput under measured acceptance
5. `measured_plain_toks(B=8)` — from research/newboxgates-20260811/RESULTS.md: **173.62 tok/s** (c=8 median)

**Decision tree**:
```
IF target_eff(8,4) > 1.5:
  # Verify columns cost < 67% of serial → amortization is REAL
  IF realistic_toks(8,4) > 173.62:
    VERDICT = "GO: batch-verify amortization is live at c=8; DSpark training gains PP-2 as second venue"
  ELSE:
    VERDICT = "HOLD: amortization exists but acceptance is too weak; focus DSpark acceptance training first, revisit after trained head"
  END
ELSE:
  VERDICT = "CLOSED: no amortization at c=8; K=0 stays correct at all batch sizes; next dollar goes to DSpark acceptance-only"
END
```

**Why B=8, gamma=4**: 
- B=8 is where plain batching already wins 1.73x (98.30 → 173.62 tok/s per newboxgates)
- gamma=4 corresponds to K=3 (the measured profitable K on single-card per spec-landscape)
- If amortization fails there, it won't save spec at higher concurrency (c≥8 is the serving target)

**Why target_eff > 1.5**:
- MoESD paper shows 2.29x on Qwen2-57B-A14B (near-free verify columns when expert unions saturate)
- Conservative threshold: 1.5x means 4 columns cost ≤ 2.67 serial steps (33% overhead per column)
- Below 1.5x, the marginal column cost is too high to justify the complexity of batched-spec scheduler + trained heads

---

## Cost and schedule

**GPU-hours**: ~1 hour on box1 (210 points * 1-5s/point + boot/setup overhead)

**Wall-clock**: ~2 hours (including lock acquisition, thermal settling between runs, summary computation)

**Rig**: box1 (2x RTX PRO 6000 Blackwell Server, the Vast verification rig per RIG DOCTRINE in MEMORY.md)

**Blocking dependencies**:
1. Owner approval of decision rule (X=1.5, Y=8, gamma=4 as pivot cell)
2. Owner confirmation that this study's NO-GO verdict (if K=0 stays correct) will NOT reopen the closed PP-2 spec verdicts

**Non-blocking follow-on** (if GO verdict):
1. Implement stage-resident multi-session pipeline (research/specmech-20260810 mechanism bill)
2. Implement DSpark-style trained head (research/dspark-plan-20260811)
3. Re-measure PP-2 c=2/c=8 with trained head + batched verify scheduler
4. Gate battery: run-spec K=1..8, serve-stress, N=5 A/B vs plain

---

**Correctness dependencies**: None (read-only study, no runtime changes, no serving defaults altered)

**Implementation readiness**: Design complete, ready for owner review and GPU-time approval
