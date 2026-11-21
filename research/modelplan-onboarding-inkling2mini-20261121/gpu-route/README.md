# Inkling 2 Mini: native GPU route, checkpoint gates (2026-11-21)

The route is `crates/memra-engine/src/inkling2_gpu.rs` with the kernels of `cu/inkling2_gpu.cu` and the
act-quant expert kernels of `cu/inkling2_actq.cu` (memra#1396). Runner: `ink2-gpu`; kernel cells:
`ink2-gpu-gate` (also in `kernel-check`). The oracle tapes are scratch evidence outside the repo
(transformers PR 51050 model code on CPU, the checkpoint's expert bytes dequantized exactly; the act-quant
arm is a CPU simulation of the declared program). Each directory holds the batch script, its log, the
receipts and the sha256 of the binaries that ran. `b6/` (the #1397 review fixes) and `b7/` (p3 against the
f64 oracle) are the current float-class evidence; `b8/` is the current declared-class evidence, after the
merge of #1396's review fixes (e6f9dc4e40: exact SiTU through f64 tables, the W8A8 chain model for
subnormals). `b3/` to `b5/` are kept as the record they replaced.

Card: RTX 5090 Laptop GPU, 24 GB, sm_120a, driver 615.71.09. Weights resident as stored: 17.594 GiB
(BF16 trunk, NVFP4 experts on layers 0 to 19, MXFP8 on 20 to 23). The drafter file is read through the
pack's auxiliary files and its 54 tensors are bound by the contract, not uploaded. Vision is not loaded.
Numeric class labels: the receipts carry `expert_activation_class` (the plan's class of the routed
experts) and `stream_outside_experts: f32`. Outside the experts this route runs an f32 stream in both
classes, which is what the plan's declared class now names (`model_plan/inkling2.rs`).

## Verdicts (b6)

| Gate | Class | Result | Receipt |
|---|---|---|---|
| Kernel cells (18) | both | ALL OK: GEMM / GEMV over BF16, NVFP4, MXFP8; embed and norms; depth mix; controller; head epilogues; static and dynamic conv with carried state; sliding, full and full-masked attention, plus `ink2g-attn-full-long` (2600 keys, three key tiles); router ties; SwiGLU, sums, softcap head; the declared expert block bit for bit (2304/2304 bf16 outputs, NVFP4 and MXFP8) with its two red arms; load refusals | `b6/gate.log` |
| Tensor contract | both | bound over 1170 census tensors (the shard and `mtp.safetensors`) before upload; red arms refused: an extra tensor, a missing one, a reshaped one, a retyped one, a loader read of an unbound name | `ink2-gpu --contract-check` (CPU; output in the PR) |
| Loader debug assertions | declared | the debug build loads every tensor and runs 2 tokens, 2/2 greedy | `b6/debug-load.log` |
| p1 (26-token chat, 57 positions) vs `exact_f32` | float | worst slot residual rel L2 8.2e-7, logits KL mean 2.0e-12, argmax 57/57, greedy 32/32 | `b6/p1-float.json` |
| p2 (1500 tokens, past the 1024 window and extent) vs `exact_f32` | float | worst slot 8.4e-4, KL mean 2.2e-7, argmax 1531/1531, greedy 32/32 | `b6/p2-float.json` |
| p2 with `log_scaling_n_floor = 512` (config-only snapshot, `make_floor_snapshot.sh`) vs `p2_floor512.exact_f32` | float | the positive log-scale cell: tau on every full layer for positions 512 to 1531; worst slot 3.8e-6, KL 4.4e-12, argmax 192/192, greedy 32/32 | `b6/p2floor512-float.json` |
| p3 (8260 tokens, crosses the 8192 floor) vs `exact_f32` | float | worst slot 0.11, KL mean 8.4e-4, argmax 178/179, greedy 32/32; the drift is the f32 oracle's (next section) | `b6/p3-float.json` |
| p2 vs the f64 oracle (288 kept positions) | float | worst slot 2.8e-6, KL 1.9e-12, router rows 72000/72000, argmax 288/288 | `b8/p2-float-vs-f64.json` |
| p3 vs the f64 oracle | float | worst slot 2.1e-6 (2.2e-6 after attention), KL mean 8.2e-13, router rows 396480/396480, argmax 148/148: the route matches the f64 program on every slot | `b7/p3-float-vs-f64.json` |
| Per MoE with forced inputs vs `actq_f32` | declared | after the #1396 merge: p1 42 of 48 slots bit-equal (41 before), worst 6.2e-4; p2 worst 6.8e-4 (4.7e-4 before), median 1.1e-4; under forcing KL 2.4e-8 / 2.1e-8, greedy 32/32. The tape is a CPU simulation with library tanh and sigmoid in SiTU; the route's SiTU is now the exact one, so the p2 slots that were bit-equal now differ by one bf16 step in places | `b8/p1-declared-forced-moe.json`, `b8/p2-declared-forced-moe.json` |
| End to end vs the float program | declared | after the #1396 merge, KL mean against the f32 oracle p1 0.050 (0.045 before), p2 0.388 (0.388); against the f64 oracle p2 0.098, p3 0.026 (0.039 before); argmax 50/57, 1339/1531, 268/288, 144/148; teacher-forced greedy 31/32 and 32/32. The CPU simulation of the same program reads 0.062 / 0.43 / 0.023 against the f32 oracle | `b8/p1-declared-vs-float.json`, `b8/p2-declared-vs-float.json`, `b8/p2-declared-vs-f64.json`, `b8/p3-declared-vs-f64.json` |
| Chunking invariance (prefill chunks vs one token per call) | both | float 26 vs 1 and 512 vs 300, declared 26 vs 1 and 512 vs 300 (after the merge): 0 slot rows, 0 logit rows differ | `b6/p1-float.json`, `b6/p2-float.json`, `b8/p1-declared.json`, `b8/p2-declared-vs-float.json` |
| Atomic forward: a decode call that fails after a slot, then the rest of the request | both | float (fails after slot 20, `b6/`) and declared (after slot 33, `b6/` and again `b8/`): 0 of 2784 slot-output and router-pick digests and 0 of 29 logit rows differ from a clean run; the failed call leaves the position unchanged | `b6/atomic-float.json`, `b6/atomic-declared.json` |
| Greedy decode on the device argmax (run-gen form) | float | 32/32 equal to the oracle's greedy ids, three takes | `b6/gen-float-*.json` |
| Admission | declared | context 1,048,576 is refused before the cache is allocated: 64 GiB of full-attention cache against 4.61 GiB driver free; a context of 71,680 loads and decodes | `b6/context-refusal.log`, `b6/context-near-max.log` |

The tiled attention and the double-buffered conv history compute the same program as before: every
score field of the b6 receipts (per-slot residual errors, router rows, KL, argmax, greedy) equals the
b3 / b4 / b5 receipt of the same cell, on p1, p2 and p3 in the float class and p1 / p2 in the declared
class.

Red arms (the gate must reject each):

| Arm | Against | Result |
|---|---|---|
| `dequant-bf16` (expert weights cast to BF16, the PR's stock load) | p1 / p2 `exact_f32` | worst slot 2.2e-2 / 0.33, KL 2.1e-4 / 0.21, against 8.2e-7 / 8.4e-4 for the route (`b3/`, `b4/p2-float-red-dequant.json`). On a declared plan the arm is refused (`b6/red-refusal.log`) |
| `conv-state` (the short convs drop their carried state between calls) | p1 `exact_f32` | worst slot 0.35, KL 2.3e-2, argmax 55/57 (`b3/`); the kernel cells move 6144/11264 outputs |
| `commit-on-error` (a failed call commits the conv histories it wrote, the pre-fix program) | p1, float | 2141 of 2784 digests and 29 of 29 logit rows differ from the clean run (`b6/atomic-float-red.json`) |
| `ties-high` (a router tie to the higher expert) | p1 `exact_f32` | not separated on the tapes, which hold no exact tie; the constructed-tie cell `ink2g-route` separates it (3/3 tied rows moved, no untied row) |
| wrong log-scale floor (the route at the shipped floor 8192 against the `p2_floor512` tape) | p2_floor512 `exact_f32` | slots 0 to 4 equal, slot 5 (the first full-attention layer) jumps to 1.7e-2, KL 8.6e-2 (`b5/`) |
| declared MoE: routing weight after the down projection | kernel cell | moves 1660/2304 (NVFP4) and 2212/2304 (MXFP8) bf16 outputs |
| declared MoE: the token row quantized at the token's index (the pre-fix program) | kernel cell | moves 2304/2304 and 2303/2304 |
| forced MoE input on a tape that keeps a subset of positions | p3 `actq_f32` | the runner refuses (179 of 8291 positions); `b4/p3-declared-forced-moe.json` ran unforced on most chunks and is void |

## The p3 drift is the f32 oracle's

The float route drifts from the f32 oracle on p3 from slot 11 (pass 1, a full-attention layer) on:
5.2e-5 against 8.6e-7 at slot 10, 4.9e-4 at slot 17, up to 0.11 on the last slots, router rows 93.7%
equal. Three measurements place it:

1. The oracle's own chunking variance (`p3-drift/`, same f32 program, prefill chunks of 2800 vs 2048,
   on the dev desk): 1e-6 on slots 10 to 17, 2 to 3% on the last slots, KL 1.4e-4. Chunking does not
   change the order of a query's key reduction, so this bounds only part of the oracle's error.
2. The f64 oracle (the same model code in f64, chunks of 1024; a first f64 run at chunks of 2800 was killed on the dev desk, bash reporting `Killed`, cause not captured, while it shared the host with the f32 run): the f32 oracle against it shows the
   route's pattern exactly: 5.7e-5 at slot 11, 5.2e-4 at slot 17, 0.06 to 0.12 on the last slots, KL
   9.5e-4, router rows 93.7% equal (both f32 chunkings read the same). The f32 oracle is the program that
   drifts.
3. The route's attention against the host, step by step (`b6/p3-attn-probe.json`): for the last 4
   queries (8257 to 8260 keys) of every full-attention slot (5, 11, 17, 23 and the pass-2 and pass-3
   copies), the route's attention output, recomputed on the host from its own queries, bias inputs, sink
   and K / V caches, equals the f64 reference to 2.4e-8 to 2.7e-8 relative L2 (one f32 rounding). The
   same inputs through an f32 eager program read 4e-6 to 2.1e-5 from f64, at slot 5 as at slot 11.
4. The route against the f64 oracle: worst slot 2.1e-6 over all 48 slots (pass-1 slots 1.6e-7 to 5.3e-7, the last slots 1.3e-6 to 2.0e-6), logits KL mean 8.2e-13, max abs 1.1e-4, every router row and every argmax equal (`b7/p3-float-vs-f64.json`). The route sits 2.1e-6 from the f64 program where the f32 oracle sits 0.12 from it.

The verdict: oracle noise, not a GPU error. The long-context float gate bound comes from the f64 tape:
the p3 float cell reads the f64 tape (`p3-drift/`; the f64 tape rounded once to f32 by `tape_f64_to_f32.py`): worst slot 2.1e-6, KL 8.2e-13. On p2 the f64 tape (`p3-drift/p2var64.sh`) reads 2.8e-6 for the route, against 8.4e-4 for the f32 tape at all 1531 positions. On the f64 tape's 288 kept positions the f32 oracle itself sits within 1.1e-5 of f64, so the f32 tape's 8.4e-4 is at positions the f64 tape does not keep (the long middle and the decoded rows); the f64 tapes are the float gate from here on.

## Memory on 24 GB (f32 KV as shipped in this route)

From `ink2-gpu`'s accounting (`MemoryAccount`, driver figures from `cuMemGetInfo`), chunk 512:

| Item | GiB |
|---|---|
| device total | 23.511 |
| driver free at open (CUDA context counted) | 23.174 |
| resident weights | 17.594 |
| sliding-window rings (40 slots, 2048 rows each) | 0.625 |
| conv histories (two buffers each) | 0.007 |
| chunk activations and head | 0.227 |
| declared expert program buffers | 0.062 |
| driver free before the full-attention caches | 4.612 |

The full-attention caches are the only part that grows with the context: 8 full slots x K and V x
1024 floats = 64 KiB a position. That puts the ceiling at 75,562 positions in the declared class
(76,586 in the float class), against the model's declared 1,048,576. A context of 71,680 loads and
decodes. The KV format stays f32 here: a KV format is a numeric program choice and needs its own per-slot
KL receipt. The context-sized f64 attention scratch of the first version (96 KB a position at chunk 512)
is gone.

## Timed decode (correctness route, not tuned)

Clamp check at the start of the lease: OK (exit 0). Telemetry over the timed window (`b6/smi.csv`,
250 ms; `judge.py`): 173 loaded samples, all P0 at 14001 MHz, none with SW thermal slowdown, 4 with the
SW power cap bit at the transitions; SM 2730-2775 MHz at the end of each take, 74-82 C. Over the whole
batch, 892 of 1760 loaded samples carry SW thermal slowdown (0x20) at up to 91 C, all during the long
untimed cells (p3). The b5 timing window had 428 such samples at 82-86 C; its clamp check returned exit 2
(unsure), and only the telemetry (every loaded sample P0, 14001 MHz) made those takes valid. The batch
scripts now treat exit 2 as not OK unless the telemetry proves full clocks.

N=3 takes per class, interleaved, 32 tokens from the p1 prompt:

| Class | tok/s (3 takes) | launches per token |
|---|---|---|
| float | 4.34, 4.34, 4.33 | 1541 |
| declared | 5.37, 5.36, 5.36 | 1637 |

The figures match b5 within 0.02 tok/s. Prefill is slower: the 8260-token p3 score takes 108.9 s against
72.8 s before, since the attention computes each score three times instead of storing it. Weight bytes a
decode token reads: about 5.55 GB (4.63 GB over the 48 slots plus the 0.93 GB head); at the measured
865 GB/s that bounds decode at 6.4 ms a token. The route runs at 186 ms (declared), 3.4% of that bound.
Tuning is lane P's.

## Declared receipts after the #1396 merge

Every declared-class receipt moved, as expected from the exact SiTU: `b8/` replaces `b6/` (forced MoE p1 and
p2, p1 declared with chunkings) and `b4/` / `b5/` / `b7/` (declared end to end). The kernel cells stay bit
for bit (the declared expert block 2304/2304 against `expert_block_*` with the sm_120a accumulators), the
act-quant gate is ALL OK (48 lines), chunking invariance and atomicity hold, and the float receipts are
unchanged (`b8/p1-float.json` equals `b6/`).

## Routing

A router-scale anomaly (scaling the router weights by 0.85 lowered NLL by 0.28 nats a token) was
traced by lane F to scoring a corpus pasted into the user turn, text the model was not trained to
predict. In distribution (raw text, and the model turn of the chat template) the reference router is at
its optimum and the float reference is calibrated. The route's routing is the reference's, gated on bit
identity to it, and stays as it is.

## Change impact

`tools/change-impact.toml` routes `cu/inkling2_gpu.cu`, `cu/inkling2_actq.cu`, `src/inkling2_gpu.rs` and
`src/inkling2_actq.rs` to the inkling2 pack, which the device battery does not cover: their evidence is
this pack's gates. The push base's rules apply, so the rule takes effect once it is on main. The lane
still reaches every target through `build.rs`, `Cargo.toml`, `lib.rs`, `kernel_check.rs`, the rules
file itself and the memra-reference fixtures from #1396, so landing it needs the sm_120a and sm_121a
cell records.

## Open


- Multi-sequence batching (several requests in one call) is not built; batched-vs-solo equality is
  pending with it. Single-request chunked prefill vs per-token decode is bit-identical.
- The drafter (DSpark) is lane S's; its tensors are bound, not resident.
- Performance is lane P's: fused decode kernels, CUDA graphs, GEMV at the DRAM roofline, attention with
  f32 scores only where a measured KL receipt allows it.
