# MiMo-V2.6 native serving reference

Owner: `codex-mimo-serving-max-20260923`, umbrella issue
`avifenesh/darklanes#1083`. This Memra worktree starts from
`origin/main@db592901708c76dffc3af919aa0d956a4f5880db`. It does not
touch the active MiMo mint issue `avifenesh/memra#623` or PR #629.

Objective: compile and execute a model-specific native reference for the
pinned MiMo-V2.6-Flash-RL minted checkpoint, then run the independent
artifact/model/hardware gates before declaring `NativeQualified`. The
current SGLang fork is a bring-up and measurement instrument. No customer
route or model fact changes in this lane.

Artifacts:

- Source: `XiaomiMiMo/MiMo-V2.6-Flash-RL` revision
  `3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
- Mint: `tiyuvta/MiMo-V2.6-Flash-RL-NVFP4` revision
  `58edbd0c60ace653512b8f423bf41813331ef9e3`.
- Source `config.json` SHA-256:
  `61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621`.
  The exact file was read from the owned research host, and all 65 source
  model-shard hashes match the prior successful mint intake.

Native program requirements from the pinned source:

| Operation | Declared shape |
| --- | --- |
| Layers | 48 trunk layers; full attention at 0, 5, 11, 17, 23, 29, 35, 41, 47; 39 sliding-window layers with window 128 |
| Attention | 64 Q heads, Q/K width 192, V width 128; global KV heads 4 and sliding KV heads 8 |
| RoPE | 64 rotary dimensions of the 192 Q/K dimensions; global theta 10,000,000 and sliding theta 10,000 |
| Sinks and scaling | Learned attention-sink bias on sliding layers only; V is multiplied by 0.707 before KV cache write |
| MLP | Layer 0 dense; layers 1–47 use 256 routed experts, sigmoid noaux top-8 with selection-only correction bias and selected-score normalization; no shared expert |
| Checkpoint | Fused QKV, ModelOpt mixed FP8/NVFP4 routed experts; MTP's three blocks and DFlash are separate artifacts and later slices |
| Context | Combined maximum 1,048,576 tokens |

Read-only Memra inspection at this head:

- `crates/memra-gguf/src/config.rs` has no `mimo_v2` architecture mapping.
  It parses the source's `n_routed_experts` spelling but currently gives
  V the Q/K width, counts three separately stored MTP blocks as trunk
  layers, and has no MiMo-specific per-layer geometry field.
- `crates/memra-gguf/src/model_plan.rs` can describe sliding attention,
  unequal Q/K and V widths, partial RoPE, and sigmoid routing, but its
  `FullAttentionPlan` cannot express learned sinks or MiMo's 0.707 value
  scale. Its canonical builder also needs the actual nine-layer full/SWA
  pattern and distinct theta/KV-head values. A direct use of
  `ArchGeometryTable` would incorrectly require QK normalization under the
  current `qk_norm_presence` rule; MiMo has no QK norm, so that arm needs
  explicit family semantics.
- `model_packs/mod.rs` has no MiMo pack. Registering a pack before the
  missing semantics and tensor contract exist would incorrectly admit
  this checkpoint, so the load path must continue to refuse it.
- `LayerTensor::AttentionSink` already exists for DeepSeek-V4 intake, and
  the generic safetensors router-bias aliases already include MiMo's
  `mlp.gate.e_score_correction_bias` spelling. These identifiers can be
  reused. The specialized DeepSeek-V4 oracle (`dsv4_forward.rs`) and CUDA
  kernel (`cu/dsv4_gpu.cu`) implement the denominator-only sink-softmax
  rule, but the generic grouped Q/K/V full/SWA path does not expose it for
  MiMo.
- The pinned index names sliding sink tensors
  `model.layers.<sliding-layer>.self_attn.attention_sink_bias`, and routed
  selection biases
  `model.layers.<moe-layer>.mlp.gate.e_score_correction_bias`. Layer 0
  has dense `gate_proj`, `up_proj`, and `down_proj`; later layers contain
  routed expert banks. The SGLang source applies `attention_value_scale`
  to V before passing it to attention, so a cache-precision parity check
  must preserve that order.
- The native executor, checkpoint binding, tokenizer/template parity,
  sampled API surface, host-cache reuse, long context, concurrency,
  output limit, and rollback gates are all still required. `NativeReference`
  requires an executed semantic fixture; `NativeQualified` requires
  pinned-checkpoint parity and serving gates on the actual hardware/image.

Current SGLang evidence for choosing native controls is under
`darklanes/research/mimo-serving-max-20260923/LANE.md`. A BF16-KV TP2
short profile passed two sampled code runs at 155/164 each, math 1274/1319,
endpoint and cache8. Its requested API seeds were ignored. The TP4/1M
BF16 profile's first code run scored 152/164; the gate therefore remains
closed at that topology. Plain FP8 TP4 C1 at 1.04M input and 98.3% prompt
reuse measured 65.36 decode and 13.85 full-wall output tok/s; its code
gate failed. None of these results qualifies Memra or customer serving.

No Memra tests, GPU gates, or server runs have been performed on the local
rig. All model-on-hardware work remains on an owned non-production provider
host, and evidence must name the artifact revision, Memra commit, binary,
GPU topology, corpus hash, request shape, and cache state.

Current fail-closed implementation slice in this worktree:

- `Arch::MiMoV2` recognizes the HF `mimo_v2` model type.
- HF config normalization keeps the separately stored MTP blocks out of
  the 48-layer trunk and reads `v_head_dim=128` rather than defaulting to
  the 192-wide key.
- The config retains the source's full/SWA pattern, SWA KV geometry, RoPE
  base, attention value scale, sink declarations, dense/MoE pattern, and
  separately stored MTP depth without fabricating missing values.
- The exact 8,068-byte pinned source `config.json` is a fixture at
  `crates/memra-gguf/src/model_packs/mimo_v2/fixtures/config.json`
  (SHA-256 above). A unit test checks those values, 64 rotary dimensions,
  256 experts/top-8, and continued refusal by `compile_for_load` while
  no MiMo pack is registered.
- Tests have **not run**. The owned research host has no Rust toolchain;
  remote or hosted checks must run before any PR/review claim. The root
  Memra checkout remains on `main` with its unrelated edits untouched.
