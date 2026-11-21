# Reference executor and TinyParity (steps 3 to 5)

The reference executor of the `inkling2` family, the oracles that pin it and what they prove.
CPU only. Nothing here ran on a GPU and nothing here is a kernel.

- Executor: `crates/memra-reference/src/inkling2.rs` (the session) and
  `crates/memra-reference/src/inkling2/ops.rs` (one function per operation). f64, unfused, from
  the values the checkpoint stores. BF16 and F32 tensors are exact in f32 and are widened. Expert
  banks are read from their stored bytes and dequantized exactly.
- Declared expert program: `crates/memra-reference/src/inkling2_actquant.rs` (the activation
  quantizers and the W4A4 / W8A8 expert block, pinned by their own goldens in
  [`../actquant/`](../actquant/)). A layer whose plan mode is `Declared` calls it on its input
  rounded to bf16; a layer in `Float` mode runs the exactly dequantized weights on float inputs.
  The mode is per physical layer.
- Tiny gate: `memra model verify tiny --against inkling2 --out <dir>`. Receipts of the run on
  this tree: [`../tiny/`](../tiny/).

## Oracles

| File | What it is |
|---|---|
| [`ink2_oracle.py`](ink2_oracle.py) | The lane's oracle program. See [What the oracle is](#what-the-oracle-is) for which parts are the PR's code and which are lane-written. Arm `actq` simulates the declared activation-quant expert program on an f32 stream. |
| [`ink2_tiny.py`](ink2_tiny.py), [`real_name_classes.json`](real_name_classes.json) | The generator of the tiny checkpoint and of its 40 token ids (seed 20261121, variant `tiny4`), and the checkpoint's name classes it checks its output against. |
| [`stock_crosscheck.py`](stock_crosscheck.py) | Runs the PR program with nothing replaced on the tiny image and reports its distance from the pinned oracle. |
| [`tiny_oracle.py`](tiny_oracle.py) | Runs `ink2_oracle.py` on a checkpoint image written by `verify tiny`, under the image's schedule, and writes the pinned TSV. |
| [`module_oracles.py`](module_oracles.py) | Runs each PR module alone in f64 on seeded inputs and writes `crates/memra-reference/src/fixtures/inkling2-module-oracles.bin` (23 records, 185,611 bytes, sha256 `90ec92fc9b7f7b461aa0c11bae1b215d786694f34dfd3985b6a87ea117864b66`). The PR's own conversion ops (`Interleave`, the message-query merge, the vocabulary unpad, the expert dequantizer) run on tensors in the checkpoint layout, so a record also pins how the stored layout maps to the module. |
| [`tiny_quant_config.py`](tiny_quant_config.py) | Writes the tiny checkpoint's `hf_quant_config.json`: the real file's groups over the tiny config's tensor names. |

Environment of every run here: Python 3.12, torch 2.15.0+cpu, numpy 2.5.5, transformers at the
PR head (`pip install "transformers @ git+https://github.com/huggingface/transformers@6173a2f679ff4ed393afc499aefc569da7e68bc1"`).

```bash
memra model verify tiny --against inkling2 --out out
python -I tiny_oracle.py --image out/tiny-checkpoint --work scratch --out inkling2.tsv
python -I tiny_oracle.py --image out/tiny-checkpoint-splits --work scratch --out splits.tsv   # the same bytes as inkling2.tsv
python -I tiny_oracle.py --image out/tiny-checkpoint-declared --work scratch --arm actq --out inkling2-declared.tsv
python -I module_oracles.py --out crates/memra-reference/src/fixtures/inkling2-module-oracles.bin
```

The oracle was run on each of the five images `verify tiny` wrote and every output is
byte-identical to its pinned TSV ([`../tiny/hashes.tsv`](../tiny/hashes.tsv),
`../tiny/oracle-run*.json`). The main run and the three schedule runs give the same bytes, so
one file, `inkling2.tsv`, is pinned for the four; `inkling2-declared.tsv` is the fifth.

## What the oracle is

The tiny gate's float-activation oracle (`ink2_oracle.py`, arm `exact`, dtype `f64`) is not the
PR program end to end. It has two parts.

- **The PR's code, transformed to f64 by text substitution** (`.float()` to `.double()`,
  `torch.float32` to `torch.float64`): the trunk (embedding norm, every attention module, the
  short convolutions, the block residual, the loop and its message modules, the router, the
  activation, the head), the PR's `DynamicCache` that the schedule runs walk, its masks, and its
  loader with the message-query merge and the vocabulary unpad.
- **Lane-written**: the expert module (`QuantExperts`). It reads the stored bytes, dequantizes
  each routed expert itself, splits gate and up as rows `0::2` and `1::2`, and runs its own f64
  GEMM. The PR's `Interleave`, dequantizer op and expert module are not in this path.

So inside the tiny pin, the interleave, the dequantization and the expert GEMM are compared with
a second implementation by the lane, not with the PR. The pins of those three facts against the
PR's own code are the module records `experts`, `dequant_nvfp4` and `dequant_mxfp8`, which call
the PR's `Interleave`, expert module and dequantizer op.

The replacement is checked against the PR program with nothing replaced, on the tiny image
([`../tiny/stock-crosscheck.json`](../tiny/stock-crosscheck.json)):

| PR program, unmodified | max abs from the pinned oracle | argmax equal |
|---|---:|---:|
| f64 | 4.8e-5 | 40 of 40 |
| f32 | 1.7e-4 | 40 of 40 |
| bf16 (the PR's stock load: dequantized experts cast to BF16) | 5.19 | 14 of 40 |

The f64 residual is the PR's f32 dequant arithmetic (two f32 roundings where the oracle has one
f64 division). The bf16 row is a different weight, not a rounding.

Bit equality with the oracle is a property of the tiny config: its clip thresholds (4.0 and 2.0)
are exact in f32. The checkpoint's Q/K threshold is not, and the executor reads it as the f32
the plan holds.

## The tiny fixture

The tiny checkpoint (`crates/memra-reference/src/fixtures/inkling2-tiny-checkpoint.bin`, 772,640
bytes, sha256 `db3bcf55117349fada275aa5e1261e13f5a703ad5b99c608ab88076326567953`) is a seeded
checkpoint image in the family's real layout: `model.llm.*` names, BF16 / F32 / U8 / F8_E4M3
storage, NVFP4 experts on layers 0 to 5 and MXFP8 on layers 6 and 7, with zero and subnormal
block scales, signed zeros and subnormal values on purpose. It was written by
[`ink2_tiny.py`](ink2_tiny.py), not by the executor it tests. Run again from the tracked script
it gives the same 772,640 bytes and the same 40 token ids as `TINY_TOKENS`
([`../tiny/generator-run.json`](../tiny/generator-run.json)):

```bash
python -I ink2_tiny.py --variant tiny4 --out regen --real-names real_name_classes.json \
  --real-quant crates/memra-gguf/src/model_packs/inkling2/fixtures/hf_quant_config.json
cmp regen/model.safetensors crates/memra-reference/src/fixtures/inkling2-tiny-checkpoint.bin
```

Its config
(`crates/memra-gguf/src/model_packs/inkling2/fixtures/tiny-config.json`) is the checkpoint's key
layout at hidden 64:

| | tiny | checkpoint |
|---|---:|---:|
| physical layers, block size | 8, 2 | 24, 6 |
| full-attention layers | 1, 3, 5, 7 | 5, 11, 17, 23 |
| loop core, passes, cache slots | 2..6, 3, 16 | 6..18, 3, 48 |
| sliding window, relative extent, log-scale floor | 8, 12, 24 | 1024, 1024, 8192 |
| experts, top-k, expert width | 8, 2, 32 | 128, 8, 1152 |
| vocabulary rows, unpadded | 320, 300 | 201024, 200058 |

Four blocks, not two: with two looped core blocks the message bank of a pass holds two states,
so the message mix is a real softmax over the bank. With one core block the bank is one state
and the mix cannot see its queries.

The request is 40 token ids. It crosses the window (8), the extent (12) and the floor (24).

## What the tiny gate runs

| Run | Plan | Schedule | Pinned to | Result |
|---|---|---|---|---|
| main | float-activation | one prefill chunk of 40 | the f64 oracle, same schedule | 12,000 logits equal to the bit after rounding to f32; gated on bit equality |
| `splits` | float-activation | prefill chunks ending at 8, 12, 24, 40 | the f64 oracle, same schedule | equal to the bit; gated on bit equality |
| `offsplits` | float-activation | prefill chunks ending at 7, 11, 23, 40 | the f64 oracle, same schedule | equal to the bit; gated on bit equality |
| `steps` | float-activation | prefill of 5, then 35 one-token calls | the f64 oracle, same schedule | equal to the bit; gated on bit equality |
| `declared` | default class, every layer on its declared quantizer | one prefill chunk of 40 | a simulation of the activation-quant program on an f32 stream (`--arm actq`) | max abs 2.26e-6, mean 2.6e-7, argmax equal at 40 of 40 positions ([`../tiny/declared-distance.json`](../tiny/declared-distance.json)) |

- The four float-activation oracles agree with each other to 9e-13 in f64 and their TSVs are the
  same bytes. The executor computes a chunk slot by slot over all of its rows and carries K, V
  and convolution state between calls, so the three schedules are three different walks of the
  state, and they give the same bits as one chunk.
- The four float-activation runs are gated on bit equality: the run's oracle text has to be
  the pinned bytes. A one-ulp move of one logit fails the gate.
- The `declared` run alone has a tolerance, 1e-5 absolute plus 1e-5 relative, with the argmax
  of every position required by the gate. It uses 18% of the band: its oracle is an f32 stream
  and the executor's trunk is f64. The band is the tiny gate's (`TINY_DECLARED_GATE`), carried
  by the fixture. It is not a checkpoint band.
- The pack declares no `checkpoint_parity`, for any class, so `verify checkpoint` refuses the
  pack before it reads an oracle. A band of the declared class has to come from that class's
  source program measured on two backends. A float-activation oracle cannot write a
  CheckpointParity pass for this pack.
- The two classes are far apart on this fixture: max abs 5.41 between the declared and the
  float-activation logits, and the argmax agrees at 5 of 40 positions. Neither oracle can pass
  for the other class.

What the `declared` oracle is and is not. It is an implementation of the declared expert
program written apart from `inkling2_actquant.rs`, from the same reading of the vendor's
serving code. Two independent implementations agreeing end to end is evidence that the reading
was implemented the same way twice. It is not a capture of a serving binary, and it does not
prove the reading. Its known gaps: exact-math tanh and sigmoid, GEMM sums in f64 rounded once to
f32, the e4m3 cast taken as round to nearest even, the routed rows added in ascending expert
id, an f32 stream outside the experts. A real oracle of the declared class needs the serving
image on a Blackwell card, and that is the checkpoint gate's job.

## Facts proven, one test each

Every test is in `crates/memra-reference/src/inkling2/tests.rs` and replays a module-oracle
record at 1e-12 in f64. A red arm is the misreading of the same fact, run against the same
record: it has to miss the oracle by more than 1e-4, which shows the record can tell the two
readings apart.

| Fact | Proven by | Red arm |
|---|---|---|
| Gate/up interleave of `w13`: row `2j` is gate `j`, row `2j + 1` is up `j` | `expert_gate_and_up_rows_are_interleaved_in_the_checkpoint`: checkpoint-layout `w13` through the PR's `Interleave(dim=1)` and its expert module | gate rows first, then up rows |
| Window edge: a sliding layer sees itself and `window - 1` positions back | `sliding_attention_sees_itself_and_window_minus_one_back`: the PR's sliding attention module with the mask the model builds, 12 positions against a window of 5 | `window - 1` (first differs exactly at position `window - 1`), and `window + 1` with an unmasked edge |
| A sliding layer's bias table ends its window; a full layer gives keys past its table a zero bias | `relative_bias_is_a_table_over_backward_distance_and_zero_outside_it`, `full_attention_gives_keys_past_the_table_a_zero_bias` | full layer with keys past the table masked (first differs exactly at position `extent`) |
| Sink form: one more logit per query and head in the softmax denominator, with no value | `the_sink_is_one_more_logit_in_the_denominator` | no sink |
| The other sink spelling (subtract the sink from the bias, zero sink) is the same function only when the sink comes off the bias of every attended key, the zero-bias keys past the table included | same test, both layer classes | `a_sink_folded_inside_the_table_only_is_another_function_on_a_full_layer`: equal through position `extent - 1`, then off the oracle by 0.3456 |
| Static short-conv taps run oldest input first: the last tap weights the current token | `key_and_value_projections_convolve_one_row_at_a_time_from_carried_history`, `output_convolution_is_a_causal_cross_correlation_with_a_residual` | taps reversed |
| Static taps of a dynamic (full-layer K, V) convolution run newest first, like its dynamic taps | same K/V test, dynamic records | static taps reversed |
| Conv state is the last `kernel - 1` inputs of the convolution (the projection, before the conv) | the same tests walk one row at a time against the PR's whole-sequence convolution | |
| `tau = 1 + alpha * ln(max(1, (pos + 1) / floor))`, on Q and R of full layers, after the clip | `query_projection_scales_clips_then_log_scales`, the full-layer relative and attention records | a floor one later (equal through position `floor - 1`, first differs at position `floor`); no tau |
| Message slots: core layer `i` reads q and mlp from rows `2i`, `2i + 1` of `middle_cycle_cross_pass_queries` and k and v from rows `2i`, `2i + 1` of `middle_cycle_cross_pass_kv_queries` | `loop_message_queries_come_from_the_two_checkpoint_tensors_in_q_k_v_mlp_order`: the two checkpoint tensors through the PR's merge op and its message module | the two tensors swapped; rows read in order |
| The message bank is the previous pass's block outputs after its first, plus its final stream; every pass restarts from the prelude | `float_activation_program_is_the_pinned_oracle_to_the_bit_under_every_schedule` (the tiny program, two core blocks, three passes) | the bank taken one block earlier ([`../tiny/session-mutations.tsv`](../tiny/session-mutations.tsv)) |
| The sink and the output gate are projected from the Q-side input; the MLP controller runs after the MLP mix and before `mlp_norm` | the same test | the sink from the K input; the controller after the norm (same receipt) |
| NVFP4 nibble order (even element in the low nibble) and e4m3 scale decode | `nvfp4_bank_dequantizes_exactly_and_its_f32_spelling_is_the_oracle_bit_for_bit`: the PR's dequantizer op on stored bytes, 1,152 values, zero, subnormal and small scales included. The bank's f32 spelling is the oracle bit for bit; the exact value the executor uses (`ExpertBank::expert`) is within two f32 roundings of it | nibbles swapped, through `ExpertBank::expert`, against the oracle |
| MXFP8 decode: `e4m3(value) * 2^(scale - 127)`, one exponent per row per 32 inputs | `mxfp8_bank_dequantizes_to_the_oracle_exactly` | |
| Router: selection on the biased sigmoid, weights from the unbiased one over the selected sum, times `route_scale` | `router_selects_on_the_biased_sigmoid_and_weights_with_the_unbiased_one` | selection without the bias picks other experts on this record |
| Softcapped SwiGLU: the sigmoid reads the raw gate | `softcapped_swiglu_caps_the_linear_gate_factor_and_up_but_not_the_sigmoid` | sigmoid of the capped gate |
| Embed norm, output norm, softcap, truncation to the unpadded rows | `embedding_norm_and_head_match_the_oracle`, `padded_vocabulary_rows_are_never_computed` (NaN in the padded rows changes nothing) | |
| Log scale at whole-program level: position `floor - 1` is unscaled, position `floor` is the first scaled | `log_scale_starts_at_the_floor_and_a_later_floor_is_another_program` | |
| Per-layer mode: one layer on its quantizer is a class of its own and leaves every earlier visit untouched | `declared_layers_run_their_quantizer_and_each_layer_moves_alone` | |

The MXFP8 scale grid of the real checkpoint (one exponent per row per 32 inputs) was closed from
the bytes in step 1 ([`../SEMANTICS.md`](../SEMANTICS.md#closed-from-the-bytes)).

## Where the references differ, and what the executor does

| Point | transformers PR | vendor serving code | Executor |
|---|---|---|---|
| Expert inputs | float | NVFP4 / MXFP8 quantized, per the checkpoint's `required_activation_quantization` | both, per layer; the declared one is the default class |
| Routing weight | after the down projection | before it, and before the down input is quantized | before, as the declared class requires. On float inputs the two orders are one function up to rounding (tested) |
| Router normalization | `+ 1e-20` in the denominator | none | none. The two differ only if every selected sigmoid underflows |
| Top-k ties | `torch.topk`, order unspecified | lower expert index | lower expert index, selected experts kept in rank order (`router_ties_go_to_the_lower_expert_and_keep_rank_order`) |
| Dequantized experts | cast to BF16 on load, which rounds `code * scale * amax / denominator` | kept quantized | exact: one f64 division. The BF16 cast is a different weight |
| Activation dtype outside the experts | the model dtype | bf16 with f32 accumulation | f64. The bf16 rounding points of a serving class belong to that class's band |
| Cache slot numbering | physical layers, then pass 1, then pass 2 | visits in execution order | the plan's slots are the first; `loop_plan.execution_order` is the second |

## Not done here

- No GPU surface. Every operation keeps an empty registry row and every rewrite is ineligible.
- The drafter (`mtp.safetensors`) and the vision tower are owned by the contract and not
  executed. `execute` and `execute_schedule` run a plan that declares the drafter and return no
  draft; the standalone draft entry point and the multimodal one refuse an Inkling 2 plan by
  name.
- No oracle of the declared class from a serving binary.
- The executor has not run the real checkpoint. `ReferenceWeights` holds every tensor in
  memory; a checkpoint runner needs a streamed source for the 14 GiB of expert banks.
