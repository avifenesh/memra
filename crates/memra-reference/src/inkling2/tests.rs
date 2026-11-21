use super::ops::*;
use super::*;
use crate::inkling2_actquant::goldens;
use memra_gguf::model_packs::{self, inkling2 as pack};
use memra_gguf::model_plan::{Inkling2NumericClass, OperationKind};

/// Module oracles: one module of the float-activation implementation per record, in f64
/// (research/modelplan-onboarding-inkling2mini-20261121/reference/module_oracles.py).
const ORACLES: &[u8] = include_bytes!("../fixtures/inkling2-module-oracles.bin");

// The oracle config (module_oracles.py CONFIG).
const HIDDEN: usize = 16;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const DIM: usize = 4;
const RANK: usize = 3;
const EXTENT: usize = 6;
const WINDOW: usize = 5;
const FLOOR: u32 = 7;
const ALPHA: f64 = 0.1;
const QK_CLIP: f64 = 2.0;
const REL_CLIP: f64 = 1.5;
const KERNEL: usize = 4;
const WIDTH: usize = 8;
const TOP_K: usize = 2;
const CAPS: (f64, f64) = (2.0, 3.0);
const TOKENS: usize = 12;
const EPSILON: f64 = 1e-6;
/// f64 against f64: only summation order and library rounding differ.
const TOLERANCE: f64 = 1e-12;

struct Record(goldens::Record);

fn record(name: &str) -> Record {
    Record(
        goldens::parse(ORACLES)
            .unwrap()
            .remove(name)
            .unwrap_or_else(|| panic!("no oracle record {name}")),
    )
}

impl Record {
    fn raw(&self, tag: &str) -> &[u8] {
        self.0.get(tag).unwrap_or_else(|| panic!("no blob {tag}"))
    }
    fn f64s(&self, tag: &str) -> Vec<f64> {
        goldens::f64s(self.raw(tag))
    }
    /// A parameter: stored as f64, exactly an f32.
    fn weight(&self, tag: &str) -> Vec<f32> {
        self.f64s(tag)
            .into_iter()
            .map(|value| {
                assert_eq!(f64::from(value as f32), value, "{tag} is not f32-exact");
                value as f32
            })
            .collect()
    }
    fn ids(&self, tag: &str) -> Vec<usize> {
        self.raw(tag)
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect()
    }
}

fn worst(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .fold(0.0, |worst, (a, e)| worst.max((a - e).abs()))
}

fn assert_matches(what: &str, actual: &[f64], expected: &[f64]) {
    let worst = worst(actual, expected);
    assert!(worst < TOLERANCE, "{what}: off the oracle by {worst}");
}

/// A red arm: the misreading is far outside the tolerance, so the oracle tells the two apart.
fn assert_differs(what: &str, actual: &[f64], expected: &[f64]) {
    let worst = worst(actual, expected);
    assert!(
        worst > 1e-4,
        "{what}: the oracle cannot see this reading ({worst})"
    );
}

fn rows(values: &[f64], width: usize) -> Vec<&[f64]> {
    values.chunks_exact(width).collect()
}

fn query(r: &Record, floor: Option<u32>) -> Vec<f64> {
    let (w, scale, x) = (r.weight("w"), r.weight("scale"), r.f64s("x"));
    let mut out = Vec::new();
    for (position, x) in rows(&x, HIDDEN).into_iter().enumerate() {
        let tau = floor.map_or(1.0, |floor| {
            log_position_scale(position as u64, floor, ALPHA)
        });
        let mut q = linear(&w, x);
        clip_heads(&mut q, DIM, QK_CLIP, EPSILON, Some(&scale), tau);
        out.extend(q);
    }
    out
}

#[test]
fn query_projection_scales_clips_then_log_scales() {
    let full = record("query_full");
    let expected = full.f64s("out");
    assert_matches("full-layer Q", &query(&full, Some(FLOOR)), &expected);
    // tau is 1 through position floor - 1 and grows after: a floor one later is another program
    // from exactly position 7 on, and no tau at all is a third.
    let later = query(&full, Some(FLOOR + 1));
    let split = FLOOR as usize * HEADS * DIM;
    assert_eq!(later[..split], query(&full, Some(FLOOR))[..split]);
    assert_differs(
        "tau floor",
        &later[split..split + HEADS * DIM],
        &expected[split..split + HEADS * DIM],
    );
    assert_differs("tau dropped", &query(&full, None), &expected);
    // The fixture engages the clip on some heads and not on others.
    let (w, scale, x) = (full.weight("w"), full.weight("scale"), full.f64s("x"));
    let mut clipped = [0usize; 2];
    for x in rows(&x, HIDDEN) {
        for (head, q) in linear(&w, x).chunks_exact(DIM).enumerate() {
            let norm: f64 = q
                .iter()
                .zip(&scale[head * DIM..])
                .map(|(v, s)| (v * f64::from(*s)).powi(2))
                .sum::<f64>()
                .sqrt();
            clipped[usize::from(norm > QK_CLIP)] += 1;
        }
    }
    assert!(clipped[0] > 4 && clipped[1] > 4, "{clipped:?}");

    let sliding = record("query_sliding");
    assert_matches(
        "sliding-layer Q",
        &query(&sliding, None),
        &sliding.f64s("out"),
    );
}

/// K or V rows through the projection, the stateful short convolution and the head rule.
fn key_or_value(r: &Record, dynamic: bool, key: bool, reverse_taps: bool) -> Vec<f64> {
    let (w, x) = (r.weight("w"), r.f64s("x"));
    let channels = KV_HEADS * DIM;
    let reverse = |taps: Vec<f32>, lag_major: bool| -> Vec<f32> {
        if !reverse_taps {
            return taps;
        }
        let mut out = taps.clone();
        for channel in 0..channels {
            for tap in 0..KERNEL {
                let (from, to) = if lag_major {
                    (
                        (KERNEL - 1 - tap) * channels + channel,
                        tap * channels + channel,
                    )
                } else {
                    (channel * KERNEL + KERNEL - 1 - tap, channel * KERNEL + tap)
                };
                out[to] = taps[from];
            }
        }
        out
    };
    let mut history = ConvHistory::default();
    let mut out = Vec::new();
    for x in rows(&x, HIDDEN) {
        let projected = linear(&w, x);
        let mut row = if dynamic {
            short_conv_dynamic(
                &projected,
                &mut history,
                &reverse(r.weight("static_w"), true),
                &linear(&r.weight("weight_proj"), x),
                KERNEL,
                KV_HEADS,
            )
        } else {
            short_conv_static(
                &projected,
                &mut history,
                &reverse(r.weight("taps"), false),
                KERNEL,
            )
        };
        if key {
            clip_heads(&mut row, DIM, QK_CLIP, EPSILON, None, 1.0);
            out.extend(row);
        } else {
            for head in row.chunks_exact(DIM) {
                out.extend(rms_norm(head, None, EPSILON));
            }
        }
    }
    out
}

#[test]
fn key_and_value_projections_convolve_one_row_at_a_time_from_carried_history() {
    for (name, dynamic, key) in [
        ("key_static", false, true),
        ("value_static", false, false),
        ("key_dynamic", true, true),
        ("value_dynamic", true, false),
    ] {
        let r = record(name);
        let expected = r.f64s("out");
        // The oracle convolves the whole sequence at once; this walks it a row at a time with
        // the last three inputs as state, so the match also pins the state.
        assert_matches(name, &key_or_value(&r, dynamic, key, false), &expected);
        // Tap order. Static taps run oldest first (the last weights the current token); the
        // static taps of a dynamic convolution run newest first. Each read the other way is a
        // different function.
        assert_differs(name, &key_or_value(&r, dynamic, key, true), &expected);
    }
}

#[test]
fn output_convolution_is_a_causal_cross_correlation_with_a_residual() {
    let r = record("conv_static");
    let (taps, x, expected) = (r.weight("taps"), r.f64s("x"), r.f64s("out"));
    let mut history = ConvHistory::default();
    let mut out = Vec::new();
    for x in rows(&x, HIDDEN) {
        out.extend(short_conv_static(x, &mut history, &taps, KERNEL));
    }
    assert_matches("static short convolution", &out, &expected);
    assert_eq!(history.flat(), x[(TOKENS - (KERNEL - 1)) * HIDDEN..]);
}

fn relative(r: &Record, extent: usize, floor: Option<u32>) -> Vec<Vec<f64>> {
    let (w, scale, proj, x) = (
        r.weight("w"),
        r.weight("scale"),
        r.weight("proj"),
        r.f64s("x"),
    );
    rows(&x, HIDDEN)
        .into_iter()
        .enumerate()
        .map(|(position, x)| {
            let tau = floor.map_or(1.0, |floor| {
                log_position_scale(position as u64, floor, ALPHA)
            });
            let mut vectors = linear(&w, x);
            clip_heads(&mut vectors, RANK, REL_CLIP, EPSILON, Some(&scale), tau);
            relative_bias(&vectors, &proj, RANK, extent)
        })
        .collect()
}

#[test]
fn relative_bias_is_a_table_over_backward_distance_and_zero_outside_it() {
    for (name, extent, floor) in [
        ("relative_full", EXTENT, Some(FLOOR)),
        ("relative_sliding", WINDOW, None),
    ] {
        let r = record(name);
        // [query, head, key]
        let expected = r.f64s("out");
        let tables = relative(&r, extent, floor);
        let mut actual = vec![0.0; expected.len()];
        let mut beyond = 0;
        for query in 0..TOKENS {
            for head in 0..HEADS {
                for key in 0..=query {
                    let distance = query - key;
                    if distance < extent {
                        actual[(query * HEADS + head) * TOKENS + key] =
                            tables[query][head * extent + distance];
                    } else {
                        beyond += 1;
                    }
                }
            }
        }
        assert!(beyond > 0, "{name} reaches past its table");
        assert_matches(name, &actual, &expected);
    }
}

struct AttentionArm {
    window: Option<usize>,
    mask_beyond_extent: bool,
    /// The sink logit as the module projects it, or folded into the bias with a zero sink.
    fold_sink: bool,
    drop_sink: bool,
}

fn attention(r: &Record, sliding: bool, arm: &AttentionArm) -> Vec<f64> {
    let extent = if sliding { WINDOW } else { EXTENT };
    let floor = (!sliding).then_some(FLOOR);
    let (xq, xk, xv) = (r.f64s("xq"), r.f64s("xk"), r.f64s("xv"));
    let (wq, q_scale, wk, wv) = (
        r.weight("wq"),
        r.weight("q_scale"),
        r.weight("wk"),
        r.weight("wv"),
    );
    let (wr, r_scale, proj) = (r.weight("wr"), r.weight("r_scale"), r.weight("proj"));
    let (sink_weight, gate, wo) = (r.weight("sink"), r.weight("gate"), r.weight("wo"));
    let shape = AttentionShape {
        query_heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: DIM,
        window: arm.window,
        extent,
        mask_beyond_extent: arm.mask_beyond_extent,
    };
    let (mut k_history, mut v_history) = (ConvHistory::default(), ConvHistory::default());
    let (mut positions, mut keys, mut values) = (Vec::new(), Vec::new(), Vec::new());
    let mut out = Vec::new();
    for position in 0..TOKENS {
        let at = |x: &[f64]| x[position * HIDDEN..(position + 1) * HIDDEN].to_vec();
        let (xq, xk, xv) = (at(&xq), at(&xk), at(&xv));
        let tau = floor.map_or(1.0, |floor| {
            log_position_scale(position as u64, floor, ALPHA)
        });
        let mut q = linear(&wq, &xq);
        clip_heads(&mut q, DIM, QK_CLIP, EPSILON, Some(&q_scale), tau);
        let mut vectors = linear(&wr, &xq);
        clip_heads(&mut vectors, RANK, REL_CLIP, EPSILON, Some(&r_scale), tau);
        let mut bias = relative_bias(&vectors, &proj, RANK, extent);
        let mut sink = linear(&sink_weight, &xq);
        let mut shape = shape;
        if arm.fold_sink {
            // The sink comes off the bias of EVERY attended key, so the table is first padded
            // with the zero bias of the keys past it.
            let padded = if shape.mask_beyond_extent {
                extent
            } else {
                extent.max(TOKENS)
            };
            let mut folded = vec![0.0; HEADS * padded];
            for head in 0..HEADS {
                folded[head * padded..head * padded + extent]
                    .copy_from_slice(&bias[head * extent..(head + 1) * extent]);
                for value in &mut folded[head * padded..(head + 1) * padded] {
                    *value -= sink[head];
                }
                sink[head] = 0.0;
            }
            bias = folded;
            shape.extent = padded;
        }
        if arm.drop_sink {
            sink.fill(f64::NEG_INFINITY);
        }
        let (mut key, value) = if sliding {
            (
                short_conv_static(
                    &linear(&wk, &xk),
                    &mut k_history,
                    &r.weight("k_taps"),
                    KERNEL,
                ),
                short_conv_static(
                    &linear(&wv, &xv),
                    &mut v_history,
                    &r.weight("v_taps"),
                    KERNEL,
                ),
            )
        } else {
            (
                short_conv_dynamic(
                    &linear(&wk, &xk),
                    &mut k_history,
                    &r.weight("k_static_w"),
                    &linear(&r.weight("k_weight_proj"), &xk),
                    KERNEL,
                    KV_HEADS,
                ),
                short_conv_dynamic(
                    &linear(&wv, &xv),
                    &mut v_history,
                    &r.weight("v_static_w"),
                    &linear(&r.weight("v_weight_proj"), &xv),
                    KERNEL,
                    KV_HEADS,
                ),
            )
        };
        clip_heads(&mut key, DIM, QK_CLIP, EPSILON, None, 1.0);
        positions.push(position as u64);
        keys.extend(key);
        for head in value.chunks_exact(DIM) {
            values.extend(rms_norm(head, None, EPSILON));
        }
        let mut attended = attend(
            &shape,
            &q,
            &bias,
            &sink,
            position as u64,
            CachedRows {
                positions: &positions,
                keys: &keys,
                values: &values,
            },
        );
        for (value, gate) in attended.iter_mut().zip(linear(&gate, &xq)) {
            *value *= sigmoid(gate);
        }
        out.extend(linear(&wo, &attended));
    }
    out
}

#[test]
fn sliding_attention_sees_itself_and_window_minus_one_back() {
    let r = record("attention_sliding");
    let expected = r.f64s("out");
    let arm = |window| AttentionArm {
        window: Some(window),
        mask_beyond_extent: true,
        fold_sink: false,
        drop_sink: false,
    };
    assert_matches(
        "sliding attention",
        &attention(&r, true, &arm(WINDOW)),
        &expected,
    );
    // The window edge: one key fewer or one more is another program, and only from the first
    // position that reaches the edge.
    let narrow = attention(&r, true, &arm(WINDOW - 1));
    let edge = (WINDOW - 1) * HIDDEN;
    assert_matches("below the edge", &narrow[..edge], &expected[..edge]);
    assert_differs(
        "window - 1",
        &narrow[edge..edge + HIDDEN],
        &expected[edge..edge + HIDDEN],
    );
    // One more key needs a bias the table does not hold: a sliding layer masks it.
    let wide = AttentionArm {
        window: Some(WINDOW + 1),
        mask_beyond_extent: true,
        fold_sink: false,
        drop_sink: false,
    };
    assert_matches(
        "the table ends the window",
        &attention(&r, true, &wide),
        &expected,
    );
    let unmasked = AttentionArm {
        window: Some(WINDOW + 1),
        mask_beyond_extent: false,
        fold_sink: false,
        drop_sink: false,
    };
    assert_differs("window + 1", &attention(&r, true, &unmasked), &expected);
}

#[test]
fn full_attention_gives_keys_past_the_table_a_zero_bias() {
    let r = record("attention_full");
    let expected = r.f64s("out");
    let arm = AttentionArm {
        window: None,
        mask_beyond_extent: false,
        fold_sink: false,
        drop_sink: false,
    };
    assert_matches("full attention", &attention(&r, false, &arm), &expected);
    let masked = AttentionArm {
        mask_beyond_extent: true,
        ..arm
    };
    let actual = attention(&r, false, &masked);
    let edge = EXTENT * HIDDEN;
    assert_matches("inside the table", &actual[..edge], &expected[..edge]);
    assert_differs(
        "masked past the table",
        &actual[edge..edge + HIDDEN],
        &expected[edge..edge + HIDDEN],
    );
}

#[test]
fn the_sink_is_one_more_logit_in_the_denominator() {
    for (name, sliding) in [("attention_sliding", true), ("attention_full", false)] {
        let r = record(name);
        let expected = r.f64s("out");
        let arm = |fold_sink, drop_sink| AttentionArm {
            window: sliding.then_some(WINDOW),
            mask_beyond_extent: sliding,
            fold_sink,
            drop_sink,
        };
        // The other spelling (the sink subtracted from the bias of every attended key, then a
        // zero sink) is the same function.
        assert_matches(name, &attention(&r, sliding, &arm(true, false)), &expected);
        assert_differs(name, &attention(&r, sliding, &arm(false, true)), &expected);
    }
}

fn mix(r: &Record, queries: usize, blocks: usize) -> Vec<f64> {
    let (q, states, partial) = (r.weight("q"), r.f64s("states"), r.f64s("partial"));
    let mut out = Vec::new();
    for row in 0..TOKENS {
        let mut list: Vec<&[f64]> = (0..blocks)
            .map(|block| &states[(row * blocks + block) * HIDDEN..][..HIDDEN])
            .collect();
        list.push(&partial[row * HIDDEN..][..HIDDEN]);
        let mixed = depth_mix(&q, &list, EPSILON);
        assert_eq!(mixed.len(), queries);
        out.extend(mixed.into_iter().flatten());
    }
    out
}

#[test]
fn block_attention_residual_mixes_normalized_states() {
    for (name, queries, blocks) in [("mix_one", 1, 2), ("mix_three", 3, 2), ("mix_first", 1, 0)] {
        let r = record(name);
        assert_matches(name, &mix(&r, queries, blocks), &r.f64s("out"));
    }
}

#[test]
fn loop_message_queries_come_from_the_two_checkpoint_tensors_in_q_k_v_mlp_order() {
    let r = record("message");
    let (queries, kv_queries) = (r.weight("queries"), r.weight("kv_queries"));
    let (states, partial, expected) = (r.f64s("states"), r.f64s("partial"), r.f64s("out"));
    let run = |assemble: &dyn Fn(usize) -> Vec<f32>| {
        let mut out = Vec::new();
        for layer in 0..2 {
            let four = assemble(layer);
            for row in 0..TOKENS {
                let list = [
                    &states[row * 2 * HIDDEN..][..HIDDEN],
                    &states[(row * 2 + 1) * HIDDEN..][..HIDDEN],
                    &partial[row * HIDDEN..][..HIDDEN],
                ];
                out.extend(depth_mix(&four, &list, EPSILON).into_iter().flatten());
            }
        }
        out
    };
    assert_matches(
        "message mix",
        &run(&|layer| message_queries(&queries, &kv_queries, layer, HIDDEN)),
        &expected,
    );
    // The two tensors swapped, or each read as consecutive (q, k) and (v, mlp) rows, feed the
    // wrong query to three of the four targets.
    assert_differs(
        "tensors swapped",
        &run(&|layer| message_queries(&kv_queries, &queries, layer, HIDDEN)),
        &expected,
    );
    assert_differs(
        "rows read in order",
        &run(&|layer| {
            let mut four = queries[2 * layer * HIDDEN..(2 * layer + 2) * HIDDEN].to_vec();
            four.extend(&kv_queries[2 * layer * HIDDEN..(2 * layer + 2) * HIDDEN]);
            four
        }),
        &expected,
    );
}

#[test]
fn loop_controller_blends_the_stream_and_its_message() {
    let r = record("controller");
    let (x, message) = (r.f64s("x"), r.f64s("message"));
    let (down, forget, gain) = (r.weight("down"), r.weight("forget"), r.weight("gain"));
    let mut out = Vec::new();
    for (x, message) in rows(&x, HIDDEN).into_iter().zip(rows(&message, HIDDEN)) {
        out.extend(controller(x, message, &down, &forget, &gain, EPSILON));
    }
    assert_matches("controller", &out, &r.f64s("out"));
}

#[test]
fn router_selects_on_the_biased_sigmoid_and_weights_with_the_unbiased_one() {
    let r = record("router");
    let (w, bias, x) = (r.weight("w"), r.weight("bias"), r.f64s("x"));
    let (selected, weights) = (r.ids("ids_selected"), r.f64s("weights"));
    let mut unbiased_differs = false;
    for (row, x) in rows(&x, HIDDEN).into_iter().enumerate() {
        let (ids, routed) = route(x, &w, Some(&bias), TOP_K, 2.0);
        let mut pairs: Vec<(usize, f64)> = ids.into_iter().zip(routed).collect();
        pairs.sort_by_key(|pair| pair.0);
        let ids: Vec<usize> = pairs.iter().map(|pair| pair.0).collect();
        assert_eq!(ids, selected[row * TOP_K..(row + 1) * TOP_K], "row {row}");
        let routed: Vec<f64> = pairs.iter().map(|pair| pair.1).collect();
        assert_matches(
            "routing weights",
            &routed,
            &weights[row * TOP_K..(row + 1) * TOP_K],
        );
        assert!((routed.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        let mut plain = route(x, &w, None, TOP_K, 2.0).0;
        plain.sort_unstable();
        unbiased_differs |= plain != ids;
    }
    assert!(unbiased_differs, "the fixture's bias moves a selection");
}

#[test]
fn router_ties_go_to_the_lower_expert_and_keep_rank_order() {
    // Rows 1 and 3 are equal, and so are 0 and 2: equal scores.
    let row_a = [0.5f32, -0.25, 0.125, 1.0];
    let row_b = [-1.0f32, 0.75, 0.5, 0.25];
    let weight: Vec<f32> = [row_b, row_a, row_b, row_a].concat();
    let x = [0.3, -0.7, 1.1, 0.4];
    let (ids, weights) = route(&x, &weight, None, 3, 8.0);
    // a scores higher than b here; rank order, ties ascending
    assert_eq!(ids, [1, 3, 0]);
    assert_eq!(weights[0], weights[1]);
    assert!(weights[2] < weights[1]);
}

#[test]
fn expert_gate_and_up_rows_are_interleaved_in_the_checkpoint() {
    let r = record("experts");
    let (w13, w2, x) = (r.f64s("w13"), r.f64s("w2"), r.f64s("x"));
    let (selected, weights, expected) = (r.ids("ids_selected"), r.f64s("weights"), r.f64s("out"));
    let run = |interleaved: bool, before_down: bool| {
        let mut out = Vec::new();
        for (row, x) in rows(&x, HIDDEN).into_iter().enumerate() {
            let mut sum = vec![0.0; HIDDEN];
            for slot in 0..TOP_K {
                let expert = selected[row * TOP_K + slot];
                let routed = expert_float(
                    x,
                    &w13[expert * 2 * WIDTH * HIDDEN..(expert + 1) * 2 * WIDTH * HIDDEN],
                    &w2[expert * HIDDEN * WIDTH..(expert + 1) * HIDDEN * WIDTH],
                    weights[row * TOP_K + slot],
                    CAPS,
                    interleaved,
                    before_down,
                );
                for (s, v) in sum.iter_mut().zip(routed) {
                    *s += v;
                }
            }
            out.extend(sum);
        }
        out
    };
    // The oracle multiplies the routing weight after the down projection; on float inputs the
    // declared order (before it) is the same function up to rounding.
    assert_matches("experts", &run(true, false), &expected);
    assert_matches("routing weight before down", &run(true, true), &expected);
    assert_differs(
        "gate rows first, then up rows",
        &run(false, false),
        &expected,
    );
}

#[test]
fn softcapped_swiglu_caps_the_linear_gate_factor_and_up_but_not_the_sigmoid() {
    let r = record("swiglu");
    let (gate, up) = (r.f64s("gate"), r.f64s("up"));
    let out: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| softcapped_swiglu(*gate, *up, CAPS.0, CAPS.1))
        .collect();
    assert_matches("softcapped SwiGLU", &out, &r.f64s("out"));
    assert!(gate.iter().any(|gate| gate.abs() > CAPS.0) && up.iter().any(|up| up.abs() > CAPS.1));
    let capped_sigmoid: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| {
            let capped = softcap(*gate, CAPS.0);
            capped * sigmoid(capped) * softcap(*up, CAPS.1)
        })
        .collect();
    assert_differs(
        "sigmoid of the capped gate",
        &capped_sigmoid,
        &r.f64s("out"),
    );
}

#[test]
fn nvfp4_bank_dequantizes_to_the_oracle_bit_for_bit() {
    let r = record("dequant_nvfp4");
    let (experts, bank_rows, inputs) = (3, 6, 64);
    let amax: Vec<f32> = r.weight("amax");
    let denominator = r.weight("denominator")[0];
    let bank = |codes: Vec<u8>| ExpertBank::Nvfp4 {
        codes,
        scales: r.raw("bytes_scales").to_vec(),
        amax: amax.clone(),
        denominator,
        rows: bank_rows,
        inputs,
    };
    let stored = bank(r.raw("bytes_codes").to_vec());
    assert!(!stored.has_nan_code());
    // The oracle's op in f32: fl32(fl32(val(code) * scale) * fl32(amax / denominator)).
    let expected = r.weight("out");
    let per_expert = bank_rows * inputs;
    let (mut nonzero, mut worst_exact) = (0, 0.0f64);
    for expert in 0..experts {
        let factor = amax[expert] / denominator;
        let exact = stored.expert(expert);
        for index in 0..per_expert {
            let want = expected[expert * per_expert + index];
            let scaled = stored.code_scale(expert, index).unwrap();
            assert_eq!(
                (scaled * factor).to_bits(),
                want.to_bits(),
                "{expert} {index}"
            );
            assert_eq!(
                exact[index],
                f64::from(scaled) * f64::from(amax[expert]) / f64::from(denominator)
            );
            // The exact value is one rounding from the product; the oracle's f32 is two.
            worst_exact = worst_exact
                .max((exact[index] - f64::from(want)).abs() / f64::from(want.abs()).max(1e-30));
            nonzero += usize::from(want != 0.0);
        }
    }
    assert!(nonzero > experts * per_expert / 2);
    assert!(worst_exact < 2.0 * f64::from(f32::EPSILON), "{worst_exact}");
    // Nibble order: the even element is the low nibble. Swapped, the bank is another tensor.
    let swapped = bank(
        r.raw("bytes_codes")
            .iter()
            .map(|b| b.rotate_left(4))
            .collect(),
    );
    let moved = (0..per_expert)
        .filter(|&index| swapped.code_scale(0, index) != stored.code_scale(0, index))
        .count();
    assert!(moved > per_expert / 2, "{moved}");
}

#[test]
fn mxfp8_bank_dequantizes_to_the_oracle_exactly() {
    let r = record("dequant_mxfp8");
    let (experts, bank_rows, inputs) = (3, 6, 64);
    let stored = ExpertBank::Mxfp8 {
        values: r.raw("bytes_values").to_vec(),
        exponents: r.raw("bytes_exponents").to_vec(),
        rows: bank_rows,
        inputs,
    };
    assert!(!stored.has_nan_code());
    let expected = r.f64s("out");
    let mut actual = Vec::new();
    for expert in 0..experts {
        actual.extend(stored.expert(expert));
    }
    assert_eq!(actual, expected);
    // The oracle's op reads one exponent per 32 inputs of each row, the shape the bank stores.
    assert_eq!(
        r.raw("bytes_exponents").len(),
        experts * bank_rows * inputs / 32
    );
    let nan = ExpertBank::Mxfp8 {
        values: vec![0x7f; 32],
        exponents: vec![127],
        rows: 1,
        inputs: 32,
    };
    assert!(nan.has_nan_code());
}

#[test]
fn embedding_norm_and_head_match_the_oracle() {
    let r = record("embedding");
    let (table, norm, ids) = (r.f64s("table"), r.weight("norm"), r.ids("ids_tokens"));
    let mut out = Vec::new();
    for id in ids {
        out.extend(rms_norm(
            &table[id * HIDDEN..(id + 1) * HIDDEN],
            Some(&norm),
            EPSILON,
        ));
    }
    assert_matches("embedding norm", &out, &r.f64s("out"));

    let r = record("head");
    let (unembed, norm, x) = (r.weight("unembed"), r.weight("norm"), r.f64s("x"));
    let keep = 20;
    assert_eq!(unembed.len(), 24 * HIDDEN);
    let mut out = Vec::new();
    for x in rows(&x, HIDDEN) {
        let normed = rms_norm(x, Some(&norm), EPSILON);
        out.extend(
            linear(&unembed[..keep * HIDDEN], &normed)
                .into_iter()
                .map(|logit| softcap(logit, 3.0)),
        );
    }
    let expected = r.f64s("out");
    assert_eq!(expected.len(), TOKENS * keep);
    assert_matches("head", &out, &expected);
    assert!(
        expected.iter().any(|logit| logit.abs() > 2.5),
        "the softcap engages"
    );
}

#[test]
fn plan_scalars_read_as_the_decimals_their_config_wrote() {
    assert_eq!(decimal(1e-6), 1e-6);
    assert_eq!(decimal(0.1), 0.1);
    assert_eq!(decimal(30.0), 30.0);
    assert!((decimal(22.627_417) - 22.627_416_997_969_522).abs() < 1e-8);
}

// ============================ the tiny program ============================

fn tiny_plan() -> ModelPlan {
    model_packs::by_alias("inkling2")
        .unwrap()
        .compile_tiny_plan()
        .unwrap()
}

fn tiny_weights() -> ReferenceWeights {
    weights_from_safetensors(TINY_CHECKPOINT).unwrap()
}

fn run(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    ends: &[usize],
    decode: Option<usize>,
) -> ReferenceOutput {
    execute_schedule(plan, weights, &TINY_TOKENS, ends, decode).unwrap()
}

#[test]
fn tiny_plan_is_the_float_activation_class_over_the_looped_program() {
    let plan = tiny_plan();
    let program = plan.inkling2.as_ref().unwrap();
    assert_eq!(
        program.numerics.class,
        Inkling2NumericClass::FloatActivation
    );
    assert_eq!(
        program.numerics.default_class,
        Inkling2NumericClass::DeclaredActivationQuant
    );
    assert_eq!(plan.layers.len(), 16);
    assert_eq!(
        program.loop_plan.execution_order,
        [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7]
    );
    let operations = plan.operations();
    for operation in [
        OperationKind::Nvfp4FloatInputExperts,
        OperationKind::Mxfp8FloatInputExperts,
        OperationKind::LoopMessageMix,
        OperationKind::LoopController,
        OperationKind::DynamicShortConv,
        OperationKind::AttentionLogScale,
        OperationKind::LogitsTruncate,
    ] {
        assert!(operations.contains(&operation), "{operation:?}");
    }
    let declared = pack::tiny_declared_plan().unwrap();
    assert_eq!(
        declared.inkling2.as_ref().unwrap().numerics.class,
        Inkling2NumericClass::DeclaredActivationQuant
    );
    assert!(
        declared
            .operations()
            .contains(&OperationKind::Nvfp4W4A4Experts)
    );
    assert!(
        declared
            .operations()
            .contains(&OperationKind::Mxfp8W8A8Experts)
    );
}

#[test]
fn every_schedule_gives_the_same_logits_and_leaves_the_same_state() {
    let (plan, weights) = (tiny_plan(), tiny_weights());
    let whole = run(&plan, &weights, &[], None);
    assert_eq!((whole.tokens, whole.vocab), (40, 300));
    assert_eq!(whole.logits.len(), 40 * 300);
    assert!(
        whole
            .logits
            .iter()
            .all(|logit| logit.is_finite() && logit.abs() < 3.0)
    );
    assert!(whole.decode.is_none());
    for (ends, decode) in [
        (vec![8, 12, 24], None),
        (vec![7, 11, 23], None),
        (vec![], Some(5)),
        (vec![3, 9], Some(31)),
    ] {
        let split = run(&plan, &weights, &ends, decode);
        assert_eq!(split.logits, whole.logits, "{ends:?} {decode:?}");
        assert_eq!(split.layer_hidden, whole.layer_hidden);
        assert_eq!(split.state, whole.state);
        assert_eq!(split.decode.map(|trace| trace.prefill_tokens), decode);
    }
    // A sliding slot keeps the 7 rows a later query can see; a full slot keeps all 40.
    for (layer, state) in plan.layers.iter().zip(&whole.state.layers) {
        let ReferenceLayerState::RelativeBias {
            positions,
            window,
            conv,
            key,
            ..
        } = state
        else {
            panic!("slot state");
        };
        match window {
            Some(8) => assert_eq!(positions[..], [33, 34, 35, 36, 37, 38, 39]),
            None => assert_eq!(positions.len(), 40),
            other => panic!("window {other:?} at slot {}", layer.index),
        }
        assert_eq!(key.len(), positions.len() * 32);
        assert_eq!(
            conv.iter().map(Vec::len).collect::<Vec<_>>(),
            [3 * 32, 3 * 32, 3 * 64, 3 * 64]
        );
    }
}

#[test]
fn log_scale_starts_at_the_floor_and_a_later_floor_is_another_program() {
    let (plan, weights) = (tiny_plan(), tiny_weights());
    let mut later = plan.clone();
    let mut full_layers = 0;
    for layer in &mut later.layers {
        if let AttentionPlan::RelativeBias(attention) = &mut layer.attention
            && let Some(scale) = &mut attention.log_scale
        {
            assert_eq!(scale.floor, 24);
            scale.floor = 25;
            full_layers += 1;
        }
    }
    assert_eq!(full_layers, 8);
    let (a, b) = (
        run(&plan, &weights, &[], None),
        run(&later, &weights, &[], None),
    );
    // Position 23 is token 24 of 24: tau is still 1. Position 24 is the first scaled one.
    assert_eq!(a.logits[..24 * 300], b.logits[..24 * 300]);
    assert_ne!(a.logits[24 * 300..25 * 300], b.logits[24 * 300..25 * 300]);
}

#[test]
fn padded_vocabulary_rows_are_never_computed() {
    let (plan, mut weights) = (tiny_plan(), tiny_weights());
    assert_eq!(plan.vocab_size, 320);
    let baseline = run(&plan, &weights, &[], None);
    let unembed = weights.get_mut(&TensorId::OutputProjection).unwrap();
    for value in &mut unembed.data[300 * 64..] {
        *value = f32::NAN;
    }
    assert_eq!(run(&plan, &weights, &[], None).logits, baseline.logits);
}

#[test]
fn declared_layers_run_their_quantizer_and_each_layer_moves_alone() {
    let weights = tiny_weights();
    let float = run(&tiny_plan(), &weights, &[], None);
    let declared_plan = pack::tiny_declared_plan().unwrap();
    let declared = run(&declared_plan, &weights, &[], None);
    assert!(declared.logits.iter().all(|logit| logit.is_finite()));
    // The two classes are different functions of the same weights: on this fixture they are
    // logits apart, orders of magnitude outside the tiny gate's band, so neither oracle can pass
    // for the other class.
    let apart = declared
        .logits
        .iter()
        .zip(&float.logits)
        .fold(0f32, |worst, (a, b)| worst.max((a - b).abs()));
    assert!(apart > 1.0, "{apart}");
    // The declared program is a function of the row alone, so it is schedule-invariant too.
    assert_eq!(
        run(&declared_plan, &weights, &[8, 12, 24], Some(30)).logits,
        declared.logits
    );
    // One layer on its quantizer (an NVFP4 layer, then an MXFP8 one): a mixed class of its own,
    // equal to neither, and the stream is untouched before that layer's first visit.
    for physical in [3usize, 7] {
        let mut modes = vec![ExpertActivationMode::Float; 8];
        modes[physical] = ExpertActivationMode::Declared;
        let plan = pack::tiny_plan_with_expert_activation(&modes).unwrap();
        assert_eq!(
            plan.inkling2.as_ref().unwrap().numerics.class,
            Inkling2NumericClass::PerLayerMixed
        );
        let mixed = run(&plan, &weights, &[], None);
        assert_ne!(mixed.logits, float.logits, "layer {physical}");
        assert_ne!(mixed.logits, declared.logits, "layer {physical}");
        let order = &plan.inkling2.as_ref().unwrap().loop_plan.execution_order;
        let first_visit = order
            .iter()
            .position(|&slot| slot as usize == physical)
            .unwrap();
        for (visit, &slot) in order.iter().enumerate() {
            let same = mixed.layer_hidden[slot as usize] == float.layer_hidden[slot as usize];
            assert_eq!(same, visit < first_visit, "layer {physical} visit {visit}");
        }
    }
}

#[test]
fn unsupported_shapes_and_broken_inputs_are_named_errors() {
    let (plan, weights) = (tiny_plan(), tiny_weights());
    let error =
        |plan: &ModelPlan, weights: &ReferenceWeights, tokens: &[u32], ends: &[usize], decode| {
            execute_schedule(plan, weights, tokens, ends, decode).unwrap_err()
        };
    assert_eq!(
        error(&plan, &weights, &[], &[], None),
        ReferenceError::EmptyInput
    );
    assert_eq!(
        error(&plan, &weights, &[1, 320], &[], None),
        ReferenceError::TokenOutOfRange {
            token: 320,
            vocab: 320
        }
    );
    for (ends, decode) in [
        (vec![0], None),
        (vec![3, 3], None),
        (vec![5], Some(4)),
        (vec![], Some(0)),
        (vec![], Some(41)),
    ] {
        assert!(matches!(
            error(&plan, &weights, &TINY_TOKENS, &ends, decode),
            ReferenceError::InvalidPlan { layer: None, .. }
        ));
    }
    // A missing tensor, and a bank plane of the wrong kind.
    let mut missing = weights.clone();
    let sink = tensor_id("model.llm.layers.5.attn.attn_sink_proj.weight");
    missing.remove(&sink).unwrap();
    assert_eq!(
        error(&plan, &missing, &TINY_TOKENS, &[], None),
        ReferenceError::MissingTensor(sink)
    );
    let mut nan = weights.clone();
    nan.get_mut(&tensor_id("model.llm.layers.6.mlp.experts.w2_weight"))
        .unwrap()
        .ints
        .as_mut()
        .unwrap()[17] = 0x7f;
    assert_eq!(
        error(&plan, &nan, &TINY_TOKENS, &[], None),
        ReferenceError::InvalidPlan {
            layer: Some(6),
            reason: "an expert bank holds an e4m3 NaN code",
        }
    );
    // Plan shapes with no implementation refuse by name before any math.
    let mut vision = plan.clone();
    vision.inkling2.as_mut().unwrap().vision_executed = true;
    assert_eq!(
        error(&vision, &weights, &TINY_TOKENS, &[], None),
        ReferenceError::UnsupportedOperation {
            layer: None,
            operation: "the Inkling 2 vision tower",
        }
    );
    let mut raw_mix = plan.clone();
    if let ResidualTopology::BlockAttention(residual) = &mut raw_mix.layers[4].residual {
        residual.normalize_values = false;
    }
    assert_eq!(
        error(&raw_mix, &weights, &TINY_TOKENS, &[], None),
        ReferenceError::UnsupportedOperation {
            layer: Some(4),
            operation: "a depth mix over unnormalized states",
        }
    );
    let mut order = pack::tiny_declared_plan().unwrap();
    order
        .inkling2
        .as_mut()
        .unwrap()
        .numerics
        .routed_weight_before_down = false;
    assert!(matches!(
        error(&order, &weights, &TINY_TOKENS, &[], None),
        ReferenceError::UnsupportedOperation { layer: Some(0), .. }
    ));
    // The generic entry points have no Inkling 2 program for images or drafts.
    assert!(matches!(
        crate::execute_mtp_standalone(&plan, &weights, &TINY_TOKENS, &[]),
        Err(ReferenceError::UnsupportedOperation { layer: None, .. })
    ));
}

#[test]
fn the_fixture_is_the_pinned_checkpoint_image_and_its_schedules() {
    let plan = tiny_plan();
    let fixture = crate::deterministic_fixture(&plan).unwrap();
    assert_eq!(fixture.token_ids, TINY_TOKENS);
    assert!(fixture.chunk_ends.is_empty() && fixture.decode_from.is_none());
    let names: Vec<&str> = fixture
        .checkpoint_files
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "model.safetensors",
            "config.json",
            "hf_quant_config.json",
            "schedule.json"
        ]
    );
    // 230 trunk tensors and the 56 planes of the 16 expert banks.
    assert_eq!(fixture.weights.len(), 286);
    let variants = crate::tiny_variants(&plan).unwrap();
    let schedules: Vec<_> = variants
        .iter()
        .map(|variant| {
            (
                variant.name,
                variant.fixture.chunk_ends.clone(),
                variant.fixture.decode_from,
            )
        })
        .collect();
    assert_eq!(
        schedules,
        [
            ("splits", vec![8, 12, 24], None),
            ("offsplits", vec![7, 11, 23], None),
            ("steps", vec![], Some(5)),
            ("declared", vec![], None),
        ]
    );
    let main = crate::execute_fixture(&plan, &fixture).unwrap();
    for variant in &variants {
        let same = crate::execute_fixture(&variant.plan, &variant.fixture)
            .unwrap()
            .logits
            == main.logits;
        assert_eq!(same, variant.plan == plan, "{}", variant.name);
    }
    assert_eq!(variants[3].plan, pack::tiny_declared_plan().unwrap());
    // A non-Inkling plan has no chunked prefill.
    let qwen = model_packs::by_alias("qwen3")
        .unwrap()
        .compile_tiny_plan()
        .unwrap();
    let mut chunked = crate::deterministic_fixture(&qwen).unwrap();
    chunked.chunk_ends = vec![1];
    assert_eq!(
        crate::execute_fixture(&qwen, &chunked).unwrap_err(),
        ReferenceError::UnsupportedOperation {
            layer: None,
            operation: "chunked reference prefill",
        }
    );
}
