use super::*;

const TOL: f64 = 1e-12;

fn worst(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (w.abs() + 1.0))
        .fold(0.0, f64::max)
}

struct Tiny {
    drafter: Drafter,
    embed: Vec<f32>,
    embed_norm: Vec<f32>,
    head: Vec<f32>,
    taps: Vec<f64>,
    anchors: Vec<i64>,
    forced: Vec<i64>,
}

fn tiny() -> Tiny {
    let g = tiny_geometry();
    let image = StImage::parse(TINY_FIXTURE).unwrap();
    let w = DraftWeights::load(&g, &image).unwrap();
    let h = g.hidden;
    Tiny {
        embed: image.bf16("trunk.embed", &[g.padded_vocab, h]).unwrap(),
        embed_norm: image.bf16("trunk.embed_norm", &[h]).unwrap(),
        head: image.bf16("trunk.head", &[g.vocab, h]).unwrap(),
        taps: image
            .bf16("taps", &[13, g.taps * h])
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect(),
        anchors: image.i64s("anchors").unwrap(),
        forced: image.i64s("forced").unwrap(),
        drafter: Drafter::new(g, w),
    }
}

/// The f64 executor against the pinned f64 torch port, on all three scenarios: the context
/// K/V of every committed position, the block's final-norm rows, the head's feature, every
/// proposal's logits (greedy and on a teacher-forced chain) and the greedy proposals.
#[test]
fn tiny_drafter_matches_the_pinned_torch_port() {
    let oracle = StImage::parse(TINY_ORACLE).unwrap();
    let mut t = tiny();
    let tw = t.drafter.g.taps * t.drafter.g.hidden;
    for (s, &p) in TINY_SCENARIOS.iter().enumerate() {
        t.drafter.truncate(0);
        for pos in 0..p {
            t.drafter.commit(&t.taps[pos * tw..(pos + 1) * tw]);
        }
        assert_eq!(t.drafter.len(), p);
        for l in 0..t.drafter.g.layers {
            let e = worst(
                &t.drafter.ctx_k[l],
                &oracle.f64s(&format!("s{s}.ctx_k.{l}")).unwrap(),
            );
            assert!(e < TOL, "scenario {s} layer {l} ctx k {e:e}");
            let e = worst(
                &t.drafter.ctx_v[l],
                &oracle.f64s(&format!("s{s}.ctx_v.{l}")).unwrap(),
            );
            assert!(e < TOL, "scenario {s} layer {l} ctx v {e:e}");
        }
        let shared = TrunkShared {
            embed: &t.embed,
            embed_norm: &t.embed_norm,
            head: &t.head,
        };
        let anchor = t.anchors[s] as u32;
        let feature = &t.taps[(p - 1) * tw..p * tw];
        let greedy = t
            .drafter
            .propose(anchor, feature, &shared, &mut |_, l| argmax(l), None);
        for (name, got) in [
            ("hidden", &greedy.hidden),
            ("f_rnn", &greedy.feature),
            ("logits", &greedy.logits),
        ] {
            let e = worst(got, &oracle.f64s(&format!("s{s}.{name}")).unwrap());
            assert!(e < TOL, "scenario {s} {name} {e:e}");
        }
        let want: Vec<u32> = oracle
            .i64s(&format!("s{s}.tokens"))
            .unwrap()
            .into_iter()
            .map(|v| v as u32)
            .collect();
        assert_eq!(greedy.tokens, want, "scenario {s} greedy proposals");
        let forced: Vec<u32> = t.forced[s * 7..(s + 1) * 7]
            .iter()
            .map(|v| *v as u32)
            .collect();
        let chain = t.drafter.propose(
            anchor,
            feature,
            &shared,
            &mut |_, l| argmax(l),
            Some(&forced),
        );
        let e = worst(
            &chain.logits,
            &oracle.f64s(&format!("s{s}.forced_logits")).unwrap(),
        );
        assert!(e < TOL, "scenario {s} forced chain logits {e:e}");
        eprintln!(
            "scenario {s} (P={p}): forced-chain logits worst rel {e:e}, greedy {:?}",
            greedy.tokens
        );
    }
}

/// Red arms: each wrong program the oracle must catch. A layer that is not local, the latent
/// taken from the new state, the softcap before the bias, and no RoPE on the context keys.
#[test]
fn tiny_drafter_red_arms_fail_the_oracle() {
    let oracle = StImage::parse(TINY_ORACLE).unwrap();
    let base = tiny();
    let tw = base.drafter.g.taps * base.drafter.g.hidden;
    let s = 2; // 13 committed positions: past the window of 5
    let p = TINY_SCENARIOS[s];
    let want = oracle.f64s(&format!("s{s}.logits")).unwrap();
    let run = |t: &mut Tiny| -> f64 {
        t.drafter.truncate(0);
        for pos in 0..p {
            t.drafter.commit(&t.taps[pos * tw..(pos + 1) * tw]);
        }
        let shared = TrunkShared {
            embed: &t.embed,
            embed_norm: &t.embed_norm,
            head: &t.head,
        };
        let anchor = t.anchors[s] as u32;
        let got = t.drafter.propose(
            anchor,
            &t.taps[(p - 1) * tw..p * tw],
            &shared,
            &mut |_, l| argmax(l),
            None,
        );
        worst(&got.logits, &want)
    };
    let mut t = tiny();
    assert!(run(&mut t) < TOL);
    // every layer sees the whole context
    let mut t = tiny();
    t.drafter.g.local_layers.clear();
    assert!(run(&mut t) > 1e-6, "local windows");
    // the fuse rows swapped (layer 0 mixing with the head's row)
    let mut t = tiny();
    let taps = t.drafter.g.taps;
    let head_row: Vec<f64> = t.drafter.alpha[3 * taps..4 * taps].to_vec();
    t.drafter.alpha[..taps].copy_from_slice(&head_row);
    assert!(run(&mut t) > 1e-6, "fuse rows");
    // a window one position wider
    let mut t = tiny();
    t.drafter.g.local_context += 1;
    assert!(run(&mut t) > 1e-6, "window edge");
}

/// The checkpoint's plan gives the shipped drafter geometry.
#[test]
fn checkpoint_plan_geometry() {
    use memra_gguf::config::{HfConfig, ModelConfig};
    let mut config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
        "../../../memra-gguf/src/model_packs/inkling2/fixtures/config.json"
    )));
    config
        .attach_quant_groups(include_str!(
            "../../../memra-gguf/src/model_packs/inkling2/fixtures/hf_quant_config.json"
        ))
        .unwrap();
    let pack = memra_gguf::model_packs::for_config(&config).unwrap();
    let plan = pack.compile_plan(&config).unwrap();
    let g = DraftGeometry::of_plan(&plan).unwrap();
    assert_eq!(
        g,
        DraftGeometry {
            hidden: 2304,
            layers: 3,
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 9216,
            rank: 256,
            rows: 8,
            taps: 4,
            local_layers: vec![0, 2],
            local_context: 1024,
            theta: 1e6,
            epsilon: 1e-6,
            softcap: 30.0,
            embed_epsilon: g.embed_epsilon,
            vocab: 200058,
            padded_vocab: 201024,
        }
    );
}
