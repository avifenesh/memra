//! Real-artifact pair oracles, deliberately ignored on ordinary test runs.
//! Run each arm in a fresh process through the lane's run-pair-cell.sh. A same-device
//! fixture cannot exercise the symmetric peer-AR program. Required inputs are
//! TP_SPEC_MODEL, TP_SPEC_PROMPT_IDS, TP_SPEC_OUT and TP_SPEC_ARM=plain|tp|pp.
//! O1 here checks the engine plain route with the spec door off/on. The companion
//! HTTP cell checks the worker's MEMRA_SPEC_K=0 dispatch.

use crate::{Engine, cache::Cache, forward::argmax, glm_spec::Glm5SpecKnobs, hybrid::HybridModel};
use memra_gguf::source::SafetensorsSource;
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

type Res<T> = Result<T, Box<dyn Error>>;

fn required(key: &str) -> Res<String> {
    Ok(std::env::var(key)?)
}

fn ids(path: &Path) -> Res<Vec<u32>> {
    fs::read_to_string(path)?
        .split_whitespace()
        .map(|x| Ok(x.parse()?))
        .collect()
}

fn write_ids(path: &Path, tokens: &[u32]) -> Res<()> {
    fs::write(
        path,
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    )?;
    Ok(())
}

fn dump(path: &Path, logits: &[f32]) -> Res<()> {
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite target logits"
    );
    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    for v in logits {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn load_logits(path: &Path) -> Res<Vec<f32>> {
    let bytes = fs::read(path)?;
    assert_eq!(bytes.len() % 4, 0);
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn cache(e: &Engine, m: &HybridModel, prompt: &[u32]) -> Res<Cache> {
    let mut c = Cache::new_planned(e, &m.cfg, &m.plan, prompt.len() + 192)?;
    m.prime_cache(e, prompt, &mut c, 0)?;
    Ok(c)
}

fn plain(e: &Engine, m: &HybridModel, prompt: &[u32]) -> Res<Vec<u32>> {
    let mut c = Cache::new_planned(e, &m.cfg, &m.plan, prompt.len() + 192)?;
    let (logits, _, _) = m.prime_cache(e, prompt, &mut c, 0)?;
    let mut out = vec![argmax(&logits) as u32];
    while out.len() < 160 {
        let logits = m.decode_step(e, *out.last().unwrap(), &mut c)?;
        out.push(argmax(&logits) as u32);
    }
    Ok(out)
}

fn o3(e: &Engine, m: &HybridModel, prompt: &[u32], tape: &[u32]) -> Res<()> {
    let mut serial = cache(e, m, prompt)?;
    let mut batch = cache(e, m, prompt)?;
    let mut offset = 0;
    let mut width = 1;
    while offset < tape.len() {
        let t = width.min(tape.len() - offset);
        let rows = &tape[offset..offset + t];
        let (ll, _, ckpt) = m.glm5_verify_rows(e, rows, &mut batch)?;
        let got = e.dtoh(&ll)?;
        let vocab = got.len() / t;
        assert!(vocab > 0);
        for (r, &token) in rows.iter().enumerate() {
            let expected = m.decode_step(e, token, &mut serial)?;
            assert_eq!(expected.len(), vocab);
            for (v, (&a, &b)) in expected
                .iter()
                .zip(&got[r * vocab..(r + 1) * vocab])
                .enumerate()
            {
                assert!(a.is_finite() && b.is_finite());
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "O3 token={} t={t} vocab={v} serial={a} verify={b}",
                    offset + r
                );
            }
        }
        m.glm5_verify_rollback(e, &mut batch, &ckpt, t)?;
        assert_eq!(batch.pos, serial.pos);
        offset += t;
        width = width % 7 + 1;
    }
    eprintln!("O3 PASS: 160 teacher-forced tokens, widths 1..7, every logit bit-identical");
    Ok(())
}

fn spec(e: &Engine, m: &HybridModel, prompt: &[u32], out: &Path) -> Res<()> {
    let mut metadata = std::io::BufWriter::new(fs::File::create(out.join("rounds.txt"))?);
    let mut accepted = Vec::new();
    let mut round = 0;
    let mut emitted = 1usize; // prompt boundary anchor
    let mut observer = |pos: usize, rows: &[u32], logits: &[f32], j: usize| -> Res<()> {
        dump(&out.join(format!("round-{round}.f32")), logits)?;
        writeln!(
            metadata,
            "{pos} {} {}",
            j + 1,
            rows.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        )?;
        accepted.extend(
            rows[1..=j]
                .iter()
                .copied()
                .take(160usize.saturating_sub(emitted)),
        );
        emitted += j + 1;
        round += 1;
        Ok(())
    };
    let knobs = Glm5SpecKnobs {
        verify_observer: Some(&mut observer),
        ..Default::default()
    };
    let (tokens, drafted, accepted_count) = m.generate_spec_glm5_gated(e, prompt, 160, 6, knobs)?;
    assert_eq!(tokens.len(), 160, "oracle tape stopped early");
    assert!(drafted > 0 && round > 0, "spec oracle did not engage");
    metadata.flush()?;
    write_ids(&out.join("spec.ids"), &tokens)?;
    write_ids(&out.join("accepted.ids"), &accepted)?;
    eprintln!("SPEC K=6 rounds={round} drafted={drafted} accepted={accepted_count}");
    Ok(())
}

fn o2_replay(e: &Engine, m: &HybridModel, prompt: &[u32], tp: &Path, out: &Path) -> Res<()> {
    let band: f32 = required("TP_SPEC_O2_BAND")?.parse()?;
    assert!(band.is_finite() && band > 0.0);
    let mut c = cache(e, m, prompt)?;
    let meta = fs::read_to_string(tp.join("rounds.txt"))?;
    assert!(!meta.is_empty());
    let mut worst = 0f32;
    for (round, line) in meta.lines().enumerate() {
        let fields: Vec<usize> = line
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let (pos, keep) = (fields[0], fields[1]);
        let rows: Vec<u32> = fields[2..].iter().map(|&x| x as u32).collect();
        assert_eq!(c.pos, pos);
        let (logits, _, ckpt) = m.glm5_verify_rows(e, &rows, &mut c)?;
        let pp = e.dtoh(&logits)?;
        let target = load_logits(&tp.join(format!("round-{round}.f32")))?;
        dump(&out.join(format!("replay-{round}.f32")), &pp)?;
        assert_eq!(pp.len(), target.len());
        let vocab = pp.len() / rows.len();
        for (r, (a, b)) in pp
            .chunks_exact(vocab)
            .zip(target.chunks_exact(vocab))
            .enumerate()
        {
            assert_eq!(argmax(a), argmax(b), "O2 argmax round={round} row={r}");
            for (&reference, &got) in a.iter().zip(b) {
                assert!(reference.is_finite() && got.is_finite());
                let delta = (reference - got).abs() / reference.abs().max(1e-6);
                worst = worst.max(delta);
                assert!(
                    delta <= band,
                    "O2 band round={round} row={r} delta={delta} limit={band}"
                );
            }
        }
        m.glm5_verify_rollback(e, &mut c, &ckpt, keep)?;
    }
    assert_eq!(
        ids(&tp.join("spec.ids"))?,
        ids(&out.join("spec.ids"))?,
        "O2 output sequence"
    );
    assert_eq!(
        ids(&tp.join("accepted.ids"))?,
        ids(&out.join("accepted.ids"))?,
        "O2 accepted-token sequence"
    );
    eprintln!(
        "O2 PASS: actual TP verify rows replayed on PP, argmax equal, max_rel={worst} band={band}, 160-token tape and accepted sequence equal"
    );
    Ok(())
}

#[test]
#[ignore = "requires the scheduled two-device pair and pinned full model/drafter artifacts"]
fn pair_arm() -> Res<()> {
    let arm = required("TP_SPEC_ARM")?;
    let base = PathBuf::from(required("TP_SPEC_OUT")?);
    let out = base.join(&arm);
    fs::create_dir_all(&out)?;
    let prompt = ids(Path::new(&required("TP_SPEC_PROMPT_IDS")?))?;
    assert!(prompt.len() >= 2);
    let e = Engine::new(0)?;
    // A real second device must exist even on the PP arm. Same-device emulation
    // cannot pass this gate or silently turn it into a single-card check.
    let second = Engine::new(1)?;
    assert_ne!(e.ctx().ordinal(), second.ctx().ordinal());
    drop(second);
    let source = SafetensorsSource::open(Path::new(&required("TP_SPEC_MODEL")?))?;
    let m = HybridModel::load_from_source(&e, &source)?;
    match arm.as_str() {
        "plain" => {
            assert_eq!(m.glm5_tp_rank_count(), Some(2));
            assert!(!crate::glm_spec::glm5_spec_tp_on());
            write_ids(&out.join("plain.ids"), &plain(&e, &m, &prompt)?)?;
        }
        "tp" => {
            assert_eq!(m.glm5_tp_spec_refusal(), None);
            let tape = plain(&e, &m, &prompt)?;
            assert_eq!(
                tape,
                ids(&base.join("plain/plain.ids"))?,
                "O1 engine plain route with spec door ON/OFF"
            );
            write_ids(&out.join("k0.ids"), &tape)?;
            o3(&e, &m, &prompt, &tape)?;
            spec(&e, &m, &prompt, &out)?;
        }
        "pp" => {
            assert!(!m.glm5_has_tp_shards());
            assert_eq!(crate::pp::pp_cuts(m.layers.len()).map(|c| c.len()), Some(3));
            spec(&e, &m, &prompt, &out)?;
            o2_replay(&e, &m, &prompt, &base.join("tp"), &out)?;
        }
        _ => return Err("TP_SPEC_ARM must be plain, tp or pp".into()),
    }
    Ok(())
}
