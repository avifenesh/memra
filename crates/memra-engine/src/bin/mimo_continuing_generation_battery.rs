//! Bounded native continuing-generation comparison for pinned MiMo raw text.
//! F32 KV is an arithmetic control. Generated text is a diagnostic output,
//! not a source quality score or serving admission.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memra_engine::Engine;
use memra_engine::mimo_text_weights::MiMoTextWeights;
use memra_gguf::source::SafetensorsSource;
use memra_tokenizer::Tokenizer;
use memra_tokenizer::detokenize::Detokenizer;
use sha2::{Digest, Sha256};

type Fail = Box<dyn Error>;
const VOCAB: usize = 152_576;
const MAX_NEW: usize = 16;
const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
const TOKENIZER_SHA256: &str = "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e";
const TOKENIZER_CONFIG_SHA256: &str =
    "413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06";
const CHAT_GOLDENS_SHA256: &str =
    "e4dc470fed7f15916185bdb4e19a783600c79f657bffb545e4d7a09937403b43";

struct Case {
    label: String,
    prompt: String,
    ids: Vec<u32>,
}

fn parse_ids(field: &str) -> Result<Vec<u32>, Fail> {
    Ok(field
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?)
}

fn decode_hex_utf8(field: &str) -> Result<String, Fail> {
    if !field.len().is_multiple_of(2) {
        return Err("MiMo generation hex has odd length".into());
    }
    let bytes = field
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair)?, 16).map_err(Into::into))
        .collect::<Result<Vec<u8>, Fail>>()?;
    Ok(String::from_utf8(bytes)?)
}

fn parse_cases(corpus: &str, references: &str, tokenizer: &Tokenizer) -> Result<Vec<Case>, Fail> {
    let mut receipts = BTreeMap::new();
    for line in references.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 || receipts.contains_key(fields[0]) {
            return Err("MiMo generation reference ID columns changed".into());
        }
        receipts.insert(fields[0], (parse_ids(fields[1])?, parse_ids(fields[2])?));
    }
    let mut cases = Vec::new();
    for line in corpus.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 || !matches!(fields[0], "natural" | "code" | "multilingual") {
            return Err("MiMo generation corpus columns or prompt changed".into());
        }
        let prompt = decode_hex_utf8(fields[1])?;
        let (without, with) = receipts
            .remove(fields[0])
            .ok_or("MiMo generation corpus has no reference IDs")?;
        if without != tokenizer.encode(&prompt, false)
            || with != tokenizer.encode(&prompt, true)
            || without.is_empty()
            || without.len() + MAX_NEW > 256
            || without.iter().any(|&id| id as usize >= VOCAB)
        {
            return Err("MiMo generation tokenizer reference parity failed".into());
        }
        cases.push(Case {
            label: fields[0].to_string(),
            prompt,
            ids: without,
        });
    }
    if cases.len() != 3 || !receipts.is_empty() {
        return Err("MiMo generation corpus or reference case count changed".into());
    }
    Ok(cases)
}

fn parse_chat_seed(text: &str, tokenizer: &Tokenizer) -> Result<Case, Fail> {
    let mut seed = None;
    for line in text.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.first() != Some(&"answer_17_plus_25") {
            continue;
        }
        if fields.len() != 3 || seed.is_some() {
            return Err("MiMo chat golden answer row is ambiguous".into());
        }
        let prompt = decode_hex_utf8(fields[1])?;
        let ids = parse_ids(fields[2])?;
        if prompt
            != "<|im_start|>user\nWhat is 17 plus 25? Answer with the number only.<|im_end|><|im_start|>assistant\n"
            || ids != tokenizer.encode(&prompt, false)
            || ids.is_empty()
            || ids.len() + MAX_NEW > 256
            || ids.iter().any(|&id| id as usize >= VOCAB)
        {
            return Err("MiMo chat golden answer bytes or IDs differ from pinned source".into());
        }
        seed = Some(Case {
            label: "answer_17_plus_25".into(),
            prompt,
            ids,
        });
    }
    seed.ok_or_else(|| "MiMo chat goldens omit the answer-bearing seed".into())
}

fn argmax(logits: &[f32]) -> Result<u32, Fail> {
    if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
        return Err("MiMo generation logits have invalid width or values".into());
    }
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index].total_cmp(&logits[best]).is_gt() {
            best = index;
        }
    }
    Ok(best as u32)
}

fn drive(
    mut token: impl FnMut(u32) -> Result<Vec<f32>, Fail>,
    ids: &[u32],
    stop: &HashSet<u32>,
    tokenizer: &Tokenizer,
) -> Result<Vec<u32>, Fail> {
    let mut logits = Vec::new();
    for &id in ids {
        logits = token(id)?;
    }
    let mut generated = Vec::new();
    for step in 0..MAX_NEW {
        let choice = argmax(&logits)?;
        if choice as usize >= tokenizer.vocab_size() {
            return Err(format!("MiMo generated padded vocabulary id {choice}").into());
        }
        generated.push(choice);
        if stop.contains(&choice) || step + 1 == MAX_NEW {
            break;
        }
        logits = token(choice)?;
    }
    Ok(generated)
}

fn digest_ids(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn run() -> Result<(), Fail> {
    let mut args = std::env::args().skip(1);
    let root = args.next().ok_or(
        "usage: mimo_continuing_generation_battery <model-dir> <corpus.tsv> <ref-ids.tsv> <chat-goldens.tsv>",
    )?;
    let corpus_path = args.next().ok_or("MiMo generation needs a pinned corpus")?;
    let ids_path = args
        .next()
        .ok_or("MiMo generation needs pinned reference IDs")?;
    let chat_path = args
        .next()
        .ok_or("MiMo generation needs pinned chat goldens")?;
    if args.next().is_some() || std::env::var("MEMRA_BF16_MMV").as_deref() != Ok("1") {
        return Err("MiMo generation needs four paths and MEMRA_BF16_MMV=1".into());
    }
    let model_dir = Path::new(&root);
    for (name, expected) in [
        ("tokenizer.json", TOKENIZER_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
    ] {
        let found = format!("{:x}", Sha256::digest(fs::read(model_dir.join(name))?));
        if found != expected {
            return Err(format!("MiMo pinned {name} changed: {found}").into());
        }
    }
    let tokenizer = Tokenizer::from_hf_dir(model_dir)?;
    let decoder = Detokenizer::from_hf_dir(model_dir)?;
    if decoder.vocab_size() != tokenizer.vocab_size() {
        return Err("MiMo source encoder and decoder vocabulary changed".into());
    }
    let corpus_bytes = fs::read(corpus_path)?;
    let reference_bytes = fs::read(ids_path)?;
    let chat_bytes = fs::read(chat_path)?;
    let chat_sha = format!("{:x}", Sha256::digest(&chat_bytes));
    if chat_sha != CHAT_GOLDENS_SHA256 {
        return Err("MiMo source chat goldens SHA changed".into());
    }
    let mut cases = parse_cases(
        std::str::from_utf8(&corpus_bytes)?,
        std::str::from_utf8(&reference_bytes)?,
        &tokenizer,
    )?;
    cases.push(parse_chat_seed(
        std::str::from_utf8(&chat_bytes)?,
        &tokenizer,
    )?);
    let stop = tokenizer.eog_ids().into_iter().collect::<HashSet<_>>();
    let source = Arc::new(SafetensorsSource::open(model_dir)?);
    let cards = [Engine::new(0)?, Engine::new(1)?];
    let engines = [&cards[0], &cards[1]];
    if cards[0].stream().context().ordinal() == cards[1].stream().context().ordinal() {
        return Err("MiMo generation requires two distinct GPUs".into());
    }
    let text = MiMoTextWeights::load(engines, source)?;
    println!("format\tmimo-continuing-generation-battery-v1");
    println!("source\tXiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4");
    println!("corpus_sha256\t{:x}", Sha256::digest(&corpus_bytes));
    println!("ref_ids_sha256\t{:x}", Sha256::digest(&reference_bytes));
    println!("chat_goldens_sha256\t{chat_sha}");
    println!("shape\traw_text_and_pinned_chat\tmax_new={MAX_NEW}\tbf16_mmv=1");
    for case in cases {
        println!(
            "prompt\t{}\tinput_tokens={}\ttext={:?}",
            case.label,
            case.ids.len(),
            case.prompt,
        );
        let mut outputs = Vec::new();
        for arm in ["f32", "nvfp4", "s5"] {
            let start = Instant::now();
            let ids = match arm {
                "f32" => {
                    let mut sequence = text.text_forward(engines)?;
                    drive(|id| sequence.token(id), &case.ids, &stop, &tokenizer)?
                }
                "nvfp4" => {
                    let mut sequence = text.compressed_text_forward(
                        engines,
                        case.ids.len() + MAX_NEW,
                        [FOUR_GIB; 2],
                    )?;
                    drive(|id| sequence.token(id), &case.ids, &stop, &tokenizer)?
                }
                "s5" => {
                    let mut sequence = text.compressed_text_forward_s5_g16(
                        engines,
                        case.ids.len() + MAX_NEW,
                        [FOUR_GIB; 2],
                    )?;
                    drive(|id| sequence.token(id), &case.ids, &stop, &tokenizer)?
                }
                _ => unreachable!(),
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
            let text_output = decoder.decode(&ids);
            println!(
                "generation\t{}\t{}\tnew_tokens={}\tids_sha256={}\tids={:?}\tdecoded={:?}\twall_ms={elapsed_ms:.6}",
                case.label,
                arm,
                ids.len(),
                digest_ids(&ids),
                ids,
                text_output,
            );
            outputs.push(ids);
        }
        println!(
            "agreement\t{}\tf32_nvfp4_prefix={}\tf32_s5_prefix={}\tnvfp4_s5_prefix={}",
            case.label,
            longest_common_prefix(&outputs[0], &outputs[1]),
            longest_common_prefix(&outputs[0], &outputs[2]),
            longest_common_prefix(&outputs[1], &outputs[2]),
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_continuing_generation_battery: {error}");
        std::process::exit(1);
    }
}
