//! Compact anchor-bounded supervision extraction from an own-generated DSpark token tape.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use std::io::Write;

const FORMAT: &str = "memra-dspark-anchors-v1";

struct Args {
    model: String,
    pairs: String,
    output_dir: String,
    anchors_per_pair: usize,
    gamma: usize,
    top_k: usize,
    chunk: usize,
    temperature: f32,
    seed: u64,
}

struct Pair {
    id: usize,
    split: String,
    mode: String,
    category: String,
    prompt_len: usize,
    response_len: usize,
    tokens: Vec<u32>,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 4 {
        return Err(
            "usage: dspark-extract <model> <pairs.tsv> <output-dir> [--anchors N] \
             [--gamma N] [--top-k N] [--chunk N] [--temperature T] [--seed N]"
                .into(),
        );
    }
    let mut args = Args {
        model: argv[1].clone(),
        pairs: argv[2].clone(),
        output_dir: argv[3].clone(),
        anchors_per_pair: 4,
        gamma: 5,
        top_k: 64,
        chunk: 512,
        temperature: 0.7,
        seed: 20_260_811,
    };
    let mut index = 4;
    while index < argv.len() {
        let value = argv
            .get(index + 1)
            .ok_or_else(|| format!("{} needs a value", argv[index]))?;
        match argv[index].as_str() {
            "--anchors" => args.anchors_per_pair = value.parse()?,
            "--gamma" => args.gamma = value.parse()?,
            "--top-k" => args.top_k = value.parse()?,
            "--chunk" => args.chunk = value.parse()?,
            "--temperature" => args.temperature = value.parse()?,
            "--seed" => args.seed = value.parse()?,
            flag => return Err(format!("unknown flag {flag}").into()),
        }
        index += 2;
    }
    if args.anchors_per_pair == 0
        || args.gamma == 0
        || args.top_k == 0
        || args.chunk < 2
        || args.temperature <= 0.0
    {
        return Err("invalid DSpark extraction parameters".into());
    }
    Ok(args)
}

fn parse_pairs(path: &str) -> Result<Vec<Pair>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut pairs = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line_no == 0 {
            if line != "# memra-dspark-pairs-v1" {
                return Err(format!("unexpected pairs header: {line}").into());
            }
            continue;
        }
        let fields: Vec<_> = line.splitn(8, '\t').collect();
        if fields.len() != 8 {
            return Err(format!("pairs line {} is incomplete", line_no + 1).into());
        }
        let prompt_len: usize = fields[4].parse()?;
        let response_len: usize = fields[5].parse()?;
        let total_len: usize = fields[6].parse()?;
        let tokens: Vec<u32> = fields[7]
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        if tokens.len() != total_len || prompt_len + response_len != total_len {
            return Err(format!(
                "pairs line {} length mismatch: prompt={prompt_len} response={response_len} total={total_len} ids={}",
                line_no + 1,
                tokens.len()
            )
            .into());
        }
        pairs.push(Pair {
            id: fields[0].parse()?,
            split: fields[1].to_string(),
            mode: fields[2].to_string(),
            category: fields[3].to_string(),
            prompt_len,
            response_len,
            tokens,
        });
    }
    if pairs.is_empty() {
        return Err("pairs file contains no records".into());
    }
    Ok(pairs)
}

fn load_model(engine: &Engine, path: &str) -> Result<HybridModel, Box<dyn std::error::Error>> {
    let model_path = std::path::Path::new(path);
    if model_path.is_dir() {
        let source = memra_gguf::source::SafetensorsSource::open(model_path)?;
        Ok(HybridModel::load_from_source(engine, &source)?)
    } else {
        let gguf = GgufFile::open(path)?;
        Ok(HybridModel::load(engine, &gguf)?)
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn choose_anchors(pair: &Pair, count: usize, gamma: usize, seed: u64) -> Vec<usize> {
    if pair.prompt_len == 0 || pair.response_len < gamma {
        return Vec::new();
    }
    // p is the anchor token. The extractor stores h[p-1], the carrier the live NextN/DSpark
    // path pairs with token[p], so p=0 is never admissible.
    let first = pair.prompt_len.saturating_sub(1).max(1);
    let last = pair.tokens.len() - gamma - 1;
    let candidates = last - first + 1;
    let take = count.min(candidates);
    let mut state = seed ^ pair.id as u64;
    let mut anchors = Vec::with_capacity(take);
    for bucket in 0..take {
        let start = first + bucket * candidates / take;
        let end = first + (bucket + 1) * candidates / take;
        let width = end - start;
        anchors.push(start + splitmix64(&mut state) as usize % width.max(1));
    }
    anchors.sort_unstable();
    anchors.dedup();
    anchors
}

fn write_u32s(file: &mut std::fs::File, values: &[u32]) -> std::io::Result<()> {
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32s(file: &mut std::fs::File, values: &[f32]) -> std::io::Result<()> {
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let pairs = parse_pairs(&args.pairs)?;
    let output_dir = std::path::Path::new(&args.output_dir);
    std::fs::create_dir_all(output_dir)?;
    let mut hidden_file = std::fs::File::create(output_dir.join("hiddens.bf16"))?;
    let mut token_file = std::fs::File::create(output_dir.join("tokens.u32"))?;
    let mut top_id_file = std::fs::File::create(output_dir.join("top_ids.u32"))?;
    let mut top_logit_file = std::fs::File::create(output_dir.join("top_logits.f32"))?;
    let mut top_prob_file = std::fs::File::create(output_dir.join("top_probs.f32"))?;
    let mut tail_file = std::fs::File::create(output_dir.join("tail_probs.f32"))?;
    let mut index_file = std::fs::File::create(output_dir.join("index.tsv"))?;
    writeln!(
        index_file,
        "record\tpair_id\tanchor_pos\tprompt_len\tsplit\tmode\tcategory"
    )?;

    let engine = Engine::new(0)?;
    let model = load_model(&engine, &args.model)?;
    let hidden_size = model.cfg.n_embd as usize;
    let mut records = 0usize;
    let mut skipped_short = 0usize;
    let started = std::time::Instant::now();
    for (pair_index, pair) in pairs.iter().enumerate() {
        let anchors = choose_anchors(pair, args.anchors_per_pair, args.gamma, args.seed);
        if anchors.is_empty() {
            skipped_short += 1;
            continue;
        }
        let extracted = model.extract_dspark_anchors(
            &engine,
            &pair.tokens,
            &anchors,
            args.gamma,
            args.top_k,
            args.chunk,
            args.temperature,
        )?;
        for record in extracted {
            if record.hidden.len() != hidden_size
                || record.tokens.len() != args.gamma + 1
                || record.target_top_ids.len() != args.gamma * args.top_k
                || record.target_top_logits.len() != args.gamma * args.top_k
                || record.target_top_probs.len() != args.gamma * args.top_k
                || record.target_tail_probs.len() != args.gamma
            {
                return Err(format!("DSpark record shape mismatch for pair {}", pair.id).into());
            }
            for slot in 0..args.gamma {
                let start = slot * args.top_k;
                let mass: f32 = record.target_top_probs[start..start + args.top_k]
                    .iter()
                    .sum::<f32>()
                    + record.target_tail_probs[slot];
                if (mass - 1.0).abs() > 2.0e-5 {
                    return Err(format!(
                        "DSpark target mass mismatch pair={} pos={} slot={} mass={mass}",
                        pair.id, record.position, slot
                    )
                    .into());
                }
            }
            for value in record.hidden {
                hidden_file.write_all(&f32_to_bf16(value).to_le_bytes())?;
            }
            write_u32s(&mut token_file, &record.tokens)?;
            write_u32s(&mut top_id_file, &record.target_top_ids)?;
            write_f32s(&mut top_logit_file, &record.target_top_logits)?;
            write_f32s(&mut top_prob_file, &record.target_top_probs)?;
            write_f32s(&mut tail_file, &record.target_tail_probs)?;
            writeln!(
                index_file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                records,
                pair.id,
                record.position,
                pair.prompt_len,
                pair.split,
                pair.mode,
                pair.category,
            )?;
            records += 1;
        }
        eprintln!(
            "[dspark-extract] pair={}/{} id={} anchors={} records={} elapsed={:.2}s",
            pair_index + 1,
            pairs.len(),
            pair.id,
            anchors.len(),
            records,
            started.elapsed().as_secs_f64(),
        );
    }
    for file in [
        &hidden_file,
        &token_file,
        &top_id_file,
        &top_logit_file,
        &top_prob_file,
        &tail_file,
        &index_file,
    ] {
        file.sync_data()?;
    }
    std::fs::write(
        output_dir.join("extraction.meta.json"),
        format!(
            "{{\"format\":\"{FORMAT}\",\"pairs\":{},\"records\":{},\"skipped_short\":{},\"hidden_size\":{},\"anchors_per_pair\":{},\"gamma\":{},\"top_k\":{},\"temperature\":{},\"chunk\":{},\"seed\":{}}}\n",
            pairs.len(),
            records,
            skipped_short,
            hidden_size,
            args.anchors_per_pair,
            args.gamma,
            args.top_k,
            args.temperature,
            args.chunk,
            args.seed,
        ),
    )?;
    println!(
        "DSPARK-EXTRACTION-DONE pairs={} records={records}",
        pairs.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Pair, choose_anchors};

    #[test]
    fn anchors_are_deterministic_unique_and_response_bounded() {
        let pair = Pair {
            id: 17,
            split: "train".into(),
            mode: "think".into(),
            category: "code".into(),
            prompt_len: 20,
            response_len: 100,
            tokens: vec![1; 120],
        };
        let anchors = choose_anchors(&pair, 4, 5, 42);
        assert_eq!(anchors, choose_anchors(&pair, 4, 5, 42));
        assert_eq!(anchors.len(), 4);
        assert!(anchors.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            anchors
                .iter()
                .all(|position| *position >= 19 && *position + 5 < 120)
        );
    }

    #[test]
    fn anchor_always_has_a_predecessor_hidden() {
        let pair = Pair {
            id: 3,
            split: "train".into(),
            mode: "nothink".into(),
            category: "chat".into(),
            prompt_len: 1,
            response_len: 10,
            tokens: vec![1; 11],
        };
        let anchors = choose_anchors(&pair, 4, 5, 9);
        assert!(!anchors.is_empty());
        assert!(anchors.iter().all(|position| *position >= 1));
    }
}
