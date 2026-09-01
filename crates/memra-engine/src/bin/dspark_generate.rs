//! Chunked, resumable own-generation corpus builder for the frozen DSpark 9B pilot.
//!
//! Prompt text is NUL-delimited in `prompts.promptpack`; aligned metadata is TSV. Each output
//! directory is one cheap-to-preempt chunk. The token-id tape is authoritative for replay, while
//! decoded response files make the prompt-response corpus directly inspectable.

use memra_engine::Engine;
use memra_engine::decode::GenParams;
use memra_engine::hybrid::HybridModel;
use memra_engine::sampler::{Sampler, SamplerConfig};
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use memra_tokenizer::chat::{ThinkMode, Turn};
use std::io::Write;

const FORMAT: &str = "memra-dspark-pairs-v1";

#[derive(Clone)]
struct PromptMeta {
    id: usize,
    split: String,
    mode: String,
    category: String,
}

struct Args {
    model: String,
    prompt_pack: String,
    prompt_meta: String,
    output_dir: String,
    offset: usize,
    limit: usize,
    max_new: usize,
    temperature: f32,
    seed: u64,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 5 {
        return Err(
            "usage: dspark-generate <model> <prompts.promptpack> <prompts.tsv> <output-dir> \
             [--offset N] [--limit N] [--max-new N] [--temperature T] [--seed N]"
                .into(),
        );
    }
    let mut args = Args {
        model: argv[1].clone(),
        prompt_pack: argv[2].clone(),
        prompt_meta: argv[3].clone(),
        output_dir: argv[4].clone(),
        offset: 0,
        limit: 100,
        max_new: 512,
        temperature: 0.7,
        seed: 20_260_811,
    };
    let mut i = 5;
    while i < argv.len() {
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{} needs a value", argv[i]))?;
        match argv[i].as_str() {
            "--offset" => args.offset = value.parse()?,
            "--limit" => args.limit = value.parse()?,
            "--max-new" => args.max_new = value.parse()?,
            "--temperature" => args.temperature = value.parse()?,
            "--seed" => args.seed = value.parse()?,
            flag => return Err(format!("unknown flag {flag}").into()),
        }
        i += 2;
    }
    if args.limit == 0 || args.max_new == 0 || args.temperature <= 0.0 {
        return Err("limit/max-new must be positive and temperature must be > 0".into());
    }
    Ok(args)
}

fn load_prompts(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if !bytes.ends_with(&[0]) {
        return Err("prompt pack must end with a NUL record delimiter".into());
    }
    let mut prompts = Vec::new();
    for record in bytes[..bytes.len() - 1].split(|byte| *byte == 0) {
        prompts.push(String::from_utf8(record.to_vec())?);
    }
    Ok(prompts)
}

fn load_metadata(path: &str) -> Result<Vec<PromptMeta>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_no, line) in text.lines().enumerate().skip(1) {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 6 {
            return Err(format!(
                "metadata line {} has {} fields, expected 6",
                line_no + 1,
                fields.len()
            )
            .into());
        }
        let id: usize = fields[0].parse()?;
        if id != rows.len() {
            return Err(format!("metadata id {id} is not contiguous at row {}", rows.len()).into());
        }
        if !matches!(fields[1], "train" | "heldout") {
            return Err(format!("metadata id {id} has invalid split {}", fields[1]).into());
        }
        if !matches!(fields[2], "think" | "nothink") {
            return Err(format!("metadata id {id} has invalid mode {}", fields[2]).into());
        }
        rows.push(PromptMeta {
            id,
            split: fields[1].to_string(),
            mode: fields[2].to_string(),
            category: fields[3].to_string(),
        });
    }
    Ok(rows)
}

fn load_model(
    engine: &Engine,
    path: &str,
) -> Result<(HybridModel, Tokenizer), Box<dyn std::error::Error>> {
    let model_path = std::path::Path::new(path);
    if model_path.is_dir() {
        let source = memra_gguf::source::SafetensorsSource::open(model_path)?;
        Ok((
            HybridModel::load_from_source(engine, &source)?,
            Tokenizer::from_hf_dir(model_path)
                .map_err(|error| format!("HF tokenizer init failed: {error}"))?,
        ))
    } else {
        let gguf = GgufFile::open(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        Ok((HybridModel::load(engine, &gguf)?, tokenizer))
    }
}

fn completed_rows(
    path: &std::path::Path,
    output_dir: &std::path::Path,
    offset: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(0);
    };
    let mut count = 0usize;
    for (line_no, line) in text.lines().enumerate() {
        if line_no == 0 {
            if line != format!("# {FORMAT}") {
                return Err(format!("unexpected pairs header: {line}").into());
            }
            continue;
        }
        let fields: Vec<_> = line.splitn(8, '\t').collect();
        if fields.len() != 8 {
            return Err(format!("incomplete pairs row at line {}", line_no + 1).into());
        }
        let id: usize = fields[0].parse()?;
        if id != offset + count {
            return Err(format!("pairs id {id}, expected {}", offset + count).into());
        }
        let response = output_dir.join("responses").join(format!("{id:08}.txt"));
        if !response.is_file() {
            return Err(format!("pairs row {id} has no decoded response {response:?}").into());
        }
        let ids: Vec<u32> = fields[7]
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let total: usize = fields[6].parse()?;
        if ids.len() != total {
            return Err(format!("pairs row {id}: {} ids != total {total}", ids.len()).into());
        }
        count += 1;
    }
    Ok(count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let prompts = load_prompts(&args.prompt_pack)?;
    let metadata = load_metadata(&args.prompt_meta)?;
    if prompts.len() != metadata.len() {
        return Err(format!(
            "{} prompts != {} metadata rows",
            prompts.len(),
            metadata.len()
        )
        .into());
    }
    let end = (args.offset + args.limit).min(prompts.len());
    if args.offset >= end {
        return Err(format!("empty requested range {}..{end}", args.offset).into());
    }

    let output_dir = std::path::Path::new(&args.output_dir);
    let response_dir = output_dir.join("responses");
    std::fs::create_dir_all(&response_dir)?;
    let pairs_path = output_dir.join("pairs.tsv");
    if !pairs_path.exists() {
        let mut pairs = std::fs::File::create(&pairs_path)?;
        writeln!(pairs, "# {FORMAT}")?;
        pairs.sync_data()?;
    }
    let done = completed_rows(&pairs_path, output_dir, args.offset)?;
    if done > end - args.offset {
        return Err(format!(
            "chunk has {done} rows but range contains only {}",
            end - args.offset
        )
        .into());
    }

    let engine = Engine::new(0)?;
    let (model, tokenizer) = load_model(&engine, &args.model)?;
    let params = GenParams {
        max_new: args.max_new,
        max_ctx: None,
        eos: vec![tokenizer.eos_id()],
    };
    eprintln!(
        "[dspark-generate] range={}..{} resume={} temp={} max_new={} model={}",
        args.offset, end, done, args.temperature, args.max_new, args.model
    );

    for id in args.offset + done..end {
        let meta = &metadata[id];
        debug_assert_eq!(meta.id, id);
        let think_mode = if meta.mode == "think" {
            ThinkMode::Think
        } else {
            ThinkMode::NoThink
        };
        let turns = [Turn {
            role: "user".to_string(),
            content: prompts[id].clone(),
            tool_calls: Vec::new(),
            ..Default::default()
        }];
        let rendered = tokenizer
            .apply_chat_template_tools(&turns, true, &[], think_mode, None)
            .map_err(|error| format!("chat render id {id}: {error}"))?;
        let prompt_ids = tokenizer.encode(&rendered, true);
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: args.temperature,
            seed: args.seed + id as u64,
            ..Default::default()
        });
        let started = std::time::Instant::now();
        let output = model.generate_with(&engine, &prompt_ids, &params, &mut sampler, |_| true)?;
        let mut full_ids = prompt_ids.clone();
        full_ids.extend_from_slice(&output.tokens);

        let response = tokenizer.decode_special(&output.tokens, false);
        let response_path = response_dir.join(format!("{id:08}.txt"));
        let response_tmp = response_dir.join(format!(".{id:08}.tmp"));
        std::fs::write(&response_tmp, response.as_bytes())?;
        std::fs::rename(&response_tmp, &response_path)?;

        let mut pairs = std::fs::OpenOptions::new().append(true).open(&pairs_path)?;
        writeln!(
            pairs,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            id,
            meta.split,
            meta.mode,
            meta.category,
            prompt_ids.len(),
            output.tokens.len(),
            full_ids.len(),
            full_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        )?;
        pairs.sync_data()?;
        eprintln!(
            "[dspark-generate] id={} {}/{} prompt={} response={} elapsed={:.3}s",
            id,
            id - args.offset + 1,
            end - args.offset,
            prompt_ids.len(),
            output.tokens.len(),
            started.elapsed().as_secs_f64(),
        );
    }

    let meta_path = output_dir.join("generation.meta.json");
    std::fs::write(
        &meta_path,
        format!(
            "{{\"format\":\"{FORMAT}\",\"offset\":{},\"end\":{},\"count\":{},\"max_new\":{},\"temperature\":{},\"seed\":{}}}\n",
            args.offset,
            end,
            end - args.offset,
            args.max_new,
            args.temperature,
            args.seed,
        ),
    )?;
    println!("DSPARK-GENERATION-DONE {} {}", args.offset, end);
    Ok(())
}
