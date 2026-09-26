//! Compare Memra's MiMo text chat rendering and BPE IDs with source Jinja/HF goldens.
//! The source files are supplied explicitly; a missing artifact never becomes a pass.

use std::path::Path;

use memra_tokenizer::Tokenizer;
use memra_tokenizer::chat::{ThinkMode, Turn};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const TOKENIZER_SHA256: &str = "ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e";
const TOKENIZER_CONFIG_SHA256: &str =
    "413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06";
const TEMPLATE_SHA256: &str = "11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059";

fn check_sha(path: &Path, expected: &str) -> Result<Vec<u8>, Fail> {
    let bytes = std::fs::read(path)?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!(
            "{}: SHA-256 differs from pinned MiMo source",
            path.display()
        )
        .into());
    }
    Ok(bytes)
}

fn from_hex(text: &str) -> Result<String, Fail> {
    if !text.len().is_multiple_of(2) {
        return Err("MiMo template golden has an odd hex length".into());
    }
    let bytes: Vec<u8> = text
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair)?;
            Ok(u8::from_str_radix(text, 16)?)
        })
        .collect::<Result<_, Fail>>()?;
    Ok(String::from_utf8(bytes)?)
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: tok-mimo-template-gate <pinned_hf_dir> <goldens.tsv>".into());
    }
    let dir = Path::new(&args[0]);
    check_sha(&dir.join("tokenizer.json"), TOKENIZER_SHA256)?;
    let config_bytes = check_sha(&dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_SHA256)?;
    let config_text = std::str::from_utf8(&config_bytes)?;
    let config = memra_tokenizer::json::parse(config_text)?;
    let template = config
        .get("chat_template")
        .and_then(|value| value.as_str())
        .ok_or("pinned MiMo source has no string chat template")?;
    if format!("{:x}", Sha256::digest(template.as_bytes())) != TEMPLATE_SHA256 {
        return Err("pinned MiMo chat template SHA-256 differs".into());
    }
    let tokenizer = Tokenizer::from_hf_dir(dir)?;
    if tokenizer.pre() != "qwen2" {
        return Err("MiMo source pre-tokenizer did not resolve to qwen2".into());
    }
    let goldens = std::fs::read_to_string(&args[1])?;
    let mut passed = 0usize;
    for line in goldens.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err("MiMo template golden row must have three fields".into());
        }
        let name = fields[0];
        let expected_text = from_hex(fields[1])?;
        let expected_ids: Vec<u32> = fields[2]
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let (turns, add_generation_prompt, no_think): (Vec<(&str, &str)>, bool, bool) = match name {
            "user_gen" => (vec![("user", "Hi")], true, false),
            "space_gen" => (vec![("user", "  pad  ")], true, false),
            "system_user_gen" => (
                vec![("system", "You are concise."), ("user", "What is 2+2?")],
                true,
                false,
            ),
            "assistant_history_gen" => (
                vec![
                    ("user", "Hello"),
                    ("assistant", "Hello!"),
                    ("user", "Again"),
                ],
                true,
                false,
            ),
            "assistant_reasoning_gen" => (
                vec![
                    ("user", "Hello"),
                    ("assistant", "Answer"),
                    ("user", "Again"),
                ],
                true,
                false,
            ),
            "unicode_gen" => (vec![("user", "שָׁלוֹם café 你好")], true, false),
            "user_no_gen" => (vec![("user", "Hi")], false, false),
            "no_think_gen" => (vec![("user", "Hi")], true, true),
            _ => return Err(format!("unknown MiMo template golden case {name}").into()),
        };
        let rendered = if no_think || name == "assistant_reasoning_gen" {
            let mut structured: Vec<Turn> = turns
                .iter()
                .map(|(role, content)| Turn {
                    role: (*role).into(),
                    content: (*content).into(),
                    ..Default::default()
                })
                .collect();
            if name == "assistant_reasoning_gen" {
                let assistant = structured
                    .iter_mut()
                    .find(|turn| turn.role == "assistant")
                    .ok_or("assistant reasoning golden has no assistant turn")?;
                assistant.reasoning = Some("Let me check.".into());
            }
            tokenizer.apply_chat_template_tools(
                &structured,
                add_generation_prompt,
                &[],
                if no_think {
                    ThinkMode::NoThink
                } else {
                    ThinkMode::Default
                },
                None,
            )?
        } else {
            tokenizer.apply_chat_template(&turns, add_generation_prompt)
        };
        if rendered != expected_text {
            return Err(
                format!("{name}: Memra render bytes differ from pinned Jinja golden").into(),
            );
        }
        let actual_ids = tokenizer.encode(&rendered, true);
        if actual_ids != expected_ids {
            return Err(format!("{name}: Memra BPE IDs differ from pinned HF golden").into());
        }
        println!("{name}\tpassed\t{}", actual_ids.len());
        passed += 1;
    }
    if passed != 8 {
        return Err(format!("MiMo template gate expected 8 cases, found {passed}").into());
    }
    println!("MiMo template and tokenizer gate passed: {passed} pinned cases");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tok-mimo-template-gate: {error}");
        std::process::exit(1);
    }
}
