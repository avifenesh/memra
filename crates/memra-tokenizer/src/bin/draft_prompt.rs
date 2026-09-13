//! Research helper: use the artifact's native chat template and tokenizer.
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: draft_prompt <model.gguf> <prompt.txt>".into());
    }
    let file = GgufFile::open(&args[1])?;
    let tokenizer = Tokenizer::from_gguf(&file)?;
    let prompt = std::fs::read_to_string(&args[2])?;
    let rendered = tokenizer.apply_chat_template(&[("user", prompt.as_str())], true);
    let ids = tokenizer.encode(&rendered, true);
    if ids.is_empty() {
        return Err("chat prompt encoded to zero tokens".into());
    }
    println!("{ids:?}");
    Ok(())
}
