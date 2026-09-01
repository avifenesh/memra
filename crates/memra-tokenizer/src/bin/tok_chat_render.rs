//! tok-chat-render: render one user turn through the model's GGUF chat template.
//!
//! Usage: tok-chat-render <model.gguf> <user.txt> <reasoning-effort> <assistant-prefix.txt> <out.txt>

use memra_gguf::GgufFile;
use memra_tokenizer::{
    Tokenizer,
    chat::{ThinkMode, Turn},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: tok-chat-render <model.gguf> <user.txt> <reasoning-effort> \
                 <assistant-prefix.txt> <out.txt>";
    let model = args.next().ok_or(usage)?;
    let user_path = args.next().ok_or(usage)?;
    let effort = args.next().ok_or(usage)?;
    let prefix_path = args.next().ok_or(usage)?;
    let out_path = args.next().ok_or(usage)?;
    if args.next().is_some() {
        return Err(usage.into());
    }
    if !matches!(effort.as_str(), "low" | "medium" | "high") {
        return Err(format!("reasoning effort must be low|medium|high, got {effort:?}").into());
    }

    let gguf = GgufFile::open(&model)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let user = std::fs::read_to_string(&user_path)?;
    let mut prompt = tokenizer.apply_chat_template_tools(
        &[Turn {
            role: "user".to_string(),
            content: user,
            tool_calls: Vec::new(),
            ..Default::default()
        }],
        true,
        &[],
        ThinkMode::Think,
        Some(&effort),
    )?;

    let expected_header = format!("<|im_start|>system\nReasoning: {effort}\n\n<|im_end|>\n");
    if !prompt.starts_with(&expected_header) {
        return Err("artifact did not render the Step35 reasoning-effort header".into());
    }
    if !prompt.ends_with("<|im_start|>assistant\n<think>\n") {
        return Err("artifact did not render the Step35 open reasoning tail".into());
    }
    let assistant_prefix = std::fs::read_to_string(&prefix_path)?;
    if !assistant_prefix.is_ascii() {
        return Err("assistant continuation prefix must be ASCII".into());
    }
    prompt.push_str(&assistant_prefix);

    let prompt_tokens = tokenizer.encode(&prompt, true).len();
    std::fs::write(&out_path, prompt.as_bytes())?;
    println!(
        "rendered={} bytes={} prompt_tokens={} reasoning_effort={} assistant_prefix={}",
        out_path,
        prompt.len(),
        prompt_tokens,
        effort,
        prefix_path
    );
    Ok(())
}
