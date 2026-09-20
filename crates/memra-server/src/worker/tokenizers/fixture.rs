use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Fixture(pub PathBuf);

impl Fixture {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "memra-tokenizer-snapshot-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("tokenizer.json"), r#"{
          "version":"1.0",
          "added_tokens":[{"id":4,"content":"<|end|>","special":true}],
          "pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated"},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false}
          ]},
          "model":{"type":"BPE","vocab":{"h":0,"e":1,"l":2,"o":3},"merges":[]}
        }"#).unwrap();
        let fixture = Self(path);
        fixture.replace_config(true);
        fixture
    }

    pub fn replace_config(&self, original: bool) {
        let (bos, template) = if original {
            ("true", "<|im_start|>")
        } else {
            ("false", "<think> add_generation_prompt")
        };
        let text = format!(
            r#"{{"eos_token":"<|end|>","bos_token":"<|end|>","add_bos_token":{bos},"chat_template":"{template}"}}"#
        );
        let pending = self.0.join("tokenizer_config.json.pending");
        std::fs::write(&pending, text).unwrap();
        std::fs::rename(pending, self.0.join("tokenizer_config.json")).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
