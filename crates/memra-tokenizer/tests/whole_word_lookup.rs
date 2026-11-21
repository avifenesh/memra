//! The whole-word lookup (`tokenizer.json` `model.ignore_merges`) on both load paths.
//!
//! A checkpoint that declares it encodes a pre-split word that is itself a vocabulary piece as
//! that one id, whatever its merge table builds. The `tokenizer.json` path reads the flag. A GGUF
//! has no key for it, so that path takes the family's value from the `pre` id, and the two paths
//! must give the same ids: the checkpoint's.
//!
//! 1. a six-piece vocabulary in which the lookup and the merge table disagree, through both
//!    paths and both values;
//! 2. GLM-5.3-Flash//! 3. GLM-5.3-Flash, where the flag decides 893 pieces: its pinned `tokenizer.json` is not in
//!    this repository, so this one runs when `MEMRA_GLM53_TOKENIZER_DIR` names a directory
//!    holding it (research/glm53-flash-bringup-20260827/parity-evidence/ignore-merges-20261121/
//!    has the corpus, the reference ids and the receipts of both paths).

use std::path::{Path, PathBuf};

use memra_gguf::GgufFile;
use memra_tokenizer::{PreSplit, Tokenizer, json};

fn research() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../research")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("memra-tok-wwl-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn string_bytes(s: &str) -> Vec<u8> {
    let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(s.as_bytes());
    bytes
}

fn string_array(items: &[String]) -> Vec<u8> {
    let mut bytes = 8u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        bytes.extend(string_bytes(item));
    }
    bytes
}

/// A GGUF holding only tokenizer metadata, as a converter writes it from `tokenizer.json`: the
/// pieces by id, their types (3 control, 4 user-defined, 1 normal), the merges as `"a b"`, the
/// `pre` id and the end-of-generation id. No key carries `ignore_merges`: GGUF has none.
fn gguf_from_tokenizer_json(tokenizer_json: &str, pre: &str, eos: u32) -> Vec<u8> {
    let tj = json::parse(tokenizer_json).unwrap();
    let model = tj.get("model").unwrap();
    let vocab = model.get("vocab").unwrap().as_obj().unwrap();
    let added = tj.get("added_tokens").unwrap().as_arr().unwrap();
    let n = vocab
        .values()
        .map(|id| id.as_u64().unwrap())
        .chain(added.iter().map(|t| t.get("id").unwrap().as_u64().unwrap()))
        .max()
        .unwrap() as usize
        + 1;
    let mut pieces = vec![String::new(); n];
    let mut types = vec![1i32; n];
    for (piece, id) in vocab {
        pieces[id.as_u64().unwrap() as usize] = piece.clone();
    }
    for token in added {
        let id = token.get("id").unwrap().as_u64().unwrap() as usize;
        pieces[id] = token.get("content").unwrap().as_str().unwrap().to_owned();
        types[id] = if token.get("special").unwrap().as_bool().unwrap() {
            3
        } else {
            4
        };
    }
    let merges: Vec<String> = model
        .get("merges")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|pair| match pair.as_arr() {
            Some(pair) => format!(
                "{} {}",
                pair[0].as_str().unwrap(),
                pair[1].as_str().unwrap()
            ),
            None => pair.as_str().unwrap().to_owned(),
        })
        .collect();
    let mut type_bytes = 5u32.to_le_bytes().to_vec();
    type_bytes.extend_from_slice(&(n as u64).to_le_bytes());
    for kind in types {
        type_bytes.extend_from_slice(&kind.to_le_bytes());
    }
    let metadata = vec![
        ("tokenizer.ggml.model", 8u32, string_bytes("gpt2")),
        ("tokenizer.ggml.pre", 8, string_bytes(pre)),
        ("tokenizer.ggml.tokens", 9, string_array(&pieces)),
        ("tokenizer.ggml.token_type", 9, type_bytes),
        ("tokenizer.ggml.merges", 9, string_array(&merges)),
        ("tokenizer.ggml.eos_token_id", 4, eos.to_le_bytes().to_vec()),
        ("tokenizer.ggml.add_bos_token", 7, vec![0]),
    ];
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, kind, value) in metadata {
        bytes.extend(string_bytes(key));
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend(value);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes
}

fn load_gguf(name: &str, bytes: &[u8]) -> Result<Tokenizer, String> {
    let dir = scratch(name);
    let path = dir.join("tokenizer.gguf");
    std::fs::write(&path, bytes).unwrap();
    let loaded = GgufFile::open(&path)
        .map_err(|e| e.to_string())
        .and_then(|file| Tokenizer::from_gguf(&file));
    let _ = std::fs::remove_dir_all(&dir);
    loaded
}

fn unhex(s: &str) -> String {
    let bytes: Vec<u8> = (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    String::from_utf8(bytes).unwrap()
}

/// The names of the corpus cases whose ids, in either encode mode, are not the reference's.
fn corpus_failures(tok: &Tokenizer, lane: &Path) -> (usize, Vec<String>) {
    let ids = |s: &str| -> Vec<u32> {
        if s.is_empty() {
            Vec::new()
        } else {
            s.split(',').map(|x| x.parse().unwrap()).collect()
        }
    };
    let corpus = read(&lane.join("corpus.tsv"));
    let reference = read(&lane.join("ref-ids.tsv"));
    assert_eq!(corpus.lines().count(), reference.lines().count());
    let mut failed = Vec::new();
    for (line, ref_line) in corpus.lines().zip(reference.lines()) {
        let (name, hexed) = line.split_once('\t').unwrap();
        let cols: Vec<&str> = ref_line.split('\t').collect();
        assert_eq!(cols[0], name);
        let text = unhex(hexed);
        if tok.encode(&text, true) != ids(cols[1]) || tok.encode(&text, false) != ids(cols[2]) {
            failed.push(name.to_string());
        }
    }
    (corpus.lines().count(), failed)
}

const QWEN2_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Six pieces: `abc` is in the vocabulary, and the one merge, `a b`, cannot build it.
fn tiny(regex: &str, ignore_merges: bool) -> String {
    let regex = regex.replace('\\', "\\\\");
    format!(
        r#"{{"version":"1.0","normalizer":null,
"added_tokens":[{{"id":5,"content":"<eos>","special":true}}],
"pre_tokenizer":{{"type":"Sequence","pretokenizers":[
 {{"type":"Split","pattern":{{"Regex":"{regex}"}},"behavior":"Isolated","invert":false}},
 {{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}}]}},
"model":{{"type":"BPE","ignore_merges":{ignore_merges},
 "vocab":{{"a":0,"b":1,"c":2,"ab":3,"abc":4}},"merges":["a b"]}}}}"#
    )
}

#[test]
fn both_load_paths_take_the_whole_word_lookup_from_the_family() {
    let glm4_regex = QWEN2_REGEX.replace(r"|\p{N}|", r"|\p{N}{1,3}|");
    let whole = vec![4u32];
    let merged = vec![3u32, 2];

    // tokenizer.json: the file's own flag, whatever the split
    let hf = |name: &str, regex: &str, flag: bool| {
        let dir = scratch(name);
        std::fs::write(dir.join("tokenizer.json"), tiny(regex, flag)).unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 5}"#).unwrap();
        let tok = Tokenizer::from_hf_dir(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        tok
    };
    assert_eq!(
        hf("hf-glm4-on", &glm4_regex, true).encode("abc", false),
        whole
    );
    assert_eq!(
        hf("hf-glm4-off", &glm4_regex, false).encode("abc", false),
        merged
    );
    assert_eq!(
        hf("hf-qwen2-on", QWEN2_REGEX, true).encode("abc", false),
        whole
    );
    assert_eq!(
        hf("hf-qwen2-off", QWEN2_REGEX, false).encode("abc", false),
        merged
    );

    // GGUF: no key for it, so the family's declared value, by pre id. The same bytes, only
    // the pre id differs.
    let gguf = |pre: &str| {
        load_gguf(
            pre,
            &gguf_from_tokenizer_json(&tiny(QWEN2_REGEX, false), pre, 5),
        )
    };
    let glm4 = gguf("glm4").unwrap();
    assert_eq!(glm4.split(), PreSplit::Glm4);
    assert_eq!(glm4.encode("abc", false), whole, "glm4 declares the lookup");
    for pre in ["qwen2", "qwen35", "deepseek-v3"] {
        assert_eq!(
            gguf(pre).unwrap().encode("abc", false),
            merged,
            "{pre} does not"
        );
    }
    // every other piece still merges as before
    assert_eq!(
        glm4.encode("ab abc ba", false),
        glm4.encode("ab", false)
            .into_iter()
            .chain(glm4.encode(" abc ba", false))
            .collect::<Vec<_>>()
    );
    assert_eq!(glm4.encode("ab", false), vec![3]);
}

/// GLM-5.3-Flash (zai-org/GLM-5.3-Flash @ 04c4e9e9, `tokenizer.json` sha256 19e77364...): the
/// 1,786 texts built on the 893 pieces its `ignore_merges` decides, through the checkpoint
/// directory and through a GGUF built from it (`pre` `glm4`), against tokenizers 0.23.3.
/// The pinned file is 20 MB and not in this repository: the test runs where
/// `MEMRA_GLM53_TOKENIZER_DIR` names a directory holding it, and says so when it does not.
#[test]
fn glm53_flash_gives_the_checkpoint_ids_on_both_load_paths() {
    let Some(dir) = std::env::var_os("MEMRA_GLM53_TOKENIZER_DIR").map(PathBuf::from) else {
        eprintln!(
            "SKIP[MEMRA_GLM53_TOKENIZER_DIR unset]: GLM-5.3-Flash whole-word-lookup parity not run"
        );
        return;
    };
    use sha2::{Digest, Sha256};
    let tokenizer_json = read(&dir.join("tokenizer.json"));
    let digest: String = Sha256::digest(tokenizer_json.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest, "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "not the pinned revision's tokenizer.json"
    );
    let lane =
        research().join("glm53-flash-bringup-20260827/parity-evidence/ignore-merges-20261121");
    let older = research().join("glm53-flash-bringup-20260827/parity-evidence");

    let hf = Tokenizer::from_hf_dir(&dir).unwrap();
    assert_eq!(hf.split(), PreSplit::Glm4);
    let gguf = load_gguf(
        "glm53",
        &gguf_from_tokenizer_json(&tokenizer_json, "glm4", hf.eos_id()),
    )
    .unwrap();
    assert_eq!(gguf.split(), PreSplit::Glm4);
    for (path, tok) in [("tokenizer.json", &hf), ("gguf", &gguf)] {
        let (cases, failed) = corpus_failures(tok, &lane);
        assert_eq!(cases, 1786);
        println!(
            "glm53 {path}: ignore-merges corpus {cases} cases, {} differ",
            failed.len()
        );
        assert!(
            failed.is_empty(),
            "{path}: {} of {cases} differ: {:?}",
            failed.len(),
            &failed[..failed.len().min(8)]
        );
        // and the lane's first corpus, which this flag does not reach, still holds
        let (cases, failed) = corpus_failures(tok, &older);
        println!(
            "glm53 {path}: pre-tokenizer corpus {cases} cases, {} differ",
            failed.len()
        );
        assert!(
            failed.is_empty(),
            "{path}: {} of {cases} differ: {failed:?}",
            failed.len()
        );
    }
}
