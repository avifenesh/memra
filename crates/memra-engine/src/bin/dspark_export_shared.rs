//! Export the frozen DSpark embedding and LM-head rows from the exact deployed GGUF.
//!
//! The output row order is the draft-to-target (`d2t`) order in `ranks-32768.txt`.
//! Rows are independently dequantized so this never materializes either 248K-row tensor.
//!
//! Usage:
//!   dspark-export-shared <target.gguf> <ranks-32768.txt> <output-dir>

use memra_gguf::{GgufFile, TensorInfo};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const DRAFT_VOCAB: usize = 32_768;

fn parse_ranks(text: &str, expected: usize, target_vocab: usize) -> Result<Vec<u32>, String> {
    let mut ranks = Vec::with_capacity(expected);
    let mut seen = HashSet::with_capacity(expected);
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let id = line
            .parse::<u32>()
            .map_err(|e| format!("rank line {} is not a u32: {e}", line_no + 1))?;
        if id as usize >= target_vocab {
            return Err(format!(
                "rank line {} id {} is outside target vocab {}",
                line_no + 1,
                id,
                target_vocab
            ));
        }
        if !seen.insert(id) {
            return Err(format!(
                "rank line {} repeats target id {}",
                line_no + 1,
                id
            ));
        }
        ranks.push(id);
    }
    if ranks.len() != expected {
        return Err(format!(
            "expected {expected} rank ids, found {}",
            ranks.len()
        ));
    }
    Ok(ranks)
}

fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn tensor_shape(t: &TensorInfo) -> Result<(usize, usize), String> {
    if t.ne.len() != 2 {
        return Err(format!("{} must be rank 2, got {:?}", t.name, t.ne));
    }
    let cols = usize::try_from(t.ne[0]).map_err(|_| format!("{} cols overflow", t.name))?;
    let rows = usize::try_from(t.ne[1]).map_err(|_| format!("{} rows overflow", t.name))?;
    Ok((rows, cols))
}

fn export_rows(
    gguf: &GgufFile,
    tensor: &TensorInfo,
    ranks: &[u32],
    expected_cols: usize,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (rows, cols) = tensor_shape(tensor)?;
    if cols != expected_cols {
        return Err(format!(
            "{} has {} columns, expected {}",
            tensor.name, cols, expected_cols
        )
        .into());
    }
    let row_bytes = usize::try_from(tensor.n_bytes)? / rows;
    if row_bytes * rows != usize::try_from(tensor.n_bytes)? {
        return Err(format!("{} byte size is not row-aligned", tensor.name).into());
    }
    let raw = gguf.tensor_data(tensor);
    let partial = PathBuf::from(format!("{}.partial", output.display()));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
    let mut encoded = vec![0u8; cols * 2];
    for &target_id in ranks {
        let row = target_id as usize;
        if row >= rows {
            return Err(format!(
                "target id {target_id} is outside {} rows for {}",
                rows, tensor.name
            )
            .into());
        }
        let start = row * row_bytes;
        let values =
            memra_gguf::dequant::dequantize(tensor.ggml_type, &raw[start..start + row_bytes], cols);
        for (dst, value) in encoded.chunks_exact_mut(2).zip(values) {
            dst.copy_from_slice(&bf16_rne(value).to_le_bytes());
        }
        writer.write_all(&encoded)?;
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(partial, output)?;
    Ok(())
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dspark-export-shared <target.gguf> <ranks-32768.txt> <output-dir>");
        std::process::exit(2);
    }
    let model_path = Path::new(&args[1]);
    let ranks_path = Path::new(&args[2]);
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir)?;
    let embedding_path = out_dir.join("embedding.bf16");
    let head_path = out_dir.join("lm_head.bf16");
    let d2t_path = out_dir.join("d2t.u32");
    let manifest_path = out_dir.join("manifest.json");
    for path in [&embedding_path, &head_path, &d2t_path, &manifest_path] {
        if path.exists() {
            return Err(format!("refusing to overwrite {}", path.display()).into());
        }
    }

    let gguf = GgufFile::open(model_path)?;
    let embedding = gguf
        .find("token_embd.weight")
        .ok_or("GGUF lacks token_embd.weight")?;
    let lm_head = gguf
        .find("output.weight")
        .ok_or("GGUF lacks output.weight")?;
    let (target_vocab, d_model) = tensor_shape(embedding)?;
    let (head_vocab, head_cols) = tensor_shape(lm_head)?;
    if (head_vocab, head_cols) != (target_vocab, d_model) {
        return Err(format!(
            "embedding/head shape mismatch: embedding=[{target_vocab},{d_model}] head=[{head_vocab},{head_cols}]"
        )
        .into());
    }
    let rank_text = std::fs::read_to_string(ranks_path)?;
    let ranks = parse_ranks(&rank_text, DRAFT_VOCAB, target_vocab)?;

    eprintln!(
        "exporting {} rows x {}: embedding={:?}, lm_head={:?}",
        ranks.len(),
        d_model,
        embedding.ggml_type,
        lm_head.ggml_type
    );
    export_rows(&gguf, embedding, &ranks, d_model, &embedding_path)?;
    export_rows(&gguf, lm_head, &ranks, d_model, &head_path)?;

    let mut d2t = BufWriter::new(File::create(&d2t_path)?);
    for id in &ranks {
        d2t.write_all(&id.to_le_bytes())?;
    }
    d2t.flush()?;

    let manifest = format!(
        concat!(
            "{{\n",
            "  \"format\": \"memra-dspark-shared-v1\",\n",
            "  \"source_gguf\": \"{}\",\n",
            "  \"ranks_source\": \"{}\",\n",
            "  \"draft_vocab\": {},\n",
            "  \"target_vocab\": {},\n",
            "  \"d_model\": {},\n",
            "  \"embedding_tensor\": \"token_embd.weight\",\n",
            "  \"embedding_qtype\": \"{:?}\",\n",
            "  \"lm_head_tensor\": \"output.weight\",\n",
            "  \"lm_head_qtype\": \"{:?}\",\n",
            "  \"dtype\": \"bfloat16-le\",\n",
            "  \"row_order\": \"d2t.u32\"\n",
            "}}\n"
        ),
        json_escape(&model_path.display().to_string()),
        json_escape(&ranks_path.display().to_string()),
        ranks.len(),
        target_vocab,
        d_model,
        embedding.ggml_type,
        lm_head.ggml_type,
    );
    std::fs::write(&manifest_path, manifest)?;
    eprintln!("wrote {}", out_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_parser_rejects_duplicates_and_out_of_range() {
        assert_eq!(parse_ranks("2\n0\n1\n", 3, 3).unwrap(), vec![2, 0, 1]);
        assert!(parse_ranks("2\n2\n", 2, 3).unwrap_err().contains("repeats"));
        assert!(parse_ranks("3\n", 1, 3).unwrap_err().contains("outside"));
        assert!(parse_ranks("0\n", 2, 3).unwrap_err().contains("expected"));
    }

    #[test]
    fn bf16_conversion_rounds_to_nearest_even() {
        for value in [0.0, -0.0, 1.0, -2.5, f32::INFINITY, f32::NEG_INFINITY] {
            let roundtrip = f32::from_bits((bf16_rne(value) as u32) << 16);
            assert_eq!(roundtrip, value);
        }
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_7fff)), 0x3f80);
        assert_eq!(bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
    }
}
