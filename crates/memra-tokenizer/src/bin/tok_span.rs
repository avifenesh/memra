//! tok-span: map byte offsets in an exact decoded token stream back to token ids.
//!
//! Usage: tok-span <model.gguf> <ids.txt> <decoded.txt> [byte-offset ...]

use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: tok-span <model.gguf> <ids.txt> <decoded.txt> [byte-offset ...]")?;
    let ids_path = args
        .next()
        .ok_or("usage: tok-span <model.gguf> <ids.txt> <decoded.txt> [byte-offset ...]")?;
    let decoded_path = args
        .next()
        .ok_or("usage: tok-span <model.gguf> <ids.txt> <decoded.txt> [byte-offset ...]")?;
    let offsets: Vec<usize> = args
        .map(|value| value.parse::<usize>())
        .collect::<Result<_, _>>()?;

    let g = GgufFile::open(model)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let ids_raw = std::fs::read_to_string(ids_path)?;
    let ids: Vec<u32> = ids_raw
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let want = std::fs::read(decoded_path)?;
    let got = tok.decode_bytes_special(&ids, true);
    if got != want {
        let first = got
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(want.len()));
        return Err(format!(
            "decoded token bytes differ from response text at byte {first} (tokens={} decoded={} response={})",
            ids.len(),
            got.len(),
            want.len()
        )
        .into());
    }

    println!("summary\t{}\t{}\tMATCH", ids.len(), got.len());
    for offset in offsets {
        if offset >= got.len() {
            return Err(format!(
                "byte offset {offset} is outside decoded length {}",
                got.len()
            )
            .into());
        }
        let mut start = 0usize;
        let mut found = None;
        for (index, &id) in ids.iter().enumerate() {
            let piece = tok.decode_bytes_special(&[id], true);
            let end = start + piece.len();
            if start <= offset && offset < end {
                found = Some((index, id, start, end, piece));
                break;
            }
            start = end;
        }
        let (index, id, start, end, piece) =
            found.ok_or_else(|| format!("no token span contains byte offset {offset}"))?;
        let hex: String = piece.iter().map(|byte| format!("{byte:02x}")).collect();
        println!(
            "offset\t{offset}\t{index}\t{id}\t{start}\t{end}\t{hex}\t{:?}",
            String::from_utf8_lossy(&piece)
        );
    }
    Ok(())
}
