#![allow(dead_code)]

mod prefix_policy;
#[path = "../router.rs"]
mod router;

use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected budget and consumed-token count".into());
    }
    let budget: usize = args[1].parse()?;
    let used: usize = args[2].parse()?;
    if used > budget || budget > prefix_policy::MAX_PREFIX_TOKENS {
        return Err("invalid prefix replay bounds".into());
    }
    let mut data = Vec::new();
    std::io::stdin().take(16385).read_to_end(&mut data)?;
    let tokens = vec![0; used];
    let selected = prefix_policy::select(&tokens, 0, budget, |_| std::mem::take(&mut data))?;
    println!(
        "{}\t{}\t{}\t{}\t{}",
        selected.decision.kind.name(),
        selected.decision.k,
        selected.token_count,
        selected.decoded_bytes,
        selected.inspected_bytes
    );
    Ok(())
}
