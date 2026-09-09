//! Pinned-source GPU rotary/query-scaling gate, including discontinuity positions.
use memra_engine::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args()
        .nth(1)
        .ok_or("usage: ministral_rope_gate <oracle.tsv>")?;
    let mut expected_q = vec![0.0; 1024];
    let mut expected_k = vec![0.0; 512];
    let mut count = 0;
    for line in std::fs::read_to_string(file)?.lines() {
        let c: Vec<_> = line.split('\t').collect();
        let i: usize = c[1].parse()?;
        let v = f32::from_bits(u32::from_str_radix(c[2], 16)?);
        match c[0] {
            "q" => expected_q[i] = v,
            "k" => expected_k[i] = v,
            _ => return Err("unknown oracle row".into()),
        };
        count += 1;
    }
    assert_eq!(count, 1536);
    let e = Engine::new(0)?;
    let q: Vec<f32> = (0..1024).map(|i| ((i % 37) as f32 - 18.0) / 19.0).collect();
    let k: Vec<f32> = (0..512).map(|i| ((i % 29) as f32 - 14.0) / 17.0).collect();
    let mut q = e.htod(&q)?;
    let mut k = e.htod(&k)?;
    let pos = e.htod_i32(&[0, 16383, 16384, 32768])?;
    let ff = e.htod(&memra_gguf::model_plan::yarn_frequency_divisors(
        128, 1e6, 16.0, 16384, 32.0, 1.0,
    ))?;
    e.rope_neox2(&mut q, &mut k, &pos, 128, 128, 2, 1, 4, 1e6, 1.0, Some(&ff))?;
    e.position_query_scale(&mut q, &pos, 256, 4, 16384, 0.1)?;
    let mut failed = false;
    for (name, actual, expected) in [
        ("q", e.dtoh(&q)?, expected_q),
        ("k", e.dtoh(&k)?, expected_k),
    ] {
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{name} positions=0,16383,16384,32768 max_abs={max_abs}");
        failed |= max_abs > 0.005;
    }
    if failed {
        return Err("pinned-source rotary gate failed".into());
    }
    println!("MINISTRAL-ROPE-GATE-PASS");
    Ok(())
}
