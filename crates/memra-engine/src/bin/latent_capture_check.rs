//! GPU diagnostic only: caller owns hardware and serialization. Never a performance gate.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: latent_capture_check <absolute-capture-dir> <absolute-new-result-dir> [device=0]".into());
    }
    let device = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0);
    memra_engine::latent_capture::check(
        std::path::Path::new(&args[0]),
        std::path::Path::new(&args[1]),
        device,
    )
}
