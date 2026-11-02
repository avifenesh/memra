//! Thin binary over the `memra-server` library. Everything lives in `lib.rs` so a
//! deployment-owned binary can link the same server and supply its own wiring; this
//! stock bin IS the reference wiring.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    memra_server::serve_main()
}
