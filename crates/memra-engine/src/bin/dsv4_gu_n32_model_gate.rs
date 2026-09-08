//! Standalone GU-only full-replay measurement; no cadence/dense helper dependency.
#[path = "../dsv4_gu_n32_model_gate.rs"]
mod gate;
fn main() {
    gate::run();
}
