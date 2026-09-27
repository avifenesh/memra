use std::{env, fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = root.join("../../../../crates/memra-engine/src/hybrid.rs");
    println!("cargo:rerun-if-changed={}", source.display());
    let text = fs::read_to_string(source).unwrap();
    let start = text.find("    /// Opt-in paired source receipt;").unwrap();
    let end = text[start..]
        .find("    /// Typed composite intake;")
        .unwrap()
        + start;
    let methods = &text[start..end];
    assert_eq!(methods.matches("Self::load_prepared_draft").count(), 2);
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("paired-entry.rs"),
        format!("impl MtpHead {{\n{methods}}}\n"),
    )
    .unwrap();
}
