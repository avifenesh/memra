use std::{env, fs, path::PathBuf};
fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let a = source.find(start).unwrap();
    let b = source[a..].find(end).unwrap() + a;
    &source[a..b]
}
fn main() {
    println!("cargo:rustc-check-cfg=cfg(memra_cutlass)");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = root.join("../../../../crates/memra-engine/src/model.rs");
    println!("cargo:rerun-if-changed={}", source.display());
    let text = fs::read_to_string(source).unwrap();
    let helper = section(
        &text,
        "pub(crate) fn quant_type_for_trim(",
        "/// Host-side split-plane repack",
    );
    let repack = section(&text, "pub fn repack_nvfp4_split(", "/// Inverse of");
    let rp = section(
        &text,
        "pub fn rp_enabled()",
        "/// FULL-PRECISION LOADER MODE",
    );
    let methods = section(
        &text,
        "    pub fn from_quant_bytes(",
        "    pub fn load_opt(",
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("quant-upload.rs"),
        format!("{helper}\n{repack}\n{rp}\nimpl GpuTensor {{\n{methods}}}\n"),
    )
    .unwrap();
}
