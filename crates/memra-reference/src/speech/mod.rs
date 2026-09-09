//! Memra-owned speech operators. Encoder execution is separate from text decoding;
//! completing an encoder does not admit an ASR generation or serving surface.

pub mod frontend;

pub mod encoder;
mod matrix;

pub mod decode;
pub mod decoder;
pub mod fastconformer;
