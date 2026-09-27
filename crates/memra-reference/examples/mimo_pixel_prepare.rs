//! Small binary bridge for the pinned Python-source pixel oracle.

use std::io::{Read, Write};

use memra_reference::mimo_pixel_prepare::prepare_mimo_rgb8_frames;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimensions: Vec<usize> = std::env::args()
        .skip(1)
        .map(|value| value.parse())
        .collect::<Result<_, _>>()?;
    if dimensions.len() != 3 {
        return Err("usage: mimo_pixel_prepare T H W < rgb8.thwc > prepared.tchw".into());
    }
    let mut rgb8 = Vec::new();
    std::io::stdin().read_to_end(&mut rgb8)?;
    let (prepared, shape) =
        prepare_mimo_rgb8_frames(&rgb8, [dimensions[0], dimensions[1], dimensions[2], 3])?;
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for dimension in shape {
        output.write_all(&u32::try_from(dimension)?.to_le_bytes())?;
    }
    for value in prepared {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
