use std::io::{self, Write};
use std::slice::ChunksExact;

use crate::{ColorType, EncodingError};
use bit_writer::BitWriter;
use huffman::{write_huffman_tree, write_single_entry_huffman_tree};

mod bit_writer;
mod huffman;

const fn length_to_symbol(len: u16) -> (u16, u8) {
    let len = len - 1;
    let highest_bit = len.ilog2() as u16;
    let second_highest_bit = (len >> (highest_bit - 1)) & 1;
    let extra_bits = highest_bit - 1;
    let symbol = 2 * highest_bit + second_highest_bit;
    (symbol, extra_bits as u8)
}

#[inline(always)]
fn count_run(
    pixel: &[u8],
    it: &mut std::iter::Peekable<ChunksExact<u8>>,
    frequencies1: &mut [u32; 280],
) {
    let mut run_length = 0;
    while run_length < 4096 && it.peek() == Some(&pixel) {
        run_length += 1;
        it.next();
    }
    if run_length > 0 {
        if run_length <= 4 {
            let symbol = 256 + run_length - 1;
            frequencies1[symbol] += 1;
        } else {
            let (symbol, _extra_bits) = length_to_symbol(run_length as u16);
            frequencies1[256 + symbol as usize] += 1;
        }
    }
}

#[inline(always)]
fn write_run<W: Write>(
    w: &mut BitWriter<W>,
    pixel: &[u8],
    it: &mut std::iter::Peekable<ChunksExact<u8>>,
    codes1: &[u16; 280],
    lengths1: &[u8; 280],
) -> io::Result<()> {
    let mut run_length = 0;
    while run_length < 4096 && it.peek() == Some(&pixel) {
        run_length += 1;
        it.next();
    }
    if run_length > 0 {
        if run_length <= 4 {
            let symbol = 256 + run_length - 1;
            w.write_bits(u64::from(codes1[symbol]), lengths1[symbol])?;
        } else {
            let (symbol, extra_bits) = length_to_symbol(run_length as u16);
            w.write_bits(
                u64::from(codes1[256 + symbol as usize]),
                lengths1[256 + symbol as usize],
            )?;
            w.write_bits(
                (run_length as u64 - 1) & ((1 << extra_bits) - 1),
                extra_bits,
            )?;
        }
    }
    Ok(())
}

/// Allows fine-tuning some encoder parameters.
///
/// Pass to [`WebPEncoder::set_params()`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncoderParams {
    /// Use a predictor transform. Enabled by default.
    pub use_predictor_transform: bool,
    /// Use the lossy encoding to encode the image using the VP8 compression format.
    pub use_lossy: bool,
    /// A quality value for the lossy encoding that must be between 0 and 100. Defaults to 95.
    pub lossy_quality: u8,
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            use_predictor_transform: true,
            use_lossy: false,
            lossy_quality: 95,
        }
    }
}

/// Encode image data losslessly with the indicated color type.
///
/// # Panics
///
/// Panics if the image data is not of the indicated dimensions.
pub(crate) fn encode_frame_lossless<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    params: EncoderParams,
    implicit_dimensions: bool,
) -> Result<(), EncodingError> {
    let w = &mut BitWriter::new(writer);

    let (is_color, is_alpha, bytes_per_pixel) = match color {
        ColorType::L8 => (false, false, 1),
        ColorType::La8 => (false, true, 2),
        ColorType::Rgb8 => (true, false, 3),
        ColorType::Rgba8 => (true, true, 4),
    };

    assert_eq!(
        (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
        data.len() as u64
    );

    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodingError::InvalidDimensions);
    }

    if !implicit_dimensions {
        w.write_bits(0x2f, 8)?; // signature
        w.write_bits(u64::from(width) - 1, 14)?;
        w.write_bits(u64::from(height) - 1, 14)?;

        w.write_bits(u64::from(is_alpha), 1)?; // alpha used
        w.write_bits(0x0, 3)?; // version
    }
    // subtract green transform
    w.write_bits(0b101, 3)?;

    // predictor transform
    if params.use_predictor_transform {
        w.write_bits(0b111001, 6)?;
        w.write_bits(0x0, 1)?; // no color cache
        write_single_entry_huffman_tree(w, 2)?;
        for _ in 0..4 {
            write_single_entry_huffman_tree(w, 0)?;
        }
    }

    // transforms done
    w.write_bits(0x0, 1)?;

    // color cache
    w.write_bits(0x0, 1)?;

    // meta-huffman codes
    w.write_bits(0x0, 1)?;

    // expand to RGBA
    let mut pixels = match color {
        ColorType::L8 => data.iter().flat_map(|&p| [p, p, p, 255]).collect(),
        ColorType::La8 => data
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        ColorType::Rgb8 => data
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        ColorType::Rgba8 => data.to_vec(),
    };

    // compute subtract green transform
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_sub(pixel[1]);
        pixel[2] = pixel[2].wrapping_sub(pixel[1]);
    }

    // compute predictor transform
    if params.use_predictor_transform {
        let row_bytes = width as usize * 4;
        for y in (1..height as usize).rev() {
            let (prev, current) =
                pixels[(y - 1) * row_bytes..][..row_bytes * 2].split_at_mut(row_bytes);
            for (c, p) in current.iter_mut().zip(prev) {
                *c = c.wrapping_sub(*p);
            }
        }
        for i in (4..row_bytes).rev() {
            pixels[i] = pixels[i].wrapping_sub(pixels[i - 4]);
        }
        pixels[3] = pixels[3].wrapping_sub(255);
    }

    // compute frequencies
    let mut frequencies0 = [0u32; 256];
    let mut frequencies1 = [0u32; 280];
    let mut frequencies2 = [0u32; 256];
    let mut frequencies3 = [0u32; 256];
    let mut it = pixels.chunks_exact(4).peekable();
    match color {
        ColorType::L8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
            frequencies3[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies1[pixel[1] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        ColorType::La8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies1[pixel[1] as usize] += 1;
                frequencies3[pixel[3] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        ColorType::Rgb8 => {
            frequencies3[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies0[pixel[0] as usize] += 1;
                frequencies1[pixel[1] as usize] += 1;
                frequencies2[pixel[2] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        ColorType::Rgba8 => {
            while let Some(pixel) = it.next() {
                frequencies0[pixel[0] as usize] += 1;
                frequencies1[pixel[1] as usize] += 1;
                frequencies2[pixel[2] as usize] += 1;
                frequencies3[pixel[3] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
    }

    // compute and write huffman codes
    let mut lengths0 = [0u8; 256];
    let mut lengths1 = [0u8; 280];
    let mut lengths2 = [0u8; 256];
    let mut lengths3 = [0u8; 256];
    let mut codes0 = [0u16; 256];
    let mut codes1 = [0u16; 280];
    let mut codes2 = [0u16; 256];
    let mut codes3 = [0u16; 256];
    write_huffman_tree(w, &frequencies1, &mut lengths1, &mut codes1)?;
    if is_color {
        write_huffman_tree(w, &frequencies0, &mut lengths0, &mut codes0)?;
        write_huffman_tree(w, &frequencies2, &mut lengths2, &mut codes2)?;
    } else {
        write_single_entry_huffman_tree(w, 0)?;
        write_single_entry_huffman_tree(w, 0)?;
    }
    if is_alpha {
        write_huffman_tree(w, &frequencies3, &mut lengths3, &mut codes3)?;
    } else if params.use_predictor_transform {
        write_single_entry_huffman_tree(w, 0)?;
    } else {
        write_single_entry_huffman_tree(w, 255)?;
    }
    write_single_entry_huffman_tree(w, 1)?;

    // Write image data
    let mut it = pixels.chunks_exact(4).peekable();
    match color {
        ColorType::L8 => {
            while let Some(pixel) = it.next() {
                w.write_bits(
                    u64::from(codes1[pixel[1] as usize]),
                    lengths1[pixel[1] as usize],
                )?;
                write_run(w, pixel, &mut it, &codes1, &lengths1)?;
            }
        }
        ColorType::La8 => {
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len3 = lengths3[pixel[3] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes3[pixel[3] as usize]) << len1);

                w.write_bits(code, len1 + len3)?;
                write_run(w, pixel, &mut it, &codes1, &lengths1)?;
            }
        }
        ColorType::Rgb8 => {
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len0 = lengths0[pixel[0] as usize];
                let len2 = lengths2[pixel[2] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes0[pixel[0] as usize]) << len1)
                    | (u64::from(codes2[pixel[2] as usize]) << (len1 + len0));

                w.write_bits(code, len1 + len0 + len2)?;
                write_run(w, pixel, &mut it, &codes1, &lengths1)?;
            }
        }
        ColorType::Rgba8 => {
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len0 = lengths0[pixel[0] as usize];
                let len2 = lengths2[pixel[2] as usize];
                let len3 = lengths3[pixel[3] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes0[pixel[0] as usize]) << len1)
                    | (u64::from(codes2[pixel[2] as usize]) << (len1 + len0))
                    | (u64::from(codes3[pixel[3] as usize]) << (len1 + len0 + len2));

                w.write_bits(code, len1 + len0 + len2 + len3)?;
                write_run(w, pixel, &mut it, &codes1, &lengths1)?;
            }
        }
    }

    w.flush()?;
    Ok(())
}
