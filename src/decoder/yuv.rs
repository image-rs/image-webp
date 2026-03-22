//! Utilities for doing the YUV -> RGB conversion
//! The images are encoded in the Y'CbCr format as detailed here: <https://en.wikipedia.org/wiki/YCbCr>
//! so need to be converted to RGB to be displayed

// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]
//! To do the YUV -> RGB conversion we need to first decide how to map the yuv values to the pixels
//! The y buffer is the same size as the pixel buffer so that maps 1-1 but the
//! u and v buffers are half the size of the pixel buffer so we need to scale it up
//! The simple way to upscale is just to take each u/v value and associate it with the 4
//! pixels around it e.g. for a 4x4 image:
//!
//! ||||||
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! ||||||
//!
//! |||||||
//! |uu|vv|
//! |uu|vv|
//! |||||||
//!
//! Then each of the 2x2 pixels would match the u/v from the same quadrant
//!
//! However fancy upsampling is the default for libwebp which does a little more work to make the values smoother
//! It interpolates u and v so that for e.g. the pixel 1 down and 1 from the left the u value
//! would be (9*u0 + 3*u1 + 3*u2 + u3 + 8) / 16 and similar for the other pixels
//! The edges are mirrored, so for the pixel 1 down and 0 from the left it uses (9*u0 + 3*u2 + 3*u0 + u2 + 8) / 16

use alloc::vec;
use alloc::vec::Vec;

use crate::common::prediction::SimdTokenType;

/// `_mm_mulhi_epu16` emulation
fn mulhi(v: u8, coeff: u16) -> i32 {
    ((u32::from(v) * u32::from(coeff)) >> 8) as i32
}

/// This function has been rewritten to encourage auto-vectorization.
///
/// Based on [src/dsp/yuv.h](https://github.com/webmproject/libwebp/blob/8534f53960befac04c9631e6e50d21dcb42dfeaf/src/dsp/yuv.h#L79)
/// from the libwebp source.
/// ```text
/// const YUV_FIX2: i32 = 6;
/// const YUV_MASK2: i32 = (256 << YUV_FIX2) - 1;
/// fn clip(v: i32) -> u8 {
///     if (v & !YUV_MASK2) == 0 {
///         (v >> YUV_FIX2) as u8
///     } else if v < 0 {
///         0
///     } else {
///         255
///     }
/// }
/// ```
// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
fn clip(v: i32) -> u8 {
    const YUV_FIX2: i32 = 6;
    (v >> YUV_FIX2).max(0).min(255) as u8
}

#[inline(always)]
fn yuv_to_r(y: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234)
}

#[inline(always)]
fn yuv_to_g(y: u8, u: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708)
}

#[inline(always)]
fn yuv_to_b(y: u8, u: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685)
}

/// Fills an rgb buffer with the image from the yuv buffers
/// Size of the buffer is assumed to be correct
/// BPP is short for bytes per pixel, allows both rgb and rgba to be decoded
pub(crate) fn fill_rgb_buffer_fancy<const BPP: usize>(
    buffer: &mut [u8],
    y_buffer: &[u8],
    u_buffer: &[u8],
    v_buffer: &[u8],
    width: usize,
    height: usize,
    buffer_width: usize,
) {
    // Summon SIMD token once for entire YUV conversion
    let simd_token: SimdTokenType = {
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        {
            use archmage::SimdToken;
            archmage::X64V3Token::summon()
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            use archmage::SimdToken;
            archmage::NeonToken::summon()
        }
        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
        )))]
        {
            None
        }
    };

    // buffer width is always even so don't need to do div_ceil
    let chroma_buffer_width = buffer_width / 2;
    let chroma_width = width.div_ceil(2);

    // fill top row first since it only uses the top u/v row
    let top_row_y = &y_buffer[..width];
    let top_row_u = &u_buffer[..chroma_width];
    let top_row_v = &v_buffer[..chroma_width];
    let top_row_buffer = &mut buffer[..width * BPP];
    fill_row_fancy_with_1_uv_row::<BPP>(top_row_buffer, top_row_y, top_row_u, top_row_v);

    let mut main_row_chunks = buffer[width * BPP..].chunks_exact_mut(width * BPP * 2);
    // the y buffer iterator limits the end of the row iterator so we need this end index
    let end_y_index = height * buffer_width;
    let mut main_y_chunks = y_buffer[buffer_width..end_y_index].chunks_exact(buffer_width * 2);
    let mut main_u_windows = u_buffer
        .windows(chroma_buffer_width * 2)
        .step_by(chroma_buffer_width);
    let mut main_v_windows = v_buffer
        .windows(chroma_buffer_width * 2)
        .step_by(chroma_buffer_width);

    for (((row_buffer, y_rows), u_rows), v_rows) in (&mut main_row_chunks)
        .zip(&mut main_y_chunks)
        .zip(&mut main_u_windows)
        .zip(&mut main_v_windows)
    {
        let (u_row_1, u_row_2) = u_rows.split_at(chroma_buffer_width);
        let (v_row_1, v_row_2) = v_rows.split_at(chroma_buffer_width);
        let (row_buf_1, row_buf_2) = row_buffer.split_at_mut(width * BPP);
        let (y_row_1, y_row_2) = y_rows.split_at(buffer_width);
        fill_row_fancy_with_2_uv_rows::<BPP>(
            row_buf_1,
            &y_row_1[..width],
            &u_row_1[..chroma_width],
            &u_row_2[..chroma_width],
            &v_row_1[..chroma_width],
            &v_row_2[..chroma_width],
            simd_token,
        );
        fill_row_fancy_with_2_uv_rows::<BPP>(
            row_buf_2,
            &y_row_2[..width],
            &u_row_2[..chroma_width],
            &u_row_1[..chroma_width],
            &v_row_2[..chroma_width],
            &v_row_1[..chroma_width],
            simd_token,
        );
    }

    let final_row_buffer = main_row_chunks.into_remainder();

    // if the image has even height there will be one final row with only one u/v row matching it
    if !final_row_buffer.is_empty() {
        let final_y_row = main_y_chunks.remainder();

        let chroma_height = height.div_ceil(2);
        let start_chroma_index = (chroma_height - 1) * chroma_buffer_width;

        let final_u_row = &u_buffer[start_chroma_index..];
        let final_v_row = &v_buffer[start_chroma_index..];
        fill_row_fancy_with_1_uv_row::<BPP>(
            final_row_buffer,
            &final_y_row[..width],
            &final_u_row[..chroma_width],
            &final_v_row[..chroma_width],
        );
    }
}

/// Fills a row with the fancy interpolation as detailed
#[allow(unused_variables)]
fn fill_row_fancy_with_2_uv_rows<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    simd_token: SimdTokenType,
) {
    // Use SIMD intrinsics for RGB (BPP=3) if available and row is wide enough
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if BPP == 3
        && y_row.len() >= 17
        && let Some(token) = simd_token
    {
        fill_row_fancy_with_2_uv_rows_simd::<BPP>(
            row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2, token,
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if BPP == 3 && y_row.len() >= 17 {
        if let Some(token) = simd_token {
            fill_row_fancy_with_2_uv_rows_neon::<BPP>(
                row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2, token,
            );
            return;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if BPP == 3 && y_row.len() >= 17 {
        if let Some(token) = simd_token {
            fill_row_fancy_with_2_uv_rows_wasm::<BPP>(
                row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2, token,
            );
            return;
        }
    }

    fill_row_fancy_with_2_uv_rows_scalar::<BPP>(
        row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2,
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn fill_row_fancy_with_2_uv_rows_simd<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    token: archmage::X64V3Token,
) {
    // Handle first pixel separately (edge case)
    {
        let rgb1 = &mut row_buffer[0..3];
        let y_value = y_row[0];
        let u_value = get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
        let v_value = get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
        set_pixel(rgb1, y_value, u_value, v_value);
    }

    let width = y_row.len();
    let mut y_offset = 1; // Start after first edge pixel
    let mut uv_offset = 0;
    let mut rgb_offset = BPP;

    // Process chunks of 32 Y pixels (16 pixel pairs) with SIMD for lower overhead
    // Need at least 32 Y pixels and 17 U/V samples
    while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
        super::yuv_simd::fancy_upsample_16_pairs_with_token(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 32;
        uv_offset += 16;
        rgb_offset += 96;
    }

    // Process remaining chunks of 16 Y pixels (8 pixel pairs) with SIMD
    // Need at least 16 Y pixels and 9 U/V samples
    while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
        super::yuv_simd::fancy_upsample_8_pairs_with_token(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 16;
        uv_offset += 8;
        rgb_offset += 48;
    }

    // Handle remaining pixels with scalar code
    // Process remaining full pairs
    while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
        let u_val_1 = &u_row_1[uv_offset..uv_offset + 2];
        let u_val_2 = &u_row_2[uv_offset..uv_offset + 2];
        let v_val_1 = &v_row_1[uv_offset..uv_offset + 2];
        let v_val_2 = &v_row_2[uv_offset..uv_offset + 2];

        {
            let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
            let y_value = y_row[y_offset];
            let u_value = get_fancy_chroma_value(u_val_1[0], u_val_1[1], u_val_2[0], u_val_2[1]);
            let v_value = get_fancy_chroma_value(v_val_1[0], v_val_1[1], v_val_2[0], v_val_2[1]);
            set_pixel(rgb1, y_value, u_value, v_value);
        }
        {
            let rgb2 = &mut row_buffer[rgb_offset + BPP..rgb_offset + BPP + 3];
            let y_value = y_row[y_offset + 1];
            let u_value = get_fancy_chroma_value(u_val_1[1], u_val_1[0], u_val_2[1], u_val_2[0]);
            let v_value = get_fancy_chroma_value(v_val_1[1], v_val_1[0], v_val_2[1], v_val_2[0]);
            set_pixel(rgb2, y_value, u_value, v_value);
        }

        y_offset += 2;
        uv_offset += 1;
        rgb_offset += BPP * 2;
    }

    // Handle final odd pixel if present
    if y_offset < width {
        let final_u_1 = *u_row_1.last().unwrap();
        let final_u_2 = *u_row_2.last().unwrap();
        let final_v_1 = *v_row_1.last().unwrap();
        let final_v_2 = *v_row_2.last().unwrap();

        let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
        let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
        let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
        set_pixel(rgb1, y_row[y_offset], u_value, v_value);
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
fn fill_row_fancy_with_2_uv_rows_neon<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    token: archmage::NeonToken,
) {
    // Handle first pixel separately (edge case)
    {
        let rgb1 = &mut row_buffer[0..3];
        let y_value = y_row[0];
        let u_value = get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
        let v_value = get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
        set_pixel(rgb1, y_value, u_value, v_value);
    }

    let width = y_row.len();
    let mut y_offset = 1;
    let mut uv_offset = 0;
    let mut rgb_offset = BPP;

    // Process chunks of 32 Y pixels (16 pixel pairs) with NEON
    while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
        super::yuv_neon::fancy_upsample_16_pairs_neon(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 32;
        uv_offset += 16;
        rgb_offset += 96;
    }

    // Process remaining chunks of 16 Y pixels (8 pixel pairs) with NEON
    while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
        super::yuv_neon::fancy_upsample_8_pairs_neon(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 16;
        uv_offset += 8;
        rgb_offset += 48;
    }

    // Handle remaining pixels with scalar code
    while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
        let u_val_1 = &u_row_1[uv_offset..uv_offset + 2];
        let u_val_2 = &u_row_2[uv_offset..uv_offset + 2];
        let v_val_1 = &v_row_1[uv_offset..uv_offset + 2];
        let v_val_2 = &v_row_2[uv_offset..uv_offset + 2];

        {
            let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
            let y_value = y_row[y_offset];
            let u_value = get_fancy_chroma_value(u_val_1[0], u_val_1[1], u_val_2[0], u_val_2[1]);
            let v_value = get_fancy_chroma_value(v_val_1[0], v_val_1[1], v_val_2[0], v_val_2[1]);
            set_pixel(rgb1, y_value, u_value, v_value);
        }
        {
            let rgb2 = &mut row_buffer[rgb_offset + BPP..rgb_offset + BPP + 3];
            let y_value = y_row[y_offset + 1];
            let u_value = get_fancy_chroma_value(u_val_1[1], u_val_1[0], u_val_2[1], u_val_2[0]);
            let v_value = get_fancy_chroma_value(v_val_1[1], v_val_1[0], v_val_2[1], v_val_2[0]);
            set_pixel(rgb2, y_value, u_value, v_value);
        }

        y_offset += 2;
        uv_offset += 1;
        rgb_offset += BPP * 2;
    }

    // Handle final odd pixel if present
    if y_offset < width {
        let final_u_1 = *u_row_1.last().unwrap();
        let final_u_2 = *u_row_2.last().unwrap();
        let final_v_1 = *v_row_1.last().unwrap();
        let final_v_2 = *v_row_2.last().unwrap();

        let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
        let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
        let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
        set_pixel(rgb1, y_row[y_offset], u_value, v_value);
    }
}

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
fn fill_row_fancy_with_2_uv_rows_wasm<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    token: archmage::Wasm128Token,
) {
    // Handle first pixel separately (edge case)
    {
        let rgb1 = &mut row_buffer[0..3];
        let y_value = y_row[0];
        let u_value = get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
        let v_value = get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
        set_pixel(rgb1, y_value, u_value, v_value);
    }

    let width = y_row.len();
    let mut y_offset = 1;
    let mut uv_offset = 0;
    let mut rgb_offset = BPP;

    // Process chunks of 32 Y pixels (16 pixel pairs) with WASM SIMD
    while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
        super::yuv_wasm::fancy_upsample_16_pairs_wasm(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 32;
        uv_offset += 16;
        rgb_offset += 96;
    }

    // Process remaining chunks of 16 Y pixels (8 pixel pairs) with WASM SIMD
    while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
        super::yuv_wasm::fancy_upsample_8_pairs_wasm(
            token,
            &y_row[y_offset..],
            &u_row_1[uv_offset..],
            &u_row_2[uv_offset..],
            &v_row_1[uv_offset..],
            &v_row_2[uv_offset..],
            &mut row_buffer[rgb_offset..],
        );
        y_offset += 16;
        uv_offset += 8;
        rgb_offset += 48;
    }

    // Handle remaining pixels with scalar code
    while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
        let u_val_1 = &u_row_1[uv_offset..uv_offset + 2];
        let u_val_2 = &u_row_2[uv_offset..uv_offset + 2];
        let v_val_1 = &v_row_1[uv_offset..uv_offset + 2];
        let v_val_2 = &v_row_2[uv_offset..uv_offset + 2];

        {
            let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
            let y_value = y_row[y_offset];
            let u_value = get_fancy_chroma_value(u_val_1[0], u_val_1[1], u_val_2[0], u_val_2[1]);
            let v_value = get_fancy_chroma_value(v_val_1[0], v_val_1[1], v_val_2[0], v_val_2[1]);
            set_pixel(rgb1, y_value, u_value, v_value);
        }
        {
            let rgb2 = &mut row_buffer[rgb_offset + BPP..rgb_offset + BPP + 3];
            let y_value = y_row[y_offset + 1];
            let u_value = get_fancy_chroma_value(u_val_1[1], u_val_1[0], u_val_2[1], u_val_2[0]);
            let v_value = get_fancy_chroma_value(v_val_1[1], v_val_1[0], v_val_2[1], v_val_2[0]);
            set_pixel(rgb2, y_value, u_value, v_value);
        }

        y_offset += 2;
        uv_offset += 1;
        rgb_offset += BPP * 2;
    }

    // Handle final odd pixel if present
    if y_offset < width {
        let final_u_1 = *u_row_1.last().unwrap();
        let final_u_2 = *u_row_2.last().unwrap();
        let final_v_1 = *v_row_1.last().unwrap();
        let final_v_2 = *v_row_2.last().unwrap();

        let rgb1 = &mut row_buffer[rgb_offset..rgb_offset + 3];
        let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
        let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
        set_pixel(rgb1, y_row[y_offset], u_value, v_value);
    }
}

maybe_autoversion! {
    fn fill_row_fancy_with_2_uv_rows_scalar<const BPP: usize>(
        row_buffer: &mut [u8],
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
    ) {
        // need to do left pixel separately since it will only have one u/v value
        {
            let rgb1 = &mut row_buffer[0..3];
            let y_value = y_row[0];
            // first pixel uses the first u/v as the main one
            let u_value = get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
            let v_value = get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
            set_pixel(rgb1, y_value, u_value, v_value);
        }

        let rest_row_buffer = &mut row_buffer[BPP..];
        let rest_y_row = &y_row[1..];

        // we do two pixels at a time since they share the same u/v values
        let mut main_row_chunks = rest_row_buffer.chunks_exact_mut(BPP * 2);
        let mut main_y_chunks = rest_y_row.chunks_exact(2);

        for (((((rgb, y_val), u_val_1), u_val_2), v_val_1), v_val_2) in (&mut main_row_chunks)
            .zip(&mut main_y_chunks)
            .zip(u_row_1.windows(2))
            .zip(u_row_2.windows(2))
            .zip(v_row_1.windows(2))
            .zip(v_row_2.windows(2))
        {
            {
                let rgb1 = &mut rgb[0..3];
                let y_value = y_val[0];
                // first pixel uses the first u/v as the main one
                let u_value = get_fancy_chroma_value(u_val_1[0], u_val_1[1], u_val_2[0], u_val_2[1]);
                let v_value = get_fancy_chroma_value(v_val_1[0], v_val_1[1], v_val_2[0], v_val_2[1]);
                set_pixel(rgb1, y_value, u_value, v_value);
            }
            {
                let rgb2 = &mut rgb[BPP..];
                let y_value = y_val[1];
                let u_value = get_fancy_chroma_value(u_val_1[1], u_val_1[0], u_val_2[1], u_val_2[0]);
                let v_value = get_fancy_chroma_value(v_val_1[1], v_val_1[0], v_val_2[1], v_val_2[0]);
                set_pixel(rgb2, y_value, u_value, v_value);
            }
        }

        let final_pixel = main_row_chunks.into_remainder();
        let final_y = main_y_chunks.remainder();

        if let (rgb, [y_value]) = (final_pixel, final_y) {
            let final_u_1 = *u_row_1.last().unwrap();
            let final_u_2 = *u_row_2.last().unwrap();

            let final_v_1 = *v_row_1.last().unwrap();
            let final_v_2 = *v_row_2.last().unwrap();

            let rgb1 = &mut rgb[0..3];
            // first pixel uses the first u/v as the main one
            let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
            let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
            set_pixel(rgb1, *y_value, u_value, v_value);
        }
    }
}

maybe_autoversion! {
    fn fill_row_fancy_with_1_uv_row<const BPP: usize>(
        row_buffer: &mut [u8],
        y_row: &[u8],
        u_row: &[u8],
        v_row: &[u8],
    ) {
        // doing left pixel first
        {
            let rgb1 = &mut row_buffer[0..3];
            let y_value = y_row[0];

            let u_value = u_row[0];
            let v_value = v_row[0];
            set_pixel(rgb1, y_value, u_value, v_value);
        }

        // two pixels at a time since they share the same u/v value
        let mut main_row_chunks = row_buffer[BPP..].chunks_exact_mut(BPP * 2);
        let mut main_y_row_chunks = y_row[1..].chunks_exact(2);

        for (((rgb, y_val), u_val), v_val) in (&mut main_row_chunks)
            .zip(&mut main_y_row_chunks)
            .zip(u_row.windows(2))
            .zip(v_row.windows(2))
        {
            {
                let rgb1 = &mut rgb[0..3];
                let y_value = y_val[0];
                // first pixel uses the first u/v as the main one
                let u_value = get_fancy_chroma_value(u_val[0], u_val[1], u_val[0], u_val[1]);
                let v_value = get_fancy_chroma_value(v_val[0], v_val[1], v_val[0], v_val[1]);
                set_pixel(rgb1, y_value, u_value, v_value);
            }
            {
                let rgb2 = &mut rgb[BPP..];
                let y_value = y_val[1];
                let u_value = get_fancy_chroma_value(u_val[1], u_val[0], u_val[1], u_val[0]);
                let v_value = get_fancy_chroma_value(v_val[1], v_val[0], v_val[1], v_val[0]);
                set_pixel(rgb2, y_value, u_value, v_value);
            }
        }

        let final_pixel = main_row_chunks.into_remainder();
        let final_y = main_y_row_chunks.remainder();

        if let (rgb, [final_y]) = (final_pixel, final_y) {
            let final_u = *u_row.last().unwrap();
            let final_v = *v_row.last().unwrap();

            set_pixel(rgb, *final_y, final_u, final_v);
        }
    }
}

#[inline]
fn get_fancy_chroma_value(main: u8, secondary1: u8, secondary2: u8, tertiary: u8) -> u8 {
    let val0 = u16::from(main);
    let val1 = u16::from(secondary1);
    let val2 = u16::from(secondary2);
    let val3 = u16::from(tertiary);
    ((9 * val0 + 3 * val1 + 3 * val2 + val3 + 8) / 16) as u8
}

#[inline]
fn set_pixel(rgb: &mut [u8], y: u8, u: u8, v: u8) {
    rgb[0] = yuv_to_r(y, v);
    rgb[1] = yuv_to_g(y, u, v);
    rgb[2] = yuv_to_b(y, u);
}

/// Simple conversion, not currently used but could add a config to allow for using the simple
#[allow(unused)]
pub(crate) fn fill_rgb_buffer_simple<const BPP: usize>(
    buffer: &mut [u8],
    y_buffer: &[u8],
    u_buffer: &[u8],
    v_buffer: &[u8],
    width: usize,
    chroma_width: usize,
    buffer_width: usize,
) {
    let u_row_twice_iter = u_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| core::iter::repeat_n(n, 2));
    let v_row_twice_iter = v_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| core::iter::repeat_n(n, 2));

    for (((row, y_row), u_row), v_row) in buffer
        .chunks_exact_mut(width * BPP)
        .zip(y_buffer.chunks_exact(buffer_width))
        .zip(u_row_twice_iter)
        .zip(v_row_twice_iter)
    {
        fill_rgba_row_simple::<BPP>(
            &y_row[..width],
            &u_row[..chroma_width],
            &v_row[..chroma_width],
            row,
        );
    }
}

fn fill_rgba_row_simple<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Use SIMD for RGB (BPP=3) if available and row is wide enough
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if BPP == 3 && y_vec.len() >= 8 && is_x86_feature_detected!("sse2") {
        fill_rgba_row_simple_simd::<BPP>(y_vec, u_vec, v_vec, rgba);
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if BPP == 3 && y_vec.len() >= 16 {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            super::yuv_neon::yuv420_to_rgb_row_neon(token, y_vec, u_vec, v_vec, rgba);
            return;
        }
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if BPP == 3 && y_vec.len() >= 16 {
        use archmage::SimdToken;
        if let Some(token) = archmage::Wasm128Token::summon() {
            super::yuv_wasm::yuv420_to_rgb_row_wasm(token, y_vec, u_vec, v_vec, rgba);
            return;
        }
    }

    fill_rgba_row_simple_scalar::<BPP>(y_vec, u_vec, v_vec, rgba);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn fill_rgba_row_simple_simd<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Use the new row-based SIMD function that processes 32 pixels at a time
    super::yuv_simd::yuv420_to_rgb_row(y_vec, u_vec, v_vec, rgba);
}

fn fill_rgba_row_simple_scalar<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Fill 2 pixels per iteration: these pixels share `u` and `v` components
    let mut rgb_chunks = rgba.chunks_exact_mut(BPP * 2);
    let mut y_chunks = y_vec.chunks_exact(2);
    let mut u_iter = u_vec.iter();
    let mut v_iter = v_vec.iter();

    for (((rgb, y), &u), &v) in (&mut rgb_chunks)
        .zip(&mut y_chunks)
        .zip(&mut u_iter)
        .zip(&mut v_iter)
    {
        let coeffs = [
            mulhi(v, 26149),
            mulhi(u, 6419),
            mulhi(v, 13320),
            mulhi(u, 33050),
        ];

        let get_r = |y: u8| clip(mulhi(y, 19077) + coeffs[0] - 14234);
        let get_g = |y: u8| clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
        let get_b = |y: u8| clip(mulhi(y, 19077) + coeffs[3] - 17685);

        let rgb1 = &mut rgb[0..3];
        rgb1[0] = get_r(y[0]);
        rgb1[1] = get_g(y[0]);
        rgb1[2] = get_b(y[0]);

        let rgb2 = &mut rgb[BPP..];
        rgb2[0] = get_r(y[1]);
        rgb2[1] = get_g(y[1]);
        rgb2[2] = get_b(y[1]);
    }

    let remainder = rgb_chunks.into_remainder();
    if remainder.len() >= 3
        && let (Some(&y), Some(&u), Some(&v)) = (
            y_chunks.remainder().iter().next(),
            u_iter.next(),
            v_iter.next(),
        )
    {
        let coeffs = [
            mulhi(v, 26149),
            mulhi(u, 6419),
            mulhi(v, 13320),
            mulhi(u, 33050),
        ];

        remainder[0] = clip(mulhi(y, 19077) + coeffs[0] - 14234);
        remainder[1] = clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
        remainder[2] = clip(mulhi(y, 19077) + coeffs[3] - 17685);
    }
}

// constants used for yuv -> rgb conversion, using ones from libwebp
const YUV_FIX: i32 = 16;
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

/// SIMD-accelerated RGB to YUV 4:2:0 conversion using the `yuv` crate.
///
/// This provides 10-150× speedup over scalar conversion using AVX2/SSE/NEON SIMD.
/// The output is compatible with VP8 encoding (BT.601 matrix, full range).
///
/// Returns (y_bytes, u_bytes, v_bytes) with macroblock-aligned dimensions.
#[cfg(feature = "fast-yuv")]
#[allow(dead_code)] // Alternative YUV conversion implementation
pub(crate) fn convert_image_yuv_simd<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use yuv::{
        YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
        rgb_to_yuv420, rgba_to_yuv420,
    };

    let width_usize = usize::from(width);
    let height_usize = usize::from(height);
    let mb_width = width_usize.div_ceil(16);
    let mb_height = height_usize.div_ceil(16);
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let y_size = 16 * mb_width * 16 * mb_height;
    let chroma_size = 8 * mb_width * 8 * mb_height;

    // Use yuv crate for fast SIMD conversion
    let mut yuv_image =
        YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

    // Convert using appropriate function based on BPP
    let result = if BPP == 4 {
        rgba_to_yuv420(
            &mut yuv_image,
            image_data,
            (stride * BPP) as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt601,
            YuvConversionMode::Balanced,
        )
    } else {
        rgb_to_yuv420(
            &mut yuv_image,
            image_data,
            (stride * BPP) as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt601,
            YuvConversionMode::Balanced,
        )
    };

    if result.is_err() {
        // Fall back to scalar implementation on error
        return convert_image_yuv::<BPP>(image_data, width, height, stride);
    }

    // Extract planes from yuv crate output
    let y_src = yuv_image.y_plane.borrow();
    let u_src = yuv_image.u_plane.borrow();
    let v_src = yuv_image.v_plane.borrow();

    let src_chroma_width = width_usize.div_ceil(2);
    let src_chroma_height = height_usize.div_ceil(2);

    // Allocate output buffers with macroblock alignment
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Copy Y plane with padding
    for y in 0..height_usize {
        let src_start = y * width_usize;
        let dst_start = y * luma_width;
        y_bytes[dst_start..dst_start + width_usize]
            .copy_from_slice(&y_src[src_start..src_start + width_usize]);

        // Horizontal padding
        let last_y = y_bytes[dst_start + width_usize - 1];
        for x in width_usize..luma_width {
            y_bytes[dst_start + x] = last_y;
        }
    }

    // Vertical padding for Y - copy the last row repeatedly
    if height_usize < mb_height * 16 {
        let last_row: Vec<u8> =
            y_bytes[(height_usize - 1) * luma_width..height_usize * luma_width].to_vec();
        for y in height_usize..(mb_height * 16) {
            let dst_row = y * luma_width;
            y_bytes[dst_row..dst_row + luma_width].copy_from_slice(&last_row);
        }
    }

    // Copy U/V planes with padding
    for y in 0..src_chroma_height {
        let src_start = y * src_chroma_width;
        let dst_start = y * chroma_width;
        u_bytes[dst_start..dst_start + src_chroma_width]
            .copy_from_slice(&u_src[src_start..src_start + src_chroma_width]);
        v_bytes[dst_start..dst_start + src_chroma_width]
            .copy_from_slice(&v_src[src_start..src_start + src_chroma_width]);

        // Horizontal padding
        let last_u = u_bytes[dst_start + src_chroma_width - 1];
        let last_v = v_bytes[dst_start + src_chroma_width - 1];
        for x in src_chroma_width..chroma_width {
            u_bytes[dst_start + x] = last_u;
            v_bytes[dst_start + x] = last_v;
        }
    }

    // Vertical padding for U/V - copy the last row repeatedly
    if src_chroma_height < mb_height * 8 {
        let last_u_row: Vec<u8> = u_bytes
            [(src_chroma_height - 1) * chroma_width..src_chroma_height * chroma_width]
            .to_vec();
        let last_v_row: Vec<u8> = v_bytes
            [(src_chroma_height - 1) * chroma_width..src_chroma_height * chroma_width]
            .to_vec();
        for y in src_chroma_height..(mb_height * 8) {
            let dst_row = y * chroma_width;
            u_bytes[dst_row..dst_row + chroma_width].copy_from_slice(&last_u_row);
            v_bytes[dst_row..dst_row + chroma_width].copy_from_slice(&last_v_row);
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

/// converts the whole image to yuv data and adds values on the end to make it match the macroblock sizes
/// downscales the u/v data as well so it's half the width and height of the y data
pub(crate) fn convert_image_yuv<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Process pairs of rows for chroma downsampling (2x2 averaging)
    let row_pairs = height / 2;
    let odd_height = height & 1 != 0;
    let col_pairs = width / 2;
    let odd_width = width & 1 != 0;

    for row_pair in 0..row_pairs {
        let src_row1 = row_pair * 2;
        let src_row2 = src_row1 + 1;
        let chroma_row = row_pair;

        // Process pairs of columns
        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            // Get 4 RGB pixels for 2x2 block
            let rgb1 = &image_data[(src_row1 * stride + src_col1) * BPP..][..BPP];
            let rgb2 = &image_data[(src_row1 * stride + src_col2) * BPP..][..BPP];
            let rgb3 = &image_data[(src_row2 * stride + src_col1) * BPP..][..BPP];
            let rgb4 = &image_data[(src_row2 * stride + src_col2) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row1 * luma_width + src_col1] = rgb_to_y(rgb1);
            y_bytes[src_row1 * luma_width + src_col2] = rgb_to_y(rgb2);
            y_bytes[src_row2 * luma_width + src_col1] = rgb_to_y(rgb3);
            y_bytes[src_row2 * luma_width + src_col2] = rgb_to_y(rgb4);

            // Convert to U/V (averaged)
            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(rgb1, rgb2, rgb3, rgb4);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(rgb1, rgb2, rgb3, rgb4);
        }

        // Handle last column if width is odd
        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            // Get 2 RGB pixels (duplicate horizontally for chroma)
            let rgb1 = &image_data[(src_row1 * stride + src_col) * BPP..][..BPP];
            let rgb3 = &image_data[(src_row2 * stride + src_col) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row1 * luma_width + src_col] = rgb_to_y(rgb1);
            y_bytes[src_row2 * luma_width + src_col] = rgb_to_y(rgb3);

            // Convert to U/V (use same pixel twice horizontally)
            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(rgb1, rgb1, rgb3, rgb3);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(rgb1, rgb1, rgb3, rgb3);
        }
    }

    // Handle last row if height is odd
    if odd_height {
        let src_row = height - 1;
        let chroma_row = row_pairs;

        // Process pairs of columns
        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            // Get 2 RGB pixels (duplicate vertically for chroma)
            let rgb1 = &image_data[(src_row * stride + src_col1) * BPP..][..BPP];
            let rgb2 = &image_data[(src_row * stride + src_col2) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row * luma_width + src_col1] = rgb_to_y(rgb1);
            y_bytes[src_row * luma_width + src_col2] = rgb_to_y(rgb2);

            // Convert to U/V (use same row twice vertically)
            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(rgb1, rgb2, rgb1, rgb2);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(rgb1, rgb2, rgb1, rgb2);
        }

        // Handle corner case: both width and height are odd
        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            // Get single RGB pixel (duplicate in all directions for chroma)
            let rgb = &image_data[(src_row * stride + src_col) * BPP..][..BPP];

            // Convert to Y
            y_bytes[src_row * luma_width + src_col] = rgb_to_y(rgb);

            // Convert to U/V (use same pixel 4 times)
            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(rgb, rgb, rgb, rgb);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(rgb, rgb, rgb, rgb);
        }
    }

    // Replicate edge pixels to fill macroblock padding
    // Horizontal padding for Y
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }

    // Vertical padding for Y (including horizontal padding area)
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }

    // Horizontal padding for U/V
    let chroma_height = height.div_ceil(2);
    let actual_chroma_width = width.div_ceil(2);
    for y in 0..chroma_height {
        let last_u = u_bytes[y * chroma_width + actual_chroma_width - 1];
        let last_v = v_bytes[y * chroma_width + actual_chroma_width - 1];
        for x in actual_chroma_width..chroma_width {
            u_bytes[y * chroma_width + x] = last_u;
            v_bytes[y * chroma_width + x] = last_v;
        }
    }

    // Vertical padding for U/V
    for y in chroma_height..(mb_height * 8) {
        for x in 0..chroma_width {
            u_bytes[y * chroma_width + x] = u_bytes[(chroma_height - 1) * chroma_width + x];
            v_bytes[y * chroma_width + x] = v_bytes[(chroma_height - 1) * chroma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

pub(crate) fn convert_image_y<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let u_bytes = vec![127u8; chroma_size];
    let v_bytes = vec![127u8; chroma_size];

    // Process all source rows
    for y in 0..height {
        let src_row = &image_data[y * stride * BPP..y * stride * BPP + width * BPP];
        for x in 0..width {
            y_bytes[y * luma_width + x] = src_row[x * BPP];
        }
    }

    // Replicate edge pixels to fill macroblock padding
    // Horizontal padding for Y
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }

    // Vertical padding for Y (including horizontal padding area)
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

// values come from libwebp
// Y = 0.2568 * R + 0.5041 * G + 0.0979 * B + 16
// U = -0.1482 * R - 0.2910 * G + 0.4392 * B + 128
// V = 0.4392 * R - 0.3678 * G - 0.0714 * B + 128

// this is converted to 16 bit fixed point by multiplying by 2^16
// and shifting back

// Encoder-only functions below (used when std feature is enabled)
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_y(rgb: &[u8]) -> u8 {
    let luma = 16839 * i32::from(rgb[0]) + 33059 * i32::from(rgb[1]) + 6420 * i32::from(rgb[2]);
    ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
}

// get the average of the four surrounding pixels
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_u_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let u1 = rgb_to_u_raw(rgb1);
    let u2 = rgb_to_u_raw(rgb2);
    let u3 = rgb_to_u_raw(rgb3);
    let u4 = rgb_to_u_raw(rgb4);

    // Add rounding before shift (matches libwebp's VP8ClipUV with YUV_HALF << 2)
    ((u1 + u2 + u3 + u4 + (YUV_HALF << 2)) >> (YUV_FIX + 2)) as u8
}

// get the average of the four surrounding pixels
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn rgb_to_v_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let v1 = rgb_to_v_raw(rgb1);
    let v2 = rgb_to_v_raw(rgb2);
    let v3 = rgb_to_v_raw(rgb3);
    let v4 = rgb_to_v_raw(rgb4);

    // Add rounding before shift (matches libwebp's VP8ClipUV with YUV_HALF << 2)
    ((v1 + v2 + v3 + v4 + (YUV_HALF << 2)) >> (YUV_FIX + 2)) as u8
}

#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn rgb_to_u_raw(rgb: &[u8]) -> i32 {
    -9719 * i32::from(rgb[0]) - 19081 * i32::from(rgb[1])
        + 28800 * i32::from(rgb[2])
        + (128 << YUV_FIX)
}

#[cfg_attr(not(feature = "std"), allow(dead_code))]
fn rgb_to_v_raw(rgb: &[u8]) -> i32 {
    28800 * i32::from(rgb[0]) - 24116 * i32::from(rgb[1]) - 4684 * i32::from(rgb[2])
        + (128 << YUV_FIX)
}

/// Convert image to YUV420 using sharp (iterative) chroma downsampling.
///
/// This produces higher-quality chroma planes at the cost of being slower.
/// Uses the `yuv` crate's sharp YUV implementation with BT.601 matrix and sRGB gamma.
///
/// Falls back to standard conversion when the `fast-yuv` feature is not enabled.
#[cfg_attr(not(feature = "fast-yuv"), allow(unused_variables))]
pub(crate) fn convert_image_sharp_yuv(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
) -> (
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
    alloc::vec::Vec<u8>,
) {
    #[cfg(feature = "fast-yuv")]
    {
        use crate::encoder::PixelLayout;

        // Sharp YUV only applies to RGB/RGBA/BGR/BGRA inputs (chroma subsampling matters).
        // For grayscale, fall back to standard conversion.
        match color {
            PixelLayout::L8 => return convert_image_y::<1>(image_data, width, height, stride),
            PixelLayout::La8 => return convert_image_y::<2>(image_data, width, height, stride),
            PixelLayout::Yuv420 => {
                // Sharp YUV doesn't apply to already-subsampled data
                unreachable!("sharp YUV should not be called with Yuv420 input");
            }
            _ => {}
        }

        let w = usize::from(width);
        let h = usize::from(height);
        let mb_width = w.div_ceil(16);
        let mb_height = h.div_ceil(16);
        let luma_width = 16 * mb_width;
        let luma_height = 16 * mb_height;
        let chroma_width = 8 * mb_width;
        let chroma_height = 8 * mb_height;

        // Allocate planar buffers at macroblock-aligned sizes
        let mut y_bytes = alloc::vec![0u8; luma_width * luma_height];
        let mut u_bytes = alloc::vec![0u8; chroma_width * chroma_height];
        let mut v_bytes = alloc::vec![0u8; chroma_width * chroma_height];

        // Create a YuvPlanarImageMut for the yuv crate
        let mut planar = yuv::YuvPlanarImageMut {
            y_plane: yuv::BufferStoreMut::Borrowed(&mut y_bytes),
            y_stride: luma_width as u32,
            u_plane: yuv::BufferStoreMut::Borrowed(&mut u_bytes),
            u_stride: chroma_width as u32,
            v_plane: yuv::BufferStoreMut::Borrowed(&mut v_bytes),
            v_stride: chroma_width as u32,
            width: w as u32,
            height: h as u32,
        };

        let bpp = match color {
            PixelLayout::Rgb8 | PixelLayout::Bgr8 => 3,
            PixelLayout::Rgba8 | PixelLayout::Bgra8 => 4,
            _ => unreachable!(),
        };
        let src_stride = (stride * bpp) as u32;

        let result = match color {
            PixelLayout::Rgb8 => yuv::rgb_to_sharp_yuv420(
                &mut planar,
                image_data,
                src_stride,
                yuv::YuvRange::Limited,
                yuv::YuvStandardMatrix::Bt601,
                yuv::SharpYuvGammaTransfer::Srgb,
            ),
            PixelLayout::Rgba8 => yuv::rgba_to_sharp_yuv420(
                &mut planar,
                image_data,
                src_stride,
                yuv::YuvRange::Limited,
                yuv::YuvStandardMatrix::Bt601,
                yuv::SharpYuvGammaTransfer::Srgb,
            ),
            PixelLayout::Bgr8 => yuv::bgr_to_sharp_yuv420(
                &mut planar,
                image_data,
                src_stride,
                yuv::YuvRange::Limited,
                yuv::YuvStandardMatrix::Bt601,
                yuv::SharpYuvGammaTransfer::Srgb,
            ),
            PixelLayout::Bgra8 => yuv::bgra_to_sharp_yuv420(
                &mut planar,
                image_data,
                src_stride,
                yuv::YuvRange::Limited,
                yuv::YuvStandardMatrix::Bt601,
                yuv::SharpYuvGammaTransfer::Srgb,
            ),
            _ => unreachable!(),
        };

        if result.is_err() {
            // Fall back to standard conversion if sharp YUV fails
            return match color {
                PixelLayout::Rgb8 => convert_image_yuv::<3>(image_data, width, height, stride),
                PixelLayout::Rgba8 => convert_image_yuv::<4>(image_data, width, height, stride),
                PixelLayout::Bgr8 => convert_image_yuv_bgr::<3>(image_data, width, height, stride),
                PixelLayout::Bgra8 => convert_image_yuv_bgr::<4>(image_data, width, height, stride),
                _ => unreachable!(),
            };
        }

        (y_bytes, u_bytes, v_bytes)
    }

    #[cfg(not(feature = "fast-yuv"))]
    {
        // Fall back to standard conversion without the fast-yuv feature
        use crate::encoder::PixelLayout;
        match color {
            PixelLayout::Rgb8 => convert_image_yuv::<3>(image_data, width, height, stride),
            PixelLayout::Rgba8 => convert_image_yuv::<4>(image_data, width, height, stride),
            PixelLayout::Bgr8 => convert_image_yuv_bgr::<3>(image_data, width, height, stride),
            PixelLayout::Bgra8 => convert_image_yuv_bgr::<4>(image_data, width, height, stride),
            PixelLayout::L8 => convert_image_y::<1>(image_data, width, height, stride),
            PixelLayout::La8 => convert_image_y::<2>(image_data, width, height, stride),
            PixelLayout::Yuv420 => unreachable!("sharp YUV should not be called with Yuv420 input"),
        }
    }
}

/// Convert BGR/BGRA image data to YUV420 with macroblock alignment.
///
/// Same as `convert_image_yuv` but reads pixels as B,G,R(,A) instead of R,G,B(,A).
/// BPP=3 for BGR, BPP=4 for BGRA.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn convert_image_yuv_bgr<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Helper to convert BGR pixel slice to RGB-ordered slice for the rgb_to_* functions.
    // We swap B and R so the existing rgb_to_y/u/v functions work correctly.
    #[inline(always)]
    fn bgr_to_rgb(bgr: &[u8]) -> [u8; 4] {
        // Return [R, G, B, ...] from [B, G, R, ...]
        [bgr[2], bgr[1], bgr[0], 0]
    }

    let row_pairs = height / 2;
    let odd_height = height & 1 != 0;
    let col_pairs = width / 2;
    let odd_width = width & 1 != 0;

    for row_pair in 0..row_pairs {
        let src_row1 = row_pair * 2;
        let src_row2 = src_row1 + 1;
        let chroma_row = row_pair;

        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            let rgb1 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col1) * BPP..]);
            let rgb2 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col2) * BPP..]);
            let rgb3 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col1) * BPP..]);
            let rgb4 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col2) * BPP..]);

            y_bytes[src_row1 * luma_width + src_col1] = rgb_to_y(&rgb1);
            y_bytes[src_row1 * luma_width + src_col2] = rgb_to_y(&rgb2);
            y_bytes[src_row2 * luma_width + src_col1] = rgb_to_y(&rgb3);
            y_bytes[src_row2 * luma_width + src_col2] = rgb_to_y(&rgb4);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb2, &rgb3, &rgb4);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb2, &rgb3, &rgb4);
        }

        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            let rgb1 = bgr_to_rgb(&image_data[(src_row1 * stride + src_col) * BPP..]);
            let rgb3 = bgr_to_rgb(&image_data[(src_row2 * stride + src_col) * BPP..]);

            y_bytes[src_row1 * luma_width + src_col] = rgb_to_y(&rgb1);
            y_bytes[src_row2 * luma_width + src_col] = rgb_to_y(&rgb3);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb1, &rgb3, &rgb3);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb1, &rgb3, &rgb3);
        }
    }

    if odd_height {
        let src_row = height - 1;
        let chroma_row = row_pairs;

        for col_pair in 0..col_pairs {
            let src_col1 = col_pair * 2;
            let src_col2 = src_col1 + 1;
            let chroma_col = col_pair;

            let rgb1 = bgr_to_rgb(&image_data[(src_row * stride + src_col1) * BPP..]);
            let rgb2 = bgr_to_rgb(&image_data[(src_row * stride + src_col2) * BPP..]);

            y_bytes[src_row * luma_width + src_col1] = rgb_to_y(&rgb1);
            y_bytes[src_row * luma_width + src_col2] = rgb_to_y(&rgb2);

            u_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_u_avg(&rgb1, &rgb2, &rgb1, &rgb2);
            v_bytes[chroma_row * chroma_width + chroma_col] =
                rgb_to_v_avg(&rgb1, &rgb2, &rgb1, &rgb2);
        }

        if odd_width {
            let src_col = width - 1;
            let chroma_col = col_pairs;

            let rgb = bgr_to_rgb(&image_data[(src_row * stride + src_col) * BPP..]);

            y_bytes[src_row * luma_width + src_col] = rgb_to_y(&rgb);

            u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(&rgb, &rgb, &rgb, &rgb);
            v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(&rgb, &rgb, &rgb, &rgb);
        }
    }

    // Replicate edge pixels to fill macroblock padding (same as convert_image_yuv)
    for y in 0..height {
        let last_y = y_bytes[y * luma_width + width - 1];
        for x in width..luma_width {
            y_bytes[y * luma_width + x] = last_y;
        }
    }
    for y in height..(mb_height * 16) {
        for x in 0..luma_width {
            y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
        }
    }
    let chroma_height = height.div_ceil(2);
    let actual_chroma_width = width.div_ceil(2);
    for y in 0..chroma_height {
        let last_u = u_bytes[y * chroma_width + actual_chroma_width - 1];
        let last_v = v_bytes[y * chroma_width + actual_chroma_width - 1];
        for x in actual_chroma_width..chroma_width {
            u_bytes[y * chroma_width + x] = last_u;
            v_bytes[y * chroma_width + x] = last_v;
        }
    }
    for y in chroma_height..(mb_height * 8) {
        for x in 0..chroma_width {
            u_bytes[y * chroma_width + x] = u_bytes[(chroma_height - 1) * chroma_width + x];
            v_bytes[y * chroma_width + x] = v_bytes[(chroma_height - 1) * chroma_width + x];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

/// Import raw YUV 4:2:0 planes into macroblock-aligned buffers.
///
/// Copies Y/U/V planes with macroblock padding (edge replication).
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) fn import_yuv420_planes(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u16,
    height: u16,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = usize::from(width);
    let h = usize::from(height);
    let mb_width = w.div_ceil(16);
    let mb_height = h.div_ceil(16);
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let y_size = luma_width * 16 * mb_height;
    let chroma_size = chroma_width * 8 * mb_height;

    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);

    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // Copy Y plane with horizontal padding
    for y in 0..h {
        let src_start = y * w;
        let dst_start = y * luma_width;
        y_bytes[dst_start..dst_start + w].copy_from_slice(&y_plane[src_start..src_start + w]);
        let last_y = y_bytes[dst_start + w - 1];
        for x in w..luma_width {
            y_bytes[dst_start + x] = last_y;
        }
    }
    // Vertical padding for Y
    for y in h..(mb_height * 16) {
        let src_row = (h - 1) * luma_width;
        let dst_row = y * luma_width;
        y_bytes.copy_within(src_row..src_row + luma_width, dst_row);
    }

    // Copy U/V planes with horizontal padding
    for y in 0..uv_h {
        let src_start = y * uv_w;
        let dst_start = y * chroma_width;
        u_bytes[dst_start..dst_start + uv_w].copy_from_slice(&u_plane[src_start..src_start + uv_w]);
        v_bytes[dst_start..dst_start + uv_w].copy_from_slice(&v_plane[src_start..src_start + uv_w]);
        let last_u = u_bytes[dst_start + uv_w - 1];
        let last_v = v_bytes[dst_start + uv_w - 1];
        for x in uv_w..chroma_width {
            u_bytes[dst_start + x] = last_u;
            v_bytes[dst_start + x] = last_v;
        }
    }
    // Vertical padding for U/V
    for y in uv_h..(mb_height * 8) {
        let src_row = (uv_h - 1) * chroma_width;
        let dst_row = y * chroma_width;
        u_bytes.copy_within(src_row..src_row + chroma_width, dst_row);
        v_bytes.copy_within(src_row..src_row + chroma_width, dst_row);
    }

    (y_bytes, u_bytes, v_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fancy_grid() {
        #[rustfmt::skip]
        let y_buffer = [
            77, 162, 202, 185,
            28, 13, 199, 182,
            135, 147, 164, 135, 
            66, 27, 171, 130,
        ];

        #[rustfmt::skip]
        let u_buffer = [
            34, 101,
            123, 163
        ];

        #[rustfmt::skip]
        let v_buffer = [
            97, 167,
            149, 23,
        ];

        let mut rgb_buffer = [0u8; 16 * 3];
        fill_rgb_buffer_fancy::<3>(&mut rgb_buffer, &y_buffer, &u_buffer, &v_buffer, 4, 4, 4);

        #[rustfmt::skip]
        let upsampled_u_buffer = [
            34, 51, 84, 101,
            56, 71, 101, 117,
            101, 112, 136, 148,
            123, 133, 153, 163,
        ];

        #[rustfmt::skip]
        let upsampled_v_buffer = [
            97, 115, 150, 167,
            110, 115, 126, 131,
            136, 117, 78, 59,
            149, 118, 55, 23,
        ];

        let mut upsampled_rgb_buffer = [0u8; 16 * 3];
        for (((rgb_val, y), u), v) in upsampled_rgb_buffer
            .chunks_exact_mut(3)
            .zip(y_buffer)
            .zip(upsampled_u_buffer)
            .zip(upsampled_v_buffer)
        {
            rgb_val[0] = yuv_to_r(y, v);
            rgb_val[1] = yuv_to_g(y, u, v);
            rgb_val[2] = yuv_to_b(y, u);
        }

        assert_eq!(rgb_buffer, upsampled_rgb_buffer);
    }

    #[test]
    fn test_yuv_conversions() {
        let (y, u, v) = (203, 40, 42);

        assert_eq!(yuv_to_r(y, v), 80);
        assert_eq!(yuv_to_g(y, u, v), 255);
        assert_eq!(yuv_to_b(y, u), 40);
    }
}
