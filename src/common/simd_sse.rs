//! SIMD-accelerated Sum of Squared Errors (SSE) computation
//!
//! This module provides optimized SSE computation for block distortion
//! measurement, similar to libwebp's SSE4x4_SSE2.
//!
//! Uses archmage for safe SIMD intrinsics with token-based CPU feature verification.

#![allow(clippy::needless_range_loop)]

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use core::arch::x86_64::*;

/// Compute Sum of Squared Errors between two 4x4 blocks
///
/// Both blocks are stored in row-major order as 16 bytes.
#[cfg(feature = "simd")]
#[inline]
pub fn sse4x4(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    // Scalar fallback for non-x86 or when no SIMD available
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse4x4_scalar(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            sse4x4_entry(token, a, b)
        } else {
            sse4x4_scalar(a, b)
        }
    }
}

/// Arcane entry point for sse4x4 dispatch (calls #[rite] inner)
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn sse4x4_entry(_token: X64V3Token, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    sse4x4_sse2(_token, a, b)
}

/// Scalar SSE computation
#[inline]
#[allow(dead_code)]
pub fn sse4x4_scalar(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let diff = i32::from(a[i]) - i32::from(b[i]);
        sum += (diff * diff) as u32;
    }
    sum
}

/// SSE2 implementation of 4x4 block SSE
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
pub(crate) fn sse4x4_sse2(_token: X64V3Token, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let zero = _mm_setzero_si128();

    // Load all 16 bytes at once
    let a_bytes = simd_mem::_mm_loadu_si128(a);
    let b_bytes = simd_mem::_mm_loadu_si128(b);

    // Unpack to 16-bit: low 8 bytes
    let a_lo = _mm_unpacklo_epi8(a_bytes, zero);
    let b_lo = _mm_unpacklo_epi8(b_bytes, zero);

    // Unpack to 16-bit: high 8 bytes
    let a_hi = _mm_unpackhi_epi8(a_bytes, zero);
    let b_hi = _mm_unpackhi_epi8(b_bytes, zero);

    // Subtract
    let d_lo = _mm_sub_epi16(a_lo, b_lo);
    let d_hi = _mm_sub_epi16(a_hi, b_hi);

    // Square and accumulate using madd (pairs of i16 multiplied and summed to i32)
    let sq_lo = _mm_madd_epi16(d_lo, d_lo);
    let sq_hi = _mm_madd_epi16(d_hi, d_hi);

    // Sum all 8 i32 values
    let sum = _mm_add_epi32(sq_lo, sq_hi);

    // Horizontal sum: sum[0]+sum[1]+sum[2]+sum[3]
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b10_11_00_01)); // swap pairs
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10)); // swap halves

    _mm_cvtsi128_si32(sum) as u32
}

/// Compute SSE between a source block and a reconstructed block (pred + residual)
///
/// This is used for RD scoring where we need SSE(src, pred + idct(quantized))
#[cfg(feature = "simd")]
#[inline]
pub fn sse4x4_with_residual(src: &[u8; 16], pred: &[u8; 16], residual: &[i32; 16]) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse4x4_with_residual_scalar(src, pred, residual)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            sse4x4_with_residual_entry(token, src, pred, residual)
        } else {
            sse4x4_with_residual_scalar(src, pred, residual)
        }
    }
}

/// Scalar implementation of SSE with residual
#[inline]
#[allow(dead_code)]
pub fn sse4x4_with_residual_scalar(src: &[u8; 16], pred: &[u8; 16], residual: &[i32; 16]) -> u32 {
    let mut sum = 0u32;
    for i in 0..16 {
        let reconstructed = (i32::from(pred[i]) + residual[i]).clamp(0, 255);
        let diff = i32::from(src[i]) - reconstructed;
        sum += (diff * diff) as u32;
    }
    sum
}

/// Arcane entry point for dispatch
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn sse4x4_with_residual_entry(
    _token: X64V3Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    sse4x4_with_residual_sse2(_token, src, pred, residual)
}

/// SSE2 implementation of SSE with residual
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
pub(crate) fn sse4x4_with_residual_sse2(
    _token: X64V3Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    let zero = _mm_setzero_si128();
    let max_255 = _mm_set1_epi16(255);

    // Load source and prediction
    let src_bytes = simd_mem::_mm_loadu_si128(src);
    let pred_bytes = simd_mem::_mm_loadu_si128(pred);

    // Unpack to 16-bit
    let src_lo = _mm_unpacklo_epi8(src_bytes, zero);
    let src_hi = _mm_unpackhi_epi8(src_bytes, zero);
    let pred_lo = _mm_unpacklo_epi8(pred_bytes, zero);
    let pred_hi = _mm_unpackhi_epi8(pred_bytes, zero);

    // Load residuals (4 i32 values at a time) and pack to i16
    let res0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[0..4]).unwrap());
    let res1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[4..8]).unwrap());
    let res2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[8..12]).unwrap());
    let res3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&residual[12..16]).unwrap());

    // Pack i32 to i16 (saturating)
    let res_lo = _mm_packs_epi32(res0, res1);
    let res_hi = _mm_packs_epi32(res2, res3);

    // Reconstruct: pred + residual, clamped to [0, 255]
    let rec_lo = _mm_add_epi16(pred_lo, res_lo);
    let rec_hi = _mm_add_epi16(pred_hi, res_hi);

    // Clamp to [0, 255]
    let rec_lo = _mm_max_epi16(rec_lo, zero);
    let rec_lo = _mm_min_epi16(rec_lo, max_255);
    let rec_hi = _mm_max_epi16(rec_hi, zero);
    let rec_hi = _mm_min_epi16(rec_hi, max_255);

    // Compute difference
    let d_lo = _mm_sub_epi16(src_lo, rec_lo);
    let d_hi = _mm_sub_epi16(src_hi, rec_hi);

    // Square and accumulate
    let sq_lo = _mm_madd_epi16(d_lo, d_lo);
    let sq_hi = _mm_madd_epi16(d_hi, d_hi);

    // Sum all values
    let sum = _mm_add_epi32(sq_lo, sq_hi);
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));

    _mm_cvtsi128_si32(sum) as u32
}

use super::prediction::{CHROMA_BLOCK_SIZE, CHROMA_STRIDE, LUMA_BLOCK_SIZE, LUMA_STRIDE};

/// Compute SSE for a 16x16 luma block between source and bordered prediction buffer
///
/// Source is in a contiguous row-major array with `src_width` stride.
/// Prediction is in a bordered buffer with LUMA_STRIDE stride and 1-pixel border.
#[cfg(feature = "simd")]
#[inline]
pub fn sse_16x16_luma(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse_16x16_luma_scalar(src_y, src_width, mbx, mby, pred)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            sse_16x16_luma_entry(token, src_y, src_width, mbx, mby, pred)
        } else {
            sse_16x16_luma_scalar(src_y, src_width, mbx, mby, pred)
        }
    }
}

/// Scalar implementation for 16x16 luma SSE
#[inline]
#[allow(dead_code)]
pub fn sse_16x16_luma_scalar(
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 16 * src_width + mbx * 16;

    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;

        for x in 0..16 {
            let diff = i32::from(src_y[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// Arcane entry point for dispatch
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn sse_16x16_luma_entry(
    _token: X64V3Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    sse_16x16_luma_sse2(_token, src_y, src_width, mbx, mby, pred)
}

/// SSE2 implementation of 16x16 luma SSE
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
pub(crate) fn sse_16x16_luma_sse2(
    _token: X64V3Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let zero = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();

    let src_base = mby * 16 * src_width + mbx * 16;

    for y in 0..16 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * LUMA_STRIDE + 1;

        // Load 16 bytes from source and prediction
        let src_bytes =
            simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&src_y[src_row..][..16]).unwrap());
        let pred_bytes =
            simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&pred[pred_row..][..16]).unwrap());

        // Unpack to 16-bit
        let src_lo = _mm_unpacklo_epi8(src_bytes, zero);
        let src_hi = _mm_unpackhi_epi8(src_bytes, zero);
        let pred_lo = _mm_unpacklo_epi8(pred_bytes, zero);
        let pred_hi = _mm_unpackhi_epi8(pred_bytes, zero);

        // Subtract
        let d_lo = _mm_sub_epi16(src_lo, pred_lo);
        let d_hi = _mm_sub_epi16(src_hi, pred_hi);

        // Square and accumulate
        let sq_lo = _mm_madd_epi16(d_lo, d_lo);
        let sq_hi = _mm_madd_epi16(d_hi, d_hi);

        total = _mm_add_epi32(total, sq_lo);
        total = _mm_add_epi32(total, sq_hi);
    }

    // Horizontal sum
    let sum = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
    _mm_cvtsi128_si32(sum) as u32
}

/// Compute SSE for an 8x8 chroma block between source and bordered prediction buffer
#[cfg(feature = "simd")]
#[inline]
pub fn sse_8x8_chroma(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        sse_8x8_chroma_scalar(src_uv, src_width, mbx, mby, pred)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            sse_8x8_chroma_entry(token, src_uv, src_width, mbx, mby, pred)
        } else {
            sse_8x8_chroma_scalar(src_uv, src_width, mbx, mby, pred)
        }
    }
}

/// Scalar implementation for 8x8 chroma SSE
#[inline]
#[allow(dead_code)]
pub fn sse_8x8_chroma_scalar(
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let mut sse = 0u32;
    let src_base = mby * 8 * src_width + mbx * 8;

    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;

        for x in 0..8 {
            let diff = i32::from(src_uv[src_row + x]) - i32::from(pred[pred_row + x]);
            sse += (diff * diff) as u32;
        }
    }
    sse
}

/// SSE2 implementation of 8x8 chroma SSE
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn sse_8x8_chroma_entry(
    _token: X64V3Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    sse_8x8_chroma_sse2(_token, src_uv, src_width, mbx, mby, pred)
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
pub(crate) fn sse_8x8_chroma_sse2(
    _token: X64V3Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let zero = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();

    let src_base = mby * 8 * src_width + mbx * 8;

    for y in 0..8 {
        let src_row = src_base + y * src_width;
        let pred_row = (y + 1) * CHROMA_STRIDE + 1;

        // Load 8 bytes from source and prediction (lower half of xmm register)
        let src_bytes =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src_uv[src_row..][..8]).unwrap());
        let pred_bytes =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pred[pred_row..][..8]).unwrap());

        // Unpack to 16-bit (only low 8 bytes are valid)
        let src_16 = _mm_unpacklo_epi8(src_bytes, zero);
        let pred_16 = _mm_unpacklo_epi8(pred_bytes, zero);

        // Subtract
        let diff = _mm_sub_epi16(src_16, pred_16);

        // Square and accumulate
        let sq = _mm_madd_epi16(diff, diff);
        total = _mm_add_epi32(total, sq);
    }

    // Horizontal sum
    let sum = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0b10_11_00_01));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_00_11_10));
    _mm_cvtsi128_si32(sum) as u32
}

//------------------------------------------------------------------------------
// TTransform - Spectral distortion helper for TDisto calculation
//
// Computes a Hadamard-like transform followed by weighted absolute value sum.
// This is used for perceptual distortion measurement.

/// Compute the TTransform for spectral distortion calculation (scalar version)
/// Returns weighted sum of absolute transform coefficients
#[inline]
#[allow(dead_code)]
pub fn t_transform_scalar(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let mut tmp = [0i32; 16];

    // Horizontal pass
    for i in 0..4 {
        let row = i * stride;
        let a0 = i32::from(input[row]) + i32::from(input[row + 2]);
        let a1 = i32::from(input[row + 1]) + i32::from(input[row + 3]);
        let a2 = i32::from(input[row + 1]) - i32::from(input[row + 3]);
        let a3 = i32::from(input[row]) - i32::from(input[row + 2]);
        tmp[i * 4] = a0 + a1;
        tmp[i * 4 + 1] = a3 + a2;
        tmp[i * 4 + 2] = a3 - a2;
        tmp[i * 4 + 3] = a0 - a1;
    }

    // Vertical pass with weighting
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;

        sum += i32::from(w[i]) * b0.abs();
        sum += i32::from(w[4 + i]) * b1.abs();
        sum += i32::from(w[8 + i]) * b2.abs();
        sum += i32::from(w[12 + i]) * b3.abs();
    }
    sum
}

/// SIMD-accelerated TTransform using SSE2
#[cfg(feature = "simd")]
#[inline]
pub fn t_transform(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        t_transform_scalar(input, stride, w)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            t_transform_entry(token, input, stride, w)
        } else {
            t_transform_scalar(input, stride, w)
        }
    }
}

/// SSE2 implementation of TTransform - Hadamard transform with weighted abs sum
/// Horizontal pass done in SIMD, vertical pass extracted for simplicity
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn t_transform_entry(_token: X64V3Token, input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    t_transform_sse2(_token, input, stride, w)
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
fn t_transform_sse2(_token: X64V3Token, input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let zero = _mm_setzero_si128();

    // Load 4 rows of 4 bytes each
    let row0 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[0..4]).unwrap());
    let row1 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride..][..4]).unwrap());
    let row2 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride * 2..][..4]).unwrap());
    let row3 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&input[stride * 3..][..4]).unwrap());

    // Expand u8 to i16: each row is [x0, x1, x2, x3, 0, 0, 0, 0]
    let in0 = _mm_unpacklo_epi8(row0, zero);
    let in1 = _mm_unpacklo_epi8(row1, zero);
    let in2 = _mm_unpacklo_epi8(row2, zero);
    let in3 = _mm_unpacklo_epi8(row3, zero);

    // Pack rows 0,1 and 2,3 into single registers for horizontal Hadamard
    // in01 = [r0x0, r0x1, r0x2, r0x3, r1x0, r1x1, r1x2, r1x3]
    // in23 = [r2x0, r2x1, r2x2, r2x3, r3x0, r3x1, r3x2, r3x3]
    let in01 = _mm_unpacklo_epi64(in0, in1);
    let in23 = _mm_unpacklo_epi64(in2, in3);

    // Horizontal Hadamard: for each row [x0,x1,x2,x3], compute:
    // a0=x0+x2, a1=x1+x3, a2=x1-x3, a3=x0-x2
    // out = [a0+a1, a3+a2, a3-a2, a0-a1]
    //
    // Shuffle to get [x0,x2,x0,x2, x1,x3,x1,x3] arrangement:
    // Use _mm_shufflelo_epi16 and _mm_shufflehi_epi16
    // in01 = [r0x0, r0x1, r0x2, r0x3, r1x0, r1x1, r1x2, r1x3]
    // Want:  [r0x0, r0x2, r0x1, r0x3, r1x0, r1x2, r1x1, r1x3] (indices 0,2,1,3)

    let shuf_02_13 = _mm_shufflelo_epi16(_mm_shufflehi_epi16(in01, 0b11_01_10_00), 0b11_01_10_00);
    let shuf_02_13_23 =
        _mm_shufflelo_epi16(_mm_shufflehi_epi16(in23, 0b11_01_10_00), 0b11_01_10_00);

    // Now: shuf_02_13 = [r0x0, r0x2, r0x1, r0x3, r1x0, r1x2, r1x1, r1x3]
    // Add adjacent pairs: [r0x0+r0x2, r0x1+r0x3, r1x0+r1x2, r1x1+r1x3] = [a0, a1, a0, a1]
    let add_pairs_01 = _mm_madd_epi16(shuf_02_13, _mm_set1_epi16(1));
    let add_pairs_23 = _mm_madd_epi16(shuf_02_13_23, _mm_set1_epi16(1));

    // Sub pairs: [r0x0-r0x2, r0x1-r0x3, r1x0-r1x2, r1x1-r1x3] = [a3, a2, a3, a2]
    // Use madd with [1, -1] pattern
    let sub_pattern = _mm_set_epi16(1, -1, 1, -1, 1, -1, 1, -1);
    let sub_pairs_01 = _mm_madd_epi16(shuf_02_13, sub_pattern);
    let sub_pairs_23 = _mm_madd_epi16(shuf_02_13_23, sub_pattern);

    // add_pairs_01 = [r0_a0, r0_a1, r1_a0, r1_a1] (i32)
    // sub_pairs_01 = [r0_a3, r0_a2, r1_a3, r1_a2] (i32)
    // Final horizontal: [a0+a1, a3+a2, a3-a2, a0-a1] for each row

    // Extract to arrays for clarity (vertical pass needs transpose anyway)
    let mut ap01 = [0i32; 4];
    let mut sp01 = [0i32; 4];
    let mut ap23 = [0i32; 4];
    let mut sp23 = [0i32; 4];
    simd_mem::_mm_storeu_si128(&mut ap01, add_pairs_01);
    simd_mem::_mm_storeu_si128(&mut sp01, sub_pairs_01);
    simd_mem::_mm_storeu_si128(&mut ap23, add_pairs_23);
    simd_mem::_mm_storeu_si128(&mut sp23, sub_pairs_23);

    // tmp[row][col] after horizontal Hadamard
    // row 0: [a0+a1, a3+a2, a3-a2, a0-a1] = [ap01[0]+ap01[1], sp01[0]+sp01[1], sp01[0]-sp01[1], ap01[0]-ap01[1]]
    // Wait, that's not right. ap01 = [r0_a0, r0_a1, r1_a0, r1_a1]
    // Row 0: a0=ap01[0], a1=ap01[1], a3=sp01[0], a2=sp01[1]
    let mut tmp = [[0i32; 4]; 4];
    // Row 0
    tmp[0][0] = ap01[0] + ap01[1]; // a0 + a1
    tmp[0][1] = sp01[0] + sp01[1]; // a3 + a2
    tmp[0][2] = sp01[0] - sp01[1]; // a3 - a2
    tmp[0][3] = ap01[0] - ap01[1]; // a0 - a1
    // Row 1
    tmp[1][0] = ap01[2] + ap01[3];
    tmp[1][1] = sp01[2] + sp01[3];
    tmp[1][2] = sp01[2] - sp01[3];
    tmp[1][3] = ap01[2] - ap01[3];
    // Row 2
    tmp[2][0] = ap23[0] + ap23[1];
    tmp[2][1] = sp23[0] + sp23[1];
    tmp[2][2] = sp23[0] - sp23[1];
    tmp[2][3] = ap23[0] - ap23[1];
    // Row 3
    tmp[3][0] = ap23[2] + ap23[3];
    tmp[3][1] = sp23[2] + sp23[3];
    tmp[3][2] = sp23[2] - sp23[3];
    tmp[3][3] = ap23[2] - ap23[3];

    // Vertical pass with weighting (same as scalar)
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = tmp[0][i] + tmp[2][i];
        let a1 = tmp[1][i] + tmp[3][i];
        let a2 = tmp[1][i] - tmp[3][i];
        let a3 = tmp[0][i] - tmp[2][i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;

        sum += w[i] as i32 * b0.abs();
        sum += w[4 + i] as i32 * b1.abs();
        sum += w[8 + i] as i32 * b2.abs();
        sum += w[12 + i] as i32 * b3.abs();
    }
    sum
}

/// TDisto for two 4x4 blocks - computes |TTransform(a) - TTransform(b)| >> 5
/// Uses fused SIMD implementation processing both blocks in parallel
#[cfg(feature = "simd")]
#[inline]
pub fn tdisto_4x4_fused(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        let sum_a = t_transform_scalar(a, stride, w);
        let sum_b = t_transform_scalar(b, stride, w);
        (sum_b - sum_a).abs() >> 5
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            tdisto_4x4_fused_entry(token, a, b, stride, w)
        } else {
            let sum_a = t_transform_scalar(a, stride, w);
            let sum_b = t_transform_scalar(b, stride, w);
            (sum_b - sum_a).abs() >> 5
        }
    }
}

/// Arcane entry point for dispatch
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn tdisto_4x4_fused_entry(
    _token: X64V3Token,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    tdisto_4x4_fused_sse2(_token, a, b, stride, w)
}

/// SSE2 fused TDisto - processes both blocks in parallel like libwebp's TTransform_SSE2
/// Based on libwebp's enc_sse2.c TTransform_SSE2 and Disto4x4_SSE2
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
pub(crate) fn tdisto_4x4_fused_sse2(
    _token: X64V3Token,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    let zero = _mm_setzero_si128();

    // Load and combine inputs (4 bytes per row per block)
    let a0 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&a[0..4]).unwrap());
    let a1 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&a[stride..][..4]).unwrap());
    let a2 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&a[stride * 2..][..4]).unwrap());
    let a3 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&a[stride * 3..][..4]).unwrap());
    let b0 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&b[0..4]).unwrap());
    let b1 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&b[stride..][..4]).unwrap());
    let b2 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&b[stride * 2..][..4]).unwrap());
    let b3 = simd_mem::_mm_loadu_si32(<&[u8; 4]>::try_from(&b[stride * 3..][..4]).unwrap());

    // Combine A and B: [a0,a1,a2,a3, b0,b1,b2,b3] for each row
    let ab0 = _mm_unpacklo_epi32(a0, b0);
    let ab1 = _mm_unpacklo_epi32(a1, b1);
    let ab2 = _mm_unpacklo_epi32(a2, b2);
    let ab3 = _mm_unpacklo_epi32(a3, b3);

    // Expand to i16
    let mut tmp0 = _mm_unpacklo_epi8(ab0, zero);
    let mut tmp1 = _mm_unpacklo_epi8(ab1, zero);
    let mut tmp2 = _mm_unpacklo_epi8(ab2, zero);
    let mut tmp3 = _mm_unpacklo_epi8(ab3, zero);
    // tmp0 = [a00 a01 a02 a03   b00 b01 b02 b03] as i16
    // tmp1 = [a10 a11 a12 a13   b10 b11 b12 b13]
    // tmp2 = [a20 a21 a22 a23   b20 b21 b22 b23]
    // tmp3 = [a30 a31 a32 a33   b30 b31 b32 b33]

    // Vertical Hadamard (processes both blocks simultaneously)
    // Since weights are symmetric, vertical-first gives same result as horizontal-first
    {
        let va0 = _mm_add_epi16(tmp0, tmp2);
        let va1 = _mm_add_epi16(tmp1, tmp3);
        let va2 = _mm_sub_epi16(tmp1, tmp3);
        let va3 = _mm_sub_epi16(tmp0, tmp2);
        let vb0 = _mm_add_epi16(va0, va1);
        let vb1 = _mm_add_epi16(va3, va2);
        let vb2 = _mm_sub_epi16(va3, va2);
        let vb3 = _mm_sub_epi16(va0, va1);

        // Transpose both 4x4 blocks (libwebp's VP8Transpose_2_4x4_16b)
        let tr0_0 = _mm_unpacklo_epi16(vb0, vb1);
        let tr0_1 = _mm_unpacklo_epi16(vb2, vb3);
        let tr0_2 = _mm_unpackhi_epi16(vb0, vb1);
        let tr0_3 = _mm_unpackhi_epi16(vb2, vb3);
        let tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
        let tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
        let tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
        let tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);
        tmp0 = _mm_unpacklo_epi64(tr1_0, tr1_1);
        tmp1 = _mm_unpackhi_epi64(tr1_0, tr1_1);
        tmp2 = _mm_unpacklo_epi64(tr1_2, tr1_3);
        tmp3 = _mm_unpackhi_epi64(tr1_2, tr1_3);
        // tmp0 = [a00 a10 a20 a30   b00 b10 b20 b30] (col0)
        // tmp1 = [a01 a11 a21 a31   b01 b11 b21 b31] (col1)
        // tmp2 = [a02 a12 a22 a32   b02 b12 b22 b32] (col2)
        // tmp3 = [a03 a13 a23 a33   b03 b13 b23 b33] (col3)
    }

    // Horizontal Hadamard (processes both blocks simultaneously)
    let ha0 = _mm_add_epi16(tmp0, tmp2);
    let ha1 = _mm_add_epi16(tmp1, tmp3);
    let ha2 = _mm_sub_epi16(tmp1, tmp3);
    let ha3 = _mm_sub_epi16(tmp0, tmp2);
    let hb0 = _mm_add_epi16(ha0, ha1);
    let hb1 = _mm_add_epi16(ha3, ha2);
    let hb2 = _mm_sub_epi16(ha3, ha2);
    let hb3 = _mm_sub_epi16(ha0, ha1);
    // hb0-hb3 contain transformed coefficients for both A and B

    // Separate transforms of A and B
    // A's coeffs are in low 4 i16 of each hbN, B's in high 4
    let a_01 = _mm_unpacklo_epi64(hb0, hb1); // A rows 0,1 (8 coeffs)
    let a_23 = _mm_unpacklo_epi64(hb2, hb3); // A rows 2,3 (8 coeffs)
    let b_01 = _mm_unpackhi_epi64(hb0, hb1); // B rows 0,1 (8 coeffs)
    let b_23 = _mm_unpackhi_epi64(hb2, hb3); // B rows 2,3 (8 coeffs)

    // Compute absolute values
    let a_abs_01 = _mm_max_epi16(a_01, _mm_sub_epi16(zero, a_01));
    let a_abs_23 = _mm_max_epi16(a_23, _mm_sub_epi16(zero, a_23));
    let b_abs_01 = _mm_max_epi16(b_01, _mm_sub_epi16(zero, b_01));
    let b_abs_23 = _mm_max_epi16(b_23, _mm_sub_epi16(zero, b_23));

    // Load weights
    let w_0 = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&w[0..8]).unwrap());
    let w_8 = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&w[8..16]).unwrap());

    // Compute weighted sums using madd (multiply-add pairs)
    let a_prod_01 = _mm_madd_epi16(a_abs_01, w_0);
    let a_prod_23 = _mm_madd_epi16(a_abs_23, w_8);
    let b_prod_01 = _mm_madd_epi16(b_abs_01, w_0);
    let b_prod_23 = _mm_madd_epi16(b_abs_23, w_8);

    // Sum up A's weighted transform
    let a_sum_01_23 = _mm_add_epi32(a_prod_01, a_prod_23);
    let a_hi = _mm_shuffle_epi32(a_sum_01_23, 0b10_11_00_01);
    let a_sum_2 = _mm_add_epi32(a_sum_01_23, a_hi);
    let a_final = _mm_shuffle_epi32(a_sum_2, 0b01_00_11_10);
    let a_sum_3 = _mm_add_epi32(a_sum_2, a_final);
    let sum_a = _mm_cvtsi128_si32(a_sum_3);

    // Sum up B's weighted transform
    let b_sum_01_23 = _mm_add_epi32(b_prod_01, b_prod_23);
    let b_hi = _mm_shuffle_epi32(b_sum_01_23, 0b10_11_00_01);
    let b_sum_2 = _mm_add_epi32(b_sum_01_23, b_hi);
    let b_final = _mm_shuffle_epi32(b_sum_2, 0b01_00_11_10);
    let b_sum_3 = _mm_add_epi32(b_sum_2, b_final);
    let sum_b = _mm_cvtsi128_si32(b_sum_3);

    (sum_b - sum_a).abs() >> 5
}

//------------------------------------------------------------------------------
// Residual Cost SIMD Precomputation
//
// Port of libwebp's GetResidualCost_SSE2 - precomputes absolute values,
// contexts, and clamped levels using SIMD for faster coefficient cost calculation.

/// Maximum variable level for context clamping (matches libwebp's MAX_VARIABLE_LEVEL)
const MAX_VARIABLE_LEVEL: u8 = 67;

/// Precomputed coefficient data for fast residual cost calculation
#[derive(Clone, Default)]
pub struct PrecomputedCoeffs {
    /// Clamped levels: min(abs(coeff), 67) for each coefficient
    pub levels: [u8; 16],
    /// Context values: min(abs(coeff), 2) for each coefficient
    pub ctxs: [u8; 16],
    /// Full absolute values (u16 to handle full range)
    pub abs_levels: [u16; 16],
}

/// Precompute coefficient data using SIMD (levels, contexts, abs_levels)
/// This is the hot path optimization from libwebp's GetResidualCost_SSE2.
#[cfg(feature = "simd")]
#[inline]
pub fn precompute_coeffs(coeffs: &[i32; 16]) -> PrecomputedCoeffs {
    #[cfg(not(target_arch = "x86_64"))]
    {
        precompute_coeffs_scalar(coeffs)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            precompute_coeffs_entry(token, coeffs)
        } else {
            precompute_coeffs_scalar(coeffs)
        }
    }
}

/// Scalar implementation of coefficient precomputation
#[inline]
#[allow(dead_code)]
pub fn precompute_coeffs_scalar(coeffs: &[i32; 16]) -> PrecomputedCoeffs {
    let mut result = PrecomputedCoeffs::default();
    for i in 0..16 {
        let abs_val = coeffs[i].unsigned_abs() as u16;
        result.abs_levels[i] = abs_val;
        result.ctxs[i] = abs_val.min(2) as u8;
        result.levels[i] = abs_val.min(MAX_VARIABLE_LEVEL as u16) as u8;
    }
    result
}

/// SSE2 implementation of coefficient precomputation
/// Matches libwebp's GetResidualCost_SSE2 precomputation block
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn precompute_coeffs_entry(_token: X64V3Token, coeffs: &[i32; 16]) -> PrecomputedCoeffs {
    precompute_coeffs_sse2(_token, coeffs)
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
fn precompute_coeffs_sse2(_token: X64V3Token, coeffs: &[i32; 16]) -> PrecomputedCoeffs {
    let zero = _mm_setzero_si128();
    let cst_2 = _mm_set1_epi8(2);
    let cst_67 = _mm_set1_epi8(MAX_VARIABLE_LEVEL as i8);

    // Load coefficients as i32, convert to i16
    // libwebp works with i16 coefficients directly, we need to convert from i32
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    // Pack i32 to i16 (saturating)
    let c0 = _mm_packs_epi32(c0_32, c1_32); // coeffs[0..8] as i16
    let c1 = _mm_packs_epi32(c2_32, c3_32); // coeffs[8..16] as i16

    // Compute absolute values: abs(v) = max(v, -v)
    let d0 = _mm_sub_epi16(zero, c0);
    let d1 = _mm_sub_epi16(zero, c1);
    let e0 = _mm_max_epi16(c0, d0); // abs(c0)
    let e1 = _mm_max_epi16(c1, d1); // abs(c1)

    // Pack to bytes for context and level computation
    let f = _mm_packs_epi16(e0, e1);

    // Context = min(abs_val, 2)
    let g = _mm_min_epu8(f, cst_2);

    // Clamped level = min(abs_val, 67)
    let h = _mm_min_epu8(f, cst_67);

    let mut result = PrecomputedCoeffs::default();

    // Store contexts and levels
    simd_mem::_mm_storeu_si128(&mut result.ctxs, g);
    simd_mem::_mm_storeu_si128(&mut result.levels, h);

    // Store absolute values as u16
    simd_mem::_mm_storeu_si128(
        <&mut [u16; 8]>::try_from(&mut result.abs_levels[0..8]).unwrap(),
        e0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u16; 8]>::try_from(&mut result.abs_levels[8..16]).unwrap(),
        e1,
    );

    result
}

/// Find the last non-zero coefficient using SIMD
/// Returns -1 if all coefficients are zero
#[cfg(feature = "simd")]
#[inline]
pub fn find_last_nonzero(coeffs: &[i32; 16], first: usize) -> i32 {
    #[cfg(not(target_arch = "x86_64"))]
    {
        find_last_nonzero_scalar(coeffs, first)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = X64V3Token::summon() {
            find_last_nonzero_entry(token, coeffs, first)
        } else {
            find_last_nonzero_scalar(coeffs, first)
        }
    }
}

/// Scalar implementation of find_last_nonzero
#[inline]
#[allow(dead_code)]
pub fn find_last_nonzero_scalar(coeffs: &[i32; 16], first: usize) -> i32 {
    for i in (first..16).rev() {
        if coeffs[i] != 0 {
            return i as i32;
        }
    }
    -1
}

/// SSE2 implementation of find_last_nonzero
/// Uses SIMD comparison and bitmask to find last non-zero
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn find_last_nonzero_entry(_token: X64V3Token, coeffs: &[i32; 16], first: usize) -> i32 {
    find_last_nonzero_sse2(_token, coeffs, first)
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
fn find_last_nonzero_sse2(_token: X64V3Token, coeffs: &[i32; 16], first: usize) -> i32 {
    let zero = _mm_setzero_si128();

    // Load and pack coefficients to bytes for comparison
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let c0 = _mm_packs_epi32(c0_32, c1_32);
    let c1 = _mm_packs_epi32(c2_32, c3_32);
    let m0 = _mm_packs_epi16(c0, c1);

    // Compare with zero
    let m1 = _mm_cmpeq_epi8(m0, zero);

    // Get bitmask: bit is 1 if byte is zero, 0 if non-zero
    let mask = _mm_movemask_epi8(m1) as u32;

    // Invert to get positions of non-zero bytes
    let nonzero_mask = 0x0000ffff ^ mask;

    // Create mask for positions >= first
    let first_mask = !((1u32 << first) - 1) & 0xffff;
    let valid_mask = nonzero_mask & first_mask;

    if valid_mask == 0 {
        -1
    } else {
        // Find highest set bit
        (31 - valid_mask.leading_zeros()) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse4x4_scalar() {
        let a = [10u8; 16];
        let b = [12u8; 16];
        // Each pixel differs by 2, so SSE = 16 * 4 = 64
        assert_eq!(sse4x4_scalar(&a, &b), 64);
    }

    #[test]
    fn test_sse4x4_identical() {
        let a = [100u8; 16];
        assert_eq!(sse4x4_scalar(&a, &a), 0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_sse4x4_simd_matches_scalar() {
        let a: [u8; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let b: [u8; 16] = [
            12, 18, 33, 38, 55, 58, 73, 78, 93, 98, 113, 118, 133, 138, 153, 158,
        ];

        let scalar = sse4x4_scalar(&a, &b);
        let simd = sse4x4(&a, &b);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse4x4_with_residual_scalar() {
        let src = [100u8; 16];
        let pred = [90u8; 16];
        let residual = [10i32; 16]; // pred + residual = 100, so SSE = 0
        assert_eq!(sse4x4_with_residual_scalar(&src, &pred, &residual), 0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_sse4x4_with_residual_simd_matches_scalar() {
        let src: [u8; 16] = [
            100, 110, 120, 130, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 1,
        ];
        let pred: [u8; 16] = [
            95, 105, 115, 125, 85, 75, 65, 55, 45, 35, 25, 15, 5, 2, 1, 0,
        ];
        let residual: [i32; 16] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 2, 1];

        let scalar = sse4x4_with_residual_scalar(&src, &pred, &residual);
        let simd = sse4x4_with_residual(&src, &pred, &residual);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse_16x16_luma_scalar() {
        // Create a 32x32 source buffer (2x2 macroblocks)
        let src_width = 32;
        let mut src_y = vec![100u8; 32 * 32];
        // Set macroblock (0,0) to values around 100
        for y in 0..16 {
            for x in 0..16 {
                src_y[y * src_width + x] = 100 + (x as u8);
            }
        }

        // Create prediction buffer with border
        let mut pred = [0u8; LUMA_BLOCK_SIZE];
        for y in 0..16 {
            for x in 0..16 {
                pred[(y + 1) * LUMA_STRIDE + 1 + x] = 102 + (x as u8); // diff of 2
            }
        }

        // Each pixel differs by 2, so SSE = 256 * 4 = 1024
        let sse = sse_16x16_luma_scalar(&src_y, src_width, 0, 0, &pred);
        assert_eq!(sse, 1024);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_sse_16x16_luma_simd_matches_scalar() {
        let src_width = 32;
        let mut src_y = vec![0u8; 32 * 32];
        // Fill with varying values
        for y in 0..16 {
            for x in 0..16 {
                src_y[y * src_width + x] = ((y * 16 + x) % 256) as u8;
            }
        }

        let mut pred = [0u8; LUMA_BLOCK_SIZE];
        for y in 0..16 {
            for x in 0..16 {
                pred[(y + 1) * LUMA_STRIDE + 1 + x] = ((y * 16 + x + 5) % 256) as u8;
            }
        }

        let scalar = sse_16x16_luma_scalar(&src_y, src_width, 0, 0, &pred);
        let simd = sse_16x16_luma(&src_y, src_width, 0, 0, &pred);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_sse_8x8_chroma_scalar() {
        let src_width = 16;
        let mut src_uv = vec![128u8; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                src_uv[y * src_width + x] = 128 + (x as u8);
            }
        }

        let mut pred = [0u8; CHROMA_BLOCK_SIZE];
        for y in 0..8 {
            for x in 0..8 {
                pred[(y + 1) * CHROMA_STRIDE + 1 + x] = 130 + (x as u8); // diff of 2
            }
        }

        // 64 pixels * 4 = 256
        let sse = sse_8x8_chroma_scalar(&src_uv, src_width, 0, 0, &pred);
        assert_eq!(sse, 256);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_sse_8x8_chroma_simd_matches_scalar() {
        let src_width = 16;
        let mut src_uv = vec![0u8; 16 * 16];
        for y in 0..8 {
            for x in 0..8 {
                src_uv[y * src_width + x] = ((y * 8 + x * 3) % 256) as u8;
            }
        }

        let mut pred = [0u8; CHROMA_BLOCK_SIZE];
        for y in 0..8 {
            for x in 0..8 {
                pred[(y + 1) * CHROMA_STRIDE + 1 + x] = ((y * 8 + x * 3 + 7) % 256) as u8;
            }
        }

        let scalar = sse_8x8_chroma_scalar(&src_uv, src_width, 0, 0, &pred);
        let simd = sse_8x8_chroma(&src_uv, src_width, 0, 0, &pred);
        assert_eq!(scalar, simd);
    }

    #[test]
    fn test_t_transform_scalar_basic() {
        // Create a simple 4x4 block with stride 16
        let mut input = [0u8; 64]; // 4 rows * 16 stride
        // Fill with a simple gradient
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = ((y * 4 + x) * 10) as u8;
            }
        }

        // Uniform weights
        let weights: [u16; 16] = [1; 16];

        let result = t_transform_scalar(&input, 16, &weights);
        // Just verify it produces a non-zero result for a non-uniform block
        assert!(result > 0);
    }

    #[test]
    fn test_t_transform_scalar_uniform() {
        // A uniform block should produce non-zero only at DC (index 0)
        let mut input = [128u8; 64];
        // Set actual 4x4 block to uniform values
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = 100;
            }
        }

        // Uniform weights
        let weights: [u16; 16] = [1; 16];

        let result = t_transform_scalar(&input, 16, &weights);
        // For uniform input, DC should be 4*100*4 = 1600, others 0
        // DC after both passes: sum of all = 400, weighted by 1 = 400
        assert!(result > 0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_t_transform_simd_matches_scalar() {
        // Create a varied 4x4 block
        let mut input = [0u8; 64];
        for y in 0..4 {
            for x in 0..4 {
                input[y * 16 + x] = ((y * 37 + x * 23 + 50) % 256) as u8;
            }
        }

        // Varied weights
        let weights: [u16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let scalar = t_transform_scalar(&input, 16, &weights);
        let simd = t_transform(&input, 16, &weights);
        assert_eq!(
            scalar, simd,
            "SIMD t_transform should match scalar: scalar={}, simd={}",
            scalar, simd
        );
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_t_transform_simd_matches_scalar_varied() {
        // Test with different strides and values
        for stride in [4, 8, 16, 32] {
            let mut input = vec![0u8; 4 * stride];
            for y in 0..4 {
                for x in 0..4 {
                    input[y * stride + x] = ((y * 53 + x * 41 + 17) % 256) as u8;
                }
            }

            let weights: [u16; 16] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1, 1];

            let scalar = t_transform_scalar(&input, stride, &weights);
            let simd = t_transform(&input, stride, &weights);
            assert_eq!(
                scalar, simd,
                "Mismatch at stride {}: scalar={}, simd={}",
                stride, scalar, simd
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_tdisto_4x4_fused_matches_scalar() {
        // Use symmetric weights like VP8_WEIGHT_Y (required for vertical-first approach)
        // Non-symmetric weights would give different results since we do vertical-first
        #[rustfmt::skip]
        let weights: [u16; 16] = [
            38, 32, 20,  9,
            32, 28, 17,  7,
            20, 17, 10,  4,
             9,  7,  4,  2,
        ];

        // Test with different strides and values
        for stride in [4, 8, 16, 32] {
            let mut a = vec![0u8; 4 * stride];
            let mut b = vec![0u8; 4 * stride];
            for y in 0..4 {
                for x in 0..4 {
                    a[y * stride + x] = ((y * 53 + x * 41 + 17) % 256) as u8;
                    b[y * stride + x] = ((y * 37 + x * 29 + 11) % 256) as u8;
                }
            }

            // Compute with two separate t_transforms (scalar reference)
            let sum_a = t_transform_scalar(&a, stride, &weights);
            let sum_b = t_transform_scalar(&b, stride, &weights);
            let scalar_result = (sum_b - sum_a).abs() >> 5;

            // Compute with fused function
            let fused_result = tdisto_4x4_fused(&a, &b, stride, &weights);

            assert_eq!(
                scalar_result, fused_result,
                "Mismatch at stride {}: scalar={}, fused={}",
                stride, scalar_result, fused_result
            );
        }

        // Test identical blocks (should be 0)
        let same: [u8; 16] = [128; 16];
        let result = tdisto_4x4_fused(&same, &same, 4, &weights);
        assert_eq!(result, 0, "Identical blocks should have 0 distortion");
    }

    #[test]
    fn test_precompute_coeffs_scalar() {
        let coeffs: [i32; 16] = [0, -1, 2, -3, 4, -5, 100, -200, 0, 0, 67, 68, 1, 2, 0, 0];
        let result = precompute_coeffs_scalar(&coeffs);

        // Check abs_levels
        assert_eq!(result.abs_levels[0], 0);
        assert_eq!(result.abs_levels[1], 1);
        assert_eq!(result.abs_levels[6], 100);
        assert_eq!(result.abs_levels[7], 200);

        // Check contexts (min(abs, 2))
        assert_eq!(result.ctxs[0], 0);
        assert_eq!(result.ctxs[1], 1);
        assert_eq!(result.ctxs[2], 2);
        assert_eq!(result.ctxs[6], 2); // 100 clamped to 2

        // Check levels (min(abs, 67))
        assert_eq!(result.levels[6], 67); // 100 clamped to 67
        assert_eq!(result.levels[7], 67); // 200 clamped to 67
        assert_eq!(result.levels[10], 67); // exactly 67
        assert_eq!(result.levels[11], 67); // 68 clamped to 67
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_precompute_coeffs_simd_matches_scalar() {
        let coeffs: [i32; 16] = [0, -1, 2, -3, 4, -5, 100, -200, 0, 0, 67, 68, 1, 2, -50, 30];

        let scalar = precompute_coeffs_scalar(&coeffs);
        let simd = precompute_coeffs(&coeffs);

        for i in 0..16 {
            assert_eq!(
                scalar.abs_levels[i], simd.abs_levels[i],
                "abs_levels mismatch at {}: scalar={}, simd={}",
                i, scalar.abs_levels[i], simd.abs_levels[i]
            );
            assert_eq!(
                scalar.ctxs[i], simd.ctxs[i],
                "ctxs mismatch at {}: scalar={}, simd={}",
                i, scalar.ctxs[i], simd.ctxs[i]
            );
            assert_eq!(
                scalar.levels[i], simd.levels[i],
                "levels mismatch at {}: scalar={}, simd={}",
                i, scalar.levels[i], simd.levels[i]
            );
        }
    }

    #[test]
    fn test_find_last_nonzero_scalar() {
        // All zeros
        let coeffs = [0i32; 16];
        assert_eq!(find_last_nonzero_scalar(&coeffs, 0), -1);

        // Single non-zero at end
        let mut coeffs = [0i32; 16];
        coeffs[15] = 1;
        assert_eq!(find_last_nonzero_scalar(&coeffs, 0), 15);

        // Multiple non-zeros
        let mut coeffs = [0i32; 16];
        coeffs[5] = 1;
        coeffs[10] = -2;
        assert_eq!(find_last_nonzero_scalar(&coeffs, 0), 10);

        // With first offset
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        coeffs[5] = 2;
        assert_eq!(find_last_nonzero_scalar(&coeffs, 1), 5);
        assert_eq!(find_last_nonzero_scalar(&coeffs, 6), -1);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_find_last_nonzero_simd_matches_scalar() {
        // Test various patterns
        let test_cases: Vec<[i32; 16]> = vec![
            [0; 16],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 2, 0, -3, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ];

        for (idx, coeffs) in test_cases.iter().enumerate() {
            for first in 0..16 {
                let scalar = find_last_nonzero_scalar(coeffs, first);
                let simd = find_last_nonzero(coeffs, first);
                assert_eq!(
                    scalar, simd,
                    "Mismatch for test case {} with first={}: scalar={}, simd={}",
                    idx, first, scalar, simd
                );
            }
        }
    }
}
