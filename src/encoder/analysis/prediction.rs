//! Prediction functions for analysis pass.
//!
//! These generate intra predictions for luma (16x16) and chroma (8x8) blocks
//! for use in DCT histogram analysis. Only DC and TM modes are used during
//! analysis (libwebp's MAX_INTRA16_MODE = 2).
//!
//! ## SIMD Optimizations
//!
//! - `pred_luma16_tm`, `pred_chroma8_tm`: TrueMotion prediction with SIMD clamping

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::{BPS, C8DC8, C8TM8, I16DC16, I16TM16};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{SimdToken, X64V3Token, arcane};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

//------------------------------------------------------------------------------
// Block fill helpers

/// Fill a block with a constant value (BPS stride)
#[inline]
fn fill_block(dst: &mut [u8], value: u8, size: usize) {
    for y in 0..size {
        for x in 0..size {
            dst[y * BPS + x] = value;
        }
    }
}

/// Vertical prediction: copy top row to all rows
#[inline]
fn vertical_pred(dst: &mut [u8], top: Option<&[u8]>, size: usize) {
    if let Some(top) = top {
        for y in 0..size {
            for x in 0..size {
                dst[y * BPS + x] = top[x];
            }
        }
    } else {
        fill_block(dst, 127, size);
    }
}

/// Horizontal prediction: copy left column to all columns
#[inline]
fn horizontal_pred(dst: &mut [u8], left: Option<&[u8]>, size: usize) {
    if let Some(left) = left {
        for y in 0..size {
            for x in 0..size {
                dst[y * BPS + x] = left[y];
            }
        }
    } else {
        fill_block(dst, 129, size);
    }
}

//------------------------------------------------------------------------------
// Luma 16x16 predictions

/// Generate DC prediction for 16x16 luma block
/// Ported from libwebp's DCMode (size=16, round=16, shift=5)
pub fn pred_luma16_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    let dc_val = match (top, left) {
        (Some(top), Some(left)) => {
            // Both borders: sum all 32 samples
            let dc: u32 = top[..16].iter().map(|&x| u32::from(x)).sum::<u32>()
                + left[..16].iter().map(|&x| u32::from(x)).sum::<u32>();
            ((dc + 16) >> 5) as u8
        }
        (Some(top), None) => {
            // Top only: sum 16 samples, double, then shift by 5
            let mut dc: u32 = top[..16].iter().map(|&x| u32::from(x)).sum();
            dc += dc; // Double
            ((dc + 16) >> 5) as u8
        }
        (None, Some(left)) => {
            // Left only: sum 16 samples, double, then shift by 5
            let mut dc: u32 = left[..16].iter().map(|&x| u32::from(x)).sum();
            dc += dc; // Double
            ((dc + 16) >> 5) as u8
        }
        (None, None) => {
            // No borders: use 128
            0x80u8
        }
    };

    // Fill 16x16 block
    for y in 0..16 {
        for x in 0..16 {
            dst[y * BPS + x] = dc_val;
        }
    }
}

/// Generate TM (TrueMotion) prediction for 16x16 luma block
/// Ported from libwebp's TrueMotion
///
/// left_with_corner\[0\] = top-left corner, left_with_corner\[1..17\] = left pixels
pub fn pred_luma16_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    match (left_with_corner, top) {
        (Some(left), Some(top)) => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                pred_luma16_tm_dispatch(dst, left, top);
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                pred_luma16_tm_scalar(dst, left, top);
            }
        }
        (Some(left), None) => {
            horizontal_pred(dst, Some(&left[1..17]), 16);
        }
        (None, Some(top)) => {
            vertical_pred(dst, Some(top), 16);
        }
        (None, None) => {
            fill_block(dst, 129, 16);
        }
    }
}

/// Scalar TrueMotion for 16x16 luma.
#[inline]
fn pred_luma16_tm_scalar(dst: &mut [u8], left: &[u8], top: &[u8]) {
    let tl = i32::from(left[0]);
    for y in 0..16 {
        let l = i32::from(left[1 + y]);
        for x in 0..16 {
            let t = i32::from(top[x]);
            dst[y * BPS + x] = (l + t - tl).clamp(0, 255) as u8;
        }
    }
}

/// SIMD dispatch for TrueMotion 16x16.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn pred_luma16_tm_dispatch(dst: &mut [u8], left: &[u8], top: &[u8]) {
    if let Some(token) = X64V3Token::summon() {
        pred_luma16_tm_sse2(token, dst, left, top);
    } else {
        pred_luma16_tm_scalar(dst, left, top);
    }
}

/// SSE2 TrueMotion: Process 16 pixels per row.
/// Formula: dst[y][x] = clamp(left[y] + top[x] - tl, 0, 255)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
fn pred_luma16_tm_sse2(_token: X64V3Token, dst: &mut [u8], left: &[u8], top: &[u8]) {
    let zero = _mm_setzero_si128();

    // Load top row (16 bytes)
    let top_arr = <&[u8; 16]>::try_from(&top[..16]).unwrap();
    let top_vec = simd_mem::_mm_loadu_si128(top_arr);
    // Unpack to i16: top_lo = top[0..8], top_hi = top[8..16]
    let top_lo = _mm_unpacklo_epi8(top_vec, zero);
    let top_hi = _mm_unpackhi_epi8(top_vec, zero);

    // Broadcast top-left corner
    let tl = _mm_set1_epi16(i16::from(left[0]));

    // Precompute (top - tl) for each column
    let top_minus_tl_lo = _mm_sub_epi16(top_lo, tl);
    let top_minus_tl_hi = _mm_sub_epi16(top_hi, tl);

    for y in 0..16 {
        // Broadcast left[y+1] to all 8 positions
        let l = _mm_set1_epi16(i16::from(left[1 + y]));

        // Compute left + (top - tl) = left + top - tl
        let sum_lo = _mm_add_epi16(l, top_minus_tl_lo);
        let sum_hi = _mm_add_epi16(l, top_minus_tl_hi);

        // Clamp to [0, 255] and pack to u8
        // packus saturates i16 to [0, 255] automatically
        let packed = _mm_packus_epi16(sum_lo, sum_hi);

        // Store 16 bytes to dst row
        let dst_arr = <&mut [u8; 16]>::try_from(&mut dst[y * BPS..y * BPS + 16]).unwrap();
        simd_mem::_mm_storeu_si128(dst_arr, packed);
    }
}

/// Generate all I16 predictions into yuv_p buffer
/// Ported from libwebp's VP8EncPredLuma16
///
/// left_with_corner: full y_left array where \[0\]=corner, \[1..17\]=left pixels
pub fn make_luma16_preds(yuv_p: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    // DC prediction at I16DC16 (only needs left pixels, not corner)
    let left_only = left_with_corner.map(|l| &l[1..17]);
    pred_luma16_dc(&mut yuv_p[I16DC16..], left_only, top);

    // TM prediction at I16TM16 (needs corner at [0] and left pixels at [1..17])
    pred_luma16_tm(&mut yuv_p[I16TM16..], left_with_corner, top);

    // Note: V and H predictions not implemented since we only test DC and TM (MAX_INTRA16_MODE=2)
}

//------------------------------------------------------------------------------
// Chroma 8x8 predictions

/// Generate DC prediction for 8x8 chroma block into BPS-strided buffer
/// Ported from libwebp's DCMode (size=8, round=8, shift=4)
pub fn pred_chroma8_dc(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    let dc_val = match (top, left) {
        (Some(top), Some(left)) => {
            // Both borders: sum all 16 samples
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(top[i]);
                dc += u32::from(left[i]);
            }
            ((dc + 8) >> 4) as u8
        }
        (Some(top), None) => {
            // Top only: sum 8 samples, double, then shift by 4
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(top[i]);
            }
            dc += dc; // Double
            ((dc + 8) >> 4) as u8
        }
        (None, Some(left)) => {
            // Left only: sum 8 samples, double, then shift by 4
            let mut dc: u32 = 0;
            for i in 0..8 {
                dc += u32::from(left[i]);
            }
            dc += dc; // Double
            ((dc + 8) >> 4) as u8
        }
        (None, None) => {
            // No borders: use 128
            0x80u8
        }
    };

    // Fill 8x8 block
    for y in 0..8 {
        for x in 0..8 {
            dst[y * BPS + x] = dc_val;
        }
    }
}

/// Generate TM prediction for 8x8 chroma block
/// Ported from libwebp's TrueMotion
///
/// left_with_corner: \[0\]=corner, \[1..9\]=left pixels (matching u_left/v_left layout)
pub fn pred_chroma8_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>) {
    match (left_with_corner, top) {
        (Some(left), Some(top)) => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                pred_chroma8_tm_dispatch(dst, left, top);
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                pred_chroma8_tm_scalar(dst, left, top);
            }
        }
        (Some(left), None) => {
            horizontal_pred(dst, Some(&left[1..9]), 8);
        }
        (None, Some(top)) => {
            vertical_pred(dst, Some(top), 8);
        }
        (None, None) => {
            fill_block(dst, 129, 8);
        }
    }
}

/// Scalar TrueMotion for 8x8 chroma.
#[inline]
fn pred_chroma8_tm_scalar(dst: &mut [u8], left: &[u8], top: &[u8]) {
    let tl = i32::from(left[0]);
    for y in 0..8 {
        let l = i32::from(left[1 + y]);
        for x in 0..8 {
            let t = i32::from(top[x]);
            dst[y * BPS + x] = (l + t - tl).clamp(0, 255) as u8;
        }
    }
}

/// SIMD dispatch for TrueMotion 8x8.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn pred_chroma8_tm_dispatch(dst: &mut [u8], left: &[u8], top: &[u8]) {
    if let Some(token) = X64V3Token::summon() {
        pred_chroma8_tm_sse2(token, dst, left, top);
    } else {
        pred_chroma8_tm_scalar(dst, left, top);
    }
}

/// SSE2 TrueMotion for 8x8 chroma: Process 8 pixels per row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
fn pred_chroma8_tm_sse2(_token: X64V3Token, dst: &mut [u8], left: &[u8], top: &[u8]) {
    let zero = _mm_setzero_si128();

    // Load 8 bytes of top using 128-bit load (safe API requires [u8; 16])
    // We pad with zeros but only use the first 8 bytes
    let mut top_padded = [0u8; 16];
    top_padded[..8].copy_from_slice(&top[..8]);
    let top_bytes = simd_mem::_mm_loadu_si128(&top_padded);
    // Unpack to i16
    let top_i16 = _mm_unpacklo_epi8(top_bytes, zero);

    // Broadcast top-left corner
    let tl = _mm_set1_epi16(i16::from(left[0]));

    // Precompute (top - tl)
    let top_minus_tl = _mm_sub_epi16(top_i16, tl);

    for y in 0..8 {
        // Broadcast left[y+1]
        let l = _mm_set1_epi16(i16::from(left[1 + y]));

        // Compute left + (top - tl)
        let sum = _mm_add_epi16(l, top_minus_tl);

        // Clamp and pack: packus saturates to [0, 255]
        let packed = _mm_packus_epi16(sum, zero);

        // Store 8 bytes using 128-bit store, but we only care about low 8
        // Use a temporary buffer to store safely
        let mut tmp = [0u8; 16];
        simd_mem::_mm_storeu_si128(&mut tmp, packed);
        dst[y * BPS..y * BPS + 8].copy_from_slice(&tmp[..8]);
    }
}

/// Generate all chroma predictions into yuv_p buffer
/// Ported from libwebp's VP8EncPredChroma8
///
/// Note: In libwebp, U and V predictions are interleaved in the buffer.
/// u_left_with_corner/v_left_with_corner: \[0\]=corner, \[1..9\]=left pixels
/// uv_top: U at [0..8], V at [8..16]
pub fn make_chroma8_preds(
    yuv_p: &mut [u8],
    u_left_with_corner: Option<&[u8]>,
    v_left_with_corner: Option<&[u8]>,
    uv_top: Option<&[u8]>,
) {
    // Extract U and V top borders if available
    let (u_top, v_top) = if let Some(uv_top) = uv_top {
        (Some(&uv_top[0..8]), Some(&uv_top[8..16]))
    } else {
        (None, None)
    };

    // For DC prediction, extract just the left pixels (no corner)
    let u_left_only = u_left_with_corner.map(|l| &l[1..9]);
    let v_left_only = v_left_with_corner.map(|l| &l[1..9]);

    // DC prediction for U at C8DC8
    pred_chroma8_dc(&mut yuv_p[C8DC8..], u_left_only, u_top);

    // DC prediction for V at C8DC8 + 8 (V is 8 pixels to the right of U)
    pred_chroma8_dc(&mut yuv_p[C8DC8 + 8..], v_left_only, v_top);

    // TM prediction for U at C8TM8 (needs corner)
    pred_chroma8_tm(&mut yuv_p[C8TM8..], u_left_with_corner, u_top);

    // TM prediction for V at C8TM8 + 8
    pred_chroma8_tm(&mut yuv_p[C8TM8 + 8..], v_left_with_corner, v_top);
}
