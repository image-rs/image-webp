//! DCT histogram collection for alpha estimation.
//!
//! This module computes DCT coefficient distributions for macroblock analysis.
//! The alpha value derived from histograms indicates block complexity/compressibility.
//!
//! ## SIMD Optimization Opportunities
//!
//! - `forward_dct_4x4`: 4x4 DCT butterfly operations (multiply-accumulate)
//! - Histogram accumulation with clamping

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::{BPS, DctHistogram, MAX_COEFF_THRESH, VP8_DSP_SCAN};

//------------------------------------------------------------------------------
// Forward DCT 4x4

/// Forward 4x4 DCT on residual (src - pred)
/// Ported from libwebp's FTransform_C
///
/// This is a SIMD optimization target - the butterfly operations
/// can be parallelized across rows/columns.
#[inline]
#[allow(clippy::identity_op)] // 0 + i patterns match libwebp's C code for clarity
pub fn forward_dct_4x4(
    src: &[u8],
    pred: &[u8],
    src_stride: usize,
    pred_stride: usize,
) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    let mut out = [0i16; 16];

    // Horizontal pass: compute differences and first butterfly
    for i in 0..4 {
        let src_row = i * src_stride;
        let pred_row = i * pred_stride;

        let d0 = i32::from(src[src_row]) - i32::from(pred[pred_row]);
        let d1 = i32::from(src[src_row + 1]) - i32::from(pred[pred_row + 1]);
        let d2 = i32::from(src[src_row + 2]) - i32::from(pred[pred_row + 2]);
        let d3 = i32::from(src[src_row + 3]) - i32::from(pred[pred_row + 3]);

        let a0 = d0 + d3;
        let a1 = d1 + d2;
        let a2 = d1 - d2;
        let a3 = d0 - d3;

        tmp[0 + i * 4] = (a0 + a1) * 8;
        tmp[2 + i * 4] = (a0 - a1) * 8;
        tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }

    // Vertical pass
    for i in 0..4 {
        let a0 = tmp[0 + i] + tmp[12 + i];
        let a1 = tmp[4 + i] + tmp[8 + i];
        let a2 = tmp[4 + i] - tmp[8 + i];
        let a3 = tmp[0 + i] - tmp[12 + i];

        out[0 + i] = ((a0 + a1 + 7) >> 4) as i16;
        out[8 + i] = ((a0 - a1 + 7) >> 4) as i16;
        out[4 + i] = (((a2 * 2217 + a3 * 5352 + 12000) >> 16) + if a3 != 0 { 1 } else { 0 }) as i16;
        out[12 + i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16;
    }

    out
}

//------------------------------------------------------------------------------
// Histogram collection

/// Collect DCT histogram for a range of 4x4 blocks
/// Ported from libwebp's CollectHistogram_C / VP8CollectHistogram
///
/// # Arguments
/// * `src` - Source pixels (BPS stride)
/// * `pred` - Prediction pixels (BPS stride)
/// * `start_block` - First block index (in VP8DspScan)
/// * `end_block` - One past last block index
pub fn collect_histogram_bps(
    src: &[u8],
    pred: &[u8],
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    for &scan_off in &VP8_DSP_SCAN[start_block..end_block] {
        let src_off = scan_off;
        let pred_off = scan_off;

        let dct_out = forward_dct_4x4(&src[src_off..], &pred[pred_off..], BPS, BPS);

        for coeff in dct_out.iter() {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            let clipped = v.min(MAX_COEFF_THRESH);
            distribution[clipped] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}

/// Collect histogram with explicit base offsets
/// Used when source and prediction are in different buffer regions
pub fn collect_histogram_with_offset(
    src_buf: &[u8],
    src_base: usize,
    pred_buf: &[u8],
    pred_base: usize,
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::X64V3Token::summon() {
            return collect_histogram_with_offset_sse2(
                token,
                src_buf,
                src_base,
                pred_buf,
                pred_base,
                start_block,
                end_block,
            );
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::summon() {
            return collect_histogram_with_offset_neon(
                token,
                src_buf,
                src_base,
                pred_buf,
                pred_base,
                start_block,
                end_block,
            );
        }
    }
    #[allow(unreachable_code)]
    collect_histogram_with_offset_scalar(
        src_buf,
        src_base,
        pred_buf,
        pred_base,
        start_block,
        end_block,
    )
}

/// Scalar fallback for histogram collection
fn collect_histogram_with_offset_scalar(
    src_buf: &[u8],
    src_base: usize,
    pred_buf: &[u8],
    pred_base: usize,
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    for j in start_block..end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        let dct_out = forward_dct_4x4(&src_buf[src_off..], &pred_buf[pred_off..], BPS, BPS);

        for coeff in dct_out.iter() {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            let clipped = v.min(MAX_COEFF_THRESH);
            distribution[clipped] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}

/// SIMD histogram collection using ftransform2 for paired-block DCT.
///
/// Processes blocks in pairs (adjacent scan offsets differ by 4), using
/// ftransform2_sse2 for double-block DCT. The DCT is the expensive part;
/// histogram scatter is scalar since bin indices are data-dependent.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[archmage::arcane]
fn collect_histogram_with_offset_sse2(
    _token: archmage::X64V3Token,
    src_buf: &[u8],
    src_base: usize,
    pred_buf: &[u8],
    pred_base: usize,
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    // Process blocks in pairs using ftransform2
    let mut j = start_block;
    while j + 1 < end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        // ftransform2 processes two adjacent 4x4 blocks (8 bytes wide per row)
        let mut dct_out = [0i16; 32];
        crate::common::transform_simd_intrinsics::ftransform2_sse2(
            _token,
            &src_buf[src_off..],
            &pred_buf[pred_off..],
            BPS,
            BPS,
            &mut dct_out,
        );

        // Scatter to histogram: abs >> 3, clamp to MAX_COEFF_THRESH
        for &coeff in &dct_out {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            distribution[v.min(MAX_COEFF_THRESH)] += 1;
        }

        j += 2;
    }

    // Handle odd trailing block (shouldn't happen with VP8_DSP_SCAN but be safe)
    if j < end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        let dct_out = forward_dct_4x4(&src_buf[src_off..], &pred_buf[pred_off..], BPS, BPS);

        for &coeff in &dct_out {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            distribution[v.min(MAX_COEFF_THRESH)] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}

/// NEON histogram collection using ftransform2 for paired-block DCT.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[archmage::arcane]
fn collect_histogram_with_offset_neon(
    _token: archmage::NeonToken,
    src_buf: &[u8],
    src_base: usize,
    pred_buf: &[u8],
    pred_base: usize,
    start_block: usize,
    end_block: usize,
) -> DctHistogram {
    let mut distribution = [0u32; MAX_COEFF_THRESH + 1];

    let mut j = start_block;
    while j + 1 < end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        let mut dct_out = [[0i32; 16]; 2];
        crate::common::transform_aarch64::ftransform2_neon(
            _token,
            &src_buf[src_off..],
            &pred_buf[pred_off..],
            BPS,
            BPS,
            &mut dct_out,
        );

        for blk in &dct_out {
            for &coeff in blk {
                let v = ((coeff.unsigned_abs() >> 3) as usize).min(MAX_COEFF_THRESH);
                distribution[v] += 1;
            }
        }

        j += 2;
    }

    if j < end_block {
        let scan_off = VP8_DSP_SCAN[j];
        let src_off = src_base + scan_off;
        let pred_off = pred_base + scan_off;

        let dct_out = forward_dct_4x4(&src_buf[src_off..], &pred_buf[pred_off..], BPS, BPS);

        for &coeff in &dct_out {
            let v = (coeff.unsigned_abs() >> 3) as usize;
            distribution[v.min(MAX_COEFF_THRESH)] += 1;
        }
    }

    DctHistogram::from_distribution(&distribution)
}
