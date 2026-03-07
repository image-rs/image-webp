//! WASM SIMD128 implementations for encoder distortion and utility functions.
//!
//! Ported from the NEON versions in simd_neon.rs using core::arch::wasm32 intrinsics.

use archmage::{Wasm128Token, arcane, rite};

use core::arch::wasm32::*;

use super::prediction::{CHROMA_BLOCK_SIZE, CHROMA_STRIDE, LUMA_BLOCK_SIZE, LUMA_STRIDE};

// =============================================================================
// Load/store helpers (safe, using lane constructors)
// =============================================================================

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> v128 {
    u8x16(
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
        a[14], a[15],
    )
}

#[inline(always)]
fn load_u8x8_low(a: &[u8; 8]) -> v128 {
    u8x16(
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], 0, 0, 0, 0, 0, 0, 0, 0,
    )
}

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> v128 {
    i32x4(a[0], a[1], a[2], a[3])
}

#[inline(always)]
fn load_u32x4(a: &[u32; 4]) -> v128 {
    u32x4(a[0], a[1], a[2], a[3])
}

#[inline(always)]
fn load_u16x8(a: &[u16; 8]) -> v128 {
    u16x8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
}

#[inline(always)]
fn store_i32x4(out: &mut [i32; 4], v: v128) {
    out[0] = i32x4_extract_lane::<0>(v);
    out[1] = i32x4_extract_lane::<1>(v);
    out[2] = i32x4_extract_lane::<2>(v);
    out[3] = i32x4_extract_lane::<3>(v);
}

// =============================================================================
// Arithmetic helpers
// =============================================================================

/// Horizontal sum of 4 i32 lanes
#[inline(always)]
fn hsum_i32x4(v: v128) -> i32 {
    let hi = i32x4_shuffle::<2, 3, 0, 1>(v, v);
    let sum = i32x4_add(v, hi);
    let hi2 = i32x4_shuffle::<1, 0, 2, 3>(sum, sum);
    let final_sum = i32x4_add(sum, hi2);
    i32x4_extract_lane::<0>(final_sum)
}

/// Horizontal sum of 8 i16 lanes (treating as unsigned, returns i32)
#[inline(always)]
fn hsum_abs_i16x8(v: v128) -> i32 {
    let abs = i16x8_abs(v);
    // Use dot product with ones to sum pairs, then hsum i32
    let ones = i16x8_splat(1);
    let pair_sums = i32x4_dot_i16x8(abs, ones);
    hsum_i32x4(pair_sums)
}

/// Compute SSE between two u8x16 vectors, returning accumulated i32x4
#[inline(always)]
fn sse_u8x16_acc(a: v128, b: v128) -> v128 {
    let mx = u8x16_max(a, b);
    let mn = u8x16_min(a, b);
    let abd = u8x16_sub(mx, mn);
    // Extend to i16 and use dot product for sum-of-squares
    let lo = u16x8_extend_low_u8x16(abd);
    let hi = u16x8_extend_high_u8x16(abd);
    let dot_lo = i32x4_dot_i16x8(lo, lo);
    let dot_hi = i32x4_dot_i16x8(hi, hi);
    i32x4_add(dot_lo, dot_hi)
}

// =============================================================================
// SSE (Sum of Squared Errors) functions
// =============================================================================

/// SSE between two 4x4 blocks (16 bytes each)
#[rite]
pub(crate) fn sse4x4_wasm(_token: Wasm128Token, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let a_vec = load_u8x16(a);
    let b_vec = load_u8x16(b);
    hsum_i32x4(sse_u8x16_acc(a_vec, b_vec)) as u32
}

/// Entry shim for sse4x4_wasm
#[arcane]
pub(crate) fn sse4x4_wasm_entry(_token: Wasm128Token, a: &[u8; 16], b: &[u8; 16]) -> u32 {
    sse4x4_wasm(_token, a, b)
}

/// SSE with fused prediction + residual reconstruction
#[rite]
pub(crate) fn sse4x4_with_residual_wasm(
    _token: Wasm128Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    let src_vec = load_u8x16(src);
    let pred_vec = load_u8x16(pred);

    // Widen prediction to i16
    let pred_lo = u16x8_extend_low_u8x16(pred_vec);
    let pred_hi = u16x8_extend_high_u8x16(pred_vec);

    // Load residuals as i32x4, narrow to i16x8
    let r0 = load_i32x4(<&[i32; 4]>::try_from(&residual[0..4]).unwrap());
    let r1 = load_i32x4(<&[i32; 4]>::try_from(&residual[4..8]).unwrap());
    let r2 = load_i32x4(<&[i32; 4]>::try_from(&residual[8..12]).unwrap());
    let r3 = load_i32x4(<&[i32; 4]>::try_from(&residual[12..16]).unwrap());
    let res_lo = i16x8_narrow_i32x4(r0, r1);
    let res_hi = i16x8_narrow_i32x4(r2, r3);

    // Reconstruct: pred + residual, clamp to [0, 255]
    let rec_lo = i16x8_add(pred_lo, res_lo);
    let rec_hi = i16x8_add(pred_hi, res_hi);
    let rec = u8x16_narrow_i16x8(rec_lo, rec_hi);

    // SSE between src and reconstructed
    hsum_i32x4(sse_u8x16_acc(src_vec, rec)) as u32
}

/// Entry shim for sse4x4_with_residual_wasm
#[arcane]
pub(crate) fn sse4x4_with_residual_wasm_entry(
    _token: Wasm128Token,
    src: &[u8; 16],
    pred: &[u8; 16],
    residual: &[i32; 16],
) -> u32 {
    sse4x4_with_residual_wasm(_token, src, pred, residual)
}

/// SSE for 16x16 luma macroblock
#[rite]
pub(crate) fn sse_16x16_luma_wasm(
    _token: Wasm128Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    let mut acc = i32x4_splat(0);
    let src_base = mby * 16 * src_width + mbx * 16;

    for row in 0..16 {
        let src_off = src_base + row * src_width;
        let pred_off = (1 + row) * LUMA_STRIDE + 1;
        let src_row = <&[u8; 16]>::try_from(&src_y[src_off..src_off + 16]).unwrap();
        let pred_row = <&[u8; 16]>::try_from(&pred[pred_off..pred_off + 16]).unwrap();

        let s = load_u8x16(src_row);
        let p = load_u8x16(pred_row);
        acc = i32x4_add(acc, sse_u8x16_acc(s, p));
    }

    hsum_i32x4(acc) as u32
}

/// Entry shim for sse_16x16_luma_wasm
#[arcane]
pub(crate) fn sse_16x16_luma_wasm_entry(
    _token: Wasm128Token,
    src_y: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; LUMA_BLOCK_SIZE],
) -> u32 {
    sse_16x16_luma_wasm(_token, src_y, src_width, mbx, mby, pred)
}

/// SSE for 8x8 chroma block
#[rite]
pub(crate) fn sse_8x8_chroma_wasm(
    _token: Wasm128Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    let chroma_width = src_width / 2;
    let mut acc = i32x4_splat(0);
    let src_base = mby * 8 * chroma_width + mbx * 8;

    for row in 0..8 {
        let src_off = src_base + row * chroma_width;
        let pred_off = (1 + row) * CHROMA_STRIDE + 1;

        let src_row = <&[u8; 8]>::try_from(&src_uv[src_off..src_off + 8]).unwrap();
        let pred_row = <&[u8; 8]>::try_from(&pred[pred_off..pred_off + 8]).unwrap();

        let s = load_u8x8_low(src_row);
        let p = load_u8x8_low(pred_row);

        // Only low 8 bytes contain data; use extend_low for 8 differences
        let mx = u8x16_max(s, p);
        let mn = u8x16_min(s, p);
        let abd = u8x16_sub(mx, mn);
        let lo = u16x8_extend_low_u8x16(abd);
        let dot = i32x4_dot_i16x8(lo, lo);
        acc = i32x4_add(acc, dot);
    }

    hsum_i32x4(acc) as u32
}

/// Entry shim for sse_8x8_chroma_wasm
#[arcane]
pub(crate) fn sse_8x8_chroma_wasm_entry(
    _token: Wasm128Token,
    src_uv: &[u8],
    src_width: usize,
    mbx: usize,
    mby: usize,
    pred: &[u8; CHROMA_BLOCK_SIZE],
) -> u32 {
    sse_8x8_chroma_wasm(_token, src_uv, src_width, mbx, mby, pred)
}

// =============================================================================
// Spectral distortion (TDisto / Hadamard transform)
// =============================================================================

/// Fused TDisto for two 4x4 blocks: |weighted_hadamard(b) - weighted_hadamard(a)| >> 5
#[rite]
pub(crate) fn tdisto_4x4_fused_wasm(
    _token: Wasm128Token,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    // Load 4 bytes per row for both blocks, combined as [a0..a3, b0..b3] per row
    let load_row = |off: usize| -> v128 {
        i16x8(
            a[off] as i16,
            a[off + 1] as i16,
            a[off + 2] as i16,
            a[off + 3] as i16,
            b[off] as i16,
            b[off + 1] as i16,
            b[off + 2] as i16,
            b[off + 3] as i16,
        )
    };

    let mut tmp0 = load_row(0);
    let mut tmp1 = load_row(stride);
    let mut tmp2 = load_row(stride * 2);
    let mut tmp3 = load_row(stride * 3);

    // Vertical Hadamard
    {
        let va0 = i16x8_add(tmp0, tmp2);
        let va1 = i16x8_add(tmp1, tmp3);
        let va2 = i16x8_sub(tmp1, tmp3);
        let va3 = i16x8_sub(tmp0, tmp2);
        let vb0 = i16x8_add(va0, va1);
        let vb1 = i16x8_add(va3, va2);
        let vb2 = i16x8_sub(va3, va2);
        let vb3 = i16x8_sub(va0, va1);

        // Transpose both 4x4 blocks using interleave operations
        // Step 1: interleave 16-bit pairs
        let t01_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(vb0, vb1);
        let t01_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(vb0, vb1);
        let t23_lo = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(vb2, vb3);
        let t23_hi = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(vb2, vb3);

        // Step 2: interleave 32-bit pairs
        tmp0 = i32x4_shuffle::<0, 4, 1, 5>(t01_lo, t23_lo);
        tmp1 = i32x4_shuffle::<2, 6, 3, 7>(t01_lo, t23_lo);
        tmp2 = i32x4_shuffle::<0, 4, 1, 5>(t01_hi, t23_hi);
        tmp3 = i32x4_shuffle::<2, 6, 3, 7>(t01_hi, t23_hi);
    }

    // Horizontal Hadamard
    let ha0 = i16x8_add(tmp0, tmp2);
    let ha1 = i16x8_add(tmp1, tmp3);
    let ha2 = i16x8_sub(tmp1, tmp3);
    let ha3 = i16x8_sub(tmp0, tmp2);
    let hb0 = i16x8_add(ha0, ha1);
    let hb1 = i16x8_add(ha3, ha2);
    let hb2 = i16x8_sub(ha3, ha2);
    let hb3 = i16x8_sub(ha0, ha1);

    // Separate A (low 4 lanes) and B (high 4 lanes), take absolute values
    let a_01 = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(hb0, hb1);
    let a_23 = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(hb2, hb3);
    let b_01 = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(hb0, hb1);
    let b_23 = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(hb2, hb3);

    let a_abs_01 = i16x8_abs(a_01);
    let a_abs_23 = i16x8_abs(a_23);
    let b_abs_01 = i16x8_abs(b_01);
    let b_abs_23 = i16x8_abs(b_23);

    // Load weights
    let w_0 = load_u16x8(<&[u16; 8]>::try_from(&w[0..8]).unwrap());
    let w_8 = load_u16x8(<&[u16; 8]>::try_from(&w[8..16]).unwrap());

    // Weighted multiply-accumulate using extending multiply
    let a_prod_0 = i32x4_extmul_low_i16x8(a_abs_01, w_0);
    let a_prod_1 = i32x4_extmul_high_i16x8(a_abs_01, w_0);
    let a_prod_2 = i32x4_extmul_low_i16x8(a_abs_23, w_8);
    let a_prod_3 = i32x4_extmul_high_i16x8(a_abs_23, w_8);

    let b_prod_0 = i32x4_extmul_low_i16x8(b_abs_01, w_0);
    let b_prod_1 = i32x4_extmul_high_i16x8(b_abs_01, w_0);
    let b_prod_2 = i32x4_extmul_low_i16x8(b_abs_23, w_8);
    let b_prod_3 = i32x4_extmul_high_i16x8(b_abs_23, w_8);

    let a_sum = i32x4_add(i32x4_add(a_prod_0, a_prod_1), i32x4_add(a_prod_2, a_prod_3));
    let b_sum = i32x4_add(i32x4_add(b_prod_0, b_prod_1), i32x4_add(b_prod_2, b_prod_3));

    let sum_a = hsum_i32x4(a_sum);
    let sum_b = hsum_i32x4(b_sum);

    (sum_b - sum_a).abs() >> 5
}

/// Entry shim for tdisto_4x4_fused_wasm
#[arcane]
pub(crate) fn tdisto_4x4_fused_wasm_entry(
    _token: Wasm128Token,
    a: &[u8],
    b: &[u8],
    stride: usize,
    w: &[u16; 16],
) -> i32 {
    tdisto_4x4_fused_wasm(_token, a, b, stride, w)
}

// =============================================================================
// Flatness check functions
// =============================================================================

/// Check if a 16x16 source block is all one color
#[rite]
pub(crate) fn is_flat_source_16_wasm(_token: Wasm128Token, src: &[u8], stride: usize) -> bool {
    let first = u8x16_splat(src[0]);
    let mut all_eq = u8x16_splat(0xff);

    for row in 0..16 {
        let off = row * stride;
        let row_data = load_u8x16(<&[u8; 16]>::try_from(&src[off..off + 16]).unwrap());
        let eq = u8x16_eq(row_data, first);
        all_eq = v128_and(all_eq, eq);
    }

    // Check if all lanes are 0xFF using bitmask
    u8x16_bitmask(all_eq) == 0xFFFF
}

/// Entry shim for is_flat_source_16_wasm
#[arcane]
pub(crate) fn is_flat_source_16_wasm_entry(
    _token: Wasm128Token,
    src: &[u8],
    stride: usize,
) -> bool {
    is_flat_source_16_wasm(_token, src, stride)
}

/// Check if coefficients are "flat" (few non-zero AC coefficients)
#[rite]
pub(crate) fn is_flat_coeffs_wasm(
    _token: Wasm128Token,
    levels: &[i16],
    num_blocks: usize,
    thresh: i32,
) -> bool {
    let zero = i16x8_splat(0);
    let one = u16x8_splat(1);
    let mut count = 0i32;

    for block in 0..num_blocks {
        let off = block * 16;
        if off + 16 > levels.len() {
            break;
        }
        let v0 = i16x8(
            levels[off],
            levels[off + 1],
            levels[off + 2],
            levels[off + 3],
            levels[off + 4],
            levels[off + 5],
            levels[off + 6],
            levels[off + 7],
        );
        let v1 = i16x8(
            levels[off + 8],
            levels[off + 9],
            levels[off + 10],
            levels[off + 11],
            levels[off + 12],
            levels[off + 13],
            levels[off + 14],
            levels[off + 15],
        );

        // Count non-zero lanes: eq gives 0xFFFF for equal, invert for non-equal
        let ne0 = v128_not(i16x8_eq(v0, zero));
        let ne1 = v128_not(i16x8_eq(v1, zero));
        let c0 = v128_and(ne0, one);
        let c1 = v128_and(ne1, one);
        let total = i16x8_add(c0, c1);
        // Sum all 8 u16 values using dot with ones
        let pair_sums = i32x4_dot_i16x8(total, i16x8_splat(1));
        let block_count = hsum_i32x4(pair_sums);
        // Skip DC (position 0): subtract 1 if DC was non-zero
        let dc_nz = if levels[off] != 0 { 1i32 } else { 0 };
        count += block_count - dc_nz;
    }

    count <= thresh
}

/// Entry shim for is_flat_coeffs_wasm
#[arcane]
pub(crate) fn is_flat_coeffs_wasm_entry(
    _token: Wasm128Token,
    levels: &[i16],
    num_blocks: usize,
    thresh: i32,
) -> bool {
    is_flat_coeffs_wasm(_token, levels, num_blocks, thresh)
}

// =============================================================================
// Quantization functions
// =============================================================================

use crate::encoder::quantize::{QFIX, VP8Matrix};
use crate::encoder::tables::MAX_LEVEL;

/// WASM quantize block: quantizes i32[16] coefficients in-place.
/// Returns true if any coefficient is non-zero.
#[rite]
pub(crate) fn quantize_block_wasm(
    _token: Wasm128Token,
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    let max_coeff = i16x8_splat(MAX_LEVEL as i16);

    // Pack i32 coefficients to i16
    let c0 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in0 = i16x8_narrow_i32x4(c0, c1);
    let in8 = i16x8_narrow_i32x4(c2, c3);

    // Extract sign and absolute value
    let sign0 = i16x8_shr(in0, 15);
    let sign8 = i16x8_shr(in8, 15);
    let mut coeff0 = i16x8_abs(in0);
    let mut coeff8 = i16x8_abs(in8);

    // Add sharpen
    if use_sharpen {
        let sh0 = load_u16x8(<&[u16; 8]>::try_from(&matrix.sharpen[0..8]).unwrap());
        let sh8 = load_u16x8(<&[u16; 8]>::try_from(&matrix.sharpen[8..16]).unwrap());
        coeff0 = i16x8_add(coeff0, sh0);
        coeff8 = i16x8_add(coeff8, sh8);
    }

    // Load iQ as u16 (fits in u16 for typical quantizers)
    let iq0 = u16x8(
        matrix.iq[0] as u16,
        matrix.iq[1] as u16,
        matrix.iq[2] as u16,
        matrix.iq[3] as u16,
        matrix.iq[4] as u16,
        matrix.iq[5] as u16,
        matrix.iq[6] as u16,
        matrix.iq[7] as u16,
    );
    let iq8 = u16x8(
        matrix.iq[8] as u16,
        matrix.iq[9] as u16,
        matrix.iq[10] as u16,
        matrix.iq[11] as u16,
        matrix.iq[12] as u16,
        matrix.iq[13] as u16,
        matrix.iq[14] as u16,
        matrix.iq[15] as u16,
    );

    // Quantize: out = (coeff * iQ + bias) >> QFIX
    // u16 * u16 → u32 using extending multiply
    let prod0_lo = u32x4_extmul_low_u16x8(coeff0, iq0);
    let prod0_hi = u32x4_extmul_high_u16x8(coeff0, iq0);
    let prod8_lo = u32x4_extmul_low_u16x8(coeff8, iq8);
    let prod8_hi = u32x4_extmul_high_u16x8(coeff8, iq8);

    // Load bias (u32[16])
    let bias0 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[0..4]).unwrap());
    let bias4 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[4..8]).unwrap());
    let bias8 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[8..12]).unwrap());
    let bias12 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[12..16]).unwrap());

    // Add bias and shift
    let out0 = u32x4_shr(i32x4_add(prod0_lo, bias0), QFIX);
    let out4 = u32x4_shr(i32x4_add(prod0_hi, bias4), QFIX);
    let out8 = u32x4_shr(i32x4_add(prod8_lo, bias8), QFIX);
    let out12 = u32x4_shr(i32x4_add(prod8_hi, bias12), QFIX);

    // Pack u32 → i16 via narrow
    let mut qout0 = i16x8_narrow_i32x4(out0, out4);
    let mut qout8 = i16x8_narrow_i32x4(out8, out12);

    // Clamp to MAX_LEVEL
    qout0 = i16x8_min(qout0, max_coeff);
    qout8 = i16x8_min(qout8, max_coeff);

    // Restore sign: (out ^ sign) - sign
    qout0 = i16x8_sub(v128_xor(qout0, sign0), sign0);
    qout8 = i16x8_sub(v128_xor(qout8, sign8), sign8);

    // Widen i16 → i32 and store
    let s0 = i32x4_extend_low_i16x8(qout0);
    let s4 = i32x4_extend_high_i16x8(qout0);
    let s8 = i32x4_extend_low_i16x8(qout8);
    let s12 = i32x4_extend_high_i16x8(qout8);

    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[0..4]).unwrap(), s0);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[4..8]).unwrap(), s4);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[8..12]).unwrap(), s8);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[12..16]).unwrap(), s12);

    // Check if any coefficient is non-zero
    let or0 = v128_or(qout0, qout8);
    hsum_abs_i16x8(or0) != 0
}

/// Entry shim for quantize_block_wasm
#[arcane]
pub(crate) fn quantize_block_wasm_entry(
    _token: Wasm128Token,
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    quantize_block_wasm(_token, coeffs, matrix, use_sharpen)
}

/// WASM dequantize block: multiply coefficients by quantizer steps
#[rite]
pub(crate) fn dequantize_block_wasm(_token: Wasm128Token, q: &[u16; 16], coeffs: &mut [i32; 16]) {
    // Load q as u16, extend to i32
    let q0_u16 = load_u16x8(<&[u16; 8]>::try_from(&q[0..8]).unwrap());
    let q8_u16 = load_u16x8(<&[u16; 8]>::try_from(&q[8..16]).unwrap());
    let q0_lo = u32x4_extend_low_u16x8(q0_u16);
    let q0_hi = u32x4_extend_high_u16x8(q0_u16);
    let q8_lo = u32x4_extend_low_u16x8(q8_u16);
    let q8_hi = u32x4_extend_high_u16x8(q8_u16);

    // Load coefficients
    let c0 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c4 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c8 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c12 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    // Multiply: WASM has i32x4_mul
    let r0 = i32x4_mul(c0, q0_lo);
    let r4 = i32x4_mul(c4, q0_hi);
    let r8 = i32x4_mul(c8, q8_lo);
    let r12 = i32x4_mul(c12, q8_hi);

    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[0..4]).unwrap(), r0);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[4..8]).unwrap(), r4);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[8..12]).unwrap(), r8);
    store_i32x4(<&mut [i32; 4]>::try_from(&mut coeffs[12..16]).unwrap(), r12);
}

/// Entry shim for dequantize_block_wasm
#[arcane]
pub(crate) fn dequantize_block_wasm_entry(
    _token: Wasm128Token,
    q: &[u16; 16],
    coeffs: &mut [i32; 16],
) {
    dequantize_block_wasm(_token, q, coeffs);
}

/// WASM fused quantize+dequantize: produces both quantized and dequantized values.
/// Returns true if any coefficient is non-zero.
#[rite]
pub(crate) fn quantize_dequantize_block_wasm(
    _token: Wasm128Token,
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let max_coeff = i16x8_splat(MAX_LEVEL as i16);

    // Pack i32 to i16
    let c0 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3 = load_i32x4(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in0 = i16x8_narrow_i32x4(c0, c1);
    let in8 = i16x8_narrow_i32x4(c2, c3);

    // Sign and absolute value
    let sign0 = i16x8_shr(in0, 15);
    let sign8 = i16x8_shr(in8, 15);
    let mut coeff0 = i16x8_abs(in0);
    let mut coeff8 = i16x8_abs(in8);

    if use_sharpen {
        let sh0 = load_u16x8(<&[u16; 8]>::try_from(&matrix.sharpen[0..8]).unwrap());
        let sh8 = load_u16x8(<&[u16; 8]>::try_from(&matrix.sharpen[8..16]).unwrap());
        coeff0 = i16x8_add(coeff0, sh0);
        coeff8 = i16x8_add(coeff8, sh8);
    }

    // Load iQ
    let iq0 = u16x8(
        matrix.iq[0] as u16,
        matrix.iq[1] as u16,
        matrix.iq[2] as u16,
        matrix.iq[3] as u16,
        matrix.iq[4] as u16,
        matrix.iq[5] as u16,
        matrix.iq[6] as u16,
        matrix.iq[7] as u16,
    );
    let iq8 = u16x8(
        matrix.iq[8] as u16,
        matrix.iq[9] as u16,
        matrix.iq[10] as u16,
        matrix.iq[11] as u16,
        matrix.iq[12] as u16,
        matrix.iq[13] as u16,
        matrix.iq[14] as u16,
        matrix.iq[15] as u16,
    );

    // Multiply and add bias
    let prod0_lo = u32x4_extmul_low_u16x8(coeff0, iq0);
    let prod0_hi = u32x4_extmul_high_u16x8(coeff0, iq0);
    let prod8_lo = u32x4_extmul_low_u16x8(coeff8, iq8);
    let prod8_hi = u32x4_extmul_high_u16x8(coeff8, iq8);

    let bias0 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[0..4]).unwrap());
    let bias4 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[4..8]).unwrap());
    let bias8 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[8..12]).unwrap());
    let bias12 = load_u32x4(<&[u32; 4]>::try_from(&matrix.bias[12..16]).unwrap());

    let out0 = u32x4_shr(i32x4_add(prod0_lo, bias0), QFIX);
    let out4 = u32x4_shr(i32x4_add(prod0_hi, bias4), QFIX);
    let out8 = u32x4_shr(i32x4_add(prod8_lo, bias8), QFIX);
    let out12 = u32x4_shr(i32x4_add(prod8_hi, bias12), QFIX);

    // Pack to i16 and clamp
    let mut qout0 = i16x8_narrow_i32x4(out0, out4);
    let mut qout8 = i16x8_narrow_i32x4(out8, out12);
    qout0 = i16x8_min(qout0, max_coeff);
    qout8 = i16x8_min(qout8, max_coeff);

    // Restore sign
    qout0 = i16x8_sub(v128_xor(qout0, sign0), sign0);
    qout8 = i16x8_sub(v128_xor(qout8, sign8), sign8);

    // Store quantized (i16 → i32)
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut quantized[0..4]).unwrap(),
        i32x4_extend_low_i16x8(qout0),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut quantized[4..8]).unwrap(),
        i32x4_extend_high_i16x8(qout0),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut quantized[8..12]).unwrap(),
        i32x4_extend_low_i16x8(qout8),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut quantized[12..16]).unwrap(),
        i32x4_extend_high_i16x8(qout8),
    );

    // Dequantize: quantized * q
    let q0 = load_u16x8(<&[u16; 8]>::try_from(&matrix.q[0..8]).unwrap());
    let q8_val = load_u16x8(<&[u16; 8]>::try_from(&matrix.q[8..16]).unwrap());

    let dq0 = i16x8_mul(qout0, q0);
    let dq8 = i16x8_mul(qout8, q8_val);

    // Store dequantized (i16 → i32)
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut dequantized[0..4]).unwrap(),
        i32x4_extend_low_i16x8(dq0),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut dequantized[4..8]).unwrap(),
        i32x4_extend_high_i16x8(dq0),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut dequantized[8..12]).unwrap(),
        i32x4_extend_low_i16x8(dq8),
    );
    store_i32x4(
        <&mut [i32; 4]>::try_from(&mut dequantized[12..16]).unwrap(),
        i32x4_extend_high_i16x8(dq8),
    );

    // Non-zero check
    let or0 = v128_or(qout0, qout8);
    hsum_abs_i16x8(or0) != 0
}

/// Entry shim for quantize_dequantize_block_wasm
#[arcane]
pub(crate) fn quantize_dequantize_block_wasm_entry(
    _token: Wasm128Token,
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    quantize_dequantize_block_wasm(_token, coeffs, matrix, use_sharpen, quantized, dequantized)
}
