//! Raw SIMD intrinsics for DCT/IDCT transforms matching libwebp.
//!
//! Uses i16 arithmetic like libwebp for maximum throughput (8 values per __m128i).
//! Runtime detection selects best available: SSE2 / AVX2+FMA / AVX-512

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use archmage::{arcane, rite, SimdToken, X64V3Token};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use archmage::intrinsics::x86_64 as simd_mem;

// =============================================================================
// Public dispatch functions
// =============================================================================

/// Forward DCT with dynamic dispatch to best available implementation
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn dct4x4_intrinsics(block: &mut [i32; 16]) {
    if let Some(token) = X64V3Token::summon() {
        dct4x4_entry(token, block);
    } else {
        crate::common::transform::dct4x4_scalar(block);
    }
}

/// Forward DCT - non-x86 fallback
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub(crate) fn dct4x4_intrinsics(block: &mut [i32; 16]) {
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            super::transform_wasm::dct4x4_wasm(token, block);
        } else {
            crate::common::transform::dct4x4_scalar(block);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            super::transform_aarch64::dct4x4_neon(token, block);
        } else {
            crate::common::transform::dct4x4_scalar(block);
        }
    }
    #[cfg(not(any(target_arch = "wasm32", target_arch = "aarch64")))]
    {
        crate::common::transform::dct4x4_scalar(block);
    }
}

/// Inverse DCT with dynamic dispatch
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn idct4x4_intrinsics(block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    if let Some(token) = X64V3Token::summon() {
        idct4x4_entry(token, block);
    } else {
        crate::common::transform::idct4x4_scalar(block);
    }
}

/// Inverse DCT with pre-summoned token (avoids per-call token summoning)
/// Note: Currently unused - decoder uses fused IDCT+add_residue.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn idct4x4_intrinsics_with_token(
    block: &mut [i32],
    simd_token: crate::common::prediction::SimdTokenType,
) {
    debug_assert!(block.len() >= 16);
    if let Some(token) = simd_token {
        idct4x4_entry(token, block);
    } else {
        crate::common::transform::idct4x4_scalar(block);
    }
}

/// Inverse DCT - non-x86 fallback
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline]
pub(crate) fn idct4x4_intrinsics(block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            super::transform_wasm::idct4x4_wasm(token, block);
        } else {
            crate::common::transform::idct4x4_scalar(block);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            super::transform_aarch64::idct4x4_neon(token, block);
        } else {
            crate::common::transform::idct4x4_scalar(block);
        }
    }
    #[cfg(not(any(target_arch = "wasm32", target_arch = "aarch64")))]
    {
        crate::common::transform::idct4x4_scalar(block);
    }
}

/// Inverse DCT with pre-summoned token - non-x86 fallback (ignores token)
/// Note: Currently unused - decoder uses fused IDCT+add_residue.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(dead_code)]
#[inline]
pub(crate) fn idct4x4_intrinsics_with_token(
    block: &mut [i32],
    _simd_token: crate::common::prediction::SimdTokenType,
) {
    idct4x4_intrinsics(block);
}

/// Process two blocks at once
#[allow(dead_code)]
pub(crate) fn dct4x4_two_intrinsics(block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        // SSE4.1 implies SSE2; summon() is now fast (no env var check)
        if let Some(token) = X64V3Token::summon() {
            dct4x4_two_entry(token, block1, block2);
        } else {
            crate::common::transform::dct4x4_scalar(block1);
            crate::common::transform::dct4x4_scalar(block2);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            super::transform_wasm::dct4x4_wasm(token, block1);
            super::transform_wasm::dct4x4_wasm(token, block2);
        } else {
            crate::common::transform::dct4x4_scalar(block1);
            crate::common::transform::dct4x4_scalar(block2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::try_new() {
            super::transform_aarch64::dct4x4_neon(token, block1);
            super::transform_aarch64::dct4x4_neon(token, block2);
        } else {
            crate::common::transform::dct4x4_scalar(block1);
            crate::common::transform::dct4x4_scalar(block2);
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "wasm32",
        target_arch = "aarch64"
    )))]
    {
        crate::common::transform::dct4x4_scalar(block1);
        crate::common::transform::dct4x4_scalar(block2);
    }
}

// =============================================================================
// SSE2 Implementation - matches libwebp's FTransform/ITransform
// =============================================================================

/// Entry shim for dct4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn dct4x4_entry(_token: X64V3Token, block: &mut [i32; 16]) {
    dct4x4_sse2(_token, block);
}

/// Forward DCT using SSE2 with i16 layout matching libwebp's FTransform
/// Uses _mm_madd_epi16 for efficient multiply-accumulate
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn dct4x4_sse2(_token: X64V3Token, block: &mut [i32; 16]) {
    // Convert i32 block to i16 and reorganize into libwebp's interleaved layout:
    // row01 = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
    // row23 = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]
    let row01 = _mm_set_epi16(
        block[7] as i16, // r1c3
        block[6] as i16, // r1c2
        block[3] as i16, // r0c3
        block[2] as i16, // r0c2
        block[5] as i16, // r1c1
        block[4] as i16, // r1c0
        block[1] as i16, // r0c1
        block[0] as i16, // r0c0
    );
    let row23 = _mm_set_epi16(
        block[15] as i16, // r3c3
        block[14] as i16, // r3c2
        block[11] as i16, // r2c3
        block[10] as i16, // r2c2
        block[13] as i16, // r3c1
        block[12] as i16, // r3c0
        block[9] as i16,  // r2c1
        block[8] as i16,  // r2c0
    );

    // Forward transform pass 1 (rows)
    let (v01, v32) = ftransform_pass1_i16(_token, row01, row23);

    // Forward transform pass 2 (columns)
    let mut out16 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01, &v32, &mut out16);

    // Convert i16 output back to i32 using SIMD sign extension
    let zero = _mm_setzero_si128();
    let out01 = simd_mem::_mm_loadu_si128(<&[i16; 8]>::try_from(&out16[0..8]).unwrap());
    let out23 = simd_mem::_mm_loadu_si128(<&[i16; 8]>::try_from(&out16[8..16]).unwrap());

    // Sign extend i16 to i32
    let sign01 = _mm_cmpgt_epi16(zero, out01);
    let sign23 = _mm_cmpgt_epi16(zero, out23);

    let out_0 = _mm_unpacklo_epi16(out01, sign01);
    let out_1 = _mm_unpackhi_epi16(out01, sign01);
    let out_2 = _mm_unpacklo_epi16(out23, sign23);
    let out_3 = _mm_unpackhi_epi16(out23, sign23);

    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[0..4]).unwrap(), out_0);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[4..8]).unwrap(), out_1);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[8..12]).unwrap(), out_2);
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut block[12..16]).unwrap(),
        out_3,
    );
}

/// FTransform Pass 1 - matches libwebp FTransformPass1_SSE2 exactly
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn ftransform_pass1_i16(_token: X64V3Token, in01: __m128i, in23: __m128i) -> (__m128i, __m128i) {
    // Constants matching libwebp exactly
    let k937 = _mm_set1_epi32(937);
    let k1812 = _mm_set1_epi32(1812);
    let k88p = _mm_set_epi16(8, 8, 8, 8, 8, 8, 8, 8);
    let k88m = _mm_set_epi16(-8, 8, -8, 8, -8, 8, -8, 8);
    let k5352_2217p = _mm_set_epi16(2217, 5352, 2217, 5352, 2217, 5352, 2217, 5352);
    let k5352_2217m = _mm_set_epi16(-5352, 2217, -5352, 2217, -5352, 2217, -5352, 2217);

    // in01 = 00 01 10 11 02 03 12 13
    // in23 = 20 21 30 31 22 23 32 33
    // Shuffle high words: swap pairs in high 64-bits
    let shuf01_p = _mm_shufflehi_epi16(in01, 0b10_11_00_01); // _MM_SHUFFLE(2,3,0,1)
    let shuf23_p = _mm_shufflehi_epi16(in23, 0b10_11_00_01);
    // 00 01 10 11 03 02 13 12
    // 20 21 30 31 23 22 33 32

    // Interleave to separate low/high parts
    let s01 = _mm_unpacklo_epi64(shuf01_p, shuf23_p);
    let s32 = _mm_unpackhi_epi64(shuf01_p, shuf23_p);
    // s01 = 00 01 10 11 20 21 30 31  (columns 0,1 for all rows)
    // s32 = 03 02 13 12 23 22 33 32  (columns 3,2 reversed for all rows)

    // a01 = [d0+d3, d1+d2] for each row pair
    // a32 = [d0-d3, d1-d2] for each row pair
    let a01 = _mm_add_epi16(s01, s32);
    let a32 = _mm_sub_epi16(s01, s32);

    // tmp0 = (a0 + a1) << 3 via madd with [8,8] - produces i32
    // tmp2 = (a0 - a1) << 3 via madd with [-8,8]
    let tmp0 = _mm_madd_epi16(a01, k88p);
    let tmp2 = _mm_madd_epi16(a01, k88m);

    // tmp1 = (a3*5352 + a2*2217 + 1812) >> 9
    // tmp3 = (a3*2217 - a2*5352 + 937) >> 9
    // Note: a32 has [d0-d3, d1-d2] = [a3, a2] for the formulas
    let tmp1_1 = _mm_madd_epi16(a32, k5352_2217p);
    let tmp3_1 = _mm_madd_epi16(a32, k5352_2217m);
    let tmp1_2 = _mm_add_epi32(tmp1_1, k1812);
    let tmp3_2 = _mm_add_epi32(tmp3_1, k937);
    let tmp1 = _mm_srai_epi32(tmp1_2, 9);
    let tmp3 = _mm_srai_epi32(tmp3_2, 9);

    // Pack back to i16
    let s03 = _mm_packs_epi32(tmp0, tmp2); // [out0_r0, out0_r1, out0_r2, out0_r3, out2_r0, ...]
    let s12 = _mm_packs_epi32(tmp1, tmp3); // [out1_r0, out1_r1, out1_r2, out1_r3, out3_r0, ...]

    // Interleave to get proper output order
    let s_lo = _mm_unpacklo_epi16(s03, s12); // 0 1 0 1 0 1 0 1
    let s_hi = _mm_unpackhi_epi16(s03, s12); // 2 3 2 3 2 3 2 3
    let v23 = _mm_unpackhi_epi32(s_lo, s_hi);
    let out01 = _mm_unpacklo_epi32(s_lo, s_hi);
    let out32 = _mm_shuffle_epi32(v23, 0b01_00_11_10); // _MM_SHUFFLE(1,0,3,2)

    (out01, out32)
}

/// FTransform Pass 2 - matches libwebp FTransformPass2_SSE2 exactly
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn ftransform_pass2_i16(_token: X64V3Token, v01: &__m128i, v32: &__m128i, out: &mut [i16; 16]) {
    let zero = _mm_setzero_si128();
    let seven = _mm_set1_epi16(7);
    let k5352_2217 = _mm_set_epi16(5352, 2217, 5352, 2217, 5352, 2217, 5352, 2217);
    let k2217_5352 = _mm_set_epi16(2217, -5352, 2217, -5352, 2217, -5352, 2217, -5352);
    let k12000_plus_one = _mm_set1_epi32(12000 + (1 << 16));
    let k51000 = _mm_set1_epi32(51000);

    // a3 = v0 - v3, a2 = v1 - v2
    let a32 = _mm_sub_epi16(*v01, *v32);
    let a22 = _mm_unpackhi_epi64(a32, a32);

    let b23 = _mm_unpacklo_epi16(a22, a32);
    let c1 = _mm_madd_epi16(b23, k5352_2217);
    let c3 = _mm_madd_epi16(b23, k2217_5352);
    let d1 = _mm_add_epi32(c1, k12000_plus_one);
    let d3 = _mm_add_epi32(c3, k51000);
    let e1 = _mm_srai_epi32(d1, 16);
    let e3 = _mm_srai_epi32(d3, 16);

    let f1 = _mm_packs_epi32(e1, e1);
    let f3 = _mm_packs_epi32(e3, e3);

    // g1 = f1 + (a3 != 0) - the +1 is already in k12000_plus_one
    // cmpeq returns 0xFFFF for equal, 0 for not equal
    // Adding 0xFFFF (-1) when equal subtracts 1, which cancels the +1 we added
    let g1 = _mm_add_epi16(f1, _mm_cmpeq_epi16(a32, zero));

    // a0 = v0 + v3, a1 = v1 + v2
    let a01 = _mm_add_epi16(*v01, *v32);
    let a01_plus_7 = _mm_add_epi16(a01, seven);
    let a11 = _mm_unpackhi_epi64(a01, a01);
    let c0 = _mm_add_epi16(a01_plus_7, a11);
    let c2 = _mm_sub_epi16(a01_plus_7, a11);

    let d0 = _mm_srai_epi16(c0, 4);
    let d2 = _mm_srai_epi16(c2, 4);

    // Combine outputs: row 0 and 1 in first register, row 2 and 3 in second
    let d0_g1 = _mm_unpacklo_epi64(d0, g1);
    let d2_f3 = _mm_unpacklo_epi64(d2, f3);

    simd_mem::_mm_storeu_si128(<&mut [i16; 8]>::try_from(&mut out[0..8]).unwrap(), d0_g1);
    simd_mem::_mm_storeu_si128(<&mut [i16; 8]>::try_from(&mut out[8..16]).unwrap(), d2_f3);
}

/// Entry shim for dct4x4_two_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
#[allow(dead_code)]
fn dct4x4_two_entry(_token: X64V3Token, block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    dct4x4_two_sse2(_token, block1, block2);
}

/// Process two blocks at once
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(dead_code)]
fn dct4x4_two_sse2(_token: X64V3Token, block1: &mut [i32; 16], block2: &mut [i32; 16]) {
    // Process both blocks using the single-block implementation
    // AVX2 can potentially do this more efficiently with 256-bit registers
    dct4x4_sse2(_token, block1);
    dct4x4_sse2(_token, block2);
}

// =============================================================================
// Fused Residual + DCT (matches libwebp's FTransform2_SSE2)
// =============================================================================

/// Fused residual computation + DCT for two adjacent 4x4 blocks
/// Takes u8 source and reference with stride, outputs i16 coefficients.
/// Matches libwebp's FTransform2_SSE2 for maximum performance.
///
/// # Arguments
/// * `src` - Source pixels (8 bytes per row = 2 blocks side by side)
/// * `ref_` - Reference/prediction pixels (same layout)
/// * `src_stride` - Stride for source rows
/// * `ref_stride` - Stride for reference rows
/// * `out` - Output: 32 i16 coefficients (2 blocks × 16 coeffs)
/// Entry shim for ftransform2_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn ftransform2_entry(
    _token: X64V3Token,
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    ftransform2_sse2(_token, src, ref_, src_stride, ref_stride, out);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn ftransform2_sse2(
    _token: X64V3Token,
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    let zero = _mm_setzero_si128();

    // Load 8 bytes per row from src (2 blocks × 4 bytes)
    let src0 = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src[..8]).unwrap());
    let src1 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&src[src_stride..src_stride + 8]).unwrap());
    let src2 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&src[src_stride * 2..src_stride * 2 + 8]).unwrap(),
    );
    let src3 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&src[src_stride * 3..src_stride * 3 + 8]).unwrap(),
    );

    // Load 8 bytes per row from ref
    let ref0 = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&ref_[..8]).unwrap());
    let ref1 =
        simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&ref_[ref_stride..ref_stride + 8]).unwrap());
    let ref2 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&ref_[ref_stride * 2..ref_stride * 2 + 8]).unwrap(),
    );
    let ref3 = simd_mem::_mm_loadu_si64(
        <&[u8; 8]>::try_from(&ref_[ref_stride * 3..ref_stride * 3 + 8]).unwrap(),
    );

    // Convert u8 to i16 (zero-extend)
    let src_0 = _mm_unpacklo_epi8(src0, zero);
    let src_1 = _mm_unpacklo_epi8(src1, zero);
    let src_2 = _mm_unpacklo_epi8(src2, zero);
    let src_3 = _mm_unpacklo_epi8(src3, zero);

    let ref_0 = _mm_unpacklo_epi8(ref0, zero);
    let ref_1 = _mm_unpacklo_epi8(ref1, zero);
    let ref_2 = _mm_unpacklo_epi8(ref2, zero);
    let ref_3 = _mm_unpacklo_epi8(ref3, zero);

    // Compute difference (src - ref)
    // Each diff register holds: [blk0_col0, blk0_col1, blk0_col2, blk0_col3, blk1_col0, ...]
    let diff0 = _mm_sub_epi16(src_0, ref_0);
    let diff1 = _mm_sub_epi16(src_1, ref_1);
    let diff2 = _mm_sub_epi16(src_2, ref_2);
    let diff3 = _mm_sub_epi16(src_3, ref_3);

    // Reorganize for processing two blocks
    // shuf01l/h splits low/high halves (block 0 and block 1)
    let shuf01l = _mm_unpacklo_epi32(diff0, diff1);
    let shuf23l = _mm_unpacklo_epi32(diff2, diff3);
    let shuf01h = _mm_unpackhi_epi32(diff0, diff1);
    let shuf23h = _mm_unpackhi_epi32(diff2, diff3);

    // First pass for block 0
    let (v01l, v32l) = ftransform_pass1_i16(_token, shuf01l, shuf23l);

    // First pass for block 1
    let (v01h, v32h) = ftransform_pass1_i16(_token, shuf01h, shuf23h);

    // Second pass for both blocks
    let mut out0 = [0i16; 16];
    let mut out1 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01l, &v32l, &mut out0);
    ftransform_pass2_i16(_token, &v01h, &v32h, &mut out1);

    // Copy to output
    out[..16].copy_from_slice(&out0);
    out[16..].copy_from_slice(&out1);
}

/// Public dispatch function for fused residual + DCT of two adjacent blocks
pub(crate) fn ftransform2_from_u8(
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if let Some(token) = X64V3Token::summon() {
            ftransform2_entry(token, src, ref_, src_stride, ref_stride, out);
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            // Process two blocks using single-block WASM DCT
            for block in 0..2 {
                let mut block_data = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let src_val = src[y * src_stride + block * 4 + x] as i32;
                        let ref_val = ref_[y * ref_stride + block * 4 + x] as i32;
                        block_data[y * 4 + x] = src_val - ref_val;
                    }
                }
                super::transform_wasm::dct4x4_wasm(token, &mut block_data);
                for (i, &val) in block_data.iter().enumerate() {
                    out[block * 16 + i] = val as i16;
                }
            }
            return;
        }
    }
    // Scalar fallback
    ftransform2_scalar(src, ref_, src_stride, ref_stride, out);
}

/// Scalar fallback for fused residual + DCT
fn ftransform2_scalar(
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [i16; 32],
) {
    // Process two blocks
    for block in 0..2 {
        let mut block_data = [0i32; 16];
        // Compute residual for this 4x4 block
        for y in 0..4 {
            for x in 0..4 {
                let src_val = src[y * src_stride + block * 4 + x] as i32;
                let ref_val = ref_[y * ref_stride + block * 4 + x] as i32;
                block_data[y * 4 + x] = src_val - ref_val;
            }
        }
        // Apply DCT
        crate::common::transform::dct4x4_scalar(&mut block_data);
        // Convert to i16
        for (i, &val) in block_data.iter().enumerate() {
            out[block * 16 + i] = val as i16;
        }
    }
}

/// Entry shim for idct4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
pub(crate) fn idct4x4_entry(_token: X64V3Token, block: &mut [i32]) {
    idct4x4_sse2(_token, block);
}

/// Inverse DCT using SSE2 - matches libwebp ITransform_One_SSE2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn idct4x4_sse2(_token: X64V3Token, block: &mut [i32]) {
    // libwebp IDCT constants:
    // K1 = sqrt(2) * cos(pi/8) * 65536 = 85627 => k1 = 85627 - 65536 = 20091
    // K2 = sqrt(2) * sin(pi/8) * 65536 = 35468 => k2 = 35468 - 65536 = -30068
    let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);

    // Load i32 values and pack to i16 using SIMD
    let i32_0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&block[0..4]).unwrap());
    let i32_1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&block[4..8]).unwrap());
    let i32_2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&block[8..12]).unwrap());
    let i32_3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&block[12..16]).unwrap());

    // Pack i32 to i16 with saturation (values should be small enough)
    let in01 = _mm_packs_epi32(i32_0, i32_1);
    let in23 = _mm_packs_epi32(i32_2, i32_3);

    // Vertical pass
    let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);

    // Horizontal pass with rounding
    let (out01, out23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

    // Unpack i16 back to i32 using sign extension
    // out01 contains 8 i16 values: [0,1,2,3,4,5,6,7]
    // out23 contains 8 i16 values: [8,9,10,11,12,13,14,15]
    let zero = _mm_setzero_si128();

    // Sign extend low 4 i16 values to i32
    let sign01_lo = _mm_cmpgt_epi16(zero, out01); // all 1s where negative
    let out_0 = _mm_unpacklo_epi16(out01, sign01_lo);
    let out_1 = _mm_unpackhi_epi16(out01, sign01_lo);

    let sign23_lo = _mm_cmpgt_epi16(zero, out23);
    let out_2 = _mm_unpacklo_epi16(out23, sign23_lo);
    let out_3 = _mm_unpackhi_epi16(out23, sign23_lo);

    // Store results
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[0..4]).unwrap(), out_0);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[4..8]).unwrap(), out_1);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut block[8..12]).unwrap(), out_2);
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut block[12..16]).unwrap(),
        out_3,
    );
}

/// ITransform vertical pass - matches libwebp
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn itransform_pass_sse2(
    _token: X64V3Token,
    in01: __m128i,
    in23: __m128i,
    k1k2: __m128i,
    k2k1: __m128i,
) -> (__m128i, __m128i) {
    // in01 = a00 a10 a20 a30 | a01 a11 a21 a31
    // in23 = a02 a12 a22 a32 | a03 a13 a23 a33

    let in1 = _mm_unpackhi_epi64(in01, in01);
    let in3 = _mm_unpackhi_epi64(in23, in23);

    // a = in0 + in2, b = in0 - in2
    let a_d3 = _mm_add_epi16(in01, in23);
    let b_c3 = _mm_sub_epi16(in01, in23);

    // c = MUL(in1, K2) - MUL(in3, K1)
    // d = MUL(in1, K1) + MUL(in3, K2)
    // Using the trick: MUL(x, K) = x + MUL(x, k) where K = k + 65536
    let c1d1 = _mm_mulhi_epi16(in1, k2k1);
    let c2d2 = _mm_mulhi_epi16(in3, k1k2);
    let c3 = _mm_unpackhi_epi64(b_c3, b_c3);
    let c4 = _mm_sub_epi16(c1d1, c2d2);
    let c = _mm_add_epi16(c3, c4);
    let d4u = _mm_add_epi16(c1d1, c2d2);
    let du = _mm_add_epi16(a_d3, d4u);
    let d = _mm_unpackhi_epi64(du, du);

    // Combine and transpose
    let comb_ab = _mm_unpacklo_epi64(a_d3, b_c3);
    let comb_dc = _mm_unpacklo_epi64(d, c);

    let tmp01 = _mm_add_epi16(comb_ab, comb_dc);
    let tmp32 = _mm_sub_epi16(comb_ab, comb_dc);
    let tmp23 = _mm_shuffle_epi32(tmp32, 0b01_00_11_10);

    let transpose_0 = _mm_unpacklo_epi16(tmp01, tmp23);
    let transpose_1 = _mm_unpackhi_epi16(tmp01, tmp23);

    let t01 = _mm_unpacklo_epi16(transpose_0, transpose_1);
    let t23 = _mm_unpackhi_epi16(transpose_0, transpose_1);

    (t01, t23)
}

/// ITransform horizontal pass with final shift
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
fn itransform_pass2_sse2(
    _token: X64V3Token,
    t01: __m128i,
    t23: __m128i,
    k1k2: __m128i,
    k2k1: __m128i,
    zero_four: __m128i,
) -> (__m128i, __m128i) {
    let t1 = _mm_unpackhi_epi64(t01, t01);
    let t3 = _mm_unpackhi_epi64(t23, t23);

    // Add rounding constant to DC
    let dc = _mm_add_epi16(t01, zero_four);

    let a_d3 = _mm_add_epi16(dc, t23);
    let b_c3 = _mm_sub_epi16(dc, t23);

    let c1d1 = _mm_mulhi_epi16(t1, k2k1);
    let c2d2 = _mm_mulhi_epi16(t3, k1k2);
    let c3 = _mm_unpackhi_epi64(b_c3, b_c3);
    let c4 = _mm_sub_epi16(c1d1, c2d2);
    let c = _mm_add_epi16(c3, c4);
    let d4u = _mm_add_epi16(c1d1, c2d2);
    let du = _mm_add_epi16(a_d3, d4u);
    let d = _mm_unpackhi_epi64(du, du);

    let comb_ab = _mm_unpacklo_epi64(a_d3, b_c3);
    let comb_dc = _mm_unpacklo_epi64(d, c);

    let tmp01 = _mm_add_epi16(comb_ab, comb_dc);
    let tmp32 = _mm_sub_epi16(comb_ab, comb_dc);
    let tmp23 = _mm_shuffle_epi32(tmp32, 0b01_00_11_10);

    // Final shift >> 3
    let shifted01 = _mm_srai_epi16(tmp01, 3);
    let shifted23 = _mm_srai_epi16(tmp23, 3);

    // Transpose back
    let transpose_0 = _mm_unpacklo_epi16(shifted01, shifted23);
    let transpose_1 = _mm_unpackhi_epi16(shifted01, shifted23);

    let out01 = _mm_unpacklo_epi16(transpose_0, transpose_1);
    let out23 = _mm_unpackhi_epi16(transpose_0, transpose_1);

    (out01, out23)
}

// =============================================================================
// Fused IDCT + Add Residue (like libwebp's ITransform_SSE2)
// =============================================================================

/// Fused IDCT + add residue + clamp, writing directly to output buffer.
/// This is the hot path optimization - avoids separate IDCT and add_residue calls.
///
/// Takes: coefficients (i32[16]), prediction block, output location
/// Does: IDCT(coeffs) + prediction → output (clamped to 0-255)
/// Also clears the coefficient block to zeros.
/// Entry shim for idct_add_residue_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_entry(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    idct_add_residue_sse2(
        _token,
        coeffs,
        pred_block,
        pred_stride,
        out_block,
        out_stride,
        pred_y0,
        pred_x0,
        out_y0,
        out_x0,
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    // IDCT constants
    let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
    let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
    let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);
    let zero = _mm_setzero_si128();

    // Load i32 coefficients and pack to i16
    let i32_0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let i32_1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let i32_2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let i32_3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in01 = _mm_packs_epi32(i32_0, i32_1);
    let in23 = _mm_packs_epi32(i32_2, i32_3);

    // Vertical pass
    let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);

    // Horizontal pass with rounding → residual in i16
    let (res01, res23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

    // res01 = row0[0-3] row1[0-3] as i16
    // res23 = row2[0-3] row3[0-3] as i16

    // Load 4 prediction bytes per row, extend to i16, add residual, pack to u8
    macro_rules! process_row {
        ($res:expr, $row_idx:expr, $hi:expr) => {{
            // Extract this row's residuals (4 x i16)
            let residual = if $hi {
                _mm_unpackhi_epi64($res, $res)
            } else {
                $res
            };

            // Load 4 prediction bytes
            let pred_pos = (pred_y0 + $row_idx) * pred_stride + pred_x0;
            let pred_bytes: [u8; 4] = pred_block[pred_pos..pred_pos + 4].try_into().unwrap();
            let pred_i32 = i32::from_ne_bytes(pred_bytes);
            let pred_vec = _mm_cvtsi32_si128(pred_i32);
            // Extend u8 → i16 (zero extend, then treat as signed for addition)
            let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);

            // Add residual to prediction (i16 + i16 → i16)
            let sum = _mm_add_epi16(pred_i16, residual);

            // Pack to u8 with saturation (clamp to 0-255)
            let packed = _mm_packus_epi16(sum, sum);

            // Store 4 bytes to output
            let out_pos = (out_y0 + $row_idx) * out_stride + out_x0;
            let result = _mm_cvtsi128_si32(packed) as u32;
            out_block[out_pos..out_pos + 4].copy_from_slice(&result.to_ne_bytes());
        }};
    }

    process_row!(res01, 0, false);
    process_row!(res01, 1, true);
    process_row!(res23, 2, false);
    process_row!(res23, 3, true);

    // Clear coefficient block
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[0..4]).unwrap(), zero);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[4..8]).unwrap(), zero);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[8..12]).unwrap(), zero);
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut coeffs[12..16]).unwrap(),
        zero,
    );
}

/// Entry shim for idct_add_residue_dc_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_dc_entry(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    idct_add_residue_dc_sse2(
        _token,
        coeffs,
        pred_block,
        pred_stride,
        out_block,
        out_stride,
        pred_y0,
        pred_x0,
        out_y0,
        out_x0,
    );
}

/// DC-only fused IDCT + add residue - when only DC coefficient is non-zero.
/// Much faster than full IDCT.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idct_add_residue_dc_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
) {
    // DC-only: output = prediction + ((DC + 4) >> 3) for all 16 pixels
    let dc = coeffs[0];
    let dc_adj = ((dc + 4) >> 3) as i16;
    let dc_vec = _mm_set1_epi16(dc_adj);
    let zero = _mm_setzero_si128();

    for row in 0..4 {
        // Load 4 prediction bytes
        let pred_pos = (pred_y0 + row) * pred_stride + pred_x0;
        let pred_bytes: [u8; 4] = pred_block[pred_pos..pred_pos + 4].try_into().unwrap();
        let pred_i32 = i32::from_ne_bytes(pred_bytes);
        let pred_vec = _mm_cvtsi32_si128(pred_i32);
        let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);

        // Add DC to prediction
        let sum = _mm_add_epi16(pred_i16, dc_vec);

        // Pack to u8 with saturation
        let packed = _mm_packus_epi16(sum, sum);

        // Store 4 bytes
        let out_pos = (out_y0 + row) * out_stride + out_x0;
        let result = _mm_cvtsi128_si32(packed) as u32;
        out_block[out_pos..out_pos + 4].copy_from_slice(&result.to_ne_bytes());
    }

    // Clear coefficient block
    coeffs.fill(0);
}

/// Wrapper with token type parameter for decoder use (separate pred/output)
/// Note: Currently unused - decoder uses in-place version instead.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(dead_code, clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn idct_add_residue_with_token(
    token: X64V3Token,
    coeffs: &mut [i32; 16],
    pred_block: &[u8],
    pred_stride: usize,
    out_block: &mut [u8],
    out_stride: usize,
    pred_y0: usize,
    pred_x0: usize,
    out_y0: usize,
    out_x0: usize,
    dc_only: bool,
) {
    if dc_only {
        idct_add_residue_dc_entry(
            token,
            coeffs,
            pred_block,
            pred_stride,
            out_block,
            out_stride,
            pred_y0,
            pred_x0,
            out_y0,
            out_x0,
        );
    } else {
        idct_add_residue_entry(
            token,
            coeffs,
            pred_block,
            pred_stride,
            out_block,
            out_stride,
            pred_y0,
            pred_x0,
            out_y0,
            out_x0,
        );
    }
}

/// In-place fused IDCT + add residue for decoder hot path.
/// Performs IDCT on raw DCT coefficients, adds to prediction block in-place, and clears coefficients.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
pub(crate) fn idct_add_residue_inplace_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    if dc_only {
        // DC-only fast path
        let dc = coeffs[0];
        let dc_adj = ((dc + 4) >> 3) as i16;
        let dc_vec = _mm_set1_epi16(dc_adj);
        let zero = _mm_setzero_si128();

        for row in 0..4 {
            let pos = (y0 + row) * stride + x0;
            let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
            let pred_i32 = i32::from_ne_bytes(pred_bytes);
            let pred_vec = _mm_cvtsi32_si128(pred_i32);
            let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);
            let sum = _mm_add_epi16(pred_i16, dc_vec);
            let packed = _mm_packus_epi16(sum, sum);
            let result = _mm_cvtsi128_si32(packed) as u32;
            block[pos..pos + 4].copy_from_slice(&result.to_ne_bytes());
        }
    } else {
        // Full IDCT path
        let k1k2 = _mm_set_epi16(-30068, -30068, -30068, -30068, 20091, 20091, 20091, 20091);
        let k2k1 = _mm_set_epi16(20091, 20091, 20091, 20091, -30068, -30068, -30068, -30068);
        let zero_four = _mm_set_epi16(0, 0, 0, 0, 4, 4, 4, 4);
        let zero = _mm_setzero_si128();

        // Load and pack coefficients
        let i32_0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
        let i32_1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
        let i32_2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
        let i32_3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

        let in01 = _mm_packs_epi32(i32_0, i32_1);
        let in23 = _mm_packs_epi32(i32_2, i32_3);

        // Vertical + horizontal passes → residuals in i16
        let (t01, t23) = itransform_pass_sse2(_token, in01, in23, k1k2, k2k1);
        let (res01, res23) = itransform_pass2_sse2(_token, t01, t23, k1k2, k2k1, zero_four);

        // Process each row: load pred, add residual, store
        macro_rules! process_row {
            ($res:expr, $row_idx:expr, $hi:expr) => {{
                let residual = if $hi {
                    _mm_unpackhi_epi64($res, $res)
                } else {
                    $res
                };
                let pos = (y0 + $row_idx) * stride + x0;
                let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
                let pred_i32 = i32::from_ne_bytes(pred_bytes);
                let pred_vec = _mm_cvtsi32_si128(pred_i32);
                let pred_i16 = _mm_unpacklo_epi8(pred_vec, zero);
                let sum = _mm_add_epi16(pred_i16, residual);
                let packed = _mm_packus_epi16(sum, sum);
                let result = _mm_cvtsi128_si32(packed) as u32;
                block[pos..pos + 4].copy_from_slice(&result.to_ne_bytes());
            }};
        }

        process_row!(res01, 0, false);
        process_row!(res01, 1, true);
        process_row!(res23, 2, false);
        process_row!(res23, 3, true);
    }

    // Clear coefficient block
    coeffs.fill(0);
}

/// Wrapper for in-place fused IDCT + add residue
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn idct_add_residue_inplace_with_token(
    token: X64V3Token,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    idct_add_residue_inplace_sse2(token, coeffs, block, y0, x0, stride, dc_only);
}

// =============================================================================
// In-place fused IDCT + add_residue for encoder
// =============================================================================

/// In-place fused IDCT + add residue with dispatch (no token needed).
/// Reads prediction from `block[y0*stride+x0..]`, performs IDCT on coefficients,
/// adds residual to prediction, clamps 0-255, stores back, and clears coefficients.
///
/// DC-only fast path when `dc_only` is true (skips full IDCT, just adds constant).
#[inline(always)]
pub(crate) fn idct_add_residue_inplace(
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if let Some(token) = X64V3Token::summon() {
            idct_add_residue_inplace_sse2(token, coeffs, block, y0, x0, stride, dc_only);
            return;
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            super::transform_aarch64::idct_add_residue_inplace_neon(
                token, coeffs, block, y0, x0, stride, dc_only,
            );
            return;
        }
    }
    // Scalar fallback: IDCT then add
    if dc_only {
        let dc = coeffs[0];
        let dc_adj = (dc + 4) >> 3;
        for row in 0..4 {
            let pos = (y0 + row) * stride + x0;
            for col in 0..4 {
                let p = block[pos + col] as i32;
                block[pos + col] = (p + dc_adj).clamp(0, 255) as u8;
            }
        }
    } else {
        crate::common::transform::idct4x4_scalar(coeffs);
        let mut pos = y0 * stride + x0;
        for row in coeffs.chunks(4) {
            for (p, &a) in block[pos..][..4].iter_mut().zip(row.iter()) {
                *p = (a + i32::from(*p)).clamp(0, 255) as u8;
            }
            pos += stride;
        }
    }
    coeffs.fill(0);
}

// =============================================================================
// Fused Residual + DCT for single 4x4 block from flat u8 arrays
// =============================================================================

/// Fused residual computation + DCT for a single 4x4 block.
/// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.
///
/// Replaces: manual residual loop + dct4x4() in I4 inner loop.
/// Benefit: avoids i32 intermediate for residuals; computes directly in i16.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline(always)]
pub(crate) fn ftransform_from_u8_4x4(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    if let Some(token) = X64V3Token::summon() {
        ftransform_from_u8_4x4_entry(token, src, ref_)
    } else {
        ftransform_from_u8_4x4_scalar(src, ref_)
    }
}

/// Non-x86 fallback: try NEON on aarch64, else scalar
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
pub(crate) fn ftransform_from_u8_4x4(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        use archmage::{NeonToken, SimdToken};
        if let Some(token) = NeonToken::summon() {
            return super::transform_aarch64::ftransform_from_u8_4x4_neon(token, src, ref_);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    {
        use archmage::{SimdToken, Wasm128Token};
        if let Some(token) = Wasm128Token::summon() {
            return super::transform_wasm::ftransform_from_u8_4x4_wasm(token, src, ref_);
        }
    }
    ftransform_from_u8_4x4_scalar(src, ref_)
}

/// Scalar implementation of fused residual+DCT
pub(crate) fn ftransform_from_u8_4x4_scalar(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    let mut block = [0i32; 16];
    for i in 0..16 {
        block[i] = src[i] as i32 - ref_[i] as i32;
    }
    crate::common::transform::dct4x4_scalar(&mut block);
    block
}

/// Entry shim for ftransform_from_u8_4x4_sse2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[arcane]
fn ftransform_from_u8_4x4_entry(_token: X64V3Token, src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    ftransform_from_u8_4x4_sse2(_token, src, ref_)
}

/// SSE2 fused residual+DCT for single 4x4 block from flat u8[16] arrays.
/// Loads bytes, computes diff as i16, runs DCT pass1+pass2, outputs i32[16].
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[rite]
pub(crate) fn ftransform_from_u8_4x4_sse2(
    _token: X64V3Token,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    let zero = _mm_setzero_si128();

    // Load all 16 source bytes and 16 reference bytes as single 128-bit loads
    let src_all = simd_mem::_mm_loadu_si128(src);
    let ref_all = simd_mem::_mm_loadu_si128(ref_);

    // Zero-extend u8 to i16
    let src_lo = _mm_unpacklo_epi8(src_all, zero); // src[0..8] as i16
    let src_hi = _mm_unpackhi_epi8(src_all, zero); // src[8..16] as i16
    let ref_lo = _mm_unpacklo_epi8(ref_all, zero); // ref[0..8] as i16
    let ref_hi = _mm_unpackhi_epi8(ref_all, zero); // ref[8..16] as i16

    // Compute difference (src - ref) as i16 (range -255..255, fits i16)
    let diff_lo = _mm_sub_epi16(src_lo, ref_lo); // diff[0..8]
    let diff_hi = _mm_sub_epi16(src_hi, ref_hi); // diff[8..16]

    // diff_lo = [d00, d01, d02, d03, d10, d11, d12, d13]
    // diff_hi = [d20, d21, d22, d23, d30, d31, d32, d33]
    // Reorganize into libwebp's interleaved layout for ftransform_pass1_i16:
    // in01 = [r0c0, r0c1, r1c0, r1c1, r0c2, r0c3, r1c2, r1c3]
    // in23 = [r2c0, r2c1, r3c0, r3c1, r2c2, r2c3, r3c2, r3c3]

    // diff_lo already has rows 0,1 interleaved as pairs of 32-bit (2 i16 per pair)
    // Need: [d00,d01,d10,d11, d02,d03,d12,d13]
    // diff_lo = [d00,d01,d02,d03, d10,d11,d12,d13]
    // Interleave 32-bit words: unpacklo_epi32 + unpackhi_epi32
    let in01 = _mm_unpacklo_epi32(diff_lo, _mm_unpackhi_epi64(diff_lo, diff_lo));
    // = [d00,d01, d10,d11, d02,d03, d12,d13] ✓

    // Same for rows 2,3
    let in23 = _mm_unpacklo_epi32(diff_hi, _mm_unpackhi_epi64(diff_hi, diff_hi));
    // = [d20,d21, d30,d31, d22,d23, d32,d33] ✓

    // Forward transform pass 1 (rows)
    let (v01, v32) = ftransform_pass1_i16(_token, in01, in23);

    // Forward transform pass 2 (columns)
    let mut out16 = [0i16; 16];
    ftransform_pass2_i16(_token, &v01, &v32, &mut out16);

    // Convert i16 output to i32 using SIMD sign extension
    let out01 = simd_mem::_mm_loadu_si128(<&[i16; 8]>::try_from(&out16[0..8]).unwrap());
    let out23 = simd_mem::_mm_loadu_si128(<&[i16; 8]>::try_from(&out16[8..16]).unwrap());

    let sign01 = _mm_cmpgt_epi16(zero, out01);
    let sign23 = _mm_cmpgt_epi16(zero, out23);

    let out_0 = _mm_unpacklo_epi16(out01, sign01);
    let out_1 = _mm_unpackhi_epi16(out01, sign01);
    let out_2 = _mm_unpacklo_epi16(out23, sign23);
    let out_3 = _mm_unpackhi_epi16(out23, sign23);

    let mut result = [0i32; 16];
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut result[0..4]).unwrap(), out_0);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut result[4..8]).unwrap(), out_1);
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut result[8..12]).unwrap(),
        out_2,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut result[12..16]).unwrap(),
        out_3,
    );
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_intrinsics_matches_scalar() {
        let input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut scalar_block = input;
        crate::common::transform::dct4x4_scalar(&mut scalar_block);

        let mut intrinsics_block = input;
        dct4x4_intrinsics(&mut intrinsics_block);

        assert_eq!(
            scalar_block, intrinsics_block,
            "Intrinsics DCT doesn't match scalar.\nScalar: {:?}\nIntrinsics: {:?}",
            scalar_block, intrinsics_block
        );
    }

    #[test]
    fn test_idct_intrinsics_matches_scalar() {
        let mut input: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        crate::common::transform::dct4x4_scalar(&mut input);

        let mut scalar_block = input;
        crate::common::transform::idct4x4_scalar(&mut scalar_block);

        let mut intrinsics_block = input;
        idct4x4_intrinsics(&mut intrinsics_block);

        assert_eq!(
            scalar_block, intrinsics_block,
            "Intrinsics IDCT doesn't match scalar.\nScalar: {:?}\nIntrinsics: {:?}",
            scalar_block, intrinsics_block
        );
    }

    #[test]
    fn test_dct_two_intrinsics() {
        let input1: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        let input2: [i32; 16] = [
            100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100,
        ];

        let mut scalar1 = input1;
        let mut scalar2 = input2;
        crate::common::transform::dct4x4_scalar(&mut scalar1);
        crate::common::transform::dct4x4_scalar(&mut scalar2);

        let mut intrinsics1 = input1;
        let mut intrinsics2 = input2;
        dct4x4_two_intrinsics(&mut intrinsics1, &mut intrinsics2);

        assert_eq!(scalar1, intrinsics1, "Two-block intrinsics block1 mismatch");
        assert_eq!(scalar2, intrinsics2, "Two-block intrinsics block2 mismatch");
    }

    #[test]
    fn test_roundtrip() {
        let original: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut block = original;
        dct4x4_intrinsics(&mut block);
        idct4x4_intrinsics(&mut block);

        // Should get back original (or very close due to rounding)
        assert_eq!(original, block, "Roundtrip failed");
    }

    #[test]
    fn test_ftransform_from_u8_4x4() {
        // Create test data: flat 4x4 blocks
        let src: [u8; 16] = [
            100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220,
        ];
        let ref_: [u8; 16] = [128; 16];

        // Use fused function
        let simd_result = ftransform_from_u8_4x4(&src, &ref_);

        // Compute expected result using scalar path
        let mut expected = [0i32; 16];
        for i in 0..16 {
            expected[i] = src[i] as i32 - ref_[i] as i32;
        }
        crate::common::transform::dct4x4_scalar(&mut expected);

        assert_eq!(
            simd_result, expected,
            "ftransform_from_u8_4x4 mismatch.\nSIMD: {:?}\nExpected: {:?}",
            simd_result, expected
        );
    }

    #[test]
    fn test_ftransform_from_u8_4x4_varied() {
        // Test with varied pixel values
        let src: [u8; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];
        let ref_: [u8; 16] = [
            100, 50, 200, 80, 60, 130, 170, 140, 230, 210, 120, 220, 200, 20, 70, 110,
        ];

        let simd_result = ftransform_from_u8_4x4(&src, &ref_);

        let mut expected = [0i32; 16];
        for i in 0..16 {
            expected[i] = src[i] as i32 - ref_[i] as i32;
        }
        crate::common::transform::dct4x4_scalar(&mut expected);

        assert_eq!(
            simd_result, expected,
            "ftransform_from_u8_4x4 varied mismatch.\nSIMD: {:?}\nExpected: {:?}",
            simd_result, expected
        );
    }

    #[test]
    fn test_ftransform2_from_u8() {
        // Create test data: 2 adjacent 4x4 blocks (8 bytes per row)
        // Each row is 8 bytes, 4 rows total
        const STRIDE: usize = 16; // Use larger stride to test stride handling
        let mut src = [128u8; STRIDE * 4];
        let mut ref_ = [128u8; STRIDE * 4];

        // Set up some test pattern in first 8 columns of each row
        for y in 0..4 {
            for x in 0..8 {
                src[y * STRIDE + x] = (y * 8 + x) as u8 + 100;
                ref_[y * STRIDE + x] = 128;
            }
        }

        // Use ftransform2_from_u8
        let mut out_simd = [0i16; 32];
        ftransform2_from_u8(&src, &ref_, STRIDE, STRIDE, &mut out_simd);

        // Compute expected result using scalar path
        let mut expected = [0i16; 32];
        for block in 0..2 {
            let mut block_data = [0i32; 16];
            for y in 0..4 {
                for x in 0..4 {
                    let src_val = src[y * STRIDE + block * 4 + x] as i32;
                    let ref_val = ref_[y * STRIDE + block * 4 + x] as i32;
                    block_data[y * 4 + x] = src_val - ref_val;
                }
            }
            crate::common::transform::dct4x4_scalar(&mut block_data);
            for (i, &val) in block_data.iter().enumerate() {
                expected[block * 16 + i] = val as i16;
            }
        }

        assert_eq!(
            out_simd, expected,
            "ftransform2_from_u8 mismatch.\nSIMD: {:?}\nExpected: {:?}",
            out_simd, expected
        );
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benchmarks {
    use super::*;
    use test::Bencher;

    const TEST_BLOCKS: [[i32; 16]; 4] = [
        [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ],
        [
            100, 50, 25, 75, 200, 150, 100, 50, 25, 75, 125, 175, 225, 200, 150, 100,
        ],
        [
            12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12,
        ],
        [
            255, 0, 128, 64, 192, 32, 224, 16, 240, 8, 248, 4, 252, 2, 254, 1,
        ],
    ];

    #[bench]
    fn bench_dct_intrinsics(b: &mut Bencher) {
        b.iter(|| {
            for input in &TEST_BLOCKS {
                let mut block = *input;
                test::black_box(dct4x4_intrinsics(&mut block));
            }
        });
    }

    #[bench]
    fn bench_idct_intrinsics(b: &mut Bencher) {
        let mut dct_blocks = TEST_BLOCKS;
        for block in &mut dct_blocks {
            crate::common::transform::dct4x4_scalar(block);
        }

        b.iter(|| {
            for input in &dct_blocks {
                let mut block = *input;
                test::black_box(idct4x4_intrinsics(&mut block));
            }
        });
    }

    #[bench]
    fn bench_dct_two_intrinsics(b: &mut Bencher) {
        b.iter(|| {
            let mut block1 = TEST_BLOCKS[0];
            let mut block2 = TEST_BLOCKS[1];
            test::black_box(dct4x4_two_intrinsics(&mut block1, &mut block2));
            let mut block3 = TEST_BLOCKS[2];
            let mut block4 = TEST_BLOCKS[3];
            test::black_box(dct4x4_two_intrinsics(&mut block3, &mut block4));
        });
    }
}
