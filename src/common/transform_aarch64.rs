//! AArch64 NEON intrinsics for DCT/IDCT transforms.
//!
//! Ported from libwebp's enc_neon.c (FTransform) and dec_neon.c (ITransform).
//! Uses archmage's #[rite] for safe NEON intrinsics (proc macro wraps body in unsafe).
//!
//! All functions operate on the same i32[16] layout as the scalar code,
//! converting internally to i16 NEON vectors for SIMD processing.

use archmage::{NeonToken, arcane, rite};

use core::arch::aarch64::*;

use archmage::intrinsics::aarch64 as simd_mem;

// =============================================================================
// Forward DCT — ported from libwebp's FTransform_NEON (intrinsics version)
// =============================================================================

/// Forward DCT using NEON intrinsics.
/// Input/output: i32[16] in row-major order.

#[arcane]
pub(crate) fn dct4x4_neon(_token: NeonToken, block: &mut [i32; 16]) {
    dct4x4_neon_inner(_token, block);
}

#[rite]
fn dct4x4_neon_inner(_token: NeonToken, block: &mut [i32; 16]) {
    // Load i32[16] as 4 × i32x4, then narrow to i16x4
    let r0 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[0..4]).unwrap());
    let r1 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[4..8]).unwrap());
    let r2 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[8..12]).unwrap());
    let r3 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[12..16]).unwrap());

    let d0 = vmovn_s32(r0);
    let d1 = vmovn_s32(r1);
    let d2 = vmovn_s32(r2);
    let d3 = vmovn_s32(r3);

    // Transpose 4x4 for first pass
    let (t0t1, t3t2) = transpose_4x4_s16_neon(_token, d0, d1, d2, d3);

    // First DCT pass (includes internal transpose at end)
    let (p0p1, p3p2) = forward_pass_1_neon(_token, t0t1, t3t2);

    // Second DCT pass with final rounding → i32 output
    // Data is already transposed by pass 1's internal transpose
    let out = forward_pass_2_neon(_token, p0p1, p3p2);

    simd_mem::vst1q_s32(<&mut [i32; 4]>::try_from(&mut block[0..4]).unwrap(), out[0]);
    simd_mem::vst1q_s32(<&mut [i32; 4]>::try_from(&mut block[4..8]).unwrap(), out[1]);
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[8..12]).unwrap(),
        out[2],
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[12..16]).unwrap(),
        out[3],
    );
}

/// Transpose 4x4 i16 matrix (matches libwebp Transpose4x4_S16_NEON).
/// Input: 4 rows as i16x4. Output: (col01, col32) as two i16x8.
/// col01 = col0 | col1, col32 = col3 | col2

#[rite]
fn transpose_4x4_s16_neon(
    _token: NeonToken,
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
    d: int16x4_t,
) -> (int16x8_t, int16x8_t) {
    let ab = vtrn_s16(a, b);
    let cd = vtrn_s16(c, d);
    let tmp02 = vtrn_s32(vreinterpret_s32_s16(ab.0), vreinterpret_s32_s16(cd.0));
    let tmp13 = vtrn_s32(vreinterpret_s32_s16(ab.1), vreinterpret_s32_s16(cd.1));
    let out01 = vreinterpretq_s16_s64(vcombine_s64(
        vreinterpret_s64_s32(tmp02.0),
        vreinterpret_s64_s32(tmp13.0),
    ));
    let out32 = vreinterpretq_s16_s64(vcombine_s64(
        vreinterpret_s64_s32(tmp13.1),
        vreinterpret_s64_s32(tmp02.1),
    ));
    (out01, out32)
}

/// Forward DCT pass 1 (from libwebp FTransform_NEON, first pass).

#[rite]
fn forward_pass_1_neon(
    _token: NeonToken,
    d0d1: int16x8_t,
    d3d2: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let k_cst937 = vdupq_n_s32(937);
    let k_cst1812 = vdupq_n_s32(1812);

    let a0a1 = vaddq_s16(d0d1, d3d2);
    let a3a2 = vsubq_s16(d0d1, d3d2);

    let a0a1_2 = vshlq_n_s16::<3>(a0a1);
    let tmp0 = vadd_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));
    let tmp2 = vsub_s16(vget_low_s16(a0a1_2), vget_high_s16(a0a1_2));

    let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);

    let tmp1 = vshrn_n_s32::<9>(vaddq_s32(a2_p_a3, k_cst1812));
    let tmp3 = vshrn_n_s32::<9>(vaddq_s32(a3_m_a2, k_cst937));

    transpose_4x4_s16_neon(_token, tmp0, tmp1, tmp2, tmp3)
}

/// Forward DCT pass 2 with final rounding. Returns i32x4[4].

#[rite]
fn forward_pass_2_neon(_token: NeonToken, d0d1: int16x8_t, d3d2: int16x8_t) -> [int32x4_t; 4] {
    let k_cst12000 = vdupq_n_s32(12000 + (1 << 16));
    let k_cst51000 = vdupq_n_s32(51000);

    let a0a1 = vaddq_s16(d0d1, d3d2);
    let a3a2 = vsubq_s16(d0d1, d3d2);

    let a0_k7 = vadd_s16(vget_low_s16(a0a1), vdup_n_s16(7));
    let out0 = vshr_n_s16::<4>(vadd_s16(a0_k7, vget_high_s16(a0a1)));
    let out2 = vshr_n_s16::<4>(vsub_s16(a0_k7, vget_high_s16(a0a1)));

    let a3_2217 = vmull_n_s16(vget_low_s16(a3a2), 2217);
    let a2_2217 = vmull_n_s16(vget_high_s16(a3a2), 2217);
    let a2_p_a3 = vmlal_n_s16(a2_2217, vget_low_s16(a3a2), 5352);
    let a3_m_a2 = vmlsl_n_s16(a3_2217, vget_high_s16(a3a2), 5352);

    let tmp1 = vaddhn_s32(a2_p_a3, k_cst12000);
    let out3 = vaddhn_s32(a3_m_a2, k_cst51000);

    // out1 += (a3 != 0): ceq returns all-ones (0xFFFF = -1), so add -1 when eq
    let a3_eq_0 = vreinterpret_s16_u16(vceq_s16(vget_low_s16(a3a2), vdup_n_s16(0)));
    let out1 = vadd_s16(tmp1, a3_eq_0);

    // Widen i16x4 → i32x4
    [
        vmovl_s16(out0),
        vmovl_s16(out1),
        vmovl_s16(out2),
        vmovl_s16(out3),
    ]
}

// =============================================================================
// Inverse DCT — ported from libwebp's TransformPass_NEON + ITransformOne_NEON
// =============================================================================

// VP8 IDCT constants for vqdmulh doubling multiply-high:
// kC1 = 20091 (cos(PI/8)*sqrt(2)-1, fixed-point)
// kC2_half = 17734 (sin(PI/8)*sqrt(2) / 2 for vqdmulh)
const KC1: i16 = 20091;
const KC2_HALF: i16 = 17734; // 35468 / 2

/// Inverse DCT using NEON intrinsics.
/// Input/output: i32[16+] in row-major order.

#[arcane]
pub(crate) fn idct4x4_neon(_token: NeonToken, block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    idct4x4_neon_inner(_token, block);
}

#[rite]
fn idct4x4_neon_inner(_token: NeonToken, block: &mut [i32]) {
    // Load i32[16] and pack to i16x8 pairs
    let r0 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[0..4]).unwrap());
    let r1 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[4..8]).unwrap());
    let r2 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[8..12]).unwrap());
    let r3 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&block[12..16]).unwrap());

    let in01 = vcombine_s16(vmovn_s32(r0), vmovn_s32(r1));
    let in23 = vcombine_s16(vmovn_s32(r2), vmovn_s32(r3));

    // Two passes of TransformPass (vertical then horizontal)
    let (t01, t23) = itransform_pass_neon(_token, in01, in23);
    let (res01, res23) = itransform_pass_neon(_token, t01, t23);

    // Final: (val + 4) >> 3 rounding
    let four = vdupq_n_s16(4);
    let res01_r = vshrq_n_s16::<3>(vaddq_s16(res01, four));
    let res23_r = vshrq_n_s16::<3>(vaddq_s16(res23, four));

    // Widen i16 → i32 and store
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[0..4]).unwrap(),
        vmovl_s16(vget_low_s16(res01_r)),
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[4..8]).unwrap(),
        vmovl_s16(vget_high_s16(res01_r)),
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[8..12]).unwrap(),
        vmovl_s16(vget_low_s16(res23_r)),
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut block[12..16]).unwrap(),
        vmovl_s16(vget_high_s16(res23_r)),
    );
}

/// IDCT TransformPass — matches libwebp's TransformPass_NEON.
/// rows packed as (row0|row1, row2|row3) in i16x8 pairs.

#[rite]
fn itransform_pass_neon(
    _token: NeonToken,
    in01: int16x8_t,
    in23: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    // B1 = in4 | in12 (high halves)
    let b1 = vcombine_s16(vget_high_s16(in01), vget_high_s16(in23));

    // C0 = kC1 * B1: vqdmulh gives (B1*kC1*2)>>16, then vsraq adds B1>>1
    let c0 = vsraq_n_s16::<1>(b1, vqdmulhq_n_s16(b1, KC1));
    // C1 = kC2 * B1 / 2^16 via vqdmulh with kC2/2
    let c1 = vqdmulhq_n_s16(b1, KC2_HALF);

    // a = in0 + in8, b = in0 - in8
    let a = vqadd_s16(vget_low_s16(in01), vget_low_s16(in23));
    let b = vqsub_s16(vget_low_s16(in01), vget_low_s16(in23));

    // c = kC2*in4 - kC1*in12, d = kC1*in4 + kC2*in12
    let c = vqsub_s16(vget_low_s16(c1), vget_high_s16(c0));
    let d = vqadd_s16(vget_low_s16(c0), vget_high_s16(c1));

    let d0 = vcombine_s16(a, b); // a | b
    let d1 = vcombine_s16(d, c); // d | c

    let e0 = vqaddq_s16(d0, d1); // a+d | b+c
    let e_tmp = vqsubq_s16(d0, d1); // a-d | b-c
    let e1 = vcombine_s16(vget_high_s16(e_tmp), vget_low_s16(e_tmp)); // b-c | a-d

    // Transpose 8x2
    let tmp = vzipq_s16(e0, e1);
    let out = vzipq_s16(tmp.0, tmp.1);
    (out.0, out.1)
}

// =============================================================================
// Fused IDCT + add_residue (decoder hot path, in-place)
// =============================================================================

/// Fused IDCT + add residue + clear coefficients.
/// Performs IDCT on raw DCT coefficients, adds to prediction block in-place,
/// clamps to [0,255], and zeros the coefficient buffer.

#[arcane]
pub(crate) fn idct_add_residue_inplace_neon(
    _token: NeonToken,
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    if dc_only {
        idct_add_residue_dc_neon(_token, coeffs, block, y0, x0, stride);
    } else {
        idct_add_residue_full_neon(_token, coeffs, block, y0, x0, stride);
    }
    coeffs.fill(0);
}

/// DC-only fast path: add constant to all 16 pixels.

#[rite]
fn idct_add_residue_dc_neon(
    _token: NeonToken,
    coeffs: &[i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let dc = coeffs[0];
    let dc_adj = ((dc + 4) >> 3) as i16;
    let dc_vec = vdupq_n_s16(dc_adj);

    for row in 0..4 {
        let pos = (y0 + row) * stride + x0;
        let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
        // Load 4 prediction bytes, zero-extend to i16
        let pred_u32 = u32::from_ne_bytes(pred_bytes);
        let pred_v = vreinterpret_u8_u32(vmov_n_u32(pred_u32));
        let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_v));
        // Add DC
        let sum = vaddq_s16(pred_i16, dc_vec);
        // Saturate to u8
        let packed = vqmovun_s16(sum);
        // Store 4 bytes
        let result_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(packed));
        block[pos..pos + 4].copy_from_slice(&result_u32.to_ne_bytes());
    }
}

/// Full IDCT + add residue.

#[rite]
fn idct_add_residue_full_neon(
    _token: NeonToken,
    coeffs: &[i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    // Load and pack coefficients to i16
    let r0 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let r1 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let r2 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let r3 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in01 = vcombine_s16(vmovn_s32(r0), vmovn_s32(r1));
    let in23 = vcombine_s16(vmovn_s32(r2), vmovn_s32(r3));

    // Two IDCT passes
    let (t01, t23) = itransform_pass_neon(_token, in01, in23);
    let (res01, res23) = itransform_pass_neon(_token, t01, t23);

    // Process each row: load pred, add residual with rounding shift, saturate, store
    // libwebp uses vrsraq_n_s16 for: pred + (residual + rounding) >> 3
    let res_rows: [(int16x8_t, bool); 4] = [
        (res01, false), // row 0 = low half of res01
        (res01, true),  // row 1 = high half of res01
        (res23, false), // row 2 = low half of res23
        (res23, true),  // row 3 = high half of res23
    ];

    for (row_idx, &(res, use_high)) in res_rows.iter().enumerate() {
        let residual = if use_high {
            vcombine_s16(vget_high_s16(res), vget_high_s16(res))
        } else {
            vcombine_s16(vget_low_s16(res), vget_low_s16(res))
        };

        let pos = (y0 + row_idx) * stride + x0;
        let pred_bytes: [u8; 4] = block[pos..pos + 4].try_into().unwrap();
        let pred_u32 = u32::from_ne_bytes(pred_bytes);
        let pred_v = vreinterpret_u8_u32(vmov_n_u32(pred_u32));
        let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_v));

        // vrsraq: pred + (residual + 4) >> 3
        let out = vrsraq_n_s16::<3>(pred_i16, residual);
        let packed = vqmovun_s16(out);
        let result_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(packed));
        block[pos..pos + 4].copy_from_slice(&result_u32.to_ne_bytes());
    }
}

// =============================================================================
// Fused residual + DCT from u8 arrays (encoder hot path)
// =============================================================================

/// Fused residual computation + DCT for a single 4x4 block.
/// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.

#[arcane]
pub(crate) fn ftransform_from_u8_4x4_neon(
    _token: NeonToken,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    ftransform_from_u8_4x4_neon_inner(_token, src, ref_)
}

#[rite]
fn ftransform_from_u8_4x4_neon_inner(
    _token: NeonToken,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    // Load src and ref as u8x16
    let s = simd_mem::vld1q_u8(src);
    let r = simd_mem::vld1q_u8(ref_);

    // Compute difference as i16: rows 0-1 and rows 2-3
    let diff_01 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
    let diff_23 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

    let d0 = vget_low_s16(diff_01);
    let d1 = vget_high_s16(diff_01);
    let d2 = vget_low_s16(diff_23);
    let d3 = vget_high_s16(diff_23);

    // Transpose for first pass
    let (t0t1, t3t2) = transpose_4x4_s16_neon(_token, d0, d1, d2, d3);

    // First DCT pass (includes internal transpose at end)
    let (p0p1, p3p2) = forward_pass_1_neon(_token, t0t1, t3t2);

    // Second DCT pass — data already transposed by pass 1
    let out = forward_pass_2_neon(_token, p0p1, p3p2);

    let mut result = [0i32; 16];
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut result[0..4]).unwrap(),
        out[0],
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut result[4..8]).unwrap(),
        out[1],
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut result[8..12]).unwrap(),
        out[2],
    );
    simd_mem::vst1q_s32(
        <&mut [i32; 4]>::try_from(&mut result[12..16]).unwrap(),
        out[3],
    );
    result
}

// =============================================================================
// ftransform2 — two adjacent 4x4 blocks
// =============================================================================

/// Process two 4x4 blocks with forward DCT.

#[arcane]
pub(crate) fn ftransform2_neon(
    _token: NeonToken,
    src: &[u8],
    ref_: &[u8],
    src_stride: usize,
    ref_stride: usize,
    out: &mut [[i32; 16]; 2],
) {
    for blk in 0..2 {
        let sx = blk * 4;
        let mut s_flat = [0u8; 16];
        let mut r_flat = [0u8; 16];
        for row in 0..4 {
            s_flat[row * 4..row * 4 + 4].copy_from_slice(&src[row * src_stride + sx..][..4]);
            r_flat[row * 4..row * 4 + 4].copy_from_slice(&ref_[row * ref_stride + sx..][..4]);
        }
        out[blk] = ftransform_from_u8_4x4_neon(_token, &s_flat, &r_flat);
    }
}

// =============================================================================
// add_residue — add IDCT residuals to prediction block
// =============================================================================

/// Add residuals (i32[16]) to prediction block (u8) with saturation.
/// Processes 4 rows of 4 pixels each.

#[arcane]
pub(crate) fn add_residue_neon(
    _token: NeonToken,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    add_residue_neon_inner(_token, pblock, rblock, y0, x0, stride);
}

#[rite]
fn add_residue_neon_inner(
    _token: NeonToken,
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    // Load residuals as i32x4 and narrow to i16x4
    let r0_i32 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&rblock[0..4]).unwrap());
    let r1_i32 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&rblock[4..8]).unwrap());
    let r2_i32 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&rblock[8..12]).unwrap());
    let r3_i32 = simd_mem::vld1q_s32(<&[i32; 4]>::try_from(&rblock[12..16]).unwrap());

    let r0_i16 = vmovn_s32(r0_i32); // int16x4
    let r1_i16 = vmovn_s32(r1_i32);
    let r2_i16 = vmovn_s32(r2_i32);
    let r3_i16 = vmovn_s32(r3_i32);

    // Process 4 rows
    for (row, r_i16) in [(0usize, r0_i16), (1, r1_i16), (2, r2_i16), (3, r3_i16)] {
        let offset = (y0 + row) * stride + x0;

        // Load 4 prediction pixels, zero-extend to i16
        let mut pred_bytes = [0u8; 8]; // load 4, pad to 8
        pred_bytes[..4].copy_from_slice(&pblock[offset..offset + 4]);
        let pred_u8 = simd_mem::vld1_u8(&pred_bytes);
        let pred_i16 = vreinterpretq_s16_u16(vmovl_u8(pred_u8));

        // Add residual (only low 4 lanes matter)
        let sum = vaddq_s16(pred_i16, vcombine_s16(r_i16, vdup_n_s16(0)));

        // Saturate to u8
        let result = vqmovun_s16(sum);

        // Store 4 pixels back
        let mut out_bytes = [0u8; 8];
        simd_mem::vst1_u8(&mut out_bytes, result);
        pblock[offset..offset + 4].copy_from_slice(&out_bytes[..4]);
    }
}
