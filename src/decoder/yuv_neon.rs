//! NEON-optimized YUV to RGB conversion for aarch64.
//!
//! Ported from libwebp's upsampling_neon.c for efficient YUV→RGB with native
//! interleaved stores (vst3q_u8). Processes 16 pixels at a time.

use archmage::{arcane, rite, NeonToken};

use archmage::intrinsics::aarch64 as simd_mem;

use core::arch::aarch64::*;

// YUV to RGB conversion constants (matching libwebp's upsampling_neon.c):
// These are used with vqdmulhq_lane_s16 which computes (a*b*2) >> 16
// So effective multiply is coefficient/32768.
//
// R = (19077 * y             + 26149 * v - 14234) >> 6
// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
// B = (19077 * y + 33050 * u             - 17685) >> 6
//
// With vqdmulhq: val << 7 gives val*128, then vqdmulhq(val*128, coeff) = (val*128*coeff*2) >> 16
// = (val * coeff) >> 8, which matches mulhi(val, coeff).
// Then shift right by 6 for final result.
const K_COEFFS1: [i16; 4] = [19077, 26149, 6419, 13320];

// Rounders for YUV→RGB (same as libwebp)
const R_ROUNDER: i16 = -14234;
const G_ROUNDER: i16 = 8708;
const B_ROUNDER: i16 = -17685;

// B channel uses 33050 which doesn't fit in i16 for vqdmulhq.
// libwebp splits it: 33050 = 32768 + 282, and 32768 * x / 32768 = x
// So B0 = vqdmulhq_n_s16(U0, 282) and then B3 = B2 + U0
// (the U0 addition accounts for the 32768/32768 = 1.0 factor)
const B_MULT_EXTRA: i16 = 282;

/// Convert 16 YUV444 pixels to RGB and store as interleaved RGBRGB...
/// Uses NEON vst3q_u8 for hardware-accelerated interleaving.
///
/// Input: 16 Y values, 16 U values (upsampled), 16 V values (upsampled)
/// Output: 48 bytes of interleaved RGB

#[rite]
fn convert_and_store_rgb16_neon(
    _token: NeonToken,
    y_vals: uint8x16_t,
    u_vals: uint8x16_t,
    v_vals: uint8x16_t,
    rgb: &mut [u8; 48],
) {
    let coeff1 = simd_mem::vld1_s16(&K_COEFFS1);
    let r_rounder = vdupq_n_s16(R_ROUNDER);
    let g_rounder = vdupq_n_s16(G_ROUNDER);
    let b_rounder = vdupq_n_s16(B_ROUNDER);

    // Process low 8 pixels
    let y_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(y_vals), 7));
    let u_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(u_vals), 7));
    let v_lo = vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(v_vals), 7));

    let y1_lo = vqdmulhq_lane_s16(y_lo, coeff1, 0); // Y * 19077
    let r0_lo = vqdmulhq_lane_s16(v_lo, coeff1, 1); // V * 26149
    let g0_lo = vqdmulhq_lane_s16(u_lo, coeff1, 2); // U * 6419
    let g1_lo = vqdmulhq_lane_s16(v_lo, coeff1, 3); // V * 13320
    let b0_lo = vqdmulhq_n_s16(u_lo, B_MULT_EXTRA); // U * 282

    let r1_lo = vqaddq_s16(y1_lo, r_rounder);
    let g2_lo = vqaddq_s16(y1_lo, g_rounder);
    let b1_lo = vqaddq_s16(y1_lo, b_rounder);

    let r2_lo = vqaddq_s16(r0_lo, r1_lo);
    let g3_lo = vqaddq_s16(g0_lo, g1_lo);
    let b2_lo = vqaddq_s16(b0_lo, b1_lo);
    let g4_lo = vqsubq_s16(g2_lo, g3_lo);
    let b3_lo = vqaddq_s16(b2_lo, u_lo); // + U accounts for 32768/32768

    let r_lo = vqshrun_n_s16(r2_lo, 6);
    let g_lo = vqshrun_n_s16(g4_lo, 6);
    let b_lo = vqshrun_n_s16(b3_lo, 6);

    // Process high 8 pixels
    let y_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(y_vals), 7));
    let u_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(u_vals), 7));
    let v_hi = vreinterpretq_s16_u16(vshll_n_u8(vget_high_u8(v_vals), 7));

    let y1_hi = vqdmulhq_lane_s16(y_hi, coeff1, 0);
    let r0_hi = vqdmulhq_lane_s16(v_hi, coeff1, 1);
    let g0_hi = vqdmulhq_lane_s16(u_hi, coeff1, 2);
    let g1_hi = vqdmulhq_lane_s16(v_hi, coeff1, 3);
    let b0_hi = vqdmulhq_n_s16(u_hi, B_MULT_EXTRA);

    let r1_hi = vqaddq_s16(y1_hi, r_rounder);
    let g2_hi = vqaddq_s16(y1_hi, g_rounder);
    let b1_hi = vqaddq_s16(y1_hi, b_rounder);

    let r2_hi = vqaddq_s16(r0_hi, r1_hi);
    let g3_hi = vqaddq_s16(g0_hi, g1_hi);
    let b2_hi = vqaddq_s16(b0_hi, b1_hi);
    let g4_hi = vqsubq_s16(g2_hi, g3_hi);
    let b3_hi = vqaddq_s16(b2_hi, u_hi);

    let r_hi = vqshrun_n_s16(r2_hi, 6);
    let g_hi = vqshrun_n_s16(g4_hi, 6);
    let b_hi = vqshrun_n_s16(b3_hi, 6);

    // Combine low/high halves
    let r16 = vcombine_u8(r_lo, r_hi);
    let g16 = vcombine_u8(g_lo, g_hi);
    let b16 = vcombine_u8(b_lo, b_hi);

    // Interleaved store: RGBRGBRGB... (48 bytes for 16 pixels)
    let rgb_val = uint8x16x3_t(r16, g16, b16);
    simd_mem::vst3q_u8(rgb, rgb_val);
}

/// Upsample 8 chroma pixels from two rows into 16 upsampled values.
/// Implements libwebp's UPSAMPLE_16PIXELS macro using NEON.
///
/// Input: 9 pixels from each of two rows (a[0..9], c[0..9])
/// The overlap (9 vs 8) provides the neighbor for bilinear filtering.
///
/// Output: 16 upsampled values stored interleaved (diag1[0], diag2[0], diag1[1], ...)
/// where diag1[i] = (9*a[i] + 3*b[i] + 3*c[i] + d[i] + 8) / 16
///       diag2[i] = (9*b[i] + 3*a[i] + 3*d[i] + c[i] + 8) / 16
/// a=row1[0..8], b=row1[1..9], c=row2[0..8], d=row2[1..9]

#[rite]
fn upsample_16pixels_neon(
    _token: NeonToken,
    a: uint8x8_t,
    b: uint8x8_t,
    c: uint8x8_t,
    d: uint8x8_t,
) -> uint8x16_t {
    // Compute (a + b + c + d)
    let ad = vaddl_u8(a, d); // u16
    let bc = vaddl_u8(b, c); // u16
    let abcd = vaddq_u16(ad, bc); // u16

    // 3a + b + c + 3d = abcd + 2*ad
    let al = vaddq_u16(abcd, vshlq_n_u16(ad, 1));
    // a + 3b + 3c + d = abcd + 2*bc
    let bl = vaddq_u16(abcd, vshlq_n_u16(bc, 1));

    // Divide by 8 (shift right 3)
    let diag2 = vshrn_n_u16(al, 3); // (3a + b + c + 3d) / 8
    let diag1 = vshrn_n_u16(bl, 3); // (a + 3b + 3c + d) / 8

    // Final result with rounding: vrhadd = (a + b + 1) / 2
    let a_out = vrhadd_u8(a, diag1); // (9a + 3b + 3c + d + 8) / 16
    let b_out = vrhadd_u8(b, diag2); // (3a + 9b + c + 3d + 8) / 16

    // Interleave A and B: [A[0], B[0], A[1], B[1], ...]
    // vzip_u8 on D-registers gives two D-registers; combine into Q-register
    let interleaved = vzip_u8(a_out, b_out);
    vcombine_u8(interleaved.0, interleaved.1)
}

/// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV→RGB.
/// This is the NEON equivalent of fancy_upsample_16_pairs_with_token.
///
/// Takes: 32 Y bytes, 17 U/V bytes per row (overlapping window), 96 RGB bytes output.

#[arcane]
pub(crate) fn fancy_upsample_16_pairs_neon(
    _token: NeonToken,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // Bounds check at entry
    let y: &[u8; 32] = y_row[..32].try_into().unwrap();
    let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
    let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
    let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
    let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
    let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

    fancy_upsample_16_pairs_inner_neon(_token, y, u1, u2, v1, v2, out);
}

#[rite]
fn fancy_upsample_16_pairs_inner_neon(
    _token: NeonToken,
    y_row: &[u8; 32],
    u_row_1: &[u8; 17],
    u_row_2: &[u8; 17],
    v_row_1: &[u8; 17],
    v_row_2: &[u8; 17],
    rgb: &mut [u8; 96],
) {
    // Load U rows: a=row1[0..8], b=row1[1..9], then second batch [8..16], [9..17]
    let u_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[0..8]).unwrap());
    let u_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[1..9]).unwrap());
    let u_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[0..8]).unwrap());
    let u_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[1..9]).unwrap());

    let u_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[8..16]).unwrap());
    let u_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[9..17]).unwrap());
    let u_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[8..16]).unwrap());
    let u_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[9..17]).unwrap());

    // Same for V
    let v_a0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[0..8]).unwrap());
    let v_b0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[1..9]).unwrap());
    let v_c0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[0..8]).unwrap());
    let v_d0 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[1..9]).unwrap());

    let v_a1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[8..16]).unwrap());
    let v_b1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[9..17]).unwrap());
    let v_c1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[8..16]).unwrap());
    let v_d1 = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[9..17]).unwrap());

    // Upsample: 8 chroma → 16 Y-aligned values for each batch
    let u_up0 = upsample_16pixels_neon(_token, u_a0, u_b0, u_c0, u_d0);
    let u_up1 = upsample_16pixels_neon(_token, u_a1, u_b1, u_c1, u_d1);
    let v_up0 = upsample_16pixels_neon(_token, v_a0, v_b0, v_c0, v_d0);
    let v_up1 = upsample_16pixels_neon(_token, v_a1, v_b1, v_c1, v_d1);

    // Load Y
    let y0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&y_row[0..16]).unwrap());
    let y1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&y_row[16..32]).unwrap());

    // Convert and store first 16 pixels
    let (rgb_0, rgb_1) = rgb.split_at_mut(48);
    convert_and_store_rgb16_neon(
        _token,
        y0,
        u_up0,
        v_up0,
        <&mut [u8; 48]>::try_from(rgb_0).unwrap(),
    );
    // Convert and store second 16 pixels
    convert_and_store_rgb16_neon(
        _token,
        y1,
        u_up1,
        v_up1,
        <&mut [u8; 48]>::try_from(rgb_1).unwrap(),
    );
}

/// Process 8 pixel pairs (16 Y pixels) with fancy upsampling and YUV→RGB.
/// This is the NEON equivalent of fancy_upsample_8_pairs_with_token.
///
/// Takes: 16 Y bytes, 9 U/V bytes per row (overlapping window), 48 RGB bytes output.

#[arcane]
pub(crate) fn fancy_upsample_8_pairs_neon(
    _token: NeonToken,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    let y: &[u8; 16] = y_row[..16].try_into().unwrap();
    let u1: &[u8; 9] = u_row_1[..9].try_into().unwrap();
    let u2: &[u8; 9] = u_row_2[..9].try_into().unwrap();
    let v1: &[u8; 9] = v_row_1[..9].try_into().unwrap();
    let v2: &[u8; 9] = v_row_2[..9].try_into().unwrap();
    let out: &mut [u8; 48] = (&mut rgb[..48]).try_into().unwrap();

    fancy_upsample_8_pairs_inner_neon(_token, y, u1, u2, v1, v2, out);
}

#[rite]
fn fancy_upsample_8_pairs_inner_neon(
    _token: NeonToken,
    y_row: &[u8; 16],
    u_row_1: &[u8; 9],
    u_row_2: &[u8; 9],
    v_row_1: &[u8; 9],
    v_row_2: &[u8; 9],
    rgb: &mut [u8; 48],
) {
    let u_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[0..8]).unwrap());
    let u_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_1[1..9]).unwrap());
    let u_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[0..8]).unwrap());
    let u_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_row_2[1..9]).unwrap());

    let v_a = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[0..8]).unwrap());
    let v_b = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_1[1..9]).unwrap());
    let v_c = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[0..8]).unwrap());
    let v_d = simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_row_2[1..9]).unwrap());

    let u_up = upsample_16pixels_neon(_token, u_a, u_b, u_c, u_d);
    let v_up = upsample_16pixels_neon(_token, v_a, v_b, v_c, v_d);

    let y_vec = simd_mem::vld1q_u8(y_row);

    convert_and_store_rgb16_neon(_token, y_vec, u_up, v_up, rgb);
}

/// Convert a row of YUV420 to RGB using NEON.
/// Simple (non-fancy) upsampling: each U/V value maps to 2 adjacent Y pixels.
/// Processes 16 pixels at a time.

#[arcane]
pub(crate) fn yuv420_to_rgb_row_neon(
    _token: NeonToken,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    dst: &mut [u8],
) {
    let len = y.len();
    assert!(u.len() >= len.div_ceil(2));
    assert!(v.len() >= len.div_ceil(2));
    assert!(dst.len() >= len * 3);

    let mut n = 0usize;

    // Process 16 pixels at a time
    while n + 16 <= len {
        let y_arr = <&[u8; 16]>::try_from(&y[n..n + 16]).unwrap();
        let u_arr = <&[u8; 8]>::try_from(&u[n / 2..n / 2 + 8]).unwrap();
        let v_arr = <&[u8; 8]>::try_from(&v[n / 2..n / 2 + 8]).unwrap();
        let dst_arr = <&mut [u8; 48]>::try_from(&mut dst[n * 3..n * 3 + 48]).unwrap();

        let y_vec = simd_mem::vld1q_u8(y_arr);

        // Replicate each U/V for 2 adjacent Y pixels: [u0,u0,u1,u1,...u7,u7]
        let u_d = simd_mem::vld1_u8(u_arr);
        let v_d = simd_mem::vld1_u8(v_arr);
        let u_zip = vzip_u8(u_d, u_d);
        let v_zip = vzip_u8(v_d, v_d);
        let u_dup = vcombine_u8(u_zip.0, u_zip.1);
        let v_dup = vcombine_u8(v_zip.0, v_zip.1);

        convert_and_store_rgb16_neon(_token, y_vec, u_dup, v_dup, dst_arr);
        n += 16;
    }

    // Scalar fallback for remainder
    while n < len {
        let y_val = y[n];
        let u_val = u[n / 2];
        let v_val = v[n / 2];
        let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
        dst[n * 3] = r;
        dst[n * 3 + 1] = g;
        dst[n * 3 + 2] = b;
        n += 1;
    }
}

/// Scalar fallback for YUV to RGB conversion (single pixel).
#[inline]
fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    fn mulhi(val: u8, coeff: u16) -> i32 {
        ((u32::from(val) * u32::from(coeff)) >> 8) as i32
    }
    fn clip(v: i32) -> u8 {
        (v >> 6).clamp(0, 255) as u8
    }
    let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
    let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
    let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
    (r, g, b)
}
