//! WASM SIMD128 YUV to RGB conversion for the decoder.
//!
//! Ported from the NEON version in yuv_neon.rs.
//! Processes 16 pixels at a time using 128-bit SIMD.

use archmage::{Wasm128Token, arcane};

use core::arch::wasm32::*;

// YUV to RGB conversion constants (matching libwebp):
// R = (19077 * y             + 26149 * v - 14234) >> 6
// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
// B = (19077 * y + 33050 * u             - 17685) >> 6
const Y_COEFF: i16 = 19077;
const V_TO_R: i16 = 26149;
const U_TO_G: i16 = 6419;
const V_TO_G: i16 = 13320;
const R_ROUNDER: i16 = -14234;
const G_ROUNDER: i16 = 8708;
const B_ROUNDER: i16 = -17685;
// B channel: 33050 = 32768 + 282. 32768/32768 = 1.0, handled by adding U directly.
const B_MULT_EXTRA: i16 = 282;

// =============================================================================
// Load/store helpers
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

// =============================================================================
// Core YUV → RGB conversion (8 pixels at a time, i16 arithmetic)
// =============================================================================

/// Multiply-high for i16: (a * b) >> 8, using extending multiply
/// This matches libwebp's mulhi pattern.
#[inline(always)]
fn mulhi_i16x8(a: v128, coeff: i16) -> v128 {
    let coeff_v = i16x8_splat(coeff);
    let lo = i32x4_extmul_low_i16x8(a, coeff_v);
    let hi = i32x4_extmul_high_i16x8(a, coeff_v);
    let lo_shifted = i32x4_shr(lo, 8);
    let hi_shifted = i32x4_shr(hi, 8);
    i16x8_narrow_i32x4(lo_shifted, hi_shifted)
}

/// Convert 8 YUV444 pixels to 8 RGB pixels (returned as three i16x8 vectors clamped to u8 range)
#[inline(always)]
fn convert_yuv_to_rgb_8(y: v128, u: v128, v: v128) -> (v128, v128, v128) {
    let y1 = mulhi_i16x8(y, Y_COEFF);
    let r0 = mulhi_i16x8(v, V_TO_R);
    let g0 = mulhi_i16x8(u, U_TO_G);
    let g1 = mulhi_i16x8(v, V_TO_G);
    let b0 = mulhi_i16x8(u, B_MULT_EXTRA);

    let r_round = i16x8_splat(R_ROUNDER);
    let g_round = i16x8_splat(G_ROUNDER);
    let b_round = i16x8_splat(B_ROUNDER);

    let r1 = i16x8_add_sat(y1, r_round);
    let g2 = i16x8_add_sat(y1, g_round);
    let b1 = i16x8_add_sat(y1, b_round);

    let r2 = i16x8_add_sat(r0, r1);
    let g3 = i16x8_add_sat(g0, g1);
    let b2 = i16x8_add_sat(b0, b1);
    let g4 = i16x8_sub_sat(g2, g3);
    let b3 = i16x8_add_sat(b2, u); // + U accounts for 32768/32768

    // Shift right 6 and clamp to [0, 255]
    let r = i16x8_shr(r2, 6);
    let g = i16x8_shr(g4, 6);
    let b = i16x8_shr(b3, 6);

    (r, g, b)
}

/// Convert 16 YUV444 pixels to RGB and store as interleaved RGBRGB... (48 bytes)
#[inline(always)]
fn convert_and_store_rgb16(y_vals: v128, u_vals: v128, v_vals: v128, rgb: &mut [u8; 48]) {
    // Process low 8 pixels: extend u8 to i16 (zero-extend, then treat as signed)
    let y_lo = u16x8_extend_low_u8x16(y_vals);
    let u_lo = u16x8_extend_low_u8x16(u_vals);
    let v_lo = u16x8_extend_low_u8x16(v_vals);
    let (r_lo, g_lo, b_lo) = convert_yuv_to_rgb_8(y_lo, u_lo, v_lo);

    // Process high 8 pixels
    let y_hi = u16x8_extend_high_u8x16(y_vals);
    let u_hi = u16x8_extend_high_u8x16(u_vals);
    let v_hi = u16x8_extend_high_u8x16(v_vals);
    let (r_hi, g_hi, b_hi) = convert_yuv_to_rgb_8(y_hi, u_hi, v_hi);

    // Pack i16 → u8 with saturation
    let r16 = u8x16_narrow_i16x8(r_lo, r_hi);
    let g16 = u8x16_narrow_i16x8(g_lo, g_hi);
    let b16 = u8x16_narrow_i16x8(b_lo, b_hi);

    // Interleave RGB: we need [R0,G0,B0, R1,G1,B1, ...]
    // WASM doesn't have vst3q_u8, so we extract lanes individually.
    // We process in chunks of 16 bytes (which holds ~5.3 RGB triplets).
    // Instead, interleave by extracting lanes.
    for i in 0..16 {
        rgb[i * 3] = u8x16_extract_lane_runtime(r16, i);
        rgb[i * 3 + 1] = u8x16_extract_lane_runtime(g16, i);
        rgb[i * 3 + 2] = u8x16_extract_lane_runtime(b16, i);
    }
}

/// Extract a lane from u8x16 at runtime index
#[inline(always)]
fn u8x16_extract_lane_runtime(v: v128, i: usize) -> u8 {
    match i {
        0 => u8x16_extract_lane::<0>(v),
        1 => u8x16_extract_lane::<1>(v),
        2 => u8x16_extract_lane::<2>(v),
        3 => u8x16_extract_lane::<3>(v),
        4 => u8x16_extract_lane::<4>(v),
        5 => u8x16_extract_lane::<5>(v),
        6 => u8x16_extract_lane::<6>(v),
        7 => u8x16_extract_lane::<7>(v),
        8 => u8x16_extract_lane::<8>(v),
        9 => u8x16_extract_lane::<9>(v),
        10 => u8x16_extract_lane::<10>(v),
        11 => u8x16_extract_lane::<11>(v),
        12 => u8x16_extract_lane::<12>(v),
        13 => u8x16_extract_lane::<13>(v),
        14 => u8x16_extract_lane::<14>(v),
        15 => u8x16_extract_lane::<15>(v),
        _ => 0,
    }
}

// =============================================================================
// Upsampling
// =============================================================================

/// Upsample 8 chroma pixels from two rows into 16 upsampled values.
/// Implements libwebp's UPSAMPLE_16PIXELS using WASM SIMD.
///
/// diag1[i] = (9*a[i] + 3*b[i] + 3*c[i] + d[i] + 8) / 16
/// diag2[i] = (9*b[i] + 3*a[i] + 3*d[i] + c[i] + 8) / 16
#[inline(always)]
fn upsample_16pixels(a: v128, b: v128, c: v128, d: v128) -> v128 {
    // a, b, c, d are u8 values in low 8 lanes (zero-extended to u16 for arithmetic)
    let a16 = u16x8_extend_low_u8x16(a);
    let b16 = u16x8_extend_low_u8x16(b);
    let c16 = u16x8_extend_low_u8x16(c);
    let d16 = u16x8_extend_low_u8x16(d);

    // ad = a + d, bc = b + c
    let ad = i16x8_add(a16, d16);
    let bc = i16x8_add(b16, c16);
    let abcd = i16x8_add(ad, bc);

    // al = abcd + 2*ad = 3a+b+c+3d, bl = abcd + 2*bc = a+3b+3c+d
    let al = i16x8_add(abcd, i16x8_shl(ad, 1));
    let bl = i16x8_add(abcd, i16x8_shl(bc, 1));

    // Divide by 8
    let diag2 = u16x8_shr(al, 3); // (3a + b + c + 3d) / 8
    let diag1 = u16x8_shr(bl, 3); // (a + 3b + 3c + d) / 8

    // Pack back to u8
    let diag2_u8 = u8x16_narrow_i16x8(diag2, diag2);
    let diag1_u8 = u8x16_narrow_i16x8(diag1, diag1);

    // Final rounding average with original values
    let a_out = u8x16_avgr(a, diag1_u8); // (9a + 3b + 3c + d + 8) / 16
    let b_out = u8x16_avgr(b, diag2_u8); // (3a + 9b + c + 3d + 8) / 16

    // Interleave: [A[0], B[0], A[1], B[1], ...]
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a_out, b_out)
}

// =============================================================================
// Public entry points
// =============================================================================

/// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV→RGB.
#[arcane]
pub(crate) fn fancy_upsample_16_pairs_wasm(
    _token: Wasm128Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    let y: &[u8; 32] = y_row[..32].try_into().unwrap();
    let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
    let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
    let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
    let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
    let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

    // Load U rows: a=row1[0..8], b=row1[1..9], etc.
    let u_a0 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[0..8]).unwrap());
    let u_b0 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[1..9]).unwrap());
    let u_c0 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[0..8]).unwrap());
    let u_d0 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[1..9]).unwrap());

    let u_a1 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[8..16]).unwrap());
    let u_b1 = load_u8x8_low(<&[u8; 8]>::try_from(&u1[9..17]).unwrap());
    let u_c1 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[8..16]).unwrap());
    let u_d1 = load_u8x8_low(<&[u8; 8]>::try_from(&u2[9..17]).unwrap());

    let v_a0 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[0..8]).unwrap());
    let v_b0 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[1..9]).unwrap());
    let v_c0 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[0..8]).unwrap());
    let v_d0 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[1..9]).unwrap());

    let v_a1 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[8..16]).unwrap());
    let v_b1 = load_u8x8_low(<&[u8; 8]>::try_from(&v1[9..17]).unwrap());
    let v_c1 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[8..16]).unwrap());
    let v_d1 = load_u8x8_low(<&[u8; 8]>::try_from(&v2[9..17]).unwrap());

    let u_up0 = upsample_16pixels(u_a0, u_b0, u_c0, u_d0);
    let u_up1 = upsample_16pixels(u_a1, u_b1, u_c1, u_d1);
    let v_up0 = upsample_16pixels(v_a0, v_b0, v_c0, v_d0);
    let v_up1 = upsample_16pixels(v_a1, v_b1, v_c1, v_d1);

    let y0 = load_u8x16(<&[u8; 16]>::try_from(&y[0..16]).unwrap());
    let y1 = load_u8x16(<&[u8; 16]>::try_from(&y[16..32]).unwrap());

    let (rgb_0, rgb_1) = out.split_at_mut(48);
    convert_and_store_rgb16(y0, u_up0, v_up0, <&mut [u8; 48]>::try_from(rgb_0).unwrap());
    convert_and_store_rgb16(y1, u_up1, v_up1, <&mut [u8; 48]>::try_from(rgb_1).unwrap());
}

/// Process 8 pixel pairs (16 Y pixels) with fancy upsampling and YUV→RGB.
#[arcane]
pub(crate) fn fancy_upsample_8_pairs_wasm(
    _token: Wasm128Token,
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

    let u_a = load_u8x8_low(<&[u8; 8]>::try_from(&u1[0..8]).unwrap());
    let u_b = load_u8x8_low(<&[u8; 8]>::try_from(&u1[1..9]).unwrap());
    let u_c = load_u8x8_low(<&[u8; 8]>::try_from(&u2[0..8]).unwrap());
    let u_d = load_u8x8_low(<&[u8; 8]>::try_from(&u2[1..9]).unwrap());

    let v_a = load_u8x8_low(<&[u8; 8]>::try_from(&v1[0..8]).unwrap());
    let v_b = load_u8x8_low(<&[u8; 8]>::try_from(&v1[1..9]).unwrap());
    let v_c = load_u8x8_low(<&[u8; 8]>::try_from(&v2[0..8]).unwrap());
    let v_d = load_u8x8_low(<&[u8; 8]>::try_from(&v2[1..9]).unwrap());

    let u_up = upsample_16pixels(u_a, u_b, u_c, u_d);
    let v_up = upsample_16pixels(v_a, v_b, v_c, v_d);
    let y_vec = load_u8x16(y);

    convert_and_store_rgb16(y_vec, u_up, v_up, out);
}

/// Convert a row of YUV420 to RGB using WASM SIMD.
/// Simple (non-fancy) upsampling: each U/V value maps to 2 adjacent Y pixels.
#[arcane]
pub(crate) fn yuv420_to_rgb_row_wasm(
    _token: Wasm128Token,
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

    while n + 16 <= len {
        let y_arr = <&[u8; 16]>::try_from(&y[n..n + 16]).unwrap();
        let u_arr = <&[u8; 8]>::try_from(&u[n / 2..n / 2 + 8]).unwrap();
        let v_arr = <&[u8; 8]>::try_from(&v[n / 2..n / 2 + 8]).unwrap();
        let dst_arr = <&mut [u8; 48]>::try_from(&mut dst[n * 3..n * 3 + 48]).unwrap();

        let y_vec = load_u8x16(y_arr);

        // Replicate each U/V for 2 adjacent Y pixels: [u0,u0,u1,u1,...u7,u7]
        let u_lo = load_u8x8_low(u_arr);
        let v_lo = load_u8x8_low(v_arr);
        let u_dup = i8x16_shuffle::<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>(u_lo, u_lo);
        let v_dup = i8x16_shuffle::<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>(v_lo, v_lo);

        convert_and_store_rgb16(y_vec, u_dup, v_dup, dst_arr);
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
