//! AArch64 NEON loop filter implementations for VP8 decoder.
//!
//! Ported from libwebp's `src/dsp/dec_neon.c` (intrinsics version).
//! Uses archmage's #[rite]/#[arcane] for safe NEON intrinsics.
//!
//! All filter functions operate on the same buffer layout as the x86 SSE2 code:
//! - Vertical filters process 16 consecutive columns across a horizontal edge
//! - Horizontal filters process 16 consecutive rows across a vertical edge
//! - UV filters pack U and V 8-pixel halves into 16-wide NEON registers

#![allow(clippy::too_many_arguments)]

use archmage::{NeonToken, arcane, rite};

use core::arch::aarch64::*;

use archmage::intrinsics::aarch64 as simd_mem;

// =============================================================================
// Core filter helpers (ported from libwebp dec_neon.c)
// =============================================================================

/// NeedsFilter: returns mask where 2*|p0-q0| + |p1-q1|/2 <= thresh

#[rite]
fn needs_filter_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    thresh: i32,
) -> uint8x16_t {
    let thresh_v = vdupq_n_u8(thresh as u8);
    let a_p0_q0 = vabdq_u8(p0, q0); // abs(p0 - q0)
    let a_p1_q1 = vabdq_u8(p1, q1); // abs(p1 - q1)
    let a_p0_q0_2 = vqaddq_u8(a_p0_q0, a_p0_q0); // 2 * abs(p0-q0) saturating
    let a_p1_q1_2 = vshrq_n_u8::<1>(a_p1_q1); // abs(p1-q1) / 2
    let sum = vqaddq_u8(a_p0_q0_2, a_p1_q1_2);
    vcgeq_u8(thresh_v, sum) // mask where thresh >= sum
}

/// NeedsFilter2: extended filter check for normal (non-simple) filter
/// Checks NeedsFilter AND all adjacent differences <= ithresh

#[rite]
fn needs_filter2_neon(
    _token: NeonToken,
    p3: uint8x16_t,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    q3: uint8x16_t,
    ithresh: i32,
    thresh: i32,
) -> uint8x16_t {
    let ithresh_v = vdupq_n_u8(ithresh as u8);
    let a_p3_p2 = vabdq_u8(p3, p2);
    let a_p2_p1 = vabdq_u8(p2, p1);
    let a_p1_p0 = vabdq_u8(p1, p0);
    let a_q3_q2 = vabdq_u8(q3, q2);
    let a_q2_q1 = vabdq_u8(q2, q1);
    let a_q1_q0 = vabdq_u8(q1, q0);
    let max1 = vmaxq_u8(a_p3_p2, a_p2_p1);
    let max2 = vmaxq_u8(a_p1_p0, a_q3_q2);
    let max3 = vmaxq_u8(a_q2_q1, a_q1_q0);
    let max12 = vmaxq_u8(max1, max2);
    let max123 = vmaxq_u8(max12, max3);
    let mask2 = vcgeq_u8(ithresh_v, max123);
    let mask1 = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    vandq_u8(mask1, mask2)
}

/// NeedsHev: returns mask where max(|p1-p0|, |q1-q0|) > hev_thresh

#[rite]
fn needs_hev_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    hev_thresh: i32,
) -> uint8x16_t {
    let hev_thresh_v = vdupq_n_u8(hev_thresh as u8);
    let a_p1_p0 = vabdq_u8(p1, p0);
    let a_q1_q0 = vabdq_u8(q1, q0);
    let a_max = vmaxq_u8(a_p1_p0, a_q1_q0);
    vcgtq_u8(a_max, hev_thresh_v) // mask where max > hev_thresh
}

/// Convert unsigned to signed by XOR with 0x80

#[rite]
fn flip_sign_neon(_token: NeonToken, v: uint8x16_t) -> int8x16_t {
    let sign_bit = vdupq_n_u8(0x80);
    vreinterpretq_s8_u8(veorq_u8(v, sign_bit))
}

/// Convert signed back to unsigned by XOR with 0x80

#[rite]
fn flip_sign_back_neon(_token: NeonToken, v: int8x16_t) -> uint8x16_t {
    let sign_bit = vdupq_n_s8(-128); // 0x80 as i8
    vreinterpretq_u8_s8(veorq_s8(v, sign_bit))
}

/// GetBaseDelta: compute (p1-q1) + 3*(q0-p0) with saturation

#[rite]
fn get_base_delta_neon(
    _token: NeonToken,
    p1s: int8x16_t,
    p0s: int8x16_t,
    q0s: int8x16_t,
    q1s: int8x16_t,
) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0s, p0s); // (q0 - p0) saturating
    let p1_q1 = vqsubq_s8(p1s, q1s); // (p1 - q1) saturating
    let s1 = vqaddq_s8(p1_q1, q0_p0); // (p1-q1) + 1*(q0-p0)
    let s2 = vqaddq_s8(q0_p0, s1); // (p1-q1) + 2*(q0-p0)
    vqaddq_s8(q0_p0, s2) // (p1-q1) + 3*(q0-p0)
}

/// GetBaseDelta0: compute 3*(q0-p0) with saturation (no p1/q1)

#[rite]
fn get_base_delta0_neon(_token: NeonToken, p0s: int8x16_t, q0s: int8x16_t) -> int8x16_t {
    let q0_p0 = vqsubq_s8(q0s, p0s);
    let s1 = vqaddq_s8(q0_p0, q0_p0); // 2*(q0-p0)
    vqaddq_s8(q0_p0, s1) // 3*(q0-p0)
}

// =============================================================================
// Filter application functions
// =============================================================================

/// ApplyFilter2NoFlip: 2-tap filter without sign flip back (stays in signed domain)

#[rite]
fn apply_filter2_no_flip_neon(
    _token: NeonToken,
    p0s: int8x16_t,
    q0s: int8x16_t,
    delta: int8x16_t,
) -> (int8x16_t, int8x16_t) {
    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta_p3 = vqaddq_s8(delta, k3);
    let delta_p4 = vqaddq_s8(delta, k4);
    let delta3 = vshrq_n_s8::<3>(delta_p3);
    let delta4 = vshrq_n_s8::<3>(delta_p4);
    let op0 = vqaddq_s8(p0s, delta3);
    let oq0 = vqsubq_s8(q0s, delta4);
    (op0, oq0)
}

/// DoFilter2: simple 2-tap filter (used for simple loop filter)
/// Returns (filtered_p0, filtered_q0) as unsigned

#[rite]
fn do_filter2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let p1s = flip_sign_neon(_token, p1);
    let p0s = flip_sign_neon(_token, p0);
    let q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let delta0 = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);
    let delta1 = vandq_s8(delta0, vreinterpretq_s8_u8(mask));

    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta_p3 = vqaddq_s8(delta1, k3);
    let delta_p4 = vqaddq_s8(delta1, k4);
    let delta3 = vshrq_n_s8::<3>(delta_p3);
    let delta4 = vshrq_n_s8::<3>(delta_p4);
    let sp0 = vqaddq_s8(p0s, delta3);
    let sq0 = vqsubq_s8(q0s, delta4);
    (
        flip_sign_back_neon(_token, sp0),
        flip_sign_back_neon(_token, sq0),
    )
}

/// DoFilter4: 4-tap filter for subblock edges (inner edges)
/// Returns (op1, op0, oq0, oq1)

#[rite]
fn do_filter4_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    let p1s = flip_sign_neon(_token, p1);
    let mut p0s = flip_sign_neon(_token, p0);
    let mut q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let simple_lf_mask = vandq_u8(mask, hev_mask);

    // do_filter2 part: simple filter on pixels with hev
    {
        let delta = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);
        let simple_lf_delta = vandq_s8(delta, vreinterpretq_s8_u8(simple_lf_mask));
        let (new_p0s, new_q0s) = apply_filter2_no_flip_neon(_token, p0s, q0s, simple_lf_delta);
        p0s = new_p0s;
        q0s = new_q0s;
    }

    // do_filter4 part: complex filter on pixels without hev
    let delta0 = get_base_delta0_neon(_token, p0s, q0s);
    // (mask & hev_mask) ^ mask = mask & !hev_mask
    let complex_lf_mask = veorq_u8(simple_lf_mask, mask);
    let complex_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(complex_lf_mask));

    // ApplyFilter4
    let k3 = vdupq_n_s8(0x03);
    let k4 = vdupq_n_s8(0x04);
    let delta1 = vqaddq_s8(complex_lf_delta, k4);
    let delta2 = vqaddq_s8(complex_lf_delta, k3);
    let a1 = vshrq_n_s8::<3>(delta1);
    let a2 = vshrq_n_s8::<3>(delta2);
    let a3 = vrshrq_n_s8::<1>(a1); // (a1 + 1) >> 1

    let op0 = flip_sign_back_neon(_token, vqaddq_s8(p0s, a2));
    let oq0 = flip_sign_back_neon(_token, vqsubq_s8(q0s, a1));
    let op1 = flip_sign_back_neon(_token, vqaddq_s8(p1s, a3));
    let oq1 = flip_sign_back_neon(_token, vqsubq_s8(q1s, a3));

    (op1, op0, oq0, oq1)
}

/// DoFilter6: 6-tap filter for macroblock edges
/// Returns (op2, op1, op0, oq0, oq1, oq2)

#[rite]
#[allow(clippy::type_complexity)]
fn do_filter6_neon(
    _token: NeonToken,
    p2: uint8x16_t,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    q2: uint8x16_t,
    mask: uint8x16_t,
    hev_mask: uint8x16_t,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    let p2s = flip_sign_neon(_token, p2);
    let p1s = flip_sign_neon(_token, p1);
    let mut p0s = flip_sign_neon(_token, p0);
    let mut q0s = flip_sign_neon(_token, q0);
    let q1s = flip_sign_neon(_token, q1);
    let q2s = flip_sign_neon(_token, q2);
    let simple_lf_mask = vandq_u8(mask, hev_mask);
    let delta0 = get_base_delta_neon(_token, p1s, p0s, q0s, q1s);

    // do_filter2 part: simple filter on pixels with hev
    {
        let simple_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(simple_lf_mask));
        let (new_p0s, new_q0s) = apply_filter2_no_flip_neon(_token, p0s, q0s, simple_lf_delta);
        p0s = new_p0s;
        q0s = new_q0s;
    }

    // do_filter6 part: 6-tap filter on pixels without hev
    let complex_lf_mask = veorq_u8(simple_lf_mask, mask);
    let complex_lf_delta = vandq_s8(delta0, vreinterpretq_s8_u8(complex_lf_mask));

    // ApplyFilter6: compute X=(9*a+63)>>7, Y=(18*a+63)>>7, Z=(27*a+63)>>7
    // Using S = 9*a - 1 trick with vqrshrn_n_s16
    let delta_lo = vget_low_s8(complex_lf_delta);
    let delta_hi = vget_high_s8(complex_lf_delta);
    let k9 = vdup_n_s8(9);
    let km1 = vdupq_n_s16(-1);
    let k18 = vdup_n_s8(18);

    let s_lo = vmlal_s8(km1, k9, delta_lo); // S = 9 * a - 1
    let s_hi = vmlal_s8(km1, k9, delta_hi);
    let z_lo = vmlal_s8(s_lo, k18, delta_lo); // S + 18 * a
    let z_hi = vmlal_s8(s_hi, k18, delta_hi);

    let a3_lo = vqrshrn_n_s16::<7>(s_lo); // (9*a + 63) >> 7
    let a3_hi = vqrshrn_n_s16::<7>(s_hi);
    let a2_lo = vqrshrn_n_s16::<6>(s_lo); // (9*a + 31) >> 6
    let a2_hi = vqrshrn_n_s16::<6>(s_hi);
    let a1_lo = vqrshrn_n_s16::<7>(z_lo); // (27*a + 63) >> 7
    let a1_hi = vqrshrn_n_s16::<7>(z_hi);

    let a1 = vcombine_s8(a1_lo, a1_hi);
    let a2 = vcombine_s8(a2_lo, a2_hi);
    let a3 = vcombine_s8(a3_lo, a3_hi);

    let op0 = flip_sign_back_neon(_token, vqaddq_s8(p0s, a1));
    let oq0 = flip_sign_back_neon(_token, vqsubq_s8(q0s, a1));
    let op1 = flip_sign_back_neon(_token, vqaddq_s8(p1s, a2));
    let oq1 = flip_sign_back_neon(_token, vqsubq_s8(q1s, a2));
    let op2 = flip_sign_back_neon(_token, vqaddq_s8(p2s, a3));
    let oq2 = flip_sign_back_neon(_token, vqsubq_s8(q2s, a3));

    (op2, op1, op0, oq0, oq1, oq2)
}

// =============================================================================
// Load/Store helpers
// =============================================================================

/// Load 16 pixels from 4 consecutive rows (for vertical filter)

#[rite]
fn load_16x4_neon(
    _token: NeonToken,
    buf: &[u8],
    point: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    let p1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 2 * stride..][..16]).unwrap());
    let p0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - stride..][..16]).unwrap());
    let q0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point..][..16]).unwrap());
    let q1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + stride..][..16]).unwrap());
    (p1, p0, q0, q1)
}

/// Load 16 pixels from 8 consecutive rows (for normal vertical filter)

#[rite]
#[allow(clippy::type_complexity)]
fn load_16x8_neon(
    _token: NeonToken,
    buf: &[u8],
    point: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    let p3 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 4 * stride..][..16]).unwrap());
    let p2 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 3 * stride..][..16]).unwrap());
    let p1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - 2 * stride..][..16]).unwrap());
    let p0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point - stride..][..16]).unwrap());
    let q0 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point..][..16]).unwrap());
    let q1 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + stride..][..16]).unwrap());
    let q2 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + 2 * stride..][..16]).unwrap());
    let q3 = simd_mem::vld1q_u8(<&[u8; 16]>::try_from(&buf[point + 3 * stride..][..16]).unwrap());
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 16 pixels to 2 consecutive rows

#[rite]
fn store_16x2_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    p0: uint8x16_t,
    q0: uint8x16_t,
) {
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        p0,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        q0,
    );
}

/// Load 8 U pixels + 8 V pixels into a single uint8x16_t per row (for UV vertical filter)
/// Loads 8 rows centered on the edge.

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x8x2_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    point: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Pack U in low half, V in high half
    let p3 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 4 * stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 4 * stride..][..8]).unwrap()),
    );
    let p2 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 3 * stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 3 * stride..][..8]).unwrap()),
    );
    let p1 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - 2 * stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - 2 * stride..][..8]).unwrap()),
    );
    let p0 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point - stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point - stride..][..8]).unwrap()),
    );
    let q0 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point..][..8]).unwrap()),
    );
    let q1 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + stride..][..8]).unwrap()),
    );
    let q2 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + 2 * stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + 2 * stride..][..8]).unwrap()),
    );
    let q3 = vcombine_u8(
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&u_buf[point + 3 * stride..][..8]).unwrap()),
        simd_mem::vld1_u8(<&[u8; 8]>::try_from(&v_buf[point + 3 * stride..][..8]).unwrap()),
    );
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 8x2x2: store U+V packed results back to separate buffers (2 rows)

#[rite]
fn store_8x2x2_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    // low half = U, high half = V
    simd_mem::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut u_buf[u_point - stride..][..8]).unwrap(),
        vget_low_u8(p0),
    );
    simd_mem::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut u_buf[u_point..][..8]).unwrap(),
        vget_low_u8(q0),
    );
    simd_mem::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut v_buf[v_point - stride..][..8]).unwrap(),
        vget_high_u8(p0),
    );
    simd_mem::vst1_u8(
        <&mut [u8; 8]>::try_from(&mut v_buf[v_point..][..8]).unwrap(),
        vget_high_u8(q0),
    );
}

/// Store 8x4x2: store U+V packed results (4 rows)

#[rite]
fn store_8x4x2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    store_8x2x2_neon(
        _token,
        p1,
        p0,
        u_buf,
        v_buf,
        u_point - stride,
        v_point - stride,
        stride,
    );
    store_8x2x2_neon(
        _token,
        q0,
        q1,
        u_buf,
        v_buf,
        u_point + stride,
        v_point + stride,
        stride,
    );
}

/// Store 6 rows (3 above, 3 below) for UV 6-tap filter

#[rite]
fn store_8x6x2_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_point: usize,
    v_point: usize,
    stride: usize,
) {
    store_8x2x2_neon(
        _token,
        op2,
        op1,
        u_buf,
        v_buf,
        u_point - 2 * stride,
        v_point - 2 * stride,
        stride,
    );
    store_8x2x2_neon(_token, op0, oq0, u_buf, v_buf, u_point, v_point, stride);
    store_8x2x2_neon(
        _token,
        oq1,
        oq2,
        u_buf,
        v_buf,
        u_point + 2 * stride,
        v_point + 2 * stride,
        stride,
    );
}

// =============================================================================
// Horizontal filter load/store (transpose-based for vertical edges)
// =============================================================================

/// Load 4 columns from 16 rows as transposed uint8x16_t vectors.
/// Loads 4 bytes per row, packs into u32x4 registers, and transposes.
/// buf[y_start + row * stride + x0 - 2 .. x0 + 2] for row 0..16

#[rite]
fn load_4x16_neon(
    _token: NeonToken,
    buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    // Load 4 bytes per row for 16 rows, pack into u32, then transpose
    let base = y_start * stride + x0 - 2;
    let mut rows: [u32; 16] = [0; 16];
    for i in 0..16 {
        let offset = base + i * stride;
        rows[i] = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
    }

    // Pack into 4 uint32x4_t (each holds 4 rows)
    let in0 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[0..4]).unwrap(),
    ));
    let in1 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[4..8]).unwrap(),
    ));
    let in2 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[8..12]).unwrap(),
    ));
    let in3 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[12..16]).unwrap(),
    ));

    // Transpose four 4x4 blocks to get columns:
    // in0: row0[c0 c1 c2 c3] row1[c0 c1 c2 c3] row2[...] row3[...]
    // After transpose: col0[r0 r1 r2 r3 | r4 r5 ... r15]
    let row01 = vtrnq_u8(in0, in1);
    let row23 = vtrnq_u8(in2, in3);
    let row02 = vtrnq_u16(vreinterpretq_u16_u8(row01.0), vreinterpretq_u16_u8(row23.0));
    let row13 = vtrnq_u16(vreinterpretq_u16_u8(row01.1), vreinterpretq_u16_u8(row23.1));

    let p1 = vreinterpretq_u8_u16(row02.0);
    let p0 = vreinterpretq_u8_u16(row13.0);
    let q0 = vreinterpretq_u8_u16(row02.1);
    let q1 = vreinterpretq_u8_u16(row13.1);

    (p1, p0, q0, q1)
}

/// Load 8 columns from 16 rows (for normal horizontal filter)

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x16_neon(
    _token: NeonToken,
    buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Load columns x0-4..x0+4 from 16 rows, transposed
    // Load the left 4 columns and right 4 columns separately
    let (p3, p2, p1, p0) = load_4x16_neon(_token, buf, x0 - 2, y_start, stride);
    let (q0, q1, q2, q3) = load_4x16_neon(_token, buf, x0 + 2, y_start, stride);
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 2 transposed columns back to 16 rows
/// Writes p0 and q0 to columns (x0-1) and (x0) for 16 rows

#[rite]
fn store_2x16_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    // Extract each lane pair and write 2 bytes per row
    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 1;
        buf[offset] = vgetq_lane_u8::<0>(vextq_u8::<0>(p0, p0)); // placeholder
        // This needs per-lane extraction. Let's use a different approach.
    }
    // Actually, extract all values at once
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    simd_mem::vst1q_u8(&mut p0_bytes, p0);
    simd_mem::vst1q_u8(&mut q0_bytes, q0);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 1;
        buf[offset] = p0_bytes[i];
        buf[offset + 1] = q0_bytes[i];
    }
}

/// Store 4 transposed columns back to 16 rows
/// Writes p1, p0, q0, q1 to columns (x0-2)..(x0+2) for 16 rows

#[rite]
fn store_4x16_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut p1_bytes = [0u8; 16];
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    let mut q1_bytes = [0u8; 16];
    simd_mem::vst1q_u8(&mut p1_bytes, p1);
    simd_mem::vst1q_u8(&mut p0_bytes, p0);
    simd_mem::vst1q_u8(&mut q0_bytes, q0);
    simd_mem::vst1q_u8(&mut q1_bytes, q1);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 2;
        buf[offset] = p1_bytes[i];
        buf[offset + 1] = p0_bytes[i];
        buf[offset + 2] = q0_bytes[i];
        buf[offset + 3] = q1_bytes[i];
    }
}

/// Store 6 transposed columns back to 16 rows (for normal h-filter edge)

#[rite]
fn store_6x16_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b2 = [0u8; 16];
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    let mut a2 = [0u8; 16];
    simd_mem::vst1q_u8(&mut b2, op2);
    simd_mem::vst1q_u8(&mut b1, op1);
    simd_mem::vst1q_u8(&mut b0, op0);
    simd_mem::vst1q_u8(&mut a0, oq0);
    simd_mem::vst1q_u8(&mut a1, oq1);
    simd_mem::vst1q_u8(&mut a2, oq2);

    for i in 0..16 {
        let offset = (y_start + i) * stride + x0 - 3;
        buf[offset] = b2[i];
        buf[offset + 1] = b1[i];
        buf[offset + 2] = b0[i];
        buf[offset + 3] = a0[i];
        buf[offset + 4] = a1[i];
        buf[offset + 5] = a2[i];
    }
}

// =============================================================================
// UV horizontal filter load/store (transpose 8 rows from U + 8 from V)
// =============================================================================

/// Load 4 columns from 8 U rows + 8 V rows, packed in uint8x16_t (transposed)

#[rite]
fn load_4x8x2_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) {
    // Load 4 bytes per row for 8 U rows + 8 V rows
    let base_u = y_start * stride + x0 - 2;
    let base_v = y_start * stride + x0 - 2;
    let mut rows: [u32; 16] = [0; 16];

    for i in 0..8 {
        let offset = base_u + i * stride;
        rows[i] = u32::from_le_bytes([
            u_buf[offset],
            u_buf[offset + 1],
            u_buf[offset + 2],
            u_buf[offset + 3],
        ]);
    }
    for i in 0..8 {
        let offset = base_v + i * stride;
        rows[8 + i] = u32::from_le_bytes([
            v_buf[offset],
            v_buf[offset + 1],
            v_buf[offset + 2],
            v_buf[offset + 3],
        ]);
    }

    let in0 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[0..4]).unwrap(),
    ));
    let in1 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[4..8]).unwrap(),
    ));
    let in2 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[8..12]).unwrap(),
    ));
    let in3 = vreinterpretq_u8_u32(simd_mem::vld1q_u32(
        <&[u32; 4]>::try_from(&rows[12..16]).unwrap(),
    ));

    let row01 = vtrnq_u8(in0, in1);
    let row23 = vtrnq_u8(in2, in3);
    let row02 = vtrnq_u16(vreinterpretq_u16_u8(row01.0), vreinterpretq_u16_u8(row23.0));
    let row13 = vtrnq_u16(vreinterpretq_u16_u8(row01.1), vreinterpretq_u16_u8(row23.1));

    let p1 = vreinterpretq_u8_u16(row02.0);
    let p0 = vreinterpretq_u8_u16(row13.0);
    let q0 = vreinterpretq_u8_u16(row02.1);
    let q1 = vreinterpretq_u8_u16(row13.1);

    (p1, p0, q0, q1)
}

/// Load 8 columns from 8 U rows + 8 V rows for normal horizontal chroma filter

#[rite]
#[allow(clippy::type_complexity)]
fn load_8x8x2_h_neon(
    _token: NeonToken,
    u_buf: &[u8],
    v_buf: &[u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) -> (
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
    uint8x16_t,
) {
    // Load left 4 and right 4 columns separately
    let (p3, p2, p1, p0) = load_4x8x2_neon(_token, u_buf, v_buf, x0 - 2, y_start, stride);
    // For the right side, shift x0 by 2 (the load_4x8x2 already does x0-2 internally)
    let (q0, q1, q2, q3) = load_4x8x2_neon(_token, u_buf, v_buf, x0 + 2, y_start, stride);
    (p3, p2, p1, p0, q0, q1, q2, q3)
}

/// Store 2 transposed columns back to 8 U rows + 8 V rows
#[allow(dead_code)]
#[rite]
fn store_2x8x2_neon(
    _token: NeonToken,
    p0: uint8x16_t,
    q0: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut p0_bytes = [0u8; 16];
    let mut q0_bytes = [0u8; 16];
    simd_mem::vst1q_u8(&mut p0_bytes, p0);
    simd_mem::vst1q_u8(&mut q0_bytes, q0);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 1;
        u_buf[offset] = p0_bytes[i];
        u_buf[offset + 1] = q0_bytes[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 1;
        v_buf[offset] = p0_bytes[8 + i];
        v_buf[offset + 1] = q0_bytes[8 + i];
    }
}

/// Store 4 transposed columns back to 8 U rows + 8 V rows

#[rite]
fn store_4x8x2_neon(
    _token: NeonToken,
    p1: uint8x16_t,
    p0: uint8x16_t,
    q0: uint8x16_t,
    q1: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    simd_mem::vst1q_u8(&mut b1, p1);
    simd_mem::vst1q_u8(&mut b0, p0);
    simd_mem::vst1q_u8(&mut a0, q0);
    simd_mem::vst1q_u8(&mut a1, q1);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 2;
        u_buf[offset] = b1[i];
        u_buf[offset + 1] = b0[i];
        u_buf[offset + 2] = a0[i];
        u_buf[offset + 3] = a1[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 2;
        v_buf[offset] = b1[8 + i];
        v_buf[offset + 1] = b0[8 + i];
        v_buf[offset + 2] = a0[8 + i];
        v_buf[offset + 3] = a1[8 + i];
    }
}

/// Store 6 transposed columns back to 8 U rows + 8 V rows (6-tap edge filter)

#[rite]
fn store_6x8x2_neon(
    _token: NeonToken,
    op2: uint8x16_t,
    op1: uint8x16_t,
    op0: uint8x16_t,
    oq0: uint8x16_t,
    oq1: uint8x16_t,
    oq2: uint8x16_t,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
) {
    let mut b2 = [0u8; 16];
    let mut b1 = [0u8; 16];
    let mut b0 = [0u8; 16];
    let mut a0 = [0u8; 16];
    let mut a1 = [0u8; 16];
    let mut a2 = [0u8; 16];
    simd_mem::vst1q_u8(&mut b2, op2);
    simd_mem::vst1q_u8(&mut b1, op1);
    simd_mem::vst1q_u8(&mut b0, op0);
    simd_mem::vst1q_u8(&mut a0, oq0);
    simd_mem::vst1q_u8(&mut a1, oq1);
    simd_mem::vst1q_u8(&mut a2, oq2);

    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 3;
        u_buf[offset] = b2[i];
        u_buf[offset + 1] = b1[i];
        u_buf[offset + 2] = b0[i];
        u_buf[offset + 3] = a0[i];
        u_buf[offset + 4] = a1[i];
        u_buf[offset + 5] = a2[i];
    }
    for i in 0..8 {
        let offset = (y_start + i) * stride + x0 - 3;
        v_buf[offset] = b2[8 + i];
        v_buf[offset + 1] = b1[8 + i];
        v_buf[offset + 2] = b0[8 + i];
        v_buf[offset + 3] = a0[8 + i];
        v_buf[offset + 4] = a1[8 + i];
        v_buf[offset + 5] = a2[8 + i];
    }
}

// =============================================================================
// Simple filter (2-tap) — luma
// =============================================================================

/// Simple vertical filter: 16 pixels across a horizontal edge

#[arcane]
pub(crate) fn simple_v_filter16_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    let (p1, p0, q0, q1) = load_16x4_neon(_token, buf, point, stride);
    let mask = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    let (op0, oq0) = do_filter2_neon(_token, p1, p0, q0, q1, mask);
    store_16x2_neon(_token, buf, point, stride, op0, oq0);
}

/// Simple horizontal filter: 16 rows across a vertical edge

#[arcane]
pub(crate) fn simple_h_filter16_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    let (p1, p0, q0, q1) = load_4x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter_neon(_token, p1, p0, q0, q1, thresh);
    let (op0, oq0) = do_filter2_neon(_token, p1, p0, q0, q1, mask);
    store_2x16_neon(_token, op0, oq0, buf, x0, y_start, stride);
}

// =============================================================================
// Normal filter (4-tap inner, 6-tap edge) — luma
// =============================================================================

/// Normal vertical filter for subblock edge (inner edge, 4-tap)

#[arcane]
pub(crate) fn normal_v_filter16_inner_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_16x8_neon(_token, buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);

    // Store 4 rows around the edge
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 2 * stride..][..16]).unwrap(),
        op1,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        op0,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        oq0,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + stride..][..16]).unwrap(),
        oq1,
    );
}

/// Normal vertical filter for macroblock edge (6-tap)

#[arcane]
pub(crate) fn normal_v_filter16_edge_neon(
    _token: NeonToken,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_16x8_neon(_token, buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);

    // Store 6 rows
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 3 * stride..][..16]).unwrap(),
        op2,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - 2 * stride..][..16]).unwrap(),
        op1,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point - stride..][..16]).unwrap(),
        op0,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point..][..16]).unwrap(),
        oq0,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + stride..][..16]).unwrap(),
        oq1,
    );
    simd_mem::vst1q_u8(
        <&mut [u8; 16]>::try_from(&mut buf[point + 2 * stride..][..16]).unwrap(),
        oq2,
    );
}

/// Normal horizontal filter for subblock edge (inner edge, 4-tap)

#[arcane]
pub(crate) fn normal_h_filter16_inner_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_4x16_neon(_token, op1, op0, oq0, oq1, buf, x0, y_start, stride);
}

/// Normal horizontal filter for macroblock edge (6-tap)

#[arcane]
pub(crate) fn normal_h_filter16_edge_neon(
    _token: NeonToken,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x16_neon(_token, buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_6x16_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, buf, x0, y_start, stride,
    );
}

// =============================================================================
// Normal filter — chroma (UV packed as 16 pixels)
// =============================================================================

/// Normal vertical filter for UV macroblock edge (6-tap)

#[arcane]
pub(crate) fn normal_v_filter_uv_edge_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x8x2_neon(_token, u_buf, v_buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_8x6x2_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, u_buf, v_buf, point, point, stride,
    );
}

/// Normal vertical filter for UV subblock edge (4-tap)

#[arcane]
pub(crate) fn normal_v_filter_uv_inner_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_8x8x2_neon(_token, u_buf, v_buf, point, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_8x4x2_neon(
        _token, op1, op0, oq0, oq1, u_buf, v_buf, point, point, stride,
    );
}

/// Normal horizontal filter for UV macroblock edge (6-tap)

#[arcane]
pub(crate) fn normal_h_filter_uv_edge_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) =
        load_8x8x2_h_neon(_token, u_buf, v_buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op2, op1, op0, oq0, oq1, oq2) =
        do_filter6_neon(_token, p2, p1, p0, q0, q1, q2, mask, hev_mask);
    store_6x8x2_neon(
        _token, op2, op1, op0, oq0, oq1, oq2, u_buf, v_buf, x0, y_start, stride,
    );
}

/// Normal horizontal filter for UV subblock edge (4-tap)

#[arcane]
pub(crate) fn normal_h_filter_uv_inner_neon(
    _token: NeonToken,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) =
        load_8x8x2_h_neon(_token, u_buf, v_buf, x0, y_start, stride);
    let mask = needs_filter2_neon(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        interior_limit,
        edge_limit,
    );
    let hev_mask = needs_hev_neon(_token, p1, p0, q0, q1, hev_threshold);
    let (op1, op0, oq0, oq1) = do_filter4_neon(_token, p1, p0, q0, q1, mask, hev_mask);
    store_4x8x2_neon(
        _token, op1, op0, oq0, oq1, u_buf, v_buf, x0, y_start, stride,
    );
}
