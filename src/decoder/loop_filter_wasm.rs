//! WASM SIMD128 loop filter implementations for VP8 decoder.
//!
//! Ported from the NEON versions in loop_filter_neon.rs.
//! Processes 16 pixels at a time using 128-bit SIMD.

#![allow(clippy::too_many_arguments)]

use archmage::{Wasm128Token, arcane};

use core::arch::wasm32::*;

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
fn store_u8x16(out: &mut [u8; 16], v: v128) {
    out[0] = u8x16_extract_lane::<0>(v);
    out[1] = u8x16_extract_lane::<1>(v);
    out[2] = u8x16_extract_lane::<2>(v);
    out[3] = u8x16_extract_lane::<3>(v);
    out[4] = u8x16_extract_lane::<4>(v);
    out[5] = u8x16_extract_lane::<5>(v);
    out[6] = u8x16_extract_lane::<6>(v);
    out[7] = u8x16_extract_lane::<7>(v);
    out[8] = u8x16_extract_lane::<8>(v);
    out[9] = u8x16_extract_lane::<9>(v);
    out[10] = u8x16_extract_lane::<10>(v);
    out[11] = u8x16_extract_lane::<11>(v);
    out[12] = u8x16_extract_lane::<12>(v);
    out[13] = u8x16_extract_lane::<13>(v);
    out[14] = u8x16_extract_lane::<14>(v);
    out[15] = u8x16_extract_lane::<15>(v);
}

/// Load 16 bytes from a buffer row
#[inline(always)]
fn load_row(buf: &[u8], offset: usize) -> v128 {
    load_u8x16(<&[u8; 16]>::try_from(&buf[offset..offset + 16]).unwrap())
}

/// Store 16 bytes to a buffer row
#[inline(always)]
fn store_row(buf: &mut [u8], offset: usize, v: v128) {
    store_u8x16(
        <&mut [u8; 16]>::try_from(&mut buf[offset..offset + 16]).unwrap(),
        v,
    );
}

/// Load a column of 16 bytes (one byte per row, stride apart)
#[inline(always)]
fn load_col(buf: &[u8], base: usize, stride: usize) -> v128 {
    u8x16(
        buf[base],
        buf[base + stride],
        buf[base + stride * 2],
        buf[base + stride * 3],
        buf[base + stride * 4],
        buf[base + stride * 5],
        buf[base + stride * 6],
        buf[base + stride * 7],
        buf[base + stride * 8],
        buf[base + stride * 9],
        buf[base + stride * 10],
        buf[base + stride * 11],
        buf[base + stride * 12],
        buf[base + stride * 13],
        buf[base + stride * 14],
        buf[base + stride * 15],
    )
}

/// Store a column of 16 bytes
#[inline(always)]
fn store_col(buf: &mut [u8], base: usize, stride: usize, v: v128) {
    buf[base] = u8x16_extract_lane::<0>(v);
    buf[base + stride] = u8x16_extract_lane::<1>(v);
    buf[base + stride * 2] = u8x16_extract_lane::<2>(v);
    buf[base + stride * 3] = u8x16_extract_lane::<3>(v);
    buf[base + stride * 4] = u8x16_extract_lane::<4>(v);
    buf[base + stride * 5] = u8x16_extract_lane::<5>(v);
    buf[base + stride * 6] = u8x16_extract_lane::<6>(v);
    buf[base + stride * 7] = u8x16_extract_lane::<7>(v);
    buf[base + stride * 8] = u8x16_extract_lane::<8>(v);
    buf[base + stride * 9] = u8x16_extract_lane::<9>(v);
    buf[base + stride * 10] = u8x16_extract_lane::<10>(v);
    buf[base + stride * 11] = u8x16_extract_lane::<11>(v);
    buf[base + stride * 12] = u8x16_extract_lane::<12>(v);
    buf[base + stride * 13] = u8x16_extract_lane::<13>(v);
    buf[base + stride * 14] = u8x16_extract_lane::<14>(v);
    buf[base + stride * 15] = u8x16_extract_lane::<15>(v);
}

// =============================================================================
// Core filter helpers
// =============================================================================

/// Absolute difference of u8x16
#[inline(always)]
fn abd_u8x16(a: v128, b: v128) -> v128 {
    u8x16_sub(u8x16_max(a, b), u8x16_min(a, b))
}

/// NeedsFilter: returns mask where 2*|p0-q0| + |p1-q1|/2 <= thresh
#[inline(always)]
fn needs_filter(p1: v128, p0: v128, q0: v128, q1: v128, thresh: i32) -> v128 {
    let thresh_v = u8x16_splat(thresh as u8);
    let a_p0_q0 = abd_u8x16(p0, q0);
    let a_p1_q1 = abd_u8x16(p1, q1);
    let a_p0_q0_2 = u8x16_add_sat(a_p0_q0, a_p0_q0);
    let a_p1_q1_2 = u8x16_shr(a_p1_q1, 1);
    let sum = u8x16_add_sat(a_p0_q0_2, a_p1_q1_2);
    u8x16_le(sum, thresh_v)
}

/// NeedsFilter2: extended check for normal filter (NeedsFilter AND all |adj| <= ithresh)
#[inline(always)]
fn needs_filter2(
    p3: v128,
    p2: v128,
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    q2: v128,
    q3: v128,
    ithresh: i32,
    thresh: i32,
) -> v128 {
    let it = u8x16_splat(ithresh as u8);
    let mask = needs_filter(p1, p0, q0, q1, thresh);
    let m1 = u8x16_le(abd_u8x16(p3, p2), it);
    let m2 = u8x16_le(abd_u8x16(p2, p1), it);
    let m3 = u8x16_le(abd_u8x16(p1, p0), it);
    let m4 = u8x16_le(abd_u8x16(q0, q1), it);
    let m5 = u8x16_le(abd_u8x16(q1, q2), it);
    let m6 = u8x16_le(abd_u8x16(q2, q3), it);
    v128_and(
        mask,
        v128_and(
            v128_and(m1, m2),
            v128_and(v128_and(m3, m4), v128_and(m5, m6)),
        ),
    )
}

/// HEV: high edge variance (|p1-p0| > thresh || |q1-q0| > thresh)
#[inline(always)]
fn hev(p1: v128, p0: v128, q0: v128, q1: v128, thresh: i32) -> v128 {
    let t = u8x16_splat(thresh as u8);
    let h1 = u8x16_gt(abd_u8x16(p1, p0), t);
    let h2 = u8x16_gt(abd_u8x16(q1, q0), t);
    v128_or(h1, h2)
}

/// DoFilter2: simple VP8 filter (modifies p0, q0)
/// a = clamp((p0 - q0 + 3*(q1 - p1) + 4) >> 3)  for q0
/// Also computes a2 = (a + 1) >> 1 for p0
#[inline(always)]
fn do_filter2(p1: v128, p0: &mut v128, q0: &mut v128, q1: v128, mask: v128) {
    let sign = u8x16_splat(0x80);
    // Convert to signed domain
    let sp1 = v128_xor(p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(q1, sign);

    // a = 3*(q0 - p0) + satu8(p1 - q1)
    let a0 = i8x16_sub_sat(sp1, sq1); // clamp(p1 - q1)
    let a1 = i8x16_sub_sat(sq0, sp0); // clamp(q0 - p0)
    let a2 = i8x16_add_sat(a1, a1); // 2*(q0 - p0)
    let a3 = i8x16_add_sat(a2, a1); // 3*(q0 - p0)
    let a = i8x16_add_sat(a0, a3); // 3*(q0-p0) + clamp(p1-q1)
    let a_masked = v128_and(a, mask);

    // Filter1 = clamp(a + 4) >> 3
    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    // Filter2 = clamp(a + 3) >> 3
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    *q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    *p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);
}

/// DoFilter4: normal inner VP8 filter (modifies p1, p0, q0, q1)
#[inline(always)]
fn do_filter4(
    p1: &mut v128,
    p0: &mut v128,
    q0: &mut v128,
    q1: &mut v128,
    mask: v128,
    hev_mask: v128,
) {
    let sign = u8x16_splat(0x80);
    let sp1 = v128_xor(*p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(*q1, sign);

    let a0 = i8x16_sub_sat(sp1, sq1);
    let a0_hev = v128_and(a0, hev_mask); // only apply p1-q1 term where HEV
    let a1 = i8x16_sub_sat(sq0, sp0);
    let a2 = i8x16_add_sat(a1, a1);
    let a3 = i8x16_add_sat(a2, a1);
    let a = i8x16_add_sat(a0_hev, a3);
    let a_masked = v128_and(a, mask);

    // Filter1, Filter2
    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    let new_q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    let new_p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);

    // For non-HEV pixels, also adjust p1 and q1
    let f3 = i8x16_add_sat(f1, i8x16_splat(1));
    let f3 = i8x16_shr(f3, 1); // (f1 + 1) >> 1
    let not_hev = v128_not(hev_mask);
    let f3_masked = v128_and(f3, not_hev);

    *q0 = new_q0;
    *p0 = new_p0;
    *q1 = v128_xor(i8x16_sub_sat(sq1, f3_masked), sign);
    *p1 = v128_xor(i8x16_add_sat(sp1, f3_masked), sign);
}

/// DoFilter6: normal edge VP8 filter (modifies p2, p1, p0, q0, q1, q2)
#[inline(always)]
fn do_filter6(
    p2: &mut v128,
    p1: &mut v128,
    p0: &mut v128,
    q0: &mut v128,
    q1: &mut v128,
    q2: &mut v128,
    mask: v128,
    hev_mask: v128,
) {
    let sign = u8x16_splat(0x80);
    let sp2 = v128_xor(*p2, sign);
    let sp1 = v128_xor(*p1, sign);
    let sp0 = v128_xor(*p0, sign);
    let sq0 = v128_xor(*q0, sign);
    let sq1 = v128_xor(*q1, sign);
    let sq2 = v128_xor(*q2, sign);

    // For HEV pixels: same as simple filter (do_filter2 logic)
    let a0 = i8x16_sub_sat(sp1, sq1);
    let a1 = i8x16_sub_sat(sq0, sp0);
    let a2 = i8x16_add_sat(a1, a1);
    let a3 = i8x16_add_sat(a2, a1);
    let a = i8x16_add_sat(a0, a3);
    let a_masked = v128_and(a, mask);

    let f1 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(4)), 3);
    let f2 = i8x16_shr(i8x16_add_sat(a_masked, i8x16_splat(3)), 3);

    let hev_q0 = v128_xor(i8x16_sub_sat(sq0, f1), sign);
    let hev_p0 = v128_xor(i8x16_add_sat(sp0, f2), sign);

    // For non-HEV pixels: wider filter using p2,p1,p0,q0,q1,q2
    // a = clamp(p0 - q0), then compute 27*a+63 >> 7, 18*a+63 >> 7, 9*a+63 >> 7
    let not_hev = v128_and(mask, v128_not(hev_mask));
    let w = i8x16_sub_sat(sp0, sq0); // clamp(p0 - q0) in signed domain

    // Widen to i16 for wider filter computation
    let w_lo = i16x8_extend_low_i8x16(w);
    let w_hi = i16x8_extend_high_i8x16(w);
    let round = i16x8_splat(63);

    // 27 * w
    let w27_lo = i16x8_mul(w_lo, i16x8_splat(27));
    let w27_hi = i16x8_mul(w_hi, i16x8_splat(27));
    let a_27_lo = i16x8_shr(i16x8_add(w27_lo, round), 7);
    let a_27_hi = i16x8_shr(i16x8_add(w27_hi, round), 7);
    let a27 = i8x16_narrow_i16x8(a_27_lo, a_27_hi);

    // 18 * w
    let w18_lo = i16x8_mul(w_lo, i16x8_splat(18));
    let w18_hi = i16x8_mul(w_hi, i16x8_splat(18));
    let a_18_lo = i16x8_shr(i16x8_add(w18_lo, round), 7);
    let a_18_hi = i16x8_shr(i16x8_add(w18_hi, round), 7);
    let a18 = i8x16_narrow_i16x8(a_18_lo, a_18_hi);

    // 9 * w
    let w9_lo = i16x8_mul(w_lo, i16x8_splat(9));
    let w9_hi = i16x8_mul(w_hi, i16x8_splat(9));
    let a_9_lo = i16x8_shr(i16x8_add(w9_lo, round), 7);
    let a_9_hi = i16x8_shr(i16x8_add(w9_hi, round), 7);
    let a9 = i8x16_narrow_i16x8(a_9_lo, a_9_hi);

    let wide_q0 = v128_xor(i8x16_sub_sat(sq0, a27), sign);
    let wide_p0 = v128_xor(i8x16_add_sat(sp0, a27), sign);
    let wide_q1 = v128_xor(i8x16_sub_sat(sq1, a18), sign);
    let wide_p1 = v128_xor(i8x16_add_sat(sp1, a18), sign);
    let wide_q2 = v128_xor(i8x16_sub_sat(sq2, a9), sign);
    let wide_p2 = v128_xor(i8x16_add_sat(sp2, a9), sign);

    // Select: HEV pixels use simple filter, non-HEV use wide filter
    *q0 = v128_bitselect(hev_q0, wide_q0, hev_mask);
    *p0 = v128_bitselect(hev_p0, wide_p0, hev_mask);
    *q1 = v128_bitselect(*q1, wide_q1, not_hev); // only modify non-HEV
    *p1 = v128_bitselect(*p1, wide_p1, not_hev);
    *q2 = v128_bitselect(*q2, wide_q2, not_hev);
    *p2 = v128_bitselect(*p2, wide_p2, not_hev);
}

// =============================================================================
// Simple filter (vertical/horizontal)
// =============================================================================

/// Simple vertical filter: processes horizontal edge at `point`, 16 columns wide
#[arcane]
pub(crate) fn simple_v_filter16_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    let p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let q1 = load_row(buf, point + stride);

    let mask = needs_filter(p1, p0, q0, q1, thresh);
    do_filter2(p1, &mut p0, &mut q0, q1, mask);

    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
}

/// Simple horizontal filter: processes vertical edge at column x0, 16 rows
#[arcane]
pub(crate) fn simple_h_filter16_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    let base = y_start * stride + x0;
    let p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let q1 = load_col(buf, base + 1, stride);

    let mask = needs_filter(p1, p0, q0, q1, thresh);
    do_filter2(p1, &mut p0, &mut q0, q1, mask);

    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
}

// =============================================================================
// Normal filter (vertical/horizontal, inner/edge)
// =============================================================================

/// Normal vertical inner filter (subblock edge)
#[arcane]
pub(crate) fn normal_v_filter16_inner_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_row(buf, point - 4 * stride);
    let p2 = load_row(buf, point - 3 * stride);
    let mut p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let mut q1 = load_row(buf, point + stride);
    let q2 = load_row(buf, point + 2 * stride);
    let q3 = load_row(buf, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_row(buf, point - 2 * stride, p1);
    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
    store_row(buf, point + stride, q1);
}

/// Normal vertical edge filter (macroblock edge)
#[arcane]
pub(crate) fn normal_v_filter16_edge_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_row(buf, point - 4 * stride);
    let mut p2 = load_row(buf, point - 3 * stride);
    let mut p1 = load_row(buf, point - 2 * stride);
    let mut p0 = load_row(buf, point - stride);
    let mut q0 = load_row(buf, point);
    let mut q1 = load_row(buf, point + stride);
    let mut q2 = load_row(buf, point + 2 * stride);
    let q3 = load_row(buf, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_row(buf, point - 3 * stride, p2);
    store_row(buf, point - 2 * stride, p1);
    store_row(buf, point - stride, p0);
    store_row(buf, point, q0);
    store_row(buf, point + stride, q1);
    store_row(buf, point + 2 * stride, q2);
}

/// Normal horizontal inner filter (subblock edge): vertical edge at x0, 16 rows
#[arcane]
pub(crate) fn normal_h_filter16_inner_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let base = y_start * stride + x0;
    let p3 = load_col(buf, base - 4, stride);
    let p2 = load_col(buf, base - 3, stride);
    let mut p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let mut q1 = load_col(buf, base + 1, stride);
    let q2 = load_col(buf, base + 2, stride);
    let q3 = load_col(buf, base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_col(buf, base - 2, stride, p1);
    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
    store_col(buf, base + 1, stride, q1);
}

/// Normal horizontal edge filter (macroblock edge): vertical edge at x0, 16 rows
#[arcane]
pub(crate) fn normal_h_filter16_edge_wasm(
    _token: Wasm128Token,
    buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let base = y_start * stride + x0;
    let p3 = load_col(buf, base - 4, stride);
    let mut p2 = load_col(buf, base - 3, stride);
    let mut p1 = load_col(buf, base - 2, stride);
    let mut p0 = load_col(buf, base - 1, stride);
    let mut q0 = load_col(buf, base, stride);
    let mut q1 = load_col(buf, base + 1, stride);
    let mut q2 = load_col(buf, base + 2, stride);
    let q3 = load_col(buf, base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_col(buf, base - 3, stride, p2);
    store_col(buf, base - 2, stride, p1);
    store_col(buf, base - 1, stride, p0);
    store_col(buf, base, stride, q0);
    store_col(buf, base + 1, stride, q1);
    store_col(buf, base + 2, stride, q2);
}

// =============================================================================
// UV (chroma) filters: pack U+V 8-pixel halves into 16-wide registers
// =============================================================================

/// Load 8 bytes from U and 8 bytes from V, packed into one u8x16
#[inline(always)]
fn load_uv_col(u_buf: &[u8], v_buf: &[u8], u_base: usize, v_base: usize, stride: usize) -> v128 {
    u8x16(
        u_buf[u_base],
        u_buf[u_base + stride],
        u_buf[u_base + stride * 2],
        u_buf[u_base + stride * 3],
        u_buf[u_base + stride * 4],
        u_buf[u_base + stride * 5],
        u_buf[u_base + stride * 6],
        u_buf[u_base + stride * 7],
        v_buf[v_base],
        v_buf[v_base + stride],
        v_buf[v_base + stride * 2],
        v_buf[v_base + stride * 3],
        v_buf[v_base + stride * 4],
        v_buf[v_base + stride * 5],
        v_buf[v_base + stride * 6],
        v_buf[v_base + stride * 7],
    )
}

/// Store 8 bytes to U and 8 bytes to V from packed u8x16
#[inline(always)]
fn store_uv_col(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    u_base: usize,
    v_base: usize,
    stride: usize,
    v: v128,
) {
    u_buf[u_base] = u8x16_extract_lane::<0>(v);
    u_buf[u_base + stride] = u8x16_extract_lane::<1>(v);
    u_buf[u_base + stride * 2] = u8x16_extract_lane::<2>(v);
    u_buf[u_base + stride * 3] = u8x16_extract_lane::<3>(v);
    u_buf[u_base + stride * 4] = u8x16_extract_lane::<4>(v);
    u_buf[u_base + stride * 5] = u8x16_extract_lane::<5>(v);
    u_buf[u_base + stride * 6] = u8x16_extract_lane::<6>(v);
    u_buf[u_base + stride * 7] = u8x16_extract_lane::<7>(v);
    v_buf[v_base] = u8x16_extract_lane::<8>(v);
    v_buf[v_base + stride] = u8x16_extract_lane::<9>(v);
    v_buf[v_base + stride * 2] = u8x16_extract_lane::<10>(v);
    v_buf[v_base + stride * 3] = u8x16_extract_lane::<11>(v);
    v_buf[v_base + stride * 4] = u8x16_extract_lane::<12>(v);
    v_buf[v_base + stride * 5] = u8x16_extract_lane::<13>(v);
    v_buf[v_base + stride * 6] = u8x16_extract_lane::<14>(v);
    v_buf[v_base + stride * 7] = u8x16_extract_lane::<15>(v);
}

/// Load UV row: 8 bytes from U row + 8 bytes from V row packed into u8x16
#[inline(always)]
fn load_uv_row(u_buf: &[u8], v_buf: &[u8], u_off: usize, v_off: usize) -> v128 {
    u8x16(
        u_buf[u_off],
        u_buf[u_off + 1],
        u_buf[u_off + 2],
        u_buf[u_off + 3],
        u_buf[u_off + 4],
        u_buf[u_off + 5],
        u_buf[u_off + 6],
        u_buf[u_off + 7],
        v_buf[v_off],
        v_buf[v_off + 1],
        v_buf[v_off + 2],
        v_buf[v_off + 3],
        v_buf[v_off + 4],
        v_buf[v_off + 5],
        v_buf[v_off + 6],
        v_buf[v_off + 7],
    )
}

/// Store UV row
#[inline(always)]
fn store_uv_row(u_buf: &mut [u8], v_buf: &mut [u8], u_off: usize, v_off: usize, v: v128) {
    u_buf[u_off] = u8x16_extract_lane::<0>(v);
    u_buf[u_off + 1] = u8x16_extract_lane::<1>(v);
    u_buf[u_off + 2] = u8x16_extract_lane::<2>(v);
    u_buf[u_off + 3] = u8x16_extract_lane::<3>(v);
    u_buf[u_off + 4] = u8x16_extract_lane::<4>(v);
    u_buf[u_off + 5] = u8x16_extract_lane::<5>(v);
    u_buf[u_off + 6] = u8x16_extract_lane::<6>(v);
    u_buf[u_off + 7] = u8x16_extract_lane::<7>(v);
    v_buf[v_off] = u8x16_extract_lane::<8>(v);
    v_buf[v_off + 1] = u8x16_extract_lane::<9>(v);
    v_buf[v_off + 2] = u8x16_extract_lane::<10>(v);
    v_buf[v_off + 3] = u8x16_extract_lane::<11>(v);
    v_buf[v_off + 4] = u8x16_extract_lane::<12>(v);
    v_buf[v_off + 5] = u8x16_extract_lane::<13>(v);
    v_buf[v_off + 6] = u8x16_extract_lane::<14>(v);
    v_buf[v_off + 7] = u8x16_extract_lane::<15>(v);
}

/// Normal vertical UV edge filter
#[arcane]
pub(crate) fn normal_v_filter_uv_edge_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_uv_row(u_buf, v_buf, point - 4 * stride, point - 4 * stride);
    let mut p2 = load_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride);
    let mut p1 = load_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride);
    let mut p0 = load_uv_row(u_buf, v_buf, point - stride, point - stride);
    let mut q0 = load_uv_row(u_buf, v_buf, point, point);
    let mut q1 = load_uv_row(u_buf, v_buf, point + stride, point + stride);
    let mut q2 = load_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride);
    let q3 = load_uv_row(u_buf, v_buf, point + 3 * stride, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride, p2);
    store_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride, p1);
    store_uv_row(u_buf, v_buf, point - stride, point - stride, p0);
    store_uv_row(u_buf, v_buf, point, point, q0);
    store_uv_row(u_buf, v_buf, point + stride, point + stride, q1);
    store_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride, q2);
}

/// Normal vertical UV inner filter
#[arcane]
pub(crate) fn normal_v_filter_uv_inner_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    point: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let p3 = load_uv_row(u_buf, v_buf, point - 4 * stride, point - 4 * stride);
    let p2 = load_uv_row(u_buf, v_buf, point - 3 * stride, point - 3 * stride);
    let mut p1 = load_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride);
    let mut p0 = load_uv_row(u_buf, v_buf, point - stride, point - stride);
    let mut q0 = load_uv_row(u_buf, v_buf, point, point);
    let mut q1 = load_uv_row(u_buf, v_buf, point + stride, point + stride);
    let q2 = load_uv_row(u_buf, v_buf, point + 2 * stride, point + 2 * stride);
    let q3 = load_uv_row(u_buf, v_buf, point + 3 * stride, point + 3 * stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_uv_row(u_buf, v_buf, point - 2 * stride, point - 2 * stride, p1);
    store_uv_row(u_buf, v_buf, point - stride, point - stride, p0);
    store_uv_row(u_buf, v_buf, point, point, q0);
    store_uv_row(u_buf, v_buf, point + stride, point + stride, q1);
}

/// Normal horizontal UV edge filter
#[arcane]
pub(crate) fn normal_h_filter_uv_edge_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let u_base = y_start * stride + x0;
    let v_base = y_start * stride + x0;
    let p3 = load_uv_col(u_buf, v_buf, u_base - 4, v_base - 4, stride);
    let mut p2 = load_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride);
    let mut p1 = load_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride);
    let mut p0 = load_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride);
    let mut q0 = load_uv_col(u_buf, v_buf, u_base, v_base, stride);
    let mut q1 = load_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride);
    let mut q2 = load_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride);
    let q3 = load_uv_col(u_buf, v_buf, u_base + 3, v_base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter6(
        &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev_mask,
    );

    store_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride, p2);
    store_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride, p1);
    store_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride, p0);
    store_uv_col(u_buf, v_buf, u_base, v_base, stride, q0);
    store_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride, q1);
    store_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride, q2);
}

/// Normal horizontal UV inner filter
#[arcane]
pub(crate) fn normal_h_filter_uv_inner_wasm(
    _token: Wasm128Token,
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    x0: usize,
    y_start: usize,
    stride: usize,
    hev_threshold: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let u_base = y_start * stride + x0;
    let v_base = y_start * stride + x0;
    let p3 = load_uv_col(u_buf, v_buf, u_base - 4, v_base - 4, stride);
    let p2 = load_uv_col(u_buf, v_buf, u_base - 3, v_base - 3, stride);
    let mut p1 = load_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride);
    let mut p0 = load_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride);
    let mut q0 = load_uv_col(u_buf, v_buf, u_base, v_base, stride);
    let mut q1 = load_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride);
    let q2 = load_uv_col(u_buf, v_buf, u_base + 2, v_base + 2, stride);
    let q3 = load_uv_col(u_buf, v_buf, u_base + 3, v_base + 3, stride);

    let mask = needs_filter2(p3, p2, p1, p0, q0, q1, q2, q3, interior_limit, edge_limit);
    let hev_mask = hev(p1, p0, q0, q1, hev_threshold);

    do_filter4(&mut p1, &mut p0, &mut q0, &mut q1, mask, hev_mask);

    store_uv_col(u_buf, v_buf, u_base - 2, v_base - 2, stride, p1);
    store_uv_col(u_buf, v_buf, u_base - 1, v_base - 1, stride, p0);
    store_uv_col(u_buf, v_buf, u_base, v_base, stride, q0);
    store_uv_col(u_buf, v_buf, u_base + 1, v_base + 1, stride, q1);
}
