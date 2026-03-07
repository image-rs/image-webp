//! Loop filter dispatch functions for VP8 decoder.
//!
//! These functions dispatch to SIMD implementations when available,
//! falling back to scalar implementations.
//!
//! All functions accept a pre-summoned `SimdTokenType` to avoid per-call
//! token summoning overhead (~4M instructions per decode eliminated).

#![allow(clippy::too_many_arguments)]
// On WASM, SimdTokenType = Option<()> and simd_token is passed but not dispatched
#![allow(unused_variables)]

use super::loop_filter;
use crate::common::prediction::SimdTokenType;

/// Helper to apply simple horizontal filter to 16 rows with SIMD when available.
/// Filters the vertical edge at column x0, processing all 16 rows at once.
#[inline]
pub(crate) fn simple_filter_horizontal_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::simple_h_filter16(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        super::loop_filter_neon::simple_h_filter16_neon(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        super::loop_filter_wasm::simple_h_filter16_wasm(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for y in 0usize..16 {
        let y0 = y_start + y;
        loop_filter::simple_segment_horizontal(edge_limit, &mut buf[y0 * stride + x0 - 4..][..8]);
    }
}

/// Helper to apply simple vertical filter to 16 columns with SIMD when available.
/// Filters the horizontal edge at row y0, processing all 16 columns at once.
#[inline]
pub(crate) fn simple_filter_vertical_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::simple_v_filter16(
            token,
            buf,
            point,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_neon::simple_v_filter16_neon(
            token,
            buf,
            point,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_wasm::simple_v_filter16_wasm(
            token,
            buf,
            point,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::simple_segment_vertical(edge_limit, buf, point, stride);
    }
}

/// Helper to apply simple vertical filter to 32 columns with AVX2.
/// Filters the horizontal edge at row y0, processing 32 columns (2 macroblocks) at once.
/// This is 2x faster than two calls to simple_filter_vertical_16_cols.
#[inline]
#[allow(dead_code)]
pub(crate) fn simple_filter_vertical_32_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::simple_v_filter32(
            token,
            buf,
            point,
            stride,
            i32::from(edge_limit),
        );
        return;
    }

    // Fallback: two 16-column filters
    simple_filter_vertical_16_cols(buf, y0, x_start, stride, edge_limit, simd_token);
    simple_filter_vertical_16_cols(buf, y0, x_start + 16, stride, edge_limit, simd_token);
}

/// Helper to apply normal vertical macroblock filter to 16 columns with SIMD when available.
/// Filters the horizontal edge at row y0, processing all 16 columns at once.
#[inline]
pub(crate) fn normal_filter_vertical_mb_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter16_edge(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_neon::normal_v_filter16_edge_neon(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_wasm::normal_v_filter16_edge_wasm(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Helper to apply normal vertical subblock filter to 16 columns with SIMD when available.
#[inline]
pub(crate) fn normal_filter_vertical_sub_16_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter16_inner(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_neon::normal_v_filter16_inner_neon(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_wasm::normal_v_filter16_inner_wasm(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for x in 0usize..16 {
        let point = y0 * stride + x_start + x;
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            buf,
            point,
            stride,
        );
    }
}

/// Helper to apply normal vertical macroblock filter to 32 columns with AVX2.
/// Filters the horizontal edge at row y0, processing 32 columns (2 macroblocks) at once.
/// 2x faster than two calls to normal_filter_vertical_mb_16_cols.
#[inline]
#[allow(dead_code)]
pub(crate) fn normal_filter_vertical_mb_32_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter32_edge(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Fallback: two 16-column filters
    normal_filter_vertical_mb_16_cols(
        buf,
        y0,
        x_start,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
    normal_filter_vertical_mb_16_cols(
        buf,
        y0,
        x_start + 16,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
}

/// Helper to apply normal vertical subblock filter to 32 columns with AVX2.
/// 2x faster than two calls to normal_filter_vertical_sub_16_cols.
#[inline]
#[allow(dead_code)]
pub(crate) fn normal_filter_vertical_sub_32_cols(
    buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter32_inner(
            token,
            buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Fallback: two 16-column filters
    normal_filter_vertical_sub_16_cols(
        buf,
        y0,
        x_start,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
    normal_filter_vertical_sub_16_cols(
        buf,
        y0,
        x_start + 16,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
}

/// Helper to apply normal horizontal macroblock filter to 16 rows with SIMD when available.
/// Filters the vertical edge at column x0, processing all 16 rows at once.
#[inline]
pub(crate) fn normal_filter_horizontal_mb_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter16_edge(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        super::loop_filter_neon::normal_h_filter16_edge_neon(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        super::loop_filter_wasm::normal_h_filter16_edge_wasm(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for y in 0usize..16 {
        let row = y_start + y;
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Helper to apply normal horizontal subblock filter to 16 rows with SIMD when available.
#[inline]
pub(crate) fn normal_filter_horizontal_sub_16_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter16_inner(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        super::loop_filter_neon::normal_h_filter16_inner_neon(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        super::loop_filter_wasm::normal_h_filter16_inner_wasm(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for y in 0usize..16 {
        let row = y_start + y;
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Helper to apply normal horizontal macroblock filter to U and V chroma planes (8 rows each).
/// Processes both planes together as 16 rows for SIMD efficiency.
#[inline]
pub(crate) fn normal_filter_horizontal_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter_uv_edge(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        super::loop_filter_neon::normal_h_filter_uv_edge_neon(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        super::loop_filter_wasm::normal_h_filter_uv_edge_wasm(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for y in 0usize..8 {
        let row = y_start + y;
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        loop_filter::macroblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Helper to apply normal horizontal subblock filter to U and V chroma planes (8 rows each).
#[inline]
pub(crate) fn normal_filter_horizontal_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter_uv_inner(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        super::loop_filter_neon::normal_h_filter_uv_inner_neon(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        super::loop_filter_wasm::normal_h_filter_uv_inner_wasm(
            token,
            u_buf,
            v_buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for y in 0usize..8 {
        let row = y_start + y;
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut u_buf[row * stride + x0 - 4..][..8],
        );
        loop_filter::subblock_filter_horizontal(
            hev_threshold,
            interior_limit,
            edge_limit,
            &mut v_buf[row * stride + x0 - 4..][..8],
        );
    }
}

/// Helper to apply normal vertical macroblock filter to U and V chroma planes (8 pixels each).
/// Filters the horizontal edge at row y0, processing 8 U + 8 V pixels together.
#[inline]
pub(crate) fn normal_filter_vertical_uv_mb(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter_uv_edge(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_neon::normal_v_filter_uv_edge_neon(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_wasm::normal_v_filter_uv_edge_wasm(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        loop_filter::macroblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

/// Helper to apply normal vertical subblock filter to U and V chroma planes (8 pixels each).
/// Filters the horizontal edge at row y0, processing 8 U + 8 V pixels together.
#[inline]
pub(crate) fn normal_filter_vertical_uv_sub(
    u_buf: &mut [u8],
    v_buf: &mut [u8],
    y0: usize,
    x_start: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_avx2::normal_v_filter_uv_inner(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_neon::normal_v_filter_uv_inner_neon(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
    if let Some(token) = simd_token {
        let point = y0 * stride + x_start;
        super::loop_filter_wasm::normal_v_filter_uv_inner_wasm(
            token,
            u_buf,
            v_buf,
            point,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Scalar fallback
    for x in 0usize..8 {
        let point = y0 * stride + x_start + x;
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            u_buf,
            point,
            stride,
        );
        loop_filter::subblock_filter_vertical(
            hev_threshold,
            interior_limit,
            edge_limit,
            v_buf,
            point,
            stride,
        );
    }
}

// ============================================================================
// AVX2 32-row horizontal filter dispatch functions
// These process 32 rows at once (e.g., two vertically adjacent macroblocks).
// Marked dead_code until decoder restructuring enables their use.
// ============================================================================

/// Helper to apply normal horizontal macroblock filter to 32 rows with AVX2.
/// Filters the vertical edge at column x0, processing 32 rows (2 macroblocks) at once.
#[inline]
#[allow(dead_code)]
pub(crate) fn normal_filter_horizontal_mb_32_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter32_edge(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Fallback: two 16-row filters
    normal_filter_horizontal_mb_16_rows(
        buf,
        y_start,
        x0,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
    normal_filter_horizontal_mb_16_rows(
        buf,
        y_start + 16,
        x0,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
}

/// Helper to apply normal horizontal subblock filter to 32 rows with AVX2.
/// Filters the vertical edge at column x0, processing 32 rows (2 macroblocks) at once.
#[inline]
#[allow(dead_code)]
pub(crate) fn normal_filter_horizontal_sub_32_rows(
    buf: &mut [u8],
    y_start: usize,
    x0: usize,
    stride: usize,
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    simd_token: SimdTokenType,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = simd_token {
        super::loop_filter_avx2::normal_h_filter32_inner(
            token,
            buf,
            x0,
            y_start,
            stride,
            i32::from(hev_threshold),
            i32::from(interior_limit),
            i32::from(edge_limit),
        );
        return;
    }

    // Fallback: two 16-row filters
    normal_filter_horizontal_sub_16_rows(
        buf,
        y_start,
        x0,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
    normal_filter_horizontal_sub_16_rows(
        buf,
        y_start + 16,
        x0,
        stride,
        hev_threshold,
        interior_limit,
        edge_limit,
        simd_token,
    );
}
