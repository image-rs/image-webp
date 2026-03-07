//! Residual coefficient cost estimation using probability-dependent tables.
//!
//! Contains SIMD-optimized GetResidualCost and block-level cost functions
//! (get_cost_luma4, get_cost_luma16, get_cost_uv).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

// SIMD imports for GetResidualCost optimization
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use core::arch::x86_64::*;

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
use archmage::{Wasm128Token, arcane, rite};
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
use core::arch::wasm32::*;

use super::cost::{LevelCosts, vp8_bit_cost};
use super::tables::VP8_ENC_BANDS;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "wasm32")))]
use super::tables::{MAX_LEVEL, MAX_VARIABLE_LEVEL, VP8_LEVEL_FIXED_COSTS};
use crate::common::types::TokenProbTables;

/// Residual coefficients for cost calculation.
/// Ported from libwebp's VP8Residual.
pub struct Residual<'a> {
    /// First coefficient to consider (0 or 1)
    pub first: usize,
    /// Last non-zero coefficient index (-1 if all zero)
    pub last: i32,
    /// Coefficient array
    pub coeffs: &'a [i32; 16],
    /// Coefficient type (0=I16DC, 1=I16AC, 2=Chroma, 3=I4)
    pub coeff_type: usize,
}

impl<'a> Residual<'a> {
    /// Create a new residual from coefficients.
    /// Automatically finds the last non-zero coefficient.
    pub fn new(coeffs: &'a [i32; 16], coeff_type: usize, first: usize) -> Self {
        let last = coeffs
            .iter()
            .rposition(|&c| c != 0)
            .map(|i| i as i32)
            .unwrap_or(-1);

        Self {
            first,
            last,
            coeffs,
            coeff_type,
        }
    }
}

/// Calculate the cost of encoding a residual block using probability-based costs.
/// Ported from libwebp's GetResidualCost_C.
///
/// On x86_64 with SIMD feature, uses SSE2 for precomputing absolute values and contexts.
/// Falls back to scalar implementation on other platforms.
///
/// # Arguments
/// * `ctx0` - Initial context (0, 1, or 2)
/// * `res` - Residual coefficients
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables (for last coefficient EOB cost)
///
/// # Returns
/// Cost in 1/256 bit units
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    if let Some(token) = X64V3Token::summon() {
        get_residual_cost_entry(token, ctx0, res, costs, probs)
    } else {
        get_residual_cost_scalar(ctx0, res, costs, probs)
    }
}

/// WASM SIMD128 dispatch for residual cost calculation.
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    use archmage::SimdToken;
    if let Some(token) = Wasm128Token::summon() {
        return get_residual_cost_wasm_entry(token, ctx0, res, costs, probs);
    }
    get_residual_cost_scalar(ctx0, res, costs, probs)
}

/// Calculate the cost of encoding a residual block using probability-based costs.
/// Scalar fallback for non-SIMD platforms.
#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "wasm32"))))]
#[inline]
pub fn get_residual_cost(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_scalar(ctx0, res, costs, probs)
}

/// Scalar implementation of residual cost calculation.
/// Ported from libwebp's GetResidualCost_C.
#[inline]
pub(crate) fn get_residual_cost_scalar(
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    let ctype = res.coeff_type;
    let mut n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0, updated after each coefficient
    let mut ctx = ctx0;

    // bit_cost(1, p0) is already incorporated in the cost tables, but only if ctx != 0.
    let mut cost = if ctx0 == 0 {
        vp8_bit_cost(true, p0) as u32
    } else {
        0
    };

    // If no non-zero coefficients, just return EOB cost
    if res.last < 0 {
        return vp8_bit_cost(false, p0) as u32;
    }

    // Process coefficients from first to last-1
    while (n as i32) < res.last {
        let v = res.coeffs[n].unsigned_abs() as usize;

        // Add cost using current context
        cost += costs.get_level_cost(ctype, n, ctx, v);

        // Update context for next position based on current value
        ctx = if v >= 2 { 2 } else { v };

        n += 1;
    }

    // Last coefficient is always non-zero
    {
        let v = res.coeffs[n].unsigned_abs() as usize;
        debug_assert!(v != 0, "Last coefficient should be non-zero");

        // Add cost using current context
        cost += costs.get_level_cost(ctype, n, ctx, v);

        // Add EOB cost for the position after the last coefficient
        if n < 15 {
            let next_band = VP8_ENC_BANDS[n + 1] as usize;
            let next_ctx = if v == 1 { 1 } else { 2 };
            let last_p0 = probs[ctype][next_band][next_ctx][0];
            cost += vp8_bit_cost(false, last_p0) as u32;
        }
    }

    cost
}

/// Entry shim for get_residual_cost_sse2
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
fn get_residual_cost_entry(
    _token: X64V3Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_sse2(_token, ctx0, res, costs, probs)
}

/// SSE2 implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
pub(crate) fn get_residual_cost_sse2(
    _token: X64V3Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    // Storage for precomputed values
    let mut ctxs: [u8; 16] = [0; 16];
    let mut levels: [u8; 16] = [0; 16];
    let mut abs_levels: [u16; 16] = [0; 16];

    let ctype = res.coeff_type;
    let mut n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let mut ctx = ctx0;

    // bit_cost(1, p0) is already incorporated in the cost tables, but only if ctx != 0.
    let mut cost = if ctx0 == 0 {
        vp8_bit_cost(true, p0) as u32
    } else {
        0
    };

    // If no non-zero coefficients, just return EOB cost
    if res.last < 0 {
        return vp8_bit_cost(false, p0) as u32;
    }

    // Precompute clamped levels and contexts using SIMD
    // libwebp uses i16 coefficients, but ours are i32. Pack them to i16.
    {
        let zero = _mm_setzero_si128();
        let k_cst2 = _mm_set1_epi8(2);
        let k_cst67 = _mm_set1_epi8(MAX_VARIABLE_LEVEL as i8);

        // Load coefficients as i32 and pack to i16
        let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[0..4]).unwrap());
        let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[4..8]).unwrap());
        let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[8..12]).unwrap());
        let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&res.coeffs[12..16]).unwrap());

        // Pack i32 to i16 (signed saturation)
        let c0 = _mm_packs_epi32(c0_32, c1_32); // 8 x i16
        let c1 = _mm_packs_epi32(c2_32, c3_32); // 8 x i16

        // Compute absolute values: abs(v) = max(v, -v)
        let d0 = _mm_sub_epi16(zero, c0);
        let d1 = _mm_sub_epi16(zero, c1);
        let e0 = _mm_max_epi16(c0, d0); // abs, 16-bit
        let e1 = _mm_max_epi16(c1, d1);

        // Pack to i8 for context and level clamping
        let f = _mm_packs_epi16(e0, e1); // 16 x i8 abs values

        // Context: min(abs, 2)
        let g = _mm_min_epu8(f, k_cst2);

        // Clamped level: min(abs, 67) for cost table lookup
        let h = _mm_min_epu8(f, k_cst67);

        // Store results
        simd_mem::_mm_storeu_si128(&mut ctxs, g);
        simd_mem::_mm_storeu_si128(&mut levels, h);

        // Store 16-bit absolute values for fixed cost lookup
        simd_mem::_mm_storeu_si128(
            <&mut [u16; 8]>::try_from(&mut abs_levels[0..8]).unwrap(),
            e0,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u16; 8]>::try_from(&mut abs_levels[8..16]).unwrap(),
            e1,
        );
    }

    // Pre-index the cost table by type (matches libwebp's CostArrayPtr pattern).
    // libwebp: t = costs[n][ctx0] then t = costs[n+1][ctx] in loop.
    // We pre-index by ctype once, then do costs_for_type[band][ctx][level] in loop.
    let costs_for_type = &costs.level_cost[ctype];
    let mut t = &costs_for_type[VP8_ENC_BANDS[n] as usize][ctx];

    // Process coefficients from first to last-1 using precomputed values
    while (n as i32) < res.last {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize;

        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;

        ctx = ctxs[n] as usize;
        n += 1;
        t = &costs_for_type[VP8_ENC_BANDS[n] as usize][ctx];
    }

    // Last coefficient is always non-zero
    {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize;

        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;

        // Add EOB cost for the position after the last coefficient
        if n < 15 {
            let next_band = VP8_ENC_BANDS[n + 1] as usize;
            let next_ctx = ctxs[n] as usize;
            let last_p0 = probs[ctype][next_band][next_ctx][0];
            cost += vp8_bit_cost(false, last_p0) as u32;
        }
    }

    cost
}

/// Entry shim for find_last_nonzero_simd
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[arcane]
#[allow(dead_code)]
fn find_last_nonzero_simd_entry(_token: X64V3Token, coeffs: &[i32; 16]) -> i32 {
    find_last_nonzero_simd(_token, coeffs)
}

/// Find last non-zero coefficient using SIMD.
/// Ported from libwebp's SetResidualCoeffs_SSE2.
///
/// Returns -1 if all coefficients are zero.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[rite]
#[allow(dead_code)]
fn find_last_nonzero_simd(_token: X64V3Token, coeffs: &[i32; 16]) -> i32 {
    let zero = _mm_setzero_si128();

    // Load coefficients as i32 and pack to i16
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    // Pack i32 to i16 (signed saturation)
    let c0 = _mm_packs_epi32(c0_32, c1_32); // 8 x i16
    let c1 = _mm_packs_epi32(c2_32, c3_32); // 8 x i16

    // Pack i16 to i8
    let m0 = _mm_packs_epi16(c0, c1); // 16 x i8

    // Compare with zero
    let m1 = _mm_cmpeq_epi8(m0, zero);

    // Get bitmask: bit is 1 if equal to zero
    let mask = 0x0000ffff_u32 ^ (_mm_movemask_epi8(m1) as u32);

    // Find position of most significant non-zero bit
    if mask == 0 {
        -1
    } else {
        (31 - mask.leading_zeros()) as i32
    }
}

// =============================================================================
// WASM SIMD128 residual cost implementation
// =============================================================================

/// Entry shim for get_residual_cost_wasm
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[arcane]
fn get_residual_cost_wasm_entry(
    _token: Wasm128Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    get_residual_cost_wasm(_token, ctx0, res, costs, probs)
}

/// WASM SIMD128 implementation of residual cost calculation.
/// Precomputes abs values, contexts, and clamped levels with SIMD.
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[rite]
pub(crate) fn get_residual_cost_wasm(
    _token: Wasm128Token,
    ctx0: usize,
    res: &Residual,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    // Storage for precomputed values
    let mut ctxs: [u8; 16] = [0; 16];
    let mut levels: [u8; 16] = [0; 16];
    let mut abs_levels: [u16; 16] = [0; 16];

    let ctype = res.coeff_type;
    let mut n = res.first;

    // Get probability p0 for the first coefficient
    let band = VP8_ENC_BANDS[n] as usize;
    let p0 = probs[ctype][band][ctx0][0];

    // Current context - starts at ctx0
    let mut ctx = ctx0;

    // bit_cost(1, p0) is already incorporated in the cost tables, but only if ctx != 0.
    let mut cost = if ctx0 == 0 {
        vp8_bit_cost(true, p0) as u32
    } else {
        0
    };

    // If no non-zero coefficients, just return EOB cost
    if res.last < 0 {
        return vp8_bit_cost(false, p0) as u32;
    }

    // Precompute clamped levels and contexts using SIMD
    {
        let k_cst2 = u8x16_splat(2);
        let k_cst67 = u8x16_splat(MAX_VARIABLE_LEVEL as u8);

        // Load coefficients as i32
        let c0 = i32x4(res.coeffs[0], res.coeffs[1], res.coeffs[2], res.coeffs[3]);
        let c1 = i32x4(res.coeffs[4], res.coeffs[5], res.coeffs[6], res.coeffs[7]);
        let c2 = i32x4(res.coeffs[8], res.coeffs[9], res.coeffs[10], res.coeffs[11]);
        let c3 = i32x4(
            res.coeffs[12],
            res.coeffs[13],
            res.coeffs[14],
            res.coeffs[15],
        );

        // Pack i32 → i16 (signed saturation)
        let s0 = i16x8_narrow_i32x4(c0, c1);
        let s1 = i16x8_narrow_i32x4(c2, c3);

        // Absolute value
        let e0 = i16x8_abs(s0);
        let e1 = i16x8_abs(s1);

        // Pack abs i16 → u8 (unsigned saturation, values are positive)
        let f = u8x16_narrow_i16x8(e0, e1);

        // Context: min(abs, 2)
        let g = u8x16_min(f, k_cst2);

        // Clamped level: min(abs, 67) for cost table lookup
        let h = u8x16_min(f, k_cst67);

        // Store ctxs (extract all 16 lanes)
        ctxs[0] = u8x16_extract_lane::<0>(g);
        ctxs[1] = u8x16_extract_lane::<1>(g);
        ctxs[2] = u8x16_extract_lane::<2>(g);
        ctxs[3] = u8x16_extract_lane::<3>(g);
        ctxs[4] = u8x16_extract_lane::<4>(g);
        ctxs[5] = u8x16_extract_lane::<5>(g);
        ctxs[6] = u8x16_extract_lane::<6>(g);
        ctxs[7] = u8x16_extract_lane::<7>(g);
        ctxs[8] = u8x16_extract_lane::<8>(g);
        ctxs[9] = u8x16_extract_lane::<9>(g);
        ctxs[10] = u8x16_extract_lane::<10>(g);
        ctxs[11] = u8x16_extract_lane::<11>(g);
        ctxs[12] = u8x16_extract_lane::<12>(g);
        ctxs[13] = u8x16_extract_lane::<13>(g);
        ctxs[14] = u8x16_extract_lane::<14>(g);
        ctxs[15] = u8x16_extract_lane::<15>(g);

        // Store levels
        levels[0] = u8x16_extract_lane::<0>(h);
        levels[1] = u8x16_extract_lane::<1>(h);
        levels[2] = u8x16_extract_lane::<2>(h);
        levels[3] = u8x16_extract_lane::<3>(h);
        levels[4] = u8x16_extract_lane::<4>(h);
        levels[5] = u8x16_extract_lane::<5>(h);
        levels[6] = u8x16_extract_lane::<6>(h);
        levels[7] = u8x16_extract_lane::<7>(h);
        levels[8] = u8x16_extract_lane::<8>(h);
        levels[9] = u8x16_extract_lane::<9>(h);
        levels[10] = u8x16_extract_lane::<10>(h);
        levels[11] = u8x16_extract_lane::<11>(h);
        levels[12] = u8x16_extract_lane::<12>(h);
        levels[13] = u8x16_extract_lane::<13>(h);
        levels[14] = u8x16_extract_lane::<14>(h);
        levels[15] = u8x16_extract_lane::<15>(h);

        // Store 16-bit absolute values
        abs_levels[0] = u16x8_extract_lane::<0>(e0);
        abs_levels[1] = u16x8_extract_lane::<1>(e0);
        abs_levels[2] = u16x8_extract_lane::<2>(e0);
        abs_levels[3] = u16x8_extract_lane::<3>(e0);
        abs_levels[4] = u16x8_extract_lane::<4>(e0);
        abs_levels[5] = u16x8_extract_lane::<5>(e0);
        abs_levels[6] = u16x8_extract_lane::<6>(e0);
        abs_levels[7] = u16x8_extract_lane::<7>(e0);
        abs_levels[8] = u16x8_extract_lane::<0>(e1);
        abs_levels[9] = u16x8_extract_lane::<1>(e1);
        abs_levels[10] = u16x8_extract_lane::<2>(e1);
        abs_levels[11] = u16x8_extract_lane::<3>(e1);
        abs_levels[12] = u16x8_extract_lane::<4>(e1);
        abs_levels[13] = u16x8_extract_lane::<5>(e1);
        abs_levels[14] = u16x8_extract_lane::<6>(e1);
        abs_levels[15] = u16x8_extract_lane::<7>(e1);
    }

    // Pre-index the cost table by type
    let costs_for_type = &costs.level_cost[ctype];
    let mut t = &costs_for_type[VP8_ENC_BANDS[n] as usize][ctx];

    // Process coefficients from first to last-1 using precomputed values
    while (n as i32) < res.last {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize;

        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;

        ctx = ctxs[n] as usize;
        n += 1;
        t = &costs_for_type[VP8_ENC_BANDS[n] as usize][ctx];
    }

    // Last coefficient is always non-zero
    {
        let level = levels[n] as usize;
        let flevel = abs_levels[n] as usize;

        cost += VP8_LEVEL_FIXED_COSTS[flevel.min(MAX_LEVEL)] as u32 + t[level] as u32;

        // Add EOB cost for the position after the last coefficient
        if n < 15 {
            let next_band = VP8_ENC_BANDS[n + 1] as usize;
            let next_ctx = ctxs[n] as usize;
            let last_p0 = probs[ctype][next_band][next_ctx][0];
            cost += vp8_bit_cost(false, last_p0) as u32;
        }
    }

    cost
}

/// Calculate the cost of encoding a 4x4 luma block (I4 mode).
/// Ported from libwebp's VP8GetCostLuma4.
///
/// # Arguments
/// * `levels` - Quantized coefficients in zigzag order
/// * `top_nz` - Whether the above block had non-zero coefficients
/// * `left_nz` - Whether the left block had non-zero coefficients
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// (cost, has_nonzero) - Cost in 1/256 bits and whether this block has non-zero coeffs
pub fn get_cost_luma4(
    levels: &[i32; 16],
    top_nz: bool,
    left_nz: bool,
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> (u32, bool) {
    // Initial context is sum of top and left non-zero flags (0, 1, or 2)
    let ctx = (top_nz as usize) + (left_nz as usize);

    // Create residual for I4 type (type 3), starting at position 0
    let res = Residual::new(levels, 3, 0);
    let has_nz = res.last >= 0;

    let cost = get_residual_cost(ctx, &res, costs, probs);

    (cost, has_nz)
}

/// Calculate the cost of encoding all 16 luma blocks in I16 mode.
/// Includes DC block (Y2) and 16 AC blocks (Y1).
///
/// # Arguments
/// * `dc_levels` - DC coefficients from WHT (16 values)
/// * `ac_levels` - AC coefficients for each 4x4 block (16 blocks × 16 coeffs)
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// Total cost in 1/256 bits
pub fn get_cost_luma16(
    dc_levels: &[i32; 16],
    ac_levels: &[[i32; 16]; 16],
    costs: &LevelCosts,
    probs: &TokenProbTables,
) -> u32 {
    let mut total_cost = 0u32;

    // DC block (type 1 = I16DC, also known as Y2)
    // Context is typically from neighboring DC blocks, but for simplicity use 0
    let dc_res = Residual::new(dc_levels, 1, 0);
    total_cost += get_residual_cost(0, &dc_res, costs, probs);

    // AC blocks (type 0 = I16AC, skipping DC coefficient which is in Y2)
    // Track non-zero context like libwebp's VP8GetCostLuma16
    let mut top_nz = [false; 4];
    let mut left_nz = [false; 4];

    for y in 0..4 {
        for x in 0..4 {
            let block_idx = y * 4 + x;
            let ctx = (top_nz[x] as usize) + (left_nz[y] as usize);
            let ac_res = Residual::new(&ac_levels[block_idx], 0, 1); // Start at position 1 (skip DC)
            total_cost += get_residual_cost(ctx, &ac_res, costs, probs);
            let has_nz = ac_res.last >= 0;
            top_nz[x] = has_nz;
            left_nz[y] = has_nz;
        }
    }

    total_cost
}

/// Compute accurate coefficient cost for UV blocks using probability-dependent tables.
///
/// Port of libwebp's VP8GetCostUV.
///
/// # Arguments
/// * `uv_levels` - 8 blocks of quantized coefficients (4 U blocks + 4 V blocks)
/// * `costs` - Precomputed level cost tables
/// * `probs` - Probability tables
///
/// # Returns
/// Total cost in 1/256 bits
pub fn get_cost_uv(uv_levels: &[[i32; 16]; 8], costs: &LevelCosts, probs: &TokenProbTables) -> u32 {
    let mut total_cost = 0u32;

    // UV blocks use coeff_type=2 (TYPE_CHROMA_A)
    // All coefficients including DC (first=0)
    for block in uv_levels.iter() {
        let res = Residual::new(block, 2, 0); // ctype=2 for UV, first=0 (include DC)
        total_cost += get_residual_cost(0, &res, costs, probs);
    }

    total_cost
}
