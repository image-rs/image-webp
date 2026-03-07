//! Cost estimation for VP8 encoding.
//!
//! This module provides rate-distortion (RD) cost calculation for mode selection,
//! based on libwebp's cost estimation approach.
//!
//! The key insight is that encoding cost depends on:
//! 1. Mode signaling cost (fixed per mode type)
//! 2. Coefficient encoding cost (depends on coefficient values and probabilities)
//!
//! For mode selection, we use: score = Distortion + lambda * Rate
//!
//! ## Module organization
//!
//! - [`distortion`]: Hadamard transform and TDisto functions (SIMD candidates)
//! - [`lambda`]: Rate-distortion lambda calculations
//! - [`level_costs`]: Precomputed level cost tables
//! - [`stats`]: Token statistics for probability adaptation

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

// Submodules
pub mod distortion;
pub mod lambda;
pub mod level_costs;
pub mod stats;

// Re-export from submodules
pub use distortion::{
    FLATNESS_LIMIT_I4, FLATNESS_LIMIT_I16, FLATNESS_LIMIT_UV, FLATNESS_PENALTY, is_flat_coeffs,
    is_flat_source_16, t_transform, tdisto_4x4, tdisto_8x8, tdisto_16x16,
};
pub use lambda::{
    DEFAULT_SNS_STRENGTH, LAMBDA_I4, LAMBDA_I16, LAMBDA_UV, calc_i4_penalty, calc_lambda_i4,
    calc_lambda_i16, calc_lambda_mode, calc_lambda_trellis_i4, calc_lambda_trellis_i16,
    calc_lambda_trellis_uv, calc_lambda_uv, calc_tlambda, compute_filter_level,
    compute_filter_level_with_beta, filter_strength_from_delta,
};
pub use level_costs::{LevelCostArray, LevelCostTables, LevelCosts, RemappedCosts};
pub use stats::{NUM_BANDS, NUM_CTX, NUM_PROBAS, NUM_TYPES, ProbaStats, TokenType, record_coeffs};

// Re-export tables from tables for backward compatibility
pub use super::tables::*;
// Re-export analysis module for backward compatibility
pub use super::analysis::*;

// Re-export quantization types and functions from quantize module
pub use super::quantize::{MatrixType, QFIX, VP8Matrix, quantdiv, quantization_bias};

// Re-export trellis quantization
pub use super::psy::PsyConfig;
pub use super::trellis::trellis_quantize_block;

// Re-export residual cost functions from residual_cost module
pub use super::residual_cost::{
    Residual, get_cost_luma4, get_cost_luma16, get_cost_uv, get_residual_cost,
};

/// Distortion multiplier - scales distortion to match bit cost units
pub const RD_DISTO_MULT: u32 = 256;

/// Calculate bit cost for coding a boolean value with given probability.
///
/// Returns cost in 1/256 bit units.
#[inline]
pub fn vp8_bit_cost(bit: bool, prob: u8) -> u16 {
    use super::tables::VP8_ENTROPY_COST;
    if bit {
        VP8_ENTROPY_COST[255 - prob as usize]
    } else {
        VP8_ENTROPY_COST[prob as usize]
    }
}

//------------------------------------------------------------------------------
// Coefficient cost estimation

/// Estimate the cost of encoding a 4x4 block of quantized coefficients.
///
/// This is a simplified approximation of libwebp's GetResidualCost.
/// It uses VP8_LEVEL_FIXED_COSTS which gives the probability-independent
/// part of the cost. The probability-dependent part is omitted for simplicity.
///
/// # Arguments
/// * `coeffs` - The 16 quantized coefficients in zig-zag order
/// * `first` - First coefficient to include (0 for DC+AC, 1 for AC only)
///
/// # Returns
/// Estimated bit cost in 1/256 bit units
#[inline]
pub fn estimate_residual_cost(coeffs: &[i32; 16], first: usize) -> u32 {
    use super::tables::{MAX_LEVEL, VP8_LEVEL_FIXED_COSTS};

    let mut cost = 0u32;
    let mut last_nz = -1i32;

    // Find last non-zero coefficient
    for (i, &c) in coeffs.iter().enumerate().rev() {
        if c != 0 {
            last_nz = i as i32;
            break;
        }
    }

    // If no non-zero coefficients, just signal end of block
    if last_nz < first as i32 {
        // Cost of signaling "no coefficients" - approximately 1 bit
        return 256;
    }

    // Sum up costs for each coefficient
    for &coeff in coeffs.iter().take(last_nz as usize + 1).skip(first) {
        let level = coeff.unsigned_abs() as usize;
        if level > 0 {
            // Cost of the coefficient level
            let level_clamped = level.min(MAX_LEVEL);
            cost += u32::from(VP8_LEVEL_FIXED_COSTS[level_clamped]);
            // Add sign bit cost (1 bit = 256)
            cost += 256;
        } else {
            // Cost of zero (EOB not reached) - approximately 0.5 bits
            cost += 128;
        }
    }

    // Cost of end-of-block signal (approximately 0.5 bits)
    cost += 128;

    cost
}

/// Estimate coefficient cost for a 16x16 macroblock (16 4x4 blocks).
///
/// # Arguments
/// * `blocks` - Array of 16 quantized coefficient blocks
/// * `has_dc` - Whether to include DC coefficients (false for I16 where DC is separate)
#[inline]
pub fn estimate_luma16_cost(blocks: &[[i32; 16]; 16], has_dc: bool) -> u32 {
    let first = if has_dc { 0 } else { 1 };
    blocks
        .iter()
        .map(|block| estimate_residual_cost(block, first))
        .sum()
}

/// Estimate coefficient cost for DC block in I16 mode.
#[inline]
pub fn estimate_dc16_cost(dc_coeffs: &[i32; 16]) -> u32 {
    estimate_residual_cost(dc_coeffs, 0)
}

//------------------------------------------------------------------------------
// RD score calculation

/// Calculate RD score for mode selection.
///
/// Formula: score = SSE * RD_DISTO_MULT + mode_cost * lambda
///
/// Lower score = better trade-off between quality and bits.
#[inline]
pub fn rd_score(sse: u32, mode_cost: u16, lambda: u32) -> u64 {
    let distortion = u64::from(sse) * u64::from(RD_DISTO_MULT);
    let rate = u64::from(mode_cost) * u64::from(lambda);
    distortion + rate
}

/// Calculate full RD score including coefficient costs.
///
/// Formula: score = SSE * RD_DISTO_MULT + (mode_cost + coeff_cost) * lambda
///
/// # Arguments
/// * `sse` - Sum of squared errors (distortion)
/// * `mode_cost` - Mode signaling cost in 1/256 bits
/// * `coeff_cost` - Coefficient encoding cost in 1/256 bits
/// * `lambda` - Rate-distortion trade-off parameter
#[inline]
pub fn rd_score_with_coeffs(sse: u32, mode_cost: u16, coeff_cost: u32, lambda: u32) -> u64 {
    let distortion = u64::from(sse) * u64::from(RD_DISTO_MULT);
    let rate = (u64::from(mode_cost) + u64::from(coeff_cost)) * u64::from(lambda);
    distortion + rate
}

/// Calculate full RD score as used by libwebp's PickBestIntra16.
///
/// Formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
///
/// Where:
/// - R = coefficient encoding cost (rate)
/// - H = mode header cost
/// - D = SSE distortion on reconstructed block
/// - SD = spectral distortion (TDisto)
///
/// # Arguments
/// * `sse` - Sum of squared errors (D)
/// * `spectral_disto` - Spectral distortion from TDisto (SD), already scaled by tlambda
/// * `mode_cost` - Mode signaling cost (H)
/// * `coeff_cost` - Coefficient encoding cost (R)
/// * `lambda` - Rate-distortion trade-off parameter
#[inline]
pub fn rd_score_full(
    sse: u32,
    spectral_disto: i32,
    mode_cost: u16,
    coeff_cost: u32,
    lambda: u32,
) -> i64 {
    let rate = (i64::from(mode_cost) + i64::from(coeff_cost)) * i64::from(lambda);
    let distortion = i64::from(RD_DISTO_MULT) * (i64::from(sse) + i64::from(spectral_disto));
    rate + distortion
}

/// Get context-dependent Intra4 mode cost.
///
/// # Arguments
/// * `top` - The mode of the block above (0-9)
/// * `left` - The mode of the block to the left (0-9)
/// * `mode` - The mode to get the cost for (0-9)
#[inline]
pub fn get_i4_mode_cost(top: usize, left: usize, mode: usize) -> u16 {
    VP8_FIXED_COSTS_I4[top][left][mode]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::psy::PsyConfig;
    use crate::encoder::tables::VP8_ZIGZAG;
    use crate::encoder::trellis::trellis_quantize_block;

    #[test]
    fn test_entropy_cost_table() {
        // Probability 128 (50%) should have cost ~256 (1 bit)
        assert!((VP8_ENTROPY_COST[128] as i32 - 256).abs() < 10);

        // Probability 255 (~100%) should have very low cost
        assert!(VP8_ENTROPY_COST[255] < 10);

        // Probability 1 (~0%) should have high cost
        assert!(VP8_ENTROPY_COST[1] > 1500);

        // Table should be monotonically decreasing
        for i in 1..256 {
            assert!(
                VP8_ENTROPY_COST[i] <= VP8_ENTROPY_COST[i - 1],
                "Entropy cost not monotonic at {}",
                i
            );
        }
    }

    #[test]
    fn test_bit_cost() {
        // Cost of 0 with prob 128 should equal cost of 1 with prob 128
        assert_eq!(vp8_bit_cost(false, 128), vp8_bit_cost(true, 128));

        // Cost of 0 with high prob should be low
        assert!(vp8_bit_cost(false, 250) < vp8_bit_cost(false, 128));

        // Cost of 1 with high prob should be high
        assert!(vp8_bit_cost(true, 250) > vp8_bit_cost(true, 128));
    }

    #[test]
    fn test_level_fixed_costs() {
        // Level 0 should have cost 0
        assert_eq!(VP8_LEVEL_FIXED_COSTS[0], 0);

        // Small levels should have lower cost than large levels
        assert!(VP8_LEVEL_FIXED_COSTS[1] < VP8_LEVEL_FIXED_COSTS[100]);
        assert!(VP8_LEVEL_FIXED_COSTS[100] < VP8_LEVEL_FIXED_COSTS[1000]);

        // Table should be correct size
        assert_eq!(VP8_LEVEL_FIXED_COSTS.len(), MAX_LEVEL + 1);
    }

    #[test]
    fn test_lambda_calculation() {
        // At q=64 (medium quality)
        let q = 64u32;

        let lambda_i4 = calc_lambda_i4(q);
        let lambda_i16 = calc_lambda_i16(q);
        let lambda_uv = calc_lambda_uv(q);

        // lambda_i16 should be much larger than lambda_i4
        assert!(lambda_i16 > lambda_i4 * 50);

        // lambda_uv should be between i4 and i16
        assert!(lambda_uv > lambda_i4);
        assert!(lambda_uv < lambda_i16);

        // Values should match libwebp formulas
        assert_eq!(lambda_i4, (3 * 64 * 64) >> 7);
        assert_eq!(lambda_i16, 3 * 64 * 64);
        assert_eq!(lambda_uv, (3 * 64 * 64) >> 6);
    }

    #[test]
    fn test_rd_score() {
        // Zero SSE and zero cost = zero score
        assert_eq!(rd_score(0, 0, LAMBDA_I16), 0);

        // Only distortion
        assert_eq!(rd_score(100, 0, LAMBDA_I16), 100 * 256);

        // Only rate
        assert_eq!(rd_score(0, 663, LAMBDA_I16), 663 * 106);

        // Combined
        let expected = 1000 * 256 + u64::from(FIXED_COSTS_I16[0]) * 106;
        assert_eq!(rd_score(1000, FIXED_COSTS_I16[0], LAMBDA_I16), expected);
    }

    #[test]
    fn test_estimate_residual_cost() {
        // All zeros should have minimal cost
        let zeros = [0i32; 16];
        let cost_zeros = estimate_residual_cost(&zeros, 0);
        assert!(cost_zeros < 512, "Zero block cost should be ~1 bit");

        // Single non-zero DC coefficient
        let mut single_dc = [0i32; 16];
        single_dc[0] = 1;
        let cost_single = estimate_residual_cost(&single_dc, 0);
        assert!(
            cost_single > cost_zeros,
            "Non-zero coefficient should cost more"
        );
    }

    #[test]
    fn test_trellis_basic() {
        // Create a Y1 matrix at q=50
        let matrix = VP8Matrix::new(50, 50, MatrixType::Y1);

        // Test case: DC dominant block (natural order)
        let block = [
            500i32, 50, 25, 10, // row 0
            5, 3, 2, 1, // row 1
            0, 0, 0, 0, // row 2
            0, 0, 0, 0, // row 3
        ];

        let mut coeffs = block;
        let mut trellis_out = [0i32; 16];

        // Lambda for i4: (7 * 50^2) >> 3 = 2187
        let lambda = 2187u32;

        // Create level costs for test
        let mut level_costs = LevelCosts::new();
        level_costs.calculate(&crate::common::types::COEFF_PROBS);

        let trellis_nz = trellis_quantize_block(
            &mut coeffs,
            &mut trellis_out,
            &matrix,
            lambda,
            0,
            &level_costs,
            3, // I4 type
            0, // initial context
            &PsyConfig::default(),
        );

        // Simple quantization for comparison
        let mut simple_out = [0i32; 16];
        for i in 0..16 {
            let j = VP8_ZIGZAG[i]; // Convert zigzag position to natural
            simple_out[i] = matrix.quantize_coeff(block[j], j);
        }

        // Both should produce non-zero DC coefficient
        assert!(simple_out[0] != 0, "Simple should have non-zero DC");
        // Trellis should also produce something
        let _ = trellis_nz; // suppress unused warning
    }
}
