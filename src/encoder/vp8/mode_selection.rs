//! Intra prediction mode selection for macroblocks.
//!
//! Contains methods for selecting optimal I16, I4, and UV prediction modes
//! using rate-distortion cost evaluation.
#![allow(clippy::too_many_arguments)]

use crate::common::prediction::*;
use crate::common::transform;
use crate::common::types::*;

use crate::encoder::cost::{
    FIXED_COSTS_I16, FIXED_COSTS_UV, FLATNESS_LIMIT_I4, FLATNESS_LIMIT_I16, FLATNESS_LIMIT_UV,
    FLATNESS_PENALTY, RD_DISTO_MULT, estimate_dc16_cost, estimate_residual_cost, get_cost_luma4,
    get_cost_luma16, get_cost_uv, is_flat_coeffs, is_flat_source_16, tdisto_4x4, tdisto_8x8,
    tdisto_16x16, trellis_quantize_block,
};

use crate::encoder::psy;

use super::{MacroblockInfo, sse_8x8_chroma, sse_16x16_luma};

// =============================================================================
// Arcane (SIMD-hoisted) inner functions for I4 mode selection
//
// These free functions run entirely within an #[arcane] context, allowing the
// compiler to inline all SIMD leaf functions (ftransform, quantize, sse4x4,
// get_residual_cost, etc.) and keep values in SIMD registers across calls.
// This eliminates the per-call dispatch overhead (~9.7M instructions) and
// enables cross-function optimizations.
// =============================================================================

/// Result of evaluating a single I4 block mode
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "wasm32")))]
struct I4BlockResult {
    mode_idx: usize,
    rd_score: u64,
    has_nz: bool,
    dequantized: [i32; 16],
    sse: u32,
    spectral_disto: i32,
    psy_cost: i32,
    coeff_cost: u32,
}

/// Pre-sort I4 prediction modes by prediction SSE (ascending).
/// Runs sse4x4 for all 10 modes using direct SIMD calls (no dispatch overhead).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[archmage::arcane]
fn presort_i4_modes_sse2(
    _token: archmage::X64V3Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
) -> [(u32, usize); 10] {
    let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
    for (mode_idx, entry) in mode_sse.iter_mut().enumerate() {
        let pred = preds.get(mode_idx);
        let sse = crate::common::simd_sse::sse4x4_sse2(_token, src_block, pred);
        *entry = (sse, mode_idx);
    }
    mode_sse.sort_unstable_by_key(|&(sse, _)| sse);
    mode_sse
}

/// Evaluate I4 block modes within a single arcane context.
///
/// This is the hot inner loop of pick_best_intra4, with all SIMD calls
/// going directly to `_sse2` variants (inlinable within arcane context).
///
/// Returns the best mode result, or None if no mode beats `best_block_score_limit`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[archmage::arcane]
fn evaluate_i4_modes_sse2(
    _token: archmage::X64V3Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
    mode_sse_order: &[(u32, usize)],
    max_modes: usize,
    mode_costs: &[u16; 10],
    nz_top: bool,
    nz_left: bool,
    // RD parameters
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    lambda_i4: u32,
    tlambda: u32,
    trellis_lambda_i4: Option<u32>,
    level_costs: &crate::encoder::cost::LevelCosts,
    probs: &TokenProbTables,
    psy_config: &crate::encoder::psy::PsyConfig,
    luma_csf: &[u16; 16],
) -> Option<I4BlockResult> {
    use crate::encoder::residual_cost::Residual;

    let mut best: Option<I4BlockResult> = None;
    let mut best_block_score = u64::MAX;

    for &(_, mode_idx) in mode_sse_order[..max_modes].iter() {
        let pred = preds.get(mode_idx);

        // Fused residual + DCT using direct SIMD call
        let mut residual = crate::common::transform_simd_intrinsics::ftransform_from_u8_4x4_sse2(
            _token, src_block, pred,
        );

        // Quantize - use trellis if enabled, otherwise fused quantize+dequantize
        let mut quantized_zigzag = [0i32; 16];
        let mut quantized_natural = [0i32; 16];
        let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
            // Trellis quantization (scalar, called from arcane context is fine)
            let ctx0 = usize::from(nz_top) + usize::from(nz_left);
            const CTYPE_I4_AC: usize = 3;
            let nz = trellis_quantize_block(
                &mut residual,
                &mut quantized_zigzag,
                y1_matrix,
                lambda,
                0,
                level_costs,
                CTYPE_I4_AC,
                ctx0,
                psy_config,
            );
            // Convert zigzag to natural order
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_natural[j] = quantized_zigzag[n];
            }
            // Dequantize + IDCT for trellis path
            let mut dq = quantized_natural;
            for (idx, val) in dq.iter_mut().enumerate() {
                *val = y1_matrix.dequantize(*val, idx);
            }
            transform::idct4x4(&mut dq);
            (nz, dq)
        } else {
            // Fused quantize+dequantize using direct SIMD call
            let mut dequant_natural = [0i32; 16];
            let nz = crate::encoder::quantize::quantize_dequantize_block_sse2(
                _token,
                &residual,
                y1_matrix,
                true,
                &mut quantized_natural,
                &mut dequant_natural,
            );
            // Convert natural to zigzag for cost estimation
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_zigzag[n] = quantized_natural[j];
            }
            // IDCT the dequantized values
            transform::idct4x4(&mut dequant_natural);
            (nz, dequant_natural)
        };

        // Get coefficient cost using direct SIMD call
        let ctx0 = (nz_top as usize) + (nz_left as usize);
        let res = Residual::new(&quantized_zigzag, 3, 0); // CTYPE_I4_AC=3, first=0
        let coeff_cost = crate::encoder::residual_cost::get_residual_cost_sse2(
            _token,
            ctx0,
            &res,
            level_costs,
            probs,
        );

        // Compute SSE using direct SIMD call
        let sse = crate::common::simd_sse::sse4x4_with_residual_sse2(
            _token,
            src_block,
            pred,
            &dequantized,
        );

        // Early exit: if base RD score already exceeds best, skip extras
        let mode_cost = mode_costs[mode_idx];
        let base_rd_score =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, coeff_cost, lambda_i4) as u64;
        if base_rd_score >= best_block_score {
            continue;
        }

        // Flatness penalty for non-DC modes
        let flatness_penalty = if mode_idx > 0 {
            let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
            if crate::encoder::cost::distortion::is_flat_coeffs_sse2(
                _token,
                &levels_i16,
                1,
                FLATNESS_LIMIT_I4,
            ) {
                FLATNESS_PENALTY
            } else {
                0
            }
        } else {
            0
        };

        // Spectral distortion + psy-rd
        let (spectral_disto, psy_cost) = if tlambda > 0 || psy_config.psy_rd_strength > 0 {
            // Build reconstructed 4x4 block
            let mut rec_block = [0u8; 16];
            for k in 0..16 {
                rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
            }

            let td = if tlambda > 0 {
                let td_raw = crate::common::simd_sse::tdisto_4x4_fused_sse2(
                    _token, src_block, &rec_block, 4, luma_csf,
                );
                (tlambda as i32 * td_raw + 128) >> 8
            } else {
                0
            };

            let psy = if psy_config.psy_rd_strength > 0 {
                let src_satd = psy::satd_4x4(src_block, 4);
                let rec_satd = psy::satd_4x4(&rec_block, 4);
                psy::psy_rd_cost(src_satd, rec_satd, psy_config.psy_rd_strength)
            } else {
                0
            };

            (td, psy)
        } else {
            (0, 0)
        };

        // Final RD score
        let total_rate_cost = coeff_cost + flatness_penalty as u32;
        let rd_score = crate::encoder::cost::rd_score_full(
            sse,
            spectral_disto + psy_cost,
            mode_cost,
            total_rate_cost,
            lambda_i4,
        ) as u64;

        if rd_score < best_block_score {
            best_block_score = rd_score;
            best = Some(I4BlockResult {
                mode_idx,
                rd_score,
                has_nz,
                dequantized,
                sse,
                spectral_disto,
                psy_cost,
                coeff_cost,
            });
        }
    }

    best
}

/// Pre-sort I4 prediction modes by prediction SSE (ascending) for wasm SIMD128.
/// Runs sse4x4 for all 10 modes using direct SIMD calls (no dispatch overhead).
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[archmage::arcane]
fn presort_i4_modes_wasm(
    _token: archmage::Wasm128Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
) -> [(u32, usize); 10] {
    let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
    for (mode_idx, entry) in mode_sse.iter_mut().enumerate() {
        let pred = preds.get(mode_idx);
        let sse = crate::common::simd_wasm::sse4x4_wasm(_token, src_block, pred);
        *entry = (sse, mode_idx);
    }
    mode_sse.sort_unstable_by_key(|&(sse, _)| sse);
    mode_sse
}

/// Evaluate I4 block modes within a single arcane context for wasm SIMD128.
///
/// This is the hot inner loop of pick_best_intra4, with all SIMD calls
/// going directly to `_wasm` #[rite] variants (inlinable within arcane context).
///
/// Returns the best mode result, or None if no mode beats `best_block_score_limit`.
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn evaluate_i4_modes_wasm(
    _token: archmage::Wasm128Token,
    src_block: &[u8; 16],
    preds: &I4Predictions,
    mode_sse_order: &[(u32, usize)],
    max_modes: usize,
    mode_costs: &[u16; 10],
    nz_top: bool,
    nz_left: bool,
    // RD parameters
    y1_matrix: &crate::encoder::quantize::VP8Matrix,
    lambda_i4: u32,
    tlambda: u32,
    trellis_lambda_i4: Option<u32>,
    level_costs: &crate::encoder::cost::LevelCosts,
    probs: &TokenProbTables,
    psy_config: &crate::encoder::psy::PsyConfig,
    luma_csf: &[u16; 16],
) -> Option<I4BlockResult> {
    use crate::encoder::residual_cost::Residual;

    let mut best: Option<I4BlockResult> = None;
    let mut best_block_score = u64::MAX;

    for &(_, mode_idx) in mode_sse_order[..max_modes].iter() {
        let pred = preds.get(mode_idx);

        // Fused residual + DCT using direct SIMD call
        let mut residual = crate::common::transform_wasm::ftransform_from_u8_4x4_wasm_impl(
            _token, src_block, pred,
        );

        // Quantize - use trellis if enabled, otherwise fused quantize+dequantize
        let mut quantized_zigzag = [0i32; 16];
        let mut quantized_natural = [0i32; 16];
        let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
            // Trellis quantization (scalar, called from arcane context is fine)
            let ctx0 = usize::from(nz_top) + usize::from(nz_left);
            const CTYPE_I4_AC: usize = 3;
            let nz = trellis_quantize_block(
                &mut residual,
                &mut quantized_zigzag,
                y1_matrix,
                lambda,
                0,
                level_costs,
                CTYPE_I4_AC,
                ctx0,
                psy_config,
            );
            // Convert zigzag to natural order
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_natural[j] = quantized_zigzag[n];
            }
            // Dequantize + IDCT for trellis path
            let mut dq = quantized_natural;
            for (idx, val) in dq.iter_mut().enumerate() {
                *val = y1_matrix.dequantize(*val, idx);
            }
            transform::idct4x4(&mut dq);
            (nz, dq)
        } else {
            // Fused quantize+dequantize using direct SIMD call
            let mut dequant_natural = [0i32; 16];
            let nz = crate::common::simd_wasm::quantize_dequantize_block_wasm(
                _token,
                &residual,
                y1_matrix,
                true,
                &mut quantized_natural,
                &mut dequant_natural,
            );
            // Convert natural to zigzag for cost estimation
            for n in 0..16 {
                let j = ZIGZAG[n] as usize;
                quantized_zigzag[n] = quantized_natural[j];
            }
            // IDCT the dequantized values
            transform::idct4x4(&mut dequant_natural);
            (nz, dequant_natural)
        };

        // Get coefficient cost using direct SIMD call
        let ctx0 = (nz_top as usize) + (nz_left as usize);
        let res = Residual::new(&quantized_zigzag, 3, 0); // CTYPE_I4_AC=3, first=0
        let coeff_cost = crate::encoder::residual_cost::get_residual_cost_wasm(
            _token,
            ctx0,
            &res,
            level_costs,
            probs,
        );

        // Compute SSE using direct SIMD call
        let sse = crate::common::simd_wasm::sse4x4_with_residual_wasm(
            _token,
            src_block,
            pred,
            &dequantized,
        );

        // Early exit: if base RD score already exceeds best, skip extras
        let mode_cost = mode_costs[mode_idx];
        let base_rd_score =
            crate::encoder::cost::rd_score_full(sse, 0, mode_cost, coeff_cost, lambda_i4) as u64;
        if base_rd_score >= best_block_score {
            continue;
        }

        // Flatness penalty for non-DC modes
        let flatness_penalty = if mode_idx > 0 {
            let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
            if crate::common::simd_wasm::is_flat_coeffs_wasm(
                _token,
                &levels_i16,
                1,
                FLATNESS_LIMIT_I4,
            ) {
                FLATNESS_PENALTY
            } else {
                0
            }
        } else {
            0
        };

        // Spectral distortion + psy-rd
        let (spectral_disto, psy_cost) = if tlambda > 0 || psy_config.psy_rd_strength > 0 {
            // Build reconstructed 4x4 block
            let mut rec_block = [0u8; 16];
            for k in 0..16 {
                rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
            }

            let td = if tlambda > 0 {
                let td_raw = crate::common::simd_wasm::tdisto_4x4_fused_wasm(
                    _token, src_block, &rec_block, 4, luma_csf,
                );
                (tlambda as i32 * td_raw + 128) >> 8
            } else {
                0
            };

            let psy = if psy_config.psy_rd_strength > 0 {
                let src_satd = psy::satd_4x4(src_block, 4);
                let rec_satd = psy::satd_4x4(&rec_block, 4);
                psy::psy_rd_cost(src_satd, rec_satd, psy_config.psy_rd_strength)
            } else {
                0
            };

            (td, psy)
        } else {
            (0, 0)
        };

        // Final RD score
        let total_rate_cost = coeff_cost + flatness_penalty as u32;
        let rd_score = crate::encoder::cost::rd_score_full(
            sse,
            spectral_disto + psy_cost,
            mode_cost,
            total_rate_cost,
            lambda_i4,
        ) as u64;

        if rd_score < best_block_score {
            best_block_score = rd_score;
            best = Some(I4BlockResult {
                mode_idx,
                rd_score,
                has_nz,
                dequantized,
                sse,
                spectral_disto,
                psy_cost,
                coeff_cost,
            });
        }
    }

    best
}

impl<'a> super::Vp8Encoder<'a> {
    /// Select the best 16x16 luma prediction mode using full RD (rate-distortion) cost.
    ///
    /// This implements libwebp's full RD path for PickBestIntra16:
    /// 1. For each mode: generate prediction, forward transform, quantize
    /// 2. Dequantize and inverse transform to get reconstructed block
    /// 3. Compute SSE between reconstructed and source (NOT prediction vs source!)
    /// 4. Compute spectral distortion (TDisto) if tlambda > 0
    /// 5. Include coefficient cost in rate term
    /// 6. Apply flat source penalty if applicable
    ///
    /// RD formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
    /// Where: R = coeff cost, H = mode cost, D = SSE, SD = spectral distortion
    ///
    /// Returns (best_mode, distortion_score) for comparison against Intra4x4.
    fn pick_best_intra16(&self, mbx: usize, mby: usize) -> (LumaMode, u64) {
        // Check for debug mode
        #[cfg(feature = "mode_debug")]
        let debug_i16 = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let _debug_i16 = false;
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // Fast path for method 0-1: DC mode only with SSE-based scoring
        // This avoids the full RD evaluation loop for maximum speed
        if self.method <= 1 {
            return self.pick_intra16_fast_dc(mbx, mby);
        }

        // The 4 modes to try for 16x16 luma prediction (order matches FIXED_COSTS_I16)
        const MODES: [LumaMode; 4] = [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM];

        let segment = self.get_segment_for_mb(mbx, mby);
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        let lambda = segment.lambda_i16;
        let tlambda = segment.tlambda;

        // Use updated probabilities if available (for consistent mode selection)
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Check if source block is flat (for flat source penalty)
        let src_base = mby * 16 * src_width + mbx * 16;
        let is_flat = is_flat_source_16(&self.frame.ybuf[src_base..], src_width);

        // Pre-extract source block for TDisto/psy-rd (avoid repeated extraction per mode)
        let need_spectral = tlambda > 0 || segment.psy_config.psy_rd_strength > 0;
        let src_block = if need_spectral {
            let mut block = [0u8; 256];
            for y in 0..16 {
                let src_row = (mby * 16 + y) * src_width + mbx * 16;
                block[y * 16..(y + 1) * 16]
                    .copy_from_slice(&self.frame.ybuf[src_row..src_row + 16]);
            }
            Some(block)
        } else {
            None
        };

        let mut best_mode = LumaMode::DC;
        let mut best_rd_score = i64::MAX;
        // Store best mode's cost components for final score recalculation with lambda_mode
        let mut best_coeff_cost = 0u32;
        let mut best_mode_cost = 0u16;
        let mut best_sse = 0u32;
        let mut best_spectral_disto = 0i32;
        let mut best_psy_cost = 0i32;

        // Pre-allocate scratch buffers outside mode loop to avoid redundant zero-init.
        // All elements are written before read in each iteration.
        let mut luma_blocks = [0i32; 256];
        let mut y1_quant = [[0i32; 16]; 16];
        let mut recon_dequant_block = [0i32; 16];
        let mut rec_block = [0u8; 256];
        let mut all_levels = [0i16; 256];

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // Skip V mode if no top row available (first row of macroblocks)
            // Note: V prediction requires real top pixels, border value 127 gives poor results
            if mode == LumaMode::V && mby == 0 {
                continue;
            }
            // Skip H mode if no left column available (first column of macroblocks)
            // Note: H prediction requires real left pixels, border value 129 gives poor results
            if mode == LumaMode::H && mbx == 0 {
                continue;
            }
            // TM mode requires both top and left pixels, but can use border values (127/129)
            // At edges, TM degenerates to H (if mby=0) or V (if mbx=0), so skip only at corner
            // where both borders are synthetic
            if mode == LumaMode::TM && mbx == 0 && mby == 0 {
                continue;
            }

            // Generate prediction for this mode
            let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);

            // === Full reconstruction for RD evaluation ===
            // 1. Compute residuals and forward DCT (luma_blocks hoisted outside mode loop)
            self.fill_luma_blocks_from_predicted_16x16(&pred, mbx, mby, &mut luma_blocks);

            // 2. Extract DC coefficients and do WHT
            let mut dc_coeffs = [0i32; 16];
            for (i, dc) in dc_coeffs.iter_mut().enumerate() {
                *dc = luma_blocks[i * 16];
            }
            let mut y2_coeffs = dc_coeffs;
            transform::wht4x4(&mut y2_coeffs);

            // 3. Quantize Y2 (DC) coefficients using SIMD
            let mut y2_quant = y2_coeffs;
            crate::encoder::quantize::quantize_block_simd(&mut y2_quant, y2_matrix, true);

            // 4. Quantize Y1 (AC) coefficients using SIMD
            // Extract each 4x4 block from luma_blocks and quantize AC coefficients
            // (y1_quant hoisted outside mode loop to avoid redundant zero-init)
            #[allow(clippy::needless_range_loop)]
            for block_idx in 0..16 {
                // Copy block from luma_blocks (DC will be zeroed by quantize_ac_only)
                let block_start = block_idx * 16;
                let mut block: [i32; 16] = luma_blocks[block_start..block_start + 16]
                    .try_into()
                    .unwrap();
                block[0] = 0; // DC is handled by Y2
                crate::encoder::quantize::quantize_ac_only_simd(&mut block, y1_matrix, true);
                y1_quant[block_idx] = block;
            }

            // 5. Compute coefficient cost using probability-dependent tables
            // This matches libwebp's VP8GetCostLuma16 which uses proper token probabilities
            let coeff_cost = get_cost_luma16(&y2_quant, &y1_quant, &self.level_costs, probs);

            // 6. Dequantize Y2 and do inverse WHT using SIMD
            let mut y2_dequant = y2_quant;
            y2_matrix.dequantize_block(&mut y2_dequant);
            transform::iwht4x4(&mut y2_dequant);

            // 7. Dequantize Y1, add DC from Y2, and do fused inverse DCT + add residue
            let mut reconstructed = pred;
            for block_idx in 0..16 {
                let bx = block_idx % 4;
                let by = block_idx / 4;

                // AC from Y1 (recon_dequant_block hoisted outside mode loop)
                for i in 1..16 {
                    recon_dequant_block[i] = y1_matrix.dequantize(y1_quant[block_idx][i], i);
                }
                // DC from Y2
                recon_dequant_block[0] = y2_dequant[block_idx];

                // Fused IDCT + add residue to prediction (reads pred, adds IDCT, clamps, stores)
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = recon_dequant_block[1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut recon_dequant_block,
                    &mut reconstructed,
                    y0,
                    x0,
                    LUMA_STRIDE,
                    dc_only,
                );
            }

            // 8. Compute SSE between source and reconstructed (NOT prediction!)
            let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &reconstructed);

            // 9. Compute spectral distortion (TDisto) + psy-rd if enabled
            let (spectral_disto, psy_cost) = if let Some(ref src_block) = src_block {
                // Extract reconstructed block only (source already cached)
                // (rec_block hoisted outside mode loop to avoid redundant zero-init)
                for y in 0..16 {
                    let rec_row = (y + 1) * LUMA_STRIDE + 1;
                    rec_block[y * 16..(y + 1) * 16]
                        .copy_from_slice(&reconstructed[rec_row..rec_row + 16]);
                }

                let td = if tlambda > 0 {
                    let td_raw =
                        tdisto_16x16(src_block, &rec_block, 16, &segment.psy_config.luma_csf);
                    (tlambda as i32 * td_raw + 128) >> 8
                } else {
                    0
                };

                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_16x16(src_block, 16);
                    let rec_satd = psy::satd_16x16(&rec_block, 16);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };

                (td, psy)
            } else {
                (0, 0)
            };

            // 10. Apply flat source penalty if applicable
            let (d_final, sd_final) = if is_flat {
                // Check if coefficients are also flat
                // (all_levels hoisted outside mode loop to avoid redundant zero-init)
                for block_idx in 0..16 {
                    for i in 1..16 {
                        all_levels[block_idx * 16 + i] = y1_quant[block_idx][i] as i16;
                    }
                }
                if is_flat_coeffs(&all_levels, 16, FLATNESS_LIMIT_I16) {
                    // Double distortion to penalize I16 for flat sources
                    (sse * 2, spectral_disto * 2)
                } else {
                    (sse, spectral_disto)
                }
            } else {
                (sse, spectral_disto)
            };

            // 11. Compute full RD score
            // score = (R + H) * lambda + RD_DISTO_MULT * (D + SD + PSY)
            let mode_cost = FIXED_COSTS_I16[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost)) * i64::from(lambda);
            let distortion = i64::from(RD_DISTO_MULT)
                * (i64::from(d_final) + i64::from(sd_final) + i64::from(psy_cost));
            let rd_score = rate + distortion;

            #[cfg(feature = "mode_debug")]
            if debug_i16 {
                eprintln!(
                    "  I16 {:?}: H={}, R={}, D={}, SD={}, rate={}, disto={}, score={}",
                    mode, mode_cost, coeff_cost, d_final, sd_final, rate, distortion, rd_score
                );
            }

            if rd_score < best_rd_score {
                best_rd_score = rd_score;
                best_mode = mode;
                // Store components for final score recalculation
                best_coeff_cost = coeff_cost;
                best_mode_cost = mode_cost;
                best_sse = d_final;
                best_spectral_disto = sd_final;
                best_psy_cost = psy_cost;
            }
        }

        // Recalculate final score using lambda_mode for I4 vs I16 comparison
        // This matches libwebp's: SetRDScore(dqm->lambda_mode, rd);
        let lambda_mode = segment.lambda_mode;
        let final_rate =
            (i64::from(best_mode_cost) + i64::from(best_coeff_cost)) * i64::from(lambda_mode);
        let final_distortion = i64::from(RD_DISTO_MULT)
            * (i64::from(best_sse) + i64::from(best_spectral_disto) + i64::from(best_psy_cost));
        let final_score = final_rate + final_distortion;

        #[cfg(feature = "mode_debug")]
        if debug_i16 {
            eprintln!(
                "  I16 FINAL: mode={:?}, H={}, R={}, D={}, SD={}, lambda_mode={}, tlambda={}, rate={}, disto={}, score={}",
                best_mode,
                best_mode_cost,
                best_coeff_cost,
                best_sse,
                best_spectral_disto,
                lambda_mode,
                tlambda,
                final_rate,
                final_distortion,
                final_score
            );
        }

        // Convert to u64 for interface compatibility (score should be positive)
        (best_mode, final_score.max(0) as u64)
    }

    /// Fast DC-only mode selection for method 0.
    ///
    /// Uses simplified scoring (SSE + fixed mode cost) without full reconstruction.
    /// This is much faster than the full RD path but may not find the optimal mode.
    fn pick_intra16_fast_dc(&self, mbx: usize, mby: usize) -> (LumaMode, u64) {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;
        let segment = self.get_segment_for_mb(mbx, mby);
        let lambda = segment.lambda_i16;

        // For method 0, we use DC mode with simple SSE scoring
        let pred = self.get_predicted_luma_block_16x16(LumaMode::DC, mbx, mby);
        let sse = sse_16x16_luma(&self.frame.ybuf, src_width, mbx, mby, &pred);

        // Simple score: SSE + lambda * mode_cost
        // DC mode has the lowest fixed cost (FIXED_COSTS_I16[0])
        let mode_cost = FIXED_COSTS_I16[0] as u32;
        let score = u64::from(sse) + u64::from(lambda) * u64::from(mode_cost);

        (LumaMode::DC, score)
    }

    /// Estimate coefficient cost for a 16x16 luma macroblock (I16 mode).
    ///
    /// Quantizes coefficients and estimates their encoding cost without
    /// permanently modifying state.
    fn estimate_luma16_coeff_cost(&self, luma_blocks: &[i32; 256], segment: &Segment) -> u32 {
        let mut total_cost = 0u32;

        // Extract DC coefficients and estimate Y2 (DC transform) cost
        let mut dc_coeffs = [0i32; 16];
        for (i, dc) in dc_coeffs.iter_mut().enumerate() {
            *dc = luma_blocks[i * 16];
        }

        // WHT transform on DC coefficients
        let mut y2_coeffs = dc_coeffs;
        transform::wht4x4(&mut y2_coeffs);

        // Quantize Y2 coefficients and estimate cost
        for (idx, coeff) in y2_coeffs.iter_mut().enumerate() {
            let quant = if idx > 0 { segment.y2ac } else { segment.y2dc };
            *coeff /= i32::from(quant);
        }
        total_cost += estimate_dc16_cost(&y2_coeffs);

        // Estimate AC coefficient cost for each 4x4 block (skip DC at index 0)
        for block_idx in 0..16 {
            let block_start = block_idx * 16;
            let mut block = [0i32; 16];

            // Copy and quantize AC coefficients (DC is handled separately in I16 mode)
            for (i, coeff) in block.iter_mut().enumerate() {
                if i == 0 {
                    *coeff = 0; // DC is in Y2 block
                } else {
                    *coeff = luma_blocks[block_start + i] / i32::from(segment.yac);
                }
            }

            // Estimate cost (starting from position 1, DC is separate)
            total_cost += estimate_residual_cost(&block, 1);
        }

        total_cost
    }

    /// Apply a 4x4 intra prediction mode to the working buffer
    #[allow(dead_code)] // Kept for future Intra4 mode selection with coefficient-level costs
    fn apply_intra4_prediction(
        ws: &mut [u8; LUMA_BLOCK_SIZE],
        mode: IntraMode,
        x0: usize,
        y0: usize,
    ) {
        let stride = LUMA_STRIDE;
        match mode {
            IntraMode::TM => predict_tmpred(ws, 4, x0, y0, stride),
            IntraMode::VE => predict_bvepred(ws, x0, y0, stride),
            IntraMode::HE => predict_bhepred(ws, x0, y0, stride),
            IntraMode::DC => predict_bdcpred(ws, x0, y0, stride),
            IntraMode::LD => predict_bldpred(ws, x0, y0, stride),
            IntraMode::RD => predict_brdpred(ws, x0, y0, stride),
            IntraMode::VR => predict_bvrpred(ws, x0, y0, stride),
            IntraMode::VL => predict_bvlpred(ws, x0, y0, stride),
            IntraMode::HD => predict_bhdpred(ws, x0, y0, stride),
            IntraMode::HU => predict_bhupred(ws, x0, y0, stride),
        }
    }

    /// Compute SSE for a 4x4 subblock between source image and prediction buffer
    #[allow(dead_code)] // Kept for future Intra4 mode selection with coefficient-level costs
    fn sse_4x4_subblock(
        &self,
        pred: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
        sbx: usize,
        sby: usize,
    ) -> u32 {
        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        let mut sse = 0u32;
        let pred_y0 = sby * 4 + 1;
        let pred_x0 = sbx * 4 + 1;
        let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;

        for y in 0..4 {
            let pred_row = (pred_y0 + y) * LUMA_STRIDE + pred_x0;
            let src_row = src_base + y * src_width;
            for x in 0..4 {
                let diff = i32::from(self.frame.ybuf[src_row + x]) - i32::from(pred[pred_row + x]);
                sse += (diff * diff) as u32;
            }
        }
        sse
    }

    /// Select the best Intra4 modes for all 16 subblocks using accurate coefficient cost estimation.
    ///
    /// Returns `Some((modes, rd_score))` if Intra4 is better than `i16_score`,
    /// or `None` if Intra16 should be used (early-exit optimization).
    ///
    /// The comparison includes an i4_penalty (1000 * q²) to account for the
    /// typically higher bit cost of Intra4 mode signaling.
    ///
    /// Uses proper probability-based coefficient cost estimation ported from
    /// libwebp's VP8GetCostLuma4 with remapped_costs tables.
    fn pick_best_intra4(
        &self,
        mbx: usize,
        mby: usize,
        i16_score: u64,
    ) -> Option<([IntraMode; 16], u64)> {
        // Check for debug mode
        #[cfg(feature = "mode_debug")]
        let debug_i4 = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let _debug_i4 = false;

        let mbw = usize::from(self.macroblock_width);
        let src_width = mbw * 16;

        // All 10 intra4 modes (used by SIMD path on x86_64 and wasm32)
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "wasm32")))]
        const MODES: [IntraMode; 10] = [
            IntraMode::DC,
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::LD,
            IntraMode::RD,
            IntraMode::VR,
            IntraMode::VL,
            IntraMode::HD,
            IntraMode::HU,
        ];

        let mut best_modes = [IntraMode::DC; 16];
        let mut best_mode_indices = [0usize; 16]; // Track indices for context lookup

        // Create working buffer with border
        let mut y_with_border =
            create_border_luma(mbx, mby, mbw, &self.top_border_y, &self.left_border_y);

        let segment = self.get_segment_for_mb(mbx, mby);

        // Get quantizer-dependent lambdas for I4 mode RD scoring
        // lambda_i4 is used for selecting the best mode within each block
        // lambda_mode is used for accumulation and comparison against I16 score
        // tlambda is used for spectral distortion (TDisto) weighting
        // (This matches libwebp's SetRDScore flow in PickBestIntra4)
        let lambda_i4 = segment.lambda_i4;
        let lambda_mode = segment.lambda_mode;
        let tlambda = segment.tlambda;

        // Initialize I4 running score with an I4 penalty
        // libwebp uses i4_penalty = 1000 * q² with fixed lambda_d_i4 = 11
        // Our scoring system uses q-dependent lambdas, so we need to scale differently
        //
        // Approach: Use a penalty proportional to lambda_mode (which is q²/128)
        // This gives: penalty ≈ SCALE * q² where SCALE is tuned empirically
        //
        // For parity with our scoring, we use: 2200 * lambda_mode
        // At q=30: lambda_mode=7, penalty = 15,400 (vs libwebp's 900,000)
        // This ratio (58x smaller) matches the ratio of our lambdas to libwebp's
        // (lambda_i4 ≈ 21 vs lambda_d_i4 = 11 is 2x, and we include coeff costs)
        let i4_penalty = 3000u64 * u64::from(lambda_mode);
        let mut running_score = i4_penalty;

        #[cfg(feature = "mode_debug")]
        if debug_i4 {
            eprintln!(
                "  I4 i4_penalty={}, running_score={}",
                i4_penalty, running_score
            );
        }

        // Track total mode cost for header bit limiting
        let mut total_mode_cost = 0u32;
        // Maximum header bits for I4 modes (from libwebp)
        let max_header_bits: u32 = 256 * 16 * 16 / 4;

        // Track non-zero context for accurate coefficient cost estimation
        // top_nz[x] = whether block above has non-zero coefficients
        // left_nz[y] = whether block to left has non-zero coefficients
        // Initialize from cross-macroblock context (top_complexity/left_complexity)
        // so edge blocks use the correct context from neighboring macroblocks
        let mut top_nz = [
            self.top_complexity[mbx].y[0] != 0,
            self.top_complexity[mbx].y[1] != 0,
            self.top_complexity[mbx].y[2] != 0,
            self.top_complexity[mbx].y[3] != 0,
        ];
        let mut left_nz = [
            self.left_complexity.y[0] != 0,
            self.left_complexity.y[1] != 0,
            self.left_complexity.y[2] != 0,
            self.left_complexity.y[3] != 0,
        ];

        // Get probability tables for coefficient cost estimation
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Process each subblock in raster order
        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                // Get mode context from neighboring blocks
                // For edge blocks, use cross-macroblock context from previous MB's I4 modes
                let top_ctx = if sby == 0 {
                    // Top edge: use mode from macroblock above (stored in top_b_pred)
                    // Index: mbx * 4 + sbx gives the correct column's top context
                    self.top_b_pred[mbx * 4 + sbx] as usize
                } else {
                    best_mode_indices[(sby - 1) * 4 + sbx]
                };
                let left_ctx = if sbx == 0 {
                    // Left edge: use mode from macroblock to the left (stored in left_b_pred)
                    // Index: sby gives the correct row's left context
                    self.left_b_pred[sby] as usize
                } else {
                    best_mode_indices[sby * 4 + (sbx - 1)]
                };

                // Get non-zero context from neighboring blocks
                // For edge blocks (sby==0 or sbx==0), the cross-macroblock context
                // is already in top_nz/left_nz from initialization above
                let nz_top = top_nz[sbx];
                let nz_left = left_nz[sby];

                // Precompute mode costs for all 10 modes (context is constant for this block)
                // This avoids repeated 3D table lookup inside the tight mode loop
                let mode_costs: [u16; 10] = core::array::from_fn(|mode_idx| {
                    crate::encoder::tables::VP8_FIXED_COSTS_I4[top_ctx][left_ctx][mode_idx]
                });

                let mut best_mode = IntraMode::DC;
                let mut best_mode_idx = 0usize;
                let mut best_block_score = u64::MAX;
                let mut best_has_nz = false;
                // Save best dequantized+IDCT result to avoid recomputing in post-loop
                let mut best_dequantized = [0i32; 16];
                // Track best block's SSE, spectral distortion, psy cost, and coeff cost for recalculating with lambda_mode
                let mut best_sse = 0u32;
                let mut best_spectral_disto = 0i32;
                let mut best_psy_cost = 0i32;
                let mut best_coeff_cost = 0u32;

                // Pre-compute all 10 I4 prediction modes at once
                let preds = I4Predictions::compute(&y_with_border, x0, y0, LUMA_STRIDE);

                // Compute source block once (row-wise copy for better cache/vectorization)
                let src_base = (mby * 16 + sby * 4) * src_width + mbx * 16 + sbx * 4;
                let mut src_block = [0u8; 16];
                for y in 0..4 {
                    let src_row = src_base + y * src_width;
                    src_block[y * 4..y * 4 + 4]
                        .copy_from_slice(&self.frame.ybuf[src_row..src_row + 4]);
                }

                let y1_matrix = segment.y1_matrix.as_ref().unwrap();

                // Number of modes to try depends on method:
                // - method 0-2: 3 modes (fast, RD_OPT_NONE equivalent)
                // - method 3+: 10 modes (full search, matches libwebp RD_OPT_BASIC+)
                let max_modes_to_try = match self.method {
                    0..=2 => 3,
                    _ => 10, // method 3+: try all modes (matches libwebp RD_OPT_BASIC+)
                };

                // Get trellis lambda if enabled for mode selection (method >= 6)
                let trellis_lambda_i4 = if self.do_trellis_i4_mode {
                    Some(segment.lambda_trellis_i4)
                } else {
                    None
                };

                // === SIMD-hoisted path: single arcane context for all mode evaluation ===
                // This eliminates per-call dispatch overhead by running all SIMD operations
                // (ftransform, quantize, dequantize, sse, tdisto, is_flat, residual_cost)
                // within a single #[target_feature] context where they can be inlined.
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                let simd_result = {
                    use archmage::SimdToken;
                    archmage::X64V3Token::summon().and_then(|token| {
                        // Pre-sort modes by prediction SSE using direct SIMD calls
                        let mode_sse = presort_i4_modes_sse2(token, &src_block, &preds);

                        // Evaluate all candidate modes in a single arcane context
                        evaluate_i4_modes_sse2(
                            token,
                            &src_block,
                            &preds,
                            &mode_sse,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            &self.level_costs,
                            probs,
                            &segment.psy_config,
                            &segment.psy_config.luma_csf,
                        )
                    })
                };

                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                if let Some(result) = simd_result {
                    // Use the arcane result
                    best_mode = MODES[result.mode_idx];
                    best_mode_idx = result.mode_idx;
                    let _ = result.rd_score; // best_block_score not needed after eval
                    best_has_nz = result.has_nz;
                    best_dequantized = result.dequantized;
                    best_sse = result.sse;
                    best_spectral_disto = result.spectral_disto;
                    best_psy_cost = result.psy_cost;
                    best_coeff_cost = result.coeff_cost;
                } else {
                    // Scalar fallback (no SIMD available at runtime or not x86_64)
                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                    {
                        self.evaluate_i4_modes_scalar(
                            &src_block,
                            &preds,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            probs,
                            segment,
                            &mut best_mode,
                            &mut best_mode_idx,
                            &mut best_block_score,
                            &mut best_has_nz,
                            &mut best_dequantized,
                            &mut best_sse,
                            &mut best_spectral_disto,
                            &mut best_psy_cost,
                            &mut best_coeff_cost,
                        );
                    }
                }

                // === WASM SIMD128 path ===
                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                let simd_result = {
                    use archmage::SimdToken;
                    archmage::Wasm128Token::summon().and_then(|token| {
                        let mode_sse = presort_i4_modes_wasm(token, &src_block, &preds);

                        evaluate_i4_modes_wasm(
                            token,
                            &src_block,
                            &preds,
                            &mode_sse,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            &self.level_costs,
                            probs,
                            &segment.psy_config,
                            &segment.psy_config.luma_csf,
                        )
                    })
                };

                #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                if let Some(result) = simd_result {
                    best_mode = MODES[result.mode_idx];
                    best_mode_idx = result.mode_idx;
                    let _ = result.rd_score;
                    best_has_nz = result.has_nz;
                    best_dequantized = result.dequantized;
                    best_sse = result.sse;
                    best_spectral_disto = result.spectral_disto;
                    best_psy_cost = result.psy_cost;
                    best_coeff_cost = result.coeff_cost;
                } else {
                    #[cfg(all(feature = "simd", target_arch = "wasm32"))]
                    {
                        self.evaluate_i4_modes_scalar(
                            &src_block,
                            &preds,
                            max_modes_to_try,
                            &mode_costs,
                            nz_top,
                            nz_left,
                            y1_matrix,
                            lambda_i4,
                            tlambda,
                            trellis_lambda_i4,
                            probs,
                            segment,
                            &mut best_mode,
                            &mut best_mode_idx,
                            &mut best_block_score,
                            &mut best_has_nz,
                            &mut best_dequantized,
                            &mut best_sse,
                            &mut best_spectral_disto,
                            &mut best_psy_cost,
                            &mut best_coeff_cost,
                        );
                    }
                }

                // Fallback: no SIMD available
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86_64", target_arch = "wasm32")
                )))]
                {
                    self.evaluate_i4_modes_scalar(
                        &src_block,
                        &preds,
                        max_modes_to_try,
                        &mode_costs,
                        nz_top,
                        nz_left,
                        y1_matrix,
                        lambda_i4,
                        tlambda,
                        trellis_lambda_i4,
                        probs,
                        segment,
                        &mut best_mode,
                        &mut best_mode_idx,
                        &mut best_block_score,
                        &mut best_has_nz,
                        &mut best_dequantized,
                        &mut best_sse,
                        &mut best_spectral_disto,
                        &mut best_psy_cost,
                        &mut best_coeff_cost,
                    );
                }

                best_modes[i] = best_mode;
                best_mode_indices[i] = best_mode_idx;

                // Update non-zero context for subsequent blocks
                top_nz[sbx] = best_has_nz;
                left_nz[sby] = best_has_nz;

                let best_mode_cost = mode_costs[best_mode_idx];
                total_mode_cost += u32::from(best_mode_cost);

                // Recalculate the block score with lambda_mode for accumulation
                // (matching libwebp's SetRDScore(lambda_mode, &rd_i4) before AddScore)
                let block_score_for_comparison = crate::encoder::cost::rd_score_full(
                    best_sse,
                    best_spectral_disto + best_psy_cost,
                    best_mode_cost,
                    best_coeff_cost,
                    lambda_mode,
                ) as u64;

                #[cfg(feature = "mode_debug")]
                if debug_i4 {
                    eprintln!(
                        "  I4 blk[{:2}]: mode={:?}, H={}, R={}, D={}, SD={}, block_score={}, running={}",
                        i,
                        best_mode,
                        best_mode_cost,
                        best_coeff_cost,
                        best_sse,
                        best_spectral_disto,
                        block_score_for_comparison,
                        running_score + block_score_for_comparison
                    );
                }

                // Add this block's score to running total
                running_score += block_score_for_comparison;

                // Early-exit: if I4 already exceeds I16, bail out
                if running_score >= i16_score {
                    return None;
                }

                // Check header bit limit
                if total_mode_cost > max_header_bits {
                    return None;
                }

                // Apply the selected mode and reconstruct for next blocks
                Self::apply_intra4_prediction(&mut y_with_border, best_mode, x0, y0);

                // Add back saved dequantized+IDCT result (already computed in inner loop)
                // This eliminates a redundant dequantize + IDCT per block
                add_residue(&mut y_with_border, &best_dequantized, y0, x0, LUMA_STRIDE);
            }
        }

        // I4 wins! Return the modes and final score
        #[cfg(feature = "mode_debug")]
        if debug_i4 {
            eprintln!(
                "  I4 FINAL: score={}, i16_score={}, margin={}",
                running_score,
                i16_score,
                i16_score as i64 - running_score as i64
            );
        }

        Some((best_modes, running_score))
    }

    /// Select the best chroma (UV) prediction mode using full RD scoring.
    ///
    /// This implements libwebp's full RD path for PickBestUV:
    /// 1. For each mode: generate prediction, forward DCT, quantize
    /// 2. Dequantize and inverse DCT to get reconstructed block
    /// 3. Compute SSE between reconstructed and source (NOT prediction vs source!)
    /// 4. Include coefficient cost in rate term
    /// 5. Apply flatness penalty for non-DC modes with flat coefficients
    ///
    /// RD formula: score = (R + H) * lambda + RD_DISTO_MULT * (D + SD + PSY)
    /// UV spectral distortion and psy-rd are enabled by PsyConfig at method >= 3.
    #[allow(clippy::needless_range_loop)] // block_idx used for both indexing and coordinate computation
    fn pick_best_uv(&self, mbx: usize, mby: usize) -> ChromaMode {
        let mbw = usize::from(self.macroblock_width);
        let chroma_width = mbw * 8;

        // Order matches FIXED_COSTS_UV
        const MODES: [ChromaMode; 4] =
            [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

        let segment = self.get_segment_for_mb(mbx, mby);
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();
        let lambda = segment.lambda_uv;
        let tlambda = segment.tlambda;

        // Use updated probabilities if available (for consistent mode selection)
        let probs = self.updated_probs.as_ref().unwrap_or(&self.token_probs);

        // Pre-extract source blocks for TDisto/psy-rd (avoid repeated extraction per mode)
        let need_spectral = tlambda > 0 || segment.psy_config.psy_rd_strength > 0;
        let (src_u_block, src_v_block) = if need_spectral {
            let mut u_block = [0u8; 64];
            let mut v_block = [0u8; 64];
            for y in 0..8 {
                let src_row = (mby * 8 + y) * chroma_width + mbx * 8;
                u_block[y * 8..(y + 1) * 8].copy_from_slice(&self.frame.ubuf[src_row..src_row + 8]);
                v_block[y * 8..(y + 1) * 8].copy_from_slice(&self.frame.vbuf[src_row..src_row + 8]);
            }
            (Some(u_block), Some(v_block))
        } else {
            (None, None)
        };

        let mut best_mode = ChromaMode::DC;
        let mut best_rd_score = i64::MAX;

        // Pre-allocate scratch buffers outside mode loop to avoid redundant zero-init.
        // All elements are written before read in each iteration.
        let mut u_blocks = [0i32; 64];
        let mut v_blocks = [0i32; 64];
        let mut uv_quant = [[0i32; 16]; 8];
        let mut uv_dequant = [[0i32; 16]; 8];
        let mut rec_u_block = [0u8; 64];
        let mut rec_v_block = [0u8; 64];
        let mut all_levels_uv = [0i16; 128];

        for (mode_idx, &mode) in MODES.iter().enumerate() {
            // Skip modes that need unavailable reference pixels
            if mode == ChromaMode::V && mby == 0 {
                continue;
            }
            if mode == ChromaMode::H && mbx == 0 {
                continue;
            }
            if mode == ChromaMode::TM && (mbx == 0 || mby == 0) {
                continue;
            }

            // Generate predictions for U and V
            let pred_u = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_u,
                &self.left_border_u,
            );
            let pred_v = self.get_predicted_chroma_block(
                mode,
                mbx,
                mby,
                &self.top_border_v,
                &self.left_border_v,
            );

            // === Full reconstruction for RD evaluation ===
            // 1. Compute residuals and forward DCT (buffers hoisted outside mode loop)
            self.fill_chroma_blocks_from_predicted(
                &pred_u,
                &self.frame.ubuf,
                mbx,
                mby,
                &mut u_blocks,
            );
            self.fill_chroma_blocks_from_predicted(
                &pred_v,
                &self.frame.vbuf,
                mbx,
                mby,
                &mut v_blocks,
            );

            // 2. Fused quantize+dequantize coefficients using SIMD
            // (uv_quant/uv_dequant hoisted outside mode loop to avoid redundant zero-init)

            // Process U blocks (indices 0-3)
            for block_idx in 0..4 {
                let block_start = block_idx * 16;
                let coeffs: [i32; 16] = u_blocks[block_start..block_start + 16].try_into().unwrap();
                crate::encoder::quantize::quantize_dequantize_block_simd(
                    &coeffs,
                    uv_matrix,
                    false,
                    &mut uv_quant[block_idx],
                    &mut uv_dequant[block_idx],
                );
            }

            // Process V blocks (indices 4-7)
            for block_idx in 0..4 {
                let block_start = block_idx * 16;
                let coeffs: [i32; 16] = v_blocks[block_start..block_start + 16].try_into().unwrap();
                crate::encoder::quantize::quantize_dequantize_block_simd(
                    &coeffs,
                    uv_matrix,
                    false,
                    &mut uv_quant[4 + block_idx],
                    &mut uv_dequant[4 + block_idx],
                );
            }

            // 3. Compute coefficient cost using probability-dependent tables
            let coeff_cost = get_cost_uv(&uv_quant, &self.level_costs, probs);

            // 4. Fused inverse DCT + add residue for reconstruction
            let mut reconstructed_u = pred_u;
            let mut reconstructed_v = pred_v;

            // Reconstruct U blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = uv_dequant[block_idx][1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut uv_dequant[block_idx],
                    &mut reconstructed_u,
                    y0,
                    x0,
                    CHROMA_STRIDE,
                    dc_only,
                );
            }

            // Reconstruct V blocks
            for block_idx in 0..4 {
                let bx = block_idx % 2;
                let by = block_idx / 2;
                let x0 = 1 + bx * 4;
                let y0 = 1 + by * 4;
                let dc_only = uv_dequant[4 + block_idx][1..].iter().all(|&c| c == 0);
                crate::common::transform::idct_add_residue_inplace(
                    &mut uv_dequant[4 + block_idx],
                    &mut reconstructed_v,
                    y0,
                    x0,
                    CHROMA_STRIDE,
                    dc_only,
                );
            }

            // 4. Compute SSE between source and reconstructed (NOT prediction!)
            let sse_u = sse_8x8_chroma(&self.frame.ubuf, chroma_width, mbx, mby, &reconstructed_u);
            let sse_v = sse_8x8_chroma(&self.frame.vbuf, chroma_width, mbx, mby, &reconstructed_v);
            let sse = sse_u + sse_v;

            // 4b. Compute UV spectral distortion + psy-rd if enabled
            let uv_spectral_disto = if let (Some(src_u), Some(src_v)) = (&src_u_block, &src_v_block)
            {
                // Extract reconstructed blocks only (source already cached)
                // (rec_u_block/rec_v_block hoisted outside mode loop)
                for y in 0..8 {
                    let rec_row = (y + 1) * CHROMA_STRIDE + 1;
                    rec_u_block[y * 8..(y + 1) * 8]
                        .copy_from_slice(&reconstructed_u[rec_row..rec_row + 8]);
                    rec_v_block[y * 8..(y + 1) * 8]
                        .copy_from_slice(&reconstructed_v[rec_row..rec_row + 8]);
                }

                let td = if tlambda > 0 {
                    let td_u = tdisto_8x8(src_u, &rec_u_block, 8, &segment.psy_config.chroma_csf);
                    let td_v = tdisto_8x8(src_v, &rec_v_block, 8, &segment.psy_config.chroma_csf);
                    let td_total = td_u + td_v;
                    (tlambda as i32 * td_total + 128) >> 8
                } else {
                    0
                };

                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_8x8(src_u, 8) + psy::satd_8x8(src_v, 8);
                    let rec_satd = psy::satd_8x8(&rec_u_block, 8) + psy::satd_8x8(&rec_v_block, 8);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };

                td + psy
            } else {
                0
            };

            // 5. Apply flatness penalty for non-DC modes
            let rate_penalty = if mode_idx > 0 {
                // Check if coefficients are flat
                // (all_levels_uv hoisted outside mode loop)
                for block_idx in 0..8 {
                    for i in 0..16 {
                        all_levels_uv[block_idx * 16 + i] = uv_quant[block_idx][i] as i16;
                    }
                }
                if is_flat_coeffs(&all_levels_uv, 8, FLATNESS_LIMIT_UV) {
                    // Add flatness penalty: FLATNESS_PENALTY * num_blocks
                    FLATNESS_PENALTY * 8
                } else {
                    0
                }
            } else {
                0
            };

            // 6. Compute full RD score
            // score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)
            let mode_cost = FIXED_COSTS_UV[mode_idx];
            let rate = (i64::from(mode_cost) + i64::from(coeff_cost) + i64::from(rate_penalty))
                * i64::from(lambda);
            let distortion =
                i64::from(RD_DISTO_MULT) * (i64::from(sse) + i64::from(uv_spectral_disto));
            let rd_score = rate + distortion;

            if rd_score < best_rd_score {
                best_rd_score = rd_score;
                best_mode = mode;
            }
        }

        best_mode
    }

    pub(super) fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        // Pick the best 16x16 luma mode using RD cost selection
        let (luma_mode, i16_score) = self.pick_best_intra16(mbx, mby);

        // Debug output for specific macroblock (check MB_DEBUG env var)
        // Set MB_DEBUG=x,y to debug mode selection for that macroblock
        #[cfg(feature = "mode_debug")]
        let debug_mb = std::env::var("MB_DEBUG")
            .ok()
            .and_then(|s| {
                let parts: alloc::vec::Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    Some((
                        parts[0].parse::<usize>().ok()?,
                        parts[1].parse::<usize>().ok()?,
                    ))
                } else {
                    None
                }
            })
            .is_some_and(|(dx, dy)| dx == mbx && dy == mby);
        #[cfg(not(feature = "mode_debug"))]
        let debug_mb = false;

        #[allow(unused_variables, clippy::needless_bool)]
        let debug_i16_details = if debug_mb {
            #[cfg(feature = "mode_debug")]
            {
                let segment = self.get_segment_for_mb(mbx, mby);
                eprintln!("=== MB({},{}) Mode Selection Debug ===", mbx, mby);
                eprintln!("I16: mode={:?}, score={}", luma_mode, i16_score);
                eprintln!(
                    "  lambda_mode={}, lambda_i4={}, lambda_i16={}",
                    segment.lambda_mode, segment.lambda_i4, segment.lambda_i16
                );
            }
            true
        } else {
            false
        };

        // Method-based I4 mode selection:
        // - method 0-1: Skip I4 entirely (fastest)
        // - method 2-4: Try I4 with fast filtering
        // - method 5-6: Full I4 search
        let (luma_mode, luma_bpred) = if self.method <= 1 {
            // Fastest: I16 only, no I4 evaluation
            (luma_mode, None)
        } else {
            // For method >= 2, try I4 with early exit optimizations
            let segment = self.get_segment_for_mb(mbx, mby);
            let skip_i4_threshold = 211 * u64::from(segment.lambda_mode);

            // Skip I4 for very flat DC blocks (method 2-4)
            // For method 5-6, always try I4 for best quality
            let should_try_i4 =
                self.method >= 5 || i16_score > skip_i4_threshold || luma_mode != LumaMode::DC;

            if should_try_i4 {
                match self.pick_best_intra4(mbx, mby, i16_score) {
                    #[allow(unused_variables)]
                    Some((modes, i4_score)) => {
                        #[cfg(feature = "mode_debug")]
                        if debug_mb {
                            eprintln!("I4: score={} (beats I16)", i4_score);
                            eprintln!("  modes={:?}", modes);
                            eprintln!(
                                "  RESULT: I4 wins by {} points",
                                i16_score.saturating_sub(i4_score)
                            );
                        }
                        (LumaMode::B, Some(modes))
                    }
                    None => {
                        #[cfg(feature = "mode_debug")]
                        if debug_mb {
                            eprintln!("I4: score >= {} (I16 wins)", i16_score);
                            eprintln!("  RESULT: I16 wins");
                        }
                        (luma_mode, None)
                    }
                }
            } else {
                #[cfg(feature = "mode_debug")]
                if debug_mb {
                    eprintln!("I4: skipped (flat DC block)");
                    eprintln!("  RESULT: I16 wins (I4 not tried)");
                }
                (luma_mode, None)
            }
        };

        // Pick the best chroma mode using RD-based selection
        let chroma_mode = self.pick_best_uv(mbx, mby);

        // Get segment ID from segment map if enabled
        let segment_id = self.get_segment_id_for_mb(mbx, mby);

        MacroblockInfo {
            luma_mode,
            luma_bpred,
            chroma_mode,
            segment_id,
            coeffs_skipped: false,
        }
    }

    /// Scalar fallback for I4 mode evaluation.
    /// Used when SIMD is not available or on non-x86_64 platforms.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_i4_modes_scalar(
        &self,
        src_block: &[u8; 16],
        preds: &I4Predictions,
        max_modes_to_try: usize,
        mode_costs: &[u16; 10],
        nz_top: bool,
        nz_left: bool,
        y1_matrix: &crate::encoder::quantize::VP8Matrix,
        lambda_i4: u32,
        tlambda: u32,
        trellis_lambda_i4: Option<u32>,
        probs: &TokenProbTables,
        segment: &Segment,
        best_mode: &mut IntraMode,
        best_mode_idx: &mut usize,
        best_block_score: &mut u64,
        best_has_nz: &mut bool,
        best_dequantized: &mut [i32; 16],
        best_sse: &mut u32,
        best_spectral_disto: &mut i32,
        best_psy_cost: &mut i32,
        best_coeff_cost: &mut u32,
    ) {
        const MODES: [IntraMode; 10] = [
            IntraMode::DC,
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::LD,
            IntraMode::RD,
            IntraMode::VR,
            IntraMode::VL,
            IntraMode::HD,
            IntraMode::HU,
        ];

        // Pre-sort modes by SSE
        let mut mode_sse: [(u32, usize); 10] = [(0, 0); 10];
        for (mode_idx, _) in MODES.iter().enumerate() {
            let pred = preds.get(mode_idx);
            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
            let sse = crate::common::simd_sse::sse4x4(src_block, pred);
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            let sse = {
                use archmage::SimdToken;
                // NEON is always available on aarch64
                let token = archmage::NeonToken::summon().unwrap();
                crate::common::simd_neon::sse4x4_neon(token, src_block, pred)
            };
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            let sse = {
                use archmage::SimdToken;
                if let Some(token) = archmage::Wasm128Token::summon() {
                    crate::common::simd_wasm::sse4x4_wasm_entry(token, src_block, pred)
                } else {
                    let mut sum = 0u32;
                    for k in 0..16 {
                        let diff = i32::from(src_block[k]) - i32::from(pred[k]);
                        sum += (diff * diff) as u32;
                    }
                    sum
                }
            };
            #[cfg(not(all(
                feature = "simd",
                any(
                    target_arch = "x86_64",
                    target_arch = "x86",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )
            )))]
            let sse = {
                let mut sum = 0u32;
                for k in 0..16 {
                    let diff = i32::from(src_block[k]) - i32::from(pred[k]);
                    sum += (diff * diff) as u32;
                }
                sum
            };
            mode_sse[mode_idx] = (sse, mode_idx);
        }
        mode_sse.sort_unstable_by_key(|&(sse, _)| sse);

        for &(_, mode_idx) in mode_sse[..max_modes_to_try].iter() {
            let pred = preds.get(mode_idx);

            let mut residual = crate::common::transform::ftransform_from_u8_4x4(src_block, pred);

            let mut quantized_zigzag = [0i32; 16];
            let mut quantized_natural = [0i32; 16];
            let (has_nz, dequantized) = if let Some(lambda) = trellis_lambda_i4 {
                let ctx0 = usize::from(nz_top) + usize::from(nz_left);
                const CTYPE_I4_AC: usize = 3;
                let nz = trellis_quantize_block(
                    &mut residual,
                    &mut quantized_zigzag,
                    y1_matrix,
                    lambda,
                    0,
                    &self.level_costs,
                    CTYPE_I4_AC,
                    ctx0,
                    &segment.psy_config,
                );
                for n in 0..16 {
                    let j = ZIGZAG[n] as usize;
                    quantized_natural[j] = quantized_zigzag[n];
                }
                let mut dq = quantized_natural;
                for (idx, val) in dq.iter_mut().enumerate() {
                    *val = y1_matrix.dequantize(*val, idx);
                }
                transform::idct4x4(&mut dq);
                (nz, dq)
            } else {
                let mut dequant_natural = [0i32; 16];
                let nz = crate::encoder::quantize::quantize_dequantize_block_simd(
                    &residual,
                    y1_matrix,
                    true,
                    &mut quantized_natural,
                    &mut dequant_natural,
                );
                for n in 0..16 {
                    let j = ZIGZAG[n] as usize;
                    quantized_zigzag[n] = quantized_natural[j];
                }
                transform::idct4x4(&mut dequant_natural);
                (nz, dequant_natural)
            };

            let (coeff_cost_val, _) =
                get_cost_luma4(&quantized_zigzag, nz_top, nz_left, &self.level_costs, probs);

            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
            let sse = crate::common::simd_sse::sse4x4_with_residual(src_block, pred, &dequantized);
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            let sse = {
                use archmage::SimdToken;
                let token = archmage::NeonToken::summon().unwrap();
                crate::common::simd_neon::sse4x4_with_residual_neon(
                    token,
                    src_block,
                    pred,
                    &dequantized,
                )
            };
            #[cfg(all(feature = "simd", target_arch = "wasm32"))]
            let sse = {
                use archmage::SimdToken;
                if let Some(token) = archmage::Wasm128Token::summon() {
                    crate::common::simd_wasm::sse4x4_with_residual_wasm_entry(
                        token,
                        src_block,
                        pred,
                        &dequantized,
                    )
                } else {
                    let mut sum = 0u32;
                    for i in 0..16 {
                        let reconstructed =
                            (i32::from(pred[i]) + dequantized[i]).clamp(0, 255) as u8;
                        let diff = i32::from(src_block[i]) - i32::from(reconstructed);
                        sum += (diff * diff) as u32;
                    }
                    sum
                }
            };
            #[cfg(not(all(
                feature = "simd",
                any(
                    target_arch = "x86_64",
                    target_arch = "x86",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )
            )))]
            let sse = {
                let mut sum = 0u32;
                for i in 0..16 {
                    let reconstructed = (i32::from(pred[i]) + dequantized[i]).clamp(0, 255) as u8;
                    let diff = i32::from(src_block[i]) - i32::from(reconstructed);
                    sum += (diff * diff) as u32;
                }
                sum
            };

            let mode_cost = mode_costs[mode_idx];
            let base_rd_score =
                crate::encoder::cost::rd_score_full(sse, 0, mode_cost, coeff_cost_val, lambda_i4)
                    as u64;
            if base_rd_score >= *best_block_score {
                continue;
            }

            let flatness_penalty = if mode_idx > 0 {
                let levels_i16: [i16; 16] = core::array::from_fn(|k| quantized_zigzag[k] as i16);
                if is_flat_coeffs(&levels_i16, 1, FLATNESS_LIMIT_I4) {
                    FLATNESS_PENALTY
                } else {
                    0
                }
            } else {
                0
            };

            let (spectral_disto, psy_cost_val) = if tlambda > 0
                || segment.psy_config.psy_rd_strength > 0
            {
                let mut rec_block = [0u8; 16];
                for k in 0..16 {
                    rec_block[k] = (i32::from(pred[k]) + dequantized[k]).clamp(0, 255) as u8;
                }
                let td = if tlambda > 0 {
                    let td_raw = tdisto_4x4(src_block, &rec_block, 4, &segment.psy_config.luma_csf);
                    (tlambda as i32 * td_raw + 128) >> 8
                } else {
                    0
                };
                let psy = if segment.psy_config.psy_rd_strength > 0 {
                    let src_satd = psy::satd_4x4(src_block, 4);
                    let rec_satd = psy::satd_4x4(&rec_block, 4);
                    psy::psy_rd_cost(src_satd, rec_satd, segment.psy_config.psy_rd_strength)
                } else {
                    0
                };
                (td, psy)
            } else {
                (0, 0)
            };

            let total_rate_cost = coeff_cost_val + flatness_penalty as u32;
            let rd_score = crate::encoder::cost::rd_score_full(
                sse,
                spectral_disto + psy_cost_val,
                mode_cost,
                total_rate_cost,
                lambda_i4,
            ) as u64;

            if rd_score < *best_block_score {
                *best_block_score = rd_score;
                *best_mode = MODES[mode_idx];
                *best_mode_idx = mode_idx;
                *best_has_nz = has_nz;
                *best_dequantized = dequantized;
                *best_sse = sse;
                *best_spectral_disto = spectral_disto;
                *best_psy_cost = psy_cost_val;
                *best_coeff_cost = coeff_cost_val;
            }
        }
    }

    /// Estimate coefficient cost for a specific Intra16 mode
    #[allow(dead_code)] // Reserved for Intra4 vs Intra16 RD comparison
    fn estimate_luma16_mode_coeff_cost(
        &self,
        mode: LumaMode,
        mbx: usize,
        mby: usize,
        segment: &Segment,
    ) -> u32 {
        // Get prediction and compute residuals
        let pred = self.get_predicted_luma_block_16x16(mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&pred, mbx, mby);

        // Use the cost estimation function
        self.estimate_luma16_coeff_cost(&luma_blocks, segment)
    }
}
