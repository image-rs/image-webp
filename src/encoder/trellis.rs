//! Trellis quantization for rate-distortion optimized coefficient selection.
//!
//! Uses dynamic programming to find optimal coefficient levels that minimize
//! the RD cost: distortion + lambda * rate.
//!
//! ## Psy-trellis integration
//!
//! When `method >= 5`, the trellis uses [`super::psy::PsyConfig`] to add
//! perceptual penalties for zeroing coefficients that carry visible texture.
//! The penalty is JND-gated: coefficients below the JND threshold are considered
//! imperceptible and can be zeroed without penalty.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::cost::{LevelCostArray, LevelCosts, RD_DISTO_MULT};
use super::psy::PsyConfig;
use super::quantize::{VP8Matrix, quantdiv, quantization_bias};
use super::tables::{
    MAX_LEVEL, MAX_VARIABLE_LEVEL, VP8_LEVEL_FIXED_COSTS, VP8_WEIGHT_TRELLIS, VP8_ZIGZAG,
};

/// Maximum cost value for RD optimization
const MAX_COST: i64 = i64::MAX / 2;

/// Trellis node for dynamic programming
#[derive(Clone, Copy, Default)]
struct TrellisNode {
    prev: i8,   // best previous node (-1, 0, or 1 for delta)
    sign: bool, // sign of coefficient
    level: i16, // quantized level
}

/// Score state for trellis traversal
/// Stores score and a reference to cost table for the *next* position.
/// The cost table is determined by the context resulting from this state's level.
#[derive(Clone, Copy)]
struct TrellisScoreState<'a> {
    score: i64,                        // partial RD score
    costs: Option<&'a LevelCostArray>, // cost table for next position (based on ctx from this level)
}

impl Default for TrellisScoreState<'_> {
    fn default() -> Self {
        Self {
            score: MAX_COST,
            costs: None,
        }
    }
}

/// RD score calculation for trellis
#[inline]
pub(crate) fn rd_score_trellis(lambda: u32, rate: i64, distortion: i64) -> i64 {
    rate * lambda as i64 + (RD_DISTO_MULT as i64) * distortion
}

/// Compute level cost using stored cost table (like libwebp's VP8LevelCost).
/// cost = VP8LevelFixedCosts[level] + table[min(level, MAX_VARIABLE_LEVEL)]
/// Note: VP8LevelFixedCosts already includes the sign bit cost (256) for non-zero levels.
#[inline]
fn level_cost_with_table(costs: Option<&LevelCostArray>, level: i32) -> u32 {
    let abs_level = level.unsigned_abs() as usize;
    let fixed = VP8_LEVEL_FIXED_COSTS[abs_level.min(MAX_LEVEL)] as u32;
    let variable = match costs {
        Some(table) => table[abs_level.min(MAX_VARIABLE_LEVEL)] as u32,
        None => 0,
    };
    fixed + variable
}

/// Fast inline level cost for trellis inner loop.
/// Assumes: level >= 0, level <= MAX_LEVEL, costs is valid.
/// Note: VP8LevelFixedCosts already includes the sign bit cost (256) for non-zero levels.
#[inline(always)]
fn level_cost_fast(costs: &LevelCostArray, level: usize) -> u32 {
    let fixed = VP8_LEVEL_FIXED_COSTS[level] as u32;
    let variable = costs[level.min(MAX_VARIABLE_LEVEL)] as u32;
    fixed + variable
}

/// Trellis-optimized quantization for a 4x4 block.
///
/// Uses dynamic programming to find optimal coefficient levels that minimize
/// the RD cost: distortion + lambda * rate.
///
/// This implementation follows libwebp's approach:
/// - Each state stores a pointer to the cost table for the *next* position
/// - Context is computed as min(level, 2) from the current level
/// - Predecessors' cost tables are used to compute transition costs
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients (will be modified with reconstructed values)
/// * `out` - Output quantized levels
/// * `mtx` - Quantization matrix
/// * `lambda` - Rate-distortion trade-off parameter
/// * `first` - First coefficient to process (1 for I16_AC mode, 0 otherwise)
/// * `level_costs` - Probability-dependent level costs
/// * `ctype` - Token type for level cost lookup (0=Y2, 1=Y_AC, 2=Y_DC, 3=UV)
/// * `ctx0` - Initial context from neighboring blocks (0, 1, or 2)
/// * `psy_config` - Perceptual config; when psy_trellis_strength > 0, adds a
///   penalty for zeroing perceptually important coefficients
///
/// # Returns
/// True if any non-zero coefficient was produced
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // p indexes multiple arrays with different semantics
#[allow(private_interfaces)] // psy_config is pub(crate), but this function is exposed for debugging
pub fn trellis_quantize_block(
    coeffs: &mut [i32; 16],
    out: &mut [i32; 16],
    mtx: &VP8Matrix,
    lambda: u32,
    first: usize,
    level_costs: &LevelCosts,
    ctype: usize,
    ctx0: usize,
    psy_config: &PsyConfig,
) -> bool {
    // Number of alternate levels to try: level-0 (MIN_DELTA=0) and level+1 (MAX_DELTA=1)
    const NUM_NODES: usize = 2; // [level, level+1]

    let mut nodes = [[TrellisNode::default(); NUM_NODES]; 16];
    let mut score_states = [[TrellisScoreState::default(); NUM_NODES]; 2];
    let mut ss_cur_idx = 0usize;
    let mut ss_prev_idx = 1usize;

    // Find last significant coefficient based on threshold
    let thresh = (mtx.q[1] as i64 * mtx.q[1] as i64 / 4) as i32;
    let mut last = first as i32 - 1;
    for n in (first..16).rev() {
        let j = VP8_ZIGZAG[n];
        let err = coeffs[j] * coeffs[j];
        if err > thresh {
            last = n as i32;
            break;
        }
    }
    // Extend by one to not lose too much
    if last < 15 {
        last += 1;
    }

    // Best path tracking: [last_pos, level_delta, prev_delta]
    let mut best_path = [-1i32; 3];

    // Initialize: compute skip score (all zeros = signal EOB immediately)
    // Skip means signaling EOB at the first position, not encoding a zero coefficient.
    let skip_cost = level_costs.get_skip_eob_cost(ctype, first, ctx0) as i64;
    let mut best_score = rd_score_trellis(lambda, skip_cost, 0);

    // Initialize source nodes with cost table for first position based on ctx0
    let initial_costs = level_costs.get_cost_table(ctype, first, ctx0);
    let init_rate = if ctx0 == 0 {
        level_costs.get_init_cost(ctype, first, ctx0) as i64
    } else {
        0
    };
    let init_score = rd_score_trellis(lambda, init_rate, 0);

    for state in &mut score_states[ss_cur_idx][..NUM_NODES] {
        *state = TrellisScoreState {
            score: init_score,
            costs: Some(initial_costs),
        };
    }

    // Traverse trellis
    for n in first..=last as usize {
        let j = VP8_ZIGZAG[n];
        let q = mtx.q[j] as i32;
        let iq = mtx.iq[j];
        let neutral_bias = quantization_bias(0x00);

        // Get sign from original coefficient
        let sign = coeffs[j] < 0;
        let abs_coeff = if sign { -coeffs[j] } else { coeffs[j] };
        let coeff_with_sharpen = abs_coeff + mtx.sharpen[j] as i32;

        // Base quantized level with neutral bias
        let level0 = quantdiv(coeff_with_sharpen as u32, iq, neutral_bias).min(MAX_LEVEL as i32);

        // Threshold level for pruning
        let thresh_bias = quantization_bias(0x80);
        let thresh_level =
            quantdiv(coeff_with_sharpen as u32, iq, thresh_bias).min(MAX_LEVEL as i32);

        // Swap score state indices
        core::mem::swap(&mut ss_cur_idx, &mut ss_prev_idx);

        // Test all alternate level values: level0 and level0+1
        for delta in 0..NUM_NODES {
            let node = &mut nodes[n][delta];
            let level = level0 + delta as i32;

            // Context for next position: min(level, 2)
            let ctx = (level as usize).min(2);

            // Store cost table for next position (based on context from this level)
            let next_costs = if n + 1 < 16 {
                Some(level_costs.get_cost_table(ctype, n + 1, ctx))
            } else {
                None
            };

            // Reset current score state
            score_states[ss_cur_idx][delta] = TrellisScoreState {
                score: MAX_COST,
                costs: next_costs,
            };

            // Skip invalid levels
            if level < 0 || level > thresh_level {
                continue;
            }

            // Compute distortion delta
            let new_error = coeff_with_sharpen - level * q;
            let orig_error_sq = (coeff_with_sharpen * coeff_with_sharpen) as i64;
            let new_error_sq = (new_error * new_error) as i64;
            let weight = VP8_WEIGHT_TRELLIS[j] as i64;
            let mut delta_distortion = weight * (new_error_sq - orig_error_sq);

            // Psy-trellis: penalize zeroing perceptually important coefficients.
            // When level == 0 but the original coefficient is non-zero, add a
            // distortion penalty proportional to the coefficient's energy and
            // its perceptual weight. This biases the DP toward retaining
            // coefficients that carry visible texture.
            //
            // JND (Just Noticeable Difference) gating (2026 algorithm):
            // Skip the penalty if the coefficient is below the JND threshold,
            // meaning zeroing it won't cause visible artifacts anyway.
            if level == 0 && abs_coeff > 0 && psy_config.psy_trellis_strength > 0 {
                // Check if coefficient is below JND threshold (imperceptible)
                // ctype: 0=Y2, 1=Y_AC, 2=Y_DC, 3=UV
                let is_chroma = ctype == 3;
                let below_jnd = if is_chroma {
                    psy_config.is_below_jnd_uv(j, coeffs[j])
                } else {
                    psy_config.is_below_jnd_y(j, coeffs[j])
                };

                // Only penalize zeroing if coefficient is perceptible
                if !below_jnd {
                    let energy = abs_coeff as i64;
                    let psy_weight = psy_config.trellis_weights[j] as i64;
                    let psy_penalty =
                        (psy_config.psy_trellis_strength as i64 * psy_weight * energy) >> 16;
                    delta_distortion += psy_penalty;
                }
            }

            let base_score = rd_score_trellis(lambda, 0, delta_distortion);

            // Find best predecessor using stored cost tables
            // Unrolled loop for NUM_NODES=2 with fast level cost
            let level_usize = level as usize;
            let (best_cur_score, best_prev) = {
                let ss_prev = &score_states[ss_prev_idx];

                // Predecessor 0
                let cost0 = if let Some(costs) = ss_prev[0].costs {
                    level_cost_fast(costs, level_usize) as i64
                } else {
                    VP8_LEVEL_FIXED_COSTS[level_usize] as i64
                };
                let score0 = ss_prev[0].score + cost0 * lambda as i64;

                // Predecessor 1
                let cost1 = if let Some(costs) = ss_prev[1].costs {
                    level_cost_fast(costs, level_usize) as i64
                } else {
                    VP8_LEVEL_FIXED_COSTS[level_usize] as i64
                };
                let score1 = ss_prev[1].score + cost1 * lambda as i64;

                // Select best
                if score1 < score0 {
                    (score1 + base_score, 1i8)
                } else {
                    (score0 + base_score, 0i8)
                }
            };

            // Store in node
            node.sign = sign;
            node.level = level as i16;
            node.prev = best_prev;
            score_states[ss_cur_idx][delta].score = best_cur_score;

            // Check if this is the best terminal node
            if level != 0 && best_cur_score < best_score {
                // Add end-of-block cost: signaling "no more coefficients"
                // Uses the context resulting from this level
                let eob_cost = if n < 15 {
                    level_costs.get_eob_cost(ctype, n, ctx) as i64
                } else {
                    0
                };
                let terminal_score = best_cur_score + rd_score_trellis(lambda, eob_cost, 0);
                if terminal_score < best_score {
                    best_score = terminal_score;
                    best_path[0] = n as i32;
                    best_path[1] = delta as i32;
                    best_path[2] = best_prev as i32;
                }
            }
        }
    }

    // Clear output
    if first == 1 {
        // Preserve DC for I16_AC mode
        out[1..].fill(0);
        coeffs[1..].fill(0);
    } else {
        out.fill(0);
        coeffs.fill(0);
    }

    // No non-zero coefficients - skip block
    if best_path[0] == -1 {
        return false;
    }

    // Unwind best path
    let mut has_nz = false;
    let mut best_node_delta = best_path[1] as usize;
    let mut n = best_path[0] as usize;

    // Patch the prev for the terminal node
    nodes[n][best_node_delta].prev = best_path[2] as i8;

    loop {
        let node = &nodes[n][best_node_delta];
        let j = VP8_ZIGZAG[n];
        let level = if node.sign {
            -node.level as i32
        } else {
            node.level as i32
        };

        out[n] = level;
        has_nz |= level != 0;

        // Reconstruct coefficient for subsequent prediction
        coeffs[j] = level * mtx.q[j] as i32;

        if n == first {
            break;
        }
        best_node_delta = node.prev as usize;
        n -= 1;
    }

    has_nz
}
