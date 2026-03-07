//! Residual coefficient encoding and statistics recording.
//!
//! Contains methods for encoding DCT coefficients into the bitstream,
//! trellis-optimized quantization dispatch, error diffusion, token buffer
//! for deferred encoding, and probability statistics recording.

use alloc::vec::Vec;

use crate::common::transform;
use crate::common::types::*;

use super::{MacroblockInfo, QuantizedMbCoeffs};
use crate::encoder::cost::{
    NUM_BANDS, NUM_CTX, NUM_PROBAS, NUM_TYPES, ProbaStats, TokenType, VP8_ENC_BANDS, record_coeffs,
    trellis_quantize_block,
};

// -----------------------------------------------------------------------
// Token buffer for deferred coefficient encoding
// -----------------------------------------------------------------------

/// Flag indicating a token has a constant (embedded) probability.
const FIXED_PROBA_BIT: u16 = 1 << 14;

/// Compute flat probability index base for `probs[type][band][ctx][0]`.
/// Add the node offset (0-10) to get the specific probability index.
#[inline]
fn token_id(coeff_type: usize, band: usize, ctx: usize) -> u16 {
    debug_assert!(coeff_type < NUM_TYPES);
    debug_assert!(band < NUM_BANDS);
    debug_assert!(ctx < NUM_CTX);
    (NUM_PROBAS * (ctx + NUM_CTX * (band + NUM_BANDS * coeff_type))) as u16
}

/// Buffer of bit-level tokens for deferred emission to the arithmetic encoder.
///
/// Each token is a u16 matching libwebp's `token_t` layout (token_enc.c):
/// - bit 15: bit value (0 or 1)
/// - bit 14: FIXED_PROBA_BIT (1 = constant probability, 0 = indexed)
/// - bits 0-13: flat probability index (if dynamic) or probability value (if constant)
///
/// During recording, tokens are stored alongside ProbaStats updates.
/// During emission, dynamic tokens look up probabilities from the (now-updated)
/// probability table, while constant tokens use their embedded probability.
pub(crate) struct TokenBuffer {
    tokens: Vec<u16>,
}

#[allow(dead_code)]
impl TokenBuffer {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn with_estimated_capacity(num_macroblocks: usize) -> Self {
        // libwebp uses 2 * 16 * 24 = 768 tokens per macroblock
        Self {
            tokens: Vec::with_capacity(num_macroblocks * 768),
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    /// Record a dynamic-probability bit decision.
    #[inline]
    fn add_token(&mut self, bit: bool, proba_idx: u16) {
        debug_assert!(
            proba_idx < FIXED_PROBA_BIT,
            "proba_idx {proba_idx} overlaps FIXED_PROBA_BIT"
        );
        self.tokens.push(((bit as u16) << 15) | proba_idx);
    }

    /// Record a fixed-probability bit decision.
    #[inline]
    fn add_constant_token(&mut self, bit: bool, proba: u8) {
        self.tokens
            .push(((bit as u16) << 15) | FIXED_PROBA_BIT | (proba as u16));
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Emit all tokens to an arithmetic encoder using the given probability table.
    ///
    /// Dynamic tokens look up their probability from `probas` using the stored
    /// flat index. Constant tokens use their embedded probability directly.
    ///
    /// Matches libwebp's `VP8EmitTokens` (token_enc.c).
    pub fn emit_tokens(
        &self,
        encoder: &mut crate::encoder::arithmetic::ArithmeticEncoder,
        probas: &TokenProbTables,
    ) {
        // Flatten 4D probability table to 1D for direct indexing.
        // Layout matches token_id(): probas[t][b][c][p] → flat[token_id(t,b,c) + p]
        // This eliminates 5 integer divisions per dynamic token in the hot loop.
        let flat_probas = Self::flatten_probas(probas);

        for &token in &self.tokens {
            let bit = (token >> 15) != 0;
            if (token & FIXED_PROBA_BIT) != 0 {
                // Constant probability embedded in token
                let proba = (token & 0xff) as u8;
                encoder.write_bool(bit, proba);
            } else {
                // Dynamic probability: direct flat index lookup (no decomposition needed)
                let idx = (token & 0x3fff) as usize;
                encoder.write_bool(bit, flat_probas[idx]);
            }
        }
    }

    /// Flatten 4D probability table to match token_id() flat indexing.
    /// TokenProbTables is [4][8][3][11] = 1056 bytes, same layout as flat array.
    fn flatten_probas(
        probas: &TokenProbTables,
    ) -> [u8; NUM_TYPES * NUM_BANDS * NUM_CTX * NUM_PROBAS] {
        let mut flat = [0u8; NUM_TYPES * NUM_BANDS * NUM_CTX * NUM_PROBAS];
        for (t, type_bands) in probas.iter().enumerate() {
            for (b, band_ctxs) in type_bands.iter().enumerate() {
                for (c, ctx_probas) in band_ctxs.iter().enumerate() {
                    let offset = NUM_PROBAS * (c + NUM_CTX * (b + NUM_BANDS * t));
                    flat[offset..offset + NUM_PROBAS].copy_from_slice(ctx_probas);
                }
            }
        }
        flat
    }

    /// Record tokens for a quantized coefficient block.
    ///
    /// Simultaneously records bit-level tokens for deferred emission AND
    /// updates ProbaStats for probability refinement.
    ///
    /// Returns true if the block has non-zero coefficients (for complexity tracking).
    ///
    /// Optimized vs libwebp's `VP8RecordCoeffTokens` (token_enc.c):
    /// - Pre-resolves stats row `&mut [u32; 11]` per band/ctx (eliminates 4D index per call)
    /// - Inlines magnitude tree (avoids passing &mut stats through function boundary)
    pub fn record_coeff_tokens(
        &mut self,
        stats: &mut ProbaStats,
        coeffs: &[i32; 16],
        coeff_type: usize,
        first_coeff: usize,
        initial_ctx: usize,
    ) -> bool {
        // Find last non-zero coefficient
        let last = coeffs[first_coeff..]
            .iter()
            .rposition(|&c| c != 0)
            .map(|i| (i + first_coeff) as i32)
            .unwrap_or(-1);

        let mut n = first_coeff;
        let mut band = VP8_ENC_BANDS[n] as usize;
        let mut ctx = initial_ctx;
        let mut base_id = token_id(coeff_type, band, ctx);

        // Node 0: Is there any non-zero coefficient? (not EOB)
        let not_eob = last >= first_coeff as i32;
        self.add_token(not_eob, base_id);
        record_stat(&mut stats.stats[coeff_type][band][ctx][0], not_eob);

        if !not_eob {
            return false; // All zero
        }

        while n < 16 {
            let c = coeffs[n];
            n += 1;
            let sign = c < 0;
            let v = c.unsigned_abs();

            // Pre-resolve stats row for current band/ctx.
            // All nodes (1-10) within one coefficient use the same band/ctx.
            let s = &mut stats.stats[coeff_type][band][ctx];

            // Node 1: Is this coefficient non-zero?
            self.add_token(v != 0, base_id + 1);
            record_stat(&mut s[1], v != 0);

            if v == 0 {
                // Zero coefficient: context = 0
                band = VP8_ENC_BANDS[n] as usize;
                ctx = 0;
                base_id = token_id(coeff_type, band, ctx);
                continue;
            }

            // Node 2: Is |coeff| > 1?
            self.add_token(v > 1, base_id + 2);
            record_stat(&mut s[2], v > 1);

            if v > 1 {
                // Magnitude tree (nodes 3-10), all same band/ctx as s
                self.add_token(v > 4, base_id + 3);
                record_stat(&mut s[3], v > 4);

                if v <= 4 {
                    self.add_token(v != 2, base_id + 4);
                    record_stat(&mut s[4], v != 2);
                    if v != 2 {
                        self.add_token(v == 4, base_id + 5);
                        record_stat(&mut s[5], v == 4);
                    }
                } else {
                    self.add_token(v > 10, base_id + 6);
                    record_stat(&mut s[6], v > 10);

                    if v <= 10 {
                        self.add_token(v > 6, base_id + 7);
                        record_stat(&mut s[7], v > 6);

                        if v <= 6 {
                            self.add_constant_token(v == 6, 159);
                        } else {
                            self.add_constant_token(v >= 9, 165);
                            self.add_constant_token((v & 1) == 0, 145);
                        }
                    } else {
                        let residue = v - 3;
                        if residue < (8 << 1) {
                            // Cat3
                            self.add_token(false, base_id + 8);
                            record_stat(&mut s[8], false);
                            self.add_token(false, base_id + 9);
                            record_stat(&mut s[9], false);
                            let r = residue - 8;
                            for (i, &prob) in PROB_DCT_CAT[2].iter().enumerate() {
                                if prob == 0 {
                                    break;
                                }
                                self.add_constant_token((r & (1 << (2 - i))) != 0, prob);
                            }
                        } else if residue < (8 << 2) {
                            // Cat4
                            self.add_token(false, base_id + 8);
                            record_stat(&mut s[8], false);
                            self.add_token(true, base_id + 9);
                            record_stat(&mut s[9], true);
                            let r = residue - (8 << 1);
                            for (i, &prob) in PROB_DCT_CAT[3].iter().enumerate() {
                                if prob == 0 {
                                    break;
                                }
                                self.add_constant_token((r & (1 << (3 - i))) != 0, prob);
                            }
                        } else if residue < (8 << 3) {
                            // Cat5
                            self.add_token(true, base_id + 8);
                            record_stat(&mut s[8], true);
                            self.add_token(false, base_id + 10);
                            record_stat(&mut s[10], false);
                            let r = residue - (8 << 2);
                            for (i, &prob) in PROB_DCT_CAT[4].iter().enumerate() {
                                if prob == 0 {
                                    break;
                                }
                                self.add_constant_token((r & (1 << (4 - i))) != 0, prob);
                            }
                        } else {
                            // Cat6
                            self.add_token(true, base_id + 8);
                            record_stat(&mut s[8], true);
                            self.add_token(true, base_id + 10);
                            record_stat(&mut s[10], true);
                            let r = residue - (8 << 3);
                            for (i, &prob) in PROB_DCT_CAT[5].iter().enumerate() {
                                if prob == 0 {
                                    break;
                                }
                                self.add_constant_token((r & (1 << (10 - i))) != 0, prob);
                            }
                        }
                    }
                }
            }

            // Update band/ctx for next coefficient
            band = VP8_ENC_BANDS[n] as usize;
            ctx = if v <= 1 { 1 } else { 2 };
            base_id = token_id(coeff_type, band, ctx);

            // Sign bit (constant probability 128)
            self.add_constant_token(sign, 128);

            // Node 0 for next position: more non-zero coefficients?
            if n == 16 {
                return true;
            }
            let not_eob_next = (n as i32) <= last;
            self.add_token(not_eob_next, base_id);
            record_stat(&mut stats.stats[coeff_type][band][ctx][0], not_eob_next);

            if !not_eob_next {
                return true; // EOB
            }
        }

        true
    }
}

/// Record a single probability statistic (inlined).
/// Matches libwebp's VP8RecordStats: upper 16 bits = total, lower 16 = count of 1s.
#[inline(always)]
fn record_stat(stat: &mut u32, bit: bool) {
    if *stat >= 0xfffe_0000 {
        *stat = ((*stat + 1) >> 1) & 0x7fff_7fff;
    }
    *stat += 0x0001_0000 + bit as u32;
}

impl<'a> super::Vp8Encoder<'a> {
    /// Apply Floyd-Steinberg-like error diffusion to chroma DC coefficients.
    ///
    /// This reduces banding artifacts in smooth gradients by spreading quantization
    /// error to neighboring blocks. The pattern for a 2x2 grid of chroma blocks:
    /// ```text
    /// | c[0] | c[1] |    errors: err0, err1
    /// | c[2] | c[3] |            err2, err3
    /// ```
    ///
    /// Modifies the DC coefficients in place and stores errors for next macroblock.
    pub(super) fn apply_chroma_error_diffusion(
        &mut self,
        u_blocks: &mut [i32; 16 * 4],
        v_blocks: &mut [i32; 16 * 4],
        mbx: usize,
        uv_matrix: &crate::encoder::cost::VP8Matrix,
    ) {
        // Diffusion constants from libwebp
        const C1: i32 = 7; // fraction from top
        const C2: i32 = 8; // fraction from left
        const DSHIFT: i32 = 4;
        const DSCALE: i32 = 1;

        let q = uv_matrix.q[0] as i32;
        let iq = uv_matrix.iq[0];
        let bias = uv_matrix.bias[0];

        // Helper: add diffused error to DC, predict quantization, return error
        // Does NOT overwrite the coefficient - leaves it adjusted for encoding
        let diffuse_dc = |dc: &mut i32, top_err: i8, left_err: i8| -> i8 {
            // Add diffused error from neighbors
            let adjustment = (C1 * top_err as i32 + C2 * left_err as i32) >> (DSHIFT - DSCALE);
            *dc += adjustment;

            // Predict what quantization will produce (to compute error)
            let sign = *dc < 0;
            let abs_dc = dc.unsigned_abs();

            let zthresh = ((1u32 << 17) - 1 - bias) / iq;
            let level = if abs_dc > zthresh {
                ((abs_dc * iq + bias) >> 17) as i32
            } else {
                0
            };

            // Error = |adjusted_input| - |reconstruction|
            // This is what we'll diffuse to neighbors
            let err = abs_dc as i32 - level * q;
            let signed_err = if sign { -err } else { err };
            (signed_err >> DSCALE).clamp(-127, 127) as i8
        };

        // Process each channel's 4 blocks in scan order
        let process_channel =
            |blocks: &mut [i32; 16 * 4], top: [i8; 2], left: [i8; 2]| -> [i8; 4] {
                // Block 0 (position 0): top-left, uses top[0] and left[0]
                let err0 = diffuse_dc(&mut blocks[0], top[0], left[0]);

                // Block 1 (position 16): top-right, uses top[1] and err0
                let err1 = diffuse_dc(&mut blocks[16], top[1], err0);

                // Block 2 (position 32): bottom-left, uses err0 and left[1]
                let err2 = diffuse_dc(&mut blocks[32], err0, left[1]);

                // Block 3 (position 48): bottom-right, uses err1 and err2
                let err3 = diffuse_dc(&mut blocks[48], err1, err2);

                [err0, err1, err2, err3]
            };

        // Process U channel
        let u_errs = process_channel(u_blocks, self.top_derr[mbx][0], self.left_derr[0]);
        // Process V channel
        let v_errs = process_channel(v_blocks, self.top_derr[mbx][1], self.left_derr[1]);

        // Store errors for next macroblock
        for (ch, errs) in [(0usize, u_errs), (1, v_errs)] {
            let [_err0, err1, err2, err3] = errs;
            // left[0] = err1, left[1] = 3/4 of err3
            self.left_derr[ch][0] = err1;
            self.left_derr[ch][1] = ((3 * err3 as i32) >> 2) as i8;
            // top[0] = err2, top[1] = 1/4 of err3
            self.top_derr[mbx][ch][0] = err2;
            self.top_derr[mbx][ch][1] = (err3 as i32 - self.left_derr[ch][1] as i32) as i8;
        }
    }

    // 13 in specification, matches read_residual_data in the decoder
    // Kept as fallback path — the token buffer path (record_residual_tokens + emit_tokens)
    // replaces this for normal encoding.
    #[allow(dead_code)]
    pub(super) fn encode_residual_data(
        &mut self,
        macroblock_info: &MacroblockInfo,
        partition_index: usize,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        // Extract VP8 matrices and trellis lambdas upfront to avoid borrow conflicts
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();
        let psy_config = segment.psy_config.clone();

        // Trellis lambda for Y1 blocks (I4 vs I16 mode)
        let is_i4 = macroblock_info.luma_mode == LumaMode::B;
        let y1_trellis_lambda = if self.do_trellis {
            Some(if is_i4 {
                segment.lambda_trellis_i4
            } else {
                segment.lambda_trellis_i16
            })
        } else {
            None
        };
        // Note: libwebp disables trellis for UV (DO_TRELLIS_UV = 0)
        // and for Y2 DC coefficients, so we use None for those

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);

            // wht here on the 0th coeffs
            transform::wht4x4(&mut coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;

            let has_coeffs = self.encode_coefficients(
                &coeffs0,
                partition_index,
                plane,
                complexity.into(),
                &y2_matrix,
                None, // No trellis for Y2 DC
                &psy_config,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            // next encode luma coefficients without the 0th coeffs
            plane = Plane::YCoeff1;
        }

        // now encode the 16 luma 4x4 subblocks in the macroblock
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &y1_matrix,
                    y1_trellis_lambda,
                    &psy_config,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            // set for the next macroblock
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

        // Note: Error diffusion is already applied in transform_chroma_blocks
        // so u_block_data and v_block_data contain the error-diffused values

        // encode the 4 u 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &uv_matrix,
                    None, // No trellis for UV (libwebp: DO_TRELLIS_UV = 0)
                    &psy_config,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // encode the 4 v 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    &uv_matrix,
                    None, // No trellis for UV (libwebp: DO_TRELLIS_UV = 0)
                    &psy_config,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    // encodes the coefficients which is the reverse procedure of read_coefficients in the decoder
    // returns whether there was any non-zero data in the block for the complexity
    // Kept as fallback — token buffer path replaces this.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_coefficients(
        &mut self,
        block: &[i32; 16],
        partition_index: usize,
        plane: Plane,
        complexity: usize,
        matrix: &crate::encoder::cost::VP8Matrix,
        trellis_lambda: Option<u32>,
        psy_config: &crate::encoder::psy::PsyConfig,
    ) -> bool {
        // transform block
        // dc is used for the 0th coefficient, ac for the others

        let encoder = &mut self.partitions[partition_index];

        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];

        assert!(complexity <= 2);
        let mut complexity = complexity;

        // convert to zigzag and quantize using VP8Matrix biased quantization
        // this is the only lossy part of the encoding
        let mut zigzag_block = [0i32; 16];

        if let Some(lambda) = trellis_lambda {
            // Trellis quantization for better RD optimization
            let mut coeffs = *block;
            let ctype = plane as usize;
            trellis_quantize_block(
                &mut coeffs,
                &mut zigzag_block,
                matrix,
                lambda,
                first_coeff,
                &self.level_costs,
                ctype,
                complexity,
                psy_config,
            );
        } else {
            // Simple quantization
            for i in first_coeff..16 {
                let zigzag_index = usize::from(ZIGZAG[i]);
                zigzag_block[i] = matrix.quantize_coeff(block[zigzag_index], zigzag_index);
            }
        }

        // get index of last coefficient that isn't 0
        let end_of_block_index =
            if let Some(last_non_zero_index) = zigzag_block.iter().rev().position(|x| *x != 0) {
                (15 - last_non_zero_index) + 1
            } else {
                // if it's all 0s then the first block is end of block
                0
            };

        let mut skip_eob = false;

        for index in first_coeff..end_of_block_index {
            let coeff = zigzag_block[index];

            let band = usize::from(COEFF_BANDS[index]);
            let probabilities = &probs[band][complexity];
            let start_index_token_tree = if skip_eob { 2 } else { 0 };
            let token_tree = &DCT_TOKEN_TREE;
            let token_probs = probabilities;

            let token = match coeff.abs() {
                0 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        DCT_0,
                        start_index_token_tree,
                    );

                    // never going to have an end of block after a 0, so skip checking next coeff
                    skip_eob = true;
                    DCT_0
                }

                // just encode as literal
                literal @ 1..=4 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        literal as i8,
                        start_index_token_tree,
                    );

                    skip_eob = false;
                    literal as i8
                }

                // encode the category
                value => {
                    let category = match value {
                        5..=6 => DCT_CAT1,
                        7..=10 => DCT_CAT2,
                        11..=18 => DCT_CAT3,
                        19..=34 => DCT_CAT4,
                        35..=66 => DCT_CAT5,
                        67..=2048 => DCT_CAT6,
                        _ => unreachable!(),
                    };

                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        category,
                        start_index_token_tree,
                    );

                    let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];

                    let extra = value - i32::from(DCT_CAT_BASE[(category - DCT_CAT1) as usize]);

                    let mut mask = if category == DCT_CAT6 {
                        1 << (11 - 1)
                    } else {
                        1 << (category - DCT_CAT1)
                    };

                    for &prob in category_probs.iter() {
                        if prob == 0 {
                            break;
                        }
                        let extra_bool = extra & mask > 0;
                        encoder.write_bool(extra_bool, prob);
                        mask >>= 1;
                    }

                    skip_eob = false;

                    category
                }
            };

            // encode sign if token is not zero
            if token != DCT_0 {
                // note flag means coeff is negative
                encoder.write_flag(!coeff.is_positive());
            }

            complexity = match token {
                DCT_0 => 0,
                DCT_1 => 1,
                _ => 2,
            };
        }

        // encode end of block
        if end_of_block_index < 16 {
            let band_index = usize::max(first_coeff, end_of_block_index);
            let band = usize::from(COEFF_BANDS[band_index]);
            let probabilities = &probs[band][complexity];
            encoder.write_with_tree(&DCT_TOKEN_TREE, probabilities, DCT_EOB);
        }

        // whether the block has a non zero coefficient
        end_of_block_index > 0
    }

    /// Check if all coefficients in a macroblock would quantize to zero.
    /// Used for skip detection.
    pub(super) fn check_all_coeffs_zero(
        &self,
        macroblock_info: &MacroblockInfo,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) -> bool {
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.as_ref().unwrap();
        let y2_matrix = segment.y2_matrix.as_ref().unwrap();
        let uv_matrix = segment.uv_matrix.as_ref().unwrap();

        // For Intra16 mode, check Y2 block (DC coefficients after WHT)
        if macroblock_info.luma_mode != LumaMode::B {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            for (idx, &val) in coeffs0.iter().enumerate() {
                if y2_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }

            // Check Y blocks (AC only for Intra16)
            for block in y_block_data.chunks_exact(16) {
                for (idx, &val) in block.iter().enumerate().skip(1) {
                    if y1_matrix.quantize_coeff(val, idx) != 0 {
                        return false;
                    }
                }
            }
        } else {
            // For Intra4 mode, check all Y coefficients (DC + AC)
            for block in y_block_data.chunks_exact(16) {
                for (idx, &val) in block.iter().enumerate() {
                    if y1_matrix.quantize_coeff(val, idx) != 0 {
                        return false;
                    }
                }
            }
        }

        // Check U blocks
        for block in u_block_data.chunks_exact(16) {
            for (idx, &val) in block.iter().enumerate() {
                if uv_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }
        }

        // Check V blocks
        for block in v_block_data.chunks_exact(16) {
            for (idx, &val) in block.iter().enumerate() {
                if uv_matrix.quantize_coeff(val, idx) != 0 {
                    return false;
                }
            }
        }

        true
    }

    /// Record token statistics for a macroblock (used in first pass of two-pass encoding).
    /// This mirrors the structure of encode_residual_data but only records stats.
    /// Kept as fallback — token buffer path replaces this.
    #[allow(dead_code)]
    pub(super) fn record_residual_stats(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        // Extract VP8 matrices
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();
        let psy_config = segment.psy_config.clone();

        let is_i4 = macroblock_info.luma_mode == LumaMode::B;

        // Y2 (DC transform) - only for I16 mode
        if !is_i4 {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            // Quantize to zigzag order
            let mut zigzag = [0i32; 16];
            for i in 0..16 {
                let zigzag_index = usize::from(ZIGZAG[i]);
                zigzag[i] = y2_matrix.quantize_coeff(coeffs0[zigzag_index], zigzag_index);
            }

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            record_coeffs(
                &zigzag,
                TokenType::I16DC,
                0,
                complexity.min(2) as usize,
                &mut self.proba_stats,
            );

            let has_coeffs = zigzag.iter().any(|&c| c != 0);
            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };
        }

        // Y1 blocks (AC only for I16, DC+AC for I4)
        let token_type = if is_i4 {
            TokenType::I4
        } else {
            TokenType::I16AC
        };
        let first_coeff = if is_i4 { 0 } else { 1 };

        // Get trellis lambda based on block type
        let trellis_lambda = if is_i4 {
            segment.lambda_trellis_i4
        } else {
            segment.lambda_trellis_i16
        };

        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block: &[i32; 16] = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                // Quantize to zigzag order
                let mut zigzag = [0i32; 16];

                let top = self.top_complexity[mbx].y[x];
                let ctx0 = (left + top).min(2) as usize;

                if self.do_trellis {
                    // Trellis quantization: optimizes coefficient levels for RD
                    let mut coeffs = *block;
                    // Use same ctype as record_coeffs: TokenType I4=3 or I16AC=0
                    let ctype = token_type as usize;
                    trellis_quantize_block(
                        &mut coeffs,
                        &mut zigzag,
                        &y1_matrix,
                        trellis_lambda,
                        first_coeff,
                        &self.level_costs,
                        ctype,
                        ctx0,
                        &psy_config,
                    );
                } else {
                    // Simple quantization
                    for i in first_coeff..16 {
                        let zigzag_index = usize::from(ZIGZAG[i]);
                        zigzag[i] = y1_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                    }
                }

                record_coeffs(
                    &zigzag,
                    token_type,
                    first_coeff,
                    ctx0,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag[first_coeff..].iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        // U blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zigzag_index = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                }

                let top = self.top_complexity[mbx].u[x];
                let complexity = (left + top).min(2);

                record_coeffs(
                    &zigzag,
                    TokenType::Chroma,
                    0,
                    complexity as usize,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag.iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // V blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zigzag_index = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zigzag_index], zigzag_index);
                }

                let top = self.top_complexity[mbx].v[x];
                let complexity = (left + top).min(2);

                record_coeffs(
                    &zigzag,
                    TokenType::Chroma,
                    0,
                    complexity as usize,
                    &mut self.proba_stats,
                );

                let has_coeffs = zigzag.iter().any(|&c| c != 0);
                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    /// Record tokens for a macroblock's residual data using the token buffer.
    ///
    /// This replaces both `encode_residual_data` (coefficient emission) and
    /// `record_residual_stats` (probability statistics) for the token buffer path.
    /// Tokens are stored for deferred emission; statistics are updated simultaneously.
    ///
    /// Includes quantization (trellis or simple) matching the encode path exactly.
    ///
    /// NOTE: Currently unused - record_residual_tokens_storing is used instead for
    /// multi-pass encoding. Kept for reference and potential future use.
    #[allow(dead_code)]
    pub(super) fn record_residual_tokens(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        // Extract VP8 matrices and trellis lambdas upfront
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();
        let psy_config = segment.psy_config.clone();

        let is_i4 = macroblock_info.luma_mode == LumaMode::B;
        let y1_trellis_lambda = if self.do_trellis {
            Some(if is_i4 {
                segment.lambda_trellis_i4
            } else {
                segment.lambda_trellis_i16
            })
        } else {
            None
        };

        let token_type_y1 = if is_i4 {
            TokenType::I4
        } else {
            TokenType::I16AC
        };
        let first_coeff_y1 = if is_i4 { 0 } else { 1 };

        // Temporarily take token buffer to avoid borrow conflicts with self.proba_stats
        let mut token_buf = self
            .token_buffer
            .take()
            .expect("token buffer not initialized");

        // Y2 (DC transform) - only for I16 mode
        if !is_i4 {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            let mut zigzag = [0i32; 16];
            for i in 0..16 {
                let zi = usize::from(ZIGZAG[i]);
                zigzag[i] = y2_matrix.quantize_coeff(coeffs0[zi], zi);
            }

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            let has_coeffs = token_buf.record_coeff_tokens(
                &mut self.proba_stats,
                &zigzag,
                TokenType::I16DC as usize,
                0,
                complexity.min(2) as usize,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };
        }

        // Y1 blocks (AC only for I16, DC+AC for I4)
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block: &[i32; 16] = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                let top = self.top_complexity[mbx].y[x];
                let ctx0 = (left + top).min(2) as usize;

                if let Some(lambda) = y1_trellis_lambda {
                    let mut coeffs = *block;
                    let ctype = token_type_y1 as usize;
                    trellis_quantize_block(
                        &mut coeffs,
                        &mut zigzag,
                        &y1_matrix,
                        lambda,
                        first_coeff_y1,
                        &self.level_costs,
                        ctype,
                        ctx0,
                        &psy_config,
                    );
                } else {
                    for i in first_coeff_y1..16 {
                        let zi = usize::from(ZIGZAG[i]);
                        zigzag[i] = y1_matrix.quantize_coeff(block[zi], zi);
                    }
                }

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &zigzag,
                    token_type_y1 as usize,
                    first_coeff_y1,
                    ctx0,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        // U blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zi = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zi], zi);
                }

                let top = self.top_complexity[mbx].u[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &zigzag,
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // V blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block: &[i32; 16] = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let mut zigzag = [0i32; 16];
                for i in 0..16 {
                    let zi = usize::from(ZIGZAG[i]);
                    zigzag[i] = uv_matrix.quantize_coeff(block[zi], zi);
                }

                let top = self.top_complexity[mbx].v[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &zigzag,
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }

        // Put token buffer back
        self.token_buffer = Some(token_buf);
    }

    /// Record tokens for a macroblock's residual data and return the quantized coefficients.
    /// Used in multi-pass encoding: pass 1 calls this to get coefficients for storage.
    #[allow(clippy::needless_range_loop)] // i is used to index both ZIGZAG and output arrays
    pub(super) fn record_residual_tokens_storing(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) -> QuantizedMbCoeffs {
        let segment = &self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let y1_matrix = segment.y1_matrix.clone().unwrap();
        let y2_matrix = segment.y2_matrix.clone().unwrap();
        let uv_matrix = segment.uv_matrix.clone().unwrap();
        let psy_config = segment.psy_config.clone();

        let is_i4 = macroblock_info.luma_mode == LumaMode::B;
        let y1_trellis_lambda = if self.do_trellis {
            Some(if is_i4 {
                segment.lambda_trellis_i4
            } else {
                segment.lambda_trellis_i16
            })
        } else {
            None
        };

        let token_type_y1 = if is_i4 {
            TokenType::I4
        } else {
            TokenType::I16AC
        };
        let first_coeff_y1 = if is_i4 { 0 } else { 1 };

        let mut token_buf = self
            .token_buffer
            .take()
            .expect("token buffer not initialized");

        // Storage for quantized coefficients
        let mut stored = QuantizedMbCoeffs {
            y2_zigzag: [0; 16],
            y1_zigzag: [[0; 16]; 16],
            u_zigzag: [[0; 16]; 4],
            v_zigzag: [[0; 16]; 4],
        };

        // Y2 (DC transform) - only for I16 mode
        if !is_i4 {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            for i in 0..16 {
                let zi = usize::from(ZIGZAG[i]);
                stored.y2_zigzag[i] = y2_matrix.quantize_coeff(coeffs0[zi], zi);
            }

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            let has_coeffs = token_buf.record_coeff_tokens(
                &mut self.proba_stats,
                &stored.y2_zigzag,
                TokenType::I16DC as usize,
                0,
                complexity.min(2) as usize,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };
        }

        // Y1 blocks
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block_idx = y * 4 + x;
                let block: &[i32; 16] = y_block_data[block_idx * 16..][..16].try_into().unwrap();

                let top = self.top_complexity[mbx].y[x];
                let ctx0 = (left + top).min(2) as usize;

                if let Some(lambda) = y1_trellis_lambda {
                    let mut coeffs = *block;
                    trellis_quantize_block(
                        &mut coeffs,
                        &mut stored.y1_zigzag[block_idx],
                        &y1_matrix,
                        lambda,
                        first_coeff_y1,
                        &self.level_costs,
                        token_type_y1 as usize,
                        ctx0,
                        &psy_config,
                    );
                } else {
                    for i in first_coeff_y1..16 {
                        let zi = usize::from(ZIGZAG[i]);
                        stored.y1_zigzag[block_idx][i] = y1_matrix.quantize_coeff(block[zi], zi);
                    }
                }

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.y1_zigzag[block_idx],
                    token_type_y1 as usize,
                    first_coeff_y1,
                    ctx0,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        // U blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block_idx = y * 2 + x;
                let block: &[i32; 16] = u_block_data[block_idx * 16..][..16].try_into().unwrap();

                for i in 0..16 {
                    let zi = usize::from(ZIGZAG[i]);
                    stored.u_zigzag[block_idx][i] = uv_matrix.quantize_coeff(block[zi], zi);
                }

                let top = self.top_complexity[mbx].u[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.u_zigzag[block_idx],
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // V blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block_idx = y * 2 + x;
                let block: &[i32; 16] = v_block_data[block_idx * 16..][..16].try_into().unwrap();

                for i in 0..16 {
                    let zi = usize::from(ZIGZAG[i]);
                    stored.v_zigzag[block_idx][i] = uv_matrix.quantize_coeff(block[zi], zi);
                }

                let top = self.top_complexity[mbx].v[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.v_zigzag[block_idx],
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }

        self.token_buffer = Some(token_buf);
        stored
    }

    /// Record tokens from pre-quantized coefficients (no re-quantization).
    /// Kept as fallback for comparison - true multi-pass uses `requantize_and_record`.
    #[allow(dead_code)]
    pub(super) fn record_from_stored_coeffs(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        stored: &QuantizedMbCoeffs,
    ) {
        let is_i4 = macroblock_info.luma_mode == LumaMode::B;

        let token_type_y1 = if is_i4 {
            TokenType::I4
        } else {
            TokenType::I16AC
        };
        let first_coeff_y1 = if is_i4 { 0 } else { 1 };

        let mut token_buf = self
            .token_buffer
            .take()
            .expect("token buffer not initialized");

        // Y2 (DC transform) - only for I16 mode
        if !is_i4 {
            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            let has_coeffs = token_buf.record_coeff_tokens(
                &mut self.proba_stats,
                &stored.y2_zigzag,
                TokenType::I16DC as usize,
                0,
                complexity.min(2) as usize,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };
        }

        // Y1 blocks
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block_idx = y * 4 + x;
                let top = self.top_complexity[mbx].y[x];
                let ctx0 = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.y1_zigzag[block_idx],
                    token_type_y1 as usize,
                    first_coeff_y1,
                    ctx0,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        // U blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block_idx = y * 2 + x;
                let top = self.top_complexity[mbx].u[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.u_zigzag[block_idx],
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // V blocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block_idx = y * 2 + x;
                let top = self.top_complexity[mbx].v[x];
                let complexity = (left + top).min(2) as usize;

                let has_coeffs = token_buf.record_coeff_tokens(
                    &mut self.proba_stats,
                    &stored.v_zigzag[block_idx],
                    TokenType::Chroma as usize,
                    0,
                    complexity,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }

        self.token_buffer = Some(token_buf);
    }
}

pub(super) fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
}
