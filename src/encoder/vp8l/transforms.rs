//! Image transforms for VP8L encoding.
//!
//! Transforms are applied to decorrelate image data before entropy coding.

#![allow(clippy::too_many_arguments)]

use alloc::vec;
use alloc::vec::Vec;

use super::types::{argb_alpha, argb_blue, argb_green, argb_red, make_argb, subsample_size};

/// Transform type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransformType {
    Predictor = 0,
    CrossColor = 1,
    SubtractGreen = 2,
    ColorIndexing = 3,
}

/// Predictor modes (14 total, per VP8L spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PredictorMode {
    Black = 0,                 // 0xff000000
    Left = 1,                  // L
    Top = 2,                   // T
    TopRight = 3,              // TR
    TopLeft = 4,               // TL
    AvgAvgLtrT = 5,            // Avg(Avg(L,TR), T)
    AvgLTl = 6,                // Avg(L, TL)
    AvgLT = 7,                 // Avg(L, T)
    AvgTlT = 8,                // Avg(TL, T)
    AvgTTr = 9,                // Avg(T, TR)
    AvgAvgLTlAvgTTr = 10,      // Avg(Avg(L,TL), Avg(T,TR))
    Select = 11,               // Select(L, T, TL)
    ClampAddSubtractFull = 12, // Clamp(L + T - TL)
    ClampAddSubtractHalf = 13, // Clamp(Avg(L,T) + (Avg(L,T)-TL)/2)
}

impl PredictorMode {
    /// Get all predictor modes.
    pub const fn all() -> [PredictorMode; 14] {
        [
            PredictorMode::Black,
            PredictorMode::Left,
            PredictorMode::Top,
            PredictorMode::TopRight,
            PredictorMode::TopLeft,
            PredictorMode::AvgAvgLtrT,
            PredictorMode::AvgLTl,
            PredictorMode::AvgLT,
            PredictorMode::AvgTlT,
            PredictorMode::AvgTTr,
            PredictorMode::AvgAvgLTlAvgTTr,
            PredictorMode::Select,
            PredictorMode::ClampAddSubtractFull,
            PredictorMode::ClampAddSubtractHalf,
        ]
    }

    /// Convert from u8 mode value.
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => PredictorMode::Black,
            1 => PredictorMode::Left,
            2 => PredictorMode::Top,
            3 => PredictorMode::TopRight,
            4 => PredictorMode::TopLeft,
            5 => PredictorMode::AvgAvgLtrT,
            6 => PredictorMode::AvgLTl,
            7 => PredictorMode::AvgLT,
            8 => PredictorMode::AvgTlT,
            9 => PredictorMode::AvgTTr,
            10 => PredictorMode::AvgAvgLTlAvgTTr,
            11 => PredictorMode::Select,
            12 => PredictorMode::ClampAddSubtractFull,
            _ => PredictorMode::ClampAddSubtractHalf,
        }
    }
}

/// Apply subtract green transform in place.
/// R -= G, B -= G
pub fn apply_subtract_green(pixels: &mut [u32]) {
    for pixel in pixels.iter_mut() {
        let a = argb_alpha(*pixel);
        let r = argb_red(*pixel);
        let g = argb_green(*pixel);
        let b = argb_blue(*pixel);

        let new_r = r.wrapping_sub(g);
        let new_b = b.wrapping_sub(g);

        *pixel = make_argb(a, new_r, g, new_b);
    }
}

/// Predict a pixel using the given mode and neighbors.
/// Must match the decoder's predictor functions exactly for lossless round-tripping.
#[inline]
fn predict(mode: PredictorMode, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        PredictorMode::Black => 0xff000000,
        PredictorMode::Left => left,
        PredictorMode::Top => top,
        PredictorMode::TopRight => top_right,
        PredictorMode::TopLeft => top_left,
        PredictorMode::AvgAvgLtrT => average2(average2(left, top_right), top),
        PredictorMode::AvgLTl => average2(left, top_left),
        PredictorMode::AvgLT => average2(left, top),
        PredictorMode::AvgTlT => average2(top_left, top),
        PredictorMode::AvgTTr => average2(top, top_right),
        PredictorMode::AvgAvgLTlAvgTTr => {
            average2(average2(left, top_left), average2(top, top_right))
        }
        PredictorMode::Select => select(left, top, top_left),
        PredictorMode::ClampAddSubtractFull => clamp_add_subtract_full(left, top, top_left),
        PredictorMode::ClampAddSubtractHalf => clamp_add_subtract_half(left, top, top_left),
    }
}

/// Average two pixels component-wise.
#[inline]
fn average2(a: u32, b: u32) -> u32 {
    let aa = argb_alpha(a) as u16 + argb_alpha(b) as u16;
    let ar = argb_red(a) as u16 + argb_red(b) as u16;
    let ag = argb_green(a) as u16 + argb_green(b) as u16;
    let ab = argb_blue(a) as u16 + argb_blue(b) as u16;
    make_argb(
        (aa / 2) as u8,
        (ar / 2) as u8,
        (ag / 2) as u8,
        (ab / 2) as u8,
    )
}

/// Select predictor: choose left or top based on gradient.
/// Matches decoder's tie-breaking: when distances are equal, returns top.
#[inline]
fn select(left: u32, top: u32, top_left: u32) -> u32 {
    // predict_left = |T - TL|: distance from prediction (L+T-TL) to L
    let pa = (argb_alpha(top) as i16 - argb_alpha(top_left) as i16).unsigned_abs();
    let pr = (argb_red(top) as i16 - argb_red(top_left) as i16).unsigned_abs();
    let pg = (argb_green(top) as i16 - argb_green(top_left) as i16).unsigned_abs();
    let pb = (argb_blue(top) as i16 - argb_blue(top_left) as i16).unsigned_abs();
    let predict_left = pa + pr + pg + pb;

    // predict_top = |L - TL|: distance from prediction (L+T-TL) to T
    let pa = (argb_alpha(left) as i16 - argb_alpha(top_left) as i16).unsigned_abs();
    let pr = (argb_red(left) as i16 - argb_red(top_left) as i16).unsigned_abs();
    let pg = (argb_green(left) as i16 - argb_green(top_left) as i16).unsigned_abs();
    let pb = (argb_blue(left) as i16 - argb_blue(top_left) as i16).unsigned_abs();
    let predict_top = pa + pr + pg + pb;

    // If prediction is closer to left, choose left; otherwise choose top.
    // On ties (predict_left == predict_top), choose top (matching decoder).
    if predict_left < predict_top {
        left
    } else {
        top
    }
}

/// Clamp a value to [0, 255].
#[inline]
fn clamp(val: i16) -> u8 {
    val.clamp(0, 255) as u8
}

/// ClampAddSubtractFull: left + top - top_left, clamped.
#[inline]
fn clamp_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let a = clamp(argb_alpha(left) as i16 + argb_alpha(top) as i16 - argb_alpha(top_left) as i16);
    let r = clamp(argb_red(left) as i16 + argb_red(top) as i16 - argb_red(top_left) as i16);
    let g = clamp(argb_green(left) as i16 + argb_green(top) as i16 - argb_green(top_left) as i16);
    let b = clamp(argb_blue(left) as i16 + argb_blue(top) as i16 - argb_blue(top_left) as i16);
    make_argb(a, r, g, b)
}

/// ClampAddSubtractHalf: clamp(avg + (avg - top_left) / 2) per component,
/// where avg = (left + top) / 2.
#[inline]
fn clamp_add_subtract_half(left: u32, top: u32, top_left: u32) -> u32 {
    let avg_a = (argb_alpha(left) as i16 + argb_alpha(top) as i16) / 2;
    let avg_r = (argb_red(left) as i16 + argb_red(top) as i16) / 2;
    let avg_g = (argb_green(left) as i16 + argb_green(top) as i16) / 2;
    let avg_b = (argb_blue(left) as i16 + argb_blue(top) as i16) / 2;
    let a = (avg_a + (avg_a - argb_alpha(top_left) as i16) / 2).clamp(0, 255) as u8;
    let r = (avg_r + (avg_r - argb_red(top_left) as i16) / 2).clamp(0, 255) as u8;
    let g = (avg_g + (avg_g - argb_green(top_left) as i16) / 2).clamp(0, 255) as u8;
    let b = (avg_b + (avg_b - argb_blue(top_left) as i16) / 2).clamp(0, 255) as u8;
    make_argb(a, r, g, b)
}

/// Compute residual (pixel - prediction) with wrapping.
#[inline]
fn residual(pixel: u32, pred: u32) -> u32 {
    let a = argb_alpha(pixel).wrapping_sub(argb_alpha(pred));
    let r = argb_red(pixel).wrapping_sub(argb_red(pred));
    let g = argb_green(pixel).wrapping_sub(argb_green(pred));
    let b = argb_blue(pixel).wrapping_sub(argb_blue(pred));
    make_argb(a, r, g, b)
}

/// Apply predictor transform.
/// Returns the predictor mode data (subsampled image of modes).
///
/// Border handling matches the VP8L decoder:
/// - (0,0): Black predictor (0xff000000)
/// - Row 0, x>0: Left predictor (fixed, regardless of mode)
/// - Col 0, y>0: Top predictor (fixed, regardless of mode)
/// - Interior (y>0, x>0): mode from predictor_data
///
/// When `max_quantization <= 1` (exact lossless), pixels are processed in
/// reverse order so neighbors remain as original values.
///
/// When `max_quantization > 1` (near-lossless), pixels are processed in
/// forward order. Each pixel's residual is quantized, and the source pixel
/// is updated with the reconstructed value (predict + quantized_residual)
/// so subsequent predictions use the reconstructed data. This matches
/// libwebp's CopyImageWithPrediction / GetResidual flow.
pub fn apply_predictor_transform(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    size_bits: u8,
    max_quantization: u32,
    used_subtract_green: bool,
) -> Vec<u32> {
    let block_size = 1usize << size_bits;
    let blocks_x = subsample_size(width as u32, size_bits) as usize;
    let blocks_y = subsample_size(height as u32, size_bits) as usize;

    // Choose best predictor for each block using entropy-based scoring.
    // Accumulated histogram tracks global residual distribution across all tiles.
    let mut predictor_data = vec![0u32; blocks_x * blocks_y];
    let mut accumulated = [[0u32; 256]; 4];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            // Get left and above modes for spatial coherence bias.
            // 0xff means no neighbor (edge tiles), ensuring no bias match.
            let left_mode = if bx > 0 {
                argb_green(predictor_data[by * blocks_x + bx - 1])
            } else {
                0xff
            };
            let above_mode = if by > 0 {
                argb_green(predictor_data[(by - 1) * blocks_x + bx])
            } else {
                0xff
            };

            let best_mode = choose_best_predictor(
                pixels,
                width,
                height,
                bx,
                by,
                block_size,
                &mut accumulated,
                left_mode,
                above_mode,
            );
            // Store mode in green channel (as per spec), alpha=0xff (matching libwebp's ARGB_BLACK | mode<<8)
            predictor_data[by * blocks_x + bx] = 0xff000000 | ((best_mode as u32) << 8);
        }
    }

    if max_quantization > 1 {
        // Forward-order processing with near-lossless residual quantization.
        // Precompute max_diffs for all interior rows from ORIGINAL pixel data
        // before any modifications.
        let mut all_max_diffs = vec![0u8; width * height];
        for y in 1..height.saturating_sub(1) {
            super::near_lossless::max_diffs_for_row(
                pixels,
                width,
                y,
                &mut all_max_diffs[y * width..],
                used_subtract_green,
            );
        }

        // Process forward, storing residuals separately since we update source
        // pixels with reconstructed values for subsequent predictions.
        let mut residuals = vec![0u32; width * height];

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pred = if y == 0 && x == 0 {
                    0xff000000
                } else if y == 0 {
                    // First row: Left predictor (uses possibly-reconstructed left)
                    pixels[x - 1]
                } else if x == 0 {
                    // First column: Top predictor (uses possibly-reconstructed top)
                    pixels[(y - 1) * width]
                } else {
                    let bx = x >> size_bits;
                    let by = y >> size_bits;
                    let mode =
                        PredictorMode::from_u8(argb_green(predictor_data[by * blocks_x + bx]));
                    let left = pixels[y * width + x - 1];
                    let top = pixels[(y - 1) * width + x];
                    let top_left = pixels[(y - 1) * width + x - 1];
                    let top_right = if x + 1 < width {
                        pixels[(y - 1) * width + x + 1]
                    } else {
                        pixels[y * width]
                    };
                    predict(mode, left, top, top_left, top_right)
                };

                // Matching libwebp's GetResidual conditions for skipping quantization:
                // border pixels, mode 0 (Black), first/last row, first/last column
                let mode_val = if y > 0 && x > 0 {
                    argb_green(predictor_data[(y >> size_bits) * blocks_x + (x >> size_bits)])
                } else {
                    0 // border uses fixed predictors, treat as mode 0
                };

                if mode_val == 0 || y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                    // Exact residual for borders and mode 0
                    residuals[idx] = residual(pixels[idx], pred);
                    // No source update needed (reconstructed == original for exact residual)
                } else {
                    let max_diff = all_max_diffs[idx];
                    let (res, recon) = super::near_lossless::near_lossless_residual(
                        pixels[idx],
                        pred,
                        max_quantization,
                        max_diff,
                        used_subtract_green,
                    );
                    residuals[idx] = res;
                    // Update source pixel with reconstructed value for subsequent predictions
                    pixels[idx] = recon;
                }
            }
        }

        // Copy residuals to output
        pixels.copy_from_slice(&residuals);
    } else {
        // Exact lossless: reverse-order processing (neighbors stay original).
        for y in (0..height).rev() {
            for x in (0..width).rev() {
                let pred = if y == 0 && x == 0 {
                    // Top-left corner: Black
                    0xff000000
                } else if y == 0 {
                    // First row: Left predictor
                    pixels[x - 1]
                } else if x == 0 {
                    // First column: Top predictor
                    pixels[(y - 1) * width]
                } else {
                    // Interior: use block's predictor mode
                    let bx = x >> size_bits;
                    let by = y >> size_bits;
                    let mode =
                        PredictorMode::from_u8(argb_green(predictor_data[by * blocks_x + bx]));
                    let left = pixels[y * width + x - 1];
                    let top = pixels[(y - 1) * width + x];
                    let top_left = pixels[(y - 1) * width + x - 1];
                    // At right edge (x == width-1), top_right wraps to the first
                    // pixel of the current row. This matches the decoder's memory
                    // layout where image_data[i - width*4 + 4] wraps to row y's start.
                    let top_right = if x + 1 < width {
                        pixels[(y - 1) * width + x + 1]
                    } else {
                        pixels[y * width]
                    };
                    predict(mode, left, top, top_left, top_right)
                };
                pixels[y * width + x] = residual(pixels[y * width + x], pred);
            }
        }
    }

    predictor_data
}

/// Choose the best predictor mode for a block using entropy-based scoring.
/// Matches libwebp's GetBestPredictorForTile + PredictionCostSpatialHistogram.
///
/// Scores each mode by building per-channel residual histograms and evaluating:
/// 1. Combined Shannon entropy (tile + accumulated global distribution)
/// 2. Prediction cost bias (favoring residuals near 0)
/// 3. Spatial coherence bonus (favoring same mode as left/above neighbors)
///
/// Updates `accumulated` with the best mode's histogram for subsequent tiles.
fn choose_best_predictor(
    pixels: &[u32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    block_size: usize,
    accumulated: &mut [[u32; 256]; 4],
    left_mode: u8,
    above_mode: u8,
) -> PredictorMode {
    let x_start = bx * block_size;
    let y_start = by * block_size;
    let x_end = (x_start + block_size).min(width);
    let y_end = (y_start + block_size).min(height);

    // Effective start: skip row 0 and col 0 (they use fixed predictors)
    let x_eff = x_start.max(1);
    let y_eff = y_start.max(1);

    // If no interior pixels to score, default to Black (mode 0, matching libwebp)
    if x_eff >= x_end || y_eff >= y_end {
        return PredictorMode::Black;
    }

    let mut best_mode = PredictorMode::Black;
    let mut best_cost = i64::MAX;
    let mut best_histo = [[0u32; 256]; 4];

    for mode in PredictorMode::all() {
        let mut tile_histo = [[0u32; 256]; 4];

        for y in y_eff..y_end {
            for x in x_eff..x_end {
                let pixel = pixels[y * width + x];
                let left = pixels[y * width + x - 1];
                let top = pixels[(y - 1) * width + x];
                let top_left = pixels[(y - 1) * width + x - 1];
                // At right edge, top_right wraps to first pixel of current row
                // (matching decoder's memory layout behavior).
                let top_right = if x + 1 < width {
                    pixels[(y - 1) * width + x + 1]
                } else {
                    pixels[y * width]
                };

                let pred = predict(mode, left, top, top_left, top_right);
                let res = residual(pixel, pred);
                update_histo(&mut tile_histo, res);
            }
        }

        let cost = prediction_cost_spatial_histogram(
            accumulated,
            &tile_histo,
            mode as u8,
            left_mode,
            above_mode,
        );

        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
            best_histo = tile_histo;
        }
    }

    // Update accumulated histogram with best mode's histogram
    for c in 0..4 {
        for i in 0..256 {
            accumulated[c][i] += best_histo[c][i];
        }
    }

    best_mode
}

/// Cross-color transform multipliers for a tile.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossColorMultipliers {
    pub green_to_red: u8,
    pub green_to_blue: u8,
    pub red_to_blue: u8,
}

/// Fixed-point precision for entropy calculations (matching libwebp).
const LOG_2_PRECISION_BITS: u32 = 23;

/// Spatial predictor bias: reward tiles that keep the same mode as neighbors.
const SPATIAL_PREDICTOR_BIAS: i64 = 15i64 << LOG_2_PRECISION_BITS;

/// ColorTransformDelta: 3.5-bit fixed point multiply (matching libwebp).
#[inline]
fn color_transform_delta(color_pred: i8, color: i8) -> i32 {
    (color_pred as i32 * color as i32) >> 5
}

/// Compute v * log2(v) in fixed-point (matching libwebp's VP8LFastSLog2).
/// Returns 0 for v == 0.
#[inline]
fn fast_slog2(v: u32) -> u64 {
    if v == 0 {
        return 0;
    }
    let vf = v as f64;
    (vf * libm::log2(vf) * (1u64 << LOG_2_PRECISION_BITS) as f64) as u64
}

/// Combined Shannon entropy of distributions X and X+Y (matching libwebp).
/// Returns SLog2(sumX) + SLog2(sumXY) - Σ(SLog2(x_i) + SLog2(x_i+y_i))
fn combined_shannon_entropy(x: &[u32; 256], y: &[u32; 256]) -> u64 {
    let mut retval: u64 = 0;
    let mut sum_x: u32 = 0;
    let mut sum_xy: u32 = 0;
    for i in 0..256 {
        let xi = x[i];
        if xi != 0 {
            let xy = xi + y[i];
            sum_x += xi;
            retval += fast_slog2(xi);
            sum_xy += xy;
            retval += fast_slog2(xy);
        } else if y[i] != 0 {
            sum_xy += y[i];
            retval += fast_slog2(y[i]);
        }
    }
    fast_slog2(sum_x) + fast_slog2(sum_xy) - retval
}

/// Rounding division matching libwebp's DivRound.
#[inline]
fn div_round(a: i64, b: i64) -> i64 {
    if (a < 0) == (b < 0) {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

/// Prediction cost bias favoring values near 0 (matching libwebp).
fn prediction_cost_bias(counts: &[u32; 256], weight_0: u64, mut exp_val: u64) -> i64 {
    let significant_symbols = 256 >> 4; // 16
    let exp_decay_factor: u64 = 6; // scaling factor 1/10
    let mut bits = (weight_0 * counts[0] as u64) << LOG_2_PRECISION_BITS;
    exp_val <<= LOG_2_PRECISION_BITS;
    for i in 1..significant_symbols {
        bits += div_round(
            (exp_val * (counts[i] as u64 + counts[256 - i] as u64)) as i64,
            100,
        ) as u64;
        exp_val = div_round((exp_decay_factor * exp_val) as i64, 10) as u64;
    }
    -div_round(bits as i64, 10)
}

/// Update a 4-channel residual histogram with an ARGB residual value.
/// Layout: [alpha_0..255, red_0..255, green_0..255, blue_0..255].
#[inline]
fn update_histo(histo: &mut [[u32; 256]; 4], argb: u32) {
    histo[0][(argb >> 24) as usize] += 1;
    histo[1][((argb >> 16) & 0xff) as usize] += 1;
    histo[2][((argb >> 8) & 0xff) as usize] += 1;
    histo[3][(argb & 0xff) as usize] += 1;
}

/// Score a predictor mode using entropy + bias + spatial coherence.
/// Matching libwebp's PredictionCostSpatialHistogram.
fn prediction_cost_spatial_histogram(
    accumulated: &[[u32; 256]; 4],
    tile: &[[u32; 256]; 4],
    mode: u8,
    left_mode: u8,
    above_mode: u8,
) -> i64 {
    let mut retval: i64 = 0;
    for i in 0..4 {
        const K_EXP_VALUE: u64 = 94;
        retval += prediction_cost_bias(&tile[i], 1, K_EXP_VALUE);
        retval += combined_shannon_entropy(&tile[i], &accumulated[i]) as i64;
    }
    if mode == left_mode {
        retval -= SPATIAL_PREDICTOR_BIAS;
    }
    if mode == above_mode {
        retval -= SPATIAL_PREDICTOR_BIAS;
    }
    retval
}

/// Cross-color prediction cost (matching libwebp's PredictionCostCrossColor).
fn prediction_cost_cross_color(accumulated: &[u32; 256], counts: &[u32; 256]) -> i64 {
    const K_EXP_VALUE: u64 = 240;
    combined_shannon_entropy(counts, accumulated) as i64
        + prediction_cost_bias(counts, 3, K_EXP_VALUE)
}

/// Collect histogram of transformed red values for a tile.
fn collect_color_red_transforms(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    green_to_red: u8,
    histo: &mut [u32; 256],
) {
    for y in start_y..end_y {
        for x in start_x..end_x {
            let pixel = argb[y * width + x];
            let green = (pixel >> 8) as u8 as i8;
            let mut new_red = (pixel >> 16) as i32;
            new_red -= color_transform_delta(green_to_red as i8, green);
            histo[(new_red & 0xff) as usize] += 1;
        }
    }
}

/// Collect histogram of transformed blue values for a tile.
fn collect_color_blue_transforms(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    green_to_blue: u8,
    red_to_blue: u8,
    histo: &mut [u32; 256],
) {
    for y in start_y..end_y {
        for x in start_x..end_x {
            let pixel = argb[y * width + x];
            let green = (pixel >> 8) as u8 as i8;
            let red = (pixel >> 16) as u8 as i8;
            let mut new_blue = pixel as i32 & 0xff;
            new_blue -= color_transform_delta(green_to_blue as i8, green);
            new_blue -= color_transform_delta(red_to_blue as i8, red);
            histo[(new_blue & 0xff) as usize] += 1;
        }
    }
}

/// Cost of a green_to_red transform on a tile (matching libwebp).
fn get_prediction_cost_red(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    prev_x: &CrossColorMultipliers,
    prev_y: &CrossColorMultipliers,
    green_to_red: u8,
    accumulated_red_histo: &[u32; 256],
) -> i64 {
    let mut histo = [0u32; 256];
    collect_color_red_transforms(
        argb,
        width,
        start_x,
        start_y,
        end_x,
        end_y,
        green_to_red,
        &mut histo,
    );
    let mut cur_diff = prediction_cost_cross_color(accumulated_red_histo, &histo);
    if green_to_red == prev_x.green_to_red {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if green_to_red == prev_y.green_to_red {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if green_to_red == 0 {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    cur_diff
}

/// Cost of green_to_blue + red_to_blue transform on a tile (matching libwebp).
fn get_prediction_cost_blue(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    prev_x: &CrossColorMultipliers,
    prev_y: &CrossColorMultipliers,
    green_to_blue: u8,
    red_to_blue: u8,
    accumulated_blue_histo: &[u32; 256],
) -> i64 {
    let mut histo = [0u32; 256];
    collect_color_blue_transforms(
        argb,
        width,
        start_x,
        start_y,
        end_x,
        end_y,
        green_to_blue,
        red_to_blue,
        &mut histo,
    );
    let mut cur_diff = prediction_cost_cross_color(accumulated_blue_histo, &histo);
    if green_to_blue == prev_x.green_to_blue {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if green_to_blue == prev_y.green_to_blue {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if red_to_blue == prev_x.red_to_blue {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if red_to_blue == prev_y.red_to_blue {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if green_to_blue == 0 {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    if red_to_blue == 0 {
        cur_diff -= 3i64 << LOG_2_PRECISION_BITS;
    }
    cur_diff
}

/// Coarse-to-fine 1D search for best green_to_red (matching libwebp).
fn get_best_green_to_red(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    prev_x: &CrossColorMultipliers,
    prev_y: &CrossColorMultipliers,
    quality: u8,
    accumulated_red_histo: &[u32; 256],
) -> u8 {
    let max_iters = 4 + ((7 * quality as i32) >> 8); // range [4..6]
    let mut green_to_red_best: i32 = 0;
    let mut best_diff = get_prediction_cost_red(
        argb,
        width,
        start_x,
        start_y,
        end_x,
        end_y,
        prev_x,
        prev_y,
        0,
        accumulated_red_histo,
    );

    for iter in 0..max_iters {
        let delta: i32 = 32 >> iter;
        for &offset in &[-delta, delta] {
            let green_to_red_cur = offset + green_to_red_best;
            let cur_diff = get_prediction_cost_red(
                argb,
                width,
                start_x,
                start_y,
                end_x,
                end_y,
                prev_x,
                prev_y,
                green_to_red_cur as u8,
                accumulated_red_histo,
            );
            if cur_diff < best_diff {
                best_diff = cur_diff;
                green_to_red_best = green_to_red_cur;
            }
        }
    }
    (green_to_red_best & 0xff) as u8
}

/// Coarse-to-fine 2D 8-axis search for best green_to_blue + red_to_blue (matching libwebp).
fn get_best_green_red_to_blue(
    argb: &[u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    prev_x: &CrossColorMultipliers,
    prev_y: &CrossColorMultipliers,
    quality: u8,
    accumulated_blue_histo: &[u32; 256],
) -> (u8, u8) {
    const OFFSETS: [[i8; 2]; 8] = [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ];
    const DELTA_LUT: [i32; 7] = [16, 16, 8, 4, 2, 2, 2];
    let iters = if quality < 25 {
        1
    } else if quality > 50 {
        7
    } else {
        4
    };

    let mut green_to_blue_best: i32 = 0;
    let mut red_to_blue_best: i32 = 0;
    let mut best_diff = get_prediction_cost_blue(
        argb,
        width,
        start_x,
        start_y,
        end_x,
        end_y,
        prev_x,
        prev_y,
        0,
        0,
        accumulated_blue_histo,
    );

    for iter in 0..iters {
        let delta = DELTA_LUT[iter as usize];
        for (axis, offset) in OFFSETS.iter().enumerate() {
            let green_to_blue_cur = offset[0] as i32 * delta + green_to_blue_best;
            let red_to_blue_cur = offset[1] as i32 * delta + red_to_blue_best;
            let cur_diff = get_prediction_cost_blue(
                argb,
                width,
                start_x,
                start_y,
                end_x,
                end_y,
                prev_x,
                prev_y,
                green_to_blue_cur as u8,
                red_to_blue_cur as u8,
                accumulated_blue_histo,
            );
            if cur_diff < best_diff {
                best_diff = cur_diff;
                green_to_blue_best = green_to_blue_cur;
                red_to_blue_best = red_to_blue_cur;
            }
            // For low quality, only axis-aligned (first 4 directions)
            if quality < 25 && axis == 3 {
                break;
            }
        }
        if delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0 {
            break;
        }
    }

    (
        (green_to_blue_best & 0xff) as u8,
        (red_to_blue_best & 0xff) as u8,
    )
}

/// Apply cross-color transform to decorrelate color channels.
///
/// Uses libwebp's entropy-based coarse-to-fine search with accumulated
/// histograms and spatial consistency bonuses.
///
/// Forward transform (encoding):
///   new_R = (R - green_to_red * G_signed / 32) & 0xFF
///   new_B = (B - green_to_blue * G_signed / 32 - red_to_blue * R_orig_signed / 32) & 0xFF
///
/// The multiplier data is stored as a sub-image where:
///   blue channel  = green_to_red
///   green channel = green_to_blue
///   red channel   = red_to_blue
///   alpha         = 0xFF
pub fn apply_cross_color_transform(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    transform_bits: u8,
    quality: u8,
) -> Vec<u32> {
    let block_size = 1usize << transform_bits;
    let tiles_x = subsample_size(width as u32, transform_bits) as usize;
    let tiles_y = subsample_size(height as u32, transform_bits) as usize;

    let mut transform_data = vec![0u32; tiles_x * tiles_y];
    let mut accumulated_red_histo = [0u32; 256];
    let mut accumulated_blue_histo = [0u32; 256];

    let mut prev_x = CrossColorMultipliers::default();
    let mut prev_y = CrossColorMultipliers::default();

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let start_x = tx * block_size;
            let start_y = ty * block_size;
            let end_x = (start_x + block_size).min(width);
            let end_y = (start_y + block_size).min(height);

            // Get prev_y from tile above
            if ty != 0 {
                let above_code = transform_data[(ty - 1) * tiles_x + tx];
                prev_y.green_to_red = above_code as u8;
                prev_y.green_to_blue = (above_code >> 8) as u8;
                prev_y.red_to_blue = (above_code >> 16) as u8;
            }

            // Find best green_to_red
            let best_g2r = get_best_green_to_red(
                pixels,
                width,
                start_x,
                start_y,
                end_x,
                end_y,
                &prev_x,
                &prev_y,
                quality,
                &accumulated_red_histo,
            );

            // Find best green_to_blue + red_to_blue
            let (best_g2b, best_r2b) = get_best_green_red_to_blue(
                pixels,
                width,
                start_x,
                start_y,
                end_x,
                end_y,
                &prev_x,
                &prev_y,
                quality,
                &accumulated_blue_histo,
            );

            prev_x = CrossColorMultipliers {
                green_to_red: best_g2r,
                green_to_blue: best_g2b,
                red_to_blue: best_r2b,
            };

            // Store as sub-image pixel (MultipliersToColorCode)
            let color_code = 0xFF000000u32
                | ((best_r2b as u32) << 16)
                | ((best_g2b as u32) << 8)
                | (best_g2r as u32);
            transform_data[ty * tiles_x + tx] = color_code;

            // Apply forward transform to this tile
            apply_cross_color_tile(pixels, width, start_x, start_y, end_x, end_y, &prev_x);

            // Gather accumulated histogram data (matching libwebp's skip logic)
            for y in start_y..end_y {
                for x in start_x..end_x {
                    let ix = y * width + x;
                    let pix = pixels[ix];
                    // Skip repeated pixels (handled by backward references)
                    if ix >= 2 && pix == pixels[ix - 2] && pix == pixels[ix - 1] {
                        continue;
                    }
                    if ix >= width + 2
                        && pixels[ix - 2] == pixels[ix - width - 2]
                        && pixels[ix - 1] == pixels[ix - width - 1]
                        && pix == pixels[ix - width]
                    {
                        continue;
                    }
                    accumulated_red_histo[((pix >> 16) & 0xff) as usize] += 1;
                    accumulated_blue_histo[(pix & 0xff) as usize] += 1;
                }
            }
        }
    }

    transform_data
}

/// Apply cross-color forward transform to a single tile (matching libwebp's VP8LTransformColor_C).
#[inline]
fn apply_cross_color_tile(
    pixels: &mut [u32],
    width: usize,
    start_x: usize,
    start_y: usize,
    end_x: usize,
    end_y: usize,
    m: &CrossColorMultipliers,
) {
    let g2r = m.green_to_red as i8;
    let g2b = m.green_to_blue as i8;
    let r2b = m.red_to_blue as i8;

    for y in start_y..end_y {
        for x in start_x..end_x {
            let idx = y * width + x;
            let argb = pixels[idx];
            let green = (argb >> 8) as u8 as i8;
            let red = (argb >> 16) as u8 as i8;
            let mut new_red = (red as i32) & 0xff;
            let mut new_blue = argb as i32 & 0xff;
            new_red -= color_transform_delta(g2r, green);
            new_red &= 0xff;
            new_blue -= color_transform_delta(g2b, green);
            new_blue -= color_transform_delta(r2b, red);
            new_blue &= 0xff;
            pixels[idx] = (argb & 0xff00ff00u32) | ((new_red as u32) << 16) | (new_blue as u32);
        }
    }
}

/// Simple predictor transform using only vertical prediction.
/// Used for quick encoding at lower quality levels.
pub fn apply_simple_predictor(pixels: &mut [u32], width: usize, height: usize) {
    // Process from bottom-right to top-left
    for y in (1..height).rev() {
        for x in (0..width).rev() {
            let idx = y * width + x;
            let top = pixels[(y - 1) * width + x];
            pixels[idx] = residual(pixels[idx], top);
        }
    }
    // First row: subtract left neighbor
    for x in (1..width).rev() {
        let left = pixels[x - 1];
        pixels[x] = residual(pixels[x], left);
    }
    // Top-left corner: subtract 0xff000000 (black with full alpha)
    pixels[0] = residual(pixels[0], 0xff000000);
}

/// Color indexing (palette) transform data.
pub struct ColorIndexTransform {
    pub palette: Vec<u32>,
}

/// Compute the xbits value for pixel packing based on palette size.
/// Returns: 3 (8 pixels/word) for <=2 colors, 2 (4/word) for <=4,
/// 1 (2/word) for <=16, 0 (no packing) otherwise.
pub fn palette_xbits(palette_size: usize) -> u8 {
    if palette_size <= 2 {
        3
    } else if palette_size <= 4 {
        2
    } else if palette_size <= 16 {
        1
    } else {
        0
    }
}

/// Compute distance between two palette colors for sorting.
/// Component distances are weighted 9x for RGB, 1x for alpha.
/// Matches libwebp's PaletteColorDistance.
fn palette_color_distance(col1: u32, col2: u32) -> u32 {
    let diff = sub_pixels_color(col1, col2);
    let component_distance = |v: u8| -> u32 { if v <= 128 { v as u32 } else { 256 - v as u32 } };
    let score = component_distance(diff as u8)
        + component_distance((diff >> 8) as u8)
        + component_distance((diff >> 16) as u8);
    score * 9 + component_distance((diff >> 24) as u8)
}

/// Subtract two pixels component-wise (wrapping). Same as sub_pixels below
/// but works on u32 directly.
fn sub_pixels_color(a: u32, b: u32) -> u32 {
    let aa = (a >> 24).wrapping_sub(b >> 24) & 0xff;
    let ar = ((a >> 16) & 0xff).wrapping_sub((b >> 16) & 0xff) & 0xff;
    let ag = ((a >> 8) & 0xff).wrapping_sub((b >> 8) & 0xff) & 0xff;
    let ab = (a & 0xff).wrapping_sub(b & 0xff) & 0xff;
    (aa << 24) | (ar << 16) | (ag << 8) | ab
}

/// Sort palette to minimize L1 deltas between consecutive entries.
/// Matches libwebp's PaletteSortMinimizeDeltas. Greedy nearest-neighbor ordering
/// weighted toward RGB channels over alpha.
fn palette_sort_minimize_deltas(palette_sorted: &[u32]) -> Vec<u32> {
    let num_colors = palette_sorted.len();
    let mut palette: Vec<u32> = palette_sorted.to_vec();

    // Check if palette already has monotonic deltas (no benefit to reorder)
    if !palette_has_non_monotonous_deltas(&palette) {
        return palette;
    }

    // For palettes > 17 colors, move 0 to the end (matching libwebp)
    if num_colors > 17 && palette[0] == 0 {
        let last = num_colors - 1;
        palette.swap(0, last);
    }

    // Greedy nearest-neighbor reordering (selection sort by distance)
    let mut predict = 0u32;
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_colors {
        let mut best_ix = i;
        let mut best_score = u32::MAX;
        for k in i..num_colors {
            let cur_score = palette_color_distance(palette[k], predict);
            if best_score > cur_score {
                best_score = cur_score;
                best_ix = k;
            }
        }
        palette.swap(best_ix, i);
        predict = palette[i];
    }

    palette
}

/// Check if palette has non-monotonous deltas (both positive and negative
/// changes in the same component). If monotonous, no benefit to reorder.
/// Matches libwebp's PaletteHasNonMonotonousDeltas.
fn palette_has_non_monotonous_deltas(palette: &[u32]) -> bool {
    let mut predict = 0u32;
    let mut sign_found = 0u8;
    for &color in palette {
        let diff = sub_pixels_color(color, predict);
        let rd = ((diff >> 16) & 0xff) as u8;
        let gd = ((diff >> 8) & 0xff) as u8;
        let bd = (diff & 0xff) as u8;
        if rd != 0 {
            sign_found |= if rd < 0x80 { 1 } else { 2 };
        }
        if gd != 0 {
            sign_found |= if gd < 0x80 { 8 } else { 16 };
        }
        if bd != 0 {
            sign_found |= if bd < 0x80 { 64 } else { 128 };
        }
        predict = color;
    }
    (sign_found & (sign_found << 1)) != 0
}

/// Bundle multiple palette indices into packed pixels.
/// For palette_size <=2: 8 indices per pixel (1 bit each)
/// For palette_size <=4: 4 indices per pixel (2 bits each)
/// For palette_size <=16: 2 indices per pixel (4 bits each)
/// Matches libwebp's VP8LBundleColorMap_C.
pub fn bundle_color_map(pixels: &[u32], width: usize, xbits: u8) -> Vec<u32> {
    if xbits == 0 {
        // No packing needed - just ensure format is 0xff000000 | (idx << 8)
        return pixels
            .iter()
            .map(|&p| 0xff000000 | ((argb_green(p) as u32) << 8))
            .collect();
    }

    let bit_depth = 1u32 << (3 - xbits);
    let mask = (1usize << xbits) - 1;
    let packed_width = subsample_size(width as u32, xbits) as usize;
    let height = pixels.len() / width;
    let mut dst = Vec::with_capacity(packed_width * height);

    for y in 0..height {
        let row_start = y * width;
        let mut code = 0xff000000u32;
        for x in 0..width {
            let xsub = x & mask;
            if xsub == 0 {
                code = 0xff000000;
            }
            let idx = argb_green(pixels[row_start + x]) as u32;
            code |= idx << (8 + bit_depth * xsub as u32);
            if xsub == mask || x == width - 1 {
                dst.push(code);
            }
        }
    }

    dst
}

impl ColorIndexTransform {
    /// Try to build a color index transform.
    /// Returns None if image has more than 256 colors.
    /// Palette is sorted to minimize deltas (default strategy).
    pub fn try_build(pixels: &[u32]) -> Option<Self> {
        Self::try_build_with_sorting(pixels, true)
    }

    /// Try to build a color index transform with a specific sorting strategy.
    /// `minimize_delta`: true = greedy nearest-neighbor, false = lexicographic.
    /// Returns None if image has more than 256 colors.
    pub fn try_build_with_sorting(pixels: &[u32], minimize_delta: bool) -> Option<Self> {
        let mut seen = alloc::collections::BTreeSet::new();

        for &pixel in pixels {
            seen.insert(pixel);
            if seen.len() > 256 {
                return None;
            }
        }

        // BTreeSet iterates in sorted order (lexicographic on u32 = ARGB)
        let palette_sorted: Vec<u32> = seen.into_iter().collect();

        let palette = if minimize_delta {
            // Sort to minimize deltas for better delta encoding (matches libwebp kMinimizeDelta)
            palette_sort_minimize_deltas(&palette_sorted)
        } else {
            // Use lexicographic order directly
            palette_sorted
        };

        Some(Self { palette })
    }

    /// Build from a pre-sorted palette (for multi-config testing).
    pub fn from_palette(palette: Vec<u32>) -> Self {
        Self { palette }
    }

    /// Get xbits value for pixel packing.
    pub fn xbits(&self) -> u8 {
        palette_xbits(self.palette.len())
    }

    /// Apply the transform: convert ARGB to palette indices in green channel.
    pub fn apply(&self, pixels: &mut [u32]) {
        // Build reverse lookup using a hash map for fast lookups
        let mut lookup = alloc::collections::BTreeMap::new();
        for (i, &color) in self.palette.iter().enumerate() {
            lookup.insert(color, i as u8);
        }

        for pixel in pixels.iter_mut() {
            let idx = lookup[pixel];
            // Store index in green channel (A=0xff, R=0, G=idx, B=0)
            *pixel = make_argb(255, 0, idx, 0);
        }
    }

    /// Apply the transform and bundle pixels into packed format.
    /// Returns the packed pixel buffer and the new (packed) width.
    pub fn apply_and_bundle(&self, pixels: &mut [u32], width: usize) -> (Vec<u32>, usize) {
        // First apply: convert ARGB to palette indices
        self.apply(pixels);

        let xbits = self.xbits();
        if xbits == 0 {
            // No packing needed
            return (pixels.to_vec(), width);
        }

        let packed_width = subsample_size(width as u32, xbits) as usize;
        let packed = bundle_color_map(pixels, width, xbits);
        (packed, packed_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtract_green() {
        let mut pixels = vec![make_argb(255, 100, 50, 150)];
        apply_subtract_green(&mut pixels);

        let p = pixels[0];
        assert_eq!(argb_alpha(p), 255);
        assert_eq!(argb_red(p), 50); // 100 - 50
        assert_eq!(argb_green(p), 50); // unchanged
        assert_eq!(argb_blue(p), 100); // 150 - 50
    }

    #[test]
    fn test_average2() {
        let a = make_argb(100, 100, 100, 100);
        let b = make_argb(200, 200, 200, 200);
        let avg = average2(a, b);

        assert_eq!(argb_alpha(avg), 150);
        assert_eq!(argb_red(avg), 150);
        assert_eq!(argb_green(avg), 150);
        assert_eq!(argb_blue(avg), 150);
    }

    #[test]
    fn test_residual() {
        let pixel = make_argb(100, 50, 80, 200);
        let pred = make_argb(90, 60, 70, 150);
        let res = residual(pixel, pred);

        assert_eq!(argb_alpha(res), 10); // 100 - 90
        assert_eq!(argb_red(res), 246); // 50 - 60 = -10 = 246 (wrapping)
        assert_eq!(argb_green(res), 10); // 80 - 70
        assert_eq!(argb_blue(res), 50); // 200 - 150
    }

    #[test]
    fn test_color_index_small_palette() {
        let pixels = vec![
            make_argb(255, 255, 0, 0), // Red
            make_argb(255, 0, 255, 0), // Green
            make_argb(255, 255, 0, 0), // Red again
        ];

        let transform = ColorIndexTransform::try_build(&pixels).unwrap();
        assert_eq!(transform.palette.len(), 2);
        assert_eq!(transform.xbits(), 3); // Can pack 8 per pixel
    }

    #[test]
    fn test_color_index_too_many_colors() {
        // Create 257 unique colors - use different channels to ensure uniqueness
        let pixels: Vec<u32> = (0..257)
            .map(|i| make_argb(255, (i % 256) as u8, (i / 256) as u8, 0))
            .collect();
        let transform = ColorIndexTransform::try_build(&pixels);
        assert!(transform.is_none());
    }

    #[test]
    fn test_palette_bundle_2_colors() {
        // 2-color palette: xbits=3, 8 pixels per packed pixel
        let indices: Vec<u32> = (0..16).map(|i| make_argb(255, 0, i & 1, 0)).collect();
        let bundled = bundle_color_map(&indices, 16, 3);
        // 16 pixels / 8 per packed = 2 packed pixels
        assert_eq!(bundled.len(), 2);
        // First packed pixel: indices 0,1,0,1,0,1,0,1
        // Each in 1-bit slots at positions 8,9,10,11,12,13,14,15
        let first = bundled[0];
        assert_eq!(first & 0xff000000, 0xff000000); // alpha=255
    }

    #[test]
    fn test_palette_bundle_16_colors() {
        // 16-color palette: xbits=1, 2 pixels per packed pixel
        let indices: Vec<u32> = (0..8)
            .map(|i| make_argb(255, 0, (i * 3) & 0xf, 0))
            .collect();
        let bundled = bundle_color_map(&indices, 8, 1);
        // 8 pixels / 2 per packed = 4 packed pixels
        assert_eq!(bundled.len(), 4);
    }

    #[test]
    fn test_palette_sort_minimize_deltas() {
        // Create palette with colors that should be reordered
        let sorted = vec![
            make_argb(255, 0, 0, 0),       // black
            make_argb(255, 128, 128, 128), // gray
            make_argb(255, 255, 0, 0),     // red
            make_argb(255, 255, 255, 255), // white
        ];
        let reordered = palette_sort_minimize_deltas(&sorted);
        // Should start from 0 (closest to predict=0), then pick nearest neighbors
        assert_eq!(reordered[0], make_argb(255, 0, 0, 0)); // black closest to 0
        assert_eq!(reordered.len(), 4);
    }
}
