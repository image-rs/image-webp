//! Segment-based quantization analysis for VP8 encoding.
//!
//! This module provides macroblock-level analysis for adaptive quantization,
//! computing complexity metrics and assigning segments using k-means clustering.
//!
//! ## Module organization
//!
//! - `classifier`: Content type detection for auto-preset selection
//! - `histogram`: DCT coefficient histogram collection (SIMD candidate)
//! - `iterator`: Macroblock iteration for analysis pass
//! - `prediction`: Intra prediction for analysis (SIMD candidate)
//! - `segment`: K-means segment assignment and quantization
//!
//! ## Analysis pipeline
//!
//! 1. For each macroblock, compute DCT histograms using best I16/UV prediction
//! 2. Derive alpha (compressibility) from histogram shape
//! 3. Cluster macroblocks into 1-4 segments using k-means on alpha
//! 4. Compute per-segment quantization based on alpha centers
//!
//! Ported from libwebp src/enc/analysis_enc.c

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

// Submodules
pub mod classifier;
pub mod histogram;
pub mod iterator;
pub mod prediction;
pub mod segment;

use alloc::vec::Vec;

// Re-exports
pub use classifier::{
    ClassifierDiag, ContentType, classify_image_type, classify_image_type_diag,
    content_type_to_tuning,
};
pub use histogram::{collect_histogram_bps, forward_dct_4x4};
pub use iterator::AnalysisIterator;
pub use segment::{assign_segments_kmeans, compute_segment_quant, smooth_segment_map};

//------------------------------------------------------------------------------
// Constants

/// Maximum alpha value for macroblock complexity
pub const MAX_ALPHA: u8 = 255;

/// Alpha scale factor (from libwebp: ALPHA_SCALE = 2 * MAX_ALPHA = 510)
const ALPHA_SCALE: u32 = 2 * MAX_ALPHA as u32;

/// Maximum coefficient threshold for histogram binning (from libwebp)
const MAX_COEFF_THRESH: usize = 31;

/// Number of segments
pub const NUM_SEGMENTS: usize = 4;

/// Number of intra16 modes to test during analysis (DC=0, TM=1)
const MAX_INTRA16_MODE: usize = 2;

/// Number of UV modes to test during analysis (DC=0, TM=1)
const MAX_UV_MODE: usize = 2;

/// Bytes per stride in libwebp's work buffers
const BPS: usize = 32;

/// Offset to Y plane in work buffer
const Y_OFF_ENC: usize = 0;

/// Offset to U plane in work buffer
const U_OFF_ENC: usize = 16;

/// Offset to V plane in work buffer
const V_OFF_ENC: usize = 24;

/// Offsets to each 4x4 block within a BPS-strided buffer
/// Ported from libwebp's VP8DspScan
/// The 0 * BPS expressions are intentional to match libwebp's pattern
#[allow(clippy::erasing_op, clippy::identity_op)]
const VP8_DSP_SCAN: [usize; 24] = [
    // Luma (16 blocks)
    0 + 0 * BPS,
    4 + 0 * BPS,
    8 + 0 * BPS,
    12 + 0 * BPS,
    0 + 4 * BPS,
    4 + 4 * BPS,
    8 + 4 * BPS,
    12 + 4 * BPS,
    0 + 8 * BPS,
    4 + 8 * BPS,
    8 + 8 * BPS,
    12 + 8 * BPS,
    0 + 12 * BPS,
    4 + 12 * BPS,
    8 + 12 * BPS,
    12 + 12 * BPS,
    // U (4 blocks)
    0 + 0 * BPS,
    4 + 0 * BPS,
    0 + 4 * BPS,
    4 + 4 * BPS,
    // V (4 blocks, offset by 8 from U)
    8 + 0 * BPS,
    12 + 0 * BPS,
    8 + 4 * BPS,
    12 + 4 * BPS,
];

/// Mode offsets for I16 predictions in yuv_p buffer
/// Ported from libwebp's VP8I16ModeOffsets
const I16DC16: usize = 0;
const I16TM16: usize = I16DC16 + 16;
#[allow(clippy::identity_op)] // 1 * 16 matches libwebp's pattern (row * block_size * stride)
const I16VE16: usize = 1 * 16 * BPS;
const I16HE16: usize = I16VE16 + 16;

const VP8_I16_MODE_OFFSETS: [usize; 4] = [I16DC16, I16TM16, I16VE16, I16HE16];

/// Mode offsets for chroma predictions in yuv_p buffer
/// Ported from libwebp's VP8UVModeOffsets
const C8DC8: usize = 2 * 16 * BPS;
const C8TM8: usize = C8DC8 + 16;
const C8VE8: usize = 2 * 16 * BPS + 8 * BPS;
const C8HE8: usize = C8VE8 + 16;

const VP8_UV_MODE_OFFSETS: [usize; 4] = [C8DC8, C8TM8, C8VE8, C8HE8];

/// Size of prediction buffer (I16 + Chroma + I4 preds)
const PRED_SIZE_ENC: usize = 32 * BPS + 16 * BPS + 8 * BPS;

/// Size of YUV work buffer
const YUV_SIZE_ENC: usize = BPS * 16;

//------------------------------------------------------------------------------
// DCT histogram

/// DCT histogram result for alpha calculation
/// Ported from libwebp's VP8Histogram struct
#[derive(Default, Clone, Copy)]
pub struct DctHistogram {
    /// Maximum count in any histogram bin
    pub max_value: u32,
    /// Highest bin index with non-zero count
    pub last_non_zero: usize,
}

impl DctHistogram {
    /// Initialize histogram with default values (from libwebp's InitHistogram)
    pub fn new() -> Self {
        Self {
            max_value: 0,
            last_non_zero: 1,
        }
    }

    /// Compute histogram data from a distribution array
    /// Ported from libwebp's VP8SetHistogramData
    pub fn from_distribution(distribution: &[u32; MAX_COEFF_THRESH + 1]) -> Self {
        let mut max_value: u32 = 0;
        let mut last_non_zero: usize = 1;

        for (k, &count) in distribution.iter().enumerate() {
            if count > 0 {
                if count > max_value {
                    max_value = count;
                }
                last_non_zero = k;
            }
        }

        Self {
            max_value,
            last_non_zero,
        }
    }

    /// Get alpha value from histogram.
    /// Ported from libwebp's GetAlpha (analysis_enc.c).
    ///
    /// Alpha measures the "compressibility" of the block:
    /// - High alpha = many non-zero coefficients relative to max (harder)
    /// - Low alpha (0) = few non-zero coefficients (easier to compress)
    ///
    /// Formula: `ALPHA_SCALE * last_non_zero / max_value`
    pub fn get_alpha(&self) -> i32 {
        if self.max_value <= 1 {
            return 0;
        }
        let value = ALPHA_SCALE * self.last_non_zero as u32 / self.max_value;
        value as i32
    }
}

//------------------------------------------------------------------------------
// Alpha finalization

/// Finalize alpha value by inverting and clipping
/// Ported from libwebp's FinalAlphaValue
#[inline]
fn final_alpha_value(alpha: i32) -> u8 {
    let alpha = MAX_ALPHA as i32 - alpha;
    alpha.clamp(0, MAX_ALPHA as i32) as u8
}

//------------------------------------------------------------------------------
// Main analysis functions

/// Analyze a single macroblock and return (mixed alpha, raw uv_alpha)
/// Ported from libwebp's MBAnalyze
///
/// Returns:
/// - mixed alpha: finalized alpha value combining luma and chroma (for segment assignment)
/// - raw uv_alpha: unfinalized chroma alpha (for UV quant delta computation)
///
/// When `method >= 4` and `sns_strength > 0`, blends in a masking-based alpha
/// from local variance to improve adaptive quantization for textured regions.
pub fn analyze_macroblock(it: &mut AnalysisIterator, method: u8, sns_strength: u8) -> (u8, i32) {
    let (best_alpha, _best_mode) = it.analyze_best_intra16_mode();
    let (best_uv_alpha, _uv_mode) = it.analyze_best_uv_mode();

    // Final susceptibility mix
    let alpha = (3 * best_alpha + best_uv_alpha + 2) >> 2;

    // Blend with masking alpha for perceptual adaptive quantization.
    // Only when method >= 4 AND sns_strength > 0 (user hasn't disabled SNS).
    let alpha = if method >= 4 && sns_strength > 0 {
        let masking = crate::encoder::psy::compute_masking_alpha(&it.yuv_in[Y_OFF_ENC..], BPS);
        crate::encoder::psy::blend_masking_alpha(alpha, masking, method)
    } else {
        alpha
    };

    // Finalize: invert and clip
    (final_alpha_value(alpha), best_uv_alpha)
}

/// Analysis result containing per-MB alphas and global statistics.
pub struct AnalysisResult {
    /// Per-macroblock alpha values (finalized, for segment assignment)
    pub mb_alphas: Vec<u8>,
    /// Histogram of alpha values
    pub alpha_histogram: [u32; 256],
    /// Average UV alpha across all macroblocks (raw, for UV quant delta)
    /// Range is typically ~30 (bad) to ~100 (ok to decimate UV more).
    pub uv_alpha_avg: i32,
}

/// Run full analysis pass on image and return alpha histogram + per-MB alphas
/// Ported from libwebp's VP8EncAnalyze / DoSegmentsJob
///
/// `method` controls whether perceptual masking is blended into alpha values.
#[allow(clippy::too_many_arguments)]
pub fn analyze_image(
    y_src: &[u8],
    u_src: &[u8],
    v_src: &[u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    method: u8,
    sns_strength: u8,
) -> AnalysisResult {
    let mut it = AnalysisIterator::new(width, height);
    it.reset();

    let total_mbs = it.mb_w * it.mb_h;
    let mut mb_alphas = Vec::with_capacity(total_mbs);
    let mut alpha_histogram = [0u32; 256];
    let mut uv_alpha_sum: i64 = 0;

    loop {
        it.import(y_src, u_src, v_src, y_stride, uv_stride);

        let (alpha, uv_alpha) = analyze_macroblock(&mut it, method, sns_strength);
        mb_alphas.push(alpha);
        alpha_histogram[alpha as usize] += 1;
        uv_alpha_sum += uv_alpha as i64;

        if !it.next() {
            break;
        }
    }

    // Compute average UV alpha (matching libwebp's enc->uv_alpha = sum / total_mb)
    let uv_alpha_avg = if total_mbs > 0 {
        (uv_alpha_sum / total_mbs as i64) as i32
    } else {
        64 // MID_ALPHA default
    };

    AnalysisResult {
        mb_alphas,
        alpha_histogram,
        uv_alpha_avg,
    }
}
