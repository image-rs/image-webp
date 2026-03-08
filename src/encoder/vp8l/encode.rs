//! Main VP8L encoder implementation.

use alloc::vec;
use alloc::vec::Vec;

use super::backward_refs::{
    apply_cache_to_refs, get_backward_references, get_backward_references_with_palette,
    strip_cache_from_refs,
};
use super::bitwriter::BitWriter;
use super::color_cache::ColorCache;
use super::histogram::{
    distance_code_extra_bits, distance_code_to_prefix, length_code_extra_bits, length_to_code,
};
use super::huffman::{
    HuffmanCode, build_huffman_codes, build_huffman_lengths, write_huffman_tree,
    write_single_entry_tree,
};
use super::meta_huffman::{MetaHuffmanInfo, build_meta_huffman, build_single_histogram};
use super::transforms::{
    ColorIndexTransform, apply_cross_color_transform, apply_predictor_transform,
    apply_subtract_green,
};
use super::types::{
    NUM_LENGTH_CODES, NUM_LITERAL_CODES, PixOrCopy, Vp8lConfig, argb_alpha, argb_blue, argb_green,
    argb_red, make_argb, subsample_size,
};

use super::entropy::vp8l_bits_entropy;
use crate::encoder::api::EncodeError;

/// Encode an image using VP8L lossless compression.
pub fn encode_vp8l(
    pixels: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    config: &Vp8lConfig,
    stop: &dyn enough::Stop,
) -> Result<Vec<u8>, EncodeError> {
    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodeError::InvalidDimensions);
    }

    let w = width as usize;
    let h = height as usize;
    let expected_len = w * h * if has_alpha { 4 } else { 3 };
    if pixels.len() != expected_len {
        return Err(EncodeError::InvalidBufferSize(alloc::format!(
            "expected {} bytes, got {}",
            expected_len,
            pixels.len()
        )));
    }

    // Convert RGBA/RGB bytes to packed ARGB u32 array.
    // On little-endian, ARGB u32 has byte order [B,G,R,A] = BGRA,
    // so we use garb's SIMD-accelerated rgba_to_bgra / rgb_to_bgra.
    let mut argb: Vec<u32> = alloc::vec![0u32; w * h];
    #[cfg(target_endian = "little")]
    {
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut argb);
        if has_alpha {
            garb::bytes::rgba_to_bgra(pixels, dst_bytes)
                .expect("encode_vp8l: validated buffer sizes");
        } else {
            garb::bytes::rgb_to_bgra(pixels, dst_bytes)
                .expect("encode_vp8l: validated buffer sizes");
        }
    }
    #[cfg(target_endian = "big")]
    {
        if has_alpha {
            for (dst, p) in argb.iter_mut().zip(pixels.chunks_exact(4)) {
                *dst = make_argb(p[3], p[0], p[1], p[2]);
            }
        } else {
            for (dst, p) in argb.iter_mut().zip(pixels.chunks_exact(3)) {
                *dst = make_argb(255, p[0], p[1], p[2]);
            }
        }
    }

    // Note: BGR/BGRA callers must swap R↔B before calling encode_vp8l.
    // The encoder API handles this in the pixel expansion step.

    // Encode with the full pipeline
    encode_argb(&mut argb, w, h, has_alpha, config, stop)
}

/// Result of entropy analysis for transform selection.
/// Matches libwebp's EntropyIx enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntropyMode {
    /// No transforms (raw pixel values).
    Direct,
    /// Predictor transform only.
    Spatial,
    /// Subtract green transform only.
    SubGreen,
    /// Subtract green + predictor transforms.
    SpatialSubGreen,
    /// Palette transform.
    Palette,
}

struct AnalyzeResult {
    mode: EntropyMode,
    red_and_blue_always_zero: bool,
}

/// Transform combination to try during multi-config encoding.
/// Extends EntropyMode with PaletteAndSpatial variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrunchMode {
    Direct,
    Spatial,
    SubGreen,
    SpatialSubGreen,
    Palette,
    PaletteAndSpatial,
}

impl CrunchMode {
    fn from_entropy_mode(mode: EntropyMode) -> Self {
        match mode {
            EntropyMode::Direct => CrunchMode::Direct,
            EntropyMode::Spatial => CrunchMode::Spatial,
            EntropyMode::SubGreen => CrunchMode::SubGreen,
            EntropyMode::SpatialSubGreen => CrunchMode::SpatialSubGreen,
            EntropyMode::Palette => CrunchMode::Palette,
        }
    }
}

/// Palette sorting strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteSorting {
    /// Lexicographic sort (BTreeSet natural order).
    Lexicographic,
    /// Greedy nearest-neighbor to minimize deltas (current default).
    MinimizeDelta,
}

/// Configuration for one encoding attempt in multi-config mode.
#[derive(Debug, Clone, Copy)]
struct CrunchConfig {
    mode: CrunchMode,
    palette_sorting: PaletteSorting,
}

/// Generate the list of configurations to try based on method and quality.
///
/// - Methods 0-4: single config (entropy-based best guess)
/// - Method 5, quality >= 75: best guess + PaletteAndSpatial variant if palette available
/// - Method 6, quality == 100: brute-force all viable modes
fn generate_crunch_configs(
    analysis: &AnalyzeResult,
    palette_available: bool,
    method: u8,
    quality: u8,
) -> Vec<CrunchConfig> {
    let best_mode = CrunchMode::from_entropy_mode(analysis.mode);
    let default_sorting = PaletteSorting::MinimizeDelta;

    // Method 6, quality 100: brute-force all viable modes
    if method >= 6 && quality >= 100 {
        let mut configs = Vec::with_capacity(8);
        // All 4 non-palette modes
        for &mode in &[
            CrunchMode::Direct,
            CrunchMode::Spatial,
            CrunchMode::SubGreen,
            CrunchMode::SpatialSubGreen,
        ] {
            configs.push(CrunchConfig {
                mode,
                palette_sorting: default_sorting,
            });
        }
        // Both palette modes if palette available, each with 2 sorting strategies
        if palette_available {
            for &mode in &[CrunchMode::Palette, CrunchMode::PaletteAndSpatial] {
                for &sorting in &[PaletteSorting::Lexicographic, PaletteSorting::MinimizeDelta] {
                    configs.push(CrunchConfig {
                        mode,
                        palette_sorting: sorting,
                    });
                }
            }
        }
        return configs;
    }

    // Method 5, quality >= 75: best guess + variant
    if method >= 5 && quality >= 75 && palette_available {
        let mut configs = vec![CrunchConfig {
            mode: best_mode,
            palette_sorting: default_sorting,
        }];
        // If best is Palette, also try PaletteAndSpatial (and vice versa)
        let variant = match best_mode {
            CrunchMode::Palette => Some(CrunchMode::PaletteAndSpatial),
            CrunchMode::PaletteAndSpatial => Some(CrunchMode::Palette),
            _ => None,
        };
        if let Some(v) = variant {
            configs.push(CrunchConfig {
                mode: v,
                palette_sorting: default_sorting,
            });
        }
        return configs;
    }

    // Default: single config
    vec![CrunchConfig {
        mode: best_mode,
        palette_sorting: default_sorting,
    }]
}

/// Analyze image entropy to determine the best transform combination.
///
/// Matches libwebp's AnalyzeEntropy: builds 13 histograms (256 buckets each)
/// in a single pass, estimates Shannon entropy for 5 transform modes, and picks
/// the mode with lowest total entropy.
fn analyze_entropy(
    argb: &[u32],
    width: usize,
    height: usize,
    use_palette: bool,
    palette_size: usize,
    transform_bits: u8,
) -> AnalyzeResult {
    // Short-circuit for small palettes (always win due to pixel packing)
    if use_palette && palette_size <= 16 {
        return AnalyzeResult {
            mode: EntropyMode::Palette,
            red_and_blue_always_zero: true,
        };
    }

    // 13 histogram categories matching libwebp's HistoIx enum
    const HISTO_ALPHA: usize = 0;
    const HISTO_RED: usize = 1;
    const HISTO_GREEN: usize = 2;
    const HISTO_BLUE: usize = 3;
    const HISTO_ALPHA_PRED: usize = 4;
    const HISTO_RED_PRED: usize = 5;
    const HISTO_GREEN_PRED: usize = 6;
    const HISTO_BLUE_PRED: usize = 7;
    const HISTO_RED_SUB_GREEN: usize = 8;
    const HISTO_BLUE_SUB_GREEN: usize = 9;
    const HISTO_RED_PRED_SUB_GREEN: usize = 10;
    const HISTO_BLUE_PRED_SUB_GREEN: usize = 11;
    const HISTO_PALETTE: usize = 12;
    const NUM_HISTO: usize = 13;
    const NUM_BUCKETS: usize = 256;

    let mut histo = [[0u32; NUM_BUCKETS]; NUM_HISTO];

    // Scan all pixels, building histograms
    let mut pix_prev = argb[0]; // Skip first pixel (no left neighbor)
    for y in 0..height {
        for x in 0..width {
            let pix = argb[y * width + x];
            let pix_diff = sub_pixels(pix, pix_prev);
            pix_prev = pix;

            // Skip pixels identical to left neighbor or above neighbor
            if pix_diff == 0 || (y > 0 && pix == argb[(y - 1) * width + x]) {
                continue;
            }

            let a = (pix >> 24) as usize;
            let r = ((pix >> 16) & 0xff) as usize;
            let g = ((pix >> 8) & 0xff) as usize;
            let b = (pix & 0xff) as usize;

            // kDirect: raw channel values
            histo[HISTO_ALPHA][a] += 1;
            histo[HISTO_RED][r] += 1;
            histo[HISTO_GREEN][g] += 1;
            histo[HISTO_BLUE][b] += 1;

            // kSpatial: per-channel prediction difference
            let da = ((pix_diff >> 24) & 0xff) as usize;
            let dr = ((pix_diff >> 16) & 0xff) as usize;
            let dg = ((pix_diff >> 8) & 0xff) as usize;
            let db = (pix_diff & 0xff) as usize;
            histo[HISTO_ALPHA_PRED][da] += 1;
            histo[HISTO_RED_PRED][dr] += 1;
            histo[HISTO_GREEN_PRED][dg] += 1;
            histo[HISTO_BLUE_PRED][db] += 1;

            // kSubGreen: R-G and B-G (mod 256)
            histo[HISTO_RED_SUB_GREEN][(r.wrapping_sub(g)) & 0xff] += 1;
            histo[HISTO_BLUE_SUB_GREEN][(b.wrapping_sub(g)) & 0xff] += 1;

            // kSpatialSubGreen: (R-G) and (B-G) of prediction difference
            histo[HISTO_RED_PRED_SUB_GREEN][(dr.wrapping_sub(dg)) & 0xff] += 1;
            histo[HISTO_BLUE_PRED_SUB_GREEN][(db.wrapping_sub(dg)) & 0xff] += 1;

            // kPalette: hash of raw pixel
            let hash = hash_pix(pix);
            histo[HISTO_PALETTE][hash as usize] += 1;
        }
    }

    // Add one zero to predicted histograms (matching libwebp's correction).
    // The pix_diff == 0 skip removes zeros too aggressively.
    histo[HISTO_RED_PRED_SUB_GREEN][0] += 1;
    histo[HISTO_BLUE_PRED_SUB_GREEN][0] += 1;
    histo[HISTO_RED_PRED][0] += 1;
    histo[HISTO_GREEN_PRED][0] += 1;
    histo[HISTO_BLUE_PRED][0] += 1;
    histo[HISTO_ALPHA_PRED][0] += 1;

    // Compute VP8LBitsEntropy for each histogram
    let mut entropy_comp = [0u64; NUM_HISTO];
    for j in 0..NUM_HISTO {
        entropy_comp[j] = vp8l_bits_entropy(&histo[j]);
    }

    // Combine into 5 mode entropies
    let mut entropy = [0u64; 5];
    entropy[0] = entropy_comp[HISTO_ALPHA]
        + entropy_comp[HISTO_RED]
        + entropy_comp[HISTO_GREEN]
        + entropy_comp[HISTO_BLUE]; // kDirect
    entropy[1] = entropy_comp[HISTO_ALPHA_PRED]
        + entropy_comp[HISTO_RED_PRED]
        + entropy_comp[HISTO_GREEN_PRED]
        + entropy_comp[HISTO_BLUE_PRED]; // kSpatial
    entropy[2] = entropy_comp[HISTO_ALPHA]
        + entropy_comp[HISTO_RED_SUB_GREEN]
        + entropy_comp[HISTO_GREEN]
        + entropy_comp[HISTO_BLUE_SUB_GREEN]; // kSubGreen
    entropy[3] = entropy_comp[HISTO_ALPHA_PRED]
        + entropy_comp[HISTO_RED_PRED_SUB_GREEN]
        + entropy_comp[HISTO_GREEN_PRED]
        + entropy_comp[HISTO_BLUE_PRED_SUB_GREEN]; // kSpatialSubGreen
    entropy[4] = entropy_comp[HISTO_PALETTE]; // kPalette

    // Add transform overhead costs (in LOG_2_PRECISION_BITS fixed-point)
    // kLog2Table[14] = log2(14) * (1 << 23): cost of storing 14 predictor modes per tile
    const LOG2_14_FIXED: u64 = 31938408;
    // kLog2Table[24] = log2(24) * (1 << 23): cost of storing color transform per tile
    const LOG2_24_FIXED: u64 = 38461453;

    let tiles_w = subsample_size(width as u32, transform_bits) as u64;
    let tiles_h = subsample_size(height as u32, transform_bits) as u64;
    entropy[1] += tiles_w * tiles_h * LOG2_14_FIXED; // kSpatial overhead
    entropy[3] += tiles_w * tiles_h * LOG2_24_FIXED; // kSpatialSubGreen overhead

    // Palette overhead: palette_size * 8 bits estimated per entry
    const LOG_2_PRECISION_BITS: u32 = 23;
    if use_palette {
        entropy[4] += (palette_size as u64 * 8) << LOG_2_PRECISION_BITS;
    }

    // Find minimum entropy mode
    let last_mode = if use_palette { 5 } else { 4 };
    let mut best_idx = 0usize;
    for k in 1..last_mode {
        if entropy[k] < entropy[best_idx] {
            best_idx = k;
        }
    }

    let mode = match best_idx {
        0 => EntropyMode::Direct,
        1 => EntropyMode::Spatial,
        2 => EntropyMode::SubGreen,
        3 => EntropyMode::SpatialSubGreen,
        4 => EntropyMode::Palette,
        _ => unreachable!(),
    };

    // Check if red and blue are always zero in the chosen mode's histograms
    let (red_histo_idx, blue_histo_idx) = match best_idx {
        0 => (HISTO_RED, HISTO_BLUE),
        1 => (HISTO_RED_PRED, HISTO_BLUE_PRED),
        2 => (HISTO_RED_SUB_GREEN, HISTO_BLUE_SUB_GREEN),
        3 => (HISTO_RED_PRED_SUB_GREEN, HISTO_BLUE_PRED_SUB_GREEN),
        4 => (HISTO_RED, HISTO_BLUE), // palette mode
        _ => unreachable!(),
    };

    let red_and_blue_always_zero =
        (1..NUM_BUCKETS).all(|i| (histo[red_histo_idx][i] | histo[blue_histo_idx][i]) == 0);

    AnalyzeResult {
        mode,
        red_and_blue_always_zero,
    }
}

/// Hash a pixel for palette entropy approximation (matching libwebp's HashPix).
#[inline]
fn hash_pix(pix: u32) -> u8 {
    let v = (pix as u64).wrapping_add((pix >> 19) as u64);
    let h = v.wrapping_mul(0x39c5fba7u64);
    ((h & 0xffffffff) >> 24) as u8
}

/// Encode ARGB pixels (internal). Orchestrates multi-config testing for high
/// quality/method settings, keeping the smallest output.
pub(crate) fn encode_argb(
    argb: &mut [u32],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &Vp8lConfig,
    stop: &dyn enough::Stop,
) -> Result<Vec<u8>, EncodeError> {
    // Determine if palette is available (needed for config generation)
    let palette_candidate = if config.use_palette {
        can_use_palette(argb)
    } else {
        false
    };

    // Run entropy analysis (used for single-config best guess)
    let palette_size_est = if palette_candidate {
        // Quick count for entropy estimation (don't build full transform yet)
        ColorIndexTransform::try_build(argb)
            .as_ref()
            .map(|p| p.palette.len())
            .unwrap_or(0)
    } else {
        0
    };
    let histo_bits_est =
        get_histo_bits_palette(width, height, config.quality.method, palette_candidate);
    let transform_bits_est = get_transform_bits(config.quality.method, histo_bits_est);
    let transform_bits_clamped = clamp_bits(
        width,
        height,
        transform_bits_est,
        MIN_TRANSFORM_BITS,
        MAX_TRANSFORM_BITS,
        MAX_PREDICTOR_IMAGE_SIZE,
    );
    let analysis = analyze_entropy(
        argb,
        width,
        height,
        palette_candidate,
        palette_size_est,
        transform_bits_clamped,
    );

    // Generate configs to try
    let configs = generate_crunch_configs(
        &analysis,
        palette_candidate,
        config.quality.method,
        config.quality.quality,
    );

    if configs.len() == 1 {
        // Single config: encode in place (no clone needed)
        encode_argb_single_config(argb, width, height, has_alpha, config, &configs[0], stop)
    } else {
        // Multi-config: clone original pixels, try each config, keep smallest
        let original = argb.to_vec();
        let mut best_output: Option<Vec<u8>> = None;

        for crunch_config in &configs {
            stop.check()?;
            argb.copy_from_slice(&original);
            match encode_argb_single_config(
                argb,
                width,
                height,
                has_alpha,
                config,
                crunch_config,
                stop,
            ) {
                Ok(output) => {
                    let keep = match &best_output {
                        None => true,
                        Some(best) => output.len() < best.len(),
                    };
                    if keep {
                        best_output = Some(output);
                    }
                }
                Err(_) => {
                    // Skip configs that fail (e.g., palette mode on non-palette image)
                }
            }
        }

        best_output.ok_or(EncodeError::InvalidBufferSize(alloc::format!(
            "all {} crunch configs failed",
            configs.len()
        )))
    }
}

/// Encode ARGB pixels with a specific transform configuration.
#[allow(clippy::too_many_arguments)]
fn encode_argb_single_config(
    argb: &mut [u32],
    width: usize,
    height: usize,
    has_alpha: bool,
    config: &Vp8lConfig,
    crunch: &CrunchConfig,
    stop: &dyn enough::Stop,
) -> Result<Vec<u8>, EncodeError> {
    let mut writer = BitWriter::with_capacity(width * height / 2);

    // Write VP8L signature
    writer.write_bits(0x2f, 8);

    // Write dimensions
    writer.write_bits((width - 1) as u64, 14);
    writer.write_bits((height - 1) as u64, 14);

    // Alpha hint and version
    writer.write_bit(has_alpha);
    writer.write_bits(0, 3); // version 0

    // Derive transform flags from CrunchMode
    let use_palette = matches!(
        crunch.mode,
        CrunchMode::Palette | CrunchMode::PaletteAndSpatial
    );
    let use_predictor = config.use_predictor
        && matches!(
            crunch.mode,
            CrunchMode::Spatial | CrunchMode::SpatialSubGreen | CrunchMode::PaletteAndSpatial
        );
    let use_subtract_green = config.use_subtract_green
        && matches!(
            crunch.mode,
            CrunchMode::SubGreen | CrunchMode::SpatialSubGreen
        );
    // Cross-color only when: not palette, R/B not always zero, and predictor is on.
    // For PaletteAndSpatial, cross-color is never used (matching libwebp).
    let use_cross_color = config.use_cross_color
        && !use_palette
        && use_predictor
        && !matches!(crunch.mode, CrunchMode::PaletteAndSpatial);

    // Build palette transform if needed
    let minimize_delta = matches!(crunch.palette_sorting, PaletteSorting::MinimizeDelta);
    let palette_transform = if use_palette {
        ColorIndexTransform::try_build_with_sorting(argb, minimize_delta)
    } else {
        None
    };

    // Apply near-lossless preprocessing if enabled.
    // Matching libwebp: only when not palette and not predictor mode.
    if config.near_lossless < 100 && !use_palette && !use_predictor {
        super::near_lossless::apply_near_lossless(argb, width, height, config.near_lossless);
    }

    // Apply transforms and signal them in bitstream.
    // Order: SubtractGreen → Predictor → CrossColor (matching libwebp).
    // For PaletteAndSpatial: ColorIndexing first, then Predictor.
    // The decoder reverses this order (LIFO).

    // For non-palette modes: SubtractGreen → Predictor → CrossColor
    if !use_palette {
        // Subtract green transform (applied first, decorrelates R/B from G)
        if use_subtract_green {
            writer.write_bit(true); // transform present
            writer.write_bits(2, 2); // subtract green = 2
            apply_subtract_green(argb);
        }

        // Predictor transform (applied second, on subtract-green'd image)
        if use_predictor {
            write_predictor_transform(&mut writer, argb, width, height, config, use_subtract_green);
        }

        // Cross-color transform (applied third, on predictor residuals)
        if use_cross_color {
            write_cross_color_transform(&mut writer, argb, width, height, config);
        }
    }

    // Color indexing transform
    // After palette, we may operate on a packed (narrower) image.
    let mut packed_buf: Option<Vec<u32>> = None;
    let mut enc_width = width; // Effective width for LZ77/Huffman (may be packed)

    if let Some(ref palette) = palette_transform {
        let palette_size = palette.palette.len();
        // Matching libwebp: if last element is 0 and palette_size > 17, omit it
        // (decoder fills with zeros). Only for non-packed palettes.
        let encoded_palette_size = if palette.palette[palette_size - 1] == 0 && palette_size > 17 {
            palette_size - 1
        } else {
            palette_size
        };

        writer.write_bit(true); // transform present
        writer.write_bits(3, 2); // color indexing = 3
        writer.write_bits((encoded_palette_size - 1) as u64, 8);

        // Write palette with delta encoding using encode_image_no_huffman
        let mut tmp_palette = Vec::with_capacity(encoded_palette_size);
        tmp_palette.push(palette.palette[0]);
        for w in palette.palette[..encoded_palette_size].windows(2) {
            tmp_palette.push(sub_pixels(w[1], w[0]));
        }
        encode_image_no_huffman(&mut writer, &tmp_palette, encoded_palette_size, 1, 20);

        // Apply transform and bundle pixels
        let xbits = palette.xbits();
        palette.apply(argb);

        if xbits > 0 {
            let packed_width = subsample_size(width as u32, xbits) as usize;
            let packed = super::transforms::bundle_color_map(argb, width, xbits);
            enc_width = packed_width;
            packed_buf = Some(packed);
        }

        // PaletteAndSpatial: apply predictor on the palette-indexed image
        if matches!(crunch.mode, CrunchMode::PaletteAndSpatial) {
            // Predictor operates on the indexed (possibly packed) image
            let pred_argb: &mut [u32] = if let Some(ref mut buf) = packed_buf {
                buf
            } else {
                argb
            };
            write_predictor_transform(
                &mut writer,
                pred_argb,
                enc_width,
                pred_argb.len() / enc_width,
                config,
                false, // no subtract green with palette
            );
        }
    }

    // No more transforms
    writer.write_bit(false);

    // Use packed buffer if palette bundling was applied
    let enc_argb: &[u32] = if let Some(ref buf) = packed_buf {
        buf
    } else {
        argb
    };
    let enc_height = enc_argb.len() / enc_width;

    // Build backward references (determines optimal cache_bits via entropy estimation)
    let cache_bits_max = match config.cache_bits {
        None => {
            // Auto-detect optimal cache size
            if let Some(ref pt) = palette_transform {
                // Cap cache bits for palette images (matching libwebp)
                let ps = pt.palette.len();
                if ps < (1 << 10) {
                    let log2_floor = 31u32.saturating_sub(ps.leading_zeros());
                    (log2_floor + 1).min(10) as u8
                } else {
                    10
                }
            } else {
                10 // Auto-detect: try all sizes up to MAX_COLOR_CACHE_BITS
            }
        }
        Some(bits) => bits.min(10),
    };
    let enc_palette_size = palette_transform
        .as_ref()
        .map(|pt| pt.palette.len())
        .unwrap_or(0);
    stop.check()?;

    let (refs, cache_bits) = if enc_palette_size > 0 {
        get_backward_references_with_palette(
            enc_argb,
            enc_width,
            enc_height,
            config.quality.quality,
            cache_bits_max,
            enc_palette_size,
        )
    } else {
        get_backward_references(
            enc_argb,
            enc_width,
            enc_height,
            config.quality.quality,
            cache_bits_max,
        )
    };

    // Encode image data from backward references.
    // If auto-detecting cache, try multiple cache sizes and keep the smallest output.
    // Entropy estimation can overshoot the optimal cache_bits because it uses a global
    // histogram, but actual encoding uses per-tile meta-Huffman codes where larger cache
    // expands the literal alphabet (280→408+ entries), increasing tree overhead per tile.
    let is_palette = palette_transform.is_some();
    let auto_cache = config.cache_bits.is_none();

    if auto_cache && cache_bits > 0 {
        // Strip cache from refs to get the base (cache-free) token sequence.
        // We'll re-apply cache at each candidate size.
        let mut base_refs = refs;
        strip_cache_from_refs(enc_argb, &mut base_refs);

        let mut best_output: Option<Vec<u8>> = None;

        // Try all cache sizes from 0 to the entropy-selected value.
        // Sizes above the entropy estimate are unlikely to be better
        // (entropy correctly identifies the direction, just overshoots).
        for cb in 0..=cache_bits {
            stop.check()?;
            let mut trial_refs = base_refs.clone();
            if cb > 0 {
                apply_cache_to_refs(enc_argb, cb, &mut trial_refs);
            }
            let output = encode_image_data(
                writer.clone(),
                &trial_refs,
                cb,
                enc_argb,
                enc_width,
                enc_height,
                config,
                is_palette,
            );
            let keep = match &best_output {
                None => true,
                Some(best) => output.len() < best.len(),
            };
            if keep {
                best_output = Some(output);
            }
        }

        Ok(best_output.unwrap())
    } else {
        Ok(encode_image_data(
            writer, &refs, cache_bits, enc_argb, enc_width, enc_height, config, is_palette,
        ))
    }
}

/// Apply and signal predictor transform.
fn write_predictor_transform(
    writer: &mut BitWriter,
    argb: &mut [u32],
    width: usize,
    height: usize,
    config: &Vp8lConfig,
    use_subtract_green: bool,
) {
    let pred_bits = if config.predictor_bits == 0 {
        let histo_bits = get_histo_bits(width, height, config.quality.method);
        let transform_bits = get_transform_bits(config.quality.method, histo_bits);
        clamp_bits(
            width,
            height,
            transform_bits,
            MIN_TRANSFORM_BITS,
            MAX_TRANSFORM_BITS,
            MAX_PREDICTOR_IMAGE_SIZE,
        )
    } else {
        config.predictor_bits.clamp(2, 8)
    };
    let max_quantization = if config.near_lossless < 100 {
        super::near_lossless::max_quantization_from_quality(config.near_lossless)
    } else {
        1
    };
    let predictor_data = apply_predictor_transform(
        argb,
        width,
        height,
        pred_bits,
        max_quantization,
        use_subtract_green,
    );

    // Signal predictor transform
    writer.write_bit(true); // transform present
    writer.write_bits(0, 2); // predictor = 0
    writer.write_bits((pred_bits - 2) as u64, 3);

    // Write predictor sub-image
    let pred_w = subsample_size(width as u32, pred_bits) as usize;
    let pred_h = subsample_size(height as u32, pred_bits) as usize;
    write_predictor_image(
        writer,
        &predictor_data,
        pred_w,
        pred_h,
        config.quality.quality,
    );
}

/// Apply and signal cross-color transform.
fn write_cross_color_transform(
    writer: &mut BitWriter,
    argb: &mut [u32],
    width: usize,
    height: usize,
    config: &Vp8lConfig,
) {
    let cc_bits = if config.cross_color_bits == 0 {
        let histo_bits = get_histo_bits(width, height, config.quality.method);
        let transform_bits = get_transform_bits(config.quality.method, histo_bits);
        clamp_bits(
            width,
            height,
            transform_bits,
            MIN_TRANSFORM_BITS,
            MAX_TRANSFORM_BITS,
            MAX_PREDICTOR_IMAGE_SIZE,
        )
    } else {
        config.cross_color_bits.clamp(2, 8)
    };
    let cross_color_data =
        apply_cross_color_transform(argb, width, height, cc_bits, config.quality.quality);

    // Signal cross-color transform
    writer.write_bit(true); // transform present
    writer.write_bits(1, 2); // cross-color = 1
    writer.write_bits((cc_bits - 2) as u64, 3);

    // Write cross-color sub-image
    let cc_w = subsample_size(width as u32, cc_bits) as usize;
    let cc_h = subsample_size(height as u32, cc_bits) as usize;
    write_cross_color_image(
        writer,
        &cross_color_data,
        cc_w,
        cc_h,
        config.quality.quality,
    );
}

/// Encode image data from backward references into a VP8L bitstream.
///
/// Handles: color cache signaling, meta-Huffman, Huffman tree construction/writing,
/// and image data encoding. Returns the finished byte buffer.
#[allow(clippy::too_many_arguments)]
fn encode_image_data(
    mut writer: BitWriter,
    refs: &super::types::BackwardRefs,
    cache_bits: u8,
    enc_argb: &[u32],
    enc_width: usize,
    enc_height: usize,
    config: &Vp8lConfig,
    is_palette: bool,
) -> Vec<u8> {
    // Write color cache info
    if cache_bits > 0 {
        writer.write_bit(true);
        writer.write_bits(cache_bits as u64, 4);
    } else {
        writer.write_bit(false);
    }

    // Build histogram(s) — either single or meta-Huffman with clustering
    let use_meta = config.use_meta_huffman && enc_width > 16 && enc_height > 16;
    let histo_bits = if use_meta {
        get_histo_bits_palette(enc_width, enc_height, config.quality.method, is_palette)
    } else {
        0u8
    };

    let meta_info = if use_meta {
        build_meta_huffman(
            refs,
            enc_width,
            enc_height,
            histo_bits,
            cache_bits,
            config.quality.quality,
        )
    } else {
        build_single_histogram(refs, cache_bits)
    };

    // Write meta-Huffman flag and prefix image
    if meta_info.num_histograms > 1 {
        writer.write_bit(true); // meta-Huffman present
        writer.write_bits((meta_info.histo_bits - 2) as u64, 3);

        // Write histogram prefix image (encoded as sub-image)
        write_histogram_image(&mut writer, &meta_info, config.quality.quality);
    } else {
        writer.write_bit(false); // no meta-Huffman
    }

    // Build Huffman codes for each histogram group (5 trees per group)
    let mut all_codes: Vec<HuffmanGroupCodes> = Vec::with_capacity(meta_info.num_histograms);
    for h in &meta_info.histograms {
        all_codes.push(build_huffman_group(h));
    }

    // Write all Huffman trees (5 per histogram group)
    for group in &all_codes {
        write_huffman_tree(&mut writer, &group.literal_lengths);
        write_huffman_tree(&mut writer, &group.red_lengths);
        write_huffman_tree(&mut writer, &group.blue_lengths);
        write_huffman_tree(&mut writer, &group.alpha_lengths);
        write_huffman_tree(&mut writer, &group.dist_lengths);
    }

    // Write image data with spatially-varying Huffman selection
    let mut color_cache = if cache_bits > 0 {
        Some(ColorCache::new(cache_bits))
    } else {
        None
    };

    let histo_xsize = if meta_info.num_histograms > 1 {
        subsample_size(enc_width as u32, meta_info.histo_bits) as usize
    } else {
        0
    };

    let mut x = 0usize;
    let mut y = 0usize;
    let mut argb_idx = 0usize;

    for token in refs.iter() {
        // Select Huffman group for current tile
        let group_idx = if meta_info.num_histograms > 1 {
            let tile_idx = (y >> meta_info.histo_bits) * histo_xsize + (x >> meta_info.histo_bits);
            meta_info.histogram_symbols[tile_idx] as usize
        } else {
            0
        };
        let codes = &all_codes[group_idx];

        match *token {
            PixOrCopy::Literal(argb_val) => {
                if !codes.literal_trivial {
                    let g = argb_green(argb_val) as usize;
                    writer.write_bits(
                        codes.literal_codes[g].code as u64,
                        codes.literal_codes[g].length,
                    );
                }
                if !codes.red_trivial {
                    let r = argb_red(argb_val) as usize;
                    writer.write_bits(codes.red_codes[r].code as u64, codes.red_codes[r].length);
                }
                if !codes.blue_trivial {
                    let b = argb_blue(argb_val) as usize;
                    writer.write_bits(codes.blue_codes[b].code as u64, codes.blue_codes[b].length);
                }
                if !codes.alpha_trivial {
                    let a = argb_alpha(argb_val) as usize;
                    writer.write_bits(
                        codes.alpha_codes[a].code as u64,
                        codes.alpha_codes[a].length,
                    );
                }

                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb_val);
                }
                x += 1;
                if x >= enc_width {
                    x = 0;
                    y += 1;
                }
                argb_idx += 1;
            }
            PixOrCopy::CacheIdx(idx) => {
                if !codes.literal_trivial {
                    let code = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
                    writer.write_bits(
                        codes.literal_codes[code].code as u64,
                        codes.literal_codes[code].length,
                    );
                }

                if let Some(ref mut cache) = color_cache {
                    cache.insert(enc_argb[argb_idx]);
                }
                x += 1;
                if x >= enc_width {
                    x = 0;
                    y += 1;
                }
                argb_idx += 1;
            }
            PixOrCopy::Copy { len, dist } => {
                let (len_prefix, len_extra) = length_to_code(len);
                let len_code = NUM_LITERAL_CODES + len_prefix as usize;
                if !codes.literal_trivial {
                    writer.write_bits(
                        codes.literal_codes[len_code].code as u64,
                        codes.literal_codes[len_code].length,
                    );
                }

                let len_extra_bits = length_code_extra_bits(len_prefix);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra as u64, len_extra_bits);
                }

                let (dist_prefix, dist_extra) = distance_code_to_prefix(dist);
                if !codes.dist_trivial {
                    writer.write_bits(
                        codes.dist_codes[dist_prefix as usize].code as u64,
                        codes.dist_codes[dist_prefix as usize].length,
                    );
                }

                let dist_extra_bits = distance_code_extra_bits(dist_prefix);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra as u64, dist_extra_bits);
                }

                if let Some(ref mut cache) = color_cache {
                    for k in 0..len as usize {
                        cache.insert(enc_argb[argb_idx + k]);
                    }
                }
                for _ in 0..len {
                    x += 1;
                    if x >= enc_width {
                        x = 0;
                        y += 1;
                    }
                }
                argb_idx += len as usize;
            }
        }
    }

    writer.finish()
}

/// Maximum number of histogram tiles (matching libwebp's MAX_HUFF_IMAGE_SIZE).
const MAX_HUFF_IMAGE_SIZE: usize = 2600;
/// Min/max Huffman bits range (VP8L spec: 3 bits, range [2, 9]).
const MIN_HUFFMAN_BITS: u8 = 2;
const MAX_HUFFMAN_BITS: u8 = 9; // 2 + (1 << 3) - 1
/// Transform bits range (VP8L spec).
const MIN_TRANSFORM_BITS: u8 = 2;
const MAX_TRANSFORM_BITS: u8 = 8;
/// Maximum predictor/cross-color sub-image size.
const MAX_PREDICTOR_IMAGE_SIZE: usize = 1 << 14; // 16384

/// Clamp bits to keep sub-image size within limits (matching libwebp's ClampBits).
fn clamp_bits(
    width: usize,
    height: usize,
    bits: u8,
    min_bits: u8,
    max_bits: u8,
    image_size_max: usize,
) -> u8 {
    let mut bits = bits.clamp(min_bits, max_bits);
    let mut image_size =
        subsample_size(width as u32, bits) as usize * subsample_size(height as u32, bits) as usize;
    while bits < max_bits && image_size > image_size_max {
        bits += 1;
        image_size = subsample_size(width as u32, bits) as usize
            * subsample_size(height as u32, bits) as usize;
    }
    // Don't reduce below needed: if image_size == 1 at current bits,
    // try going smaller
    while bits > min_bits {
        let smaller_size = subsample_size(width as u32, bits - 1) as usize
            * subsample_size(height as u32, bits - 1) as usize;
        if smaller_size != 1 {
            break;
        }
        bits -= 1;
    }
    bits
}

/// Calculate optimal histogram bits based on method and image size.
/// Matches libwebp's GetHistoBits + ClampBits.
fn get_histo_bits(width: usize, height: usize, method: u8) -> u8 {
    get_histo_bits_palette(width, height, method, false)
}

/// Calculate histogram bits with palette awareness.
/// Matches libwebp's GetHistoBits: use_palette adds +2 to base bits.
fn get_histo_bits_palette(width: usize, height: usize, method: u8, use_palette: bool) -> u8 {
    let base = if use_palette { 9i32 } else { 7i32 };
    let histo_bits = base - method as i32;
    clamp_bits(
        width,
        height,
        histo_bits.clamp(MIN_HUFFMAN_BITS as i32, MAX_HUFFMAN_BITS as i32) as u8,
        MIN_HUFFMAN_BITS,
        MAX_HUFFMAN_BITS,
        MAX_HUFF_IMAGE_SIZE,
    )
}

/// Calculate optimal transform bits (predictor/cross-color) based on method.
/// Matches libwebp's GetTransformBits.
fn get_transform_bits(method: u8, histo_bits: u8) -> u8 {
    let max_transform_bits: u8 = if method < 4 {
        6
    } else if method > 4 {
        4
    } else {
        5
    };
    histo_bits.min(max_transform_bits)
}

/// Pre-built Huffman codes for one histogram group (5 trees).
struct HuffmanGroupCodes {
    literal_lengths: Vec<u8>,
    literal_codes: Vec<HuffmanCode>,
    literal_trivial: bool,
    red_lengths: Vec<u8>,
    red_codes: Vec<HuffmanCode>,
    red_trivial: bool,
    blue_lengths: Vec<u8>,
    blue_codes: Vec<HuffmanCode>,
    blue_trivial: bool,
    alpha_lengths: Vec<u8>,
    alpha_codes: Vec<HuffmanCode>,
    alpha_trivial: bool,
    dist_lengths: Vec<u8>,
    dist_codes: Vec<HuffmanCode>,
    dist_trivial: bool,
}

/// Build Huffman codes for one histogram.
fn build_huffman_group(h: &super::histogram::Histogram) -> HuffmanGroupCodes {
    let literal_lengths = build_huffman_lengths(&h.literal, 15);
    let literal_codes = build_huffman_codes(&literal_lengths);
    let literal_trivial = literal_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let red_lengths = build_huffman_lengths(&h.red, 15);
    let red_codes = build_huffman_codes(&red_lengths);
    let red_trivial = red_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let blue_lengths = build_huffman_lengths(&h.blue, 15);
    let blue_codes = build_huffman_codes(&blue_lengths);
    let blue_trivial = blue_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let alpha_lengths = build_huffman_lengths(&h.alpha, 15);
    let alpha_codes = build_huffman_codes(&alpha_lengths);
    let alpha_trivial = alpha_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let dist_lengths = build_huffman_lengths(&h.distance, 15);
    let dist_codes = build_huffman_codes(&dist_lengths);
    let dist_trivial = dist_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    HuffmanGroupCodes {
        literal_lengths,
        literal_codes,
        literal_trivial,
        red_lengths,
        red_codes,
        red_trivial,
        blue_lengths,
        blue_codes,
        blue_trivial,
        alpha_lengths,
        alpha_codes,
        alpha_trivial,
        dist_lengths,
        dist_codes,
        dist_trivial,
    }
}

/// Write the histogram prefix image (meta-Huffman sub-image).
/// Uses full LZ77+Huffman encoding matching libwebp's EncodeImageNoHuffman.
fn write_histogram_image(writer: &mut BitWriter, meta_info: &MetaHuffmanInfo, quality: u8) {
    // Convert histogram symbols to ARGB pixels: green=low byte, red=high byte
    let argb: Vec<u32> = meta_info
        .histogram_symbols
        .iter()
        .map(|&sym| make_argb(0, (sym >> 8) as u8, (sym & 0xFF) as u8, 0))
        .collect();

    let histo_w = subsample_size(meta_info.image_width as u32, meta_info.histo_bits) as usize;
    let histo_h = subsample_size(meta_info.image_height as u32, meta_info.histo_bits) as usize;

    encode_image_no_huffman(writer, &argb, histo_w, histo_h, quality);
}

/// Check if the image can use a palette (≤256 unique colors).
/// Matches libwebp's GetColorPalette: returns true if unique color count ≤ MAX_PALETTE_SIZE.
fn can_use_palette(argb: &[u32]) -> bool {
    // Use hash-based approach matching libwebp's GetColorPalette for speed
    const HASH_SIZE: usize = 256 * 4; // COLOR_HASH_SIZE
    const HASH_SHIFT: u32 = 22; // 32 - log2(HASH_SIZE)

    let mut in_use = [false; HASH_SIZE];
    let mut colors = [0u32; HASH_SIZE];
    let mut num_colors = 0;
    let mut last_pix = !argb.first().copied().unwrap_or(0);

    for &pix in argb {
        if pix == last_pix {
            continue;
        }
        last_pix = pix;
        // Hash matching libwebp's VP8LHashPix
        let mut key = ((pix.wrapping_mul(0x1e35a7bd)) >> HASH_SHIFT) as usize;
        loop {
            if !in_use[key] {
                colors[key] = pix;
                in_use[key] = true;
                num_colors += 1;
                if num_colors > 256 {
                    return false;
                }
                break;
            } else if colors[key] == pix {
                break;
            } else {
                key = (key + 1) & (HASH_SIZE - 1);
            }
        }
    }

    true
}

/// Write predictor sub-image (variable modes per block).
///
/// The sub-image encodes one pixel per block with the predictor mode in the
/// green channel (0-13). Other channels are zero. Uses Huffman coding for
/// the green channel and trivial single-entry trees for other channels.
/// Encode a sub-image (predictor modes, cross-color data, histogram image, palette)
/// using full LZ77 + Huffman but no meta-Huffman or transforms.
/// Matches libwebp's EncodeImageNoHuffman.
fn encode_image_no_huffman(
    writer: &mut BitWriter,
    argb: &[u32],
    width: usize,
    height: usize,
    quality: u8,
) {
    use super::backward_refs::{
        apply_2d_locality, backward_references_lz77, backward_references_rle,
    };
    use super::entropy::estimate_histogram_bits;
    use super::hash_chain::HashChain;
    use super::histogram::Histogram;

    let size = argb.len();
    if size == 0 {
        // Write no-cache flag and trivial trees
        writer.write_bit(false); // no color cache
        for _ in 0..5 {
            write_single_entry_tree(writer, 0);
        }
        return;
    }

    // No color cache for sub-images (matching libwebp: cache_bits=0)
    let cache_bits: u8 = 0;

    // Build hash chain for the sub-image
    let hash_chain = HashChain::new(argb, quality, width);

    // Try LZ77 Standard and RLE, pick best by entropy
    let refs_lz77 = backward_references_lz77(argb, width, height, cache_bits, &hash_chain);
    let refs_rle = backward_references_rle(argb, width, height, cache_bits);

    let histo_lz77 = Histogram::from_refs_with_plane_codes(&refs_lz77, cache_bits, width);
    let histo_rle = Histogram::from_refs_with_plane_codes(&refs_rle, cache_bits, width);

    let cost_lz77 = estimate_histogram_bits(&histo_lz77);
    let cost_rle = estimate_histogram_bits(&histo_rle);

    let mut best_refs = if cost_rle < cost_lz77 {
        refs_rle
    } else {
        refs_lz77
    };

    // Apply 2D locality transform
    apply_2d_locality(&mut best_refs, width);

    // Build single histogram from refs
    let histo = Histogram::from_refs(&best_refs, cache_bits);

    // Build Huffman codes for each channel
    let lit_lengths = build_huffman_lengths(&histo.literal, 15);
    let lit_codes = build_huffman_codes(&lit_lengths);
    let lit_trivial = lit_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let red_lengths = build_huffman_lengths(&histo.red, 15);
    let red_codes = build_huffman_codes(&red_lengths);
    let red_trivial = red_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let blue_lengths = build_huffman_lengths(&histo.blue, 15);
    let blue_codes = build_huffman_codes(&blue_lengths);
    let blue_trivial = blue_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let alpha_lengths = build_huffman_lengths(&histo.alpha, 15);
    let alpha_codes = build_huffman_codes(&alpha_lengths);
    let alpha_trivial = alpha_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    let dist_lengths = build_huffman_lengths(&histo.distance, 15);
    let dist_codes = build_huffman_codes(&dist_lengths);
    let dist_trivial = dist_lengths.iter().filter(|&&l| l > 0).count() <= 1;

    // Write no color cache flag
    writer.write_bit(false);

    // Write 5 Huffman trees
    write_huffman_tree(writer, &lit_lengths);
    write_huffman_tree(writer, &red_lengths);
    write_huffman_tree(writer, &blue_lengths);
    write_huffman_tree(writer, &alpha_lengths);
    write_huffman_tree(writer, &dist_lengths);

    // Write image data with LZ77 tokens
    for token in best_refs.iter() {
        match *token {
            PixOrCopy::Literal(argb_val) => {
                let g = argb_green(argb_val) as usize;
                let r = argb_red(argb_val) as usize;
                let b = argb_blue(argb_val) as usize;
                let a = argb_alpha(argb_val) as usize;

                if !lit_trivial {
                    writer.write_bits(lit_codes[g].code as u64, lit_codes[g].length);
                }
                if !red_trivial {
                    writer.write_bits(red_codes[r].code as u64, red_codes[r].length);
                }
                if !blue_trivial {
                    writer.write_bits(blue_codes[b].code as u64, blue_codes[b].length);
                }
                if !alpha_trivial {
                    writer.write_bits(alpha_codes[a].code as u64, alpha_codes[a].length);
                }
            }
            PixOrCopy::CacheIdx(_) => {
                // No color cache for sub-images, shouldn't happen
            }
            PixOrCopy::Copy { len, dist } => {
                // Write length code
                let (len_code, len_extra_bits) = super::histogram::length_to_code(len);
                let len_extra_bits_count = super::histogram::length_code_extra_bits(len_code);
                if !lit_trivial {
                    let lit_idx = NUM_LITERAL_CODES + len_code as usize;
                    writer.write_bits(lit_codes[lit_idx].code as u64, lit_codes[lit_idx].length);
                }
                if len_extra_bits_count > 0 {
                    writer.write_bits(len_extra_bits as u64, len_extra_bits_count);
                }

                // Write distance code
                let (dist_code, dist_extra_bits) = super::histogram::distance_code_to_prefix(dist);
                let dist_extra_bits_count = super::histogram::distance_code_extra_bits(dist_code);
                if !dist_trivial {
                    writer.write_bits(
                        dist_codes[dist_code as usize].code as u64,
                        dist_codes[dist_code as usize].length,
                    );
                }
                if dist_extra_bits_count > 0 {
                    writer.write_bits(dist_extra_bits as u64, dist_extra_bits_count);
                }
            }
        }
    }
}

fn write_predictor_image(
    writer: &mut BitWriter,
    predictor_data: &[u32],
    width: usize,
    height: usize,
    quality: u8,
) {
    encode_image_no_huffman(writer, predictor_data, width, height, quality);
}

/// Write cross-color sub-image (multiplier data).
fn write_cross_color_image(
    writer: &mut BitWriter,
    cross_color_data: &[u32],
    width: usize,
    height: usize,
    quality: u8,
) {
    encode_image_no_huffman(writer, cross_color_data, width, height, quality);
}

/// Subtract two pixels component-wise (wrapping).
#[inline]
fn sub_pixels(a: u32, b: u32) -> u32 {
    let aa = argb_alpha(a).wrapping_sub(argb_alpha(b));
    let ar = argb_red(a).wrapping_sub(argb_red(b));
    let ag = argb_green(a).wrapping_sub(argb_green(b));
    let ab = argb_blue(a).wrapping_sub(argb_blue(b));
    make_argb(aa, ar, ag, ab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple() {
        // 4x4 red image
        let pixels: Vec<u8> = [255, 0, 0].repeat(16);
        let config = Vp8lConfig::default();

        let result = encode_vp8l(&pixels, 4, 4, false, &config, &enough::Unstoppable);
        assert!(result.is_ok());

        let data = result.unwrap();
        // Check VP8L signature
        assert_eq!(data[0], 0x2f);
    }

    #[test]
    fn test_encode_with_alpha() {
        // 4x4 semi-transparent red
        let pixels: Vec<u8> = [255, 0, 0, 128].repeat(16);
        let config = Vp8lConfig::default();

        let result = encode_vp8l(&pixels, 4, 4, true, &config, &enough::Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_gradient() {
        // 16x16 gradient
        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for y in 0..16 {
            for x in 0..16 {
                pixels.push((x * 16) as u8);
                pixels.push((y * 16) as u8);
                pixels.push(128);
            }
        }

        let config = Vp8lConfig::default();
        let result = encode_vp8l(&pixels, 16, 16, false, &config, &enough::Unstoppable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sub_pixels() {
        let a = make_argb(100, 50, 200, 10);
        let b = make_argb(50, 100, 100, 5);
        let diff = sub_pixels(a, b);

        assert_eq!(argb_alpha(diff), 50); // 100 - 50
        assert_eq!(argb_red(diff), 206); // 50 - 100 = -50 = 206 (wrapping)
        assert_eq!(argb_green(diff), 100); // 200 - 100
        assert_eq!(argb_blue(diff), 5); // 10 - 5
    }

    #[test]
    fn test_generate_configs_single() {
        // m4, q75: single config
        let analysis = AnalyzeResult {
            mode: EntropyMode::SpatialSubGreen,
            red_and_blue_always_zero: false,
        };
        let configs = generate_crunch_configs(&analysis, false, 4, 75);
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].mode, CrunchMode::SpatialSubGreen);
    }

    #[test]
    fn test_generate_configs_m5_palette() {
        // m5, q75, palette available: best guess + variant
        let analysis = AnalyzeResult {
            mode: EntropyMode::Palette,
            red_and_blue_always_zero: true,
        };
        let configs = generate_crunch_configs(&analysis, true, 5, 75);
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].mode, CrunchMode::Palette);
        assert_eq!(configs[1].mode, CrunchMode::PaletteAndSpatial);
    }

    #[test]
    fn test_generate_configs_m5_no_palette() {
        // m5, q75, no palette: single config
        let analysis = AnalyzeResult {
            mode: EntropyMode::Spatial,
            red_and_blue_always_zero: false,
        };
        let configs = generate_crunch_configs(&analysis, false, 5, 75);
        assert_eq!(configs.len(), 1);
    }

    #[test]
    fn test_generate_configs_m6_q100_brute() {
        // m6, q100, palette available: 4 non-palette + 2*2 palette = 8
        let analysis = AnalyzeResult {
            mode: EntropyMode::Direct,
            red_and_blue_always_zero: false,
        };
        let configs = generate_crunch_configs(&analysis, true, 6, 100);
        assert_eq!(configs.len(), 8);
    }

    #[test]
    fn test_generate_configs_m6_q100_no_palette() {
        // m6, q100, no palette: 4 non-palette modes
        let analysis = AnalyzeResult {
            mode: EntropyMode::Direct,
            red_and_blue_always_zero: false,
        };
        let configs = generate_crunch_configs(&analysis, false, 6, 100);
        assert_eq!(configs.len(), 4);
    }

    #[test]
    fn test_multi_config_palette_image() {
        // 8x8 image with 4 colors: should trigger palette mode.
        // Test that multi-config mode (m5 q75) produces valid output.
        let colors: [[u8; 3]; 4] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let mut pixels = Vec::with_capacity(8 * 8 * 3);
        for i in 0..64 {
            let c = colors[i % 4];
            pixels.extend_from_slice(&c);
        }

        let mut config = Vp8lConfig::default();
        config.quality.method = 5;
        config.quality.quality = 75;
        let result = encode_vp8l(&pixels, 8, 8, false, &config, &enough::Unstoppable);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0], 0x2f); // VP8L signature
    }

    #[test]
    fn test_brute_force_all_modes() {
        // 16x16 image with few colors: m6 q100 brute-force
        let colors: [[u8; 3]; 4] = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
        let mut pixels = Vec::with_capacity(16 * 16 * 3);
        for i in 0..256 {
            let c = colors[i % 4];
            pixels.extend_from_slice(&c);
        }

        let mut config = Vp8lConfig::default();
        config.quality.method = 6;
        config.quality.quality = 100;
        let result = encode_vp8l(&pixels, 16, 16, false, &config, &enough::Unstoppable);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0], 0x2f); // VP8L signature
    }

    #[test]
    fn test_palette_and_spatial_forced() {
        // Force PaletteAndSpatial mode directly and verify output is valid VP8L.
        let colors: [[u8; 3]; 8] = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [100, 110, 120],
            [130, 140, 150],
            [160, 170, 180],
            [190, 200, 210],
            [220, 230, 240],
        ];
        let w = 16usize;
        let h = 16usize;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            let c = colors[i % 8];
            pixels.extend_from_slice(&c);
        }

        // Convert to ARGB
        let mut argb: Vec<u32> = pixels
            .chunks_exact(3)
            .map(|p| make_argb(255, p[0], p[1], p[2]))
            .collect();

        let config = Vp8lConfig::default();
        let crunch = CrunchConfig {
            mode: CrunchMode::PaletteAndSpatial,
            palette_sorting: PaletteSorting::MinimizeDelta,
        };

        let result = encode_argb_single_config(
            &mut argb,
            w,
            h,
            false,
            &config,
            &crunch,
            &enough::Unstoppable,
        );
        assert!(result.is_ok(), "PaletteAndSpatial encoding failed");
        let data = result.unwrap();
        assert_eq!(data[0], 0x2f); // VP8L signature
        assert!(data.len() > 10); // Non-trivial output
    }

    #[test]
    fn test_palette_sorting_lexicographic() {
        // Test lexicographic sorting produces different (but valid) output
        let colors: [[u8; 3]; 4] = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]];
        let w = 8usize;
        let h = 8usize;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            let c = colors[i % 4];
            pixels.extend_from_slice(&c);
        }

        let mut argb: Vec<u32> = pixels
            .chunks_exact(3)
            .map(|p| make_argb(255, p[0], p[1], p[2]))
            .collect();

        let config = Vp8lConfig::default();

        // MinimizeDelta sorting
        let mut argb_md = argb.clone();
        let crunch_md = CrunchConfig {
            mode: CrunchMode::Palette,
            palette_sorting: PaletteSorting::MinimizeDelta,
        };
        let result_md = encode_argb_single_config(
            &mut argb_md,
            w,
            h,
            false,
            &config,
            &crunch_md,
            &enough::Unstoppable,
        )
        .unwrap();

        // Lexicographic sorting
        let crunch_lex = CrunchConfig {
            mode: CrunchMode::Palette,
            palette_sorting: PaletteSorting::Lexicographic,
        };
        let result_lex = encode_argb_single_config(
            &mut argb,
            w,
            h,
            false,
            &config,
            &crunch_lex,
            &enough::Unstoppable,
        )
        .unwrap();

        // Both should be valid VP8L
        assert_eq!(result_md[0], 0x2f);
        assert_eq!(result_lex[0], 0x2f);
    }
}
