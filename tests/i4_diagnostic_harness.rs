// Requires C libwebp (webpx) — not available on wasm32
#![cfg(not(target_arch = "wasm32"))]
//! I4 Encoding Efficiency Decomposition Harness
//!
//! Compares zenwebp and libwebp VP8 bitstreams stage-by-stage to identify
//! where encoding efficiency diverges. Decodes both outputs with diagnostic
//! capture to compare modes, coefficient levels, and probability tables.
//!
//! Run with: cargo test --release --features _corpus_tests --test i4_diagnostic_harness -- --nocapture

#![cfg(feature = "_corpus_tests")]

use std::fs;
use std::io::BufReader;
use webpx::Unstoppable;
use zenwebp::decoder::LumaMode;
use zenwebp::decoder::vp8::{DiagnosticFrame, Vp8Decoder};
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};

// ============================================================================
// Image Loading
// ============================================================================

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

/// Create a synthetic test image with known patterns for predictable encoding
fn create_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Checkerboard with noise pattern - good for I4 mode selection
            let checker = ((x / 4) + (y / 4)) % 2 == 0;
            let base = if checker { 200u8 } else { 50u8 };
            // Add slight variation to avoid degenerate DCT
            let noise = ((x * 7 + y * 13) % 20) as u8;
            rgb[idx] = base.saturating_add(noise);
            rgb[idx + 1] = base.saturating_add(noise / 2);
            rgb[idx + 2] = base.saturating_sub(noise / 2);
        }
    }
    rgb
}

// ============================================================================
// VP8 Chunk Extraction
// ============================================================================

/// Extract raw VP8 data from a WebP container
fn extract_vp8_chunk(webp: &[u8]) -> Option<&[u8]> {
    // WebP format: RIFF header (12 bytes) + chunks
    // RIFF <size> WEBP
    if webp.len() < 12 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return None;
    }

    let mut pos = 12;
    while pos + 8 <= webp.len() {
        let chunk_fourcc = &webp[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(webp[pos + 4..pos + 8].try_into().ok()?) as usize;

        if chunk_fourcc == b"VP8 " {
            // VP8 lossy chunk - data starts at pos + 8
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(webp.len());
            return Some(&webp[data_start..data_end]);
        }

        // Move to next chunk (chunks are padded to even size)
        pos += 8 + chunk_size + (chunk_size & 1);
    }

    None
}

// ============================================================================
// Encoding Helpers with Matched Settings
// ============================================================================

/// Encode with zenwebp using diagnostic-friendly settings
fn encode_zenwebp(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = EncoderConfig::with_preset(Preset::Default, quality)
        .with_method(method)
        // Disable SNS and filtering for cleaner comparison
        .with_sns_strength(0)
        .with_filter_strength(0)
        .with_filter_sharpness(0)
        .with_segments(1); // Single segment simplifies quantizer comparison
    EncodeRequest::new(&config, rgb, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("zenwebp encoding failed")
}

/// Encode with libwebp (via webpx) using matched settings
fn encode_libwebp(rgb: &[u8], width: u32, height: u32, quality: f32, method: u8) -> Vec<u8> {
    let config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, quality)
        .method(method)
        .sns_strength(0)
        .filter_strength(0)
        .filter_sharpness(0)
        .segments(1);

    config
        .encode_rgb(rgb, width, height, Unstoppable)
        .expect("libwebp encoding failed")
}

// ============================================================================
// Diagnostic Comparison
// ============================================================================

/// Statistics from comparing two DiagnosticFrames
#[derive(Debug, Default)]
struct ComparisonStats {
    /// Number of macroblocks with matching luma mode
    luma_mode_matches: usize,
    /// Number of macroblocks with different luma mode
    luma_mode_mismatches: usize,

    /// I4 blocks where both chose I4
    i4_blocks_both: usize,
    /// I4 sub-block mode matches (within blocks where both are I4)
    i4_mode_matches: usize,
    /// I4 sub-block mode mismatches
    i4_mode_mismatches: usize,

    /// Coefficient level comparisons
    coeff_blocks_compared: usize,
    coeff_blocks_exact: usize,
    total_level_diff: i64,
    max_level_diff: i32,

    /// First few mismatches for debugging
    first_mismatches: Vec<String>,
}

impl ComparisonStats {
    fn report(&self) {
        let total_mbs = self.luma_mode_matches + self.luma_mode_mismatches;
        let mode_pct = if total_mbs > 0 {
            100.0 * self.luma_mode_matches as f64 / total_mbs as f64
        } else {
            0.0
        };

        println!("\n=== Diagnostic Comparison Results ===");
        println!(
            "Mode decisions: {}/{} match ({:.1}%)",
            self.luma_mode_matches, total_mbs, mode_pct
        );

        if self.i4_blocks_both > 0 {
            let i4_total = self.i4_mode_matches + self.i4_mode_mismatches;
            let i4_pct = 100.0 * self.i4_mode_matches as f64 / i4_total as f64;
            println!(
                "I4 sub-block modes: {}/{} match ({:.1}%)",
                self.i4_mode_matches, i4_total, i4_pct
            );
        }

        if self.coeff_blocks_compared > 0 {
            let coeff_pct =
                100.0 * self.coeff_blocks_exact as f64 / self.coeff_blocks_compared as f64;
            println!(
                "Coefficient blocks: {}/{} exact ({:.1}%)",
                self.coeff_blocks_exact, self.coeff_blocks_compared, coeff_pct
            );
            println!(
                "Total |level_diff|: {}, max: {}",
                self.total_level_diff, self.max_level_diff
            );
        }

        if !self.first_mismatches.is_empty() {
            println!("\nFirst mismatches:");
            for m in &self.first_mismatches {
                println!("  {}", m);
            }
        }
    }
}

/// Compare two diagnostic frames and collect statistics
fn compare_diagnostics(zen: &DiagnosticFrame, lib: &DiagnosticFrame) -> ComparisonStats {
    let mut stats = ComparisonStats::default();

    // Check dimensions match
    assert_eq!(zen.mb_width, lib.mb_width, "MB width mismatch");
    assert_eq!(zen.mb_height, lib.mb_height, "MB height mismatch");
    assert_eq!(
        zen.macroblocks.len(),
        lib.macroblocks.len(),
        "MB count mismatch"
    );

    // Compare segment quantizers
    println!("\nSegment quantizers:");
    for (i, (z, l)) in zen.segments.iter().zip(lib.segments.iter()).enumerate() {
        let match_str = if z == l { "[MATCH]" } else { "[DIFFER]" };
        println!("  Seg {}: zen={:?} lib={:?} {}", i, z, l, match_str);
    }

    // Compare macroblocks
    for (mb_idx, (z_mb, l_mb)) in zen
        .macroblocks
        .iter()
        .zip(lib.macroblocks.iter())
        .enumerate()
    {
        let mbx = mb_idx % zen.mb_width as usize;
        let mby = mb_idx / zen.mb_width as usize;

        // Compare luma mode
        if z_mb.luma_mode == l_mb.luma_mode {
            stats.luma_mode_matches += 1;
        } else {
            stats.luma_mode_mismatches += 1;
            if stats.first_mismatches.len() < 5 {
                stats.first_mismatches.push(format!(
                    "MB({},{}) luma mode: zen={:?} lib={:?}",
                    mbx, mby, z_mb.luma_mode, l_mb.luma_mode
                ));
            }
        }

        // If both are I4 (LumaMode::B), compare sub-block modes
        if z_mb.luma_mode == LumaMode::B && l_mb.luma_mode == LumaMode::B {
            stats.i4_blocks_both += 1;
            for (blk_idx, (z_mode, l_mode)) in z_mb
                .bpred_modes
                .iter()
                .zip(l_mb.bpred_modes.iter())
                .enumerate()
            {
                if z_mode == l_mode {
                    stats.i4_mode_matches += 1;
                } else {
                    stats.i4_mode_mismatches += 1;
                    if stats.first_mismatches.len() < 5 {
                        stats.first_mismatches.push(format!(
                            "MB({},{}) I4 block {}: zen={:?} lib={:?}",
                            mbx, mby, blk_idx, z_mode, l_mode
                        ));
                    }
                }
            }

            // Compare Y block coefficients (only when both are I4)
            for (blk_idx, (z_blk, l_blk)) in
                z_mb.y_blocks.iter().zip(l_mb.y_blocks.iter()).enumerate()
            {
                stats.coeff_blocks_compared += 1;
                let mut exact = true;
                let mut block_diff = 0i64;

                for pos in 0..16 {
                    let z_level = z_blk.levels[pos];
                    let l_level = l_blk.levels[pos];
                    let diff = (z_level - l_level).abs();
                    if diff != 0 {
                        exact = false;
                        block_diff += diff as i64;
                        stats.max_level_diff = stats.max_level_diff.max(diff);

                        if stats.first_mismatches.len() < 5 {
                            stats.first_mismatches.push(format!(
                                "MB({},{}) Y block {} pos {}: zen={} lib={}",
                                mbx, mby, blk_idx, pos, z_level, l_level
                            ));
                        }
                    }
                }

                if exact {
                    stats.coeff_blocks_exact += 1;
                }
                stats.total_level_diff += block_diff;
            }
        }
    }

    stats
}

// ============================================================================
// Test Cases
// ============================================================================

#[test]
fn single_mb_diagnostic() {
    println!("\n=== Single MB Diagnostic (16x16, method 2, Q75) ===");

    let rgb = create_test_image(16, 16);
    let quality = 75.0;
    let method = 2; // No trellis

    let zen_webp = encode_zenwebp(&rgb, 16, 16, quality, method);
    let lib_webp = encode_libwebp(&rgb, 16, 16, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    // Extract VP8 chunks
    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    // Decode with diagnostics
    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    println!(
        "Partition 0: zenwebp={} bytes, libwebp={} bytes",
        zen_diag.partition0_size, lib_diag.partition0_size
    );

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

#[test]
fn small_image_diagnostic() {
    println!("\n=== Small Image Diagnostic (64x64, method 2, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Count mode breakdown
    let zen_i4 = zen_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let lib_i4 = lib_diag
        .macroblocks
        .iter()
        .filter(|m| m.luma_mode == LumaMode::B)
        .count();
    let total = zen_diag.macroblocks.len();

    println!(
        "\nMode breakdown: zenwebp {} I4 / {} total, libwebp {} I4 / {} total",
        zen_i4, total, lib_i4, total
    );
}

#[test]
fn small_image_m4_diagnostic() {
    println!("\n=== Small Image Diagnostic (64x64, method 4 with trellis, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 4; // With trellis

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

#[test]
fn benchmark_image_diagnostic() {
    println!("\n=== Benchmark Image Diagnostic (792079.png, method 2, Q75) ===");

    // Try to load the benchmark image
    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, width, height, quality, method);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Aggregate nonzero coefficient stats
    let mut zen_nonzero = 0usize;
    let mut lib_nonzero = 0usize;

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            zen_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }
    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            lib_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }

    println!(
        "\nNonzero Y coefficients: zenwebp={}, libwebp={}, ratio={:.3}x",
        zen_nonzero,
        lib_nonzero,
        zen_nonzero as f64 / lib_nonzero as f64
    );
}

#[test]
fn benchmark_image_m4_diagnostic() {
    println!("\n=== Benchmark Image Diagnostic (792079.png, method 4 with trellis, Q75) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;
    let method = 4;

    let zen_webp = encode_zenwebp(&rgb, width, height, quality, method);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, method);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();

    // Aggregate nonzero coefficient stats
    let mut zen_nonzero = 0usize;
    let mut lib_nonzero = 0usize;

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            zen_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }
    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            lib_nonzero += blk.levels.iter().filter(|&&l| l != 0).count();
        }
    }

    println!(
        "\nNonzero Y coefficients: zenwebp={}, libwebp={}, ratio={:.3}x",
        zen_nonzero,
        lib_nonzero,
        zen_nonzero as f64 / lib_nonzero as f64
    );
}

/// Compare H vs TM prediction SSE for MB(8,0)
/// This helps understand why we pick different modes than libwebp
#[test]
#[allow(clippy::needless_range_loop)]
fn compare_h_tm_sse_mb_8_0() {
    println!("\n=== Compare H vs TM SSE for MB(8,0) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, _height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Convert to YUV
    let y_buf: Vec<u8> = rgb
        .chunks_exact(3)
        .map(|p| {
            let r = p[0] as i32;
            let g = p[1] as i32;
            let b = p[2] as i32;
            ((66 * r + 129 * g + 25 * b + 128) >> 8).clamp(0, 255) as u8 + 16
        })
        .collect();

    let src_width = width as usize;
    let mbx = 8usize;
    let mby = 0usize;

    // Get the source 16x16 block
    let mut src_block = [0u8; 256];
    for y in 0..16 {
        for x in 0..16 {
            let src_idx = (mby * 16 + y) * src_width + mbx * 16 + x;
            src_block[y * 16 + x] = y_buf[src_idx];
        }
    }

    // Get border pixels for prediction (top row and left column)
    let mut top_row = [128u8; 16];
    let mut left_col = [128u8; 16];

    if mby > 0 {
        for x in 0..16 {
            let idx = (mby * 16 - 1) * src_width + mbx * 16 + x;
            top_row[x] = y_buf[idx];
        }
    }
    if mbx > 0 {
        for y in 0..16 {
            let idx = (mby * 16 + y) * src_width + mbx * 16 - 1;
            left_col[y] = y_buf[idx];
        }
    }
    let top_left = if mby > 0 && mbx > 0 {
        y_buf[(mby * 16 - 1) * src_width + mbx * 16 - 1]
    } else {
        128
    };

    // Compute H prediction
    let mut h_pred = [0u8; 256];
    for y in 0..16 {
        let left = left_col[y];
        for x in 0..16 {
            h_pred[y * 16 + x] = left;
        }
    }

    // Compute TM prediction
    let mut tm_pred = [0u8; 256];
    for y in 0..16 {
        let l = left_col[y] as i32;
        let p = top_left as i32;
        for x in 0..16 {
            let a = top_row[x] as i32;
            tm_pred[y * 16 + x] = (l + a - p).clamp(0, 255) as u8;
        }
    }

    // Compute SSE for each prediction
    let h_sse: u64 = src_block
        .iter()
        .zip(h_pred.iter())
        .map(|(&s, &p)| (s as i64 - p as i64).pow(2) as u64)
        .sum();
    let tm_sse: u64 = src_block
        .iter()
        .zip(tm_pred.iter())
        .map(|(&s, &p)| (s as i64 - p as i64).pow(2) as u64)
        .sum();

    println!("Top-left (P) = {}", top_left);
    println!("Top row (first 8): {:?}", &top_row[..8]);
    println!("Left col (first 8): {:?}", &left_col[..8]);
    println!("Source block (first row): {:?}", &src_block[..16]);
    println!("\nH prediction SSE: {}", h_sse);
    println!("TM prediction SSE: {}", tm_sse);
    println!("Difference (H - TM): {}", h_sse as i64 - tm_sse as i64);

    if tm_sse < h_sse {
        println!("\n=> TM has lower SSE, libwebp's choice is correct for this prediction");
    } else {
        println!("\n=> H has lower or equal SSE, our choice of H is correct for this prediction");
    }
}

/// Debug mode selection for specific MB to understand H vs TM decision
#[test]
fn debug_mode_selection_mb_8_0() {
    println!("\n=== Debug Mode Selection for MB(8,0) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Encode with zenwebp at m4
    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 4);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // Look at MB(8,0) - first mismatch
    let mb_idx = 8; // row 0, col 8
    let zen_mb = &zen_diag.macroblocks[mb_idx];
    let lib_mb = &lib_diag.macroblocks[mb_idx];

    println!("MB(8,0):");
    println!("  zenwebp luma mode: {:?}", zen_mb.luma_mode);
    println!("  libwebp luma mode: {:?}", lib_mb.luma_mode);
    println!("  zenwebp segment: {}", zen_mb.segment_id);
    println!("  libwebp segment: {}", lib_mb.segment_id);

    // Compare coefficient levels for this MB
    if zen_mb.luma_mode == LumaMode::B && lib_mb.luma_mode == LumaMode::B {
        println!("  Both use I4, comparing bpred modes...");
        for i in 0..16 {
            if zen_mb.bpred_modes[i] != lib_mb.bpred_modes[i] {
                println!(
                    "    Block {}: zen={:?} lib={:?}",
                    i, zen_mb.bpred_modes[i], lib_mb.bpred_modes[i]
                );
            }
        }
    } else {
        println!(
            "  Different I16 modes (H={:?} vs TM={:?})",
            zen_mb.luma_mode, lib_mb.luma_mode
        );

        // For I16 modes, show Y2 (WHT/DC) coefficients
        println!("\n  Y2 (WHT/DC) coefficients:");
        print!("    zen: ");
        for i in 0..16 {
            print!("{:4} ", zen_mb.y2_block.levels[i]);
        }
        println!();
        print!("    lib: ");
        for i in 0..16 {
            print!("{:4} ", lib_mb.y2_block.levels[i]);
        }
        println!();

        // Count Y2 nonzero
        let zen_y2_nz = zen_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
        let lib_y2_nz = lib_mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
        println!("  Nonzero Y2 coeffs: zen={}, lib={}", zen_y2_nz, lib_y2_nz);

        // Y1 AC coefficients
        let zen_y1_nz: usize = zen_mb
            .y_blocks
            .iter()
            .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
            .sum();
        let lib_y1_nz: usize = lib_mb
            .y_blocks
            .iter()
            .map(|b| b.levels.iter().filter(|&&l| l != 0).count())
            .sum();
        println!(
            "  Nonzero Y1 (AC) coeffs: zen={}, lib={}",
            zen_y1_nz, lib_y1_nz
        );
    }

    // Also check the next few MBs
    println!("\nMBs 8-12 in row 0:");
    for col in 8..=12 {
        let zm = &zen_diag.macroblocks[col];
        let lm = &lib_diag.macroblocks[col];
        println!(
            "  MB({},0): zen={:?} lib={:?}",
            col, zm.luma_mode, lm.luma_mode
        );
    }
}

/// Quick corpus size comparison across multiple CID22 images
/// Uses fair comparisons: trellis vs trellis, non-trellis vs non-trellis
#[test]
fn quick_corpus_comparison() {
    println!("\n=== Quick CID22 Corpus Comparison (Q75) ===");

    let images = [
        "792079.png",
        "7062177.png",
        "670530.png",
        "2020432.png",
        "5894435.png",
    ];

    // Fair comparison 1: our m4 (trellis) vs libwebp m6 (trellis)
    println!("\n--- Trellis comparison: zenwebp m4 vs libwebp m6 ---");
    let mut total_zen = 0usize;
    let mut total_lib = 0usize;

    for img_name in images {
        let path = format!("/tmp/CID22/original/{}", img_name);
        let (rgb, width, height) = match load_png(&path) {
            Some(data) => data,
            None => {
                println!("Skipping: {} not found", img_name);
                continue;
            }
        };

        let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
        let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 6);

        let ratio = zen_webp.len() as f64 / lib_webp.len() as f64;
        println!(
            "  {}: zen(m4)={} lib(m6)={} ratio={:.3}x",
            img_name,
            zen_webp.len(),
            lib_webp.len(),
            ratio
        );

        total_zen += zen_webp.len();
        total_lib += lib_webp.len();
    }

    let total_ratio = total_zen as f64 / total_lib as f64;
    println!(
        "Aggregate (trellis): zen={} lib={} ratio={:.3}x",
        total_zen, total_lib, total_ratio
    );

    // Fair comparison 2: our m2 (no trellis) vs libwebp m4 (no trellis)
    println!("\n--- Non-trellis comparison: zenwebp m2 vs libwebp m4 ---");
    total_zen = 0;
    total_lib = 0;

    for img_name in images {
        let path = format!("/tmp/CID22/original/{}", img_name);
        let (rgb, width, height) = match load_png(&path) {
            Some(data) => data,
            None => continue,
        };

        let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 2);
        let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 4);

        let ratio = zen_webp.len() as f64 / lib_webp.len() as f64;
        println!(
            "  {}: zen(m2)={} lib(m4)={} ratio={:.3}x",
            img_name,
            zen_webp.len(),
            lib_webp.len(),
            ratio
        );

        total_zen += zen_webp.len();
        total_lib += lib_webp.len();
    }

    let total_ratio = total_zen as f64 / total_lib as f64;
    println!(
        "Aggregate (non-trellis): zen={} lib={} ratio={:.3}x",
        total_zen, total_lib, total_ratio
    );
}

/// Fair trellis comparison: our m4 (has trellis) vs libwebp m6 (has trellis)
#[test]
fn benchmark_trellis_fair_comparison() {
    println!("\n=== Fair Trellis: zenwebp m4 vs libwebp m6 (both with trellis) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;

    // Our m4 has trellis; libwebp needs m6 for trellis
    let zen_webp = encode_zenwebp(&rgb, width, height, quality, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, 6);

    println!(
        "File sizes: zenwebp(m4)={} bytes, libwebp(m6)={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );
}

/// Test method 6 (both have trellis) on real image
#[test]
fn benchmark_image_m6_vs_libwebp_m6() {
    println!("\n=== Benchmark Image: zenwebp m6 vs libwebp m6 (both with trellis) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image size: {}x{}", width, height);

    let quality = 75.0;

    // Both at method 6 - both should have trellis
    let zen_webp = encode_zenwebp(&rgb, width, height, quality, 6);
    let lib_webp = encode_libwebp(&rgb, width, height, quality, 6);

    println!(
        "File sizes: zenwebp={} bytes, libwebp={} bytes, ratio={:.3}x",
        zen_webp.len(),
        lib_webp.len(),
        zen_webp.len() as f64 / lib_webp.len() as f64
    );

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    let stats = compare_diagnostics(&zen_diag, &lib_diag);
    stats.report();
}

/// Compare coefficient level distributions between encoders
#[test]
fn coefficient_distribution_analysis() {
    println!("\n=== Coefficient Level Distribution Analysis (zen m4 vs lib m6) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Fair comparison: our m4 (trellis) vs libwebp m6 (trellis)
    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 6);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // Count coefficient level distribution for both
    let mut zen_level_counts = [0usize; 128];
    let mut lib_level_counts = [0usize; 128];

    for mb in &zen_diag.macroblocks {
        for blk in &mb.y_blocks {
            for &level in &blk.levels {
                let idx = level.unsigned_abs().min(127) as usize;
                zen_level_counts[idx] += 1;
            }
        }
    }

    for mb in &lib_diag.macroblocks {
        for blk in &mb.y_blocks {
            for &level in &blk.levels {
                let idx = level.unsigned_abs().min(127) as usize;
                lib_level_counts[idx] += 1;
            }
        }
    }

    println!("Level distribution (non-zero levels only):");
    println!("Level | zenwebp | libwebp | delta");
    for i in 1..32 {
        if zen_level_counts[i] > 0 || lib_level_counts[i] > 0 {
            let delta = zen_level_counts[i] as i64 - lib_level_counts[i] as i64;
            let delta_sign = if delta > 0 { "+" } else { "" };
            println!(
                "{:5} | {:7} | {:7} | {}{}",
                i, zen_level_counts[i], lib_level_counts[i], delta_sign, delta
            );
        }
    }

    // Total nonzero
    let zen_total: usize = zen_level_counts[1..].iter().sum();
    let lib_total: usize = lib_level_counts[1..].iter().sum();
    println!(
        "\nTotal nonzero: zenwebp={}, libwebp={}",
        zen_total, lib_total
    );
}

/// Compare probability tables on real benchmark image
/// Fair comparison: zenwebp m4 (trellis) vs libwebp m6 (trellis)
#[test]
fn probability_table_comparison_real_image() {
    println!("\n=== Probability Table Comparison (792079.png, zen m4 vs lib m6) ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Fair comparison: our m4 (trellis) vs libwebp m6 (trellis)
    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 6);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // Compare token probability tables
    let mut total_entries = 0usize;
    let mut matching_entries = 0usize;
    let mut total_diff = 0u64;
    let mut max_diff = 0u8;

    println!("Probability differences by plane/band (showing divergent entries):");
    for plane in 0..4 {
        for band in 0..8 {
            for ctx in 0..3 {
                for tok in 0..11 {
                    total_entries += 1;
                    let z_prob = zen_diag.token_probs[plane][band][ctx][tok].prob;
                    let l_prob = lib_diag.token_probs[plane][band][ctx][tok].prob;

                    if z_prob == l_prob {
                        matching_entries += 1;
                    } else {
                        let diff = z_prob.abs_diff(l_prob);
                        total_diff += diff as u64;
                        max_diff = max_diff.max(diff);
                        if diff >= 30 {
                            println!(
                                "  [P{} B{} C{} T{}] zen={} lib={} diff={}",
                                plane, band, ctx, tok, z_prob, l_prob, diff
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "\nToken probabilities: {}/{} match ({:.1}%)",
        matching_entries,
        total_entries,
        100.0 * matching_entries as f64 / total_entries as f64
    );
    println!("Total |prob_diff|: {}, max: {}", total_diff, max_diff);
}

#[test]
fn probability_table_comparison() {
    println!("\n=== Probability Table Comparison (64x64, method 2, Q75) ===");

    let rgb = create_test_image(64, 64);
    let quality = 75.0;
    let method = 2;

    let zen_webp = encode_zenwebp(&rgb, 64, 64, quality, method);
    let lib_webp = encode_libwebp(&rgb, 64, 64, quality, method);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("Failed to extract VP8 from zenwebp");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("Failed to extract VP8 from libwebp");

    let (_, zen_diag) =
        Vp8Decoder::decode_diagnostic(zen_vp8).expect("Failed to decode zenwebp with diagnostics");
    let (_, lib_diag) =
        Vp8Decoder::decode_diagnostic(lib_vp8).expect("Failed to decode libwebp with diagnostics");

    // Compare token probability tables
    let mut total_entries = 0usize;
    let mut matching_entries = 0usize;
    let mut total_diff = 0u64;
    let mut max_diff = 0u8;

    for plane in 0..4 {
        for band in 0..8 {
            for ctx in 0..3 {
                for tok in 0..11 {
                    total_entries += 1;
                    let z_prob = zen_diag.token_probs[plane][band][ctx][tok].prob;
                    let l_prob = lib_diag.token_probs[plane][band][ctx][tok].prob;
                    if z_prob == l_prob {
                        matching_entries += 1;
                    } else {
                        let diff = z_prob.abs_diff(l_prob);
                        total_diff += diff as u64;
                        max_diff = max_diff.max(diff);
                    }
                }
            }
        }
    }

    println!(
        "Token probabilities: {}/{} match ({:.1}%)",
        matching_entries,
        total_entries,
        100.0 * matching_entries as f64 / total_entries as f64
    );
    println!("Total |prob_diff|: {}, max: {}", total_diff, max_diff);
}

/// True apples-to-apples comparison: same method number for both encoders
/// This isolates the encoding efficiency difference from trellis configuration differences
#[test]
fn apples_to_apples_same_method() {
    println!("\n=== Apples-to-Apples: Same Method Number for Both Encoders ===");
    println!("Settings: SNS=0, filter=0, segments=1, Q75");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Image: 792079.png ({}x{})\n", width, height);
    println!("Method | zenwebp | libwebp | Ratio  | Notes");
    println!("-------|---------|---------|--------|-------");

    for method in 0..=6u8 {
        let zen = encode_zenwebp(&rgb, width, height, 75.0, method);
        let lib = encode_libwebp(&rgb, width, height, 75.0, method);
        let ratio = zen.len() as f64 / lib.len() as f64;

        let notes = match method {
            0 => "I16-only, fastest",
            1 => "I16-only",
            2 => "I4 limited modes, no trellis",
            3 => "I4 limited modes, no trellis",
            4 => "I4 full, lib:no-trellis zen:trellis",
            5 => "I4 full, both have trellis",
            6 => "I4 full, both have trellis+extra",
            _ => "",
        };

        println!(
            "   {}   | {:>7} | {:>7} | {:.3}x | {}",
            method,
            zen.len(),
            lib.len(),
            ratio,
            notes
        );
    }
}

/// Detailed I16 vs I4 breakdown to isolate where the gap comes from
#[test]
fn i16_vs_i4_breakdown() {
    println!("\n=== I16 vs I4 Breakdown ===");
    println!("Comparing coefficient counts and sizes by block type");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Use method 4 for both (our trellis vs libwebp no-trellis)
    // and method 6 for both (both have trellis)
    for method in [4u8, 6] {
        println!("\n--- Method {} ---", method);

        let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, method);
        let lib_webp = encode_libwebp(&rgb, width, height, 75.0, method);

        let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("VP8");
        let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("VP8");

        let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
        let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

        // Count I16 vs I4 blocks
        let zen_i16 = zen_diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode != LumaMode::B)
            .count();
        let zen_i4 = zen_diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode == LumaMode::B)
            .count();
        let lib_i16 = lib_diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode != LumaMode::B)
            .count();
        let lib_i4 = lib_diag
            .macroblocks
            .iter()
            .filter(|m| m.luma_mode == LumaMode::B)
            .count();

        println!("Mode distribution:");
        println!("  zenwebp: {} I16, {} I4", zen_i16, zen_i4);
        println!("  libwebp: {} I16, {} I4", lib_i16, lib_i4);

        // Count total nonzero coefficients
        let mut zen_y2_nz = 0usize;
        let mut zen_y1_nz = 0usize;
        let mut lib_y2_nz = 0usize;
        let mut lib_y1_nz = 0usize;

        for mb in &zen_diag.macroblocks {
            zen_y2_nz += mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
            for blk in &mb.y_blocks {
                zen_y1_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
        }
        for mb in &lib_diag.macroblocks {
            lib_y2_nz += mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
            for blk in &mb.y_blocks {
                lib_y1_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
        }

        println!("\nNonzero coefficient counts:");
        println!(
            "  Y2 (DC): zenwebp={}, libwebp={}, ratio={:.3}x",
            zen_y2_nz,
            lib_y2_nz,
            zen_y2_nz as f64 / lib_y2_nz as f64
        );
        println!(
            "  Y1 (AC): zenwebp={}, libwebp={}, ratio={:.3}x",
            zen_y1_nz,
            lib_y1_nz,
            zen_y1_nz as f64 / lib_y1_nz as f64
        );

        // Count MBs where mode matches vs differs
        let mut mode_match = 0usize;
        let mut mode_differ = 0usize;
        let mut match_coeff_diff = 0i64;
        let mut differ_coeff_diff = 0i64;

        for (zmb, lmb) in zen_diag.macroblocks.iter().zip(lib_diag.macroblocks.iter()) {
            let modes_match = zmb.luma_mode == lmb.luma_mode;

            // Sum coefficient differences
            let mut coeff_diff = 0i64;
            for (zblk, lblk) in zmb.y_blocks.iter().zip(lmb.y_blocks.iter()) {
                for (zl, ll) in zblk.levels.iter().zip(lblk.levels.iter()) {
                    coeff_diff += (zl.abs() as i64) - (ll.abs() as i64);
                }
            }

            if modes_match {
                mode_match += 1;
                match_coeff_diff += coeff_diff;
            } else {
                mode_differ += 1;
                differ_coeff_diff += coeff_diff;
            }
        }

        println!("\nMode agreement:");
        println!(
            "  Matching modes: {} MBs, sum(|zen|-|lib|) coeffs: {:+}",
            mode_match, match_coeff_diff
        );
        println!(
            "  Different modes: {} MBs, sum(|zen|-|lib|) coeffs: {:+}",
            mode_differ, differ_coeff_diff
        );

        println!(
            "\nFile sizes: zenwebp={} libwebp={} ratio={:.3}x",
            zen_webp.len(),
            lib_webp.len(),
            zen_webp.len() as f64 / lib_webp.len() as f64
        );
    }
}

/// Analyze blocks where both encoders chose the same mode
/// This isolates coefficient quantization differences from mode selection
#[test]
fn same_mode_coefficient_analysis() {
    println!("\n=== Same-Mode Coefficient Analysis ===");
    println!("For blocks where both encoders chose the same mode,");
    println!("compare coefficient level distributions.\n");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, 6);
    let lib_webp = encode_libwebp(&rgb, width, height, 75.0, 6);

    let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("VP8");
    let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("VP8");

    let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
    let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

    // For same-mode blocks, compare level distributions
    let mut zen_levels = [0usize; 32];
    let mut lib_levels = [0usize; 32];
    let mut same_mode_blocks = 0usize;
    let mut exact_match_blocks = 0usize;

    for (zmb, lmb) in zen_diag.macroblocks.iter().zip(lib_diag.macroblocks.iter()) {
        if zmb.luma_mode != lmb.luma_mode {
            continue;
        }

        // Same mode - compare Y blocks
        for (zblk, lblk) in zmb.y_blocks.iter().zip(lmb.y_blocks.iter()) {
            same_mode_blocks += 1;
            let mut exact = true;

            for (zl, ll) in zblk.levels.iter().zip(lblk.levels.iter()) {
                let zabs = zl.unsigned_abs() as usize;
                let labs = ll.unsigned_abs() as usize;

                if zabs < 32 {
                    zen_levels[zabs] += 1;
                }
                if labs < 32 {
                    lib_levels[labs] += 1;
                }

                if zl != ll {
                    exact = false;
                }
            }

            if exact {
                exact_match_blocks += 1;
            }
        }
    }

    println!(
        "Same-mode blocks: {}, exact coefficient match: {} ({:.1}%)",
        same_mode_blocks,
        exact_match_blocks,
        100.0 * exact_match_blocks as f64 / same_mode_blocks as f64
    );

    println!("\nLevel distribution (same-mode blocks only):");
    println!("Level | zenwebp | libwebp | Delta");
    println!("------|---------|---------|------");

    for i in 0..16 {
        let zc = zen_levels[i];
        let lc = lib_levels[i];
        if zc > 0 || lc > 0 {
            let delta = zc as i64 - lc as i64;
            let sign = if delta >= 0 { "+" } else { "" };
            println!("  {:>3} | {:>7} | {:>7} | {}{}", i, zc, lc, sign, delta);
        }
    }

    // Show higher levels if present
    let zen_high: usize = zen_levels[16..].iter().sum();
    let lib_high: usize = lib_levels[16..].iter().sum();
    if zen_high > 0 || lib_high > 0 {
        println!(
            " 16+  | {:>7} | {:>7} | {:+}",
            zen_high,
            lib_high,
            zen_high as i64 - lib_high as i64
        );
    }
}

/// Compute bits per nonzero coefficient to isolate entropy coding efficiency
#[test]
fn bits_per_coefficient_analysis() {
    println!("\n=== Bits Per Coefficient Analysis ===");
    println!("Isolating entropy coding efficiency from quantization\n");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    println!("Method | zen_bytes | lib_bytes | zen_nz | lib_nz | zen_bpc | lib_bpc | bpc_ratio");
    println!("-------|-----------|-----------|--------|--------|---------|---------|----------");

    for method in [4u8, 5, 6] {
        let zen_webp = encode_zenwebp(&rgb, width, height, 75.0, method);
        let lib_webp = encode_libwebp(&rgb, width, height, 75.0, method);

        let zen_vp8 = extract_vp8_chunk(&zen_webp).expect("VP8");
        let lib_vp8 = extract_vp8_chunk(&lib_webp).expect("VP8");

        let (_, zen_diag) = Vp8Decoder::decode_diagnostic(zen_vp8).expect("decode");
        let (_, lib_diag) = Vp8Decoder::decode_diagnostic(lib_vp8).expect("decode");

        // Count nonzero coefficients (Y2 + Y1 + UV)
        let mut zen_nz = 0usize;
        let mut lib_nz = 0usize;

        for mb in &zen_diag.macroblocks {
            zen_nz += mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
            for blk in &mb.y_blocks {
                zen_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
            for blk in &mb.uv_blocks {
                zen_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
        }
        for mb in &lib_diag.macroblocks {
            lib_nz += mb.y2_block.levels.iter().filter(|&&l| l != 0).count();
            for blk in &mb.y_blocks {
                lib_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
            for blk in &mb.uv_blocks {
                lib_nz += blk.levels.iter().filter(|&&l| l != 0).count();
            }
        }

        // Bits per coefficient (approximation - includes header overhead)
        let zen_bits = zen_webp.len() * 8;
        let lib_bits = lib_webp.len() * 8;
        let zen_bpc = if zen_nz > 0 {
            zen_bits as f64 / zen_nz as f64
        } else {
            0.0
        };
        let lib_bpc = if lib_nz > 0 {
            lib_bits as f64 / lib_nz as f64
        } else {
            0.0
        };
        let bpc_ratio = zen_bpc / lib_bpc;

        println!(
            "   {}   | {:>9} | {:>9} | {:>6} | {:>6} | {:>7.2} | {:>7.2} | {:>8.3}x",
            method,
            zen_webp.len(),
            lib_webp.len(),
            zen_nz,
            lib_nz,
            zen_bpc,
            lib_bpc,
            bpc_ratio
        );
    }

    println!("\nNote: bpc_ratio > 1.0 means we use more bits per coefficient (less efficient)");
}

#[test]
#[ignore] // Requires dwebp to be installed
fn verify_decode_with_dwebp() {
    println!("\n=== Verifying Decode with dwebp ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    for method in [4u8, 5, 6] {
        let webp = encode_zenwebp(&rgb, width, height, 75.0, method);

        // Save to file
        let webp_path = format!("/tmp/test_m{}.webp", method);
        std::fs::write(&webp_path, &webp).expect("write webp");

        // Try to decode with dwebp
        let output = std::process::Command::new("dwebp")
            .args([&webp_path, "-o", "/tmp/test_decoded.ppm"])
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    println!(
                        "Method {}: dwebp decode SUCCESS ({} bytes)",
                        method,
                        webp.len()
                    );
                } else {
                    println!(
                        "Method {}: dwebp decode FAILED: {}",
                        method,
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
            }
            Err(e) => {
                println!("Method {}: dwebp not available: {}", method, e);
                break;
            }
        }
    }
}

#[test]
fn compare_decoded_pixels() {
    println!("\n=== Comparing Decoded Pixels Between Methods ===");

    let path = "/tmp/CID22/original/792079.png";
    let (rgb, width, height) = match load_png(path) {
        Some(data) => data,
        None => {
            println!("Skipping: {} not found", path);
            return;
        }
    };

    // Encode with m4 and m5
    let webp_m4 = encode_zenwebp(&rgb, width, height, 75.0, 4);
    let webp_m5 = encode_zenwebp(&rgb, width, height, 75.0, 5);

    // Extract VP8 chunks
    let vp8_m4 = extract_vp8_chunk(&webp_m4).expect("VP8");
    let vp8_m5 = extract_vp8_chunk(&webp_m5).expect("VP8");

    // Decode both
    let (frame_m4, _) = Vp8Decoder::decode_diagnostic(vp8_m4).expect("decode m4");
    let (frame_m5, _) = Vp8Decoder::decode_diagnostic(vp8_m5).expect("decode m5");

    // Compare Y planes
    let mut diff_count = 0usize;
    let mut max_diff = 0i32;
    for (y4, y5) in frame_m4.ybuf.iter().zip(frame_m5.ybuf.iter()) {
        let diff = (*y4 as i32 - *y5 as i32).abs();
        if diff > 0 {
            diff_count += 1;
            max_diff = max_diff.max(diff);
        }
    }

    let total = frame_m4.ybuf.len();
    println!(
        "Y plane: {} total pixels, {} differ, max diff = {}",
        total, diff_count, max_diff
    );

    // If most pixels differ significantly, there's a bug
    if diff_count > total / 2 {
        println!("WARNING: More than 50% of pixels differ between m4 and m5!");
        // Check if m5 decoded to garbage
        let m5_avg: f64 = frame_m5.ybuf.iter().map(|&p| p as f64).sum::<f64>() / total as f64;
        println!("M5 average Y value: {:.1}", m5_avg);
    }
}
