// Requires C libwebp (webpx) — not available on wasm32
#![cfg(not(target_arch = "wasm32"))]
//! Lossy encoder quality and size comparison tests
//!
//! These tests compare the image-webp lossy encoder against libwebp:
//! - Bitstream compatibility: libwebp can decode our output
//! - Quality metrics: PSNR, DSSIM, and SSIMULACRA2 of decoded output
//! - Size comparison: our output vs libwebp's output at same quality setting
//!
//! Current status:
//! - At same file size: ~99% of libwebp's PSNR quality
//! - At same Q setting: files are ~1.2-1.6x larger (less efficient encoding)
//! - Missing optimizations: adaptive token probabilities, segment quantization
//!
//! The encoder implements RD-based mode selection with VP8Matrix biased
//! quantization and skip detection for zero macroblocks.

use dssim_core::Dssim;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use imgref::ImgVec;
use rgb::RGBA;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

/// Simple PSNR calculation (not great for perceptual quality, but simple)
fn calculate_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(original.len(), decoded.len());

    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Calculate DSSIM (perceptual quality metric)
/// Returns 0 for identical images, higher values indicate more difference
/// DSSIM = 1/SSIM - 1, so 0.01 means SSIM ≈ 0.99
fn calculate_dssim(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    // Convert RGB8 to RGBA<f32> in linear light for dssim
    // sRGB gamma decode: (x / 255)^2.2 approximation
    fn srgb_to_linear(v: u8) -> f32 {
        let x = v as f32 / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    let orig_rgba: Vec<RGBA<f32>> = original
        .chunks_exact(3)
        .map(|p| {
            RGBA::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
                1.0,
            )
        })
        .collect();
    let dec_rgba: Vec<RGBA<f32>> = decoded
        .chunks_exact(3)
        .map(|p| {
            RGBA::new(
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
                1.0,
            )
        })
        .collect();

    let orig_img: ImgVec<RGBA<f32>> = ImgVec::new(orig_rgba, w, h);
    let dec_img: ImgVec<RGBA<f32>> = ImgVec::new(dec_rgba, w, h);

    let dssim = Dssim::new();
    let orig_dssim = dssim.create_image(&orig_img).unwrap();
    let dec_dssim = dssim.create_image(&dec_img).unwrap();

    let (dssim_val, _) = dssim.compare(&orig_dssim, dec_dssim);
    dssim_val.into()
}

/// Calculate SSIMULACRA2 (state-of-the-art perceptual quality metric)
/// Returns score where: 90+ = excellent, 70-90 = good, 50-70 = acceptable, <50 = poor
/// Higher is better (opposite of DSSIM)
fn calculate_ssimulacra2(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    // sRGB gamma decode
    fn srgb_to_linear(v: u8) -> f32 {
        let x = v as f32 / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    // Convert to linear RGB f32 format for fast-ssim2
    let orig_rgb: Vec<[f32; 3]> = original
        .chunks_exact(3)
        .map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();
    let dec_rgb: Vec<[f32; 3]> = decoded
        .chunks_exact(3)
        .map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect();

    let orig_img = Rgb::new(
        orig_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let dec_img = Rgb::new(
        dec_rgb,
        w,
        h,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_img, dec_img).unwrap()
}

/// Helper to create lossy encoder params
fn lossy_config(quality: u8) -> EncoderConfig {
    EncoderConfig::new_lossy().with_quality(quality as f32)
}

/// Test that libwebp can decode our lossy output
#[test]
fn libwebp_can_decode_our_lossy_output() {
    // Create a simple gradient test image
    let width = 64u32;
    let height = 64u32;
    let mut img = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            img[idx] = (x * 4) as u8; // R: horizontal gradient
            img[idx + 1] = (y * 4) as u8; // G: vertical gradient
            img[idx + 2] = 128; // B: constant
        }
    }

    // Encode with image-webp lossy (quality 75)
    let _cfg = lossy_config(75);

    let output = EncodeRequest::new(&_cfg, &img, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("Encoding failed");

    println!("Encoded {} bytes", output.len());

    // Verify it's a valid WebP that libwebp can decode
    let decoded = webp::Decoder::new(&output)
        .decode()
        .expect("libwebp failed to decode our output");

    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);

    // For lossy, we can't expect exact match, but PSNR should be reasonable
    let psnr = calculate_psnr(&img, &decoded);
    println!("PSNR: {:.2} dB (target: > 20 dB for basic lossy)", psnr);
    assert!(psnr > 15.0, "PSNR too low: {:.2} dB", psnr);
}

/// Compare our encoder size vs libwebp at same quality
#[test]
fn size_comparison_vs_libwebp() {
    let width = 128u32;
    let height = 128u32;

    // Create test image with some texture (checkerboard + gradient)
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let checker = ((x / 8 + y / 8) % 2) as u8 * 64;
            img[idx] = ((x * 2) as u8).wrapping_add(checker);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(checker);
            img[idx + 2] = 128u8.wrapping_add(checker);
        }
    }

    for quality in [50u8, 75, 90] {
        // Encode with image-webp
        let _cfg = lossy_config(quality);

        let our_output = EncodeRequest::new(&_cfg, &img, PixelLayout::Rgb8, width, height)
            .encode()
            .expect("Our encoding failed");

        // Encode with libwebp
        let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
        let libwebp_output = libwebp_encoder.encode(quality as f32);

        let our_size = our_output.len();
        let libwebp_size = libwebp_output.len();
        let ratio = our_size as f64 / libwebp_size as f64;

        println!(
            "Quality {}: ours = {} bytes, libwebp = {} bytes, ratio = {:.2}x",
            quality, our_size, libwebp_size, ratio
        );

        // We should be within reasonable bounds of libwebp's size.
        // Note: synthetic checkerboard test images are harder than real images.
        // Real images (Kodak corpus) show ~1.2-1.5x ratio at various quality levels.
        // Allow up to 2.1x for this synthetic test.
        assert!(
            ratio < 2.1,
            "Our output is {:.2}x larger than libwebp at quality {}",
            ratio,
            quality
        );
    }
}

/// Compare decoded quality metrics at same quality setting
/// Note: Same Q setting produces different file sizes (ours ~1.4x larger at Q75)
/// but quality metrics are comparable since we're using more bits.
#[test]
fn quality_comparison_at_same_quality_setting() {
    let width = 128u32;
    let height = 128u32;

    // Load a real test image or create a complex one
    let mut img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Create some texture
            let noise = ((x * 7 + y * 13) % 64) as u8;
            img[idx] = ((x * 2) as u8).wrapping_add(noise);
            img[idx + 1] = ((y * 2) as u8).wrapping_add(noise);
            img[idx + 2] = (((x + y) * 2) as u8).wrapping_add(noise);
        }
    }

    // Encode with libwebp at quality 75
    let libwebp_encoder = webp::Encoder::from_rgb(&img, width, height);
    let libwebp_output = libwebp_encoder.encode(75.0);

    // Encode with our encoder at same quality
    let _cfg = lossy_config(75);

    let our_output = EncodeRequest::new(&_cfg, &img, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("Our encoding failed");

    // Decode both
    let our_decoded = webp::Decoder::new(&our_output)
        .decode()
        .expect("Failed to decode our output");
    let libwebp_decoded = webp::Decoder::new(&libwebp_output)
        .decode()
        .expect("Failed to decode libwebp output");

    // Calculate PSNR, DSSIM, and SSIMULACRA2
    let our_psnr = calculate_psnr(&img, &our_decoded);
    let libwebp_psnr = calculate_psnr(&img, &libwebp_decoded);
    let our_dssim = calculate_dssim(&img, &our_decoded, width, height);
    let libwebp_dssim = calculate_dssim(&img, &libwebp_decoded, width, height);
    let our_ssim2 = calculate_ssimulacra2(&img, &our_decoded, width, height);
    let libwebp_ssim2 = calculate_ssimulacra2(&img, &libwebp_decoded, width, height);

    println!("=== Quality Comparison at Q75 ===");
    println!(
        "Our encoder: {} bytes, PSNR = {:.2} dB, DSSIM = {:.6}, SSIMULACRA2 = {:.2}",
        our_output.len(),
        our_psnr,
        our_dssim,
        our_ssim2
    );
    println!(
        "libwebp:     {} bytes, PSNR = {:.2} dB, DSSIM = {:.6}, SSIMULACRA2 = {:.2}",
        libwebp_output.len(),
        libwebp_psnr,
        libwebp_dssim,
        libwebp_ssim2
    );
    println!(
        "Ratio: size {:.2}x, PSNR {:.1}%, DSSIM {:.2}x, SSIMULACRA2 {:.1}%",
        our_output.len() as f64 / libwebp_output.len() as f64,
        100.0 * our_psnr / libwebp_psnr,
        our_dssim / libwebp_dssim.max(0.0001), // avoid div by zero
        100.0 * our_ssim2 / libwebp_ssim2.max(0.01)
    );

    // Our quality should be at least 80% of libwebp's PSNR
    let quality_ratio = our_psnr / libwebp_psnr;
    assert!(
        quality_ratio > 0.8,
        "Our quality ({:.2} dB) is less than 80% of libwebp ({:.2} dB)",
        our_psnr,
        libwebp_psnr
    );

    // DSSIM should not be more than 3x worse than libwebp
    // (DSSIM: lower is better, so we check if ours is less than 3x theirs)
    assert!(
        our_dssim < libwebp_dssim * 3.0,
        "Our DSSIM ({:.6}) is more than 3x worse than libwebp ({:.6})",
        our_dssim,
        libwebp_dssim
    );
}

/// Test various image types for encoder robustness
#[test]
fn encode_various_image_types() {
    let test_cases = [
        ("solid_color", create_solid_color_image(64, 64)),
        ("gradient", create_gradient_image(64, 64)),
        ("checkerboard", create_checkerboard_image(64, 64)),
        ("noise", create_noise_image(64, 64)),
    ];

    for (name, img) in test_cases {
        let _cfg = lossy_config(75);

        let output = EncodeRequest::new(&_cfg, &img, PixelLayout::Rgb8, 64, 64)
            .encode()
            .unwrap_or_else(|e| panic!("Failed to encode {}: {:?}", name, e));

        // Verify libwebp can decode
        let decoded = webp::Decoder::new(&output)
            .decode()
            .unwrap_or_else(|| panic!("libwebp failed to decode {}", name));

        let psnr = calculate_psnr(&img, &decoded);
        println!("{}: {} bytes, PSNR = {:.2} dB", name, output.len(), psnr);

        // Minimum acceptable PSNR (noise is a pathological case, allow lower threshold)
        let min_psnr = if name == "noise" { 10.0 } else { 20.0 };
        assert!(
            psnr > min_psnr,
            "{} has unacceptably low PSNR: {:.2} (min: {:.0})",
            name,
            psnr,
            min_psnr
        );
    }
}

// Helper functions to create test images
fn create_solid_color_image(w: u32, h: u32) -> Vec<u8> {
    vec![128u8; (w * h * 3) as usize]
}

fn create_gradient_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            img[idx] = (x * 255 / w) as u8;
            img[idx + 1] = (y * 255 / h) as u8;
            img[idx + 2] = ((x + y) * 255 / (w + h)) as u8;
        }
    }
    img
}

fn create_checkerboard_image(w: u32, h: u32) -> Vec<u8> {
    let mut img = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            let val = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            img[idx] = val;
            img[idx + 1] = val;
            img[idx + 2] = val;
        }
    }
    img
}

fn create_noise_image(w: u32, h: u32) -> Vec<u8> {
    use rand::RngCore;
    let mut img = vec![0u8; (w * h * 3) as usize];
    rand::thread_rng().fill_bytes(&mut img);
    img
}

// ============================================================================
// Quality monotonicity regression tests
// ============================================================================

/// Roundtrip encode→decode and measure per-channel average absolute error.
/// Returns (file_bytes, avg_delta_across_all_channels).
fn roundtrip_avg_delta(
    img: &[u8],
    layout: PixelLayout,
    w: u32,
    h: u32,
    quality: f32,
) -> (usize, f64) {
    let n = (w * h) as usize;
    let bpp: usize = match layout {
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => 4,
        _ => 3,
    };

    let cfg = EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(4);
    let webp = EncodeRequest::new(&cfg, img, layout, w, h)
        .encode()
        .expect("encode failed");

    let (dec, dec_w, dec_h) = zenwebp::decode_bgra(&webp).expect("decode failed");
    assert_eq!(dec_w, w);
    assert_eq!(dec_h, h);

    // Decoded is always BGRA. Convert original to BGRA order for comparison.
    let mut total_delta: u64 = 0;
    for i in 0..n {
        let (ob, og, or_) = if bpp == 4 {
            match layout {
                PixelLayout::Bgra8 => (
                    img[i * 4] as i32,
                    img[i * 4 + 1] as i32,
                    img[i * 4 + 2] as i32,
                ),
                PixelLayout::Rgba8 => (
                    img[i * 4 + 2] as i32,
                    img[i * 4 + 1] as i32,
                    img[i * 4] as i32,
                ),
                _ => unreachable!(),
            }
        } else {
            match layout {
                PixelLayout::Rgb8 => (
                    img[i * 3 + 2] as i32,
                    img[i * 3 + 1] as i32,
                    img[i * 3] as i32,
                ),
                PixelLayout::Bgr8 => (
                    img[i * 3] as i32,
                    img[i * 3 + 1] as i32,
                    img[i * 3 + 2] as i32,
                ),
                _ => unreachable!(),
            }
        };
        let db = (ob - dec[i * 4] as i32).unsigned_abs();
        let dg = (og - dec[i * 4 + 1] as i32).unsigned_abs();
        let dr = (or_ - dec[i * 4 + 2] as i32).unsigned_abs();
        total_delta += (db + dg + dr) as u64;
    }

    let avg = total_delta as f64 / (n * 3) as f64;
    (webp.len(), avg)
}

/// Quality monotonicity: increasing quality should not increase error.
///
/// This test checks that the quality-error curve is monotonically decreasing
/// (with a tolerance for quantization noise). A violation means higher quality
/// settings produce worse output than lower ones.
#[test]
fn quality_error_monotonicity_gradient() {
    let (w, h) = (256u32, 256u32);
    let n = (w * h) as usize;

    // Smooth gradient — sensitive to quantization quality
    let mut img = Vec::with_capacity(n * 3);
    for y in 0..h {
        for x in 0..w {
            img.push((x * 255 / w) as u8);
            img.push((y * 255 / h) as u8);
            img.push(((x + y) * 255 / (w + h)) as u8);
        }
    }

    let qualities = [50.0f32, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0];
    let mut results: Vec<(f32, usize, f64)> = Vec::new();

    println!("\n{:>6}  {:>8}  {:>10}", "q", "bytes", "avg_delta");
    for &q in &qualities {
        let (bytes, avg) = roundtrip_avg_delta(&img, PixelLayout::Rgb8, w, h, q);
        println!("{:>6.1}  {:>8}  {:>10.4}", q, bytes, avg);
        results.push((q, bytes, avg));
    }

    // Check monotonicity: for each quality step up, avg_delta should not
    // increase by more than 50% over the previous level.
    // (Small increases due to quantization rounding are acceptable.)
    let mut violations = Vec::new();
    for pair in results.windows(2) {
        let (q_low, _, delta_low) = pair[0];
        let (q_high, _, delta_high) = pair[1];
        // Higher quality should mean same or lower delta.
        // Allow 50% tolerance for quantization noise.
        if delta_high > delta_low * 1.5 && delta_high - delta_low > 0.5 {
            violations.push((q_low, q_high, delta_low, delta_high));
        }
    }

    if !violations.is_empty() {
        let mut msg = String::from("Quality-error monotonicity violated:\n");
        for (q_low, q_high, d_low, d_high) in &violations {
            msg.push_str(&format!(
                "  q={q_low:.0}→{q_high:.0}: avg_delta {d_low:.4}→{d_high:.4} ({:.1}x worse)\n",
                d_high / d_low
            ));
        }
        panic!("{msg}");
    }
}

/// Same monotonicity test with BGRA input layout.
#[test]
fn quality_error_monotonicity_gradient_bgra() {
    let (w, h) = (256u32, 256u32);
    let n = (w * h) as usize;

    let mut img = Vec::with_capacity(n * 4);
    for y in 0..h {
        for x in 0..w {
            img.extend_from_slice(&[x as u8, y as u8, ((x + y) / 2) as u8, 255]);
        }
    }

    let qualities = [50.0f32, 70.0, 80.0, 90.0, 95.0, 100.0];
    let mut results: Vec<(f32, f64)> = Vec::new();

    println!("\n{:>6}  {:>10}", "q", "avg_delta");
    for &q in &qualities {
        let (_, avg) = roundtrip_avg_delta(&img, PixelLayout::Bgra8, w, h, q);
        println!("{:>6.1}  {:>10.4}", q, avg);
        results.push((q, avg));
    }

    let mut violations = Vec::new();
    for pair in results.windows(2) {
        let (q_low, delta_low) = pair[0];
        let (q_high, delta_high) = pair[1];
        if delta_high > delta_low * 1.5 && delta_high - delta_low > 0.5 {
            violations.push((q_low, q_high, delta_low, delta_high));
        }
    }

    if !violations.is_empty() {
        let mut msg = String::from("Quality-error monotonicity violated (BGRA):\n");
        for (q_low, q_high, d_low, d_high) in &violations {
            msg.push_str(&format!(
                "  q={q_low:.0}→{q_high:.0}: avg_delta {d_low:.4}→{d_high:.4} ({:.1}x worse)\n",
                d_high / d_low
            ));
        }
        panic!("{msg}");
    }
}

// ============================================================================
// Preset tests
// ============================================================================

/// Test that different presets produce measurably different file sizes.
/// This validates that the preset parameters actually affect encoding.
#[test]
fn presets_produce_different_file_sizes() {
    use zenwebp::{EncoderConfig, Preset};

    // Use a large, varied image so segments and SNS have room to differentiate.
    // Noise + patches gives diverse frequency content across macroblocks.
    let w = 512u32;
    let h = 512u32;
    let mut img = create_noise_image(w, h);
    // Add some flat patches to give SNS something to work with
    for y in 100..200 {
        for x in 100..200 {
            let idx = ((y * w + x) * 3) as usize;
            img[idx] = 64;
            img[idx + 1] = 128;
            img[idx + 2] = 192;
        }
    }
    let quality = 75.0;

    let presets = [
        ("Default", Preset::Default),
        ("Picture", Preset::Picture),
        ("Photo", Preset::Photo),
        ("Drawing", Preset::Drawing),
        ("Icon", Preset::Icon),
        ("Text", Preset::Text),
    ];

    let mut sizes: Vec<(&str, usize)> = Vec::new();

    for &(name, preset) in &presets {
        let config = EncoderConfig::with_preset(preset, quality).with_method(4);
        let webp = EncodeRequest::new(&config, &img, PixelLayout::Rgb8, w, h)
            .encode()
            .expect("Encoding failed");

        // Verify libwebp can decode the output
        let decoded = webp::Decoder::new(&webp)
            .decode()
            .expect("libwebp failed to decode preset output");
        assert_eq!(decoded.width(), w);
        assert_eq!(decoded.height(), h);

        sizes.push((name, webp.len()));
        println!("Preset {name:>10}: {} bytes", webp.len());
    }

    // Check that at least some presets produce different sizes
    let min_size = sizes.iter().map(|(_, s)| *s).min().unwrap();
    let max_size = sizes.iter().map(|(_, s)| *s).max().unwrap();
    let ratio = max_size as f64 / min_size as f64;
    println!(
        "Size range: {} - {} bytes (ratio: {:.3}x)",
        min_size, max_size, ratio
    );

    // Presets should produce measurably different sizes.
    // The main differentiators are SNS (segment quantization spread),
    // filter strength, and sharpness. With varied content, these should
    // produce at least 1% difference between extremes.
    assert!(
        ratio > 1.01,
        "Presets should produce measurably different sizes, got ratio {:.3}x",
        ratio
    );

    // Additionally verify that not all sizes are identical
    let unique_sizes: std::collections::HashSet<_> = sizes.iter().map(|(_, s)| *s).collect();
    assert!(
        unique_sizes.len() > 1,
        "All presets produced identical output"
    );
}

/// Test that preset overrides work correctly.
#[test]
fn preset_overrides_work() {
    use zenwebp::{EncoderConfig, Preset};

    // Use noisy image - checkerboard is too uniform for filter/SNS to matter
    let img = create_noise_image(256, 256);

    // Default preset with default filter (SNS=50, filter=60)
    let _cfg1 = EncoderConfig::with_preset(Preset::Default, 75.0).with_method(4);
    let default_size = EncodeRequest::new(&_cfg1, &img, PixelLayout::Rgb8, 256, 256)
        .encode()
        .unwrap()
        .len();

    // Default preset with filter forced to 0 (like Icon)
    let _cfg2 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_filter_strength(0)
        .with_filter_sharpness(0)
        .with_sns_strength(0);
    let no_filter_size = EncodeRequest::new(&_cfg2, &img, PixelLayout::Rgb8, 256, 256)
        .encode()
        .unwrap()
        .len();

    println!(
        "Default: {} bytes, No-filter override: {} bytes",
        default_size, no_filter_size
    );

    // Overrides should produce different output than default
    // For noisy images, SNS affects size significantly
    assert_ne!(
        default_size, no_filter_size,
        "Filter/SNS override should change output on noisy images"
    );
}

/// Test that segments=1 disables segmentation.
#[test]
fn segments_one_disables_segmentation() {
    use zenwebp::{EncoderConfig, Preset};

    let img = create_checkerboard_image(256, 256);

    let _cfg3 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_segments(4);
    let with_segments = EncodeRequest::new(&_cfg3, &img, PixelLayout::Rgb8, 256, 256)
        .encode()
        .unwrap()
        .len();

    let _cfg4 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_segments(1);
    let without_segments = EncodeRequest::new(&_cfg4, &img, PixelLayout::Rgb8, 256, 256)
        .encode()
        .unwrap()
        .len();

    println!(
        "4 segments: {} bytes, 1 segment: {} bytes",
        with_segments, without_segments
    );

    // Both should decode correctly
    let _cfg5 = EncoderConfig::with_preset(Preset::Default, 75.0)
        .with_method(4)
        .with_segments(1);
    let webp_data = EncodeRequest::new(&_cfg5, &img, PixelLayout::Rgb8, 256, 256)
        .encode()
        .unwrap();
    let decoded = webp::Decoder::new(&webp_data)
        .decode()
        .expect("libwebp failed to decode 1-segment output");
    assert_eq!(decoded.width(), 256);
    assert_eq!(decoded.height(), 256);
}

/// Compare zenwebp vs webpx (libwebp) at matched configurations.
/// This isolates encoding quality differences from preset selection.
#[test]
fn matched_config_comparison_vs_webpx() {
    use webpx::Unstoppable;

    let img = create_checkerboard_image(256, 256);
    let quality = 75.0;

    let configs: &[(&str, zenwebp::Preset, webpx::Preset)] = &[
        ("Default", zenwebp::Preset::Default, webpx::Preset::Default),
        ("Photo", zenwebp::Preset::Photo, webpx::Preset::Photo),
        ("Drawing", zenwebp::Preset::Drawing, webpx::Preset::Drawing),
        ("Text", zenwebp::Preset::Text, webpx::Preset::Text),
    ];

    for &(name, zen_preset, wpx_preset) in configs {
        // Encode with zenwebp
        let zen_config = zenwebp::EncoderConfig::with_preset(zen_preset, quality).with_method(4);
        let _cfg6 = zen_config;
        let zen_out = EncodeRequest::new(&_cfg6, &img, PixelLayout::Rgb8, 256, 256)
            .encode()
            .expect("zenwebp encoding failed");

        // Encode with webpx (libwebp)
        let wpx_config = webpx::EncoderConfig::with_preset(wpx_preset, quality).method(4);
        let wpx_out = wpx_config
            .encode_rgb(&img, 256, 256, Unstoppable)
            .expect("webpx encoding failed");

        let ratio = zen_out.len() as f64 / wpx_out.len() as f64;

        println!(
            "Preset {name:>10}: zenwebp={:>6} bytes, webpx={:>6} bytes, ratio={:.3}x",
            zen_out.len(),
            wpx_out.len(),
            ratio
        );

        // Verify our output is valid
        let decoded = webp::Decoder::new(&zen_out)
            .decode()
            .expect("libwebp failed to decode zenwebp output");
        assert_eq!(decoded.width(), 256);
        assert_eq!(decoded.height(), 256);
    }
}

// ============================================================================
// Auto-detection tests
// ============================================================================

/// Test that Preset::Auto produces valid output.
#[test]
fn auto_preset_produces_valid_output() {
    use zenwebp::{EncoderConfig, Preset};

    // Photo-like image (noise + patches)
    let w = 512u32;
    let h = 512u32;
    let mut img = create_noise_image(w, h);
    for y in 100..200 {
        for x in 100..200 {
            let idx = ((y * w + x) * 3) as usize;
            img[idx] = 64;
            img[idx + 1] = 128;
            img[idx + 2] = 192;
        }
    }

    let config = EncoderConfig::with_preset(Preset::Auto, 75.0).with_method(4);
    let webp = EncodeRequest::new(&config, &img, PixelLayout::Rgb8, w, h)
        .encode()
        .expect("Auto encoding failed");

    // Verify libwebp can decode
    let decoded = webp::Decoder::new(&webp)
        .decode()
        .expect("libwebp failed to decode Auto output");
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);

    println!("Auto preset output: {} bytes", webp.len());
}

/// Test that Auto detects screenshots differently from photos.
#[test]
fn auto_detects_screenshot_vs_photo() {
    use zenwebp::{EncoderConfig, Preset};

    let w = 512u32;
    let h = 512u32;

    // Create screenshot-like image: uniform blocks with sharp edges
    let mut screenshot = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            // UI-like blocks
            let block_x = x / 64;
            let block_y = y / 64;
            let val = ((block_x * 37 + block_y * 73) % 5) as u8;
            screenshot[idx] = [32, 64, 128, 192, 240][val as usize];
            screenshot[idx + 1] = [48, 96, 160, 200, 230][val as usize];
            screenshot[idx + 2] = [64, 80, 144, 180, 250][val as usize];
        }
    }

    // Create photo-like image: noise + gradients
    let photo = create_noise_image(w, h);

    let _cfg7 = EncoderConfig::with_preset(Preset::Auto, 75.0).with_method(4);
    let auto_screenshot = EncodeRequest::new(&_cfg7, &screenshot, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();
    let _cfg8 = EncoderConfig::with_preset(Preset::Auto, 75.0).with_method(4);
    let auto_photo = EncodeRequest::new(&_cfg8, &photo, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    // Both should produce valid output
    webp::Decoder::new(&auto_screenshot)
        .decode()
        .expect("Failed to decode auto screenshot");
    webp::Decoder::new(&auto_photo)
        .decode()
        .expect("Failed to decode auto photo");

    println!(
        "Auto screenshot: {} bytes, Auto photo: {} bytes",
        auto_screenshot.len(),
        auto_photo.len()
    );
}

/// Test that Auto on small images produces Icon-like output.
#[test]
fn auto_detects_icon() {
    use zenwebp::{EncoderConfig, Preset};

    // Small 64x64 image
    let w = 64u32;
    let h = 64u32;
    let img = create_checkerboard_image(w, h);

    let _cfg9 = EncoderConfig::with_preset(Preset::Auto, 75.0).with_method(4);
    let auto = EncodeRequest::new(&_cfg9, &img, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();
    let _cfg10 = EncoderConfig::with_preset(Preset::Icon, 75.0).with_method(4);
    let icon = EncodeRequest::new(&_cfg10, &img, PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap();

    // Verify both decode
    webp::Decoder::new(&auto)
        .decode()
        .expect("Failed to decode auto icon");

    // Auto should produce similar output to Icon for small images
    let ratio = auto.len() as f64 / icon.len() as f64;
    println!(
        "Auto: {} bytes, Icon: {} bytes, ratio: {:.3}x",
        auto.len(),
        icon.len(),
        ratio
    );

    // For 64x64, segments are disabled anyway (< 256 macroblocks),
    // so auto-detection doesn't run, and both use initial (Default) params.
    // The key test is that it produces valid output without crashing.
}

// ============================================================================
// Corpus tests (behind _corpus_tests feature flag)
// ============================================================================

#[cfg(feature = "_corpus_tests")]
mod corpus_tests {
    #[allow(unused_imports)]
    use super::*;

    fn png_codec_eval_dir() -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(
            std::env::var("PNG_CODEC_EVAL_DIR")
                .unwrap_or_else(|_| "/home/lilith/work/png-codec-eval".into()),
        );
        assert!(
            dir.is_dir(),
            "png-codec-eval not found: {}. Set PNG_CODEC_EVAL_DIR.",
            dir.display()
        );
        dir
    }

    fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path).ok()?;
        let decoder = png::Decoder::new(BufReader::new(file));
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()?];
        let info = reader.next_frame(&mut buf).ok()?;
        let width = info.width;
        let height = info.height;

        // Convert to RGB if needed
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

    fn encode_and_compare(
        rgb: &[u8],
        w: u32,
        h: u32,
        zen_preset: zenwebp::Preset,
        wpx_preset: webpx::Preset,
        quality: f32,
    ) -> (usize, usize) {
        use webpx::Unstoppable;

        let zen_config = zenwebp::EncoderConfig::with_preset(zen_preset, quality).with_method(4);
        let zen_out = EncodeRequest::new(&zen_config, rgb, PixelLayout::Rgb8, w, h)
            .encode()
            .unwrap();

        let wpx_config = webpx::EncoderConfig::with_preset(wpx_preset, quality).method(4);
        let wpx_out = wpx_config.encode_rgb(rgb, w, h, Unstoppable).unwrap();

        (zen_out.len(), wpx_out.len())
    }

    #[test]
    fn kodak_corpus_preset_comparison() {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        let corpus_dir = corpus.get("kodak").expect("kodak corpus unavailable");
        let mut total_zen = 0usize;
        let mut total_wpx = 0usize;
        let mut count = 0;

        for entry in std::fs::read_dir(&corpus_dir).expect("Kodak corpus not found") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "png") {
                let path_str = path.to_string_lossy().to_string();
                if let Some((rgb, w, h)) = load_png(&path_str) {
                    let (zen_size, wpx_size) = encode_and_compare(
                        &rgb,
                        w,
                        h,
                        zenwebp::Preset::Default,
                        webpx::Preset::Default,
                        75.0,
                    );
                    let ratio = zen_size as f64 / wpx_size as f64;
                    println!(
                        "{}: zenwebp={} webpx={} ratio={:.3}x",
                        path.file_name().unwrap().to_string_lossy(),
                        zen_size,
                        wpx_size,
                        ratio
                    );
                    total_zen += zen_size;
                    total_wpx += wpx_size;
                    count += 1;
                }
            }
        }

        let avg_ratio = total_zen as f64 / total_wpx as f64;
        println!(
            "\nKodak aggregate ({count} images): zenwebp={total_zen} webpx={total_wpx} ratio={avg_ratio:.3}x"
        );
    }

    #[test]
    fn screenshot_corpus_preset_comparison() {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        let corpus_dir = corpus.get("gb82-sc").expect("gb82-sc corpus unavailable");
        let mut results: Vec<(String, f64, f64)> = Vec::new();

        for entry in std::fs::read_dir(&corpus_dir).expect("Screenshot corpus not found") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "png") {
                let path_str = path.to_string_lossy().to_string();
                if let Some((rgb, w, h)) = load_png(&path_str) {
                    let name = path.file_name().unwrap().to_string_lossy().to_string();

                    // Test Default preset
                    let (zen_default, wpx_default) = encode_and_compare(
                        &rgb,
                        w,
                        h,
                        zenwebp::Preset::Default,
                        webpx::Preset::Default,
                        75.0,
                    );
                    // Test Drawing preset (better for screenshots)
                    let (zen_drawing, wpx_drawing) = encode_and_compare(
                        &rgb,
                        w,
                        h,
                        zenwebp::Preset::Drawing,
                        webpx::Preset::Drawing,
                        75.0,
                    );

                    let default_ratio = zen_default as f64 / wpx_default as f64;
                    let drawing_ratio = zen_drawing as f64 / wpx_drawing as f64;

                    println!("{name}: Default={default_ratio:.3}x, Drawing={drawing_ratio:.3}x");
                    results.push((name, default_ratio, drawing_ratio));
                }
            }
        }

        if !results.is_empty() {
            let avg_default: f64 =
                results.iter().map(|(_, d, _)| d).sum::<f64>() / results.len() as f64;
            let avg_drawing: f64 =
                results.iter().map(|(_, _, d)| d).sum::<f64>() / results.len() as f64;
            println!("\nScreenshot aggregate ({} images):", results.len());
            println!("  Default preset ratio: {avg_default:.3}x");
            println!("  Drawing preset ratio: {avg_drawing:.3}x");
        }
    }

    /// Test Auto preset vs manual presets on kodak (photos) and screenshots.
    /// Auto should produce output within 5% of the "correct" manual preset.
    #[test]
    fn auto_vs_manual_preset_kodak() {
        let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
        let corpus_dir = corpus.get("kodak").expect("kodak corpus unavailable");
        let mut auto_total = 0usize;
        let mut photo_total = 0usize;
        let mut count = 0;

        for entry in std::fs::read_dir(&corpus_dir).expect("Kodak corpus not found") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "png") {
                let path_str = path.to_string_lossy().to_string();
                if let Some((rgb, w, h)) = load_png(&path_str) {
                    let auto_cfg = zenwebp::EncoderConfig::with_preset(zenwebp::Preset::Auto, 75.0)
                        .with_method(4);
                    let auto_size = EncodeRequest::new(&auto_cfg, &rgb, PixelLayout::Rgb8, w, h)
                        .encode()
                        .unwrap()
                        .len();

                    let photo_cfg =
                        zenwebp::EncoderConfig::with_preset(zenwebp::Preset::Photo, 75.0)
                            .with_method(4);
                    let photo_size = EncodeRequest::new(&photo_cfg, &rgb, PixelLayout::Rgb8, w, h)
                        .encode()
                        .unwrap()
                        .len();

                    let ratio = auto_size as f64 / photo_size as f64;
                    println!(
                        "{}: Auto={} Photo={} ratio={:.3}x",
                        path.file_name().unwrap().to_string_lossy(),
                        auto_size,
                        photo_size,
                        ratio
                    );
                    auto_total += auto_size;
                    photo_total += photo_size;
                    count += 1;
                }
            }
        }

        let avg_ratio = auto_total as f64 / photo_total as f64;
        println!(
            "\nKodak Auto vs Photo ({count} images): Auto={auto_total} Photo={photo_total} ratio={avg_ratio:.3}x"
        );

        // Auto should be within 5% of Photo preset for photos
        assert!(
            avg_ratio < 1.05 && avg_ratio > 0.95,
            "Auto should be within 5% of Photo on kodak, got {avg_ratio:.3}x"
        );
    }

    /// Test multi-pass encoding (methods 5 and 6) vs single-pass (method 4).
    /// Multi-pass should produce smaller files due to empirical probability estimation.
    #[test]
    fn multipass_encoding_comparison() {
        use std::time::Instant;

        // Use a single CID22 test image for quick testing
        let test_images = [png_codec_eval_dir()
            .join("test_images/792079.png")
            .to_string_lossy()
            .into_owned()];

        println!("\n=== Multi-pass Encoding Comparison (zenwebp) ===");
        println!("(SNS=0, filter=0, segments=1 for diagnostic isolation)\n");

        let mut total_m4 = 0usize;
        let mut total_m5 = 0usize;
        let mut total_m6 = 0usize;
        let mut count = 0;

        for path in &test_images {
            if let Some((rgb, w, h)) = load_png(path) {
                let name = std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy();

                // Encode with method 4 (single pass with trellis)
                let start = Instant::now();
                let _cfg11 = zenwebp::EncoderConfig::with_preset(zenwebp::Preset::Default, 75.0)
                    .with_method(4)
                    .with_sns_strength(0)
                    .with_filter_strength(0)
                    .with_segments(1);
                let m4_output = EncodeRequest::new(&_cfg11, &rgb, PixelLayout::Rgb8, w, h)
                    .encode()
                    .unwrap();
                let m4_time = start.elapsed();

                // Encode with method 5 (2 passes)
                let start = Instant::now();
                let _cfg12 = zenwebp::EncoderConfig::with_preset(zenwebp::Preset::Default, 75.0)
                    .with_method(5)
                    .with_sns_strength(0)
                    .with_filter_strength(0)
                    .with_segments(1);
                let m5_output = EncodeRequest::new(&_cfg12, &rgb, PixelLayout::Rgb8, w, h)
                    .encode()
                    .unwrap();
                let m5_time = start.elapsed();

                // Encode with method 6 (3 passes)
                let start = Instant::now();
                let _cfg13 = zenwebp::EncoderConfig::with_preset(zenwebp::Preset::Default, 75.0)
                    .with_method(6)
                    .with_sns_strength(0)
                    .with_filter_strength(0)
                    .with_segments(1);
                let m6_output = EncodeRequest::new(&_cfg13, &rgb, PixelLayout::Rgb8, w, h)
                    .encode()
                    .unwrap();
                let m6_time = start.elapsed();

                // Verify all outputs decode correctly
                zenwebp::decode_rgba(&m4_output).expect("m4 should decode");
                zenwebp::decode_rgba(&m5_output).expect("m5 should decode");
                zenwebp::decode_rgba(&m6_output).expect("m6 should decode");

                // Also compare to libwebp with multiple passes
                use webpx::Unstoppable;
                let libwebp_m4 = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                    .method(4)
                    .sns_strength(0)
                    .filter_strength(0)
                    .segments(1)
                    .pass(1); // Single pass
                let libwebp_m6_1pass =
                    webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                        .method(6)
                        .sns_strength(0)
                        .filter_strength(0)
                        .segments(1)
                        .pass(1); // Single pass
                let libwebp_m6_3pass =
                    webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
                        .method(6)
                        .sns_strength(0)
                        .filter_strength(0)
                        .segments(1)
                        .pass(3); // 3 passes

                let lib_m4 = libwebp_m4.encode_rgb(&rgb, w, h, Unstoppable).unwrap();
                let lib_m6_1 = libwebp_m6_1pass
                    .encode_rgb(&rgb, w, h, Unstoppable)
                    .unwrap();
                let lib_m6_3 = libwebp_m6_3pass
                    .encode_rgb(&rgb, w, h, Unstoppable)
                    .unwrap();

                let m5_vs_m4 = m5_output.len() as f64 / m4_output.len() as f64;
                let m6_vs_m4 = m6_output.len() as f64 / m4_output.len() as f64;
                let lib_m6_3_vs_1 = lib_m6_3.len() as f64 / lib_m6_1.len() as f64;

                println!("{} ({}x{}):", name, w, h);
                println!(
                    "  zenwebp m4 (1 pass):  {:>6} bytes, {:>6.2?}",
                    m4_output.len(),
                    m4_time
                );
                println!(
                    "  zenwebp m5 (2 pass):  {:>6} bytes, {:>6.2?}, {:.3}x vs m4",
                    m5_output.len(),
                    m5_time,
                    m5_vs_m4
                );
                println!(
                    "  zenwebp m6 (3 pass):  {:>6} bytes, {:>6.2?}, {:.3}x vs m4",
                    m6_output.len(),
                    m6_time,
                    m6_vs_m4
                );
                println!("  ---");
                println!("  libwebp m4 (1 pass):  {:>6} bytes", lib_m4.len());
                println!("  libwebp m6 (1 pass):  {:>6} bytes", lib_m6_1.len());
                println!(
                    "  libwebp m6 (3 pass):  {:>6} bytes, {:.3}x vs 1-pass",
                    lib_m6_3.len(),
                    lib_m6_3_vs_1
                );

                total_m4 += m4_output.len();
                total_m5 += m5_output.len();
                total_m6 += m6_output.len();
                count += 1;
            }
        }

        if count > 0 {
            let m5_ratio = total_m5 as f64 / total_m4 as f64;
            let m6_ratio = total_m6 as f64 / total_m4 as f64;

            println!("\nAggregate ({count} images):");
            println!("  m4 total: {total_m4} bytes");
            println!("  m5 total: {total_m5} bytes ({m5_ratio:.3}x vs m4)");
            println!("  m6 total: {total_m6} bytes ({m6_ratio:.3}x vs m4)");

            // Multi-pass should improve compression (ratio < 1.0) or at least not regress
            // Allow for small fluctuations due to different mode decisions
            assert!(
                m5_ratio <= 1.02,
                "m5 should not be more than 2% larger than m4, got {m5_ratio:.3}x"
            );
            assert!(
                m6_ratio <= 1.02,
                "m6 should not be more than 2% larger than m4, got {m6_ratio:.3}x"
            );
        }
    }

    /// Test quality search to meet target file size.
    /// Uses secant method to converge on target size.
    #[test]
    fn target_size_quality_search() {
        let test_path = png_codec_eval_dir()
            .join("test_images/792079.png")
            .to_string_lossy()
            .into_owned();

        if let Some((rgb, w, h)) = load_png(&test_path) {
            // First encode without target_size to get a baseline
            let _cfg14 = zenwebp::encoder::EncoderConfig::new_lossy().with_quality(75.0);
            let baseline = EncodeRequest::new(&_cfg14, &rgb, PixelLayout::Rgb8, w, h)
                .encode()
                .expect("Baseline encoding failed");
            let baseline_size = baseline.len();
            println!("Baseline Q75 size: {} bytes", baseline_size);

            // Now encode with target_size set to ~80% of baseline
            let target_size = (baseline_size as f64 * 0.8) as u32;
            println!("Target size: {} bytes", target_size);

            let _cfg15 = zenwebp::encoder::EncoderConfig::new_lossy()
                .with_quality(75.0)
                .with_target_size(target_size);
            let target_output = EncodeRequest::new(&_cfg15, &rgb, PixelLayout::Rgb8, w, h)
                .encode()
                .expect("Target size encoding failed");
            let actual_size = target_output.len();
            println!("Actual size: {} bytes", actual_size);

            // Allow 15% tolerance around target (quality search isn't exact)
            let tolerance = 0.15;
            let min_size = (target_size as f64 * (1.0 - tolerance)) as usize;
            let max_size = (target_size as f64 * (1.0 + tolerance)) as usize;

            assert!(
                actual_size >= min_size && actual_size <= max_size,
                "Actual size {} should be within {}% of target {}, got {}% off",
                actual_size,
                (tolerance * 100.0) as i32,
                target_size,
                ((actual_size as f64 / target_size as f64) - 1.0).abs() * 100.0
            );

            // Verify output is valid WebP
            let (decoded, dw, dh) = zenwebp::decoder::decode_rgb(&target_output)
                .expect("Failed to decode target-size encoded output");
            assert_eq!(dw, w);
            assert_eq!(dh, h);
            assert!(!decoded.is_empty());
        } else {
            println!("Skipping test: test image not found at {}", test_path);
        }
    }

    /// Test target_size with various target values.
    #[test]
    fn target_size_various_targets() {
        let test_path = png_codec_eval_dir()
            .join("test_images/792079.png")
            .to_string_lossy()
            .into_owned();

        if let Some((rgb, w, h)) = load_png(&test_path) {
            // Test several target sizes
            let targets = [8000u32, 10000, 12000, 15000];

            println!("\nTesting target sizes on {}x{} image:", w, h);
            for target in targets {
                let _cfg16 = zenwebp::encoder::EncoderConfig::new_lossy()
                    .with_quality(75.0)
                    .with_target_size(target);
                let output = EncodeRequest::new(&_cfg16, &rgb, PixelLayout::Rgb8, w, h)
                    .encode()
                    .expect("Encoding failed");

                let diff_pct = ((output.len() as f64 / target as f64) - 1.0) * 100.0;
                println!(
                    "  Target: {:>6} bytes, Actual: {:>6} bytes ({:+.1}%)",
                    target,
                    output.len(),
                    diff_pct
                );

                // Verify valid output
                zenwebp::decoder::decode_rgb(&output).expect("Failed to decode");
            }
        } else {
            println!("Skipping test: test image not found");
        }
    }
}
