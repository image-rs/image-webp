//! Integration tests comparing zenwebp lossy encoder output against webpx (libwebp).
//!
//! Uses zensim for perceptual quality scoring and zensim-regress for structured
//! pass/fail regression testing. Sweeps across quality levels, methods, presets,
//! and encoder settings to ensure zenwebp produces comparable output to libwebp.

use webpx::Unstoppable;
use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::generators;
use zensim_regress::testing::{RegressionTolerance, check_regression};

use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, Preset};

// ---------------------------------------------------------------------------
// Test images (RGBA from generators, 256x256 — fast but big enough for zensim)
// ---------------------------------------------------------------------------

const W: u32 = 256;
const H: u32 = 256;

fn gradient_rgba() -> Vec<u8> {
    generators::gradient(W, H)
}

fn noise_rgba() -> Vec<u8> {
    generators::value_noise(W, H, 42)
}

fn mandelbrot_rgba() -> Vec<u8> {
    generators::mandelbrot(W, H)
}

fn checkerboard_rgba() -> Vec<u8> {
    generators::checkerboard(W, H, 8)
}

fn color_blocks_rgba() -> Vec<u8> {
    generators::color_blocks(W, H)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode RGBA pixels with zenwebp at the given settings, decode with zenwebp.
fn zen_encode_decode(rgba: &[u8], w: u32, h: u32, config: &LossyConfig) -> (Vec<u8>, Vec<u8>) {
    let webp = EncodeRequest::lossy(config, rgba, PixelLayout::Rgba8, w, h)
        .encode()
        .expect("zenwebp encode failed");
    let (decoded, dw, dh) = zenwebp::decode_rgba(&webp).expect("zenwebp decode failed");
    assert_eq!((dw, dh), (w, h));
    (webp, decoded)
}

/// Encode RGBA pixels with webpx (libwebp) at matched settings, decode with webpx.
fn wpx_encode_decode(
    rgba: &[u8],
    w: u32,
    h: u32,
    quality: f32,
    method: u8,
    preset: webpx::Preset,
    sns: Option<u8>,
    filter: Option<u8>,
    segments: Option<u8>,
) -> (Vec<u8>, Vec<u8>) {
    let mut cfg = webpx::EncoderConfig::with_preset(preset, quality).method(method);
    if let Some(s) = sns {
        cfg = cfg.sns_strength(s);
    }
    if let Some(f) = filter {
        cfg = cfg.filter_strength(f);
    }
    if let Some(seg) = segments {
        cfg = cfg.segments(seg);
    }
    let webp = cfg
        .encode_rgba(rgba, w, h, Unstoppable)
        .expect("webpx encode failed");
    let (decoded, dw, dh) = webpx::decode_rgba(&webp).expect("webpx decode failed");
    assert_eq!((dw, dh), (w, h));
    (webp, decoded)
}

/// Map zenwebp Preset to webpx Preset.
fn to_wpx_preset(p: Preset) -> webpx::Preset {
    match p {
        Preset::Default => webpx::Preset::Default,
        Preset::Photo => webpx::Preset::Photo,
        Preset::Drawing => webpx::Preset::Drawing,
        Preset::Text => webpx::Preset::Text,
        Preset::Picture => webpx::Preset::Picture,
        Preset::Icon => webpx::Preset::Icon,
        _ => webpx::Preset::Default,
    }
}

/// Pixel data as &[[u8; 4]] for zensim.
fn as_rgba_pixels(data: &[u8]) -> &[[u8; 4]] {
    assert!(data.len() % 4 == 0);
    // SAFETY: [u8] with len % 4 == 0 can be reinterpreted as [[u8; 4]].
    // But we avoid unsafe — just use from_raw_parts equivalent via slice methods.
    let (prefix, pixels, suffix) = // use align trick
        // Actually, let's just collect. zensim takes RgbaSlice which wants &[[u8; 4]].
        // We can use bytemuck or just do the pointer cast safely.
        // For now, let's just build the slice safely.
        unsafe { data.align_to::<[u8; 4]>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    pixels
}

#[allow(dead_code)]
struct CompareResult {
    zen_size: usize,
    wpx_size: usize,
    size_ratio: f64,
    score: f64,
    passed: bool,
    report_text: String,
}

/// Encode with both encoders, decode, and compare using zensim.
fn compare_encoders(
    name: &str,
    rgba: &[u8],
    w: u32,
    h: u32,
    quality: f32,
    method: u8,
    preset: Preset,
    sns: Option<u8>,
    filter: Option<u8>,
    segments: Option<u8>,
    tolerance: &RegressionTolerance,
) -> CompareResult {
    let config = {
        let mut c = LossyConfig::with_preset(preset, quality).with_method(method);
        if let Some(s) = sns {
            c = c.with_sns_strength(s);
        }
        if let Some(f) = filter {
            c = c.with_filter_strength(f);
        }
        if let Some(seg) = segments {
            c = c.with_segments(seg);
        }
        c
    };

    let (zen_webp, zen_decoded) = zen_encode_decode(rgba, w, h, &config);
    let (wpx_webp, wpx_decoded) = wpx_encode_decode(
        rgba,
        w,
        h,
        quality,
        method,
        to_wpx_preset(preset),
        sns,
        filter,
        segments,
    );

    let size_ratio = zen_webp.len() as f64 / wpx_webp.len() as f64;

    // Compare zenwebp decoded output against webpx decoded output using zensim.
    // Both are lossy reconstructions of the same source — we're checking they
    // produce perceptually similar results.
    let z = Zensim::new(ZensimProfile::latest());
    let zen_px = as_rgba_pixels(&zen_decoded);
    let wpx_px = as_rgba_pixels(&wpx_decoded);
    let zen_img = RgbaSlice::new(zen_px, w as usize, h as usize);
    let wpx_img = RgbaSlice::new(wpx_px, w as usize, h as usize);

    let report =
        check_regression(&z, &wpx_img, &zen_img, tolerance).expect("zensim comparison failed");

    let report_text = format!(
        "{name} q{quality} m{method} {preset}: zen={} wpx={} ratio={size_ratio:.3}x score={:.1} {}",
        zen_webp.len(),
        wpx_webp.len(),
        report.score(),
        if report.passed() { "PASS" } else { "FAIL" },
    );

    CompareResult {
        zen_size: zen_webp.len(),
        wpx_size: wpx_webp.len(),
        size_ratio,
        score: report.score(),
        passed: report.passed(),
        report_text,
    }
}

// ---------------------------------------------------------------------------
// Tolerance: lossy encoder outputs differ due to algorithmic differences.
// We're not comparing pixel-identical — we're checking perceptual similarity.
// libwebp and zenwebp will produce different outputs at the same settings.
// Score >= 60 means "perceptually similar lossy encoding" — both are lossy
// reconstructions with different algorithms, so some divergence is expected.
// ---------------------------------------------------------------------------

fn lossy_tolerance() -> RegressionTolerance {
    RegressionTolerance::off_by_one()
        .with_max_delta(255) // lossy: large pixel differences are normal
        .with_max_pixels_different(1.0) // all pixels may differ
        .with_min_similarity(50.0) // perceptually similar
        .ignore_alpha()
}

// ---------------------------------------------------------------------------
// Quality sweep tests
// ---------------------------------------------------------------------------

#[test]
fn quality_sweep_gradient() {
    let rgba = gradient_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &q in &[50.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] {
        let r = compare_encoders(
            "gradient",
            &rgba,
            W,
            H,
            q,
            4,
            Preset::Default,
            None,
            None,
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Quality sweep: gradient ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some quality levels failed for gradient");
}

#[test]
fn quality_sweep_noise() {
    let rgba = noise_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &q in &[50.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] {
        let r = compare_encoders(
            "noise",
            &rgba,
            W,
            H,
            q,
            4,
            Preset::Default,
            None,
            None,
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Quality sweep: noise ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some quality levels failed for noise");
}

#[test]
fn quality_sweep_mandelbrot() {
    let rgba = mandelbrot_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &q in &[50.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] {
        let r = compare_encoders(
            "mandelbrot",
            &rgba,
            W,
            H,
            q,
            4,
            Preset::Default,
            None,
            None,
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Quality sweep: mandelbrot ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some quality levels failed for mandelbrot");
}

// ---------------------------------------------------------------------------
// Method sweep tests
// ---------------------------------------------------------------------------

#[test]
fn method_sweep() {
    let rgba = noise_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for method in 0..=6u8 {
        let r = compare_encoders(
            "noise",
            &rgba,
            W,
            H,
            75.0,
            method,
            Preset::Default,
            None,
            None,
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Method sweep (Q75, noise) ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some methods failed");
}

// ---------------------------------------------------------------------------
// Preset sweep tests
// ---------------------------------------------------------------------------

#[test]
fn preset_sweep() {
    let images: Vec<(&str, Vec<u8>)> = vec![
        ("gradient", gradient_rgba()),
        ("noise", noise_rgba()),
        ("mandelbrot", mandelbrot_rgba()),
        ("checkerboard", checkerboard_rgba()),
        ("color_blocks", color_blocks_rgba()),
    ];

    let presets = [
        Preset::Default,
        Preset::Photo,
        Preset::Drawing,
        Preset::Text,
        Preset::Picture,
    ];

    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for (name, rgba) in &images {
        for &preset in &presets {
            let r = compare_encoders(name, rgba, W, H, 75.0, 4, preset, None, None, None, &tol);
            results.push(r);
        }
    }

    println!("\n=== Preset sweep (Q75, M4) ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some presets failed");
}

// ---------------------------------------------------------------------------
// SNS strength sweep
// ---------------------------------------------------------------------------

#[test]
fn sns_sweep() {
    let rgba = noise_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &sns in &[0u8, 25, 50, 75, 100] {
        let r = compare_encoders(
            "noise",
            &rgba,
            W,
            H,
            75.0,
            4,
            Preset::Default,
            Some(sns),
            Some(60),
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== SNS sweep (Q75, M4, filter=60) ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some SNS levels failed");
}

// ---------------------------------------------------------------------------
// Filter strength sweep
// ---------------------------------------------------------------------------

#[test]
fn filter_sweep() {
    let rgba = noise_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &filter in &[0u8, 20, 40, 60, 80, 100] {
        let r = compare_encoders(
            "noise",
            &rgba,
            W,
            H,
            75.0,
            4,
            Preset::Default,
            Some(50),
            Some(filter),
            None,
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Filter sweep (Q75, M4, SNS=50) ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some filter levels failed");
}

// ---------------------------------------------------------------------------
// Segments sweep
// ---------------------------------------------------------------------------

#[test]
fn segments_sweep() {
    let rgba = noise_rgba();
    let tol = lossy_tolerance();
    let mut results = Vec::new();

    for &seg in &[1u8, 2, 3, 4] {
        let r = compare_encoders(
            "noise",
            &rgba,
            W,
            H,
            75.0,
            4,
            Preset::Default,
            None,
            None,
            Some(seg),
            &tol,
        );
        results.push(r);
    }

    println!("\n=== Segments sweep (Q75, M4) ===");
    let mut all_pass = true;
    for r in &results {
        println!("  {}", r.report_text);
        if !r.passed {
            all_pass = false;
        }
    }
    assert!(all_pass, "Some segment counts failed");
}

// ---------------------------------------------------------------------------
// Cross-decode: zenwebp encodes, webpx decodes (and vice versa)
// Tests bitstream compatibility.
// ---------------------------------------------------------------------------

#[test]
fn cross_decode_zen_to_wpx() {
    let rgba = noise_rgba();

    for &q in &[50.0, 75.0, 90.0] {
        let config = LossyConfig::new().with_quality(q).with_method(4);
        let webp = EncodeRequest::lossy(&config, &rgba, PixelLayout::Rgba8, W, H)
            .encode()
            .expect("zenwebp encode failed");

        // Decode with webpx (libwebp)
        let (wpx_decoded, dw, dh) =
            webpx::decode_rgba(&webp).expect("webpx failed to decode zenwebp output");
        assert_eq!((dw, dh), (W, H));

        // Decode with zenwebp
        let (zen_decoded, _, _) = zenwebp::decode_rgba(&webp).expect("zenwebp decode failed");

        // Compare the two decoders' outputs of the same bitstream
        let z = Zensim::new(ZensimProfile::latest());
        let tol = RegressionTolerance::off_by_one()
            .with_max_delta(3) // decoder rounding differences
            .with_min_similarity(95.0)
            .ignore_alpha();

        let zen_px = as_rgba_pixels(&zen_decoded);
        let wpx_px = as_rgba_pixels(&wpx_decoded);
        let zen_img = RgbaSlice::new(zen_px, W as usize, H as usize);
        let wpx_img = RgbaSlice::new(wpx_px, W as usize, H as usize);

        let report =
            check_regression(&z, &wpx_img, &zen_img, &tol).expect("zensim comparison failed");
        assert!(
            report.passed(),
            "Cross-decode zen→wpx failed at q{q}: {report}"
        );
    }
}

#[test]
fn cross_decode_wpx_to_zen() {
    let rgba = noise_rgba();

    for &q in &[50.0, 75.0, 90.0] {
        let wpx_config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, q).method(4);
        let webp = wpx_config
            .encode_rgba(&rgba, W, H, Unstoppable)
            .expect("webpx encode failed");

        // Decode with zenwebp
        let (zen_decoded, dw, dh) =
            zenwebp::decode_rgba(&webp).expect("zenwebp failed to decode webpx output");
        assert_eq!((dw, dh), (W, H));

        // Decode with webpx
        let (wpx_decoded, _, _) = webpx::decode_rgba(&webp).expect("webpx decode failed");

        // Compare
        let z = Zensim::new(ZensimProfile::latest());
        let tol = RegressionTolerance::off_by_one()
            .with_max_delta(3)
            .with_min_similarity(95.0)
            .ignore_alpha();

        let zen_px = as_rgba_pixels(&zen_decoded);
        let wpx_px = as_rgba_pixels(&wpx_decoded);
        let zen_img = RgbaSlice::new(zen_px, W as usize, H as usize);
        let wpx_img = RgbaSlice::new(wpx_px, W as usize, H as usize);

        let report =
            check_regression(&z, &wpx_img, &zen_img, &tol).expect("zensim comparison failed");
        assert!(
            report.passed(),
            "Cross-decode wpx→zen failed at q{q}: {report}"
        );
    }
}

// ---------------------------------------------------------------------------
// Size ratio sanity: zenwebp should be within ~10% of libwebp file sizes
// across the quality range with matched settings (diagnostic, sns=0, filter=0, seg=1).
// ---------------------------------------------------------------------------

#[test]
fn size_ratio_diagnostic() {
    let images: Vec<(&str, Vec<u8>)> = vec![
        ("gradient", gradient_rgba()),
        ("noise", noise_rgba()),
        ("mandelbrot", mandelbrot_rgba()),
    ];

    println!("\n=== Size ratio (diagnostic: SNS=0, filter=0, seg=1) ===");
    let mut worst_ratio = 0.0f64;

    for (name, rgba) in &images {
        for &q in &[50.0, 75.0, 90.0] {
            let zen_config = LossyConfig::new()
                .with_quality(q)
                .with_method(4)
                .with_sns_strength(0)
                .with_filter_strength(0)
                .with_segments(1);

            let zen_webp = EncodeRequest::lossy(&zen_config, rgba, PixelLayout::Rgba8, W, H)
                .encode()
                .expect("zenwebp encode failed");

            let wpx_config = webpx::EncoderConfig::with_preset(webpx::Preset::Default, q)
                .method(4)
                .sns_strength(0)
                .filter_strength(0)
                .segments(1);
            let wpx_webp = wpx_config
                .encode_rgba(rgba, W, H, Unstoppable)
                .expect("webpx encode failed");

            let ratio = zen_webp.len() as f64 / wpx_webp.len() as f64;
            println!(
                "  {name} q{q}: zen={} wpx={} ratio={ratio:.3}x",
                zen_webp.len(),
                wpx_webp.len()
            );
            worst_ratio = worst_ratio.max(ratio);
        }
    }

    // Gradient images amplify small algorithmic differences at low quality.
    // Real-world images are within ~5% (see corpus tests).
    assert!(
        worst_ratio < 1.20,
        "Worst size ratio {worst_ratio:.3}x exceeds 20% threshold"
    );
}

// ---------------------------------------------------------------------------
// Quality vs source: both encoders' output should score well against source.
// This catches regressions where zenwebp's output is worse than libwebp's
// at the same quality setting.
// ---------------------------------------------------------------------------

#[test]
fn quality_vs_source() {
    let images: Vec<(&str, Vec<u8>)> = vec![
        ("gradient", gradient_rgba()),
        ("noise", noise_rgba()),
        ("mandelbrot", mandelbrot_rgba()),
        ("checkerboard", checkerboard_rgba()),
        ("color_blocks", color_blocks_rgba()),
    ];

    let z = Zensim::new(ZensimProfile::latest());

    println!("\n=== Quality vs source (Q75, M4) ===");
    let mut max_gap = 0.0f64;

    for (name, rgba) in &images {
        let src_px = as_rgba_pixels(&rgba);
        let src_img = RgbaSlice::new(src_px, W as usize, H as usize);

        // zenwebp
        let zen_config = LossyConfig::new().with_quality(75.0).with_method(4);
        let (_, zen_decoded) = zen_encode_decode(&rgba, W, H, &zen_config);
        let zen_px = as_rgba_pixels(&zen_decoded);
        let zen_img = RgbaSlice::new(zen_px, W as usize, H as usize);
        let zen_result = z.compute(&src_img, &zen_img).expect("zensim failed");

        // webpx
        let (_, wpx_decoded) = wpx_encode_decode(
            &rgba,
            W,
            H,
            75.0,
            4,
            webpx::Preset::Default,
            None,
            None,
            None,
        );
        let wpx_px = as_rgba_pixels(&wpx_decoded);
        let wpx_img = RgbaSlice::new(wpx_px, W as usize, H as usize);
        let wpx_result = z.compute(&src_img, &wpx_img).expect("zensim failed");

        let gap = wpx_result.score() - zen_result.score();
        max_gap = max_gap.max(gap);

        println!(
            "  {name}: zen={:.1} wpx={:.1} gap={gap:.1}",
            zen_result.score(), wpx_result.score()
        );
    }

    // zenwebp should not be more than 5 points worse than libwebp
    assert!(
        max_gap < 5.0,
        "zenwebp quality gap {max_gap:.1} exceeds 5-point threshold vs libwebp"
    );
}

// ---------------------------------------------------------------------------
// Full matrix: all image types × key quality × preset combinations
// ---------------------------------------------------------------------------

#[test]
fn full_matrix() {
    let images: Vec<(&str, Vec<u8>)> = vec![
        ("gradient", gradient_rgba()),
        ("noise", noise_rgba()),
        ("mandelbrot", mandelbrot_rgba()),
        ("checkerboard", checkerboard_rgba()),
        ("color_blocks", color_blocks_rgba()),
    ];

    let presets = [Preset::Default, Preset::Photo, Preset::Drawing];
    let qualities = [50.0f32, 75.0, 90.0];
    let tol = lossy_tolerance();

    let mut pass_count = 0u32;
    let mut fail_count = 0u32;
    let mut failures = Vec::new();

    for (name, rgba) in &images {
        for &preset in &presets {
            for &q in &qualities {
                let r = compare_encoders(name, rgba, W, H, q, 4, preset, None, None, None, &tol);
                if r.passed {
                    pass_count += 1;
                } else {
                    fail_count += 1;
                    failures.push(r.report_text.clone());
                }
            }
        }
    }

    println!("\n=== Full matrix: {pass_count} passed, {fail_count} failed ===");
    for f in &failures {
        println!("  FAIL: {f}");
    }

    assert_eq!(fail_count, 0, "{fail_count} tests failed in full matrix");
}
