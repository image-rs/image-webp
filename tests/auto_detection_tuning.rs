#![cfg(not(target_arch = "wasm32"))]
//! Auto-detection tuning tests for CID22 and screenshot corpora.
//! Run with: cargo test --release --features _corpus_tests --test auto_detection_tuning -- --nocapture

#![cfg(feature = "_corpus_tests")]

use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::fs;
use std::io::BufReader;
use zenwebp::encoder::analysis::{ContentType, analyze_image, classify_image_type_diag};

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

fn encode_zen_bytes(rgb: &[u8], w: u32, h: u32, preset: zenwebp::Preset) -> Vec<u8> {
    let config = zenwebp::EncoderConfig::with_preset(preset, 75.0).with_method(4);
    zenwebp::EncodeRequest::new(&config, rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .unwrap()
}

fn encode_wpx_bytes(rgb: &[u8], w: u32, h: u32, preset: webpx::Preset) -> Vec<u8> {
    webpx::EncoderConfig::with_preset(preset, 75.0)
        .method(4)
        .encode_rgb(rgb, w, h, webpx::Unstoppable)
        .unwrap()
}

fn srgb_to_linear(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn rgb8_to_linear(rgb: &[u8]) -> Vec<[f32; 3]> {
    rgb.chunks_exact(3)
        .map(|p| {
            [
                srgb_to_linear(p[0]),
                srgb_to_linear(p[1]),
                srgb_to_linear(p[2]),
            ]
        })
        .collect()
}

fn ssim2(original_rgb: &[u8], webp_bytes: &[u8], w: u32, h: u32) -> f64 {
    let (decoded, dw, dh) = zenwebp::decode_rgb(webp_bytes).unwrap();
    assert_eq!((dw, dh), (w, h));

    let orig_linear = rgb8_to_linear(original_rgb);
    let dec_linear = rgb8_to_linear(&decoded);

    let orig = Rgb::new(
        orig_linear,
        w as usize,
        h as usize,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let dec = Rgb::new(
        dec_linear,
        w as usize,
        h as usize,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig, dec).unwrap()
}

fn butteraugli_score(original_rgb: &[u8], webp_bytes: &[u8], w: u32, h: u32) -> f64 {
    use butteraugli::{ButteraugliParams, RGB8, butteraugli};
    use imgref::Img;

    let (decoded, dw, dh) = zenwebp::decode_rgb(webp_bytes).unwrap();
    assert_eq!((dw, dh), (w, h));

    // Convert to RGB8 pixel slices
    let orig_pixels: Vec<RGB8> = original_rgb
        .chunks_exact(3)
        .map(|p| RGB8::new(p[0], p[1], p[2]))
        .collect();
    let dec_pixels: Vec<RGB8> = decoded
        .chunks_exact(3)
        .map(|p| RGB8::new(p[0], p[1], p[2]))
        .collect();

    let orig_img = Img::new(orig_pixels, w as usize, h as usize);
    let dec_img = Img::new(dec_pixels, w as usize, h as usize);

    let params = ButteraugliParams::default();
    butteraugli(orig_img.as_ref(), dec_img.as_ref(), &params)
        .map(|r| r.score)
        .unwrap_or(f64::MAX)
}

/// Get classifier diagnostics for an RGB image by converting to YUV and running analysis.
fn classify_rgb(rgb: &[u8], w: u32, h: u32) -> zenwebp::ClassifierDiag {
    let width = w as usize;
    let height = h as usize;
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let y_stride = mb_w * 16;
    let uv_stride = mb_w * 8;

    let mut y_buf = vec![0u8; y_stride * mb_h * 16];
    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 3;
            let r = rgb[idx] as i32;
            let g = rgb[idx + 1] as i32;
            let b = rgb[idx + 2] as i32;
            let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_buf[row * y_stride + col] = y.clamp(0, 255) as u8;
        }
    }

    let u_buf = vec![128u8; uv_stride * mb_h * 8];
    let v_buf = vec![128u8; uv_stride * mb_h * 8];

    let analysis = analyze_image(
        &y_buf, &u_buf, &v_buf, width, height, y_stride, uv_stride, 4, 50,
    );

    classify_image_type_diag(&y_buf, width, height, y_stride, &analysis.alpha_histogram)
}

fn ct_label(ct: ContentType) -> &'static str {
    match ct {
        ContentType::Photo => "Photo",
        ContentType::Drawing => "Draw",
        ContentType::Text => "Text",
        ContentType::Icon => "Icon",
        _ => "Unknown",
    }
}

/// Per-image result for aggregation.
#[allow(dead_code)]
struct ImageResult {
    name: String,
    det: ContentType,
    // zenwebp
    zen_auto_size: usize,
    zen_default_size: usize,
    zen_photo_size: usize,
    zen_auto_ss: f64,
    zen_default_ss: f64,
    zen_photo_ss: f64,
    zen_auto_ba: f64,
    zen_default_ba: f64,
    zen_photo_ba: f64,
    // libwebp (via webpx)
    wpx_default_size: usize,
    wpx_photo_size: usize,
    wpx_default_ss: f64,
    wpx_photo_ss: f64,
    wpx_default_ba: f64,
    wpx_photo_ba: f64,
    // classifier
    uniformity: f32,
}

#[test]
fn auto_detection_cid22() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let corpus_dir = corpus
        .get("CID22/CID22-512/validation")
        .expect("CID22 corpus unavailable");

    println!("\n=== CID22 Validation Set — Size & Quality Comparison ===");
    println!(
        "{:<14} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}  {:>4}",
        "Image",
        "Det",
        "zAuto",
        "zDflt",
        "zPhot",
        "wDflt",
        "wPhot",
        "sZA",
        "sZD",
        "sZP",
        "sWD",
        "sWP",
        "unif",
        ""
    );
    println!("{}", "-".repeat(130));

    let mut results: Vec<ImageResult> = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(corpus_dir)
        .expect("CID22 corpus not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if let Some((rgb, w, h)) = load_png(&path.to_string_lossy()) {
            let diag = classify_rgb(&rgb, w, h);

            let zen_auto = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Auto);
            let zen_default = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Default);
            let zen_photo = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Photo);
            let wpx_default = encode_wpx_bytes(&rgb, w, h, webpx::Preset::Default);
            let wpx_photo = encode_wpx_bytes(&rgb, w, h, webpx::Preset::Photo);

            let ss_zen_auto = ssim2(&rgb, &zen_auto, w, h);
            let ss_zen_default = ssim2(&rgb, &zen_default, w, h);
            let ss_zen_photo = ssim2(&rgb, &zen_photo, w, h);
            let ss_wpx_default = ssim2(&rgb, &wpx_default, w, h);
            let ss_wpx_photo = ssim2(&rgb, &wpx_photo, w, h);

            // Compute butteraugli scores
            let ba_zen_auto = butteraugli_score(&rgb, &zen_auto, w, h);
            let ba_zen_default = butteraugli_score(&rgb, &zen_default, w, h);
            let ba_zen_photo = butteraugli_score(&rgb, &zen_photo, w, h);
            let ba_wpx_default = butteraugli_score(&rgb, &wpx_default, w, h);
            let ba_wpx_photo = butteraugli_score(&rgb, &wpx_photo, w, h);

            println!(
                "{:<14} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>5.1} {:>5.1} {:>5.1} {:>5.1} {:>5.1}  {:>4.2}",
                &name[..name.len().min(14)],
                ct_label(diag.content_type),
                zen_auto.len(),
                zen_default.len(),
                zen_photo.len(),
                wpx_default.len(),
                wpx_photo.len(),
                ss_zen_auto,
                ss_zen_default,
                ss_zen_photo,
                ss_wpx_default,
                ss_wpx_photo,
                diag.uniformity,
            );

            results.push(ImageResult {
                name,
                det: diag.content_type,
                zen_auto_size: zen_auto.len(),
                zen_default_size: zen_default.len(),
                zen_photo_size: zen_photo.len(),
                zen_auto_ss: ss_zen_auto,
                zen_default_ss: ss_zen_default,
                zen_photo_ss: ss_zen_photo,
                zen_auto_ba: ba_zen_auto,
                zen_default_ba: ba_zen_default,
                zen_photo_ba: ba_zen_photo,
                wpx_default_size: wpx_default.len(),
                wpx_photo_size: wpx_photo.len(),
                wpx_default_ss: ss_wpx_default,
                wpx_photo_ss: ss_wpx_photo,
                wpx_default_ba: ba_wpx_default,
                wpx_photo_ba: ba_wpx_photo,
                uniformity: diag.uniformity,
            });
        }
    }

    let n = results.len() as f64;
    let total = |f: fn(&ImageResult) -> usize| -> usize { results.iter().map(f).sum() };
    let avg = |f: fn(&ImageResult) -> f64| -> f64 { results.iter().map(f).sum::<f64>() / n };

    let t_zen_auto = total(|r| r.zen_auto_size);
    let t_zen_default = total(|r| r.zen_default_size);
    let t_zen_photo = total(|r| r.zen_photo_size);
    let t_wpx_default = total(|r| r.wpx_default_size);
    let t_wpx_photo = total(|r| r.wpx_photo_size);

    let a_zen_auto = avg(|r| r.zen_auto_ss);
    let a_zen_default = avg(|r| r.zen_default_ss);
    let a_zen_photo = avg(|r| r.zen_photo_ss);
    let a_wpx_default = avg(|r| r.wpx_default_ss);
    let a_wpx_photo = avg(|r| r.wpx_photo_ss);

    // Butteraugli averages
    let ba_zen_auto = avg(|r| r.zen_auto_ba);
    let ba_zen_default = avg(|r| r.zen_default_ba);
    let ba_zen_photo = avg(|r| r.zen_photo_ba);
    let ba_wpx_default = avg(|r| r.wpx_default_ba);
    let ba_wpx_photo = avg(|r| r.wpx_photo_ba);

    println!("{}", "-".repeat(130));
    println!(
        "\n  {:>20} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "", "Size", "SSIM2 avg", "BA avg", "Size Δ", "SSIM2 Δ", "BA Δ"
    );

    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2}",
        "zenwebp Default", t_zen_default, a_zen_default, ba_zen_default
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "zenwebp Photo",
        t_zen_photo,
        a_zen_photo,
        ba_zen_photo,
        t_zen_photo as f64 / t_zen_default as f64,
        a_zen_photo - a_zen_default,
        ba_zen_photo - ba_zen_default,
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "zenwebp Auto",
        t_zen_auto,
        a_zen_auto,
        ba_zen_auto,
        t_zen_auto as f64 / t_zen_default as f64,
        a_zen_auto - a_zen_default,
        ba_zen_auto - ba_zen_default,
    );
    println!();
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2}",
        "libwebp Default", t_wpx_default, a_wpx_default, ba_wpx_default
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "libwebp Photo",
        t_wpx_photo,
        a_wpx_photo,
        ba_wpx_photo,
        t_wpx_photo as f64 / t_wpx_default as f64,
        a_wpx_photo - a_wpx_default,
        ba_wpx_photo - ba_wpx_default,
    );

    // Per-image Photo-vs-Default delta comparison
    println!("\n  Photo-vs-Default tradeoff per image (Photo-detected only):");
    println!(
        "  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Image", "zen Δsz%", "zen Δss", "zen Δba", "wpx Δsz%", "wpx Δss", "wpx Δba"
    );
    for r in &results {
        if r.det != ContentType::Photo {
            continue;
        }
        let zen_size_delta = (r.zen_photo_size as f64 / r.zen_default_size as f64 - 1.0) * 100.0;
        let zen_ss_delta = r.zen_photo_ss - r.zen_default_ss;
        let zen_ba_delta = r.zen_photo_ba - r.zen_default_ba;
        let wpx_size_delta = (r.wpx_photo_size as f64 / r.wpx_default_size as f64 - 1.0) * 100.0;
        let wpx_ss_delta = r.wpx_photo_ss - r.wpx_default_ss;
        let wpx_ba_delta = r.wpx_photo_ba - r.wpx_default_ba;
        println!(
            "  {:<14} {:>+7.1}% {:>+8.2} {:>+8.2} {:>+7.1}% {:>+8.2} {:>+8.2}",
            &r.name[..r.name.len().min(14)],
            zen_size_delta,
            zen_ss_delta,
            zen_ba_delta,
            wpx_size_delta,
            wpx_ss_delta,
            wpx_ba_delta,
        );
    }
}

#[test]
fn auto_detection_screenshots() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    let corpus_dir = corpus.get("gb82-sc").expect("gb82-sc corpus unavailable");

    println!("\n=== Screenshot Corpus — Size & Quality Comparison ===");
    println!(
        "{:<14} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>5} {:>5} {:>5} {:>5} {:>5}",
        "Image", "zAuto", "zDflt", "zPhot", "wDflt", "wPhot", "sZA", "sZD", "sZP", "sWD", "sWP",
    );
    println!("{}", "-".repeat(110));

    let mut results: Vec<ImageResult> = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(corpus_dir)
        .expect("Screenshot corpus not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if let Some((rgb, w, h)) = load_png(&path.to_string_lossy()) {
            let diag = classify_rgb(&rgb, w, h);

            let zen_auto = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Auto);
            let zen_default = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Default);
            let zen_photo = encode_zen_bytes(&rgb, w, h, zenwebp::Preset::Photo);
            // libwebp with explicit tuning (bypasses webpx preset bug)
            let wpx_default = encode_wpx_bytes(&rgb, w, h, webpx::Preset::Default);
            let wpx_photo = encode_wpx_bytes(&rgb, w, h, webpx::Preset::Photo);

            let ss_zen_auto = ssim2(&rgb, &zen_auto, w, h);
            let ss_zen_default = ssim2(&rgb, &zen_default, w, h);
            let ss_zen_photo = ssim2(&rgb, &zen_photo, w, h);
            let ss_wpx_default = ssim2(&rgb, &wpx_default, w, h);
            let ss_wpx_photo = ssim2(&rgb, &wpx_photo, w, h);

            // Compute butteraugli scores
            let ba_zen_auto = butteraugli_score(&rgb, &zen_auto, w, h);
            let ba_zen_default = butteraugli_score(&rgb, &zen_default, w, h);
            let ba_zen_photo = butteraugli_score(&rgb, &zen_photo, w, h);
            let ba_wpx_default = butteraugli_score(&rgb, &wpx_default, w, h);
            let ba_wpx_photo = butteraugli_score(&rgb, &wpx_photo, w, h);

            println!(
                "{:<14} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>5.1} {:>5.1} {:>5.1} {:>5.1} {:>5.1}",
                &name[..name.len().min(14)],
                zen_auto.len(),
                zen_default.len(),
                zen_photo.len(),
                wpx_default.len(),
                wpx_photo.len(),
                ss_zen_auto,
                ss_zen_default,
                ss_zen_photo,
                ss_wpx_default,
                ss_wpx_photo,
            );

            results.push(ImageResult {
                name,
                det: diag.content_type,
                zen_auto_size: zen_auto.len(),
                zen_default_size: zen_default.len(),
                zen_photo_size: zen_photo.len(),
                zen_auto_ss: ss_zen_auto,
                zen_default_ss: ss_zen_default,
                zen_photo_ss: ss_zen_photo,
                zen_auto_ba: ba_zen_auto,
                zen_default_ba: ba_zen_default,
                zen_photo_ba: ba_zen_photo,
                wpx_default_size: wpx_default.len(),
                wpx_photo_size: wpx_photo.len(),
                wpx_default_ss: ss_wpx_default,
                wpx_photo_ss: ss_wpx_photo,
                wpx_default_ba: ba_wpx_default,
                wpx_photo_ba: ba_wpx_photo,
                uniformity: diag.uniformity,
            });
        }
    }

    let n = results.len() as f64;
    let total = |f: fn(&ImageResult) -> usize| -> usize { results.iter().map(f).sum() };
    let avg = |f: fn(&ImageResult) -> f64| -> f64 { results.iter().map(f).sum::<f64>() / n };

    let t_zen_auto = total(|r| r.zen_auto_size);
    let t_zen_default = total(|r| r.zen_default_size);
    let t_zen_photo = total(|r| r.zen_photo_size);
    let t_wpx_default = total(|r| r.wpx_default_size);
    let t_wpx_photo = total(|r| r.wpx_photo_size);

    let a_zen_auto = avg(|r| r.zen_auto_ss);
    let a_zen_default = avg(|r| r.zen_default_ss);
    let a_zen_photo = avg(|r| r.zen_photo_ss);
    let a_wpx_default = avg(|r| r.wpx_default_ss);
    let a_wpx_photo = avg(|r| r.wpx_photo_ss);

    // Butteraugli averages
    let ba_zen_auto = avg(|r| r.zen_auto_ba);
    let ba_zen_default = avg(|r| r.zen_default_ba);
    let ba_zen_photo = avg(|r| r.zen_photo_ba);
    let ba_wpx_default = avg(|r| r.wpx_default_ba);
    let ba_wpx_photo = avg(|r| r.wpx_photo_ba);

    println!("{}", "-".repeat(110));
    println!(
        "\n  {:>20} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "", "Size", "SSIM2 avg", "BA avg", "Size Δ", "SSIM2 Δ", "BA Δ"
    );

    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2}",
        "zenwebp Default", t_zen_default, a_zen_default, ba_zen_default
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "zenwebp Photo",
        t_zen_photo,
        a_zen_photo,
        ba_zen_photo,
        t_zen_photo as f64 / t_zen_default as f64,
        a_zen_photo - a_zen_default,
        ba_zen_photo - ba_zen_default,
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "zenwebp Auto",
        t_zen_auto,
        a_zen_auto,
        ba_zen_auto,
        t_zen_auto as f64 / t_zen_default as f64,
        a_zen_auto - a_zen_default,
        ba_zen_auto - ba_zen_default,
    );
    println!();
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2}",
        "libwebp Default", t_wpx_default, a_wpx_default, ba_wpx_default
    );
    println!(
        "  {:>20} {:>12} {:>12.2} {:>12.2} {:>9.3}x {:>+10.2} {:>+10.2}",
        "libwebp Photo",
        t_wpx_photo,
        a_wpx_photo,
        ba_wpx_photo,
        t_wpx_photo as f64 / t_wpx_default as f64,
        a_wpx_photo - a_wpx_default,
        ba_wpx_photo - ba_wpx_default,
    );

    // Per-image comparison
    println!("\n  Photo-vs-Default tradeoff per image:");
    println!(
        "  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Image", "zen Δsz%", "zen Δss", "zen Δba", "wpx Δsz%", "wpx Δss", "wpx Δba"
    );
    for r in &results {
        let zen_size_delta = (r.zen_photo_size as f64 / r.zen_default_size as f64 - 1.0) * 100.0;
        let zen_ss_delta = r.zen_photo_ss - r.zen_default_ss;
        let zen_ba_delta = r.zen_photo_ba - r.zen_default_ba;
        let wpx_size_delta = (r.wpx_photo_size as f64 / r.wpx_default_size as f64 - 1.0) * 100.0;
        let wpx_ss_delta = r.wpx_photo_ss - r.wpx_default_ss;
        let wpx_ba_delta = r.wpx_photo_ba - r.wpx_default_ba;
        println!(
            "  {:<14} {:>+7.1}% {:>+8.2} {:>+8.2} {:>+7.1}% {:>+8.2} {:>+8.2}",
            &r.name[..r.name.len().min(14)],
            zen_size_delta,
            zen_ss_delta,
            zen_ba_delta,
            wpx_size_delta,
            wpx_ss_delta,
            wpx_ba_delta,
        );
    }
}
