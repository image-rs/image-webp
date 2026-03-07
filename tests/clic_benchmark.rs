#![cfg(not(target_arch = "wasm32"))]
//! Comprehensive benchmark comparing image-webp vs libwebp on CLIC 2025 validation set.
//!
//! Measures: encoding speed, file size, and SSIMULACRA2 quality.
//!
//! Run with: cargo test --release clic_benchmark -- --nocapture --ignored

use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::fs;
use std::path::Path;
use std::time::Instant;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

fn clic_path() -> Option<std::path::PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    corpus.get("clic2025/validation").ok()
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;

    let width = info.width;
    let height = info.height;

    // Convert to RGB if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for pixel in rgba.chunks_exact(4) {
                rgb.extend_from_slice(&pixel[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let gray = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for &g in gray {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let ga = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for pixel in ga.chunks_exact(2) {
                let g = pixel[0];
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, width, height))
}

fn compute_ssim2(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    // Convert u8 RGB to Vec<[f32; 3]>
    let orig_f32: Vec<[f32; 3]> = original
        .chunks_exact(3)
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let dec_f32: Vec<[f32; 3]> = decoded
        .chunks_exact(3)
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let orig_img = Rgb::new(
        orig_f32,
        width as usize,
        height as usize,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dec_img = Rgb::new(
        dec_f32,
        width as usize,
        height as usize,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_img, dec_img).unwrap()
}

fn decode_webp(data: &[u8]) -> Option<Vec<u8>> {
    let decoder = webp::Decoder::new(data);
    let image = decoder.decode()?;
    Some(image.to_vec())
}

#[allow(dead_code)]
#[derive(Debug)]
struct BenchResult {
    quality: u8,
    ours_size: usize,
    ours_time_ms: f64,
    ours_ssim2: f64,
    libwebp_size: usize,
    libwebp_time_ms: f64,
    libwebp_ssim2: f64,
}

fn benchmark_image(rgb: &[u8], width: u32, height: u32, quality: u8) -> Option<BenchResult> {
    // Encode with our encoder
    let start = Instant::now();
    let config = EncoderConfig::new_lossy().with_quality(quality as f32);
    let ours_output =
        match EncodeRequest::new(&config, rgb, PixelLayout::Rgb8, width, height).encode() {
            Ok(v) => v,
            Err(_) => return None,
        };
    let ours_time = start.elapsed();

    // Encode with libwebp
    let start = Instant::now();
    let libwebp_encoder = webp::Encoder::from_rgb(rgb, width, height);
    let libwebp_output = libwebp_encoder.encode(quality as f32);
    let libwebp_time = start.elapsed();

    // Decode both and compute SSIM2
    let ours_decoded = decode_webp(&ours_output)?;
    let libwebp_decoded = decode_webp(&libwebp_output)?;

    let ours_ssim2 = compute_ssim2(rgb, &ours_decoded, width, height);
    let libwebp_ssim2 = compute_ssim2(rgb, &libwebp_decoded, width, height);

    Some(BenchResult {
        quality,
        ours_size: ours_output.len(),
        ours_time_ms: ours_time.as_secs_f64() * 1000.0,
        ours_ssim2,
        libwebp_size: libwebp_output.len(),
        libwebp_time_ms: libwebp_time.as_secs_f64() * 1000.0,
        libwebp_ssim2,
    })
}

#[test]
#[ignore]
fn clic_benchmark() {
    let clic_dir = match clic_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: CLIC corpus unavailable");
            return;
        }
    };

    let mut images: Vec<_> = fs::read_dir(&clic_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .collect();
    images.sort_by_key(|e| e.path());

    let qualities = [50, 75, 90];

    println!("\n=== CLIC 2025 Validation Benchmark ===");
    println!("Images: {}", images.len());
    println!("Quality levels: {:?}\n", qualities);

    // Aggregate stats
    let mut total_ours_size = [0usize; 3];
    let mut total_libwebp_size = [0usize; 3];
    let mut total_ours_time = [0.0f64; 3];
    let mut total_libwebp_time = [0.0f64; 3];
    let mut total_ours_ssim2 = [0.0f64; 3];
    let mut total_libwebp_ssim2 = [0.0f64; 3];
    let mut image_count = 0;

    for entry in &images {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_str().unwrap();

        let Some((rgb, width, height)) = load_png(&path) else {
            eprintln!("Failed to load: {}", path.display());
            continue;
        };

        print!("{} ({}x{}): ", &name[..8], width, height);

        let mut all_ok = true;
        for (qi, &quality) in qualities.iter().enumerate() {
            let Some(result) = benchmark_image(&rgb, width, height, quality) else {
                print!("q{}[FAIL] ", quality);
                all_ok = false;
                continue;
            };

            total_ours_size[qi] += result.ours_size;
            total_libwebp_size[qi] += result.libwebp_size;
            total_ours_time[qi] += result.ours_time_ms;
            total_libwebp_time[qi] += result.libwebp_time_ms;
            total_ours_ssim2[qi] += result.ours_ssim2;
            total_libwebp_ssim2[qi] += result.libwebp_ssim2;

            print!(
                "q{}[{:+.1}dB,{:+.0}%] ",
                quality,
                result.ours_ssim2 - result.libwebp_ssim2,
                (result.ours_size as f64 / result.libwebp_size as f64 - 1.0) * 100.0
            );
        }
        println!();

        if all_ok {
            image_count += 1;
        }
    }

    println!("\n=== Summary ({} images) ===\n", image_count);

    println!(
        "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "Quality", "Ours Size", "Lib Size", "Size Δ%", "Ours SSIM2", "Lib SSIM2", "SSIM2 Δ"
    );
    println!("{}", "-".repeat(82));

    for (qi, &quality) in qualities.iter().enumerate() {
        let n = image_count as f64;
        let avg_ours_ssim2 = total_ours_ssim2[qi] / n;
        let avg_libwebp_ssim2 = total_libwebp_ssim2[qi] / n;
        let size_ratio = total_ours_size[qi] as f64 / total_libwebp_size[qi] as f64;

        println!(
            "{:>8} {:>12} {:>12} {:>+9.1}% {:>10.2} {:>10.2} {:>+10.2}",
            quality,
            format_size(total_ours_size[qi]),
            format_size(total_libwebp_size[qi]),
            (size_ratio - 1.0) * 100.0,
            avg_ours_ssim2,
            avg_libwebp_ssim2,
            avg_ours_ssim2 - avg_libwebp_ssim2
        );
    }

    println!(
        "\n{:>8} {:>12} {:>12} {:>10}",
        "Quality", "Ours Time", "Lib Time", "Speedup"
    );
    println!("{}", "-".repeat(52));

    for (qi, &quality) in qualities.iter().enumerate() {
        let speedup = total_libwebp_time[qi] / total_ours_time[qi];
        println!(
            "{:>8} {:>11.1}s {:>11.1}s {:>9.2}x",
            quality,
            total_ours_time[qi] / 1000.0,
            total_libwebp_time[qi] / 1000.0,
            speedup
        );
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

#[test]
#[ignore]
fn clic_detailed_quality_sweep() {
    let clic_dir = match clic_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: CLIC corpus unavailable");
            return;
        }
    };

    // Pick first 5 images for detailed sweep
    let mut images: Vec<_> = fs::read_dir(&clic_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .take(5)
        .collect();
    images.sort_by_key(|e| e.path());

    let qualities: Vec<u8> = (10..=100).step_by(10).collect();

    println!("\n=== Detailed Quality Sweep (5 images) ===\n");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Quality", "Ours KB", "Lib KB", "Size Δ%", "Ours SSIM2", "Lib SSIM2", "SSIM2 Δ"
    );
    println!("{}", "-".repeat(78));

    for quality in qualities {
        let mut total_ours_size = 0;
        let mut total_libwebp_size = 0;
        let mut total_ours_ssim2 = 0.0;
        let mut total_libwebp_ssim2 = 0.0;
        let mut count = 0;

        for entry in &images {
            let path = entry.path();
            let Some((rgb, width, height)) = load_png(&path) else {
                continue;
            };

            let Some(result) = benchmark_image(&rgb, width, height, quality) else {
                continue;
            };
            total_ours_size += result.ours_size;
            total_libwebp_size += result.libwebp_size;
            total_ours_ssim2 += result.ours_ssim2;
            total_libwebp_ssim2 += result.libwebp_ssim2;
            count += 1;
        }

        let n = count as f64;
        let size_ratio = total_ours_size as f64 / total_libwebp_size as f64;

        println!(
            "{:>8} {:>10.1} {:>10.1} {:>+9.1}% {:>10.2} {:>10.2} {:>+10.2}",
            quality,
            total_ours_size as f64 / 1000.0,
            total_libwebp_size as f64 / 1000.0,
            (size_ratio - 1.0) * 100.0,
            total_ours_ssim2 / n,
            total_libwebp_ssim2 / n,
            total_ours_ssim2 / n - total_libwebp_ssim2 / n
        );
    }
}
