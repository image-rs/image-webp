#![cfg(not(target_arch = "wasm32"))]
//! Edge-specific SSIMULACRA2 comparison: our encoder vs libwebp
//!
//! Isolates edge handling quality by:
//! 1. Extracting partial macroblock edge pixels (rightmost columns / bottom rows)
//! 2. Tiling them to create a full image of just edge content
//! 3. Encoding/decoding with both our encoder and libwebp
//! 4. Comparing SSIMULACRA2 of the tiled edge regions
//!
//! WebP uses 16x16 macroblocks, so we test non-16-aligned dimensions.
//!
//! Run with: cargo test --release --test edge_tile_ssim2_comparison -- --nocapture --ignored

use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic, compute_frame_ssimulacra2};
use std::path::{Path, PathBuf};
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

fn kodak_path() -> Option<std::path::PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    corpus.get("kodak").ok()
}

/// Load PNG and return RGB data with dimensions
fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
            for chunk in rgba.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

/// Crop image to specified dimensions (top-left origin)
fn crop_image(
    rgb: &[u8],
    src_width: u32,
    _src_height: u32,
    new_width: u32,
    new_height: u32,
) -> Vec<u8> {
    let new_width = new_width as usize;
    let new_height = new_height as usize;
    let src_width = src_width as usize;

    let mut cropped = Vec::with_capacity(new_width * new_height * 3);
    for y in 0..new_height {
        let start = y * src_width * 3;
        cropped.extend_from_slice(&rgb[start..start + new_width * 3]);
    }
    cropped
}

/// Extract rightmost N columns and tile them leftward to fill width
fn tile_right_edge(rgb: &[u8], width: usize, height: usize, edge_width: usize) -> Vec<u8> {
    let mut tiled = vec![0u8; width * height * 3];

    for y in 0..height {
        let row_start = y * width * 3;
        let edge_start = row_start + (width - edge_width) * 3;
        let edge_pixels = &rgb[edge_start..row_start + width * 3];

        let dst_row_start = y * width * 3;
        for x in 0..width {
            let src_x = x % edge_width;
            let src_idx = src_x * 3;
            let dst_idx = dst_row_start + x * 3;
            tiled[dst_idx] = edge_pixels[src_idx];
            tiled[dst_idx + 1] = edge_pixels[src_idx + 1];
            tiled[dst_idx + 2] = edge_pixels[src_idx + 2];
        }
    }
    tiled
}

/// Extract bottom N rows and tile them upward to fill height
fn tile_bottom_edge(rgb: &[u8], width: usize, height: usize, edge_height: usize) -> Vec<u8> {
    let mut tiled = vec![0u8; width * height * 3];

    let edge_start_row = height - edge_height;

    for y in 0..height {
        let src_y = edge_start_row + (y % edge_height);
        let src_row_start = src_y * width * 3;
        let dst_row_start = y * width * 3;
        tiled[dst_row_start..dst_row_start + width * 3]
            .copy_from_slice(&rgb[src_row_start..src_row_start + width * 3]);
    }
    tiled
}

/// Encode with our encoder
fn encode_ours(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::new_lossy().with_quality(quality as f32);
    EncodeRequest::new(&config, rgb, PixelLayout::Rgb8, width, height)
        .encode()
        .expect("Encoding failed")
}

/// Encode with libwebp
fn encode_libwebp(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let encoder = webp::Encoder::from_rgb(rgb, width, height);
    encoder.encode(quality).to_vec()
}

/// Decode WebP
fn decode_webp(data: &[u8]) -> Vec<u8> {
    webp::Decoder::new(data)
        .decode()
        .expect("Decode failed")
        .to_vec()
}

/// sRGB to linear conversion
fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Calculate SSIMULACRA2 between two RGB images
fn calculate_ssim2(rgb1: &[u8], rgb2: &[u8], width: usize, height: usize) -> f64 {
    let src: Vec<[f32; 3]> = rgb1
        .chunks(3)
        .map(|c| {
            [
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            ]
        })
        .collect();
    let dst: Vec<[f32; 3]> = rgb2
        .chunks(3)
        .map(|c| {
            [
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            ]
        })
        .collect();

    let src_img = Rgb::new(
        src,
        width,
        height,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dst_img = Rgb::new(
        dst,
        width,
        height,
        TransferCharacteristic::Linear,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(src_img, dst_img).unwrap_or(-1.0)
}

#[test]
#[ignore]
fn test_edge_tile_ssim2_comparison() {
    let kodak_dir = match kodak_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: kodak corpus unavailable");
            return;
        }
    };

    // Find test images (use first 6 for reasonable test time)
    let mut images: Vec<PathBuf> = (1..=6)
        .map(|i| kodak_dir.join(format!("{}.png", i)))
        .filter(|p| p.exists())
        .collect();
    images.sort();

    if images.is_empty() {
        panic!("No PNG images found in corpus");
    }

    println!("\n=== EDGE TILE SSIMULACRA2 COMPARISON (WebP) ===\n");
    println!("Testing edge handling quality by tiling partial macroblock (16x16) pixels\n");

    // Test configurations: (crop_width, crop_height, edge_width, edge_height)
    // WebP uses 16x16 macroblocks, so test 1-15 pixel partial edges
    let test_configs = [
        // Width edge tests (partial pixels beyond 16-alignment)
        (257, 256, 1, 0),  // 1 pixel partial width (256+1)
        (260, 256, 4, 0),  // 4 pixels partial width (256+4)
        (264, 256, 8, 0),  // 8 pixels partial width (256+8)
        (270, 256, 14, 0), // 14 pixels partial width (256+14)
        // Height edge tests
        (256, 257, 0, 1),  // 1 pixel partial height
        (256, 260, 0, 4),  // 4 pixels partial height
        (256, 264, 0, 8),  // 8 pixels partial height
        (256, 270, 0, 14), // 14 pixels partial height
        // Combined tests
        (257, 257, 1, 1),   // Both edges minimal
        (264, 264, 8, 8),   // Both edges half
        (270, 270, 14, 14), // Both edges near-maximal
    ];

    // Quality levels to test
    let qualities = [50u8, 75, 90];

    for quality in qualities {
        println!("\n=== Quality {} ===\n", quality);
        println!(
            "{:>15} {:>10} {:>6} {:>6} {:>10} {:>10} {:>10}",
            "Image", "Dims", "EdgeW", "EdgeH", "Ours", "libwebp", "Diff"
        );
        println!("{}", "-".repeat(80));

        let mut total_ours_ssim2 = 0.0;
        let mut total_lib_ssim2 = 0.0;
        let mut count = 0;

        for image_path in &images {
            let (rgb, orig_width, orig_height) = match load_png(image_path) {
                Some(data) => data,
                None => continue,
            };

            let image_name = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            for &(crop_w, crop_h, edge_w, edge_h) in &test_configs {
                if crop_w > orig_width || crop_h > orig_height {
                    continue;
                }

                // Crop image
                let cropped = crop_image(&rgb, orig_width, orig_height, crop_w, crop_h);
                let width = crop_w as usize;
                let height = crop_h as usize;

                // Encode and decode with both encoders
                let ours_webp = encode_ours(&cropped, crop_w, crop_h, quality);
                let ours_decoded = decode_webp(&ours_webp);

                let lib_webp = encode_libwebp(&cropped, crop_w, crop_h, quality as f32);
                let lib_decoded = decode_webp(&lib_webp);

                // Calculate SSIM2 based on edge type
                let (ours_ssim2, lib_ssim2) = if edge_w > 0 && edge_h == 0 {
                    // Width edge: tile rightmost columns
                    let orig_tiled = tile_right_edge(&cropped, width, height, edge_w as usize);
                    let ours_tiled = tile_right_edge(&ours_decoded, width, height, edge_w as usize);
                    let lib_tiled = tile_right_edge(&lib_decoded, width, height, edge_w as usize);

                    (
                        calculate_ssim2(&orig_tiled, &ours_tiled, width, height),
                        calculate_ssim2(&orig_tiled, &lib_tiled, width, height),
                    )
                } else if edge_h > 0 && edge_w == 0 {
                    // Height edge: tile bottom rows
                    let orig_tiled = tile_bottom_edge(&cropped, width, height, edge_h as usize);
                    let ours_tiled =
                        tile_bottom_edge(&ours_decoded, width, height, edge_h as usize);
                    let lib_tiled = tile_bottom_edge(&lib_decoded, width, height, edge_h as usize);

                    (
                        calculate_ssim2(&orig_tiled, &ours_tiled, width, height),
                        calculate_ssim2(&orig_tiled, &lib_tiled, width, height),
                    )
                } else if edge_w > 0 && edge_h > 0 {
                    // Both edges: test right edge
                    let orig_tiled = tile_right_edge(&cropped, width, height, edge_w as usize);
                    let ours_tiled = tile_right_edge(&ours_decoded, width, height, edge_w as usize);
                    let lib_tiled = tile_right_edge(&lib_decoded, width, height, edge_w as usize);

                    (
                        calculate_ssim2(&orig_tiled, &ours_tiled, width, height),
                        calculate_ssim2(&orig_tiled, &lib_tiled, width, height),
                    )
                } else {
                    continue;
                };

                let diff = ours_ssim2 - lib_ssim2;
                let diff_str = if diff.abs() < 0.5 {
                    format!("{:+.2}", diff)
                } else if diff > 0.0 {
                    format!("{:+.2} +", diff)
                } else {
                    format!("{:+.2} -", diff)
                };

                println!(
                    "{:>15} {:>10} {:>6} {:>6} {:>10.2} {:>10.2} {:>10}",
                    image_name,
                    format!("{}x{}", crop_w, crop_h),
                    if edge_w > 0 {
                        edge_w.to_string()
                    } else {
                        "-".to_string()
                    },
                    if edge_h > 0 {
                        edge_h.to_string()
                    } else {
                        "-".to_string()
                    },
                    ours_ssim2,
                    lib_ssim2,
                    diff_str
                );

                total_ours_ssim2 += ours_ssim2;
                total_lib_ssim2 += lib_ssim2;
                count += 1;
            }
        }

        println!("{}", "-".repeat(80));
        if count > 0 {
            let avg_ours = total_ours_ssim2 / count as f64;
            let avg_lib = total_lib_ssim2 / count as f64;
            let avg_diff = avg_ours - avg_lib;
            println!(
                "{:>15} {:>10} {:>6} {:>6} {:>10.2} {:>10.2} {:>10}",
                "AVERAGE",
                "",
                "",
                "",
                avg_ours,
                avg_lib,
                format!("{:+.2}", avg_diff)
            );
        }
    }

    println!("\nNote: SSIMULACRA2 scores are for TILED EDGE PIXELS ONLY");
    println!("Higher = better quality. Positive diff = our encoder handles edges better.\n");
}

/// Additional test: compare full image quality at non-aligned sizes
#[test]
#[ignore]
fn test_edge_full_image_comparison() {
    let kodak_dir = match kodak_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: kodak corpus unavailable");
            return;
        }
    };

    println!("\n=== FULL IMAGE QUALITY AT NON-ALIGNED SIZES ===\n");

    let image_path = kodak_dir.join("1.png");
    let (rgb, orig_width, orig_height) = load_png(&image_path).expect("Failed to load image");

    // Test various non-aligned sizes
    let sizes = [
        (256, 256), // Aligned
        (257, 256), // +1 width
        (256, 257), // +1 height
        (257, 257), // +1 both
        (264, 264), // +8 both
        (270, 270), // +14 both
        (271, 271), // +15 both
    ];

    let quality = 75u8;

    println!(
        "{:>12} {:>12} {:>12} {:>10} {:>10}",
        "Size", "Ours Size", "libwebp Size", "Ours SSIM2", "lib SSIM2"
    );
    println!("{}", "-".repeat(60));

    for (w, h) in sizes {
        if w > orig_width || h > orig_height {
            continue;
        }

        let cropped = crop_image(&rgb, orig_width, orig_height, w, h);

        let ours_webp = encode_ours(&cropped, w, h, quality);
        let ours_decoded = decode_webp(&ours_webp);
        let ours_ssim2 = calculate_ssim2(&cropped, &ours_decoded, w as usize, h as usize);

        let lib_webp = encode_libwebp(&cropped, w, h, quality as f32);
        let lib_decoded = decode_webp(&lib_webp);
        let lib_ssim2 = calculate_ssim2(&cropped, &lib_decoded, w as usize, h as usize);

        println!(
            "{:>12} {:>12} {:>12} {:>10.2} {:>10.2}",
            format!("{}x{}", w, h),
            ours_webp.len(),
            lib_webp.len(),
            ours_ssim2,
            lib_ssim2
        );
    }
}
