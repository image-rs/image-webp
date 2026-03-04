//! Lossless VP8L encoder benchmark against libwebp.
//! Usage: cargo run --release --example lossless_benchmark [image_dir]

use std::fs;
use std::path::Path;
use std::process::Command;

fn libwebp_bin(name: &str) -> String {
    let dir = std::env::var("LIBWEBP_DIR").unwrap_or_else(|_| "/home/lilith/work/libwebp".into());
    format!("{dir}/examples/{name}")
}

fn wrap_vp8l_in_riff(vp8l_data: &[u8]) -> Vec<u8> {
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + vp8l_data.len() + (vp8l_data.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(vp8l_data.len() as u32).to_le_bytes());
    webp.extend_from_slice(vp8l_data);
    if !vp8l_data.len().is_multiple_of(2) {
        webp.push(0);
    }
    webp
}

fn encode_libwebp_lossless(png_path: &str, quality: u8) -> Option<u64> {
    // Use a unique temp file based on PID to avoid stale file issues
    let output = format!("/tmp/libwebp_lossless_bench_{}.webp", std::process::id());
    let result = Command::new(libwebp_bin("cwebp"))
        .args([
            "-lossless",
            "-q",
            &quality.to_string(),
            "-o",
            &output,
            png_path,
        ])
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    let size = fs::metadata(&output).map(|m| m.len()).ok();
    let _ = fs::remove_file(&output); // Clean up
    size
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/CID22/original".to_string());

    let mut pngs: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension()?.to_str()? == "png" {
                Some(p.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    pngs.sort();

    // Take first N for quick benchmark
    let max_images = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    pngs.truncate(max_images);

    let quality = 75u8;
    println!(
        "Lossless VP8L benchmark: {} images, q={}",
        pngs.len(),
        quality
    );
    println!("{:-<95}", "");
    println!(
        "{:<25} {:>10} {:>8} {:>8} {:>8} {:>6}",
        "Image", "dims", "zenwebp", "libwebp", "ratio", "valid"
    );
    println!("{:-<95}", "");

    let mut total_zen = 0u64;
    let mut total_lib = 0u64;
    let mut count = 0usize;

    for png_path in &pngs {
        let name = Path::new(png_path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Load PNG
        let file = match fs::File::open(png_path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let decoder = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = match decoder.read_info() {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) => continue,
        };

        let (rgb_pixels, has_alpha) = match info.color_type {
            png::ColorType::Rgba => {
                let rgba = buf[..info.buffer_size()].to_vec();
                (rgba, true)
            }
            png::ColorType::Rgb => (buf[..info.buffer_size()].to_vec(), false),
            _ => continue,
        };
        let width = info.width;
        let height = info.height;

        // Encode with zenwebp (meta-Huffman enabled)
        let config = zenwebp::encoder::vp8l::Vp8lConfig {
            quality: zenwebp::encoder::vp8l::Vp8lQuality { quality, method: 4 },
            use_predictor: true,
            use_cross_color: true,
            use_subtract_green: true,
            use_palette: true,
            ..Default::default()
        };

        let vp8l = match zenwebp::encoder::vp8l::encode_vp8l(
            &rgb_pixels,
            width,
            height,
            has_alpha,
            &config,
            &enough::Unstoppable,
        ) {
            Ok(d) => d,
            Err(e) => {
                println!("{:<30} ERROR: {:?}", name, e);
                continue;
            }
        };

        // Also encode WITHOUT meta-Huffman for comparison
        let config_nomh = zenwebp::encoder::vp8l::Vp8lConfig {
            use_meta_huffman: false,
            ..config.clone()
        };
        let vp8l_nomh = zenwebp::encoder::vp8l::encode_vp8l(
            &rgb_pixels,
            width,
            height,
            has_alpha,
            &config_nomh,
            &enough::Unstoppable,
        )
        .ok();

        let webp = wrap_vp8l_in_riff(&vp8l);
        let zen_size = webp.len() as u64;

        // Verify with our decoder
        let valid = match zenwebp::decode_rgba(&webp) {
            Ok((decoded, dw, dh)) => {
                if dw != width || dh != height {
                    "dim!"
                } else {
                    let pixel_count = (width * height) as usize;
                    let bpp = if has_alpha { 4 } else { 3 };
                    let mismatches = (0..pixel_count)
                        .filter(|&i| {
                            decoded[i * 4] != rgb_pixels[i * bpp]
                                || decoded[i * 4 + 1] != rgb_pixels[i * bpp + 1]
                                || decoded[i * 4 + 2] != rgb_pixels[i * bpp + 2]
                        })
                        .count();
                    if mismatches == 0 {
                        "OK"
                    } else {
                        "FAIL"
                    }
                }
            }
            Err(_) => "ERR",
        };

        // Encode with libwebp
        let lib_size = encode_libwebp_lossless(png_path, quality).unwrap_or(0);

        let ratio = if lib_size > 0 {
            zen_size as f64 / lib_size as f64
        } else {
            0.0
        };

        // Show meta-Huffman benefit
        let mh_note = if let Some(ref nomh) = vp8l_nomh {
            let nomh_size = wrap_vp8l_in_riff(nomh).len() as u64;
            if nomh_size > zen_size {
                format!(
                    "MH-{:.0}%",
                    (1.0 - zen_size as f64 / nomh_size as f64) * 100.0
                )
            } else {
                format!(
                    "MH+{:.0}%",
                    (zen_size as f64 / nomh_size as f64 - 1.0) * 100.0
                )
            }
        } else {
            String::new()
        };

        let dims = format!("{}x{}", width, height);
        println!(
            "{:<25} {:>10} {:>8} {:>8} {:>8.3}x {:>6} {}",
            &name[..name.len().min(25)],
            dims,
            zen_size,
            lib_size,
            ratio,
            valid,
            mh_note,
        );

        total_zen += zen_size;
        total_lib += lib_size;
        count += 1;
    }

    println!("{:-<95}", "");
    if count > 0 && total_lib > 0 {
        println!(
            "{:<25} {:>10} {:>8} {:>8} {:>8.3}x",
            "TOTAL",
            "",
            total_zen,
            total_lib,
            total_zen as f64 / total_lib as f64
        );
    }
}
