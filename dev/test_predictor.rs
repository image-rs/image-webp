//! Test VP8L encoder with full 14 predictor modes.

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

fn verify_dwebp(path: &str) -> bool {
    let verify = Command::new(libwebp_bin("dwebp"))
        .args([path, "-o", &format!("{}.ppm", path)])
        .output();
    match verify {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            println!("  dwebp FAILED: {}", String::from_utf8_lossy(&out.stderr));
            false
        }
        Err(e) => {
            println!("  dwebp error: {}", e);
            false
        }
    }
}

fn test_gradient() {
    println!("=== Test 1: Gradient 64x64 ===");
    let width = 64u32;
    let height = 64u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 4) as u8);
            rgb.push((y * 4) as u8);
            rgb.push(128);
        }
    }

    // With predictor enabled (all 14 modes)
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: true,
        use_subtract_green: true,
        use_palette: false,
        predictor_bits: 4, // 16x16 blocks
        ..Default::default()
    };
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb,
        width,
        height,
        false,
        &config,
        &enough::Unstoppable,
    )
    .unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    std::fs::write("/tmp/test_predictor_gradient.webp", &webp).unwrap();
    println!("  Size with predictor: {} bytes", webp.len());

    // Without predictor
    let config_no = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: false,
        use_subtract_green: true,
        use_palette: false,
        ..Default::default()
    };
    let vp8l_no = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb,
        width,
        height,
        false,
        &config_no,
        &enough::Unstoppable,
    )
    .unwrap();
    let webp_no = wrap_vp8l_in_riff(&vp8l_no);
    println!("  Size without predictor: {} bytes", webp_no.len());
    println!(
        "  Improvement: {:.1}%",
        (1.0 - webp.len() as f64 / webp_no.len() as f64) * 100.0
    );

    // Verify with dwebp
    let ok = verify_dwebp("/tmp/test_predictor_gradient.webp");
    println!("  dwebp: {}", if ok { "OK" } else { "FAILED" });

    // Verify with our decoder
    match zenwebp::decode_rgba(&webp) {
        Ok((data, w, h)) => {
            println!("  Our decoder: OK ({}x{}, {} bytes)", w, h, data.len());
            // Verify lossless: check first few pixels
            let mut mismatches = 0;
            for y in 0..height.min(8) {
                for x in 0..width.min(8) {
                    let idx = (y * width + x) as usize;
                    let expected_r = (x * 4) as u8;
                    let expected_g = (y * 4) as u8;
                    let expected_b = 128u8;
                    if data[idx * 4] != expected_r
                        || data[idx * 4 + 1] != expected_g
                        || data[idx * 4 + 2] != expected_b
                    {
                        if mismatches < 3 {
                            println!(
                                "  MISMATCH at ({},{}): expected ({},{},{}) got ({},{},{})",
                                x,
                                y,
                                expected_r,
                                expected_g,
                                expected_b,
                                data[idx * 4],
                                data[idx * 4 + 1],
                                data[idx * 4 + 2]
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
            if mismatches == 0 {
                println!("  Lossless: VERIFIED (checked 64 pixels)");
            } else {
                println!("  Lossless: FAILED ({} mismatches)", mismatches);
            }
        }
        Err(e) => println!("  Our decoder: FAILED: {:?}", e),
    }
}

fn test_photo() {
    let path = "/tmp/CID22/original/1001682.png";
    if !std::path::Path::new(path).exists() {
        println!("\n=== Test 2: Photo (SKIPPED - no test image) ===");
        return;
    }
    println!("\n=== Test 2: Photo (CID22) ===");

    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let width = info.width;
    let height = info.height;

    let rgb: Vec<u8> = if info.color_type == png::ColorType::Rgba {
        buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]].into_iter())
            .collect()
    } else {
        buf[..info.buffer_size()].to_vec()
    };

    println!("  Image: {}x{}", width, height);

    // With predictor
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: true,
        use_subtract_green: true,
        use_palette: false,
        predictor_bits: 4,
        ..Default::default()
    };
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb,
        width,
        height,
        false,
        &config,
        &enough::Unstoppable,
    )
    .unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    std::fs::write("/tmp/test_predictor_photo.webp", &webp).unwrap();
    println!("  Size with predictor: {} bytes", webp.len());

    // Without predictor
    let config_no = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: false,
        use_subtract_green: true,
        use_palette: false,
        ..Default::default()
    };
    let vp8l_no = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb,
        width,
        height,
        false,
        &config_no,
        &enough::Unstoppable,
    )
    .unwrap();
    let webp_no = wrap_vp8l_in_riff(&vp8l_no);
    println!("  Size without predictor: {} bytes", webp_no.len());
    println!(
        "  Improvement: {:.1}%",
        (1.0 - webp.len() as f64 / webp_no.len() as f64) * 100.0
    );

    // Compare with libwebp
    let _ = Command::new(libwebp_bin("cwebp"))
        .args([
            "-lossless",
            "-q",
            "75",
            "-o",
            "/tmp/libwebp_lossless_photo.webp",
            path,
        ])
        .output();
    if let Ok(meta) = std::fs::metadata("/tmp/libwebp_lossless_photo.webp") {
        println!("  libwebp lossless: {} bytes", meta.len());
        println!(
            "  Ratio vs libwebp: {:.3}x",
            webp.len() as f64 / meta.len() as f64
        );
    }

    // Verify with dwebp
    let ok = verify_dwebp("/tmp/test_predictor_photo.webp");
    println!("  dwebp: {}", if ok { "OK" } else { "FAILED" });

    // Verify with our decoder (lossless check)
    match zenwebp::decode_rgba(&webp) {
        Ok((data, w, h)) => {
            println!("  Our decoder: OK ({}x{}, {} bytes)", w, h, data.len());
            // Compare decoded RGB against original
            let mut mismatches = 0;
            for i in 0..(width * height) as usize {
                if data[i * 4] != rgb[i * 3]
                    || data[i * 4 + 1] != rgb[i * 3 + 1]
                    || data[i * 4 + 2] != rgb[i * 3 + 2]
                {
                    mismatches += 1;
                }
            }
            if mismatches == 0 {
                println!("  Lossless: VERIFIED (all {} pixels match)", width * height);
            } else {
                println!(
                    "  Lossless: FAILED ({} of {} pixels differ)",
                    mismatches,
                    width * height
                );
            }
        }
        Err(e) => println!("  Our decoder: FAILED: {:?}", e),
    }
}

fn main() {
    test_gradient();
    test_photo();
}
