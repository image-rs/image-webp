//! Test VP8L lossless encoder - focus on getting a valid bitstream.

use std::fs;
use std::path::Path;
use std::process::Command;
use zenwebp::{EncodeRequest, PixelLayout};

fn libwebp_bin(name: &str) -> String {
    let dir = std::env::var("LIBWEBP_DIR").unwrap_or_else(|_| "/home/lilith/work/libwebp".into());
    format!("{dir}/examples/{name}")
}

fn main() {
    let test_path = "/tmp/CID22/original/1001682.png";

    if !Path::new(test_path).exists() {
        eprintln!("Test image not found: {}", test_path);
        return;
    }

    let file = fs::File::open(test_path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb_pixels: Vec<u8> = if info.color_type == png::ColorType::Rgba {
        buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]].into_iter())
            .collect()
    } else {
        buf[..info.buffer_size()].to_vec()
    };
    let width = info.width;
    let height = info.height;

    println!("Image: {}x{}", width, height);

    // Test the EXISTING encoder which produces valid output
    println!("\n=== Existing integrated encoder ===");
    let _cfg = zenwebp::EncoderConfig::new_lossy().with_lossless(true);
    let existing = EncodeRequest::new(&_cfg, &rgb_pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap();

    println!("Size: {} bytes", existing.len());
    std::fs::write("/tmp/zenwebp_existing.webp", &existing).unwrap();

    print!("Verifying... ");
    let verify = Command::new(libwebp_bin("dwebp"))
        .args([
            "/tmp/zenwebp_existing.webp",
            "-o",
            "/tmp/zenwebp_existing_decoded.ppm",
        ])
        .output();
    match verify {
        Ok(out) if out.status.success() => println!("OK"),
        Ok(out) => println!("FAILED: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => println!("Error: {}", e),
    }

    // Compare with libwebp
    let _ = Command::new(libwebp_bin("cwebp"))
        .args([
            "-lossless",
            "-q",
            "75",
            "-o",
            "/tmp/libwebp_lossless.webp",
            test_path,
        ])
        .output();

    let libwebp_size = std::fs::metadata("/tmp/libwebp_lossless.webp")
        .map(|m| m.len())
        .unwrap_or(0);

    println!("\nlibwebp lossless: {} bytes", libwebp_size);
    println!("Ratio: {:.3}x", existing.len() as f64 / libwebp_size as f64);

    // Now test our new VP8L encoder
    println!("\n=== New VP8L encoder ===");
    let config = zenwebp::encoder::vp8l::Vp8lConfig {
        use_predictor: false,
        use_subtract_green: false,
        ..Default::default()
    };

    let new_vp8l = zenwebp::encoder::vp8l::encode_vp8l(
        &rgb_pixels,
        width,
        height,
        false,
        &config,
        &enough::Unstoppable,
    )
    .unwrap();
    println!("Size (raw VP8L): {} bytes", new_vp8l.len());

    // Wrap in RIFF container
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + new_vp8l.len() + (new_vp8l.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(new_vp8l.len() as u32).to_le_bytes());
    webp.extend_from_slice(&new_vp8l);
    if !new_vp8l.len().is_multiple_of(2) {
        webp.push(0);
    }

    std::fs::write("/tmp/zenwebp_new_vp8l.webp", &webp).unwrap();

    print!("Verifying new encoder... ");
    let verify = Command::new(libwebp_bin("dwebp"))
        .args([
            "/tmp/zenwebp_new_vp8l.webp",
            "-o",
            "/tmp/zenwebp_new_decoded.ppm",
        ])
        .output();
    match verify {
        Ok(out) if out.status.success() => println!("OK"),
        Ok(out) => println!("FAILED: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => println!("Error: {}", e),
    }
}
