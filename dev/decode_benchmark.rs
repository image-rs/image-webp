//! Decoder speed comparison between zenwebp and libwebp.
//! Usage: cargo run --release --example decode_benchmark [image.png]

use std::env;
use std::fs;
use std::time::Instant;

use webpx::decode_rgb as lib_decode;
use zenwebp::decode_rgb as zen_decode;

fn zenwebp_test_images_dir() -> String {
    env::var("ZENWEBP_TEST_IMAGES_DIR").unwrap_or_else(|_| "/mnt/v".into())
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let default_path = format!("{}/tiled_image.png", zenwebp_test_images_dir());
    let img_path = args.get(1).unwrap_or(&default_path);

    // Load and encode PNG to WebP
    let file = fs::File::open(img_path).expect("Failed to open image");
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let rgba = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity(rgba.len() * 3 / 4);
            for chunk in rgba.chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => panic!("Unsupported color type"),
    };

    // Encode with libwebp
    let webp_data = webpx::EncoderConfig::with_preset(webpx::Preset::Default, 75.0)
        .method(5)
        .encode_rgb(&rgb, info.width, info.height, webpx::Unstoppable)
        .expect("Failed to encode");

    println!("Image: {}x{} ({})", info.width, info.height, img_path);
    println!("WebP size: {} bytes", webp_data.len());

    // Warm up
    let _ = zen_decode(&webp_data);
    let _ = lib_decode(&webp_data);

    // Benchmark zenwebp
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = zen_decode(&webp_data).unwrap();
    }
    let zen_time = start.elapsed() / iterations;

    // Benchmark libwebp
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = lib_decode(&webp_data).unwrap();
    }
    let lib_time = start.elapsed() / iterations;

    let ratio = zen_time.as_secs_f64() / lib_time.as_secs_f64();
    let zen_mpix = (info.width as f64 * info.height as f64) / zen_time.as_secs_f64() / 1_000_000.0;
    let lib_mpix = (info.width as f64 * info.height as f64) / lib_time.as_secs_f64() / 1_000_000.0;

    println!("\nDecoder Performance:");
    println!("  zenwebp: {:?} ({:.1} MPix/s)", zen_time, zen_mpix);
    println!("  libwebp: {:?} ({:.1} MPix/s)", lib_time, lib_mpix);
    println!("  Ratio: {:.2}x slower", ratio);
}
