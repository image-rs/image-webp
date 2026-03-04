//! Compare decoder speed: zenwebp vs image-webp (upstream) vs libwebp

use std::env;
use std::fs;
use std::time::Instant;

fn zenwebp_test_images_dir() -> String {
    env::var("ZENWEBP_TEST_IMAGES_DIR").unwrap_or_else(|_| "/mnt/v".into())
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let default_path = format!("{}/tiled_image.png", zenwebp_test_images_dir());
    let img_path = args.get(1).unwrap_or(&default_path);

    // Load PNG and encode to WebP
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

    println!("Image: {}x{}", info.width, info.height);
    println!("WebP size: {} bytes", webp_data.len());
    println!();

    // Warm up all decoders
    let _ = zenwebp::decode_rgb(&webp_data);
    let _ = webpx::decode_rgb(&webp_data);
    let _ = image_webp::WebPDecoder::new(std::io::Cursor::new(&webp_data)).map(|mut d| {
        let mut buf = vec![0u8; d.output_buffer_size().unwrap_or(0)];
        let _ = d.read_image(&mut buf);
    });

    let iterations = 50;

    // Benchmark zenwebp
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = zenwebp::decode_rgb(&webp_data).unwrap();
    }
    let zen_time = start.elapsed() / iterations;

    // Benchmark image-webp (upstream)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(&webp_data)).unwrap();
        let mut buf = vec![0u8; decoder.output_buffer_size().unwrap()];
        decoder.read_image(&mut buf).unwrap();
    }
    let upstream_time = start.elapsed() / iterations;

    // Benchmark libwebp
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = webpx::decode_rgb(&webp_data).unwrap();
    }
    let lib_time = start.elapsed() / iterations;

    let pixels = info.width as f64 * info.height as f64;
    println!("Decoder Performance:");
    println!(
        "  image-webp (upstream): {:?} ({:.1} MPix/s) - {:.2}x vs libwebp",
        upstream_time,
        pixels / upstream_time.as_secs_f64() / 1e6,
        upstream_time.as_secs_f64() / lib_time.as_secs_f64()
    );
    println!(
        "  zenwebp:               {:?} ({:.1} MPix/s) - {:.2}x vs libwebp",
        zen_time,
        pixels / zen_time.as_secs_f64() / 1e6,
        zen_time.as_secs_f64() / lib_time.as_secs_f64()
    );
    println!(
        "  libwebp:               {:?} ({:.1} MPix/s) - 1.00x",
        lib_time,
        pixels / lib_time.as_secs_f64() / 1e6
    );
}
