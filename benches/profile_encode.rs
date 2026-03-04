use std::path::Path;
use std::time::Instant;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => panic!("Unsupported color type"),
    };

    (rgb_data, info.width, info.height)
}

fn codec_corpus_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR").unwrap_or_else(|_| "/home/lilith/work/codec-corpus".into()),
    );
    assert!(dir.is_dir(), "Codec corpus not found: {}. Set CODEC_CORPUS_DIR.", dir.display());
    dir
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_path = codec_corpus_dir().join("CID22/CID22-512/validation/792079.png").to_string_lossy().into_owned();
    let image_path = args.get(1).unwrap_or(&default_path);
    let quality: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(75.0);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let method: u8 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);

    let path = Path::new(image_path);
    println!("Loading: {}", path.display());
    let (rgb_data, width, height) = load_png(path);
    println!("Image: {}x{} ({} pixels)", width, height, width * height);
    println!(
        "Quality: {}, Method: {}, Iterations: {}",
        quality, method, iterations
    );

    // Warmup
    let config = EncoderConfig::new_lossy()
        .with_quality(quality)
        .with_method(method);
    let output = EncodeRequest::new(&config, &rgb_data, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap();
    println!("Output size: {} bytes", output.len());

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iterations {
        let config = EncoderConfig::new_lossy()
            .with_quality(quality)
            .with_method(method);
        let _output = EncodeRequest::new(&config, &rgb_data, PixelLayout::Rgb8, width, height)
            .encode()
            .unwrap();
    }
    let elapsed = start.elapsed();

    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let mpix_per_sec =
        (width as f64 * height as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;

    println!("\n=== Results ===");
    println!(
        "Total time: {:.2}ms for {} iterations",
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );
    println!("Per iteration: {:.2}ms", ms_per_iter);
    println!("Throughput: {:.2} MPix/s", mpix_per_sec);
}
