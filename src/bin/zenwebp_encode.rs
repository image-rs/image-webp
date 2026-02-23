use std::path::Path;

fn load_png(path: &Path) -> (Vec<u8>, u32, u32, zenwebp::PixelLayout) {
    let file = std::fs::File::open(path).expect("failed to open input PNG");
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().expect("failed to read PNG header");
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).expect("failed to decode PNG");
    let data = buf[..info.buffer_size()].to_vec();

    let layout = match info.color_type {
        png::ColorType::Rgb => zenwebp::PixelLayout::Rgb8,
        png::ColorType::Rgba => zenwebp::PixelLayout::Rgba8,
        png::ColorType::Grayscale => zenwebp::PixelLayout::L8,
        png::ColorType::GrayscaleAlpha => zenwebp::PixelLayout::La8,
        other => panic!("unsupported PNG color type: {other:?}"),
    };

    (data, info.width, info.height, layout)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: zenwebp-encode <INPUT.png> <OUTPUT.webp> [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --quality Q    Lossy quality 0-100 (default: 75)");
        eprintln!("  --method M     Compression method 0-6 (default: 4)");
        eprintln!("  --lossless     Use lossless encoding");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    let mut quality: f32 = 75.0;
    let mut method: u8 = 4;
    let mut lossless = false;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" => {
                quality = args.get(i + 1).expect("--quality needs a value").parse().expect("invalid quality");
                i += 2;
            }
            "--method" => {
                method = args.get(i + 1).expect("--method needs a value").parse().expect("invalid method");
                i += 2;
            }
            "--lossless" => {
                lossless = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }

    let (pixels, width, height, layout) = load_png(input_path);

    let output = if lossless {
        let config = zenwebp::EncoderConfig::new_lossless()
            .with_quality(quality)
            .with_method(method);
        zenwebp::EncodeRequest::new(&config, &pixels, layout, width, height)
            .encode()
            .expect("encoding failed")
    } else {
        let config = zenwebp::EncoderConfig::new_lossy()
            .with_quality(quality)
            .with_method(method);
        zenwebp::EncodeRequest::new(&config, &pixels, layout, width, height)
            .encode()
            .expect("encoding failed")
    };

    std::fs::write(output_path, &output).expect("failed to write output");
    eprintln!("{} bytes", output.len());
}
