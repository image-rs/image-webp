#![cfg(not(target_arch = "wasm32"))]
//! Criterion benchmarks for zenwebp decoding performance.
//!
//! Run with: cargo bench --bench decode_benchmark
//! Run with native: RUSTFLAGS="-C target-cpu=native" cargo bench --bench decode_benchmark

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::Path;
use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};

/// Load a PNG image, encode to WebP, return WebP data.
fn make_webp(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;

    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity((info.width * info.height * 3) as usize);
            for chunk in buf[..info.buffer_size()].chunks(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };

    let config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    let webp = EncodeRequest::new(
        &config,
        &rgb_data,
        PixelLayout::Rgb8,
        info.width,
        info.height,
    )
    .encode()
    .ok()?;

    Some((webp, info.width, info.height))
}

fn zenwebp_test_images_dir() -> String {
    std::env::var("ZENWEBP_TEST_IMAGES_DIR").unwrap_or_else(|_| "/mnt/v".into())
}

/// Test image with metadata
struct TestImage {
    name: String,
    path: String,
}

fn test_images() -> Vec<TestImage> {
    let base = zenwebp_test_images_dir();
    vec![
        TestImage {
            name: "tiled_517x517".into(),
            path: format!("{base}/tiled_image.png"),
        },
        TestImage {
            name: "photo_512".into(),
            path: format!("{base}/cid22/792079.png"),
        },
    ]
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for test in &test_images() {
        let path = Path::new(&test.path);
        if !path.exists() {
            eprintln!("Skipping {}: file not found", test.name);
            continue;
        }

        let (webp_data, width, height) = match make_webp(path) {
            Some(data) => data,
            None => {
                eprintln!("Skipping {}: failed to encode", test.name);
                continue;
            }
        };

        let pixels = (width * height) as u64;
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(
            BenchmarkId::new("zenwebp", &test.name),
            &webp_data,
            |b, data| {
                b.iter(|| {
                    let result = zenwebp::decode_rgb(black_box(data)).unwrap();
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("libwebp", &test.name),
            &webp_data,
            |b, data| {
                b.iter(|| {
                    let decoder = webp::Decoder::new(black_box(data));
                    let result = decoder.decode().unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_decode_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_large");
    group.sample_size(20);

    let base = zenwebp_test_images_dir();
    let clic_paths = [
        format!("{base}/clic/clic2025_test_image_000001.png"),
        format!("{base}/clic/clic2025_test_image_000010.png"),
    ];

    for path_str in &clic_paths {
        let path = Path::new(path_str);
        if !path.exists() {
            continue;
        }

        let name = path.file_stem().unwrap().to_str().unwrap();
        let (webp_data, width, height) = match make_webp(path) {
            Some(data) => data,
            None => continue,
        };

        let pixels = (width * height) as u64;
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(BenchmarkId::new("zenwebp", name), &webp_data, |b, data| {
            b.iter(|| {
                let result = zenwebp::decode_rgb(black_box(data)).unwrap();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("libwebp", name), &webp_data, |b, data| {
            b.iter(|| {
                let decoder = webp::Decoder::new(black_box(data));
                let result = decoder.decode().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_decode, bench_decode_large);
criterion_main!(benches);
