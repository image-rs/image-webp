//! WASM benchmark binary for comparing native vs WASM performance.
//!
//! Build and run:
//! - Native: `cargo run --release --bin profile_wasm --features _wasm_profiling -- [path.webp] [iterations]`
//! - WASM: `cargo build --release --target wasm32-wasip1 --features _wasm_profiling --bin profile_wasm`
//!   then: `wasmtime run --dir=. target/wasm32-wasip1/release/profile_wasm.wasm -- [path.webp] [iterations]`

use std::time::Instant;

fn benchmark_decode(webp_data: &[u8], iterations: usize, label: &str) {
    // Decode once to get dimensions
    let decoder = zenwebp::WebPDecoder::new(webp_data).expect("Failed to create decoder");
    let (width, height) = decoder.dimensions();
    let output_size = decoder.output_buffer_size().unwrap();
    let pixels = width as u64 * height as u64;

    println!("=== {} ===", label);
    println!("Image: {}x{} ({} pixels)", width, height, pixels);
    println!("WebP size: {} bytes", webp_data.len());
    println!("Iterations: {}", iterations);

    // Warmup
    for _ in 0..3.min(iterations) {
        let mut decoder = zenwebp::WebPDecoder::new(webp_data).unwrap();
        let mut output = vec![0u8; output_size];
        decoder.read_image(&mut output).unwrap();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut decoder = zenwebp::WebPDecoder::new(webp_data).unwrap();
        let mut output = vec![0u8; output_size];
        decoder.read_image(&mut output).unwrap();
    }
    let elapsed = start.elapsed();

    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let total_pixels = pixels * iterations as u64;
    let mpix_per_sec = total_pixels as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Time per decode: {:.3} ms", ms_per_iter);
    println!("Throughput: {:.2} MPix/s", mpix_per_sec);
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Detect platform
    #[cfg(target_arch = "wasm32")]
    println!("Platform: WASM32 (SIMD128)");
    #[cfg(target_arch = "x86_64")]
    println!("Platform: x86_64 (SSE2/AVX2)");
    #[cfg(target_arch = "aarch64")]
    println!("Platform: aarch64 (NEON)");
    #[cfg(not(any(
        target_arch = "wasm32",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    println!("Platform: unknown");
    println!();

    let webp_path = args.get(1);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    if let Some(path) = webp_path {
        let webp_data = std::fs::read(path).expect("Failed to read WebP file");
        benchmark_decode(&webp_data, iterations, &format!("Decoding: {}", path));
    } else {
        eprintln!("Usage: profile_wasm <path.webp> [iterations]");
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  Native: cargo run --release --bin profile_wasm --features _wasm_profiling -- image.webp 100"
        );
        eprintln!(
            "  WASM:   wasmtime run --dir=. target/wasm32-wasip1/release/profile_wasm.wasm -- image.webp 100"
        );
        std::process::exit(1);
    }
}
