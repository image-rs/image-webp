//! Tests for memory/throughput estimation heuristics.

use zenwebp::EncoderConfig;
use zenwebp::heuristics::*;

#[test]
fn estimate_lossy_1mp() {
    let config = EncoderConfig::new_lossy().with_method(4);
    let est = estimate_encode(1024, 1024, 4, &config);
    // Peak memory should be reasonable for 1MP (order of magnitude check)
    assert!(
        est.peak_memory_bytes > 1_000_000,
        "too low: {}",
        est.peak_memory_bytes
    );
    assert!(
        est.peak_memory_bytes < 500_000_000,
        "too high: {}",
        est.peak_memory_bytes
    );
    assert!(est.time_ms > 0.0);
}

#[test]
fn estimate_lossy_4k() {
    let config = EncoderConfig::new_lossy().with_method(4);
    let est = estimate_encode(3840, 2160, 4, &config);
    let est_1mp = estimate_encode(1024, 1024, 4, &config);
    // 4K should use more memory than 1MP
    assert!(est.peak_memory_bytes > est_1mp.peak_memory_bytes);
    // And take longer
    assert!(est.time_ms > est_1mp.time_ms);
}

#[test]
fn estimate_lossless_1mp() {
    let config = EncoderConfig::new_lossless();
    let est = estimate_encode(1024, 1024, 4, &config);
    assert!(est.peak_memory_bytes > 1_000_000);
    assert!(est.time_ms > 0.0);
}

#[test]
fn estimate_memory_range_valid() {
    let config = EncoderConfig::new_lossy().with_method(4);
    let est = estimate_encode(1920, 1080, 4, &config);
    // Min <= peak <= max
    assert!(
        est.peak_memory_bytes_min <= est.peak_memory_bytes,
        "min {} > peak {}",
        est.peak_memory_bytes_min,
        est.peak_memory_bytes
    );
    assert!(
        est.peak_memory_bytes <= est.peak_memory_bytes_max,
        "peak {} > max {}",
        est.peak_memory_bytes,
        est.peak_memory_bytes_max
    );
}
