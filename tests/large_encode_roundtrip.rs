//! Test that large image encode/decode roundtrips produce valid bitstreams,
//! and that the encoder returns a clean error (not corrupt output) when the
//! VP8 partition 0 size limit is exceeded.
//!
//! VP8 frame tag: partition_0_size is 19 bits (max 524,287 bytes).
//! For high-entropy images above ~25MP, macroblock mode headers push
//! partition 0 past this limit. The encoder must return
//! `EncodeError::Partition0Overflow` rather than producing a corrupt bitstream.
//!
//! See VP8 RFC 6386 Section 9.2 for the frame tag format.

use zenwebp::{DecodeRequest, EncodeError, EncodeRequest, LossyConfig, PixelLayout};

/// Generate a smooth gradient image (RGB, low entropy — compresses small).
fn generate_gradient_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = ((x + y) * 128 / (width + height)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Generate a noisy image (RGB, high entropy — compresses large).
/// This pushes the compressed partition size past the 19-bit limit.
fn generate_noisy_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let hash =
                ((x as u64).wrapping_mul(2654435761) ^ (y as u64).wrapping_mul(2246822519)) as u32;
            let noise = (hash >> 16) as u8;
            let r = ((x * 255 / width) as u8).wrapping_add(noise & 0x3F);
            let g = ((y * 255 / height) as u8).wrapping_add((noise >> 2) & 0x3F);
            let b = (((x + y) * 128 / (width + height)) as u8).wrapping_add((noise >> 4) & 0x3F);
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Encode and decode an RGB image, asserting both succeed and dimensions match.
fn roundtrip_rgb(pixels: &[u8], width: u32, height: u32, quality: f32) {
    let config = LossyConfig::new().with_quality(quality);
    let encoded = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, width, height)
        .encode()
        .unwrap_or_else(|e| panic!("encode failed at {width}x{height}: {e}"));

    assert!(
        !encoded.is_empty(),
        "encoded data is empty at {width}x{height}"
    );

    let dc = zenwebp::DecodeConfig::default();
    let result = DecodeRequest::new(&dc, &encoded).decode();
    match result {
        Ok((_, dw, dh, _)) => {
            assert_eq!(dw, width, "width mismatch");
            assert_eq!(dh, height, "height mismatch");
        }
        Err(e) => {
            panic!(
                "decode failed at {width}x{height} q{quality} (encoded {} bytes): {e}",
                encoded.len()
            );
        }
    }
}

/// Attempt to encode, asserting that it fails with Partition0Overflow.
fn expect_partition0_overflow(pixels: &[u8], width: u32, height: u32, quality: f32) {
    let config = LossyConfig::new().with_quality(quality);
    let result = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, width, height).encode();
    match result {
        Err(e) => {
            assert!(
                matches!(e.error(), EncodeError::Partition0Overflow { .. }),
                "expected Partition0Overflow at {width}x{height} q{quality}, got: {e}"
            );
        }
        Ok(data) => {
            panic!(
                "expected Partition0Overflow at {width}x{height} q{quality}, \
                 but encode succeeded ({} bytes)",
                data.len()
            );
        }
    }
}

// --- Low-entropy gradient images: partition stays small, all pass ---

#[test]
fn gradient_4096x4096() {
    let pixels = generate_gradient_rgb(4096, 4096);
    roundtrip_rgb(&pixels, 4096, 4096, 80.0);
}

#[test]
fn gradient_7680x5760() {
    let pixels = generate_gradient_rgb(7680, 5760);
    roundtrip_rgb(&pixels, 7680, 5760, 80.0);
}

#[test]
fn gradient_8192x8192() {
    let pixels = generate_gradient_rgb(8192, 8192);
    roundtrip_rgb(&pixels, 8192, 8192, 80.0);
}

// --- High-entropy noisy images: partition exceeds 19-bit limit above ~25MP ---

#[test]
fn noisy_4096x4096() {
    // 16.8MP — OK, partition fits in 19 bits
    let pixels = generate_noisy_rgb(4096, 4096);
    roundtrip_rgb(&pixels, 4096, 4096, 80.0);
}

#[test]
fn noisy_5000x5000() {
    // 25.0MP — OK, just under threshold at q75
    let pixels = generate_noisy_rgb(5000, 5000);
    roundtrip_rgb(&pixels, 5000, 5000, 75.0);
}

#[test]
fn noisy_5824x4368() {
    // 25.4MP — OK, just under threshold at q70
    let pixels = generate_noisy_rgb(5824, 4368);
    roundtrip_rgb(&pixels, 5824, 4368, 70.0);
}

#[test]
fn noisy_5888x4416_overflow() {
    // 26.0MP — partition_0_size exceeds 19-bit field
    let pixels = generate_noisy_rgb(5888, 4416);
    expect_partition0_overflow(&pixels, 5888, 4416, 80.0);
}

#[test]
fn noisy_7680x5760_overflow() {
    // 44.2MP — well over limit
    let pixels = generate_noisy_rgb(7680, 5760);
    expect_partition0_overflow(&pixels, 7680, 5760, 80.0);
}

#[test]
fn noisy_8192x8192_overflow() {
    // 67.1MP — well over limit
    let pixels = generate_noisy_rgb(8192, 8192);
    expect_partition0_overflow(&pixels, 8192, 8192, 80.0);
}

// --- Shape doesn't matter, only total pixel count ---

#[test]
fn noisy_8192x3456_wide_overflow() {
    // 28.3MP wide — overflows
    let pixels = generate_noisy_rgb(8192, 3456);
    expect_partition0_overflow(&pixels, 8192, 3456, 80.0);
}

#[test]
fn noisy_3456x8192_tall_overflow() {
    // 28.3MP tall — overflows
    let pixels = generate_noisy_rgb(3456, 8192);
    expect_partition0_overflow(&pixels, 3456, 8192, 80.0);
}

// --- Quality affects compression ratio, thus partition size ---

#[test]
fn noisy_7680x5760_q10() {
    // q10 compresses aggressively — partition stays small enough. Passes.
    let pixels = generate_noisy_rgb(7680, 5760);
    roundtrip_rgb(&pixels, 7680, 5760, 10.0);
}

#[test]
fn noisy_7680x5760_q50_overflow() {
    // q50 — partition too large. Overflows.
    let pixels = generate_noisy_rgb(7680, 5760);
    expect_partition0_overflow(&pixels, 7680, 5760, 50.0);
}
