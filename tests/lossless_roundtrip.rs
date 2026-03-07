//! Lossless VP8L encoder roundtrip tests.
//!
//! Verifies that encode→decode produces pixel-identical output for various
//! synthetic images and VP8L configuration combinations.

use zenwebp::encoder::vp8l::{Vp8lConfig, encode_vp8l};

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

/// Encode RGB pixels to VP8L, wrap in RIFF, decode, and verify pixel-identical roundtrip.
fn assert_lossless_roundtrip(rgb: &[u8], w: u32, h: u32, config: &Vp8lConfig) {
    let vp8l =
        encode_vp8l(rgb, w, h, false, config, &enough::Unstoppable).expect("VP8L encoding failed");
    let webp = wrap_vp8l_in_riff(&vp8l);
    let (decoded, dw, dh) = zenwebp::decode_rgba(&webp).expect("decode failed");
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    let total = (w * h) as usize;
    let mut mismatches = 0;
    for i in 0..total {
        if decoded[i * 4] != rgb[i * 3]
            || decoded[i * 4 + 1] != rgb[i * 3 + 1]
            || decoded[i * 4 + 2] != rgb[i * 3 + 2]
        {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{total} pixel mismatches");
}

/// Encode with the high-level API and verify lossless roundtrip.
fn assert_lossless_roundtrip_highlevel(rgb: &[u8], w: u32, h: u32) {
    let cfg = zenwebp::EncoderConfig::new_lossless();
    let webp = zenwebp::EncodeRequest::new(&cfg, rgb, zenwebp::PixelLayout::Rgb8, w, h)
        .encode()
        .expect("high-level encode failed");
    let (decoded, dw, dh) = zenwebp::decode_rgba(&webp).expect("decode failed");
    assert_eq!(dw, w);
    assert_eq!(dh, h);
    let total = (w * h) as usize;
    let mut mismatches = 0;
    for i in 0..total {
        if decoded[i * 4] != rgb[i * 3]
            || decoded[i * 4 + 1] != rgb[i * 3 + 1]
            || decoded[i * 4 + 2] != rgb[i * 3 + 2]
        {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{total} pixel mismatches");
}

fn deterministic_noise(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut seed = 42u64;
    for b in rgb.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (seed >> 33) as u8;
    }
    rgb
}

fn horizontal_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for _y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[((x * 256 / w) & 0xFF) as u8, 100, 100]);
        }
    }
    rgb
}

fn vertical_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for _x in 0..w {
            rgb.extend_from_slice(&[100, ((y * 256 / h) & 0xFF) as u8, 100]);
        }
    }
    rgb
}

fn bidirectional_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[
                ((x * 256 / w) & 0xFF) as u8,
                ((y * 256 / h) & 0xFF) as u8,
                100,
            ]);
        }
    }
    rgb
}

// --- Two-color image ---

#[test]
fn two_color_no_transforms() {
    let (w, h) = (4, 4);
    let mut rgb = Vec::with_capacity(4 * 4 * 3);
    for i in 0..16u32 {
        if i % 2 == 0 {
            rgb.extend_from_slice(&[255, 0, 0]);
        } else {
            rgb.extend_from_slice(&[0, 0, 255]);
        }
    }
    let mut config = Vp8lConfig {
        use_predictor: false,
        use_subtract_green: false,
        use_palette: false,
        ..Default::default()
    };
    config.quality.quality = 0;
    assert_lossless_roundtrip(&rgb, w, h, &config);
}

// --- Gradient (16x16) ---

#[test]
fn gradient_16x16_no_transforms() {
    let rgb = bidirectional_gradient(16, 16);
    let config = Vp8lConfig {
        use_predictor: false,
        use_subtract_green: false,
        use_palette: false,
        ..Default::default()
    };
    assert_lossless_roundtrip(&rgb, 16, 16, &config);
}

// --- Noise roundtrip with various transform combos ---

#[test]
fn noise_512_no_transforms() {
    let rgb = deterministic_noise(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: false,
            use_subtract_green: false,
            use_palette: false,
            ..Default::default()
        },
    );
}

#[test]
fn noise_512_subtract_green_only() {
    let rgb = deterministic_noise(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: false,
            use_subtract_green: true,
            use_palette: false,
            ..Default::default()
        },
    );
}

#[test]
fn noise_512_predictor_bits9() {
    let rgb = deterministic_noise(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 9,
            ..Default::default()
        },
    );
}

#[test]
fn noise_512_predictor_bits4() {
    let rgb = deterministic_noise(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

#[test]
fn noise_smaller_sizes() {
    for sz in [64, 128, 256] {
        let rgb = deterministic_noise(sz, sz);
        assert_lossless_roundtrip(
            &rgb,
            sz,
            sz,
            &Vp8lConfig {
                use_predictor: true,
                use_subtract_green: false,
                use_palette: false,
                predictor_bits: 4,
                ..Default::default()
            },
        );
    }
}

// --- Predictor on 512x512 synthetic images ---

#[test]
fn pred_512_solid() {
    let rgb = [128u8, 128, 128].repeat(512 * 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

#[test]
fn pred_512_horizontal_gradient() {
    let rgb = horizontal_gradient(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

#[test]
fn pred_512_noise() {
    let rgb = deterministic_noise(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

#[test]
fn pred_512_two_halves() {
    let mut rgb = Vec::with_capacity(512 * 512 * 3);
    for y in 0..512u32 {
        for x in 0..512u32 {
            if y < 256 {
                rgb.extend_from_slice(&[(x / 2) as u8, 100, 100]);
            } else {
                rgb.extend_from_slice(&[100, (y / 2) as u8, 100]);
            }
        }
    }
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

#[test]
fn pred_512_bidirectional_gradient() {
    let rgb = bidirectional_gradient(512, 512);
    assert_lossless_roundtrip(
        &rgb,
        512,
        512,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );
}

// --- Predictor mode forcing (32x32 images) ---

#[test]
fn pred_simple_single_horizontal_mode() {
    let rgb = horizontal_gradient(32, 32);
    assert_lossless_roundtrip(
        &rgb,
        32,
        32,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 3,
            ..Default::default()
        },
    );
}

#[test]
fn pred_simple_single_vertical_mode() {
    let rgb = vertical_gradient(32, 32);
    assert_lossless_roundtrip(
        &rgb,
        32,
        32,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 3,
            ..Default::default()
        },
    );
}

#[test]
fn pred_simple_two_modes() {
    let (w, h) = (32u32, 32u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            if y < h / 2 {
                rgb.extend_from_slice(&[(x * 8) as u8, 100, 100]);
            } else {
                rgb.extend_from_slice(&[100, (y * 8) as u8, 100]);
            }
        }
    }
    assert_lossless_roundtrip(
        &rgb,
        w,
        h,
        &Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 3,
            ..Default::default()
        },
    );
}

#[test]
fn pred_simple_three_modes() {
    let (w, h) = (32u32, 32u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            if y < h / 2 && x < w / 2 {
                rgb.extend_from_slice(&[(x * 8) as u8, 100, 100]);
            } else if y < h / 2 {
                rgb.extend_from_slice(&[100, (y * 8) as u8, 100]);
            } else {
                rgb.extend_from_slice(&[200, 200, 200]);
            }
        }
    }
    for bits in [3, 4] {
        assert_lossless_roundtrip(
            &rgb,
            w,
            h,
            &Vp8lConfig {
                use_predictor: true,
                use_subtract_green: false,
                use_palette: false,
                predictor_bits: bits,
                ..Default::default()
            },
        );
    }
}

// --- Multi-quadrant predictor debug (32x32) ---

#[test]
fn pred_debug_multi_quadrant() {
    let (w, h) = (32u32, 32u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            if x < 16 && y < 16 {
                rgb.extend_from_slice(&[(x * 16) as u8, 128, 64]);
            } else if x >= 16 && y < 16 {
                rgb.extend_from_slice(&[64, (y * 16) as u8, 128]);
            } else if x < 16 {
                rgb.extend_from_slice(&[200, 100, 50]);
            } else {
                let c = if (x + y) % 2 == 0 { 255u8 } else { 0 };
                rgb.extend_from_slice(&[c, c, c]);
            }
        }
    }
    for bits in 2..=5u8 {
        assert_lossless_roundtrip(
            &rgb,
            w,
            h,
            &Vp8lConfig {
                use_predictor: true,
                use_subtract_green: false,
                use_palette: false,
                predictor_bits: bits,
                ..Default::default()
            },
        );
    }
}

// --- Full 14 predictor modes: gradient + subtract green ---

#[test]
fn predictor_gradient_with_subtract_green() {
    let (w, h) = (64u32, 64u32);
    let rgb = bidirectional_gradient(w, h);
    let config = Vp8lConfig {
        use_predictor: true,
        use_subtract_green: true,
        use_palette: false,
        predictor_bits: 4,
        ..Default::default()
    };
    assert_lossless_roundtrip(&rgb, w, h, &config);

    // Also verify the file is smaller with predictor than without
    let vp8l_with = encode_vp8l(&rgb, w, h, false, &config, &enough::Unstoppable).unwrap();
    let config_no = Vp8lConfig {
        use_predictor: false,
        use_subtract_green: true,
        use_palette: false,
        ..Default::default()
    };
    let vp8l_without = encode_vp8l(&rgb, w, h, false, &config_no, &enough::Unstoppable).unwrap();
    assert!(
        vp8l_with.len() < vp8l_without.len(),
        "predictor should reduce size for gradient: {} vs {}",
        vp8l_with.len(),
        vp8l_without.len()
    );
}

// --- High-level API roundtrip ---

#[test]
fn highlevel_lossless_gradient() {
    let rgb = bidirectional_gradient(64, 64);
    assert_lossless_roundtrip_highlevel(&rgb, 64, 64);
}

#[test]
fn highlevel_lossless_noise() {
    let rgb = deterministic_noise(128, 128);
    assert_lossless_roundtrip_highlevel(&rgb, 128, 128);
}
