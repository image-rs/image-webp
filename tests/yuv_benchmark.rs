#![cfg(not(target_arch = "wasm32"))]
//! Benchmark comparing scalar vs SIMD YUV conversion.
//!
//! Run with: cargo test --release yuv_benchmark -- --nocapture

use std::time::Instant;

// Scalar implementation (copied from yuv.rs for comparison)
mod scalar {
    const YUV_FIX: i32 = 16;
    const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

    fn rgb_to_y(rgb: &[u8]) -> u8 {
        let luma = 16839 * i32::from(rgb[0]) + 33059 * i32::from(rgb[1]) + 6420 * i32::from(rgb[2]);
        ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
    }

    fn rgb_to_u_raw(rgb: &[u8]) -> i32 {
        -9719 * i32::from(rgb[0]) - 19081 * i32::from(rgb[1])
            + 28800 * i32::from(rgb[2])
            + (128 << YUV_FIX)
    }

    fn rgb_to_v_raw(rgb: &[u8]) -> i32 {
        28800 * i32::from(rgb[0]) - 24116 * i32::from(rgb[1]) - 4684 * i32::from(rgb[2])
            + (128 << YUV_FIX)
    }

    fn rgb_to_u_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
        let u1 = rgb_to_u_raw(rgb1);
        let u2 = rgb_to_u_raw(rgb2);
        let u3 = rgb_to_u_raw(rgb3);
        let u4 = rgb_to_u_raw(rgb4);
        ((u1 + u2 + u3 + u4 + (YUV_HALF << 2)) >> (YUV_FIX + 2)) as u8
    }

    fn rgb_to_v_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
        let v1 = rgb_to_v_raw(rgb1);
        let v2 = rgb_to_v_raw(rgb2);
        let v3 = rgb_to_v_raw(rgb3);
        let v4 = rgb_to_v_raw(rgb4);
        ((v1 + v2 + v3 + v4 + (YUV_HALF << 2)) >> (YUV_FIX + 2)) as u8
    }

    pub fn convert_image_yuv_scalar(
        image_data: &[u8],
        width: u16,
        height: u16,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        const BPP: usize = 3;
        let width = usize::from(width);
        let height = usize::from(height);
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let y_size = 16 * mb_width * 16 * mb_height;
        let luma_width = 16 * mb_width;
        let chroma_width = 8 * mb_width;
        let chroma_size = 8 * mb_width * 8 * mb_height;
        let mut y_bytes = vec![0u8; y_size];
        let mut u_bytes = vec![0u8; chroma_size];
        let mut v_bytes = vec![0u8; chroma_size];

        let row_pairs = height / 2;
        let odd_height = height & 1 != 0;
        let col_pairs = width / 2;
        let odd_width = width & 1 != 0;

        for row_pair in 0..row_pairs {
            let src_row1 = row_pair * 2;
            let src_row2 = src_row1 + 1;
            let chroma_row = row_pair;

            for col_pair in 0..col_pairs {
                let src_col1 = col_pair * 2;
                let src_col2 = src_col1 + 1;
                let chroma_col = col_pair;

                let rgb1 = &image_data[(src_row1 * width + src_col1) * BPP..][..BPP];
                let rgb2 = &image_data[(src_row1 * width + src_col2) * BPP..][..BPP];
                let rgb3 = &image_data[(src_row2 * width + src_col1) * BPP..][..BPP];
                let rgb4 = &image_data[(src_row2 * width + src_col2) * BPP..][..BPP];

                y_bytes[src_row1 * luma_width + src_col1] = rgb_to_y(rgb1);
                y_bytes[src_row1 * luma_width + src_col2] = rgb_to_y(rgb2);
                y_bytes[src_row2 * luma_width + src_col1] = rgb_to_y(rgb3);
                y_bytes[src_row2 * luma_width + src_col2] = rgb_to_y(rgb4);

                u_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_u_avg(rgb1, rgb2, rgb3, rgb4);
                v_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_v_avg(rgb1, rgb2, rgb3, rgb4);
            }

            if odd_width {
                let src_col = width - 1;
                let chroma_col = col_pairs;
                let rgb1 = &image_data[(src_row1 * width + src_col) * BPP..][..BPP];
                let rgb3 = &image_data[(src_row2 * width + src_col) * BPP..][..BPP];
                y_bytes[src_row1 * luma_width + src_col] = rgb_to_y(rgb1);
                y_bytes[src_row2 * luma_width + src_col] = rgb_to_y(rgb3);
                u_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_u_avg(rgb1, rgb1, rgb3, rgb3);
                v_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_v_avg(rgb1, rgb1, rgb3, rgb3);
            }
        }

        if odd_height {
            let src_row = height - 1;
            let chroma_row = row_pairs;
            for col_pair in 0..col_pairs {
                let src_col1 = col_pair * 2;
                let src_col2 = src_col1 + 1;
                let chroma_col = col_pair;
                let rgb1 = &image_data[(src_row * width + src_col1) * BPP..][..BPP];
                let rgb2 = &image_data[(src_row * width + src_col2) * BPP..][..BPP];
                y_bytes[src_row * luma_width + src_col1] = rgb_to_y(rgb1);
                y_bytes[src_row * luma_width + src_col2] = rgb_to_y(rgb2);
                u_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_u_avg(rgb1, rgb2, rgb1, rgb2);
                v_bytes[chroma_row * chroma_width + chroma_col] =
                    rgb_to_v_avg(rgb1, rgb2, rgb1, rgb2);
            }
            if odd_width {
                let src_col = width - 1;
                let chroma_col = col_pairs;
                let rgb = &image_data[(src_row * width + src_col) * BPP..][..BPP];
                y_bytes[src_row * luma_width + src_col] = rgb_to_y(rgb);
                u_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_u_avg(rgb, rgb, rgb, rgb);
                v_bytes[chroma_row * chroma_width + chroma_col] = rgb_to_v_avg(rgb, rgb, rgb, rgb);
            }
        }

        // Edge padding
        for y in 0..height {
            let last_y = y_bytes[y * luma_width + width - 1];
            for x in width..luma_width {
                y_bytes[y * luma_width + x] = last_y;
            }
        }
        for y in height..(mb_height * 16) {
            for x in 0..luma_width {
                y_bytes[y * luma_width + x] = y_bytes[(height - 1) * luma_width + x];
            }
        }

        let chroma_height = height.div_ceil(2);
        let actual_chroma_width = width.div_ceil(2);
        for y in 0..chroma_height {
            let last_u = u_bytes[y * chroma_width + actual_chroma_width - 1];
            let last_v = v_bytes[y * chroma_width + actual_chroma_width - 1];
            for x in actual_chroma_width..chroma_width {
                u_bytes[y * chroma_width + x] = last_u;
                v_bytes[y * chroma_width + x] = last_v;
            }
        }
        for y in chroma_height..(mb_height * 8) {
            for x in 0..chroma_width {
                u_bytes[y * chroma_width + x] = u_bytes[(chroma_height - 1) * chroma_width + x];
                v_bytes[y * chroma_width + x] = v_bytes[(chroma_height - 1) * chroma_width + x];
            }
        }

        (y_bytes, u_bytes, v_bytes)
    }
}

// SIMD implementation using yuv crate
mod simd {
    use yuv::{
        YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
        rgb_to_yuv420,
    };

    pub fn convert_image_yuv_simd(
        image_data: &[u8],
        width: u16,
        height: u16,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let width_usize = usize::from(width);
        let height_usize = usize::from(height);
        let mb_width = width_usize.div_ceil(16);
        let mb_height = height_usize.div_ceil(16);
        let luma_width = 16 * mb_width;
        let chroma_width = 8 * mb_width;
        let y_size = 16 * mb_width * 16 * mb_height;
        let chroma_size = 8 * mb_width * 8 * mb_height;

        let mut yuv_image = YuvPlanarImageMut::<u8>::alloc(
            width as u32,
            height as u32,
            YuvChromaSubsampling::Yuv420,
        );

        rgb_to_yuv420(
            &mut yuv_image,
            image_data,
            (width_usize * 3) as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt601,
            YuvConversionMode::Balanced,
        )
        .expect("yuv conversion failed");

        let y_src = yuv_image.y_plane.borrow();
        let u_src = yuv_image.u_plane.borrow();
        let v_src = yuv_image.v_plane.borrow();

        let src_chroma_width = width_usize.div_ceil(2);
        let src_chroma_height = height_usize.div_ceil(2);

        let mut y_bytes = vec![0u8; y_size];
        let mut u_bytes = vec![0u8; chroma_size];
        let mut v_bytes = vec![0u8; chroma_size];

        // Copy Y plane with padding
        for y in 0..height_usize {
            let src_start = y * width_usize;
            let dst_start = y * luma_width;
            y_bytes[dst_start..dst_start + width_usize]
                .copy_from_slice(&y_src[src_start..src_start + width_usize]);
            let last_y = y_bytes[dst_start + width_usize - 1];
            for x in width_usize..luma_width {
                y_bytes[dst_start + x] = last_y;
            }
        }
        if height_usize < mb_height * 16 {
            let last_row: Vec<u8> =
                y_bytes[(height_usize - 1) * luma_width..height_usize * luma_width].to_vec();
            for y in height_usize..(mb_height * 16) {
                let dst_row = y * luma_width;
                y_bytes[dst_row..dst_row + luma_width].copy_from_slice(&last_row);
            }
        }

        // Copy U/V planes with padding
        for y in 0..src_chroma_height {
            let src_start = y * src_chroma_width;
            let dst_start = y * chroma_width;
            u_bytes[dst_start..dst_start + src_chroma_width]
                .copy_from_slice(&u_src[src_start..src_start + src_chroma_width]);
            v_bytes[dst_start..dst_start + src_chroma_width]
                .copy_from_slice(&v_src[src_start..src_start + src_chroma_width]);
            let last_u = u_bytes[dst_start + src_chroma_width - 1];
            let last_v = v_bytes[dst_start + src_chroma_width - 1];
            for x in src_chroma_width..chroma_width {
                u_bytes[dst_start + x] = last_u;
                v_bytes[dst_start + x] = last_v;
            }
        }
        if src_chroma_height < mb_height * 8 {
            let last_u_row: Vec<u8> = u_bytes
                [(src_chroma_height - 1) * chroma_width..src_chroma_height * chroma_width]
                .to_vec();
            let last_v_row: Vec<u8> = v_bytes
                [(src_chroma_height - 1) * chroma_width..src_chroma_height * chroma_width]
                .to_vec();
            for y in src_chroma_height..(mb_height * 8) {
                let dst_row = y * chroma_width;
                u_bytes[dst_row..dst_row + chroma_width].copy_from_slice(&last_u_row);
                v_bytes[dst_row..dst_row + chroma_width].copy_from_slice(&last_v_row);
            }
        }

        (y_bytes, u_bytes, v_bytes)
    }
}

fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 7 + y * 13) % 256) as u8; // R
            data[idx + 1] = ((x * 11 + y * 5) % 256) as u8; // G
            data[idx + 2] = ((x * 3 + y * 17) % 256) as u8; // B
        }
    }
    data
}

#[test]
fn benchmark_yuv_conversion() {
    let sizes = [
        (640, 480, "VGA"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K"),
    ];

    let iterations = 10;

    println!("\n=== YUV Conversion Benchmark ===");
    println!("Iterations per size: {}", iterations);
    println!();

    for (width, height, name) in &sizes {
        let image = generate_test_image(*width, *height);
        let pixels = width * height;
        let megapixels = pixels as f64 / 1_000_000.0;

        // Warm up
        let _ = scalar::convert_image_yuv_scalar(&image, *width as u16, *height as u16);
        let _ = simd::convert_image_yuv_simd(&image, *width as u16, *height as u16);

        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = scalar::convert_image_yuv_scalar(&image, *width as u16, *height as u16);
        }
        let scalar_elapsed = start.elapsed();
        let scalar_mpps = (megapixels * iterations as f64) / scalar_elapsed.as_secs_f64();

        // Benchmark SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd::convert_image_yuv_simd(&image, *width as u16, *height as u16);
        }
        let simd_elapsed = start.elapsed();
        let simd_mpps = (megapixels * iterations as f64) / simd_elapsed.as_secs_f64();

        let speedup = scalar_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64();

        println!("{} ({}x{}):", name, width, height);
        println!(
            "  Scalar: {:>8.2} ms ({:.1} Mpix/s)",
            scalar_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            scalar_mpps
        );
        println!(
            "  SIMD:   {:>8.2} ms ({:.1} Mpix/s)",
            simd_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            simd_mpps
        );
        println!("  Speedup: {:.1}x", speedup);
        println!();
    }
}

#[test]
fn verify_yuv_output_similarity() {
    // Test that SIMD and scalar produce similar results
    // Note: They won't be identical due to different rounding/coefficient precision
    let width = 64u16;
    let height = 64u16;
    let image = generate_test_image(width as usize, height as usize);

    let (y_scalar, u_scalar, v_scalar) = scalar::convert_image_yuv_scalar(&image, width, height);
    let (y_simd, u_simd, v_simd) = simd::convert_image_yuv_simd(&image, width, height);

    // Calculate max differences
    let y_max_diff = y_scalar
        .iter()
        .zip(y_simd.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);
    let u_max_diff = u_scalar
        .iter()
        .zip(u_simd.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);
    let v_max_diff = v_scalar
        .iter()
        .zip(v_simd.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap_or(0);

    println!("\n=== YUV Output Comparison ===");
    println!("Max Y difference: {}", y_max_diff);
    println!("Max U difference: {}", u_max_diff);
    println!("Max V difference: {}", v_max_diff);

    // Allow up to 2 levels difference (yuv crate uses different precision)
    assert!(y_max_diff <= 2, "Y difference too large: {}", y_max_diff);
    assert!(u_max_diff <= 2, "U difference too large: {}", u_max_diff);
    assert!(v_max_diff <= 2, "V difference too large: {}", v_max_diff);
}
