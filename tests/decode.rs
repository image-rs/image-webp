use std::fs::create_dir_all;
use std::io::{Cursor, Write};
use std::path::PathBuf;

// Write images to `out/` directory on test failure - useful for diffing with reference images.
const WRITE_IMAGES_ON_FAILURE: bool = false;

fn save_image(data: &[u8], file: &str, i: Option<u32>, has_alpha: bool, width: u32, height: u32) {
    if !WRITE_IMAGES_ON_FAILURE {
        return;
    }

    let path = PathBuf::from(match i {
        Some(i) => format!("tests/out/{file}-{i}.png"),
        None => format!("tests/out/{file}.png"),
    });

    let directory = path.parent().unwrap();
    if !directory.exists() {
        create_dir_all(directory).unwrap();
    }

    let mut f = std::fs::File::create(path).unwrap();

    let mut encoder = png::Encoder::new(&mut f, width, height);
    if has_alpha {
        encoder.set_color(png::ColorType::Rgba);
    } else {
        encoder.set_color(png::ColorType::Rgb);
    }
    encoder
        .write_header()
        .unwrap()
        .write_image_data(data)
        .unwrap();
    f.flush().unwrap();
}

fn reference_test(file: &str) {
    reference_test_with_options(file, image_webp::WebPDecodeOptions::default(), None);
}

fn reference_test_with_options(
    file: &str,
    options: image_webp::WebPDecodeOptions,
    custom_reference_file: Option<&str>,
) {
    // Prepare WebP decoder
    let contents = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
    let mut decoder =
        image_webp::WebPDecoder::new_with_options(Cursor::new(contents), options).unwrap();
    let (width, height) = decoder.dimensions();

    // Decode reference PNG
    let reference_file = custom_reference_file.unwrap_or(file);
    let reference_path = if decoder.is_animated() {
        format!("tests/reference/{reference_file}-1.png")
    } else {
        format!("tests/reference/{reference_file}.png")
    };
    let reference_contents = std::fs::read(reference_path).unwrap();
    let mut reference_decoder = png::Decoder::new(Cursor::new(reference_contents))
        .read_info()
        .unwrap();
    assert_eq!(reference_decoder.info().bit_depth, png::BitDepth::Eight);
    let mut reference_data = vec![0; reference_decoder.output_buffer_size()];
    reference_decoder.next_frame(&mut reference_data).unwrap();

    // Compare metadata
    assert_eq!(width, reference_decoder.info().width);
    assert_eq!(height, reference_decoder.info().height);
    match reference_decoder.info().color_type {
        png::ColorType::Rgb => assert!(!decoder.has_alpha()),
        png::ColorType::Rgba => assert!(decoder.has_alpha()),
        _ => unreachable!(),
    }

    // Decode WebP
    let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
    let mut data = vec![0; width as usize * height as usize * bytes_per_pixel];
    decoder.read_image(&mut data).unwrap();

    // Compare pixels
    if !decoder.is_lossy() {
        if data != reference_data {
            save_image(
                &data,
                file,
                if decoder.is_animated() { Some(1) } else { None },
                decoder.has_alpha(),
                width,
                height,
            );
            panic!("Pixel mismatch")
        }
    } else {
        // NOTE: WebP lossy images are stored in YUV format. The conversion to RGB is not precisely
        // defined, but we currently attempt to match the dwebp's default conversion option.
        let num_bytes_different = data
            .iter()
            .zip(reference_data.iter())
            .filter(|(a, b)| a != b)
            .count();
        if num_bytes_different > 0 {
            save_image(
                &data,
                file,
                if decoder.is_animated() { Some(1) } else { None },
                decoder.has_alpha(),
                width,
                height,
            );
        }
        assert_eq!(num_bytes_different, 0, "Pixel mismatch");
    }

    // If the file is animated, then check all frames.
    if decoder.is_animated() {
        for i in 1..=decoder.num_frames() {
            let reference_path = PathBuf::from(format!("tests/reference/{file}-{i}.png"));
            assert!(reference_path.exists());

            let reference_contents = std::fs::read(reference_path).unwrap();
            let mut reference_decoder = png::Decoder::new(Cursor::new(reference_contents))
                .read_info()
                .unwrap();
            assert_eq!(reference_decoder.info().bit_depth, png::BitDepth::Eight);
            let mut reference_data = vec![0; reference_decoder.output_buffer_size()];
            reference_decoder.next_frame(&mut reference_data).unwrap();

            // Decode WebP
            let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
            let mut data = vec![0; width as usize * height as usize * bytes_per_pixel];
            decoder.read_frame(&mut data).unwrap();

            if decoder.is_lossy() {
                let num_bytes_different = data
                    .iter()
                    .zip(reference_data.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                if num_bytes_different > 0 {
                    save_image(&data, file, Some(i), decoder.has_alpha(), width, height);
                }
                assert_eq!(num_bytes_different, 0, "Pixel mismatch");
            } else if data != reference_data {
                save_image(&data, file, Some(i), decoder.has_alpha(), width, height);
                panic!("Pixel mismatch")
            }
        }
    }
}

macro_rules! reftest {
    ($basename:expr, $name:expr) => {
        paste::paste! {
            #[test]
            fn [<reftest_ $basename _ $name>]() {
                reference_test(concat!(stringify!($basename), "/", stringify!($name)));
            }
        }
    };
    ($basename:expr, $name:expr, $($tail:expr),+) => {
        reftest!( $basename, $name );
        reftest!( $basename, $($tail),+ );
    }
}

macro_rules! reftest_nofancy {
    ($basename:expr, $name:expr) => {
        paste::paste! {
            #[test]
            fn [<reftest_nofancy_ $basename _ $name>]() {
                let mut options = image_webp::WebPDecodeOptions::default();
                options.lossy_upsampling = image_webp::UpsamplingMethod::Simple;
                reference_test_with_options(
                    concat!(stringify!($basename), "/", stringify!($name)),
                    options,
                    Some(concat!(stringify!($basename), "_nofancy", "/", stringify!($name)))
                );
            }
        }
    };
    ($basename:expr, $name:expr, $($tail:expr),+) => {
        reftest_nofancy!( $basename, $name );
        reftest_nofancy!( $basename, $($tail),+ );
    }
}

reftest!(gallery1, 1, 2, 3, 4, 5);
reftest_nofancy!(gallery1, 1, 2, 3, 4, 5);
reftest!(gallery2, 1_webp_ll, 2_webp_ll, 3_webp_ll, 4_webp_ll, 5_webp_ll);
reftest!(gallery2, 1_webp_a, 2_webp_a, 3_webp_a, 4_webp_a, 5_webp_a);
reftest!(animated, random_lossless, random_lossy);
reftest!(
    regression,
    color_index,
    dark,
    tiny,
    lossless_indexed_1bit_palette,
    lossless_indexed_2bit_palette,
    lossless_indexed_4bit_palette
);

// Builds a minimal RIFF WebP containing only a VP8L header (no image data).
// This is enough to test dimension parsing without needing valid pixel data.
fn vp8l_dimensions_only_webp(width_minus_one: u32, height_minus_one: u32) -> Vec<u8> {
    let packed = (width_minus_one & 0x3FFF) | ((height_minus_one & 0x3FFF) << 14);
    let packed_le = packed.to_le_bytes();
    // VP8L payload: 1-byte signature + 4-byte packed header = 5 bytes
    let payload: [u8; 5] = [0x2F, packed_le[0], packed_le[1], packed_le[2], packed_le[3]];
    let chunk_size: u32 = payload.len() as u32;
    // RIFF file_size = "WEBP"(4) + "VP8L"(4) + size_field(4) + payload(5) + padding(1) = 18
    let riff_size: u32 = 4 + 4 + 4 + chunk_size + (chunk_size % 2);
    [
        b"RIFF".as_slice(),
        &riff_size.to_le_bytes(),
        b"WEBP",
        b"VP8L",
        &chunk_size.to_le_bytes(),
        &payload,
        &[0x00], // padding byte (chunk_size=5 is odd)
    ]
    .concat()
}

#[test]
fn test_vp8l_max_width_dimensions() {
    // Regression test: VP8L encodes (width - 1) in bits 0-13 and (height - 1) in bits 14-27.
    // The buggy formula `(1 + header) & 0x3FFF` returns 0 instead of 16384 when the encoded
    // value is 0x3FFF, because adding 1 carries into bit 14 before the mask is applied.
    // The correct formula is `(header & 0x3FFF) + 1`.
    let data = vp8l_dimensions_only_webp(16383, 0);
    let decoder = image_webp::WebPDecoder::new(Cursor::new(data)).unwrap();
    assert_eq!(decoder.dimensions(), (16384, 1));
}

#[test]
fn test_vp8l_max_height_dimensions() {
    let data = vp8l_dimensions_only_webp(0, 16383);
    let decoder = image_webp::WebPDecoder::new(Cursor::new(data)).unwrap();
    assert_eq!(decoder.dimensions(), (1, 16384));
}

#[test]
fn test_vp8l_max_width_and_height_dimensions() {
    let data = vp8l_dimensions_only_webp(16383, 16383);
    let decoder = image_webp::WebPDecoder::new(Cursor::new(data)).unwrap();
    assert_eq!(decoder.dimensions(), (16384, 16384));
}
