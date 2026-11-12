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

#[test]
fn lossless_huffman_respects_memory_limit() {
    let bytes = std::fs::read("tests/images/gallery2/1_webp_ll.webp").unwrap();
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).unwrap();
    let mut output = vec![0; decoder.output_buffer_size().unwrap()];
    decoder.set_memory_limit(0);
    assert!(matches!(
        decoder.read_image(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
}

#[test]
fn lossless_indexed_transform_respects_memory_limit() {
    let bytes =
        std::fs::read("tests/images/regression/lossless_indexed_1bit_palette.webp").unwrap();
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    let mut output = vec![0; decoder.output_buffer_size().unwrap()];
    // Fixture-specific budget that reaches the indexed transform but cannot fit its scratch space.
    const LIMIT_BELOW_INDEXED_SCRATCH: usize = 121_222;
    decoder.set_memory_limit(LIMIT_BELOW_INDEXED_SCRATCH);
    assert!(matches!(
        decoder.read_image(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));

    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    // Fixture-specific budget verified to fit all temporary allocations.
    const LIMIT_FOR_INDEXED_IMAGE: usize = 200_000;
    decoder.set_memory_limit(LIMIT_FOR_INDEXED_IMAGE);
    decoder.read_image(&mut output).unwrap();
}

#[test]
fn compressed_alpha_respects_memory_limit() {
    let bytes = std::fs::read("tests/images/gallery2/1_webp_a.webp").unwrap();
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    let mut output = vec![0; decoder.output_buffer_size().unwrap()];
    decoder.set_memory_limit(0);
    assert!(matches!(
        decoder.read_image(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    decoder.read_image(&mut output).unwrap();
}

#[test]
fn animated_lossless_frame_respects_memory_limit() {
    let bytes = std::fs::read("tests/images/animated/random_lossless.webp").unwrap();
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    let mut output = vec![0; decoder.output_buffer_size().unwrap()];
    decoder.set_memory_limit(0);
    assert!(matches!(
        decoder.read_frame(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(&bytes)).unwrap();
    decoder.read_frame(&mut output).unwrap();
}

#[test]
fn animated_lossy_frame_respects_memory_limit() {
    let bytes = std::fs::read("tests/images/animated/random_lossy.webp")
        .expect("animated lossy fixture exists");
    let mut decoder =
        image_webp::WebPDecoder::new(Cursor::new(bytes)).expect("animated lossy fixture is valid");
    let mut output = vec![0; decoder.output_buffer_size().expect("output size fits")];
    decoder.set_memory_limit(0);
    assert!(matches!(
        decoder.read_frame(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
}

#[test]
fn alpha_animation_shares_budget_and_reuses_canvas() {
    const RIFF_HEADER_SIZE: usize = 12;
    const VP8X_CHUNK_SIZE: usize = 18;
    const VP8X_FLAGS_OFFSET: usize = RIFF_HEADER_SIZE + 8;
    const VP8X_DIMENSIONS_OFFSET: usize = RIFF_HEADER_SIZE + 12;
    const ANIMATION_FLAG: u8 = 2;
    const FRAME_HEADER_SIZE: usize = 16;
    const FRAME_OFFSET_SIZE: usize = 6;
    const FRAME_DIMENSIONS_SIZE: usize = 6;
    const FRAME_DURATION_OFFSET: usize = FRAME_OFFSET_SIZE + FRAME_DIMENSIONS_SIZE;
    const FRAME_COUNT: usize = 2;
    const FIXTURE_MEMORY_LIMIT: usize = 4 * 1024 * 1024;

    let still = std::fs::read("tests/images/gallery2/1_webp_a.webp").expect("alpha fixture exists");
    let mut webp = still[..RIFF_HEADER_SIZE + VP8X_CHUNK_SIZE].to_vec();
    webp[VP8X_FLAGS_OFFSET] |= ANIMATION_FLAG;
    webp.extend_from_slice(b"ANIM\x06\0\0\0\0\0\0\0\0\0");
    let chunks = &still[RIFF_HEADER_SIZE + VP8X_CHUNK_SIZE..];
    let frame_size =
        u32::try_from(FRAME_HEADER_SIZE + chunks.len()).expect("fixture frame size fits u32");
    for _ in 0..FRAME_COUNT {
        webp.extend_from_slice(b"ANMF");
        webp.extend_from_slice(&frame_size.to_le_bytes());
        let mut frame_header = [0; FRAME_HEADER_SIZE];
        frame_header[FRAME_OFFSET_SIZE..FRAME_OFFSET_SIZE + FRAME_DIMENSIONS_SIZE].copy_from_slice(
            &still[VP8X_DIMENSIONS_OFFSET..VP8X_DIMENSIONS_OFFSET + FRAME_DIMENSIONS_SIZE],
        );
        frame_header[FRAME_DURATION_OFFSET] = 1;
        webp.extend_from_slice(&frame_header);
        webp.extend_from_slice(chunks);
    }
    let riff_size = u32::try_from(webp.len() - 8).expect("fixture RIFF size fits u32");
    webp[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(webp))
        .expect("constructed alpha animation is valid");
    let mut output = vec![0; decoder.output_buffer_size().expect("output size fits")];
    let frame_and_canvas_limit = output.len() * 2 - 1;
    decoder.set_memory_limit(frame_and_canvas_limit);
    assert!(matches!(
        decoder.read_frame(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
    decoder.set_memory_limit(FIXTURE_MEMORY_LIMIT);
    decoder
        .read_frame(&mut output)
        .expect("first frame decodes");
    decoder.set_memory_limit(0);
    assert!(matches!(
        decoder.read_frame(&mut output),
        Err(image_webp::DecodingError::MemoryLimitExceeded)
    ));
    decoder.set_memory_limit(FIXTURE_MEMORY_LIMIT);
    decoder
        .read_frame(&mut output)
        .expect("retry decodes second frame");
}
