use std::{
    fs::create_dir_all,
    io::{Cursor, Write},
    path::PathBuf,
};

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
        .write_image_data(&data)
        .unwrap();
    f.flush().unwrap();
}

fn reference_test(file: &str) {
    // Prepare WebP decoder
    let contents = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(contents)).unwrap();
    let (width, height) = decoder.dimensions();

    // Decode reference PNG
    let reference_path = if decoder.is_animated() {
        format!("tests/reference/{file}-1.png")
    } else {
        format!("tests/reference/{file}.png")
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
        // defined, but we currently attempt to match the dwebp's "-nofancy" conversion option.
        //
        // TODO: Investigate why we don't get bit exact output for all pixels.
        let num_bytes_different = data
            .iter()
            .zip(reference_data.iter())
            .filter(|(a, b)| a != b)
            .count();
        let percentage_different = 100 * num_bytes_different / data.len();
        if percentage_different >= 10 {
            save_image(
                &data,
                file,
                if decoder.is_animated() { Some(1) } else { None },
                decoder.has_alpha(),
                width,
                height,
            );
        }
        assert!(percentage_different < 10, "More than 10% of pixels differ");
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

            if !decoder.is_lossy() {
                if data != reference_data {
                    save_image(&data, file, Some(i), decoder.has_alpha(), width, height);
                    panic!("Pixel mismatch")
                }
            } else {
                let num_bytes_different = data
                    .iter()
                    .zip(reference_data.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                let percentage_different = 100 * num_bytes_different / data.len();
                if percentage_different >= 10 {
                    save_image(&data, file, Some(i), decoder.has_alpha(), width, height);
                }
                assert!(percentage_different < 10, "More than 10% of pixels differ");
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

reftest!(gallery1, 1, 2, 3, 4, 5);
reftest!(gallery2, 1_webp_ll, 2_webp_ll, 3_webp_ll, 4_webp_ll, 5_webp_ll);
reftest!(gallery2, 1_webp_a, 2_webp_a, 3_webp_a, 4_webp_a, 5_webp_a);
reftest!(animated, random_lossless, random_lossy);
reftest!(regression, color_index, dark);
