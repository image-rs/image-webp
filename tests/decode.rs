use std::io::Cursor;

fn refence_test(file: &str) {
    // Decode reference PNG
    let reference_path = format!("tests/reference/{file}.png",);
    let reference_contents = std::fs::read(reference_path).unwrap();
    let mut reference_decoder = png::Decoder::new(Cursor::new(reference_contents))
        .read_info()
        .unwrap();
    assert_eq!(reference_decoder.info().bit_depth, png::BitDepth::Eight);
    let mut reference_data = vec![0; reference_decoder.output_buffer_size()];
    reference_decoder.next_frame(&mut reference_data).unwrap();

    // Prepare WebP decoder
    let contents = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
    let mut decoder = webp::WebPDecoder::new(Cursor::new(contents)).unwrap();
    let (width, height) = decoder.dimensions();

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

    // // Save mismatching images
    // if data != reference_data {
    //     let mut f = std::fs::File::create(format!("tests/out/{file}.png")).unwrap();
    //     let mut encoder = png::Encoder::new(&mut f, width, height);
    //     encoder.set_color(png::ColorType::Rgba);
    //     encoder
    //         .write_header()
    //         .unwrap()
    //         .write_image_data(&data)
    //         .unwrap();
    //     f.flush().unwrap();
    // }

    // Compare pixels
    if !decoder.is_lossy() {
        if data != reference_data {
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
        assert!(
            100 * num_bytes_different / data.len() < 10,
            "More than 10% of pixels differ"
        );
    }
}

macro_rules! reftest {
    ($name:expr) => {
        paste::paste! {
            #[test]
            fn [<reftest_ $name>]() {
                refence_test(stringify!($name));
            }
        }
    };
    ($name:expr, $($tail:expr),+) => {
        reftest!( $name );
        reftest!( $($tail),+ );
    }
}

reftest!(1, 2, 3, 4, 5);
reftest!(1_webp_ll, 2_webp_ll, 3_webp_ll, 4_webp_ll, 5_webp_ll);
reftest!(1_webp_a, 2_webp_a, 3_webp_a, 4_webp_a, 5_webp_a);
