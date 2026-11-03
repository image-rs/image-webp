#![no_main]

use libfuzzer_sys::{fuzz_target, Corpus};
use std::io::Cursor;

const COLOR_TYPES: [image_webp::ColorType; 4] = [
    image_webp::ColorType::L8,
    image_webp::ColorType::La8,
    image_webp::ColorType::Rgb8,
    image_webp::ColorType::Rgba8,
];

fuzz_target!(|input: &[u8]| -> Corpus {
    let Some((config, data)) = input.split_first_chunk::<8>() else {
        return Corpus::Reject;
    };

    // Configure input parameters
    let width = u32::from(u16::from_le_bytes([config[0], config[1]]));
    let height = u32::from(u16::from_le_bytes([config[2], config[3]]));
    if width == 0 || height == 0 || width >= 16384 || height >= 16384 {
        return Corpus::Reject;
    }
    let depth_minus_1 = config[4];
    let Some(color_type) = COLOR_TYPES.get(depth_minus_1 as usize) else {
        return Corpus::Reject;
    };
    let depth = u32::from(depth_minus_1 + 1);
    if width.strict_mul(height).strict_mul(depth)
        != data.len().min(u32::MAX as usize).try_into().unwrap()
    {
        return Corpus::Reject;
    }
    let lossy = config[5] & (1 << 0) == 0;
    let predictor_tf = config[5] & (1 << 1) == 0;
    let mut params = image_webp::EncoderParams::default();
    params.use_lossy = lossy;
    params.use_predictor_transform = predictor_tf;
    params.lossy_quality = config[6];
    if params.lossy_quality > 100 {
        return Corpus::Reject;
    }

    // Encode data as image
    let mut image = Vec::new();
    let mut encoder = image_webp::WebPEncoder::new(&mut image);
    encoder.set_params(params);
    encoder
        .encode(data, width, height, *color_type)
        .expect("Every valid input should be encodable");

    // Check encoded image can be decoded
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(image))
        .expect("Encoder should produce valid header");
    let (dec_width, dec_height) = decoder.dimensions();
    assert_eq!((width, height), (dec_width, dec_height));

    let dec_bpp = if decoder.has_alpha() { 4 } else { 3 };
    let decoded_bytes = width.strict_mul(height).strict_mul(dec_bpp) as usize;

    // This allocation is bounded by a constant multiple of `input.len()`.
    let mut dec_data = vec![0; decoded_bytes];
    decoder
        .read_image(&mut dec_data)
        .expect("Encoder should produce valid frame");

    if !lossy
        && matches!(
            color_type,
            image_webp::ColorType::Rgb8 | image_webp::ColorType::Rgba8
        )
    {
        assert!(data == dec_data, "Lossless encoder failed to roundtrip")
    }
    // NOTE: if lossy encoding makes any hard accuracy guarantees in
    // the future, these should be checked here

    Corpus::Keep
});
