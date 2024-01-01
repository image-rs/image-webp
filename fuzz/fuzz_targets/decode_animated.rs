#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|input: &[u8]| {
    let decoder = webp::WebPDecoder::new(Cursor::new(input));
    if let Ok(mut decoder) = decoder {
        let (width, height) = decoder.dimensions();
        let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
        let total_bytes = width as usize * height as usize * bytes_per_pixel;
        if total_bytes <= 1024 * 1024 * 1024 {
            if decoder.is_animated() {
                let mut data = vec![0; total_bytes];
                while let Ok(_delay) = decoder.read_frame(&mut data) {};
            }
        }
    }
});
