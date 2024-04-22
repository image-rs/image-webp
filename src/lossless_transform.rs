use crate::decoder::DecodingError;

use super::lossless::subsample_size;

#[derive(Debug, Clone)]
pub(crate) enum TransformType {
    PredictorTransform {
        size_bits: u8,
        predictor_data: Vec<u8>,
    },
    ColorTransform {
        size_bits: u8,
        transform_data: Vec<u8>,
    },
    SubtractGreen,
    ColorIndexingTransform {
        table_size: u16,
        table_data: Vec<u8>,
    },
}

pub(crate) fn apply_predictor_transform(
    image_data: &mut [u8],
    width: u16,
    height: u16,
    size_bits: u8,
    predictor_data: &[u8],
) -> Result<(), DecodingError> {
    let block_xsize = usize::from(subsample_size(width, size_bits));
    let width = usize::from(width);
    let height = usize::from(height);

    //handle top and left borders specially
    //this involves ignoring mode and just setting prediction values like this
    image_data[3] = image_data[3].wrapping_add(255);

    for x in 4..width * 4 {
        image_data[x] = image_data[x].wrapping_add(image_data[x - 4]);
    }

    for y in 1..height {
        for i in 0..4 {
            image_data[y * width * 4 + i] =
                image_data[y * width * 4 + i].wrapping_add(image_data[(y - 1) * width * 4 + i]);
        }
    }

    for y in 1..height {
        for block_x in 0..block_xsize {
            let block_index = (y >> size_bits) * block_xsize + block_x;
            let predictor = predictor_data[block_index * 4 + 1];
            let start_index = (y * width + (block_x << size_bits).max(1)) * 4;
            let end_index = (y * width + ((block_x + 1) << size_bits).min(width)) * 4;

            match predictor {
                0 => {
                    for i in ((start_index + 3)..end_index).step_by(4) {
                        image_data[i] = image_data[i].wrapping_add(0xff);
                    }
                }
                1 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(image_data[i - 4]);
                    }
                }
                2 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(image_data[i - width * 4]);
                    }
                }
                3 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(image_data[i - width * 4 + 4]);
                    }
                }
                4 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(image_data[i - width * 4 - 4]);
                    }
                }
                5 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(average2(
                            average2(image_data[i - 4], image_data[i - width * 4 + 4]),
                            image_data[i - width * 4],
                        ));
                    }
                }
                6 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(average2(
                            image_data[i - 4],
                            image_data[i - width * 4 - 4],
                        ));
                    }
                }
                7 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i]
                            .wrapping_add(average2(image_data[i - 4], image_data[i - width * 4]));
                    }
                }
                8 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(average2(
                            image_data[i - width * 4 - 4],
                            image_data[i - width * 4],
                        ));
                    }
                }
                9 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(average2(
                            image_data[i - width * 4],
                            image_data[i - width * 4 + 4],
                        ));
                    }
                }
                10 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(average2(
                            average2(image_data[i - 4], image_data[i - width * 4 - 4]),
                            average2(image_data[i - width * 4], image_data[i - width * 4 + 4]),
                        ));
                    }
                }
                11 => {
                    for i in (start_index..end_index).step_by(4) {
                        let lred = image_data[i - 4];
                        let lgreen = image_data[i - 3];
                        let lblue = image_data[i - 2];
                        let lalpha = image_data[i - 1];

                        let tred = image_data[i - width * 4];
                        let tgreen = image_data[i - width * 4 + 1];
                        let tblue = image_data[i - width * 4 + 2];
                        let talpha = image_data[i - width * 4 + 3];

                        let tlred = image_data[i - width * 4 - 4];
                        let tlgreen = image_data[i - width * 4 - 3];
                        let tlblue = image_data[i - width * 4 - 2];
                        let tlalpha = image_data[i - width * 4 - 1];

                        let predict_red = i16::from(lred) + i16::from(tred) - i16::from(tlred);
                        let predict_green =
                            i16::from(lgreen) + i16::from(tgreen) - i16::from(tlgreen);
                        let predict_blue = i16::from(lblue) + i16::from(tblue) - i16::from(tlblue);
                        let predict_alpha =
                            i16::from(lalpha) + i16::from(talpha) - i16::from(tlalpha);

                        let predict_left = i16::abs(predict_red - i16::from(lred))
                            + i16::abs(predict_green - i16::from(lgreen))
                            + i16::abs(predict_blue - i16::from(lblue))
                            + i16::abs(predict_alpha - i16::from(lalpha));
                        let predict_top = i16::abs(predict_red - i16::from(tred))
                            + i16::abs(predict_green - i16::from(tgreen))
                            + i16::abs(predict_blue - i16::from(tblue))
                            + i16::abs(predict_alpha - i16::from(talpha));

                        if predict_left < predict_top {
                            image_data[i] = image_data[i].wrapping_add(lred);
                            image_data[i + 1] = image_data[i + 1].wrapping_add(lgreen);
                            image_data[i + 2] = image_data[i + 2].wrapping_add(lblue);
                            image_data[i + 3] = image_data[i + 3].wrapping_add(lalpha);
                        } else {
                            image_data[i] = image_data[i].wrapping_add(tred);
                            image_data[i + 1] = image_data[i + 1].wrapping_add(tgreen);
                            image_data[i + 2] = image_data[i + 2].wrapping_add(tblue);
                            image_data[i + 3] = image_data[i + 3].wrapping_add(talpha);
                        }
                    }
                }
                12 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(clamp_add_subtract_full(
                            i16::from(image_data[i - 4]),
                            i16::from(image_data[i - width * 4]),
                            i16::from(image_data[i - width * 4 - 4]),
                        ));
                    }
                }
                13 => {
                    for i in start_index..end_index {
                        image_data[i] = image_data[i].wrapping_add(clamp_add_subtract_half(
                            i16::from(average2(image_data[i - 4], image_data[i - width * 4])),
                            i16::from(image_data[i - width * 4 - 4]),
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

pub(crate) fn apply_color_transform(
    image_data: &mut [u8],
    width: u16,
    size_bits: u8,
    transform_data: &[u8],
) {
    let block_xsize = usize::from(subsample_size(width, size_bits));
    let width = usize::from(width);

    for (y, row) in image_data.chunks_exact_mut(width * 4).enumerate() {
        for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
            let block_index = (y >> size_bits) * block_xsize + (x >> size_bits);
            let red_to_blue = transform_data[block_index * 4];
            let green_to_blue = transform_data[block_index * 4 + 1];
            let green_to_red = transform_data[block_index * 4 + 2];

            let green = u32::from(pixel[1]);
            let mut temp_red = u32::from(pixel[0]);
            let mut temp_blue = u32::from(pixel[2]);

            temp_red += color_transform_delta(green_to_red as i8, green as i8);
            temp_blue += color_transform_delta(green_to_blue as i8, green as i8);
            temp_blue += color_transform_delta(red_to_blue as i8, temp_red as i8);

            pixel[0] = (temp_red & 0xff) as u8;
            pixel[2] = (temp_blue & 0xff) as u8;
        }
    }
}

pub(crate) fn apply_subtract_green_transform(image_data: &mut [u8]) {
    for pixel in image_data.chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_add(pixel[1]);
        pixel[2] = pixel[2].wrapping_add(pixel[1]);
    }
}

pub(crate) fn apply_color_indexing_transform(
    image_data: &mut Vec<u8>,
    width: u16,
    height: u16,
    table_size: u16,
    table_data: &[u8],
) -> Result<(), DecodingError> {
    let mut new_image_data = Vec::with_capacity(usize::from(width) * usize::from(height) * 4);

    todo!();

    // let width_bits: u8 = if table_size <= 2 {
    //     3
    // } else if table_size <= 4 {
    //     2
    // } else if table_size <= 16 {
    //     1
    // } else {
    //     0
    // };

    // let bits_per_pixel = 8 >> width_bits;
    // let mask = (1 << bits_per_pixel) - 1;

    // let mut src = 0;
    // let width = usize::from(width);

    // let pixels_per_byte = 1 << width_bits;
    // let count_mask = pixels_per_byte - 1;
    // let mut packed_pixels = 0;

    // for _y in 0..usize::from(height) {
    //     for x in 0..width {
    //         if (x & count_mask) == 0 {
    //             packed_pixels = (image_data[src] >> 8) & 0xff;
    //             src += 1;
    //         }

    //         let pixels: usize = (packed_pixels & mask).try_into().unwrap();
    //         let new_val = if pixels >= table_size.into() {
    //             0x00000000
    //         } else {
    //             table_data[pixels]
    //         };

    //         new_image_data.push(new_val);

    //         packed_pixels >>= bits_per_pixel;
    //     }
    // }

    *image_data = new_image_data;
    Ok(())
}

//predictor functions

/// Get average of 2 bytes
fn average2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) / 2).try_into().unwrap()
}

/// Clamp add subtract full on one part
fn clamp_add_subtract_full(a: i16, b: i16, c: i16) -> u8 {
    (a + b - c).clamp(0, 255) as u8
}

/// Clamp add subtract half on one part
fn clamp_add_subtract_half(a: i16, b: i16) -> u8 {
    (a + (a - b) / 2).clamp(0, 255) as u8
}

/// Does color transform on 2 numbers
fn color_transform_delta(t: i8, c: i8) -> u32 {
    ((i16::from(t) * i16::from(c)) as u32) >> 5
}
