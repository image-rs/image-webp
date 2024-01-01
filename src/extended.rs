use super::lossless::LosslessDecoder;
use crate::decoder::DecodingError;
use byteorder::ReadBytesExt;
use std::convert::TryInto;
use std::io::Read;

#[derive(Debug, Clone)]
pub(crate) struct WebPExtendedInfo {
    pub(crate) alpha: bool,

    pub(crate) canvas_width: u32,
    pub(crate) canvas_height: u32,

    pub(crate) icc_profile: bool,
    pub(crate) exif_metadata: bool,
    pub(crate) xmp_metadata: bool,
    pub(crate) animation: bool,

    pub(crate) background_color: [u8; 4],
}

/// Composites a frame onto a canvas.
///
/// Starts by filling the canvas with the background color, if provided. Then copies or blends the
/// frame onto the canvas.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_frame(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    clear_color: Option<[u8; 4]>,
    frame: &[u8],
    frame_offset_x: u32,
    frame_offset_y: u32,
    frame_width: u32,
    frame_height: u32,
    frame_has_alpha: bool,
    frame_use_alpha_blending: bool,
) {
    if frame_offset_x == 0
        && frame_offset_y == 0
        && frame_width == canvas_width
        && frame_height == canvas_height
        && !frame_use_alpha_blending
    {
        if frame_has_alpha {
            canvas.copy_from_slice(frame);
        } else {
            for (input, output) in frame.chunks_exact(3).zip(canvas.chunks_exact_mut(4)) {
                output[..3].copy_from_slice(input);
                output[3] = 255;
            }
        }
        return;
    }

    if let Some(clear_color) = clear_color {
        if frame_has_alpha {
            for pixel in canvas.chunks_exact_mut(4) {
                pixel.copy_from_slice(&clear_color);
            }
        } else {
            for pixel in canvas.chunks_exact_mut(3) {
                pixel.copy_from_slice(&clear_color[..3]);
            }
        }
    }

    let width = frame_width.min(canvas_width.saturating_sub(frame_offset_x)) as usize;
    let height = frame_height.min(canvas_height.saturating_sub(frame_offset_y)) as usize;

    if frame_has_alpha && frame_use_alpha_blending {
        for y in 0..height {
            for x in 0..width {
                let frame_index = (x + y * frame_width as usize) * 4;
                let canvas_index = ((x + frame_offset_x as usize)
                    + (y + frame_offset_y as usize) * canvas_width as usize)
                    * 4;

                let input = &frame[frame_index..][..4];
                let output = &mut canvas[canvas_index..][..4];

                let blended =
                    do_alpha_blending(input.try_into().unwrap(), output.try_into().unwrap());
                output.copy_from_slice(&blended);
            }
        }
    } else if frame_has_alpha {
        for y in 0..height {
            let frame_index = (y * frame_width as usize) * 4;
            let canvas_index = (y + frame_offset_y as usize) * canvas_width as usize * 4;

            canvas[canvas_index..][..width * 4].copy_from_slice(&frame[frame_index..][..width * 4]);
        }
    } else {
        for y in 0..height {
            let index = (y * frame_width as usize) * 3;
            let canvas_index = (y + frame_offset_y as usize) * canvas_width as usize * 4;

            let input = &frame[index..][..width * 3];
            let output = &mut canvas[canvas_index..][..width * 4];

            for (input, output) in input.chunks_exact(3).zip(output.chunks_exact_mut(4)) {
                output[..3].copy_from_slice(input);
                output[3] = 255;
            }
        }
    }
}

fn do_alpha_blending(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    let canvas_alpha = f64::from(canvas[3]);
    let buffer_alpha = f64::from(buffer[3]);
    let blend_alpha_f64 = buffer_alpha + canvas_alpha * (1.0 - buffer_alpha / 255.0);
    //value should be between 0 and 255, this truncates the fractional part
    let blend_alpha: u8 = blend_alpha_f64 as u8;

    let blend_rgb: [u8; 3] = if blend_alpha == 0 {
        [0, 0, 0]
    } else {
        let mut rgb = [0u8; 3];
        for i in 0..3 {
            let canvas_f64 = f64::from(canvas[i]);
            let buffer_f64 = f64::from(buffer[i]);

            let val = (buffer_f64 * buffer_alpha
                + canvas_f64 * canvas_alpha * (1.0 - buffer_alpha / 255.0))
                / blend_alpha_f64;
            //value should be between 0 and 255, this truncates the fractional part
            rgb[i] = val as u8;
        }

        rgb
    };

    [blend_rgb[0], blend_rgb[1], blend_rgb[2], blend_alpha]
}

pub(crate) fn get_alpha_predictor(
    x: usize,
    y: usize,
    width: usize,
    filtering_method: FilteringMethod,
    image_slice: &[u8],
) -> u8 {
    match filtering_method {
        FilteringMethod::None => 0,
        FilteringMethod::Horizontal => {
            if x == 0 && y == 0 {
                0
            } else if x == 0 {
                let index = (y - 1) * width + x;
                image_slice[index * 4 + 3]
            } else {
                let index = y * width + x - 1;
                image_slice[index * 4 + 3]
            }
        }
        FilteringMethod::Vertical => {
            if x == 0 && y == 0 {
                0
            } else if y == 0 {
                let index = y * width + x - 1;
                image_slice[index * 4 + 3]
            } else {
                let index = (y - 1) * width + x;
                image_slice[index * 4 + 3]
            }
        }
        FilteringMethod::Gradient => {
            let (left, top, top_left) = match (x, y) {
                (0, 0) => (0, 0, 0),
                (0, y) => {
                    let above_index = (y - 1) * width + x;
                    let val = image_slice[above_index * 4 + 3];
                    (val, val, val)
                }
                (x, 0) => {
                    let before_index = y * width + x - 1;
                    let val = image_slice[before_index * 4 + 3];
                    (val, val, val)
                }
                (x, y) => {
                    let left_index = y * width + x - 1;
                    let left = image_slice[left_index * 4 + 3];
                    let top_index = (y - 1) * width + x;
                    let top = image_slice[top_index * 4 + 3];
                    let top_left_index = (y - 1) * width + x - 1;
                    let top_left = image_slice[top_left_index * 4 + 3];

                    (left, top, top_left)
                }
            };

            let combination = i16::from(left) + i16::from(top) - i16::from(top_left);
            i16::clamp(combination, 0, 255).try_into().unwrap()
        }
    }
}

pub(crate) fn read_extended_header<R: Read>(
    reader: &mut R,
) -> Result<WebPExtendedInfo, DecodingError> {
    let chunk_flags = reader.read_u8()?;

    let reserved_first = chunk_flags & 0b11000000;
    let icc_profile = chunk_flags & 0b00100000 != 0;
    let alpha = chunk_flags & 0b00010000 != 0;
    let exif_metadata = chunk_flags & 0b00001000 != 0;
    let xmp_metadata = chunk_flags & 0b00000100 != 0;
    let animation = chunk_flags & 0b00000010 != 0;
    let reserved_second = chunk_flags & 0b00000001;

    let reserved_third = read_3_bytes(reader)?;

    if reserved_first != 0 || reserved_second != 0 || reserved_third != 0 {
        return Err(DecodingError::ReservedBitSet);
    }

    let canvas_width = read_3_bytes(reader)? + 1;
    let canvas_height = read_3_bytes(reader)? + 1;

    //product of canvas dimensions cannot be larger than u32 max
    if u32::checked_mul(canvas_width, canvas_height).is_none() {
        return Err(DecodingError::ImageTooLarge);
    }

    let info = WebPExtendedInfo {
        icc_profile,
        alpha,
        exif_metadata,
        xmp_metadata,
        animation,
        canvas_width,
        canvas_height,
        background_color: [0; 4],
    };

    Ok(info)
}

pub(crate) fn read_3_bytes<R: Read>(reader: &mut R) -> Result<u32, DecodingError> {
    let mut buffer: [u8; 3] = [0; 3];
    reader.read_exact(&mut buffer)?;
    let value: u32 =
        (u32::from(buffer[2]) << 16) | (u32::from(buffer[1]) << 8) | u32::from(buffer[0]);
    Ok(value)
}

#[derive(Debug)]
pub(crate) struct AlphaChunk {
    _preprocessing: bool,
    pub(crate) filtering_method: FilteringMethod,
    pub(crate) data: Vec<u8>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum FilteringMethod {
    None,
    Horizontal,
    Vertical,
    Gradient,
}

pub(crate) fn read_alpha_chunk<R: Read>(
    reader: &mut R,
    width: u16,
    height: u16,
) -> Result<AlphaChunk, DecodingError> {
    let info_byte = reader.read_u8()?;

    let reserved = info_byte & 0b11000000;
    let preprocessing = (info_byte & 0b00110000) >> 4;
    let filtering = (info_byte & 0b00001100) >> 2;
    let compression = info_byte & 0b00000011;

    if reserved != 0 {
        return Err(DecodingError::ReservedBitSet);
    }

    let preprocessing = match preprocessing {
        0 => false,
        1 => true,
        _ => return Err(DecodingError::ReservedBitSet),
    };

    let filtering_method = match filtering {
        0 => FilteringMethod::None,
        1 => FilteringMethod::Horizontal,
        2 => FilteringMethod::Vertical,
        3 => FilteringMethod::Gradient,
        _ => unreachable!(),
    };

    let lossless_compression = match compression {
        0 => false,
        1 => true,
        _ => return Err(DecodingError::InvalidCompressionMethod),
    };

    let data = if lossless_compression {
        let mut decoder = LosslessDecoder::new(reader);
        let frame = decoder.decode_frame(Some((width, height)))?;

        let mut data = vec![0u8; usize::from(width) * usize::from(height)];
        frame.fill_green(&mut data);
        data
    } else {
        let mut framedata = vec![0; width as usize * height as usize];
        reader.read_exact(&mut framedata)?;
        framedata
    };

    let chunk = AlphaChunk {
        _preprocessing: preprocessing,
        filtering_method,
        data,
    };

    Ok(chunk)
}
