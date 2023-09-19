use std::convert::TryInto;
use std::io::{self, Cursor, Error, Read};

use crate::decoder::DecodingError;

// use super::decoder::{
//     read_chunk, read_fourcc, read_len_cursor, DecodingError::ChunkHeaderInvalid, WebPRiffChunk,
// };
use super::lossless::{LosslessDecoder, LosslessFrame};
use super::vp8::{Frame as VP8Frame, Vp8Decoder};
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Clone)]
struct RgbImage {
    _width: u32,
    _height: u32,
    data: Vec<u8>,
}

#[derive(Clone)]
struct RgbaImage {
    _width: u32,
    _height: u32,
    data: Vec<u8>,
}

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

// enum ExtendedImageData {
//     Animation {
//         frames: Vec<AnimatedFrame>,
//         anim_info: WebPAnimatedInfo,
//     },
//     Static(WebPStatic),
// }

// pub(crate) struct ExtendedImage {
//     info: WebPExtendedInfo,
//     image: ExtendedImageData,

//     canvas: Option<Vec<u8>>,
//     next_frame: usize,
// }

// impl ExtendedImage {
//     pub(crate) fn dimensions(&self) -> (u32, u32) {
//         (self.info.canvas_width, self.info.canvas_height)
//     }

//     pub(crate) fn has_animation(&self) -> bool {
//         self.info.animation
//     }

//     pub(crate) fn icc_profile(&self) -> Option<Vec<u8>> {
//         self.info.icc_profile_contents.clone()
//     }

//     pub(crate) fn has_alpha(&self) -> bool {
//         match &self.image {
//             ExtendedImageData::Animation { frames, .. } => &frames[0].image,
//             ExtendedImageData::Static(image) => image,
//         }
//         .has_alpha()
//     }

//     pub(crate) fn next_frame(&mut self) -> Result<Option<(&[u8], u32)>, DecodingError> {
//         if let ExtendedImageData::Animation { frames, anim_info } = &self.image {
//             match frames.get(self.next_frame) {
//                 Some(anim_image) => {
//                     if self.canvas.is_none() {
//                         self.canvas = Some(vec![
//                             0;
//                             self.info.canvas_width as usize
//                                 * self.info.canvas_height as usize
//                         ]);
//                     }

//                     let dispose_previous =
//                         self.next_frame == 0 || frames[self.next_frame - 1].dispose;
//                     let bg_color = if self.info.alpha {
//                         [0; 4]
//                     } else {
//                         anim_info.background_color
//                     };

//                     Self::composite_frame(
//                         self.canvas.as_mut().unwrap(),
//                         self.info.canvas_width,
//                         self.info.canvas_height,
//                         dispose_previous.then_some(bg_color),
//                         anim_image,
//                     );

//                     self.next_frame += 1;
//                     if self.next_frame == frames.len() {
//                         self.next_frame = 0;
//                     }

//                     Ok(self.canvas.as_ref().map(|c| (&**c, anim_image.duration)))
//                 }
//                 None => Ok(None),
//             }
//         } else {
//             Ok(None)
//         }
//     }

//     pub(crate) fn read_extended_chunks<R: Read>(
//         reader: &mut R,
//         mut info: WebPExtendedInfo,
//     ) -> Result<ExtendedImage, DecodingError> {
//         let mut anim_info: Option<WebPAnimatedInfo> = None;
//         let mut anim_frames: Vec<AnimatedFrame> = Vec::new();
//         let mut static_frame: Option<WebPStatic> = None;
//         //go until end of file and while chunk headers are valid
//         while let Some((mut cursor, chunk)) = read_extended_chunk(reader)? {
//             match chunk {
//                 WebPRiffChunk::EXIF | WebPRiffChunk::XMP => {
//                     //ignore these chunks
//                 }
//                 WebPRiffChunk::ANIM => {
//                     if anim_info.is_none() {
//                         anim_info = Some(Self::read_anim_info(&mut cursor)?);
//                     }
//                 }
//                 WebPRiffChunk::ANMF => {
//                     let frame = read_anim_frame(cursor, info.canvas_width, info.canvas_height)?;
//                     anim_frames.push(frame);
//                 }
//                 WebPRiffChunk::ALPH => {
//                     if static_frame.is_none() {
//                         let alpha_chunk =
//                             read_alpha_chunk(&mut cursor, info.canvas_width, info.canvas_height)?;

//                         let vp8_frame = read_lossy_with_chunk(reader)?;

//                         let img = WebPStatic::from_alpha_lossy(alpha_chunk, vp8_frame)?;

//                         static_frame = Some(img);
//                     }
//                 }
//                 WebPRiffChunk::ICCP => {
//                     let mut icc_profile = Vec::new();
//                     cursor.read_to_end(&mut icc_profile)?;
//                     info.icc_profile_contents = Some(icc_profile);
//                 }
//                 WebPRiffChunk::VP8 => {
//                     if static_frame.is_none() {
//                         let vp8_frame = read_lossy(cursor)?;

//                         let img = WebPStatic::from_lossy(vp8_frame)?;

//                         static_frame = Some(img);
//                     }
//                 }
//                 WebPRiffChunk::VP8L => {
//                     if static_frame.is_none() {
//                         let mut lossless_decoder = LosslessDecoder::new(cursor);
//                         let frame = lossless_decoder.decode_frame()?;
//                         let image = WebPStatic::Lossless(frame.clone());

//                         static_frame = Some(image);
//                     }
//                 }
//                 _ => return Err(ChunkHeaderInvalid(chunk.to_fourcc()).into()),
//             }
//         }

//         let image = if let Some(info) = anim_info {
//             if anim_frames.is_empty() {
//                 return Err(DecodingError::IoError(io::ErrorKind::UnexpectedEof.into()));
//             }
//             ExtendedImageData::Animation {
//                 frames: anim_frames,
//                 anim_info: info,
//             }
//         } else if let Some(frame) = static_frame {
//             ExtendedImageData::Static(frame)
//         } else {
//             //reached end of file too early before image data was reached
//             return Err(DecodingError::IoError(io::ErrorKind::UnexpectedEof.into()));
//         };

//         let image = ExtendedImage {
//             image,
//             info,
//             canvas: None,
//             next_frame: 0,
//         };

//         Ok(image)
//     }

//     fn read_anim_info<R: Read>(reader: &mut R) -> Result<WebPAnimatedInfo, DecodingError> {
//         let mut colors: [u8; 4] = [0; 4];
//         reader.read_exact(&mut colors)?;

//         //background color is [blue, green, red, alpha]
//         let background_color = [colors[2], colors[1], colors[0], colors[3]];

//         let loop_count = reader.read_u16::<LittleEndian>()?;

//         let info = WebPAnimatedInfo {
//             background_color,
//             _loop_count: loop_count,
//         };

//         Ok(info)
//     }

//     pub(crate) fn fill_buf(&self, buf: &mut [u8]) {
//         match &self.image {
//             ExtendedImageData::Animation { frames, anim_info } => {
//                 let bg_color = if self.info.alpha {
//                     [0; 4]
//                 } else {
//                     anim_info.background_color
//                 };
//                 Self::composite_frame(
//                     buf,
//                     self.info.canvas_width,
//                     self.info.canvas_height,
//                     Some(bg_color),
//                     &frames[0], // will always have at least one frame
//                 );
//             }
//             ExtendedImageData::Static(image) => {
//                 image.fill_buf(buf);
//             }
//         }
//     }

//     pub(crate) fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodingError> {
//         match &mut self.image {
//             ExtendedImageData::Animation { anim_info, .. } => {
//                 anim_info.background_color = color;
//                 Ok(())
//             }
//             _ => Err(DecodingError::InvalidParameter(
//                 "Background color can only be set on animated webp".to_owned(),
//             )),
//         }
//     }
// }

/// Composites a frame onto a canvas.
///
/// Starts by filling the canvas with the background color, if provided. Then copies or blends
/// the frame onto the canvas.
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
            let frame_index = (y * frame_width as usize) * 3;
            let canvas_index = (y + frame_offset_y as usize) * canvas_width as usize * 4;

            canvas[canvas_index..][..width * 4].copy_from_slice(&frame[frame_index..][..width * 3]);
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

#[derive(Debug)]
struct WebPAnimatedInfo {
    background_color: [u8; 4],
    _loop_count: u16,
}

// pub(crate) struct AnimatedFrame {
//     offset_x: u32,
//     offset_y: u32,
//     width: u32,
//     height: u32,
//     pub(crate) duration: u32,
//     use_alpha_blending: bool,
//     dispose: bool,
//     image: WebPStatic,
// }

// /// Reads a chunk, but silently ignores unknown chunks at the end of a file
// fn read_extended_chunk<R>(
//     r: &mut R,
// ) -> Result<Option<(Cursor<Vec<u8>>, WebPRiffChunk)>, DecodingError>
// where
//     R: Read,
// {
//     let mut unknown_chunk = Ok(());

//     while let Some(chunk) = read_fourcc(r)? {
//         let cursor = read_len_cursor(r)?;
//         match chunk {
//             Ok(chunk) => return unknown_chunk.and(Ok(Some((cursor, chunk)))),
//             Err(err) => unknown_chunk = unknown_chunk.and(Err(err)),
//         }
//     }

//     Ok(None)
// }

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
        let value: u32 = if reserved_first != 0 {
            reserved_first.into()
        } else if reserved_second != 0 {
            reserved_second.into()
        } else {
            reserved_third
        };
        return Err(DecodingError::InfoBitsInvalid {
            name: "reserved",
            value,
        }
        .into());
    }

    let canvas_width = read_3_bytes(reader)? + 1;
    let canvas_height = read_3_bytes(reader)? + 1;

    //product of canvas dimensions cannot be larger than u32 max
    if u32::checked_mul(canvas_width, canvas_height).is_none() {
        return Err(DecodingError::ImageTooLarge.into());
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

// pub(crate) fn read_anim_frame<R: Read>(
//     mut reader: R,
//     canvas_width: u32,
//     canvas_height: u32,
// ) -> Result<AnimatedFrame, DecodingError> {
//     //offsets for the frames are twice the values
//     let frame_x = read_3_bytes(&mut reader)? * 2;
//     let frame_y = read_3_bytes(&mut reader)? * 2;

//     let frame_width = read_3_bytes(&mut reader)? + 1;
//     let frame_height = read_3_bytes(&mut reader)? + 1;

//     if frame_x + frame_width > canvas_width || frame_y + frame_height > canvas_height {
//         return Err(DecodingError::FrameOutsideImage.into());
//     }

//     let duration = read_3_bytes(&mut reader)?;

//     let frame_info = reader.read_u8()?;
//     let reserved = frame_info & 0b11111100;
//     if reserved != 0 {
//         return Err(DecodingError::InfoBitsInvalid {
//             name: "reserved",
//             value: reserved.into(),
//         }
//         .into());
//     }
//     let use_alpha_blending = frame_info & 0b00000010 == 0;
//     let dispose = frame_info & 0b00000001 != 0;

//     //read normal bitstream now
//     let static_image = read_image(&mut reader, frame_width, frame_height)?;

//     let frame = AnimatedFrame {
//         offset_x: frame_x,
//         offset_y: frame_y,
//         width: frame_width,
//         height: frame_height,
//         duration,
//         use_alpha_blending,
//         dispose,
//         image: static_image,
//     };

//     Ok(frame)
// }

pub(crate) fn read_3_bytes<R: Read>(reader: &mut R) -> Result<u32, DecodingError> {
    let mut buffer: [u8; 3] = [0; 3];
    reader.read_exact(&mut buffer)?;
    let value: u32 =
        (u32::from(buffer[2]) << 16) | (u32::from(buffer[1]) << 8) | u32::from(buffer[0]);
    Ok(value)
}

// fn read_lossy_with_chunk<R: Read>(reader: &mut R) -> Result<VP8Frame, DecodingError> {
//     let (cursor, chunk) =
//         read_chunk(reader)?.ok_or_else(|| Error::from(io::ErrorKind::UnexpectedEof))?;

//     if chunk != WebPRiffChunk::VP8 {
//         return Err(ChunkHeaderInvalid(chunk.to_fourcc()).into());
//     }

//     read_lossy(cursor)
// }

// fn read_lossy(cursor: Cursor<Vec<u8>>) -> Result<VP8Frame, DecodingError> {
//     let mut vp8_decoder = Vp8Decoder::new(cursor);
//     let frame = vp8_decoder.decode_frame()?;

//     Ok(frame.clone())
// }

// fn read_image<R: Read>(
//     reader: &mut R,
//     width: u32,
//     height: u32,
// ) -> Result<WebPStatic, DecodingError> {
//     let chunk = read_chunk(reader)?;

//     match chunk {
//         Some((cursor, WebPRiffChunk::VP8)) => {
//             let mut vp8_decoder = Vp8Decoder::new(cursor);
//             let frame = vp8_decoder.decode_frame()?;

//             let img = WebPStatic::from_lossy(frame.clone())?;

//             Ok(img)
//         }
//         Some((cursor, WebPRiffChunk::VP8L)) => {
//             let mut lossless_decoder = LosslessDecoder::new(cursor);
//             let frame = lossless_decoder.decode_frame()?;

//             let img = WebPStatic::Lossless(frame.clone());

//             Ok(img)
//         }
//         Some((mut cursor, WebPRiffChunk::ALPH)) => {
//             let alpha_chunk = read_alpha_chunk(&mut cursor, width, height)?;

//             let vp8_frame = read_lossy_with_chunk(reader)?;

//             let img = WebPStatic::from_alpha_lossy(alpha_chunk, vp8_frame)?;

//             Ok(img)
//         }
//         None => Err(DecodingError::IoError(io::ErrorKind::UnexpectedEof.into())),
//         Some((_, chunk)) => Err(ChunkHeaderInvalid(chunk.to_fourcc()).into()),
//     }
// }

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
    width: u32,
    height: u32,
) -> Result<AlphaChunk, DecodingError> {
    let info_byte = reader.read_u8()?;

    let reserved = info_byte & 0b11000000;
    let preprocessing = (info_byte & 0b00110000) >> 4;
    let filtering = (info_byte & 0b00001100) >> 2;
    let compression = info_byte & 0b00000011;

    if reserved != 0 {
        return Err(DecodingError::InfoBitsInvalid {
            name: "reserved",
            value: reserved.into(),
        }
        .into());
    }

    let preprocessing = match preprocessing {
        0 => false,
        1 => true,
        _ => {
            return Err(DecodingError::InfoBitsInvalid {
                name: "reserved",
                value: preprocessing.into(),
            }
            .into())
        }
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
        _ => {
            return Err(DecodingError::InfoBitsInvalid {
                name: "lossless compression",
                value: compression.into(),
            }
            .into())
        }
    };

    let mut framedata = Vec::new();
    reader.read_to_end(&mut framedata)?;

    let data = if lossless_compression {
        let cursor = io::Cursor::new(framedata);

        let mut decoder = LosslessDecoder::new(cursor);
        //this is a potential problem for large images; would require rewriting lossless decoder to use u32 for width and height
        let width: u16 = width.try_into().map_err(|_| DecodingError::ImageTooLarge)?;
        let height: u16 = height
            .try_into()
            .map_err(|_| DecodingError::ImageTooLarge)?;
        let frame = decoder.decode_frame_implicit_dims(width, height)?;

        let mut data = vec![0u8; usize::from(width) * usize::from(height)];

        frame.fill_green(&mut data);

        data
    } else {
        framedata
    };

    let chunk = AlphaChunk {
        _preprocessing: preprocessing,
        filtering_method,
        data,
    };

    Ok(chunk)
}
