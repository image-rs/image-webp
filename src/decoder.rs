use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::io::{self, BufReader, Cursor, Read, Seek};
use std::mem;
use std::ops::Range;
use thiserror::Error;

use crate::extended::{self, get_alpha_predictor, read_alpha_chunk, WebPExtendedInfo};

use super::lossless::{LosslessDecoder, LosslessFrame};
use super::vp8::{Frame as VP8Frame, Vp8Decoder};

/// All errors that can occur when attempting to parse a WEBP container
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DecodingError {
    /// An IO error occurred while reading the file
    #[error("IO Error: {0}")]
    IoError(#[from] io::Error),

    /// RIFF's "RIFF" signature not found or invalid
    #[error("Invalid RIFF signature: {0:x?}")]
    RiffSignatureInvalid([u8; 4]),

    /// WebP's "WEBP" signature not found or invalid
    #[error("Invalid WebP signature: {0:x?}")]
    WebpSignatureInvalid([u8; 4]),

    /// An expected chunk was missing
    #[error("An expected chunk was missing")]
    ChunkMissing,

    /// Chunk Header was incorrect or invalid in its usage
    #[error("Invalid Chunk header: {0:?}")]
    ChunkHeaderInvalid([u8; 4]),

    /// Some bits were invalid
    #[error("Invalid info bits: {name} {value}")]
    InfoBitsInvalid { name: &'static str, value: u32 },

    /// Alpha chunk doesn't match the frame's size
    #[error("Alpha chunk size mismatch")]
    AlphaChunkSizeMismatch,

    /// Image is too large, either for the platform's pointer size or generally
    #[error("Image too large")]
    ImageTooLarge,

    /// Frame would go out of the canvas
    #[error("Frame outside image")]
    FrameOutsideImage,

    /// Signature of 0x2f not found
    #[error("Invalid lossless signature: {0:x?}")]
    LosslessSignatureInvalid(u8),

    /// Version Number was not zero
    #[error("Invalid lossless version number: {0}")]
    VersionNumberInvalid(u8),

    #[error("Invalid color cache bits: {0}")]
    InvalidColorCacheBits(u8),

    #[error("Invalid Huffman code")]
    HuffmanError,

    #[error("Corrupt bitstream")]
    BitStreamError,

    #[error("Invalid transform")]
    TransformError,

    /// VP8's `[0x9D, 0x01, 0x2A]` magic not found or invalid
    #[error("Invalid VP8 magic: {0:x?}")]
    Vp8MagicInvalid([u8; 3]),

    /// VP8 Decoder initialisation wasn't provided with enough data
    #[error("Not enough VP8 init data")]
    NotEnoughInitData,

    /// At time of writing, only the YUV colour-space encoded as `0` is specified
    #[error("Invalid VP8 color space: {0}")]
    ColorSpaceInvalid(u8),

    /// LUMA prediction mode was not recognised
    #[error("Invalid VP8 luma prediction mode: {0}")]
    LumaPredictionModeInvalid(i8),

    /// Intra-prediction mode was not recognised
    #[error("Invalid VP8 intra prediction mode: {0}")]
    IntraPredictionModeInvalid(i8),

    /// Chroma prediction mode was not recognised
    #[error("Invalid VP8 chroma prediction mode: {0}")]
    ChromaPredictionModeInvalid(i8),

    /// Inconsistent image sizes
    #[error("Inconsistent image sizes")]
    InconsistentImageSizes,

    /// The file may be valid, but this crate doesn't support decoding it.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Invalid function call or parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// All possible RIFF chunks in a WebP image file
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub(crate) enum WebPRiffChunk {
    RIFF,
    WEBP,
    VP8,
    VP8L,
    VP8X,
    ANIM,
    ANMF,
    ALPH,
    ICCP,
    EXIF,
    XMP,
    Unknown([u8; 4]),
}

impl WebPRiffChunk {
    pub(crate) fn from_fourcc(chunk_fourcc: [u8; 4]) -> Self {
        match &chunk_fourcc {
            b"RIFF" => Self::RIFF,
            b"WEBP" => Self::WEBP,
            b"VP8 " => Self::VP8,
            b"VP8L" => Self::VP8L,
            b"VP8X" => Self::VP8X,
            b"ANIM" => Self::ANIM,
            b"ANMF" => Self::ANMF,
            b"ALPH" => Self::ALPH,
            b"ICCP" => Self::ICCP,
            b"EXIF" => Self::EXIF,
            b"XMP " => Self::XMP,
            _ => Self::Unknown(chunk_fourcc),
        }
    }

    pub(crate) fn to_fourcc(&self) -> [u8; 4] {
        match self {
            Self::RIFF => *b"RIFF",
            Self::WEBP => *b"WEBP",
            Self::VP8 => *b"VP8 ",
            Self::VP8L => *b"VP8L",
            Self::VP8X => *b"VP8X",
            Self::ANIM => *b"ANIM",
            Self::ANMF => *b"ANMF",
            Self::ALPH => *b"ALPH",
            Self::ICCP => *b"ICCP",
            Self::EXIF => *b"EXIF",
            Self::XMP => *b"XMP ",
            Self::Unknown(fourcc) => *fourcc,
        }
    }

    pub(crate) fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
}

// enum WebPImage {
//     Lossy(VP8Frame),
//     Lossless(LosslessFrame),
//     Extended(ExtendedImage),
// }

enum ImageKind {
    Lossy,
    Lossless,
    Extended(WebPExtendedInfo),
}

struct AnimationState {
    next_frame: usize,
    loops_before_done: Option<u16>,
    next_frame_start: u64,
    dispose_next_frame: bool,
    canvas: Option<Vec<u8>>,
}
impl Default for AnimationState {
    fn default() -> Self {
        Self {
            next_frame: 0,
            loops_before_done: None,
            next_frame_start: 0,
            dispose_next_frame: true,
            canvas: None,
        }
    }
}

/// WebP Image format decoder. Currently only supports lossy RGB images or lossless RGBA images.
pub struct WebPDecoder<R> {
    r: R,

    width: u32,
    height: u32,

    num_frames: usize,
    animation: AnimationState,

    kind: ImageKind,

    chunks: HashMap<WebPRiffChunk, Range<u64>>,
}

impl<R: Read + Seek> WebPDecoder<R> {
    /// Create a new WebPDecoder from the Reader ```r```.
    /// This function takes ownership of the Reader.
    pub fn new(r: R) -> Result<WebPDecoder<R>, DecodingError> {
        let mut decoder = WebPDecoder {
            r,
            width: 0,
            height: 0,
            num_frames: 0,
            kind: ImageKind::Lossy,
            chunks: HashMap::new(),
            animation: Default::default(),
        };
        decoder.read_data()?;
        Ok(decoder)
    }

    fn read_data(&mut self) -> Result<(), DecodingError> {
        let (WebPRiffChunk::RIFF, riff_size, _) = read_chunk_header(&mut self.r)? else {
            return Err(DecodingError::ChunkHeaderInvalid(*b"RIFF"));
        };

        match &read_fourcc(&mut self.r)? {
            WebPRiffChunk::WEBP => {}
            fourcc => return Err(DecodingError::WebpSignatureInvalid(fourcc.to_fourcc()).into()),
        }

        let (chunk, chunk_size, chunk_size_rounded) = read_chunk_header(&mut self.r)?;
        let start = self.r.stream_position()?;

        match chunk {
            WebPRiffChunk::VP8 => {
                let tag = self.r.read_u24::<LittleEndian>()?;

                let keyframe = tag & 1 == 0;
                if !keyframe {
                    return Err(DecodingError::UnsupportedFeature(
                        "Non-keyframe frames".to_owned(),
                    ));
                }

                let mut tag = [0u8; 3];
                self.r.read_exact(&mut tag)?;
                if tag != [0x9d, 0x01, 0x2a] {
                    return Err(DecodingError::Vp8MagicInvalid(tag).into());
                }

                let w = self.r.read_u16::<LittleEndian>()?;
                let h = self.r.read_u16::<LittleEndian>()?;

                self.width = (w & 0x3FFF) as u32;
                self.height = (h & 0x3FFF) as u32;
                self.chunks
                    .insert(WebPRiffChunk::VP8, start..start + chunk_size as u64);
                self.kind = ImageKind::Lossy;
            }
            WebPRiffChunk::VP8L => {
                let signature = self.r.read_u8()?;
                if signature != 0x2f {
                    return Err(DecodingError::LosslessSignatureInvalid(signature));
                }

                let header = self.r.read_u24::<LittleEndian>()?;
                let version = header >> 29;
                if version != 0 {
                    return Err(DecodingError::VersionNumberInvalid(version as u8));
                }

                self.width = 1 + (header >> 6) & 0x3FFF;
                self.height = 1 + (header << 8) & 0x3FFF;
                self.chunks
                    .insert(WebPRiffChunk::VP8L, start..start + chunk_size as u64);
                self.kind = ImageKind::Lossless;
            }
            WebPRiffChunk::VP8X => {
                let mut info = extended::read_extended_header(&mut self.r)?;
                self.width = info.canvas_width;
                self.height = info.canvas_height;

                let next_chunk = if info.icc_profile {
                    WebPRiffChunk::ICCP
                } else if info.animation {
                    WebPRiffChunk::ANIM
                } else {
                    WebPRiffChunk::VP8
                };

                let mut position = start + chunk_size_rounded as u64;
                let max_position = position + riff_size.saturating_sub(12) as u64;
                self.r.seek(io::SeekFrom::Start(position))?;

                // Resist denial of service attacks by using a BufReader. In most images there
                // should be a very small number of chunks. However, nothing prevents a malicious
                // image from having an extremely large number of "unknown" chunks. Issuing
                // millions of reads and seeks against the underlying reader might be very
                // expensive.
                let mut reader = BufReader::with_capacity(64 << 10, &mut self.r);

                while position < max_position {
                    match read_chunk_header(&mut reader) {
                        Ok((chunk, chunk_size, chunk_size_rounded)) => {
                            if !chunk.is_unknown() {
                                let range = position..position + chunk_size as u64;
                                self.chunks.entry(chunk).or_insert(range);
                                if let WebPRiffChunk::ANMF = chunk {
                                    self.num_frames += 1;
                                }
                            }

                            reader.seek_relative(chunk_size_rounded as i64)?;
                            position += chunk_size_rounded as u64;
                        }
                        Err(DecodingError::IoError(e))
                            if e.kind() == io::ErrorKind::UnexpectedEof =>
                        {
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }

                if info.animation && !self.chunks.contains_key(&WebPRiffChunk::ANIM)
                    || info.animation && !self.chunks.contains_key(&WebPRiffChunk::ANMF)
                    || info.icc_profile && !self.chunks.contains_key(&WebPRiffChunk::ICCP)
                    || info.exif_metadata && !self.chunks.contains_key(&WebPRiffChunk::EXIF)
                    || info.xmp_metadata && !self.chunks.contains_key(&WebPRiffChunk::XMP)
                    || self.chunks.contains_key(&WebPRiffChunk::VP8)
                        == self.chunks.contains_key(&WebPRiffChunk::VP8L)
                {
                    return Err(DecodingError::ChunkMissing);
                }

                if let Some(chunk) = self.read_chunk(WebPRiffChunk::ANIM)? {
                    let mut cursor = Cursor::new(chunk);
                    cursor.read_exact(&mut info.background_color)?;
                    match cursor.read_u16::<LittleEndian>()? {
                        0 => self.animation.loops_before_done = None,
                        n => self.animation.loops_before_done = Some(n),
                    }
                    self.animation.next_frame_start =
                        self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start;
                }

                self.kind = ImageKind::Extended(info);
            }
            _ => return Err(DecodingError::ChunkHeaderInvalid(chunk.to_fourcc()).into()),
        };

        Ok(())
    }

    /// Returns true if the image as described by the bitstream is animated.
    pub fn has_animation(&self) -> bool {
        match &self.kind {
            ImageKind::Lossy | ImageKind::Lossless => false,
            ImageKind::Extended(extended) => extended.animation,
        }
    }

    /// Returns whether the image has an alpha channel.
    ///
    /// If so, the pixel format is Rgba8 and otherwise Rgb8.
    pub fn has_alpha(&self) -> bool {
        match &self.kind {
            ImageKind::Lossy => false,
            ImageKind::Lossless => true,
            ImageKind::Extended(extended) => extended.alpha,
        }
    }

    /// Sets the background color if the image is an extended and animated webp.
    pub fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodingError> {
        if let ImageKind::Extended(info) = &mut self.kind {
            info.background_color = color;
            Ok(())
        } else {
            Err(DecodingError::InvalidParameter(
                "Background color can only be set on animated webp".to_owned(),
            ))
        }
    }

    /// Returns the (width, height) of the image in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn read_chunk(&mut self, chunk: WebPRiffChunk) -> Result<Option<Vec<u8>>, DecodingError> {
        match self.chunks.get(&chunk) {
            Some(range) => {
                self.r.seek(io::SeekFrom::Start(range.start))?;
                let mut data = vec![0; (range.end - range.start) as usize];
                self.r.read_exact(&mut data)?;
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    pub fn icc_profile(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(WebPRiffChunk::ICCP)
    }

    pub fn exif_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(WebPRiffChunk::EXIF)
    }

    pub fn xmp_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        self.read_chunk(WebPRiffChunk::XMP)
    }

    /// Returns the raw bytes of the image.
    ///
    /// For animated images, this is the first frame.
    pub fn read_image(&mut self, buf: &mut [u8]) -> Result<(), DecodingError> {
        assert_eq!(
            u64::try_from(buf.len()),
            Ok(self.width as u64 * self.height as u64 * 4)
        );

        if self.has_animation() {
            let state = mem::take(&mut self.animation);
            self.animation.next_frame_start = self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start;
            self.read_frame(buf)?;
            self.animation = state;
        } else if let Some(range) = self.chunks.get(&WebPRiffChunk::VP8L) {
            let mut frame = LosslessDecoder::new(range_reader(&mut self.r, range.clone())?);
            let frame = frame.decode_frame()?;
            if u32::from(frame.width) != self.width || u32::from(frame.height) != self.height {
                return Err(DecodingError::InconsistentImageSizes);
            }

            frame.fill_rgba(buf);
        } else {
            let range = self
                .chunks
                .get(&WebPRiffChunk::VP8)
                .ok_or(DecodingError::ChunkMissing)?;
            // TODO: avoid cloning frame
            let frame = Vp8Decoder::new(range_reader(&mut self.r, range.clone())?)
                .decode_frame()?
                .clone();
            if u32::from(frame.width) != self.width || u32::from(frame.height) != self.height {
                return Err(DecodingError::InconsistentImageSizes);
            }

            if self.has_alpha() {
                frame.fill_rgba(buf);

                let range = self
                    .chunks
                    .get(&WebPRiffChunk::ALPH)
                    .ok_or(DecodingError::ChunkMissing)?
                    .clone();
                let alpha_chunk = read_alpha_chunk(
                    &mut range_reader(&mut self.r, range.clone())?,
                    self.width,
                    self.height,
                )?;

                for y in 0..frame.height {
                    for x in 0..frame.width {
                        let predictor: u8 = get_alpha_predictor(
                            x.into(),
                            y.into(),
                            frame.width.into(),
                            alpha_chunk.filtering_method,
                            &buf,
                        );

                        let alpha_index =
                            usize::from(y) * usize::from(frame.width) + usize::from(x);
                        let buffer_index = alpha_index * 4 + 3;

                        buf[buffer_index] = predictor.wrapping_add(alpha_chunk.data[alpha_index]);
                    }
                }
            } else {
                frame.fill_rgb(buf);
            }
        }

        Ok(())
    }

    /// Reads the next frame of the animation.
    ///
    /// Writes the frame contents into `buf` and returns the delay of the frame in milliseconds.
    /// If there are no more frames, the method returns `None` and `buf` is left unchanged.
    ///
    /// Panics if the image is not animated.
    pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<Option<u32>, DecodingError> {
        assert!(self.has_animation());

        if self.animation.loops_before_done == Some(0) {
            return Ok(None);
        }

        let ImageKind::Extended(info) = &self.kind else {
            unreachable!()
        };

        self.r
            .seek(io::SeekFrom::Start(self.animation.next_frame_start))?;

        let anmf_size = match read_chunk_header(&mut self.r)? {
            (WebPRiffChunk::ANMF, size, _) if size >= 32 => size,
            _ => return Err(DecodingError::ChunkHeaderInvalid(*b"ANMF")),
        };

        // Read ANMF chunk
        let frame_x = extended::read_3_bytes(&mut self.r)? * 2;
        let frame_y = extended::read_3_bytes(&mut self.r)? * 2;
        let frame_width = extended::read_3_bytes(&mut self.r)? + 1;
        let frame_height = extended::read_3_bytes(&mut self.r)? + 1;
        if frame_x + frame_width > self.width || frame_y + frame_height > self.height {
            return Err(DecodingError::FrameOutsideImage.into());
        }
        let duration = extended::read_3_bytes(&mut self.r)?;
        let frame_info = self.r.read_u8()?;
        let reserved = frame_info & 0b11111100;
        if reserved != 0 {
            return Err(DecodingError::InfoBitsInvalid {
                name: "reserved",
                value: reserved.into(),
            }
            .into());
        }
        let use_alpha_blending = frame_info & 0b00000010 == 0;
        let dispose = frame_info & 0b00000001 != 0;

        let clear_color = if self.animation.dispose_next_frame {
            Some(info.background_color)
        } else {
            None
        };

        //read normal bitstream now
        let (chunk, chunk_size, chunk_size_rounded) = read_chunk_header(&mut self.r)?;
        if chunk_size_rounded + 32 < anmf_size {
            return Err(DecodingError::ChunkHeaderInvalid(chunk.to_fourcc()).into());
        }

        let (frame, frame_has_alpha): (Vec<u8>, bool) = match chunk {
            WebPRiffChunk::VP8 => {
                let reader = (&mut self.r).take(chunk_size as u64);
                let mut vp8_decoder = Vp8Decoder::new(reader);
                let raw_frame = vp8_decoder.decode_frame()?;
                if raw_frame.width as u32 != frame_width || raw_frame.height as u32 != frame_height
                {
                    return Err(DecodingError::InconsistentImageSizes);
                }
                let mut rgb_frame = vec![0; frame_width as usize * frame_height as usize * 3];
                raw_frame.fill_rgb(&mut rgb_frame);
                (rgb_frame, false)
            }
            WebPRiffChunk::VP8L => {
                let reader = (&mut self.r).take(chunk_size as u64);
                let mut lossless_decoder = LosslessDecoder::new(reader);
                let frame = lossless_decoder.decode_frame()?;
                if frame.width as u32 != frame_width || frame.height as u32 != frame_height {
                    return Err(DecodingError::InconsistentImageSizes);
                }
                let mut rgba_frame = vec![0; frame_width as usize * frame_height as usize * 4];
                frame.fill_rgba(&mut rgba_frame);
                (rgba_frame, true)
            }
            WebPRiffChunk::ALPH => {
                if chunk_size_rounded + 40 < anmf_size {
                    return Err(DecodingError::ChunkHeaderInvalid(chunk.to_fourcc()).into());
                }

                // read alpha
                let next_chunk_start = self.r.stream_position()? + chunk_size_rounded as u64;
                let mut reader = (&mut self.r).take(chunk_size as u64);
                let alpha_chunk = read_alpha_chunk(&mut reader, frame_width, frame_height)?;

                // read opaque
                self.r.seek(io::SeekFrom::Start(next_chunk_start))?;
                let (next_chunk, next_chunk_size, _) = read_chunk_header(&mut self.r)?;
                if chunk_size + next_chunk_size + 40 > anmf_size {
                    return Err(DecodingError::ChunkHeaderInvalid(next_chunk.to_fourcc()).into());
                }

                let mut vp8_decoder = Vp8Decoder::new((&mut self.r).take(chunk_size as u64));
                let frame = vp8_decoder.decode_frame()?;

                let mut rgba_frame = vec![0; frame_width as usize * frame_height as usize * 4];
                frame.fill_rgba(&mut rgba_frame);

                todo!("apply alpha");

                (rgba_frame, true)
            }
            _ => return Err(DecodingError::ChunkHeaderInvalid(chunk.to_fourcc()).into()),
        };

        if self.animation.canvas.is_none() {
            self.animation.canvas = Some(vec![0; (self.width * self.height * 4) as usize]);
        }
        extended::composite_frame(
            self.animation.canvas.as_mut().unwrap(),
            self.width,
            self.height,
            clear_color,
            &frame,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            frame_has_alpha,
            use_alpha_blending,
        );

        self.animation.dispose_next_frame = dispose;
        self.animation.next_frame_start += anmf_size as u64 + 8;
        self.animation.next_frame += 1;

        if self.animation.next_frame >= self.num_frames {
            self.animation.next_frame = 0;
            if self.animation.loops_before_done.is_some() {
                *self.animation.loops_before_done.as_mut().unwrap() -= 1;
            }
            self.animation.next_frame_start = self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start;
            self.animation.dispose_next_frame = true;
        }

        Ok(Some(duration))
    }
}

pub(crate) fn range_reader<R: Read + Seek>(
    mut r: R,
    range: Range<u64>,
) -> Result<impl Read, DecodingError> {
    r.seek(io::SeekFrom::Start(range.start))?;
    Ok(r.take(range.end - range.start))
}

pub(crate) fn read_fourcc<R: Read>(mut r: R) -> Result<WebPRiffChunk, DecodingError> {
    let mut chunk_fourcc = [0; 4];
    r.read_exact(&mut chunk_fourcc)?;
    Ok(WebPRiffChunk::from_fourcc(chunk_fourcc))
}

pub(crate) fn read_chunk_header<R: Read>(
    mut r: R,
) -> Result<(WebPRiffChunk, u32, u32), DecodingError> {
    let chunk = read_fourcc(&mut r)?;
    let chunk_size = r.read_u32::<LittleEndian>()?;
    let chunk_size_rounded = chunk_size.saturating_add(chunk_size & 1);
    Ok((chunk, chunk_size, chunk_size_rounded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_with_overflow_size() {
        let bytes = vec![
            0x52, 0x49, 0x46, 0x46, 0xaf, 0x37, 0x80, 0x47, 0x57, 0x45, 0x42, 0x50, 0x6c, 0x64,
            0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfb, 0x7e, 0x73, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x40, 0xfb, 0xff, 0xff, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49,
            0x49, 0x54, 0x55, 0x50, 0x4c, 0x54, 0x59, 0x50, 0x45, 0x33, 0x37, 0x44, 0x4d, 0x46,
        ];

        let data = std::io::Cursor::new(bytes);

        let _ = WebPDecoder::new(data);
    }
}
