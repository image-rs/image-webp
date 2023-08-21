use byteorder::{LittleEndian, ReadBytesExt};
use std::convert::TryFrom;
use std::io::{self, Cursor, Read};
use thiserror::Error;

use crate::extended::ExtendedImageData;

use super::lossless::{LosslessDecoder, LosslessFrame};
use super::vp8::{Frame as VP8Frame, Vp8Decoder};

use super::extended::{read_extended_header, ExtendedImage};

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

    /// The file may be valid, but this crate doesn't support decoding it.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Invalid function call or parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// All possible RIFF chunks in a WebP image file
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

impl WebPRiffChunk {
    pub(crate) fn from_fourcc(chunk_fourcc: [u8; 4]) -> Result<Self, DecodingError> {
        match &chunk_fourcc {
            b"RIFF" => Ok(Self::RIFF),
            b"WEBP" => Ok(Self::WEBP),
            b"VP8 " => Ok(Self::VP8),
            b"VP8L" => Ok(Self::VP8L),
            b"VP8X" => Ok(Self::VP8X),
            b"ANIM" => Ok(Self::ANIM),
            b"ANMF" => Ok(Self::ANMF),
            b"ALPH" => Ok(Self::ALPH),
            b"ICCP" => Ok(Self::ICCP),
            b"EXIF" => Ok(Self::EXIF),
            b"XMP " => Ok(Self::XMP),
            _ => Err(DecodingError::ChunkHeaderInvalid(chunk_fourcc).into()),
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
        }
    }
}

enum WebPImage {
    Lossy(VP8Frame),
    Lossless(LosslessFrame),
    Extended(ExtendedImage),
}

/// WebP Image format decoder. Currently only supports lossy RGB images or lossless RGBA images.
pub struct WebPDecoder<R> {
    r: R,
    image: WebPImage,
}

impl<R: Read> WebPDecoder<R> {
    /// Create a new WebPDecoder from the Reader ```r```.
    /// This function takes ownership of the Reader.
    pub fn new(r: R) -> Result<WebPDecoder<R>, DecodingError> {
        let image = WebPImage::Lossy(Default::default());

        let mut decoder = WebPDecoder { r, image };
        decoder.read_data()?;
        Ok(decoder)
    }

    //reads the 12 bytes of the WebP file header
    fn read_riff_header(&mut self) -> Result<u32, DecodingError> {
        let mut riff = [0; 4];
        self.r.read_exact(&mut riff)?;
        if &riff != b"RIFF" {
            return Err(DecodingError::RiffSignatureInvalid(riff).into());
        }

        let size = self.r.read_u32::<LittleEndian>()?;

        let mut webp = [0; 4];
        self.r.read_exact(&mut webp)?;
        if &webp != b"WEBP" {
            return Err(DecodingError::WebpSignatureInvalid(webp).into());
        }

        Ok(size)
    }

    fn read_data(&mut self) -> Result<(), DecodingError> {
        let _size = self.read_riff_header()?;

        let chunk = read_chunk(&mut self.r)?;
        self.image = match chunk {
            Some((cursor, WebPRiffChunk::VP8)) => {
                let mut vp8_decoder = Vp8Decoder::new(cursor);
                let frame = vp8_decoder.decode_frame()?;

                WebPImage::Lossy(frame.clone())
            }
            Some((cursor, WebPRiffChunk::VP8L)) => {
                let mut lossless_decoder = LosslessDecoder::new(cursor);
                let frame = lossless_decoder.decode_frame()?;

                WebPImage::Lossless(frame.clone())
            }
            Some((mut cursor, WebPRiffChunk::VP8X)) => {
                let info = read_extended_header(&mut cursor)?;

                let image = ExtendedImage::read_extended_chunks(&mut self.r, info)?;

                WebPImage::Extended(image)
            }
            None => return Err(DecodingError::IoError(io::ErrorKind::UnexpectedEof.into())),
            Some((_, chunk)) => {
                return Err(DecodingError::ChunkHeaderInvalid(chunk.to_fourcc()).into())
            }
        };

        Ok(())
    }

    /// Returns true if the image as described by the bitstream is animated.
    pub fn has_animation(&self) -> Result<bool, DecodingError> {
        Ok(match &self.image {
            WebPImage::Lossy(_) => false,
            WebPImage::Lossless(_) => false,
            WebPImage::Extended(extended) => extended.has_animation(),
        })
    }

    /// Returns whether the image has an alpha channel.
    ///
    /// If so, the pixel format is Rgba8 and otherwise Rgb8.
    pub fn has_alpha(&self) -> bool {
        match &self.image {
            WebPImage::Lossy(_) => false,
            WebPImage::Lossless(_) => true,
            WebPImage::Extended(extended) => extended.has_alpha(),
        }
    }

    /// Sets the background color if the image is an extended and animated webp.
    pub fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodingError> {
        match &mut self.image {
            WebPImage::Extended(image) => image.set_background_color(color),
            _ => Err(DecodingError::InvalidParameter(
                "Background color can only be set on animated webp".to_owned(),
            )),
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        match &self.image {
            WebPImage::Lossy(vp8_frame) => {
                (u32::from(vp8_frame.width), u32::from(vp8_frame.height))
            }
            WebPImage::Lossless(lossless_frame) => (
                u32::from(lossless_frame.width),
                u32::from(lossless_frame.height),
            ),
            WebPImage::Extended(extended) => extended.dimensions(),
        }
    }

    pub fn read_image(self, buf: &mut [u8]) -> Result<(), DecodingError> {
        let (width, height) = self.dimensions();
        assert_eq!(
            u64::try_from(buf.len()),
            Ok(width as u64 * height as u64 * 4)
        );

        match &self.image {
            WebPImage::Lossy(vp8_frame) => {
                vp8_frame.fill_rgb(buf);
            }
            WebPImage::Lossless(lossless_frame) => {
                lossless_frame.fill_rgba(buf);
            }
            WebPImage::Extended(extended) => {
                extended.fill_buf(buf);
            }
        }
        Ok(())
    }

    pub fn icc_profile(&mut self) -> Result<Option<Vec<u8>>, DecodingError> {
        if let WebPImage::Extended(extended) = &self.image {
            Ok(extended.icc_profile())
        } else {
            Ok(None)
        }
    }

    // pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<u32, DecodingError> {
    //     assert!(self.has_animation());

    //     match self.image {
    //         WebPImage::Extended(extended_image) => extended_image.,
    //         _ => unreachable!()
    //     }
    // }

    pub fn into_frames(self) -> impl Iterator<Item = Result<Vec<u8>, DecodingError>> {
        match self.image {
            WebPImage::Lossy(_) | WebPImage::Lossless(_) => {
                Box::new(std::iter::empty()) as Box<dyn Iterator<Item = _>>
            }
            WebPImage::Extended(extended_image) => Box::new(extended_image.into_frames()),
        }
    }
}

pub(crate) fn read_len_cursor<R>(r: &mut R) -> Result<Cursor<Vec<u8>>, DecodingError>
where
    R: Read,
{
    let unpadded_len = u64::from(r.read_u32::<LittleEndian>()?);

    // RIFF chunks containing an uneven number of bytes append
    // an extra 0x00 at the end of the chunk
    //
    // The addition cannot overflow since we have a u64 that was created from a u32
    let len = unpadded_len + (unpadded_len % 2);

    let mut framedata = Vec::new();
    r.by_ref().take(len).read_to_end(&mut framedata)?;

    //remove padding byte
    if unpadded_len % 2 == 1 {
        framedata.pop();
    }

    Ok(io::Cursor::new(framedata))
}

/// Reads a chunk header FourCC
/// Returns None if and only if we hit end of file reading the four character code of the chunk
/// The inner error is `Err` if and only if the chunk header FourCC is present but unknown
pub(crate) fn read_fourcc<R: Read>(
    r: &mut R,
) -> Result<Option<Result<WebPRiffChunk, DecodingError>>, DecodingError> {
    let mut chunk_fourcc = [0; 4];
    let result = r.read_exact(&mut chunk_fourcc);

    match result {
        Ok(()) => {}
        Err(err) => {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                return Ok(None);
            } else {
                return Err(err.into());
            }
        }
    }

    let chunk = WebPRiffChunk::from_fourcc(chunk_fourcc);
    Ok(Some(chunk))
}

/// Reads a chunk
/// Returns an error if the chunk header is not a valid webp header or some other reading error
/// Returns None if and only if we hit end of file reading the four character code of the chunk
pub(crate) fn read_chunk<R>(
    r: &mut R,
) -> Result<Option<(Cursor<Vec<u8>>, WebPRiffChunk)>, DecodingError>
where
    R: Read,
{
    if let Some(chunk) = read_fourcc(r)? {
        let chunk = chunk?;
        let cursor = read_len_cursor(r)?;
        Ok(Some((cursor, chunk)))
    } else {
        Ok(None)
    }
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
