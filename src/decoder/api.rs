use alloc::string::String;
use thiserror::Error;

/// Errors that can occur when attempting to decode a WebP image
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecodeError {
    /// An IO error occurred while reading the file
    #[cfg(feature = "std")]
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

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
    #[error("Invalid Chunk header: {0:x?}")]
    ChunkHeaderInvalid([u8; 4]),

    /// The ALPH chunk preprocessing info flag was invalid
    #[error("Alpha chunk preprocessing flag invalid")]
    InvalidAlphaPreprocessing,

    /// Invalid compression method
    #[error("Invalid compression method")]
    InvalidCompressionMethod,

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

    /// Invalid color cache bits
    #[error("Invalid color cache bits: {0}")]
    InvalidColorCacheBits(u8),

    /// An invalid Huffman code was encountered
    #[error("Invalid Huffman code")]
    HuffmanError,

    /// The bitstream was somehow corrupt
    #[error("Corrupt bitstream")]
    BitStreamError,

    /// The transforms specified were invalid
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

    /// Memory limit exceeded
    #[error("Memory limit exceeded")]
    MemoryLimitExceeded,

    /// Invalid chunk size
    #[error("Invalid chunk size")]
    InvalidChunkSize,

    /// No more frames in image
    #[error("No more frames")]
    NoMoreFrames,

    /// Decoding was cancelled via a [`enough::Stop`] token.
    #[error("Decoding cancelled: {0}")]
    Cancelled(enough::StopReason),

    /// Unsupported codec operation.
    #[error(transparent)]
    UnsupportedOperation(#[from] zc::UnsupportedOperation),
}

/// Result type alias using `At<DecodeError>` for automatic location tracking.
///
/// Errors wrapped in `At<>` automatically capture file and line information,
/// making debugging easier in production environments.
pub type DecodeResult<T> = core::result::Result<T, whereat::At<DecodeError>>;

impl From<enough::StopReason> for DecodeError {
    fn from(reason: enough::StopReason) -> Self {
        Self::Cancelled(reason)
    }
}

// Core decoder implementation using SliceReader for no_std compatibility
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::ops::Range;

use hashbrown::HashMap;

use super::extended::{self, WebPExtendedInfo, get_alpha_predictor, read_alpha_chunk};
use super::lossless::LosslessDecoder;
use super::vp8::Vp8Decoder;
use crate::slice_reader::SliceReader;

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
    pub(crate) const fn from_fourcc(chunk_fourcc: [u8; 4]) -> Self {
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

    pub(crate) const fn to_fourcc(self) -> [u8; 4] {
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
            Self::Unknown(fourcc) => fourcc,
        }
    }

    pub(crate) const fn is_unknown(self) -> bool {
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
    next_frame: u32,
    next_frame_start: u64,
    dispose_next_frame: bool,
    previous_frame_width: u32,
    previous_frame_height: u32,
    previous_frame_x_offset: u32,
    previous_frame_y_offset: u32,
    canvas: Option<Vec<u8>>,
    /// Reusable scratch buffer for per-frame decode data.
    /// Avoids allocating a fresh Vec<u8> for every animation frame.
    frame_scratch: Vec<u8>,
}
impl Default for AnimationState {
    fn default() -> Self {
        Self {
            next_frame: 0,
            next_frame_start: 0,
            dispose_next_frame: true,
            previous_frame_width: 0,
            previous_frame_height: 0,
            previous_frame_x_offset: 0,
            previous_frame_y_offset: 0,
            canvas: None,
            frame_scratch: Vec::new(),
        }
    }
}

/// Number of times that an animation loops.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LoopCount {
    /// The animation loops forever.
    Forever,
    /// Each frame of the animation is displayed the specified number of times.
    Times(NonZeroU16),
}

impl core::fmt::Display for LoopCount {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LoopCount::Forever => f.write_str("infinite"),
            LoopCount::Times(n) => write!(f, "{} time{}", n, if n.get() == 1 { "" } else { "s" }),
        }
    }
}

impl From<u16> for LoopCount {
    fn from(n: u16) -> Self {
        match NonZeroU16::new(n) {
            None => LoopCount::Forever,
            Some(n) => LoopCount::Times(n),
        }
    }
}

/// WebP decoder configuration. Reusable across requests.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeConfig {
    /// Upsampling method for lossy chroma reconstruction. Default: `Bilinear`.
    pub upsampling: UpsamplingMethod,

    /// Decode limits for dimensions, memory, frame count, etc.
    pub limits: super::limits::Limits,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            upsampling: UpsamplingMethod::Bilinear,
            limits: super::limits::Limits::default(),
        }
    }
}

impl DecodeConfig {
    /// Set the upsampling method.
    #[must_use]
    pub fn upsampling(mut self, method: UpsamplingMethod) -> Self {
        self.upsampling = method;
        self
    }

    /// Set decode limits.
    #[must_use]
    pub fn limits(mut self, limits: super::limits::Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set maximum dimensions.
    #[must_use]
    pub fn max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.limits = self.limits.max_dimensions(width, height);
        self
    }

    /// Set maximum memory usage.
    #[must_use]
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.limits = self.limits.max_memory(bytes);
        self
    }

    /// Disable fancy upsampling.
    #[must_use]
    pub fn no_fancy_upsampling(mut self) -> Self {
        self.upsampling = UpsamplingMethod::Simple;
        self
    }

    pub(crate) fn to_options(&self) -> WebPDecodeOptions {
        WebPDecodeOptions {
            lossy_upsampling: self.upsampling,
        }
    }
}

/// Decoding request that borrows configuration and input data.
///
/// # Example
///
/// ```rust,no_run
/// use zenwebp::{DecodeConfig, DecodeRequest};
///
/// let config = DecodeConfig::default();
/// let webp_data: &[u8] = &[]; // your WebP data
/// let (pixels, w, h) = DecodeRequest::new(&config, webp_data).decode_rgba()?;
/// # Ok::<(), zenwebp::DecodeError>(())
/// ```
pub struct DecodeRequest<'a> {
    config: &'a DecodeConfig,
    data: &'a [u8],
    stop: Option<&'a dyn enough::Stop>,
    stride_pixels: Option<u32>,
}

impl<'a> DecodeRequest<'a> {
    /// Create a new decoding request.
    #[must_use]
    pub fn new(config: &'a DecodeConfig, data: &'a [u8]) -> Self {
        Self {
            config,
            data,
            stop: None,
            stride_pixels: None,
        }
    }

    /// Set a cooperative cancellation token.
    #[must_use]
    pub fn stop(mut self, stop: &'a dyn enough::Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set row stride in pixels for `_into` methods. Must be >= image width.
    #[must_use]
    pub fn stride(mut self, stride_pixels: u32) -> Self {
        self.stride_pixels = Some(stride_pixels);
        self
    }

    /// Decode to the image's native pixel format (RGB or RGBA).
    ///
    /// Returns RGBA if the image has alpha, RGB otherwise. This avoids
    /// both unnecessary alpha expansion and alpha stripping.
    ///
    /// The returned [`PixelLayout`](crate::PixelLayout) indicates the format.
    pub fn decode(self) -> Result<(Vec<u8>, u32, u32, crate::PixelLayout), DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(self.data, self.config.to_options())?;
        decoder.set_limits(self.config.limits.clone());
        decoder.set_stop(self.stop);
        let (w, h) = decoder.dimensions();
        let output_size = decoder
            .output_buffer_size()
            .ok_or(DecodeError::ImageTooLarge)?;
        let mut pixels = alloc::vec![0u8; output_size];
        decoder.read_image(&mut pixels)?;
        let layout = if decoder.has_alpha() {
            crate::PixelLayout::Rgba8
        } else {
            crate::PixelLayout::Rgb8
        };
        Ok((pixels, w, h, layout))
    }

    /// Decode to RGBA pixels. If the image has no alpha channel, alpha is set to 255.
    pub fn decode_rgba(self) -> Result<(Vec<u8>, u32, u32), DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(self.data, self.config.to_options())?;
        decoder.set_limits(self.config.limits.clone());
        decoder.set_stop(self.stop);
        let (w, h) = decoder.dimensions();
        let output_size = decoder
            .output_buffer_size()
            .ok_or(DecodeError::ImageTooLarge)?;
        let mut native = alloc::vec![0u8; output_size];
        decoder.read_image(&mut native)?;

        if decoder.has_alpha() {
            Ok((native, w, h))
        } else {
            // Expand RGB to RGBA
            let pixel_count = (w as usize) * (h as usize);
            let mut rgba = alloc::vec![0u8; pixel_count * 4];
            garb::bytes::rgb_to_rgba(&native, &mut rgba).map_err(garb_err)?;
            Ok((rgba, w, h))
        }
    }

    /// Decode to RGB pixels (no alpha). If the image has alpha, it is discarded.
    pub fn decode_rgb(self) -> Result<(Vec<u8>, u32, u32), DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(self.data, self.config.to_options())?;
        decoder.set_limits(self.config.limits.clone());
        decoder.set_stop(self.stop);
        let (w, h) = decoder.dimensions();
        let output_size = decoder
            .output_buffer_size()
            .ok_or(DecodeError::ImageTooLarge)?;
        let mut native = alloc::vec![0u8; output_size];
        decoder.read_image(&mut native)?;

        if !decoder.has_alpha() {
            Ok((native, w, h))
        } else {
            // Strip alpha from RGBA
            let pixel_count = (w as usize) * (h as usize);
            let mut rgb = alloc::vec![0u8; pixel_count * 3];
            garb::bytes::rgba_to_rgb(&native, &mut rgb).map_err(garb_err)?;
            Ok((rgb, w, h))
        }
    }

    /// Decode to RGBA, writing into a pre-allocated buffer.
    ///
    /// If [`stride`](Self::stride) is set, rows are written with that pixel stride.
    /// Otherwise rows are packed (stride == width).
    pub fn decode_rgba_into(self, output: &mut [u8]) -> Result<(u32, u32), DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(self.data, self.config.to_options())?;
        decoder.set_limits(self.config.limits.clone());
        decoder.set_stop(self.stop);
        let (w, h) = decoder.dimensions();

        if let Some(stride_px) = self.stride_pixels {
            if stride_px < w {
                return Err(DecodeError::InvalidParameter(format!(
                    "stride_pixels {} < width {}",
                    stride_px, w
                )));
            }
            let dst_stride = stride_px as usize * 4;
            let required = dst_stride * h as usize;
            if output.len() < required {
                return Err(DecodeError::InvalidParameter(format!(
                    "output buffer too small: got {}, need {}",
                    output.len(),
                    required
                )));
            }
            let buf_size = decoder
                .output_buffer_size()
                .ok_or(DecodeError::ImageTooLarge)?;
            let mut temp = alloc::vec![0u8; buf_size];
            decoder.read_image(&mut temp)?;

            if decoder.has_alpha() {
                let src_stride = w as usize * 4;
                for y in 0..(h as usize) {
                    output[y * dst_stride..][..w as usize * 4]
                        .copy_from_slice(&temp[y * src_stride..][..w as usize * 4]);
                }
            } else {
                garb::bytes::rgb_to_rgba_strided(
                    &temp,
                    output,
                    w as usize,
                    h as usize,
                    w as usize * 3,
                    dst_stride,
                )
                .map_err(garb_err)?;
            }
        } else {
            decoder.read_image(output)?;
        }
        Ok((w, h))
    }

    /// Decode to RGB, writing into a pre-allocated buffer.
    ///
    /// If [`stride`](Self::stride) is set, rows are written with that pixel stride.
    /// Otherwise rows are packed (stride == width).
    pub fn decode_rgb_into(self, output: &mut [u8]) -> Result<(u32, u32), DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(self.data, self.config.to_options())?;
        decoder.set_limits(self.config.limits.clone());
        decoder.set_stop(self.stop);
        let (w, h) = decoder.dimensions();
        let wu = w as usize;
        let hu = h as usize;

        let stride_px = self.stride_pixels.unwrap_or(w) as usize;
        if stride_px < wu {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "stride_pixels {} < width {}",
                stride_px,
                w
            )));
        }
        let dst_stride = stride_px * 3;
        let required = dst_stride * hu;
        if output.len() < required {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "output buffer too small: got {}, need {}",
                output.len(),
                required
            )));
        }

        let no_stride = self.stride_pixels.is_none() || stride_px == wu;
        if no_stride && !decoder.has_alpha() {
            // No stride padding and native format is RGB — decode directly into output.
            let buf_size = decoder
                .output_buffer_size()
                .ok_or(DecodeError::ImageTooLarge)?;
            decoder.read_image(&mut output[..buf_size])?;
        } else {
            // Need temp buffer — either for RGBA→RGB conversion or stride scatter
            let buf_size = decoder
                .output_buffer_size()
                .ok_or(DecodeError::ImageTooLarge)?;
            let mut temp = alloc::vec![0u8; buf_size];
            decoder.read_image(&mut temp)?;

            if decoder.has_alpha() {
                garb::bytes::rgba_to_rgb_strided(&temp, output, wu, hu, wu * 4, dst_stride)
                    .map_err(garb_err)?;
            } else {
                // RGB with stride padding — row scatter
                let src_stride = wu * 3;
                for y in 0..hu {
                    output[y * dst_stride..][..wu * 3]
                        .copy_from_slice(&temp[y * src_stride..][..wu * 3]);
                }
            }
        }
        Ok((w, h))
    }

    /// Read image info without decoding pixel data.
    pub fn info(self) -> Result<ImageInfo, DecodeError> {
        ImageInfo::from_webp(self.data)
    }

    /// Decode to YUV 4:2:0 planes (lossy only).
    pub fn decode_yuv420(self) -> Result<YuvPlanes, DecodeError> {
        decode_yuv420(self.data)
    }
}

/// WebP decoder configuration options (internal, used by AnimationDecoder)
#[derive(Clone)]
#[non_exhaustive]
pub(crate) struct WebPDecodeOptions {
    /// The upsampling method used in conversion from lossy yuv to rgb
    pub lossy_upsampling: UpsamplingMethod,
}

impl Default for WebPDecodeOptions {
    fn default() -> Self {
        Self {
            lossy_upsampling: UpsamplingMethod::Bilinear,
        }
    }
}

/// Methods for upsampling the chroma values in lossy decoding
///
/// The chroma red and blue planes are encoded in VP8 as half the size of the luma plane
/// Therefore we need to upsample these values up to fit each pixel in the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpsamplingMethod {
    /// Fancy upsampling
    ///
    /// Does bilinear interpolation using the 4 values nearest to the pixel, weighting based on the distance
    /// from the pixel.
    #[default]
    Bilinear,
    /// Simple upsampling, just uses the closest u/v value to the pixel when upsampling
    ///
    /// Matches the -nofancy option in dwebp.
    /// Should be faster but may lead to slightly jagged edges.
    Simple,
}

/// WebP image format decoder.
pub struct WebPDecoder<'a> {
    r: SliceReader<'a>,
    memory_limit: usize,
    limits: super::limits::Limits,

    width: u32,
    height: u32,

    kind: ImageKind,
    animation: AnimationState,

    is_lossy: bool,
    has_alpha: bool,
    num_frames: u32,
    loop_count: LoopCount,
    loop_duration: u64,

    chunks: HashMap<WebPRiffChunk, Range<u64>>,

    webp_decode_options: WebPDecodeOptions,

    stop: Option<&'a dyn enough::Stop>,
}

impl<'a> WebPDecoder<'a> {
    /// Create a new `WebPDecoder` from the data slice (alias for [`new`](Self::new)).
    ///
    /// This method parses the WebP headers and prepares for decoding. Use [`info()`](Self::info)
    /// to inspect metadata before calling decode methods.
    ///
    /// # Example - Two-phase decoding
    ///
    /// ```rust,no_run
    /// use zenwebp::WebPDecoder;
    ///
    /// # let webp_data: &[u8] = &[]; // your WebP data
    /// // Phase 1: Parse headers
    /// let mut decoder = WebPDecoder::build(webp_data)?;
    ///
    /// // Phase 2: Inspect metadata
    /// let info = decoder.info();
    /// println!("{}x{}, alpha={}", info.width, info.height, info.has_alpha);
    ///
    /// // Phase 3: Decode (no re-parsing)
    /// let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
    /// decoder.read_image(&mut output)?;
    /// # Ok::<(), zenwebp::DecodeError>(())
    /// ```
    pub fn build(data: &'a [u8]) -> Result<Self, DecodeError> {
        Self::new(data)
    }

    /// Create a new `WebPDecoder` from the data slice.
    pub fn new(data: &'a [u8]) -> Result<Self, DecodeError> {
        Self::new_with_options(data, WebPDecodeOptions::default())
    }

    /// Get image information without decoding the full image.
    ///
    /// Returns metadata that was parsed during construction. This is a zero-cost
    /// operation that doesn't re-parse or decode any data.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use zenwebp::WebPDecoder;
    ///
    /// let webp_data: &[u8] = &[]; // your WebP data
    /// let decoder = WebPDecoder::new(webp_data)?;
    /// let info = decoder.info();
    /// println!("Format: {:?}, {}x{}", info.format, info.width, info.height);
    /// # Ok::<(), zenwebp::DecodeError>(())
    /// ```
    pub fn info(&self) -> ImageInfo {
        let icc_profile = self
            .read_chunk_direct(WebPRiffChunk::ICCP, self.memory_limit)
            .unwrap_or(None);
        let exif = self
            .read_chunk_direct(WebPRiffChunk::EXIF, self.memory_limit)
            .unwrap_or(None);
        let xmp = self
            .read_chunk_direct(WebPRiffChunk::XMP, self.memory_limit)
            .unwrap_or(None);
        ImageInfo {
            width: self.width,
            height: self.height,
            has_alpha: self.has_alpha,
            is_lossy: self.is_lossy,
            has_animation: self.is_animated(),
            frame_count: self.num_frames,
            format: if self.is_lossy {
                BitstreamFormat::Lossy
            } else {
                BitstreamFormat::Lossless
            },
            icc_profile,
            exif,
            xmp,
        }
    }

    /// Create a new `WebPDecoder` from the data slice with the given options.
    pub(crate) fn new_with_options(
        data: &'a [u8],
        webp_decode_options: WebPDecodeOptions,
    ) -> Result<Self, DecodeError> {
        let mut decoder = Self {
            r: SliceReader::new(data),
            width: 0,
            height: 0,
            num_frames: 0,
            kind: ImageKind::Lossy,
            chunks: HashMap::new(),
            animation: Default::default(),
            memory_limit: usize::MAX,
            limits: super::limits::Limits::none(), // No limits by default
            is_lossy: false,
            has_alpha: false,
            loop_count: LoopCount::Times(NonZeroU16::new(1).unwrap()),
            loop_duration: 0,
            webp_decode_options,
            stop: None,
        };
        decoder.read_data()?;
        Ok(decoder)
    }

    fn read_data(&mut self) -> Result<(), DecodeError> {
        let (WebPRiffChunk::RIFF, riff_size, _) = read_chunk_header(&mut self.r)? else {
            return Err(DecodeError::ChunkHeaderInvalid(*b"RIFF"));
        };

        match &read_fourcc(&mut self.r)? {
            WebPRiffChunk::WEBP => {}
            fourcc => return Err(DecodeError::WebpSignatureInvalid(fourcc.to_fourcc())),
        }

        let (chunk, chunk_size, chunk_size_rounded) = read_chunk_header(&mut self.r)?;
        let start = self.r.stream_position();

        match chunk {
            WebPRiffChunk::VP8 => {
                let tag = self.r.read_u24_le()?;

                let keyframe = tag & 1 == 0;
                if !keyframe {
                    return Err(DecodeError::UnsupportedFeature(
                        "Non-keyframe frames".into(),
                    ));
                }

                let mut tag = [0u8; 3];
                self.r.read_exact(&mut tag)?;
                if tag != [0x9d, 0x01, 0x2a] {
                    return Err(DecodeError::Vp8MagicInvalid(tag));
                }

                let w = self.r.read_u16_le()?;
                let h = self.r.read_u16_le()?;

                self.width = u32::from(w & 0x3FFF);
                self.height = u32::from(h & 0x3FFF);
                if self.width == 0 || self.height == 0 {
                    return Err(DecodeError::InconsistentImageSizes);
                }

                self.limits.check_dimensions(self.width, self.height)?;

                self.chunks
                    .insert(WebPRiffChunk::VP8, start..start + chunk_size);
                self.kind = ImageKind::Lossy;
                self.is_lossy = true;
            }
            WebPRiffChunk::VP8L => {
                let signature = self.r.read_u8()?;
                if signature != 0x2f {
                    return Err(DecodeError::LosslessSignatureInvalid(signature));
                }

                let header = self.r.read_u32_le()?;
                let version = header >> 29;
                if version != 0 {
                    return Err(DecodeError::VersionNumberInvalid(version as u8));
                }

                self.width = (1 + header) & 0x3FFF;
                self.height = (1 + (header >> 14)) & 0x3FFF;
                self.limits.check_dimensions(self.width, self.height)?;
                self.chunks
                    .insert(WebPRiffChunk::VP8L, start..start + chunk_size);
                self.kind = ImageKind::Lossless;
                self.has_alpha = (header >> 28) & 1 != 0;
            }
            WebPRiffChunk::VP8X => {
                let mut info = extended::read_extended_header(&mut self.r)?;
                self.width = info.canvas_width;
                self.height = info.canvas_height;
                self.limits.check_dimensions(self.width, self.height)?;

                let mut position = start + chunk_size_rounded;
                let max_position = position + riff_size.saturating_sub(12);
                self.r.seek_from_start(position)?;

                while position < max_position {
                    match read_chunk_header(&mut self.r) {
                        Ok((chunk, chunk_size, chunk_size_rounded)) => {
                            let range = position + 8..position + 8 + chunk_size;
                            position += 8 + chunk_size_rounded;

                            if !chunk.is_unknown() {
                                self.chunks.entry(chunk).or_insert(range);
                            }

                            if chunk == WebPRiffChunk::ANMF {
                                self.num_frames += 1;
                                if chunk_size < 24 {
                                    return Err(DecodeError::InvalidChunkSize);
                                }

                                self.r.seek_relative(12)?;
                                let duration = self.r.read_u32_le()? & 0xffffff;
                                self.loop_duration =
                                    self.loop_duration.wrapping_add(u64::from(duration));

                                // If the image is animated, the image data chunk will be inside the
                                // ANMF chunks, so we must inspect them to determine whether the
                                // image contains any lossy image data. VP8 chunks store lossy data
                                // and the spec says that lossless images SHOULD NOT contain ALPH
                                // chunks, so we treat both as indicators of lossy images.
                                if !self.is_lossy {
                                    let (subchunk, ..) = read_chunk_header(&mut self.r)?;
                                    if let WebPRiffChunk::VP8 | WebPRiffChunk::ALPH = subchunk {
                                        self.is_lossy = true;
                                    }
                                    self.r.seek_relative(chunk_size_rounded as i64 - 24)?;
                                } else {
                                    self.r.seek_relative(chunk_size_rounded as i64 - 16)?;
                                }

                                continue;
                            }

                            self.r.seek_relative(chunk_size_rounded as i64)?;
                        }
                        Err(DecodeError::BitStreamError) => {
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }
                self.is_lossy = self.is_lossy || self.chunks.contains_key(&WebPRiffChunk::VP8);

                // NOTE: We allow malformed images that have `info.icc_profile` set without a ICCP chunk,
                // because this is relatively common.
                if info.animation
                    && (!self.chunks.contains_key(&WebPRiffChunk::ANIM)
                        || !self.chunks.contains_key(&WebPRiffChunk::ANMF))
                    || info.exif_metadata && !self.chunks.contains_key(&WebPRiffChunk::EXIF)
                    || info.xmp_metadata && !self.chunks.contains_key(&WebPRiffChunk::XMP)
                    || !info.animation
                        && self.chunks.contains_key(&WebPRiffChunk::VP8)
                            == self.chunks.contains_key(&WebPRiffChunk::VP8L)
                {
                    return Err(DecodeError::ChunkMissing);
                }

                // Decode ANIM chunk.
                if info.animation {
                    match self.read_chunk(WebPRiffChunk::ANIM, 6) {
                        Ok(Some(chunk)) => {
                            let mut cursor = SliceReader::new(&chunk);
                            cursor.read_exact(&mut info.background_color_hint)?;
                            self.loop_count = match cursor.read_u16_le()? {
                                0 => LoopCount::Forever,
                                n => LoopCount::Times(NonZeroU16::new(n).unwrap()),
                            };
                            self.animation.next_frame_start =
                                self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start - 8;
                        }
                        Ok(None) => return Err(DecodeError::ChunkMissing),
                        Err(DecodeError::MemoryLimitExceeded) => {
                            return Err(DecodeError::InvalidChunkSize);
                        }
                        Err(e) => return Err(e),
                    }
                }

                // If the image is animated, the image data chunk will be inside the ANMF chunks. We
                // store the ALPH, VP8, and VP8L chunks (as applicable) of the first frame in the
                // hashmap so that we can read them later.
                if let Some(range) = self.chunks.get(&WebPRiffChunk::ANMF).cloned() {
                    let mut position = range.start + 16;
                    self.r.seek_from_start(position)?;
                    for _ in 0..2 {
                        let (subchunk, subchunk_size, subchunk_size_rounded) =
                            read_chunk_header(&mut self.r)?;
                        let subrange = position + 8..position + 8 + subchunk_size;
                        self.chunks.entry(subchunk).or_insert(subrange.clone());

                        position += 8 + subchunk_size_rounded;
                        if position + 8 > range.end {
                            break;
                        }
                    }
                }

                self.has_alpha = info.alpha;
                self.kind = ImageKind::Extended(info);
            }
            _ => return Err(DecodeError::ChunkHeaderInvalid(chunk.to_fourcc())),
        };

        Ok(())
    }

    /// Sets the maximum amount of memory that the decoder is allowed to allocate at once.
    ///
    /// TODO: Some allocations currently ignore this limit.
    /// Set a cooperative cancellation token for decoding.
    pub fn set_stop(&mut self, stop: Option<&'a dyn enough::Stop>) {
        self.stop = stop;
    }

    /// Sets the memory limit in bytes for decoded image buffers.
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.memory_limit = limit;
    }

    /// Set decode limits for validation.
    pub fn set_limits(&mut self, limits: super::limits::Limits) {
        self.limits = limits;
    }

    /// Get the background color specified in the image file if the image is extended and animated webp.
    pub fn background_color_hint(&self) -> Option<[u8; 4]> {
        if let ImageKind::Extended(info) = &self.kind {
            Some(info.background_color_hint)
        } else {
            None
        }
    }

    /// Sets the background color if the image is an extended and animated webp.
    pub fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodeError> {
        if let ImageKind::Extended(info) = &mut self.kind {
            info.background_color = Some(color);
            Ok(())
        } else {
            Err(DecodeError::InvalidParameter(
                "Background color can only be set on animated webp".into(),
            ))
        }
    }

    /// Returns the (width, height) of the image in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns whether the image has an alpha channel. If so, the pixel format is Rgba8 and
    /// otherwise Rgb8.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Returns true if the image is animated.
    pub fn is_animated(&self) -> bool {
        match &self.kind {
            ImageKind::Lossy | ImageKind::Lossless => false,
            ImageKind::Extended(extended) => extended.animation,
        }
    }

    /// Returns whether the image is lossy. For animated images, this is true if any frame is lossy.
    pub fn is_lossy(&self) -> bool {
        self.is_lossy
    }

    /// Returns the number of frames of a single loop of the animation, or zero if the image is not
    /// animated.
    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    /// Returns the number of times the animation should loop.
    pub fn loop_count(&self) -> LoopCount {
        self.loop_count
    }

    /// Returns the total duration of one loop through the animation in milliseconds, or zero if the
    /// image is not animated.
    ///
    /// This is the sum of the durations of all individual frames of the image.
    pub fn loop_duration(&self) -> u64 {
        self.loop_duration
    }

    fn read_chunk(
        &mut self,
        chunk: WebPRiffChunk,
        max_size: usize,
    ) -> Result<Option<Vec<u8>>, DecodeError> {
        self.read_chunk_direct(chunk, max_size)
    }

    fn read_chunk_direct(
        &self,
        chunk: WebPRiffChunk,
        max_size: usize,
    ) -> Result<Option<Vec<u8>>, DecodeError> {
        match self.chunks.get(&chunk) {
            Some(range) => {
                let len = (range.end - range.start) as usize;
                if len > max_size {
                    return Err(DecodeError::MemoryLimitExceeded);
                }
                let start = range.start as usize;
                let end = range.end as usize;
                Ok(Some(self.r.get_ref()[start..end].to_vec()))
            }
            None => Ok(None),
        }
    }

    /// Returns the raw bytes of the ICC profile, or None if there is no ICC profile.
    pub fn icc_profile(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.read_chunk(WebPRiffChunk::ICCP, self.memory_limit)
    }

    /// Returns the raw bytes of the EXIF metadata, or None if there is no EXIF metadata.
    pub fn exif_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.read_chunk(WebPRiffChunk::EXIF, self.memory_limit)
    }

    /// Returns the raw bytes of the XMP metadata, or None if there is no XMP metadata.
    pub fn xmp_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.read_chunk(WebPRiffChunk::XMP, self.memory_limit)
    }

    /// Returns the number of bytes required to store the image or a single frame, or None if that
    /// would take more than `usize::MAX` bytes.
    pub fn output_buffer_size(&self) -> Option<usize> {
        let bytes_per_pixel = if self.has_alpha() { 4 } else { 3 };
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(bytes_per_pixel)
    }

    /// Returns the raw bytes of the image. For animated images, this is the first frame.
    ///
    /// Fails with `ImageTooLarge` if `buf` has length different than `output_buffer_size()`
    pub fn read_image(&mut self, buf: &mut [u8]) -> Result<(), DecodeError> {
        if Some(buf.len()) != self.output_buffer_size() {
            return Err(DecodeError::ImageTooLarge);
        }

        if self.is_animated() {
            let saved = core::mem::take(&mut self.animation);
            self.animation.next_frame_start =
                self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start - 8;
            let result = self.read_frame(buf);
            self.animation = saved;
            result?;
        } else if let Some(range) = self.chunks.get(&WebPRiffChunk::VP8L) {
            let data_slice = &self.r.get_ref()[range.start as usize..range.end as usize];
            let mut decoder = LosslessDecoder::new(data_slice);
            decoder.set_stop(self.stop);

            if self.has_alpha {
                decoder.decode_frame(self.width, self.height, false, buf)?;
            } else {
                let alloc_size = self.width as usize * self.height as usize * 4;
                self.limits.check_memory(alloc_size)?;
                let mut data = vec![0; alloc_size];
                decoder.decode_frame(self.width, self.height, false, &mut data)?;
                garb::bytes::rgba_to_rgb(&data, buf).map_err(garb_err)?;
            }
        } else {
            let range = self
                .chunks
                .get(&WebPRiffChunk::VP8)
                .ok_or(DecodeError::ChunkMissing)?;
            let data_slice = &self.r.get_ref()[range.start as usize..range.end as usize];
            let frame = Vp8Decoder::decode_frame_with_stop(data_slice, self.stop)?;
            if u32::from(frame.width) != self.width || u32::from(frame.height) != self.height {
                return Err(DecodeError::InconsistentImageSizes);
            }

            if self.has_alpha() {
                frame.fill_rgba(buf, self.webp_decode_options.lossy_upsampling);

                let range = self
                    .chunks
                    .get(&WebPRiffChunk::ALPH)
                    .ok_or(DecodeError::ChunkMissing)?
                    .clone();
                let alpha_slice = &self.r.get_ref()[range.start as usize..range.end as usize];
                let alpha_chunk =
                    read_alpha_chunk(alpha_slice, self.width as u16, self.height as u16)?;

                for y in 0..frame.height {
                    for x in 0..frame.width {
                        let predictor: u8 = get_alpha_predictor(
                            x.into(),
                            y.into(),
                            frame.width.into(),
                            alpha_chunk.filtering_method,
                            buf,
                        );

                        let alpha_index =
                            usize::from(y) * usize::from(frame.width) + usize::from(x);
                        let buffer_index = alpha_index * 4 + 3;

                        buf[buffer_index] = predictor.wrapping_add(alpha_chunk.data[alpha_index]);
                    }
                }
            } else {
                frame.fill_rgb(buf, self.webp_decode_options.lossy_upsampling);
            }
        }

        Ok(())
    }

    /// Reads the next frame of the animation.
    ///
    /// The frame contents are written into `buf` and the method returns the duration of the frame
    /// in milliseconds. If there are no more frames, the method returns
    /// `DecodeError::NoMoreFrames` and `buf` is left unchanged.
    ///
    pub fn read_frame(&mut self, buf: &mut [u8]) -> Result<u32, DecodeError> {
        if !self.is_animated() {
            return Err(DecodeError::InvalidParameter(String::from(
                "not an animated WebP",
            )));
        }
        if Some(buf.len()) != self.output_buffer_size() {
            return Err(DecodeError::ImageTooLarge);
        }

        if self.animation.next_frame == self.num_frames {
            return Err(DecodeError::NoMoreFrames);
        }

        let ImageKind::Extended(info) = &self.kind else {
            unreachable!()
        };

        self.r.seek_from_start(self.animation.next_frame_start)?;

        let anmf_size = match read_chunk_header(&mut self.r)? {
            (WebPRiffChunk::ANMF, size, _) if size >= 32 => size,
            _ => return Err(DecodeError::ChunkHeaderInvalid(*b"ANMF")),
        };

        // Read ANMF chunk
        let frame_x = extended::read_3_bytes(&mut self.r)? * 2;
        let frame_y = extended::read_3_bytes(&mut self.r)? * 2;
        let frame_width = extended::read_3_bytes(&mut self.r)? + 1;
        let frame_height = extended::read_3_bytes(&mut self.r)? + 1;
        if frame_width > 16384 || frame_height > 16384 {
            return Err(DecodeError::ImageTooLarge);
        }
        if frame_x + frame_width > self.width || frame_y + frame_height > self.height {
            return Err(DecodeError::FrameOutsideImage);
        }
        let duration = extended::read_3_bytes(&mut self.r)?;
        let frame_info = self.r.read_u8()?;
        let use_alpha_blending = frame_info & 0b00000010 == 0;
        let dispose = frame_info & 0b00000001 != 0;

        // Read normal bitstream now
        let (chunk, chunk_size, chunk_size_rounded) = read_chunk_header(&mut self.r)?;
        if chunk_size_rounded + 24 > anmf_size {
            return Err(DecodeError::ChunkHeaderInvalid(chunk.to_fourcc()));
        }

        let frame_has_alpha: bool = match chunk {
            WebPRiffChunk::VP8 => {
                let data_slice = self.r.take_slice(chunk_size as usize)?;
                let raw_frame = Vp8Decoder::decode_frame_with_stop(data_slice, self.stop)?;
                if u32::from(raw_frame.width) != frame_width
                    || u32::from(raw_frame.height) != frame_height
                {
                    return Err(DecodeError::InconsistentImageSizes);
                }
                let frame_alloc = frame_width as usize * frame_height as usize * 3;
                self.limits.check_memory(frame_alloc)?;
                self.animation.frame_scratch.resize(frame_alloc, 0);
                raw_frame.fill_rgb(
                    &mut self.animation.frame_scratch,
                    self.webp_decode_options.lossy_upsampling,
                );
                false
            }
            WebPRiffChunk::VP8L => {
                let data_slice = self.r.take_slice(chunk_size as usize)?;
                let mut lossless_decoder = LosslessDecoder::new(data_slice);
                lossless_decoder.set_stop(self.stop);
                let frame_alloc = frame_width as usize * frame_height as usize * 4;
                self.limits.check_memory(frame_alloc)?;
                self.animation.frame_scratch.resize(frame_alloc, 0);
                lossless_decoder.decode_frame(
                    frame_width,
                    frame_height,
                    false,
                    &mut self.animation.frame_scratch,
                )?;
                true
            }
            WebPRiffChunk::ALPH => {
                if chunk_size_rounded + 32 > anmf_size {
                    return Err(DecodeError::ChunkHeaderInvalid(chunk.to_fourcc()));
                }

                // read alpha
                let alpha_slice = self.r.take_slice(chunk_size as usize)?;
                // Skip padding if chunk_size is odd
                if chunk_size_rounded > chunk_size {
                    self.r
                        .seek_relative((chunk_size_rounded - chunk_size) as i64)?;
                }
                let alpha_chunk =
                    read_alpha_chunk(alpha_slice, frame_width as u16, frame_height as u16)?;

                // read opaque
                let (next_chunk, next_chunk_size, _) = read_chunk_header(&mut self.r)?;
                if chunk_size + next_chunk_size + 32 > anmf_size {
                    return Err(DecodeError::ChunkHeaderInvalid(next_chunk.to_fourcc()));
                }

                let vp8_slice = self.r.take_slice(next_chunk_size as usize)?;
                let raw_frame = Vp8Decoder::decode_frame_with_stop(vp8_slice, self.stop)?;

                let frame_alloc = frame_width as usize * frame_height as usize * 4;
                self.limits.check_memory(frame_alloc)?;
                self.animation.frame_scratch.resize(frame_alloc, 0);
                raw_frame.fill_rgba(
                    &mut self.animation.frame_scratch,
                    self.webp_decode_options.lossy_upsampling,
                );

                for y in 0..raw_frame.height {
                    for x in 0..raw_frame.width {
                        let predictor: u8 = get_alpha_predictor(
                            x.into(),
                            y.into(),
                            raw_frame.width.into(),
                            alpha_chunk.filtering_method,
                            &self.animation.frame_scratch,
                        );

                        let alpha_index =
                            usize::from(y) * usize::from(raw_frame.width) + usize::from(x);
                        let buffer_index = alpha_index * 4 + 3;

                        self.animation.frame_scratch[buffer_index] =
                            predictor.wrapping_add(alpha_chunk.data[alpha_index]);
                    }
                }

                true
            }
            _ => return Err(DecodeError::ChunkHeaderInvalid(chunk.to_fourcc())),
        };

        let clear_color = if self.animation.dispose_next_frame {
            match (info.background_color, frame_has_alpha) {
                (color @ Some(_), _) => color,
                (_, true) => Some([0, 0, 0, 0]),
                _ => None,
            }
        } else {
            None
        };

        // fill starting canvas with clear color
        if self.animation.canvas.is_none() {
            self.animation.canvas = {
                let canvas_alloc = self.width as usize * self.height as usize * 4;
                self.limits.check_memory(canvas_alloc)?;
                let mut canvas = vec![0; canvas_alloc];
                if let Some(color) = info.background_color.as_ref() {
                    canvas
                        .chunks_exact_mut(4)
                        .for_each(|c| c.copy_from_slice(color))
                }
                Some(canvas)
            }
        }
        extended::composite_frame(
            self.animation.canvas.as_mut().unwrap(),
            self.width,
            self.height,
            clear_color,
            &self.animation.frame_scratch,
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            frame_has_alpha,
            use_alpha_blending,
            self.animation.previous_frame_width,
            self.animation.previous_frame_height,
            self.animation.previous_frame_x_offset,
            self.animation.previous_frame_y_offset,
        )?;

        self.animation.previous_frame_width = frame_width;
        self.animation.previous_frame_height = frame_height;
        self.animation.previous_frame_x_offset = frame_x;
        self.animation.previous_frame_y_offset = frame_y;

        self.animation.dispose_next_frame = dispose;
        self.animation.next_frame_start += anmf_size + 8;
        self.animation.next_frame += 1;

        if self.has_alpha() {
            buf.copy_from_slice(self.animation.canvas.as_ref().unwrap());
        } else {
            garb::bytes::rgba_to_rgb(self.animation.canvas.as_ref().unwrap(), buf)
                .map_err(garb_err)?;
        }

        Ok(duration)
    }

    /// Resets the animation to the first frame.
    ///
    pub fn reset_animation(&mut self) -> Result<(), DecodeError> {
        if !self.is_animated() {
            return Err(DecodeError::InvalidParameter(String::from(
                "not an animated WebP",
            )));
        }
        self.animation.next_frame = 0;
        self.animation.next_frame_start = self.chunks.get(&WebPRiffChunk::ANMF).unwrap().start - 8;
        self.animation.dispose_next_frame = true;
        Ok(())
    }

    /// Sets the upsampling method that is used in lossy decoding
    pub fn set_lossy_upsampling(&mut self, upsampling_method: UpsamplingMethod) {
        self.webp_decode_options.lossy_upsampling = upsampling_method;
    }
}

/// Convert a garb SizeError into a DecodeError.
fn garb_err(e: garb::SizeError) -> DecodeError {
    DecodeError::InvalidParameter(alloc::format!("pixel conversion: {e}"))
}

pub(crate) fn read_fourcc(r: &mut SliceReader) -> Result<WebPRiffChunk, DecodeError> {
    let mut chunk_fourcc = [0; 4];
    r.read_exact(&mut chunk_fourcc)?;
    Ok(WebPRiffChunk::from_fourcc(chunk_fourcc))
}

pub(crate) fn read_chunk_header(
    r: &mut SliceReader,
) -> Result<(WebPRiffChunk, u64, u64), DecodeError> {
    let chunk = read_fourcc(r)?;
    let chunk_size = r.read_u32_le()?;
    let chunk_size_rounded = chunk_size.saturating_add(chunk_size & 1);
    Ok((chunk, chunk_size.into(), chunk_size_rounded.into()))
}

// ============================================================================
// Convenience decode functions (webpx-compatible API)
// ============================================================================

/// Decode WebP data to RGBA pixels.
///
/// Returns the decoded pixels and dimensions.
///
/// # Example
///
/// ```rust,no_run
/// let webp_data: &[u8] = &[]; // your WebP data
/// let (pixels, width, height) = zenwebp::decode_rgba(webp_data)?;
/// # Ok::<(), zenwebp::DecodeError>(())
/// ```
pub fn decode_rgba(data: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;

    // Get output in native format (RGB or RGBA)
    let mut output = vec![0u8; output_size];
    decoder.read_image(&mut output)?;

    // If the decoder outputs RGB, convert to RGBA
    if !decoder.has_alpha() {
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        garb::bytes::rgb_to_rgba(&output, &mut rgba).map_err(garb_err)?;
        return Ok((rgba, width, height));
    }

    Ok((output, width, height))
}

/// Decode WebP data to RGB pixels (no alpha).
///
/// Returns the decoded pixels and dimensions.
///
/// # Example
///
/// ```rust,no_run
/// let webp_data: &[u8] = &[]; // your WebP data
/// let (pixels, width, height) = zenwebp::decode_rgb(webp_data)?;
/// # Ok::<(), zenwebp::DecodeError>(())
/// ```
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;

    let mut output = vec![0u8; output_size];
    decoder.read_image(&mut output)?;

    // If the decoder outputs RGBA, convert to RGB
    if decoder.has_alpha() {
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        garb::bytes::rgba_to_rgb(&output, &mut rgb).map_err(garb_err)?;
        return Ok((rgb, width, height));
    }

    Ok((output, width, height))
}

/// Decode WebP data directly into a pre-allocated RGBA buffer.
///
/// # Arguments
/// * `data` - WebP encoded data
/// * `output` - Pre-allocated output buffer (must be at least `stride_pixels * height * 4` bytes)
/// * `stride_pixels` - Row stride in pixels (must be >= width)
///
/// # Returns
/// Width and height of the decoded image.
pub fn decode_rgba_into(
    data: &[u8],
    output: &mut [u8],
    stride_pixels: u32,
) -> Result<(u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let w = width as usize;
    let h = height as usize;

    if stride_pixels < width {
        return Err(DecodeError::InvalidParameter(format!(
            "stride_pixels {} < width {}",
            stride_pixels, width
        )));
    }

    let dst_stride = stride_pixels as usize * 4;
    let required = dst_stride * h;
    if output.len() < required {
        return Err(DecodeError::InvalidParameter(format!(
            "output buffer too small: got {}, need {}",
            output.len(),
            required
        )));
    }

    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut temp = vec![0u8; output_size];
    decoder.read_image(&mut temp)?;

    if decoder.has_alpha() {
        let src_stride = w * 4;
        if src_stride == dst_stride {
            output[..w * h * 4].copy_from_slice(&temp);
        } else {
            for y in 0..h {
                output[y * dst_stride..][..w * 4].copy_from_slice(&temp[y * src_stride..][..w * 4]);
            }
        }
    } else {
        garb::bytes::rgb_to_rgba_strided(&temp, output, w, h, w * 3, dst_stride)
            .map_err(garb_err)?;
    }

    Ok((width, height))
}

/// Decode WebP data directly into a pre-allocated RGB buffer.
///
/// # Arguments
/// * `data` - WebP encoded data
/// * `output` - Pre-allocated output buffer (must be at least `stride_pixels * height * 3` bytes)
/// * `stride_pixels` - Row stride in pixels (must be >= width)
///
/// # Returns
/// Width and height of the decoded image.
pub fn decode_rgb_into(
    data: &[u8],
    output: &mut [u8],
    stride_pixels: u32,
) -> Result<(u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let w = width as usize;
    let h = height as usize;

    if stride_pixels < width {
        return Err(DecodeError::InvalidParameter(format!(
            "stride_pixels {} < width {}",
            stride_pixels, width
        )));
    }

    let dst_stride = stride_pixels as usize * 3;
    let required = dst_stride * h;
    if output.len() < required {
        return Err(DecodeError::InvalidParameter(format!(
            "output buffer too small: got {}, need {}",
            output.len(),
            required
        )));
    }

    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut temp = vec![0u8; output_size];
    decoder.read_image(&mut temp)?;

    if decoder.has_alpha() {
        garb::bytes::rgba_to_rgb_strided(&temp, output, w, h, w * 4, dst_stride)
            .map_err(garb_err)?;
    } else {
        let src_stride = w * 3;
        if src_stride == dst_stride {
            output[..w * h * 3].copy_from_slice(&temp);
        } else {
            for y in 0..h {
                output[y * dst_stride..][..w * 3].copy_from_slice(&temp[y * src_stride..][..w * 3]);
            }
        }
    }

    Ok((width, height))
}

/// Image information obtained from WebP data header.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// Whether the image uses lossy compression.
    pub is_lossy: bool,
    /// Whether the image is animated.
    pub has_animation: bool,
    /// Number of frames (1 for static images).
    pub frame_count: u32,
    /// Bitstream format (lossy or lossless).
    pub format: BitstreamFormat,
    /// ICC color profile, if present.
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF metadata, if present.
    pub exif: Option<Vec<u8>>,
    /// XMP metadata, if present.
    pub xmp: Option<Vec<u8>>,
}

impl ImageInfo {
    /// Minimum bytes needed to probe WebP metadata.
    ///
    /// This is a conservative estimate that covers the RIFF header, VP8/VP8L chunk
    /// header, and enough data to read basic image metadata. Actual images may be
    /// larger, but this is sufficient for probing.
    pub const PROBE_BYTES: usize = 64;

    /// Parse image information from WebP data (alias for [`from_webp`](Self::from_webp)).
    ///
    /// This is a fast probing operation that only parses headers without decoding
    /// the full image data.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use zenwebp::ImageInfo;
    ///
    /// let webp_data: &[u8] = &[]; // your WebP data
    /// let info = ImageInfo::from_bytes(webp_data)?;
    /// println!("{}x{}, alpha={}", info.width, info.height, info.has_alpha);
    /// # Ok::<(), zenwebp::DecodeError>(())
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, DecodeError> {
        Self::from_webp(data)
    }

    /// Parse image information from WebP data.
    ///
    /// Extracts dimensions, format info, and metadata (ICC, EXIF, XMP) in a single
    /// pass. This replaces the need to use both [`WebPDecoder`] and
    /// [`WebPDemuxer`](crate::WebPDemuxer) for probing.
    pub fn from_webp(data: &[u8]) -> Result<Self, DecodeError> {
        let mut decoder = WebPDecoder::new(data)?;
        let (width, height) = decoder.dimensions();
        let is_lossy = decoder.is_lossy();
        let is_animated = decoder.is_animated();
        let frame_count = if is_animated { decoder.num_frames() } else { 1 };
        let format = if is_lossy {
            BitstreamFormat::Lossy
        } else {
            BitstreamFormat::Lossless
        };
        let icc_profile = decoder.icc_profile().unwrap_or(None);
        let exif = decoder.exif_metadata().unwrap_or(None);
        let xmp = decoder.xmp_metadata().unwrap_or(None);
        Ok(Self {
            width,
            height,
            has_alpha: decoder.has_alpha(),
            is_lossy,
            has_animation: is_animated,
            frame_count,
            format,
            icc_profile,
            exif,
            xmp,
        })
    }

    /// Estimate resource consumption for decoding this image.
    ///
    /// Returns memory, time, and output size estimates. See
    /// [`heuristics::estimate_decode`](crate::heuristics::estimate_decode) for details.
    #[must_use]
    pub fn estimate_decode(&self, output_bpp: u8) -> crate::heuristics::DecodeEstimate {
        if self.has_animation {
            crate::heuristics::estimate_animation_decode(self.width, self.height, self.frame_count)
        } else {
            crate::heuristics::estimate_decode(self.width, self.height, output_bpp)
        }
    }
}

/// Bitstream compression format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BitstreamFormat {
    /// Lossy compression (VP8).
    #[default]
    Lossy,
    /// Lossless compression (VP8L).
    Lossless,
}

impl core::fmt::Display for BitstreamFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BitstreamFormat::Lossy => f.write_str("lossy"),
            BitstreamFormat::Lossless => f.write_str("lossless"),
        }
    }
}

/// Decoded YUV 4:2:0 planar image data.
///
/// Contains separate Y, U, and V planes at their native resolutions.
/// Y is full resolution, U and V are half resolution in each dimension.
#[derive(Debug, Clone)]
pub struct YuvPlanes {
    /// Luma plane (full resolution).
    pub y: Vec<u8>,
    /// Chroma blue plane (half resolution in each dimension).
    pub u: Vec<u8>,
    /// Chroma red plane (half resolution in each dimension).
    pub v: Vec<u8>,
    /// Width of the luma plane in pixels.
    pub y_width: u32,
    /// Height of the luma plane in pixels.
    pub y_height: u32,
    /// Width of each chroma plane in pixels.
    pub uv_width: u32,
    /// Height of each chroma plane in pixels.
    pub uv_height: u32,
}

/// Decode WebP data to BGRA pixels (blue, green, red, alpha order).
///
/// Returns the decoded pixels and dimensions.
pub fn decode_bgra(data: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut native = vec![0u8; output_size];
    decoder.read_image(&mut native)?;

    if decoder.has_alpha() {
        // RGBA → BGRA in-place
        garb::bytes::rgba_to_bgra_inplace(&mut native).map_err(garb_err)?;
        Ok((native, width, height))
    } else {
        // RGB → BGRA (adds alpha=255, swaps R↔B in one pass)
        let mut bgra = vec![0u8; (width * height * 4) as usize];
        garb::bytes::rgb_to_bgra(&native, &mut bgra).map_err(garb_err)?;
        Ok((bgra, width, height))
    }
}

/// Decode WebP data to BGR pixels (blue, green, red order, no alpha).
///
/// Returns the decoded pixels and dimensions.
pub fn decode_bgr(data: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut native = vec![0u8; output_size];
    decoder.read_image(&mut native)?;

    if decoder.has_alpha() {
        // RGBA → BGR (drop alpha + swap R↔B)
        let mut bgr = vec![0u8; (width * height * 3) as usize];
        garb::bytes::rgba_to_bgr(&native, &mut bgr).map_err(garb_err)?;
        Ok((bgr, width, height))
    } else {
        // RGB → BGR in-place
        garb::bytes::rgb_to_bgr_inplace(&mut native).map_err(garb_err)?;
        Ok((native, width, height))
    }
}

/// Decode WebP data directly into a pre-allocated BGRA buffer.
///
/// Also suitable for BGRX output — alpha bytes are set to 255 for opaque images.
///
/// # Arguments
/// * `data` - WebP encoded data
/// * `output` - Pre-allocated output buffer (must be at least `stride_pixels * height * 4` bytes)
/// * `stride_pixels` - Row stride in pixels (must be >= width)
///
/// # Returns
/// Width and height of the decoded image.
pub fn decode_bgra_into(
    data: &[u8],
    output: &mut [u8],
    stride_pixels: u32,
) -> Result<(u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let w = width as usize;
    let h = height as usize;

    if stride_pixels < width {
        return Err(DecodeError::InvalidParameter(format!(
            "stride_pixels {} < width {}",
            stride_pixels, width
        )));
    }
    let dst_stride = stride_pixels as usize * 4;
    let required = dst_stride * h;
    if output.len() < required {
        return Err(DecodeError::InvalidParameter(format!(
            "output buffer too small: got {}, need {}",
            output.len(),
            required
        )));
    }

    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut temp = vec![0u8; output_size];
    decoder.read_image(&mut temp)?;

    if decoder.has_alpha() {
        // RGBA → BGRA with stride (swap R↔B + scatter in one pass)
        garb::bytes::rgba_to_bgra_strided(&temp, output, w, h, w * 4, dst_stride)
            .map_err(garb_err)?;
    } else {
        // RGB → BGRA with stride (add alpha=255 + swap R↔B in one pass)
        garb::bytes::rgb_to_bgra_strided(&temp, output, w, h, w * 3, dst_stride)
            .map_err(garb_err)?;
    }

    Ok((width, height))
}

/// Decode WebP data directly into a pre-allocated BGR buffer.
///
/// # Arguments
/// * `data` - WebP encoded data
/// * `output` - Pre-allocated output buffer (must be at least `stride_pixels * height * 3` bytes)
/// * `stride_pixels` - Row stride in pixels (must be >= width)
///
/// # Returns
/// Width and height of the decoded image.
pub fn decode_bgr_into(
    data: &[u8],
    output: &mut [u8],
    stride_pixels: u32,
) -> Result<(u32, u32), DecodeError> {
    let mut decoder = WebPDecoder::new(data)?;
    let (width, height) = decoder.dimensions();
    let w = width as usize;
    let h = height as usize;

    if stride_pixels < width {
        return Err(DecodeError::InvalidParameter(format!(
            "stride_pixels {} < width {}",
            stride_pixels, width
        )));
    }
    let dst_stride = stride_pixels as usize * 3;
    let required = dst_stride * h;
    if output.len() < required {
        return Err(DecodeError::InvalidParameter(format!(
            "output buffer too small: got {}, need {}",
            output.len(),
            required
        )));
    }

    let output_size = decoder
        .output_buffer_size()
        .ok_or(DecodeError::ImageTooLarge)?;
    let mut temp = vec![0u8; output_size];
    decoder.read_image(&mut temp)?;

    if decoder.has_alpha() {
        // RGBA → BGR with stride (drop alpha + swap R↔B in one pass)
        garb::bytes::rgba_to_bgr_strided(&temp, output, w, h, w * 4, dst_stride)
            .map_err(garb_err)?;
    } else {
        // RGB → BGR with stride (swap R↔B in one pass)
        garb::bytes::rgb_to_bgr_strided(&temp, output, w, h, w * 3, dst_stride)
            .map_err(garb_err)?;
    }

    Ok((width, height))
}

/// Decode WebP data to raw YUV 4:2:0 planes.
///
/// For VP8 lossy images, returns the native YUV planes without upsampling.
/// For VP8L lossless images, decodes to RGBA then converts to YUV.
///
/// # Returns
/// [`YuvPlanes`] containing separate Y, U, and V buffers.
pub fn decode_yuv420(data: &[u8]) -> Result<YuvPlanes, DecodeError> {
    let decoder = WebPDecoder::new(data)?;

    if decoder.is_lossy() && !decoder.is_animated() {
        // For lossy images, extract the native YUV planes from the VP8 frame
        if let Some(range) = decoder.chunks.get(&WebPRiffChunk::VP8) {
            let data_slice = &decoder.r.get_ref()[range.start as usize..range.end as usize];
            let frame = Vp8Decoder::decode_frame(data_slice)?;

            let w = u32::from(frame.width);
            let h = u32::from(frame.height);
            let uv_w = w.div_ceil(2);
            let uv_h = h.div_ceil(2);

            // Macroblock-aligned buffer width (same as frame.buffer_width())
            let buffer_width = {
                let diff = w % 16;
                if diff > 0 {
                    (w + 16 - diff) as usize
                } else {
                    w as usize
                }
            };
            let chroma_bw = buffer_width / 2;

            // Crop from macroblock-aligned buffers to actual dimensions
            let mut y = Vec::with_capacity((w * h) as usize);
            for row in 0..h as usize {
                y.extend_from_slice(
                    &frame.ybuf[row * buffer_width..row * buffer_width + w as usize],
                );
            }

            let mut u = Vec::with_capacity((uv_w * uv_h) as usize);
            let mut v = Vec::with_capacity((uv_w * uv_h) as usize);
            for row in 0..uv_h as usize {
                u.extend_from_slice(&frame.ubuf[row * chroma_bw..row * chroma_bw + uv_w as usize]);
                v.extend_from_slice(&frame.vbuf[row * chroma_bw..row * chroma_bw + uv_w as usize]);
            }

            return Ok(YuvPlanes {
                y,
                u,
                v,
                y_width: w,
                y_height: h,
                uv_width: uv_w,
                uv_height: uv_h,
            });
        }
    }

    // For lossless or animated images, decode to RGBA then convert to YUV
    let (rgba, w, h) = decode_rgba(data)?;
    let (y_bytes, u_bytes, v_bytes) =
        super::yuv::convert_image_yuv::<4>(&rgba, w as u16, h as u16, w as usize);

    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);
    let mb_width = (w as usize).div_ceil(16);

    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;

    // Crop from macroblock-aligned buffers
    let mut y = Vec::with_capacity((w * h) as usize);
    for row in 0..h as usize {
        y.extend_from_slice(&y_bytes[row * luma_width..row * luma_width + w as usize]);
    }

    let mut u = Vec::with_capacity((uv_w * uv_h) as usize);
    let mut v = Vec::with_capacity((uv_w * uv_h) as usize);
    for row in 0..uv_h as usize {
        u.extend_from_slice(&u_bytes[row * chroma_width..row * chroma_width + uv_w as usize]);
        v.extend_from_slice(&v_bytes[row * chroma_width..row * chroma_width + uv_w as usize]);
    }

    Ok(YuvPlanes {
        y,
        u,
        v,
        y_width: w,
        y_height: h,
        uv_width: uv_w,
        uv_height: uv_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const RGB_BPP: usize = 3;

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

        let _ = WebPDecoder::new(&bytes);
    }

    #[test]
    fn decode_2x2_single_color_image() {
        // Image data created from imagemagick and output of xxd:
        // $ convert -size 2x2 xc:#f00 red.webp
        // $ xxd -g 1 red.webp | head

        const NUM_PIXELS: usize = 2 * 2 * RGB_BPP;
        // 2x2 red pixel image
        let bytes = [
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x02, 0x00,
            0x02, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let mut data = [0; NUM_PIXELS];
        let mut decoder = WebPDecoder::new(&bytes).unwrap();
        decoder.read_image(&mut data).unwrap();

        // All pixels are the same value
        let first_pixel = &data[..RGB_BPP];
        assert!(data.chunks_exact(3).all(|ch| ch.iter().eq(first_pixel)));
    }

    #[test]
    fn decode_3x3_single_color_image() {
        // Test that any odd pixel "tail" is decoded properly

        const NUM_PIXELS: usize = 3 * 3 * RGB_BPP;
        // 3x3 red pixel image
        let bytes = [
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x03, 0x00,
            0x03, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let mut data = [0; NUM_PIXELS];
        let mut decoder = WebPDecoder::new(&bytes).unwrap();
        decoder.read_image(&mut data).unwrap();

        // All pixels are the same value
        let first_pixel = &data[..RGB_BPP];
        assert!(data.chunks_exact(3).all(|ch| ch.iter().eq(first_pixel)));
    }
}
