//! Type-safe pixel format traits for decoding and encoding.
//!
//! When the `pixel-types` feature is enabled, you can use generic pixel types
//! from the [`rgb`] crate instead of raw byte slices:
//!
//! ```rust,no_run
//! use zenwebp::pixel;
//! use rgb::{Rgba, Rgb};
//!
//! // Decode to typed pixels
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, w, h): (Vec<Rgba<u8>>, u32, u32) = pixel::decode(webp_data)?;
//!
//! // Encode from typed pixels
//! use zenwebp::{EncoderConfig, EncodeRequest, PixelLayout};
//! let pixels: Vec<Rgb<u8>> = vec![Rgb::new(255, 0, 0); 4 * 4];
//! let config = EncoderConfig::new_lossy();
//! let webp = config.encode_pixels(&pixels, 4, 4)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use alloc::vec::Vec;

use rgb::{AsPixels, ComponentBytes, Rgb, Rgba};

use crate::decoder::DecodeError;
use crate::encoder::{EncodeError, EncodeRequest, EncoderConfig, PixelLayout};

mod private {
    pub trait Sealed {}
}

/// Pixel type that can be decoded from WebP.
pub trait DecodePixel: Copy + 'static + private::Sealed {
    /// Number of color channels (3 for RGB/BGR, 4 for RGBA/BGRA).
    const CHANNELS: usize;
    /// Whether this pixel type includes alpha.
    const HAS_ALPHA: bool;
    /// Whether this pixel type uses BGR channel order.
    const IS_BGR: bool;
}

/// Pixel type that can be encoded to WebP.
pub trait EncodePixel: Copy + 'static + private::Sealed {
    /// Number of color channels.
    const CHANNELS: usize;
    /// The corresponding [`PixelLayout`] for encoding.
    fn color_type() -> PixelLayout;
}

// --- Sealed impls ---

impl private::Sealed for Rgb<u8> {}
impl private::Sealed for Rgba<u8> {}
impl private::Sealed for rgb::Bgr<u8> {}
impl private::Sealed for rgb::Bgra<u8> {}
impl private::Sealed for rgb::Gray<u8> {}
impl private::Sealed for rgb::GrayAlpha<u8> {}

// --- DecodePixel impls ---

impl DecodePixel for Rgb<u8> {
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
    const IS_BGR: bool = false;
}

impl DecodePixel for Rgba<u8> {
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;
    const IS_BGR: bool = false;
}

impl DecodePixel for rgb::Bgr<u8> {
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
    const IS_BGR: bool = true;
}

impl DecodePixel for rgb::Bgra<u8> {
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;
    const IS_BGR: bool = true;
}

// --- EncodePixel impls ---

impl EncodePixel for Rgb<u8> {
    const CHANNELS: usize = 3;
    fn color_type() -> PixelLayout {
        PixelLayout::Rgb8
    }
}

impl EncodePixel for Rgba<u8> {
    const CHANNELS: usize = 4;
    fn color_type() -> PixelLayout {
        PixelLayout::Rgba8
    }
}

impl EncodePixel for rgb::Bgr<u8> {
    const CHANNELS: usize = 3;
    fn color_type() -> PixelLayout {
        PixelLayout::Bgr8
    }
}

impl EncodePixel for rgb::Bgra<u8> {
    const CHANNELS: usize = 4;
    fn color_type() -> PixelLayout {
        PixelLayout::Bgra8
    }
}

impl EncodePixel for rgb::Gray<u8> {
    const CHANNELS: usize = 1;
    fn color_type() -> PixelLayout {
        PixelLayout::L8
    }
}

impl EncodePixel for rgb::GrayAlpha<u8> {
    const CHANNELS: usize = 2;
    fn color_type() -> PixelLayout {
        PixelLayout::La8
    }
}

/// Decode WebP data into a vector of typed pixels.
///
/// The pixel type determines the output format:
/// - [`Rgb<u8>`] — RGB, no alpha
/// - [`Rgba<u8>`] — RGBA with alpha
/// - [`Bgr<u8>`](rgb::Bgr) — BGR, no alpha
/// - [`Bgra<u8>`](rgb::Bgra) — BGRA with alpha
///
/// Returns `(pixels, width, height)`.
pub fn decode<P: DecodePixel>(data: &[u8]) -> Result<(Vec<P>, u32, u32), DecodeError>
where
    [u8]: AsPixels<P>,
{
    let (bytes, w, h) = match (P::IS_BGR, P::HAS_ALPHA) {
        (false, false) => crate::decode_rgb(data)?,
        (false, true) => crate::decode_rgba(data)?,
        (true, false) => crate::decode_bgr(data)?,
        (true, true) => crate::decode_bgra(data)?,
    };
    let pixels: &[P] = bytes.as_pixels();
    Ok((pixels.to_vec(), w, h))
}

/// Decode WebP data into a pre-allocated pixel buffer.
///
/// The buffer must be at least `width * height` pixels. Returns `(width, height)`.
pub fn decode_into<P: DecodePixel>(
    data: &[u8],
    output: &mut [P],
    stride_pixels: u32,
) -> Result<(u32, u32), DecodeError>
where
    [P]: ComponentBytes<u8>,
{
    let buf: &mut [u8] = output.as_bytes_mut();
    match (P::IS_BGR, P::HAS_ALPHA) {
        (false, false) => crate::decode_rgb_into(data, buf, stride_pixels),
        (false, true) => crate::decode_rgba_into(data, buf, stride_pixels),
        (true, false) => crate::decode_bgr_into(data, buf, stride_pixels),
        (true, true) => crate::decode_bgra_into(data, buf, stride_pixels),
    }
}

/// Decode WebP data, appending typed pixels to an existing [`Vec`].
///
/// Useful for reusing an existing buffer or decoding multiple images
/// into the same Vec.
///
/// Returns `(width, height)` of the decoded image.
pub fn decode_append<P: DecodePixel>(
    data: &[u8],
    output: &mut Vec<P>,
) -> Result<(u32, u32), DecodeError>
where
    [u8]: AsPixels<P>,
{
    let (bytes, w, h) = match (P::IS_BGR, P::HAS_ALPHA) {
        (false, false) => crate::decode_rgb(data)?,
        (false, true) => crate::decode_rgba(data)?,
        (true, false) => crate::decode_bgr(data)?,
        (true, true) => crate::decode_bgra(data)?,
    };
    let pixels: &[P] = bytes.as_pixels();
    output.extend_from_slice(pixels);
    Ok((w, h))
}

/// Decode WebP data to an [`imgref::ImgVec`].
///
/// Returns a 2D image buffer with typed pixels and dimensions.
///
/// # Example
///
/// ```rust,no_run
/// use rgb::Rgba;
///
/// let webp_data: &[u8] = &[]; // your WebP data
/// let img: imgref::ImgVec<Rgba<u8>> = zenwebp::pixel::decode_to_img(webp_data)?;
/// println!("{}x{}", img.width(), img.height());
/// # Ok::<(), zenwebp::DecodeError>(())
/// ```
#[cfg(feature = "imgref")]
pub fn decode_to_img<P: DecodePixel>(data: &[u8]) -> Result<imgref::ImgVec<P>, DecodeError>
where
    [u8]: AsPixels<P>,
{
    let (pixels, w, h) = decode::<P>(data)?;
    Ok(imgref::ImgVec::new(pixels, w as usize, h as usize))
}

/// Encode an [`imgref::ImgRef`] to WebP with default settings.
///
/// # Example
///
/// ```rust,no_run
/// use rgb::Rgba;
/// use imgref::ImgVec;
///
/// let img = ImgVec::new(vec![Rgba::new(255, 0, 0, 255); 4 * 4], 4, 4);
/// let webp = zenwebp::pixel::encode_img(img.as_ref())?;
/// # Ok::<(), zenwebp::EncodeError>(())
/// ```
#[cfg(feature = "imgref")]
pub fn encode_img<P: EncodePixel>(img: imgref::ImgRef<'_, P>) -> Result<Vec<u8>, EncodeError>
where
    [P]: ComponentBytes<u8>,
{
    let width = img.width() as u32;
    let height = img.height() as u32;
    let buf: &[u8] = img.buf().as_bytes();
    let config = EncoderConfig::new_lossy();
    EncodeRequest::new(&config, buf, P::color_type(), width, height)
        .with_stride(img.stride())
        .encode()
}

/// Encode typed pixel data to WebP with default settings.
///
/// For custom settings, use [`EncoderConfig::encode_pixels`].
pub fn encode<P: EncodePixel>(pixels: &[P], width: u32, height: u32) -> Result<Vec<u8>, EncodeError>
where
    [P]: ComponentBytes<u8>,
{
    EncoderConfig::new_lossy().encode_pixels(pixels, width, height)
}

impl EncoderConfig {
    /// Encode typed pixel data to WebP.
    ///
    /// Supports [`Rgb<u8>`], [`Rgba<u8>`], [`Bgr<u8>`](rgb::Bgr),
    /// [`Bgra<u8>`](rgb::Bgra), [`Gray<u8>`](rgb::Gray), and
    /// [`GrayAlpha<u8>`](rgb::GrayAlpha).
    pub fn encode_pixels<P: EncodePixel>(
        &self,
        pixels: &[P],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, EncodeError>
    where
        [P]: ComponentBytes<u8>,
    {
        let bytes: &[u8] = pixels.as_bytes();
        let color = P::color_type();
        EncodeRequest::new(self, bytes, color, width, height).encode()
    }

    /// Encode an [`imgref::ImgRef`] to WebP.
    #[cfg(feature = "imgref")]
    pub fn encode_img<P: EncodePixel>(
        &self,
        img: imgref::ImgRef<'_, P>,
    ) -> Result<Vec<u8>, EncodeError>
    where
        [P]: ComponentBytes<u8>,
    {
        let width = img.width() as u32;
        let height = img.height() as u32;
        let buf: &[u8] = img.buf().as_bytes();
        EncodeRequest::new(self, buf, P::color_type(), width, height)
            .with_stride(img.stride())
            .encode()
    }
}
