//! zencodec-types trait implementations for zenwebp.
//!
//! Provides [`WebpEncoderConfig`] and [`WebpDecoderConfig`] types that implement
//! the 4-layer trait hierarchy from zencodec-types, wrapping the native zenwebp API.
//!
//! The native API remains untouched — this is a thin adapter layer.
//!
//! # Trait mapping
//!
//! | zencodec-types | zenwebp adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`WebpEncoderConfig`] |
//! | `EncodeJob<'a>` | [`WebpEncodeJob`] |
//! | `EncodeRgb8` etc. | [`WebpEncoder`] |
//! | `FrameEncodeRgb8` etc. | [`WebpFrameEncoder`] |
//! | `DecoderConfig` | [`WebpDecoderConfig`] |
//! | `DecodeJob<'a>` | [`WebpDecodeJob`] |
//! | `Decode` | [`WebpDecoder`] |
//! | `FrameDecode` | [`WebpFrameDecoder`] |

use alloc::sync::Arc;
use alloc::vec::Vec;

use rgb::{Gray, Rgb, Rgba};
use zencodec_types::{
    DecodeFrame, DecodeOutput, EncodeOutput, ImageFormat, ImageInfo, MetadataView, OutputInfo,
    PixelBuffer, PixelBufferConvertExt as _, PixelDescriptor, PixelSlice, PixelSliceMut,
    ResourceLimits, Stop,
};
// Import trait names as _ to avoid conflict with crate-internal names.
use zencodec_types::DecodeJob as _;
use zencodec_types::DecoderConfig as _;
use zencodec_types::EncodeJob as _;
use zencodec_types::EncoderConfig as _;

use crate::encoder::config::EncoderConfig;
use crate::mux::{
    AnimationConfig, AnimationDecoder, AnimationEncoder, BlendMethod, DisposeMethod, MuxError,
};
use crate::{DecodeConfig, DecodeError, DecodeRequest, EncodeError, EncodeRequest, PixelLayout};

// ── Encoding ────────────────────────────────────────────────────────────────

/// WebP encoder configuration implementing [`zencodec_types::EncoderConfig`].
///
/// Wraps the native [`EncoderConfig`] (lossy/lossless enum) and tracks
/// universal quality/effort settings for the trait interface.
///
/// # Examples
///
/// ```rust
/// use zencodec_types::EncoderConfig;
/// use zenwebp::WebpEncoderConfig;
///
/// let enc = WebpEncoderConfig::lossy()
///     .with_quality(85.0)
///     .with_sharp_yuv(true);
/// ```
#[derive(Clone, Debug)]
pub struct WebpEncoderConfig {
    inner: EncoderConfig,
    /// Trait-level effort (0-10 signed scale).
    trait_effort: Option<i32>,
    /// Trait-level calibrated quality (0.0-100.0).
    trait_quality: Option<f32>,
}

impl WebpEncoderConfig {
    /// Create a lossy encoder config with defaults.
    #[must_use]
    pub fn lossy() -> Self {
        Self {
            inner: EncoderConfig::new_lossy(),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Create a lossless encoder config with defaults.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            inner: EncoderConfig::new_lossless(),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Create from a preset with the given quality.
    #[must_use]
    pub fn with_preset(preset: crate::Preset, quality: f32) -> Self {
        Self {
            inner: EncoderConfig::with_preset(preset, quality),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Set encoding quality (0.0 = smallest, 100.0 = best).
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.inner = self.inner.with_quality(quality);
        self
    }

    /// Set quality/speed tradeoff (0-10, mapped to WebP method 0-6).
    #[must_use]
    pub fn with_effort_u32(mut self, effort: u32) -> Self {
        let method = ((effort as u64 * 6) / 10).min(6) as u8;
        self.inner = self.inner.with_method(method);
        self
    }

    /// Enable or disable lossless encoding (inherent method).
    #[must_use]
    pub fn with_lossless_mode(mut self, lossless: bool) -> Self {
        self.inner = self.inner.with_lossless(lossless);
        self
    }

    /// Set alpha plane quality (0.0-100.0).
    #[must_use]
    pub fn with_alpha_quality(mut self, quality: f32) -> Self {
        let aq = quality.clamp(0.0, 100.0) as u8;
        match &mut self.inner {
            EncoderConfig::Lossy(cfg) => cfg.alpha_quality = aq,
            EncoderConfig::Lossless(cfg) => cfg.alpha_quality = aq,
        }
        self
    }

    /// Enable sharp YUV conversion (lossy only).
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        self.inner = self.inner.with_sharp_yuv(enable);
        self
    }

    /// Set SNS strength (lossy only, 0-100).
    #[must_use]
    pub fn with_sns_strength(mut self, strength: u8) -> Self {
        self.inner = self.inner.with_sns_strength(strength);
        self
    }

    /// Set near-lossless preprocessing (lossless only, 0-100, 100=off).
    #[must_use]
    pub fn with_near_lossless(mut self, value: u8) -> Self {
        self.inner = self.inner.with_near_lossless(value);
        self
    }

    /// Set filter strength (lossy only, 0-100).
    #[must_use]
    pub fn with_filter_strength(mut self, strength: u8) -> Self {
        self.inner = self.inner.with_filter_strength(strength);
        self
    }

    /// Set filter sharpness (lossy only, 0-7).
    #[must_use]
    pub fn with_filter_sharpness(mut self, sharpness: u8) -> Self {
        self.inner = self.inner.with_filter_sharpness(sharpness);
        self
    }

    /// Access the underlying [`EncoderConfig`].
    #[must_use]
    pub fn inner(&self) -> &EncoderConfig {
        &self.inner
    }

    /// Mutably access the underlying [`EncoderConfig`].
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        &mut self.inner
    }

    // --- Convenience methods (inherent, use trait flow internally) ---

    /// Set calibrated quality (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_calibrated_quality(self, quality: f32) -> Self {
        <Self as zencodec_types::EncoderConfig>::with_generic_quality(self, quality)
    }

    /// Get calibrated quality (inherent convenience, delegates to trait).
    pub fn calibrated_quality(&self) -> Option<f32> {
        <Self as zencodec_types::EncoderConfig>::generic_quality(self)
    }

    /// Set effort (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_effort(self, effort: i32) -> Self {
        <Self as zencodec_types::EncoderConfig>::with_generic_effort(self, effort)
    }

    /// Get effort (inherent convenience, delegates to trait).
    pub fn effort(&self) -> Option<i32> {
        <Self as zencodec_types::EncoderConfig>::generic_effort(self)
    }

    /// Convenience: encode RGB8 pixels with this config.
    pub fn encode_rgb8(
        &self,
        img: zencodec_types::ImgRef<'_, Rgb<u8>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeRgb8;
        self.job().encoder()?.encode_rgb8(PixelSlice::from(img))
    }

    /// Convenience: encode RGBA8 pixels with this config.
    pub fn encode_rgba8(
        &self,
        img: zencodec_types::ImgRef<'_, Rgba<u8>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeRgba8;
        self.job().encoder()?.encode_rgba8(PixelSlice::from(img))
    }

    /// Convenience: encode Gray8 pixels with this config.
    pub fn encode_gray8(
        &self,
        img: zencodec_types::ImgRef<'_, Gray<u8>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeGray8;
        self.job().encoder()?.encode_gray8(PixelSlice::from(img))
    }

    /// Convenience: encode RGB f32 pixels with this config.
    pub fn encode_rgb_f32(
        &self,
        img: zencodec_types::ImgRef<'_, Rgb<f32>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeRgbF32;
        self.job().encoder()?.encode_rgb_f32(PixelSlice::from(img))
    }

    /// Convenience: encode RGBA f32 pixels with this config.
    pub fn encode_rgba_f32(
        &self,
        img: zencodec_types::ImgRef<'_, Rgba<f32>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeRgbaF32;
        self.job().encoder()?.encode_rgba_f32(PixelSlice::from(img))
    }

    /// Convenience: encode Gray f32 pixels with this config.
    pub fn encode_gray_f32(
        &self,
        img: zencodec_types::ImgRef<'_, Gray<f32>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use zencodec_types::EncodeGrayF32;
        self.job().encoder()?.encode_gray_f32(PixelSlice::from(img))
    }
}

static ENCODE_DESCRIPTORS: &[PixelDescriptor] =
    &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB];

impl zencodec_types::EncoderConfig for WebpEncoderConfig {
    type Error = EncodeError;
    type Job<'a> = WebpEncodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::WebP
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn with_generic_effort(mut self, effort: i32) -> Self {
        let clamped = effort.clamp(0, 10);
        self.trait_effort = Some(clamped);
        let method = ((clamped as u64 * 6) / 10).min(6) as u8;
        self.inner = self.inner.with_method(method);
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        self.trait_effort
    }

    fn with_generic_quality(mut self, quality: f32) -> Self {
        let clamped = quality.clamp(0.0, 100.0);
        self.trait_quality = Some(clamped);
        self.inner = self.inner.with_quality(clamped);
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        self.trait_quality
    }

    fn with_lossless(mut self, lossless: bool) -> Self {
        self.inner = self.inner.with_lossless(lossless);
        self
    }

    fn is_lossless(&self) -> Option<bool> {
        Some(matches!(self.inner, EncoderConfig::Lossless(_)))
    }

    fn job(&self) -> WebpEncodeJob<'_> {
        WebpEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limits: ResourceLimits::none(),
        }
    }
}

// ── Encode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP encode job.
pub struct WebpEncodeJob<'a> {
    config: &'a WebpEncoderConfig,
    stop: Option<&'a dyn Stop>,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    limits: ResourceLimits,
}

impl<'a> WebpEncodeJob<'a> {
    /// Set ICC profile data.
    #[must_use]
    pub fn with_icc(mut self, icc: &'a [u8]) -> Self {
        self.icc = Some(icc);
        self
    }

    /// Set EXIF data.
    #[must_use]
    pub fn with_exif(mut self, exif: &'a [u8]) -> Self {
        self.exif = Some(exif);
        self
    }

    /// Set XMP data.
    #[must_use]
    pub fn with_xmp(mut self, xmp: &'a [u8]) -> Self {
        self.xmp = Some(xmp);
        self
    }

    fn build_inner_config(&self) -> EncoderConfig {
        let mut inner = self.config.inner.clone();
        let mut limits = crate::Limits::none();
        if let Some(px) = self.limits.max_pixels {
            limits = limits.max_total_pixels(px);
        }
        if let Some(mem) = self.limits.max_memory_bytes {
            limits = limits.max_memory(mem);
        }
        if self.limits.max_width.is_some() || self.limits.max_height.is_some() {
            limits = limits.max_dimensions(
                self.limits.max_width.unwrap_or(u32::MAX),
                self.limits.max_height.unwrap_or(u32::MAX),
            );
        }
        inner = inner.limits(limits);
        inner
    }

    fn build_metadata(&self) -> crate::ImageMetadata<'a> {
        let mut meta = crate::ImageMetadata::new();
        if let Some(icc) = self.icc {
            meta = meta.with_icc_profile(icc);
        }
        if let Some(exif) = self.exif {
            meta = meta.with_exif(exif);
        }
        if let Some(xmp) = self.xmp {
            meta = meta.with_xmp(xmp);
        }
        meta
    }

    fn has_metadata(&self) -> bool {
        self.icc.is_some() || self.exif.is_some() || self.xmp.is_some()
    }
}

impl<'a> zencodec_types::EncodeJob<'a> for WebpEncodeJob<'a> {
    type Error = EncodeError;
    type Enc = WebpEncoder<'a>;
    type FrameEnc = WebpFrameEncoder<'a>;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a MetadataView<'a>) -> Self {
        if let Some(icc) = meta.icc_profile {
            self.icc = Some(icc);
        }
        if let Some(exif) = meta.exif {
            self.exif = Some(exif);
        }
        if let Some(xmp) = meta.xmp {
            self.xmp = Some(xmp);
        }
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn encoder(self) -> Result<WebpEncoder<'a>, EncodeError> {
        let inner_config = self.build_inner_config();
        let metadata = if self.has_metadata() {
            Some(self.build_metadata())
        } else {
            None
        };
        Ok(WebpEncoder {
            inner_config,
            stop: self.stop,
            metadata,
            limits: self.limits,
        })
    }

    fn frame_encoder(self) -> Result<WebpFrameEncoder<'a>, EncodeError> {
        let inner_config = self.build_inner_config();
        Ok(WebpFrameEncoder {
            inner_config,
            stop: self.stop,
            anim_enc: None,
            cumulative_ms: 0,
        })
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image WebP encoder.
///
/// Implements per-format encode traits (`EncodeRgb8`, `EncodeRgba8`, etc.)
/// and provides inherent `encode()` for type-erased single-shot encode.
pub struct WebpEncoder<'a> {
    inner_config: EncoderConfig,
    stop: Option<&'a dyn Stop>,
    metadata: Option<crate::ImageMetadata<'a>>,
    limits: ResourceLimits,
}

impl<'a> WebpEncoder<'a> {
    fn do_encode(
        self,
        pixels: &[u8],
        layout: PixelLayout,
        w: u32,
        h: u32,
    ) -> Result<EncodeOutput, EncodeError> {
        // Pre-flight limit checks
        self.limits
            .check_dimensions(w, h)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;
        let bpp = layout.bytes_per_pixel() as u64;
        let estimated_mem = w as u64 * h as u64 * bpp;
        self.limits
            .check_memory(estimated_mem)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;

        let mut req = EncodeRequest::new(&self.inner_config, pixels, layout, w, h);
        if let Some(stop) = self.stop {
            req = req.with_stop(stop);
        }
        if let Some(meta) = self.metadata {
            req = req.with_metadata(meta);
        }
        let data = req.encode()?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }

    /// Type-erased single-shot encode (inherent method for backwards compat).
    pub fn encode<P>(self, pixels: PixelSlice<'_, P>) -> Result<EncodeOutput, EncodeError> {
        let pixels = pixels.erase();
        let (buf, layout, w, h) = pixels_to_webp_input(&pixels)?;
        self.do_encode(&buf, layout, w, h)
    }

    /// Push rows — not supported by WebP (returns error).
    pub fn push_rows<P>(&mut self, _rows: PixelSlice<'_, P>) -> Result<(), EncodeError> {
        Err(EncodeError::InvalidBufferSize(
            "WebP does not support row-level push encoding".into(),
        ))
    }

    /// Finish streaming encode — not supported by WebP (returns error).
    pub fn finish(self) -> Result<EncodeOutput, EncodeError> {
        Err(EncodeError::InvalidBufferSize(
            "WebP does not support row-level push encoding".into(),
        ))
    }

    /// Encode from a pull source — not supported by WebP (returns error).
    pub fn encode_from(
        self,
        _source: &mut dyn FnMut(u32, PixelSliceMut<'_>) -> usize,
    ) -> Result<EncodeOutput, EncodeError> {
        Err(EncodeError::InvalidBufferSize(
            "WebP does not support pull-from-source encoding".into(),
        ))
    }
}

/// Convert a PixelSlice to raw bytes + PixelLayout for the native WebP API.
///
/// Returns `Cow::Borrowed` (zero-copy) for RGB8/RGBA8 when tightly packed,
/// `Cow::Owned` when pixel format conversion or stride stripping is needed.
#[allow(clippy::type_complexity)]
fn pixels_to_webp_input<'a>(
    pixels: &PixelSlice<'a>,
) -> Result<(alloc::borrow::Cow<'a, [u8]>, PixelLayout, u32, u32), EncodeError> {
    use alloc::borrow::Cow;

    let desc = pixels.descriptor();
    let w = pixels.width();
    let h = pixels.rows();

    if desc == PixelDescriptor::RGB8_SRGB {
        Ok((pixels.contiguous_bytes(), PixelLayout::Rgb8, w, h))
    } else if desc == PixelDescriptor::RGBA8_SRGB {
        Ok((pixels.contiguous_bytes(), PixelLayout::Rgba8, w, h))
    } else if desc == PixelDescriptor::BGRA8_SRGB {
        // Swizzle BGRA → RGBA
        let raw = pixels.contiguous_bytes();
        let rgba: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|c| [c[2], c[1], c[0], c[3]])
            .collect();
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h))
    } else if desc == PixelDescriptor::GRAY8_SRGB {
        // Expand grayscale → RGB
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw.iter().flat_map(|&g| [g, g, g]).collect();
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h))
    } else if desc == PixelDescriptor::RGBF32_LINEAR {
        use linear_srgb::default::linear_to_srgb_u8;
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw
            .chunks_exact(12)
            .flat_map(|c| {
                let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
                [
                    linear_to_srgb_u8(r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(b.clamp(0.0, 1.0)),
                ]
            })
            .collect();
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h))
    } else if desc == PixelDescriptor::RGBAF32_LINEAR {
        use linear_srgb::default::linear_to_srgb_u8;
        let raw = pixels.contiguous_bytes();
        let rgba: Vec<u8> = raw
            .chunks_exact(16)
            .flat_map(|c| {
                let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
                let a = f32::from_le_bytes([c[12], c[13], c[14], c[15]]);
                [
                    linear_to_srgb_u8(r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(b.clamp(0.0, 1.0)),
                    (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ]
            })
            .collect();
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h))
    } else if desc == PixelDescriptor::GRAYF32_LINEAR {
        use linear_srgb::default::linear_to_srgb_u8;
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let s = linear_to_srgb_u8(v.clamp(0.0, 1.0));
                [s, s, s]
            })
            .collect();
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h))
    } else {
        Err(EncodeError::InvalidBufferSize(alloc::format!(
            "unsupported pixel format for WebP encode: {:?}",
            desc
        )))
    }
}

// ── Per-format encode trait impls ────────────────────────────────────────────

impl zencodec_types::EncodeRgb8 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_rgb8(self, pixels: PixelSlice<'_, Rgb<u8>>) -> Result<EncodeOutput, EncodeError> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.do_encode(&data, PixelLayout::Rgb8, w, h)
    }
}

impl zencodec_types::EncodeRgba8 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_rgba8(self, pixels: PixelSlice<'_, Rgba<u8>>) -> Result<EncodeOutput, EncodeError> {
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        self.do_encode(&data, PixelLayout::Rgba8, w, h)
    }
}

impl zencodec_types::EncodeGray8 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_gray8(self, pixels: PixelSlice<'_, Gray<u8>>) -> Result<EncodeOutput, EncodeError> {
        // Expand grayscale → RGB for WebP
        let w = pixels.width();
        let h = pixels.rows();
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw.iter().flat_map(|&g| [g, g, g]).collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }
}

impl zencodec_types::EncodeRgbF32 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_rgb_f32(self, pixels: PixelSlice<'_, Rgb<f32>>) -> Result<EncodeOutput, EncodeError> {
        use linear_srgb::default::linear_to_srgb_u8;
        let w = pixels.width();
        let h = pixels.rows();
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw
            .chunks_exact(12)
            .flat_map(|c| {
                let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
                [
                    linear_to_srgb_u8(r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(b.clamp(0.0, 1.0)),
                ]
            })
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }
}

impl zencodec_types::EncodeRgbaF32 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_rgba_f32(
        self,
        pixels: PixelSlice<'_, Rgba<f32>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use linear_srgb::default::linear_to_srgb_u8;
        let w = pixels.width();
        let h = pixels.rows();
        let raw = pixels.contiguous_bytes();
        let rgba: Vec<u8> = raw
            .chunks_exact(16)
            .flat_map(|c| {
                let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
                let a = f32::from_le_bytes([c[12], c[13], c[14], c[15]]);
                [
                    linear_to_srgb_u8(r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(b.clamp(0.0, 1.0)),
                    (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ]
            })
            .collect();
        self.do_encode(&rgba, PixelLayout::Rgba8, w, h)
    }
}

impl zencodec_types::EncodeGrayF32 for WebpEncoder<'_> {
    type Error = EncodeError;
    fn encode_gray_f32(
        self,
        pixels: PixelSlice<'_, Gray<f32>>,
    ) -> Result<EncodeOutput, EncodeError> {
        use linear_srgb::default::linear_to_srgb_u8;
        let w = pixels.width();
        let h = pixels.rows();
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let s = linear_to_srgb_u8(v.clamp(0.0, 1.0));
                [s, s, s]
            })
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }
}

// ── Frame Encoder ───────────────────────────────────────────────────────────

/// Animation WebP frame encoder.
///
/// Wraps [`AnimationEncoder`] and accepts frames via the per-format frame
/// encode traits. The animation encoder is created lazily on the first frame
/// push (to determine canvas dimensions).
///
/// Implements [`FrameEncodeRgb8`](zencodec_types::FrameEncodeRgb8) and
/// [`FrameEncodeRgba8`](zencodec_types::FrameEncodeRgba8) for the trait
/// interface. Also provides inherent methods for backwards-compatible
/// type-erased frame pushing.
pub struct WebpFrameEncoder<'a> {
    inner_config: EncoderConfig,
    stop: Option<&'a dyn Stop>,
    anim_enc: Option<AnimationEncoder>,
    cumulative_ms: u32,
}

/// Convert a [`MuxError`] to an [`EncodeError`].
fn mux_to_encode_err(e: MuxError) -> EncodeError {
    match e {
        MuxError::EncodeError(e) => e,
        MuxError::InvalidDimensions { .. } => EncodeError::InvalidDimensions,
        other => EncodeError::InvalidBufferSize(other.to_string()),
    }
}

impl<'a> WebpFrameEncoder<'a> {
    /// Push a type-erased frame (inherent method for backwards compat).
    pub fn push_frame<P>(
        &mut self,
        pixels: PixelSlice<'_, P>,
        duration_ms: u32,
    ) -> Result<(), EncodeError> {
        let pixels = pixels.erase();
        if let Some(stop) = self.stop {
            stop.check().map_err(EncodeError::Cancelled)?;
        }
        let (buf, layout, w, h) = pixels_to_webp_input(&pixels)?;
        // Initialize the animation encoder lazily on first frame
        if self.anim_enc.is_none() {
            let config = AnimationConfig::default();
            let enc = AnimationEncoder::new(w, h, config).map_err(mux_to_encode_err)?;
            self.anim_enc = Some(enc);
        }
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&buf, layout, timestamp_ms, &self.inner_config)
            .map_err(mux_to_encode_err)?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        Ok(())
    }

    /// Push an advanced encode frame with disposal/blend hints (inherent).
    pub fn push_encode_frame(
        &mut self,
        frame: zencodec_types::EncodeFrame<'_>,
    ) -> Result<(), EncodeError> {
        use zencodec_types::{FrameBlend, FrameDisposal};

        if let Some(stop) = self.stop {
            stop.check().map_err(EncodeError::Cancelled)?;
        }
        let (buf, layout, w, h) = pixels_to_webp_input(&frame.pixels)?;

        // Initialize the animation encoder lazily on first frame
        if self.anim_enc.is_none() {
            let config = AnimationConfig::default();
            let enc = AnimationEncoder::new(w, h, config).map_err(mux_to_encode_err)?;
            self.anim_enc = Some(enc);
        }

        // Map blend/disposal from zencodec-types to zenwebp types
        let dispose = match frame.disposal {
            FrameDisposal::None => DisposeMethod::None,
            FrameDisposal::RestoreBackground => DisposeMethod::Background,
            // WebP only supports None/Background; RestorePrevious maps to None
            FrameDisposal::RestorePrevious => DisposeMethod::None,
            _ => DisposeMethod::None,
        };
        let blend = match frame.blend {
            FrameBlend::Source => BlendMethod::Overwrite,
            FrameBlend::Over => BlendMethod::AlphaBlend,
            _ => BlendMethod::Overwrite,
        };

        // Use frame_rect for sub-canvas positioning, or full canvas
        let (frame_w, frame_h, x_off, y_off) = if let Some([x, y, fw, fh]) = frame.frame_rect {
            (fw, fh, x, y)
        } else {
            (w, h, 0, 0)
        };

        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame_advanced(
            &buf,
            layout,
            frame_w,
            frame_h,
            x_off,
            y_off,
            timestamp_ms,
            &self.inner_config,
            dispose,
            blend,
        )
        .map_err(mux_to_encode_err)?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(frame.duration_ms);
        Ok(())
    }

    /// Finalize the animation (inherent, type-erased finish).
    pub fn finish_animation(self) -> Result<EncodeOutput, EncodeError> {
        let enc = self
            .anim_enc
            .ok_or_else(|| EncodeError::InvalidBufferSize("no frames added".into()))?;
        let last_duration = 100; // default last frame duration
        let data = enc.finalize(last_duration).map_err(mux_to_encode_err)?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }
}

// ── Per-format frame encode trait impls ──────────────────────────────────────

impl zencodec_types::FrameEncodeRgb8 for WebpFrameEncoder<'_> {
    type Error = EncodeError;

    fn push_frame_rgb8(
        &mut self,
        pixels: PixelSlice<'_, Rgb<u8>>,
        duration_ms: u32,
    ) -> Result<(), EncodeError> {
        if let Some(stop) = self.stop {
            stop.check().map_err(EncodeError::Cancelled)?;
        }
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        if self.anim_enc.is_none() {
            let config = AnimationConfig::default();
            let enc = AnimationEncoder::new(w, h, config).map_err(mux_to_encode_err)?;
            self.anim_enc = Some(enc);
        }
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&data, PixelLayout::Rgb8, timestamp_ms, &self.inner_config)
            .map_err(mux_to_encode_err)?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        Ok(())
    }

    fn finish_rgb8(self) -> Result<EncodeOutput, EncodeError> {
        self.finish_animation()
    }
}

impl zencodec_types::FrameEncodeRgba8 for WebpFrameEncoder<'_> {
    type Error = EncodeError;

    fn push_frame_rgba8(
        &mut self,
        pixels: PixelSlice<'_, Rgba<u8>>,
        duration_ms: u32,
    ) -> Result<(), EncodeError> {
        if let Some(stop) = self.stop {
            stop.check().map_err(EncodeError::Cancelled)?;
        }
        let w = pixels.width();
        let h = pixels.rows();
        let data = pixels.contiguous_bytes();
        if self.anim_enc.is_none() {
            let config = AnimationConfig::default();
            let enc = AnimationEncoder::new(w, h, config).map_err(mux_to_encode_err)?;
            self.anim_enc = Some(enc);
        }
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&data, PixelLayout::Rgba8, timestamp_ms, &self.inner_config)
            .map_err(mux_to_encode_err)?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        Ok(())
    }

    fn finish_rgba8(self) -> Result<EncodeOutput, EncodeError> {
        self.finish_animation()
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// WebP decoder configuration implementing [`zencodec_types::DecoderConfig`].
///
/// Wraps [`DecodeConfig`] for the trait interface.
#[derive(Clone, Debug)]
pub struct WebpDecoderConfig {
    inner: DecodeConfig,
}

impl WebpDecoderConfig {
    /// Create a new decoder config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: DecodeConfig::default(),
        }
    }

    /// Set the chroma upsampling method.
    #[must_use]
    pub fn with_upsampling(mut self, method: crate::UpsamplingMethod) -> Self {
        self.inner = self.inner.upsampling(method);
        self
    }

    /// Set resource limits on the inner decode config.
    #[must_use]
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        if let Some(px) = limits.max_pixels {
            self.inner.limits = self.inner.limits.max_total_pixels(px);
        }
        if let Some(mem) = limits.max_memory_bytes {
            self.inner.limits = self.inner.limits.max_memory(mem);
        }
        if limits.max_width.is_some() || limits.max_height.is_some() {
            self.inner.limits = self.inner.limits.max_dimensions(
                limits.max_width.unwrap_or(u32::MAX),
                limits.max_height.unwrap_or(u32::MAX),
            );
        }
        self
    }

    /// Access the underlying [`DecodeConfig`].
    #[must_use]
    pub fn inner(&self) -> &DecodeConfig {
        &self.inner
    }

    /// Mutably access the underlying [`DecodeConfig`].
    pub fn inner_mut(&mut self) -> &mut DecodeConfig {
        &mut self.inner
    }

    // --- Convenience methods (formerly on trait, now inherent) ---

    /// Convenience: probe image header.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, DecodeError> {
        self.job().probe(data)
    }

    /// Convenience: probe full image metadata (may be expensive).
    pub fn probe_full(&self, data: &[u8]) -> Result<ImageInfo, DecodeError> {
        self.job().probe_full(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        use zencodec_types::Decode;
        self.job().decoder()?.decode(data, &[])
    }

    /// Convenience: decode into a pre-allocated RGB8 buffer.
    pub fn decode_into_rgb8(
        &self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, Rgb<u8>>,
    ) -> Result<ImageInfo, DecodeError> {
        self.job()
            .decoder()?
            .decode_into(data, PixelSliceMut::from(dst))
    }

    /// Convenience: decode into a pre-allocated RGBA8 buffer.
    pub fn decode_into_rgba8(
        &self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, Rgba<u8>>,
    ) -> Result<ImageInfo, DecodeError> {
        self.job()
            .decoder()?
            .decode_into(data, PixelSliceMut::from(dst))
    }

    /// Convenience: decode into a pre-allocated RGB f32 buffer.
    pub fn decode_into_rgb_f32(
        &self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, Rgb<f32>>,
    ) -> Result<ImageInfo, DecodeError> {
        self.job()
            .decoder()?
            .decode_into(data, PixelSliceMut::from(dst))
    }

    /// Convenience: decode into a pre-allocated RGBA f32 buffer.
    pub fn decode_into_rgba_f32(
        &self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, Rgba<f32>>,
    ) -> Result<ImageInfo, DecodeError> {
        self.job()
            .decoder()?
            .decode_into(data, PixelSliceMut::from(dst))
    }

    /// Convenience: decode into a pre-allocated Gray f32 buffer.
    pub fn decode_into_gray_f32(
        &self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, Gray<f32>>,
    ) -> Result<ImageInfo, DecodeError> {
        self.job()
            .decoder()?
            .decode_into(data, PixelSliceMut::from(dst))
    }
}

impl Default for WebpDecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

static DECODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
];

impl zencodec_types::DecoderConfig for WebpDecoderConfig {
    type Error = DecodeError;
    type Job<'a> = WebpDecodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::WebP
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn job(&self) -> WebpDecodeJob<'_> {
        WebpDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
        }
    }
}

// ── Decode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP decode job.
pub struct WebpDecodeJob<'a> {
    config: &'a WebpDecoderConfig,
    stop: Option<&'a dyn Stop>,
    limits: ResourceLimits,
}

impl<'a> WebpDecodeJob<'a> {
    fn build_config(&self) -> DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(px) = self.limits.max_pixels {
            cfg.limits = cfg.limits.max_total_pixels(px);
        }
        if let Some(mem) = self.limits.max_memory_bytes {
            cfg.limits = cfg.limits.max_memory(mem);
        }
        if self.limits.max_width.is_some() || self.limits.max_height.is_some() {
            cfg.limits = cfg.limits.max_dimensions(
                self.limits.max_width.unwrap_or(u32::MAX),
                self.limits.max_height.unwrap_or(u32::MAX),
            );
        }
        cfg
    }

    fn effective_file_size_limit(&self) -> Option<u64> {
        self.limits
            .max_file_size
            .or(self.config.inner.limits.max_file_size)
    }
}

impl<'a> zencodec_types::DecodeJob<'a> for WebpDecodeJob<'a> {
    type Error = DecodeError;
    type Dec = WebpDecoder<'a>;
    type FrameDec = WebpFrameDecoder;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, DecodeError> {
        let native = crate::ImageInfo::from_webp(data)?;
        Ok(to_image_info(&native))
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, DecodeError> {
        let native = crate::ImageInfo::from_webp(data)?;
        let desc = if native.has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };
        Ok(OutputInfo::full_decode(native.width, native.height, desc))
    }

    fn decoder(self) -> Result<WebpDecoder<'a>, DecodeError> {
        let cfg = self.build_config();
        Ok(WebpDecoder {
            config: cfg,
            stop: self.stop,
            file_size_limit: self.effective_file_size_limit(),
            limits: self.limits,
        })
    }

    fn frame_decoder(self, data: &[u8]) -> Result<WebpFrameDecoder, DecodeError> {
        if let Some(max) = self.effective_file_size_limit() {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }

        let cfg = self.build_config();
        let mut anim = AnimationDecoder::new_with_config(data, &cfg)?;
        if let Some(stop) = self.stop {
            anim.set_stop(stop);
        }

        let anim_info = anim.info();
        let native_info = crate::ImageInfo::from_webp(data).ok();
        let base_info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(
                anim_info.canvas_width,
                anim_info.canvas_height,
                ImageFormat::WebP,
            )
            .with_alpha(anim_info.has_alpha)
            .with_animation(true)
            .with_frame_count(anim_info.frame_count)
        };
        let shared_info = Arc::new(base_info);
        let total_frames = anim_info.frame_count;

        // Eagerly decode all frames (required because AnimationDecoder borrows data)
        let mut frames = Vec::new();
        while let Some(frame) = anim.next_frame()? {
            let rgba = bytes_to_rgba(&frame.data);
            let buf = PixelBuffer::from_pixels(rgba, frame.width, frame.height)
                .map_err(|_| DecodeError::InvalidParameter("frame size mismatch".into()))?
                .with_descriptor(PixelDescriptor::RGBA8_SRGB);
            frames.push((buf.into(), frame.duration_ms));
        }

        Ok(WebpFrameDecoder {
            frames,
            index: 0,
            info: shared_info,
            total_frames,
        })
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// Single-image WebP decoder.
pub struct WebpDecoder<'a> {
    config: DecodeConfig,
    stop: Option<&'a dyn Stop>,
    file_size_limit: Option<u64>,
    limits: ResourceLimits,
}

impl WebpDecoder<'_> {
    fn check_file_size(&self, data: &[u8]) -> Result<(), DecodeError> {
        if let Some(max) = self.file_size_limit {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }
        Ok(())
    }

    /// Decode into a pre-allocated buffer (inherent method).
    pub fn decode_into<P>(
        self,
        data: &[u8],
        dst: PixelSliceMut<'_, P>,
    ) -> Result<ImageInfo, DecodeError> {
        let mut dst = dst.erase();
        self.check_file_size(data)?;

        let desc = dst.descriptor();
        let w = dst.width();
        let h = dst.rows();

        if desc == PixelDescriptor::RGB8_SRGB {
            // Zero-copy decode into RGB8
            let cfg = self.config.clone();
            let mut req = DecodeRequest::new(&cfg, data);
            if let Some(stop) = self.stop {
                req = req.stop(stop);
            }
            // Use stride if dst has padding between rows
            let stride = dst.stride();
            let expected_stride = w as usize * 3;
            if stride != expected_stride {
                req = req.stride((stride / 3) as u32);
            }
            let total_bytes = stride * h as usize;
            // Build a contiguous mutable byte slice from the PixelSliceMut
            let mut buf = alloc::vec![0u8; total_bytes];
            req.decode_rgb_into(&mut buf)?;
            // Copy back into dst row by row
            let row_bytes = w as usize * 3;
            for y in 0..h {
                let src_start = y as usize * stride;
                let src_row = &buf[src_start..src_start + row_bytes];
                dst.row_mut(y)[..row_bytes].copy_from_slice(src_row);
            }
        } else if desc == PixelDescriptor::RGBA8_SRGB {
            // Zero-copy decode into RGBA8
            let cfg = self.config.clone();
            let mut req = DecodeRequest::new(&cfg, data);
            if let Some(stop) = self.stop {
                req = req.stop(stop);
            }
            let stride = dst.stride();
            let expected_stride = w as usize * 4;
            if stride != expected_stride {
                req = req.stride((stride / 4) as u32);
            }
            let total_bytes = stride * h as usize;
            let mut buf = alloc::vec![0u8; total_bytes];
            req.decode_rgba_into(&mut buf)?;
            let row_bytes = w as usize * 4;
            for y in 0..h {
                let src_start = y as usize * stride;
                let src_row = &buf[src_start..src_start + row_bytes];
                dst.row_mut(y)[..row_bytes].copy_from_slice(src_row);
            }
        } else if desc == PixelDescriptor::BGRA8_SRGB {
            // Decode as RGBA, swizzle to BGRA
            let output = self.do_decode(data)?;
            let pixels = output.into_pixels();
            let src = pixels.to_rgba8();
            let src = src.as_imgref();
            let row_bytes = w as usize * 4;
            for y in 0..h {
                let src_row = &src.buf()[y as usize * src.stride()..][..src.width()];
                let dst_row = &mut dst.row_mut(y)[..row_bytes];
                for (i, px) in src_row.iter().enumerate() {
                    let off = i * 4;
                    dst_row[off] = px.b;
                    dst_row[off + 1] = px.g;
                    dst_row[off + 2] = px.r;
                    dst_row[off + 3] = px.a;
                }
            }
        } else if desc == PixelDescriptor::GRAY8_SRGB {
            // Decode as RGB, convert to luma
            let output = self.do_decode(data)?;
            let pixels = output.into_pixels();
            let src = pixels.to_rgb8();
            let src = src.as_imgref();
            for y in 0..h {
                let src_row = &src.buf()[y as usize * src.stride()..][..src.width()];
                let dst_row = &mut dst.row_mut(y)[..w as usize];
                for (i, px) in src_row.iter().enumerate() {
                    // ITU-R BT.601 luma
                    let luma =
                        ((px.r as u16 * 77 + px.g as u16 * 150 + px.b as u16 * 29) >> 8) as u8;
                    dst_row[i] = luma;
                }
            }
        } else if desc == PixelDescriptor::RGBF32_LINEAR {
            use linear_srgb::default::srgb_u8_to_linear;
            let output = self.do_decode(data)?;
            let pixels = output.into_pixels();
            let src = pixels.to_rgb8();
            let src = src.as_imgref();
            let row_bytes = w as usize * 12;
            for y in 0..h {
                let src_row = &src.buf()[y as usize * src.stride()..][..src.width()];
                let dst_row = &mut dst.row_mut(y)[..row_bytes];
                for (i, px) in src_row.iter().enumerate() {
                    let off = i * 12;
                    dst_row[off..off + 4].copy_from_slice(&srgb_u8_to_linear(px.r).to_le_bytes());
                    dst_row[off + 4..off + 8]
                        .copy_from_slice(&srgb_u8_to_linear(px.g).to_le_bytes());
                    dst_row[off + 8..off + 12]
                        .copy_from_slice(&srgb_u8_to_linear(px.b).to_le_bytes());
                }
            }
        } else if desc == PixelDescriptor::RGBAF32_LINEAR {
            use linear_srgb::default::srgb_u8_to_linear;
            let output = self.do_decode(data)?;
            let pixels = output.into_pixels();
            let src = pixels.to_rgba8();
            let src = src.as_imgref();
            let row_bytes = w as usize * 16;
            for y in 0..h {
                let src_row = &src.buf()[y as usize * src.stride()..][..src.width()];
                let dst_row = &mut dst.row_mut(y)[..row_bytes];
                for (i, px) in src_row.iter().enumerate() {
                    let off = i * 16;
                    dst_row[off..off + 4].copy_from_slice(&srgb_u8_to_linear(px.r).to_le_bytes());
                    dst_row[off + 4..off + 8]
                        .copy_from_slice(&srgb_u8_to_linear(px.g).to_le_bytes());
                    dst_row[off + 8..off + 12]
                        .copy_from_slice(&srgb_u8_to_linear(px.b).to_le_bytes());
                    dst_row[off + 12..off + 16]
                        .copy_from_slice(&(px.a as f32 / 255.0).to_le_bytes());
                }
            }
        } else if desc == PixelDescriptor::GRAYF32_LINEAR {
            use linear_srgb::default::srgb_u8_to_linear;
            let output = self.do_decode(data)?;
            let pixels = output.into_pixels();
            let src = pixels.to_rgb8();
            let src = src.as_imgref();
            let row_bytes = w as usize * 4;
            for y in 0..h {
                let src_row = &src.buf()[y as usize * src.stride()..][..src.width()];
                let dst_row = &mut dst.row_mut(y)[..row_bytes];
                for (i, px) in src_row.iter().enumerate() {
                    let r = srgb_u8_to_linear(px.r);
                    let g = srgb_u8_to_linear(px.g);
                    let b = srgb_u8_to_linear(px.b);
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    dst_row[i * 4..(i + 1) * 4].copy_from_slice(&luma.to_le_bytes());
                }
            }
        } else {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "unsupported pixel format for WebP decode_into: {:?}",
                desc
            )));
        }

        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP).with_alpha(desc.has_alpha())
        };
        Ok(info)
    }

    /// Internal: perform a full decode and return DecodeOutput.
    fn do_decode(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        self.check_file_size(data)?;

        // Pre-flight dimension check: probe dimensions before full decode
        if let Ok(info) = crate::ImageInfo::from_webp(data) {
            self.limits
                .check_dimensions(info.width, info.height)
                .map_err(|e| DecodeError::InvalidParameter(alloc::format!("{e}")))?;
        }

        let mut req = DecodeRequest::new(&self.config, data);
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }

        let (pixels, w, h, layout) = req.decode()?;

        let (buf, has_alpha): (PixelBuffer, bool) = match layout {
            PixelLayout::Rgb8 => {
                let rgb = bytes_to_rgb(&pixels);
                let pb = PixelBuffer::from_pixels(rgb, w, h)
                    .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
                    .with_descriptor(PixelDescriptor::RGB8_SRGB);
                (pb.into(), false)
            }
            PixelLayout::Rgba8 => {
                let rgba = bytes_to_rgba(&pixels);
                let pb = PixelBuffer::from_pixels(rgba, w, h)
                    .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
                    .with_descriptor(PixelDescriptor::RGBA8_SRGB);
                (pb.into(), true)
            }
            _ => {
                // Fallback: decode as RGBA
                let rgba_req = DecodeRequest::new(&self.config, data);
                let (rgba_pixels, rw, rh) = if let Some(stop) = self.stop {
                    rgba_req.stop(stop).decode_rgba()?
                } else {
                    rgba_req.decode_rgba()?
                };
                let rgba = bytes_to_rgba(&rgba_pixels);
                let pb = PixelBuffer::from_pixels(rgba, rw, rh)
                    .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
                    .with_descriptor(PixelDescriptor::RGBA8_SRGB);
                (pb.into(), true)
            }
        };

        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP).with_alpha(has_alpha)
        };

        Ok(DecodeOutput::new(buf, info))
    }
}

/// Apply preferred format negotiation to decoded output.
fn negotiate_format(pixels: PixelBuffer, preferred: &[PixelDescriptor]) -> PixelBuffer {
    if preferred.is_empty() {
        return pixels;
    }
    // Check if any preferred descriptor is BGRA.
    if preferred.contains(&PixelDescriptor::BGRA8_SRGB) {
        return pixels.to_bgra8().into();
    }
    pixels
}

impl zencodec_types::Decode for WebpDecoder<'_> {
    type Error = DecodeError;

    fn decode(
        self,
        data: &[u8],
        preferred: &[PixelDescriptor],
    ) -> Result<DecodeOutput, DecodeError> {
        let output = self.do_decode(data)?;
        if preferred.is_empty() {
            return Ok(output);
        }
        let info = output.info().clone();
        let pixels = negotiate_format(output.into_pixels(), preferred);
        Ok(DecodeOutput::new(pixels, info))
    }
}

// ── Frame Decoder ───────────────────────────────────────────────────────────

/// Animation WebP frame decoder.
///
/// Pre-decodes all frames eagerly (required because the underlying
/// [`AnimationDecoder`] borrows the input data).
pub struct WebpFrameDecoder {
    frames: Vec<(PixelBuffer, u32)>,
    index: usize,
    info: Arc<ImageInfo>,
    total_frames: u32,
}

impl WebpFrameDecoder {
    /// Decode into a pre-allocated buffer for the next frame (inherent method).
    pub fn next_frame_into<P>(
        &mut self,
        _dst: PixelSliceMut<'_, P>,
        _prior_frame: Option<u32>,
    ) -> Result<Option<ImageInfo>, DecodeError> {
        Err(DecodeError::InvalidParameter(
            "WebP animation decode_into not yet supported".into(),
        ))
    }
}

impl zencodec_types::FrameDecode for WebpFrameDecoder {
    type Error = DecodeError;

    fn frame_count(&self) -> Option<u32> {
        Some(self.total_frames)
    }

    fn next_frame(
        &mut self,
        preferred: &[PixelDescriptor],
    ) -> Result<Option<DecodeFrame>, DecodeError> {
        if self.index >= self.frames.len() {
            return Ok(None);
        }
        let (pixels, duration_ms) = self.frames.remove(0);
        let pixels = negotiate_format(pixels, preferred);
        let idx = self.index as u32;
        self.index += 1;
        Ok(Some(DecodeFrame::new(
            pixels,
            Arc::clone(&self.info),
            duration_ms,
            idx,
        )))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a native `crate::ImageInfo` to a `zencodec_types::ImageInfo`.
fn to_image_info(native: &crate::ImageInfo) -> ImageInfo {
    let mut info = ImageInfo::new(native.width, native.height, ImageFormat::WebP)
        .with_alpha(native.has_alpha)
        .with_animation(native.has_animation)
        .with_frame_count(native.frame_count);
    if let Some(ref icc) = native.icc_profile {
        info = info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = native.exif {
        info = info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = native.xmp {
        info = info.with_xmp(xmp.clone());
    }
    info
}

/// Convert raw RGB bytes to `Vec<Rgb<u8>>`.
fn bytes_to_rgb(data: &[u8]) -> Vec<Rgb<u8>> {
    data.chunks_exact(3)
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect()
}

/// Convert raw RGBA bytes to `Vec<Rgba<u8>>`.
fn bytes_to_rgba(data: &[u8]) -> Vec<Rgba<u8>> {
    data.chunks_exact(4)
        .map(|c| Rgba {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec_types::{Decode, DecodeJob, DecoderConfig, EncodeJob, EncoderConfig, ImgVec};

    #[test]
    fn roundtrip_rgb8_lossy() {
        let pixels: Vec<Rgb<u8>> = (0..64 * 64)
            .map(|i| Rgb {
                r: (i % 256) as u8,
                g: ((i * 7) % 256) as u8,
                b: ((i * 13) % 256) as u8,
            })
            .collect();
        let img = ImgVec::new(pixels, 64, 64);

        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.bytes()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn roundtrip_rgba8_lossless() {
        let pixels: Vec<Rgba<u8>> = (0..32 * 32)
            .map(|i| Rgba {
                r: (i % 256) as u8,
                g: ((i * 3) % 256) as u8,
                b: ((i * 7) % 256) as u8,
                a: 255,
            })
            .collect();
        let img = ImgVec::new(pixels, 32, 32);

        let enc = WebpEncoderConfig::lossless();
        let output = enc.encode_rgba8(img.as_ref()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.bytes()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    #[test]
    fn probe_header() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            };
            16 * 16
        ];
        let img = ImgVec::new(pixels, 16, 16);
        let output = WebpEncoderConfig::lossy()
            .encode_rgb8(img.as_ref())
            .unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.probe_header(output.bytes()).unwrap();
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.format, ImageFormat::WebP);
    }

    #[test]
    fn encode_gray8() {
        let pixels: Vec<Gray<u8>> = (0..16 * 16).map(|i| Gray((i % 256) as u8)).collect();
        let img = ImgVec::new(pixels, 16, 16);

        let enc = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = enc.encode_gray8(img.as_ref()).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn job_with_stop() {
        use zencodec_types::Unstoppable;
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 8 * 8];
        let img = ImgVec::new(pixels, 8, 8);
        let enc = WebpEncoderConfig::lossy();
        let output = enc
            .job()
            .with_stop(&Unstoppable)
            .encoder()
            .unwrap()
            .encode(PixelSlice::from(img.as_ref()))
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn f32_roundtrip_all_simd_tiers() {
        use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            // Encode linear f32 → WebP → decode back to f32
            let pixels: Vec<Rgb<f32>> = (0..16 * 16)
                .map(|i| {
                    let t = i as f32 / 255.0;
                    Rgb {
                        r: t,
                        g: (t * 0.7),
                        b: (t * 0.3),
                    }
                })
                .collect();
            let img = ImgVec::new(pixels, 16, 16);

            let enc = WebpEncoderConfig::lossless();
            let output = enc.encode_rgb_f32(img.as_ref()).unwrap();
            assert!(!output.is_empty());

            let dec = WebpDecoderConfig::new();
            let dst = vec![
                Rgb {
                    r: 0.0f32,
                    g: 0.0,
                    b: 0.0,
                };
                16 * 16
            ];
            let mut dst_img = ImgVec::new(dst, 16, 16);
            let _info = dec
                .decode_into_rgb_f32(output.bytes(), dst_img.as_mut())
                .unwrap();

            // Verify values are in valid range
            for p in dst_img.buf().iter() {
                assert!(p.r >= 0.0 && p.r <= 1.0, "r out of range: {}", p.r);
                assert!(p.g >= 0.0 && p.g <= 1.0, "g out of range: {}", p.g);
                assert!(p.b >= 0.0 && p.b <= 1.0, "b out of range: {}", p.b);
            }

            // Also test gray f32 path
            let gray_pixels: Vec<Gray<f32>> = (0..8 * 8).map(|i| Gray(i as f32 / 63.0)).collect();
            let gray_img = ImgVec::new(gray_pixels, 8, 8);
            let gray_out = WebpEncoderConfig::lossless()
                .encode_gray_f32(gray_img.as_ref())
                .unwrap();
            assert!(!gray_out.is_empty());
        });
        assert!(report.permutations_run >= 1);
    }

    #[test]
    fn f32_rgba_roundtrip_lossless() {
        let pixels: Vec<Rgba<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgba {
                    r: t,
                    g: (t * 0.7),
                    b: (t * 0.3),
                    a: 1.0 - t * 0.5,
                }
            })
            .collect();
        let img = ImgVec::new(pixels, 16, 16);

        let enc = WebpEncoderConfig::lossless();
        let output = enc.encode_rgba_f32(img.as_ref()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let mut dst_img = ImgVec::new(
            vec![
                Rgba {
                    r: 0.0f32,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0
                };
                16 * 16
            ],
            16,
            16,
        );
        dec.decode_into_rgba_f32(output.bytes(), dst_img.as_mut())
            .unwrap();

        for p in dst_img.buf().iter() {
            assert!(p.r >= 0.0 && p.r <= 1.0, "r out of range: {}", p.r);
            assert!(p.g >= 0.0 && p.g <= 1.0, "g out of range: {}", p.g);
            assert!(p.b >= 0.0 && p.b <= 1.0, "b out of range: {}", p.b);
            assert!(p.a >= 0.0 && p.a <= 1.0, "a out of range: {}", p.a);
        }
    }

    #[test]
    fn f32_gray_decode_values() {
        use linear_srgb::default::srgb_u8_to_linear;

        // Encode known sRGB gray values as RGB, decode to gray f32
        let pixels = vec![
            Rgb { r: 0u8, g: 0, b: 0 },
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
        ];
        let img = ImgVec::new(pixels, 2, 2);
        let enc = WebpEncoderConfig::lossless();
        let output = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = WebpDecoderConfig::new();
        let mut dst = ImgVec::new(vec![Gray(0.0f32); 4], 2, 2);
        dec.decode_into_gray_f32(output.bytes(), dst.as_mut())
            .unwrap();

        let buf = dst.buf();
        // Black → 0.0
        assert!(buf[0].value().abs() < 0.02, "black: {}", buf[0].value());
        // White → 1.0
        assert!(
            (buf[2].value() - 1.0).abs() < 0.02,
            "white: {}",
            buf[2].value()
        );
        // Mid-gray → should be close to srgb_u8_to_linear(128)
        let expected = srgb_u8_to_linear(128);
        assert!(
            (buf[1].value() - expected).abs() < 0.05,
            "mid: {} vs {}",
            buf[1].value(),
            expected
        );
    }

    #[test]
    fn four_layer_encode_flow() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 100,
                g: 150,
                b: 200
            };
            8 * 8
        ];
        let img = ImgVec::new(pixels, 8, 8);

        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode(PixelSlice::from(img.as_ref()))
            .unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);
    }

    #[test]
    fn four_layer_decode_flow() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 100,
                g: 150,
                b: 200
            };
            8 * 8
        ];
        let img = ImgVec::new(pixels, 8, 8);
        let encoded = WebpEncoderConfig::lossy()
            .encode_rgb8(img.as_ref())
            .unwrap();

        let config = WebpDecoderConfig::new();
        let job = config.job();
        let info = job.output_info(encoded.bytes()).unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);

        let decoded = config
            .job()
            .decoder()
            .unwrap()
            .decode(encoded.bytes(), &[])
            .unwrap();
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
    }

    #[test]
    fn effort_and_quality_getters() {
        let config = WebpEncoderConfig::lossy()
            .with_calibrated_quality(75.0)
            .with_effort(5);

        assert_eq!(config.calibrated_quality(), Some(75.0));
        assert_eq!(config.effort(), Some(5));
        assert_eq!(
            <WebpEncoderConfig as zencodec_types::EncoderConfig>::is_lossless(&config),
            Some(false)
        );

        let lossless = WebpEncoderConfig::lossless();
        assert_eq!(
            <WebpEncoderConfig as zencodec_types::EncoderConfig>::is_lossless(&lossless),
            Some(true)
        );
    }

    #[test]
    fn output_info_minimal() {
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 4 * 4];
        let img = ImgVec::new(pixels, 4, 4);
        let encoded = WebpEncoderConfig::lossy()
            .encode_rgb8(img.as_ref())
            .unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.job().output_info(encoded.bytes()).unwrap();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 4);
    }
}
