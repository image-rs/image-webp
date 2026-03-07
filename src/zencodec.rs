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
//! | `Encoder` | [`WebpEncoder`] |
//! | `FullFrameEncoder` | [`WebpFullFrameEncoder`] |
//! | `DecoderConfig` | [`WebpDecoderConfig`] |
//! | `DecodeJob<'a>` | [`WebpDecodeJob`] |
//! | `Decode` | [`WebpDecoder`] |
//! | `FullFrameDecoder` | [`WebpFullFrameDecoder`] |

use alloc::borrow::Cow;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use whereat::{At, ResultAtExt};
use zc::decode::{DecodeOutput, FullFrame, OwnedFullFrame, OutputInfo, SinkError};
use zc::encode::EncodeOutput;
use zc::{ImageFormat, ImageInfo, MetadataView, ResourceLimits, UnsupportedOperation};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

use crate::encoder::config::EncoderConfig;
use crate::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder, MuxError};
use crate::{DecodeConfig, DecodeError, DecodeRequest, EncodeError, EncodeRequest, PixelLayout};

// ── Encoding ────────────────────────────────────────────────────────────────

/// WebP encoder configuration implementing [`zc::encode::EncoderConfig`].
///
/// Wraps the native [`EncoderConfig`] (lossy/lossless enum) and tracks
/// universal quality/effort settings for the trait interface.
///
/// # Examples
///
/// ```rust,ignore
/// use zc::encode::EncoderConfig;
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
    pub fn with_alpha_quality_value(mut self, quality: f32) -> Self {
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

    /// Set calibrated quality (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_calibrated_quality(self, quality: f32) -> Self {
        <Self as zc::encode::EncoderConfig>::with_generic_quality(self, quality)
    }

    /// Get calibrated quality (inherent convenience, delegates to trait).
    pub fn calibrated_quality(&self) -> Option<f32> {
        <Self as zc::encode::EncoderConfig>::generic_quality(self)
    }

    /// Set effort (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_effort(self, effort: i32) -> Self {
        <Self as zc::encode::EncoderConfig>::with_generic_effort(self, effort)
    }

    /// Get effort (inherent convenience, delegates to trait).
    pub fn effort(&self) -> Option<i32> {
        <Self as zc::encode::EncoderConfig>::generic_effort(self)
    }
}

static ENCODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
    PixelDescriptor::GRAY8_SRGB,
    PixelDescriptor::RGBF32_LINEAR,
    PixelDescriptor::RGBAF32_LINEAR,
    PixelDescriptor::GRAYF32_LINEAR,
];

static ENCODE_CAPABILITIES: zc::encode::EncodeCapabilities = zc::encode::EncodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    .with_lossy(true)
    .with_lossless(true)
    .with_animation(true)
    .with_native_alpha(true)
    .with_effort_range(0, 10)
    .with_quality_range(0.0, 100.0);

impl zc::encode::EncoderConfig for WebpEncoderConfig {
    type Error = At<EncodeError>;
    type Job<'a> = WebpEncodeJob<'a>;

    fn format() -> ImageFormat {
        ImageFormat::WebP
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static zc::encode::EncodeCapabilities {
        &ENCODE_CAPABILITIES
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

    fn with_alpha_quality(self, quality: f32) -> Self {
        self.with_alpha_quality_value(quality)
    }

    fn alpha_quality(&self) -> Option<f32> {
        let aq = match &self.inner {
            EncoderConfig::Lossy(cfg) => cfg.alpha_quality,
            EncoderConfig::Lossless(cfg) => cfg.alpha_quality,
        };
        Some(aq as f32)
    }

    fn job(&self) -> WebpEncodeJob<'_> {
        WebpEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limits: ResourceLimits::none(),
            canvas_size: None,
            loop_count: None,
        }
    }
}

// ── Encode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP encode job.
pub struct WebpEncodeJob<'a> {
    config: &'a WebpEncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    limits: ResourceLimits,
    canvas_size: Option<(u32, u32)>,
    loop_count: Option<Option<u32>>,
}

impl<'a> WebpEncodeJob<'a> {
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

impl<'a> zc::encode::EncodeJob<'a> for WebpEncodeJob<'a> {
    type Error = At<EncodeError>;
    type Enc = WebpEncoder<'a>;
    type FullFrameEnc = WebpFullFrameEncoder;

    fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
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

    fn with_canvas_size(mut self, width: u32, height: u32) -> Self {
        self.canvas_size = Some((width, height));
        self
    }

    fn with_loop_count(mut self, count: Option<u32>) -> Self {
        self.loop_count = Some(count);
        self
    }

    fn encoder(self) -> Result<WebpEncoder<'a>, At<EncodeError>> {
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

    fn full_frame_encoder(self) -> Result<WebpFullFrameEncoder, At<EncodeError>> {
        let inner_config = self.build_inner_config();
        let loop_count = match self.loop_count {
            Some(Some(0)) | None => crate::LoopCount::Forever,
            Some(None) => crate::LoopCount::Forever,
            Some(Some(n)) => {
                let n16 = (n.min(u16::MAX as u32)) as u16;
                crate::LoopCount::Times(
                    core::num::NonZeroU16::new(n16)
                        .unwrap_or(core::num::NonZeroU16::new(1).unwrap()),
                )
            }
        };
        Ok(WebpFullFrameEncoder {
            inner_config,
            anim_enc: None,
            cumulative_ms: 0,
            canvas_size: self.canvas_size,
            loop_count,
        })
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Single-image WebP encoder.
pub struct WebpEncoder<'a> {
    inner_config: EncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
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
}

/// Convert a type-erased PixelSlice to raw bytes + PixelLayout for the native WebP API.
fn pixels_to_webp_input<'a>(
    pixels: &'a PixelSlice<'a>,
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
        let raw = pixels.contiguous_bytes();
        let rgba: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|c| [c[2], c[1], c[0], c[3]])
            .collect();
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h))
    } else if desc == PixelDescriptor::GRAY8_SRGB {
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

impl zc::encode::Encoder for WebpEncoder<'_> {
    type Error = At<EncodeError>;

    fn reject(op: UnsupportedOperation) -> At<EncodeError> {
        At::from(EncodeError::from(op))
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, At<EncodeError>> {
        let (buf, layout, w, h) = pixels_to_webp_input(&pixels).map_err(whereat::at)?;
        self.do_encode(&buf, layout, w, h).map_err(whereat::at)
    }
}

// ── Full Frame Encoder ──────────────────────────────────────────────────────

/// Animation WebP full-frame encoder.
///
/// The animation encoder is created lazily on the first frame push
/// (to determine canvas dimensions).
pub struct WebpFullFrameEncoder {
    inner_config: EncoderConfig,
    anim_enc: Option<AnimationEncoder>,
    cumulative_ms: u32,
    canvas_size: Option<(u32, u32)>,
    loop_count: crate::LoopCount,
}

/// Convert a [`MuxError`] to an [`EncodeError`].
fn mux_to_encode_err(e: MuxError) -> EncodeError {
    use alloc::string::ToString;
    match e {
        MuxError::EncodeError(e) => e,
        MuxError::InvalidDimensions { .. } => EncodeError::InvalidDimensions,
        other => EncodeError::InvalidBufferSize(other.to_string()),
    }
}

impl WebpFullFrameEncoder {
    fn ensure_encoder(&mut self, frame_w: u32, frame_h: u32) -> Result<(), At<EncodeError>> {
        if self.anim_enc.is_none() {
            let (cw, ch) = self.canvas_size.unwrap_or((frame_w, frame_h));
            let config = AnimationConfig {
                loop_count: self.loop_count,
                ..AnimationConfig::default()
            };
            let enc = AnimationEncoder::new(cw, ch, config)
                .map_err(|e| whereat::at(mux_to_encode_err(e)))?;
            self.anim_enc = Some(enc);
        }
        Ok(())
    }
}

impl zc::encode::FullFrameEncoder for WebpFullFrameEncoder {
    type Error = At<EncodeError>;

    fn reject(op: UnsupportedOperation) -> At<EncodeError> {
        At::from(EncodeError::from(op))
    }

    fn push_frame(
        &mut self,
        pixels: PixelSlice<'_>,
        duration_ms: u32,
    ) -> Result<(), At<EncodeError>> {
        let (buf, layout, w, h) = pixels_to_webp_input(&pixels).map_err(whereat::at)?;
        self.ensure_encoder(w, h)?;
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&buf, layout, timestamp_ms, &self.inner_config)
            .map_err(|e| whereat::at(mux_to_encode_err(e)))?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        Ok(())
    }

    fn finish(self) -> Result<EncodeOutput, At<EncodeError>> {
        let enc = self
            .anim_enc
            .ok_or_else(|| EncodeError::InvalidBufferSize("no frames added".into()))
            .map_err(whereat::at)?;
        let last_duration = 100;
        let data = enc
            .finalize(last_duration)
            .map_err(|e| whereat::at(mux_to_encode_err(e)))?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// WebP decoder configuration implementing [`zc::decode::DecoderConfig`].
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

    /// Convenience: probe image header.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, At<DecodeError>> {
        use zc::decode::{DecodeJob, DecoderConfig};
        <Self as DecoderConfig>::job(self).probe(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, At<DecodeError>> {
        use zc::decode::{Decode, DecodeJob, DecoderConfig};
        <Self as DecoderConfig>::job(self)
            .decoder(Cow::Borrowed(data), &[])
            .at()?
            .decode()
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

static DECODE_CAPABILITIES: zc::decode::DecodeCapabilities = zc::decode::DecodeCapabilities::new()
    .with_icc(true)
    .with_exif(true)
    .with_xmp(true)
    .with_animation(true)
    .with_cheap_probe(true)
    .with_native_alpha(true);

impl zc::decode::DecoderConfig for WebpDecoderConfig {
    type Error = At<DecodeError>;
    type Job<'a> = WebpDecodeJob<'a>;

    fn formats() -> &'static [ImageFormat] {
        &[ImageFormat::WebP]
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static zc::decode::DecodeCapabilities {
        &DECODE_CAPABILITIES
    }

    fn job(&self) -> WebpDecodeJob<'_> {
        WebpDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            start_frame_index: 0,
        }
    }
}

// ── Decode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP decode job.
pub struct WebpDecodeJob<'a> {
    config: &'a WebpDecoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    limits: ResourceLimits,
    start_frame_index: u32,
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

    fn effective_input_size_limit(&self) -> Option<u64> {
        self.limits
            .max_input_bytes
            .or(self.config.inner.limits.max_file_size)
    }
}

impl<'a> zc::decode::DecodeJob<'a> for WebpDecodeJob<'a> {
    type Error = At<DecodeError>;
    type Dec = WebpDecoder<'a>;
    type StreamDec = zc::Unsupported<At<DecodeError>>;
    type FullFrameDec = WebpFullFrameDecoder;

    fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_start_frame_index(mut self, index: u32) -> Self {
        self.start_frame_index = index;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, At<DecodeError>> {
        let native = crate::ImageInfo::from_webp(data).map_err(whereat::at)?;
        Ok(to_image_info(&native))
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, At<DecodeError>> {
        let native = crate::ImageInfo::from_webp(data).map_err(whereat::at)?;
        let desc = if native.has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };
        Ok(OutputInfo::full_decode(native.width, native.height, desc))
    }

    fn decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<WebpDecoder<'a>, At<DecodeError>> {
        let cfg = self.build_config();
        Ok(WebpDecoder {
            config: cfg,
            stop: self.stop,
            input_size_limit: self.effective_input_size_limit(),
            limits: self.limits,
            data,
            preferred: preferred.to_vec(),
        })
    }

    fn streaming_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<zc::Unsupported<At<DecodeError>>, At<DecodeError>> {
        Err(whereat::at(DecodeError::InvalidParameter(
            "WebP does not support streaming decode".into(),
        )))
    }

    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn zc::decode::DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        zc::decode::push_decoder_via_full_decode(self, data, sink, preferred, |e| {
            whereat::at(DecodeError::InvalidParameter(alloc::format!("{e}")))
        })
    }

    fn full_frame_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<WebpFullFrameDecoder, At<DecodeError>> {
        if let Some(max) = self.effective_input_size_limit() {
            if data.len() as u64 > max {
                return Err(whereat::at(DecodeError::InvalidParameter(alloc::format!(
                    "input size {} exceeds limit {}",
                    data.len(),
                    max
                ))));
            }
        }

        let cfg = self.build_config();
        let mut anim = AnimationDecoder::new_with_config(&data, &cfg).map_err(whereat::at)?;
        if let Some(stop) = self.stop {
            anim.set_stop(stop);
        }

        let anim_info = anim.info();
        let native_info = crate::ImageInfo::from_webp(&data).ok();
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
        let anim_loop_count = match anim_info.loop_count {
            crate::LoopCount::Forever => Some(0),
            crate::LoopCount::Times(n) => Some(n.get() as u32),
        };

        // Eagerly decode all frames (required because AnimationDecoder borrows data).
        // Apply format negotiation upfront so render_next_frame can borrow in place.
        // Frames before start_frame_index are decoded (needed for compositing) but
        // not stored, reducing peak memory when skipping early frames.
        let start = self.start_frame_index as usize;
        let mut frames = VecDeque::new();
        let mut decoded_count: usize = 0;
        while let Some(frame) = anim.next_frame().map_err(whereat::at)? {
            let frame_idx = decoded_count;
            decoded_count += 1;
            if frame_idx < start {
                // Decode for compositing state but don't store.
                continue;
            }
            let buf = PixelBuffer::from_vec(
                frame.data,
                frame.width,
                frame.height,
                PixelDescriptor::RGBA8_SRGB,
            )
            .map_err(|_| {
                whereat::at(DecodeError::InvalidParameter("frame size mismatch".into()))
            })?;
            let buf = negotiate_format(buf, preferred);
            frames.push_back((buf, frame.duration_ms));
        }

        Ok(WebpFullFrameDecoder {
            frames,
            current_frame: None,
            next_frame_index: start as u32,
            info: shared_info,
            total_frames,
            anim_loop_count,
        })
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// Single-image WebP decoder.
pub struct WebpDecoder<'a> {
    config: DecodeConfig,
    stop: Option<&'a dyn enough::Stop>,
    input_size_limit: Option<u64>,
    limits: ResourceLimits,
    data: Cow<'a, [u8]>,
    preferred: Vec<PixelDescriptor>,
}

impl WebpDecoder<'_> {
    fn check_input_size(&self, data: &[u8]) -> Result<(), DecodeError> {
        if let Some(max) = self.input_size_limit {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "input size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }
        Ok(())
    }

    fn do_decode(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        self.check_input_size(data)?;

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

        let buf: PixelBuffer = match layout {
            PixelLayout::Rgb8 => PixelBuffer::from_vec(pixels, w, h, PixelDescriptor::RGB8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?,
            PixelLayout::Rgba8 => PixelBuffer::from_vec(pixels, w, h, PixelDescriptor::RGBA8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?,
            _ => {
                // Fallback: decode as RGBA
                let rgba_req = DecodeRequest::new(&self.config, data);
                let (rgba_pixels, rw, rh) = if let Some(stop) = self.stop {
                    rgba_req.stop(stop).decode_rgba()?
                } else {
                    rgba_req.decode_rgba()?
                };
                PixelBuffer::from_vec(rgba_pixels, rw, rh, PixelDescriptor::RGBA8_SRGB)
                    .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
            }
        };

        let has_alpha = buf.has_alpha();
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
    let desc = pixels.descriptor();
    // Check if BGRA is preferred and we have RGBA
    if preferred.contains(&PixelDescriptor::BGRA8_SRGB) && desc == PixelDescriptor::RGBA8_SRGB {
        let w = pixels.width();
        let h = pixels.height();
        // Swizzle RGBA → BGRA in-place (avoids extra allocation)
        let mut raw = pixels.into_vec();
        for chunk in raw.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        return PixelBuffer::from_vec(raw, w, h, PixelDescriptor::BGRA8_SRGB)
            .expect("negotiate_format: dimensions unchanged");
    }
    pixels
}

impl zc::decode::Decode for WebpDecoder<'_> {
    type Error = At<DecodeError>;

    fn decode(self) -> Result<DecodeOutput, At<DecodeError>> {
        let output = self.do_decode(&self.data).map_err(whereat::at)?;
        if self.preferred.is_empty() {
            return Ok(output);
        }
        let info = output.info().clone();
        let pixels = negotiate_format(output.into_buffer(), &self.preferred);
        Ok(DecodeOutput::new(pixels, info))
    }
}

// ── Full Frame Decoder ──────────────────────────────────────────────────────

/// Animation WebP full-frame decoder.
///
/// Pre-decodes all frames eagerly (required because the underlying
/// [`AnimationDecoder`] borrows the input data). Consumed frames are
/// freed immediately via `VecDeque::pop_front`, so peak memory is
/// proportional to remaining (not total) frames.
pub struct WebpFullFrameDecoder {
    frames: VecDeque<(PixelBuffer, u32)>,
    /// Holds the most recently yielded frame for borrowing via `render_next_frame`.
    current_frame: Option<(PixelBuffer, u32)>,
    /// Frame index of the next frame to yield (accounts for `start_frame_index`).
    next_frame_index: u32,
    info: Arc<ImageInfo>,
    total_frames: u32,
    anim_loop_count: Option<u32>,
}

impl zc::decode::FullFrameDecoder for WebpFullFrameDecoder {
    type Error = At<DecodeError>;

    fn wrap_sink_error(err: SinkError) -> At<DecodeError> {
        whereat::at(DecodeError::InvalidParameter(alloc::format!("{err}")))
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn frame_count(&self) -> Option<u32> {
        Some(self.total_frames)
    }

    fn loop_count(&self) -> Option<u32> {
        self.anim_loop_count
    }

    fn render_next_frame(&mut self) -> Result<Option<FullFrame<'_>>, At<DecodeError>> {
        let frame = self.frames.pop_front();
        let Some((pixels, duration_ms)) = frame else {
            return Ok(None);
        };
        let idx = self.next_frame_index;
        self.next_frame_index += 1;
        self.current_frame = Some((pixels, duration_ms));
        let (ref pixels, duration_ms) = *self.current_frame.as_ref().unwrap();
        Ok(Some(FullFrame::new(pixels.as_slice(), duration_ms, idx)))
    }

    fn render_next_frame_owned(&mut self) -> Result<Option<OwnedFullFrame>, At<DecodeError>> {
        let Some((pixels, duration_ms)) = self.frames.pop_front() else {
            return Ok(None);
        };
        let idx = self.next_frame_index;
        self.next_frame_index += 1;
        // Drop any previously held frame to free memory before returning.
        self.current_frame = None;
        Ok(Some(OwnedFullFrame::new(pixels, duration_ms, idx)))
    }

    fn render_next_frame_to_sink(
        &mut self,
        sink: &mut dyn zc::decode::DecodeRowSink,
    ) -> Result<Option<OutputInfo>, Self::Error> {
        zc::decode::render_frame_to_sink_via_copy(self, sink)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a native `crate::ImageInfo` to a `zc::ImageInfo`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use zc::decode::{Decode, DecodeJob, DecoderConfig};
    use zc::encode::{EncodeJob, Encoder, EncoderConfig};

    fn make_rgb8_pixels(w: u32, h: u32) -> PixelBuffer {
        let mut buf = PixelBuffer::new(w, h, PixelDescriptor::RGB8_SRGB);
        let mut s = buf.as_slice_mut();
        for y in 0..h {
            let row = s.row_mut(y);
            for (i, b) in row.iter_mut().enumerate() {
                *b = ((y * w * 3 + i as u32) % 256) as u8;
            }
        }
        buf
    }

    fn make_rgba8_pixels(w: u32, h: u32) -> PixelBuffer {
        let mut buf = PixelBuffer::new(w, h, PixelDescriptor::RGBA8_SRGB);
        let mut s = buf.as_slice_mut();
        for y in 0..h {
            let row = s.row_mut(y);
            for chunk in row.chunks_exact_mut(4) {
                chunk[0] = 100;
                chunk[1] = 150;
                chunk[2] = 200;
                chunk[3] = 255;
            }
        }
        buf
    }

    #[test]
    fn roundtrip_rgb8_lossy() {
        let buf = make_rgb8_pixels(64, 64);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn roundtrip_rgba8_lossless() {
        let buf = make_rgba8_pixels(32, 32);
        let enc = WebpEncoderConfig::lossless();
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    #[test]
    fn probe_header() {
        let buf = make_rgb8_pixels(16, 16);
        let enc = WebpEncoderConfig::lossy();
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.probe_header(output.data()).unwrap();
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.format, ImageFormat::WebP);
    }

    #[test]
    fn job_with_stop() {
        let buf = make_rgb8_pixels(8, 8);
        let enc = WebpEncoderConfig::lossy();
        let output = enc
            .job()
            .with_stop(&enough::Unstoppable)
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn four_layer_encode_flow() {
        let buf = make_rgb8_pixels(8, 8);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);
    }

    #[test]
    fn four_layer_decode_flow() {
        let buf = make_rgb8_pixels(8, 8);
        let encoded = WebpEncoderConfig::lossy()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        let config = WebpDecoderConfig::new();
        let job = config.job();
        let info = job.output_info(encoded.data()).unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);

        let decoded = config
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode()
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
            <WebpEncoderConfig as EncoderConfig>::is_lossless(&config),
            Some(false)
        );

        let lossless = WebpEncoderConfig::lossless();
        assert_eq!(
            <WebpEncoderConfig as EncoderConfig>::is_lossless(&lossless),
            Some(true)
        );
    }

    #[test]
    fn output_info_minimal() {
        let buf = make_rgb8_pixels(4, 4);
        let encoded = WebpEncoderConfig::lossy()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.job().output_info(encoded.data()).unwrap();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 4);
    }

    #[test]
    fn encoder_trait_roundtrip() {
        let buf = make_rgb8_pixels(16, 16);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let encoder = config.job().encoder().unwrap();
        let output = encoder.encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert!(decoded.width() > 0);
        assert!(decoded.height() > 0);
    }

    #[test]
    fn encoder_trait_gray8() {
        let mut buf = PixelBuffer::new(16, 16, PixelDescriptor::GRAY8_SRGB);
        {
            let mut s = buf.as_slice_mut();
            for y in 0..16u32 {
                let row = s.row_mut(y);
                for (i, b) in row.iter_mut().enumerate() {
                    *b = ((y as usize * 16 + i) % 256) as u8;
                }
            }
        }
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn dyn_encoder_path() {
        let buf = make_rgb8_pixels(32, 32);
        let config = WebpEncoderConfig::lossy().with_quality(75.0);
        let dyn_enc = config.job().dyn_encoder().unwrap();
        let output = dyn_enc.encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);
    }

    #[test]
    fn encode_srgba8() {
        let mut data = vec![0u8; 16 * 16 * 4];
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = 100;
            chunk[1] = 150;
            chunk[2] = 200;
            chunk[3] = 255;
        }
        let config = WebpEncoderConfig::lossy().with_quality(75.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode_srgba8(&mut data, false, 16, 16, 16)
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn capabilities_encode() {
        let caps = WebpEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.lossy());
        assert!(caps.lossless());
        assert!(caps.animation());
        assert!(caps.native_alpha());
    }

    #[test]
    fn capabilities_decode() {
        let caps = WebpDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.animation());
        assert!(caps.cheap_probe());
        assert!(caps.native_alpha());
    }
}
