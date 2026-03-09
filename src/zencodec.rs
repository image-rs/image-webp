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
use alloc::sync::Arc;
use alloc::vec::Vec;

use whereat::{At, ResultAtExt};
use zc::decode::{DecodeOutput, FullFrame, OutputInfo, OwnedFullFrame, SinkError};
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
    .with_cancel(true)
    .with_lossy(true)
    .with_lossless(true)
    .with_animation(true)
    .with_row_level(true)
    .with_native_alpha(true)
    .with_effort_range(0, 10)
    .with_quality_range(0.0, 100.0)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true);

/// Map generic quality (libjpeg-turbo scale) to WebP native quality.
///
/// Calibrated on CID22-512 corpus (209 images) to produce the same median
/// SSIMULACRA2 as libjpeg-turbo at each quality level.
fn calibrated_webp_quality(generic_q: f32) -> f32 {
    const TABLE: &[(f32, f32)] = &[
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 5.0),
        (20.0, 10.4),
        (25.0, 18.0),
        (30.0, 25.4),
        (35.0, 32.3),
        (40.0, 37.8),
        (45.0, 43.4),
        (50.0, 49.2),
        (55.0, 54.3),
        (60.0, 59.5),
        (65.0, 65.8),
        (70.0, 73.4),
        (72.0, 76.0),
        (75.0, 78.1),
        (78.0, 80.3),
        (80.0, 81.8),
        (82.0, 83.4),
        (85.0, 85.9),
        (87.0, 87.5),
        (90.0, 90.5),
        (92.0, 92.2),
        (95.0, 97.4),
        (97.0, 99.0),
        (99.0, 99.0),
    ];
    interp_quality(TABLE, generic_q)
}

/// Piecewise linear interpolation with clamping at table bounds.
fn interp_quality(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

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
        let native = calibrated_webp_quality(clamped);
        self.inner = self.inner.with_quality(native);
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
            canvas_size: self.canvas_size,
            stream: None,
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

/// Streaming accumulation state for row-level encoding.
///
/// Initialized on the first `push_rows` call based on encoder config and input format.
enum StreamAccum {
    /// Lossy opaque RGB8: convert to YUV420 during push for lower peak memory
    /// (1.5 bytes/pixel vs 3 bytes/pixel for RGB8 input).
    Yuv {
        y_plane: Vec<u8>,
        u_plane: Vec<u8>,
        v_plane: Vec<u8>,
        /// Held-back RGB row for chroma pairing when odd rows received.
        pending_row: Vec<u8>,
        width: u32,
        total_rows: u32,
    },
    /// Generic path: accumulate converted pixel bytes.
    Raw {
        pixels: Vec<u8>,
        layout: PixelLayout,
        width: u32,
        total_rows: u32,
    },
}

/// Single-image WebP encoder.
pub struct WebpEncoder<'a> {
    inner_config: EncoderConfig,
    stop: Option<&'a dyn enough::Stop>,
    metadata: Option<crate::ImageMetadata<'a>>,
    limits: ResourceLimits,
    canvas_size: Option<(u32, u32)>,
    stream: Option<StreamAccum>,
}

impl<'a> WebpEncoder<'a> {
    fn do_encode(
        self,
        pixels: &[u8],
        layout: PixelLayout,
        w: u32,
        h: u32,
        stride_pixels: usize,
    ) -> Result<EncodeOutput, EncodeError> {
        self.limits
            .check_dimensions(w, h)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;
        let bpp = layout.bytes_per_pixel() as u64;
        let estimated_mem = w as u64 * h as u64 * bpp;
        self.limits
            .check_memory(estimated_mem)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;

        let mut req = EncodeRequest::new(&self.inner_config, pixels, layout, w, h)
            .with_stride(stride_pixels);
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
///
/// Returns `(bytes, layout, width, height, stride_pixels)`.
/// For passthrough formats (RGB8, RGBA8), bytes may be borrowed zero-copy.
/// For converted formats, bytes are always owned and contiguous (stride = width).
fn pixels_to_webp_input<'a>(
    pixels: &'a PixelSlice<'a>,
) -> Result<(alloc::borrow::Cow<'a, [u8]>, PixelLayout, u32, u32, usize), EncodeError> {
    use alloc::borrow::Cow;

    let desc = pixels.descriptor();
    let w = pixels.width();
    let h = pixels.rows();

    let stride_pixels = pixels.stride() / desc.bytes_per_pixel();

    if desc == PixelDescriptor::RGB8_SRGB {
        Ok((Cow::Borrowed(pixels.as_strided_bytes()), PixelLayout::Rgb8, w, h, stride_pixels))
    } else if desc == PixelDescriptor::RGBA8_SRGB {
        Ok((Cow::Borrowed(pixels.as_strided_bytes()), PixelLayout::Rgba8, w, h, stride_pixels))
    } else if desc == PixelDescriptor::BGRA8_SRGB {
        let raw = pixels.contiguous_bytes();
        let mut rgba = alloc::vec![0u8; raw.len()];
        garb::bytes::bgra_to_rgba(&raw, &mut rgba)
            .map_err(|e| EncodeError::InvalidBufferSize(alloc::format!("pixel conversion: {e}")))?;
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h, w as usize))
    } else if desc == PixelDescriptor::GRAY8_SRGB {
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw.iter().flat_map(|&g| [g, g, g]).collect();
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else if desc == PixelDescriptor::RGBF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats: &[f32] = bytemuck::cast_slice(&raw);
        let mut rgb = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_slice(floats, &mut rgb);
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else if desc == PixelDescriptor::RGBAF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats: &[f32] = bytemuck::cast_slice(&raw);
        let mut rgba = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_rgba_slice(floats, &mut rgba);
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h, w as usize))
    } else if desc == PixelDescriptor::GRAYF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats: &[f32] = bytemuck::cast_slice(&raw);
        let mut gray_u8 = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_slice(floats, &mut gray_u8);
        // gray→rgb: no garb::gray_to_rgb without experimental feature, expand manually
        let mut rgb = alloc::vec![0u8; floats.len() * 3];
        for (i, &g) in gray_u8.iter().enumerate() {
            rgb[i * 3] = g;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = g;
        }
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else {
        Err(EncodeError::InvalidBufferSize(alloc::format!(
            "unsupported pixel format for WebP encode: {:?}",
            desc
        )))
    }
}

/// Convert a pair of RGB8 rows to Y/U/V planes.
///
/// Appends Y samples for both rows and one row of U/V chroma samples
/// (2:1 subsampling in both directions).
fn convert_row_pair_to_yuv(
    row0: &[u8],
    row1: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    u_plane: &mut Vec<u8>,
    v_plane: &mut Vec<u8>,
) {
    use crate::decoder::yuv::{rgb_to_u_avg, rgb_to_v_avg, rgb_to_y};

    // Y for both rows
    for x in 0..width {
        y_plane.push(rgb_to_y(&row0[x * 3..x * 3 + 3]));
    }
    for x in 0..width {
        y_plane.push(rgb_to_y(&row1[x * 3..x * 3 + 3]));
    }

    // U/V: average of 2×2 blocks
    let uv_w = width.div_ceil(2);
    for cx in 0..uv_w {
        let x0 = cx * 2;
        let x1 = (cx * 2 + 1).min(width - 1);
        let p00 = &row0[x0 * 3..x0 * 3 + 3];
        let p01 = &row0[x1 * 3..x1 * 3 + 3];
        let p10 = &row1[x0 * 3..x0 * 3 + 3];
        let p11 = &row1[x1 * 3..x1 * 3 + 3];
        u_plane.push(rgb_to_u_avg(p00, p01, p10, p11));
        v_plane.push(rgb_to_v_avg(p00, p01, p10, p11));
    }
}

/// Convert a single RGB8 row to Y plane + U/V chroma (edge replication).
///
/// Used for the last row when image height is odd.
fn convert_single_row_to_yuv(
    row: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    u_plane: &mut Vec<u8>,
    v_plane: &mut Vec<u8>,
) {
    use crate::decoder::yuv::{rgb_to_u_avg, rgb_to_v_avg, rgb_to_y};

    for x in 0..width {
        y_plane.push(rgb_to_y(&row[x * 3..x * 3 + 3]));
    }
    let uv_w = width.div_ceil(2);
    for cx in 0..uv_w {
        let x0 = cx * 2;
        let x1 = (cx * 2 + 1).min(width - 1);
        let p0 = &row[x0 * 3..x0 * 3 + 3];
        let p1 = &row[x1 * 3..x1 * 3 + 3];
        u_plane.push(rgb_to_u_avg(p0, p1, p0, p1));
        v_plane.push(rgb_to_v_avg(p0, p1, p0, p1));
    }
}

impl zc::encode::Encoder for WebpEncoder<'_> {
    type Error = At<EncodeError>;

    fn reject(op: UnsupportedOperation) -> At<EncodeError> {
        At::from(EncodeError::from(op))
    }

    fn preferred_strip_height(&self) -> u32 {
        match &self.inner_config {
            // Row pair for chroma subsampling (YUV420 processes 2 rows at a time)
            EncoderConfig::Lossy(_) => 2,
            // Lossless has no row-level preference
            EncoderConfig::Lossless(_) => 1,
        }
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, At<EncodeError>> {
        let (buf, layout, w, h, stride) = pixels_to_webp_input(&pixels).map_err(whereat::at)?;
        self.do_encode(&buf, layout, w, h, stride).map_err(whereat::at)
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), At<EncodeError>> {
        let desc = rows.descriptor();
        let strip_w = rows.width();
        let strip_h = rows.rows();

        if strip_h == 0 {
            return Ok(());
        }

        // Initialize streaming state on first push
        if self.stream.is_none() {
            let is_lossy = matches!(self.inner_config, EncoderConfig::Lossy(_));
            let is_rgb8 = desc == PixelDescriptor::RGB8_SRGB;
            let sharp_yuv = match &self.inner_config {
                EncoderConfig::Lossy(cfg) => cfg.sharp_yuv,
                EncoderConfig::Lossless(_) => false,
            };

            if is_lossy && is_rgb8 && !sharp_yuv {
                // YUV conversion path: lower peak memory
                let (cw, ch) = self.canvas_size.unwrap_or((strip_w, 0));
                let y_cap = cw as usize * ch as usize;
                let uv_w = (cw as usize).div_ceil(2);
                let uv_h = (ch as usize).div_ceil(2);
                self.stream = Some(StreamAccum::Yuv {
                    y_plane: Vec::with_capacity(y_cap),
                    u_plane: Vec::with_capacity(uv_w * uv_h),
                    v_plane: Vec::with_capacity(uv_w * uv_h),
                    pending_row: Vec::new(),
                    width: cw,
                    total_rows: 0,
                });
            } else {
                // Generic path: convert per-strip, accumulate bytes
                let (_, layout, _, _, _) = pixels_to_webp_input(&rows).map_err(whereat::at)?;
                let (cw, ch) = self.canvas_size.unwrap_or((strip_w, 0));
                let bpp = layout.bytes_per_pixel();
                let cap = cw as usize * ch as usize * bpp;
                self.stream = Some(StreamAccum::Raw {
                    pixels: Vec::with_capacity(cap),
                    layout,
                    width: cw,
                    total_rows: 0,
                });
                // First strip already converted; store it
                let stream = self.stream.as_mut().unwrap();
                if let StreamAccum::Raw {
                    pixels, total_rows, ..
                } = stream
                {
                    let (buf, _, _, _, _) = pixels_to_webp_input(&rows).map_err(whereat::at)?;
                    pixels.extend_from_slice(&buf);
                    *total_rows += strip_h;
                }
                return Ok(());
            }
        }

        match self.stream.as_mut().unwrap() {
            StreamAccum::Yuv {
                y_plane,
                u_plane,
                v_plane,
                pending_row,
                width,
                total_rows,
            } => {
                let w = *width as usize;
                let data = rows.contiguous_bytes();
                let row_bytes = w * 3;
                let input_row_count = data.len() / row_bytes;
                let mut i = 0;

                // Pair pending row from previous push with first new row
                if !pending_row.is_empty() && input_row_count > 0 {
                    let mut pr = core::mem::take(pending_row);
                    let row1 = &data[0..row_bytes];
                    convert_row_pair_to_yuv(&pr, row1, w, y_plane, u_plane, v_plane);
                    pr.clear();
                    *pending_row = pr; // return allocation
                    i = 1;
                }

                // Process complete row pairs
                while i + 1 < input_row_count {
                    let r0 = &data[i * row_bytes..(i + 1) * row_bytes];
                    let r1 = &data[(i + 1) * row_bytes..(i + 2) * row_bytes];
                    convert_row_pair_to_yuv(r0, r1, w, y_plane, u_plane, v_plane);
                    i += 2;
                }

                // Leftover row → pending
                if i < input_row_count {
                    pending_row.clear();
                    pending_row.extend_from_slice(&data[i * row_bytes..(i + 1) * row_bytes]);
                }

                *total_rows += input_row_count as u32;
            }
            StreamAccum::Raw {
                pixels, total_rows, ..
            } => {
                let (buf, _, _, _, _) = pixels_to_webp_input(&rows).map_err(whereat::at)?;
                pixels.extend_from_slice(&buf);
                *total_rows += strip_h;
            }
        }

        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, At<EncodeError>> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| whereat::at(EncodeError::InvalidBufferSize("no rows pushed".into())))?;

        match stream {
            StreamAccum::Yuv {
                mut y_plane,
                mut u_plane,
                mut v_plane,
                pending_row,
                width,
                total_rows,
            } => {
                // Process pending row (odd image height)
                if !pending_row.is_empty() {
                    convert_single_row_to_yuv(
                        &pending_row,
                        width as usize,
                        &mut y_plane,
                        &mut u_plane,
                        &mut v_plane,
                    );
                }

                // Pack Y+U+V into a single buffer for the Yuv420 encode path
                let mut yuv_buf = Vec::with_capacity(y_plane.len() + u_plane.len() + v_plane.len());
                yuv_buf.append(&mut y_plane);
                yuv_buf.append(&mut u_plane);
                yuv_buf.append(&mut v_plane);

                self.do_encode(&yuv_buf, PixelLayout::Yuv420, width, total_rows, width as usize)
                    .map_err(whereat::at)
            }
            StreamAccum::Raw {
                pixels,
                layout,
                width,
                total_rows,
            } => self
                .do_encode(&pixels, layout, width, total_rows, width as usize)
                .map_err(whereat::at),
        }
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
        stop: Option<&dyn enough::Stop>,
    ) -> Result<(), At<EncodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| whereat::at(EncodeError::from(e)))?;
        }
        let (buf, layout, w, h, _stride) = pixels_to_webp_input(&pixels).map_err(whereat::at)?;
        self.ensure_encoder(w, h)?;
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&buf, layout, timestamp_ms, &self.inner_config)
            .map_err(|e| whereat::at(mux_to_encode_err(e)))?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        Ok(())
    }

    fn finish(self, stop: Option<&dyn enough::Stop>) -> Result<EncodeOutput, At<EncodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| whereat::at(EncodeError::from(e)))?;
        }
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
    .with_cancel(true)
    .with_animation(true)
    .with_cheap_probe(true)
    .with_native_alpha(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true);

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
        let mut info = to_image_info(&native);
        if let Ok(probe) = crate::detect::probe(data) {
            info = info.with_source_encoding_details(probe);
        }
        Ok(info)
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

        // Parse container info before moving data into the self-referential struct.
        let native_info = crate::ImageInfo::from_webp(&data).ok();

        // Probe animation metadata with a temporary decoder.
        let probe_anim = AnimationDecoder::new_with_config(&data, &cfg).map_err(whereat::at)?;
        let anim_info = probe_anim.info();
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
        drop(probe_anim);

        // Build the self-referential struct: owned data + borrowing AnimationDecoder.
        // The AnimationDecoder borrows &[u8] from the owned Vec, so we use self_cell
        // to safely express this relationship without unsafe code.
        let owned_data = data.into_owned();
        let decoder = OwnedAnimDecoder::try_new(
            AnimDecoderOwner {
                data: owned_data,
                config: cfg,
            },
            |owner| AnimationDecoder::new_with_config(&owner.data, &owner.config),
        )
        .map_err(whereat::at)?;

        let has_alpha = anim_info.has_alpha;
        let canvas_width = anim_info.canvas_width;
        let canvas_height = anim_info.canvas_height;
        let bpp: usize = if has_alpha { 4 } else { 3 };
        let buf_size = canvas_width as usize * canvas_height as usize * bpp;
        let source_desc = if has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };

        Ok(WebpFullFrameDecoder {
            decoder,
            preferred: preferred.to_vec(),
            frame_buf: alloc::vec![0u8; buf_size],
            has_alpha,
            canvas_width,
            canvas_height,
            current_duration_ms: 0,
            current_descriptor: source_desc,
            next_frame_index: 0,
            start_frame_index: self.start_frame_index,
            frames_skipped: false,
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

        let mut output = DecodeOutput::new(buf, info);
        if let Ok(probe) = crate::detect::probe(data) {
            output = output.with_source_encoding_details(probe);
        }
        Ok(output)
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
        let mut raw = pixels.into_vec();
        garb::bytes::rgba_to_bgra_inplace(&mut raw)
            .expect("negotiate_format: validated 4bpp buffer");
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

/// Owned data for the self-referential animation decoder.
struct AnimDecoderOwner {
    data: Vec<u8>,
    config: DecodeConfig,
}

// Safety: self_cell uses only safe Rust internally. AnimationDecoder<'a> is
// covariant in 'a, so the #[covariant] annotation is correct.
self_cell::self_cell! {
    struct OwnedAnimDecoder {
        owner: AnimDecoderOwner,
        #[covariant]
        dependent: AnimationDecoder,
    }
}

/// Animation WebP full-frame decoder.
///
/// Decodes frames lazily — each `render_next_frame` call decodes exactly one
/// frame via the underlying [`AnimationDecoder`]. Zero per-frame allocations
/// on the borrowing path (`render_next_frame`); one allocation on the owned
/// path (`render_next_frame_owned`).
///
/// Memory usage is O(canvas_size) instead of O(N_frames × canvas_size).
///
/// The [`AnimationDecoder`] borrows `&[u8]` from the input data. A self-referential
/// struct ([`OwnedAnimDecoder`] via `self_cell`) stores the owned data alongside
/// the borrowing decoder, satisfying the `'static` requirement on `FullFrameDec`.
pub struct WebpFullFrameDecoder {
    decoder: OwnedAnimDecoder,
    preferred: Vec<PixelDescriptor>,
    /// Reusable buffer for composited frame data. Pre-allocated to canvas size;
    /// reused across frames without reallocating.
    frame_buf: Vec<u8>,
    /// Whether the animation has alpha (RGBA vs RGB output).
    has_alpha: bool,
    /// Canvas dimensions.
    canvas_width: u32,
    canvas_height: u32,
    /// Duration of the most recently decoded frame.
    current_duration_ms: u32,
    /// Descriptor after format negotiation for the current frame.
    current_descriptor: PixelDescriptor,
    /// Frame index of the next frame to yield.
    next_frame_index: u32,
    /// Frames before this index are decoded (for compositing) but not returned.
    start_frame_index: u32,
    /// Whether we've already skipped past start_frame_index.
    frames_skipped: bool,
    info: Arc<ImageInfo>,
    total_frames: u32,
    anim_loop_count: Option<u32>,
}

impl WebpFullFrameDecoder {
    /// Decode and discard frames before `start_frame_index` (needed for
    /// correct compositing state). Called lazily on first `render_next_frame`.
    fn skip_to_start(&mut self) -> Result<(), At<DecodeError>> {
        if self.frames_skipped {
            return Ok(());
        }
        self.frames_skipped = true;
        let skip_count = self.start_frame_index as usize;
        for _ in 0..skip_count {
            let done = self
                .decoder
                .with_dependent_mut(|_, anim| match anim.decode_next() {
                    Ok(Some(_)) => Ok(false),
                    Ok(None) => Ok(true),
                    Err(e) => Err(e),
                })
                .map_err(whereat::at)?;
            if done {
                break;
            }
        }
        self.next_frame_index = self.start_frame_index;
        Ok(())
    }

    /// Decode next frame into `self.frame_buf` (zero-alloc: reuses buffer).
    /// Returns `true` if a frame was decoded, `false` if no more frames.
    fn decode_next_into_buf(&mut self) -> Result<bool, At<DecodeError>> {
        // Re-allocate if frame_buf was taken by render_next_frame_owned.
        let expected_size = self.stride() * self.canvas_height as usize;
        if self.frame_buf.len() < expected_size {
            self.frame_buf.resize(expected_size, 0);
        }

        let frame_buf = &mut self.frame_buf;
        let result = self
            .decoder
            .with_dependent_mut(|_, anim| {
                match anim.decode_next() {
                    Ok(Some(info)) => {
                        let data = anim.current_frame_data();
                        // frame_buf is pre-allocated to canvas size; this is a memcpy, no alloc.
                        frame_buf[..data.len()].copy_from_slice(data);
                        Ok(Some(info.duration_ms))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            })
            .map_err(whereat::at)?;

        match result {
            Some(duration_ms) => {
                self.current_duration_ms = duration_ms;
                // Apply RGBA→BGRA swizzle in-place if preferred.
                let source = if self.has_alpha {
                    PixelDescriptor::RGBA8_SRGB
                } else {
                    PixelDescriptor::RGB8_SRGB
                };
                self.current_descriptor =
                    negotiate_format_inplace(&mut self.frame_buf, source, &self.preferred);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Byte stride for the current frame format.
    fn stride(&self) -> usize {
        let bpp: usize = if self.has_alpha { 4 } else { 3 };
        self.canvas_width as usize * bpp
    }
}

/// Apply RGBA→BGRA swizzle in-place if the preferred list requests BGRA.
/// Returns the descriptor that matches the buffer contents after the call.
fn negotiate_format_inplace(
    data: &mut [u8],
    source: PixelDescriptor,
    preferred: &[PixelDescriptor],
) -> PixelDescriptor {
    if !preferred.is_empty()
        && preferred.contains(&PixelDescriptor::BGRA8_SRGB)
        && source == PixelDescriptor::RGBA8_SRGB
    {
        garb::bytes::rgba_to_bgra_inplace(data)
            .expect("negotiate_format_inplace: validated 4bpp buffer");
        return PixelDescriptor::BGRA8_SRGB;
    }
    source
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

    fn render_next_frame(
        &mut self,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Option<FullFrame<'_>>, At<DecodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| whereat::at(DecodeError::from(e)))?;
        }
        self.skip_to_start()?;

        if !self.decode_next_into_buf()? {
            return Ok(None);
        }
        let idx = self.next_frame_index;
        self.next_frame_index += 1;

        // Zero-alloc: create PixelSlice directly from the reusable frame_buf.
        let stride = self.stride();
        let slice = PixelSlice::new(
            &self.frame_buf,
            self.canvas_width,
            self.canvas_height,
            stride,
            self.current_descriptor,
        )
        .map_err(|_| {
            whereat::at(DecodeError::InvalidParameter(
                "frame buffer mismatch".into(),
            ))
        })?;
        Ok(Some(FullFrame::new(slice, self.current_duration_ms, idx)))
    }

    fn render_next_frame_owned(
        &mut self,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Option<OwnedFullFrame>, At<DecodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| whereat::at(DecodeError::from(e)))?;
        }
        self.skip_to_start()?;

        if !self.decode_next_into_buf()? {
            return Ok(None);
        }
        let idx = self.next_frame_index;
        self.next_frame_index += 1;

        // Take the frame_buf for the owned PixelBuffer. A fresh buffer will
        // be allocated on the next call (unavoidable for owned output).
        let data = core::mem::take(&mut self.frame_buf);
        let buf = PixelBuffer::from_vec(
            data,
            self.canvas_width,
            self.canvas_height,
            self.current_descriptor,
        )
        .map_err(|_| whereat::at(DecodeError::InvalidParameter("frame size mismatch".into())))?;
        Ok(Some(OwnedFullFrame::new(
            buf,
            self.current_duration_ms,
            idx,
        )))
    }

    fn render_next_frame_to_sink(
        &mut self,
        stop: Option<&dyn enough::Stop>,
        sink: &mut dyn zc::decode::DecodeRowSink,
    ) -> Result<Option<OutputInfo>, Self::Error> {
        zc::decode::render_frame_to_sink_via_copy(self, stop, sink)
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
        assert!(caps.cancel());
        assert!(caps.lossy());
        assert!(caps.lossless());
        assert!(caps.animation());
        assert!(caps.row_level());
        assert!(caps.native_alpha());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert_eq!(caps.effort_range(), Some([0, 10]));
        assert_eq!(caps.quality_range(), Some([0.0, 100.0]));
        // WebP doesn't natively support these
        assert!(!caps.cicp());
        assert!(!caps.hdr());
        assert!(!caps.native_16bit());
        assert!(!caps.pull());
    }

    #[test]
    fn capabilities_decode() {
        let caps = WebpDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.cancel());
        assert!(caps.animation());
        assert!(caps.cheap_probe());
        assert!(caps.native_alpha());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        // WebP doesn't natively support these
        assert!(!caps.cicp());
        assert!(!caps.hdr());
        assert!(!caps.native_16bit());
        assert!(!caps.decode_into());
        assert!(!caps.row_level());
    }

    #[test]
    fn animation_roundtrip_lazy_decode() {
        use zc::decode::FullFrameDecoder;
        use zc::encode::FullFrameEncoder;

        // Encode a 3-frame animation.
        let frame1 = make_rgba8_pixels(16, 16);
        let frame2 = make_rgba8_pixels(16, 16);
        let frame3 = make_rgba8_pixels(16, 16);
        let enc_config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut enc = enc_config.job().full_frame_encoder().unwrap();
        enc.push_frame(frame1.as_slice(), 100, None).unwrap();
        enc.push_frame(frame2.as_slice(), 100, None).unwrap();
        enc.push_frame(frame3.as_slice(), 100, None).unwrap();
        let output = enc.finish(None).unwrap();
        assert!(!output.is_empty());

        // Decode lazily through FullFrameDecoder.
        let dec_config = WebpDecoderConfig::new();
        let mut dec = dec_config
            .job()
            .full_frame_decoder(Cow::Owned(output.data().to_vec()), &[])
            .unwrap();

        assert_eq!(dec.frame_count(), Some(3));

        // Each render_next_frame decodes one frame lazily.
        let f1 = dec.render_next_frame(None).unwrap().expect("frame 1");
        assert_eq!(f1.frame_index(), 0);
        assert_eq!(f1.duration_ms(), 100);

        let f2 = dec.render_next_frame_owned(None).unwrap().expect("frame 2");
        assert_eq!(f2.frame_index(), 1);

        let f3 = dec.render_next_frame(None).unwrap().expect("frame 3");
        assert_eq!(f3.frame_index(), 2);

        // No more frames.
        assert!(dec.render_next_frame(None).unwrap().is_none());
    }

    #[test]
    fn animation_lazy_decode_with_start_frame() {
        use zc::decode::FullFrameDecoder;
        use zc::encode::FullFrameEncoder;

        // Encode 4 frames.
        let enc_config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut enc = enc_config.job().full_frame_encoder().unwrap();
        for _ in 0..4 {
            let frame = make_rgba8_pixels(8, 8);
            enc.push_frame(frame.as_slice(), 50, None).unwrap();
        }
        let output = enc.finish(None).unwrap();

        // Decode starting from frame 2.
        let dec_config = WebpDecoderConfig::new();
        let mut dec = dec_config
            .job()
            .with_start_frame_index(2)
            .full_frame_decoder(Cow::Owned(output.data().to_vec()), &[])
            .unwrap();

        // First yielded frame should be index 2.
        let f = dec.render_next_frame(None).unwrap().expect("frame 2");
        assert_eq!(f.frame_index(), 2);

        let f = dec.render_next_frame(None).unwrap().expect("frame 3");
        assert_eq!(f.frame_index(), 3);

        assert!(dec.render_next_frame(None).unwrap().is_none());
    }

    #[test]
    fn streaming_encode_lossy_rgb8() {
        // Push rows in strips of 2, verify output decodes correctly.
        let buf = make_rgb8_pixels(64, 64);
        let config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut encoder = config.job().with_canvas_size(64, 64).encoder().unwrap();
        assert_eq!(encoder.preferred_strip_height(), 2);

        // Push 32 strips of 2 rows each
        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 64 * 3;
        for y in (0..64).step_by(2) {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 2) * row_bytes],
                64,
                2,
                row_bytes,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        // Verify decodable
        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn streaming_encode_lossy_rgb8_odd_height() {
        // Image with odd height: tests the pending row chroma handling.
        let buf = make_rgb8_pixels(32, 33);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let mut encoder = config.job().with_canvas_size(32, 33).encoder().unwrap();

        // Push one row at a time: 16 pairs + 1 leftover
        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 32 * 3;
        for y in 0..33usize {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 1) * row_bytes],
                32,
                1,
                row_bytes,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 33);
    }

    #[test]
    fn streaming_encode_lossless_rgba8() {
        // Lossless RGBA: uses the Raw accumulation path.
        let buf = make_rgba8_pixels(16, 16);
        let config = WebpEncoderConfig::lossless();
        let mut encoder = config.job().with_canvas_size(16, 16).encoder().unwrap();
        assert_eq!(encoder.preferred_strip_height(), 1);

        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 16 * 4;
        for y in 0..16usize {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 1) * row_bytes],
                16,
                1,
                row_bytes,
                PixelDescriptor::RGBA8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 16);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn streaming_encode_matches_oneshot() {
        // Verify streaming produces output of similar size to one-shot.
        let buf = make_rgb8_pixels(64, 64);
        let config = WebpEncoderConfig::lossy().with_quality(75.0);

        // One-shot
        let oneshot_output = config
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        // Streaming (full image in one push)
        let mut encoder = config.job().with_canvas_size(64, 64).encoder().unwrap();
        encoder.push_rows(buf.as_slice()).unwrap();
        let stream_output = encoder.finish().unwrap();

        // Streaming via YUV path may differ slightly from one-shot RGB path
        // due to different conversion rounding, but sizes should be similar.
        let oneshot_len = oneshot_output.data().len();
        let stream_len = stream_output.data().len();
        let ratio = stream_len as f64 / oneshot_len as f64;
        assert!(
            (0.8..1.25).contains(&ratio),
            "streaming output size {stream_len} too different from one-shot {oneshot_len} (ratio {ratio:.3})"
        );
    }

    #[test]
    fn streaming_encode_capabilities() {
        let caps = WebpEncoderConfig::capabilities();
        assert!(caps.row_level());
    }
}
