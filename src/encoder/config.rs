//! Type-safe encoder configuration API.
//!
//! This module provides separate configuration types for lossy and lossless
//! encoding, ensuring compile-time prevention of invalid parameter combinations.
//!
//! # Primary API (Recommended)
//!
//! Use concrete types directly:
//!
//! ```rust
//! use zenwebp::{LossyConfig, EncodeRequest, PixelLayout};
//!
//! let config = LossyConfig::new()
//!     .with_quality(75.0)
//!     .with_sns_strength(50);
//!
//! let pixels = vec![0u8; 64 * 64 * 4];
//! let webp = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgba8, 64, 64)
//!     .encode()?;
//! # Ok::<(), whereat::At<zenwebp::EncodeError>>(())
//! ```
//!
//! # Runtime Mode Selection
//!
//! When you need to choose the mode at runtime, use the enum wrapper:
//!
//! ```rust
//! use zenwebp::{EncoderConfig, LossyConfig, LosslessConfig};
//!
//! fn get_config(has_transparency: bool) -> EncoderConfig {
//!     if has_transparency {
//!         EncoderConfig::Lossless(LosslessConfig::new())
//!     } else {
//!         EncoderConfig::Lossy(LossyConfig::new())
//!     }
//! }
//! ```

use crate::{Limits, Preset};

/// Configuration for lossy (VP8) encoding.
///
/// Lossy encoding uses DCT-based compression similar to JPEG, with additional
/// features like loop filtering, spatial noise shaping, and segmentation.
#[derive(Clone)]
#[non_exhaustive]
pub struct LossyConfig {
    /// Encoding quality (0.0 = smallest, 100.0 = best). Default: 75.0.
    pub quality: f32,
    /// Quality/speed tradeoff (0 = fast, 6 = slower but better). Default: 4.
    pub method: u8,
    /// Alpha channel quality (0-100, 100 = lossless alpha). Default: 100.
    pub alpha_quality: u8,
    /// Target file size in bytes (0 = disabled). Default: 0.
    pub target_size: u32,
    /// Target PSNR in dB (0.0 = disabled). Default: 0.0.
    pub target_psnr: f32,
    /// Content-aware preset (affects SNS, filter, sharpness). Default: None.
    pub preset: Option<Preset>,
    /// Use sharp (iterative) YUV conversion. Default: false.
    pub sharp_yuv: bool,
    /// Spatial noise shaping strength (0-100). None = use preset default.
    pub sns_strength: Option<u8>,
    /// Loop filter strength (0-100). None = use preset default.
    pub filter_strength: Option<u8>,
    /// Loop filter sharpness (0-7). None = use preset default.
    pub filter_sharpness: Option<u8>,
    /// Number of segments (1-4). None = use preset default.
    pub segments: Option<u8>,
    /// Resource limits for validation.
    pub limits: Limits,
}

impl Default for LossyConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LossyConfig {
    /// Create a new lossy encoder configuration with defaults.
    ///
    /// Default: quality 75, method 4, no preset overrides.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality: 75.0,
            method: 4,
            alpha_quality: 100,
            target_size: 0,
            target_psnr: 0.0,
            preset: None,
            sharp_yuv: false,
            sns_strength: None,
            filter_strength: None,
            filter_sharpness: None,
            segments: None,
            limits: Limits::none(),
        }
    }

    /// Create from a preset with the given quality.
    #[must_use]
    pub fn with_preset(preset: Preset, quality: f32) -> Self {
        Self {
            quality,
            preset: Some(preset),
            ..Self::new()
        }
    }

    /// Set encoding quality (0.0 = smallest file, 100.0 = best quality).
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(0.0, 100.0);
        self
    }

    /// Set method (0 = fastest, 6 = slowest but best compression).
    #[must_use]
    pub fn with_method(mut self, method: u8) -> Self {
        self.method = method.min(6);
        self
    }

    /// Set alpha channel quality (0-100, 100 = lossless alpha).
    #[must_use]
    pub fn with_alpha_quality(mut self, quality: u8) -> Self {
        self.alpha_quality = quality.min(100);
        self
    }

    /// Target output file size in bytes (encoder will adjust quality).
    #[must_use]
    pub fn with_target_size(mut self, size: u32) -> Self {
        self.target_size = size;
        self
    }

    /// Target PSNR in dB (encoder will adjust quality).
    #[must_use]
    pub fn with_target_psnr(mut self, psnr: f32) -> Self {
        self.target_psnr = psnr;
        self
    }

    /// Apply a content-aware preset (overrides SNS, filter, sharpness defaults).
    #[must_use]
    pub fn with_preset_value(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Enable sharp (iterative) YUV conversion for better quality.
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        self.sharp_yuv = enable;
        self
    }

    /// Override spatial noise shaping strength (0-100).
    /// Higher values preserve more texture detail.
    #[must_use]
    pub fn with_sns_strength(mut self, strength: u8) -> Self {
        self.sns_strength = Some(strength.min(100));
        self
    }

    /// Override loop filter strength (0-100).
    /// Higher values produce smoother output.
    #[must_use]
    pub fn with_filter_strength(mut self, strength: u8) -> Self {
        self.filter_strength = Some(strength.min(100));
        self
    }

    /// Override loop filter sharpness (0-7).
    #[must_use]
    pub fn with_filter_sharpness(mut self, sharpness: u8) -> Self {
        self.filter_sharpness = Some(sharpness.min(7));
        self
    }

    /// Override number of segments for adaptive quantization (1-4).
    #[must_use]
    pub fn with_segments(mut self, segments: u8) -> Self {
        self.segments = Some(segments.clamp(1, 4));
        self
    }

    /// Set resource limits for validation.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set maximum dimensions.
    #[must_use]
    pub fn with_max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.limits = self.limits.max_dimensions(width, height);
        self
    }

    /// Set maximum memory usage in bytes.
    #[must_use]
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.limits = self.limits.max_memory(bytes);
        self
    }

    /// Estimate peak memory usage for encoding an image.
    ///
    /// Returns the typical peak memory consumption in bytes.
    #[must_use]
    pub fn estimate_memory(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(
            width,
            height,
            bpp,
            &crate::EncoderConfig::Lossy(self.clone()),
        )
        .peak_memory_bytes
    }

    /// Estimate worst-case peak memory usage for encoding an image.
    ///
    /// Returns the maximum expected peak memory (high-entropy content).
    #[must_use]
    pub fn estimate_memory_ceiling(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(
            width,
            height,
            bpp,
            &crate::EncoderConfig::Lossy(self.clone()),
        )
        .peak_memory_bytes_max
    }
}

/// Configuration for lossless (VP8L) encoding.
///
/// Lossless encoding uses prediction, color transforms, and LZ77 compression
/// to achieve perfect reconstruction of the original pixels.
#[derive(Clone)]
#[non_exhaustive]
pub struct LosslessConfig {
    /// Encoding effort (0.0 = fastest, 100.0 = best compression). Default: 75.0.
    pub quality: f32,
    /// Quality/speed tradeoff (0 = fast, 6 = slower but better). Default: 4.
    pub method: u8,
    /// Alpha channel quality (0-100, 100 = lossless alpha). Default: 100.
    pub alpha_quality: u8,
    /// Target file size in bytes (0 = disabled). Default: 0.
    pub target_size: u32,
    /// Near-lossless preprocessing (0 = max preprocessing, 100 = off). Default: 100.
    pub near_lossless: u8,
    /// Preserve exact RGB values under transparent areas. Default: false.
    pub exact: bool,
    /// Resource limits for validation.
    pub limits: Limits,
}

impl Default for LosslessConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LosslessConfig {
    /// Create a new lossless encoder configuration with defaults.
    ///
    /// Default: quality 75, method 4, fully lossless (near_lossless = 100).
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality: 75.0,
            method: 4,
            alpha_quality: 100,
            target_size: 0,
            near_lossless: 100,
            exact: false,
            limits: Limits::none(),
        }
    }

    /// Set encoding effort (0.0 = fastest, 100.0 = best compression).
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(0.0, 100.0);
        self
    }

    /// Set method (0 = fastest, 6 = slowest but best compression).
    #[must_use]
    pub fn with_method(mut self, method: u8) -> Self {
        self.method = method.min(6);
        self
    }

    /// Set alpha channel quality (0-100, 100 = lossless alpha).
    #[must_use]
    pub fn with_alpha_quality(mut self, quality: u8) -> Self {
        self.alpha_quality = quality.min(100);
        self
    }

    /// Target output file size in bytes (encoder will adjust effort).
    #[must_use]
    pub fn with_target_size(mut self, size: u32) -> Self {
        self.target_size = size;
        self
    }

    /// Enable near-lossless preprocessing (0 = max preprocessing, 100 = off).
    ///
    /// Values < 100 allow slight color changes to improve compression while
    /// maintaining the illusion of lossless quality.
    #[must_use]
    pub fn with_near_lossless(mut self, value: u8) -> Self {
        self.near_lossless = value.min(100);
        self
    }

    /// Preserve exact RGB values even under fully transparent pixels.
    ///
    /// By default, RGB values under alpha=0 may be modified for better compression.
    /// Enable this to preserve them exactly (e.g., for alpha compositing workflows).
    #[must_use]
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Set resource limits for validation.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set maximum dimensions.
    #[must_use]
    pub fn with_max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.limits = self.limits.max_dimensions(width, height);
        self
    }

    /// Set maximum memory usage in bytes.
    #[must_use]
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.limits = self.limits.max_memory(bytes);
        self
    }

    /// Estimate peak memory usage for encoding an image.
    ///
    /// Returns the typical peak memory consumption in bytes.
    #[must_use]
    pub fn estimate_memory(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(
            width,
            height,
            bpp,
            &crate::EncoderConfig::Lossless(self.clone()),
        )
        .peak_memory_bytes
    }

    /// Estimate worst-case peak memory usage for encoding an image.
    ///
    /// Returns the maximum expected peak memory (high-entropy content).
    #[must_use]
    pub fn estimate_memory_ceiling(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(
            width,
            height,
            bpp,
            &crate::EncoderConfig::Lossless(self.clone()),
        )
        .peak_memory_bytes_max
    }
}

/// Encoder configuration enum for runtime mode selection.
///
/// Use this when you need to choose between lossy and lossless at runtime.
/// For compile-time mode selection, use [`LossyConfig`] or [`LosslessConfig`] directly.
///
/// # Example
///
/// ```rust
/// use zenwebp::{EncoderConfig, LossyConfig, LosslessConfig};
///
/// fn choose_config(has_transparency: bool) -> EncoderConfig {
///     if has_transparency {
///         EncoderConfig::Lossless(LosslessConfig::new())
///     } else {
///         EncoderConfig::Lossy(LossyConfig::new().with_quality(80.0))
///     }
/// }
/// ```
#[derive(Clone)]
pub enum EncoderConfig {
    /// Lossy (VP8) encoding configuration.
    Lossy(LossyConfig),
    /// Lossless (VP8L) encoding configuration.
    Lossless(LosslessConfig),
}

impl EncoderConfig {
    /// Create a new lossy encoder configuration.
    ///
    /// Convenience wrapper for `EncoderConfig::Lossy(LossyConfig::new())`.
    #[must_use]
    pub fn new_lossy() -> Self {
        Self::Lossy(LossyConfig::new())
    }

    /// Create a new lossless encoder configuration.
    ///
    /// Convenience wrapper for `EncoderConfig::Lossless(LosslessConfig::new())`.
    #[must_use]
    pub fn new_lossless() -> Self {
        Self::Lossless(LosslessConfig::new())
    }

    /// Create from a preset with the given quality.
    #[must_use]
    pub fn with_preset(preset: Preset, quality: f32) -> Self {
        Self::Lossy(LossyConfig::with_preset(preset, quality))
    }

    /// Set encoding quality (0.0 = smallest, 100.0 = best).
    ///
    /// Works for both lossy and lossless configurations.
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        match &mut self {
            Self::Lossy(cfg) => cfg.quality = quality,
            Self::Lossless(cfg) => cfg.quality = quality,
        }
        self
    }

    /// Set encoding method (0-6, higher = better compression but slower).
    ///
    /// Works for both lossy and lossless configurations.
    #[must_use]
    pub fn with_method(mut self, method: u8) -> Self {
        match &mut self {
            Self::Lossy(cfg) => cfg.method = method,
            Self::Lossless(cfg) => cfg.method = method,
        }
        self
    }

    /// Set SNS strength (lossy only, 0-100). No effect on lossless.
    #[must_use]
    pub fn with_sns_strength(mut self, strength: u8) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.sns_strength = Some(strength);
        }
        self
    }

    /// Set filter strength (lossy only, 0-100). No effect on lossless.
    #[must_use]
    pub fn with_filter_strength(mut self, strength: u8) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.filter_strength = Some(strength);
        }
        self
    }

    /// Set filter sharpness (lossy only, 0-7). No effect on lossless.
    #[must_use]
    pub fn with_filter_sharpness(mut self, sharpness: u8) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.filter_sharpness = Some(sharpness);
        }
        self
    }

    /// Set number of segments (lossy only, 1-4). No effect on lossless.
    #[must_use]
    pub fn with_segments(mut self, segments: u8) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.segments = Some(segments);
        }
        self
    }

    /// Set near-lossless preprocessing (lossless only, 0-100). No effect on lossy.
    #[must_use]
    pub fn with_near_lossless(mut self, value: u8) -> Self {
        if let Self::Lossless(cfg) = &mut self {
            cfg.near_lossless = value;
        }
        self
    }

    /// Set target file size in bytes. 0 = disabled.
    ///
    /// Works for both lossy and lossless configurations.
    #[must_use]
    pub fn with_target_size(mut self, bytes: u32) -> Self {
        match &mut self {
            Self::Lossy(cfg) => cfg.target_size = bytes,
            Self::Lossless(cfg) => cfg.target_size = bytes,
        }
        self
    }

    /// Set target PSNR in dB (lossy only). 0.0 = disabled. No effect on lossless.
    #[must_use]
    pub fn with_target_psnr(mut self, psnr: f32) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.target_psnr = psnr;
        }
        self
    }

    /// Set resource limits for dimensions and memory validation.
    ///
    /// Works for both lossy and lossless configurations.
    #[must_use]
    pub fn limits(mut self, limits: crate::decoder::Limits) -> Self {
        match &mut self {
            Self::Lossy(cfg) => cfg.limits = limits,
            Self::Lossless(cfg) => cfg.limits = limits,
        }
        self
    }

    /// Enable/disable sharp YUV conversion (lossy only). No effect on lossless.
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        if let Self::Lossy(cfg) = &mut self {
            cfg.sharp_yuv = enable;
        }
        self
    }

    /// Switch between lossy and lossless encoding.
    ///
    /// When switching, preserves common settings (quality, method, limits, etc.).
    #[must_use]
    pub fn with_lossless(self, enable: bool) -> Self {
        match (&self, enable) {
            (Self::Lossy(_), true) => {
                // Switch to lossless
                let q = self.get_quality();
                let m = self.get_method();
                let l = self.get_limits().clone();
                Self::Lossless(LosslessConfig {
                    quality: q,
                    method: m,
                    limits: l,
                    ..LosslessConfig::new()
                })
            }
            (Self::Lossless(_), false) => {
                // Switch to lossy
                let q = self.get_quality();
                let m = self.get_method();
                let l = self.get_limits().clone();
                Self::Lossy(LossyConfig {
                    quality: q,
                    method: m,
                    limits: l,
                    ..LossyConfig::new()
                })
            }
            _ => self, // Already in requested mode
        }
    }

    /// Estimate resource consumption for encoding an image with this config.
    ///
    /// Returns memory, time, and output size estimates.
    #[must_use]
    pub fn estimate(&self, width: u32, height: u32, bpp: u8) -> crate::heuristics::EncodeEstimate {
        crate::heuristics::estimate_encode(width, height, bpp, self)
    }

    /// Estimate peak memory usage for encoding an image.
    ///
    /// Returns the typical peak memory consumption in bytes.
    #[must_use]
    pub fn estimate_memory(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(width, height, bpp, self).peak_memory_bytes
    }

    /// Estimate worst-case peak memory usage for encoding an image.
    ///
    /// Returns the maximum expected peak memory (high-entropy content).
    #[must_use]
    pub fn estimate_memory_ceiling(&self, width: u32, height: u32, bpp: u8) -> u64 {
        crate::heuristics::estimate_encode(width, height, bpp, self).peak_memory_bytes_max
    }

    /// Check if this is a lossless configuration.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Lossless(_))
    }

    /// Get the underlying lossy config, if this is a lossy configuration.
    #[must_use]
    pub fn as_lossy(&self) -> Option<&LossyConfig> {
        match self {
            Self::Lossy(cfg) => Some(cfg),
            Self::Lossless(_) => None,
        }
    }

    /// Get the underlying lossless config, if this is a lossless configuration.
    #[must_use]
    pub fn as_lossless(&self) -> Option<&LosslessConfig> {
        match self {
            Self::Lossy(_) => None,
            Self::Lossless(cfg) => Some(cfg),
        }
    }
}

// Manual Debug implementations (Stop trait doesn't implement Debug)
impl core::fmt::Debug for LossyConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LossyConfig")
            .field("quality", &self.quality)
            .field("method", &self.method)
            .field("alpha_quality", &self.alpha_quality)
            .field("target_size", &self.target_size)
            .field("target_psnr", &self.target_psnr)
            .field("preset", &self.preset)
            .field("sharp_yuv", &self.sharp_yuv)
            .field("sns_strength", &self.sns_strength)
            .field("filter_strength", &self.filter_strength)
            .field("filter_sharpness", &self.filter_sharpness)
            .field("segments", &self.segments)
            .field("limits", &self.limits)
            .finish()
    }
}

impl core::fmt::Debug for LosslessConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LosslessConfig")
            .field("quality", &self.quality)
            .field("method", &self.method)
            .field("alpha_quality", &self.alpha_quality)
            .field("target_size", &self.target_size)
            .field("near_lossless", &self.near_lossless)
            .field("exact", &self.exact)
            .field("limits", &self.limits)
            .finish()
    }
}

impl core::fmt::Debug for EncoderConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Lossy(cfg) => f.debug_tuple("Lossy").field(cfg).finish(),
            Self::Lossless(cfg) => f.debug_tuple("Lossless").field(cfg).finish(),
        }
    }
}

// Conversion to internal EncoderParams
use super::api::EncoderParams;
use super::fast_math;

impl LossyConfig {
    pub(crate) fn to_params(&self) -> EncoderParams {
        // Base tuning values from preset (matches libwebp config_enc.c)
        let (sns, filter, sharp, segs) = match self.preset {
            Some(Preset::Default) => (50, 60, 0, 4),
            Some(Preset::Photo) => (80, 30, 3, 4),
            Some(Preset::Drawing) => (25, 10, 6, 4),
            Some(Preset::Icon) => (0, 0, 0, 4),
            Some(Preset::Text) => (0, 0, 0, 2),
            Some(Preset::Picture) => (80, 35, 4, 4),
            Some(Preset::Auto) | None => (50, 60, 0, 4),
        };

        EncoderParams {
            use_predictor_transform: true,
            use_lossy: true,
            lossy_quality: fast_math::roundf(self.quality) as u8,
            method: self.method,
            sns_strength: self.sns_strength.unwrap_or(sns),
            filter_strength: self.filter_strength.unwrap_or(filter),
            filter_sharpness: self.filter_sharpness.unwrap_or(sharp),
            num_segments: self.segments.unwrap_or(segs),
            preset: self.preset.unwrap_or(Preset::Default),
            target_size: self.target_size,
            target_psnr: self.target_psnr,
            use_sharp_yuv: self.sharp_yuv,
            alpha_quality: self.alpha_quality,
        }
    }
}

impl LosslessConfig {
    pub(crate) fn to_params(&self) -> EncoderParams {
        EncoderParams {
            use_predictor_transform: true,
            use_lossy: false,
            lossy_quality: fast_math::roundf(self.quality) as u8,
            method: self.method,
            sns_strength: 0,
            filter_strength: 0,
            filter_sharpness: 0,
            num_segments: 1,
            preset: Preset::Default, // Preset not used for lossless encoding
            target_size: self.target_size,
            target_psnr: 0.0,
            use_sharp_yuv: false,
            alpha_quality: self.alpha_quality,
        }
    }
}

impl EncoderConfig {
    pub(crate) fn to_params(&self) -> EncoderParams {
        match self {
            Self::Lossy(cfg) => cfg.to_params(),
            Self::Lossless(cfg) => cfg.to_params(),
        }
    }

    /// Get the quality value from either variant.
    pub(crate) fn get_quality(&self) -> f32 {
        match self {
            Self::Lossy(cfg) => cfg.quality,
            Self::Lossless(cfg) => cfg.quality,
        }
    }

    /// Get the method value from either variant.
    pub(crate) fn get_method(&self) -> u8 {
        match self {
            Self::Lossy(cfg) => cfg.method,
            Self::Lossless(cfg) => cfg.method,
        }
    }

    /// Get the limits from either variant.
    pub(crate) fn get_limits(&self) -> &Limits {
        match self {
            Self::Lossy(cfg) => &cfg.limits,
            Self::Lossless(cfg) => &cfg.limits,
        }
    }
}
