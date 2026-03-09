//! Decoding and Encoding of WebP Images
//!
//! Copyright (C) 2025 Imazen LLC
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Affero General Public License as published
//! by the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! For commercial licensing inquiries: support@imazen.io
//!
//! This crate provides both encoding and decoding of WebP images.
//!
//! # Features
//!
//! - `std` (default): Enables `encode_to_writer()`. Everything else works without it.
//! - `simd` (default): SIMD optimizations (SSE2/SSE4.1/AVX2 on x86, SIMD128 on WASM).
//! - `fast-yuv` (default): Optimized YUV conversion via the `yuv` crate.
//! - `pixel-types`: Type-safe pixel formats via the `rgb` crate.
//!
//! # no_std Support
//!
//! Both encoding and decoding work in `no_std` environments (requires `alloc`):
//! ```toml
//! [dependencies]
//! zenwebp = { version = "...", default-features = false }
//! ```
//!
//! Only [`EncodeRequest::encode_to`] requires `std` (for `std::io::Write`).
//!
//! # Encoding
//!
//! Use [`LossyConfig`] or [`LosslessConfig`] with [`EncodeRequest`]:
//!
//! ```rust
//! use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};
//!
//! let config = LossyConfig::new().with_quality(85.0).with_method(4);
//! let rgba_data = vec![255u8; 4 * 4 * 4];
//! let webp = EncodeRequest::lossy(&config, &rgba_data, PixelLayout::Rgba8, 4, 4)
//!     .encode()?;
//! # Ok::<(), whereat::At<zenwebp::EncodeError>>(())
//! ```
//!
//! # Decoding
//!
//! Use the convenience functions:
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::decode_rgba(webp_data)?;
//! # Ok::<(), whereat::At<zenwebp::DecodeError>>(())
//! ```
//!
//! Or [`WebPDecoder`] for two-phase decoding (inspect headers before allocating):
//!
//! ```rust,no_run
//! use zenwebp::WebPDecoder;
//!
//! let webp_data: &[u8] = &[]; // your WebP data
//! let mut decoder = WebPDecoder::build(webp_data)?;
//! let info = decoder.info();
//! println!("{}x{}, alpha={}", info.width, info.height, info.has_alpha);
//!
//! let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
//! decoder.read_image(&mut output)?;
//! # Ok::<(), zenwebp::DecodeError>(())
//! ```
//!
//! # Safety
//!
//! This crate uses `#![forbid(unsafe_code)]` to prevent direct unsafe usage in source.
//! However, when the `simd` feature is enabled, we rely on the [`archmage`] crate for
//! safe SIMD intrinsics. The `#[arcane]` proc macro generates unsafe blocks internally
//! (which bypass the `forbid` lint due to proc-macro span handling). The soundness of
//! our SIMD code depends on archmage's token-based safety model being correct.
//!
//! Without the `simd` feature, this crate contains no unsafe code whatsoever.
//!
//! [`archmage`]: https://docs.rs/archmage

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]

extern crate alloc;

whereat::define_at_crate_info!();

#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

/// Conditionally applies `#[autoversion]` and adds a `SimdToken` parameter
/// when the `simd` feature is enabled, otherwise defines the function as-is.
macro_rules! maybe_autoversion {
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    ) => {
        #[cfg(feature = "simd")]
        #[archmage::autoversion]
        $(#[$attr])*
        $vis fn $name(_token: archmage::SimdToken, $($arg : $ty),*) $(-> $ret)? $body

        #[cfg(not(feature = "simd"))]
        $(#[$attr])*
        $vis fn $name($($arg : $ty),*) $(-> $ret)? $body
    };
    (
        $(#[$attr:meta])*
        $vis:vis fn $name:ident<const $C:ident : $CT:ty>($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block
    ) => {
        #[cfg(feature = "simd")]
        #[archmage::autoversion]
        $(#[$attr])*
        $vis fn $name<const $C: $CT>(_token: archmage::SimdToken, $($arg : $ty),*) $(-> $ret)? $body

        #[cfg(not(feature = "simd"))]
        $(#[$attr])*
        $vis fn $name<const $C: $CT>($($arg : $ty),*) $(-> $ret)? $body
    };
}

// Core modules
pub mod common;
pub mod decoder;
/// Encoder detection and quality estimation from WebP file headers.
pub mod detect;
pub mod encoder;
/// WebP mux/demux and animation encoding.
pub mod mux;

// Slice reader utility (used by decoder and mux)
mod slice_reader;

/// Type-safe pixel format traits for decoding and encoding.
#[cfg(feature = "pixel-types")]
pub mod pixel;

/// Resource estimation heuristics for encoding and decoding operations.
pub mod heuristics;

// Re-export decoder public API
pub use decoder::{
    BitstreamFormat, DecodeConfig, DecodeError, DecodeRequest, DecodeResult, ImageInfo, Limits,
    LoopCount, StreamStatus, StreamingDecoder, UpsamplingMethod, WebPDecoder, YuvPlanes,
    decode_bgr, decode_bgr_into, decode_bgra, decode_bgra_into, decode_rgb, decode_rgb_into,
    decode_rgba, decode_rgba_into, decode_yuv420,
};

// Re-export encoder public API
pub use encoder::{
    ClassifierDiag, ContentType, EncodeError, EncodeProgress, EncodeRequest, EncodeResult,
    EncodeStats, EncoderConfig, ImageMetadata, LosslessConfig, LossyConfig, NoProgress,
    PixelLayout, Preset,
};

// Re-export mux/demux public API
pub use mux::{
    AnimFrame, AnimationConfig, AnimationDecoder, AnimationEncoder, AnimationInfo, BlendMethod,
    DemuxFrame, DisposeMethod, MuxError, MuxFrame, MuxResult, WebPDemuxer, WebPMux,
};

// Re-export cooperative cancellation types
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export VP8 decoder (public module)
pub use decoder::vp8;

#[cfg(feature = "zencodec")]
mod zencodec;
#[cfg(feature = "zencodec")]
pub use zencodec::{
    WebpDecodeJob, WebpDecoder, WebpDecoderConfig, WebpEncodeJob, WebpEncoder, WebpEncoderConfig,
    WebpFullFrameDecoder, WebpFullFrameEncoder,
};

/// Standalone metadata convenience functions for already-encoded WebP data.
///
/// For embedding metadata during encoding, use
/// [`EncodeRequest::with_metadata`] instead.
pub mod metadata;
