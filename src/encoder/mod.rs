//! WebP encoder implementation.
//!
//! This module provides lossy (VP8) and lossless (VP8L) WebP encoding with
//! perceptual optimizations for improved rate-quality trade-offs.
//!
//! # Module Organization
//!
//! **Core encoding:**
//! - [`vp8`]: Lossy VP8 encoding (DCT-based, like JPEG)
//! - [`vp8l`]: Lossless VP8L encoding (LZ77 + entropy coding)
//!
//! **Rate-distortion optimization:**
//! - [`analysis`]: Segment-based adaptive quantization (k-means clustering)
//! - [`cost`]: RD cost estimation for mode selection
//! - trellis: Trellis quantization for optimal coefficient selection
//!
//! **Perceptual models:**
//! - psy: CSF weighting, JND thresholds, masking-based AQ
//!
//! **Low-level utilities:**
//! - [`quantize`]: Quantization matrices and coefficient quantization
//! - [`tables`]: Lookup tables (entropy costs, zigzag order, etc.)
//! - arithmetic: Arithmetic/range coding
//! - residual_cost: SIMD-optimized residual cost estimation

/// Image analysis and auto-detection
pub mod analysis;
mod api;
mod arithmetic;
/// Type-safe encoder configuration
pub mod config;
pub mod cost;
pub(crate) mod fast_math;
/// Perceptual distortion model (CSF tables, psy-rd, psy-trellis)
pub(crate) mod psy;
/// Quantization matrix and coefficient quantization
pub mod quantize;
/// Residual cost estimation (SIMD-optimized)
mod residual_cost;
pub mod tables;
/// Trellis quantization for RD-optimized coefficient selection
mod trellis;
mod vec_writer;
/// VP8 encoder implementation
pub mod vp8;
/// VP8L (lossless) encoder implementation
pub mod vp8l;

// Re-export public API
pub use analysis::{ClassifierDiag, ContentType};
pub(crate) use api::EncoderParams;
pub use api::{
    EncodeError, EncodeProgress, EncodeRequest, EncodeResult, EncodeStats, ImageMetadata,
    NoProgress, PixelLayout, Preset,
};
pub use config::{EncoderConfig, LosslessConfig, LossyConfig};
pub use vp8l::{Vp8lConfig, Vp8lQuality, encode_vp8l};

// Crate-internal re-exports for mux module
pub(crate) use api::{chunk_size, encode_alpha_lossless, encode_frame_lossless, write_chunk};
pub(crate) use vec_writer::VecWriter;
