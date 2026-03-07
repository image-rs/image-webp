//! WebP decoder implementation

mod alpha_blending;
mod api;
pub(crate) mod arithmetic;
mod bit_reader;
mod extended;
mod huffman;
mod limits;
mod loop_filter;
mod lossless;
mod lossless_transform;
mod streaming;
pub(crate) mod yuv;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod loop_filter_avx2;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod loop_filter_neon;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
mod loop_filter_wasm;

mod loop_filter_dispatch;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod yuv_simd;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod yuv_neon;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
mod yuv_wasm;

// Public VP8 decoder module
pub mod vp8;

// Re-export public API
pub use api::{
    BitstreamFormat, DecodeConfig, DecodeError, DecodeRequest, DecodeResult, ImageInfo, LoopCount,
    UpsamplingMethod, WebPDecoder, YuvPlanes, decode_bgr, decode_bgr_into, decode_bgra,
    decode_bgra_into, decode_rgb, decode_rgb_into, decode_rgba, decode_rgba_into, decode_yuv420,
};
#[allow(deprecated)]
pub use limits::Limits;
pub use streaming::{StreamStatus, StreamingDecoder};

// Re-export diagnostic types for tests (hidden from public docs)
#[doc(hidden)]
pub use vp8::{BlockDiagnostic, DiagnosticFrame, MacroblockDiagnostic, TreeNode};

// Re-export common types used in diagnostics
#[doc(hidden)]
pub use crate::common::types::{ChromaMode, IntraMode, LumaMode};
