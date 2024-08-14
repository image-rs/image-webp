//! Decoding and Encoding of WebP Images

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.
#![recursion_limit = "256"]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]
#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

pub use self::decoder::{DecodingError, LoopCount, WebPDecoder};
pub use self::encoder::{ColorType, EncodingError, WebPEncoder};

mod decoder;
mod encoder;
mod extended;
mod huffman;
mod loop_filter;
mod lossless;
mod lossless_transform;
mod transform;

pub mod vp8;
