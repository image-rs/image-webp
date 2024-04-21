//! Decoding and Encoding of WebP Images

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.
#![recursion_limit = "256"]

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
