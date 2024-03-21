//! Decoding and Encoding of WebP Images

#![forbid(unsafe_code)]
#![deny(missing_docs)]

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
