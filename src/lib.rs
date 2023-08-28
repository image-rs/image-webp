//! Decoding and Encoding of WebP Images

#![forbid(unsafe_code)]

pub use self::decoder::WebPDecoder;

mod decoder;
mod extended;
mod huffman;
mod loop_filter;
mod lossless;
mod lossless_transform;
mod transform;

pub mod vp8;
