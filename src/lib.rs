//! Decoding and Encoding of WebP Images

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.
#![recursion_limit = "256"]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]
#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

pub use self::decoder::{
    DecodingError, LoopCount, UpsamplingMethod, WebPDecodeOptions, WebPDecoder,
};
pub use self::encoder::{ColorType, EncoderParams, EncodingError, WebPEncoder};

mod alpha_blending;
mod decoder;
mod encoder;
mod extended;
mod lossless;
mod lossy;

/// An implementation of the VP8 Video Codec
///
/// This module contains a partial implementation of the
/// VP8 video format as defined in RFC-6386.
///
/// It decodes Keyframes only.
/// VP8 is the underpinning of the WebP image format
///
/// # Related Links
/// * [rfc-6386](http://tools.ietf.org/html/rfc6386) - The VP8 Data Format and Decoding Guide
/// * [VP8.pdf](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37073.pdf) - An overview of of the VP8 format
pub mod vp8 {
    /// Representation of a decoded video frame
    #[deprecated = "This is planned for removal; please open an issue if you are relying on it"]
    pub type Frame = crate::lossy::Frame;

    /// VP8 Decoder that only decodes keyframes
    #[deprecated = "This is planned for removal; please open an issue if you are relying on it"]
    pub type Vp8Decoder<R> = crate::lossy::Vp8Decoder<R>;
}
