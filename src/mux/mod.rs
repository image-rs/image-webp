//! WebP mux/demux and animation encoding.
//!
//! This module provides three capabilities:
//!
//! - **Demux** ([`WebPDemuxer`]): Parse WebP files at the chunk level, iterate
//!   frames and access raw bitstream data without decoding pixels.
//! - **Mux** ([`WebPMux`]): Assemble WebP containers from pre-encoded chunks
//!   and metadata (ICC, EXIF, XMP).
//! - **Animation** ([`AnimationEncoder`]): Encode animated WebP files
//!   frame-by-frame using the existing VP8/VP8L encoder.
//!
//! All types work in `no_std + alloc` environments.

mod anim;
mod anim_decode;
mod assemble;
mod demux;
mod error;

pub use anim::{AnimationConfig, AnimationEncoder};
pub use anim_decode::{AnimFrame, AnimationDecoder, AnimationInfo, FrameInfo};
pub use assemble::{MuxFrame, WebPMux};
pub use demux::{BlendMethod, DemuxFrame, DemuxFrameIter, DisposeMethod, WebPDemuxer};
pub use error::{MuxError, MuxResult};
