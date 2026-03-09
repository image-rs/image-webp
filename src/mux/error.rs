//! Error types for mux/demux operations.

use alloc::string::String;
use thiserror::Error;

use crate::decoder::DecodeError;
use crate::encoder::EncodeError;

/// Result type alias using `At<MuxError>` for automatic location tracking.
pub type MuxResult<T> = core::result::Result<T, whereat::At<MuxError>>;

/// Errors that can occur during mux/demux operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MuxError {
    /// The data is not a valid WebP file.
    #[error("Invalid WebP format: {0}")]
    InvalidFormat(String),

    /// Frame dimensions are invalid (zero, too large, or don't fit the canvas).
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// The invalid width.
        width: u32,
        /// The invalid height.
        height: u32,
    },

    /// A frame index is out of bounds.
    #[error("Frame {index} out of bounds (total: {total})")]
    FrameOutOfBounds {
        /// The requested frame index.
        index: u32,
        /// The total number of frames.
        total: u32,
    },

    /// An error occurred during encoding.
    #[error("Encoding error: {0}")]
    EncodeError(#[from] EncodeError),

    /// An error occurred during decoding/parsing.
    #[error("Decoding error: {0}")]
    DecodeError(#[from] DecodeError),

    /// No frames were added before assembly.
    #[error("No frames to assemble")]
    NoFrames,

    /// Frame offset is not a multiple of 2 (WebP spec requirement).
    #[error("Frame offset must be even: ({x}, {y})")]
    OddFrameOffset {
        /// The invalid x offset.
        x: u32,
        /// The invalid y offset.
        y: u32,
    },

    /// Frame extends beyond the canvas boundary.
    #[error(
        "Frame at ({x}, {y}) size {width}x{height} exceeds canvas {canvas_width}x{canvas_height}"
    )]
    FrameOutsideCanvas {
        /// Frame x offset.
        x: u32,
        /// Frame y offset.
        y: u32,
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Canvas width.
        canvas_width: u32,
        /// Canvas height.
        canvas_height: u32,
    },
}
