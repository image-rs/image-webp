//! Animation decoder.
//!
//! High-level API for decoding animated WebP files frame-by-frame.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::AnimationDecoder;
//!
//! let webp_data: &[u8] = &[]; // your animated WebP data
//! let mut decoder = AnimationDecoder::new(webp_data)?;
//! let info = decoder.info();
//! println!("{}x{}, {} frames", info.canvas_width, info.canvas_height, info.frame_count);
//!
//! while let Some(frame) = decoder.next_frame()? {
//!     println!("frame at {}ms, duration {}ms", frame.timestamp_ms, frame.duration_ms);
//! }
//! # Ok::<(), zenwebp::DecodeError>(())
//! ```

use alloc::vec;
use alloc::vec::Vec;

use crate::decoder::{DecodeConfig, DecodeError, LoopCount, UpsamplingMethod, WebPDecoder};

/// Frame metadata without pixel data (returned by [`AnimationDecoder::decode_next`]).
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Cumulative presentation timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// Display duration of this frame in milliseconds.
    pub duration_ms: u32,
}

/// A decoded animation frame with owned RGBA pixel data.
#[derive(Debug, Clone)]
pub struct AnimFrame {
    /// RGBA pixel data (canvas_width * canvas_height * 4 bytes).
    pub data: Vec<u8>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Cumulative presentation timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// Display duration of this frame in milliseconds.
    pub duration_ms: u32,
}

/// Metadata about an animated WebP image.
#[derive(Debug, Clone)]
pub struct AnimationInfo {
    /// Canvas width in pixels.
    pub canvas_width: u32,
    /// Canvas height in pixels.
    pub canvas_height: u32,
    /// Total number of frames.
    pub frame_count: u32,
    /// Loop count for the animation.
    pub loop_count: LoopCount,
    /// Background color hint (BGRA byte order, per WebP spec).
    pub background_color: [u8; 4],
    /// Whether the image has alpha.
    pub has_alpha: bool,
}

/// High-level animated WebP decoder.
///
/// Wraps [`WebPDecoder`] and provides an iterator-like interface over frames,
/// returning owned RGBA snapshots of the composited canvas at each frame.
pub struct AnimationDecoder<'a> {
    decoder: WebPDecoder<'a>,
    buf: Vec<u8>,
    cumulative_ms: u32,
    frames_read: u32,
    total_frames: u32,
    stop: Option<&'a dyn enough::Stop>,
}

impl<'a> AnimationDecoder<'a> {
    /// Create a new animation decoder from WebP data.
    ///
    /// Returns an error if the data is not a valid animated WebP.
    pub fn new(data: &'a [u8]) -> Result<Self, DecodeError> {
        Self::new_with_config(data, &DecodeConfig::default())
    }

    /// Create a new animation decoder with custom decode configuration.
    ///
    /// Returns an error if the data is not a valid animated WebP.
    pub fn new_with_config(data: &'a [u8], config: &DecodeConfig) -> Result<Self, DecodeError> {
        let mut decoder = WebPDecoder::new_with_options(data, config.to_options())?;
        decoder.set_limits(config.limits.clone());
        if !decoder.is_animated() {
            return Err(DecodeError::InvalidParameter(alloc::string::String::from(
                "not an animated WebP",
            )));
        }
        let total_frames = decoder.num_frames();
        let buf_size = decoder
            .output_buffer_size()
            .ok_or(DecodeError::ImageTooLarge)?;
        let buf = vec![0u8; buf_size];
        Ok(Self {
            decoder,
            buf,
            cumulative_ms: 0,
            frames_read: 0,
            total_frames,
            stop: None,
        })
    }

    /// Get metadata about the animation.
    pub fn info(&self) -> AnimationInfo {
        let (w, h) = self.decoder.dimensions();
        AnimationInfo {
            canvas_width: w,
            canvas_height: h,
            frame_count: self.total_frames,
            loop_count: self.decoder.loop_count(),
            background_color: self.decoder.background_color_hint().unwrap_or([0, 0, 0, 0]),
            has_alpha: self.decoder.has_alpha(),
        }
    }

    /// Returns `true` if there are more frames to decode.
    pub fn has_more_frames(&self) -> bool {
        self.frames_read < self.total_frames
    }

    /// Returns the number of frames decoded so far.
    pub fn frames_read(&self) -> u32 {
        self.frames_read
    }

    /// Set memory limit for the underlying decoder.
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.decoder.set_memory_limit(limit);
    }

    /// Set background color for compositing.
    pub fn set_background_color(&mut self, color: [u8; 4]) -> Result<(), DecodeError> {
        self.decoder.set_background_color(color)
    }

    /// Set a cooperative cancellation token.
    pub fn set_stop(&mut self, stop: &'a dyn enough::Stop) {
        self.stop = Some(stop);
        self.decoder.set_stop(Some(stop));
    }

    /// Set upsampling method for lossy frames.
    pub fn set_lossy_upsampling(&mut self, method: UpsamplingMethod) {
        self.decoder.set_lossy_upsampling(method);
    }

    /// Read the embedded ICC profile, if any.
    pub fn icc_profile(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.decoder.icc_profile()
    }

    /// Read the embedded EXIF metadata, if any.
    pub fn exif_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.decoder.exif_metadata()
    }

    /// Read the embedded XMP metadata, if any.
    pub fn xmp_metadata(&mut self) -> Result<Option<Vec<u8>>, DecodeError> {
        self.decoder.xmp_metadata()
    }

    /// Total duration of all frames in milliseconds.
    ///
    /// Only accurate after all frames have been read, or from the container
    /// header if available.
    pub fn loop_duration(&self) -> u64 {
        self.decoder.loop_duration()
    }

    /// Decode the next frame, returning `None` when all frames have been read.
    pub fn next_frame(&mut self) -> Result<Option<AnimFrame>, DecodeError> {
        let info = match self.decode_next()? {
            Some(info) => info,
            None => return Ok(None),
        };
        Ok(Some(AnimFrame {
            data: self.buf.clone(),
            width: info.width,
            height: info.height,
            timestamp_ms: info.timestamp_ms,
            duration_ms: info.duration_ms,
        }))
    }

    /// Decode the next frame without allocating.
    ///
    /// Returns frame metadata on success. The composited pixel data is
    /// available via [`current_frame_data()`](Self::current_frame_data)
    /// until the next call to `decode_next` or `next_frame`.
    pub fn decode_next(&mut self) -> Result<Option<FrameInfo>, DecodeError> {
        if let Some(stop) = self.stop {
            stop.check()?;
        }
        match self.decoder.read_frame(&mut self.buf) {
            Ok(duration_ms) => {
                let timestamp_ms = self.cumulative_ms;
                self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
                self.frames_read += 1;
                let (w, h) = self.decoder.dimensions();
                Ok(Some(FrameInfo {
                    width: w,
                    height: h,
                    timestamp_ms,
                    duration_ms,
                }))
            }
            Err(DecodeError::NoMoreFrames) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Borrow the current composited frame data.
    ///
    /// Valid after a successful [`decode_next()`](Self::decode_next) call.
    /// The format is RGBA (4 bytes/pixel) if `has_alpha`, otherwise RGB
    /// (3 bytes/pixel), matching [`output_buffer_size`](WebPDecoder::output_buffer_size).
    #[inline]
    pub fn current_frame_data(&self) -> &[u8] {
        &self.buf
    }

    /// Reset the decoder to the first frame.
    pub fn reset(&mut self) -> Result<(), DecodeError> {
        self.decoder.reset_animation()?;
        self.cumulative_ms = 0;
        self.frames_read = 0;
        Ok(())
    }

    /// Decode all remaining frames at once.
    pub fn decode_all(&mut self) -> Result<Vec<AnimFrame>, DecodeError> {
        self.reset()?;
        let mut frames = Vec::with_capacity(self.total_frames as usize);
        while let Some(frame) = self.next_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}

impl Iterator for AnimationDecoder<'_> {
    type Item = Result<AnimFrame, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
