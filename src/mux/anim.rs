//! Animation encoder.
//!
//! High-level API for encoding animated WebP files frame-by-frame.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::mux::{AnimationEncoder, AnimationConfig};
//! use zenwebp::{EncoderConfig, PixelLayout, LoopCount};
//!
//! let config = AnimationConfig {
//!     background_color: [0, 0, 0, 0],
//!     loop_count: LoopCount::Forever,
//!     ..Default::default()
//! };
//! let mut anim = AnimationEncoder::new(320, 240, config)?;
//!
//! let frame_config = EncoderConfig::new_lossy().with_quality(75.0);
//!
//! // Frame pixels (320x240 RGBA)
//! let pixels = vec![255u8; 320 * 240 * 4];
//! anim.add_frame(&pixels, PixelLayout::Rgba8, 0, &frame_config)?;
//! anim.add_frame(&pixels, PixelLayout::Rgba8, 100, &frame_config)?;
//!
//! let webp = anim.finalize(100)?;
//! # Ok::<(), zenwebp::mux::MuxError>(())
//! ```

use alloc::vec::Vec;

use super::assemble::{MuxFrame, WebPMux};
use super::demux::{BlendMethod, DisposeMethod};
use super::error::MuxError;
use crate::decoder::LoopCount;
use crate::encoder::vp8::encode_frame_lossy;
use crate::encoder::{
    EncoderConfig, EncoderParams, NoProgress, PixelLayout, encode_alpha_lossless,
    encode_frame_lossless,
};
use enough::Unstoppable;

/// Configuration for an animated WebP.
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Background color in BGRA byte order (as per the WebP spec).
    pub background_color: [u8; 4],
    /// Loop count for the animation.
    pub loop_count: LoopCount,
    /// Enable sub-frame delta compression. When `true`, only the changed
    /// region between consecutive frames is encoded, reducing file size for
    /// animations with localized per-frame changes. Default: `true`.
    pub minimize_size: bool,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            background_color: [0, 0, 0, 0],
            loop_count: LoopCount::Forever,
            minimize_size: true,
        }
    }
}

/// Pending frame waiting for the next timestamp to compute its duration.
struct PendingFrame {
    mux_frame: MuxFrame,
    timestamp_ms: u32,
}

/// Animated WebP encoder.
///
/// Encodes individual frames using the existing VP8/VP8L encoder and assembles
/// them into an animated WebP container.
pub struct AnimationEncoder {
    width: u32,
    height: u32,
    mux: WebPMux,
    pending: Option<PendingFrame>,
    minimize_size: bool,
    /// Previous frame's RGBA canvas for delta compression.
    prev_canvas: Option<Vec<u8>>,
}

impl AnimationEncoder {
    /// Create a new animation encoder.
    ///
    /// The canvas dimensions must be between 1 and 16384 (inclusive).
    pub fn new(width: u32, height: u32, config: AnimationConfig) -> Result<Self, MuxError> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(MuxError::InvalidDimensions { width, height });
        }
        let mut mux = WebPMux::new(width, height);
        mux.set_animation(config.background_color, config.loop_count);
        Ok(Self {
            width,
            height,
            mux,
            pending: None,
            minimize_size: config.minimize_size,
            prev_canvas: None,
        })
    }

    /// Flush the previous pending frame and enqueue a new encoded frame.
    #[allow(clippy::too_many_arguments)]
    fn push_encoded_frame(
        &mut self,
        encoded: EncodedFrame,
        frame_width: u32,
        frame_height: u32,
        x_offset: u32,
        y_offset: u32,
        timestamp_ms: u32,
        dispose: DisposeMethod,
        blend: BlendMethod,
    ) -> Result<(), MuxError> {
        // Flush previous pending frame
        if let Some(prev) = self.pending.take() {
            let duration = timestamp_ms.saturating_sub(prev.timestamp_ms);
            let mut frame = prev.mux_frame;
            frame.duration_ms = duration;
            self.mux.push_frame(frame)?;
        }

        let mux_frame = MuxFrame {
            x_offset,
            y_offset,
            width: frame_width,
            height: frame_height,
            duration_ms: 0, // set when next frame arrives or on finalize
            dispose,
            blend,
            bitstream: encoded.bitstream,
            alpha_data: encoded.alpha_data,
            is_lossless: encoded.is_lossless,
        };

        self.pending = Some(PendingFrame {
            mux_frame,
            timestamp_ms,
        });

        Ok(())
    }

    /// Add a frame to the animation.
    ///
    /// Each frame is encoded using the provided [`EncoderConfig`]. The
    /// `timestamp_ms` is the presentation time of this frame; the previous
    /// frame's duration is computed as `this_timestamp - previous_timestamp`.
    ///
    /// The first frame's timestamp is typically 0.
    ///
    /// When [`AnimationConfig::minimize_size`] is enabled (the default), only
    /// the changed region between consecutive frames is encoded, reducing
    /// file size for animations with localized changes. Identical frames
    /// become a tiny 1x1 sub-frame.
    ///
    /// For manual sub-frame control, use
    /// [`add_frame_advanced`](Self::add_frame_advanced).
    pub fn add_frame(
        &mut self,
        pixels: &[u8],
        color_type: PixelLayout,
        timestamp_ms: u32,
        encoder_config: &EncoderConfig,
    ) -> Result<(), MuxError> {
        // If optimization is disabled, or input is YUV420 (can't diff planar
        // data pixel-by-pixel), fall through to full-canvas encoding.
        if !self.minimize_size || color_type == PixelLayout::Yuv420 {
            let result = self.add_frame_advanced(
                pixels,
                color_type,
                self.width,
                self.height,
                0,
                0,
                timestamp_ms,
                encoder_config,
                DisposeMethod::None,
                BlendMethod::Overwrite,
            );
            // Invalidate canvas tracking for YUV420 (can't convert)
            if color_type == PixelLayout::Yuv420 {
                self.prev_canvas = None;
            }
            return result;
        }

        let rgba = to_rgba(pixels, color_type, self.width, self.height)
            .expect("non-YUV420 color types always convert to RGBA");

        let params = encoder_config.to_params();

        if let Some(ref prev) = self.prev_canvas {
            // Delta frame: find changed region
            match find_diff_rect(prev, &rgba, self.width, self.height) {
                None => {
                    // Identical frame — encode a tiny 1x1 sub-frame
                    let one_pixel: &[u8] = if color_type.has_alpha() {
                        &rgba[..4]
                    } else {
                        &rgba[..3]
                    };
                    let sub_color = if color_type.has_alpha() {
                        PixelLayout::Rgba8
                    } else {
                        PixelLayout::Rgb8
                    };
                    let encoded = encode_frame_data(one_pixel, 1, 1, sub_color, &params)?;
                    self.push_encoded_frame(
                        encoded,
                        1,
                        1,
                        0,
                        0,
                        timestamp_ms,
                        DisposeMethod::None,
                        BlendMethod::Overwrite,
                    )?;
                }
                Some((x, y, w, h)) => {
                    let include_alpha = color_type.has_alpha();
                    let sub_pixels = extract_sub_rect(&rgba, self.width, x, y, w, h, include_alpha);
                    let sub_color = if include_alpha {
                        PixelLayout::Rgba8
                    } else {
                        PixelLayout::Rgb8
                    };
                    let encoded = encode_frame_data(&sub_pixels, w, h, sub_color, &params)?;
                    self.push_encoded_frame(
                        encoded,
                        w,
                        h,
                        x,
                        y,
                        timestamp_ms,
                        DisposeMethod::None,
                        BlendMethod::Overwrite,
                    )?;
                }
            }
        } else {
            // First frame or after canvas invalidation — encode full canvas
            let encoded = encode_frame_data(pixels, self.width, self.height, color_type, &params)?;
            self.push_encoded_frame(
                encoded,
                self.width,
                self.height,
                0,
                0,
                timestamp_ms,
                DisposeMethod::None,
                BlendMethod::Overwrite,
            )?;
        }

        self.prev_canvas = Some(rgba);
        Ok(())
    }

    /// Add a frame with full control over placement, dispose, and blend.
    ///
    /// Frame offsets must be even. The frame region must fit within the canvas.
    ///
    /// Using this method invalidates canvas tracking for sub-frame
    /// optimization. The next [`add_frame`](Self::add_frame) call will encode
    /// a full keyframe.
    #[allow(clippy::too_many_arguments)]
    pub fn add_frame_advanced(
        &mut self,
        pixels: &[u8],
        color_type: PixelLayout,
        frame_width: u32,
        frame_height: u32,
        x_offset: u32,
        y_offset: u32,
        timestamp_ms: u32,
        encoder_config: &EncoderConfig,
        dispose: DisposeMethod,
        blend: BlendMethod,
    ) -> Result<(), MuxError> {
        let params = encoder_config.to_params();
        let encoded = encode_frame_data(pixels, frame_width, frame_height, color_type, &params)?;
        self.push_encoded_frame(
            encoded,
            frame_width,
            frame_height,
            x_offset,
            y_offset,
            timestamp_ms,
            dispose,
            blend,
        )?;
        // Invalidate canvas tracking — user controls sub-framing manually
        self.prev_canvas = None;
        Ok(())
    }

    /// Finalize the animation and return the assembled WebP bytes.
    ///
    /// `last_frame_duration_ms` is the display duration for the final frame,
    /// since there is no subsequent timestamp to derive it from.
    pub fn finalize(mut self, last_frame_duration_ms: u32) -> Result<Vec<u8>, MuxError> {
        // Flush the last pending frame
        if let Some(prev) = self.pending.take() {
            let mut frame = prev.mux_frame;
            frame.duration_ms = last_frame_duration_ms;
            self.mux.push_frame(frame)?;
        }

        self.mux.assemble()
    }

    /// Set ICC profile on the output.
    pub fn icc_profile(&mut self, data: Vec<u8>) {
        self.mux.set_icc_profile(data);
    }

    /// Set EXIF metadata on the output.
    pub fn exif(&mut self, data: Vec<u8>) {
        self.mux.set_exif(data);
    }

    /// Set XMP metadata on the output.
    pub fn xmp(&mut self, data: Vec<u8>) {
        self.mux.set_xmp(data);
    }
}

/// Convert pixel data to RGBA. Returns `None` for YUV420 (planar data
/// can't be compared pixel-by-pixel for delta compression).
fn to_rgba(pixels: &[u8], color_type: PixelLayout, width: u32, height: u32) -> Option<Vec<u8>> {
    let npixels = (width as usize) * (height as usize);
    match color_type {
        PixelLayout::L8 => Some(
            pixels[..npixels]
                .iter()
                .flat_map(|&p| [p, p, p, 255])
                .collect(),
        ),
        PixelLayout::La8 => Some(
            pixels[..npixels * 2]
                .chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
        ),
        PixelLayout::Rgb8 => Some(
            pixels[..npixels * 3]
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        ),
        PixelLayout::Rgba8 => Some(pixels[..npixels * 4].to_vec()),
        PixelLayout::Bgr8 => Some(
            pixels[..npixels * 3]
                .chunks_exact(3)
                .flat_map(|p| [p[2], p[1], p[0], 255])
                .collect(),
        ),
        PixelLayout::Bgra8 => Some(
            pixels[..npixels * 4]
                .chunks_exact(4)
                .flat_map(|p| [p[2], p[1], p[0], p[3]])
                .collect(),
        ),
        PixelLayout::Yuv420 => None,
    }
}

/// Find the minimal bounding rectangle of pixels that differ between two
/// RGBA canvases. Returns `(x, y, w, h)` snapped to even coordinates, or
/// `None` if the frames are identical.
fn find_diff_rect(
    prev: &[u8],
    curr: &[u8],
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let w = width as usize;
    let h = height as usize;
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut min_y = h;
    let mut max_y = 0usize;

    for y in 0..h {
        let row_off = y * w * 4;
        for x in 0..w {
            let off = row_off + x * 4;
            if prev[off..off + 4] != curr[off..off + 4] {
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }

    if min_x > max_x {
        // No differences found
        return None;
    }

    // Snap to even coordinates (WebP requires even offsets)
    let x0 = (min_x as u32) & !1;
    let y0 = (min_y as u32) & !1;
    // End coordinate, rounded up to even
    let x1 = ((max_x as u32 + 2) & !1).min(width);
    let y1 = ((max_y as u32 + 2) & !1).min(height);

    Some((x0, y0, x1 - x0, y1 - y0))
}

/// Extract a sub-rectangle from an RGBA canvas. When `include_alpha` is
/// `false`, the alpha channel is stripped and the output is RGB.
fn extract_sub_rect(
    rgba: &[u8],
    canvas_width: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    include_alpha: bool,
) -> Vec<u8> {
    let cw = canvas_width as usize;
    let bpp = if include_alpha { 4 } else { 3 };
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * bpp);
    for row in 0..h as usize {
        let src_y = y as usize + row;
        let src_off = (src_y * cw + x as usize) * 4;
        for col in 0..w as usize {
            let off = src_off + col * 4;
            out.push(rgba[off]);
            out.push(rgba[off + 1]);
            out.push(rgba[off + 2]);
            if include_alpha {
                out.push(rgba[off + 3]);
            }
        }
    }
    out
}

/// Encoded frame data: bitstream, optional alpha data, and whether it's lossless.
struct EncodedFrame {
    bitstream: Vec<u8>,
    alpha_data: Option<Vec<u8>>,
    is_lossless: bool,
}

/// Encode a single frame to raw bitstream data (no RIFF container).
fn encode_frame_data(
    pixels: &[u8],
    width: u32,
    height: u32,
    color: PixelLayout,
    params: &EncoderParams,
) -> Result<EncodedFrame, MuxError> {
    let mut bitstream = Vec::new();
    let lossy_with_alpha = params.use_lossy && color.has_alpha();

    if params.use_lossy {
        encode_frame_lossy(
            &mut bitstream,
            pixels,
            width,
            height,
            color,
            params,
            &Unstoppable,
            &NoProgress,
        )?;

        let alpha_data = if lossy_with_alpha {
            let mut alpha = Vec::new();
            encode_alpha_lossless(
                &mut alpha,
                pixels,
                width,
                height,
                color,
                params.alpha_quality,
                &Unstoppable,
            )?;
            Some(alpha)
        } else {
            None
        };

        Ok(EncodedFrame {
            bitstream,
            alpha_data,
            is_lossless: false,
        })
    } else {
        encode_frame_lossless(
            &mut bitstream,
            pixels,
            width,
            height,
            color,
            params.clone(),
            false,
            &Unstoppable,
        )?;
        Ok(EncodedFrame {
            bitstream,
            alpha_data: None,
            is_lossless: true,
        })
    }
}
