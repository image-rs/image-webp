//! WebP container assembler.
//!
//! Assembles a valid WebP file from pre-encoded bitstream chunks and metadata.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::mux::{WebPMux, MuxFrame, BlendMethod, DisposeMethod};
//! use zenwebp::LoopCount;
//!
//! let mut mux = WebPMux::new(320, 240);
//! mux.set_animation([0, 0, 0, 0], LoopCount::Forever);
//!
//! // Add pre-encoded frames
//! mux.push_frame(MuxFrame {
//!     x_offset: 0,
//!     y_offset: 0,
//!     width: 320,
//!     height: 240,
//!     duration_ms: 100,
//!     dispose: DisposeMethod::Background,
//!     blend: BlendMethod::Overwrite,
//!     bitstream: vec![], // VP8L data here
//!     alpha_data: None,
//!     is_lossless: true,
//! })?;
//!
//! let webp_bytes = mux.assemble()?;
//! # Ok::<(), zenwebp::mux::MuxError>(())
//! ```

use alloc::vec::Vec;

use super::demux::{BlendMethod, DisposeMethod, WebPDemuxer};
use super::error::MuxError;
use crate::decoder::LoopCount;
use crate::encoder::{VecWriter, chunk_size, write_chunk};

/// A single frame to be muxed into a WebP container.
#[derive(Debug, Clone)]
pub struct MuxFrame {
    /// Horizontal offset on the canvas. Must be even.
    pub x_offset: u32,
    /// Vertical offset on the canvas. Must be even.
    pub y_offset: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame duration in milliseconds (max 16777215).
    pub duration_ms: u32,
    /// How the frame area is disposed after rendering.
    pub dispose: DisposeMethod,
    /// How the frame is blended onto the canvas.
    pub blend: BlendMethod,
    /// Raw VP8 or VP8L bitstream data.
    pub bitstream: Vec<u8>,
    /// Raw ALPH chunk payload (for lossy frames with separate alpha).
    pub alpha_data: Option<Vec<u8>>,
    /// Whether the bitstream is VP8L (lossless). `false` means VP8 (lossy).
    pub is_lossless: bool,
}

/// WebP container assembler.
///
/// Builds a complete WebP file from pre-encoded frame data and optional metadata.
/// Supports both single-image and animated WebP output.
pub struct WebPMux {
    canvas_width: u32,
    canvas_height: u32,
    // Animation parameters (None = not animated)
    animation: Option<AnimationParams>,
    // Frames (only used for animated)
    frames: Vec<MuxFrame>,
    // Single image (only used for non-animated)
    single_image: Option<MuxFrame>,
    // Metadata
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct AnimationParams {
    background_color: [u8; 4],
    loop_count: LoopCount,
}

impl WebPMux {
    /// Create a new mux assembler with the given canvas dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            canvas_width: width,
            canvas_height: height,
            animation: None,
            frames: Vec::new(),
            single_image: None,
            icc_profile: None,
            exif: None,
            xmp: None,
        }
    }

    /// Parse an existing WebP file into a mux assembler.
    ///
    /// This allows modifying an existing file (e.g., adding metadata, replacing frames).
    pub fn from_data(data: &[u8]) -> Result<Self, MuxError> {
        let demuxer = WebPDemuxer::new(data)?;
        let mut mux = Self::new(demuxer.canvas_width(), demuxer.canvas_height());

        if let Some(icc) = demuxer.icc_profile() {
            mux.icc_profile = Some(icc.to_vec());
        }
        if let Some(exif) = demuxer.exif() {
            mux.exif = Some(exif.to_vec());
        }
        if let Some(xmp) = demuxer.xmp() {
            mux.xmp = Some(xmp.to_vec());
        }

        if demuxer.is_animated() {
            mux.animation = Some(AnimationParams {
                background_color: demuxer.background_color(),
                loop_count: demuxer.loop_count(),
            });

            for frame in demuxer.frames() {
                mux.frames.push(MuxFrame {
                    x_offset: frame.x_offset,
                    y_offset: frame.y_offset,
                    width: frame.width,
                    height: frame.height,
                    duration_ms: frame.duration_ms,
                    dispose: frame.dispose,
                    blend: frame.blend,
                    bitstream: frame.bitstream.to_vec(),
                    alpha_data: frame.alpha_data.map(|d| d.to_vec()),
                    is_lossless: !frame.is_lossy,
                });
            }
        } else if let Some(frame) = demuxer.frame(1) {
            mux.single_image = Some(MuxFrame {
                x_offset: 0,
                y_offset: 0,
                width: frame.width,
                height: frame.height,
                duration_ms: 0,
                dispose: DisposeMethod::None,
                blend: BlendMethod::Overwrite,
                bitstream: frame.bitstream.to_vec(),
                alpha_data: frame.alpha_data.map(|d| d.to_vec()),
                is_lossless: !frame.is_lossy,
            });
        }

        Ok(mux)
    }

    /// Set ICC profile metadata.
    pub fn set_icc_profile(&mut self, data: Vec<u8>) {
        self.icc_profile = Some(data);
    }

    /// Get ICC profile metadata.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Set EXIF metadata.
    pub fn set_exif(&mut self, data: Vec<u8>) {
        self.exif = Some(data);
    }

    /// Get EXIF metadata.
    pub fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// Set XMP metadata.
    pub fn set_xmp(&mut self, data: Vec<u8>) {
        self.xmp = Some(data);
    }

    /// Get XMP metadata.
    pub fn xmp(&self) -> Option<&[u8]> {
        self.xmp.as_deref()
    }

    /// Remove ICC profile metadata.
    pub fn clear_icc_profile(&mut self) {
        self.icc_profile = None;
    }

    /// Remove EXIF metadata.
    pub fn clear_exif(&mut self) {
        self.exif = None;
    }

    /// Remove XMP metadata.
    pub fn clear_xmp(&mut self) {
        self.xmp = None;
    }

    /// Configure this mux for animation output.
    ///
    /// The `background_color` is in BGRA byte order as per the WebP spec.
    pub fn set_animation(&mut self, background_color: [u8; 4], loop_count: LoopCount) {
        self.animation = Some(AnimationParams {
            background_color,
            loop_count,
        });
    }

    /// Add a frame to the animation.
    ///
    /// Frame offsets must be even. The frame must fit within the canvas.
    pub fn push_frame(&mut self, frame: MuxFrame) -> Result<(), MuxError> {
        if !frame.x_offset.is_multiple_of(2) || !frame.y_offset.is_multiple_of(2) {
            return Err(MuxError::OddFrameOffset {
                x: frame.x_offset,
                y: frame.y_offset,
            });
        }
        if frame.width == 0 || frame.height == 0 || frame.width > 16384 || frame.height > 16384 {
            return Err(MuxError::InvalidDimensions {
                width: frame.width,
                height: frame.height,
            });
        }
        if frame.x_offset + frame.width > self.canvas_width
            || frame.y_offset + frame.height > self.canvas_height
        {
            return Err(MuxError::FrameOutsideCanvas {
                x: frame.x_offset,
                y: frame.y_offset,
                width: frame.width,
                height: frame.height,
                canvas_width: self.canvas_width,
                canvas_height: self.canvas_height,
            });
        }
        self.frames.push(frame);
        Ok(())
    }

    /// Set a single (non-animated) image.
    pub fn set_image(&mut self, frame: MuxFrame) {
        self.single_image = Some(frame);
    }

    /// Number of animation frames.
    pub fn num_frames(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Assemble the final WebP file.
    pub fn assemble(&self) -> Result<Vec<u8>, MuxError> {
        if self.animation.is_some() {
            self.assemble_animated()
        } else {
            self.assemble_single()
        }
    }

    fn assemble_single(&self) -> Result<Vec<u8>, MuxError> {
        let frame = self.single_image.as_ref().ok_or(MuxError::NoFrames)?;

        let frame_chunk = if frame.is_lossless { b"VP8L" } else { b"VP8 " };
        let has_alpha = frame.alpha_data.is_some() || frame.is_lossless;

        let needs_extended = self.icc_profile.is_some()
            || self.exif.is_some()
            || self.xmp.is_some()
            || frame.alpha_data.is_some();

        if !needs_extended {
            // Simple container
            let mut out = Vec::new();
            out.write_all(b"RIFF");
            out.write_u32_le(chunk_size(frame.bitstream.len()) + 4);
            out.write_all(b"WEBP");
            write_chunk(&mut out, frame_chunk, &frame.bitstream);
            return Ok(out);
        }

        // Extended container
        let mut total = 4u32 + chunk_size(10); // "WEBP" + VP8X

        if let Some(icc) = &self.icc_profile {
            total += chunk_size(icc.len());
        }
        if let Some(alpha) = &frame.alpha_data {
            total += chunk_size(alpha.len());
        }
        total += chunk_size(frame.bitstream.len());
        if let Some(exif) = &self.exif {
            total += chunk_size(exif.len());
        }
        if let Some(xmp) = &self.xmp {
            total += chunk_size(xmp.len());
        }

        let mut out = Vec::with_capacity(total as usize + 8);
        out.write_all(b"RIFF");
        out.write_u32_le(total);
        out.write_all(b"WEBP");

        // VP8X
        let mut flags = 0u8;
        if self.xmp.is_some() {
            flags |= 1 << 2;
        }
        if self.exif.is_some() {
            flags |= 1 << 3;
        }
        if has_alpha {
            flags |= 1 << 4;
        }
        if self.icc_profile.is_some() {
            flags |= 1 << 5;
        }

        let mut vp8x = Vec::with_capacity(10);
        vp8x.push(flags);
        vp8x.write_all(&[0; 3]); // reserved
        vp8x.write_u24_le(self.canvas_width - 1);
        vp8x.write_u24_le(self.canvas_height - 1);
        write_chunk(&mut out, b"VP8X", &vp8x);

        if let Some(icc) = &self.icc_profile {
            write_chunk(&mut out, b"ICCP", icc);
        }
        if let Some(alpha) = &frame.alpha_data {
            write_chunk(&mut out, b"ALPH", alpha);
        }
        write_chunk(&mut out, frame_chunk, &frame.bitstream);
        if let Some(exif) = &self.exif {
            write_chunk(&mut out, b"EXIF", exif);
        }
        if let Some(xmp) = &self.xmp {
            write_chunk(&mut out, b"XMP ", xmp);
        }

        Ok(out)
    }

    fn assemble_animated(&self) -> Result<Vec<u8>, MuxError> {
        if self.frames.is_empty() {
            return Err(MuxError::NoFrames);
        }

        let anim = self.animation.as_ref().unwrap();

        // Calculate total size
        let mut total = 4u32 + chunk_size(10); // "WEBP" + VP8X
        if let Some(icc) = &self.icc_profile {
            total += chunk_size(icc.len());
        }
        total += chunk_size(6); // ANIM chunk

        for frame in &self.frames {
            let anmf_payload_size = self.anmf_payload_size(frame);
            total += chunk_size(anmf_payload_size);
        }

        if let Some(exif) = &self.exif {
            total += chunk_size(exif.len());
        }
        if let Some(xmp) = &self.xmp {
            total += chunk_size(xmp.len());
        }

        let mut out = Vec::with_capacity(total as usize + 8);
        out.write_all(b"RIFF");
        out.write_u32_le(total);
        out.write_all(b"WEBP");

        // VP8X flags
        let has_alpha = self
            .frames
            .iter()
            .any(|f| f.alpha_data.is_some() || f.is_lossless);
        let mut flags = 0u8;
        if self.xmp.is_some() {
            flags |= 1 << 2;
        }
        if self.exif.is_some() {
            flags |= 1 << 3;
        }
        if has_alpha {
            flags |= 1 << 4;
        }
        if self.icc_profile.is_some() {
            flags |= 1 << 5;
        }
        flags |= 1 << 1; // animation flag

        let mut vp8x = Vec::with_capacity(10);
        vp8x.push(flags);
        vp8x.write_all(&[0; 3]);
        vp8x.write_u24_le(self.canvas_width - 1);
        vp8x.write_u24_le(self.canvas_height - 1);
        write_chunk(&mut out, b"VP8X", &vp8x);

        if let Some(icc) = &self.icc_profile {
            write_chunk(&mut out, b"ICCP", icc);
        }

        // ANIM chunk
        let mut anim_data = Vec::with_capacity(6);
        anim_data.write_all(&anim.background_color);
        let lc = match anim.loop_count {
            LoopCount::Forever => 0u16,
            LoopCount::Times(n) => n.get(),
        };
        anim_data.write_u16_le(lc);
        write_chunk(&mut out, b"ANIM", &anim_data);

        // ANMF chunks
        for frame in &self.frames {
            self.write_anmf(&mut out, frame);
        }

        if let Some(exif) = &self.exif {
            write_chunk(&mut out, b"EXIF", exif);
        }
        if let Some(xmp) = &self.xmp {
            write_chunk(&mut out, b"XMP ", xmp);
        }

        Ok(out)
    }

    fn anmf_payload_size(&self, frame: &MuxFrame) -> usize {
        let mut size = 16usize; // ANMF header fields (6*3 + 1 = 16 bytes... no: 3+3+3+3+3+1 = 16)
        if let Some(alpha) = &frame.alpha_data {
            size += 8 + alpha.len() + (alpha.len() & 1); // ALPH chunk
        }
        let frame_chunk_tag = if frame.is_lossless { b"VP8L" } else { b"VP8 " };
        let _ = frame_chunk_tag; // used for clarity
        size += 8 + frame.bitstream.len() + (frame.bitstream.len() & 1); // VP8/VP8L chunk
        size
    }

    fn write_anmf(&self, out: &mut Vec<u8>, frame: &MuxFrame) {
        let payload_size = self.anmf_payload_size(frame);

        // ANMF chunk header
        out.write_all(b"ANMF");
        out.write_u32_le(payload_size as u32);

        // ANMF payload: offsets stored in 2-pixel units
        out.write_u24_le(frame.x_offset / 2);
        out.write_u24_le(frame.y_offset / 2);
        out.write_u24_le(frame.width - 1);
        out.write_u24_le(frame.height - 1);
        out.write_u24_le(frame.duration_ms);

        let mut flags = 0u8;
        if matches!(frame.dispose, DisposeMethod::Background) {
            flags |= 1;
        }
        if matches!(frame.blend, BlendMethod::Overwrite) {
            flags |= 2;
        }
        out.push(flags);

        // Sub-chunks
        if let Some(alpha) = &frame.alpha_data {
            write_chunk(out, b"ALPH", alpha);
        }

        let frame_chunk = if frame.is_lossless { b"VP8L" } else { b"VP8 " };
        write_chunk(out, frame_chunk, &frame.bitstream);
    }
}
