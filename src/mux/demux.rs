//! Zero-copy WebP demuxer.
//!
//! Parses a WebP file at the chunk level, exposing frame metadata and raw
//! bitstream data without decoding pixels.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::mux::WebPDemuxer;
//!
//! let data: &[u8] = &[]; // your WebP data
//! let demuxer = WebPDemuxer::new(data)?;
//! println!("{}x{}, {} frame(s)", demuxer.canvas_width(), demuxer.canvas_height(), demuxer.num_frames());
//!
//! for frame in demuxer.frames() {
//!     println!("  frame {}: {}x{} at ({},{}) duration={}ms",
//!         frame.frame_num, frame.width, frame.height,
//!         frame.x_offset, frame.y_offset, frame.duration_ms);
//! }
//! # Ok::<(), whereat::At<zenwebp::mux::MuxError>>(())
//! ```

use alloc::vec::Vec;

use super::error::{MuxError, MuxResult};
use crate::decoder::LoopCount;
use crate::slice_reader::SliceReader;

/// How the frame area is disposed after rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisposeMethod {
    /// Do not dispose. The frame remains on the canvas.
    None,
    /// Fill the frame rectangle with the background color.
    Background,
}

/// How the frame is blended with the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMethod {
    /// Use alpha blending with the existing canvas content.
    AlphaBlend,
    /// Overwrite the canvas region with the frame data.
    Overwrite,
}

/// Metadata for a single frame extracted by the demuxer.
///
/// The `bitstream` field contains the raw VP8 or VP8L data (not including
/// any RIFF container framing). For lossy frames with separate alpha, the
/// `alpha_data` field contains the raw ALPH chunk payload.
#[derive(Debug, Clone)]
pub struct DemuxFrame<'a> {
    /// 1-based frame number.
    pub frame_num: u32,
    /// Horizontal offset of the frame on the canvas (always even).
    pub x_offset: u32,
    /// Vertical offset of the frame on the canvas (always even).
    pub y_offset: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame duration in milliseconds.
    pub duration_ms: u32,
    /// How the frame area is disposed after rendering.
    pub dispose: DisposeMethod,
    /// How the frame is blended onto the canvas.
    pub blend: BlendMethod,
    /// Whether the frame contains alpha data.
    pub has_alpha: bool,
    /// Whether the frame uses lossy (VP8) encoding. `false` means lossless (VP8L).
    pub is_lossy: bool,
    /// Raw VP8 or VP8L bitstream data for this frame.
    pub bitstream: &'a [u8],
    /// Raw ALPH chunk payload, if present (lossy frames with separate alpha).
    pub alpha_data: Option<&'a [u8]>,
}

/// Parsed frame record (byte ranges into the original data).
#[derive(Debug, Clone)]
struct FrameRecord {
    /// Byte offset of the ANMF payload start (after the 8-byte chunk header).
    anmf_payload_start: usize,
    /// Total ANMF payload size.
    anmf_payload_size: usize,
}

/// Zero-copy WebP demuxer.
///
/// Parses a WebP file and provides access to frame metadata, raw bitstreams,
/// and container-level metadata (ICC, EXIF, XMP) without decoding pixel data.
pub struct WebPDemuxer<'a> {
    data: &'a [u8],
    canvas_width: u32,
    canvas_height: u32,
    loop_count: LoopCount,
    background_color: [u8; 4],
    has_alpha: bool,
    is_animated: bool,
    frames: Vec<FrameRecord>,
    // Byte ranges for metadata chunks (start..end within data)
    icc_range: Option<(usize, usize)>,
    exif_range: Option<(usize, usize)>,
    xmp_range: Option<(usize, usize)>,
    // For non-animated files: the single image bitstream info
    single_bitstream_range: Option<(usize, usize)>,
    single_alpha_range: Option<(usize, usize)>,
    single_is_lossy: bool,
    single_width: u32,
    single_height: u32,
}

impl<'a> WebPDemuxer<'a> {
    /// Parse a WebP file from a byte slice.
    ///
    /// This only parses the container structure and records byte ranges.
    /// No pixel decoding is performed.
    #[track_caller]
    pub fn new(data: &'a [u8]) -> MuxResult<Self> {
        Self::new_inner(data).map_err(|e| whereat::at!(e))
    }

    fn new_inner(data: &'a [u8]) -> Result<Self, MuxError> {
        if data.len() < 12 {
            return Err(MuxError::InvalidFormat("File too small".into()));
        }

        let mut r = SliceReader::new(data);

        // Read RIFF header
        let mut sig = [0u8; 4];
        r.read_exact(&mut sig)?;
        if &sig != b"RIFF" {
            return Err(MuxError::InvalidFormat("Missing RIFF signature".into()));
        }

        let riff_size = r.read_u32_le()? as usize;

        r.read_exact(&mut sig)?;
        if &sig != b"WEBP" {
            return Err(MuxError::InvalidFormat("Missing WEBP signature".into()));
        }

        // Read first chunk fourcc
        let mut fourcc = [0u8; 4];
        r.read_exact(&mut fourcc)?;
        let chunk_size = r.read_u32_le()? as usize;

        match &fourcc {
            b"VP8 " => Self::parse_simple_lossy(data, r, chunk_size),
            b"VP8L" => Self::parse_simple_lossless(data, r, chunk_size),
            b"VP8X" => Self::parse_extended(data, r, chunk_size, riff_size),
            _ => Err(MuxError::InvalidFormat(alloc::format!(
                "Unknown first chunk: {:?}",
                fourcc
            ))),
        }
    }

    fn parse_simple_lossy(
        data: &'a [u8],
        mut r: SliceReader<'a>,
        chunk_size: usize,
    ) -> Result<Self, MuxError> {
        let bitstream_start = r.position() as usize;

        // Parse VP8 frame header to get dimensions
        if chunk_size < 10 {
            return Err(MuxError::InvalidFormat("VP8 chunk too small".into()));
        }

        let frame_tag = r.read_u24_le()?;
        let is_keyframe = frame_tag & 1 == 0;
        if !is_keyframe {
            return Err(MuxError::InvalidFormat("Not a keyframe".into()));
        }

        // Skip to magic bytes
        let mut magic = [0u8; 3];
        r.read_exact(&mut magic)?;
        if magic != [0x9D, 0x01, 0x2A] {
            return Err(MuxError::InvalidFormat("Invalid VP8 magic".into()));
        }

        let w = r.read_u16_le()?;
        let h = r.read_u16_le()?;
        let width = u32::from(w & 0x3FFF);
        let height = u32::from(h & 0x3FFF);

        Ok(Self {
            data,
            canvas_width: width,
            canvas_height: height,
            loop_count: LoopCount::Times(core::num::NonZeroU16::new(1).unwrap()),
            background_color: [0; 4],
            has_alpha: false,
            is_animated: false,
            frames: Vec::new(),
            icc_range: None,
            exif_range: None,
            xmp_range: None,
            single_bitstream_range: Some((bitstream_start, bitstream_start + chunk_size)),
            single_alpha_range: None,
            single_is_lossy: true,
            single_width: width,
            single_height: height,
        })
    }

    fn parse_simple_lossless(
        data: &'a [u8],
        mut r: SliceReader<'a>,
        chunk_size: usize,
    ) -> Result<Self, MuxError> {
        let bitstream_start = r.position() as usize;

        // Parse VP8L header to get dimensions
        if chunk_size < 5 {
            return Err(MuxError::InvalidFormat("VP8L chunk too small".into()));
        }

        let signature = r.read_u8()?;
        if signature != 0x2f {
            return Err(MuxError::InvalidFormat("Invalid VP8L signature".into()));
        }

        let header = r.read_u32_le()?;
        let width = (1 + header) & 0x3FFF;
        let height = (1 + (header >> 14)) & 0x3FFF;
        let has_alpha = (header >> 28) & 1 != 0;

        Ok(Self {
            data,
            canvas_width: width,
            canvas_height: height,
            loop_count: LoopCount::Times(core::num::NonZeroU16::new(1).unwrap()),
            background_color: [0; 4],
            has_alpha,
            is_animated: false,
            frames: Vec::new(),
            icc_range: None,
            exif_range: None,
            xmp_range: None,
            single_bitstream_range: Some((bitstream_start, bitstream_start + chunk_size)),
            single_alpha_range: None,
            single_is_lossy: false,
            single_width: width,
            single_height: height,
        })
    }

    fn parse_extended(
        data: &'a [u8],
        mut r: SliceReader<'a>,
        vp8x_size: usize,
        riff_size: usize,
    ) -> Result<Self, MuxError> {
        if vp8x_size < 10 {
            return Err(MuxError::InvalidFormat("VP8X chunk too small".into()));
        }

        let flags = r.read_u8()?;
        let has_icc = flags & 0b00100000 != 0;
        let has_alpha = flags & 0b00010000 != 0;
        let has_exif = flags & 0b00001000 != 0;
        let has_xmp = flags & 0b00000100 != 0;
        let is_animated = flags & 0b00000010 != 0;

        // Skip 3 reserved bytes
        r.seek_relative(3)?;

        // Canvas dimensions (24-bit LE, stored as value-1)
        let canvas_width = r.read_u24_le()? + 1;
        let canvas_height = r.read_u24_le()? + 1;

        let mut demuxer = Self {
            data,
            canvas_width,
            canvas_height,
            loop_count: LoopCount::Times(core::num::NonZeroU16::new(1).unwrap()),
            background_color: [0; 4],
            has_alpha,
            is_animated,
            frames: Vec::new(),
            icc_range: None,
            exif_range: None,
            xmp_range: None,
            single_bitstream_range: None,
            single_alpha_range: None,
            single_is_lossy: false,
            single_width: canvas_width,
            single_height: canvas_height,
        };

        // Skip any remaining VP8X payload padding
        let vp8x_rounded = vp8x_size + (vp8x_size & 1);
        r.seek_from_start(12 + 8 + vp8x_rounded as u64)?;

        let max_pos = 8 + riff_size; // RIFF header (8 bytes) + declared size

        // Scan all chunks
        while (r.position() as usize) + 8 <= max_pos && (r.position() as usize) + 8 <= data.len() {
            let chunk_start = r.position() as usize;
            let mut fourcc = [0u8; 4];
            if r.read_exact(&mut fourcc).is_err() {
                break;
            }
            let size = match r.read_u32_le() {
                Ok(s) => s as usize,
                Err(_) => break,
            };
            let payload_start = r.position() as usize;
            let rounded = size + (size & 1);

            match &fourcc {
                b"ICCP" if has_icc => {
                    demuxer.icc_range = Some((payload_start, payload_start + size));
                }
                b"ANIM" if is_animated => {
                    // ANIM: 4 bytes background color + 2 bytes loop count
                    if size >= 6 {
                        let bg_start = payload_start;
                        demuxer.background_color = [
                            data[bg_start],
                            data[bg_start + 1],
                            data[bg_start + 2],
                            data[bg_start + 3],
                        ];
                        let lc = u16::from_le_bytes([data[bg_start + 4], data[bg_start + 5]]);
                        demuxer.loop_count = match lc {
                            0 => LoopCount::Forever,
                            n => LoopCount::Times(core::num::NonZeroU16::new(n).unwrap()),
                        };
                    }
                }
                b"ANMF" if is_animated => {
                    demuxer.frames.push(FrameRecord {
                        anmf_payload_start: payload_start,
                        anmf_payload_size: size,
                    });
                }
                b"VP8 " if !is_animated => {
                    demuxer.single_bitstream_range = Some((payload_start, payload_start + size));
                    demuxer.single_is_lossy = true;
                }
                b"VP8L" if !is_animated => {
                    demuxer.single_bitstream_range = Some((payload_start, payload_start + size));
                    demuxer.single_is_lossy = false;
                }
                b"ALPH" if !is_animated => {
                    demuxer.single_alpha_range = Some((payload_start, payload_start + size));
                }
                b"EXIF" if has_exif => {
                    demuxer.exif_range = Some((payload_start, payload_start + size));
                }
                b"XMP " if has_xmp => {
                    demuxer.xmp_range = Some((payload_start, payload_start + size));
                }
                _ => {}
            }

            // Advance past this chunk's payload
            if r.seek_from_start((chunk_start + 8 + rounded) as u64)
                .is_err()
            {
                break;
            }
        }

        Ok(demuxer)
    }

    /// Canvas width in pixels.
    pub fn canvas_width(&self) -> u32 {
        self.canvas_width
    }

    /// Canvas height in pixels.
    pub fn canvas_height(&self) -> u32 {
        self.canvas_height
    }

    /// Number of frames. Non-animated images return 1.
    pub fn num_frames(&self) -> u32 {
        if self.is_animated {
            self.frames.len() as u32
        } else {
            1
        }
    }

    /// Number of frames (alias for [`num_frames()`](Self::num_frames)).
    pub fn frame_count(&self) -> u32 {
        self.num_frames()
    }

    /// Loop count for animated images.
    pub fn loop_count(&self) -> LoopCount {
        self.loop_count
    }

    /// Background color for animated images (BGRA byte order as per the spec).
    pub fn background_color(&self) -> [u8; 4] {
        self.background_color
    }

    /// Whether the image is animated.
    pub fn is_animated(&self) -> bool {
        self.is_animated
    }

    /// Whether the image has alpha data.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Get a specific frame by 1-based index.
    ///
    /// Returns `None` if the index is out of range.
    pub fn frame(&self, n: u32) -> Option<DemuxFrame<'a>> {
        if n == 0 {
            return None;
        }

        if !self.is_animated {
            if n != 1 {
                return None;
            }
            return self.single_frame();
        }

        let idx = (n - 1) as usize;
        if idx >= self.frames.len() {
            return None;
        }

        self.parse_anmf_frame(idx)
    }

    /// Iterate over all frames.
    pub fn frames(&self) -> DemuxFrameIter<'a, '_> {
        DemuxFrameIter {
            demuxer: self,
            current: 1,
        }
    }

    /// ICC profile data, if present.
    pub fn icc_profile(&self) -> Option<&'a [u8]> {
        self.icc_range.map(|(s, e)| &self.data[s..e])
    }

    /// EXIF metadata, if present.
    pub fn exif(&self) -> Option<&'a [u8]> {
        self.exif_range.map(|(s, e)| &self.data[s..e])
    }

    /// XMP metadata, if present.
    pub fn xmp(&self) -> Option<&'a [u8]> {
        self.xmp_range.map(|(s, e)| &self.data[s..e])
    }

    fn single_frame(&self) -> Option<DemuxFrame<'a>> {
        let (bs_start, bs_end) = self.single_bitstream_range?;
        let alpha_data = self.single_alpha_range.map(|(s, e)| &self.data[s..e]);

        Some(DemuxFrame {
            frame_num: 1,
            x_offset: 0,
            y_offset: 0,
            width: self.single_width,
            height: self.single_height,
            duration_ms: 0,
            dispose: DisposeMethod::None,
            blend: BlendMethod::Overwrite,
            has_alpha: self.has_alpha,
            is_lossy: self.single_is_lossy,
            bitstream: &self.data[bs_start..bs_end],
            alpha_data,
        })
    }

    fn parse_anmf_frame(&self, idx: usize) -> Option<DemuxFrame<'a>> {
        let record = &self.frames[idx];
        let start = record.anmf_payload_start;
        let size = record.anmf_payload_size;

        if size < 16 {
            return None;
        }

        let d = &self.data[start..start + size];

        // ANMF payload layout:
        // 3 bytes: Frame X (in 2-pixel units)
        // 3 bytes: Frame Y (in 2-pixel units)
        // 3 bytes: Frame Width Minus One
        // 3 bytes: Frame Height Minus One
        // 3 bytes: Frame Duration
        // 1 byte:  Flags (dispose[0], blend[1], reserved[2-7])
        // Then: sub-chunks (ALPH + VP8, or VP8L)

        let x_offset = read_u24_le(&d[0..3]) * 2;
        let y_offset = read_u24_le(&d[3..6]) * 2;
        let width = read_u24_le(&d[6..9]) + 1;
        let height = read_u24_le(&d[9..12]) + 1;
        let duration_ms = read_u24_le(&d[12..15]);
        let flags = d[15];
        let dispose = if flags & 1 != 0 {
            DisposeMethod::Background
        } else {
            DisposeMethod::None
        };
        let blend = if flags & 2 != 0 {
            BlendMethod::Overwrite
        } else {
            BlendMethod::AlphaBlend
        };

        // Parse sub-chunks starting at offset 16
        let sub = &d[16..];
        if sub.len() < 8 {
            return None;
        }

        let sub_fourcc = &sub[0..4];
        let sub_size = u32::from_le_bytes([sub[4], sub[5], sub[6], sub[7]]) as usize;
        let sub_payload_offset = start + 16 + 8; // absolute offset in data

        match sub_fourcc {
            b"VP8L" => {
                let end = (sub_payload_offset + sub_size).min(self.data.len());
                Some(DemuxFrame {
                    frame_num: idx as u32 + 1,
                    x_offset,
                    y_offset,
                    width,
                    height,
                    duration_ms,
                    dispose,
                    blend,
                    has_alpha: true, // VP8L can always carry alpha
                    is_lossy: false,
                    bitstream: &self.data[sub_payload_offset..end],
                    alpha_data: None,
                })
            }
            b"VP8 " => {
                let end = (sub_payload_offset + sub_size).min(self.data.len());
                Some(DemuxFrame {
                    frame_num: idx as u32 + 1,
                    x_offset,
                    y_offset,
                    width,
                    height,
                    duration_ms,
                    dispose,
                    blend,
                    has_alpha: false,
                    is_lossy: true,
                    bitstream: &self.data[sub_payload_offset..end],
                    alpha_data: None,
                })
            }
            b"ALPH" => {
                // ALPH chunk followed by VP8 chunk
                let alpha_end = (sub_payload_offset + sub_size).min(self.data.len());
                let alpha_data = &self.data[sub_payload_offset..alpha_end];

                let sub_rounded = sub_size + (sub_size & 1);
                let vp8_header_start = start + 16 + 8 + sub_rounded;
                if vp8_header_start + 8 > start + size {
                    return None;
                }

                let vp8_fourcc = &self.data[vp8_header_start..vp8_header_start + 4];
                if vp8_fourcc != b"VP8 " {
                    return None;
                }
                let vp8_size = u32::from_le_bytes([
                    self.data[vp8_header_start + 4],
                    self.data[vp8_header_start + 5],
                    self.data[vp8_header_start + 6],
                    self.data[vp8_header_start + 7],
                ]) as usize;
                let vp8_start = vp8_header_start + 8;
                let vp8_end = (vp8_start + vp8_size).min(self.data.len());

                Some(DemuxFrame {
                    frame_num: idx as u32 + 1,
                    x_offset,
                    y_offset,
                    width,
                    height,
                    duration_ms,
                    dispose,
                    blend,
                    has_alpha: true,
                    is_lossy: true,
                    bitstream: &self.data[vp8_start..vp8_end],
                    alpha_data: Some(alpha_data),
                })
            }
            _ => None,
        }
    }
}

/// Iterator over demuxed frames.
pub struct DemuxFrameIter<'a, 'b> {
    demuxer: &'b WebPDemuxer<'a>,
    current: u32,
}

impl<'a, 'b> Iterator for DemuxFrameIter<'a, 'b> {
    type Item = DemuxFrame<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = self.demuxer.frame(self.current)?;
        self.current += 1;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.demuxer.num_frames().saturating_sub(self.current - 1) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DemuxFrameIter<'_, '_> {}

/// Read a 24-bit little-endian value from 3 bytes.
fn read_u24_le(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}
