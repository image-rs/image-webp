//! WebP encoder detection and quality estimation.
//!
//! Identifies whether a WebP file is lossy or lossless, estimates the quality
//! level for lossy files, and provides re-encoding recommendations — all from
//! header-only parsing without pixel decoding.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenwebp::detect::{probe, BitstreamType};
//!
//! let webp_data = std::fs::read("photo.webp").unwrap();
//! let info = probe(&webp_data).unwrap();
//!
//! match info.bitstream {
//!     BitstreamType::Lossy { quality_estimate, quantizer_index } => {
//!         println!("Lossy WebP, estimated quality: {:.0}", quality_estimate);
//!         println!("VP8 base quantizer: {}", quantizer_index);
//!     }
//!     BitstreamType::Lossless => {
//!         println!("Lossless WebP — no quality to detect");
//!     }
//! }
//!
//! if let Some(q) = info.recommended_quality() {
//!     println!("Re-encode at quality {:.0} to match", q);
//! }
//! ```

use alloc::vec::Vec;

/// Result of probing a WebP file.
#[derive(Debug, Clone)]
pub struct WebPProbe {
    /// Image dimensions.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// Whether the image is animated.
    pub has_animation: bool,
    /// Number of frames (1 for static images).
    pub frame_count: u32,
    /// Bitstream type with quality details.
    pub bitstream: BitstreamType,
    /// ICC color profile, if present.
    pub icc_profile: Option<Vec<u8>>,
}

/// Bitstream format with quality estimation details.
#[derive(Debug, Clone)]
pub enum BitstreamType {
    /// Lossy VP8 compression with quality estimation.
    Lossy {
        /// Estimated quality (0-100) that produced this file.
        ///
        /// Reversed from the VP8 base quantizer using libwebp's quality mapping.
        /// Exact when the file was produced with libwebp/zenwebp; approximate
        /// for other encoders.
        quality_estimate: f32,
        /// VP8 base quantizer index (0-127).
        ///
        /// Lower = higher quality. 0 = best quality, 127 = worst.
        quantizer_index: u8,
        /// Whether segments with per-segment quantizer overrides are used.
        has_segment_quant: bool,
        /// Filter level (0-63). Higher = more deblocking.
        filter_level: u8,
        /// Sharpness level (0-7).
        sharpness_level: u8,
    },
    /// Lossless VP8L compression — no quality level to detect.
    Lossless,
}

/// Errors that can occur during WebP probing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeError {
    /// Data is too short to be a WebP file.
    TooShort,
    /// Missing RIFF signature.
    NotWebP,
    /// VP8 bitstream header is truncated or malformed.
    Truncated,
    /// VP8 magic bytes not found.
    InvalidVP8Magic,
}

impl core::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort => write!(f, "data too short to be a WebP file"),
            Self::NotWebP => write!(f, "not a WebP file (missing RIFF/WEBP signature)"),
            Self::Truncated => write!(f, "truncated WebP bitstream header"),
            Self::InvalidVP8Magic => write!(f, "invalid VP8 magic bytes"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProbeError {}

/// Probe a WebP file from its raw bytes.
///
/// Reads only container and bitstream headers. No pixel decoding is performed.
/// For lossy files, extracts the VP8 base quantizer and reverse-maps it to an
/// estimated quality level.
pub fn probe(data: &[u8]) -> Result<WebPProbe, ProbeError> {
    // Need at least RIFF header (12 bytes) + chunk header (8 bytes) + VP8 frame header (10 bytes)
    if data.len() < 12 {
        return Err(ProbeError::TooShort);
    }

    // Validate RIFF/WEBP container
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(ProbeError::NotWebP);
    }

    // Parse chunks to find the bitstream
    let mut pos = 12;
    let mut is_extended = false;
    let mut has_alpha = false;
    let mut has_animation = false;
    let mut frame_count = 1u32;
    let mut canvas_width = 0u32;
    let mut canvas_height = 0u32;
    let mut icc_profile = None;
    let mut bitstream_result = None;

    while pos + 8 <= data.len() {
        let fourcc = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let payload_start = pos + 8;
        let payload_end = (payload_start + chunk_size).min(data.len());

        match fourcc {
            b"VP8 " => {
                // Lossy bitstream — parse frame header for quantizer
                if bitstream_result.is_none() {
                    bitstream_result = Some(parse_vp8_header(&data[payload_start..payload_end])?);
                }
            }
            b"VP8L" => {
                // Lossless bitstream
                if bitstream_result.is_none() {
                    // Parse VP8L header for dimensions
                    if payload_end - payload_start >= 5 {
                        let signature = data[payload_start];
                        if signature == 0x2f {
                            let bits = u32::from_le_bytes([
                                data[payload_start + 1],
                                data[payload_start + 2],
                                data[payload_start + 3],
                                data[payload_start + 4],
                            ]);
                            let w = (bits & 0x3FFF) + 1;
                            let h = ((bits >> 14) & 0x3FFF) + 1;
                            has_alpha = ((bits >> 28) & 1) != 0;
                            canvas_width = w;
                            canvas_height = h;
                        }
                    }
                    bitstream_result = Some(ParsedBitstream::Lossless);
                }
            }
            b"VP8X" => {
                // Extended format header
                is_extended = true;
                if payload_end - payload_start >= 10 {
                    let flags = data[payload_start];
                    has_alpha = (flags & 0x10) != 0;
                    has_animation = (flags & 0x02) != 0;
                    let has_icc = (flags & 0x20) != 0;
                    let _ = has_icc; // used to know if ICCP chunk expected
                    canvas_width = u32::from_le_bytes([
                        data[payload_start + 4],
                        data[payload_start + 5],
                        data[payload_start + 6],
                        0,
                    ]) + 1;
                    canvas_height = u32::from_le_bytes([
                        data[payload_start + 7],
                        data[payload_start + 8],
                        data[payload_start + 9],
                        0,
                    ]) + 1;
                }
            }
            b"ANIM" => {
                // Animation global parameters — frame count comes from ANMF chunks
            }
            b"ANMF" => {
                // Animation frame — count them and parse the first one's bitstream
                if has_animation {
                    frame_count = frame_count.max(1);
                    // Each ANMF has a sub-bitstream; parse the first one for quality
                    if bitstream_result.is_none() && payload_end - payload_start >= 24 {
                        // ANMF payload: 16 bytes header + sub-chunks
                        let sub_start = payload_start + 16;
                        if sub_start + 8 <= payload_end {
                            let sub_fourcc = &data[sub_start..sub_start + 4];
                            let sub_size = u32::from_le_bytes([
                                data[sub_start + 4],
                                data[sub_start + 5],
                                data[sub_start + 6],
                                data[sub_start + 7],
                            ]) as usize;
                            let sub_payload_start = sub_start + 8;
                            let sub_payload_end = (sub_payload_start + sub_size).min(payload_end);
                            match sub_fourcc {
                                b"VP8 " => {
                                    bitstream_result = Some(parse_vp8_header(
                                        &data[sub_payload_start..sub_payload_end],
                                    )?);
                                }
                                b"VP8L" => {
                                    bitstream_result = Some(ParsedBitstream::Lossless);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            b"ICCP" => {
                if payload_end > payload_start {
                    icc_profile = Some(data[payload_start..payload_end].to_vec());
                }
            }
            _ => {}
        }

        // Advance to next chunk (chunks are 2-byte aligned)
        pos = payload_start + ((chunk_size + 1) & !1);
    }

    // Count ANMF chunks for animated images
    if has_animation {
        frame_count = 0;
        let mut scan_pos = 12;
        while scan_pos + 8 <= data.len() {
            let fourcc = &data[scan_pos..scan_pos + 4];
            let chunk_size = u32::from_le_bytes([
                data[scan_pos + 4],
                data[scan_pos + 5],
                data[scan_pos + 6],
                data[scan_pos + 7],
            ]) as usize;
            if fourcc == b"ANMF" {
                frame_count += 1;
            }
            scan_pos = scan_pos + 8 + ((chunk_size + 1) & !1);
        }
    }

    let parsed = bitstream_result.ok_or(ProbeError::Truncated)?;

    let bitstream = match parsed {
        ParsedBitstream::Lossy(info) => {
            if !is_extended {
                canvas_width = info.width;
                canvas_height = info.height;
            }
            BitstreamType::Lossy {
                quality_estimate: quant_index_to_quality(info.yac_abs),
                quantizer_index: info.yac_abs,
                has_segment_quant: info.has_segment_quant,
                filter_level: info.filter_level,
                sharpness_level: info.sharpness_level,
            }
        }
        ParsedBitstream::Lossless => BitstreamType::Lossless,
    };

    Ok(WebPProbe {
        width: canvas_width,
        height: canvas_height,
        has_alpha,
        has_animation,
        frame_count,
        bitstream,
        icc_profile,
    })
}

impl WebPProbe {
    /// Estimated source quality (0-100), or `None` for lossless.
    pub fn estimated_quality(&self) -> Option<f32> {
        match &self.bitstream {
            BitstreamType::Lossy {
                quality_estimate, ..
            } => Some(*quality_estimate),
            BitstreamType::Lossless => None,
        }
    }

    /// Recommended zenwebp quality for re-encoding that matches the source.
    ///
    /// Returns `None` for lossless files (use lossless re-encoding instead).
    /// For lossy files, returns a quality value that should produce similar
    /// perceptual quality without unnecessary size bloat.
    pub fn recommended_quality(&self) -> Option<f32> {
        match &self.bitstream {
            BitstreamType::Lossy {
                quality_estimate, ..
            } => {
                // Use the same quality — zenwebp uses the same mapping as libwebp
                Some(*quality_estimate)
            }
            BitstreamType::Lossless => None,
        }
    }

    /// Whether this file is lossless.
    pub fn is_lossless(&self) -> bool {
        matches!(self.bitstream, BitstreamType::Lossless)
    }
}

// ── Internal parsing ────────────────────────────────────────────────

/// Intermediate result from parsing a VP8 bitstream header.
enum ParsedBitstream {
    Lossy(Vp8HeaderInfo),
    Lossless,
}

struct Vp8HeaderInfo {
    width: u32,
    height: u32,
    yac_abs: u8,
    has_segment_quant: bool,
    filter_level: u8,
    sharpness_level: u8,
}

/// Parse a VP8 lossy bitstream header to extract the base quantizer.
///
/// This reads the uncompressed frame header (10 bytes) and then decodes
/// just enough of the boolean-coded header to reach the quantizer fields.
fn parse_vp8_header(data: &[u8]) -> Result<ParsedBitstream, ProbeError> {
    if data.len() < 10 {
        return Err(ProbeError::Truncated);
    }

    // Uncompressed data chunk (3 bytes)
    let tag = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
    let keyframe = tag & 1 == 0;
    if !keyframe {
        // Non-keyframe — can't reliably extract quantizer
        return Err(ProbeError::Truncated);
    }
    let first_partition_size = (tag >> 5) as usize;

    // VP8 magic
    if data[3..6] != [0x9d, 0x01, 0x2a] {
        return Err(ProbeError::InvalidVP8Magic);
    }

    // Dimensions
    let w = u16::from_le_bytes([data[6], data[7]]) & 0x3FFF;
    let h = u16::from_le_bytes([data[8], data[9]]) & 0x3FFF;

    // Boolean decoder for the first partition
    let part_start = 10;
    let part_end = (part_start + first_partition_size).min(data.len());
    if part_end <= part_start {
        return Err(ProbeError::Truncated);
    }
    let part_data = &data[part_start..part_end];

    // Minimal boolean decoder — just enough to reach quantizer fields
    let mut reader = MiniBoolReader::new(part_data)?;

    // color_space (1 bit) + pixel_type (1 bit)
    reader.read_literal(1)?;
    reader.read_literal(1)?;

    // Segment updates
    let segments_enabled = reader.read_flag()?;
    let mut has_segment_quant = false;
    if segments_enabled {
        let update_map = reader.read_flag()?;
        let update_data = reader.read_flag()?;

        if update_data {
            has_segment_quant = true;
            let absolute_delta = reader.read_flag()?;
            let _ = absolute_delta;

            // 4 segments × (quantizer_level + loopfilter_level)
            for _ in 0..4 {
                let present = reader.read_flag()?;
                if present {
                    reader.read_literal(7)?; // quantizer value
                    reader.read_literal(1)?; // sign
                }
            }
            for _ in 0..4 {
                let present = reader.read_flag()?;
                if present {
                    reader.read_literal(6)?; // loopfilter value
                    reader.read_literal(1)?; // sign
                }
            }
        }

        if update_map {
            for _ in 0..3 {
                let present = reader.read_flag()?;
                if present {
                    reader.read_literal(8)?;
                }
            }
        }
    }

    // Filter parameters
    let _filter_type = reader.read_flag()?;
    let filter_level = reader.read_literal(6)?;
    let sharpness_level = reader.read_literal(3)?;

    // Loop filter adjustments
    let lf_adj_enabled = reader.read_flag()?;
    if lf_adj_enabled {
        let lf_adj_update = reader.read_flag()?;
        if lf_adj_update {
            // 4 ref_deltas + 4 mode_deltas
            for _ in 0..4 {
                let present = reader.read_flag()?;
                if present {
                    reader.read_literal(6)?;
                    reader.read_literal(1)?;
                }
            }
            for _ in 0..4 {
                let present = reader.read_flag()?;
                if present {
                    reader.read_literal(6)?;
                    reader.read_literal(1)?;
                }
            }
        }
    }

    // Number of partitions (2 bits → 1 << value)
    reader.read_literal(2)?;

    // QUANTIZER INDICES — this is what we came for
    let yac_abs = reader.read_literal(7)?;

    Ok(ParsedBitstream::Lossy(Vp8HeaderInfo {
        width: w as u32,
        height: h as u32,
        yac_abs,
        has_segment_quant,
        filter_level,
        sharpness_level,
    }))
}

/// Reverse the libwebp quality-to-quantizer mapping.
///
/// libwebp maps quality → compression → quant_index:
///   compression = cbrt(linear_c)  where linear_c = piecewise(quality/100)
///   quant_index = round(127 * (1 - compression))
///
/// We invert this to recover the approximate quality from a quant index.
fn quant_index_to_quality(qi: u8) -> f32 {
    if qi == 0 {
        return 100.0;
    }
    if qi >= 127 {
        return 0.0;
    }

    // quant_index = round(127 * (1 - compression))
    // compression = 1 - quant_index / 127
    let compression = 1.0 - (qi as f64 / 127.0);

    // compression = cbrt(linear_c)
    // linear_c = compression³
    let linear_c = compression * compression * compression;

    // linear_c = if quality < 75: (quality/100) * (2/3)
    //            else:             2*(quality/100) - 1
    //
    // Invert:
    // if linear_c < 0.5: quality = linear_c * 150
    // else:              quality = (linear_c + 1) * 50
    let quality = if linear_c < 0.5 {
        linear_c * 150.0
    } else {
        (linear_c + 1.0) * 50.0
    };

    quality.clamp(0.0, 100.0) as f32
}

// ── Minimal boolean reader ──────────────────────────────────────────

/// Minimal VP8 boolean (arithmetic) decoder for header parsing only.
///
/// This is a stripped-down version that reads just enough to parse
/// the quantizer fields. It doesn't need to be fast — it runs once.
struct MiniBoolReader<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    value: u32,
    bits_left: i32,
}

impl<'a> MiniBoolReader<'a> {
    fn new(data: &'a [u8]) -> Result<Self, ProbeError> {
        if data.len() < 2 {
            return Err(ProbeError::Truncated);
        }
        // Initialize with the first two bytes
        let value = (u32::from(data[0]) << 8) | u32::from(data[1]);
        Ok(Self {
            data,
            pos: 2,
            range: 255,
            value,
            bits_left: 16,
        })
    }

    fn read_flag(&mut self) -> Result<bool, ProbeError> {
        Ok(self.read_bool(128)? != 0)
    }

    fn read_bool(&mut self, prob: u8) -> Result<u8, ProbeError> {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let big = self.value >= (split << 8);
        let (new_range, bit) = if big {
            (self.range - split, 1u8)
        } else {
            (split, 0u8)
        };
        if big {
            self.value -= split << 8;
        }
        self.range = new_range;

        // Renormalize
        while self.range < 128 {
            self.range <<= 1;
            self.value <<= 1;
            self.bits_left -= 1;
            if self.bits_left <= 0 {
                if self.pos < self.data.len() {
                    self.value |= u32::from(self.data[self.pos]);
                    self.pos += 1;
                }
                self.bits_left = 8;
            }
        }

        Ok(bit)
    }

    fn read_literal(&mut self, n: u8) -> Result<u8, ProbeError> {
        let mut val = 0u8;
        for _ in 0..n {
            val = (val << 1) | self.read_bool(128)?;
        }
        Ok(val)
    }
}

#[cfg(feature = "zencodec")]
impl zencodec::SourceEncodingDetails for WebPProbe {
    fn source_generic_quality(&self) -> Option<f32> {
        self.estimated_quality()
    }

    fn is_lossless(&self) -> bool {
        matches!(self.bitstream, BitstreamType::Lossless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_roundtrip() {
        // Verify that quality → quant → quality roundtrips approximately
        use crate::encoder::fast_math::quality_to_quant_index;

        for q in [0, 10, 25, 50, 75, 80, 85, 90, 95, 100] {
            let qi = quality_to_quant_index(q);
            let recovered = quant_index_to_quality(qi);
            let diff = (recovered - q as f32).abs();
            assert!(
                diff < 3.0,
                "Q{q} → qi={qi} → recovered={recovered:.1}, diff={diff:.1}"
            );
        }
    }

    #[test]
    fn test_quant_to_quality_boundaries() {
        assert_eq!(quant_index_to_quality(0), 100.0);
        assert_eq!(quant_index_to_quality(127), 0.0);

        // Mid-range should be reasonable (cubic mapping is non-linear)
        let mid = quant_index_to_quality(63);
        assert!(mid > 10.0 && mid < 80.0, "qi=63 → {mid}");
    }

    #[test]
    fn test_probe_too_short() {
        assert_eq!(probe(&[]).unwrap_err(), ProbeError::TooShort);
        assert_eq!(probe(&[0; 11]).unwrap_err(), ProbeError::TooShort);
    }

    #[test]
    fn test_probe_not_webp() {
        let bad = b"RIFF\x00\x00\x00\x00NOPE";
        assert_eq!(probe(bad).unwrap_err(), ProbeError::NotWebP);
    }
}
