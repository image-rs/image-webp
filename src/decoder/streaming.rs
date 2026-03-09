//! Streaming WebP decoder for incremental data delivery.
//!
//! Buffers incoming data and reports when enough is available to extract
//! image information or perform a full decode. Useful for network streams
//! and memory-constrained environments where data arrives in chunks.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::{StreamingDecoder, StreamStatus};
//!
//! let mut decoder = StreamingDecoder::new();
//!
//! // Feed data as it arrives
//! let network_chunks: Vec<Vec<u8>> = vec![]; // your data chunks
//! for chunk in &network_chunks {
//!     match decoder.append(chunk)? {
//!         StreamStatus::NeedMoreData => continue,
//!         StreamStatus::HeaderReady => {
//!             let info = decoder.info()?;
//!             println!("{}x{}", info.width, info.height);
//!         }
//!         StreamStatus::Complete => break,
//!         _ => {}
//!     }
//! }
//!
//! let (pixels, w, h) = decoder.finish_rgba()?;
//! # Ok::<(), whereat::At<zenwebp::DecodeError>>(())
//! ```

use alloc::vec::Vec;

use super::api::{DecodeConfig, DecodeError, DecodeResult, ImageInfo, WebPDecoder};

/// Status returned from [`StreamingDecoder::append`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamStatus {
    /// More data is needed before any useful work can be done.
    NeedMoreData,
    /// Enough data has been received to parse image headers.
    /// Call [`StreamingDecoder::info`] to inspect dimensions and format.
    HeaderReady,
    /// The declared RIFF size has been fully received.
    /// Call a `finish_*` method to decode.
    Complete,
}

/// Incremental WebP decoder that buffers data from a stream.
///
/// Data is appended via [`append`](Self::append). Once the full RIFF payload
/// has arrived ([`StreamStatus::Complete`]), call [`finish_rgba`](Self::finish_rgba)
/// or another finish method to decode the image.
pub struct StreamingDecoder {
    buf: Vec<u8>,
    riff_size: Option<u32>,
    header_parsed: bool,
    config: DecodeConfig,
}

impl StreamingDecoder {
    /// Create a new streaming decoder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            riff_size: None,
            header_parsed: false,
            config: DecodeConfig::default(),
        }
    }

    /// Create a new streaming decoder with custom configuration.
    #[must_use]
    pub fn with_config(config: DecodeConfig) -> Self {
        Self {
            buf: Vec::new(),
            riff_size: None,
            header_parsed: false,
            config,
        }
    }

    /// Append data to the internal buffer and check progress.
    ///
    /// Returns the current status after incorporating the new data.
    pub fn append(&mut self, data: &[u8]) -> Result<StreamStatus, DecodeError> {
        self.buf.extend_from_slice(data);

        // Try to parse RIFF header (12 bytes minimum)
        if self.riff_size.is_none() && self.buf.len() >= 12 {
            if &self.buf[0..4] != b"RIFF" {
                return Err(DecodeError::RiffSignatureInvalid(
                    self.buf[0..4].try_into().unwrap(),
                ));
            }
            let size = u32::from_le_bytes(self.buf[4..8].try_into().unwrap());
            if &self.buf[8..12] != b"WEBP" {
                return Err(DecodeError::WebpSignatureInvalid(
                    self.buf[8..12].try_into().unwrap(),
                ));
            }
            self.riff_size = Some(size);
        }

        // Check if we have enough for headers (RIFF + first chunk header)
        if !self.header_parsed && self.buf.len() >= 30 {
            // Try parsing — if it succeeds, headers are ready
            if WebPDecoder::new(&self.buf).is_ok() {
                self.header_parsed = true;
            }
        }

        // Check if all data has arrived
        if let Some(riff_size) = self.riff_size {
            let total = riff_size as usize + 8; // RIFF header is 8 bytes before payload
            if self.buf.len() >= total {
                return Ok(StreamStatus::Complete);
            }
        }

        if self.header_parsed {
            Ok(StreamStatus::HeaderReady)
        } else {
            Ok(StreamStatus::NeedMoreData)
        }
    }

    /// Get image information, available after [`StreamStatus::HeaderReady`].
    pub fn info(&self) -> DecodeResult<ImageInfo> {
        if !self.header_parsed {
            return Err(whereat::at!(DecodeError::InvalidParameter(
                alloc::string::String::from("headers not yet available"),
            )));
        }
        ImageInfo::from_webp(&self.buf)
    }

    /// Returns the number of bytes buffered so far.
    #[must_use]
    pub fn bytes_buffered(&self) -> usize {
        self.buf.len()
    }

    /// Returns the total expected size in bytes, if the RIFF header has been parsed.
    #[must_use]
    pub fn total_size(&self) -> Option<usize> {
        self.riff_size.map(|s| s as usize + 8)
    }

    /// Returns true if all data has been received.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.total_size()
            .is_some_and(|total| self.buf.len() >= total)
    }

    /// Consume the decoder and decode to RGBA pixels.
    ///
    /// Returns an error if data is incomplete.
    pub fn finish_rgba(self) -> DecodeResult<(Vec<u8>, u32, u32)> {
        self.ensure_complete().map_err(|e| whereat::at!(e))?;
        super::api::DecodeRequest::new(&self.config, &self.buf).decode_rgba()
    }

    /// Consume the decoder and decode to RGB pixels.
    ///
    /// Returns an error if data is incomplete.
    pub fn finish_rgb(self) -> DecodeResult<(Vec<u8>, u32, u32)> {
        self.ensure_complete().map_err(|e| whereat::at!(e))?;
        super::api::DecodeRequest::new(&self.config, &self.buf).decode_rgb()
    }

    /// Consume the decoder and decode RGBA into a pre-allocated buffer.
    ///
    /// Returns an error if data is incomplete.
    pub fn finish_rgba_into(self, output: &mut [u8]) -> DecodeResult<(u32, u32)> {
        self.ensure_complete().map_err(|e| whereat::at!(e))?;
        super::api::DecodeRequest::new(&self.config, &self.buf).decode_rgba_into(output)
    }

    /// Take ownership of the buffered data without decoding.
    ///
    /// Useful if you want to decode later or pass to another decoder.
    #[must_use]
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    fn ensure_complete(&self) -> Result<(), DecodeError> {
        if !self.is_complete() {
            return Err(DecodeError::InvalidParameter(alloc::string::String::from(
                "data incomplete",
            )));
        }
        Ok(())
    }
}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new()
    }
}
