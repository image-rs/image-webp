//! Encoding of WebP images.
//!
//! # API
//!
//! ```rust
//! use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout, Preset};
//!
//! let config = EncoderConfig::with_preset(Preset::Photo, 85.0)
//!     .with_method(4);
//!
//! let rgba_data = vec![255u8; 4 * 4 * 4]; // 4x4 RGBA image
//! let webp = EncodeRequest::new(&config, &rgba_data, PixelLayout::Rgba8, 4, 4)
//!     .encode()?;
//! # Ok::<(), zenwebp::EncodeError>(())
//! ```
use alloc::collections::BinaryHeap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::iter::Peekable;
use core::slice::ChunksExact;
use thiserror::Error;

use super::config;
use super::vec_writer::VecWriter;
use super::vp8::encode_frame_lossy;

/// Error that can occur during encoding.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EncodeError {
    /// An IO error occurred.
    #[cfg(feature = "std")]
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// The image dimensions are not allowed by the WebP format.
    #[error("Invalid dimensions")]
    InvalidDimensions,

    /// The input buffer is too small.
    #[error("Invalid buffer size: {0}")]
    InvalidBufferSize(String),

    /// Encoding was cancelled via a [`enough::Stop`] or [`EncodeProgress`] callback.
    #[error("Encoding cancelled: {0}")]
    Cancelled(enough::StopReason),

    /// A resource limit was exceeded (dimensions, memory, etc.).
    #[error("Limit exceeded: {0}")]
    LimitExceeded(String),

    /// The compressed header (partition 0) exceeds the VP8 19-bit size limit (524,287 bytes).
    ///
    /// This occurs with very large, high-entropy images where macroblock prediction
    /// mode headers exceed the VP8 frame tag capacity. Try reducing image dimensions,
    /// lowering quality, or splitting into tiles.
    #[error("Partition 0 overflow: {size} bytes exceeds VP8 limit of {max} bytes")]
    Partition0Overflow {
        /// Actual partition 0 size in bytes.
        size: u32,
        /// Maximum allowed size (524,287).
        max: u32,
    },

    /// Unsupported codec operation.
    #[error(transparent)]
    UnsupportedOperation(#[from] zencodec_types::UnsupportedOperation),
}

/// Result type alias using `At<EncodeError>` for automatic location tracking.
///
/// Errors wrapped in `At<>` automatically capture file and line information,
/// making debugging easier in production environments.
pub type EncodeResult<T> = core::result::Result<T, whereat::At<EncodeError>>;

impl From<enough::StopReason> for EncodeError {
    fn from(reason: enough::StopReason) -> Self {
        Self::Cancelled(reason)
    }
}

/// Content-aware encoding presets.
///
/// These presets configure the encoder for different types of content,
/// optimizing the balance between file size and visual quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Preset {
    /// Default preset, balanced for general use.
    #[default]
    Default,
    /// Digital picture (portrait, indoor shot).
    /// Optimizes for smooth skin tones and indoor lighting.
    Picture,
    /// Outdoor photograph with natural lighting.
    /// Best for landscapes, nature, and outdoor scenes.
    Photo,
    /// Hand or line drawing with high-contrast details.
    /// Preserves sharp edges and fine lines.
    Drawing,
    /// Small-sized colorful images like icons or sprites.
    /// Optimizes for small dimensions and sharp edges.
    Icon,
    /// Text-heavy images.
    /// Preserves text readability and sharp character edges.
    Text,
    /// Auto-detect content type from image statistics.
    /// Analyzes the image to choose the best preset automatically.
    Auto,
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Preset::Default => f.write_str("default"),
            Preset::Picture => f.write_str("picture"),
            Preset::Photo => f.write_str("photo"),
            Preset::Drawing => f.write_str("drawing"),
            Preset::Icon => f.write_str("icon"),
            Preset::Text => f.write_str("text"),
            Preset::Auto => f.write_str("auto"),
        }
    }
}

/// Color type of the image.
///
/// Note that the WebP format doesn't have a concept of color type. All images are encoded as RGBA
/// and some decoders may treat them as such. This enum is used to indicate the color type of the
/// input data provided to the encoder, which can help improve compression ratio.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelLayout {
    /// Opaque image with a single luminance byte per pixel.
    L8,
    /// Image with a luminance and alpha byte per pixel.
    La8,
    /// Opaque image with a red, green, and blue byte per pixel.
    Rgb8,
    /// Image with a red, green, blue, and alpha byte per pixel.
    Rgba8,
    /// Opaque image with a blue, green, and red byte per pixel.
    Bgr8,
    /// Image with a blue, green, red, and alpha byte per pixel.
    Bgra8,
    /// YUV 4:2:0 planar data (3 separate planes packed as \[Y, U, V\]).
    Yuv420,
}

impl PixelLayout {
    pub(crate) fn has_alpha(self) -> bool {
        matches!(
            self,
            PixelLayout::La8 | PixelLayout::Rgba8 | PixelLayout::Bgra8
        )
    }

    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            PixelLayout::L8 => 1,
            PixelLayout::La8 => 2,
            PixelLayout::Rgb8 | PixelLayout::Bgr8 => 3,
            PixelLayout::Rgba8 | PixelLayout::Bgra8 => 4,
            PixelLayout::Yuv420 => 1, // not meaningful for planar; validated separately
        }
    }
}

impl fmt::Display for PixelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelLayout::L8 => f.write_str("L8"),
            PixelLayout::La8 => f.write_str("LA8"),
            PixelLayout::Rgb8 => f.write_str("RGB8"),
            PixelLayout::Rgba8 => f.write_str("RGBA8"),
            PixelLayout::Bgr8 => f.write_str("BGR8"),
            PixelLayout::Bgra8 => f.write_str("BGRA8"),
            PixelLayout::Yuv420 => f.write_str("YUV420"),
        }
    }
}

/// Bit writer that writes to a Vec<u8> - infallible operations.
struct BitWriter<'a> {
    writer: &'a mut Vec<u8>,
    buffer: u64,
    nbits: u8,
}

impl<'a> BitWriter<'a> {
    fn new(writer: &'a mut Vec<u8>) -> Self {
        Self {
            writer,
            buffer: 0,
            nbits: 0,
        }
    }

    fn write_bits(&mut self, bits: u64, nbits: u8) {
        debug_assert!(nbits <= 64);

        self.buffer |= bits << self.nbits;
        self.nbits += nbits;

        if self.nbits >= 64 {
            self.writer.write_all(&self.buffer.to_le_bytes());
            self.nbits -= 64;
            self.buffer = bits.checked_shr(u32::from(nbits - self.nbits)).unwrap_or(0);
        }
        debug_assert!(self.nbits < 64);
    }

    fn flush(&mut self) {
        if !self.nbits.is_multiple_of(8) {
            self.write_bits(0, 8 - self.nbits % 8);
        }
        if self.nbits > 0 {
            self.writer
                .write_all(&self.buffer.to_le_bytes()[..self.nbits as usize / 8]);
            self.buffer = 0;
            self.nbits = 0;
        }
    }
}

fn write_single_entry_huffman_tree(w: &mut BitWriter<'_>, symbol: u8) {
    w.write_bits(1, 2);
    if symbol <= 1 {
        w.write_bits(0, 1);
        w.write_bits(u64::from(symbol), 1);
    } else {
        w.write_bits(1, 1);
        w.write_bits(u64::from(symbol), 8);
    }
}

fn build_huffman_tree(
    frequencies: &[u32],
    lengths: &mut [u8],
    codes: &mut [u16],
    length_limit: u8,
) -> bool {
    assert_eq!(frequencies.len(), lengths.len());
    assert_eq!(frequencies.len(), codes.len());

    if frequencies.iter().filter(|&&f| f > 0).count() <= 1 {
        lengths.fill(0);
        codes.fill(0);
        return false;
    }

    #[derive(Eq, PartialEq, Copy, Clone, Debug)]
    struct Item(u32, u16);
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> Ordering {
            other.0.cmp(&self.0)
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // Build a huffman tree
    let mut internal_nodes = Vec::new();
    let mut nodes = BinaryHeap::from_iter(
        frequencies
            .iter()
            .enumerate()
            .filter(|(_, &frequency)| frequency > 0)
            .map(|(i, &frequency)| Item(frequency, i as u16)),
    );
    while nodes.len() > 1 {
        let Item(frequency1, index1) = nodes.pop().unwrap();
        let mut root = nodes.peek_mut().unwrap();
        internal_nodes.push((index1, root.1));
        *root = Item(
            frequency1 + root.0,
            internal_nodes.len() as u16 + frequencies.len() as u16 - 1,
        );
    }

    // Walk the tree to assign code lengths
    lengths.fill(0);
    let mut stack = Vec::new();
    stack.push((nodes.pop().unwrap().1, 0));
    while let Some((node, depth)) = stack.pop() {
        let node = node as usize;
        if node < frequencies.len() {
            lengths[node] = depth as u8;
        } else {
            let (left, right) = internal_nodes[node - frequencies.len()];
            stack.push((left, depth + 1));
            stack.push((right, depth + 1));
        }
    }

    // Limit the codes to length length_limit
    let mut max_length = 0;
    for &length in lengths.iter() {
        max_length = max_length.max(length);
    }
    if max_length > length_limit {
        let mut counts = [0u32; 16];
        for &length in lengths.iter() {
            counts[length.min(length_limit) as usize] += 1;
        }

        let mut total = 0;
        for (i, count) in counts
            .iter()
            .enumerate()
            .skip(1)
            .take(length_limit as usize)
        {
            total += count << (length_limit as usize - i);
        }

        while total > 1u32 << length_limit {
            let mut i = length_limit as usize - 1;
            while counts[i] == 0 {
                i -= 1;
            }
            counts[i] -= 1;
            counts[length_limit as usize] -= 1;
            counts[i + 1] += 2;
            total -= 1;
        }

        // assign new lengths
        let mut len = length_limit;
        let mut indexes = frequencies.iter().copied().enumerate().collect::<Vec<_>>();
        indexes.sort_unstable_by_key(|&(_, frequency)| frequency);
        for &(i, frequency) in &indexes {
            if frequency > 0 {
                while counts[len as usize] == 0 {
                    len -= 1;
                }
                lengths[i] = len;
                counts[len as usize] -= 1;
            }
        }
    }

    // Assign codes
    codes.fill(0);
    let mut code = 0u32;
    for len in 1..=length_limit {
        for (i, &length) in lengths.iter().enumerate() {
            if length == len {
                codes[i] = (code as u16).reverse_bits() >> (16 - len);
                code += 1;
            }
        }
        code <<= 1;
    }
    assert_eq!(code, 2 << length_limit);

    true
}

fn write_huffman_tree(
    w: &mut BitWriter<'_>,
    frequencies: &[u32],
    lengths: &mut [u8],
    codes: &mut [u16],
) {
    if !build_huffman_tree(frequencies, lengths, codes, 15) {
        let symbol = frequencies
            .iter()
            .position(|&frequency| frequency > 0)
            .unwrap_or(0);
        write_single_entry_huffman_tree(w, symbol as u8);
        return;
    }

    let mut code_length_lengths = [0u8; 16];
    let mut code_length_codes = [0u16; 16];
    let mut code_length_frequencies = [0u32; 16];
    for &length in lengths.iter() {
        code_length_frequencies[length as usize] += 1;
    }
    let single_code_length_length = !build_huffman_tree(
        &code_length_frequencies,
        &mut code_length_lengths,
        &mut code_length_codes,
        7,
    );

    const CODE_LENGTH_ORDER: [usize; 19] = [
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];

    // Write the huffman tree
    w.write_bits(0, 1); // normal huffman tree
    w.write_bits(19 - 4, 4); // num_code_lengths - 4

    for i in CODE_LENGTH_ORDER {
        if i > 15 || code_length_frequencies[i] == 0 {
            w.write_bits(0, 3);
        } else if single_code_length_length {
            w.write_bits(1, 3);
        } else {
            w.write_bits(u64::from(code_length_lengths[i]), 3);
        }
    }

    match lengths.len() {
        256 => {
            w.write_bits(1, 1); // max_symbol is stored
            w.write_bits(3, 3); // max_symbol_nbits / 2 - 2
            w.write_bits(254, 8); // max_symbol - 2
        }
        280 => w.write_bits(0, 1),
        _ => unreachable!(),
    }

    // Write the huffman codes
    if !single_code_length_length {
        for &len in lengths.iter() {
            w.write_bits(
                u64::from(code_length_codes[len as usize]),
                code_length_lengths[len as usize],
            );
        }
    }
}

const fn length_to_symbol(len: u16) -> (u16, u8) {
    let len = len - 1;
    let highest_bit = len.ilog2() as u16;
    let second_highest_bit = (len >> (highest_bit - 1)) & 1;
    let extra_bits = highest_bit - 1;
    let symbol = 2 * highest_bit + second_highest_bit;
    (symbol, extra_bits as u8)
}

#[inline(always)]
fn count_run(pixel: &[u8], it: &mut Peekable<ChunksExact<u8>>, frequencies1: &mut [u32; 280]) {
    let mut run_length = 0;
    while run_length < 4096 && it.peek() == Some(&pixel) {
        run_length += 1;
        it.next();
    }
    if run_length > 0 {
        if run_length <= 4 {
            let symbol = 256 + run_length - 1;
            frequencies1[symbol] += 1;
        } else {
            let (symbol, _extra_bits) = length_to_symbol(run_length as u16);
            frequencies1[256 + symbol as usize] += 1;
        }
    }
}

#[inline(always)]
fn write_run(
    w: &mut BitWriter<'_>,
    pixel: &[u8],
    it: &mut Peekable<ChunksExact<u8>>,
    codes1: &[u16; 280],
    lengths1: &[u8; 280],
) {
    let mut run_length = 0;
    while run_length < 4096 && it.peek() == Some(&pixel) {
        run_length += 1;
        it.next();
    }
    if run_length > 0 {
        if run_length <= 4 {
            let symbol = 256 + run_length - 1;
            w.write_bits(u64::from(codes1[symbol]), lengths1[symbol]);
        } else {
            let (symbol, extra_bits) = length_to_symbol(run_length as u16);
            w.write_bits(
                u64::from(codes1[256 + symbol as usize]),
                lengths1[256 + symbol as usize],
            );
            w.write_bits(
                (run_length as u64 - 1) & ((1 << extra_bits) - 1),
                extra_bits,
            );
        }
    }
}

/// Allows fine-tuning some encoder parameters.
///
/// Pass to [`WebPEncoder::set_params()`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncoderParams {
    /// Use a predictor transform. Enabled by default.
    pub use_predictor_transform: bool,
    /// Use the lossy encoding to encode the image using the VP8 compression format.
    pub use_lossy: bool,
    /// A quality value for the lossy encoding that must be between 0 and 100. Defaults to 95.
    pub lossy_quality: u8,
    /// Encoding method (0-6). Higher values = better quality/compression but slower.
    /// - 0: Fastest encoding, minimal mode search
    /// - 1-2: Fast encoding with limited mode search
    /// - 3-4: Balanced speed/quality (default: 4)
    /// - 5-6: Best quality, full mode search
    pub method: u8,
    /// Spatial noise shaping strength (0-100). Controls per-segment quantization spread.
    /// Higher values give more differentiation between flat/textured regions.
    pub(crate) sns_strength: u8,
    /// Loop filter strength (0-100). Controls deblocking filter intensity.
    pub(crate) filter_strength: u8,
    /// Loop filter sharpness (0-7). Higher values reduce filter effect near edges.
    pub(crate) filter_sharpness: u8,
    /// Number of segments (1-4). More segments allow finer per-region quantization.
    pub(crate) num_segments: u8,
    /// The selected preset. When Auto, the encoder will detect content type.
    pub(crate) preset: Preset,
    /// Target file size in bytes. When > 0, enables quality search to hit target.
    /// Multi-pass encoding will adjust quality to converge on this size.
    pub(crate) target_size: u32,
    /// Target PSNR in dB. When > 0.0, enables quality search to hit target PSNR.
    pub(crate) target_psnr: f32,
    /// Use sharp (iterative) YUV conversion for higher chroma fidelity.
    /// Requires the `fast-yuv` feature. Falls back to standard conversion without it.
    pub(crate) use_sharp_yuv: bool,
    /// Alpha channel quality (0-100). 100 = lossless alpha, <100 = quantize alpha levels.
    pub(crate) alpha_quality: u8,
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            use_predictor_transform: true,
            use_lossy: false,
            lossy_quality: 95,
            method: 4, // Balanced speed/quality
            sns_strength: 50,
            filter_strength: 60,
            filter_sharpness: 0,
            num_segments: 4,
            preset: Preset::Default,
            target_size: 0,
            target_psnr: 0.0,
            use_sharp_yuv: false,
            alpha_quality: 100,
        }
    }
}

/// Progress callback for encoding. Return `Err(StopReason)` to cancel.
pub trait EncodeProgress: Send + Sync {
    /// Called with encoding progress percentage (0-100).
    fn on_progress(&self, percent: u8) -> Result<(), enough::StopReason>;
}

/// Default progress callback that does nothing.
pub struct NoProgress;

impl EncodeProgress for NoProgress {
    #[inline(always)]
    fn on_progress(&self, _: u8) -> Result<(), enough::StopReason> {
        Ok(())
    }
}

/// Statistics from an encoding operation.
///
/// Provides information about the encoded output, including PSNR measurements,
/// block counts, segment assignments, and size breakdowns.
#[derive(Debug, Clone, Default)]
pub struct EncodeStats {
    // --- Common ---
    /// Total coded output size in bytes.
    pub coded_size: u32,

    // --- Lossy VP8 ---
    /// PSNR values: \[Y, U, V, All, Alpha\] in dB.
    /// Computed from accumulated SSE during encoding (no decode needed).
    pub psnr: [f32; 5],
    /// Number of macroblocks using I4 prediction mode.
    pub block_count_i4: u32,
    /// Number of macroblocks using I16 prediction mode.
    pub block_count_i16: u32,
    /// Number of macroblocks with all-zero coefficients (skipped).
    pub block_count_skip: u32,
    /// Size of the compressed frame header in bytes.
    pub header_bytes: u32,
    /// Size of the mode partition (macroblock headers) in bytes.
    pub mode_partition_bytes: u32,
    /// Per-segment output sizes in bytes.
    pub segment_size: [u32; 4],
    /// Per-segment quantization index.
    pub segment_quant: [u8; 4],
    /// Per-segment loop filter level.
    pub segment_level: [u8; 4],
    /// Size of alpha plane data in bytes.
    pub alpha_data_size: u32,

    // --- Lossless VP8L ---
    /// Bit flags for lossless features: predictor, cross-color, sub-green, palette.
    pub lossless_features: u32,
    /// Histogram bits used for meta-Huffman coding.
    pub histogram_bits: u8,
    /// Transform bits for predictor/cross-color block size.
    pub transform_bits: u8,
    /// Color cache bits (0 = disabled).
    pub cache_bits: u8,
    /// Palette size (0 if no palette).
    pub palette_size: u16,
    /// Total lossless data size in bytes.
    pub lossless_size: u32,
    /// Lossless header size in bytes.
    pub lossless_hdr_size: u32,
    /// Lossless image data size in bytes.
    pub lossless_data_size: u32,
}

// ============================================================================
// Builder-style Encoder API (webpx-compatible)
// ============================================================================

/// WebP encoder configuration. Dimension-independent, reusable across images.
///
/// All fields are public for direct construction and inspection.
/// Builder methods are provided for ergonomic chained construction.
///
/// # Example
///
/// ```rust
/// use zenwebp::{EncoderConfig, EncodeRequest, PixelLayout, Preset};
///
/// let config = EncoderConfig::with_preset(Preset::Photo, 85.0)
///     .with_method(4);
///
/// // Reuse config for multiple images
/// let image1 = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
/// let image2 = vec![0u8; 8 * 6 * 4]; // 8x6 RGBA
/// let webp1 = EncodeRequest::new(&config, &image1, PixelLayout::Rgba8, 4, 4).encode()?;
/// let webp2 = EncodeRequest::new(&config, &image2, PixelLayout::Rgba8, 8, 6).encode()?;
/// # Ok::<(), zenwebp::EncodeError>(())
/// ```
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Encoding quality (0.0 = smallest, 100.0 = best). Default: 75.0.
    pub quality: f32,
    /// Content-aware preset. Default: `Preset::Default`.
    pub preset: Preset,
    /// Whether to use lossless compression. Default: `false`.
    pub lossless: bool,
    /// Quality/speed tradeoff (0 = fast, 6 = slower but better). Default: 4.
    pub method: u8,
    /// Near-lossless preprocessing (0 = max preprocessing, 100 = off). Default: 100.
    pub near_lossless: u8,
    /// Alpha channel quality (0-100, 100 = lossless alpha). Default: 100.
    pub alpha_quality: u8,
    /// Preserve exact RGB values under transparent areas. Default: `false`.
    pub exact: bool,
    /// Target file size in bytes (0 = disabled). Default: 0.
    pub target_size: u32,
    /// Target PSNR in dB (0.0 = disabled). Default: 0.0.
    pub target_psnr: f32,
    /// Use sharp (iterative) YUV conversion. Default: `false`.
    pub sharp_yuv: bool,
    /// Spatial noise shaping strength override (0-100). `None` = use preset default.
    pub sns_strength: Option<u8>,
    /// Loop filter strength override (0-100). `None` = use preset default.
    pub filter_strength: Option<u8>,
    /// Loop filter sharpness override (0-7). `None` = use preset default.
    pub filter_sharpness: Option<u8>,
    /// Number of segments override (1-4). `None` = use preset default.
    pub segments: Option<u8>,
    /// Encode limits for dimensions and memory validation.
    pub limits: crate::Limits,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quality: 75.0,
            preset: Preset::Default,
            lossless: false,
            method: 4,
            near_lossless: 100,
            alpha_quality: 100,
            exact: false,
            target_size: 0,
            target_psnr: 0.0,
            sharp_yuv: false,
            sns_strength: None,
            filter_strength: None,
            filter_sharpness: None,
            segments: None,
            limits: crate::Limits::none(), // No limits by default
        }
    }
}

#[allow(dead_code)]
impl EncoderConfig {
    /// Create a new encoder configuration with default settings.
    ///
    /// Default: lossy encoding at quality 75, method 4, no preset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lossless encoder configuration.
    #[must_use]
    pub fn new_lossless() -> Self {
        Self {
            lossless: true,
            quality: 75.0,
            ..Self::default()
        }
    }

    /// Create a configuration with preset and quality.
    #[must_use]
    pub fn with_preset(preset: Preset, quality: f32) -> Self {
        Self {
            preset,
            quality,
            ..Self::default()
        }
    }

    /// Set encoding quality (0.0 = smallest, 100.0 = best).
    #[must_use]
    pub fn quality(mut self, quality: f32) -> Self {
        self.quality = quality.clamp(0.0, 100.0);
        self
    }

    /// Set content-aware preset.
    #[must_use]
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = preset;
        self
    }

    /// Enable or disable lossless compression.
    #[must_use]
    pub fn lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Set quality/speed tradeoff (0 = fast, 6 = slower but better).
    #[must_use]
    pub fn method(mut self, method: u8) -> Self {
        self.method = method.min(6);
        self
    }

    /// Set near-lossless preprocessing (0 = max preprocessing, 100 = off).
    #[must_use]
    pub fn near_lossless(mut self, value: u8) -> Self {
        self.near_lossless = value.min(100);
        self
    }

    /// Set alpha plane quality (0-100, default 100).
    #[must_use]
    pub fn alpha_quality(mut self, quality: u8) -> Self {
        self.alpha_quality = quality.min(100);
        self
    }

    /// Preserve exact RGB values under transparent areas.
    #[must_use]
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }

    /// Set target file size in bytes (0 = disabled).
    #[must_use]
    pub fn target_size(mut self, size: u32) -> Self {
        self.target_size = size;
        self
    }

    /// Set target PSNR in dB (0.0 = disabled).
    /// When set, the encoder iterates quality to converge on the target PSNR.
    #[must_use]
    pub fn target_psnr(mut self, psnr: f32) -> Self {
        self.target_psnr = if psnr > 0.0 { psnr } else { 0.0 };
        self
    }

    /// Use sharp YUV conversion (slower but better quality).
    #[must_use]
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.sharp_yuv = enable;
        self
    }

    /// Set spatial noise shaping strength (0-100).
    #[must_use]
    pub fn sns_strength(mut self, strength: u8) -> Self {
        self.sns_strength = Some(strength.min(100));
        self
    }

    /// Set loop filter strength (0-100).
    #[must_use]
    pub fn filter_strength(mut self, strength: u8) -> Self {
        self.filter_strength = Some(strength.min(100));
        self
    }

    /// Set loop filter sharpness (0-7).
    #[must_use]
    pub fn filter_sharpness(mut self, sharpness: u8) -> Self {
        self.filter_sharpness = Some(sharpness.min(7));
        self
    }

    /// Set number of segments (1-4).
    #[must_use]
    pub fn segments(mut self, segments: u8) -> Self {
        self.segments = Some(segments.clamp(1, 4));
        self
    }

    /// Set encode limits for validation.
    #[must_use]
    pub fn limits(mut self, limits: crate::Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set maximum input dimensions.
    #[must_use]
    pub fn max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.limits = self.limits.max_dimensions(width, height);
        self
    }

    /// Set maximum memory usage during encoding.
    #[must_use]
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.limits = self.limits.max_memory(bytes);
        self
    }

    /// Convert to internal EncoderParams.
    pub(crate) fn to_params(&self) -> EncoderParams {
        // Base tuning values from preset (matches libwebp config_enc.c)
        // Auto uses Default values initially; the encoder overrides after analysis.
        let (sns, filter, sharp, segs) = match self.preset {
            Preset::Default | Preset::Auto => (50, 60, 0, 4),
            Preset::Picture => (80, 35, 4, 4),
            Preset::Photo => (80, 30, 3, 4),
            Preset::Drawing => (25, 10, 6, 4),
            Preset::Icon => (0, 0, 0, 4),
            Preset::Text => (0, 0, 0, 2),
        };

        EncoderParams {
            use_predictor_transform: true,
            use_lossy: !self.lossless,
            lossy_quality: super::fast_math::roundf(self.quality) as u8,
            method: self.method,
            sns_strength: self.sns_strength.unwrap_or(sns),
            filter_strength: self.filter_strength.unwrap_or(filter),
            filter_sharpness: self.filter_sharpness.unwrap_or(sharp),
            num_segments: self.segments.unwrap_or(segs),
            preset: self.preset,
            target_size: self.target_size,
            target_psnr: self.target_psnr,
            use_sharp_yuv: self.sharp_yuv,
            alpha_quality: self.alpha_quality,
        }
    }

    /// Estimate resource consumption for encoding an image with this config.
    ///
    /// Returns memory, time, and output size estimates. See
    /// [`heuristics::estimate_encode`](crate::heuristics::estimate_encode) for details.
    #[must_use]
    #[deprecated(
        since = "0.3.0",
        note = "Use LossyConfig or LosslessConfig with heuristics module"
    )]
    #[doc(hidden)]
    pub fn estimate(
        &self,
        _width: u32,
        _height: u32,
        _bpp: u8,
    ) -> crate::heuristics::EncodeEstimate {
        // Deprecated - no longer functional with new config types
        crate::heuristics::EncodeEstimate {
            peak_memory_bytes_min: 0,
            peak_memory_bytes: 0,
            peak_memory_bytes_max: 0,
            time_ms: 0.0,
            output_bytes: 0,
        }
    }
}

/// Static default progress callback (does nothing).
static NO_PROGRESS: NoProgress = NoProgress;

/// Image metadata (ICC profile, EXIF, XMP).
///
/// Use this to attach metadata to encoded images. All fields are optional.
/// This struct borrows data with lifetime `'a` to match EncodeRequest's pattern.
///
/// # Example
///
/// ```rust
/// use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ImageMetadata};
///
/// let config = LossyConfig::new();
/// let pixels = vec![255u8; 4 * 4 * 4];
/// let icc_data = vec![/* ICC data */];
/// let metadata = ImageMetadata::new()
///     .with_icc_profile(&icc_data);
///
/// let webp = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgba8, 4, 4)
///     .with_metadata(metadata)
///     .encode()?;
/// # Ok::<(), zenwebp::EncodeError>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageMetadata<'a> {
    /// ICC color profile data.
    pub icc_profile: Option<&'a [u8]>,
    /// EXIF metadata.
    pub exif: Option<&'a [u8]>,
    /// XMP metadata.
    pub xmp: Option<&'a [u8]>,
}

impl<'a> ImageMetadata<'a> {
    /// Create empty metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set ICC color profile.
    #[must_use]
    pub fn with_icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Set EXIF metadata.
    #[must_use]
    pub fn with_exif(mut self, data: &'a [u8]) -> Self {
        self.exif = Some(data);
        self
    }

    /// Set XMP metadata.
    #[must_use]
    pub fn with_xmp(mut self, data: &'a [u8]) -> Self {
        self.xmp = Some(data);
        self
    }
}

/// Internal enum to hold different configuration types.
enum ConfigKind<'a> {
    Lossy(&'a config::LossyConfig),
    Lossless(&'a config::LosslessConfig),
    Enum(&'a config::EncoderConfig),
}

impl<'a> ConfigKind<'a> {
    /// Convert to EncoderParams for internal use.
    fn to_params(&self) -> EncoderParams {
        match self {
            Self::Lossy(cfg) => cfg.to_params(),
            Self::Lossless(cfg) => cfg.to_params(),
            Self::Enum(cfg) => cfg.to_params(),
        }
    }

    /// Get the limits from the configuration.
    fn get_limits(&self) -> &crate::decoder::Limits {
        match self {
            Self::Lossy(cfg) => &cfg.limits,
            Self::Lossless(cfg) => &cfg.limits,
            Self::Enum(cfg) => cfg.get_limits(),
        }
    }
}

/// Encoding request that borrows configuration, pixel data, and optional
/// callbacks. Short-lived — created, configured, and consumed in one call chain.
///
/// # Example
///
/// ```rust
/// use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};
///
/// let config = LossyConfig::new().with_quality(85.0);
/// let rgba = vec![0u8; 640 * 480 * 4];
/// let webp = EncodeRequest::lossy(&config, &rgba, PixelLayout::Rgba8, 640, 480)
///     .encode()?;
/// # Ok::<(), zenwebp::EncodeError>(())
/// ```
pub struct EncodeRequest<'a> {
    config: ConfigKind<'a>,
    pixels: &'a [u8],
    color_type: PixelLayout,
    width: u32,
    height: u32,
    stride_pixels: Option<usize>,
    icc_profile: Option<&'a [u8]>,
    exif_metadata: Option<&'a [u8]>,
    xmp_metadata: Option<&'a [u8]>,
    stop: &'a dyn enough::Stop,
    progress: &'a dyn EncodeProgress,
}

impl<'a> EncodeRequest<'a> {
    /// Create an encoding request with a lossy configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};
    ///
    /// let config = LossyConfig::new().with_quality(85.0).with_method(4);
    /// let rgba = vec![0u8; 640 * 480 * 4];
    /// let webp = EncodeRequest::lossy(&config, &rgba, PixelLayout::Rgba8, 640, 480)
    ///     .encode()?;
    /// # Ok::<(), zenwebp::EncodeError>(())
    /// ```
    #[must_use]
    pub fn lossy(
        config: &'a config::LossyConfig,
        pixels: &'a [u8],
        color_type: PixelLayout,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            config: ConfigKind::Lossy(config),
            pixels,
            color_type,
            width,
            height,
            stride_pixels: None,
            icc_profile: None,
            exif_metadata: None,
            xmp_metadata: None,
            stop: &enough::Unstoppable,
            progress: &NO_PROGRESS,
        }
    }

    /// Create an encoding request with a lossless configuration.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenwebp::{EncodeRequest, LosslessConfig, PixelLayout};
    ///
    /// let config = LosslessConfig::new().with_quality(90.0);
    /// let rgba = vec![0u8; 640 * 480 * 4];
    /// let webp = EncodeRequest::lossless(&config, &rgba, PixelLayout::Rgba8, 640, 480)
    ///     .encode()?;
    /// # Ok::<(), zenwebp::EncodeError>(())
    /// ```
    #[must_use]
    pub fn lossless(
        config: &'a config::LosslessConfig,
        pixels: &'a [u8],
        color_type: PixelLayout,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            config: ConfigKind::Lossless(config),
            pixels,
            color_type,
            width,
            height,
            stride_pixels: None,
            icc_profile: None,
            exif_metadata: None,
            xmp_metadata: None,
            stop: &enough::Unstoppable,
            progress: &NO_PROGRESS,
        }
    }

    /// Create an encoding request with an enum configuration (runtime selection).
    ///
    /// Use this when you need to choose between lossy and lossless at runtime.
    /// For compile-time mode selection, use [`lossy()`](Self::lossy) or
    /// [`lossless()`](Self::lossless) instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenwebp::{EncodeRequest, EncoderConfig, PixelLayout};
    ///
    /// let config = EncoderConfig::new_lossy().with_quality(85.0);
    /// let rgba = vec![0u8; 640 * 480 * 4];
    /// let webp = EncodeRequest::new(&config, &rgba, PixelLayout::Rgba8, 640, 480)
    ///     .encode()?;
    /// # Ok::<(), zenwebp::EncodeError>(())
    /// ```
    #[must_use]
    pub fn new(
        config: &'a config::EncoderConfig,
        pixels: &'a [u8],
        color_type: PixelLayout,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            config: ConfigKind::Enum(config),
            pixels,
            color_type,
            width,
            height,
            stride_pixels: None,
            icc_profile: None,
            exif_metadata: None,
            xmp_metadata: None,
            stop: &enough::Unstoppable,
            progress: &NO_PROGRESS,
        }
    }

    /// Set a cooperative cancellation token.
    #[must_use]
    pub fn with_stop(mut self, stop: &'a dyn enough::Stop) -> Self {
        self.stop = stop;
        self
    }

    /// Set a progress callback.
    #[must_use]
    pub fn with_progress(mut self, progress: &'a dyn EncodeProgress) -> Self {
        self.progress = progress;
        self
    }

    /// Set row stride in pixels. Must be >= `width`.
    #[must_use]
    pub fn with_stride(mut self, stride_pixels: usize) -> Self {
        self.stride_pixels = Some(stride_pixels);
        self
    }

    /// Set ICC profile to embed.
    #[must_use]
    pub fn with_icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Set EXIF metadata to embed.
    #[must_use]
    pub fn with_exif(mut self, data: &'a [u8]) -> Self {
        self.exif_metadata = Some(data);
        self
    }

    /// Set XMP metadata to embed.
    #[must_use]
    pub fn with_xmp(mut self, data: &'a [u8]) -> Self {
        self.xmp_metadata = Some(data);
        self
    }

    /// Set all metadata at once from an [`ImageMetadata`] struct.
    ///
    /// This is a convenience method that sets ICC profile, EXIF, and XMP metadata
    /// in a single call. For setting individual fields, use [`with_icc_profile()`](Self::with_icc_profile),
    /// [`with_exif()`](Self::with_exif), or [`with_xmp()`](Self::with_xmp).
    ///
    /// # Example
    ///
    /// ```rust
    /// use zenwebp::{EncodeRequest, LossyConfig, PixelLayout, ImageMetadata};
    ///
    /// let config = LossyConfig::new();
    /// let pixels = vec![255u8; 4 * 4 * 4];
    /// let icc_data = vec![/* ICC data */];
    /// let metadata = ImageMetadata::new()
    ///     .with_icc_profile(&icc_data);
    ///
    /// let webp = EncodeRequest::lossy(&config, &pixels, PixelLayout::Rgba8, 4, 4)
    ///     .with_metadata(metadata)
    ///     .encode()?;
    /// # Ok::<(), zenwebp::EncodeError>(())
    /// ```
    #[must_use]
    pub fn with_metadata(mut self, meta: ImageMetadata<'a>) -> Self {
        if let Some(icc) = meta.icc_profile {
            self.icc_profile = Some(icc);
        }
        if let Some(exif) = meta.exif {
            self.exif_metadata = Some(exif);
        }
        if let Some(xmp) = meta.xmp {
            self.xmp_metadata = Some(xmp);
        }
        self
    }

    /// Encode to WebP bytes.
    pub fn encode(self) -> Result<Vec<u8>, EncodeError> {
        let (output, _stats) = self.encode_inner()?;
        Ok(output)
    }

    /// Encode to WebP bytes, appending to an existing Vec.
    pub fn encode_into(self, output: &mut Vec<u8>) -> Result<(), EncodeError> {
        let encoded = self.encode()?;
        output.extend_from_slice(&encoded);
        Ok(())
    }

    /// Encode to WebP bytes and return encoding statistics.
    pub fn encode_with_stats(self) -> Result<(Vec<u8>, EncodeStats), EncodeError> {
        self.encode_inner()
    }

    /// Encode to WebP, writing to an [`io::Write`](std::io::Write) implementor.
    #[cfg(feature = "std")]
    pub fn encode_to<W: std::io::Write>(self, mut writer: W) -> Result<(), EncodeError> {
        let encoded = self.encode()?;
        writer.write_all(&encoded)?;
        Ok(())
    }

    fn encode_inner(self) -> Result<(Vec<u8>, EncodeStats), EncodeError> {
        // Validate dimensions against limits
        self.config
            .get_limits()
            .check_dimensions(self.width, self.height)
            .map_err(|e| EncodeError::InvalidBufferSize(format!("{}", e)))?;

        let bpp = self.color_type.bytes_per_pixel();
        let row_bytes = self.width as usize * bpp;

        // If stride is set and differs from row width, compact the data
        let compacted;
        let encode_data = if let Some(stride_px) = self.stride_pixels {
            if stride_px < self.width as usize {
                return Err(EncodeError::InvalidBufferSize(format!(
                    "stride_pixels {} < width {}",
                    stride_px, self.width
                )));
            }
            let stride_bytes = stride_px * bpp;
            if stride_bytes == row_bytes {
                self.pixels
            } else {
                compacted = (0..self.height as usize)
                    .flat_map(|y| &self.pixels[y * stride_bytes..y * stride_bytes + row_bytes])
                    .copied()
                    .collect::<Vec<u8>>();
                &compacted
            }
        } else {
            self.pixels
        };

        if self.color_type != PixelLayout::Yuv420 {
            validate_buffer_size(encode_data.len(), self.width, self.height, bpp as u32)?;
        }

        let mut output = Vec::new();
        let stats;
        {
            let mut encoder = WebPEncoder::new(&mut output);
            encoder.set_params(self.config.to_params());
            encoder.set_stop(self.stop);
            encoder.set_progress(self.progress);
            if let Some(icc) = self.icc_profile {
                encoder.set_icc_profile(icc.to_vec());
            }
            if let Some(exif) = self.exif_metadata {
                encoder.set_exif_metadata(exif.to_vec());
            }
            if let Some(xmp) = self.xmp_metadata {
                encoder.set_xmp_metadata(xmp.to_vec());
            }
            stats = encoder.encode(encode_data, self.width, self.height, self.color_type)?;
        }
        Ok((output, stats))
    }
}

/// Validate buffer size for encoding.
fn validate_buffer_size(size: usize, width: u32, height: u32, bpp: u32) -> Result<(), EncodeError> {
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(bpp as usize);

    if size < expected {
        return Err(EncodeError::InvalidBufferSize(format!(
            "buffer too small: got {}, expected {}",
            size, expected
        )));
    }
    Ok(())
}

// ============================================================================
// Lossless encoding implementation
// ============================================================================

/// Encode image data losslessly with the indicated color type.
///
/// # Panics
///
/// Panics if the image data is not of the indicated dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame_lossless(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color: PixelLayout,
    params: EncoderParams,
    implicit_dimensions: bool,
    stop: &dyn enough::Stop,
) -> Result<(), EncodeError> {
    let w = &mut BitWriter::new(writer);

    let (is_color, is_alpha, bytes_per_pixel) = match color {
        PixelLayout::L8 => (false, false, 1),
        PixelLayout::La8 => (false, true, 2),
        PixelLayout::Rgb8 | PixelLayout::Bgr8 => (true, false, 3),
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => (true, true, 4),
        PixelLayout::Yuv420 => {
            return Err(EncodeError::InvalidBufferSize(
                "YUV 4:2:0 input only supports lossy encoding".into(),
            ));
        }
    };

    assert_eq!(
        (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
        data.len() as u64
    );

    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodeError::InvalidDimensions);
    }

    if !implicit_dimensions {
        w.write_bits(0x2f, 8); // signature
        w.write_bits(u64::from(width) - 1, 14);
        w.write_bits(u64::from(height) - 1, 14);

        w.write_bits(u64::from(is_alpha), 1); // alpha used
        w.write_bits(0x0, 3); // version
    }
    // subtract green transform
    w.write_bits(0b101, 3);

    // predictor transform
    if params.use_predictor_transform {
        w.write_bits(0b111001, 6);
        w.write_bits(0x0, 1); // no color cache
        write_single_entry_huffman_tree(w, 2);
        for _ in 0..4 {
            write_single_entry_huffman_tree(w, 0);
        }
    }

    // transforms done
    w.write_bits(0x0, 1);

    // color cache
    w.write_bits(0x0, 1);

    // meta-huffman codes
    w.write_bits(0x0, 1);

    // expand to RGBA
    let mut pixels = match color {
        PixelLayout::L8 => data.iter().flat_map(|&p| [p, p, p, 255]).collect(),
        PixelLayout::La8 => data
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        PixelLayout::Rgb8 => data
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        PixelLayout::Rgba8 => data.to_vec(),
        PixelLayout::Bgr8 => data
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0], 255]) // B,G,R → R,G,B,A
            .collect(),
        PixelLayout::Bgra8 => data
            .chunks_exact(4)
            .flat_map(|p| [p[2], p[1], p[0], p[3]]) // B,G,R,A → R,G,B,A
            .collect(),
        PixelLayout::Yuv420 => unreachable!(), // already rejected above
    };

    // compute subtract green transform
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_sub(pixel[1]);
        pixel[2] = pixel[2].wrapping_sub(pixel[1]);
    }

    // compute predictor transform
    if params.use_predictor_transform {
        let row_bytes = width as usize * 4;
        for y in (1..height as usize).rev() {
            let (prev, current) =
                pixels[(y - 1) * row_bytes..][..row_bytes * 2].split_at_mut(row_bytes);
            for (c, p) in current.iter_mut().zip(prev) {
                *c = c.wrapping_sub(*p);
            }
        }
        for i in (4..row_bytes).rev() {
            pixels[i] = pixels[i].wrapping_sub(pixels[i - 4]);
        }
        pixels[3] = pixels[3].wrapping_sub(255);
    }

    stop.check()?;

    // compute frequencies
    let mut frequencies0 = [0u32; 256];
    let mut frequencies1 = [0u32; 280];
    let mut frequencies2 = [0u32; 256];
    let mut frequencies3 = [0u32; 256];
    let mut it = pixels.chunks_exact(4).peekable();
    match color {
        PixelLayout::L8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
            frequencies3[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies1[pixel[1] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        PixelLayout::La8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies1[pixel[1] as usize] += 1;
                frequencies3[pixel[3] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        PixelLayout::Rgb8 | PixelLayout::Bgr8 => {
            // BGR already converted to RGB in pixel expansion above
            frequencies3[0] = 1;
            while let Some(pixel) = it.next() {
                frequencies0[pixel[0] as usize] += 1;
                frequencies1[pixel[1] as usize] += 1;
                frequencies2[pixel[2] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => {
            // BGRA already converted to RGBA in pixel expansion above
            while let Some(pixel) = it.next() {
                frequencies0[pixel[0] as usize] += 1;
                frequencies1[pixel[1] as usize] += 1;
                frequencies2[pixel[2] as usize] += 1;
                frequencies3[pixel[3] as usize] += 1;
                count_run(pixel, &mut it, &mut frequencies1);
            }
        }
        PixelLayout::Yuv420 => unreachable!(),
    }

    // compute and write huffman codes
    let mut lengths0 = [0u8; 256];
    let mut lengths1 = [0u8; 280];
    let mut lengths2 = [0u8; 256];
    let mut lengths3 = [0u8; 256];
    let mut codes0 = [0u16; 256];
    let mut codes1 = [0u16; 280];
    let mut codes2 = [0u16; 256];
    let mut codes3 = [0u16; 256];
    write_huffman_tree(w, &frequencies1, &mut lengths1, &mut codes1);
    if is_color {
        write_huffman_tree(w, &frequencies0, &mut lengths0, &mut codes0);
        write_huffman_tree(w, &frequencies2, &mut lengths2, &mut codes2);
    } else {
        write_single_entry_huffman_tree(w, 0);
        write_single_entry_huffman_tree(w, 0);
    }
    if is_alpha {
        write_huffman_tree(w, &frequencies3, &mut lengths3, &mut codes3);
    } else if params.use_predictor_transform {
        write_single_entry_huffman_tree(w, 0);
    } else {
        write_single_entry_huffman_tree(w, 255);
    }
    write_single_entry_huffman_tree(w, 1);

    // Write image data
    let mut it = pixels.chunks_exact(4).peekable();
    match color {
        PixelLayout::L8 => {
            while let Some(pixel) = it.next() {
                w.write_bits(
                    u64::from(codes1[pixel[1] as usize]),
                    lengths1[pixel[1] as usize],
                );
                write_run(w, pixel, &mut it, &codes1, &lengths1);
            }
        }
        PixelLayout::La8 => {
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len3 = lengths3[pixel[3] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes3[pixel[3] as usize]) << len1);

                w.write_bits(code, len1 + len3);
                write_run(w, pixel, &mut it, &codes1, &lengths1);
            }
        }
        PixelLayout::Rgb8 | PixelLayout::Bgr8 => {
            // BGR already converted to RGB in pixel expansion above
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len0 = lengths0[pixel[0] as usize];
                let len2 = lengths2[pixel[2] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes0[pixel[0] as usize]) << len1)
                    | (u64::from(codes2[pixel[2] as usize]) << (len1 + len0));

                w.write_bits(code, len1 + len0 + len2);
                write_run(w, pixel, &mut it, &codes1, &lengths1);
            }
        }
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => {
            // BGRA already converted to RGBA in pixel expansion above
            while let Some(pixel) = it.next() {
                let len1 = lengths1[pixel[1] as usize];
                let len0 = lengths0[pixel[0] as usize];
                let len2 = lengths2[pixel[2] as usize];
                let len3 = lengths3[pixel[3] as usize];

                let code = u64::from(codes1[pixel[1] as usize])
                    | (u64::from(codes0[pixel[0] as usize]) << len1)
                    | (u64::from(codes2[pixel[2] as usize]) << (len1 + len0))
                    | (u64::from(codes3[pixel[3] as usize]) << (len1 + len0 + len2));

                w.write_bits(code, len1 + len0 + len2 + len3);
                write_run(w, pixel, &mut it, &codes1, &lengths1);
            }
        }
        PixelLayout::Yuv420 => unreachable!(),
    }

    w.flush();
    Ok(())
}

/// Quantize alpha values to `num_levels` distinct values using k-means clustering.
///
/// Matches libwebp's `QuantizeLevels()` behavior: reduces distinct alpha values
/// for better compression while maintaining perceptual quality.
fn quantize_alpha_levels(data: &mut [u8], num_levels: u16) {
    if num_levels >= 256 || data.is_empty() {
        return;
    }
    let num_levels = num_levels.max(2) as usize;

    // Build histogram
    let mut histogram = [0u32; 256];
    for &v in data.iter() {
        histogram[v as usize] += 1;
    }

    // Find min/max
    let min_val = histogram.iter().position(|&c| c > 0).unwrap_or(0);
    let max_val = histogram.iter().rposition(|&c| c > 0).unwrap_or(255);

    if min_val == max_val {
        return; // single value, nothing to quantize
    }

    // Initialize cluster centers uniformly across [min, max]
    let mut centers = Vec::with_capacity(num_levels);
    for i in 0..num_levels {
        centers
            .push(min_val as f32 + (max_val - min_val) as f32 * i as f32 / (num_levels - 1) as f32);
    }

    // K-means iteration (max 6 iterations, matching libwebp)
    let mut map = [0u8; 256];
    for _ in 0..6 {
        // Assign each value to nearest center
        for val in min_val..=max_val {
            if histogram[val] == 0 {
                continue;
            }
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            for (i, &c) in centers.iter().enumerate() {
                let d = (val as f32 - c).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            map[val] = best_idx as u8;
        }

        // Recompute centers
        let mut sums = alloc::vec![0.0f64; num_levels];
        let mut counts = alloc::vec![0u64; num_levels];
        for val in min_val..=max_val {
            if histogram[val] == 0 {
                continue;
            }
            let idx = map[val] as usize;
            sums[idx] += val as f64 * histogram[val] as f64;
            counts[idx] += histogram[val] as u64;
        }

        let mut converged = true;
        for i in 0..num_levels {
            if counts[i] > 0 {
                let new_center = (sums[i] / counts[i] as f64) as f32;
                if (new_center - centers[i]).abs() > 0.5 {
                    converged = false;
                }
                centers[i] = new_center;
            }
        }

        if converged {
            break;
        }
    }

    // Build final lookup table: value → quantized value (rounded center)
    let mut lut = [0u8; 256];
    for val in min_val..=max_val {
        lut[val] = (centers[map[val] as usize] + 0.5) as u8;
    }
    // Values below min_val map to min_val's center
    for val in 0..min_val {
        lut[val] = lut[min_val];
    }
    // Values above max_val map to max_val's center
    for val in (max_val + 1)..=255 {
        lut[val] = lut[max_val];
    }

    // Apply quantization
    for v in data.iter_mut() {
        *v = lut[*v as usize];
    }
}

/// Encodes the alpha part of the image data losslessly.
/// Used for lossy images that include transparency.
///
/// # Panics
///
/// Panics if the image data is not of the indicated dimensions.
pub(crate) fn encode_alpha_lossless(
    writer: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color: PixelLayout,
    alpha_quality: u8,
    stop: &dyn enough::Stop,
) -> Result<(), EncodeError> {
    let bytes_per_pixel = match color {
        PixelLayout::La8 => 2,
        PixelLayout::Rgba8 | PixelLayout::Bgra8 => 4,
        _ => unreachable!(),
    };
    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodeError::InvalidDimensions);
    }

    let filtering_method = 0u8;
    // 0 is raw alpha data
    // 1 is using the lossless format to encode alpha data
    let compression_method = 1u8;

    // Extract alpha channel
    let mut alpha_data: Vec<u8> = data
        .iter()
        .skip(bytes_per_pixel - 1)
        .step_by(bytes_per_pixel)
        .copied()
        .collect();

    debug_assert_eq!(alpha_data.len(), (width * height) as usize);

    // Apply lossy alpha quantization if alpha_quality < 100
    let preprocessing = if alpha_quality < 100 {
        let num_levels = 1u16 + u16::from(alpha_quality) * 255 / 100;
        quantize_alpha_levels(&mut alpha_data, num_levels);
        1u8 // preprocessing = 1 signals quantized levels
    } else {
        0u8
    };

    let initial_byte = preprocessing << 4 | filtering_method << 2 | compression_method;
    writer.push(initial_byte);

    encode_frame_lossless(
        writer,
        &alpha_data,
        width,
        height,
        PixelLayout::L8,
        EncoderParams::default(),
        true,
        stop,
    )?;

    Ok(())
}

pub(crate) const fn chunk_size(inner_bytes: usize) -> u32 {
    if inner_bytes % 2 == 1 {
        (inner_bytes + 1) as u32 + 8
    } else {
        inner_bytes as u32 + 8
    }
}

pub(crate) fn write_chunk(w: &mut Vec<u8>, name: &[u8], data: &[u8]) {
    debug_assert!(name.len() == 4);

    w.write_all(name);
    w.write_u32_le(data.len() as u32);
    w.write_all(data);
    if data.len() % 2 == 1 {
        w.push(0);
    }
}

/// Internal WebP encoder (not public API — use [`EncodeRequest`] instead).
pub(crate) struct WebPEncoder<'a> {
    writer: &'a mut Vec<u8>,
    icc_profile: Vec<u8>,
    exif_metadata: Vec<u8>,
    xmp_metadata: Vec<u8>,
    params: EncoderParams,
    stop: &'a dyn enough::Stop,
    progress: &'a dyn EncodeProgress,
}

impl<'a> WebPEncoder<'a> {
    /// Create a new encoder that writes its output to `w`.
    ///
    /// Only supports "VP8L" lossless encoding.
    pub fn new(w: &'a mut Vec<u8>) -> Self {
        Self {
            writer: w,
            icc_profile: Vec::new(),
            exif_metadata: Vec::new(),
            xmp_metadata: Vec::new(),
            params: EncoderParams::default(),
            stop: &enough::Unstoppable,
            progress: &NO_PROGRESS,
        }
    }

    /// Set the ICC profile to use for the image.
    pub fn set_icc_profile(&mut self, icc_profile: Vec<u8>) {
        self.icc_profile = icc_profile;
    }

    /// Set the EXIF metadata to use for the image.
    pub fn set_exif_metadata(&mut self, exif_metadata: Vec<u8>) {
        self.exif_metadata = exif_metadata;
    }

    /// Set the XMP metadata to use for the image.
    pub fn set_xmp_metadata(&mut self, xmp_metadata: Vec<u8>) {
        self.xmp_metadata = xmp_metadata;
    }

    /// Set the `EncoderParams` to use.
    pub fn set_params(&mut self, params: EncoderParams) {
        self.params = params;
    }

    /// Set a cooperative cancellation token.
    pub fn set_stop(&mut self, stop: &'a dyn enough::Stop) {
        self.stop = stop;
    }

    /// Set a progress callback.
    pub fn set_progress(&mut self, progress: &'a dyn EncodeProgress) {
        self.progress = progress;
    }

    /// Encode image data with the indicated color type.
    ///
    /// Returns [`EncodeStats`] with information about the encoded output.
    ///
    /// # Panics
    ///
    /// Panics if the image data is not of the indicated dimensions.
    pub fn encode(
        self,
        data: &[u8],
        width: u32,
        height: u32,
        color: PixelLayout,
    ) -> Result<EncodeStats, EncodeError> {
        let mut frame = Vec::new();

        let lossy_with_alpha = self.params.use_lossy && color.has_alpha();
        let alpha_quality = self.params.alpha_quality;

        let mut stats = EncodeStats::default();

        let frame_chunk = if self.params.use_lossy {
            stats = encode_frame_lossy(
                &mut frame,
                data,
                width,
                height,
                color,
                &self.params,
                self.stop,
                self.progress,
            )?;
            b"VP8 "
        } else {
            encode_frame_lossless(
                &mut frame,
                data,
                width,
                height,
                color,
                self.params,
                false,
                self.stop,
            )?;
            b"VP8L"
        };

        // If the image has no metadata and isn't lossy with alpha,
        // it can be encoded with the "simple" WebP container format.
        let use_simple_container = self.icc_profile.is_empty()
            && self.exif_metadata.is_empty()
            && self.xmp_metadata.is_empty()
            && !lossy_with_alpha;

        if use_simple_container {
            self.writer.write_all(b"RIFF");
            self.writer.write_u32_le(chunk_size(frame.len()) + 4);
            self.writer.write_all(b"WEBP");
            write_chunk(self.writer, frame_chunk, &frame);
        } else {
            let mut total_bytes = 22 + chunk_size(frame.len());
            if !self.icc_profile.is_empty() {
                total_bytes += chunk_size(self.icc_profile.len());
            }
            if !self.exif_metadata.is_empty() {
                total_bytes += chunk_size(self.exif_metadata.len());
            }
            if !self.xmp_metadata.is_empty() {
                total_bytes += chunk_size(self.xmp_metadata.len());
            }

            let alpha_chunk_data = if lossy_with_alpha {
                let mut alpha_chunk = Vec::new();
                encode_alpha_lossless(
                    &mut alpha_chunk,
                    data,
                    width,
                    height,
                    color,
                    alpha_quality,
                    self.stop,
                )?;

                total_bytes += chunk_size(alpha_chunk.len());
                Some(alpha_chunk)
            } else {
                None
            };

            let mut flags = 0;
            if !self.xmp_metadata.is_empty() {
                flags |= 1 << 2;
            }
            if !self.exif_metadata.is_empty() {
                flags |= 1 << 3;
            }
            if color.has_alpha() {
                flags |= 1 << 4;
            }
            if !self.icc_profile.is_empty() {
                flags |= 1 << 5;
            }

            self.writer.write_all(b"RIFF");
            self.writer.write_u32_le(total_bytes);
            self.writer.write_all(b"WEBP");

            let mut vp8x = Vec::new();
            vp8x.push(flags); // flags
            vp8x.write_all(&[0; 3]); // reserved
            vp8x.write_all(&(width - 1).to_le_bytes()[..3]); // canvas width
            vp8x.write_all(&(height - 1).to_le_bytes()[..3]); // canvas height
            write_chunk(self.writer, b"VP8X", &vp8x);

            if !self.icc_profile.is_empty() {
                write_chunk(self.writer, b"ICCP", &self.icc_profile);
            }

            if let Some(alpha_chunk) = alpha_chunk_data {
                write_chunk(self.writer, b"ALPH", &alpha_chunk);
            }

            write_chunk(self.writer, frame_chunk, &frame);

            if !self.exif_metadata.is_empty() {
                write_chunk(self.writer, b"EXIF", &self.exif_metadata);
            }

            if !self.xmp_metadata.is_empty() {
                write_chunk(self.writer, b"XMP ", &self.xmp_metadata);
            }
        }

        stats.coded_size = self.writer.len() as u32;
        Ok(stats)
    }
}

#[cfg(all(test, feature = "std", not(target_arch = "wasm32")))]
mod tests {
    use rand::RngCore;

    use super::*;

    #[test]
    fn write_webp() {
        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        WebPEncoder::new(&mut output)
            .encode(&img, 256, 256, crate::PixelLayout::Rgba8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(&output).unwrap();
        let mut img2 = vec![0; 256 * 256 * 4];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);
    }

    #[test]
    fn write_webp_exif() {
        let mut img = vec![0; 256 * 256 * 3];
        rand::thread_rng().fill_bytes(&mut img);

        let mut exif = vec![0; 10];
        rand::thread_rng().fill_bytes(&mut exif);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_exif_metadata(exif.clone());
        encoder
            .encode(&img, 256, 256, crate::PixelLayout::Rgb8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(&output).unwrap();

        let mut img2 = vec![0; 256 * 256 * 3];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);

        let exif2 = decoder.exif_metadata().unwrap();
        assert_eq!(Some(exif), exif2);
    }

    #[test]
    fn roundtrip_libwebp() {
        roundtrip_libwebp_params(EncoderParams::default());
        roundtrip_libwebp_params(EncoderParams {
            use_predictor_transform: false,
            ..Default::default()
        });
    }

    fn roundtrip_libwebp_params(params: EncoderParams) {
        println!("Testing {params:?}");

        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img[..256 * 256 * 3], 256, 256, crate::PixelLayout::Rgb8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img[..256 * 256 * 3], *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img, 256, 256, crate::PixelLayout::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_icc_profile(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::PixelLayout::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_exif_metadata(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::PixelLayout::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params);
        encoder.set_xmp_metadata(vec![0; 7]);
        encoder.set_icc_profile(vec![0; 8]);
        encoder.set_icc_profile(vec![0; 9]);
        encoder
            .encode(&img, 256, 256, crate::PixelLayout::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);
    }
}
