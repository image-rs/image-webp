//! Configurable limits for WebP decoding.
//!
//! These limits protect against malicious or malformed inputs that could
//! cause excessive memory usage or processing time.

use super::api::DecodeError;

/// Configuration for decode limits.
///
/// All limits are optional; `None` means unlimited.
///
/// # Example
///
/// ```rust
/// use zenwebp::Limits;
///
/// // Start with defaults and customize
/// let limits = Limits::default()
///     .max_dimensions(4096, 4096)
///     .max_memory(256 * 1024 * 1024);  // 256 MB
///
/// // Or start with no limits for trusted inputs
/// let unlimited = Limits::none();
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Limits {
    /// Maximum image width in pixels.
    pub max_width: Option<u32>,

    /// Maximum image height in pixels.
    pub max_height: Option<u32>,

    /// Maximum total pixels (width * height).
    /// Useful for limiting memory even with odd aspect ratios.
    pub max_total_pixels: Option<u64>,

    /// Maximum number of frames in an animation.
    pub max_frame_count: Option<u64>,

    /// Maximum input file size in bytes.
    pub max_file_size: Option<u64>,

    /// Maximum memory usage in bytes during decoding.
    pub max_memory: Option<u64>,
}

impl Default for Limits {
    /// Default limits suitable for server-side use.
    ///
    /// - Max dimensions: 16384 x 16384 (WebP format max)
    /// - Max total pixels: 100 megapixels
    /// - Max frames: 10,000
    /// - Max file size: 100 MB
    /// - Max memory: 1 GB
    fn default() -> Self {
        Self {
            max_width: Some(16384),
            max_height: Some(16384),
            max_total_pixels: Some(100_000_000),
            max_frame_count: Some(10_000),
            max_file_size: Some(100 * 1024 * 1024),
            max_memory: Some(1024 * 1024 * 1024),
        }
    }
}

impl Limits {
    /// Create limits with no restrictions.
    ///
    /// **Warning**: Only use this for trusted inputs!
    #[must_use]
    pub fn none() -> Self {
        Self {
            max_width: None,
            max_height: None,
            max_total_pixels: None,
            max_frame_count: None,
            max_file_size: None,
            max_memory: None,
        }
    }

    /// Set maximum dimensions.
    #[must_use]
    pub fn max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.max_width = Some(width);
        self.max_height = Some(height);
        self
    }

    /// Set maximum total pixels.
    #[must_use]
    pub fn max_total_pixels(mut self, pixels: u64) -> Self {
        self.max_total_pixels = Some(pixels);
        self
    }

    /// Set maximum frame count.
    #[must_use]
    pub fn max_frame_count(mut self, count: u64) -> Self {
        self.max_frame_count = Some(count);
        self
    }

    /// Set maximum file size in bytes.
    #[must_use]
    pub fn max_file_size(mut self, bytes: u64) -> Self {
        self.max_file_size = Some(bytes);
        self
    }

    /// Set maximum memory usage in bytes.
    #[must_use]
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = Some(bytes);
        self
    }

    /// Check if dimensions are within limits.
    pub fn check_dimensions(&self, width: u32, height: u32) -> Result<(), DecodeError> {
        if let Some(max_w) = self.max_width
            && width > max_w
        {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "width {} exceeds limit {}",
                width,
                max_w
            )));
        }

        if let Some(max_h) = self.max_height
            && height > max_h
        {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "height {} exceeds limit {}",
                height,
                max_h
            )));
        }

        let total_pixels = width as u64 * height as u64;
        if let Some(max_pixels) = self.max_total_pixels
            && total_pixels > max_pixels
        {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "total pixels {} exceeds limit {}",
                total_pixels,
                max_pixels
            )));
        }

        Ok(())
    }

    /// Check if frame count is within limits.
    pub fn check_frame_count(&self, count: usize) -> Result<(), DecodeError> {
        if let Some(max) = self.max_frame_count
            && count as u64 >= max
        {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "frame count {} exceeds limit {}",
                count,
                max
            )));
        }
        Ok(())
    }

    /// Check if file size is within limits.
    pub fn check_file_size(&self, size: u64) -> Result<(), DecodeError> {
        if let Some(max) = self.max_file_size
            && size > max
        {
            return Err(DecodeError::InvalidParameter(alloc::format!(
                "file size {} bytes exceeds limit {} bytes",
                size,
                max
            )));
        }
        Ok(())
    }

    /// Check if memory usage is within limits.
    pub fn check_memory(&self, bytes: usize) -> Result<(), DecodeError> {
        if let Some(max) = self.max_memory
            && bytes as u64 > max
        {
            return Err(DecodeError::MemoryLimitExceeded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits() {
        let limits = Limits::default();
        assert!(limits.max_width.is_some());
        assert!(limits.max_height.is_some());
    }

    #[test]
    fn check_dimensions_ok() {
        let limits = Limits::default().max_dimensions(1000, 1000);
        assert!(limits.check_dimensions(500, 500).is_ok());
        assert!(limits.check_dimensions(1000, 1000).is_ok());
    }

    #[test]
    fn check_dimensions_too_large() {
        let limits = Limits::default().max_dimensions(1000, 1000);
        let result = limits.check_dimensions(1001, 500);
        assert!(result.is_err());
    }

    #[test]
    fn check_total_pixels() {
        let limits = Limits::default().max_total_pixels(1_000_000);
        assert!(limits.check_dimensions(1000, 1000).is_ok());
        assert!(limits.check_dimensions(1001, 1000).is_err());
    }

    #[test]
    fn no_limits() {
        let limits = Limits::none();
        assert!(limits.check_dimensions(u32::MAX, u32::MAX).is_ok());
        assert!(limits.check_frame_count(usize::MAX).is_ok());
    }

    #[test]
    fn builder_pattern() {
        let limits = Limits::default()
            .max_dimensions(4096, 4096)
            .max_frame_count(100)
            .max_file_size(10 * 1024 * 1024)
            .max_memory(512 * 1024 * 1024);

        assert_eq!(limits.max_width, Some(4096));
        assert_eq!(limits.max_frame_count, Some(100));
    }
}
