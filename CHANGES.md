# Release Notes

### Version 0.2.1

Changes:
 - Increased the required Rust compiler version to v1.80

Optimizations:
 - Removed bounds checks from hot loops in `read_coefficients()` (#121)
 - Faster YUV -> RGBA conversion for a 7% speedup on lossy RGBA images (#122)
 - Faster alpha blending for up to 20% speedup on animated images (#123)
 - Much faster arithmetic decoding for up to 30% speedup on lossy images (#124)
 - Avoid unnecessarily cloning image data for a 4% speedup (#126)

### Version 0.2.0

Breaking Changes:
- `WebPDecoder` now requires the passed reader implement `BufRead`.

Changes:
- Add `EncoderParams` to make predictor transform optional.

Bug Fixes:
- Several bug fixes in animation compositing.
- Fix indexing for filling image regions with tivial huffman codes.
- Properly update the color cache when trivial huffman codes are used.

Optimizations:
- Substantially faster decoding of lossless images, by switching to a
  table-based Huffman decoder and a variety of smaller optimizations.

### Version 0.1.3

Changes:
- Accept files with out-of-order "unknown" chunks.
- Switched to `quick-error` crate for faster compliation.

Bug Fixes:
- Fixed decoding of animations with ALPH chunks.
- Fixed encoding bug for extended WebP files.
- Resolved compliation problem with fuzz targets.

Optimizations:
- Faster YUV to RGB conversion.
- Improved `BitReader` logic.
- In-place decoding of lossless RGBA images.

### Version 0.1.2

- Export `decoder::LoopCount`.
- Fix decode bug in `read_quantization_indices` that caused some lossy images to
  appear washed out.
- Switch to `byteorder-lite` crate for byte order conversions.

### Version 0.1.1

- Fix RIFF size calculation in encoder.

### Version 0.1.0

- Initial release
