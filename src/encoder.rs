//! Encoding of WebP images.
use std::io::{self, Write};

use quick_error::quick_error;

use crate::{
    lossless::{encode_frame_lossless, EncoderParams},
    lossy::encoder::encode_frame_lossy,
};

/// Color type of the image.
///
/// Note that the WebP format doesn't have a concept of color type. All images are encoded as RGBA
/// and some decoders may treat them as such. This enum is used to indicate the color type of the
/// input data provided to the encoder, which can help improve compression ratio.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorType {
    /// Opaque image with a single luminance byte per pixel.
    L8,
    /// Image with a luminance and alpha byte per pixel.
    La8,
    /// Opaque image with a red, green, and blue byte per pixel.
    Rgb8,
    /// Image with a red, green, blue, and alpha byte per pixel.
    Rgba8,
}

impl ColorType {
    fn has_alpha(self) -> bool {
        self == ColorType::La8 || self == ColorType::Rgba8
    }
}

quick_error! {
    /// Error that can occur during encoding.
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum EncodingError {
        /// An IO error occurred.
        IoError(err: io::Error) {
            from()
            display("IO error: {}", err)
            source(err)
        }

        /// The image dimensions are not allowed by the WebP format.
        InvalidDimensions {
            display("Invalid dimensions")
        }
    }
}

/// Encodes the alpha part of the image data losslessly.
/// Used for lossy images that include transparency.
///
/// # Panics
///
/// Panics if the image data is not of the indicated dimensions.
fn encode_alpha_lossless<W: Write>(
    mut writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
) -> Result<(), EncodingError> {
    let bytes_per_pixel = match color {
        ColorType::La8 => 2,
        ColorType::Rgba8 => 4,
        _ => unreachable!(),
    };
    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodingError::InvalidDimensions);
    }

    let preprocessing = 0u8;
    let filtering_method = 0u8;
    // 0 is raw alpha data
    // 1 is using the lossless format to encode alpha data
    let compression_method = 1u8;

    let initial_byte = preprocessing << 4 | filtering_method << 2 | compression_method;

    writer.write_all(&[initial_byte])?;

    // uncompressed raw alpha data
    let alpha_data: Vec<u8> = data
        .iter()
        .skip(bytes_per_pixel - 1)
        .step_by(bytes_per_pixel)
        .copied()
        .collect();

    debug_assert_eq!(alpha_data.len(), (width * height) as usize);

    encode_frame_lossless(
        writer,
        &alpha_data,
        width,
        height,
        ColorType::L8,
        EncoderParams::default(),
        true,
    )?;

    Ok(())
}

const fn chunk_size(inner_bytes: usize) -> u32 {
    if inner_bytes % 2 == 1 {
        (inner_bytes + 1) as u32 + 8
    } else {
        inner_bytes as u32 + 8
    }
}

fn write_chunk<W: Write>(mut w: W, name: &[u8], data: &[u8]) -> io::Result<()> {
    debug_assert!(name.len() == 4);

    w.write_all(name)?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(data)?;
    if data.len() % 2 == 1 {
        w.write_all(&[0])?;
    }
    Ok(())
}

/// WebP Encoder.
pub struct WebPEncoder<W> {
    writer: W,
    icc_profile: Vec<u8>,
    exif_metadata: Vec<u8>,
    xmp_metadata: Vec<u8>,
    params: EncoderParams,
}

impl<W: Write> WebPEncoder<W> {
    /// Create a new encoder that writes its output to `w`.
    ///
    /// Only supports "VP8L" lossless encoding.
    pub fn new(w: W) -> Self {
        Self {
            writer: w,
            icc_profile: Vec::new(),
            exif_metadata: Vec::new(),
            xmp_metadata: Vec::new(),
            params: EncoderParams::default(),
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

    /// Encode image data with the indicated color type.
    ///
    /// # Panics
    ///
    /// Panics if the image data is not of the indicated dimensions.
    pub fn encode(
        mut self,
        data: &[u8],
        width: u32,
        height: u32,
        color: ColorType,
    ) -> Result<(), EncodingError> {
        let mut frame = Vec::new();

        let lossy_with_alpha = self.params.use_lossy && color.has_alpha();

        let frame_chunk = if self.params.use_lossy {
            encode_frame_lossy(
                &mut frame,
                data,
                width,
                height,
                color,
                self.params.lossy_quality,
            )?;
            b"VP8 "
        } else {
            encode_frame_lossless(&mut frame, data, width, height, color, self.params, false)?;
            b"VP8L"
        };

        // If the image has no metadata and isn't lossy with alpha,
        // it can be encoded with the "simple" WebP container format.
        let use_simple_container = self.icc_profile.is_empty()
            && self.exif_metadata.is_empty()
            && self.xmp_metadata.is_empty()
            && !lossy_with_alpha;

        if use_simple_container {
            self.writer.write_all(b"RIFF")?;
            self.writer
                .write_all(&(chunk_size(frame.len()) + 4).to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;
            write_chunk(&mut self.writer, frame_chunk, &frame)?;
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
                encode_alpha_lossless(&mut alpha_chunk, data, width, height, color)?;

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

            self.writer.write_all(b"RIFF")?;
            self.writer.write_all(&total_bytes.to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;

            let mut vp8x = Vec::new();
            vp8x.write_all(&[flags])?; // flags
            vp8x.write_all(&[0; 3])?; // reserved
            vp8x.write_all(&(width - 1).to_le_bytes()[..3])?; // canvas width
            vp8x.write_all(&(height - 1).to_le_bytes()[..3])?; // canvas height
            write_chunk(&mut self.writer, b"VP8X", &vp8x)?;

            if !self.icc_profile.is_empty() {
                write_chunk(&mut self.writer, b"ICCP", &self.icc_profile)?;
            }

            if let Some(alpha_chunk) = alpha_chunk_data {
                write_chunk(&mut self.writer, b"ALPH", &alpha_chunk)?;
            }

            write_chunk(&mut self.writer, frame_chunk, &frame)?;

            if !self.exif_metadata.is_empty() {
                write_chunk(&mut self.writer, b"EXIF", &self.exif_metadata)?;
            }

            if !self.xmp_metadata.is_empty() {
                write_chunk(&mut self.writer, b"XMP ", &self.xmp_metadata)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    #[test]
    fn write_webp() {
        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        WebPEncoder::new(&mut output)
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();
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
            .encode(&img, 256, 256, crate::ColorType::Rgb8)
            .unwrap();

        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();

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
            .encode(&img[..256 * 256 * 3], 256, 256, crate::ColorType::Rgb8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img[..256 * 256 * 3], *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_icc_profile(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_exif_metadata(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
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
            .encode(&img, 256, 256, crate::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);
    }
}
