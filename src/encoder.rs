//! Encoding of WebP images.
use std::collections::BinaryHeap;
use std::io::{self, Write};
use std::iter::FromIterator;
use std::slice::ChunksExact;

use thiserror::Error;

/// Color type of the image.
pub enum ColorType {
    L8,
    La8,
    Rgb8,
    Rgba8,
}

/// Error that can occur during encoding.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EncodingError {
    /// An IO error occurred.
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// The image dimensions are not allowed by the WebP format.
    #[error("Invalid dimensions")]
    InvalidDimensions,
}

/// WebP Encoder.
pub struct WebPEncoder<W> {
    writer: W,

    chunk_buffer: Vec<u8>,
    buffer: u64,
    nbits: u8,
}

impl<W: Write> WebPEncoder<W> {
    /// Create a new encoder that writes its output to `w`.
    ///
    /// Only supports "VP8L" lossless encoding.
    pub fn new(w: W) -> Self {
        Self {
            writer: w,
            chunk_buffer: Vec::new(),
            buffer: 0,
            nbits: 0,
        }
    }

    fn write_bits(&mut self, bits: u64, nbits: u8) -> io::Result<()> {
        debug_assert!(nbits <= 64);

        self.buffer |= bits << self.nbits;
        self.nbits += nbits;

        if self.nbits >= 64 {
            self.chunk_buffer.write_all(&self.buffer.to_le_bytes())?;
            self.nbits -= 64;
            self.buffer = bits.checked_shr((nbits - self.nbits) as u32).unwrap_or(0);
        }
        debug_assert!(self.nbits < 64);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.nbits % 8 != 0 {
            self.write_bits(0, 8 - self.nbits % 8)?;
        }
        if self.nbits > 0 {
            self.chunk_buffer
                .write_all(&self.buffer.to_le_bytes()[..self.nbits as usize / 8])
                .unwrap();
            self.buffer = 0;
            self.nbits = 0;
        }
        Ok(())
    }

    fn write_single_entry_huffman_tree(&mut self, symbol: u8) -> io::Result<()> {
        self.write_bits(1, 2)?;
        if symbol <= 1 {
            self.write_bits(0, 1)?;
            self.write_bits(symbol as u64, 1)?;
        } else {
            self.write_bits(1, 1)?;
            self.write_bits(symbol as u64, 8)?;
        }
        Ok(())
    }

    fn build_huffman_tree(
        &mut self,
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
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.0.cmp(&self.0)
            }
        }
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
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
            for &(i, frequency) in indexes.iter() {
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
        &mut self,
        frequencies: &[u32],
        lengths: &mut [u8],
        codes: &mut [u16],
    ) -> io::Result<()> {
        if !self.build_huffman_tree(frequencies, lengths, codes, 15) {
            let symbol = frequencies
                .iter()
                .position(|&frequency| frequency > 0)
                .unwrap_or(0);
            return self.write_single_entry_huffman_tree(symbol as u8);
        }

        let mut code_length_lengths = [0u8; 16];
        let mut code_length_codes = [0u16; 16];
        let mut code_length_frequencies = [0u32; 16];
        for &length in lengths.iter() {
            code_length_frequencies[length as usize] += 1;
        }
        let single_code_length_length = !self.build_huffman_tree(
            &code_length_frequencies,
            &mut code_length_lengths,
            &mut code_length_codes,
            7,
        );

        const CODE_LENGTH_ORDER: [usize; 19] = [
            17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];

        // Write the huffman tree
        self.write_bits(0, 1)?; // normal huffman tree
        self.write_bits(19 - 4, 4)?; // num_code_lengths - 4

        for &i in CODE_LENGTH_ORDER.iter() {
            if i > 15 || code_length_frequencies[i] == 0 {
                self.write_bits(0, 3)?;
            } else if single_code_length_length {
                self.write_bits(1, 3)?;
            } else {
                self.write_bits(code_length_lengths[i] as u64, 3)?;
            }
        }

        match lengths.len() {
            256 => {
                self.write_bits(1, 1)?; // max_symbol is stored
                self.write_bits(3, 3)?; // max_symbol_nbits / 2 - 2
                self.write_bits(254, 8)?; // max_symbol - 2
            }
            280 => self.write_bits(0, 1)?,
            _ => unreachable!(),
        }

        // Write the huffman codes
        if !single_code_length_length {
            for &len in lengths.iter() {
                self.write_bits(
                    code_length_codes[len as usize] as u64,
                    code_length_lengths[len as usize],
                )?;
            }
        }

        Ok(())
    }

    fn length_to_symbol(len: u16) -> (u16, u8) {
        let len = len - 1;
        let highest_bit = 15 - len.leading_zeros() as u16; // TODO: use ilog2 once MSRV >= 1.67
        let second_highest_bit = (len >> (highest_bit - 1)) & 1;
        let extra_bits = highest_bit - 1;
        let symbol = 2 * highest_bit + second_highest_bit;
        (symbol, extra_bits as u8)
    }

    #[inline(always)]
    fn count_run(
        pixel: &[u8],
        it: &mut std::iter::Peekable<ChunksExact<u8>>,
        frequencies1: &mut [u32; 280],
    ) {
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
                let (symbol, _extra_bits) = Self::length_to_symbol(run_length as u16);
                frequencies1[256 + symbol as usize] += 1;
            }
        }
    }

    #[inline(always)]
    fn write_run(
        &mut self,
        pixel: &[u8],
        it: &mut std::iter::Peekable<ChunksExact<u8>>,
        codes1: &[u16; 280],
        lengths1: &[u8; 280],
    ) -> io::Result<()> {
        let mut run_length = 0;
        while run_length < 4096 && it.peek() == Some(&pixel) {
            run_length += 1;
            it.next();
        }
        if run_length > 0 {
            if run_length <= 4 {
                let symbol = 256 + run_length - 1;
                self.write_bits(codes1[symbol] as u64, lengths1[symbol])?;
            } else {
                let (symbol, extra_bits) = Self::length_to_symbol(run_length as u16);
                self.write_bits(
                    codes1[256 + symbol as usize] as u64,
                    lengths1[256 + symbol as usize],
                )?;
                self.write_bits(
                    (run_length as u64 - 1) & ((1 << extra_bits) - 1),
                    extra_bits,
                )?;
            }
        }
        Ok(())
    }

    /// Encode image data with the indicated color type.
    ///
    /// # Panics
    ///
    /// Panics if the image data is not of the indicated dimensions.
    pub fn encode(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        color: ColorType,
    ) -> Result<(), EncodingError> {
        let (is_color, is_alpha, bytes_per_pixel) = match color {
            ColorType::L8 => (false, false, 1),
            ColorType::La8 => (false, true, 2),
            ColorType::Rgb8 => (true, false, 3),
            ColorType::Rgba8 => (true, true, 4),
        };

        assert_eq!(
            (width as u64 * height as u64).saturating_mul(bytes_per_pixel),
            data.len() as u64
        );

        if width == 0 || width > 16384 || height == 0 || height > 16384 {
            return Err(EncodingError::InvalidDimensions);
        }

        self.write_bits(0x2f, 8)?; // signature
        self.write_bits(width as u64 - 1, 14)?;
        self.write_bits(height as u64 - 1, 14)?;

        self.write_bits(is_alpha as u64, 1)?; // alpha used
        self.write_bits(0x0, 3)?; // version

        // subtract green transform
        self.write_bits(0b101, 3)?;

        // predictor transform
        self.write_bits(0b111001, 6)?;
        self.write_bits(0x0, 1)?; // no color cache
        self.write_single_entry_huffman_tree(2)?;
        for _ in 0..4 {
            self.write_single_entry_huffman_tree(0)?;
        }

        // transforms done
        self.write_bits(0x0, 1)?;

        // color cache
        self.write_bits(0x0, 1)?;

        // meta-huffman codes
        self.write_bits(0x0, 1)?;

        // expand to RGBA
        let mut pixels = match color {
            ColorType::L8 => data.iter().flat_map(|&p| [p, p, p, 255]).collect(),
            ColorType::La8 => data
                .chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
            ColorType::Rgb8 => data
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            ColorType::Rgba8 => data.to_vec(),
        };

        // compute subtract green transform
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[0] = pixel[0].wrapping_sub(pixel[1]);
            pixel[2] = pixel[2].wrapping_sub(pixel[1]);
        }

        // compute predictor transform
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

        // compute frequencies
        let mut frequencies0 = [0u32; 256];
        let mut frequencies1 = [0u32; 280];
        let mut frequencies2 = [0u32; 256];
        let mut frequencies3 = [0u32; 256];
        let mut it = pixels.chunks_exact(4).peekable();
        match color {
            ColorType::L8 => {
                frequencies0[0] = 1;
                frequencies2[0] = 1;
                frequencies3[0] = 1;
                while let Some(pixel) = it.next() {
                    frequencies1[pixel[1] as usize] += 1;
                    Self::count_run(pixel, &mut it, &mut frequencies1);
                }
            }
            ColorType::La8 => {
                frequencies0[0] = 1;
                frequencies2[0] = 1;
                while let Some(pixel) = it.next() {
                    frequencies1[pixel[1] as usize] += 1;
                    frequencies3[pixel[3] as usize] += 1;
                    Self::count_run(pixel, &mut it, &mut frequencies1);
                }
            }
            ColorType::Rgb8 => {
                frequencies3[0] = 1;
                while let Some(pixel) = it.next() {
                    frequencies0[pixel[0] as usize] += 1;
                    frequencies1[pixel[1] as usize] += 1;
                    frequencies2[pixel[2] as usize] += 1;
                    Self::count_run(pixel, &mut it, &mut frequencies1);
                }
            }
            ColorType::Rgba8 => {
                while let Some(pixel) = it.next() {
                    frequencies0[pixel[0] as usize] += 1;
                    frequencies1[pixel[1] as usize] += 1;
                    frequencies2[pixel[2] as usize] += 1;
                    frequencies3[pixel[3] as usize] += 1;
                    Self::count_run(pixel, &mut it, &mut frequencies1);
                }
            }
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
        self.write_huffman_tree(&frequencies1, &mut lengths1, &mut codes1)?;
        if is_color {
            self.write_huffman_tree(&frequencies0, &mut lengths0, &mut codes0)?;
            self.write_huffman_tree(&frequencies2, &mut lengths2, &mut codes2)?;
        } else {
            self.write_single_entry_huffman_tree(0)?;
            self.write_single_entry_huffman_tree(0)?;
        }
        if is_alpha {
            self.write_huffman_tree(&frequencies3, &mut lengths3, &mut codes3)?;
        } else {
            self.write_single_entry_huffman_tree(0)?;
        }
        self.write_single_entry_huffman_tree(1)?;

        // Write image data
        let mut it = pixels.chunks_exact(4).peekable();
        match color {
            ColorType::L8 => {
                while let Some(pixel) = it.next() {
                    self.write_bits(
                        codes1[pixel[1] as usize] as u64,
                        lengths1[pixel[1] as usize],
                    )?;
                    self.write_run(pixel, &mut it, &codes1, &lengths1)?;
                }
            }
            ColorType::La8 => {
                while let Some(pixel) = it.next() {
                    let len1 = lengths1[pixel[1] as usize];
                    let len3 = lengths3[pixel[3] as usize];

                    let code = codes1[pixel[1] as usize] as u64
                        | (codes3[pixel[3] as usize] as u64) << len1;

                    self.write_bits(code, len1 + len3)?;
                    self.write_run(pixel, &mut it, &codes1, &lengths1)?;
                }
            }
            ColorType::Rgb8 => {
                while let Some(pixel) = it.next() {
                    let len1 = lengths1[pixel[1] as usize];
                    let len0 = lengths0[pixel[0] as usize];
                    let len2 = lengths2[pixel[2] as usize];

                    let code = codes1[pixel[1] as usize] as u64
                        | (codes0[pixel[0] as usize] as u64) << len1
                        | (codes2[pixel[2] as usize] as u64) << (len1 + len0);

                    self.write_bits(code, len1 + len0 + len2)?;
                    self.write_run(pixel, &mut it, &codes1, &lengths1)?;
                }
            }
            ColorType::Rgba8 => {
                while let Some(pixel) = it.next() {
                    let len1 = lengths1[pixel[1] as usize];
                    let len0 = lengths0[pixel[0] as usize];
                    let len2 = lengths2[pixel[2] as usize];
                    let len3 = lengths3[pixel[3] as usize];

                    let code = codes1[pixel[1] as usize] as u64
                        | (codes0[pixel[0] as usize] as u64) << len1
                        | (codes2[pixel[2] as usize] as u64) << (len1 + len0)
                        | (codes3[pixel[3] as usize] as u64) << (len1 + len0 + len2);

                    self.write_bits(code, len1 + len0 + len2 + len3)?;
                    self.write_run(pixel, &mut it, &codes1, &lengths1)?;
                }
            }
        }

        // flush writer
        self.flush()?;
        if self.chunk_buffer.len() % 2 == 1 {
            self.chunk_buffer.push(0);
        }

        // write container
        self.writer.write_all(b"RIFF")?;
        self.writer
            .write_all(&(self.chunk_buffer.len() as u32 + 12).to_le_bytes())?;
        self.writer.write_all(b"WEBP")?;
        self.writer.write_all(b"VP8L")?;
        self.writer
            .write_all(&(self.chunk_buffer.len() as u32).to_le_bytes())?;
        self.writer.write_all(&self.chunk_buffer)?;

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
}
