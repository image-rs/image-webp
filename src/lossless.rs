//! Decoding of lossless WebP images
//!
//! [Lossless spec](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)
//!

use std::{io::Read, mem};

use byteorder_lite::ReadBytesExt;

use crate::decoder::DecodingError;

use super::huffman::HuffmanTree;
use super::lossless_transform::{add_pixels, TransformType};

const CODE_LENGTH_CODES: usize = 19;
const CODE_LENGTH_CODE_ORDER: [usize; CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];

const GREEN: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DIST: usize = 4;

const HUFFMAN_CODES_PER_META_CODE: usize = 5;

type HuffmanCodeGroup = [HuffmanTree; HUFFMAN_CODES_PER_META_CODE];

const ALPHABET_SIZE: [u16; HUFFMAN_CODES_PER_META_CODE] = [256 + 24, 256, 256, 256, 40];

#[inline]
pub(crate) fn subsample_size(size: u16, bits: u8) -> u16 {
    ((u32::from(size) + (1u32 << bits) - 1) >> bits)
        .try_into()
        .unwrap()
}

const NUM_TRANSFORM_TYPES: usize = 4;

//Decodes lossless WebP images
#[derive(Debug)]
pub(crate) struct LosslessDecoder<R> {
    bit_reader: BitReader<R>,
    frame: LosslessFrame,
    transforms: [Option<TransformType>; NUM_TRANSFORM_TYPES],
    transform_order: Vec<u8>,
}

impl<R: Read> LosslessDecoder<R> {
    /// Create a new decoder
    pub(crate) fn new(r: R) -> LosslessDecoder<R> {
        LosslessDecoder {
            bit_reader: BitReader::new(r),
            frame: Default::default(),
            transforms: [None, None, None, None],
            transform_order: Vec::new(),
        }
    }

    /// Decodes a frame.
    ///
    /// In an alpha chunk the width and height are not included in the header, so they should be
    /// provided by setting the `implicit_dimensions` argument. Otherwise that argument should be
    /// `None` and the frame dimensions will be determined by reading the VP8L header.
    pub(crate) fn decode_frame(
        &mut self,
        implicit_dimensions: Option<(u16, u16)>,
    ) -> Result<&LosslessFrame, DecodingError> {
        match implicit_dimensions {
            Some((width, height)) => {
                self.frame.width = width;
                self.frame.height = height;
            }
            None => {
                let signature = self.bit_reader.read_bits::<u8>(8)?;
                if signature != 0x2f {
                    return Err(DecodingError::LosslessSignatureInvalid(signature));
                }

                self.frame.width = self.bit_reader.read_bits::<u16>(14)? + 1;
                self.frame.height = self.bit_reader.read_bits::<u16>(14)? + 1;

                let _alpha_used = self.bit_reader.read_bits::<u8>(1)?;
                let version_num = self.bit_reader.read_bits::<u8>(3)?;
                if version_num != 0 {
                    return Err(DecodingError::VersionNumberInvalid(version_num));
                }
            }
        }

        let transformed_width = self.read_transforms()?;
        let mut data = self.decode_image_stream(transformed_width, self.frame.height, true)?;

        let mut width = transformed_width;
        for &trans_index in self.transform_order.iter().rev() {
            let transform = self.transforms[usize::from(trans_index)].as_ref().unwrap();
            if let TransformType::ColorIndexingTransform { .. } = transform {
                width = self.frame.width;
            }
            transform.apply_transform(&mut data, width, self.frame.height)?;
        }

        self.frame.buf = data;
        Ok(&self.frame)
    }

    /// Reads Image data from the bitstream
    ///
    /// Can be in any of the 5 roles described in the Specification. ARGB Image role has different
    /// behaviour to the other 4. xsize and ysize describe the size of the blocks where each block
    /// has its own entropy code
    fn decode_image_stream(
        &mut self,
        xsize: u16,
        ysize: u16,
        is_argb_img: bool,
    ) -> Result<Vec<u32>, DecodingError> {
        let color_cache_bits = self.read_color_cache()?;
        let color_cache = color_cache_bits.map(|bits| {
            let size = 1 << bits;
            let cache = vec![0u32; size];
            ColorCache {
                color_cache_bits: bits,
                color_cache: cache,
            }
        });

        let huffman_info = self.read_huffman_codes(is_argb_img, xsize, ysize, color_cache)?;
        self.decode_image_data(xsize, ysize, huffman_info)
    }

    /// Reads transforms and their data from the bitstream
    fn read_transforms(&mut self) -> Result<u16, DecodingError> {
        let mut xsize = self.frame.width;

        while self.bit_reader.read_bits::<u8>(1)? == 1 {
            let transform_type_val = self.bit_reader.read_bits::<u8>(2)?;

            if self.transforms[usize::from(transform_type_val)].is_some() {
                //can only have one of each transform, error
                return Err(DecodingError::TransformError);
            }

            self.transform_order.push(transform_type_val);

            let transform_type = match transform_type_val {
                0 => {
                    //predictor

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.frame.height, size_bits);

                    let data = self.decode_image_stream(block_xsize, block_ysize, false)?;

                    TransformType::PredictorTransform {
                        size_bits,
                        predictor_data: data,
                    }
                }
                1 => {
                    //color transform

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.frame.height, size_bits);

                    let data = self.decode_image_stream(block_xsize, block_ysize, false)?;

                    TransformType::ColorTransform {
                        size_bits,
                        transform_data: data,
                    }
                }
                2 => {
                    //subtract green

                    TransformType::SubtractGreen
                }
                3 => {
                    let color_table_size = self.bit_reader.read_bits::<u16>(8)? + 1;

                    let mut color_map = self.decode_image_stream(color_table_size, 1, false)?;

                    let bits = if color_table_size <= 2 {
                        3
                    } else if color_table_size <= 4 {
                        2
                    } else if color_table_size <= 16 {
                        1
                    } else {
                        0
                    };
                    xsize = subsample_size(xsize, bits);

                    Self::adjust_color_map(&mut color_map);

                    TransformType::ColorIndexingTransform {
                        table_size: color_table_size,
                        table_data: color_map,
                    }
                }
                _ => unreachable!(),
            };

            self.transforms[usize::from(transform_type_val)] = Some(transform_type);
        }

        Ok(xsize)
    }

    /// Adjusts the color map since it's subtraction coded
    fn adjust_color_map(color_map: &mut [u32]) {
        for i in 1..color_map.len() {
            color_map[i] = add_pixels(color_map[i], color_map[i - 1]);
        }
    }

    /// Reads huffman codes associated with an image
    fn read_huffman_codes(
        &mut self,
        read_meta: bool,
        xsize: u16,
        ysize: u16,
        color_cache: Option<ColorCache>,
    ) -> Result<HuffmanInfo, DecodingError> {
        let mut num_huff_groups = 1;

        let mut huffman_bits = 0;
        let mut huffman_xsize = 1;
        let mut huffman_ysize = 1;
        let mut entropy_image = Vec::new();

        if read_meta && self.bit_reader.read_bits::<u8>(1)? == 1 {
            //meta huffman codes
            huffman_bits = self.bit_reader.read_bits::<u8>(3)? + 2;
            huffman_xsize = subsample_size(xsize, huffman_bits);
            huffman_ysize = subsample_size(ysize, huffman_bits);

            entropy_image = self.decode_image_stream(huffman_xsize, huffman_ysize, false)?;

            for pixel in entropy_image.iter_mut() {
                let meta_huff_code = (*pixel >> 8) & 0xffff;

                *pixel = meta_huff_code;

                if meta_huff_code >= num_huff_groups {
                    num_huff_groups = meta_huff_code + 1;
                }
            }
        }

        let mut hufftree_groups = Vec::new();

        for _i in 0..num_huff_groups {
            let mut group: HuffmanCodeGroup = Default::default();
            for j in 0..HUFFMAN_CODES_PER_META_CODE {
                let mut alphabet_size = ALPHABET_SIZE[j];
                if j == 0 {
                    if let Some(color_cache) = color_cache.as_ref() {
                        alphabet_size += 1 << color_cache.color_cache_bits;
                    }
                }

                let tree = self.read_huffman_code(alphabet_size)?;
                group[j] = tree;
            }
            hufftree_groups.push(group);
        }

        let huffman_mask = if huffman_bits == 0 {
            !0
        } else {
            (1 << huffman_bits) - 1
        };

        let info = HuffmanInfo {
            xsize: huffman_xsize,
            _ysize: huffman_ysize,
            color_cache,
            image: entropy_image,
            bits: huffman_bits,
            mask: huffman_mask,
            huffman_code_groups: hufftree_groups,
        };

        Ok(info)
    }

    /// Decodes and returns a single huffman tree
    fn read_huffman_code(&mut self, alphabet_size: u16) -> Result<HuffmanTree, DecodingError> {
        let simple = self.bit_reader.read_bits::<u8>(1)? == 1;

        if simple {
            let num_symbols = self.bit_reader.read_bits::<u8>(1)? + 1;

            let mut code_lengths = vec![u16::from(num_symbols - 1)];
            let mut codes = vec![0];
            let mut symbols = Vec::new();

            let is_first_8bits = self.bit_reader.read_bits::<u8>(1)?;
            symbols.push(self.bit_reader.read_bits::<u16>(1 + 7 * is_first_8bits)?);

            if num_symbols == 2 {
                symbols.push(self.bit_reader.read_bits::<u16>(8)?);
                code_lengths.push(1);
                codes.push(1);
            }

            if symbols.iter().any(|&s| s > alphabet_size) {
                return Err(DecodingError::BitStreamError);
            }

            HuffmanTree::build_explicit(code_lengths, codes, symbols)
        } else {
            let mut code_length_code_lengths = vec![0; CODE_LENGTH_CODES];

            let num_code_lengths = 4 + self.bit_reader.read_bits::<usize>(4)?;
            for i in 0..num_code_lengths {
                code_length_code_lengths[CODE_LENGTH_CODE_ORDER[i]] =
                    self.bit_reader.read_bits(3)?;
            }

            let new_code_lengths =
                self.read_huffman_code_lengths(code_length_code_lengths, alphabet_size)?;

            HuffmanTree::build_implicit(new_code_lengths)
        }
    }

    /// Reads huffman code lengths
    fn read_huffman_code_lengths(
        &mut self,
        code_length_code_lengths: Vec<u16>,
        num_symbols: u16,
    ) -> Result<Vec<u16>, DecodingError> {
        let table = HuffmanTree::build_implicit(code_length_code_lengths)?;

        let mut max_symbol = if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let length_nbits = 2 + 2 * self.bit_reader.read_bits::<u8>(3)?;
            let max_minus_two = self.bit_reader.read_bits::<u16>(length_nbits)?;
            if max_minus_two > num_symbols - 2 {
                return Err(DecodingError::BitStreamError);
            }
            2 + max_minus_two
        } else {
            num_symbols
        };

        let mut code_lengths = vec![0; usize::from(num_symbols)];
        let mut prev_code_len = 8; //default code length

        let mut symbol = 0;
        while symbol < num_symbols {
            if max_symbol == 0 {
                break;
            }
            max_symbol -= 1;

            let code_len = table.read_symbol(&mut self.bit_reader)?;

            if code_len < 16 {
                code_lengths[usize::from(symbol)] = code_len;
                symbol += 1;
                if code_len != 0 {
                    prev_code_len = code_len;
                }
            } else {
                let use_prev = code_len == 16;
                let slot = code_len - 16;
                let extra_bits = match slot {
                    0 => 2,
                    1 => 3,
                    2 => 7,
                    _ => return Err(DecodingError::BitStreamError),
                };
                let repeat_offset = match slot {
                    0 | 1 => 3,
                    2 => 11,
                    _ => return Err(DecodingError::BitStreamError),
                };

                let mut repeat = self.bit_reader.read_bits::<u16>(extra_bits)? + repeat_offset;

                if symbol + repeat > num_symbols {
                    return Err(DecodingError::BitStreamError);
                } else {
                    let length = if use_prev { prev_code_len } else { 0 };
                    while repeat > 0 {
                        repeat -= 1;
                        code_lengths[usize::from(symbol)] = length;
                        symbol += 1;
                    }
                }
            }
        }

        Ok(code_lengths)
    }

    /// Decodes the image data using the huffman trees and either of the 3 methods of decoding
    fn decode_image_data(
        &mut self,
        width: u16,
        height: u16,
        mut huffman_info: HuffmanInfo,
    ) -> Result<Vec<u32>, DecodingError> {
        let num_values = usize::from(width) * usize::from(height);
        let mut data = vec![0; num_values];

        let huff_index = huffman_info.get_huff_index(0, 0);
        let mut tree = &huffman_info.huffman_code_groups[huff_index];
        let mut last_cached = 0;
        let mut index = 0;

        let mut next_block_start = 0;
        while index < num_values {
            if index >= next_block_start {
                let x = index % usize::from(width);
                let y = index / usize::from(width);
                next_block_start = (x | usize::from(huffman_info.mask)).min(usize::from(width - 1))
                    + y * usize::from(width)
                    + 1;

                let huff_index = huffman_info.get_huff_index(x as u16, y as u16);
                tree = &huffman_info.huffman_code_groups[huff_index];

                // Fast path: If all the codes each contain only a single
                // symbol, then the pixel data isn't written to the bitstream
                // and we can just fill the output buffer with the symbol
                // directly.
                if tree[..4].iter().all(|t| t.is_single_node()) {
                    let code = tree[GREEN].read_symbol(&mut self.bit_reader)?;
                    if code < 256 {
                        let n = if huffman_info.bits == 0 {
                            num_values
                        } else {
                            next_block_start - index
                        };

                        let red = tree[RED].read_symbol(&mut self.bit_reader)?;
                        let blue = tree[BLUE].read_symbol(&mut self.bit_reader)?;
                        let alpha = tree[ALPHA].read_symbol(&mut self.bit_reader)?;
                        let value = (u32::from(alpha) << 24)
                            + (u32::from(red) << 16)
                            + (u32::from(code) << 8)
                            + u32::from(blue);

                        data[index..][..n].fill(value);
                        index += n;
                        continue;
                    }
                }
            }

            let code = tree[GREEN].read_symbol(&mut self.bit_reader)?;

            //check code
            if code < 256 {
                //literal, so just use huffman codes and read as argb
                let red = tree[RED].read_symbol(&mut self.bit_reader)?;
                let blue = tree[BLUE].read_symbol(&mut self.bit_reader)?;
                let alpha = tree[ALPHA].read_symbol(&mut self.bit_reader)?;

                data[index] = (u32::from(alpha) << 24)
                    + (u32::from(red) << 16)
                    + (u32::from(code) << 8)
                    + u32::from(blue);

                index += 1;
            } else if code < 256 + 24 {
                //backward reference, so go back and use that to add image data
                let length_symbol = code - 256;
                let length = Self::get_copy_distance(&mut self.bit_reader, length_symbol)?;

                let dist_symbol = tree[DIST].read_symbol(&mut self.bit_reader)?;
                let dist_code = Self::get_copy_distance(&mut self.bit_reader, dist_symbol)?;
                let dist = Self::plane_code_to_distance(width, dist_code);

                if index < dist || num_values - index < length {
                    return Err(DecodingError::BitStreamError);
                }

                for i in 0..length {
                    data[index + i] = data[index + i - dist];
                }
                index += length;
            } else {
                //color cache, so use previously stored pixels to get this pixel
                let key = code - 256 - 24;

                if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                    //cache old colors
                    while last_cached < index {
                        color_cache.insert(data[last_cached]);
                        last_cached += 1;
                    }
                    data[index] = color_cache.lookup(key.into())?;
                } else {
                    return Err(DecodingError::BitStreamError);
                }
                index += 1;
            }
        }

        Ok(data)
    }

    /// Reads color cache data from the bitstream
    fn read_color_cache(&mut self) -> Result<Option<u8>, DecodingError> {
        if self.bit_reader.read_bits::<u8>(1)? == 1 {
            let code_bits = self.bit_reader.read_bits::<u8>(4)?;

            if !(1..=11).contains(&code_bits) {
                return Err(DecodingError::InvalidColorCacheBits(code_bits));
            }

            Ok(Some(code_bits))
        } else {
            Ok(None)
        }
    }

    /// Gets the copy distance from the prefix code and bitstream
    fn get_copy_distance(
        bit_reader: &mut BitReader<R>,
        prefix_code: u16,
    ) -> Result<usize, DecodingError> {
        if prefix_code < 4 {
            return Ok(usize::from(prefix_code + 1));
        }
        let extra_bits: u8 = ((prefix_code - 2) >> 1).try_into().unwrap();
        let offset = (2 + (usize::from(prefix_code) & 1)) << extra_bits;

        Ok(offset + bit_reader.read_bits::<usize>(extra_bits)? + 1)
    }

    /// Gets distance to pixel
    fn plane_code_to_distance(xsize: u16, plane_code: usize) -> usize {
        if plane_code > 120 {
            plane_code - 120
        } else {
            let (xoffset, yoffset) = DISTANCE_MAP[plane_code - 1];

            let dist = i32::from(xoffset) + i32::from(yoffset) * i32::from(xsize);
            if dist < 1 {
                return 1;
            }
            dist.try_into().unwrap()
        }
    }
}

#[derive(Debug, Clone)]
struct HuffmanInfo {
    xsize: u16,
    _ysize: u16,
    color_cache: Option<ColorCache>,
    image: Vec<u32>,
    bits: u8,
    mask: u16,
    huffman_code_groups: Vec<HuffmanCodeGroup>,
}

impl HuffmanInfo {
    fn get_huff_index(&self, x: u16, y: u16) -> usize {
        if self.bits == 0 {
            return 0;
        }
        let position =
            usize::from(y >> self.bits) * usize::from(self.xsize) + usize::from(x >> self.bits);
        let meta_huff_code: usize = self.image[position].try_into().unwrap();
        meta_huff_code
    }
}

#[derive(Debug, Clone)]
struct ColorCache {
    color_cache_bits: u8,
    color_cache: Vec<u32>,
}

impl ColorCache {
    fn insert(&mut self, color: u32) {
        let index = (0x1e35a7bdu32.overflowing_mul(color).0) >> (32 - self.color_cache_bits);
        self.color_cache[index as usize] = color;
    }

    fn lookup(&self, index: usize) -> Result<u32, DecodingError> {
        match self.color_cache.get(index) {
            Some(&value) => Ok(value),
            None => Err(DecodingError::BitStreamError),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BitReader<R> {
    reader: R,
    buffer: u64,
    nbits: u8,
}

impl<R: Read> BitReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            nbits: 0,
        }
    }

    pub(crate) fn read_bits<T: TryFrom<u32>>(&mut self, num: u8) -> Result<T, DecodingError> {
        debug_assert!(num as usize <= 8 * mem::size_of::<T>());
        debug_assert!(num <= 32);

        while self.nbits < num {
            self.buffer |= u64::from(self.reader.read_u8()?) << self.nbits;
            self.nbits += 8;
        }

        let value = (self.buffer & ((1 << num) - 1)) as u32;
        self.buffer >>= num;
        self.nbits -= num;

        match value.try_into() {
            Ok(value) => Ok(value),
            Err(_) => unreachable!("Value too large to fit in type"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LosslessFrame {
    pub(crate) width: u16,
    pub(crate) height: u16,

    pub(crate) buf: Vec<u32>,
}

impl LosslessFrame {
    /// Fills a buffer by converting from argb to rgba
    pub(crate) fn fill_rgba(&self, buf: &mut [u8]) {
        for (&argb_val, chunk) in self.buf.iter().zip(buf.chunks_exact_mut(4)) {
            chunk[0] = ((argb_val >> 16) & 0xff).try_into().unwrap();
            chunk[1] = ((argb_val >> 8) & 0xff).try_into().unwrap();
            chunk[2] = (argb_val & 0xff).try_into().unwrap();
            chunk[3] = ((argb_val >> 24) & 0xff).try_into().unwrap();
        }
    }

    pub(crate) fn fill_rgb(&self, buf: &mut [u8]) {
        for (&argb_val, chunk) in self.buf.iter().zip(buf.chunks_exact_mut(3)) {
            chunk[0] = ((argb_val >> 16) & 0xff).try_into().unwrap();
            chunk[1] = ((argb_val >> 8) & 0xff).try_into().unwrap();
            chunk[2] = (argb_val & 0xff).try_into().unwrap();
        }
    }

    /// Fills a buffer with just the green values from the lossless decoding
    /// Used in extended alpha decoding
    pub(crate) fn fill_green(&self, buf: &mut [u8]) {
        for (&argb_val, buf_value) in self.buf.iter().zip(buf.iter_mut()) {
            *buf_value = ((argb_val >> 8) & 0xff).try_into().unwrap();
        }
    }
}

#[cfg(test)]
mod test {

    use std::io::Cursor;

    use super::BitReader;

    #[test]
    fn bit_read_test() {
        //10011100 01000001 11100001
        let mut bit_reader = BitReader::new(Cursor::new(vec![0x9C, 0x41, 0xE1]));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 4); //100
        assert_eq!(bit_reader.read_bits::<u8>(2).unwrap(), 3); //11
        assert_eq!(bit_reader.read_bits::<u8>(6).unwrap(), 12); //001100
        assert_eq!(bit_reader.read_bits::<u16>(10).unwrap(), 40); //0000101000
        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 7); //111
    }

    #[test]
    fn bit_read_error_test() {
        //01101010
        let mut bit_reader = BitReader::new(Cursor::new(vec![0x6A]));

        assert_eq!(bit_reader.read_bits::<u8>(3).unwrap(), 2); //010
        assert_eq!(bit_reader.read_bits::<u8>(5).unwrap(), 13); //01101
        assert!(bit_reader.read_bits::<u8>(4).is_err()); //error
    }
}
