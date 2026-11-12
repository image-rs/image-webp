//! Decoding of lossless WebP images
//!
//! [Lossless spec](https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification)

mod huffman;
mod reverse_transform;

use std::io::BufRead;
use std::mem;

use super::{CODE_LENGTH_CODES, CODE_LENGTH_CODE_ORDER, DISTANCE_MAP};
use crate::decoder::DecodingError;
use crate::memory_budget::{MemoryAccount, MemoryBudget};
use huffman::HuffmanTree;
use reverse_transform::{
    apply_color_indexing_transform, apply_color_transform, apply_predictor_transform,
    apply_subtract_green_transform, TransformType,
};

const GREEN: usize = 0;
const RED: usize = 1;
const BLUE: usize = 2;
const ALPHA: usize = 3;
const DIST: usize = 4;

const HUFFMAN_CODES_PER_META_CODE: usize = 5;
const COLOR_BYTE_VALUES: usize = u8::MAX as usize + 1;
const ENTROPY_IMAGE_BUFFERS: usize = 2;
const TWO_SYMBOL_TABLE_SIZE: usize = 2;
const TWO_COLOR_PACKING_BITS: u8 = 3;
const FOUR_COLOR_PACKING_BITS: u8 = 2;
const SIXTEEN_COLOR_PACKING_BITS: u8 = 1;

// The result is log2 of the number of palette indices packed into one byte.
fn color_index_bits(table_size: u16) -> u8 {
    if table_size <= 2 {
        TWO_COLOR_PACKING_BITS
    } else if table_size <= 4 {
        FOUR_COLOR_PACKING_BITS
    } else if table_size <= 16 {
        SIXTEEN_COLOR_PACKING_BITS
    } else {
        0
    }
}

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
    transforms: [Option<TransformType>; NUM_TRANSFORM_TYPES],
    transform_order: Vec<u8>,
    width: u16,
    height: u16,
    memory_used: MemoryAccount,
}

impl<R: BufRead> LosslessDecoder<R> {
    /// Create a new decoder
    pub(crate) fn new(r: R, memory_budget: MemoryBudget) -> Self {
        Self {
            bit_reader: BitReader::new(r),
            transforms: [None, None, None, None],
            transform_order: Vec::new(),
            width: 0,
            height: 0,
            memory_used: MemoryAccount::new(memory_budget),
        }
    }

    fn allocate(&mut self, bytes: usize) -> Result<(), DecodingError> {
        self.memory_used.allocate(bytes)
    }

    /// Decodes a frame.
    ///
    /// In an alpha chunk the width and height are not included in the header, so they should be
    /// provided by setting the `implicit_dimensions` argument. Otherwise that argument should be
    /// `None` and the frame dimensions will be determined by reading the VP8L header.
    pub(crate) fn decode_frame(
        &mut self,
        width: u32,
        height: u32,
        implicit_dimensions: bool,
        buf: &mut [u8],
    ) -> Result<(), DecodingError> {
        if implicit_dimensions {
            self.width = width as u16;
            self.height = height as u16;
        } else {
            let signature = self.bit_reader.read_bits::<u8>(8)?;
            if signature != 0x2f {
                return Err(DecodingError::LosslessSignatureInvalid(signature));
            }

            self.width = self.bit_reader.read_bits::<u16>(14)? + 1;
            self.height = self.bit_reader.read_bits::<u16>(14)? + 1;
            if u32::from(self.width) != width || u32::from(self.height) != height {
                return Err(DecodingError::InconsistentImageSizes);
            }

            let _alpha_used = self.bit_reader.read_bits::<u8>(1)?;
            let version_num = self.bit_reader.read_bits::<u8>(3)?;
            if version_num != 0 {
                return Err(DecodingError::VersionNumberInvalid(version_num));
            }
        }

        let transformed_width = self.read_transforms()?;
        let transformed_size = usize::from(transformed_width) * usize::from(self.height) * 4;
        self.decode_image_stream(
            transformed_width,
            self.height,
            true,
            &mut buf[..transformed_size],
        )?;

        let mut image_size = transformed_size;
        let mut width = transformed_width;
        for order_index in (0..self.transform_order.len()).rev() {
            let trans_index = self.transform_order[order_index];
            let scratch_size = match self.transforms[usize::from(trans_index)]
                .as_ref()
                .ok_or(DecodingError::TransformError)?
            {
                TransformType::ColorIndexingTransform { table_size, .. } => {
                    let bits = color_index_bits(*table_size);
                    let lookup_size = COLOR_BYTE_VALUES * mem::size_of::<[u8; 4]>() * (1 << bits);
                    if bits == 0 {
                        lookup_size
                    } else {
                        lookup_size + usize::from(subsample_size(self.width, bits))
                    }
                }
                _ => 0,
            };
            self.allocate(scratch_size)?;
            let transform = self.transforms[usize::from(trans_index)]
                .as_ref()
                .ok_or(DecodingError::TransformError)?;
            match transform {
                TransformType::PredictorTransform {
                    size_bits,
                    predictor_data,
                } => apply_predictor_transform(
                    &mut buf[..image_size],
                    width,
                    self.height,
                    *size_bits,
                    predictor_data,
                )?,
                TransformType::ColorTransform {
                    size_bits,
                    transform_data,
                } => {
                    apply_color_transform(
                        &mut buf[..image_size],
                        width,
                        *size_bits,
                        transform_data,
                    );
                }
                TransformType::SubtractGreen => {
                    apply_subtract_green_transform(&mut buf[..image_size]);
                }
                TransformType::ColorIndexingTransform {
                    table_size,
                    table_data,
                } => {
                    width = self.width;
                    image_size = usize::from(width) * usize::from(self.height) * 4;
                    apply_color_indexing_transform(
                        buf,
                        width,
                        self.height,
                        *table_size,
                        table_data,
                    );
                }
            }
            self.memory_used.release(scratch_size);
        }

        Ok(())
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
        data: &mut [u8],
    ) -> Result<(), DecodingError> {
        let initial_memory = self.memory_used.used();
        let color_cache_bits = self.read_color_cache()?;
        if let Some(bits) = color_cache_bits {
            self.allocate((1usize << bits) * mem::size_of::<[u8; 4]>())?;
        }
        let color_cache = color_cache_bits.map(|bits| ColorCache {
            color_cache_bits: bits,
            color_cache: vec![[0; 4]; 1 << bits],
        });

        let huffman_info = self.read_huffman_codes(is_argb_img, xsize, ysize, color_cache)?;
        let result = self.decode_image_data(xsize, ysize, huffman_info, data);
        self.memory_used
            .release(self.memory_used.used() - initial_memory);
        result
    }

    /// Reads transforms and their data from the bitstream
    fn read_transforms(&mut self) -> Result<u16, DecodingError> {
        let mut xsize = self.width;

        while self.bit_reader.read_bits::<u8>(1)? == 1 {
            let transform_type_val = self.bit_reader.read_bits::<u8>(2)?;

            if self.transforms[usize::from(transform_type_val)].is_some() {
                //can only have one of each transform, error
                return Err(DecodingError::TransformError);
            }

            if self.transform_order.is_empty() {
                self.allocate(NUM_TRANSFORM_TYPES)?;
                self.transform_order = Vec::with_capacity(NUM_TRANSFORM_TYPES);
            }
            self.transform_order.push(transform_type_val);

            let transform_type = match transform_type_val {
                0 => {
                    //predictor

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let size = usize::from(block_xsize)
                        .checked_mul(usize::from(block_ysize))
                        .and_then(|pixels| pixels.checked_mul(mem::size_of::<[u8; 4]>()))
                        .ok_or(DecodingError::MemoryLimitExceeded)?;
                    self.allocate(size)?;
                    let mut predictor_data = vec![0; size];
                    self.decode_image_stream(block_xsize, block_ysize, false, &mut predictor_data)?;

                    TransformType::PredictorTransform {
                        size_bits,
                        predictor_data,
                    }
                }
                1 => {
                    //color transform

                    let size_bits = self.bit_reader.read_bits::<u8>(3)? + 2;

                    let block_xsize = subsample_size(xsize, size_bits);
                    let block_ysize = subsample_size(self.height, size_bits);

                    let size = usize::from(block_xsize)
                        .checked_mul(usize::from(block_ysize))
                        .and_then(|pixels| pixels.checked_mul(mem::size_of::<[u8; 4]>()))
                        .ok_or(DecodingError::MemoryLimitExceeded)?;
                    self.allocate(size)?;
                    let mut transform_data = vec![0; size];
                    self.decode_image_stream(block_xsize, block_ysize, false, &mut transform_data)?;

                    TransformType::ColorTransform {
                        size_bits,
                        transform_data,
                    }
                }
                2 => {
                    //subtract green

                    TransformType::SubtractGreen
                }
                3 => {
                    let color_table_size = self.bit_reader.read_bits::<u16>(8)? + 1;

                    self.allocate(usize::from(color_table_size) * mem::size_of::<[u8; 4]>())?;
                    let mut color_map =
                        vec![0; usize::from(color_table_size) * mem::size_of::<[u8; 4]>()];
                    self.decode_image_stream(color_table_size, 1, false, &mut color_map)?;

                    let bits = color_index_bits(color_table_size);
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
    fn adjust_color_map(color_map: &mut [u8]) {
        for i in 4..color_map.len() {
            color_map[i] = color_map[i].wrapping_add(color_map[i - 4]);
        }
    }

    /// Reads huffman codes associated with an image
    #[inline(never)]
    fn read_huffman_codes(
        &mut self,
        read_meta: bool,
        xsize: u16,
        ysize: u16,
        color_cache: Option<ColorCache>,
    ) -> Result<HuffmanInfo, DecodingError> {
        let mut num_huff_groups = 1u32;
        let mut huffman_bits = 0;
        let mut huffman_xsize = 1;
        let mut huffman_ysize = 1;
        let mut entropy_image = Vec::new();
        let mut used_groups;

        if read_meta && self.bit_reader.read_bits::<u8>(1)? == 1 {
            huffman_bits = self.bit_reader.read_bits::<u8>(3)? + 2;
            huffman_xsize = subsample_size(xsize, huffman_bits);
            huffman_ysize = subsample_size(ysize, huffman_bits);
            let count = usize::from(huffman_xsize)
                .checked_mul(usize::from(huffman_ysize))
                .ok_or(DecodingError::MemoryLimitExceeded)?;
            let data_bytes = count
                .checked_mul(mem::size_of::<[u8; 4]>())
                .ok_or(DecodingError::MemoryLimitExceeded)?;
            self.allocate(data_bytes)?;
            let mut data = vec![0; data_bytes];
            self.decode_image_stream(huffman_xsize, huffman_ysize, false, &mut data)?;
            let entropy_bytes = count
                .checked_mul(ENTROPY_IMAGE_BUFFERS * mem::size_of::<u16>())
                .ok_or(DecodingError::MemoryLimitExceeded)?;
            self.allocate(entropy_bytes)?;
            entropy_image = Vec::with_capacity(count);
            used_groups = Vec::with_capacity(count);
            for pixel in data.chunks_exact(4) {
                let id = (u16::from(pixel[0]) << 8) | u16::from(pixel[1]);
                entropy_image.push(id);
                used_groups.push(id);
                num_huff_groups = num_huff_groups.max(u32::from(id) + 1);
            }
            drop(data);
            self.memory_used.release(data_bytes);
            used_groups.sort_unstable();
            used_groups.dedup();
            for id in &mut entropy_image {
                *id = used_groups
                    .binary_search(id)
                    .map_err(|_| DecodingError::HuffmanError)? as u16;
            }
        } else {
            self.allocate(mem::size_of::<u16>())?;
            used_groups = vec![0];
        }

        let group_bytes = used_groups
            .len()
            .checked_mul(mem::size_of::<HuffmanCodeGroup>())
            .ok_or(DecodingError::MemoryLimitExceeded)?;
        self.allocate(group_bytes)?;
        let mut hufftree_groups = Vec::with_capacity(used_groups.len());
        let mut used_index = 0;
        for i in 0..num_huff_groups {
            let retain = used_groups.get(used_index).copied() == Some(i as u16);
            let mut group: HuffmanCodeGroup = Default::default();
            for j in 0..HUFFMAN_CODES_PER_META_CODE {
                let mut alphabet_size = ALPHABET_SIZE[j];
                if j == 0 {
                    if let Some(color_cache) = color_cache.as_ref() {
                        alphabet_size += 1 << color_cache.color_cache_bits;
                    }
                }
                let tree = self.read_huffman_code(alphabet_size, retain)?;
                group[j] = tree;
            }
            if retain {
                hufftree_groups.push(group);
                used_index += 1;
            }
        }
        let used_groups_bytes = used_groups.capacity() * mem::size_of::<u16>();
        drop(used_groups);
        self.memory_used.release(used_groups_bytes);

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

    /// Decodes and returns a single Huffman tree.
    fn read_huffman_code(
        &mut self,
        alphabet_size: u16,
        retain: bool,
    ) -> Result<HuffmanTree, DecodingError> {
        let simple = self.bit_reader.read_bits::<u8>(1)? == 1;

        let tree = if simple {
            let num_symbols = self.bit_reader.read_bits::<u8>(1)? + 1;

            let is_first_8bits = self.bit_reader.read_bits::<u8>(1)?;
            let zero_symbol = self.bit_reader.read_bits::<u16>(1 + 7 * is_first_8bits)?;

            if zero_symbol >= alphabet_size {
                return Err(DecodingError::BitStreamError);
            }

            if num_symbols == 1 {
                HuffmanTree::build_single_node(zero_symbol)
            } else {
                let one_symbol = self.bit_reader.read_bits::<u16>(8)?;
                if one_symbol >= alphabet_size {
                    return Err(DecodingError::BitStreamError);
                }
                if retain {
                    self.allocate(TWO_SYMBOL_TABLE_SIZE * mem::size_of::<u16>())?;
                    HuffmanTree::build_two_node(zero_symbol, one_symbol)
                } else {
                    HuffmanTree::default()
                }
            }
        } else {
            self.allocate(CODE_LENGTH_CODES * mem::size_of::<u16>())?;
            let mut code_length_code_lengths = vec![0; CODE_LENGTH_CODES];

            let num_code_lengths = 4 + self.bit_reader.read_bits::<usize>(4)?;
            for i in 0..num_code_lengths {
                code_length_code_lengths[CODE_LENGTH_CODE_ORDER[i]] =
                    self.bit_reader.read_bits(3)?;
            }

            let new_code_lengths =
                self.read_huffman_code_lengths(code_length_code_lengths, alphabet_size)?;

            HuffmanTree::build_implicit(new_code_lengths, &mut self.memory_used, retain)?
        };

        Ok(tree)
    }

    /// Reads huffman code lengths
    fn read_huffman_code_lengths(
        &mut self,
        code_length_code_lengths: Vec<u16>,
        num_symbols: u16,
    ) -> Result<Vec<u16>, DecodingError> {
        let table =
            HuffmanTree::build_implicit(code_length_code_lengths, &mut self.memory_used, true)?;

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

        self.allocate(usize::from(num_symbols) * mem::size_of::<u16>())?;
        let mut code_lengths = vec![0; usize::from(num_symbols)];
        let mut prev_code_len = 8; //default code length

        let mut symbol = 0;
        while symbol < num_symbols {
            if max_symbol == 0 {
                break;
            }
            max_symbol -= 1;

            self.bit_reader.fill()?;
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
                }

                let length = if use_prev { prev_code_len } else { 0 };
                while repeat > 0 {
                    repeat -= 1;
                    code_lengths[usize::from(symbol)] = length;
                    symbol += 1;
                }
            }
        }

        let table_bytes = table.allocated_bytes();
        drop(table);
        self.memory_used.release(table_bytes);
        Ok(code_lengths)
    }

    /// Decodes the image data using the huffman trees and either of the 3 methods of decoding
    fn decode_image_data(
        &mut self,
        width: u16,
        height: u16,
        mut huffman_info: HuffmanInfo,
        data: &mut [u8],
    ) -> Result<(), DecodingError> {
        let num_values = usize::from(width) * usize::from(height);

        let huff_index = huffman_info.get_huff_index(0, 0);
        let mut tree = &huffman_info.huffman_code_groups[huff_index];
        let mut index = 0;

        let mut next_block_start = 0;
        while index < num_values {
            self.bit_reader.fill()?;

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
                        let value = [red as u8, code as u8, blue as u8, alpha as u8];

                        for i in 0..n {
                            data[index * 4 + i * 4..][..4].copy_from_slice(&value);
                        }

                        if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                            color_cache.insert(value);
                        }

                        index += n;
                        continue;
                    }
                }
            }

            let code = tree[GREEN].read_symbol(&mut self.bit_reader)?;

            //check code
            if code < 256 {
                //literal, so just use huffman codes and read as argb
                let green = code as u8;
                let red = tree[RED].read_symbol(&mut self.bit_reader)? as u8;
                let blue = tree[BLUE].read_symbol(&mut self.bit_reader)? as u8;
                if self.bit_reader.nbits < 15 {
                    self.bit_reader.fill()?;
                }
                let alpha = tree[ALPHA].read_symbol(&mut self.bit_reader)? as u8;

                data[index * 4] = red;
                data[index * 4 + 1] = green;
                data[index * 4 + 2] = blue;
                data[index * 4 + 3] = alpha;

                if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                    color_cache.insert([red, green, blue, alpha]);
                }
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

                if dist == 1 {
                    let value: [u8; 4] = data[(index - dist) * 4..][..4].try_into().unwrap();
                    for i in 0..length {
                        data[index * 4 + i * 4..][..4].copy_from_slice(&value);
                    }
                } else {
                    if index + length + 3 <= num_values {
                        let start = (index - dist) * 4;
                        data.copy_within(start..start + 16, index * 4);

                        if length > 4 || dist < 4 {
                            for i in (0..length * 4).step_by((dist * 4).min(16)).skip(1) {
                                data.copy_within(start + i..start + i + 16, index * 4 + i);
                            }
                        }
                    } else {
                        for i in 0..length * 4 {
                            data[index * 4 + i] = data[index * 4 + i - dist * 4];
                        }
                    }

                    if let Some(color_cache) = huffman_info.color_cache.as_mut() {
                        for pixel in data[index * 4..][..length * 4].chunks_exact(4) {
                            color_cache.insert(pixel.try_into().unwrap());
                        }
                    }
                }
                index += length;
            } else {
                //color cache, so use previously stored pixels to get this pixel
                let color_cache = huffman_info
                    .color_cache
                    .as_mut()
                    .ok_or(DecodingError::BitStreamError)?;
                let color = color_cache.lookup((code - 280).into());
                data[index * 4..][..4].copy_from_slice(&color);
                index += 1;

                if index < next_block_start {
                    if let Some((bits, code)) = tree[GREEN].peek_symbol(&self.bit_reader) {
                        if code >= 280 {
                            self.bit_reader.consume(bits)?;
                            data[index * 4..][..4]
                                .copy_from_slice(&color_cache.lookup((code - 280).into()));
                            index += 1;
                        }
                    }
                }
            }
        }

        Ok(())
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

        let bits = bit_reader.peek(extra_bits) as usize;
        bit_reader.consume(extra_bits)?;

        Ok(offset + bits + 1)
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
    image: Vec<u16>,
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
        let meta_huff_code: usize = usize::from(self.image[position]);
        meta_huff_code
    }
}

#[derive(Debug, Clone)]
struct ColorCache {
    color_cache_bits: u8,
    color_cache: Vec<[u8; 4]>,
}

impl ColorCache {
    #[inline(always)]
    fn insert(&mut self, color: [u8; 4]) {
        let [r, g, b, a] = color;
        let color_u32 =
            (u32::from(r) << 16) | (u32::from(g) << 8) | (u32::from(b)) | (u32::from(a) << 24);
        let index = (0x1e35a7bdu32.wrapping_mul(color_u32)) >> (32 - self.color_cache_bits);
        self.color_cache[index as usize] = color;
    }

    #[inline(always)]
    fn lookup(&self, index: usize) -> [u8; 4] {
        self.color_cache[index]
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BitReader<R> {
    reader: R,
    buffer: u64,
    nbits: u8,
}

impl<R: BufRead> BitReader<R> {
    const fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            nbits: 0,
        }
    }

    /// Fills the buffer with bits from the input stream.
    ///
    /// After this function, the internal buffer will contain 64-bits or have reached the end of
    /// the input stream.
    pub(crate) fn fill(&mut self) -> Result<(), DecodingError> {
        debug_assert!(self.nbits < 64);

        let mut buf = self.reader.fill_buf()?;
        if buf.len() >= 8 {
            let lookahead = u64::from_le_bytes(buf[..8].try_into().unwrap());
            self.reader.consume(usize::from((63 - self.nbits) / 8));
            self.buffer |= lookahead << self.nbits;
            self.nbits |= 56;
        } else {
            while !buf.is_empty() && self.nbits < 56 {
                self.buffer |= u64::from(buf[0]) << self.nbits;
                self.nbits += 8;
                self.reader.consume(1);
                buf = self.reader.fill_buf()?;
            }
        }

        Ok(())
    }

    /// Peeks at the next `num` bits in the buffer.
    pub(crate) const fn peek(&self, num: u8) -> u64 {
        self.buffer & ((1 << num) - 1)
    }

    /// Peeks at the full buffer.
    pub(crate) const fn peek_full(&self) -> u64 {
        self.buffer
    }

    /// Consumes `num` bits from the buffer returning an error if there are not enough bits.
    pub(crate) fn consume(&mut self, num: u8) -> Result<(), DecodingError> {
        if self.nbits < num {
            return Err(DecodingError::BitStreamError);
        }

        self.buffer >>= num;
        self.nbits -= num;
        Ok(())
    }

    /// Convenience function to read a number of bits and convert them to a type.
    pub(crate) fn read_bits<T: TryFrom<u32>>(&mut self, num: u8) -> Result<T, DecodingError> {
        debug_assert!(num as usize <= 8 * mem::size_of::<T>());
        debug_assert!(num <= 32);

        if self.nbits < num {
            self.fill()?;
        }
        let value = self.peek(num) as u32;
        self.consume(num)?;

        value.try_into().map_err(|_| {
            debug_assert!(false, "Value too large to fit in type");
            DecodingError::BitStreamError
        })
    }
}

#[cfg(test)]
mod test {

    use std::io::Cursor;

    use super::BitReader;

    struct Bits(Vec<u8>, usize);

    impl Bits {
        fn write(&mut self, value: u32, count: usize) {
            for bit in 0..count {
                if self.1 % 8 == 0 {
                    self.0.push(0);
                }
                self.0[self.1 / 8] |= (((value >> bit) & 1) as u8) << (self.1 % 8);
                self.1 += 1;
            }
        }
        fn single(&mut self, symbol: u8) {
            self.write(1, 1);
            self.write(0, 1);
            self.write(u32::from(symbol > 1), 1);
            self.write(u32::from(symbol), if symbol > 1 { 8 } else { 1 });
        }
    }

    fn sparse_groups(invalid: bool) -> Vec<u8> {
        let mut bits = Bits(Vec::new(), 0);
        bits.write(1, 1);
        bits.write(0, 3);
        bits.write(0, 1);
        for symbol in [255, 255, 0, 0, 0] {
            bits.single(symbol);
        }
        for group in 0..=u16::MAX {
            for tree in 0..5 {
                bits.single(if invalid && group == 0 && tree == 4 {
                    255
                } else {
                    0
                });
            }
        }
        bits.0
    }

    #[test]
    fn nonadjacent_groups_decode_their_own_colors() {
        let mut bits = Bits(Vec::new(), 0);
        bits.write(1, 1); // entropy image present
        bits.write(0, 3); // blocks are four pixels wide
        bits.write(0, 1); // entropy image has no color cache
        bits.write(1, 1); // simple green tree
        bits.write(1, 1); // two symbols
        bits.write(0, 1); // first symbol uses one bit
        bits.write(1, 1);
        bits.write(3, 8);
        for symbol in [0, 0, 0, 0] {
            bits.single(symbol);
        }
        bits.write(1, 1); // entropy group 3
        bits.write(0, 1); // entropy group 1
        bits.write(1, 1); // entropy group 3
        for group in 0..4 {
            for symbol in [
                0,
                if group == 1 {
                    10
                } else if group == 3 {
                    20
                } else {
                    0
                },
                0,
                255,
                0,
            ] {
                bits.single(symbol);
            }
        }
        bits.write(0, 16); // padding for the bit reader
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(bits.0),
            crate::memory_budget::MemoryBudget::new(usize::MAX),
        );
        let info = decoder.read_huffman_codes(true, 12, 1, None).unwrap();
        assert_eq!(info.image, [1, 0, 1]);
        assert_eq!(info.huffman_code_groups.len(), 2);
        let mut pixels = [0; 48];
        decoder.decode_image_data(12, 1, info, &mut pixels).unwrap();
        assert_eq!(&pixels[..16], [20, 0, 0, 255].repeat(4));
        assert_eq!(&pixels[16..32], [10, 0, 0, 255].repeat(4));
        assert_eq!(&pixels[32..], [20, 0, 0, 255].repeat(4));
    }

    #[test]
    fn sparse_group_ids_are_dense_and_do_not_retain_unused_groups() {
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(sparse_groups(false)),
            crate::memory_budget::MemoryBudget::new(4096),
        );
        let info = decoder.read_huffman_codes(true, 1, 1, None).unwrap();
        assert_eq!(info.image, [0]);
        assert_eq!(info.huffman_code_groups.len(), 1);
        let mut pixel = [1; 4];
        decoder.decode_image_data(1, 1, info, &mut pixel).unwrap();
        assert_eq!(pixel, [0; 4]);
    }

    #[test]
    fn skipped_groups_are_validated() {
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(sparse_groups(true)),
            crate::memory_budget::MemoryBudget::new(usize::MAX),
        );
        assert!(decoder.read_huffman_codes(true, 1, 1, None).is_err());
        let mut truncated = sparse_groups(false);
        truncated.truncate(truncated.len() / 2);
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(truncated),
            crate::memory_budget::MemoryBudget::new(usize::MAX),
        );
        assert!(decoder.read_huffman_codes(true, 1, 1, None).is_err());
    }

    #[test]
    fn skipped_group_rejects_invalid_code_lengths() {
        let mut bits = Bits(Vec::new(), 0);
        bits.write(1, 1);
        bits.write(0, 3);
        bits.write(0, 1);
        for symbol in [255, 255, 0, 0, 0] {
            bits.single(symbol);
        }
        bits.write(0, 1); // complex tree
        bits.write(0, 4); // four code-length code lengths
        for length in [0, 0, 1, 1] {
            bits.write(length, 3);
        }
        bits.write(1, 1); // only two symbols described
        bits.write(0, 3);
        bits.write(0, 2);
        bits.write(0, 2); // both code lengths are zero
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(bits.0),
            crate::memory_budget::MemoryBudget::new(usize::MAX),
        );
        assert!(matches!(
            decoder.read_huffman_codes(true, 1, 1, None),
            Err(crate::decoder::DecodingError::HuffmanError)
        ));
    }

    fn limit_for_huffman_test(lengths: &[u16]) -> usize {
        let primary_table_size = 1usize << super::huffman::MAX_TABLE_BITS;
        (lengths.len() * 2 + primary_table_size) * std::mem::size_of::<u16>()
    }

    #[test]
    fn secondary_huffman_table_respects_memory_limit() {
        let mut lengths: Vec<u16> = (1..=10).collect();
        lengths.extend([11, 11]);
        let budget = crate::memory_budget::MemoryBudget::new(limit_for_huffman_test(&lengths));
        let mut used = crate::memory_budget::MemoryAccount::new(budget);
        used.allocate(lengths.len() * std::mem::size_of::<u16>())
            .expect("initial lengths fit");
        assert!(matches!(
            super::HuffmanTree::build_implicit(lengths, &mut used, true),
            Err(crate::decoder::DecodingError::MemoryLimitExceeded)
        ));
    }

    #[test]
    fn huffman_table_respects_memory_limit() {
        let mut bits = Bits(Vec::new(), 0);
        bits.write(1, 1);
        bits.write(1, 1);
        bits.write(0, 1);
        bits.write(0, 1);
        bits.write(1, 8);
        let mut decoder = super::LosslessDecoder::new(
            Cursor::new(bits.0),
            crate::memory_budget::MemoryBudget::new(3),
        );
        assert!(matches!(
            decoder.read_huffman_code(256, true),
            Err(crate::decoder::DecodingError::MemoryLimitExceeded)
        ));
    }

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
