use std::io::BufRead;

use crate::decoder::DecodingError;
use crate::lossless::BitReader;

const MAX_ALLOWED_CODE_LENGTH: usize = 15;
const MAX_TABLE_BITS: u8 = 10;

#[derive(Clone, Debug)]
enum HuffmanTreeInner {
    Single(u16),
    Tree {
        table_mask: u16,
        primary_table: Vec<u16>,
        secondary_table: Vec<u16>,
    },
}

/// Huffman tree
#[derive(Clone, Debug)]
pub(crate) struct HuffmanTree(HuffmanTreeInner);

impl Default for HuffmanTree {
    fn default() -> Self {
        Self(HuffmanTreeInner::Single(0))
    }
}

impl HuffmanTree {
    /// Return the next code, or if the codeword is already all ones (which is the final code), return
    /// the same code again.
    fn next_codeword(mut codeword: u16, table_size: u16) -> u16 {
        if codeword == table_size - 1 {
            return codeword;
        }

        let adv = (u16::BITS - 1) - (codeword ^ (table_size - 1)).leading_zeros();
        let bit = 1 << adv;
        codeword &= bit - 1;
        codeword |= bit;
        codeword
    }

    /// Builds a tree implicitly, just from code lengths
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn build_implicit(code_lengths: Vec<u16>) -> Result<Self, DecodingError> {
        // Count symbols and build histogram
        let mut num_symbols = 0;
        let mut histogram = [0; MAX_ALLOWED_CODE_LENGTH + 1];
        for &length in code_lengths.iter() {
            histogram[usize::from(length)] += 1;
            if length != 0 {
                num_symbols += 1;
            }
        }

        // Handle special cases
        if num_symbols == 0 {
            return Err(DecodingError::HuffmanError);
        } else if num_symbols == 1 {
            let root_symbol = code_lengths.iter().position(|&x| x != 0).unwrap() as u16;
            return Ok(Self::build_single_node(root_symbol));
        };

        // Determine the maximum code length.
        let mut max_length = MAX_ALLOWED_CODE_LENGTH;
        while max_length > 1 && histogram[max_length] == 0 {
            max_length -= 1;
        }

        // Sort symbols by code length. Given the histogram, we can determine the starting offset
        // for each code length.
        let mut offsets = [0; 16];
        let mut codespace_used = 0;
        offsets[1] = histogram[0];
        for i in 1..max_length {
            offsets[i + 1] = offsets[i] + histogram[i];
            codespace_used = (codespace_used << 1) + histogram[i];
        }
        codespace_used = (codespace_used << 1) + histogram[max_length];

        // Confirm that the huffman tree is valid
        if codespace_used != (1 << max_length) {
            return Err(DecodingError::HuffmanError);
        }

        // Calculate table/tree parameters
        let table_bits = (max_length as u16).min(u16::from(MAX_TABLE_BITS));
        let table_size = (1 << table_bits) as usize;
        let table_mask = table_size as u16 - 1;
        let mut primary_table = vec![0; table_size];

        // Sort the symbols by code length.
        let mut next_index = offsets;
        let mut sorted_symbols = vec![0u16; code_lengths.len()];
        for symbol in 0..code_lengths.len() {
            let length = code_lengths[symbol];
            sorted_symbols[next_index[length as usize]] = symbol as u16;
            next_index[length as usize] += 1;
        }

        let mut codeword = 0u16;
        let mut i = histogram[0];

        // Populate the primary decoding table
        let primary_table_bits = primary_table.len().ilog2() as usize;
        let primary_table_mask = (1 << primary_table_bits) - 1;
        for length in 1..=primary_table_bits {
            let current_table_end = 1 << length;

            // Loop over all symbols with the current code length and set their table entries.
            for _ in 0..histogram[length] {
                let symbol = sorted_symbols[i];
                i += 1;

                let entry = ((length as u16) << 12) | symbol;
                primary_table[codeword as usize] = entry;

                codeword = Self::next_codeword(codeword, current_table_end as u16);
            }

            // If we aren't at the maximum table size, double the size of the table.
            if length < primary_table_bits {
                primary_table.copy_within(0..current_table_end, current_table_end);
            }
        }

        // Populate the secondary decoding table.
        let mut secondary_table = Vec::new();
        if max_length > primary_table_bits {
            let mut subtable_start = 0;
            let mut subtable_prefix = !0;
            for length in (primary_table_bits + 1)..=max_length {
                let subtable_size = 1 << (length - primary_table_bits);
                for _ in 0..histogram[length] {
                    // If the codeword's prefix doesn't match the current subtable, create a new
                    // subtable.
                    if codeword & primary_table_mask != subtable_prefix {
                        subtable_prefix = codeword & primary_table_mask;
                        subtable_start = secondary_table.len();
                        primary_table[subtable_prefix as usize] =
                            ((length as u16) << 12) | subtable_start as u16;
                        secondary_table.resize(subtable_start + subtable_size, 0);
                    }

                    // Lookup the symbol.
                    let symbol = sorted_symbols[i];
                    i += 1;

                    // Insert the symbol into the secondary table and advance to the next codeword.
                    secondary_table[subtable_start + (codeword >> primary_table_bits) as usize] =
                        (symbol << 4) | (length as u16);
                    codeword = Self::next_codeword(codeword, 1 << length);
                }

                // If there are more codes with the same subtable prefix, extend the subtable.
                if length < max_length && codeword & primary_table_mask == subtable_prefix {
                    secondary_table.extend_from_within(subtable_start..);
                    primary_table[subtable_prefix as usize] =
                        (((length + 1) as u16) << 12) | subtable_start as u16;
                }
            }
        }

        // Ensure indexes into the secondary table fit in 12 bits.
        assert!(secondary_table.len() <= 4096);

        Ok(Self(HuffmanTreeInner::Tree {
            table_mask,
            primary_table,
            secondary_table,
        }))
    }

    pub(crate) const fn build_single_node(symbol: u16) -> Self {
        Self(HuffmanTreeInner::Single(symbol))
    }

    pub(crate) fn build_two_node(zero: u16, one: u16) -> Self {
        Self(HuffmanTreeInner::Tree {
            primary_table: vec![(1 << 12) | zero, (1 << 12) | one],
            table_mask: 0x1,
            secondary_table: Vec::new(),
        })
    }

    pub(crate) const fn is_single_node(&self) -> bool {
        matches!(self.0, HuffmanTreeInner::Single(_))
    }

    #[inline(never)]
    fn read_symbol_slowpath<R: BufRead>(
        secondary_table: &[u16],
        v: u16,
        primary_table_entry: u16,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        let length = primary_table_entry >> 12;
        let mask = (1 << (length - MAX_TABLE_BITS as u16)) - 1;
        let secondary_index = ((primary_table_entry & 0xfff) as usize)
            + ((v >> MAX_TABLE_BITS) as usize & mask as usize);
        let secondary_entry = secondary_table[secondary_index];
        bit_reader.consume((secondary_entry & 0xf) as u8)?;
        Ok(secondary_entry >> 4)
    }

    /// Reads a symbol using the bit reader.
    ///
    /// You must call call `bit_reader.fill()` before calling this function or it may erroroneosly
    /// detect the end of the stream and return a bitstream error.
    pub(crate) fn read_symbol<R: BufRead>(
        &self,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        match &self.0 {
            HuffmanTreeInner::Tree {
                primary_table,
                secondary_table,
                table_mask,
            } => {
                let v = bit_reader.peek_full() as u16;
                let entry = primary_table[(v & table_mask) as usize];
                if (entry >> 12) <= MAX_TABLE_BITS as u16 {
                    bit_reader.consume((entry >> 12) as u8)?;
                    return Ok(entry & 0xfff);
                }

                Self::read_symbol_slowpath(secondary_table, v, entry, bit_reader)
            }
            HuffmanTreeInner::Single(symbol) => Ok(*symbol),
        }
    }

    /// Peek at the next symbol in the bitstream if it can be read with only a primary table lookup.
    ///
    /// Returns a tuple of the codelength and symbol value. This function may return wrong
    /// information if there aren't enough bits in the bit reader to read the next symbol.
    pub(crate) fn peek_symbol<R: BufRead>(&self, bit_reader: &BitReader<R>) -> Option<(u8, u16)> {
        match &self.0 {
            HuffmanTreeInner::Tree {
                primary_table,
                table_mask,
                ..
            } => {
                let v = bit_reader.peek_full() as u16;
                let entry = primary_table[(v & table_mask) as usize];
                if (entry >> 12) <= MAX_TABLE_BITS as u16 {
                    return Some(((entry >> 12) as u8, entry & 0xfff));
                }
                None
            }
            HuffmanTreeInner::Single(symbol) => Some((0, *symbol)),
        }
    }
}
