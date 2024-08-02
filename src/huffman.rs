use std::io::Read;

use crate::decoder::DecodingError;

use super::lossless::BitReader;

/// Rudimentary utility for reading Canonical Huffman Codes.
/// Based off https://github.com/webmproject/libwebp/blob/7f8472a610b61ec780ef0a8873cd954ac512a505/src/utils/huffman.c
///

const MAX_ALLOWED_CODE_LENGTH: usize = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HuffmanTreeNode {
    Branch(usize), //offset in vector to children
    Leaf(u16),     //symbol stored in leaf
    Empty,
}

/// Huffman tree
#[derive(Clone, Debug)]
pub(crate) enum HuffmanTree {
    Single(u16),
    Tree {
        tree: Vec<HuffmanTreeNode>,
        table: Vec<u32>,
        table_mask: u16,
    },
}

impl Default for HuffmanTree {
    fn default() -> Self {
        HuffmanTree::Single(0)
    }
}

impl HuffmanTree {
    /// Builds a tree implicitly, just from code lengths
    pub(crate) fn build_implicit(code_lengths: Vec<u16>) -> Result<HuffmanTree, DecodingError> {
        let mut num_symbols = 0;
        let mut root_symbol = 0;

        for (symbol, length) in code_lengths.iter().enumerate() {
            if *length > 0 {
                num_symbols += 1;
                root_symbol = symbol.try_into().unwrap();
            }
        }

        if num_symbols == 0 {
            return Err(DecodingError::HuffmanError);
        } else if num_symbols == 1 {
            return Ok(Self::build_single_node(root_symbol));
        };

        let max_code_length = *code_lengths
            .iter()
            .reduce(|a, b| if a >= b { a } else { b })
            .unwrap();
        if max_code_length > MAX_ALLOWED_CODE_LENGTH.try_into().unwrap() {
            return Err(DecodingError::HuffmanError);
        }

        // Build histogram
        let mut code_length_hist = [0; MAX_ALLOWED_CODE_LENGTH + 1];
        for &length in code_lengths.iter() {
            code_length_hist[usize::from(length)] += 1;
        }
        code_length_hist[0] = 0;

        // Ensure code lengths produce a valid huffman tree
        let mut total = 0;
        for (code_len, &count) in code_length_hist.iter().enumerate() {
            total += (count as u32) << (MAX_ALLOWED_CODE_LENGTH - code_len);
        }
        if total != 1 << MAX_ALLOWED_CODE_LENGTH {
            return Err(DecodingError::HuffmanError);
        }

        // Assign codes
        let mut curr_code = 0;
        let mut next_codes = [0; MAX_ALLOWED_CODE_LENGTH + 1];
        for code_len in 1..=usize::from(max_code_length) {
            curr_code = (curr_code + code_length_hist[code_len - 1]) << 1;
            next_codes[code_len] = curr_code;
        }
        let mut huff_codes = vec![0u16; code_lengths.len()];
        for (symbol, &length) in code_lengths.iter().enumerate() {
            let length = usize::from(length);
            if length > 0 {
                huff_codes[symbol] = next_codes[length];
                next_codes[length] += 1;
            }
        }

        // Populate decoding table
        let table_bits = max_code_length.min(10);
        let table_size = (1 << table_bits) as usize;
        let table_mask = table_size as u16 - 1;
        let mut table = vec![0; table_size];
        for (symbol, (&code, &length)) in huff_codes.iter().zip(code_lengths.iter()).enumerate() {
            if length != 0 && length <= table_bits {
                let mut j = ((code as u16).reverse_bits() >> (16 - length)) as usize;
                let entry = ((length as u32) << 16) | symbol as u32;
                while j < table_size {
                    table[j] = entry;
                    j += 1 << length as usize;
                }
            }
        }

        // If the longest code is larger than the table size, build a tree as a fallback.
        let mut tree = Vec::new();
        if max_code_length > table_bits {
            tree = vec![HuffmanTreeNode::Empty; 2 * num_symbols - 1];

            let mut num_nodes = 1;
            for (symbol, &length) in code_lengths.iter().enumerate() {
                let code = huff_codes[symbol];
                let code_length = length;
                let symbol = symbol.try_into().unwrap();

                if length > 0 {
                    let mut node_index = 0;
                    let code = usize::from(code);

                    for length in (0..code_length).rev() {
                        let node = tree[node_index];

                        let offset = match node {
                            HuffmanTreeNode::Empty => {
                                // Turns a node from empty into a branch and assigns its children
                                let offset_index = num_nodes - node_index;
                                tree[node_index] = HuffmanTreeNode::Branch(offset_index);
                                num_nodes += 2;
                                offset_index
                            }
                            HuffmanTreeNode::Leaf(_) => return Err(DecodingError::HuffmanError),
                            HuffmanTreeNode::Branch(offset) => offset,
                        };

                        node_index += offset + ((code >> length) & 1);
                    }

                    match tree[node_index] {
                        HuffmanTreeNode::Empty => tree[node_index] = HuffmanTreeNode::Leaf(symbol),
                        HuffmanTreeNode::Leaf(_) => return Err(DecodingError::HuffmanError),
                        HuffmanTreeNode::Branch(_offset) => {
                            return Err(DecodingError::HuffmanError)
                        }
                    }
                }
            }
        }

        Ok(HuffmanTree::Tree {
            tree,
            table,
            table_mask,
        })
    }

    pub(crate) fn build_single_node(symbol: u16) -> HuffmanTree {
        HuffmanTree::Single(symbol)
    }

    pub(crate) fn build_two_node(zero: u16, one: u16) -> HuffmanTree {
        HuffmanTree::Tree {
            tree: vec![
                HuffmanTreeNode::Leaf(zero),
                HuffmanTreeNode::Leaf(one),
                HuffmanTreeNode::Empty,
            ],
            table: vec![1 << 16 | zero as u32, 1 << 16 | one as u32],
            table_mask: 0x1,
        }
    }

    pub(crate) fn is_single_node(&self) -> bool {
        matches!(self, HuffmanTree::Single(_))
    }

    #[inline(never)]
    fn read_symbol_slowpath<R: Read>(
        tree: &[HuffmanTreeNode],
        mut v: usize,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        let mut depth = 0;
        let mut index = 0;
        loop {
            match &tree[index] {
                HuffmanTreeNode::Branch(children_offset) => {
                    index += children_offset + (v & 1);
                    depth += 1;
                    v >>= 1;
                }
                HuffmanTreeNode::Leaf(symbol) => {
                    bit_reader.consume(depth)?;
                    return Ok(*symbol);
                }
                HuffmanTreeNode::Empty => return Err(DecodingError::HuffmanError),
            }
        }
    }

    /// Reads a symbol using the bitstream.
    ///
    /// You must call call `bit_reader.fill()` before calling this function or it may erroroneosly
    /// detect the end of the stream and return a bitstream error.
    pub(crate) fn read_symbol<R: Read>(
        &self,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        match self {
            HuffmanTree::Tree {
                tree,
                table,
                table_mask,
            } => {
                let v = bit_reader.peek_full() as u16;
                let entry = table[(v & table_mask) as usize];
                if entry != 0 {
                    bit_reader.consume((entry >> 16) as u8)?;
                    return Ok(entry as u16);
                }

                Self::read_symbol_slowpath(tree, v as usize, bit_reader)
            }
            HuffmanTree::Single(symbol) => Ok(*symbol),
        }
    }
}
