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

#[derive(Clone, Debug)]
pub(crate) struct HuffmanTreeInner {
    tree: Vec<HuffmanTreeNode>,
    num_nodes: usize,

    table: Vec<u32>,
    table_mask: u16,
}

/// Huffman tree
#[derive(Clone, Debug)]
pub(crate) enum HuffmanTree {
    Single(u16),
    Tree(HuffmanTreeInner),
}

impl Default for HuffmanTree {
    fn default() -> Self {
        HuffmanTree::Single(0)
    }
}

impl HuffmanTreeInner {
    /// Adds a symbol to a huffman tree
    fn add_symbol(
        &mut self,
        symbol: u16,
        code: u16,
        code_length: u16,
    ) -> Result<(), DecodingError> {
        let mut node_index = 0;
        let code = usize::from(code);

        for length in (0..code_length).rev() {
            let node = self.tree[node_index];

            let offset = match node {
                HuffmanTreeNode::Empty => {
                    // Turns a node from empty into a branch and assigns its children
                    let offset_index = self.num_nodes - node_index;
                    self.tree[node_index] = HuffmanTreeNode::Branch(offset_index);
                    self.num_nodes += 2;
                    offset_index
                }
                HuffmanTreeNode::Leaf(_) => return Err(DecodingError::HuffmanError),
                HuffmanTreeNode::Branch(offset) => offset,
            };

            node_index += offset + ((code >> length) & 1);
        }

        match self.tree[node_index] {
            HuffmanTreeNode::Empty => self.tree[node_index] = HuffmanTreeNode::Leaf(symbol),
            HuffmanTreeNode::Leaf(_) => return Err(DecodingError::HuffmanError),
            HuffmanTreeNode::Branch(_offset) => return Err(DecodingError::HuffmanError),
        }

        Ok(())
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
        if max_code_length > table_bits {
            let max_nodes = 2 * num_symbols - 1;
            let mut tree = HuffmanTreeInner {
                tree: vec![HuffmanTreeNode::Empty; max_nodes],
                num_nodes: 1,
                table,
                table_mask,
            };

            for (symbol, &length) in code_lengths.iter().enumerate() {
                if length > 0 {
                    tree.add_symbol(symbol.try_into().unwrap(), huff_codes[symbol], length)?;
                }
            }
            Ok(HuffmanTree::Tree(tree))
        } else {
            Ok(HuffmanTree::Tree(HuffmanTreeInner {
                tree: Vec::new(),
                num_nodes: 1,
                table,
                table_mask,
            }))
        }
    }

    pub(crate) fn build_single_node(symbol: u16) -> HuffmanTree {
        HuffmanTree::Single(symbol)
    }

    pub(crate) fn build_two_node(zero: u16, one: u16) -> HuffmanTree {
        // HuffmanTree::Pair(zero, one)
        HuffmanTree::Tree(HuffmanTreeInner {
            tree: vec![
                HuffmanTreeNode::Leaf(zero),
                HuffmanTreeNode::Leaf(one),
                HuffmanTreeNode::Empty,
            ],
            num_nodes: 3,
            table: vec![1 << 16 | zero as u32, 1 << 16 | one as u32],
            table_mask: 0x1,
        })
    }

    pub(crate) fn is_single_node(&self) -> bool {
        matches!(self, HuffmanTree::Single(_))
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
            HuffmanTree::Single(symbol) => Ok(*symbol),
            HuffmanTree::Tree(inner) => {
                let mut v = bit_reader.peek(15) as usize;
                let mut depth = 0;

                let entry = inner.table[v & inner.table_mask as usize];
                if entry != 0 {
                    bit_reader.consume((entry >> 16) as u8)?;
                    return Ok(entry as u16);
                }

                let mut index = 0;
                loop {
                    match &inner.tree[index] {
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
        }
    }
}
