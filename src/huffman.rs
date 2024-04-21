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
#[derive(Clone, Debug, Default)]
pub(crate) struct HuffmanTree {
    tree: Vec<HuffmanTreeNode>,
    max_nodes: usize,
    num_nodes: usize,
}

impl HuffmanTree {
    fn is_full(&self) -> bool {
        self.num_nodes == self.max_nodes
    }

    /// Turns a node from empty into a branch and assigns its children
    fn assign_children(&mut self, node_index: usize) -> usize {
        let offset_index = self.num_nodes - node_index;
        self.tree[node_index] = HuffmanTreeNode::Branch(offset_index);
        self.num_nodes += 2;

        offset_index
    }

    /// Init a huffman tree
    fn init(num_leaves: usize) -> Result<HuffmanTree, DecodingError> {
        if num_leaves == 0 {
            return Err(DecodingError::HuffmanError);
        }

        let max_nodes = 2 * num_leaves - 1;
        let tree = vec![HuffmanTreeNode::Empty; max_nodes];
        let num_nodes = 1;

        let tree = HuffmanTree {
            tree,
            max_nodes,
            num_nodes,
        };

        Ok(tree)
    }

    /// Converts code lengths to codes
    fn code_lengths_to_codes(code_lengths: &[u16]) -> Result<Vec<Option<u16>>, DecodingError> {
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
        let mut next_codes = [None; MAX_ALLOWED_CODE_LENGTH + 1];
        for code_len in 1..=usize::from(max_code_length) {
            curr_code = (curr_code + code_length_hist[code_len - 1]) << 1;
            next_codes[code_len] = Some(curr_code);
        }
        let mut huff_codes = vec![None; code_lengths.len()];
        for (symbol, &length) in code_lengths.iter().enumerate() {
            let length = usize::from(length);
            if length > 0 {
                huff_codes[symbol] = next_codes[length];
                if let Some(value) = next_codes[length].as_mut() {
                    *value += 1;
                }
            } else {
                huff_codes[symbol] = None;
            }
        }

        Ok(huff_codes)
    }

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
            if node_index >= self.max_nodes {
                return Err(DecodingError::HuffmanError);
            }

            let node = self.tree[node_index];

            let offset = match node {
                HuffmanTreeNode::Empty => {
                    if self.is_full() {
                        return Err(DecodingError::HuffmanError);
                    }
                    self.assign_children(node_index)
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

        let mut tree = HuffmanTree::init(num_symbols)?;

        if num_symbols == 1 {
            tree.add_symbol(root_symbol, 0, 0)?;
        } else {
            let codes = HuffmanTree::code_lengths_to_codes(&code_lengths)?;

            for (symbol, &length) in code_lengths.iter().enumerate() {
                if length > 0 && codes[symbol].is_some() {
                    tree.add_symbol(symbol.try_into().unwrap(), codes[symbol].unwrap(), length)?;
                }
            }
        }

        Ok(tree)
    }

    /// Builds a tree explicitly from lengths, codes and symbols
    pub(crate) fn build_explicit(
        code_lengths: Vec<u16>,
        codes: Vec<u16>,
        symbols: Vec<u16>,
    ) -> Result<HuffmanTree, DecodingError> {
        let mut tree = HuffmanTree::init(symbols.len())?;

        for i in 0..symbols.len() {
            tree.add_symbol(symbols[i], codes[i], code_lengths[i])?;
        }

        Ok(tree)
    }

    pub(crate) fn is_single_node(&self) -> bool {
        self.num_nodes == 1
    }

    /// Reads a symbol using the bitstream.
    ///
    /// You must call call `bit_reader.fill()` before calling this function or it may erroroneosly
    /// detect the end of the stream and return a bitstream error.
    pub(crate) fn read_symbol<R: Read>(
        &self,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        let mut v = bit_reader.peek(15) as usize;
        let mut depth = 0;

        let mut index = 0;
        loop {
            match &self.tree[index] {
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
