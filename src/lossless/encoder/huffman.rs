use std::collections::BinaryHeap;
use std::io::{self, Write};

use super::BitWriter;
use crate::lossless::CODE_LENGTH_CODE_ORDER;

pub(crate) fn write_single_entry_huffman_tree<W: Write>(
    w: &mut BitWriter<W>,
    symbol: u8,
) -> io::Result<()> {
    w.write_bits(1, 2)?;
    if symbol <= 1 {
        w.write_bits(0, 1)?;
        w.write_bits(u64::from(symbol), 1)?;
    } else {
        w.write_bits(1, 1)?;
        w.write_bits(u64::from(symbol), 8)?;
    }
    Ok(())
}

pub(crate) fn write_huffman_tree<W: Write>(
    w: &mut BitWriter<W>,
    frequencies: &[u32],
    lengths: &mut [u8],
    codes: &mut [u16],
) -> io::Result<()> {
    if !build_huffman_tree(frequencies, lengths, codes, 15) {
        let symbol = frequencies
            .iter()
            .position(|&frequency| frequency > 0)
            .unwrap_or(0);
        return write_single_entry_huffman_tree(w, symbol as u8);
    }

    let mut code_length_lengths = [0u8; 16];
    let mut code_length_codes = [0u16; 16];
    let mut code_length_frequencies = [0u32; 16];
    for &length in lengths.iter() {
        code_length_frequencies[length as usize] += 1;
    }
    let single_code_length_length = !build_huffman_tree(
        &code_length_frequencies,
        &mut code_length_lengths,
        &mut code_length_codes,
        7,
    );

    // Write the huffman tree
    w.write_bits(0, 1)?; // normal huffman tree
    w.write_bits(19 - 4, 4)?; // num_code_lengths - 4

    for i in CODE_LENGTH_CODE_ORDER {
        if i > 15 || code_length_frequencies[i] == 0 {
            w.write_bits(0, 3)?;
        } else if single_code_length_length {
            w.write_bits(1, 3)?;
        } else {
            w.write_bits(u64::from(code_length_lengths[i]), 3)?;
        }
    }

    match lengths.len() {
        256 => {
            w.write_bits(1, 1)?; // max_symbol is stored
            w.write_bits(3, 3)?; // max_symbol_nbits / 2 - 2
            w.write_bits(254, 8)?; // max_symbol - 2
        }
        280 => w.write_bits(0, 1)?,
        _ => unreachable!(),
    }

    // Write the huffman codes
    if !single_code_length_length {
        for &len in lengths.iter() {
            w.write_bits(
                u64::from(code_length_codes[len as usize]),
                code_length_lengths[len as usize],
            )?;
        }
    }

    Ok(())
}

fn build_huffman_tree(
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
        indexes.sort_unstable_by_key(|&(index, frequency)| (frequency, index));
        for &(i, frequency) in &indexes {
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

#[cfg(test)]
mod tests {
    use super::build_huffman_tree;

    #[test]
    fn length_limited_ties_follow_symbol_order() {
        let frequencies = [
            36460, 6724, 2938, 2015, 1496, 1187, 1063, 967, 829, 745, 693, 672, 592, 555, 503, 489,
            465, 436, 428, 329, 335, 310, 314, 277, 234, 222, 221, 233, 223, 178, 170, 183, 161,
            156, 122, 118, 104, 87, 109, 84, 90, 70, 93, 71, 62, 59, 68, 78, 63, 53, 65, 59, 58,
            55, 53, 56, 39, 40, 43, 49, 42, 32, 31, 45, 43, 39, 34, 34, 36, 54, 57, 44, 26, 32, 25,
            27, 19, 18, 21, 17, 9, 10, 16, 12, 11, 13, 11, 14, 11, 14, 17, 17, 8, 12, 6, 5, 10, 13,
            8, 16, 14, 12, 13, 9, 9, 9, 12, 9, 8, 3, 7, 3, 3, 4, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 3, 2, 1, 3, 6, 4, 1, 3, 2, 6, 7, 11,
            11, 12, 20, 19, 20, 17, 13, 6, 6, 12, 12, 9, 8, 8, 9, 10, 13, 11, 11, 14, 15, 20, 9, 9,
            15, 15, 15, 19, 21, 34, 12, 39, 37, 67, 63, 41, 43, 31, 35, 32, 27, 42, 45, 50, 37, 38,
            48, 42, 51, 57, 62, 66, 44, 68, 45, 48, 60, 55, 68, 70, 75, 68, 91, 78, 103, 124, 105,
            119, 114, 126, 169, 145, 184, 156, 174, 229, 209, 217, 223, 232, 273, 270, 297, 358,
            390, 382, 411, 430, 459, 481, 531, 587, 641, 728, 746, 840, 912, 958, 1236, 1483, 2075,
            3201, 7209, 16415, 1725, 2751, 495, 1061, 579, 518, 274, 244, 247, 80, 24, 39, 33, 52,
            62, 73, 253, 112, 197, 193, 0, 2, 30,
        ];
        let mut lengths = [0; 280];
        let mut codes = [0; 280];
        assert!(build_huffman_tree(
            &frequencies,
            &mut lengths,
            &mut codes,
            15
        ));
        assert_eq!(lengths.iter().copied().max(), Some(15));
        for i in 0..frequencies.len() {
            for j in i + 1..frequencies.len() {
                if frequencies[i] > 0 && frequencies[i] == frequencies[j] {
                    assert!(lengths[i] >= lengths[j], "symbols {i} and {j}");
                }
            }
        }
    }
}
