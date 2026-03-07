//! Histogram building and manipulation for VP8L encoding.
//!
//! Histograms track symbol frequencies for Huffman code construction.

use alloc::vec;
use alloc::vec::Vec;

use super::types::{
    ALPHABET_SIZE_ALPHA, ALPHABET_SIZE_BLUE, ALPHABET_SIZE_DISTANCE, ALPHABET_SIZE_GREEN,
    ALPHABET_SIZE_RED, BackwardRefs, NUM_LENGTH_CODES, NUM_LITERAL_CODES, PixOrCopy, argb_alpha,
    argb_blue, argb_green, argb_red,
};

/// VP8L histogram for a single Huffman code group.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Green channel + length codes + cache codes (256 + 24 + cache_size).
    pub literal: Vec<u32>,
    /// Red channel (256).
    pub red: [u32; ALPHABET_SIZE_RED],
    /// Blue channel (256).
    pub blue: [u32; ALPHABET_SIZE_BLUE],
    /// Alpha channel (256).
    pub alpha: [u32; ALPHABET_SIZE_ALPHA],
    /// Distance codes (40).
    pub distance: [u32; ALPHABET_SIZE_DISTANCE],
    /// Cached bit cost.
    pub bit_cost: u64,
    /// Cache bits (0 = no cache).
    pub cache_bits: u8,
}

impl Histogram {
    /// Create a new histogram with the given cache bits.
    pub fn new(cache_bits: u8) -> Self {
        let literal_size = literal_alphabet_size(cache_bits);
        Self {
            literal: vec![0; literal_size],
            red: [0; ALPHABET_SIZE_RED],
            blue: [0; ALPHABET_SIZE_BLUE],
            alpha: [0; ALPHABET_SIZE_ALPHA],
            distance: [0; ALPHABET_SIZE_DISTANCE],
            bit_cost: 0,
            cache_bits,
        }
    }

    /// Clear all counts.
    pub fn clear(&mut self) {
        self.literal.fill(0);
        self.red.fill(0);
        self.blue.fill(0);
        self.alpha.fill(0);
        self.distance.fill(0);
        self.bit_cost = 0;
    }

    /// Add a literal ARGB pixel.
    #[inline]
    pub fn add_literal(&mut self, argb: u32) {
        let g = argb_green(argb) as usize;
        let r = argb_red(argb) as usize;
        let b = argb_blue(argb) as usize;
        let a = argb_alpha(argb) as usize;

        self.literal[g] += 1;
        self.red[r] += 1;
        self.blue[b] += 1;
        self.alpha[a] += 1;
    }

    /// Add a cache index.
    #[inline]
    pub fn add_cache_idx(&mut self, idx: u16) {
        let code = NUM_LITERAL_CODES + NUM_LENGTH_CODES + idx as usize;
        debug_assert!(code < self.literal.len());
        self.literal[code] += 1;
    }

    /// Add a backward reference.
    #[inline]
    pub fn add_copy(&mut self, len: u16, dist_code: u32) {
        let (len_code, _) = length_to_code(len);
        let (dist_idx, _) = distance_code_to_prefix(dist_code);

        self.literal[NUM_LITERAL_CODES + len_code as usize] += 1;
        self.distance[dist_idx as usize] += 1;
    }

    /// Build histogram from backward references.
    /// Assumes distances in refs are already plane codes.
    pub fn from_refs(refs: &BackwardRefs, cache_bits: u8) -> Self {
        let mut h = Self::new(cache_bits);
        for token in refs.iter() {
            match *token {
                PixOrCopy::Literal(argb) => h.add_literal(argb),
                PixOrCopy::CacheIdx(idx) => h.add_cache_idx(idx),
                PixOrCopy::Copy { len, dist } => h.add_copy(len, dist),
            }
        }
        h
    }

    /// Build histogram from backward references with raw distances,
    /// converting to plane codes on the fly for accurate distance statistics.
    pub fn from_refs_with_plane_codes(refs: &BackwardRefs, cache_bits: u8, xsize: usize) -> Self {
        let mut h = Self::new(cache_bits);
        for token in refs.iter() {
            match *token {
                PixOrCopy::Literal(argb) => h.add_literal(argb),
                PixOrCopy::CacheIdx(idx) => h.add_cache_idx(idx),
                PixOrCopy::Copy { len, dist } => {
                    let plane_code =
                        super::backward_refs::distance_to_plane_code(xsize, dist as usize);
                    h.add_copy(len, plane_code);
                }
            }
        }
        h
    }

    /// Merge another histogram into this one.
    pub fn add(&mut self, other: &Histogram) {
        debug_assert_eq!(self.cache_bits, other.cache_bits);
        for (a, b) in self.literal.iter_mut().zip(other.literal.iter()) {
            *a += b;
        }
        for i in 0..ALPHABET_SIZE_RED {
            self.red[i] += other.red[i];
        }
        for i in 0..ALPHABET_SIZE_BLUE {
            self.blue[i] += other.blue[i];
        }
        for i in 0..ALPHABET_SIZE_ALPHA {
            self.alpha[i] += other.alpha[i];
        }
        for i in 0..ALPHABET_SIZE_DISTANCE {
            self.distance[i] += other.distance[i];
        }
    }

    /// Check if a symbol array has only one non-zero entry.
    fn is_trivial(counts: &[u32]) -> Option<usize> {
        let mut symbol = None;
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 {
                if symbol.is_some() {
                    return None; // More than one symbol
                }
                symbol = Some(i);
            }
        }
        symbol
    }

    /// Get trivial symbol for literal if only one is used.
    pub fn trivial_literal(&self) -> Option<usize> {
        Self::is_trivial(&self.literal)
    }

    /// Get trivial symbol for red if only one is used.
    pub fn trivial_red(&self) -> Option<usize> {
        Self::is_trivial(&self.red)
    }

    /// Get trivial symbol for blue if only one is used.
    pub fn trivial_blue(&self) -> Option<usize> {
        Self::is_trivial(&self.blue)
    }

    /// Get trivial symbol for alpha if only one is used.
    pub fn trivial_alpha(&self) -> Option<usize> {
        Self::is_trivial(&self.alpha)
    }

    /// Get trivial symbol for distance if only one is used.
    pub fn trivial_distance(&self) -> Option<usize> {
        Self::is_trivial(&self.distance)
    }
}

/// Calculate literal alphabet size including cache codes.
#[inline]
pub fn literal_alphabet_size(cache_bits: u8) -> usize {
    ALPHABET_SIZE_GREEN + if cache_bits > 0 { 1 << cache_bits } else { 0 }
}

/// Convert length to prefix code and extra bits.
/// Returns (prefix_code, extra_bits_value).
/// Uses the same encoding as libwebp's VP8LPrefixEncode.
pub fn length_to_code(len: u16) -> (u8, u16) {
    debug_assert!(len >= 1);

    // Lengths 1-4 map directly to codes 0-3 with no extra bits
    if len <= 4 {
        return ((len - 1) as u8, 0);
    }

    // For len >= 5, use logarithmic prefix encoding
    // libwebp does --distance before computing, so distance = len - 1
    let distance = (len - 1) as u32;
    let highest_bit = 31 - distance.leading_zeros(); // BitsLog2Floor(distance)
    let second_highest_bit = (distance >> (highest_bit - 1)) & 1;
    let extra_bits_count = highest_bit - 1;
    let extra_bits_value = distance & ((1u32 << extra_bits_count) - 1);
    let code = 2 * highest_bit + second_highest_bit;

    (code as u8, extra_bits_value as u16)
}

/// Get number of extra bits for a length prefix code.
pub fn length_code_extra_bits(code: u8) -> u8 {
    if code < 4 {
        0
    } else {
        // For code >= 4: extra_bits = highest_bit - 1 = (code / 2) - 1
        (code / 2).saturating_sub(1)
    }
}

/// Convert distance code to prefix and extra bits.
/// Same encoding scheme as length codes.
pub fn distance_code_to_prefix(dist_code: u32) -> (u8, u32) {
    debug_assert!(dist_code >= 1);

    if dist_code <= 4 {
        return ((dist_code - 1) as u8, 0);
    }

    // For dist >= 5, use logarithmic prefix encoding
    let distance = dist_code - 1;
    let highest_bit = 31 - distance.leading_zeros();
    let second_highest_bit = (distance >> (highest_bit - 1)) & 1;
    let extra_bits_count = highest_bit - 1;
    let extra_bits_value = distance & ((1 << extra_bits_count) - 1);
    let code = 2 * highest_bit + second_highest_bit;

    (code as u8, extra_bits_value)
}

/// Get number of extra bits for a distance prefix code.
pub fn distance_code_extra_bits(code: u8) -> u8 {
    if code < 4 {
        0
    } else {
        (code / 2).saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_codes() {
        // Length 1-4 should be codes 0-3
        assert_eq!(length_to_code(1), (0, 0));
        assert_eq!(length_to_code(2), (1, 0));
        assert_eq!(length_to_code(3), (2, 0));
        assert_eq!(length_to_code(4), (3, 0));

        // Length 5: after --distance, distance=3, highest_bit=1, second=1, code=3, extra=0
        // Wait, that conflicts. Let me trace libwebp for length 5:
        // distance = 5, --distance = 4
        // highest_bit = BitsLog2Floor(4) = 2
        // second_highest_bit = (4 >> 1) & 1 = 0
        // code = 2*2 + 0 = 4
        assert_eq!(length_to_code(5), (4, 0));
        // Length 6: --distance=5, highest=2, second=(5>>1)&1=0, code=4, extra=5&1=1
        assert_eq!(length_to_code(6), (4, 1));
        // Length 7: --distance=6, highest=2, second=(6>>1)&1=1, code=5, extra=6&1=0
        assert_eq!(length_to_code(7), (5, 0));

        // Verify code ranges are reasonable
        for len in 1..=100u16 {
            let (code, _extra) = length_to_code(len);
            assert!(code < 24, "Code {} too large for len {}", code, len);
        }
    }

    #[test]
    fn test_distance_codes() {
        // Distance 1-4 should be codes 0-3
        assert_eq!(distance_code_to_prefix(1), (0, 0));
        assert_eq!(distance_code_to_prefix(2), (1, 0));
        assert_eq!(distance_code_to_prefix(3), (2, 0));
        assert_eq!(distance_code_to_prefix(4), (3, 0));
    }

    #[test]
    fn test_histogram_literal() {
        let mut h = Histogram::new(0);
        h.add_literal(0xFF112233); // A=FF, R=11, G=22, B=33

        assert_eq!(h.literal[0x22], 1); // Green
        assert_eq!(h.red[0x11], 1);
        assert_eq!(h.blue[0x33], 1);
        assert_eq!(h.alpha[0xFF], 1);
    }

    #[test]
    fn test_literal_alphabet_size() {
        assert_eq!(literal_alphabet_size(0), 280); // 256 + 24
        assert_eq!(literal_alphabet_size(1), 282); // 280 + 2
        assert_eq!(literal_alphabet_size(8), 536); // 280 + 256
        assert_eq!(literal_alphabet_size(11), 2328); // 280 + 2048
    }
}
