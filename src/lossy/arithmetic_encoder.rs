// currently just a direct translation of the encoder given in the vp8 specification
#[derive(Default)]
pub(crate) struct ArithmeticEncoder {
    /// the entropy values that have been encoded so far
    writer: Vec<u8>,
    /// value of the current bytes being encoded
    bottom: u32,
    /// the range for the next bit, must be between 128 and 255 inclusive
    range: u32,
    /// number of bits that have been encoded in the current byte
    bit_num: i32,
}

impl ArithmeticEncoder {
    pub fn new() -> Self {
        Self {
            writer: vec![],
            bottom: 0,
            range: 255,
            bit_num: 24,
        }
    }

    // A carry out of the current byte (`bottom`'s bit 31 set before the
    // shift) has to ripple backwards through every already-written byte that
    // is already saturated at 0xFF, turning each of them into 0x00, until it
    // reaches one that isn't and can absorb the +1 (standard arithmetic-coder
    // carry propagation - the same rule as decimal "999 + 1 = 1000").
    //
    // The previous version of this loop popped 0xFF bytes and, since they're
    // not `< 255`, simply discarded them instead of writing back 0x00 and
    // continuing the carry - i.e. it deleted bytes from the output instead of
    // zeroing them. That shortens the encoded stream by however many 0xFF
    // bytes were in the carry chain, desyncing every read the decoder does
    // from that point on. It stayed latent because it only fires when a
    // carry actually has to cross one or more 0xFF bytes, which needs a
    // long-enough run of near-maximal `bottom` values; this crate's encoder
    // wrote so little through this path before RD mode decision + skip
    // signalling (a single fixed DC/DC mode, and `mb_no_skip_coeff` always
    // disabled) that the condition was essentially never exercised.
    fn add_one_to_output(&mut self) {
        for value in self.writer.iter_mut().rev() {
            if *value == 255 {
                *value = 0;
            } else {
                *value += 1;
                return;
            }
        }
    }

    // writes a flag
    pub(crate) fn write_flag(&mut self, flag_bool: bool) {
        self.write_bool(flag_bool, 128);
    }

    pub(crate) fn write_bool(&mut self, bool_to_write: bool, probability: u8) {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);

        if bool_to_write {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }

        while self.range < 128 {
            self.range <<= 1;

            if self.bottom & (1 << 31) != 0 {
                self.add_one_to_output();
            }
            self.bottom <<= 1;

            self.bit_num -= 1;
            // we have a byte now so can write it
            if self.bit_num == 0 {
                let new_value = (self.bottom >> 24) as u8;
                self.writer.push(new_value);
                // only keep low 3 bytes
                self.bottom &= (1 << 24) - 1;
                self.bit_num = 8;
            }
        }
    }

    pub(crate) fn write_literal(&mut self, num_bits: u8, value: u8) {
        for bit in (0..num_bits).rev() {
            let bool_encode = (1 << bit) & value > 0;
            self.write_bool(bool_encode, 128);
        }
    }

    pub(crate) fn write_optional_signed_value(&mut self, num_bits: u8, value: Option<i8>) {
        self.write_flag(value.is_some());
        if let Some(value) = value {
            let abs_value = value.unsigned_abs();
            self.write_literal(num_bits, abs_value);
            let sign = value >= 0;
            self.write_flag(sign);
        }
    }

    pub(crate) fn write_with_tree(&mut self, tree: &[i8], probabilities: &[u8], value: i8) {
        self.write_with_tree_start_index(tree, probabilities, value, 0);
    }

    // Could optimise to be faster by processing the trees as const, similar to the decoder
    // especially encoding of trees
    pub(crate) fn write_with_tree_start_index(
        &mut self,
        tree: &[i8],
        probabilities: &[u8],
        value: i8,
        start_index: usize,
    ) {
        assert_eq!(tree.len(), probabilities.len() * 2);
        // the values are encoded as negative or zero in the tree, positive values are indexes
        let mut current_index = tree.iter().position(|x| *x == -value).unwrap();

        let mut to_encode: Vec<(bool, u8)> = vec![];

        loop {
            if current_index == start_index {
                // just write the 0 using the prob
                to_encode.push((false, probabilities[current_index / 2]));
                break;
            }
            if current_index == start_index + 1 {
                to_encode.push((true, probabilities[current_index / 2]));
                break;
            }

            // even => encode false
            let encode_val = if current_index % 2 == 0 {
                false
            } else {
                current_index -= 1;
                true
            };

            to_encode.push((encode_val, probabilities[current_index / 2]));

            let previous_index = tree
                .iter()
                .position(|x| *x == (current_index as i8))
                .unwrap_or_else(|| {
                    panic!("Failed to encode {value} for tree {tree:?} and probs {probabilities:?}")
                });
            current_index = previous_index;
        }

        // write bools backwards
        for (encode_bool, prob) in to_encode.iter().rev() {
            self.write_bool(*encode_bool, *prob);
        }
    }

    /// Flushes any remaining bits to the writer and consumes the encoder altogether
    pub(crate) fn flush_and_get_buffer(mut self) -> Vec<u8> {
        let mut c = self.bit_num;
        let mut v = self.bottom;
        if self.bottom & (1 << (32 - self.bit_num)) != 0 {
            self.add_one_to_output();
        }
        v <<= c & 0b111;
        c = (c >> 3) - 1;
        while c >= 0 {
            v <<= 8;
            c -= 1;
        }
        c = 3;
        while c >= 0 {
            self.writer.push((v >> 24) as u8);
            v <<= 8;
            c -= 1;
        }
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use crate::lossy::arithmetic_decoder::ArithmeticDecoder;
    use crate::lossy::common::*;

    use super::*;

    fn convert_buffer_for_decoding(buffer: &[u8]) -> Vec<[u8; 4]> {
        let mut new_buf = vec![[0u8; 4]; (buffer.len() + 3) / 4];
        new_buf.as_mut_slice().as_flattened_mut()[..buffer.len()].copy_from_slice(buffer);
        new_buf
    }

    #[test]
    fn test_arithmetic_encoder_short() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_flag(false);
        encoder.write_bool(true, 10);
        encoder.write_bool(false, 250);
        encoder.write_literal(1, 1);
        encoder.write_literal(3, 5);
        encoder.write_literal(8, 64);
        encoder.write_literal(8, 185);
        let bytes = encoder.flush_and_get_buffer();
        assert_eq!(&[104, 101, 107, 128], &*bytes);
    }

    #[test]
    fn test_arithmetic_encoder_hello() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_flag(false);
        encoder.write_bool(true, 10);
        encoder.write_bool(false, 250);
        encoder.write_literal(1, 1);
        encoder.write_literal(3, 5);
        encoder.write_literal(8, 64);
        encoder.write_literal(8, 185);
        encoder.write_literal(8, 31);
        encoder.write_literal(8, 134);
        encoder.write_optional_signed_value(2, None);
        encoder.write_optional_signed_value(2, Some(1));
        let data = b"hello";
        let bytes = encoder.flush_and_get_buffer();
        assert_eq!(data, &bytes[..data.len()]);
    }

    #[test]
    fn test_encoder_with_decoder() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_bool(true, 40);
        encoder.write_bool(true, 110);
        encoder.write_bool(false, 70);
        encoder.write_bool(false, 10);
        encoder.write_bool(true, 5);
        let write_buffer = encoder.flush_and_get_buffer();

        let decode_buffer = convert_buffer_for_decoding(&write_buffer);

        let mut decoder = ArithmeticDecoder::new();
        decoder.init(decode_buffer, write_buffer.len()).unwrap();

        let mut res = decoder.start_accumulated_result();
        assert_eq!(decoder.read_bool(40).or_accumulate(&mut res), true);
        assert_eq!(decoder.read_bool(110).or_accumulate(&mut res), true);
        assert_eq!(decoder.read_bool(70).or_accumulate(&mut res), false);
        assert_eq!(decoder.read_bool(10).or_accumulate(&mut res), false);
        assert_eq!(decoder.read_bool(5).or_accumulate(&mut res), true);
        decoder.check(res, ()).unwrap();
    }

    #[test]
    fn test_encoder_tree() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.write_with_tree(&KEYFRAME_YMODE_TREE, &KEYFRAME_YMODE_PROBS, TM_PRED);
        let write_buffer = encoder.flush_and_get_buffer();
        assert_eq!(&[233, 64, 0, 0], &*write_buffer);
    }

    // Regression test for a carry-propagation bug in `add_one_to_output`:
    // when a carry from `write_bool` needs to ripple back through one or
    // more already-written 0xFF bytes, each of them must become 0x00 and the
    // carry keeps propagating until it reaches a byte it can increment
    // in-place - the same rule as decimal "1099 + 1 = 1100". The buggy
    // version instead treated "can't add 1 without overflowing" as "discard
    // this byte", which silently shortened the encoded stream by however
    // many 0xFF bytes were in the chain: every read the decoder did from
    // that point on landed on the wrong bits. This is exactly the mechanism
    // behind a known bitstream corruption this crate had for
    // `examples/kodim02.png` at `lossy_quality = 85` (see the encoder's
    // commit history / rd_eval notes) - it just needed a long enough run of
    // near-saturated `bottom` values to fire, which a real image only
    // produces occasionally.
    #[test]
    fn test_carry_propagates_through_saturated_bytes() {
        let mut encoder = ArithmeticEncoder::new();
        encoder.writer = vec![0, 10, 255, 255, 255];
        encoder.add_one_to_output();
        assert_eq!(encoder.writer, vec![0, 11, 0, 0, 0]);

        // No chain to cross: the last byte simply absorbs the carry.
        let mut encoder = ArithmeticEncoder::new();
        encoder.writer = vec![0x10];
        encoder.add_one_to_output();
        assert_eq!(encoder.writer, vec![0x11]);

        // Every byte saturated, so the carry falls off the front. The VP8
        // encoder's own invariants make this unreachable in practice, but the
        // buffer must still keep its length rather than being emptied - which
        // is the failure mode this whole function is about.
        let mut encoder = ArithmeticEncoder::new();
        encoder.writer = vec![255, 255];
        encoder.add_one_to_output();
        assert_eq!(encoder.writer, vec![0, 0]);
    }
}
