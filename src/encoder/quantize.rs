//! Quantization matrix and coefficient quantization.
//!
//! Contains VP8Matrix for quantize/dequantize operations on 4x4 DCT blocks.
//! SIMD-optimized quantization using SSE2 intrinsics.

// Many loops in this file match libwebp's C patterns for clarity when comparing
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use archmage::{arcane, rite, SimdToken, X64V3Token};
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use core::arch::x86_64::*;

use super::tables::{MAX_LEVEL, VP8_FREQ_SHARPENING};

//------------------------------------------------------------------------------
// Quantization constants

/// Fixed-point precision for quantization
pub const QFIX: u32 = 17;

/// Bias calculation macro equivalent
#[inline]
pub const fn quantization_bias(b: u32) -> u32 {
    (((b) << (QFIX)) + 128) >> 8
}

/// Quantization division: (coeff * iq + bias) >> QFIX
#[inline]
pub fn quantdiv(coeff: u32, iq: u32, bias: u32) -> i32 {
    ((coeff as u64 * iq as u64 + bias as u64) >> QFIX) as i32
}

//------------------------------------------------------------------------------
// Quantization matrix

/// Quantization matrix for a coefficient type (Y1, Y2, UV)
#[derive(Clone, Debug)]
pub struct VP8Matrix {
    /// Quantizer steps for each coefficient position
    pub q: [u16; 16],
    /// Reciprocals (1 << QFIX) / q, for fast division
    pub iq: [u32; 16],
    /// Rounding bias for quantization
    pub bias: [u32; 16],
    /// Zero threshold: coefficients below this are quantized to 0
    pub zthresh: [u32; 16],
    /// Sharpening boost for high-frequency coefficients
    pub sharpen: [u16; 16],
}

impl VP8Matrix {
    /// Create a new quantization matrix from DC and AC quantizer values
    pub fn new(q_dc: u16, q_ac: u16, matrix_type: MatrixType) -> Self {
        let bias_values = match matrix_type {
            MatrixType::Y1 => (96, 110),  // luma-ac
            MatrixType::Y2 => (96, 108),  // luma-dc
            MatrixType::UV => (110, 115), // chroma
        };

        let mut m = Self {
            q: [0; 16],
            iq: [0; 16],
            bias: [0; 16],
            zthresh: [0; 16],
            sharpen: [0; 16],
        };

        // Set DC (index 0) and AC (index 1+) values
        m.q[0] = q_dc;
        m.q[1] = q_ac;

        // Calculate reciprocals, bias, and zero thresholds for DC and AC
        for i in 0..2 {
            let is_ac = i > 0;
            let bias = if is_ac { bias_values.1 } else { bias_values.0 };
            m.iq[i] = ((1u64 << QFIX) / m.q[i] as u64) as u32;
            m.bias[i] = quantization_bias(bias);
            // zthresh: value such that quantdiv(coeff, iq, bias) is 0 if coeff <= zthresh
            m.zthresh[i] = ((1 << QFIX) - 1 - m.bias[i]) / m.iq[i];
        }

        // Replicate AC values for positions 2-15
        for i in 2..16 {
            m.q[i] = m.q[1];
            m.iq[i] = m.iq[1];
            m.bias[i] = m.bias[1];
            m.zthresh[i] = m.zthresh[1];
        }

        // Apply sharpening for Y1 matrix (luma AC)
        if matches!(matrix_type, MatrixType::Y1) {
            const SHARPEN_BITS: u32 = 11;
            for (i, &freq_sharpen) in VP8_FREQ_SHARPENING.iter().enumerate() {
                m.sharpen[i] = ((freq_sharpen as u32 * m.q[i] as u32) >> SHARPEN_BITS) as u16;
            }
        }

        m
    }

    /// Get the average quantizer value (for lambda calculations)
    pub fn average_q(&self) -> u32 {
        let sum: u32 = self.q.iter().map(|&x| x as u32).sum();
        (sum + 8) >> 4
    }

    /// Quantize a single coefficient (matches libwebp's QuantizeBlock_C)
    ///
    /// Adds sharpen boost to absolute coefficient value, then checks against
    /// zthresh to skip coefficients guaranteed to quantize to zero.
    #[inline]
    pub fn quantize_coeff(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = (if sign { -coeff } else { coeff } as u32) + self.sharpen[pos] as u32;
        if abs_coeff <= self.zthresh[pos] {
            return 0;
        }
        let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Quantize a single coefficient with neutral bias (for trellis)
    #[inline]
    pub fn quantize_neutral(&self, coeff: i32, pos: usize) -> i32 {
        let sign = coeff < 0;
        let abs_coeff = if sign { -coeff } else { coeff } as u32;
        let neutral_bias = quantization_bias(0x00); // neutral
        let level = quantdiv(abs_coeff, self.iq[pos], neutral_bias);
        if sign {
            -level
        } else {
            level
        }
    }

    /// Dequantize a coefficient
    #[inline]
    pub fn dequantize(&self, level: i32, pos: usize) -> i32 {
        level * self.q[pos] as i32
    }

    /// Quantize an entire 4x4 block of coefficients in place
    ///
    /// Includes sharpen boost and zthresh check matching libwebp's QuantizeBlock_C.
    #[inline]
    pub fn quantize(&self, coeffs: &mut [i32; 16]) {
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            let sign = *coeff < 0;
            let abs_coeff = (if sign { -*coeff } else { *coeff } as u32) + self.sharpen[pos] as u32;
            if abs_coeff <= self.zthresh[pos] {
                *coeff = 0;
                continue;
            }
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
            *coeff = if sign { -level } else { level };
        }
    }

    /// Quantize only AC coefficients (positions 1-15) in place, leaving DC unchanged
    /// This is used for Y1 blocks where the DC goes to the Y2 block
    ///
    /// Includes sharpen boost and zthresh check matching libwebp's QuantizeBlock_C.
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.iq/self.bias
    pub fn quantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            let sign = coeffs[pos] < 0;
            let abs_coeff =
                (if sign { -coeffs[pos] } else { coeffs[pos] } as u32) + self.sharpen[pos] as u32;
            if abs_coeff <= self.zthresh[pos] {
                coeffs[pos] = 0;
                continue;
            }
            let level = quantdiv(abs_coeff, self.iq[pos], self.bias[pos]).min(MAX_LEVEL as i32);
            coeffs[pos] = if sign { -level } else { level };
        }
    }

    /// Dequantize an entire 4x4 block of coefficients in place
    #[inline(always)]
    pub fn dequantize_block(&self, coeffs: &mut [i32; 16]) {
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
        {
            if let Some(token) = X64V3Token::summon() {
                dequantize_block_entry(token, &self.q, coeffs);
                return;
            }
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            use archmage::SimdToken;
            let token = archmage::NeonToken::summon().unwrap();
            crate::common::simd_neon::dequantize_block_neon(token, &self.q, coeffs);
            return;
        }
        #[cfg(all(feature = "simd", target_arch = "wasm32"))]
        {
            use archmage::SimdToken;
            if let Some(token) = archmage::Wasm128Token::summon() {
                crate::common::simd_wasm::dequantize_block_wasm_entry(token, &self.q, coeffs);
                return;
            }
        }
        #[allow(unreachable_code)]
        for (pos, coeff) in coeffs.iter_mut().enumerate() {
            *coeff *= self.q[pos] as i32;
        }
    }

    /// Dequantize only AC coefficients (positions 1-15) in place
    #[allow(clippy::needless_range_loop)] // pos indexes both coeffs and self.q
    pub fn dequantize_ac_only(&self, coeffs: &mut [i32; 16]) {
        for pos in 1..16 {
            coeffs[pos] *= self.q[pos] as i32;
        }
    }
}

/// Entry shim for dequantize_block_sse2
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[arcane]
fn dequantize_block_entry(_token: X64V3Token, q: &[u16; 16], coeffs: &mut [i32; 16]) {
    dequantize_block_sse2(_token, q, coeffs);
}

/// SIMD dequantization using SSE2
/// Multiplies each coefficient by its quantizer step.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[rite]
pub(crate) fn dequantize_block_sse2(_token: X64V3Token, q: &[u16; 16], coeffs: &mut [i32; 16]) {
    // Load quantizers as u16, zero-extend to i32
    let q_lo = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&q[0..8]).unwrap());
    let q_hi = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&q[8..16]).unwrap());

    let zero = _mm_setzero_si128();

    // Zero-extend u16 to i32: unpack low/high with zeros
    let q0_32 = _mm_unpacklo_epi16(q_lo, zero); // q[0..4] as i32
    let q1_32 = _mm_unpackhi_epi16(q_lo, zero); // q[4..8] as i32
    let q2_32 = _mm_unpacklo_epi16(q_hi, zero); // q[8..12] as i32
    let q3_32 = _mm_unpackhi_epi16(q_hi, zero); // q[12..16] as i32

    // Load coefficients as i32
    let c0 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    // Multiply using SSE2 workaround (no _mm_mullo_epi32 until SSE4.1)
    // For i32*i32, use _mm_mul_epu32 on even/odd elements separately
    // Macro to inline the multiplication pattern
    macro_rules! mul_epi32_sse2 {
        ($a:expr, $b:expr) => {{
            // Multiply even elements (positions 0, 2)
            let even = _mm_mul_epu32($a, $b);
            // Shuffle to get odd elements (1, 3) to even positions
            let a_odd = _mm_shuffle_epi32($a, 0xF5); // [1,1,3,3]
            let b_odd = _mm_shuffle_epi32($b, 0xF5);
            let odd = _mm_mul_epu32(a_odd, b_odd);
            // Extract low 32 bits of each 64-bit result and interleave
            let even_lo = _mm_shuffle_epi32(even, 0x08); // [r0, _, r2, _]
            let odd_lo = _mm_shuffle_epi32(odd, 0x08); // [r1, _, r3, _]
            _mm_unpacklo_epi32(even_lo, odd_lo) // [r0, r1, r2, r3]
        }};
    }

    let r0 = mul_epi32_sse2!(c0, q0_32);
    let r1 = mul_epi32_sse2!(c1, q1_32);
    let r2 = mul_epi32_sse2!(c2, q2_32);
    let r3 = mul_epi32_sse2!(c3, q3_32);

    // Store results
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[0..4]).unwrap(), r0);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[4..8]).unwrap(), r1);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[8..12]).unwrap(), r2);
    simd_mem::_mm_storeu_si128(<&mut [i32; 4]>::try_from(&mut coeffs[12..16]).unwrap(), r3);
}

/// Matrix type for bias selection
#[derive(Clone, Copy, Debug)]
pub enum MatrixType {
    /// Luma AC coefficients
    Y1,
    /// Luma DC (WHT) coefficients
    Y2,
    /// Chroma coefficients
    UV,
}

// =============================================================================
// SIMD Quantization - Ported from libwebp's DoQuantizeBlock_SSE2
// =============================================================================

/// SIMD-optimized quantization of a 4x4 block.
/// Returns true if any coefficient is non-zero.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn quantize_block_simd(coeffs: &mut [i32; 16], matrix: &VP8Matrix, use_sharpen: bool) -> bool {
    if let Some(token) = X64V3Token::summon() {
        quantize_block_entry(token, coeffs, matrix, use_sharpen)
    } else {
        matrix.quantize(coeffs);
        coeffs.iter().any(|&c| c != 0)
    }
}

/// NEON-optimized quantization
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn quantize_block_simd(coeffs: &mut [i32; 16], matrix: &VP8Matrix, use_sharpen: bool) -> bool {
    use archmage::SimdToken;
    let token = archmage::NeonToken::summon().unwrap();
    crate::common::simd_neon::quantize_block_neon(token, coeffs, matrix, use_sharpen)
}

/// WASM SIMD128-optimized quantization
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub fn quantize_block_simd(coeffs: &mut [i32; 16], matrix: &VP8Matrix, use_sharpen: bool) -> bool {
    use archmage::SimdToken;
    if let Some(token) = archmage::Wasm128Token::summon() {
        return crate::common::simd_wasm::quantize_block_wasm_entry(
            token,
            coeffs,
            matrix,
            use_sharpen,
        );
    }
    matrix.quantize(coeffs);
    coeffs.iter().any(|&c| c != 0)
}

/// Scalar fallback for non-SIMD platforms
#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
)))]
pub fn quantize_block_simd(coeffs: &mut [i32; 16], matrix: &VP8Matrix, _use_sharpen: bool) -> bool {
    matrix.quantize(coeffs);
    coeffs.iter().any(|&c| c != 0)
}

/// Entry shim for quantize_block_sse2
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[arcane]
fn quantize_block_entry(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    quantize_block_sse2(_token, coeffs, matrix, use_sharpen)
}

/// SSE2 implementation of block quantization.
/// Matches libwebp's DoQuantizeBlock_SSE2 algorithm.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[rite]
pub(crate) fn quantize_block_sse2(
    _token: X64V3Token,
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    let max_coeff = _mm_set1_epi16(MAX_LEVEL as i16);
    let zero = _mm_setzero_si128();

    // Pack i32 coefficients to i16 (safe for typical DCT range)
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in0 = _mm_packs_epi32(c0_32, c1_32); // coeffs[0..8] as i16
    let in8 = _mm_packs_epi32(c2_32, c3_32); // coeffs[8..16] as i16

    // Load quantization parameters (need to convert u32 iq to u16 for SIMD)
    // Since iq values fit in 16 bits for typical quantizers, this is safe
    let iq0 = _mm_set_epi16(
        matrix.iq[7] as i16,
        matrix.iq[6] as i16,
        matrix.iq[5] as i16,
        matrix.iq[4] as i16,
        matrix.iq[3] as i16,
        matrix.iq[2] as i16,
        matrix.iq[1] as i16,
        matrix.iq[0] as i16,
    );
    let iq8 = _mm_set_epi16(
        matrix.iq[15] as i16,
        matrix.iq[14] as i16,
        matrix.iq[13] as i16,
        matrix.iq[12] as i16,
        matrix.iq[11] as i16,
        matrix.iq[10] as i16,
        matrix.iq[9] as i16,
        matrix.iq[8] as i16,
    );

    // Extract sign (0x0000 if positive, 0xffff if negative)
    let sign0 = _mm_cmpgt_epi16(zero, in0);
    let sign8 = _mm_cmpgt_epi16(zero, in8);

    // coeff = abs(in) = (in ^ sign) - sign
    let mut coeff0 = _mm_sub_epi16(_mm_xor_si128(in0, sign0), sign0);
    let mut coeff8 = _mm_sub_epi16(_mm_xor_si128(in8, sign8), sign8);

    // Add sharpen if enabled
    if use_sharpen {
        let sharpen0 = _mm_set_epi16(
            matrix.sharpen[7] as i16,
            matrix.sharpen[6] as i16,
            matrix.sharpen[5] as i16,
            matrix.sharpen[4] as i16,
            matrix.sharpen[3] as i16,
            matrix.sharpen[2] as i16,
            matrix.sharpen[1] as i16,
            matrix.sharpen[0] as i16,
        );
        let sharpen8 = _mm_set_epi16(
            matrix.sharpen[15] as i16,
            matrix.sharpen[14] as i16,
            matrix.sharpen[13] as i16,
            matrix.sharpen[12] as i16,
            matrix.sharpen[11] as i16,
            matrix.sharpen[10] as i16,
            matrix.sharpen[9] as i16,
            matrix.sharpen[8] as i16,
        );
        coeff0 = _mm_add_epi16(coeff0, sharpen0);
        coeff8 = _mm_add_epi16(coeff8, sharpen8);
    }

    // out = (coeff * iQ + B) >> QFIX
    // Using mulhi_epu16 + mullo_epi16 to get 32-bit result
    let coeff_iq0_h = _mm_mulhi_epu16(coeff0, iq0);
    let coeff_iq0_l = _mm_mullo_epi16(coeff0, iq0);
    let coeff_iq8_h = _mm_mulhi_epu16(coeff8, iq8);
    let coeff_iq8_l = _mm_mullo_epi16(coeff8, iq8);

    // Unpack to 32-bit
    let out_00 = _mm_unpacklo_epi16(coeff_iq0_l, coeff_iq0_h);
    let out_04 = _mm_unpackhi_epi16(coeff_iq0_l, coeff_iq0_h);
    let out_08 = _mm_unpacklo_epi16(coeff_iq8_l, coeff_iq8_h);
    let out_12 = _mm_unpackhi_epi16(coeff_iq8_l, coeff_iq8_h);

    // Add bias
    let bias_00 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[0..4]).unwrap());
    let bias_04 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[4..8]).unwrap());
    let bias_08 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[8..12]).unwrap());
    let bias_12 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[12..16]).unwrap());

    let out_00 = _mm_add_epi32(out_00, bias_00);
    let out_04 = _mm_add_epi32(out_04, bias_04);
    let out_08 = _mm_add_epi32(out_08, bias_08);
    let out_12 = _mm_add_epi32(out_12, bias_12);

    // Shift by QFIX (17)
    let out_00 = _mm_srai_epi32(out_00, QFIX as i32);
    let out_04 = _mm_srai_epi32(out_04, QFIX as i32);
    let out_08 = _mm_srai_epi32(out_08, QFIX as i32);
    let out_12 = _mm_srai_epi32(out_12, QFIX as i32);

    // Pack back to i16
    let mut out0 = _mm_packs_epi32(out_00, out_04);
    let mut out8 = _mm_packs_epi32(out_08, out_12);

    // Clamp to MAX_LEVEL
    out0 = _mm_min_epi16(out0, max_coeff);
    out8 = _mm_min_epi16(out8, max_coeff);

    // Apply sign back: (out ^ sign) - sign
    out0 = _mm_sub_epi16(_mm_xor_si128(out0, sign0), sign0);
    out8 = _mm_sub_epi16(_mm_xor_si128(out8, sign8), sign8);

    // Unpack i16 to i32 for output
    let sign0_ext = _mm_cmpgt_epi16(zero, out0);
    let sign8_ext = _mm_cmpgt_epi16(zero, out8);

    let out0_lo = _mm_unpacklo_epi16(out0, sign0_ext);
    let out0_hi = _mm_unpackhi_epi16(out0, sign0_ext);
    let out8_lo = _mm_unpacklo_epi16(out8, sign8_ext);
    let out8_hi = _mm_unpackhi_epi16(out8, sign8_ext);

    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut coeffs[0..4]).unwrap(),
        out0_lo,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut coeffs[4..8]).unwrap(),
        out0_hi,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut coeffs[8..12]).unwrap(),
        out8_lo,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut coeffs[12..16]).unwrap(),
        out8_hi,
    );

    // Return true if any coefficient is non-zero
    let packed = _mm_packs_epi16(out0, out8);
    _mm_movemask_epi8(_mm_cmpeq_epi8(packed, zero)) != 0xffff
}

/// SIMD-optimized AC-only quantization of a 4x4 block (DC at pos 0 unchanged).
/// Returns true if any AC coefficient is non-zero.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn quantize_ac_only_simd(
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    let dc = coeffs[0];
    let has_nz = quantize_block_simd(coeffs, matrix, use_sharpen);
    coeffs[0] = dc; // Restore DC
                    // Check AC coefficients only
    coeffs[1..].iter().any(|&c| c != 0) || has_nz
}

/// NEON-optimized AC-only quantization
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn quantize_ac_only_simd(
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    let dc = coeffs[0];
    let has_nz = quantize_block_simd(coeffs, matrix, use_sharpen);
    coeffs[0] = dc;
    coeffs[1..].iter().any(|&c| c != 0) || has_nz
}

/// WASM SIMD128-optimized AC-only quantization
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub fn quantize_ac_only_simd(
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
) -> bool {
    let dc = coeffs[0];
    let has_nz = quantize_block_simd(coeffs, matrix, use_sharpen);
    coeffs[0] = dc;
    coeffs[1..].iter().any(|&c| c != 0) || has_nz
}

/// Scalar fallback for non-SIMD platforms
#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
)))]
pub fn quantize_ac_only_simd(
    coeffs: &mut [i32; 16],
    matrix: &VP8Matrix,
    _use_sharpen: bool,
) -> bool {
    matrix.quantize_ac_only(coeffs);
    coeffs[1..].iter().any(|&c| c != 0)
}

// =============================================================================
// Fused Quantize + Dequantize SIMD
// =============================================================================

/// Fused quantize+dequantize: produces both quantized levels and dequantized
/// values (quantized * q) in a single SIMD pass. Returns true if any coefficient
/// is non-zero. Coefficients are in natural (raster) order.
///
/// This matches libwebp's DoQuantizeBlock_SSE2 dual-output pattern where
/// dequantized values are computed immediately after quantization.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn quantize_dequantize_block_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    if let Some(token) = X64V3Token::summon() {
        quantize_dequantize_block_entry(token, coeffs, matrix, use_sharpen, quantized, dequantized)
    } else {
        quantize_dequantize_block_scalar(coeffs, matrix, quantized, dequantized)
    }
}

/// NEON fused quantize+dequantize
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn quantize_dequantize_block_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    use archmage::SimdToken;
    let token = archmage::NeonToken::summon().unwrap();
    crate::common::simd_neon::quantize_dequantize_block_neon(
        token,
        coeffs,
        matrix,
        use_sharpen,
        quantized,
        dequantized,
    )
}

/// WASM SIMD128 fused quantize+dequantize
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub fn quantize_dequantize_block_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    use archmage::SimdToken;
    if let Some(token) = archmage::Wasm128Token::summon() {
        return crate::common::simd_wasm::quantize_dequantize_block_wasm_entry(
            token,
            coeffs,
            matrix,
            use_sharpen,
            quantized,
            dequantized,
        );
    }
    quantize_dequantize_block_scalar(coeffs, matrix, quantized, dequantized)
}

/// Scalar fallback for non-SIMD platforms
#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
)))]
pub fn quantize_dequantize_block_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    _use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    quantize_dequantize_block_scalar(coeffs, matrix, quantized, dequantized)
}

/// Scalar implementation of fused quantize+dequantize
pub(crate) fn quantize_dequantize_block_scalar(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let mut has_nz = false;
    for pos in 0..16 {
        quantized[pos] = matrix.quantize_coeff(coeffs[pos], pos);
        dequantized[pos] = quantized[pos] * matrix.q[pos] as i32;
        if quantized[pos] != 0 {
            has_nz = true;
        }
    }
    has_nz
}

/// Entry shim for quantize_dequantize_block_sse2
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[arcane]
fn quantize_dequantize_block_entry(
    _token: X64V3Token,
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    quantize_dequantize_block_sse2(_token, coeffs, matrix, use_sharpen, quantized, dequantized)
}

/// SSE2 fused quantize+dequantize.
/// Quantizes coefficients, then immediately multiplies by q to get dequantized values.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[rite]
pub(crate) fn quantize_dequantize_block_sse2(
    _token: X64V3Token,
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let max_coeff = _mm_set1_epi16(MAX_LEVEL as i16);
    let zero = _mm_setzero_si128();

    // Pack i32 coefficients to i16
    let c0_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[0..4]).unwrap());
    let c1_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[4..8]).unwrap());
    let c2_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[8..12]).unwrap());
    let c3_32 = simd_mem::_mm_loadu_si128(<&[i32; 4]>::try_from(&coeffs[12..16]).unwrap());

    let in0 = _mm_packs_epi32(c0_32, c1_32);
    let in8 = _mm_packs_epi32(c2_32, c3_32);

    // Load quantization parameters
    let iq0 = _mm_set_epi16(
        matrix.iq[7] as i16,
        matrix.iq[6] as i16,
        matrix.iq[5] as i16,
        matrix.iq[4] as i16,
        matrix.iq[3] as i16,
        matrix.iq[2] as i16,
        matrix.iq[1] as i16,
        matrix.iq[0] as i16,
    );
    let iq8 = _mm_set_epi16(
        matrix.iq[15] as i16,
        matrix.iq[14] as i16,
        matrix.iq[13] as i16,
        matrix.iq[12] as i16,
        matrix.iq[11] as i16,
        matrix.iq[10] as i16,
        matrix.iq[9] as i16,
        matrix.iq[8] as i16,
    );

    // Extract sign
    let sign0 = _mm_cmpgt_epi16(zero, in0);
    let sign8 = _mm_cmpgt_epi16(zero, in8);

    // Absolute value
    let mut coeff0 = _mm_sub_epi16(_mm_xor_si128(in0, sign0), sign0);
    let mut coeff8 = _mm_sub_epi16(_mm_xor_si128(in8, sign8), sign8);

    // Add sharpen if enabled
    if use_sharpen {
        let sharpen0 = _mm_set_epi16(
            matrix.sharpen[7] as i16,
            matrix.sharpen[6] as i16,
            matrix.sharpen[5] as i16,
            matrix.sharpen[4] as i16,
            matrix.sharpen[3] as i16,
            matrix.sharpen[2] as i16,
            matrix.sharpen[1] as i16,
            matrix.sharpen[0] as i16,
        );
        let sharpen8 = _mm_set_epi16(
            matrix.sharpen[15] as i16,
            matrix.sharpen[14] as i16,
            matrix.sharpen[13] as i16,
            matrix.sharpen[12] as i16,
            matrix.sharpen[11] as i16,
            matrix.sharpen[10] as i16,
            matrix.sharpen[9] as i16,
            matrix.sharpen[8] as i16,
        );
        coeff0 = _mm_add_epi16(coeff0, sharpen0);
        coeff8 = _mm_add_epi16(coeff8, sharpen8);
    }

    // Quantize: out = (coeff * iQ + B) >> QFIX
    let coeff_iq0_h = _mm_mulhi_epu16(coeff0, iq0);
    let coeff_iq0_l = _mm_mullo_epi16(coeff0, iq0);
    let coeff_iq8_h = _mm_mulhi_epu16(coeff8, iq8);
    let coeff_iq8_l = _mm_mullo_epi16(coeff8, iq8);

    let out_00 = _mm_unpacklo_epi16(coeff_iq0_l, coeff_iq0_h);
    let out_04 = _mm_unpackhi_epi16(coeff_iq0_l, coeff_iq0_h);
    let out_08 = _mm_unpacklo_epi16(coeff_iq8_l, coeff_iq8_h);
    let out_12 = _mm_unpackhi_epi16(coeff_iq8_l, coeff_iq8_h);

    let bias_00 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[0..4]).unwrap());
    let bias_04 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[4..8]).unwrap());
    let bias_08 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[8..12]).unwrap());
    let bias_12 = simd_mem::_mm_loadu_si128(<&[u32; 4]>::try_from(&matrix.bias[12..16]).unwrap());

    let out_00 = _mm_srai_epi32(_mm_add_epi32(out_00, bias_00), QFIX as i32);
    let out_04 = _mm_srai_epi32(_mm_add_epi32(out_04, bias_04), QFIX as i32);
    let out_08 = _mm_srai_epi32(_mm_add_epi32(out_08, bias_08), QFIX as i32);
    let out_12 = _mm_srai_epi32(_mm_add_epi32(out_12, bias_12), QFIX as i32);

    // Pack to i16 and clamp
    let mut qout0 = _mm_packs_epi32(out_00, out_04);
    let mut qout8 = _mm_packs_epi32(out_08, out_12);
    qout0 = _mm_min_epi16(qout0, max_coeff);
    qout8 = _mm_min_epi16(qout8, max_coeff);

    // Apply sign back
    qout0 = _mm_sub_epi16(_mm_xor_si128(qout0, sign0), sign0);
    qout8 = _mm_sub_epi16(_mm_xor_si128(qout8, sign8), sign8);

    // === Dequantize: dequantized = quantized * q ===
    // Load q values as i16
    let q0 = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&matrix.q[0..8]).unwrap());
    let q8 = simd_mem::_mm_loadu_si128(<&[u16; 8]>::try_from(&matrix.q[8..16]).unwrap());

    // Multiply i16 * i16 (result fits in i16 for typical ranges)
    let dq0 = _mm_mullo_epi16(qout0, q0);
    let dq8 = _mm_mullo_epi16(qout8, q8);

    // Store quantized values (sign-extend i16 to i32)
    let qsign0 = _mm_cmpgt_epi16(zero, qout0);
    let qsign8 = _mm_cmpgt_epi16(zero, qout8);

    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut quantized[0..4]).unwrap(),
        _mm_unpacklo_epi16(qout0, qsign0),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut quantized[4..8]).unwrap(),
        _mm_unpackhi_epi16(qout0, qsign0),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut quantized[8..12]).unwrap(),
        _mm_unpacklo_epi16(qout8, qsign8),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut quantized[12..16]).unwrap(),
        _mm_unpackhi_epi16(qout8, qsign8),
    );

    // Store dequantized values (sign-extend i16 to i32)
    let dsign0 = _mm_cmpgt_epi16(zero, dq0);
    let dsign8 = _mm_cmpgt_epi16(zero, dq8);

    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut dequantized[0..4]).unwrap(),
        _mm_unpacklo_epi16(dq0, dsign0),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut dequantized[4..8]).unwrap(),
        _mm_unpackhi_epi16(dq0, dsign0),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut dequantized[8..12]).unwrap(),
        _mm_unpacklo_epi16(dq8, dsign8),
    );
    simd_mem::_mm_storeu_si128(
        <&mut [i32; 4]>::try_from(&mut dequantized[12..16]).unwrap(),
        _mm_unpackhi_epi16(dq8, dsign8),
    );

    // Check if any coefficient is non-zero
    let packed = _mm_packs_epi16(qout0, qout8);
    _mm_movemask_epi8(_mm_cmpeq_epi8(packed, zero)) != 0xffff
}

/// Fused quantize+dequantize for AC-only (preserves DC at position 0).
/// Used for Y1 blocks in I16 mode where DC goes to Y2 block.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn quantize_dequantize_ac_only_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let has_nz =
        quantize_dequantize_block_simd(coeffs, matrix, use_sharpen, quantized, dequantized);
    // Restore DC from input
    quantized[0] = coeffs[0];
    dequantized[0] = coeffs[0]; // DC will be handled by Y2 path
    has_nz || quantized[1..].iter().any(|&c| c != 0)
}

/// NEON fused quantize+dequantize for AC-only
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn quantize_dequantize_ac_only_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let has_nz =
        quantize_dequantize_block_simd(coeffs, matrix, use_sharpen, quantized, dequantized);
    quantized[0] = coeffs[0];
    dequantized[0] = coeffs[0];
    has_nz || quantized[1..].iter().any(|&c| c != 0)
}

/// WASM SIMD128 fused quantize+dequantize for AC-only
#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub fn quantize_dequantize_ac_only_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let has_nz =
        quantize_dequantize_block_simd(coeffs, matrix, use_sharpen, quantized, dequantized);
    quantized[0] = coeffs[0];
    dequantized[0] = coeffs[0];
    has_nz || quantized[1..].iter().any(|&c| c != 0)
}

/// Scalar fallback for non-SIMD platforms
#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
)))]
pub fn quantize_dequantize_ac_only_simd(
    coeffs: &[i32; 16],
    matrix: &VP8Matrix,
    _use_sharpen: bool,
    quantized: &mut [i32; 16],
    dequantized: &mut [i32; 16],
) -> bool {
    let has_nz = quantize_dequantize_block_scalar(coeffs, matrix, quantized, dequantized);
    quantized[0] = coeffs[0];
    dequantized[0] = coeffs[0];
    has_nz || quantized[1..].iter().any(|&c| c != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_block_matches_separate() {
        let matrix = VP8Matrix::new(4, 6, MatrixType::Y1);
        let coeffs: [i32; 16] = [120, -45, 30, -15, 22, -8, 5, -3, 10, -6, 4, -2, 3, -1, 1, 0];

        let mut fused_quantized = [0i32; 16];
        let mut fused_dequantized = [0i32; 16];
        let fused_nz = quantize_dequantize_block_simd(
            &coeffs,
            &matrix,
            true,
            &mut fused_quantized,
            &mut fused_dequantized,
        );

        let mut sep_quantized = coeffs;
        quantize_block_simd(&mut sep_quantized, &matrix, true);
        let mut sep_dequantized = sep_quantized;
        matrix.dequantize_block(&mut sep_dequantized);
        let sep_nz = sep_quantized.iter().any(|&c| c != 0);

        assert_eq!(
            fused_quantized, sep_quantized,
            "Quantized mismatch.\nFused: {:?}\nSep: {:?}",
            fused_quantized, sep_quantized
        );
        assert_eq!(
            fused_dequantized, sep_dequantized,
            "Dequantized mismatch.\nFused: {:?}\nSep: {:?}",
            fused_dequantized, sep_dequantized
        );
        assert_eq!(fused_nz, sep_nz, "has_nz mismatch");
    }

    #[test]
    fn test_quantize_dequantize_block_all_zero() {
        let matrix = VP8Matrix::new(80, 100, MatrixType::UV);
        let coeffs = [0i32; 16];

        let mut quantized = [0i32; 16];
        let mut dequantized = [0i32; 16];
        let has_nz = quantize_dequantize_block_simd(
            &coeffs,
            &matrix,
            false,
            &mut quantized,
            &mut dequantized,
        );

        assert!(!has_nz);
        assert_eq!(quantized, [0; 16]);
        assert_eq!(dequantized, [0; 16]);
    }
}
