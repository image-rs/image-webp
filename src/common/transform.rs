// Allow dead code when std is disabled - some functions are encoder-only
#![cfg_attr(not(feature = "std"), allow(dead_code))]


/// 16 bit fixed point version of cos(PI/8) * sqrt(2) - 1
const CONST1: i64 = 20091;
/// 16 bit fixed point version of sin(PI/8) * sqrt(2)
const CONST2: i64 = 35468;


/// DC-only inverse transform: fills all 16 positions with (DC+4)>>3
/// Used when a block has only DC coefficient (no AC), avoiding full IDCT.
/// The input block[0] contains the quantized DC value; AC positions are ignored.
#[inline(always)]
pub(crate) fn idct4x4_dc(block: &mut [i32; 16]) {
    let dc = (block[0] + 4) >> 3;
    block.fill(dc);
}

// inverse discrete cosine transform, used in decoding
#[inline(always)]
pub(crate) fn idct4x4(block: &mut [i32]) {
    #[cfg(feature = "simd")]
    {
        super::transform_simd_intrinsics::idct4x4_intrinsics(block);
    }
    #[cfg(all(feature = "simd", not(feature = "simd")))]
    {
        super::transform_simd::idct4x4_simd(block);
    }
    #[cfg(not(any(feature = "simd", feature = "simd")))]
    {
        idct4x4_scalar(block);
    }
}

/// Inverse DCT with pre-summoned SIMD token (avoids per-call token summoning)
#[inline(always)]
#[allow(dead_code)] // May be useful for alternative decode paths
pub(crate) fn idct4x4_with_token(block: &mut [i32], simd_token: super::prediction::SimdTokenType) {
    #[cfg(feature = "simd")]
    {
        super::transform_simd_intrinsics::idct4x4_intrinsics_with_token(block, simd_token);
    }
    #[cfg(not(feature = "simd"))]
    {
        let _ = simd_token;
        idct4x4_scalar(block);
    }
}

maybe_autoversion! {
    #[cfg_attr(feature = "simd", allow(dead_code))]
    pub(crate) fn idct4x4_scalar(block: &mut [i32]) {
        // The intermediate results may overflow the types, so we stretch the type.
        fn fetch(block: &[i32], idx: usize) -> i64 {
            i64::from(block[idx])
        }

        // Perform one length check up front to avoid subsequent bounds checks in this function
        assert!(block.len() >= 16);

        for i in 0usize..4 {
            let a1 = fetch(block, i) + fetch(block, 8 + i);
            let b1 = fetch(block, i) - fetch(block, 8 + i);

            let t1 = (fetch(block, 4 + i) * CONST2) >> 16;
            let t2 = fetch(block, 12 + i) + ((fetch(block, 12 + i) * CONST1) >> 16);
            let c1 = t1 - t2;

            let t1 = fetch(block, 4 + i) + ((fetch(block, 4 + i) * CONST1) >> 16);
            let t2 = (fetch(block, 12 + i) * CONST2) >> 16;
            let d1 = t1 + t2;

            block[i] = (a1 + d1) as i32;
            block[4 + i] = (b1 + c1) as i32;
            block[4 * 3 + i] = (a1 - d1) as i32;
            block[4 * 2 + i] = (b1 - c1) as i32;
        }

        for i in 0usize..4 {
            let a1 = fetch(block, 4 * i) + fetch(block, 4 * i + 2);
            let b1 = fetch(block, 4 * i) - fetch(block, 4 * i + 2);

            let t1 = (fetch(block, 4 * i + 1) * CONST2) >> 16;
            let t2 = fetch(block, 4 * i + 3) + ((fetch(block, 4 * i + 3) * CONST1) >> 16);
            let c1 = t1 - t2;

            let t1 = fetch(block, 4 * i + 1) + ((fetch(block, 4 * i + 1) * CONST1) >> 16);
            let t2 = (fetch(block, 4 * i + 3) * CONST2) >> 16;
            let d1 = t1 + t2;

            block[4 * i] = ((a1 + d1 + 4) >> 3) as i32;
            block[4 * i + 3] = ((a1 - d1 + 4) >> 3) as i32;
            block[4 * i + 1] = ((b1 + c1 + 4) >> 3) as i32;
            block[4 * i + 2] = ((b1 - c1 + 4) >> 3) as i32;
        }
    }
}

// 14.3 inverse walsh-hadamard transform, used in decoding
maybe_autoversion! {
    pub(crate) fn iwht4x4(block: &mut [i32]) {
        // Perform one length check up front to avoid subsequent bounds checks in this function
        assert!(block.len() >= 16);

        for i in 0usize..4 {
            let a1 = block[i] + block[12 + i];
            let b1 = block[4 + i] + block[8 + i];
            let c1 = block[4 + i] - block[8 + i];
            let d1 = block[i] - block[12 + i];

            block[i] = a1 + b1;
            block[4 + i] = c1 + d1;
            block[8 + i] = a1 - b1;
            block[12 + i] = d1 - c1;
        }

        for block in block.chunks_exact_mut(4) {
            let a1 = block[0] + block[3];
            let b1 = block[1] + block[2];
            let c1 = block[1] - block[2];
            let d1 = block[0] - block[3];

            let a2 = a1 + b1;
            let b2 = c1 + d1;
            let c2 = a1 - b1;
            let d2 = d1 - c1;

            block[0] = (a2 + 3) >> 3;
            block[1] = (b2 + 3) >> 3;
            block[2] = (c2 + 3) >> 3;
            block[3] = (d2 + 3) >> 3;
        }
    }
}

maybe_autoversion! {
    pub(crate) fn wht4x4(block: &mut [i32; 16]) {
        // The intermediate results may overflow the types, so we stretch the type.
        fn fetch(block: &[i32], idx: usize) -> i64 {
            i64::from(block[idx])
        }

        // vertical
        for i in 0..4 {
            let a = fetch(block, i * 4) + fetch(block, i * 4 + 3);
            let b = fetch(block, i * 4 + 1) + fetch(block, i * 4 + 2);
            let c = fetch(block, i * 4 + 1) - fetch(block, i * 4 + 2);
            let d = fetch(block, i * 4) - fetch(block, i * 4 + 3);

            block[i * 4] = (a + b) as i32;
            block[i * 4 + 1] = (c + d) as i32;
            block[i * 4 + 2] = (a - b) as i32;
            block[i * 4 + 3] = (d - c) as i32;
        }

        // horizontal
        for i in 0..4 {
            let a1 = fetch(block, i) + fetch(block, i + 12);
            let b1 = fetch(block, i + 4) + fetch(block, i + 8);
            let c1 = fetch(block, i + 4) - fetch(block, i + 8);
            let d1 = fetch(block, i) - fetch(block, i + 12);

            let a2 = a1 + b1;
            let b2 = c1 + d1;
            let c2 = a1 - b1;
            let d2 = d1 - c1;

            let a3 = (a2 + if a2 > 0 { 1 } else { 0 }) / 2;
            let b3 = (b2 + if b2 > 0 { 1 } else { 0 }) / 2;
            let c3 = (c2 + if c2 > 0 { 1 } else { 0 }) / 2;
            let d3 = (d2 + if d2 > 0 { 1 } else { 0 }) / 2;

            block[i] = a3 as i32;
            block[i + 4] = b3 as i32;
            block[i + 8] = c3 as i32;
            block[i + 12] = d3 as i32;
        }
    }
}

#[inline(always)]
pub(crate) fn dct4x4(block: &mut [i32; 16]) {
    #[cfg(feature = "simd")]
    {
        super::transform_simd_intrinsics::dct4x4_intrinsics(block);
    }
    #[cfg(all(feature = "simd", not(feature = "simd")))]
    {
        super::transform_simd::dct4x4_simd(block);
    }
    #[cfg(not(any(feature = "simd", feature = "simd")))]
    {
        dct4x4_scalar(block);
    }
}

// Scalar DCT implementation for reference and non-SIMD builds
maybe_autoversion! {
    #[cfg_attr(feature = "simd", allow(dead_code))]
    pub(crate) fn dct4x4_scalar(block: &mut [i32; 16]) {
        // The intermediate results may overflow the types, so we stretch the type.
        fn fetch(block: &[i32], idx: usize) -> i64 {
            i64::from(block[idx])
        }

        // vertical
        for i in 0..4 {
            let a = (fetch(block, i * 4) + fetch(block, i * 4 + 3)) * 8;
            let b = (fetch(block, i * 4 + 1) + fetch(block, i * 4 + 2)) * 8;
            let c = (fetch(block, i * 4 + 1) - fetch(block, i * 4 + 2)) * 8;
            let d = (fetch(block, i * 4) - fetch(block, i * 4 + 3)) * 8;

            block[i * 4] = (a + b) as i32;
            block[i * 4 + 2] = (a - b) as i32;
            block[i * 4 + 1] = ((c * 2217 + d * 5352 + 14500) >> 12) as i32;
            block[i * 4 + 3] = ((d * 2217 - c * 5352 + 7500) >> 12) as i32;
        }

        // horizontal
        for i in 0..4 {
            let a = fetch(block, i) + fetch(block, i + 12);
            let b = fetch(block, i + 4) + fetch(block, i + 8);
            let c = fetch(block, i + 4) - fetch(block, i + 8);
            let d = fetch(block, i) - fetch(block, i + 12);

            block[i] = ((a + b + 7) >> 4) as i32;
            block[i + 8] = ((a - b + 7) >> 4) as i32;
            block[i + 4] = (((c * 2217 + d * 5352 + 12000) >> 16) + if d != 0 { 1 } else { 0 }) as i32;
            block[i + 12] = ((d * 2217 - c * 5352 + 51000) >> 16) as i32;
        }
    }
}

// =============================================================================
// Dispatch wrappers (always available, delegates to SIMD or scalar)
// =============================================================================

/// Fused residual computation + DCT for a single 4x4 block from flat u8 arrays.
/// Dispatches to SIMD when available, otherwise scalar.
#[inline(always)]
pub(crate) fn ftransform_from_u8_4x4(src: &[u8; 16], ref_: &[u8; 16]) -> [i32; 16] {
    #[cfg(feature = "simd")]
    {
        super::transform_simd_intrinsics::ftransform_from_u8_4x4(src, ref_)
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut block = [0i32; 16];
        for i in 0..16 {
            block[i] = src[i] as i32 - ref_[i] as i32;
        }
        dct4x4_scalar(&mut block);
        block
    }
}

/// Fused IDCT + add residue inplace + clear coefficients.
/// Dispatches to SIMD when available, otherwise scalar.
pub(crate) fn idct_add_residue_inplace(
    coeffs: &mut [i32; 16],
    block: &mut [u8],
    y0: usize,
    x0: usize,
    stride: usize,
    dc_only: bool,
) {
    #[cfg(feature = "simd")]
    {
        super::transform_simd_intrinsics::idct_add_residue_inplace(
            coeffs, block, y0, x0, stride, dc_only,
        );
    }
    #[cfg(not(feature = "simd"))]
    {
        if dc_only {
            let dc = coeffs[0];
            let dc_adj = (dc + 4) >> 3;
            for row in 0..4 {
                let pos = (y0 + row) * stride + x0;
                for col in 0..4 {
                    let p = block[pos + col] as i32;
                    block[pos + col] = (p + dc_adj).clamp(0, 255) as u8;
                }
            }
        } else {
            idct4x4_scalar(coeffs);
            let mut pos = y0 * stride + x0;
            for row in coeffs.chunks(4) {
                for (p, &a) in block[pos..][..4].iter_mut().zip(row.iter()) {
                    *p = (a + i32::from(*p)).clamp(0, 255) as u8;
                }
                pos += stride;
            }
        }
        coeffs.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_inverse() {
        const BLOCK: [i32; 16] = [
            38, 6, 210, 107, 42, 125, 185, 151, 241, 224, 125, 233, 227, 8, 57, 96,
        ];

        let mut dct_block = BLOCK;

        dct4x4(&mut dct_block);

        let mut inverse_dct_block = dct_block;
        idct4x4(&mut inverse_dct_block);
        assert_eq!(BLOCK, inverse_dct_block);
    }
}
