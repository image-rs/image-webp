//! WASM SIMD128 intrinsics for DCT/IDCT transforms.
//!
//! Uses i32x4 arithmetic matching the scalar implementation for correctness,
//! with SIMD parallelism across 4 columns/rows at once.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[cfg(target_arch = "wasm32")]
use archmage::{Wasm128Token, arcane, rite};

// =============================================================================
// Public dispatch functions
// =============================================================================

/// Forward DCT using WASM SIMD128 (entry shim)
#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn dct4x4_wasm(_token: Wasm128Token, block: &mut [i32; 16]) {
    dct4x4_wasm_impl(_token, block);
}

/// Inverse DCT using WASM SIMD128 (entry shim)
#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn idct4x4_wasm(_token: Wasm128Token, block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    idct4x4_wasm_impl(_token, block);
}

// =============================================================================
// Helpers
// =============================================================================

/// Load 4 i32 values from a block row
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn load_row(block: &[i32], row: usize) -> v128 {
    let off = row * 4;
    i32x4(block[off], block[off + 1], block[off + 2], block[off + 3])
}

/// Store v128 as 4 i32 values into a block row
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn store_row(block: &mut [i32], row: usize, v: v128) {
    let off = row * 4;
    block[off] = i32x4_extract_lane::<0>(v);
    block[off + 1] = i32x4_extract_lane::<1>(v);
    block[off + 2] = i32x4_extract_lane::<2>(v);
    block[off + 3] = i32x4_extract_lane::<3>(v);
}

/// Compute (v * constant) >> 16 using 64-bit intermediates.
/// Matches the scalar `(val * CONST) >> 16` pattern used in VP8 transforms.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn mulhi32x4(v: v128, c: i32) -> v128 {
    let c_vec = i32x4_splat(c);
    let lo_prod = i64x2_extmul_low_i32x4(v, c_vec);
    let hi_prod = i64x2_extmul_high_i32x4(v, c_vec);
    let lo_shifted = i64x2_shr(lo_prod, 16);
    let hi_shifted = i64x2_shr(hi_prod, 16);
    // Narrow back to i32x4 by extracting low 32 bits of each i64
    i32x4(
        i64x2_extract_lane::<0>(lo_shifted) as i32,
        i64x2_extract_lane::<1>(lo_shifted) as i32,
        i64x2_extract_lane::<0>(hi_shifted) as i32,
        i64x2_extract_lane::<1>(hi_shifted) as i32,
    )
}

/// Transpose a 4x4 matrix of i32 values stored as 4 row vectors.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn transpose4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
    // Step 1: interleave pairs of 32-bit elements
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1); // r0[0], r1[0], r0[1], r1[1]
    let t1 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1); // r0[2], r1[2], r0[3], r1[3]
    let t2 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3); // r2[0], r3[0], r2[1], r3[1]
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3); // r2[2], r3[2], r2[3], r3[3]

    // Step 2: interleave 64-bit pairs
    let o0 = i64x2_shuffle::<0, 2>(t0, t2); // col 0: r0[0], r1[0], r2[0], r3[0]
    let o1 = i64x2_shuffle::<1, 3>(t0, t2); // col 1: r0[1], r1[1], r2[1], r3[1]
    let o2 = i64x2_shuffle::<0, 2>(t1, t3); // col 2: r0[2], r1[2], r2[2], r3[2]
    let o3 = i64x2_shuffle::<1, 3>(t1, t3); // col 3: r0[3], r1[3], r2[3], r3[3]

    (o0, o1, o2, o3)
}

// =============================================================================
// Inverse DCT
// =============================================================================

// VP8 IDCT constants
const CONST1: i32 = 20091; // sqrt(2)*cos(pi/8)*65536 - 65536
const CONST2: i32 = 35468; // sqrt(2)*sin(pi/8)*65536

/// One pass of the IDCT butterfly. Processes 4 elements in parallel.
/// Matches the scalar: a=in0+in2, b=in0-in2, c=MUL(in1,K2)-MUL(in3,K1), d=MUL(in1,K1)+MUL(in3,K2)
/// where MUL(x,K) = x + (x*k)>>16 for K1, or just (x*k)>>16 for K2.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn idct_butterfly(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
    let a = i32x4_add(in0, in2);
    let b = i32x4_sub(in0, in2);

    // t1 = (in1 * CONST2) >> 16
    let t1_c = mulhi32x4(in1, CONST2);
    // t2 = in3 + (in3 * CONST1) >> 16  = MUL(in3, K1)
    let t2_c = i32x4_add(in3, mulhi32x4(in3, CONST1));
    let c = i32x4_sub(t1_c, t2_c);

    // t1 = in1 + (in1 * CONST1) >> 16  = MUL(in1, K1)
    let t1_d = i32x4_add(in1, mulhi32x4(in1, CONST1));
    // t2 = (in3 * CONST2) >> 16
    let t2_d = mulhi32x4(in3, CONST2);
    let d = i32x4_add(t1_d, t2_d);

    let out0 = i32x4_add(a, d);
    let out1 = i32x4_add(b, c);
    let out2 = i32x4_sub(b, c);
    let out3 = i32x4_sub(a, d);

    (out0, out1, out2, out3)
}

/// WASM SIMD128 inverse DCT. Uses i32x4 arithmetic matching the scalar implementation.
///
/// Scalar pass order: column pass (for i in 0..4, reads block[i], block[4+i], ...),
/// then row pass (for i in 0..4, reads block[4*i], block[4*i+1], ...).
///
/// With row vectors r0..r3, `idct_butterfly(r0,r1,r2,r3)` processes lane j as
/// butterfly(row0[j], row1[j], row2[j], row3[j]) = column pass (no transpose needed).
/// For the row pass, we transpose so each vector holds a row's elements as lanes.
#[cfg(target_arch = "wasm32")]
#[rite]
pub(crate) fn idct4x4_wasm_impl(_token: Wasm128Token, block: &mut [i32]) {
    // Load 4 rows
    let r0 = load_row(block, 0);
    let r1 = load_row(block, 1);
    let r2 = load_row(block, 2);
    let r3 = load_row(block, 3);

    // Pass 1: column pass — butterfly on row vectors processes columns in parallel
    let (v0, v1, v2, v3) = idct_butterfly(r0, r1, r2, r3);

    // Pass 2: row pass — transpose so each vector holds one row's elements,
    // butterfly processes rows, then transpose back to row-major
    let (c0, c1, c2, c3) = transpose4x4(v0, v1, v2, v3);
    let (t0, t1, t2, t3) = idct_butterfly(c0, c1, c2, c3);
    let (f0, f1, f2, f3) = transpose4x4(t0, t1, t2, t3);

    // Final rounding: (val + 4) >> 3
    let four = i32x4_splat(4);
    let o0 = i32x4_shr(i32x4_add(f0, four), 3);
    let o1 = i32x4_shr(i32x4_add(f1, four), 3);
    let o2 = i32x4_shr(i32x4_add(f2, four), 3);
    let o3 = i32x4_shr(i32x4_add(f3, four), 3);

    // Store results
    store_row(block, 0, o0);
    store_row(block, 1, o1);
    store_row(block, 2, o2);
    store_row(block, 3, o3);
}

// =============================================================================
// Forward DCT
// =============================================================================

/// First pass of the forward DCT butterfly (row pass). Processes 4 elements in parallel.
///
/// Scalar row pass (for each row i):
///   a = (block[i*4+0] + block[i*4+3]) * 8
///   b = (block[i*4+1] + block[i*4+2]) * 8
///   c = (block[i*4+1] - block[i*4+2]) * 8
///   d = (block[i*4+0] - block[i*4+3]) * 8
///   out[i*4]   = a + b
///   out[i*4+1] = (c*2217 + d*5352 + 14500) >> 12
///   out[i*4+2] = a - b
///   out[i*4+3] = (d*2217 - c*5352 + 7500) >> 12
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn dct_butterfly(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
    let eight = i32x4_splat(8);

    let a = i32x4_mul(i32x4_add(in0, in3), eight);
    let b = i32x4_mul(i32x4_add(in1, in2), eight);
    let c = i32x4_mul(i32x4_sub(in1, in2), eight);
    let d = i32x4_mul(i32x4_sub(in0, in3), eight);

    let k2217 = i32x4_splat(2217);
    let k5352 = i32x4_splat(5352);

    let out0 = i32x4_add(a, b);

    // out1 = (c * 2217 + d * 5352 + 14500) >> 12
    let k14500 = i32x4_splat(14500);
    let out1 = i32x4_shr(
        i32x4_add(i32x4_add(i32x4_mul(c, k2217), i32x4_mul(d, k5352)), k14500),
        12,
    );

    // out2 = a - b
    let out2 = i32x4_sub(a, b);

    // out3 = (d * 2217 - c * 5352 + 7500) >> 12
    let k7500 = i32x4_splat(7500);
    let out3 = i32x4_shr(
        i32x4_add(i32x4_sub(i32x4_mul(d, k2217), i32x4_mul(c, k5352)), k7500),
        12,
    );

    (out0, out1, out2, out3)
}

/// Second pass of forward DCT (column pass). Same butterfly pairing as pass 1,
/// but different constants and rounding.
///
/// Scalar column pass (for column i):
///   a = block[i] + block[i+12]         (row0 + row3)
///   b = block[i+4] + block[i+8]        (row1 + row2)
///   c = block[i+4] - block[i+8]        (row1 - row2)
///   d = block[i] - block[i+12]         (row0 - row3)
///   out_row0 = (a + b + 7) >> 4
///   out_row1 = (c*2217 + d*5352 + 12000) >> 16 + (d != 0)
///   out_row2 = (a - b + 7) >> 4
///   out_row3 = (d*2217 - c*5352 + 51000) >> 16
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn dct_butterfly_pass2(in0: v128, in1: v128, in2: v128, in3: v128) -> (v128, v128, v128, v128) {
    let a = i32x4_add(in0, in3);
    let b = i32x4_add(in1, in2);
    let c = i32x4_sub(in1, in2);
    let d = i32x4_sub(in0, in3);

    let k2217 = i32x4_splat(2217);
    let k5352 = i32x4_splat(5352);
    let k7 = i32x4_splat(7);
    let k12000 = i32x4_splat(12000);
    let k51000 = i32x4_splat(51000);

    // out0 = (a + b + 7) >> 4
    let out0 = i32x4_shr(i32x4_add(i32x4_add(a, b), k7), 4);

    // out1 = (c * 2217 + d * 5352 + 12000) >> 16 + (d != 0)
    let out1_raw = i32x4_shr(
        i32x4_add(i32x4_add(i32x4_mul(c, k2217), i32x4_mul(d, k5352)), k12000),
        16,
    );
    // d_ne_0: 0xFFFFFFFF when d != 0, 0 when d == 0
    // We need +1 when d != 0, so negate the mask (-(-1) = 1)
    let d_ne_0 = v128_not(i32x4_eq(d, i32x4_splat(0)));
    let out1 = i32x4_sub(out1_raw, d_ne_0);

    // out2 = (a - b + 7) >> 4
    let out2 = i32x4_shr(i32x4_add(i32x4_sub(a, b), k7), 4);

    // out3 = (d * 2217 - c * 5352 + 51000) >> 16
    let out3 = i32x4_shr(
        i32x4_add(i32x4_sub(i32x4_mul(d, k2217), i32x4_mul(c, k5352)), k51000),
        16,
    );

    (out0, out1, out2, out3)
}

/// Fused residual computation + forward DCT for a single 4x4 block.
/// Takes flat u8 source and reference arrays (stride=4), outputs i32 coefficients.
///
/// Fuses: residual = src - ref, then DCT on residual.
/// Avoids intermediate i32 storage by computing diff in i16 then widening to i32.
#[cfg(target_arch = "wasm32")]
#[arcane]
pub(crate) fn ftransform_from_u8_4x4_wasm(
    _token: Wasm128Token,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    ftransform_from_u8_4x4_wasm_impl(_token, src, ref_)
}

/// Inner #[rite] implementation of fused residual+DCT for wasm.
#[cfg(target_arch = "wasm32")]
#[rite]
pub(crate) fn ftransform_from_u8_4x4_wasm_impl(
    _token: Wasm128Token,
    src: &[u8; 16],
    ref_: &[u8; 16],
) -> [i32; 16] {
    // Load src and ref as u8x16
    let src_vec = u8x16(
        src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7], src[8], src[9], src[10],
        src[11], src[12], src[13], src[14], src[15],
    );
    let ref_vec = u8x16(
        ref_[0], ref_[1], ref_[2], ref_[3], ref_[4], ref_[5], ref_[6], ref_[7], ref_[8], ref_[9],
        ref_[10], ref_[11], ref_[12], ref_[13], ref_[14], ref_[15],
    );

    // Zero-extend to i16 and compute diff (src - ref)
    let src_lo = u16x8_extend_low_u8x16(src_vec);
    let src_hi = u16x8_extend_high_u8x16(src_vec);
    let ref_lo = u16x8_extend_low_u8x16(ref_vec);
    let ref_hi = u16x8_extend_high_u8x16(ref_vec);
    let diff_lo = i16x8_sub(src_lo, ref_lo);
    let diff_hi = i16x8_sub(src_hi, ref_hi);

    // Widen i16 → i32 for DCT (4 rows of 4 values)
    let r0 = i32x4_extend_low_i16x8(diff_lo);
    let r1 = i32x4_extend_high_i16x8(diff_lo);
    let r2 = i32x4_extend_low_i16x8(diff_hi);
    let r3 = i32x4_extend_high_i16x8(diff_hi);

    // Forward DCT pass 1: row pass (transpose, butterfly, transpose back)
    let (c0, c1, c2, c3) = transpose4x4(r0, r1, r2, r3);
    let (v0, v1, v2, v3) = dct_butterfly(c0, c1, c2, c3);
    let (t0, t1, t2, t3) = transpose4x4(v0, v1, v2, v3);

    // Forward DCT pass 2: column pass
    let (o0, o1, o2, o3) = dct_butterfly_pass2(t0, t1, t2, t3);

    // Store results
    let mut result = [0i32; 16];
    store_row(&mut result, 0, o0);
    store_row(&mut result, 1, o1);
    store_row(&mut result, 2, o2);
    store_row(&mut result, 3, o3);
    result
}

/// WASM SIMD128 forward DCT. Uses i32x4 arithmetic matching the scalar implementation.
///
/// Scalar pass order: row pass first (for i in 0..4, reads block[i*4..i*4+3]),
/// then column pass (for i in 0..4, reads block[i], block[i+4], ...).
///
/// With row vectors, `dct_butterfly(r0,r1,r2,r3)` processes lane j as
/// butterfly(row0[j], row1[j], row2[j], row3[j]) = column pass.
/// For the row pass, we transpose so each vector holds one row's elements.
#[cfg(target_arch = "wasm32")]
#[rite]
pub(crate) fn dct4x4_wasm_impl(_token: Wasm128Token, block: &mut [i32; 16]) {
    // Load 4 rows
    let r0 = load_row(block, 0);
    let r1 = load_row(block, 1);
    let r2 = load_row(block, 2);
    let r3 = load_row(block, 3);

    // Pass 1: row pass — transpose so each vector holds one row's elements,
    // butterfly processes rows, then transpose back to row-major
    let (c0, c1, c2, c3) = transpose4x4(r0, r1, r2, r3);
    let (v0, v1, v2, v3) = dct_butterfly(c0, c1, c2, c3);
    let (t0, t1, t2, t3) = transpose4x4(v0, v1, v2, v3);

    // Pass 2: column pass — butterfly on row vectors processes columns in parallel
    let (o0, o1, o2, o3) = dct_butterfly_pass2(t0, t1, t2, t3);

    // Store results
    store_row(block, 0, o0);
    store_row(block, 1, o1);
    store_row(block, 2, o2);
    store_row(block, 3, o3);
}
