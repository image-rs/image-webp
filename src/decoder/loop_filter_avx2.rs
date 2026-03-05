//! AVX2-optimized VP8 loop filter.
//!
//! Processes 16 pixels at once by loading entire rows, matching libwebp's approach.
//! For horizontal filtering (vertical edges), uses the transpose technique:
//! load 16 rows × 8 columns, transpose to 8 × 16, filter as vertical, transpose back.
//!
//! Uses archmage for safe SIMD intrinsics with token-based CPU feature verification.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::intrinsics::x86_64 as simd_mem;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{arcane, X64V3Token};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::convert::TryFrom;

/// Maximum stride for bounds-check-free filtering.
/// WebP max dimension is 16383, rounded up to MB boundary = 16384.
const MAX_STRIDE: usize = 16384;

/// Fixed region size for simple vertical filter: 4 rows (p1, p0, q0, q1) plus 16 bytes.
const V_FILTER_REGION: usize = 3 * MAX_STRIDE + 16;

/// Fixed region size for normal vertical filter: 8 rows (p3-p0, q0-q3) plus 16 bytes.
const V_FILTER_NORMAL_REGION: usize = 7 * MAX_STRIDE + 16;

/// Compute the "needs filter" mask for simple filter.
/// Returns a mask where each byte is 0xFF if the pixel should be filtered, 0x00 otherwise.
/// Condition: |p0 - q0| * 2 + |p1 - q1| / 2 <= thresh
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn needs_filter_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    thresh: i32,
) -> __m128i {
    let t = _mm_set1_epi8(thresh as i8);

    // |p0 - q0|
    let abs_p0_q0 = _mm_or_si128(_mm_subs_epu8(p0, q0), _mm_subs_epu8(q0, p0));

    // |p1 - q1|
    let abs_p1_q1 = _mm_or_si128(_mm_subs_epu8(p1, q1), _mm_subs_epu8(q1, p1));

    // |p0 - q0| * 2
    let doubled = _mm_adds_epu8(abs_p0_q0, abs_p0_q0);

    // |p1 - q1| / 2
    let halved = _mm_and_si128(_mm_srli_epi16(abs_p1_q1, 1), _mm_set1_epi8(0x7F));

    // |p0 - q0| * 2 + |p1 - q1| / 2
    let sum = _mm_adds_epu8(doubled, halved);

    // sum <= thresh  =>  !(sum > thresh)  =>  (thresh - sum) >= 0 using saturating sub
    let exceeds = _mm_subs_epu8(sum, t);
    _mm_cmpeq_epi8(exceeds, _mm_setzero_si128())
}

/// Get the base delta for the simple filter: clamp(p1 - q1 + 3*(q0 - p0))
/// Uses signed arithmetic with sign bit flipping.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn get_base_delta_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
) -> __m128i {
    // Convert to signed by XOR with 0x80
    let sign = _mm_set1_epi8(-128i8);
    let p1s = _mm_xor_si128(p1, sign);
    let p0s = _mm_xor_si128(p0, sign);
    let q0s = _mm_xor_si128(q0, sign);
    let q1s = _mm_xor_si128(q1, sign);

    // p1 - q1 (saturating)
    let p1_q1 = _mm_subs_epi8(p1s, q1s);

    // q0 - p0 (saturating)
    let q0_p0 = _mm_subs_epi8(q0s, p0s);

    // p1 - q1 + 3*(q0 - p0) = p1 - q1 + (q0 - p0) + (q0 - p0) + (q0 - p0)
    let s1 = _mm_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm_adds_epi8(s1, q0_p0);
    _mm_adds_epi8(s2, q0_p0)
}

/// Signed right shift by 3 for packed bytes (in signed domain).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn signed_shift_right_3(_token: X64V3Token, v: __m128i) -> __m128i {
    // For signed bytes, we need to handle sign extension properly.
    // Unpack to 16-bit, shift, pack back.
    let lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 11); // sign-extend and shift
    let hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 11);
    _mm_packs_epi16(lo, hi)
}

/// Apply the simple filter to p0 and q0 given the filter value.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn do_simple_filter_16(_token: X64V3Token, p0: &mut __m128i, q0: &mut __m128i, fl: __m128i) {
    let sign = _mm_set1_epi8(-128i8);
    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);

    // v3 = (fl + 3) >> 3
    // v4 = (fl + 4) >> 3
    let v3 = _mm_adds_epi8(fl, k3);
    let v4 = _mm_adds_epi8(fl, k4);

    let v3 = signed_shift_right_3(_token, v3);
    let v4 = signed_shift_right_3(_token, v4);

    // Convert p0, q0 to signed
    let mut p0s = _mm_xor_si128(*p0, sign);
    let mut q0s = _mm_xor_si128(*q0, sign);

    // q0 -= v4, p0 += v3
    q0s = _mm_subs_epi8(q0s, v4);
    p0s = _mm_adds_epi8(p0s, v3);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(p0s, sign);
    *q0 = _mm_xor_si128(q0s, sign);
}

/// Apply simple vertical filter to 16 pixels across a horizontal edge.
///
/// This filters the edge between row (point - stride) and row (point).
/// Processes 16 consecutive pixels in a single call.
///
/// Buffer must have at least point + stride + 16 bytes.
/// point must be >= 2 * stride (for p1 access).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_v_filter16(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    // Requires pixel buffer to have FILTER_PADDING bytes at the end.
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 2 * stride;
    let region: &mut [u8; V_FILTER_REGION] =
        <&mut [u8; V_FILTER_REGION]>::try_from(&mut pixels[start..start + V_FILTER_REGION])
            .expect("simple_v_filter16: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region - compiler proves these are in-bounds
    let off_p1 = 0;
    let off_p0 = stride;
    let off_q0 = 2 * stride;
    let off_q1 = 3 * stride;

    // Load 16 pixels from each row - NO per-access bounds checks
    let p1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let q1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());

    // Check which pixels need filtering
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);

    // Get filter delta
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);

    // Apply filter
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Store results - NO per-access bounds checks
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
}

/// UNCHECKED: Simple vertical filter with all bounds checks eliminated.
///
/// # Safety
/// Caller must ensure:
/// - `point >= 2 * stride`
/// - `pixels.len() >= point + stride + 16`
/// - `stride <= MAX_STRIDE`
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
pub unsafe fn simple_v_filter16_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Compute row pointers directly - NO bounds checks
    let (p1_ptr, p0_ptr, q0_ptr, q1_ptr) = unsafe {
        (
            ptr.add(point - 2 * stride),
            ptr.add(point - stride),
            ptr.add(point),
            ptr.add(point + stride),
        )
    };

    // Load 16 pixels from each row
    let (p1, mut p0, mut q0, q1) = unsafe {
        (
            _mm_loadu_si128(p1_ptr as *const __m128i),
            _mm_loadu_si128(p0_ptr as *const __m128i),
            _mm_loadu_si128(q0_ptr as *const __m128i),
            _mm_loadu_si128(q1_ptr as *const __m128i),
        )
    };

    // Check which pixels need filtering
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);

    // Get filter delta
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);

    // Apply filter
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Store results
    unsafe {
        _mm_storeu_si128(p0_ptr as *mut __m128i, p0);
        _mm_storeu_si128(q0_ptr as *mut __m128i, q0);
    }
}

// ============================================================================
// AVX2 32-pixel filters (2x throughput vs SSE2)
// ============================================================================

/// Fixed region size for 32-pixel simple vertical filter.
const V_FILTER_REGION_32: usize = 3 * MAX_STRIDE + 32;

/// Compute "needs filter" mask for 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn needs_filter_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    thresh: i32,
) -> __m256i {
    let t = _mm256_set1_epi8(thresh as i8);

    // |p0 - q0|
    let abs_p0_q0 = _mm256_or_si256(_mm256_subs_epu8(p0, q0), _mm256_subs_epu8(q0, p0));

    // |p1 - q1|
    let abs_p1_q1 = _mm256_or_si256(_mm256_subs_epu8(p1, q1), _mm256_subs_epu8(q1, p1));

    // |p0 - q0| * 2
    let doubled = _mm256_adds_epu8(abs_p0_q0, abs_p0_q0);

    // |p1 - q1| / 2
    let halved = _mm256_and_si256(_mm256_srli_epi16(abs_p1_q1, 1), _mm256_set1_epi8(0x7F));

    // |p0 - q0| * 2 + |p1 - q1| / 2
    let sum = _mm256_adds_epu8(doubled, halved);

    // sum <= thresh
    let exceeds = _mm256_subs_epu8(sum, t);
    _mm256_cmpeq_epi8(exceeds, _mm256_setzero_si256())
}

/// Get base delta for 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn get_base_delta_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
) -> __m256i {
    // Convert to signed by XOR with 0x80
    let sign = _mm256_set1_epi8(-128i8);
    let p1s = _mm256_xor_si256(p1, sign);
    let p0s = _mm256_xor_si256(p0, sign);
    let q0s = _mm256_xor_si256(q0, sign);
    let q1s = _mm256_xor_si256(q1, sign);

    // p1 - q1 (saturating)
    let p1_q1 = _mm256_subs_epi8(p1s, q1s);

    // q0 - p0 (saturating)
    let q0_p0 = _mm256_subs_epi8(q0s, p0s);

    // p1 - q1 + 3*(q0 - p0)
    let s1 = _mm256_adds_epi8(p1_q1, q0_p0);
    let s2 = _mm256_adds_epi8(s1, q0_p0);
    _mm256_adds_epi8(s2, q0_p0)
}

/// Signed right shift by 3 for packed bytes using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn signed_shift_right_3_avx2(_token: X64V3Token, v: __m256i) -> __m256i {
    // Unpack to 16-bit, shift, pack back
    let lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(v, v), 11);
    let hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(v, v), 11);
    _mm256_packs_epi16(lo, hi)
}

/// Apply simple filter to 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn do_simple_filter_32(_token: X64V3Token, p0: &mut __m256i, q0: &mut __m256i, fl: __m256i) {
    let sign = _mm256_set1_epi8(-128i8);
    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);

    // v3 = (fl + 3) >> 3, v4 = (fl + 4) >> 3
    let v3 = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(fl, k3));
    let v4 = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(fl, k4));

    // Convert p0, q0 to signed
    let mut p0s = _mm256_xor_si256(*p0, sign);
    let mut q0s = _mm256_xor_si256(*q0, sign);

    // q0 -= v4, p0 += v3
    q0s = _mm256_subs_epi8(q0s, v4);
    p0s = _mm256_adds_epi8(p0s, v3);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(p0s, sign);
    *q0 = _mm256_xor_si256(q0s, sign);
}

/// Apply simple vertical filter to 32 pixels using AVX2.
/// Processes 32 consecutive pixels in a single call - 2x throughput vs SSE2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_v_filter32(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    thresh: i32,
) {
    // Fixed-region approach for bounds check elimination
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 2 * stride;
    let region: &mut [u8; V_FILTER_REGION_32] =
        <&mut [u8; V_FILTER_REGION_32]>::try_from(&mut pixels[start..start + V_FILTER_REGION_32])
            .expect("simple_v_filter32: buffer too small");

    let off_p1 = 0;
    let off_p0 = stride;
    let off_q0 = 2 * stride;
    let off_q1 = 3 * stride;

    // Load 32 pixels from each row using AVX2
    let p1 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let q1 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());

    // Check which pixels need filtering
    let mask = needs_filter_32(_token, p1, p0, q0, q1, thresh);

    // Get filter delta and mask it
    let fl = get_base_delta_32(_token, p1, p0, q0, q1);
    let fl_masked = _mm256_and_si256(fl, mask);

    // Apply filter
    do_simple_filter_32(_token, &mut p0, &mut q0, fl_masked);

    // Store results
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
}

// =============================================================================
// AVX2 32-pixel Normal Filter Helpers
// =============================================================================

/// Fixed region size for normal vertical filter with 32 pixels.
const V_FILTER_NORMAL_REGION_32: usize = 7 * MAX_STRIDE + 32;

/// Check if normal filtering is needed for 32 pixels using AVX2.
/// Returns a mask where each byte is 0xFF if the pixel should be filtered.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn needs_filter_normal_32(
    _token: X64V3Token,
    p3: __m256i,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
    q3: __m256i,
    edge_limit: i32,
    interior_limit: i32,
) -> __m256i {
    // First check simple threshold
    let simple_mask = needs_filter_32(_token, p1, p0, q0, q1, edge_limit);

    let i_limit = _mm256_set1_epi8(interior_limit as i8);

    // Helper macro for abs diff
    macro_rules! abs_diff {
        ($a:expr, $b:expr) => {
            _mm256_or_si256(_mm256_subs_epu8($a, $b), _mm256_subs_epu8($b, $a))
        };
    }

    let d_p3_p2 = abs_diff!(p3, p2);
    let d_p2_p1 = abs_diff!(p2, p1);
    let d_p1_p0 = abs_diff!(p1, p0);
    let d_q0_q1 = abs_diff!(q0, q1);
    let d_q1_q2 = abs_diff!(q1, q2);
    let d_q2_q3 = abs_diff!(q2, q3);

    // Take max of all differences
    let max1 = _mm256_max_epu8(d_p3_p2, d_p2_p1);
    let max2 = _mm256_max_epu8(d_p1_p0, d_q0_q1);
    let max3 = _mm256_max_epu8(d_q1_q2, d_q2_q3);
    let max4 = _mm256_max_epu8(max1, max2);
    let max_diff = _mm256_max_epu8(max3, max4);

    // Check if max_diff <= interior_limit
    let exceeds = _mm256_subs_epu8(max_diff, i_limit);
    let interior_ok = _mm256_cmpeq_epi8(exceeds, _mm256_setzero_si256());

    // Both conditions must be true
    _mm256_and_si256(simple_mask, interior_ok)
}

/// Check high edge variance for 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn high_edge_variance_32(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    hev_thresh: i32,
) -> __m256i {
    let t = _mm256_set1_epi8(hev_thresh as i8);

    // |p1 - p0|
    let d_p1_p0 = _mm256_or_si256(_mm256_subs_epu8(p1, p0), _mm256_subs_epu8(p0, p1));

    // |q1 - q0|
    let d_q1_q0 = _mm256_or_si256(_mm256_subs_epu8(q1, q0), _mm256_subs_epu8(q0, q1));

    // Check if either > thresh
    let p_exceeds = _mm256_subs_epu8(d_p1_p0, t);
    let q_exceeds = _mm256_subs_epu8(d_q1_q0, t);

    // hev = true if exceeds > 0
    let p_hev = _mm256_xor_si256(
        _mm256_cmpeq_epi8(p_exceeds, _mm256_setzero_si256()),
        _mm256_set1_epi8(-1),
    );
    let q_hev = _mm256_xor_si256(
        _mm256_cmpeq_epi8(q_exceeds, _mm256_setzero_si256()),
        _mm256_set1_epi8(-1),
    );

    _mm256_or_si256(p_hev, q_hev)
}

/// Signed right shift by 1 for packed bytes using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn signed_shift_right_1_avx2(_token: X64V3Token, v: __m256i) -> __m256i {
    let lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(v, v), 9);
    let hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(v, v), 9);
    _mm256_packs_epi16(lo, hi)
}

/// Apply the subblock/inner filter (DoFilter4) for 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn do_filter4_32(
    _token: X64V3Token,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    mask: __m256i,
    hev: __m256i,
) {
    let sign = _mm256_set1_epi8(-128i8);

    // Convert to signed
    let p1s = _mm256_xor_si256(*p1, sign);
    let p0s = _mm256_xor_si256(*p0, sign);
    let q0s = _mm256_xor_si256(*q0, sign);
    let q1s = _mm256_xor_si256(*q1, sign);

    // Compute base filter value
    let outer = _mm256_subs_epi8(p1s, q1s);
    let outer_masked = _mm256_and_si256(outer, hev);

    let q0_p0 = _mm256_subs_epi8(q0s, p0s);

    // a = outer + 3*(q0 - p0)
    let a = _mm256_adds_epi8(outer_masked, q0_p0);
    let a = _mm256_adds_epi8(a, q0_p0);
    let a = _mm256_adds_epi8(a, q0_p0);

    // Apply mask
    let a = _mm256_and_si256(a, mask);

    // Compute filter1 = (a + 4) >> 3 and filter2 = (a + 3) >> 3
    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);

    let f1 = _mm256_adds_epi8(a, k4);
    let f2 = _mm256_adds_epi8(a, k3);

    let f1 = signed_shift_right_3_avx2(_token, f1);
    let f2 = signed_shift_right_3_avx2(_token, f2);

    // Update p0, q0
    let new_p0s = _mm256_adds_epi8(p0s, f2);
    let new_q0s = _mm256_subs_epi8(q0s, f1);

    // For !hev case, also update p1, q1
    let a2 = _mm256_adds_epi8(f1, _mm256_set1_epi8(1));
    let a2 = signed_shift_right_1_avx2(_token, a2);
    let a2 = _mm256_andnot_si256(hev, a2);
    let a2 = _mm256_and_si256(a2, mask);

    let new_p1s = _mm256_adds_epi8(p1s, a2);
    let new_q1s = _mm256_subs_epi8(q1s, a2);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(new_p0s, sign);
    *q0 = _mm256_xor_si256(new_q0s, sign);
    *p1 = _mm256_xor_si256(new_p1s, sign);
    *q1 = _mm256_xor_si256(new_q1s, sign);
}

/// Helper for filter6 wide path using AVX2 - processes 16 pixels in 16-bit precision.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn filter6_wide_half_avx2(
    _token: X64V3Token,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i, __m256i, __m256i) {
    // Sign extend to 16-bit
    let p2_16 = _mm256_srai_epi16(p2, 8);
    let p1_16 = _mm256_srai_epi16(p1, 8);
    let p0_16 = _mm256_srai_epi16(p0, 8);
    let q0_16 = _mm256_srai_epi16(q0, 8);
    let q1_16 = _mm256_srai_epi16(q1, 8);
    let q2_16 = _mm256_srai_epi16(q2, 8);

    // w = clamp(p1 - q1 + 3*(q0 - p0))
    let p1_q1 = _mm256_sub_epi16(p1_16, q1_16);
    let q0_p0 = _mm256_sub_epi16(q0_16, p0_16);
    let three_q0_p0 = _mm256_add_epi16(_mm256_add_epi16(q0_p0, q0_p0), q0_p0);
    let w = _mm256_add_epi16(p1_q1, three_q0_p0);
    let w = _mm256_max_epi16(
        _mm256_min_epi16(w, _mm256_set1_epi16(127)),
        _mm256_set1_epi16(-128),
    );

    // a0 = (27*w + 63) >> 7
    let k27 = _mm256_set1_epi16(27);
    let k18 = _mm256_set1_epi16(18);
    let k9 = _mm256_set1_epi16(9);
    let k63 = _mm256_set1_epi16(63);

    let a0 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k27), k63), 7);
    let a1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k18), k63), 7);
    let a2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(w, k9), k63), 7);

    // Apply adjustments
    let new_p0 = _mm256_add_epi16(p0_16, a0);
    let new_q0 = _mm256_sub_epi16(q0_16, a0);
    let new_p1 = _mm256_add_epi16(p1_16, a1);
    let new_q1 = _mm256_sub_epi16(q1_16, a1);
    let new_p2 = _mm256_add_epi16(p2_16, a2);
    let new_q2 = _mm256_sub_epi16(q2_16, a2);

    // Clamp to [-128, 127]
    let clamp = |v: __m256i| {
        _mm256_max_epi16(
            _mm256_min_epi16(v, _mm256_set1_epi16(127)),
            _mm256_set1_epi16(-128),
        )
    };

    (
        clamp(new_p2),
        clamp(new_p1),
        clamp(new_p0),
        clamp(new_q0),
        clamp(new_q1),
        clamp(new_q2),
    )
}

/// Apply the macroblock/outer filter (DoFilter6) for 32 pixels using AVX2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn do_filter6_32(
    _token: X64V3Token,
    p2: &mut __m256i,
    p1: &mut __m256i,
    p0: &mut __m256i,
    q0: &mut __m256i,
    q1: &mut __m256i,
    q2: &mut __m256i,
    mask: __m256i,
    hev: __m256i,
) {
    let sign = _mm256_set1_epi8(-128i8);
    let not_hev = _mm256_andnot_si256(hev, _mm256_set1_epi8(-1));

    // Convert to signed
    let p2s = _mm256_xor_si256(*p2, sign);
    let p1s = _mm256_xor_si256(*p1, sign);
    let p0s = _mm256_xor_si256(*p0, sign);
    let q0s = _mm256_xor_si256(*q0, sign);
    let q1s = _mm256_xor_si256(*q1, sign);
    let q2s = _mm256_xor_si256(*q2, sign);

    // For hev path: same as simple filter
    let outer = _mm256_subs_epi8(p1s, q1s);
    let outer_hev = _mm256_and_si256(outer, hev);

    let q0_p0 = _mm256_subs_epi8(q0s, p0s);
    let a_hev = _mm256_adds_epi8(outer_hev, q0_p0);
    let a_hev = _mm256_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm256_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm256_and_si256(a_hev, _mm256_and_si256(mask, hev));

    let k3 = _mm256_set1_epi8(3);
    let k4 = _mm256_set1_epi8(4);
    let f1_hev = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(a_hev, k4));
    let f2_hev = signed_shift_right_3_avx2(_token, _mm256_adds_epi8(a_hev, k3));

    // For !hev path: wide filter using 16-bit precision
    // Process low and high halves separately
    let (new_p2_lo, new_p1_lo, new_p0_lo, new_q0_lo, new_q1_lo, new_q2_lo) = filter6_wide_half_avx2(
        _token,
        _mm256_unpacklo_epi8(p2s, p2s),
        _mm256_unpacklo_epi8(p1s, p1s),
        _mm256_unpacklo_epi8(p0s, p0s),
        _mm256_unpacklo_epi8(q0s, q0s),
        _mm256_unpacklo_epi8(q1s, q1s),
        _mm256_unpacklo_epi8(q2s, q2s),
    );

    let (new_p2_hi, new_p1_hi, new_p0_hi, new_q0_hi, new_q1_hi, new_q2_hi) = filter6_wide_half_avx2(
        _token,
        _mm256_unpackhi_epi8(p2s, p2s),
        _mm256_unpackhi_epi8(p1s, p1s),
        _mm256_unpackhi_epi8(p0s, p0s),
        _mm256_unpackhi_epi8(q0s, q0s),
        _mm256_unpackhi_epi8(q1s, q1s),
        _mm256_unpackhi_epi8(q2s, q2s),
    );

    // Pack back to bytes
    let new_p2_wide = _mm256_packs_epi16(new_p2_lo, new_p2_hi);
    let new_p1_wide = _mm256_packs_epi16(new_p1_lo, new_p1_hi);
    let new_p0_wide = _mm256_packs_epi16(new_p0_lo, new_p0_hi);
    let new_q0_wide = _mm256_packs_epi16(new_q0_lo, new_q0_hi);
    let new_q1_wide = _mm256_packs_epi16(new_q1_lo, new_q1_hi);
    let new_q2_wide = _mm256_packs_epi16(new_q2_lo, new_q2_hi);

    // Blend hev and !hev results
    let mask_not_hev = _mm256_and_si256(mask, not_hev);

    // For p0, q0: use hev result where hev, wide result where !hev
    let new_p0s = _mm256_adds_epi8(p0s, f2_hev);
    let new_q0s = _mm256_subs_epi8(q0s, f1_hev);

    // Blend using blendv
    let final_p0s = _mm256_blendv_epi8(new_p0s, new_p0_wide, mask_not_hev);
    let final_q0s = _mm256_blendv_epi8(new_q0s, new_q0_wide, mask_not_hev);

    // For p1, q1, p2, q2: only update when !hev
    let final_p1s = _mm256_blendv_epi8(p1s, new_p1_wide, mask_not_hev);
    let final_q1s = _mm256_blendv_epi8(q1s, new_q1_wide, mask_not_hev);
    let final_p2s = _mm256_blendv_epi8(p2s, new_p2_wide, mask_not_hev);
    let final_q2s = _mm256_blendv_epi8(q2s, new_q2_wide, mask_not_hev);

    // Convert back to unsigned
    *p0 = _mm256_xor_si256(final_p0s, sign);
    *q0 = _mm256_xor_si256(final_q0s, sign);
    *p1 = _mm256_xor_si256(final_p1s, sign);
    *q1 = _mm256_xor_si256(final_q1s, sign);
    *p2 = _mm256_xor_si256(final_p2s, sign);
    *q2 = _mm256_xor_si256(final_q2s, sign);
}

/// Apply normal vertical filter (DoFilter4) to 32 pixels across a horizontal edge.
/// This is for subblock edges within a macroblock - 2x throughput vs SSE2.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter32_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION_32] =
        <&mut [u8; V_FILTER_NORMAL_REGION_32]>::try_from(
            &mut pixels[start..start + V_FILTER_NORMAL_REGION_32],
        )
        .expect("normal_v_filter32_inner: buffer too small");

    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 32 pixels each
    let p3 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p3..][..32]).unwrap());
    let p2 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p2..][..32]).unwrap());
    let mut p1 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let mut q1 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());
    let q2 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q2..][..32]).unwrap());
    let q3 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q3..][..32]).unwrap());

    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    do_filter4_32(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p1..][..32]).unwrap(),
        p1,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q1..][..32]).unwrap(),
        q1,
    );
}

/// Apply normal vertical filter (DoFilter6) to 32 pixels across a horizontal macroblock edge.
/// 2x throughput vs SSE2 version.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter32_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION_32] =
        <&mut [u8; V_FILTER_NORMAL_REGION_32]>::try_from(
            &mut pixels[start..start + V_FILTER_NORMAL_REGION_32],
        )
        .expect("normal_v_filter32_edge: buffer too small");

    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 32 pixels each
    let p3 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p3..][..32]).unwrap());
    let mut p2 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p2..][..32]).unwrap());
    let mut p1 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p1..][..32]).unwrap());
    let mut p0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_p0..][..32]).unwrap());
    let mut q0 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q0..][..32]).unwrap());
    let mut q1 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q1..][..32]).unwrap());
    let mut q2 =
        simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q2..][..32]).unwrap());
    let q3 = simd_mem::_mm256_loadu_si256(<&[u8; 32]>::try_from(&region[off_q3..][..32]).unwrap());

    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    do_filter6_32(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p2..][..32]).unwrap(),
        p2,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p1..][..32]).unwrap(),
        p1,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_p0..][..32]).unwrap(),
        p0,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q0..][..32]).unwrap(),
        q0,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q1..][..32]).unwrap(),
        q1,
    );
    simd_mem::_mm256_storeu_si256(
        <&mut [u8; 32]>::try_from(&mut region[off_q2..][..32]).unwrap(),
        q2,
    );
}

/// Transpose an 8x16 matrix of bytes to 16x8.
/// Input: 16 __m128i values, each containing 8 bytes (low 64 bits used).
/// Output: 8 __m128i values, each containing 16 bytes.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_8x16_to_16x8(_token: X64V3Token, rows: &[__m128i; 16]) -> [__m128i; 8] {
    // Stage 1: interleave pairs
    let t0 = _mm_unpacklo_epi8(rows[0], rows[1]);
    let t1 = _mm_unpacklo_epi8(rows[2], rows[3]);
    let t2 = _mm_unpacklo_epi8(rows[4], rows[5]);
    let t3 = _mm_unpacklo_epi8(rows[6], rows[7]);
    let t4 = _mm_unpacklo_epi8(rows[8], rows[9]);
    let t5 = _mm_unpacklo_epi8(rows[10], rows[11]);
    let t6 = _mm_unpacklo_epi8(rows[12], rows[13]);
    let t7 = _mm_unpacklo_epi8(rows[14], rows[15]);

    // Stage 2: interleave 16-bit pairs
    let u0 = _mm_unpacklo_epi16(t0, t1);
    let u1 = _mm_unpackhi_epi16(t0, t1);
    let u2 = _mm_unpacklo_epi16(t2, t3);
    let u3 = _mm_unpackhi_epi16(t2, t3);
    let u4 = _mm_unpacklo_epi16(t4, t5);
    let u5 = _mm_unpackhi_epi16(t4, t5);
    let u6 = _mm_unpacklo_epi16(t6, t7);
    let u7 = _mm_unpackhi_epi16(t6, t7);

    // Stage 3: interleave 32-bit pairs
    let v0 = _mm_unpacklo_epi32(u0, u2);
    let v1 = _mm_unpackhi_epi32(u0, u2);
    let v2 = _mm_unpacklo_epi32(u4, u6);
    let v3 = _mm_unpackhi_epi32(u4, u6);
    let v4 = _mm_unpacklo_epi32(u1, u3);
    let v5 = _mm_unpackhi_epi32(u1, u3);
    let v6 = _mm_unpacklo_epi32(u5, u7);
    let v7 = _mm_unpackhi_epi32(u5, u7);

    // Stage 4: interleave 64-bit to get final columns
    [
        _mm_unpacklo_epi64(v0, v2), // column 0 (p3)
        _mm_unpackhi_epi64(v0, v2), // column 1 (p2)
        _mm_unpacklo_epi64(v1, v3), // column 2 (p1)
        _mm_unpackhi_epi64(v1, v3), // column 3 (p0)
        _mm_unpacklo_epi64(v4, v6), // column 4 (q0)
        _mm_unpackhi_epi64(v4, v6), // column 5 (q1)
        _mm_unpacklo_epi64(v5, v7), // column 6 (q2)
        _mm_unpackhi_epi64(v5, v7), // column 7 (q3)
    ]
}

/// Transpose 4 columns (p1, p0, q0, q1) of 16 bytes each back to 16 rows of 4 bytes.
/// Returns values suitable for storing as 32-bit integers per row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_4x16_to_16x4(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
) -> [i32; 16] {
    // Interleave p1,p0 and q0,q1
    let p1p0_lo = _mm_unpacklo_epi8(p1, p0); // rows 0-7: p1[i], p0[i]
    let p1p0_hi = _mm_unpackhi_epi8(p1, p0); // rows 8-15
    let q0q1_lo = _mm_unpacklo_epi8(q0, q1);
    let q0q1_hi = _mm_unpackhi_epi8(q0, q1);

    // Combine to 32-bit: p1, p0, q0, q1
    let r0 = _mm_unpacklo_epi16(p1p0_lo, q0q1_lo); // rows 0-3
    let r1 = _mm_unpackhi_epi16(p1p0_lo, q0q1_lo); // rows 4-7
    let r2 = _mm_unpacklo_epi16(p1p0_hi, q0q1_hi); // rows 8-11
    let r3 = _mm_unpackhi_epi16(p1p0_hi, q0q1_hi); // rows 12-15

    // Extract 32-bit values
    let mut result = [0i32; 16];
    result[0] = _mm_extract_epi32(r0, 0);
    result[1] = _mm_extract_epi32(r0, 1);
    result[2] = _mm_extract_epi32(r0, 2);
    result[3] = _mm_extract_epi32(r0, 3);
    result[4] = _mm_extract_epi32(r1, 0);
    result[5] = _mm_extract_epi32(r1, 1);
    result[6] = _mm_extract_epi32(r1, 2);
    result[7] = _mm_extract_epi32(r1, 3);
    result[8] = _mm_extract_epi32(r2, 0);
    result[9] = _mm_extract_epi32(r2, 1);
    result[10] = _mm_extract_epi32(r2, 2);
    result[11] = _mm_extract_epi32(r2, 3);
    result[12] = _mm_extract_epi32(r3, 0);
    result[13] = _mm_extract_epi32(r3, 1);
    result[14] = _mm_extract_epi32(r3, 2);
    result[15] = _mm_extract_epi32(r3, 3);
    result
}

/// Apply simple horizontal filter to 16 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses the transpose technique: load 16 rows × 8 columns, transpose,
/// apply vertical filter logic, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_h_filter16(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 15) * stride + x + 4 <= pixels.len(),
        "simple_h_filter16: bounds check failed"
    );

    // Load 16 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    // The edge is between p0 (x-1) and q0 (x), so we load from x-4 to x+3
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8: now we have 8 columns of 16 pixels each
    let cols = transpose_8x16_to_16x8(_token, &rows);
    // cols[2] = p1 (16 pixels from 16 different rows)
    // cols[3] = p0
    // cols[4] = q0
    // cols[5] = q1

    let p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let q1 = cols[5];

    // Apply simple filter (same logic as vertical, but on transposed data)
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Transpose back: convert 4 columns of 16 to 16 rows of 4
    let packed = transpose_4x16_to_16x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row (p1, p0, q0, q1) using safe store
    for (i, &val) in packed.iter().enumerate() {
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
}

/// Apply simple vertical filter to entire macroblock edge (16 pixels).
/// This is the main entry point for filtering horizontal edges between macroblocks.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_filter_mb_edge_v(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let point = mb_y * 16 * stride + mb_x * 16;
    simple_v_filter16(_token, pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to entire macroblock edge (16 rows).
/// This is the main entry point for filtering vertical edges between macroblocks.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_filter_mb_edge_h(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16;
    let y_start = mb_y * 16;
    simple_h_filter16(_token, pixels, x, y_start, stride, thresh);
}

/// Apply simple vertical filter to a subblock edge within a macroblock.
/// y_offset is the row offset within the macroblock (4, 8, or 12).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_filter_subblock_edge_v(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    y_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let point = (mb_y * 16 + y_offset) * stride + mb_x * 16;
    simple_v_filter16(_token, pixels, point, stride, thresh);
}

/// Apply simple horizontal filter to a subblock edge within a macroblock.
/// x_offset is the column offset within the macroblock (4, 8, or 12).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn simple_filter_subblock_edge_h(
    _token: X64V3Token,
    pixels: &mut [u8],
    mb_y: usize,
    mb_x: usize,
    x_offset: usize,
    stride: usize,
    thresh: i32,
) {
    let x = mb_x * 16 + x_offset;
    let y_start = mb_y * 16;
    simple_h_filter16(_token, pixels, x, y_start, stride, thresh);
}

// =============================================================================
// Normal filter implementation (DoFilter4 equivalent from libwebp)
// =============================================================================

/// Check if pixels need filtering using the full normal filter threshold.
/// Condition: simple_threshold AND all interior differences <= interior_limit
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn needs_filter_normal_16(
    _token: X64V3Token,
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    edge_limit: i32,
    interior_limit: i32,
) -> __m128i {
    // First check simple threshold
    let simple_mask = needs_filter_16(_token, p1, p0, q0, q1, edge_limit);

    let i_limit = _mm_set1_epi8(interior_limit as i8);

    // Check interior differences: all must be <= interior_limit
    // |p3-p2|, |p2-p1|, |p1-p0|, |q0-q1|, |q1-q2|, |q2-q3|

    // Helper macro for abs diff
    macro_rules! abs_diff {
        ($a:expr, $b:expr) => {
            _mm_or_si128(_mm_subs_epu8($a, $b), _mm_subs_epu8($b, $a))
        };
    }

    let d_p3_p2 = abs_diff!(p3, p2);
    let d_p2_p1 = abs_diff!(p2, p1);
    let d_p1_p0 = abs_diff!(p1, p0);
    let d_q0_q1 = abs_diff!(q0, q1);
    let d_q1_q2 = abs_diff!(q1, q2);
    let d_q2_q3 = abs_diff!(q2, q3);

    // Take max of all differences
    let max1 = _mm_max_epu8(d_p3_p2, d_p2_p1);
    let max2 = _mm_max_epu8(d_p1_p0, d_q0_q1);
    let max3 = _mm_max_epu8(d_q1_q2, d_q2_q3);
    let max4 = _mm_max_epu8(max1, max2);
    let max_diff = _mm_max_epu8(max3, max4);

    // Check if max_diff <= interior_limit
    let exceeds = _mm_subs_epu8(max_diff, i_limit);
    let interior_ok = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());

    // Both conditions must be true
    _mm_and_si128(simple_mask, interior_ok)
}

/// Check high edge variance: |p1 - p0| > thresh OR |q1 - q0| > thresh
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn high_edge_variance_16(
    _token: X64V3Token,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    hev_thresh: i32,
) -> __m128i {
    let t = _mm_set1_epi8(hev_thresh as i8);

    // |p1 - p0|
    let d_p1_p0 = _mm_or_si128(_mm_subs_epu8(p1, p0), _mm_subs_epu8(p0, p1));

    // |q1 - q0|
    let d_q1_q0 = _mm_or_si128(_mm_subs_epu8(q1, q0), _mm_subs_epu8(q0, q1));

    // Check if either > thresh
    // subs_epu8 saturates at 0, so d > t means d - t > 0
    let p_exceeds = _mm_subs_epu8(d_p1_p0, t);
    let q_exceeds = _mm_subs_epu8(d_q1_q0, t);

    // hev = true if exceeds > 0 (i.e., not equal to zero)
    let p_hev = _mm_xor_si128(
        _mm_cmpeq_epi8(p_exceeds, _mm_setzero_si128()),
        _mm_set1_epi8(-1),
    );
    let q_hev = _mm_xor_si128(
        _mm_cmpeq_epi8(q_exceeds, _mm_setzero_si128()),
        _mm_set1_epi8(-1),
    );

    _mm_or_si128(p_hev, q_hev)
}

/// Apply the subblock/inner filter (DoFilter4 from libwebp).
/// When hev=true: only modify p0, q0 (use outer taps)
/// When hev=false: modify p1, p0, q0, q1
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn do_filter4_16(
    _token: X64V3Token,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    mask: __m128i,
    hev: __m128i,
) {
    let sign = _mm_set1_epi8(-128i8);

    // Convert to signed
    let p1s = _mm_xor_si128(*p1, sign);
    let p0s = _mm_xor_si128(*p0, sign);
    let q0s = _mm_xor_si128(*q0, sign);
    let q1s = _mm_xor_si128(*q1, sign);

    // Compute base filter value
    // When hev: use outer taps (p1 - q1)
    // When !hev: no outer taps
    let outer = _mm_subs_epi8(p1s, q1s);
    let outer_masked = _mm_and_si128(outer, hev); // Only use outer when hev

    // q0 - p0 (will multiply by 3)
    let q0_p0 = _mm_subs_epi8(q0s, p0s);

    // a = outer + 3*(q0 - p0)
    let a = _mm_adds_epi8(outer_masked, q0_p0);
    let a = _mm_adds_epi8(a, q0_p0);
    let a = _mm_adds_epi8(a, q0_p0);

    // Apply mask
    let a = _mm_and_si128(a, mask);

    // Compute filter1 = (a + 4) >> 3 and filter2 = (a + 3) >> 3
    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);

    let f1 = _mm_adds_epi8(a, k4);
    let f2 = _mm_adds_epi8(a, k3);

    let f1 = signed_shift_right_3(_token, f1);
    let f2 = signed_shift_right_3(_token, f2);

    // Update p0, q0
    let new_p0s = _mm_adds_epi8(p0s, f2);
    let new_q0s = _mm_subs_epi8(q0s, f1);

    // For !hev case, also update p1, q1
    // a2 = (f1 + 1) >> 1 -- spread the filter to outer pixels
    let a2 = _mm_adds_epi8(f1, _mm_set1_epi8(1));
    let a2 = signed_shift_right_1(_token, a2);
    let a2 = _mm_andnot_si128(hev, a2); // Only when !hev
    let a2 = _mm_and_si128(a2, mask); // Only when filtered

    let new_p1s = _mm_adds_epi8(p1s, a2);
    let new_q1s = _mm_subs_epi8(q1s, a2);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(new_p0s, sign);
    *q0 = _mm_xor_si128(new_q0s, sign);
    *p1 = _mm_xor_si128(new_p1s, sign);
    *q1 = _mm_xor_si128(new_q1s, sign);
}

/// Signed right shift by 1 for packed bytes
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn signed_shift_right_1(_token: X64V3Token, v: __m128i) -> __m128i {
    let lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 9);
    let hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 9);
    _mm_packs_epi16(lo, hi)
}

/// Apply the macroblock/outer filter (DoFilter6 from libwebp).
/// When hev=true: only modify p0, q0 (use outer taps)
/// When hev=false: modify p2, p1, p0, q0, q1, q2 with weighted filter
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn do_filter6_16(
    _token: X64V3Token,
    p2: &mut __m128i,
    p1: &mut __m128i,
    p0: &mut __m128i,
    q0: &mut __m128i,
    q1: &mut __m128i,
    q2: &mut __m128i,
    mask: __m128i,
    hev: __m128i,
) {
    let sign = _mm_set1_epi8(-128i8);
    let not_hev = _mm_andnot_si128(hev, _mm_set1_epi8(-1));

    // Convert to signed
    let p2s = _mm_xor_si128(*p2, sign);
    let p1s = _mm_xor_si128(*p1, sign);
    let p0s = _mm_xor_si128(*p0, sign);
    let q0s = _mm_xor_si128(*q0, sign);
    let q1s = _mm_xor_si128(*q1, sign);
    let q2s = _mm_xor_si128(*q2, sign);

    // For hev path: same as simple filter
    let outer = _mm_subs_epi8(p1s, q1s);
    let outer_hev = _mm_and_si128(outer, hev);

    let q0_p0 = _mm_subs_epi8(q0s, p0s);
    let a_hev = _mm_adds_epi8(outer_hev, q0_p0);
    let a_hev = _mm_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm_adds_epi8(a_hev, q0_p0);
    let a_hev = _mm_and_si128(a_hev, _mm_and_si128(mask, hev));

    let k3 = _mm_set1_epi8(3);
    let k4 = _mm_set1_epi8(4);
    let f1_hev = signed_shift_right_3(_token, _mm_adds_epi8(a_hev, k4));
    let f2_hev = signed_shift_right_3(_token, _mm_adds_epi8(a_hev, k3));

    // For !hev path: wide filter using 16-bit precision
    // w = clamp(p1 - q1 + 3*(q0 - p0))
    // a0 = (27*w + 63) >> 7  (applied to p0, q0)
    // a1 = (18*w + 63) >> 7  (applied to p1, q1)
    // a2 = (9*w + 63) >> 7   (applied to p2, q2)

    // We need 16-bit precision for the multiply
    // Process low and high halves separately
    let (new_p2_lo, new_p1_lo, new_p0_lo, new_q0_lo, new_q1_lo, new_q2_lo) = filter6_wide_half(
        _token,
        _mm_unpacklo_epi8(p2s, p2s),
        _mm_unpacklo_epi8(p1s, p1s),
        _mm_unpacklo_epi8(p0s, p0s),
        _mm_unpacklo_epi8(q0s, q0s),
        _mm_unpacklo_epi8(q1s, q1s),
        _mm_unpacklo_epi8(q2s, q2s),
    );

    let (new_p2_hi, new_p1_hi, new_p0_hi, new_q0_hi, new_q1_hi, new_q2_hi) = filter6_wide_half(
        _token,
        _mm_unpackhi_epi8(p2s, p2s),
        _mm_unpackhi_epi8(p1s, p1s),
        _mm_unpackhi_epi8(p0s, p0s),
        _mm_unpackhi_epi8(q0s, q0s),
        _mm_unpackhi_epi8(q1s, q1s),
        _mm_unpackhi_epi8(q2s, q2s),
    );

    // Pack back to bytes
    let new_p2_wide = _mm_packs_epi16(new_p2_lo, new_p2_hi);
    let new_p1_wide = _mm_packs_epi16(new_p1_lo, new_p1_hi);
    let new_p0_wide = _mm_packs_epi16(new_p0_lo, new_p0_hi);
    let new_q0_wide = _mm_packs_epi16(new_q0_lo, new_q0_hi);
    let new_q1_wide = _mm_packs_epi16(new_q1_lo, new_q1_hi);
    let new_q2_wide = _mm_packs_epi16(new_q2_lo, new_q2_hi);

    // Blend hev and !hev results
    let mask_not_hev = _mm_and_si128(mask, not_hev);

    // For p0, q0: use hev result where hev, wide result where !hev
    let new_p0s = _mm_adds_epi8(p0s, f2_hev); // hev path
    let new_q0s = _mm_subs_epi8(q0s, f1_hev);

    // Blend: select wide where !hev, hev result where hev (and filtered)
    let final_p0s = _mm_blendv_epi8(new_p0s, new_p0_wide, mask_not_hev);
    let final_q0s = _mm_blendv_epi8(new_q0s, new_q0_wide, mask_not_hev);

    // For p1, q1, p2, q2: only update when !hev
    let final_p1s = _mm_blendv_epi8(p1s, new_p1_wide, mask_not_hev);
    let final_q1s = _mm_blendv_epi8(q1s, new_q1_wide, mask_not_hev);
    let final_p2s = _mm_blendv_epi8(p2s, new_p2_wide, mask_not_hev);
    let final_q2s = _mm_blendv_epi8(q2s, new_q2_wide, mask_not_hev);

    // Convert back to unsigned
    *p0 = _mm_xor_si128(final_p0s, sign);
    *q0 = _mm_xor_si128(final_q0s, sign);
    *p1 = _mm_xor_si128(final_p1s, sign);
    *q1 = _mm_xor_si128(final_q1s, sign);
    *p2 = _mm_xor_si128(final_p2s, sign);
    *q2 = _mm_xor_si128(final_q2s, sign);
}

/// Helper for filter6 wide path - processes 8 pixels in 16-bit precision
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn filter6_wide_half(
    _token: X64V3Token,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // Sign extend to 16-bit (values are duplicated, take every other)
    let p2_16 = _mm_srai_epi16(p2, 8);
    let p1_16 = _mm_srai_epi16(p1, 8);
    let p0_16 = _mm_srai_epi16(p0, 8);
    let q0_16 = _mm_srai_epi16(q0, 8);
    let q1_16 = _mm_srai_epi16(q1, 8);
    let q2_16 = _mm_srai_epi16(q2, 8);

    // w = clamp(p1 - q1 + 3*(q0 - p0))
    let p1_q1 = _mm_sub_epi16(p1_16, q1_16);
    let q0_p0 = _mm_sub_epi16(q0_16, p0_16);
    let three_q0_p0 = _mm_add_epi16(_mm_add_epi16(q0_p0, q0_p0), q0_p0);
    let w = _mm_add_epi16(p1_q1, three_q0_p0);
    let w = _mm_max_epi16(_mm_min_epi16(w, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    // a0 = (27*w + 63) >> 7
    let k27 = _mm_set1_epi16(27);
    let k18 = _mm_set1_epi16(18);
    let k9 = _mm_set1_epi16(9);
    let k63 = _mm_set1_epi16(63);

    let a0 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k27), k63), 7);
    let a1 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k18), k63), 7);
    let a2 = _mm_srai_epi16(_mm_add_epi16(_mm_mullo_epi16(w, k9), k63), 7);

    // Apply adjustments
    let new_p0 = _mm_add_epi16(p0_16, a0);
    let new_q0 = _mm_sub_epi16(q0_16, a0);
    let new_p1 = _mm_add_epi16(p1_16, a1);
    let new_q1 = _mm_sub_epi16(q1_16, a1);
    let new_p2 = _mm_add_epi16(p2_16, a2);
    let new_q2 = _mm_sub_epi16(q2_16, a2);

    // Clamp to [-128, 127]
    let clamp =
        |v: __m128i| _mm_max_epi16(_mm_min_epi16(v, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    (
        clamp(new_p2),
        clamp(new_p1),
        clamp(new_p0),
        clamp(new_q0),
        clamp(new_q1),
        clamp(new_q2),
    )
}

/// Apply normal vertical filter (DoFilter4) to 16 pixels across a horizontal edge.
/// This is for subblock edges within a macroblock.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter16_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION] = <&mut [u8; V_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[start..start + V_FILTER_NORMAL_REGION],
    )
    .expect("normal_v_filter16_inner: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region - compiler proves these are in-bounds
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 16 pixels each - NO per-access bounds checks
    let p3 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p3..][..16]).unwrap());
    let p2 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p2..][..16]).unwrap());
    let mut p1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let mut q1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());
    let q2 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q2..][..16]).unwrap());
    let q3 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q3..][..16]).unwrap());

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store results - NO per-access bounds checks
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p1..][..16]).unwrap(),
        p1,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q1..][..16]).unwrap(),
        q1,
    );
}

/// UNCHECKED: Normal vertical filter (DoFilter4) with all bounds checks eliminated.
///
/// # Safety
/// Caller must ensure:
/// - `point >= 4 * stride`
/// - `pixels.len() >= point + 4 * stride`
/// - `stride <= MAX_STRIDE`
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
pub unsafe fn normal_v_filter16_inner_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Compute row pointers directly - NO bounds checks
    let (p3_ptr, p2_ptr, p1_ptr, p0_ptr, q0_ptr, q1_ptr, q2_ptr, q3_ptr) = unsafe {
        (
            ptr.add(point - 4 * stride),
            ptr.add(point - 3 * stride),
            ptr.add(point - 2 * stride),
            ptr.add(point - stride),
            ptr.add(point),
            ptr.add(point + stride),
            ptr.add(point + 2 * stride),
            ptr.add(point + 3 * stride),
        )
    };

    // Load 8 rows of 16 pixels
    let (p3, p2, mut p1, mut p0, mut q0, mut q1, q2, q3) = unsafe {
        (
            _mm_loadu_si128(p3_ptr as *const __m128i),
            _mm_loadu_si128(p2_ptr as *const __m128i),
            _mm_loadu_si128(p1_ptr as *const __m128i),
            _mm_loadu_si128(p0_ptr as *const __m128i),
            _mm_loadu_si128(q0_ptr as *const __m128i),
            _mm_loadu_si128(q1_ptr as *const __m128i),
            _mm_loadu_si128(q2_ptr as *const __m128i),
            _mm_loadu_si128(q3_ptr as *const __m128i),
        )
    };

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store results
    unsafe {
        _mm_storeu_si128(p1_ptr as *mut __m128i, p1);
        _mm_storeu_si128(p0_ptr as *mut __m128i, p0);
        _mm_storeu_si128(q0_ptr as *mut __m128i, q0);
        _mm_storeu_si128(q1_ptr as *mut __m128i, q1);
    }
}

/// Apply normal vertical filter (DoFilter6) to 16 pixels across a horizontal macroblock edge.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter16_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Fixed-region approach: single bounds check, then all interior accesses are check-free.
    debug_assert!(stride <= MAX_STRIDE, "stride exceeds MAX_STRIDE");
    let start = point - 4 * stride;
    let region: &mut [u8; V_FILTER_NORMAL_REGION] = <&mut [u8; V_FILTER_NORMAL_REGION]>::try_from(
        &mut pixels[start..start + V_FILTER_NORMAL_REGION],
    )
    .expect("normal_v_filter16_edge: buffer too small (missing FILTER_PADDING?)");

    // Offsets within fixed region
    let off_p3 = 0;
    let off_p2 = stride;
    let off_p1 = 2 * stride;
    let off_p0 = 3 * stride;
    let off_q0 = 4 * stride;
    let off_q1 = 5 * stride;
    let off_q2 = 6 * stride;
    let off_q3 = 7 * stride;

    // Load 8 rows of 16 pixels each - NO per-access bounds checks
    let p3 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p3..][..16]).unwrap());
    let mut p2 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p2..][..16]).unwrap());
    let mut p1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p1..][..16]).unwrap());
    let mut p0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_p0..][..16]).unwrap());
    let mut q0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q0..][..16]).unwrap());
    let mut q1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q1..][..16]).unwrap());
    let mut q2 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q2..][..16]).unwrap());
    let q3 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&region[off_q3..][..16]).unwrap());

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store results - NO per-access bounds checks
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p2..][..16]).unwrap(),
        p2,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p1..][..16]).unwrap(),
        p1,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_p0..][..16]).unwrap(),
        p0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q0..][..16]).unwrap(),
        q0,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q1..][..16]).unwrap(),
        q1,
    );
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut region[off_q2..][..16]).unwrap(),
        q2,
    );
}

/// UNCHECKED: Normal vertical filter (DoFilter6) with all bounds checks eliminated.
///
/// # Safety
/// Caller must ensure:
/// - `point >= 4 * stride`
/// - `pixels.len() >= point + 4 * stride`
/// - `stride <= MAX_STRIDE`
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
pub unsafe fn normal_v_filter16_edge_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Compute row pointers directly - NO bounds checks
    let (p3_ptr, p2_ptr, p1_ptr, p0_ptr, q0_ptr, q1_ptr, q2_ptr, q3_ptr) = unsafe {
        (
            ptr.add(point - 4 * stride),
            ptr.add(point - 3 * stride),
            ptr.add(point - 2 * stride),
            ptr.add(point - stride),
            ptr.add(point),
            ptr.add(point + stride),
            ptr.add(point + 2 * stride),
            ptr.add(point + 3 * stride),
        )
    };

    // Load 8 rows of 16 pixels
    let (p3, mut p2, mut p1, mut p0, mut q0, mut q1, mut q2, q3) = unsafe {
        (
            _mm_loadu_si128(p3_ptr as *const __m128i),
            _mm_loadu_si128(p2_ptr as *const __m128i),
            _mm_loadu_si128(p1_ptr as *const __m128i),
            _mm_loadu_si128(p0_ptr as *const __m128i),
            _mm_loadu_si128(q0_ptr as *const __m128i),
            _mm_loadu_si128(q1_ptr as *const __m128i),
            _mm_loadu_si128(q2_ptr as *const __m128i),
            _mm_loadu_si128(q3_ptr as *const __m128i),
        )
    };

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store results
    unsafe {
        _mm_storeu_si128(p2_ptr as *mut __m128i, p2);
        _mm_storeu_si128(p1_ptr as *mut __m128i, p1);
        _mm_storeu_si128(p0_ptr as *mut __m128i, p0);
        _mm_storeu_si128(q0_ptr as *mut __m128i, q0);
        _mm_storeu_si128(q1_ptr as *mut __m128i, q1);
        _mm_storeu_si128(q2_ptr as *mut __m128i, q2);
    }
}

/// UNCHECKED: Simple horizontal filter with all bounds checks eliminated.
///
/// # Safety
/// - `x >= 4` and `(y_start + 15) * stride + x + 4 <= pixels.len()`
/// - Caller must verify input is trusted (e.g., self-generated WebP files)
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
pub unsafe fn simple_h_filter16_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    thresh: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Load 16 rows of 8 pixels each using raw pointer arithmetic
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 4);
            rows[i] = _mm_loadu_si64(row_ptr);
        }
    }

    // Transpose 8x16 to 16x8: now we have 8 columns of 16 pixels each
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let q1 = cols[5];

    // Apply simple filter
    let mask = needs_filter_16(_token, p1, p0, q0, q1, thresh);
    let fl = get_base_delta_16(_token, p1, p0, q0, q1);
    let fl_masked = _mm_and_si128(fl, mask);
    do_simple_filter_16(_token, &mut p0, &mut q0, fl_masked);

    // Transpose back: convert 4 columns of 16 to 16 rows of 4
    let packed = transpose_4x16_to_16x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row using raw pointer arithmetic
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 2);
            core::ptr::write_unaligned(row_ptr as *mut i32, packed[i]);
        }
    }
}

/// UNCHECKED: Normal horizontal filter (DoFilter4) for inner edges with all bounds checks eliminated.
///
/// # Safety
/// - `x >= 4` and `(y_start + 15) * stride + x + 4 <= pixels.len()`
/// - Caller must verify input is trusted (e.g., self-generated WebP files)
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
#[inline(always)]
pub unsafe fn normal_h_filter16_inner_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Load 16 rows of 8 pixels each using raw pointer arithmetic
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 4);
            rows[i] = _mm_loadu_si64(row_ptr);
        }
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Transpose back: convert 4 columns of 16 to 16 rows of 4
    let packed = transpose_4x16_to_16x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row using raw pointer arithmetic
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 2);
            core::ptr::write_unaligned(row_ptr as *mut i32, packed[i]);
        }
    }
}

/// UNCHECKED: Normal horizontal filter (DoFilter6) for MB edges with all bounds checks eliminated.
///
/// # Safety
/// - `x >= 4` and `(y_start + 15) * stride + x + 4 <= pixels.len()`
/// - Caller must verify input is trusted (e.g., self-generated WebP files)
#[cfg(all(feature = "simd", feature = "unchecked", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
#[inline(always)]
pub unsafe fn normal_h_filter16_edge_unchecked(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    let ptr = pixels.as_mut_ptr();

    // Load 16 rows of 8 pixels each using raw pointer arithmetic
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 4);
            rows[i] = _mm_loadu_si64(row_ptr);
        }
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Transpose back: convert 6 columns of 16 to 16 rows of 6
    let (packed4, packed2) = transpose_6x16_to_16x6(_token, p2, p1, p0, q0, q1, q2);

    // Store 6 bytes per row using raw pointer arithmetic
    for i in 0..16 {
        unsafe {
            let row_ptr = ptr.add((y_start + i) * stride + x - 3);
            // Store first 4 bytes (p2, p1, p0, q0)
            core::ptr::write_unaligned(row_ptr as *mut i32, packed4[i]);
            // Store next 2 bytes (q1, q2)
            core::ptr::write_unaligned(row_ptr.add(4) as *mut i16, packed2[i]);
        }
    }
}

/// Transpose 6 columns (p2, p1, p0, q0, q1, q2) of 16 bytes each back to 16 rows of 6 bytes.
/// Returns values suitable for storing back to memory.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_6x16_to_16x6(
    _token: X64V3Token,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
) -> ([i32; 16], [i16; 16]) {
    // We need to return 6 bytes per row: p2, p1, p0, q0, q1, q2
    // We'll return two arrays: one with 4-byte values (p2,p1,p0,q0) and one with 2-byte values (q1,q2)

    // Interleave pairs
    let p2p1_lo = _mm_unpacklo_epi8(p2, p1); // rows 0-7: p2[i], p1[i]
    let p2p1_hi = _mm_unpackhi_epi8(p2, p1); // rows 8-15
    let p0q0_lo = _mm_unpacklo_epi8(p0, q0);
    let p0q0_hi = _mm_unpackhi_epi8(p0, q0);
    let q1q2_lo = _mm_unpacklo_epi8(q1, q2);
    let q1q2_hi = _mm_unpackhi_epi8(q1, q2);

    // Combine p2p1 and p0q0 to 32-bit
    let r0_lo = _mm_unpacklo_epi16(p2p1_lo, p0q0_lo); // rows 0-3
    let r1_lo = _mm_unpackhi_epi16(p2p1_lo, p0q0_lo); // rows 4-7
    let r0_hi = _mm_unpacklo_epi16(p2p1_hi, p0q0_hi); // rows 8-11
    let r1_hi = _mm_unpackhi_epi16(p2p1_hi, p0q0_hi); // rows 12-15

    // Extract 32-bit values for p2,p1,p0,q0
    let mut result4 = [0i32; 16];
    result4[0] = _mm_extract_epi32(r0_lo, 0);
    result4[1] = _mm_extract_epi32(r0_lo, 1);
    result4[2] = _mm_extract_epi32(r0_lo, 2);
    result4[3] = _mm_extract_epi32(r0_lo, 3);
    result4[4] = _mm_extract_epi32(r1_lo, 0);
    result4[5] = _mm_extract_epi32(r1_lo, 1);
    result4[6] = _mm_extract_epi32(r1_lo, 2);
    result4[7] = _mm_extract_epi32(r1_lo, 3);
    result4[8] = _mm_extract_epi32(r0_hi, 0);
    result4[9] = _mm_extract_epi32(r0_hi, 1);
    result4[10] = _mm_extract_epi32(r0_hi, 2);
    result4[11] = _mm_extract_epi32(r0_hi, 3);
    result4[12] = _mm_extract_epi32(r1_hi, 0);
    result4[13] = _mm_extract_epi32(r1_hi, 1);
    result4[14] = _mm_extract_epi32(r1_hi, 2);
    result4[15] = _mm_extract_epi32(r1_hi, 3);

    // Extract 16-bit values for q1,q2
    let mut result2 = [0i16; 16];
    result2[0] = _mm_extract_epi16(q1q2_lo, 0) as i16;
    result2[1] = _mm_extract_epi16(q1q2_lo, 1) as i16;
    result2[2] = _mm_extract_epi16(q1q2_lo, 2) as i16;
    result2[3] = _mm_extract_epi16(q1q2_lo, 3) as i16;
    result2[4] = _mm_extract_epi16(q1q2_lo, 4) as i16;
    result2[5] = _mm_extract_epi16(q1q2_lo, 5) as i16;
    result2[6] = _mm_extract_epi16(q1q2_lo, 6) as i16;
    result2[7] = _mm_extract_epi16(q1q2_lo, 7) as i16;
    result2[8] = _mm_extract_epi16(q1q2_hi, 0) as i16;
    result2[9] = _mm_extract_epi16(q1q2_hi, 1) as i16;
    result2[10] = _mm_extract_epi16(q1q2_hi, 2) as i16;
    result2[11] = _mm_extract_epi16(q1q2_hi, 3) as i16;
    result2[12] = _mm_extract_epi16(q1q2_hi, 4) as i16;
    result2[13] = _mm_extract_epi16(q1q2_hi, 5) as i16;
    result2[14] = _mm_extract_epi16(q1q2_hi, 6) as i16;
    result2[15] = _mm_extract_epi16(q1q2_hi, 7) as i16;

    (result4, result2)
}

/// Apply normal horizontal filter (DoFilter4) to 16 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses the transpose technique: load 16 rows × 8 columns, transpose,
/// apply DoFilter4 logic, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub fn normal_h_filter16_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 15) * stride + x + 4 <= pixels.len(),
        "normal_h_filter16_inner: bounds check failed"
    );

    // Load 16 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    // The edge is between p0 (x-1) and q0 (x), so we load from x-4 to x+3
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8: now we have 8 columns of 16 pixels each
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Transpose back: convert 4 columns of 16 to 16 rows of 4
    let packed = transpose_4x16_to_16x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row (p1, p0, q0, q1)
    for (i, &val) in packed.iter().enumerate() {
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
}

/// Apply normal horizontal filter (DoFilter6) to 16 rows at a vertical macroblock edge.
///
/// This filters the edge between column (x-1) and column (x).
/// Uses the transpose technique: load 16 rows × 8 columns, transpose,
/// apply DoFilter6 logic, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub fn normal_h_filter16_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 15) * stride + x + 4 <= pixels.len(),
        "normal_h_filter16_edge: bounds check failed"
    );

    // Load 16 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, row) in rows.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Transpose back: convert 6 columns of 16 to 16 rows of 6
    let (packed4, packed2) = transpose_6x16_to_16x6(_token, p2, p1, p0, q0, q1, q2);

    // Store 6 bytes per row (p2, p1, p0, q0, q1, q2)
    for (i, (&val4, &val2)) in packed4.iter().zip(packed2.iter()).enumerate() {
        let row_start = (y_start + i) * stride + x - 3;
        // Store first 4 bytes (p2, p1, p0, q0)
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val4),
        );
        // Store next 2 bytes (q1, q2)
        simd_mem::_mm_storeu_si16(
            <&mut [u8; 2]>::try_from(&mut pixels[row_start + 4..][..2]).unwrap(),
            _mm_cvtsi32_si128(val2 as i32),
        );
    }
}

// ============================================================================
// 8-row chroma filter functions
// These process both U and V planes together as 16 rows for efficiency.
// ============================================================================

/// Apply normal horizontal filter (DoFilter6) to U and V planes together.
/// Processes 8 U rows and 8 V rows as a single 16-row operation.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_h_filter_uv_edge(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 7) * stride + x + 4 <= u_pixels.len(),
        "normal_h_filter_uv_edge: u_pixels bounds check failed"
    );
    assert!(
        x >= 4 && (y_start + 7) * stride + x + 4 <= v_pixels.len(),
        "normal_h_filter_uv_edge: v_pixels bounds check failed"
    );

    // Load 8 U rows and 8 V rows into a 16-row array
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..8 {
        let row_start = (y_start + i) * stride + x - 4;
        rows[i] =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_pixels[row_start..][..8]).unwrap());
        rows[i + 8] =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter6
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Transpose 6 columns back to 16 rows of 6 bytes
    let (vals4, vals2) = transpose_6x16_to_16x6(_token, p2, p1, p0, q0, q1, q2);

    // Store 8 U rows and 8 V rows
    for i in 0..8 {
        let row_start = (y_start + i) * stride + x - 3;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut u_pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(vals4[i]),
        );
        simd_mem::_mm_storeu_si16(
            <&mut [u8; 2]>::try_from(&mut u_pixels[row_start + 4..][..2]).unwrap(),
            _mm_cvtsi32_si128(vals2[i] as i32),
        );
    }
    for i in 0..8 {
        let row_start = (y_start + i) * stride + x - 3;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut v_pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(vals4[i + 8]),
        );
        simd_mem::_mm_storeu_si16(
            <&mut [u8; 2]>::try_from(&mut v_pixels[row_start + 4..][..2]).unwrap(),
            _mm_cvtsi32_si128(vals2[i + 8] as i32),
        );
    }
}

/// Apply normal horizontal filter (DoFilter4) to U and V planes together at subblock edges.
/// Processes 8 U rows and 8 V rows as a single 16-row operation.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_h_filter_uv_inner(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 7) * stride + x + 4 <= u_pixels.len(),
        "normal_h_filter_uv_inner: u_pixels bounds check failed"
    );
    assert!(
        x >= 4 && (y_start + 7) * stride + x + 4 <= v_pixels.len(),
        "normal_h_filter_uv_inner: v_pixels bounds check failed"
    );

    // Load 8 U rows and 8 V rows
    let mut rows = [_mm_setzero_si128(); 16];
    for i in 0..8 {
        let row_start = (y_start + i) * stride + x - 4;
        rows[i] =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_pixels[row_start..][..8]).unwrap());
        rows[i + 8] =
            simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x16 to 16x8
    let cols = transpose_8x16_to_16x8(_token, &rows);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter4
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Transpose 4 columns back to 16 rows of 4 bytes
    let packed = transpose_4x16_to_16x4(_token, p1, p0, q0, q1);

    // Store 8 U rows and 8 V rows
    for i in 0..8 {
        let val = packed[i];
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut u_pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
    for i in 0..8 {
        let val = packed[i + 8];
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut v_pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
}

// ============================================================================
// AVX2 32-row horizontal filter functions (filtering vertical edges)
// These process 32 rows at once by transposing two 16-row groups, filtering
// with AVX2, then transposing back. Useful for filtering across two vertically
// adjacent macroblocks.
// ============================================================================

/// Transpose 32 rows of 8 bytes each into 8 columns of 32 bytes each.
/// Uses the existing 16-row transpose twice and combines into __m256i.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_8x32_to_32x8(
    _token: X64V3Token,
    rows_lo: &[__m128i; 16],
    rows_hi: &[__m128i; 16],
) -> [__m256i; 8] {
    // Transpose each half using existing SSE transpose
    let cols_lo = transpose_8x16_to_16x8(_token, rows_lo);
    let cols_hi = transpose_8x16_to_16x8(_token, rows_hi);

    // Combine low and high halves into AVX2 registers
    [
        _mm256_set_m128i(cols_hi[0], cols_lo[0]),
        _mm256_set_m128i(cols_hi[1], cols_lo[1]),
        _mm256_set_m128i(cols_hi[2], cols_lo[2]),
        _mm256_set_m128i(cols_hi[3], cols_lo[3]),
        _mm256_set_m128i(cols_hi[4], cols_lo[4]),
        _mm256_set_m128i(cols_hi[5], cols_lo[5]),
        _mm256_set_m128i(cols_hi[6], cols_lo[6]),
        _mm256_set_m128i(cols_hi[7], cols_lo[7]),
    ]
}

/// Transpose 4 columns of 32 bytes each back to 32 rows of 4 bytes.
/// Returns values suitable for storing as 32-bit integers per row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_4x32_to_32x4(
    _token: X64V3Token,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
) -> [i32; 32] {
    // Split into low and high halves
    let p1_lo = _mm256_castsi256_si128(p1);
    let p1_hi = _mm256_extracti128_si256(p1, 1);
    let p0_lo = _mm256_castsi256_si128(p0);
    let p0_hi = _mm256_extracti128_si256(p0, 1);
    let q0_lo = _mm256_castsi256_si128(q0);
    let q0_hi = _mm256_extracti128_si256(q0, 1);
    let q1_lo = _mm256_castsi256_si128(q1);
    let q1_hi = _mm256_extracti128_si256(q1, 1);

    // Use existing 16-row transpose on each half
    let lo = transpose_4x16_to_16x4(_token, p1_lo, p0_lo, q0_lo, q1_lo);
    let hi = transpose_4x16_to_16x4(_token, p1_hi, p0_hi, q0_hi, q1_hi);

    // Combine results
    let mut result = [0i32; 32];
    result[..16].copy_from_slice(&lo);
    result[16..].copy_from_slice(&hi);
    result
}

/// Transpose 6 columns of 32 bytes each back to 32 rows of 6 bytes.
/// Returns (4-byte values, 2-byte values) suitable for storing per row.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
#[inline(always)]
fn transpose_6x32_to_32x6(
    _token: X64V3Token,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
) -> ([i32; 32], [i16; 32]) {
    // Split into low and high halves
    let p2_lo = _mm256_castsi256_si128(p2);
    let p2_hi = _mm256_extracti128_si256(p2, 1);
    let p1_lo = _mm256_castsi256_si128(p1);
    let p1_hi = _mm256_extracti128_si256(p1, 1);
    let p0_lo = _mm256_castsi256_si128(p0);
    let p0_hi = _mm256_extracti128_si256(p0, 1);
    let q0_lo = _mm256_castsi256_si128(q0);
    let q0_hi = _mm256_extracti128_si256(q0, 1);
    let q1_lo = _mm256_castsi256_si128(q1);
    let q1_hi = _mm256_extracti128_si256(q1, 1);
    let q2_lo = _mm256_castsi256_si128(q2);
    let q2_hi = _mm256_extracti128_si256(q2, 1);

    // Use existing 16-row transpose on each half
    let (lo4, lo2) = transpose_6x16_to_16x6(_token, p2_lo, p1_lo, p0_lo, q0_lo, q1_lo, q2_lo);
    let (hi4, hi2) = transpose_6x16_to_16x6(_token, p2_hi, p1_hi, p0_hi, q0_hi, q1_hi, q2_hi);

    // Combine results
    let mut result4 = [0i32; 32];
    let mut result2 = [0i16; 32];
    result4[..16].copy_from_slice(&lo4);
    result4[16..].copy_from_slice(&hi4);
    result2[..16].copy_from_slice(&lo2);
    result2[16..].copy_from_slice(&hi2);
    (result4, result2)
}

/// Apply normal horizontal filter (DoFilter4) to 32 rows at a vertical edge.
///
/// This filters the edge between column (x-1) and column (x), processing
/// 32 consecutive rows (e.g., two vertically adjacent macroblocks).
/// Uses the transpose technique: load 32 rows × 8 columns, transpose,
/// apply DoFilter4 logic with AVX2, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub fn normal_h_filter32_inner(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 31) * stride + x + 4 <= pixels.len(),
        "normal_h_filter32_inner: bounds check failed"
    );

    // Load 32 rows of 8 pixels each (p3,p2,p1,p0,q0,q1,q2,q3)
    // Split into two groups of 16 for the transpose
    let mut rows_lo = [_mm_setzero_si128(); 16];
    let mut rows_hi = [_mm_setzero_si128(); 16];

    for (i, row) in rows_lo.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }
    for (i, row) in rows_hi.iter_mut().enumerate() {
        let row_start = (y_start + 16 + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x32 to 32x8
    let cols = transpose_8x32_to_32x8(_token, &rows_lo, &rows_hi);
    let p3 = cols[0];
    let p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter4_32(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Transpose back: convert 4 columns of 32 to 32 rows of 4
    let packed = transpose_4x32_to_32x4(_token, p1, p0, q0, q1);

    // Store 4 bytes per row (p1, p0, q0, q1)
    for (i, &val) in packed.iter().enumerate() {
        let row_start = (y_start + i) * stride + x - 2;
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val),
        );
    }
}

/// Apply normal horizontal filter (DoFilter6) to 32 rows at a vertical macroblock edge.
///
/// This filters the edge between column (x-1) and column (x), processing
/// 32 consecutive rows (e.g., two vertically adjacent macroblocks).
/// Uses the transpose technique: load 32 rows × 8 columns, transpose,
/// apply DoFilter6 logic with AVX2, transpose back, store.
///
/// Each row must have at least 4 bytes before and after the edge point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[arcane]
pub fn normal_h_filter32_edge(
    _token: X64V3Token,
    pixels: &mut [u8],
    x: usize,
    y_start: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(
        x >= 4 && (y_start + 31) * stride + x + 4 <= pixels.len(),
        "normal_h_filter32_edge: bounds check failed"
    );

    // Load 32 rows of 8 pixels each
    let mut rows_lo = [_mm_setzero_si128(); 16];
    let mut rows_hi = [_mm_setzero_si128(); 16];

    for (i, row) in rows_lo.iter_mut().enumerate() {
        let row_start = (y_start + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }
    for (i, row) in rows_hi.iter_mut().enumerate() {
        let row_start = (y_start + 16 + i) * stride + x - 4;
        *row = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&pixels[row_start..][..8]).unwrap());
    }

    // Transpose 8x32 to 32x8
    let cols = transpose_8x32_to_32x8(_token, &rows_lo, &rows_hi);
    let p3 = cols[0];
    let mut p2 = cols[1];
    let mut p1 = cols[2];
    let mut p0 = cols[3];
    let mut q0 = cols[4];
    let mut q1 = cols[5];
    let mut q2 = cols[6];
    let q3 = cols[7];

    // Check if filtering is needed
    let mask = needs_filter_normal_32(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_32(_token, p1, p0, q0, q1, hev_thresh);

    // Apply filter
    do_filter6_32(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Transpose back: convert 6 columns of 32 to 32 rows of 6
    let (packed4, packed2) = transpose_6x32_to_32x6(_token, p2, p1, p0, q0, q1, q2);

    // Store 6 bytes per row (p2, p1, p0, q0, q1, q2)
    for (i, (&val4, &val2)) in packed4.iter().zip(packed2.iter()).enumerate() {
        let row_start = (y_start + i) * stride + x - 3;
        // Store first 4 bytes (p2, p1, p0, q0)
        simd_mem::_mm_storeu_si32(
            <&mut [u8; 4]>::try_from(&mut pixels[row_start..][..4]).unwrap(),
            _mm_cvtsi32_si128(val4),
        );
        // Store next 2 bytes (q1, q2)
        simd_mem::_mm_storeu_si16(
            <&mut [u8; 2]>::try_from(&mut pixels[row_start + 4..][..2]).unwrap(),
            _mm_cvtsi32_si128(val2 as i32),
        );
    }
}

// ============================================================================
// 8-pixel chroma vertical filter functions (filtering horizontal edges)
// These process both U and V planes together by packing 8 U + 8 V pixels
// into each __m128i register.
// ============================================================================

/// Apply normal vertical filter (DoFilter6) to U and V planes together at macroblock edges.
/// Processes 8 U pixels and 8 V pixels per row, packed into __m128i registers.
///
/// Must have at least 4 rows before and after the point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter_uv_edge(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    // Loads from point - 4*stride to point + 3*stride (8 bytes each)
    // Stores from point - 3*stride to point + 2*stride (need 16-byte slice for storel_epi64)
    assert!(
        point >= 4 * stride && point + 3 * stride + 8 <= u_pixels.len(),
        "normal_v_filter_uv_edge: u_pixels bounds check failed"
    );
    assert!(
        point >= 4 * stride && point + 3 * stride + 8 <= v_pixels.len(),
        "normal_v_filter_uv_edge: v_pixels bounds check failed"
    );

    // Helper to load a row with 8 U + 8 V pixels packed together
    let load_uv_row = |u_pix: &[u8], v_pix: &[u8], offset: usize| -> __m128i {
        let u = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_pix[offset..][..8]).unwrap());
        let v = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_pix[offset..][..8]).unwrap());
        _mm_unpacklo_epi64(u, v)
    };

    let p3 = load_uv_row(u_pixels, v_pixels, point - 4 * stride);
    let mut p2 = load_uv_row(u_pixels, v_pixels, point - 3 * stride);
    let mut p1 = load_uv_row(u_pixels, v_pixels, point - 2 * stride);
    let mut p0 = load_uv_row(u_pixels, v_pixels, point - stride);
    let mut q0 = load_uv_row(u_pixels, v_pixels, point);
    let mut q1 = load_uv_row(u_pixels, v_pixels, point + stride);
    let mut q2 = load_uv_row(u_pixels, v_pixels, point + 2 * stride);
    let q3 = load_uv_row(u_pixels, v_pixels, point + 3 * stride);

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter6
    do_filter6_16(
        _token, &mut p2, &mut p1, &mut p0, &mut q0, &mut q1, &mut q2, mask, hev,
    );

    // Store results back - low 64 bits to U, high 64 bits to V
    let store_uv_row = |u_pix: &mut [u8], v_pix: &mut [u8], offset: usize, reg: __m128i| {
        simd_mem::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut u_pix[offset..][..16]).unwrap(),
            reg,
        );
        simd_mem::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut v_pix[offset..][..16]).unwrap(),
            _mm_srli_si128(reg, 8),
        );
    };

    store_uv_row(u_pixels, v_pixels, point - 3 * stride, p2);
    store_uv_row(u_pixels, v_pixels, point - 2 * stride, p1);
    store_uv_row(u_pixels, v_pixels, point - stride, p0);
    store_uv_row(u_pixels, v_pixels, point, q0);
    store_uv_row(u_pixels, v_pixels, point + stride, q1);
    store_uv_row(u_pixels, v_pixels, point + 2 * stride, q2);
}

/// Apply normal vertical filter (DoFilter4) to U and V planes together at subblock edges.
/// Processes 8 U pixels and 8 V pixels per row, packed into __m128i registers.
///
/// Must have at least 4 rows before and after the point.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane]
pub fn normal_v_filter_uv_inner(
    _token: X64V3Token,
    u_pixels: &mut [u8],
    v_pixels: &mut [u8],
    point: usize,
    stride: usize,
    hev_thresh: i32,
    interior_limit: i32,
    edge_limit: i32,
) {
    // Assert bounds upfront to elide checks in SIMD loads/stores
    // Loads from point - 4*stride to point + 3*stride (8 bytes each)
    // Stores from point - 2*stride to point + stride (need 16-byte slice for storel_epi64)
    assert!(
        point >= 4 * stride && point + 3 * stride + 8 <= u_pixels.len(),
        "normal_v_filter_uv_inner: u_pixels bounds check failed"
    );
    assert!(
        point >= 4 * stride && point + 3 * stride + 8 <= v_pixels.len(),
        "normal_v_filter_uv_inner: v_pixels bounds check failed"
    );

    // Helper to load a row with 8 U + 8 V pixels packed together
    let load_uv_row = |u_pix: &[u8], v_pix: &[u8], offset: usize| -> __m128i {
        let u = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&u_pix[offset..][..8]).unwrap());
        let v = simd_mem::_mm_loadu_si64(<&[u8; 8]>::try_from(&v_pix[offset..][..8]).unwrap());
        _mm_unpacklo_epi64(u, v)
    };

    let p3 = load_uv_row(u_pixels, v_pixels, point - 4 * stride);
    let p2 = load_uv_row(u_pixels, v_pixels, point - 3 * stride);
    let mut p1 = load_uv_row(u_pixels, v_pixels, point - 2 * stride);
    let mut p0 = load_uv_row(u_pixels, v_pixels, point - stride);
    let mut q0 = load_uv_row(u_pixels, v_pixels, point);
    let mut q1 = load_uv_row(u_pixels, v_pixels, point + stride);
    let q2 = load_uv_row(u_pixels, v_pixels, point + 2 * stride);
    let q3 = load_uv_row(u_pixels, v_pixels, point + 3 * stride);

    // Check if filtering is needed
    let mask = needs_filter_normal_16(
        _token,
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        edge_limit,
        interior_limit,
    );

    // Check high edge variance
    let hev = high_edge_variance_16(_token, p1, p0, q0, q1, hev_thresh);

    // Apply DoFilter4
    do_filter4_16(_token, &mut p1, &mut p0, &mut q0, &mut q1, mask, hev);

    // Store results back - low 64 bits to U, high 64 bits to V
    let store_uv_row = |u_pix: &mut [u8], v_pix: &mut [u8], offset: usize, reg: __m128i| {
        simd_mem::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut u_pix[offset..][..16]).unwrap(),
            reg,
        );
        simd_mem::_mm_storel_epi64(
            <&mut [u8; 16]>::try_from(&mut v_pix[offset..][..16]).unwrap(),
            _mm_srli_si128(reg, 8),
        );
    };

    store_uv_row(u_pixels, v_pixels, point - 2 * stride, p1);
    store_uv_row(u_pixels, v_pixels, point - stride, p0);
    store_uv_row(u_pixels, v_pixels, point, q0);
    store_uv_row(u_pixels, v_pixels, point + stride, q1);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    use archmage::SimdToken;

    /// Reference scalar simple filter for comparison
    fn scalar_simple_filter(p1: u8, p0: u8, q0: u8, q1: u8, thresh: i32) -> (u8, u8) {
        // Check threshold
        let diff_p0_q0 = (p0 as i32 - q0 as i32).abs();
        let diff_p1_q1 = (p1 as i32 - q1 as i32).abs();
        if diff_p0_q0 * 2 + diff_p1_q1 / 2 > thresh {
            return (p0, q0); // No filtering
        }

        // Convert to signed
        let p1s = p1 as i32 - 128;
        let p0s = p0 as i32 - 128;
        let q0s = q0 as i32 - 128;
        let q1s = q1 as i32 - 128;

        // Compute filter value
        let a = (p1s - q1s + 3 * (q0s - p0s)).clamp(-128, 127);
        let a_plus_4 = (a + 4) >> 3;
        let a_plus_3 = (a + 3) >> 3;

        // Apply
        let new_q0 = (q0s - a_plus_4).clamp(-128, 127) + 128;
        let new_p0 = (p0s + a_plus_3).clamp(-128, 127) + 128;

        (new_p0 as u8, new_q0 as u8)
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_simple_v_filter16_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 8 + V_FILTER_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up a gradient that should be filtered
        // Row 0 = p1, Row 1 = p0, Row 2 = q0, Row 3 = q1
        for x in 0..16 {
            pixels[x] = 100; // row 0: p1
            pixels[stride + x] = 110; // row 1: p0
            pixels[2 * stride + x] = 140; // row 2: q0
            pixels[3 * stride + x] = 150; // row 3: q1

            pixels_scalar[x] = 100;
            pixels_scalar[stride + x] = 110;
            pixels_scalar[2 * stride + x] = 140;
            pixels_scalar[3 * stride + x] = 150;
        }

        let thresh = 40;

        // Apply SIMD filter (edge between row 1 and row 2, so point = 2 * stride)
        simple_v_filter16(token, &mut pixels, 2 * stride, stride, thresh);

        // Apply scalar filter
        for x in 0..16 {
            let p1 = pixels_scalar[x];
            let p0 = pixels_scalar[stride + x];
            let q0 = pixels_scalar[2 * stride + x];
            let q1 = pixels_scalar[3 * stride + x];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[stride + x] = new_p0;
            pixels_scalar[2 * stride + x] = new_q0;
        }

        // Compare
        for x in 0..16 {
            assert_eq!(
                pixels[stride + x],
                pixels_scalar[stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "q0 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_simple_h_filter16_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        let mut pixels = vec![128u8; stride * 20];
        let mut pixels_scalar = pixels.clone();

        // Set up a vertical edge with gradient
        for y in 0..16 {
            pixels[y * stride + 2] = 100; // p1
            pixels[y * stride + 3] = 110; // p0
            pixels[y * stride + 4] = 140; // q0
            pixels[y * stride + 5] = 150; // q1

            pixels_scalar[y * stride + 2] = 100;
            pixels_scalar[y * stride + 3] = 110;
            pixels_scalar[y * stride + 4] = 140;
            pixels_scalar[y * stride + 5] = 150;
        }

        let thresh = 40;

        // Apply SIMD filter (edge at x=4)
        simple_h_filter16(token, &mut pixels, 4, 0, stride, thresh);

        // Apply scalar filter
        for y in 0..16 {
            let p1 = pixels_scalar[y * stride + 2];
            let p0 = pixels_scalar[y * stride + 3];
            let q0 = pixels_scalar[y * stride + 4];
            let q1 = pixels_scalar[y * stride + 5];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[y * stride + 3] = new_p0;
            pixels_scalar[y * stride + 4] = new_q0;
        }

        // Compare
        for y in 0..16 {
            assert_eq!(
                pixels[y * stride + 3],
                pixels_scalar[y * stride + 3],
                "p0 mismatch at y={}",
                y
            );
            assert_eq!(
                pixels[y * stride + 4],
                pixels_scalar[y * stride + 4],
                "q0 mismatch at y={}",
                y
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_normal_v_filter16_inner_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_NORMAL_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient data for all 8 rows around the edge
        for x in 0..16 {
            pixels[0 * stride + x] = 100; // p3
            pixels[1 * stride + x] = 105; // p2
            pixels[2 * stride + x] = 110; // p1
            pixels[3 * stride + x] = 115; // p0
            pixels[4 * stride + x] = 145; // q0 (edge here)
            pixels[5 * stride + x] = 150; // q1
            pixels[6 * stride + x] = 155; // q2
            pixels[7 * stride + x] = 160; // q3

            pixels_scalar[0 * stride + x] = 100;
            pixels_scalar[1 * stride + x] = 105;
            pixels_scalar[2 * stride + x] = 110;
            pixels_scalar[3 * stride + x] = 115;
            pixels_scalar[4 * stride + x] = 145;
            pixels_scalar[5 * stride + x] = 150;
            pixels_scalar[6 * stride + x] = 155;
            pixels_scalar[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply SIMD filter
        normal_v_filter16_inner(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply scalar filter
        for x in 0..16 {
            crate::decoder::loop_filter::subblock_filter_vertical(
                hev_thresh as u8,
                interior_limit as u8,
                edge_limit as u8,
                &mut pixels_scalar,
                4 * stride + x,
                stride,
            );
        }

        // Compare - we compare p1, p0, q0, q1 which can be modified
        for x in 0..16 {
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_scalar[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_scalar[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_scalar[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_normal_v_filter16_edge_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 32;
        // Allocate with V_FILTER_NORMAL_REGION padding for fixed-region bounds check elimination
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient data for all 8 rows around the edge
        for x in 0..16 {
            pixels[0 * stride + x] = 100; // p3
            pixels[1 * stride + x] = 105; // p2
            pixels[2 * stride + x] = 110; // p1
            pixels[3 * stride + x] = 115; // p0
            pixels[4 * stride + x] = 145; // q0 (edge here)
            pixels[5 * stride + x] = 150; // q1
            pixels[6 * stride + x] = 155; // q2
            pixels[7 * stride + x] = 160; // q3

            pixels_scalar[0 * stride + x] = 100;
            pixels_scalar[1 * stride + x] = 105;
            pixels_scalar[2 * stride + x] = 110;
            pixels_scalar[3 * stride + x] = 115;
            pixels_scalar[4 * stride + x] = 145;
            pixels_scalar[5 * stride + x] = 150;
            pixels_scalar[6 * stride + x] = 155;
            pixels_scalar[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 40;

        // Apply SIMD filter
        normal_v_filter16_edge(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply scalar filter
        for x in 0..16 {
            crate::decoder::loop_filter::macroblock_filter_vertical(
                hev_thresh as u8,
                interior_limit as u8,
                edge_limit as u8,
                &mut pixels_scalar,
                4 * stride + x,
                stride,
            );
        }

        // Compare - p2, p1, p0, q0, q1, q2 can be modified
        for x in 0..16 {
            assert_eq!(
                pixels[1 * stride + x],
                pixels_scalar[1 * stride + x],
                "p2 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_scalar[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_scalar[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_scalar[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[6 * stride + x],
                pixels_scalar[6 * stride + x],
                "q2 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_simple_v_filter32_matches_scalar() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_REGION_32 padding
        let mut pixels = vec![128u8; stride * 8 + V_FILTER_REGION_32];
        let mut pixels_scalar = pixels.clone();

        // Set up gradient for 32 pixels
        for x in 0..32 {
            pixels[x] = 100;
            pixels[stride + x] = 110;
            pixels[2 * stride + x] = 140;
            pixels[3 * stride + x] = 150;

            pixels_scalar[x] = 100;
            pixels_scalar[stride + x] = 110;
            pixels_scalar[2 * stride + x] = 140;
            pixels_scalar[3 * stride + x] = 150;
        }

        let thresh = 40;

        // Apply AVX2 32-pixel filter
        simple_v_filter32(token, &mut pixels, 2 * stride, stride, thresh);

        // Apply scalar filter
        for x in 0..32 {
            let p1 = pixels_scalar[x];
            let p0 = pixels_scalar[stride + x];
            let q0 = pixels_scalar[2 * stride + x];
            let q1 = pixels_scalar[3 * stride + x];
            let (new_p0, new_q0) = scalar_simple_filter(p1, p0, q0, q1, thresh);
            pixels_scalar[stride + x] = new_p0;
            pixels_scalar[2 * stride + x] = new_q0;
        }

        // Compare
        for x in 0..32 {
            assert_eq!(
                pixels[stride + x],
                pixels_scalar[stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_scalar[2 * stride + x],
                "q0 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_normal_v_filter32_inner_matches_two_16() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_NORMAL_REGION_32 padding
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION_32];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 8 rows around the edge (32 pixels wide)
        for x in 0..32 {
            pixels[0 * stride + x] = 100;
            pixels[1 * stride + x] = 105;
            pixels[2 * stride + x] = 110;
            pixels[3 * stride + x] = 115;
            pixels[4 * stride + x] = 145;
            pixels[5 * stride + x] = 150;
            pixels[6 * stride + x] = 155;
            pixels[7 * stride + x] = 160;

            pixels_16[0 * stride + x] = 100;
            pixels_16[1 * stride + x] = 105;
            pixels_16[2 * stride + x] = 110;
            pixels_16[3 * stride + x] = 115;
            pixels_16[4 * stride + x] = 145;
            pixels_16[5 * stride + x] = 150;
            pixels_16[6 * stride + x] = 155;
            pixels_16[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-pixel filter
        normal_v_filter32_inner(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-pixel filter
        normal_v_filter16_inner(
            token,
            &mut pixels_16,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        normal_v_filter16_inner(
            token,
            &mut pixels_16,
            4 * stride + 16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 pixels for modified rows (p1, p0, q0, q1)
        for x in 0..32 {
            assert_eq!(
                pixels[2 * stride + x],
                pixels_16[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_16[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_16[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_16[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_normal_v_filter32_edge_matches_two_16() {
        let Some(token) = archmage::X64V3Token::summon() else {
            return;
        };

        let stride = 64;
        // Allocate with V_FILTER_NORMAL_REGION_32 padding
        let mut pixels = vec![128u8; stride * 12 + V_FILTER_NORMAL_REGION_32];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 8 rows around the edge (32 pixels wide)
        for x in 0..32 {
            pixels[0 * stride + x] = 100;
            pixels[1 * stride + x] = 105;
            pixels[2 * stride + x] = 110;
            pixels[3 * stride + x] = 115;
            pixels[4 * stride + x] = 145;
            pixels[5 * stride + x] = 150;
            pixels[6 * stride + x] = 155;
            pixels[7 * stride + x] = 160;

            pixels_16[0 * stride + x] = 100;
            pixels_16[1 * stride + x] = 105;
            pixels_16[2 * stride + x] = 110;
            pixels_16[3 * stride + x] = 115;
            pixels_16[4 * stride + x] = 145;
            pixels_16[5 * stride + x] = 150;
            pixels_16[6 * stride + x] = 155;
            pixels_16[7 * stride + x] = 160;
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-pixel filter
        normal_v_filter32_edge(
            token,
            &mut pixels,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-pixel filter
        normal_v_filter16_edge(
            token,
            &mut pixels_16,
            4 * stride,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        normal_v_filter16_edge(
            token,
            &mut pixels_16,
            4 * stride + 16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 pixels for modified rows (p2, p1, p0, q0, q1, q2)
        for x in 0..32 {
            assert_eq!(
                pixels[1 * stride + x],
                pixels_16[1 * stride + x],
                "p2 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[2 * stride + x],
                pixels_16[2 * stride + x],
                "p1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[3 * stride + x],
                pixels_16[3 * stride + x],
                "p0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[4 * stride + x],
                pixels_16[4 * stride + x],
                "q0 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[5 * stride + x],
                pixels_16[5 * stride + x],
                "q1 mismatch at x={}",
                x
            );
            assert_eq!(
                pixels[6 * stride + x],
                pixels_16[6 * stride + x],
                "q2 mismatch at x={}",
                x
            );
        }
    }

    #[test]
    fn test_normal_h_filter32_inner_matches_two_16() {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("AVX2 not available, skipping test");
            return;
        };

        let width = 64;
        let height = 48; // Need 32 rows + padding
        let stride = width;
        let mut pixels = vec![128u8; stride * height];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 32 rows around the vertical edge at x=16
        for y in 0..32 {
            for x in 0..8 {
                let base = y * stride + 12 + x; // columns 12-19 around edge at x=16
                pixels[base] = (100 + x * 10) as u8;
                pixels_16[base] = (100 + x * 10) as u8;
            }
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-row filter (processing 32 rows at once)
        normal_h_filter32_inner(
            token,
            &mut pixels,
            16, // x
            0,  // y_start
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-row filter (rows 0-15 and rows 16-31)
        normal_h_filter16_inner(
            token,
            &mut pixels_16,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        normal_h_filter16_inner(
            token,
            &mut pixels_16,
            16,
            16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 rows for modified columns (p1, p0, q0, q1 = x-2 to x+1)
        for y in 0..32 {
            for x in 14..18 {
                assert_eq!(
                    pixels[y * stride + x],
                    pixels_16[y * stride + x],
                    "mismatch at y={}, x={}",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    fn test_normal_h_filter32_edge_matches_two_16() {
        let Some(token) = X64V3Token::summon() else {
            eprintln!("AVX2 not available, skipping test");
            return;
        };

        let width = 64;
        let height = 48;
        let stride = width;
        let mut pixels = vec![128u8; stride * height];
        let mut pixels_16 = pixels.clone();

        // Set up gradient data for all 32 rows around the vertical edge at x=16
        for y in 0..32 {
            for x in 0..8 {
                let base = y * stride + 12 + x;
                pixels[base] = (100 + x * 10) as u8;
                pixels_16[base] = (100 + x * 10) as u8;
            }
        }

        let hev_thresh = 5;
        let interior_limit = 15;
        let edge_limit = 25;

        // Apply AVX2 32-row filter
        normal_h_filter32_edge(
            token,
            &mut pixels,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Apply 2x 16-row filter
        normal_h_filter16_edge(
            token,
            &mut pixels_16,
            16,
            0,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );
        normal_h_filter16_edge(
            token,
            &mut pixels_16,
            16,
            16,
            stride,
            hev_thresh,
            interior_limit,
            edge_limit,
        );

        // Compare all 32 rows for modified columns (p2, p1, p0, q0, q1, q2 = x-3 to x+2)
        for y in 0..32 {
            for x in 13..19 {
                assert_eq!(
                    pixels[y * stride + x],
                    pixels_16[y * stride + x],
                    "mismatch at y={}, x={}",
                    y,
                    x
                );
            }
        }
    }
}
