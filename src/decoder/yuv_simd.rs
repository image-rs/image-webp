//! SIMD-optimized YUV to RGB conversion using x86 SSE2/SSE4.1 intrinsics.
//!
//! Ported from libwebp's yuv_sse2.c for efficient 16-bit arithmetic.
//! Processes 32 pixels at a time with SIMD RGB interleaving.

#[cfg(target_arch = "x86_64")]
use archmage::{arcane, SimdToken, X64V3Token};

#[cfg(target_arch = "x86_64")]
use archmage::intrinsics::x86_64 as simd_mem;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// YUV to RGB conversion constants (14-bit fixed-point, matching libwebp).
// R = (19077 * y             + 26149 * v - 14234) >> 6
// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
// B = (19077 * y + 33050 * u             - 17685) >> 6

/// Load 8 bytes into the upper 8 bits of 16-bit words (equivalent to << 8).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn load_hi_16(_token: X64V3Token, src: &[u8; 8]) -> __m128i {
    let zero = _mm_setzero_si128();
    // Load 8 bytes as i64 then convert to __m128i
    let val = i64::from_le_bytes(*src);
    let data = _mm_cvtsi64_si128(val);
    _mm_unpacklo_epi8(zero, data)
}

/// Load 4 U/V bytes and replicate each to get 8 values for 4:2:0.
/// Result: [u0,u0,u1,u1,u2,u2,u3,u3] in upper 8 bits of 16-bit words.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn load_uv_hi_8(_token: X64V3Token, src: &[u8; 4]) -> __m128i {
    let zero = _mm_setzero_si128();
    // Load 4 bytes as i32
    let val = i32::from_le_bytes(*src);
    let tmp0 = _mm_cvtsi32_si128(val);
    // Unpack to 16-bit with zeros in low bytes: [0,u0,0,u1,0,u2,0,u3,...]
    let tmp1 = _mm_unpacklo_epi8(zero, tmp0);
    // Replicate: [0,u0,0,u0,0,u1,0,u1,...]
    _mm_unpacklo_epi16(tmp1, tmp1)
}

/// Convert 8 YUV444 pixels to R, G, B (16-bit results).
/// Input Y, U, V are in upper 8 bits of 16-bit words.
/// Output R, G, B are signed 16-bit values (will be clamped later).
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn convert_yuv444_to_rgb(
    _token: X64V3Token,
    y: __m128i,
    u: __m128i,
    v: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let k19077 = _mm_set1_epi16(19077);
    let k26149 = _mm_set1_epi16(26149);
    let k14234 = _mm_set1_epi16(14234);
    // 33050 doesn't fit in signed i16, use unsigned arithmetic
    let k33050 = _mm_set1_epi16(33050u16 as i16);
    let k17685 = _mm_set1_epi16(17685);
    let k6419 = _mm_set1_epi16(6419);
    let k13320 = _mm_set1_epi16(13320);
    let k8708 = _mm_set1_epi16(8708);

    // Y contribution (same for all channels)
    let y1 = _mm_mulhi_epu16(y, k19077);

    // R = Y1 + V*26149 - 14234
    let r0 = _mm_mulhi_epu16(v, k26149);
    let r1 = _mm_sub_epi16(y1, k14234);
    let r2 = _mm_add_epi16(r1, r0);

    // G = Y1 - U*6419 - V*13320 + 8708
    let g0 = _mm_mulhi_epu16(u, k6419);
    let g1 = _mm_mulhi_epu16(v, k13320);
    let g2 = _mm_add_epi16(y1, k8708);
    let g3 = _mm_add_epi16(g0, g1);
    let g4 = _mm_sub_epi16(g2, g3);

    // B = Y1 + U*33050 - 17685 (careful with unsigned arithmetic)
    let b0 = _mm_mulhi_epu16(u, k33050);
    let b1 = _mm_adds_epu16(b0, y1);
    let b2 = _mm_subs_epu16(b1, k17685);

    // Final shift by 6
    // R and G can be negative, use arithmetic shift
    // B is always positive (due to unsigned ops), use logical shift
    let r = _mm_srai_epi16(r2, 6);
    let g = _mm_srai_epi16(g4, 6);
    let b = _mm_srli_epi16(b2, 6);

    (r, g, b)
}

/// Pack R, G, B, A (8 pixels each, 16-bit) into 32 bytes of RGBA output.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
#[allow(dead_code)]
fn pack_and_store_rgba(
    _token: X64V3Token,
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
    dst: &mut [u8; 32],
) {
    let rb = _mm_packus_epi16(r, b);
    let ga = _mm_packus_epi16(g, a);
    let rg = _mm_unpacklo_epi8(rb, ga);
    let ba = _mm_unpackhi_epi8(rb, ga);
    let rgba_lo = _mm_unpacklo_epi16(rg, ba);
    let rgba_hi = _mm_unpackhi_epi16(rg, ba);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), rgba_lo);
    simd_mem::_mm_storeu_si128(
        <&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(),
        rgba_hi,
    );
}

/// Helper macro for VP8PlanarTo24b - splits even/odd bytes
macro_rules! planar_to_24b_helper {
    ($in0:expr, $in1:expr, $in2:expr, $in3:expr, $in4:expr, $in5:expr,
     $out0:expr, $out1:expr, $out2:expr, $out3:expr, $out4:expr, $out5:expr) => {
        let v_mask = _mm_set1_epi16(0x00ff);
        // Take even bytes (lower 8 bits of each 16-bit word)
        $out0 = _mm_packus_epi16(_mm_and_si128($in0, v_mask), _mm_and_si128($in1, v_mask));
        $out1 = _mm_packus_epi16(_mm_and_si128($in2, v_mask), _mm_and_si128($in3, v_mask));
        $out2 = _mm_packus_epi16(_mm_and_si128($in4, v_mask), _mm_and_si128($in5, v_mask));
        // Take odd bytes (upper 8 bits of each 16-bit word)
        $out3 = _mm_packus_epi16(_mm_srli_epi16($in0, 8), _mm_srli_epi16($in1, 8));
        $out4 = _mm_packus_epi16(_mm_srli_epi16($in2, 8), _mm_srli_epi16($in3, 8));
        $out5 = _mm_packus_epi16(_mm_srli_epi16($in4, 8), _mm_srli_epi16($in5, 8));
    };
}

/// Convert planar RRRR...GGGG...BBBB... to interleaved RGBRGBRGB...
/// Input: 6 registers (R0, R1, G0, G1, B0, B1) with 16 bytes each = 32 R, 32 G, 32 B
/// Output: 6 registers with 96 bytes of interleaved RGB
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn planar_to_24b(
    _token: X64V3Token,
    in0: __m128i,
    in1: __m128i,
    in2: __m128i,
    in3: __m128i,
    in4: __m128i,
    in5: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // 5 passes of the permutation to convert:
    // Input: r0r1r2r3... | r16r17... | g0g1g2g3... | g16g17... | b0b1b2b3... | b16b17...
    // Output: r0g0b0r1g1b1... (interleaved RGB)

    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5);
    let (mut o0, mut o1, mut o2, mut o3, mut o4, mut o5);

    // Pass 1
    planar_to_24b_helper!(in0, in1, in2, in3, in4, in5, t0, t1, t2, t3, t4, t5);
    // Pass 2
    planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
    // Pass 3
    planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);
    // Pass 4
    planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
    // Pass 5
    planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);

    (t0, t1, t2, t3, t4, t5)
}

/// Convert 32 YUV444 pixels to 96 bytes of RGB.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn yuv444_to_rgb_32(
    _token: X64V3Token,
    y: &[u8; 32],
    u: &[u8; 32],
    v: &[u8; 32],
    dst: &mut [u8; 96],
) {
    // Process 4 groups of 8 pixels
    let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[..8]).unwrap());
    let u0 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[..8]).unwrap());
    let v0 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[..8]).unwrap());
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y0, u0, v0);

    let y1 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[8..16]).unwrap());
    let u1 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[8..16]).unwrap());
    let v1 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[8..16]).unwrap());
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y1, u1, v1);

    let y2 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[16..24]).unwrap());
    let u2 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[16..24]).unwrap());
    let v2 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[16..24]).unwrap());
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y2, u2, v2);

    let y3 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[24..32]).unwrap());
    let u3 = load_hi_16(_token, <&[u8; 8]>::try_from(&u[24..32]).unwrap());
    let v3 = load_hi_16(_token, <&[u8; 8]>::try_from(&v[24..32]).unwrap());
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y3, u3, v3);

    // Pack to 8-bit and arrange as RRRRGGGGBBBB
    let rgb0 = _mm_packus_epi16(r0, r1); // R0-R15
    let rgb1 = _mm_packus_epi16(r2, r3); // R16-R31
    let rgb2 = _mm_packus_epi16(g0, g1); // G0-G15
    let rgb3 = _mm_packus_epi16(g2, g3); // G16-G31
    let rgb4 = _mm_packus_epi16(b0, b1); // B0-B15
    let rgb5 = _mm_packus_epi16(b2, b3); // B16-B31

    // Interleave to RGBRGBRGB...
    let (out0, out1, out2, out3, out4, out5) =
        planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store 96 bytes
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[32..48]).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[48..64]).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[64..80]).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[80..96]).unwrap(), out5);
}

/// Convert 32 YUV420 pixels (32 Y, 16 U, 16 V) to 96 bytes of RGB.
/// Each U/V value is replicated for 2 adjacent Y pixels.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn yuv420_to_rgb_32(
    _token: X64V3Token,
    y: &[u8; 32],
    u: &[u8; 16],
    v: &[u8; 16],
    dst: &mut [u8; 96],
) {
    // Process 4 groups of 8 pixels, with U/V replication
    let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[..8]).unwrap());
    let u0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[..4]).unwrap());
    let v0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[..4]).unwrap());
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y0, u0, v0);

    let y1 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[8..16]).unwrap());
    let u1 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[4..8]).unwrap());
    let v1 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[4..8]).unwrap());
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y1, u1, v1);

    let y2 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[16..24]).unwrap());
    let u2 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[8..12]).unwrap());
    let v2 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[8..12]).unwrap());
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y2, u2, v2);

    let y3 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[24..32]).unwrap());
    let u3 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[12..16]).unwrap());
    let v3 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[12..16]).unwrap());
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y3, u3, v3);

    // Pack to 8-bit and arrange as RRRRGGGGBBBB
    let rgb0 = _mm_packus_epi16(r0, r1);
    let rgb1 = _mm_packus_epi16(r2, r3);
    let rgb2 = _mm_packus_epi16(g0, g1);
    let rgb3 = _mm_packus_epi16(g2, g3);
    let rgb4 = _mm_packus_epi16(b0, b1);
    let rgb5 = _mm_packus_epi16(b2, b3);

    // Interleave to RGBRGBRGB...
    let (out0, out1, out2, out3, out4, out5) =
        planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store 96 bytes
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[..16]).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[16..32]).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[32..48]).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[48..64]).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[64..80]).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut dst[80..96]).unwrap(), out5);
}

/// Scalar fallback for YUV to RGB conversion (single pixel).
#[inline]
fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    fn mulhi(val: u8, coeff: u16) -> i32 {
        ((u32::from(val) * u32::from(coeff)) >> 8) as i32
    }
    fn clip(v: i32) -> u8 {
        (v >> 6).clamp(0, 255) as u8
    }
    let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
    let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
    let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
    (r, g, b)
}

/// Convert a row of YUV420 to RGB using SIMD.
/// Processes 32 pixels at a time, with scalar fallback for remainder.
///
/// Requires SSE2. y must have `len` elements, u/v must have `len/2` elements.
/// dst must have `len * 3` bytes.
#[cfg(target_arch = "x86_64")]
pub fn yuv420_to_rgb_row(y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    yuv420_to_rgb_row_inner(token, y, u, v, dst);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn yuv420_to_rgb_row_inner(_token: X64V3Token, y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    let len = y.len();
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(u.len() >= len.div_ceil(2));
    assert!(v.len() >= len.div_ceil(2));
    assert!(dst.len() >= len * 3);

    let mut n = 0usize;

    // Process 32 pixels at a time
    while n + 32 <= len {
        let y_arr = <&[u8; 32]>::try_from(&y[n..n + 32]).unwrap();
        let u_arr = <&[u8; 16]>::try_from(&u[n / 2..n / 2 + 16]).unwrap();
        let v_arr = <&[u8; 16]>::try_from(&v[n / 2..n / 2 + 16]).unwrap();
        let dst_arr = <&mut [u8; 96]>::try_from(&mut dst[n * 3..n * 3 + 96]).unwrap();
        yuv420_to_rgb_32(_token, y_arr, u_arr, v_arr, dst_arr);
        n += 32;
    }

    // Scalar fallback for remainder
    while n < len {
        let y_val = y[n];
        let u_val = u[n / 2];
        let v_val = v[n / 2];
        let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
        dst[n * 3] = r;
        dst[n * 3 + 1] = g;
        dst[n * 3 + 2] = b;
        n += 1;
    }
}

/// Convert a row of YUV420 to RGBA using SIMD.
/// Alpha is set to 255.
///
/// Requires SSE2. y must have `len` elements, u/v must have `len/2` elements.
/// dst must have `len * 4` bytes.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn yuv420_to_rgba_row(y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    yuv420_to_rgba_row_inner(token, y, u, v, dst);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn yuv420_to_rgba_row_inner(_token: X64V3Token, y: &[u8], u: &[u8], v: &[u8], dst: &mut [u8]) {
    let len = y.len();
    // Assert bounds upfront to elide checks in SIMD loads/stores
    assert!(u.len() >= len.div_ceil(2));
    assert!(v.len() >= len.div_ceil(2));
    assert!(dst.len() >= len * 4);

    let k_alpha = _mm_set1_epi16(255);
    let mut n = 0usize;

    // Process 8 pixels at a time for RGBA
    while n + 8 <= len {
        let y0 = load_hi_16(_token, <&[u8; 8]>::try_from(&y[n..n + 8]).unwrap());
        let u0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&u[n / 2..n / 2 + 4]).unwrap());
        let v0 = load_uv_hi_8(_token, <&[u8; 4]>::try_from(&v[n / 2..n / 2 + 4]).unwrap());
        let (r, g, b) = convert_yuv444_to_rgb(_token, y0, u0, v0);
        pack_and_store_rgba(
            _token,
            r,
            g,
            b,
            k_alpha,
            <&mut [u8; 32]>::try_from(&mut dst[n * 4..n * 4 + 32]).unwrap(),
        );

        n += 8;
    }

    // Scalar fallback for remainder
    while n < len {
        let y_val = y[n];
        let u_val = u[n / 2];
        let v_val = v[n / 2];
        let (r, g, b) = yuv_to_rgb_scalar(y_val, u_val, v_val);
        dst[n * 4] = r;
        dst[n * 4 + 1] = g;
        dst[n * 4 + 2] = b;
        dst[n * 4 + 3] = 255;
        n += 1;
    }
}

// =============================================================================
// Fancy upsampling (bilinear interpolation of chroma)
// =============================================================================

/// Compute fancy chroma interpolation for 16 pixels using SIMD.
/// Formula: (9*a + 3*b + 3*c + d + 8) / 16
///
/// Uses libwebp's efficient approach with _mm_avg_epu8.
#[cfg(target_arch = "x86_64")]
#[arcane]
#[inline]
fn fancy_upsample_16(
    _token: X64V3Token,
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i) {
    let one = _mm_set1_epi8(1);

    // s = (a + d + 1) / 2
    let s = _mm_avg_epu8(a, d);
    // t = (b + c + 1) / 2
    let t = _mm_avg_epu8(b, c);
    // st = s ^ t
    let st = _mm_xor_si128(s, t);

    // ad = a ^ d, bc = b ^ c
    let ad = _mm_xor_si128(a, d);
    let bc = _mm_xor_si128(b, c);

    // k = (a + b + c + d) / 4 with proper rounding
    // k = (s + t + 1) / 2 - ((a^d) | (b^c) | (s^t)) & 1
    let t1 = _mm_or_si128(ad, bc);
    let t2 = _mm_or_si128(t1, st);
    let t3 = _mm_and_si128(t2, one);
    let t4 = _mm_avg_epu8(s, t);
    let k = _mm_sub_epi8(t4, t3);

    // diag1 = (a + 3*b + 3*c + d) / 8 then (9*a + 3*b + 3*c + d) / 16 = (a + diag1 + 1) / 2
    // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
    let tmp1 = _mm_avg_epu8(k, t);
    let tmp2 = _mm_and_si128(bc, st);
    let tmp3 = _mm_xor_si128(k, t);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m1 = _mm_sub_epi8(tmp1, tmp5);

    // diag2 = (3*a + b + c + 3*d) / 8 then (3*a + 9*b + c + 3*d) / 16 = (b + diag2 + 1) / 2
    // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
    let tmp1 = _mm_avg_epu8(k, s);
    let tmp2 = _mm_and_si128(ad, st);
    let tmp3 = _mm_xor_si128(k, s);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m2 = _mm_sub_epi8(tmp1, tmp5);

    // Final results
    let diag1 = _mm_avg_epu8(a, m1); // (9*a + 3*b + 3*c + d + 8) / 16
    let diag2 = _mm_avg_epu8(b, m2); // (3*a + 9*b + c + 3*d + 8) / 16

    (diag1, diag2)
}

/// Upsample 32 chroma pixels from two rows.
/// Input: 17 pixels from each row (r1, r2)
/// Output: 32 upsampled pixels for top row, 32 for bottom row
#[cfg(target_arch = "x86_64")]
#[arcane]
#[allow(dead_code)]
fn upsample_32_pixels(_token: X64V3Token, r1: &[u8; 17], r2: &[u8; 17], out: &mut [u8; 128]) {
    let one = _mm_set1_epi8(1);

    let a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r1[..16]).unwrap());
    let b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r1[1..17]).unwrap());
    let c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r2[..16]).unwrap());
    let d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&r2[1..17]).unwrap());

    let s = _mm_avg_epu8(a, d);
    let t = _mm_avg_epu8(b, c);
    let st = _mm_xor_si128(s, t);
    let ad = _mm_xor_si128(a, d);
    let bc = _mm_xor_si128(b, c);

    let t1 = _mm_or_si128(ad, bc);
    let t2 = _mm_or_si128(t1, st);
    let t3 = _mm_and_si128(t2, one);
    let t4 = _mm_avg_epu8(s, t);
    let k = _mm_sub_epi8(t4, t3);

    // m1 for diag1
    let tmp1 = _mm_avg_epu8(k, t);
    let tmp2 = _mm_and_si128(bc, st);
    let tmp3 = _mm_xor_si128(k, t);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let diag1 = _mm_sub_epi8(tmp1, tmp5);

    // m2 for diag2
    let tmp1 = _mm_avg_epu8(k, s);
    let tmp2 = _mm_and_si128(ad, st);
    let tmp3 = _mm_xor_si128(k, s);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let diag2 = _mm_sub_epi8(tmp1, tmp5);

    // Pack alternating pixels for top row: (9a+3b+3c+d)/16, (3a+9b+c+3d)/16
    let t_a = _mm_avg_epu8(a, diag1);
    let t_b = _mm_avg_epu8(b, diag2);
    let t_1 = _mm_unpacklo_epi8(t_a, t_b);
    let t_2 = _mm_unpackhi_epi8(t_a, t_b);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[..16]).unwrap(), t_1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[16..32]).unwrap(), t_2);

    // Pack for bottom row: roles of diag1/diag2 swapped
    let b_a = _mm_avg_epu8(c, diag2);
    let b_b = _mm_avg_epu8(d, diag1);
    let b_1 = _mm_unpacklo_epi8(b_a, b_b);
    let b_2 = _mm_unpackhi_epi8(b_a, b_b);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[64..80]).unwrap(), b_1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut out[80..96]).unwrap(), b_2);
}

/// Process 8 pixel pairs with fancy upsampling and YUV->RGB conversion.
/// This is the main entry point for fancy upsampling used by yuv.rs.
///
/// Requires SSE2. All input slices must have the required minimum lengths.
#[cfg(target_arch = "x86_64")]
/// Process 8 pixel pairs with fancy upsampling and YUV->RGB conversion.
/// This version accepts a pre-summoned token to avoid repeated summon() calls in loops.
#[inline(always)]
pub fn fancy_upsample_8_pairs_with_token(
    token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    fancy_upsample_8_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

#[allow(dead_code)] // Used in tests
pub fn fancy_upsample_8_pairs(
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // SSE4.1 implies SSE2; summon() is now fast (no env var check)
    let token = X64V3Token::summon().expect("SSE4.1 required for SIMD YUV");
    fancy_upsample_8_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

/// Optimized inner function taking fixed-size arrays to eliminate bounds checks.
/// Takes: 16 Y bytes, 9 U/V bytes per row (for overlapping window), 48 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_8_pairs_inner_opt(
    _token: X64V3Token,
    y_row: &[u8; 16],
    u_row_1: &[u8; 9],
    u_row_2: &[u8; 9],
    v_row_1: &[u8; 9],
    v_row_2: &[u8; 9],
    rgb: &mut [u8; 48],
) {
    // Load 8 chroma values from fixed-size arrays - no bounds checks needed
    // Using a macro instead of a nested function because macros expand at call site,
    // so _mm_cvtsi64_si128 is called within the #[arcane] function's target_feature context.
    // A nested function wouldn't inherit target_feature and would fail to compile.
    macro_rules! load_8_from_9 {
        ($arr:expr, 0) => {{
            let bytes: [u8; 8] = [
                $arr[0], $arr[1], $arr[2], $arr[3], $arr[4], $arr[5], $arr[6], $arr[7],
            ];
            let val = i64::from_le_bytes(bytes);
            _mm_cvtsi64_si128(val)
        }};
        ($arr:expr, 1) => {{
            let bytes: [u8; 8] = [
                $arr[1], $arr[2], $arr[3], $arr[4], $arr[5], $arr[6], $arr[7], $arr[8],
            ];
            let val = i64::from_le_bytes(bytes);
            _mm_cvtsi64_si128(val)
        }};
    }

    let u_a = load_8_from_9!(u_row_1, 0);
    let u_b = load_8_from_9!(u_row_1, 1);
    let u_c = load_8_from_9!(u_row_2, 0);
    let u_d = load_8_from_9!(u_row_2, 1);

    let v_a = load_8_from_9!(v_row_1, 0);
    let v_b = load_8_from_9!(v_row_1, 1);
    let v_c = load_8_from_9!(v_row_2, 0);
    let v_d = load_8_from_9!(v_row_2, 1);

    // Compute fancy upsampled U/V
    let (u_diag1, u_diag2) = fancy_upsample_16(_token, u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(_token, v_a, v_b, v_c, v_d);

    // Interleave: [diag1[0], diag2[0], diag1[1], diag2[1], ...]
    let u_interleaved = _mm_unpacklo_epi8(u_diag1, u_diag2);
    let v_interleaved = _mm_unpacklo_epi8(v_diag1, v_diag2);

    // Load Y directly from fixed-size array - no bounds check
    let y_vec = simd_mem::_mm_loadu_si128(y_row);
    let zero = _mm_setzero_si128();

    // Process first 8 pixels
    let y_lo = _mm_unpacklo_epi8(zero, y_vec);
    let u_lo = _mm_unpacklo_epi8(zero, u_interleaved);
    let v_lo = _mm_unpacklo_epi8(zero, v_interleaved);
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y_lo, u_lo, v_lo);

    // Process second 8 pixels
    let y_hi = _mm_unpackhi_epi8(zero, y_vec);
    let u_hi = _mm_unpackhi_epi8(zero, u_interleaved);
    let v_hi = _mm_unpackhi_epi8(zero, v_interleaved);
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y_hi, u_hi, v_hi);

    // Pack to 8-bit
    let r8 = _mm_packus_epi16(r0, r1);
    let g8 = _mm_packus_epi16(g0, g1);
    let b8 = _mm_packus_epi16(b0, b1);

    // Interleave RGB using partial planar_to_24b
    let rgb0 = r8;
    let rgb1 = _mm_setzero_si128();
    let rgb2 = g8;
    let rgb3 = _mm_setzero_si128();
    let rgb4 = b8;
    let rgb5 = _mm_setzero_si128();

    let (out0, out1, out2, _, _, _) = planar_to_24b(_token, rgb0, rgb1, rgb2, rgb3, rgb4, rgb5);

    // Store to fixed-size output - use split_at_mut for non-overlapping borrows
    let (rgb_0, rest) = rgb.split_at_mut(16);
    let (rgb_1, rgb_2) = rest.split_at_mut(16);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_0).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_1).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_2).unwrap(), out2);
}

/// Wrapper that converts slices to fixed-size arrays with a single bounds check.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_8_pairs_inner(
    _token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // Single bounds check at entry - arrays are then passed by reference
    let y: &[u8; 16] = y_row[..16].try_into().unwrap();
    let u1: &[u8; 9] = u_row_1[..9].try_into().unwrap();
    let u2: &[u8; 9] = u_row_2[..9].try_into().unwrap();
    let v1: &[u8; 9] = v_row_1[..9].try_into().unwrap();
    let v2: &[u8; 9] = v_row_2[..9].try_into().unwrap();
    let out: &mut [u8; 48] = (&mut rgb[..48]).try_into().unwrap();

    fancy_upsample_8_pairs_inner_opt(_token, y, u1, u2, v1, v2, out);
}

/// Process 16 pixel pairs (32 Y pixels) with fancy upsampling and YUV->RGB conversion.
/// Processes 2x the pixels of fancy_upsample_8_pairs, reducing loop overhead.
/// Takes: 32 Y bytes, 17 U/V bytes per row (for overlapping window), 96 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn fancy_upsample_16_pairs_with_token(
    token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    fancy_upsample_16_pairs_inner(token, y_row, u_row_1, u_row_2, v_row_1, v_row_2, rgb);
}

/// Optimized inner function for 32 Y pixels (16 chroma pairs).
/// Takes: 32 Y bytes, 17 U/V bytes per row, 96 RGB bytes output.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_16_pairs_inner(
    _token: X64V3Token,
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    rgb: &mut [u8],
) {
    // Single bounds check at entry
    let y: &[u8; 32] = y_row[..32].try_into().unwrap();
    let u1: &[u8; 17] = u_row_1[..17].try_into().unwrap();
    let u2: &[u8; 17] = u_row_2[..17].try_into().unwrap();
    let v1: &[u8; 17] = v_row_1[..17].try_into().unwrap();
    let v2: &[u8; 17] = v_row_2[..17].try_into().unwrap();
    let out: &mut [u8; 96] = (&mut rgb[..96]).try_into().unwrap();

    fancy_upsample_16_pairs_inner_opt(_token, y, u1, u2, v1, v2, out);
}

/// Core SIMD implementation for 32 Y pixels (16 chroma pairs).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn fancy_upsample_16_pairs_inner_opt(
    _token: X64V3Token,
    y_row: &[u8; 32],
    u_row_1: &[u8; 17],
    u_row_2: &[u8; 17],
    v_row_1: &[u8; 17],
    v_row_2: &[u8; 17],
    rgb: &mut [u8; 96],
) {
    // Load 16 U/V values for each position (a=offset0, b=offset1)
    let u_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_1[0..16]).unwrap());
    let u_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_1[1..17]).unwrap());
    let u_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_2[0..16]).unwrap());
    let u_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&u_row_2[1..17]).unwrap());

    let v_a = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_1[0..16]).unwrap());
    let v_b = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_1[1..17]).unwrap());
    let v_c = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_2[0..16]).unwrap());
    let v_d = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&v_row_2[1..17]).unwrap());

    // Compute fancy upsampled U/V - produces 16 diag1 + 16 diag2 values each
    let (u_diag1, u_diag2) = fancy_upsample_16(_token, u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(_token, v_a, v_b, v_c, v_d);

    // Interleave diag1/diag2 to get 32 U and 32 V values
    // First 16: unpacklo gives [d1[0],d2[0],d1[1],d2[1],...d1[7],d2[7]]
    // Second 16: unpackhi gives [d1[8],d2[8],...d1[15],d2[15]]
    let u_lo = _mm_unpacklo_epi8(u_diag1, u_diag2); // U values for Y[0..16]
    let u_hi = _mm_unpackhi_epi8(u_diag1, u_diag2); // U values for Y[16..32]
    let v_lo = _mm_unpacklo_epi8(v_diag1, v_diag2);
    let v_hi = _mm_unpackhi_epi8(v_diag1, v_diag2);

    // Load 32 Y values
    let y_0 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y_row[0..16]).unwrap());
    let y_1 = simd_mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&y_row[16..32]).unwrap());

    let zero = _mm_setzero_si128();

    // Process first 16 Y pixels (4 groups of 8 for YUV conversion)
    // Group 0: Y[0..8], U_lo[0..8], V_lo[0..8]
    let y_0_lo = _mm_unpacklo_epi8(zero, y_0);
    let u_0_lo = _mm_unpacklo_epi8(zero, u_lo);
    let v_0_lo = _mm_unpacklo_epi8(zero, v_lo);
    let (r0, g0, b0) = convert_yuv444_to_rgb(_token, y_0_lo, u_0_lo, v_0_lo);

    // Group 1: Y[8..16], U_lo[8..16], V_lo[8..16]
    let y_0_hi = _mm_unpackhi_epi8(zero, y_0);
    let u_0_hi = _mm_unpackhi_epi8(zero, u_lo);
    let v_0_hi = _mm_unpackhi_epi8(zero, v_lo);
    let (r1, g1, b1) = convert_yuv444_to_rgb(_token, y_0_hi, u_0_hi, v_0_hi);

    // Group 2: Y[16..24], U_hi[0..8], V_hi[0..8]
    let y_1_lo = _mm_unpacklo_epi8(zero, y_1);
    let u_1_lo = _mm_unpacklo_epi8(zero, u_hi);
    let v_1_lo = _mm_unpacklo_epi8(zero, v_hi);
    let (r2, g2, b2) = convert_yuv444_to_rgb(_token, y_1_lo, u_1_lo, v_1_lo);

    // Group 3: Y[24..32], U_hi[8..16], V_hi[8..16]
    let y_1_hi = _mm_unpackhi_epi8(zero, y_1);
    let u_1_hi = _mm_unpackhi_epi8(zero, u_hi);
    let v_1_hi = _mm_unpackhi_epi8(zero, v_hi);
    let (r3, g3, b3) = convert_yuv444_to_rgb(_token, y_1_hi, u_1_hi, v_1_hi);

    // Pack R/G/B to 8-bit: each packus gives 16 bytes
    let r_0 = _mm_packus_epi16(r0, r1); // R for Y[0..16]
    let r_1 = _mm_packus_epi16(r2, r3); // R for Y[16..32]
    let g_0 = _mm_packus_epi16(g0, g1);
    let g_1 = _mm_packus_epi16(g2, g3);
    let b_0 = _mm_packus_epi16(b0, b1);
    let b_1 = _mm_packus_epi16(b2, b3);

    // Interleave RGB using planar_to_24b (32 pixels -> 96 bytes)
    let (out0, out1, out2, out3, out4, out5) = planar_to_24b(_token, r_0, r_1, g_0, g_1, b_0, b_1);

    // Store 96 bytes
    let (rgb_0, rest) = rgb.split_at_mut(16);
    let (rgb_1, rest) = rest.split_at_mut(16);
    let (rgb_2, rest) = rest.split_at_mut(16);
    let (rgb_3, rest) = rest.split_at_mut(16);
    let (rgb_4, rgb_5) = rest.split_at_mut(16);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_0).unwrap(), out0);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_1).unwrap(), out1);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_2).unwrap(), out2);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_3).unwrap(), out3);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_4).unwrap(), out4);
    simd_mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(rgb_5).unwrap(), out5);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_to_rgb_matches_scalar() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let test_cases: [(u8, u8, u8); 8] = [
            (128, 128, 128),
            (255, 128, 128),
            (0, 128, 128),
            (203, 40, 42),
            (77, 34, 97),
            (162, 101, 167),
            (202, 84, 150),
            (185, 101, 167),
        ];

        let y: Vec<u8> = test_cases.iter().map(|(y, _, _)| *y).collect();
        let u: Vec<u8> = test_cases.iter().map(|(_, u, _)| *u).collect();
        let v: Vec<u8> = test_cases.iter().map(|(_, _, v)| *v).collect();

        let u_420: Vec<u8> = u.iter().step_by(2).copied().collect();
        let v_420: Vec<u8> = v.iter().step_by(2).copied().collect();

        let mut rgb_simd = vec![0u8; 24];
        yuv420_to_rgb_row(&y, &u_420, &v_420, &mut rgb_simd);

        for i in 0..8 {
            let y_val = y[i];
            let u_val = u_420[i / 2];
            let v_val = v_420[i / 2];
            let (r_scalar, g_scalar, b_scalar) = yuv_to_rgb_scalar(y_val, u_val, v_val);

            assert_eq!(rgb_simd[i * 3], r_scalar);
            assert_eq!(rgb_simd[i * 3 + 1], g_scalar);
            assert_eq!(rgb_simd[i * 3 + 2], b_scalar);
        }
    }

    #[test]
    fn test_yuv_to_rgb_32_pixels() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let y: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
        let u: Vec<u8> = (0..16).map(|i| (128 + i * 4) as u8).collect();
        let v: Vec<u8> = (0..16).map(|i| (128 - i * 4) as u8).collect();

        let mut rgb_simd = vec![0u8; 96];
        yuv420_to_rgb_row(&y, &u, &v, &mut rgb_simd);

        for i in 0..32 {
            let y_val = y[i];
            let u_val = u[i / 2];
            let v_val = v[i / 2];
            let (r_scalar, g_scalar, b_scalar) = yuv_to_rgb_scalar(y_val, u_val, v_val);

            assert_eq!(rgb_simd[i * 3], r_scalar);
            assert_eq!(rgb_simd[i * 3 + 1], g_scalar);
            assert_eq!(rgb_simd[i * 3 + 2], b_scalar);
        }
    }

    fn get_fancy_chroma_value(main: u8, secondary1: u8, secondary2: u8, tertiary: u8) -> u8 {
        let val0 = u16::from(main);
        let val1 = u16::from(secondary1);
        let val2 = u16::from(secondary2);
        let val3 = u16::from(tertiary);
        ((9 * val0 + 3 * val1 + 3 * val2 + val3 + 8) / 16) as u8
    }

    #[test]
    fn test_fancy_upsample_8_pairs() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        let y_row: [u8; 16] = [
            77, 162, 202, 185, 28, 13, 199, 182, 135, 147, 164, 135, 66, 27, 171, 130,
        ];
        let u_row_1: [u8; 9] = [34, 101, 84, 123, 163, 90, 110, 140, 120];
        let u_row_2: [u8; 9] = [123, 163, 133, 150, 100, 80, 95, 105, 115];
        let v_row_1: [u8; 9] = [97, 167, 150, 149, 23, 45, 67, 89, 100];
        let v_row_2: [u8; 9] = [149, 23, 86, 100, 120, 55, 75, 95, 110];

        let mut rgb_simd = [0u8; 48];
        fancy_upsample_8_pairs(
            &y_row,
            &u_row_1,
            &u_row_2,
            &v_row_1,
            &v_row_2,
            &mut rgb_simd,
        );

        let mut rgb_scalar = [0u8; 48];
        for i in 0..8 {
            let u_diag1 =
                get_fancy_chroma_value(u_row_1[i], u_row_1[i + 1], u_row_2[i], u_row_2[i + 1]);
            let v_diag1 =
                get_fancy_chroma_value(v_row_1[i], v_row_1[i + 1], v_row_2[i], v_row_2[i + 1]);
            let u_diag2 =
                get_fancy_chroma_value(u_row_1[i + 1], u_row_1[i], u_row_2[i + 1], u_row_2[i]);
            let v_diag2 =
                get_fancy_chroma_value(v_row_1[i + 1], v_row_1[i], v_row_2[i + 1], v_row_2[i]);

            let (r1, g1, b1) = yuv_to_rgb_scalar(y_row[i * 2], u_diag1, v_diag1);
            let (r2, g2, b2) = yuv_to_rgb_scalar(y_row[i * 2 + 1], u_diag2, v_diag2);

            rgb_scalar[i * 6] = r1;
            rgb_scalar[i * 6 + 1] = g1;
            rgb_scalar[i * 6 + 2] = b1;
            rgb_scalar[i * 6 + 3] = r2;
            rgb_scalar[i * 6 + 4] = g2;
            rgb_scalar[i * 6 + 5] = b2;
        }

        for i in 0..16 {
            assert_eq!(rgb_simd[i * 3], rgb_scalar[i * 3]);
            assert_eq!(rgb_simd[i * 3 + 1], rgb_scalar[i * 3 + 1]);
            assert_eq!(rgb_simd[i * 3 + 2], rgb_scalar[i * 3 + 2]);
        }
    }
}
