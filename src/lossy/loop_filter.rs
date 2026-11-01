//! Does loop filtering on webp lossy images
//!
//! The filters work on 16 pixels of an edge at a time. Every pixel computes both of its branches
//! and keeps one with a select, and all the arithmetic stays in the byte lanes, so that the
//! compiler can turn each loop into SIMD for any target.

/// The 4 pixels either side of an edge: `p3, p2, p1, p0, q0, q1, q2, q3`, each 16 pixels along it.
pub(crate) type Edge = [[u8; 16]; 8];

#[inline(always)]
const fn u2s(val: u8) -> i8 {
    (val ^ 0x80) as i8
}

#[inline(always)]
const fn s2u(val: i8) -> u8 {
    (val as u8) ^ 0x80
}

/// `clamp(outer + 3 * (q0 - p0))` on signed pixels
///
/// Saturating three times gives the same result as clamping once: `q0 - p0` can only saturate when
/// the sum is past the limit anyway, and from there the remaining additions cannot bring it back.
#[inline(always)]
const fn base_delta(outer: i8, p0: i8, q0: i8) -> i8 {
    let d = q0.saturating_sub(p0);
    outer.saturating_add(d).saturating_add(d).saturating_add(d)
}

/// All ones if `condition`, otherwise zero. A plain `if` here can come back as a branch per pixel.
#[inline(always)]
const fn mask(condition: bool) -> i8 {
    -(condition as i8)
}

#[inline(always)]
fn simple_threshold(edge_limit: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> bool {
    let d = p0.abs_diff(q0);
    d.saturating_add(d).saturating_add(p1.abs_diff(q1) >> 1) <= edge_limit
}

#[inline(always)]
fn should_filter(interior_limit: u8, edge_limit: u8, p: [u8; 4], q: [u8; 4]) -> bool {
    let [p3, p2, p1, p0] = p;
    let [q0, q1, q2, q3] = q;
    // There are only 6 diff comparisons required as per the spec:
    // https://www.rfc-editor.org/rfc/rfc6386#section-20.6
    let interior = p3
        .abs_diff(p2)
        .max(p2.abs_diff(p1))
        .max(p1.abs_diff(p0))
        .max(q3.abs_diff(q2))
        .max(q2.abs_diff(q1))
        .max(q1.abs_diff(q0));
    (interior <= interior_limit) & simple_threshold(edge_limit, p1, p0, q0, q1)
}

#[inline(always)]
fn high_edge_variance(threshold: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> bool {
    p1.abs_diff(p0).max(q1.abs_diff(q0)) > threshold
}

/// The simple filter described in 15.2 in the specification
///
/// Affects 4 pixels on an edge(2 each side)
pub(crate) fn simple_filter(edge_limit: u8, edge: &mut Edge) {
    let [_, _, p1, p0, q0, q1, _, _] = edge;

    for i in 0..16 {
        let filter = simple_threshold(edge_limit, p1[i], p0[i], q0[i], q1[i]);

        let (sp0, sq0) = (u2s(p0[i]), u2s(q0[i]));
        let a = base_delta(u2s(p1[i]).saturating_sub(u2s(q1[i])), sp0, sq0);
        let b = a.saturating_add(3) >> 3;
        let a = a.saturating_add(4) >> 3;

        let filter = mask(filter);
        q0[i] = s2u(sq0.saturating_sub(a & filter));
        p0[i] = s2u(sp0.saturating_add(b & filter));
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the edges between subblocks inside a macroblock
pub(crate) fn subblock_filter(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    edge: &mut Edge,
) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = edge;

    for i in 0..16 {
        let p = [p3[i], p2[i], p1[i], p0[i]];
        let q = [q0[i], q1[i], q2[i], q3[i]];
        let filter = should_filter(interior_limit, edge_limit, p, q);
        let hev = high_edge_variance(hev_threshold, p1[i], p0[i], q0[i], q1[i]);

        let (sp1, sp0, sq0, sq1) = (u2s(p1[i]), u2s(p0[i]), u2s(q0[i]), u2s(q1[i]));
        let outer = sp1.saturating_sub(sq1) & mask(hev);
        let a = base_delta(outer, sp0, sq0);
        let b = a.saturating_add(3) >> 3;
        let a = a.saturating_add(4) >> 3;
        let outer = (a + 1) >> 1;

        let filter = mask(filter);
        q0[i] = s2u(sq0.saturating_sub(a & filter));
        p0[i] = s2u(sp0.saturating_add(b & filter));
        let outer = outer & filter & !mask(hev);
        q1[i] = s2u(sq1.saturating_sub(outer));
        p1[i] = s2u(sp1.saturating_add(outer));
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the edges between macroblocks
pub(crate) fn macroblock_filter(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    edge: &mut Edge,
) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = edge;

    for i in 0..16 {
        let p = [p3[i], p2[i], p1[i], p0[i]];
        let q = [q0[i], q1[i], q2[i], q3[i]];
        let filter = should_filter(interior_limit, edge_limit, p, q);
        let hev = high_edge_variance(hev_threshold, p1[i], p0[i], q0[i], q1[i]);

        let (sp2, sp1, sp0) = (u2s(p2[i]), u2s(p1[i]), u2s(p0[i]));
        let (sq0, sq1, sq2) = (u2s(q0[i]), u2s(q1[i]), u2s(q2[i]));

        let w = base_delta(sp1.saturating_sub(sq1), sp0, sq0);
        let wide = i16::from(w);
        // These cannot leave the range of an i8: `w` is at most 128 away from 0.
        let a0 = ((27 * wide + 63) >> 7) as i8;
        let a1 = ((18 * wide + 63) >> 7) as i8;
        let a2 = ((9 * wide + 63) >> 7) as i8;

        let hev = mask(hev);
        let a = (w.saturating_add(4) >> 3) & hev | a0 & !hev;
        let b = (w.saturating_add(3) >> 3) & hev | a0 & !hev;
        let (a1, a2) = (a1 & !hev, a2 & !hev);

        let filter = mask(filter);
        q0[i] = s2u(sq0.saturating_sub(a & filter));
        p0[i] = s2u(sp0.saturating_add(b & filter));
        let a1 = a1 & filter;
        q1[i] = s2u(sq1.saturating_sub(a1));
        p1[i] = s2u(sp1.saturating_add(a1));
        let a2 = a2 & filter;
        q2[i] = s2u(sq2.saturating_sub(a2));
        p2[i] = s2u(sp2.saturating_add(a2));
    }
}

/// Swaps the `mask` bits of each lane of `b` with the bits `SHIFT` above them in `a`
#[inline(always)]
fn swap_bits<const SHIFT: u32>(a: &mut [u64; 2], b: &mut [u64; 2], mask: u64) {
    for i in 0..2 {
        let swapped = ((a[i] >> SHIFT) ^ b[i]) & mask;
        b[i] ^= swapped;
        a[i] ^= swapped << SHIFT;
    }
}

/// Transposes the first 8 bytes of the rows, and the last 8 bytes of the rows
#[inline(always)]
fn transpose(edge: &mut Edge) {
    const BYTES: u64 = 0x00ff_00ff_00ff_00ff;
    const SHORTS: u64 = 0x0000_ffff_0000_ffff;
    const WORDS: u64 = 0x0000_0000_ffff_ffff;

    let mut lanes = edge.map(|row| {
        let (low, high) = row.split_at(8);
        [
            u64::from_le_bytes(low.try_into().unwrap()),
            u64::from_le_bytes(high.try_into().unwrap()),
        ]
    });

    let [r0, r1, r2, r3, r4, r5, r6, r7] = &mut lanes;
    swap_bits::<8>(r0, r1, BYTES);
    swap_bits::<8>(r2, r3, BYTES);
    swap_bits::<8>(r4, r5, BYTES);
    swap_bits::<8>(r6, r7, BYTES);
    swap_bits::<16>(r0, r2, SHORTS);
    swap_bits::<16>(r1, r3, SHORTS);
    swap_bits::<16>(r4, r6, SHORTS);
    swap_bits::<16>(r5, r7, SHORTS);
    swap_bits::<32>(r0, r4, WORDS);
    swap_bits::<32>(r1, r5, WORDS);
    swap_bits::<32>(r2, r6, WORDS);
    swap_bits::<32>(r3, r7, WORDS);

    for (row, [low, high]) in edge.iter_mut().zip(lanes) {
        row[..8].copy_from_slice(&low.to_le_bytes());
        row[8..].copy_from_slice(&high.to_le_bytes());
    }
}

/// Filters across the columns of 16 rows of `pixels`, starting at the edge at `point`
#[inline(always)]
pub(crate) fn filter_across_columns(
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    filter: impl FnOnce(&mut Edge),
) {
    let start = point - 4;
    let mut edge = [[0; 16]; 8];
    for (i, row) in edge.iter_mut().enumerate() {
        row[..8].copy_from_slice(&pixels[start + i * stride..][..8]);
        row[8..].copy_from_slice(&pixels[start + (i + 8) * stride..][..8]);
    }
    transpose(&mut edge);
    filter(&mut edge);
    transpose(&mut edge);
    for (i, row) in edge.iter().enumerate() {
        pixels[start + i * stride..][..8].copy_from_slice(&row[..8]);
        pixels[start + (i + 8) * stride..][..8].copy_from_slice(&row[8..]);
    }
}

/// Filters across the rows of 16 columns of `pixels`, starting at the edge at `point`
#[inline(always)]
pub(crate) fn filter_across_rows(
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    filter: impl FnOnce(&mut Edge),
) {
    let start = point - 4 * stride;
    let mut edge = [[0; 16]; 8];
    for (i, row) in edge.iter_mut().enumerate() {
        row.copy_from_slice(&pixels[start + i * stride..][..16]);
    }
    filter(&mut edge);
    // `p3` and `q3` are only read.
    for (i, row) in edge.iter().enumerate().take(7).skip(1) {
        pixels[start + i * stride..][..16].copy_from_slice(row);
    }
}

/// Filters across the columns of 8 rows of both chroma planes, starting at the edge at `point`
#[inline(always)]
pub(crate) fn filter_chroma_across_columns(
    u: &mut [u8],
    v: &mut [u8],
    point: usize,
    stride: usize,
    filter: impl FnOnce(&mut Edge),
) {
    let start = point - 4;
    let mut edge = [[0; 16]; 8];
    for (i, row) in edge.iter_mut().enumerate() {
        row[..8].copy_from_slice(&u[start + i * stride..][..8]);
        row[8..].copy_from_slice(&v[start + i * stride..][..8]);
    }
    transpose(&mut edge);
    filter(&mut edge);
    transpose(&mut edge);
    for (i, row) in edge.iter().enumerate() {
        u[start + i * stride..][..8].copy_from_slice(&row[..8]);
        v[start + i * stride..][..8].copy_from_slice(&row[8..]);
    }
}

/// Filters across the rows of 8 columns of both chroma planes, starting at the edge at `point`
#[inline(always)]
pub(crate) fn filter_chroma_across_rows(
    u: &mut [u8],
    v: &mut [u8],
    point: usize,
    stride: usize,
    filter: impl FnOnce(&mut Edge),
) {
    let start = point - 4 * stride;
    let mut edge = [[0; 16]; 8];
    for (i, row) in edge.iter_mut().enumerate() {
        row[..8].copy_from_slice(&u[start + i * stride..][..8]);
        row[8..].copy_from_slice(&v[start + i * stride..][..8]);
    }
    filter(&mut edge);
    // `p3` and `q3` are only read.
    for (i, row) in edge.iter().enumerate().take(7).skip(1) {
        u[start + i * stride..][..8].copy_from_slice(&row[..8]);
        v[start + i * stride..][..8].copy_from_slice(&row[8..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn c(val: i32) -> i32 {
        val.clamp(-128, 127)
    }

    fn signed(val: u8) -> i32 {
        i32::from(val) - 128
    }

    fn unsigned(val: i32) -> u8 {
        (c(val) + 128) as u8
    }

    /// Per pixel filters written out as in the specification
    fn common_adjust(use_outer_taps: bool, pixels: &mut [u8; 8]) -> i32 {
        let p1 = signed(pixels[2]);
        let p0 = signed(pixels[3]);
        let q0 = signed(pixels[4]);
        let q1 = signed(pixels[5]);

        let outer = if use_outer_taps { c(p1 - q1) } else { 0 };
        let a = c(outer + 3 * (q0 - p0));
        let b = (c(a + 3)) >> 3;
        let a = (c(a + 4)) >> 3;

        pixels[4] = unsigned(q0 - a);
        pixels[3] = unsigned(p0 + b);

        a
    }

    fn simple_threshold_scalar(filter_limit: i32, pixels: &[u8; 8]) -> bool {
        i32::from(pixels[3].abs_diff(pixels[4])) * 2 + i32::from(pixels[2].abs_diff(pixels[5])) / 2
            <= filter_limit
    }

    fn should_filter_scalar(interior_limit: u8, edge_limit: u8, pixels: &[u8; 8]) -> bool {
        simple_threshold_scalar(i32::from(edge_limit), pixels)
            && pixels[0].abs_diff(pixels[1]) <= interior_limit
            && pixels[1].abs_diff(pixels[2]) <= interior_limit
            && pixels[2].abs_diff(pixels[3]) <= interior_limit
            && pixels[7].abs_diff(pixels[6]) <= interior_limit
            && pixels[6].abs_diff(pixels[5]) <= interior_limit
            && pixels[5].abs_diff(pixels[4]) <= interior_limit
    }

    fn hev_scalar(threshold: u8, pixels: &[u8; 8]) -> bool {
        pixels[2].abs_diff(pixels[3]) > threshold || pixels[5].abs_diff(pixels[4]) > threshold
    }

    fn simple_scalar(edge_limit: u8, pixels: &mut [u8; 8]) {
        if simple_threshold_scalar(i32::from(edge_limit), pixels) {
            common_adjust(true, pixels);
        }
    }

    fn subblock_scalar(hev: u8, interior: u8, edge_limit: u8, pixels: &mut [u8; 8]) {
        if should_filter_scalar(interior, edge_limit, pixels) {
            let hv = hev_scalar(hev, pixels);
            let a = (common_adjust(hv, pixels) + 1) >> 1;
            if !hv {
                pixels[5] = unsigned(signed(pixels[5]) - a);
                pixels[2] = unsigned(signed(pixels[2]) + a);
            }
        }
    }

    fn macroblock_scalar(hev: u8, interior: u8, edge_limit: u8, pixels: &mut [u8; 8]) {
        if should_filter_scalar(interior, edge_limit, pixels) {
            if !hev_scalar(hev, pixels) {
                let p2 = signed(pixels[1]);
                let p1 = signed(pixels[2]);
                let p0 = signed(pixels[3]);
                let q0 = signed(pixels[4]);
                let q1 = signed(pixels[5]);
                let q2 = signed(pixels[6]);

                let w = c(c(p1 - q1) + 3 * (q0 - p0));

                let a = c((27 * w + 63) >> 7);
                pixels[4] = unsigned(q0 - a);
                pixels[3] = unsigned(p0 + a);

                let a = c((18 * w + 63) >> 7);
                pixels[5] = unsigned(q1 - a);
                pixels[2] = unsigned(p1 + a);

                let a = c((9 * w + 63) >> 7);
                pixels[6] = unsigned(q2 - a);
                pixels[1] = unsigned(p2 + a);
            } else {
                common_adjust(true, pixels);
            }
        }
    }

    fn lane(edge: &Edge, i: usize) -> [u8; 8] {
        std::array::from_fn(|row| edge[row][i])
    }

    /// An edge whose pixels sit close enough for the filters to act on many of them
    fn random_edge(rng: &mut impl Rng) -> Edge {
        let spread = [1, 2, 4, 8, 16, 32, 64, 256][rng.gen_range(0..8)];
        let base = rng.gen_range(0..256);
        let mut edge = [[0; 16]; 8];
        for row in &mut edge {
            for pixel in row {
                *pixel = (base + rng.gen_range(0..spread)).clamp(0, 255) as u8;
            }
        }
        edge
    }

    #[test]
    fn kernels_match_per_pixel_filters() {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..10_000 {
            let hev = rng.gen_range(0..=2);
            let interior = rng.gen_range(1..=63);
            let edge_limit = rng.gen_range(0..=193);
            let edge = random_edge(&mut rng);

            let mut simple = edge;
            simple_filter(edge_limit, &mut simple);
            let mut subblock = edge;
            subblock_filter(hev, interior, edge_limit, &mut subblock);
            let mut macroblock = edge;
            macroblock_filter(hev, interior, edge_limit, &mut macroblock);

            for i in 0..16 {
                let mut pixels = lane(&edge, i);
                simple_scalar(edge_limit, &mut pixels);
                assert_eq!(pixels, lane(&simple, i));

                let mut pixels = lane(&edge, i);
                subblock_scalar(hev, interior, edge_limit, &mut pixels);
                assert_eq!(pixels, lane(&subblock, i));

                let mut pixels = lane(&edge, i);
                macroblock_scalar(hev, interior, edge_limit, &mut pixels);
                assert_eq!(pixels, lane(&macroblock, i));
            }
        }
    }

    #[test]
    fn filters_read_the_edge_they_are_given() {
        const STRIDE: usize = 24;
        let mut rng = StdRng::seed_from_u64(1);
        let mut u = [0u8; 32 * STRIDE];
        let mut v = [0u8; 32 * STRIDE];
        rng.fill_bytes(&mut u);
        rng.fill_bytes(&mut v);
        let point = 8 * STRIDE + 5;

        let (mut u_columns, mut u_rows) = (u, u);
        filter_across_columns(&mut u_columns, point, STRIDE, |edge| {
            for (row, pixels) in edge.iter_mut().enumerate() {
                for (i, pixel) in pixels.iter_mut().enumerate() {
                    assert_eq!(*pixel, u[point - 4 + row + i * STRIDE]);
                    *pixel = !*pixel;
                }
            }
        });
        filter_across_rows(&mut u_rows, point, STRIDE, |edge| {
            for (row, pixels) in edge.iter_mut().enumerate() {
                for (i, pixel) in pixels.iter_mut().enumerate() {
                    assert_eq!(*pixel, u[point - 4 * STRIDE + row * STRIDE + i]);
                    *pixel = !*pixel;
                }
            }
        });
        for y in 0..32 {
            for x in 0..STRIDE {
                let i = y * STRIDE + x;
                let on_rows = (5..21).contains(&x) && (4..12).contains(&y);
                // The outermost pixels of an edge are only read across rows.
                let on_rows = on_rows && (5..11).contains(&y);
                let on_columns = (1..9).contains(&x) && (8..24).contains(&y);
                assert_eq!(u_columns[i], if on_columns { !u[i] } else { u[i] });
                assert_eq!(u_rows[i], if on_rows { !u[i] } else { u[i] });
            }
        }

        let (mut u2, mut v2) = (u, v);
        filter_chroma_across_columns(&mut u2, &mut v2, point, STRIDE, |edge| {
            for (row, pixels) in edge.iter().enumerate() {
                for i in 0..8 {
                    assert_eq!(pixels[i], u[point - 4 + row + i * STRIDE]);
                    assert_eq!(pixels[8 + i], v[point - 4 + row + i * STRIDE]);
                }
            }
        });
        filter_chroma_across_rows(&mut u2, &mut v2, point, STRIDE, |edge| {
            for (row, pixels) in edge.iter().enumerate() {
                for i in 0..8 {
                    assert_eq!(pixels[i], u[point - 4 * STRIDE + row * STRIDE + i]);
                    assert_eq!(pixels[8 + i], v[point - 4 * STRIDE + row * STRIDE + i]);
                }
            }
        });
        assert_eq!(u, u2);
        assert_eq!(v, v2);
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benches {
    use super::*;
    use test::{black_box, Bencher};

    #[rustfmt::skip]
    const TEST_DATA: [[u8; 8]; 8] = [
        [177, 192, 179, 181, 185, 174, 186, 193],
        [185, 180, 175, 179, 175, 190, 189, 190],
        [185, 181, 177, 190, 190, 174, 176, 188],
        [192, 179, 186, 175, 190, 184, 190, 175],
        [175, 183, 183, 190, 187, 186, 176, 181],
        [183, 177, 182, 185, 183, 179, 178, 181],
        [191, 183, 188, 181, 180, 193, 185, 180],
        [177, 182, 177, 178, 179, 178, 191, 178],
    ];

    fn pixels() -> [u8; 16 * 16] {
        std::array::from_fn(|i| TEST_DATA[i / 16 % 8][i % 8])
    }

    #[bench]
    fn measure_macroblock_filter(b: &mut Bencher) {
        let mut edge: Edge = std::array::from_fn(|i| pixels()[i * 16..][..16].try_into().unwrap());
        b.iter(|| black_box(macroblock_filter(5, 15, 15, black_box(&mut edge))));
    }

    #[bench]
    fn measure_subblock_filter(b: &mut Bencher) {
        let mut edge: Edge = std::array::from_fn(|i| pixels()[i * 16..][..16].try_into().unwrap());
        b.iter(|| black_box(subblock_filter(5, 15, 15, black_box(&mut edge))));
    }

    #[bench]
    fn measure_simple_filter(b: &mut Bencher) {
        let mut edge: Edge = std::array::from_fn(|i| pixels()[i * 16..][..16].try_into().unwrap());
        b.iter(|| black_box(simple_filter(15, black_box(&mut edge))));
    }

    #[bench]
    fn measure_filter_across_columns(b: &mut Bencher) {
        let mut data = pixels();
        b.iter(|| filter_across_columns(black_box(&mut data), 4, 16, |e| black_box(e).reverse()));
    }

    #[bench]
    fn measure_filter_across_rows(b: &mut Bencher) {
        let mut data = pixels();
        b.iter(|| filter_across_rows(black_box(&mut data), 4 * 16, 16, |e| black_box(e).reverse()));
    }
}
