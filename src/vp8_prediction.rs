//! Methods related to vp8 prediction used in both the decoder and the encoder
//!
//! Functions for doing prediction and for setting up buffers for prediction

/// Luma prediction block includes the 1 pixel border to the left and on top
/// as well as 4 pixels to the top right of the block
pub(crate) const LUMA_BLOCK_SIZE: usize = (1 + 16 + 4) * (1 + 16);
pub(crate) const LUMA_STRIDE: usize = 1 + 16 + 4;

// creates the luma block with the border from the buffers
pub(crate) fn create_border_luma(
    mbx: usize,
    mby: usize,
    mbw: usize,
    top: &[u8],
    left: &[u8],
) -> [u8; LUMA_BLOCK_SIZE] {
    let stride = LUMA_STRIDE;
    let mut ws = [0u8; LUMA_BLOCK_SIZE];

    // A
    {
        let above = &mut ws[1..stride];
        if mby == 0 {
            for above in above.iter_mut() {
                *above = 127;
            }
        } else {
            for (above, &top) in above[..16].iter_mut().zip(&top[mbx * 16..]) {
                *above = top;
            }

            if mbx == mbw - 1 {
                for above in &mut above[16..] {
                    *above = top[mbx * 16 + 15];
                }
            } else {
                for (above, &top) in above[16..].iter_mut().zip(&top[mbx * 16 + 16..]) {
                    *above = top;
                }
            }
        }
    }

    for i in 17usize..stride {
        ws[4 * stride + i] = ws[i];
        ws[8 * stride + i] = ws[i];
        ws[12 * stride + i] = ws[i];
    }

    // L
    if mbx == 0 {
        for i in 0usize..16 {
            ws[(i + 1) * stride] = 129;
        }
    } else {
        for (i, &left) in (0usize..16).zip(&left[1..]) {
            ws[(i + 1) * stride] = left;
        }
    }

    // P
    ws[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };

    ws
}

/// Chroma blocks only need the 1 pixel border left and above block
pub(crate) const CHROMA_BLOCK_SIZE: usize = (8 + 1) * (8 + 1);
pub(crate) const CHROMA_STRIDE: usize = 8 + 1;

// creates the left and top border for chroma prediction
pub(crate) fn create_border_chroma(
    mbx: usize,
    mby: usize,
    top: &[u8],
    left: &[u8],
) -> [u8; CHROMA_BLOCK_SIZE] {
    let mut chroma_block = [0u8; CHROMA_BLOCK_SIZE];

    // above
    {
        let above = &mut chroma_block[1..CHROMA_STRIDE];
        if mby == 0 {
            for above in above.iter_mut() {
                *above = 127;
            }
        } else {
            for (above, &top) in above.iter_mut().zip(&top[mbx * 8..]) {
                *above = top;
            }
        }
    }

    // left
    if mbx == 0 {
        for y in 0usize..8 {
            chroma_block[(y + 1) * CHROMA_STRIDE] = 129;
        }
    } else {
        for (y, &left) in (0usize..8).zip(&left[1..]) {
            chroma_block[(y + 1) * CHROMA_STRIDE] = left;
        }
    }

    chroma_block[0] = if mby == 0 {
        127
    } else if mbx == 0 {
        129
    } else {
        left[0]
    };

    chroma_block
}

// Helper function for adding residue to a 4x4 sub-block
// from a contiguous block of 16
//
// Only 16 elements from rblock are used to add residue, so it is restricted to 16 elements
// to enable SIMD and other optimizations.
//
// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
pub(crate) fn add_residue(
    pblock: &mut [u8],
    rblock: &[i32; 16],
    y0: usize,
    x0: usize,
    stride: usize,
) {
    let mut pos = y0 * stride + x0;
    for row in rblock.chunks(4) {
        for (p, &a) in pblock[pos..][..4].iter_mut().zip(row.iter()) {
            *p = (a + i32::from(*p)).max(0).min(255) as u8;
        }
        pos += stride;
    }
}

fn avg3(left: u8, this: u8, right: u8) -> u8 {
    let avg = (u16::from(left) + 2 * u16::from(this) + u16::from(right) + 2) >> 2;
    avg as u8
}

fn avg2(this: u8, right: u8) -> u8 {
    let avg = (u16::from(this) + u16::from(right) + 1) >> 1;
    avg as u8
}

pub(crate) fn predict_vpred(a: &mut [u8], size: usize, x0: usize, y0: usize, stride: usize) {
    // This pass copies the top row to the rows below it.
    let (above, curr) = a.split_at_mut(stride * y0);
    let above_slice = &above[x0..];

    for curr_chunk in curr.chunks_exact_mut(stride).take(size) {
        for (curr, &above) in curr_chunk[1..].iter_mut().zip(above_slice) {
            *curr = above;
        }
    }
}

pub(crate) fn predict_hpred(a: &mut [u8], size: usize, x0: usize, y0: usize, stride: usize) {
    // This pass copies the first value of a row to the values right of it.
    for chunk in a.chunks_exact_mut(stride).skip(y0).take(size) {
        let left = chunk[x0 - 1];
        chunk[x0..].iter_mut().for_each(|a| *a = left);
    }
}

pub(crate) fn predict_dcpred(a: &mut [u8], size: usize, stride: usize, above: bool, left: bool) {
    let mut sum = 0;
    let mut shf = if size == 8 { 2 } else { 3 };

    if left {
        for y in 0usize..size {
            sum += u32::from(a[(y + 1) * stride]);
        }

        shf += 1;
    }

    if above {
        sum += a[1..=size].iter().fold(0, |acc, &x| acc + u32::from(x));

        shf += 1;
    }

    let dcval = if !left && !above {
        128
    } else {
        (sum + (1 << (shf - 1))) >> shf
    };

    for y in 0usize..size {
        a[1 + stride * (y + 1)..][..size]
            .iter_mut()
            .for_each(|a| *a = dcval as u8);
    }
}

// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
pub(crate) fn predict_tmpred(a: &mut [u8], size: usize, x0: usize, y0: usize, stride: usize) {
    // The formula for tmpred is:
    // X_ij = L_i + A_j - P (i, j=0, 1, 2, 3)
    //
    // |-----|-----|-----|-----|-----|
    // | P   | A0  | A1  | A2  | A3  |
    // |-----|-----|-----|-----|-----|
    // | L0  | X00 | X01 | X02 | X03 |
    // |-----|-----|-----|-----|-----|
    // | L1  | X10 | X11 | X12 | X13 |
    // |-----|-----|-----|-----|-----|
    // | L2  | X20 | X21 | X22 | X23 |
    // |-----|-----|-----|-----|-----|
    // | L3  | X30 | X31 | X32 | X33 |
    // |-----|-----|-----|-----|-----|
    // Diagram from p. 52 of RFC 6386

    // Split at L0
    let (above, x_block) = a.split_at_mut(y0 * stride + (x0 - 1));
    let p = i32::from(above[(y0 - 1) * stride + x0 - 1]);
    let above_slice = &above[(y0 - 1) * stride + x0..];

    for y in 0usize..size {
        let left_minus_p = i32::from(x_block[y * stride]) - p;

        // Add 1 to skip over L0 byte
        x_block[y * stride + 1..][..size]
            .iter_mut()
            .zip(above_slice)
            .for_each(|(cur, &abv)| *cur = (left_minus_p + i32::from(abv)).max(0).min(255) as u8);
    }
}

pub(crate) fn predict_bdcpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let mut v = 4;

    a[(y0 - 1) * stride + x0..][..4]
        .iter()
        .for_each(|&a| v += u32::from(a));

    for i in 0usize..4 {
        v += u32::from(a[(y0 + i) * stride + x0 - 1]);
    }

    v >>= 3;
    for chunk in a.chunks_exact_mut(stride).skip(y0).take(4) {
        for ch in &mut chunk[x0..][..4] {
            *ch = v as u8;
        }
    }
}

fn topleft_pixel(a: &[u8], x0: usize, y0: usize, stride: usize) -> u8 {
    a[(y0 - 1) * stride + x0 - 1]
}

fn top_pixels(a: &[u8], x0: usize, y0: usize, stride: usize) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let pos = (y0 - 1) * stride + x0;
    let a_slice = &a[pos..pos + 8];
    let a0 = a_slice[0];
    let a1 = a_slice[1];
    let a2 = a_slice[2];
    let a3 = a_slice[3];
    let a4 = a_slice[4];
    let a5 = a_slice[5];
    let a6 = a_slice[6];
    let a7 = a_slice[7];

    (a0, a1, a2, a3, a4, a5, a6, a7)
}

fn left_pixels(a: &[u8], x0: usize, y0: usize, stride: usize) -> (u8, u8, u8, u8) {
    let l0 = a[y0 * stride + x0 - 1];
    let l1 = a[(y0 + 1) * stride + x0 - 1];
    let l2 = a[(y0 + 2) * stride + x0 - 1];
    let l3 = a[(y0 + 3) * stride + x0 - 1];

    (l0, l1, l2, l3)
}

fn edge_pixels(
    a: &[u8],
    x0: usize,
    y0: usize,
    stride: usize,
) -> (u8, u8, u8, u8, u8, u8, u8, u8, u8) {
    let pos = (y0 - 1) * stride + x0 - 1;
    let a_slice = &a[pos..=pos + 4];
    let e0 = a[pos + 4 * stride];
    let e1 = a[pos + 3 * stride];
    let e2 = a[pos + 2 * stride];
    let e3 = a[pos + stride];
    let e4 = a_slice[0];
    let e5 = a_slice[1];
    let e6 = a_slice[2];
    let e7 = a_slice[3];
    let e8 = a_slice[4];

    (e0, e1, e2, e3, e4, e5, e6, e7, e8)
}

pub(crate) fn predict_bvepred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let p = topleft_pixel(a, x0, y0, stride);
    let (a0, a1, a2, a3, a4, ..) = top_pixels(a, x0, y0, stride);
    let avg_1 = avg3(p, a0, a1);
    let avg_2 = avg3(a0, a1, a2);
    let avg_3 = avg3(a1, a2, a3);
    let avg_4 = avg3(a2, a3, a4);

    let avg = [avg_1, avg_2, avg_3, avg_4];

    let mut pos = y0 * stride + x0;
    for _ in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avg);
        pos += stride;
    }
}

pub(crate) fn predict_bhepred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let p = topleft_pixel(a, x0, y0, stride);
    let (l0, l1, l2, l3) = left_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(p, l0, l1),
        avg3(l0, l1, l2),
        avg3(l1, l2, l3),
        avg3(l2, l3, l3),
    ];

    let mut pos = y0 * stride + x0;
    for avg in avgs {
        for a_p in &mut a[pos..=pos + 3] {
            *a_p = avg;
        }
        pos += stride;
    }
}

pub(crate) fn predict_bldpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (a0, a1, a2, a3, a4, a5, a6, a7) = top_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(a0, a1, a2),
        avg3(a1, a2, a3),
        avg3(a2, a3, a4),
        avg3(a3, a4, a5),
        avg3(a4, a5, a6),
        avg3(a5, a6, a7),
        avg3(a6, a7, a7),
    ];

    let mut pos = y0 * stride + x0;

    for i in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avgs[i..=i + 3]);
        pos += stride;
    }
}

pub(crate) fn predict_brdpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (e0, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(a, x0, y0, stride);

    let avgs = [
        avg3(e0, e1, e2),
        avg3(e1, e2, e3),
        avg3(e2, e3, e4),
        avg3(e3, e4, e5),
        avg3(e4, e5, e6),
        avg3(e5, e6, e7),
        avg3(e6, e7, e8),
    ];
    let mut pos = y0 * stride + x0;

    for i in 0..4 {
        a[pos..=pos + 3].copy_from_slice(&avgs[3 - i..7 - i]);
        pos += stride;
    }
}

pub(crate) fn predict_bvrpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (_, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(a, x0, y0, stride);

    a[(y0 + 3) * stride + x0] = avg3(e1, e2, e3);
    a[(y0 + 2) * stride + x0] = avg3(e2, e3, e4);
    a[(y0 + 3) * stride + x0 + 1] = avg3(e3, e4, e5);
    a[(y0 + 1) * stride + x0] = avg3(e3, e4, e5);
    a[(y0 + 2) * stride + x0 + 1] = avg2(e4, e5);
    a[y0 * stride + x0] = avg2(e4, e5);
    a[(y0 + 3) * stride + x0 + 2] = avg3(e4, e5, e6);
    a[(y0 + 1) * stride + x0 + 1] = avg3(e4, e5, e6);
    a[(y0 + 2) * stride + x0 + 2] = avg2(e5, e6);
    a[y0 * stride + x0 + 1] = avg2(e5, e6);
    a[(y0 + 3) * stride + x0 + 3] = avg3(e5, e6, e7);
    a[(y0 + 1) * stride + x0 + 2] = avg3(e5, e6, e7);
    a[(y0 + 2) * stride + x0 + 3] = avg2(e6, e7);
    a[y0 * stride + x0 + 2] = avg2(e6, e7);
    a[(y0 + 1) * stride + x0 + 3] = avg3(e6, e7, e8);
    a[y0 * stride + x0 + 3] = avg2(e7, e8);
}

pub(crate) fn predict_bvlpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (a0, a1, a2, a3, a4, a5, a6, a7) = top_pixels(a, x0, y0, stride);

    a[y0 * stride + x0] = avg2(a0, a1);
    a[(y0 + 1) * stride + x0] = avg3(a0, a1, a2);
    a[(y0 + 2) * stride + x0] = avg2(a1, a2);
    a[y0 * stride + x0 + 1] = avg2(a1, a2);
    a[(y0 + 1) * stride + x0 + 1] = avg3(a1, a2, a3);
    a[(y0 + 3) * stride + x0] = avg3(a1, a2, a3);
    a[(y0 + 2) * stride + x0 + 1] = avg2(a2, a3);
    a[y0 * stride + x0 + 2] = avg2(a2, a3);
    a[(y0 + 3) * stride + x0 + 1] = avg3(a2, a3, a4);
    a[(y0 + 1) * stride + x0 + 2] = avg3(a2, a3, a4);
    a[(y0 + 2) * stride + x0 + 2] = avg2(a3, a4);
    a[y0 * stride + x0 + 3] = avg2(a3, a4);
    a[(y0 + 3) * stride + x0 + 2] = avg3(a3, a4, a5);
    a[(y0 + 1) * stride + x0 + 3] = avg3(a3, a4, a5);
    a[(y0 + 2) * stride + x0 + 3] = avg3(a4, a5, a6);
    a[(y0 + 3) * stride + x0 + 3] = avg3(a5, a6, a7);
}

pub(crate) fn predict_bhdpred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (e0, e1, e2, e3, e4, e5, e6, e7, _) = edge_pixels(a, x0, y0, stride);

    a[(y0 + 3) * stride + x0] = avg2(e0, e1);
    a[(y0 + 3) * stride + x0 + 1] = avg3(e0, e1, e2);
    a[(y0 + 2) * stride + x0] = avg2(e1, e2);
    a[(y0 + 3) * stride + x0 + 2] = avg2(e1, e2);
    a[(y0 + 2) * stride + x0 + 1] = avg3(e1, e2, e3);
    a[(y0 + 3) * stride + x0 + 3] = avg3(e1, e2, e3);
    a[(y0 + 2) * stride + x0 + 2] = avg2(e2, e3);
    a[(y0 + 1) * stride + x0] = avg2(e2, e3);
    a[(y0 + 2) * stride + x0 + 3] = avg3(e2, e3, e4);
    a[(y0 + 1) * stride + x0 + 1] = avg3(e2, e3, e4);
    a[(y0 + 1) * stride + x0 + 2] = avg2(e3, e4);
    a[y0 * stride + x0] = avg2(e3, e4);
    a[(y0 + 1) * stride + x0 + 3] = avg3(e3, e4, e5);
    a[y0 * stride + x0 + 1] = avg3(e3, e4, e5);
    a[y0 * stride + x0 + 2] = avg3(e4, e5, e6);
    a[y0 * stride + x0 + 3] = avg3(e5, e6, e7);
}

pub(crate) fn predict_bhupred(a: &mut [u8], x0: usize, y0: usize, stride: usize) {
    let (l0, l1, l2, l3) = left_pixels(a, x0, y0, stride);

    a[y0 * stride + x0] = avg2(l0, l1);
    a[y0 * stride + x0 + 1] = avg3(l0, l1, l2);
    a[y0 * stride + x0 + 2] = avg2(l1, l2);
    a[(y0 + 1) * stride + x0] = avg2(l1, l2);
    a[y0 * stride + x0 + 3] = avg3(l1, l2, l3);
    a[(y0 + 1) * stride + x0 + 1] = avg3(l1, l2, l3);
    a[(y0 + 1) * stride + x0 + 2] = avg2(l2, l3);
    a[(y0 + 2) * stride + x0] = avg2(l2, l3);
    a[(y0 + 1) * stride + x0 + 3] = avg3(l2, l3, l3);
    a[(y0 + 2) * stride + x0 + 1] = avg3(l2, l3, l3);
    a[(y0 + 2) * stride + x0 + 2] = l3;
    a[(y0 + 2) * stride + x0 + 3] = l3;
    a[(y0 + 3) * stride + x0] = l3;
    a[(y0 + 3) * stride + x0 + 1] = l3;
    a[(y0 + 3) * stride + x0 + 2] = l3;
    a[(y0 + 3) * stride + x0 + 3] = l3;
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benches {
    use super::*;
    use test::{black_box, Bencher};

    const W: usize = 256;
    const H: usize = 256;

    fn make_sample_image() -> Vec<u8> {
        let mut v = Vec::with_capacity((W * H * 4) as usize);
        for c in 0u8..=255 {
            for k in 0u8..=255 {
                v.push(c);
                v.push(0);
                v.push(0);
                v.push(k);
            }
        }
        v
    }

    #[bench]
    fn bench_predict_4x4(b: &mut Bencher) {
        let mut v = black_box(make_sample_image());

        let res_data = vec![1i32; W * H * 4];
        let modes = [
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::DC,
            IntraMode::LD,
            IntraMode::RD,
            IntraMode::VR,
            IntraMode::VL,
            IntraMode::HD,
            IntraMode::HU,
            IntraMode::TM,
            IntraMode::VE,
            IntraMode::HE,
            IntraMode::DC,
            IntraMode::LD,
            IntraMode::RD,
        ];

        b.iter(|| {
            black_box(predict_4x4(&mut v, W * 2, &modes, &res_data));
        });
    }

    #[bench]
    fn bench_predict_bvepred(b: &mut Bencher) {
        let mut v = make_sample_image();

        b.iter(|| {
            predict_bvepred(black_box(&mut v), 5, 5, W * 2);
        });
    }

    #[bench]
    fn bench_predict_bldpred(b: &mut Bencher) {
        let mut v = black_box(make_sample_image());

        b.iter(|| {
            black_box(predict_bldpred(black_box(&mut v), 5, 5, W * 2));
        });
    }

    #[bench]
    fn bench_predict_brdpred(b: &mut Bencher) {
        let mut v = black_box(make_sample_image());

        b.iter(|| {
            black_box(predict_brdpred(black_box(&mut v), 5, 5, W * 2));
        });
    }

    #[bench]
    fn bench_predict_bhepred(b: &mut Bencher) {
        let mut v = black_box(make_sample_image());

        b.iter(|| {
            black_box(predict_bhepred(black_box(&mut v), 5, 5, W * 2));
        });
    }

    #[bench]
    fn bench_top_pixels(b: &mut Bencher) {
        let v = black_box(make_sample_image());

        b.iter(|| {
            black_box(top_pixels(black_box(&v), 5, 5, W * 2));
        });
    }

    #[bench]
    fn bench_edge_pixels(b: &mut Bencher) {
        let v = black_box(make_sample_image());

        b.iter(|| {
            black_box(edge_pixels(black_box(&v), 5, 5, W * 2));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg2() {
        for i in 0u8..=255 {
            for j in 0u8..=255 {
                let ceil_avg = (f32::from(i) + f32::from(j)) / 2.0;
                let ceil_avg = ceil_avg.ceil() as u8;
                assert_eq!(
                    ceil_avg,
                    avg2(i, j),
                    "avg2({}, {}), expected {}, got {}.",
                    i,
                    j,
                    ceil_avg,
                    avg2(i, j)
                );
            }
        }
    }

    #[test]
    fn test_avg2_specific() {
        assert_eq!(
            255,
            avg2(255, 255),
            "avg2(255, 255), expected 255, got {}.",
            avg2(255, 255)
        );
        assert_eq!(1, avg2(1, 1), "avg2(1, 1), expected 1, got {}.", avg2(1, 1));
        assert_eq!(2, avg2(2, 1), "avg2(2, 1), expected 2, got {}.", avg2(2, 1));
    }

    #[test]
    fn test_avg3() {
        for i in 0u8..=255 {
            for j in 0u8..=255 {
                for k in 0u8..=255 {
                    let floor_avg =
                        (2.0f32.mul_add(f32::from(j), f32::from(i)) + { f32::from(k) } + 2.0) / 4.0;
                    let floor_avg = floor_avg.floor() as u8;
                    assert_eq!(
                        floor_avg,
                        avg3(i, j, k),
                        "avg3({}, {}, {}), expected {}, got {}.",
                        i,
                        j,
                        k,
                        floor_avg,
                        avg3(i, j, k)
                    );
                }
            }
        }
    }

    #[test]
    fn test_edge_pixels() {
        #[rustfmt::skip]
        let im = vec![5, 6, 7, 8, 9,
                      4, 0, 0, 0, 0,
                      3, 0, 0, 0, 0,
                      2, 0, 0, 0, 0,
                      1, 0, 0, 0, 0];
        let (e0, e1, e2, e3, e4, e5, e6, e7, e8) = edge_pixels(&im, 1, 1, 5);
        assert_eq!(e0, 1);
        assert_eq!(e1, 2);
        assert_eq!(e2, 3);
        assert_eq!(e3, 4);
        assert_eq!(e4, 5);
        assert_eq!(e5, 6);
        assert_eq!(e6, 7);
        assert_eq!(e7, 8);
        assert_eq!(e8, 9);
    }

    #[test]
    fn test_top_pixels() {
        #[rustfmt::skip]
        let im = vec![1, 2, 3, 4, 5, 6, 7, 8,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0];
        let (e0, e1, e2, e3, e4, e5, e6, e7) = top_pixels(&im, 0, 1, 8);
        assert_eq!(e0, 1);
        assert_eq!(e1, 2);
        assert_eq!(e2, 3);
        assert_eq!(e3, 4);
        assert_eq!(e4, 5);
        assert_eq!(e5, 6);
        assert_eq!(e6, 7);
        assert_eq!(e7, 8);
    }

    #[test]
    fn test_predict_bhepred() {
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![5, 0, 0, 0, 0,
              4, 4, 4, 4, 4,
              3, 3, 3, 3, 3,
              2, 2, 2, 2, 2,
              1, 1, 1, 1, 1];

        #[rustfmt::skip]
        let mut im = vec![5, 0, 0, 0, 0,
                      4, 0, 0, 0, 0,
                      3, 0, 0, 0, 0,
                      2, 0, 0, 0, 0,
                      1, 0, 0, 0, 0];
        predict_bhepred(&mut im, 1, 1, 5);
        for (&e, i) in expected.iter().zip(im) {
            assert_eq!(e, i);
        }
    }

    #[test]
    fn test_predict_brdpred() {
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![5, 6, 7, 8, 9,
              4, 5, 6, 7, 8,
              3, 4, 5, 6, 7,
              2, 3, 4, 5, 6,
              1, 2, 3, 4, 5];

        #[rustfmt::skip]
        let mut im = vec![5, 6, 7, 8, 9,
                      4, 0, 0, 0, 0,
                      3, 0, 0, 0, 0,
                      2, 0, 0, 0, 0,
                      1, 0, 0, 0, 0];
        predict_brdpred(&mut im, 1, 1, 5);
        for (&e, i) in expected.iter().zip(im) {
            assert_eq!(e, i);
        }
    }

    #[test]
    fn test_predict_bldpred() {
        #[rustfmt::skip]
        let mut im: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0];
        let avg_1 = 2u8;
        let avg_2 = 3u8;
        let avg_3 = 4u8;
        let avg_4 = 5u8;
        let avg_5 = 6u8;
        let avg_6 = 7u8;
        let avg_7 = 8u8;

        predict_bldpred(&mut im, 0, 1, 8);

        assert_eq!(im[8], avg_1);
        assert_eq!(im[9], avg_2);
        assert_eq!(im[10], avg_3);
        assert_eq!(im[11], avg_4);
        assert_eq!(im[16], avg_2);
        assert_eq!(im[17], avg_3);
        assert_eq!(im[18], avg_4);
        assert_eq!(im[19], avg_5);
        assert_eq!(im[24], avg_3);
        assert_eq!(im[25], avg_4);
        assert_eq!(im[26], avg_5);
        assert_eq!(im[27], avg_6);
        assert_eq!(im[32], avg_4);
        assert_eq!(im[33], avg_5);
        assert_eq!(im[34], avg_6);
        assert_eq!(im[35], avg_7);
    }

    #[test]
    fn test_predict_bvepred() {
        #[rustfmt::skip]
        let mut im: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0];
        let avg_1 = 2u8;
        let avg_2 = 3u8;
        let avg_3 = 4u8;
        let avg_4 = 5u8;

        predict_bvepred(&mut im, 1, 1, 9);

        assert_eq!(im[10], avg_1);
        assert_eq!(im[11], avg_2);
        assert_eq!(im[12], avg_3);
        assert_eq!(im[13], avg_4);
        assert_eq!(im[19], avg_1);
        assert_eq!(im[20], avg_2);
        assert_eq!(im[21], avg_3);
        assert_eq!(im[22], avg_4);
        assert_eq!(im[28], avg_1);
        assert_eq!(im[29], avg_2);
        assert_eq!(im[30], avg_3);
        assert_eq!(im[31], avg_4);
        assert_eq!(im[37], avg_1);
        assert_eq!(im[38], avg_2);
        assert_eq!(im[39], avg_3);
        assert_eq!(im[40], avg_4);
    }
}
