//! Quantization parity test: zenwebp vs libwebp's QuantizeBlock_C
//!
//! Verifies that our `quantize_coeff` produces identical results to libwebp's
//! reference implementation when given the same inputs (DCT coefficients,
//! quantization matrix parameters).

use zenwebp::encoder::cost::{MatrixType, QFIX, VP8Matrix, quantdiv};

/// Reference implementation of libwebp's QuantizeBlock_C
/// From libwebp enc.c:688
///
/// ```c
/// coeff = abs(in[j]) + mtx->sharpen[j];
/// if (coeff > mtx->zthresh[j]) {
///     level = QUANTDIV(coeff, iQ, B);
/// ```
fn reference_quantize_coeff(
    coeff: i32,
    _pos: usize,
    q: u16,
    iq: u32,
    bias: u32,
    zthresh: u32,
    sharpen: u16,
) -> (i32, i32) {
    let sign = coeff < 0;
    let abs_coeff = (if sign { -coeff } else { coeff } as u32) + sharpen as u32;
    if abs_coeff <= zthresh {
        return (0, 0); // level=0, dequantized=0
    }
    let level = quantdiv(abs_coeff, iq, bias).min(2047);
    let signed_level = if sign { -level } else { level };
    let dequantized = signed_level * q as i32;
    (signed_level, dequantized)
}

/// Test that quantize_coeff matches the reference for a range of typical coefficient values
#[test]
fn quantize_coeff_matches_reference() {
    // Test across several quality levels (different quantizer steps)
    for q_index in [10, 30, 50, 75, 90, 120] {
        // Use real AC quantizer table values
        let q_dc = zenwebp::encoder::cost::VP8_DC_TABLE[q_index as usize];
        let q_ac = zenwebp::encoder::cost::VP8_AC_TABLE[q_index as usize];

        let matrix = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);

        let mut mismatch_count = 0;
        let mut total_tested = 0;

        // Test coefficient values from -2048 to +2048 at every position
        for pos in 0..16 {
            for coeff in -2048..=2048 {
                total_tested += 1;

                let our_level = matrix.quantize_coeff(coeff, pos);

                let (ref_level, _ref_dequant) = reference_quantize_coeff(
                    coeff,
                    pos,
                    matrix.q[pos],
                    matrix.iq[pos],
                    matrix.bias[pos],
                    matrix.zthresh[pos],
                    matrix.sharpen[pos],
                );

                if our_level != ref_level {
                    mismatch_count += 1;
                    if mismatch_count <= 5 {
                        eprintln!(
                            "MISMATCH q_index={q_index} pos={pos} coeff={coeff}: \
                             ours={our_level} ref={ref_level} \
                             sharpen={} zthresh={}",
                            matrix.sharpen[pos], matrix.zthresh[pos]
                        );
                    }
                }
            }
        }

        assert_eq!(
            mismatch_count, 0,
            "q_index={q_index}: {mismatch_count}/{total_tested} coefficient mismatches"
        );
    }
}

/// Test that the SIMD block quantize produces the same results as quantize_coeff
#[test]
fn block_quantize_matches_scalar() {
    for q_index in [10, 50, 90] {
        let q_dc = zenwebp::encoder::cost::VP8_DC_TABLE[q_index as usize];
        let q_ac = zenwebp::encoder::cost::VP8_AC_TABLE[q_index as usize];

        let matrix = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);

        // Test with various coefficient patterns
        let test_patterns: Vec<[i32; 16]> = vec![
            // Typical DCT output pattern (large DC, decaying AC)
            [
                500, -120, 80, -40, 60, -30, 20, -15, 10, -8, 5, -3, 2, -1, 1, 0,
            ],
            // All zeros
            [0; 16],
            // All same value
            [100; 16],
            // Near-zero values (should trigger zthresh)
            [1, 2, 3, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            // Large negative values
            [
                -800, -400, -200, -100, -50, -25, -12, -6, -3, -2, -1, -1, 0, 0, 0, 0,
            ],
            // Mixed signs, various magnitudes
            [
                -300, 250, -180, 150, -90, 70, -45, 30, -20, 15, -8, 5, -3, 2, -1, 1,
            ],
        ];

        for pattern in &test_patterns {
            let mut block = *pattern;
            let scalar_results: Vec<i32> = (0..16)
                .map(|pos| matrix.quantize_coeff(pattern[pos], pos))
                .collect();

            matrix.quantize(&mut block);

            for pos in 0..16 {
                assert_eq!(
                    block[pos], scalar_results[pos],
                    "Block quantize mismatch at pos={pos}, q_index={q_index}, \
                     coeff={}, block={}, scalar={}",
                    pattern[pos], block[pos], scalar_results[pos]
                );
            }
        }
    }
}

/// Test that quantize_ac_only produces the same as quantize_coeff for AC positions
#[test]
fn ac_only_quantize_matches_scalar() {
    for q_index in [10, 50, 90] {
        let q_dc = zenwebp::encoder::cost::VP8_DC_TABLE[q_index as usize];
        let q_ac = zenwebp::encoder::cost::VP8_AC_TABLE[q_index as usize];

        let matrix = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);

        let test_patterns: Vec<[i32; 16]> = vec![
            [
                500, -120, 80, -40, 60, -30, 20, -15, 10, -8, 5, -3, 2, -1, 1, 0,
            ],
            [0; 16],
            [
                -300, 250, -180, 150, -90, 70, -45, 30, -20, 15, -8, 5, -3, 2, -1, 1,
            ],
        ];

        for pattern in &test_patterns {
            let mut block = *pattern;
            let original_dc = block[0]; // DC should be preserved

            let scalar_results: Vec<i32> = (0..16)
                .map(|pos| {
                    if pos == 0 {
                        pattern[0] // DC unchanged
                    } else {
                        matrix.quantize_coeff(pattern[pos], pos)
                    }
                })
                .collect();

            matrix.quantize_ac_only(&mut block);

            assert_eq!(
                block[0], original_dc,
                "DC was modified by quantize_ac_only!"
            );
            for pos in 1..16 {
                assert_eq!(
                    block[pos], scalar_results[pos],
                    "AC-only quantize mismatch at pos={pos}, q_index={q_index}"
                );
            }
        }
    }
}

/// Verify sharpen values are non-zero for Y1 matrix (luma) at typical quality
/// and zero for Y2/UV matrices
#[test]
fn sharpen_values_correct() {
    // Use Q75 quantizer values (index ~50) where sharpen is meaningful
    let q_dc = zenwebp::encoder::cost::VP8_DC_TABLE[50];
    let q_ac = zenwebp::encoder::cost::VP8_AC_TABLE[50];
    let y1 = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);
    let y2 = VP8Matrix::new(q_dc, q_ac, MatrixType::Y2);
    let uv = VP8Matrix::new(q_dc, q_ac, MatrixType::UV);

    // Y1 should have non-zero sharpen for high-frequency positions
    let y1_sharpen_sum: u32 = y1.sharpen.iter().map(|&s| s as u32).sum();
    assert!(
        y1_sharpen_sum > 0,
        "Y1 matrix should have non-zero sharpen values"
    );

    // Y2 and UV should have zero sharpen
    assert!(
        y2.sharpen.iter().all(|&s| s == 0),
        "Y2 matrix should have zero sharpen"
    );
    assert!(
        uv.sharpen.iter().all(|&s| s == 0),
        "UV matrix should have zero sharpen"
    );
}

/// Verify zthresh values are computed correctly
/// zthresh = ((1 << QFIX) - 1 - bias) / iq
/// Any coefficient with abs_coeff + sharpen <= zthresh should quantize to 0
#[test]
fn zthresh_correctly_computed() {
    for q_index in [10, 30, 50, 75, 90, 120] {
        let q_dc = zenwebp::encoder::cost::VP8_DC_TABLE[q_index as usize];
        let q_ac = zenwebp::encoder::cost::VP8_AC_TABLE[q_index as usize];

        let matrix = VP8Matrix::new(q_dc, q_ac, MatrixType::Y1);

        for pos in 0..16 {
            let zt = matrix.zthresh[pos];
            let iq = matrix.iq[pos];
            let bias = matrix.bias[pos];

            // Verify: quantdiv(zthresh, iq, bias) should be 0
            let level_at_thresh = quantdiv(zt, iq, bias);
            assert_eq!(
                level_at_thresh, 0,
                "pos={pos} q_index={q_index}: zthresh={zt} should give level=0, got {level_at_thresh}"
            );

            // Verify: quantdiv(zthresh+1, iq, bias) should be >= 1
            // (unless zthresh is at maximum, which shouldn't happen for reasonable q values)
            if zt < (1 << QFIX) {
                let level_above = quantdiv(zt + 1, iq, bias);
                assert!(
                    level_above >= 1,
                    "pos={pos} q_index={q_index}: zthresh+1={} should give level>=1, got {level_above}",
                    zt + 1
                );
            }
        }
    }
}
