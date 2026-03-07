//! Macroblock iterator for analysis pass.
//!
//! This module provides iteration over macroblocks during the analysis pass,
//! managing boundary samples and YUV import buffers.
//!
//! Ported from libwebp's VP8EncIterator (analysis subset).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::histogram::collect_histogram_with_offset;
use super::prediction::{make_chroma8_preds, make_luma16_preds};
use super::{
    BPS, MAX_INTRA16_MODE, MAX_UV_MODE, PRED_SIZE_ENC, U_OFF_ENC, V_OFF_ENC, VP8_I16_MODE_OFFSETS,
    VP8_UV_MODE_OFFSETS, Y_OFF_ENC, YUV_SIZE_ENC,
};

//------------------------------------------------------------------------------
// Block import helpers

/// Import a block of pixels into BPS-strided work buffer
/// Ported from libwebp's ImportBlock
#[inline]
fn import_block(src: &[u8], src_stride: usize, dst: &mut [u8], w: usize, h: usize, size: usize) {
    for y in 0..h {
        // Copy row using memcpy
        dst[y * BPS..y * BPS + w].copy_from_slice(&src[y * src_stride..y * src_stride + w]);
        // Replicate last pixel if width < size
        if w < size {
            let last_pixel = dst[y * BPS + w - 1];
            for x in w..size {
                dst[y * BPS + x] = last_pixel;
            }
        }
    }
    // Replicate last row if height < size
    if h < size {
        // Copy last valid row to a temp buffer to avoid overlapping borrow
        let last_row_start = (h - 1) * BPS;
        let mut last_row = [0u8; 16]; // max size is 16
        last_row[..size].copy_from_slice(&dst[last_row_start..last_row_start + size]);
        for y in h..size {
            dst[y * BPS..y * BPS + size].copy_from_slice(&last_row[..size]);
        }
    }
}

/// Import vertical line into left border array
#[inline]
fn import_line(src: &[u8], src_stride: usize, dst: &mut [u8], len: usize, total_len: usize) {
    for i in 0..len {
        dst[i] = src[i * src_stride];
    }
    // Replicate last value
    let last = dst[len.saturating_sub(1).max(0)];
    for i in len..total_len {
        dst[i] = last;
    }
}

//------------------------------------------------------------------------------
// Analysis iterator

/// Analysis iterator state
/// Simplified version of libwebp's VP8EncIterator for analysis pass
pub struct AnalysisIterator {
    /// Current macroblock x position
    pub x: usize,
    /// Current macroblock y position
    pub y: usize,
    /// Image width in macroblocks
    pub mb_w: usize,
    /// Image height in macroblocks
    pub mb_h: usize,
    /// Picture width in pixels
    pub width: usize,
    /// Picture height in pixels
    pub height: usize,
    /// YUV input work buffer (BPS stride)
    pub yuv_in: Vec<u8>,
    /// Prediction work buffer (BPS stride)
    pub yuv_p: Vec<u8>,
    /// Left Y boundary samples (17 bytes: [-1] at index 0, [0..15] at indices 1..17)
    pub y_left: [u8; 17],
    /// Left U boundary samples (9 bytes)
    pub u_left: [u8; 9],
    /// Left V boundary samples (9 bytes)
    pub v_left: [u8; 9],
    /// Top Y boundary samples (mb_w * 16 + 4)
    pub y_top: Vec<u8>,
    /// Top UV boundary samples (mb_w * 16, interleaved U[0..8] V[8..16])
    pub uv_top: Vec<u8>,
}

impl AnalysisIterator {
    /// Create a new analysis iterator
    pub fn new(width: usize, height: usize) -> Self {
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);

        Self {
            x: 0,
            y: 0,
            mb_w,
            mb_h,
            width,
            height,
            yuv_in: vec![0u8; YUV_SIZE_ENC],
            yuv_p: vec![0u8; PRED_SIZE_ENC],
            y_left: [129u8; 17],
            u_left: [129u8; 9],
            v_left: [129u8; 9],
            y_top: vec![127u8; mb_w * 16 + 4],
            uv_top: vec![127u8; mb_w * 16],
        }
    }

    /// Reset iterator to start
    pub fn reset(&mut self) {
        self.x = 0;
        self.y = 0;
        self.init_left();
        self.init_top();
    }

    /// Initialize left border to defaults
    fn init_left(&mut self) {
        let corner = if self.y > 0 { 129u8 } else { 127u8 };
        self.y_left[0] = corner;
        self.u_left[0] = corner;
        self.v_left[0] = corner;
        for i in 1..17 {
            self.y_left[i] = 129;
        }
        for i in 1..9 {
            self.u_left[i] = 129;
            self.v_left[i] = 129;
        }
    }

    /// Initialize top border to defaults
    fn init_top(&mut self) {
        self.y_top.fill(127);
        self.uv_top.fill(127);
    }

    /// Set iterator to start of row y
    pub fn set_row(&mut self, y: usize) {
        self.x = 0;
        self.y = y;
        self.init_left();
    }

    /// Import source samples into work buffer
    /// Ported from libwebp's VP8IteratorImport
    pub fn import(
        &mut self,
        y_src: &[u8],
        u_src: &[u8],
        v_src: &[u8],
        y_stride: usize,
        uv_stride: usize,
    ) {
        let x = self.x;
        let y = self.y;

        let y_offset = y * 16 * y_stride + x * 16;
        let uv_offset = y * 8 * uv_stride + x * 8;

        let w = (self.width - x * 16).min(16);
        let h = (self.height - y * 16).min(16);
        let uv_w = w.div_ceil(2);
        let uv_h = h.div_ceil(2);

        // Import Y
        import_block(
            &y_src[y_offset..],
            y_stride,
            &mut self.yuv_in[Y_OFF_ENC..],
            w,
            h,
            16,
        );

        // Import U
        import_block(
            &u_src[uv_offset..],
            uv_stride,
            &mut self.yuv_in[U_OFF_ENC..],
            uv_w,
            uv_h,
            8,
        );

        // Import V
        import_block(
            &v_src[uv_offset..],
            uv_stride,
            &mut self.yuv_in[V_OFF_ENC..],
            uv_w,
            uv_h,
            8,
        );

        // Import boundary samples
        if x == 0 {
            self.init_left();
        } else {
            // Top-left corner
            if y == 0 {
                self.y_left[0] = 127;
                self.u_left[0] = 127;
                self.v_left[0] = 127;
            } else {
                self.y_left[0] = y_src[y_offset - 1 - y_stride];
                self.u_left[0] = u_src[uv_offset - 1 - uv_stride];
                self.v_left[0] = v_src[uv_offset - 1 - uv_stride];
            }

            // Left column
            import_line(
                &y_src[y_offset - 1..],
                y_stride,
                &mut self.y_left[1..],
                h,
                16,
            );
            import_line(
                &u_src[uv_offset - 1..],
                uv_stride,
                &mut self.u_left[1..],
                uv_h,
                8,
            );
            import_line(
                &v_src[uv_offset - 1..],
                uv_stride,
                &mut self.v_left[1..],
                uv_h,
                8,
            );
        }

        // Top row
        if y == 0 {
            // First row: use 127
            for i in 0..16 {
                self.y_top[x * 16 + i] = 127;
            }
            for i in 0..8 {
                self.uv_top[x * 16 + i] = 127;
                self.uv_top[x * 16 + 8 + i] = 127;
            }
        } else {
            // Import from source
            for i in 0..w {
                self.y_top[x * 16 + i] = y_src[y_offset - y_stride + i];
            }
            // Replicate last pixel
            let last_y = self.y_top[x * 16 + w - 1];
            for i in w..16 {
                self.y_top[x * 16 + i] = last_y;
            }

            for i in 0..uv_w {
                self.uv_top[x * 16 + i] = u_src[uv_offset - uv_stride + i];
                self.uv_top[x * 16 + 8 + i] = v_src[uv_offset - uv_stride + i];
            }
            // Replicate
            let last_u = self.uv_top[x * 16 + uv_w - 1];
            let last_v = self.uv_top[x * 16 + 8 + uv_w - 1];
            for i in uv_w..8 {
                self.uv_top[x * 16 + i] = last_u;
                self.uv_top[x * 16 + 8 + i] = last_v;
            }
        }
    }

    /// Advance to next macroblock
    /// Returns true if more macroblocks remain
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> bool {
        self.x += 1;
        if self.x >= self.mb_w {
            self.set_row(self.y + 1);
        }
        self.y < self.mb_h
    }

    /// Check if iteration is done
    pub fn is_done(&self) -> bool {
        self.y >= self.mb_h
    }

    /// Get left boundary for Y with corner at index 0
    /// Returns full y_left array: [0]=corner, [1..17]=left pixels
    #[allow(dead_code)]
    fn get_y_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.y_left[..])
        } else {
            None
        }
    }

    /// Get top boundary for Y
    #[allow(dead_code)]
    fn get_y_top(&self) -> Option<&[u8]> {
        if self.y > 0 {
            Some(&self.y_top[self.x * 16..self.x * 16 + 16])
        } else {
            None
        }
    }

    /// Get left boundary for U with corner at index 0
    /// Returns full u_left array: [0]=corner, [1..9]=left pixels
    #[allow(dead_code)]
    fn get_u_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.u_left[..])
        } else {
            None
        }
    }

    /// Get left boundary for V with corner at index 0
    /// Returns full v_left array: [0]=corner, [1..9]=left pixels
    #[allow(dead_code)]
    fn get_v_left_with_corner(&self) -> Option<&[u8]> {
        if self.x > 0 {
            Some(&self.v_left[..])
        } else {
            None
        }
    }

    /// Get top boundary for UV (interleaved)
    #[allow(dead_code)]
    fn get_uv_top(&self) -> Option<&[u8]> {
        if self.y > 0 {
            Some(&self.uv_top[self.x * 16..self.x * 16 + 16])
        } else {
            None
        }
    }

    /// Analyze best I16 mode and return alpha
    /// Ported from libwebp's MBAnalyzeBestIntra16Mode
    pub fn analyze_best_intra16_mode(&mut self) -> (i32, usize) {
        let mut best_alpha = -1i32;
        let mut best_mode = 0usize;

        // Extract boundary data before mutable borrow
        let has_left = self.x > 0;
        let has_top = self.y > 0;
        let y_left_copy: [u8; 17] = self.y_left;
        let y_top_start = self.x * 16;

        // Generate all predictions
        // Pass copies/slices to avoid borrow conflicts
        let y_left_opt = if has_left {
            Some(&y_left_copy[..])
        } else {
            None
        };
        let y_top_opt = if has_top {
            Some(&self.y_top[y_top_start..y_top_start + 16])
        } else {
            None
        };
        make_luma16_preds(&mut self.yuv_p, y_left_opt, y_top_opt);

        // Test DC (mode 0) and TM (mode 1)
        for mode in 0..MAX_INTRA16_MODE {
            let pred_offset = VP8_I16_MODE_OFFSETS[mode];

            // Collect histogram comparing yuv_in+Y_OFF vs yuv_p+mode_offset
            let histo = collect_histogram_with_offset(
                &self.yuv_in,
                Y_OFF_ENC,
                &self.yuv_p,
                pred_offset,
                0,
                16,
            );

            let alpha = histo.get_alpha();
            if alpha > best_alpha {
                best_alpha = alpha;
                best_mode = mode;
            }
        }

        (best_alpha, best_mode)
    }

    /// Analyze best UV mode and return alpha
    /// Ported from libwebp's MBAnalyzeBestUVMode
    pub fn analyze_best_uv_mode(&mut self) -> (i32, usize) {
        let mut best_alpha = -1i32;
        let mut smallest_alpha = i32::MAX;
        let mut best_mode = 0usize;

        // Extract boundary data before mutable borrow
        let has_left = self.x > 0;
        let has_top = self.y > 0;
        let u_left_copy: [u8; 9] = self.u_left;
        let v_left_copy: [u8; 9] = self.v_left;
        let uv_top_start = self.x * 16;

        // Generate all chroma predictions
        let u_left_opt = if has_left {
            Some(&u_left_copy[..])
        } else {
            None
        };
        let v_left_opt = if has_left {
            Some(&v_left_copy[..])
        } else {
            None
        };
        let uv_top_opt = if has_top {
            Some(&self.uv_top[uv_top_start..uv_top_start + 16])
        } else {
            None
        };
        make_chroma8_preds(&mut self.yuv_p, u_left_opt, v_left_opt, uv_top_opt);

        // Test DC (mode 0) and TM (mode 1)
        for mode in 0..MAX_UV_MODE {
            let pred_offset = VP8_UV_MODE_OFFSETS[mode];

            // Collect histogram for U+V blocks (blocks 16-24 in VP8DspScan)
            let histo = collect_histogram_with_offset(
                &self.yuv_in,
                U_OFF_ENC,
                &self.yuv_p,
                pred_offset,
                16,
                24,
            );

            let alpha = histo.get_alpha();
            if alpha > best_alpha {
                best_alpha = alpha;
            }
            // Best prediction mode is the one with smallest alpha
            if mode == 0 || alpha < smallest_alpha {
                smallest_alpha = alpha;
                best_mode = mode;
            }
        }

        (best_alpha, best_mode)
    }
}
