//! An implementation of the VP8 Video Codec
//!
//! This module contains a partial implementation of the
//! VP8 video format as defined in RFC-6386.
//!
//! It decodes Keyframes only.
//! VP8 is the underpinning of the WebP image format
//!
//! # Related Links
//! * [rfc-6386](http://tools.ietf.org/html/rfc6386) - The VP8 Data Format and Decoding Guide
//! * [VP8.pdf](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37073.pdf) - An overview of of the VP8 format

use byteorder_lite::{LittleEndian, ReadBytesExt};
use std::default::Default;
use std::io::Read;

use crate::decoder::{DecodingError, UpsamplingMethod};
use crate::vp8_common::*;
use crate::vp8_prediction::*;
use crate::yuv;

use super::vp8_arithmetic_decoder::ArithmeticDecoder;
use super::{loop_filter, transform};

#[derive(Clone, Copy)]
pub(crate) struct TreeNode {
    pub left: u8,
    pub right: u8,
    pub prob: Prob,
    pub index: u8,
}

impl TreeNode {
    const UNINIT: TreeNode = TreeNode {
        left: 0,
        right: 0,
        prob: 0,
        index: 0,
    };

    const fn prepare_branch(t: i8) -> u8 {
        if t > 0 {
            (t as u8) / 2
        } else {
            let value = -t;
            0x80 | (value as u8)
        }
    }

    pub(crate) const fn value_from_branch(t: u8) -> i8 {
        (t & !0x80) as i8
    }
}

const fn tree_nodes_from<const N: usize, const M: usize>(
    tree: [i8; N],
    probs: [Prob; M],
) -> [TreeNode; M] {
    if N != 2 * M {
        panic!("invalid tree with probs");
    }
    let mut nodes = [TreeNode::UNINIT; M];
    let mut i = 0;
    while i < M {
        nodes[i].left = TreeNode::prepare_branch(tree[2 * i]);
        nodes[i].right = TreeNode::prepare_branch(tree[2 * i + 1]);
        nodes[i].prob = probs[i];
        nodes[i].index = i as u8;
        i += 1;
    }
    nodes
}

const SEGMENT_TREE_NODE_DEFAULTS: [TreeNode; 3] = tree_nodes_from(SEGMENT_ID_TREE, [255; 3]);

const KEYFRAME_YMODE_NODES: [TreeNode; 4] =
    tree_nodes_from(KEYFRAME_YMODE_TREE, KEYFRAME_YMODE_PROBS);

const KEYFRAME_BPRED_MODE_NODES: [[[TreeNode; 9]; 10]; 10] = {
    let mut output = [[[TreeNode::UNINIT; 9]; 10]; 10];
    let mut i = 0;
    while i < output.len() {
        let mut j = 0;
        while j < output[i].len() {
            output[i][j] =
                tree_nodes_from(KEYFRAME_BPRED_MODE_TREE, KEYFRAME_BPRED_MODE_PROBS[i][j]);
            j += 1;
        }
        i += 1;
    }
    output
};

const KEYFRAME_UV_MODE_NODES: [TreeNode; 3] =
    tree_nodes_from(KEYFRAME_UV_MODE_TREE, KEYFRAME_UV_MODE_PROBS);

type TokenProbTreeNodes = [[[[TreeNode; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];

const COEFF_PROB_NODES: TokenProbTreeNodes = {
    let mut output = [[[[TreeNode::UNINIT; 11]; 3]; 8]; 4];
    let mut i = 0;
    while i < output.len() {
        let mut j = 0;
        while j < output[i].len() {
            let mut k = 0;
            while k < output[i][j].len() {
                output[i][j][k] = tree_nodes_from(DCT_TOKEN_TREE, COEFF_PROBS[i][j][k]);
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    output
};

#[derive(Default, Clone, Copy)]
struct MacroBlock {
    bpred: [IntraMode; 16],
    complexity: [u8; 9],
    luma_mode: LumaMode,
    chroma_mode: ChromaMode,
    segmentid: u8,
    coeffs_skipped: bool,
    non_zero_dct: bool,
}

/// A Representation of the last decoded video frame
#[derive(Default, Debug, Clone)]
pub struct Frame {
    /// The width of the luma plane
    pub width: u16,

    /// The height of the luma plane
    pub height: u16,

    /// The luma plane of the frame
    pub ybuf: Vec<u8>,

    /// The blue plane of the frame
    pub ubuf: Vec<u8>,

    /// The red plane of the frame
    pub vbuf: Vec<u8>,

    pub(crate) version: u8,

    /// Indicates whether this frame is intended for display
    pub for_display: bool,

    // Section 9.2
    /// The pixel type of the frame as defined by Section 9.2
    /// of the VP8 Specification
    pub pixel_type: u8,

    // Section 9.4 and 15
    pub(crate) filter_type: bool, //if true uses simple filter // if false uses normal filter
    pub(crate) filter_level: u8,
    pub(crate) sharpness_level: u8,
}

impl Frame {
    const fn chroma_width(&self) -> u16 {
        self.width.div_ceil(2)
    }

    const fn buffer_width(&self) -> u16 {
        let difference = self.width % 16;
        if difference > 0 {
            self.width + (16 - difference % 16)
        } else {
            self.width
        }
    }

    /// Fills an rgb buffer from the YUV buffers
    pub(crate) fn fill_rgb(&self, buf: &mut [u8], upsampling_method: UpsamplingMethod) {
        const BPP: usize = 3;

        match upsampling_method {
            UpsamplingMethod::Bilinear => {
                yuv::fill_rgb_buffer_fancy::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.height),
                    usize::from(self.buffer_width()),
                );
            }
            UpsamplingMethod::Simple => {
                yuv::fill_rgb_buffer_simple::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.chroma_width()),
                    usize::from(self.buffer_width()),
                );
            }
        }
    }

    /// Fills an rgba buffer from the YUV buffers
    pub(crate) fn fill_rgba(&self, buf: &mut [u8], upsampling_method: UpsamplingMethod) {
        const BPP: usize = 4;

        match upsampling_method {
            UpsamplingMethod::Bilinear => {
                yuv::fill_rgb_buffer_fancy::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.height),
                    usize::from(self.buffer_width()),
                );
            }
            UpsamplingMethod::Simple => {
                yuv::fill_rgb_buffer_simple::<BPP>(
                    buf,
                    &self.ybuf,
                    &self.ubuf,
                    &self.vbuf,
                    usize::from(self.width),
                    usize::from(self.chroma_width()),
                    usize::from(self.buffer_width()),
                );
            }
        }
    }
    /// Gets the buffer size
    #[must_use]
    pub fn get_buf_size(&self) -> usize {
        self.ybuf.len() * 3
    }
}

/// VP8 Decoder
///
/// Only decodes keyframes
pub struct Vp8Decoder<R> {
    r: R,
    b: ArithmeticDecoder,

    mbwidth: u16,
    mbheight: u16,
    macroblocks: Vec<MacroBlock>,

    frame: Frame,

    segments_enabled: bool,
    segments_update_map: bool,
    segment: [Segment; MAX_SEGMENTS],

    loop_filter_adjustments_enabled: bool,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],

    partitions: [ArithmeticDecoder; 8],
    num_partitions: u8,

    segment_tree_nodes: [TreeNode; 3],
    token_probs: Box<TokenProbTreeNodes>,

    // Section 9.11
    prob_skip_false: Option<Prob>,

    top: Vec<MacroBlock>,
    left: MacroBlock,

    // The borders from the previous macroblock, used for predictions
    // See Section 12
    // Note that the left border contains the top left pixel
    top_border_y: Vec<u8>,
    left_border_y: Vec<u8>,

    top_border_u: Vec<u8>,
    left_border_u: Vec<u8>,

    top_border_v: Vec<u8>,
    left_border_v: Vec<u8>,
}

impl<R: Read> Vp8Decoder<R> {
    /// Create a new decoder.
    /// The reader must present a raw vp8 bitstream to the decoder
    fn new(r: R) -> Self {
        let f = Frame::default();
        let s = Segment::default();
        let m = MacroBlock::default();

        Self {
            r,
            b: ArithmeticDecoder::new(),

            mbwidth: 0,
            mbheight: 0,
            macroblocks: Vec::new(),

            frame: f,
            segments_enabled: false,
            segments_update_map: false,
            segment: [s; MAX_SEGMENTS],

            loop_filter_adjustments_enabled: false,
            ref_delta: [0; 4],
            mode_delta: [0; 4],

            partitions: [
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
                ArithmeticDecoder::new(),
            ],

            num_partitions: 1,

            segment_tree_nodes: SEGMENT_TREE_NODE_DEFAULTS,
            token_probs: Box::new(COEFF_PROB_NODES),

            // Section 9.11
            prob_skip_false: None,

            top: Vec::new(),
            left: m,

            top_border_y: Vec::new(),
            left_border_y: Vec::new(),

            top_border_u: Vec::new(),
            left_border_u: Vec::new(),

            top_border_v: Vec::new(),
            left_border_v: Vec::new(),
        }
    }

    fn update_token_probabilities(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if self.b.read_bool(*prob).or_accumulate(&mut res) {
                            let v = self.b.read_literal(8).or_accumulate(&mut res);
                            self.token_probs[i][j][k][t].prob = v;
                        }
                    }
                }
            }
        }
        self.b.check(res, ())
    }

    fn init_partitions(&mut self, n: usize) -> Result<(), DecodingError> {
        if n > 1 {
            let mut sizes = vec![0; 3 * n - 3];
            self.r.read_exact(sizes.as_mut_slice())?;

            for (i, s) in sizes.chunks(3).enumerate() {
                let size = { s }
                    .read_u24::<LittleEndian>()
                    .expect("Reading from &[u8] can't fail and the chunk is complete");

                let size = size as usize;
                let mut buf = vec![[0; 4]; size.div_ceil(4)];
                let bytes: &mut [u8] = buf.as_mut_slice().as_flattened_mut();
                self.r.read_exact(&mut bytes[..size])?;
                self.partitions[i].init(buf, size)?;
            }
        }

        let mut buf = Vec::new();
        self.r.read_to_end(&mut buf)?;
        let size = buf.len();
        let mut chunks = vec![[0; 4]; size.div_ceil(4)];
        chunks.as_mut_slice().as_flattened_mut()[..size].copy_from_slice(&buf);
        self.partitions[n - 1].init(chunks, size)?;

        Ok(())
    }

    fn read_quantization_indices(&mut self) -> Result<(), DecodingError> {
        fn dc_quant(index: i32) -> i16 {
            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let mut res = self.b.start_accumulated_result();

        let yac_abs = self.b.read_literal(7).or_accumulate(&mut res);
        let ydc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2dc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let y2ac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvdc_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);
        let uvac_delta = self.b.read_optional_signed_value(4).or_accumulate(&mut res);

        let n = if self.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for i in 0usize..n {
            let base = i32::from(if self.segments_enabled {
                if self.segment[i].delta_values {
                    i16::from(self.segment[i].quantizer_level) + i16::from(yac_abs)
                } else {
                    i16::from(self.segment[i].quantizer_level)
                }
            } else {
                i16::from(yac_abs)
            });

            self.segment[i].ydc = dc_quant(base + ydc_delta);
            self.segment[i].yac = ac_quant(base);

            self.segment[i].y2dc = dc_quant(base + y2dc_delta) * 2;
            // The intermediate result (max`284*155`) can be larger than the `i16` range.
            self.segment[i].y2ac = (i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16;

            self.segment[i].uvdc = dc_quant(base + uvdc_delta);
            self.segment[i].uvac = ac_quant(base + uvac_delta);

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        self.b.check(res, ())
    }

    fn read_loop_filter_adjustments(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        if self.b.read_flag().or_accumulate(&mut res) {
            for i in 0usize..4 {
                self.ref_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }

            for i in 0usize..4 {
                self.mode_delta[i] = self.b.read_optional_signed_value(6).or_accumulate(&mut res);
            }
        }

        self.b.check(res, ())
    }

    fn read_segment_updates(&mut self) -> Result<(), DecodingError> {
        let mut res = self.b.start_accumulated_result();

        // Section 9.3
        self.segments_update_map = self.b.read_flag().or_accumulate(&mut res);
        let update_segment_feature_data = self.b.read_flag().or_accumulate(&mut res);

        if update_segment_feature_data {
            let segment_feature_mode = self.b.read_flag().or_accumulate(&mut res);

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].quantizer_level =
                    self.b.read_optional_signed_value(7).or_accumulate(&mut res) as i8;
            }

            for i in 0usize..MAX_SEGMENTS {
                self.segment[i].loopfilter_level =
                    self.b.read_optional_signed_value(6).or_accumulate(&mut res) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0usize..3 {
                let update = self.b.read_flag().or_accumulate(&mut res);

                let prob = if update {
                    self.b.read_literal(8).or_accumulate(&mut res)
                } else {
                    255
                };
                self.segment_tree_nodes[i].prob = prob;
            }
        }

        self.b.check(res, ())
    }

    fn read_frame_header(&mut self) -> Result<(), DecodingError> {
        let tag = self.r.read_u24::<LittleEndian>()?;

        let keyframe = tag & 1 == 0;
        if !keyframe {
            return Err(DecodingError::UnsupportedFeature(
                "Non-keyframe frames".to_owned(),
            ));
        }

        self.frame.version = ((tag >> 1) & 7) as u8;
        self.frame.for_display = (tag >> 4) & 1 != 0;

        let first_partition_size = tag >> 5;

        let mut tag = [0u8; 3];
        self.r.read_exact(&mut tag)?;

        if tag != [0x9d, 0x01, 0x2a] {
            return Err(DecodingError::Vp8MagicInvalid(tag));
        }

        let w = self.r.read_u16::<LittleEndian>()?;
        let h = self.r.read_u16::<LittleEndian>()?;

        self.frame.width = w & 0x3FFF;
        self.frame.height = h & 0x3FFF;

        self.mbwidth = self.frame.width.div_ceil(16);
        self.mbheight = self.frame.height.div_ceil(16);

        self.top = vec![MacroBlock::default(); self.mbwidth.into()];
        // Almost always the first macro block, except when non exists (i.e. `width == 0`)
        self.left = self.top.first().copied().unwrap_or_default();

        self.frame.ybuf =
            vec![0u8; usize::from(self.mbwidth) * 16 * usize::from(self.mbheight) * 16];
        self.frame.ubuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];
        self.frame.vbuf = vec![0u8; usize::from(self.mbwidth) * 8 * usize::from(self.mbheight) * 8];

        self.top_border_y = vec![127u8; self.frame.width as usize + 4 + 16];
        self.left_border_y = vec![129u8; 1 + 16];

        // 8 pixels per macroblock
        self.top_border_u = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_u = vec![129u8; 1 + 8];

        self.top_border_v = vec![127u8; 8 * self.mbwidth as usize];
        self.left_border_v = vec![129u8; 1 + 8];

        let size = first_partition_size as usize;
        let mut buf = vec![[0; 4]; size.div_ceil(4)];
        let bytes: &mut [u8] = buf.as_mut_slice().as_flattened_mut();
        self.r.read_exact(&mut bytes[..size])?;

        // initialise binary decoder
        self.b.init(buf, size)?;

        let mut res = self.b.start_accumulated_result();
        let color_space = self.b.read_literal(1).or_accumulate(&mut res);
        self.frame.pixel_type = self.b.read_literal(1).or_accumulate(&mut res);

        if color_space != 0 {
            return Err(DecodingError::ColorSpaceInvalid(color_space));
        }

        self.segments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.segments_enabled {
            self.read_segment_updates()?;
        }

        self.frame.filter_type = self.b.read_flag().or_accumulate(&mut res);
        self.frame.filter_level = self.b.read_literal(6).or_accumulate(&mut res);
        self.frame.sharpness_level = self.b.read_literal(3).or_accumulate(&mut res);

        self.loop_filter_adjustments_enabled = self.b.read_flag().or_accumulate(&mut res);
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments()?;
        }

        let num_partitions = 1 << self.b.read_literal(2).or_accumulate(&mut res) as usize;
        self.b.check(res, ())?;

        self.num_partitions = num_partitions as u8;
        self.init_partitions(num_partitions)?;

        self.read_quantization_indices()?;

        // Refresh entropy probs ?????
        let _ = self.b.read_literal(1);

        self.update_token_probabilities()?;

        let mut res = self.b.start_accumulated_result();
        let mb_no_skip_coeff = self.b.read_literal(1).or_accumulate(&mut res);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(self.b.read_literal(8).or_accumulate(&mut res))
        } else {
            None
        };
        self.b.check(res, ())?;
        Ok(())
    }

    fn read_macroblock_header(&mut self, mbx: usize) -> Result<MacroBlock, DecodingError> {
        let mut mb = MacroBlock::default();
        let mut res = self.b.start_accumulated_result();

        if self.segments_enabled && self.segments_update_map {
            mb.segmentid =
                (self.b.read_with_tree(&self.segment_tree_nodes)).or_accumulate(&mut res) as u8;
        };

        mb.coeffs_skipped = if let Some(prob) = self.prob_skip_false {
            self.b.read_bool(prob).or_accumulate(&mut res)
        } else {
            false
        };

        // intra prediction
        let luma = (self.b.read_with_tree(&KEYFRAME_YMODE_NODES)).or_accumulate(&mut res);
        mb.luma_mode =
            LumaMode::from_i8(luma).ok_or(DecodingError::LumaPredictionModeInvalid(luma))?;

        match mb.luma_mode.into_intra() {
            // `LumaMode::B` - This is predicted individually
            None => {
                for y in 0usize..4 {
                    for x in 0usize..4 {
                        let top = self.top[mbx].bpred[12 + x];
                        let left = self.left.bpred[y];
                        let intra = self.b.read_with_tree(
                            &KEYFRAME_BPRED_MODE_NODES[top as usize][left as usize],
                        );
                        let intra = intra.or_accumulate(&mut res);
                        let bmode = IntraMode::from_i8(intra)
                            .ok_or(DecodingError::IntraPredictionModeInvalid(intra))?;
                        mb.bpred[x + y * 4] = bmode;

                        self.top[mbx].bpred[12 + x] = bmode;
                        self.left.bpred[y] = bmode;
                    }
                }
            }
            Some(mode) => {
                for i in 0usize..4 {
                    mb.bpred[12 + i] = mode;
                    self.left.bpred[i] = mode;
                }
            }
        }

        let chroma = (self.b.read_with_tree(&KEYFRAME_UV_MODE_NODES)).or_accumulate(&mut res);
        mb.chroma_mode = ChromaMode::from_i8(chroma)
            .ok_or(DecodingError::ChromaPredictionModeInvalid(chroma))?;

        self.top[mbx].chroma_mode = mb.chroma_mode;
        self.top[mbx].luma_mode = mb.luma_mode;
        self.top[mbx].bpred = mb.bpred;

        self.b.check(res, mb)
    }

    fn intra_predict_luma(&mut self, mbx: usize, mby: usize, mb: &MacroBlock, resdata: &[i32]) {
        let stride = 1usize + 16 + 4;
        let mw = self.mbwidth as usize;
        let mut ws = create_border_luma(mbx, mby, mw, &self.top_border_y, &self.left_border_y);

        match mb.luma_mode {
            LumaMode::V => predict_vpred(&mut ws, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut ws, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut ws, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut ws, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => predict_4x4(&mut ws, stride, &mb.bpred, resdata),
        }

        if mb.luma_mode != LumaMode::B {
            for y in 0usize..4 {
                for x in 0usize..4 {
                    let i = x + y * 4;
                    // Create a reference to a [i32; 16] array for add_residue (slices of size 16 do not work).
                    let rb: &[i32; 16] = resdata[i * 16..][..16].try_into().unwrap();
                    let y0 = 1 + y * 4;
                    let x0 = 1 + x * 4;

                    add_residue(&mut ws, rb, y0, x0, stride);
                }
            }
        }

        self.left_border_y[0] = ws[16];

        for (i, left) in self.left_border_y[1..][..16].iter_mut().enumerate() {
            *left = ws[(i + 1) * stride + 16];
        }

        for (top, &w) in self.top_border_y[mbx * 16..][..16]
            .iter_mut()
            .zip(&ws[16 * stride + 1..][..16])
        {
            *top = w;
        }

        for y in 0usize..16 {
            for (ybuf, &ws) in self.frame.ybuf[(mby * 16 + y) * mw * 16 + mbx * 16..][..16]
                .iter_mut()
                .zip(ws[(1 + y) * stride + 1..][..16].iter())
            {
                *ybuf = ws;
            }
        }
    }

    fn intra_predict_chroma(&mut self, mbx: usize, mby: usize, mb: &MacroBlock, resdata: &[i32]) {
        let stride = 1usize + 8;

        let mw = self.mbwidth as usize;

        //8x8 with left top border of 1
        let mut uws = create_border_chroma(mbx, mby, &self.top_border_u, &self.left_border_u);
        let mut vws = create_border_chroma(mbx, mby, &self.top_border_v, &self.left_border_v);

        match mb.chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(&mut uws, 8, stride, mby != 0, mbx != 0);
                predict_dcpred(&mut vws, 8, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(&mut uws, 8, 1, 1, stride);
                predict_vpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(&mut uws, 8, 1, 1, stride);
                predict_hpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut uws, 8, 1, 1, stride);
                predict_tmpred(&mut vws, 8, 1, 1, stride);
            }
        }

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = resdata[16 * 16 + i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut uws, urb, y0, x0, stride);

                let vrb: &[i32; 16] = resdata[20 * 16 + i * 16..][..16].try_into().unwrap();

                add_residue(&mut vws, vrb, y0, x0, stride);
            }
        }

        set_chroma_border(&mut self.left_border_u, &mut self.top_border_u, &uws, mbx);
        set_chroma_border(&mut self.left_border_v, &mut self.top_border_v, &vws, mbx);

        for y in 0usize..8 {
            let uv_buf_index = (mby * 8 + y) * mw * 8 + mbx * 8;
            let ws_index = (1 + y) * stride + 1;

            for (((ub, vb), &uw), &vw) in self.frame.ubuf[uv_buf_index..][..8]
                .iter_mut()
                .zip(self.frame.vbuf[uv_buf_index..][..8].iter_mut())
                .zip(uws[ws_index..][..8].iter())
                .zip(vws[ws_index..][..8].iter())
            {
                *ub = uw;
                *vb = vw;
            }
        }
    }

    fn read_coefficients(
        &mut self,
        block: &mut [i32; 16],
        p: usize,
        plane: Plane,
        complexity: usize,
        dcq: i16,
        acq: i16,
    ) -> Result<bool, DecodingError> {
        // perform bounds checks once up front,
        // so that the compiler doesn't have to insert them in the hot loop below
        assert!(complexity <= 2);

        let first_coeff = if plane == Plane::YCoeff1 {
            1usize
        } else {
            0usize
        };
        let probs = &self.token_probs[plane as usize];
        let decoder = &mut self.partitions[p];

        let mut res = decoder.start_accumulated_result();

        let mut complexity = complexity;
        let mut has_coefficients = false;
        let mut skip = false;

        for i in first_coeff..16usize {
            let band = COEFF_BANDS[i] as usize;
            let tree = &probs[band][complexity];

            let token = decoder
                .read_with_tree_with_first_node(tree, tree[skip as usize])
                .or_accumulate(&mut res);

            let mut abs_value = i32::from(match token {
                DCT_EOB => break,

                DCT_0 => {
                    skip = true;
                    has_coefficients = true;
                    complexity = 0;
                    continue;
                }

                literal @ DCT_1..=DCT_4 => i16::from(literal),

                category @ DCT_CAT1..=DCT_CAT6 => {
                    let probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];

                    let mut extra = 0i16;

                    for t in probs.iter().copied() {
                        if t == 0 {
                            break;
                        }
                        let b = decoder.read_bool(t).or_accumulate(&mut res);
                        extra = extra + extra + i16::from(b);
                    }

                    i16::from(DCT_CAT_BASE[(category - DCT_CAT1) as usize]) + extra
                }

                c => panic!("unknown token: {c}"),
            });

            skip = false;

            complexity = if abs_value == 0 {
                0
            } else if abs_value == 1 {
                1
            } else {
                2
            };

            if decoder.read_flag().or_accumulate(&mut res) {
                abs_value = -abs_value;
            }

            let zigzag = ZIGZAG[i] as usize;
            block[zigzag] = abs_value * i32::from(if zigzag > 0 { acq } else { dcq });

            has_coefficients = true;
        }

        decoder.check(res, has_coefficients)
    }

    fn read_residual_data(
        &mut self,
        mb: &mut MacroBlock,
        mbx: usize,
        p: usize,
    ) -> Result<[i32; 384], DecodingError> {
        let sindex = mb.segmentid as usize;
        let mut blocks = [0i32; 384];
        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = self.top[mbx].complexity[0] + self.left.complexity[0];
            let mut block = [0i32; 16];
            let dcq = self.segment[sindex].y2dc;
            let acq = self.segment[sindex].y2ac;
            let n = self.read_coefficients(&mut block, p, plane, complexity as usize, dcq, acq)?;

            self.left.complexity[0] = if n { 1 } else { 0 };
            self.top[mbx].complexity[0] = if n { 1 } else { 0 };

            transform::iwht4x4(&mut block);

            for k in 0usize..16 {
                blocks[16 * k] = block[k];
            }

            plane = Plane::YCoeff1;
        }

        for y in 0usize..4 {
            let mut left = self.left.complexity[y + 1];
            for x in 0usize..4 {
                let i = x + y * 4;
                let block = &mut blocks[i * 16..][..16];
                let block: &mut [i32; 16] = block.try_into().unwrap();

                let complexity = self.top[mbx].complexity[x + 1] + left;
                let dcq = self.segment[sindex].ydc;
                let acq = self.segment[sindex].yac;

                let n = self.read_coefficients(block, p, plane, complexity as usize, dcq, acq)?;

                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    transform::idct4x4(block);
                }

                left = if n { 1 } else { 0 };
                self.top[mbx].complexity[x + 1] = if n { 1 } else { 0 };
            }

            self.left.complexity[y + 1] = left;
        }

        plane = Plane::Chroma;

        for &j in &[5usize, 7usize] {
            for y in 0usize..2 {
                let mut left = self.left.complexity[y + j];

                for x in 0usize..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let block = &mut blocks[i * 16..][..16];
                    let block: &mut [i32; 16] = block.try_into().unwrap();

                    let complexity = self.top[mbx].complexity[x + j] + left;
                    let dcq = self.segment[sindex].uvdc;
                    let acq = self.segment[sindex].uvac;

                    let n =
                        self.read_coefficients(block, p, plane, complexity as usize, dcq, acq)?;
                    if block[0] != 0 || n {
                        mb.non_zero_dct = true;
                        transform::idct4x4(block);
                    }

                    left = if n { 1 } else { 0 };
                    self.top[mbx].complexity[x + j] = if n { 1 } else { 0 };
                }

                self.left.complexity[y + j] = left;
            }
        }

        Ok(blocks)
    }

    /// Does loop filtering on the macroblock
    fn loop_filter(&mut self, mbx: usize, mby: usize, mb: &MacroBlock) {
        let luma_w = self.mbwidth as usize * 16;
        let chroma_w = self.mbwidth as usize * 8;

        let (filter_level, interior_limit, hev_threshold) = self.calculate_filter_parameters(mb);

        if filter_level > 0 {
            let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
            let sub_bedge_limit = (filter_level * 2) + interior_limit;

            // we skip subblock filtering if the coding mode isn't B_PRED and there's no DCT coefficient coded
            let do_subblock_filtering =
                mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped && mb.non_zero_dct);

            //filter across left of macroblock
            if mbx > 0 {
                //simple loop filtering
                if self.frame.filter_type {
                    for y in 0usize..16 {
                        let y0 = mby * 16 + y;
                        let x0 = mbx * 16;

                        loop_filter::simple_segment_horizontal(
                            mbedge_limit,
                            &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                        );
                    }
                } else {
                    for y in 0usize..16 {
                        let y0 = mby * 16 + y;
                        let x0 = mbx * 16;

                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                        );
                    }

                    for y in 0usize..8 {
                        let y0 = mby * 8 + y;
                        let x0 = mbx * 8;

                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ubuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                        loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.vbuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                    }
                }
            }

            //filter across vertical subblocks in macroblock
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for x in (4usize..16 - 1).step_by(4) {
                        for y in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            loop_filter::simple_segment_horizontal(
                                sub_bedge_limit,
                                &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                            );
                        }
                    }
                } else {
                    for x in (4usize..16 - 3).step_by(4) {
                        for y in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            loop_filter::subblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                            );
                        }
                    }

                    for y in 0usize..8 {
                        let y0 = mby * 8 + y;
                        let x0 = mbx * 8 + 4;

                        loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.ubuf[y0 * chroma_w + x0 - 4..][..8],
                        );

                        loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.vbuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                    }
                }
            }

            //filter across top of macroblock
            if mby > 0 {
                if self.frame.filter_type {
                    for x in 0usize..16 {
                        let y0 = mby * 16;
                        let x0 = mbx * 16 + x;

                        loop_filter::simple_segment_vertical(
                            mbedge_limit,
                            &mut self.frame.ybuf[..],
                            y0 * luma_w + x0,
                            luma_w,
                        );
                    }
                } else {
                    //if bottom macroblock, can only filter if there is 3 pixels below
                    for x in 0usize..16 {
                        let y0 = mby * 16;
                        let x0 = mbx * 16 + x;

                        loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ybuf[..],
                            y0 * luma_w + x0,
                            luma_w,
                        );
                    }

                    for x in 0usize..8 {
                        let y0 = mby * 8;
                        let x0 = mbx * 8 + x;

                        loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ubuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                        loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.vbuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                    }
                }
            }

            //filter across horizontal subblock edges within the macroblock
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for y in (4usize..16 - 1).step_by(4) {
                        for x in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            loop_filter::simple_segment_vertical(
                                sub_bedge_limit,
                                &mut self.frame.ybuf[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }
                    }
                } else {
                    for y in (4usize..16 - 3).step_by(4) {
                        for x in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.frame.ybuf[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }
                    }

                    for x in 0..8 {
                        let y0 = mby * 8 + 4;
                        let x0 = mbx * 8 + x;

                        loop_filter::subblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.ubuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );

                        loop_filter::subblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.vbuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                    }
                }
            }
        }
    }

    //return values are the filter level, interior limit and hev threshold
    fn calculate_filter_parameters(&self, macroblock: &MacroBlock) -> (u8, u8, u8) {
        let segment = self.segment[macroblock.segmentid as usize];
        let mut filter_level = i32::from(self.frame.filter_level);

        // if frame level filter level is 0, we must skip loop filter
        if filter_level == 0 {
            return (0, 0, 0);
        }

        if self.segments_enabled {
            if segment.delta_values {
                filter_level += i32::from(segment.loopfilter_level);
            } else {
                filter_level = i32::from(segment.loopfilter_level);
            }
        }

        filter_level = filter_level.clamp(0, 63);

        if self.loop_filter_adjustments_enabled {
            filter_level += self.ref_delta[0];
            if macroblock.luma_mode == LumaMode::B {
                filter_level += self.mode_delta[0];
            }
        }

        let filter_level = filter_level.clamp(0, 63) as u8;

        //interior limit
        let mut interior_limit = filter_level;

        if self.frame.sharpness_level > 0 {
            interior_limit >>= if self.frame.sharpness_level > 4 { 2 } else { 1 };

            if interior_limit > 9 - self.frame.sharpness_level {
                interior_limit = 9 - self.frame.sharpness_level;
            }
        }

        if interior_limit == 0 {
            interior_limit = 1;
        }

        // high edge variance threshold
        let hev_threshold = if filter_level >= 40 {
            2
        } else if filter_level >= 15 {
            1
        } else {
            0
        };

        (filter_level, interior_limit, hev_threshold)
    }

    /// Decodes the current frame
    pub fn decode_frame(r: R) -> Result<Frame, DecodingError> {
        let decoder = Self::new(r);
        decoder.decode_frame_()
    }

    fn decode_frame_(mut self) -> Result<Frame, DecodingError> {
        self.read_frame_header()?;

        for mby in 0..self.mbheight as usize {
            let p = mby % self.num_partitions as usize;
            self.left = MacroBlock::default();

            for mbx in 0..self.mbwidth as usize {
                let mut mb = self.read_macroblock_header(mbx)?;
                let blocks = if !mb.coeffs_skipped {
                    self.read_residual_data(&mut mb, mbx, p)?
                } else {
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        self.top[mbx].complexity[0] = 0;
                    }

                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        self.top[mbx].complexity[i] = 0;
                    }

                    [0i32; 384]
                };

                self.intra_predict_luma(mbx, mby, &mb, &blocks);
                self.intra_predict_chroma(mbx, mby, &mb, &blocks);

                self.macroblocks.push(mb);
            }

            self.left_border_y = vec![129u8; 1 + 16];
            self.left_border_u = vec![129u8; 1 + 8];
            self.left_border_v = vec![129u8; 1 + 8];
        }

        //do loop filtering
        for mby in 0..self.mbheight as usize {
            for mbx in 0..self.mbwidth as usize {
                let mb = self.macroblocks[mby * self.mbwidth as usize + mbx];
                self.loop_filter(mbx, mby, &mb);
            }
        }

        Ok(self.frame)
    }
}

// set border
fn set_chroma_border(
    left_border: &mut [u8],
    top_border: &mut [u8],
    chroma_block: &[u8],
    mbx: usize,
) {
    let stride = 1usize + 8;
    // top left is top right of previous chroma block
    left_border[0] = chroma_block[8];

    // left border
    for (i, left) in left_border[1..][..8].iter_mut().enumerate() {
        *left = chroma_block[(i + 1) * stride + 8];
    }

    for (top, &w) in top_border[mbx * 8..][..8]
        .iter_mut()
        .zip(&chroma_block[8 * stride + 1..][..8])
    {
        *top = w;
    }
}
