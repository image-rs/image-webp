use std::io::Write;

use byteorder_lite::{LittleEndian, WriteBytesExt};

use super::arithmetic_encoder::ArithmeticEncoder;
use super::common::*;
use super::prediction::*;
use super::transform;
use super::yuv::convert_image_y;
use super::yuv::convert_image_yuv;
use super::Frame;
use crate::ColorType;
use crate::EncodingError;

// currently in decoder it actually stores this information on the macroblock but that's confusing
// because it doesn't update the macroblock, just the complexity values as we decode
// this is used as the complexity per 13.3 in the decoder
#[derive(Clone, Copy, Default)]
struct Complexity {
    y2: u8,
    y: [u8; 4],
    u: [u8; 2],
    v: [u8; 2],
}

impl Complexity {
    fn clear(&mut self, include_y2: bool) {
        self.y = [0; 4];
        self.u = [0; 2];
        self.v = [0; 2];
        if include_y2 {
            self.y2 = 0;
        }
    }
}

#[derive(Default)]
struct QuantizationIndices {
    yac_abs: u8,
    ydc_delta: Option<i8>,
    y2dc_delta: Option<i8>,
    y2ac_delta: Option<i8>,
    uvdc_delta: Option<i8>,
    uvac_delta: Option<i8>,
}

/// TODO: Consider merging this with the MacroBlock from the decoder
#[derive(Clone, Copy, Default)]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,

    coeffs_skipped: bool,
}

struct Luma16x16Coeffs {
    y2_coeffs: [i32; 16],
    y_coeffs: LumaYCoeffs,
}

type LumaYCoeffs = [i32; 16 * 16];

type ChromaCoeffs = [i32; 16 * 4];

struct Vp8Encoder<W> {
    writer: W,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,

    loop_filter_adjustments: bool,
    macroblock_no_skip_coeff: Option<u8>,
    quantization_indices: QuantizationIndices,

    token_probs: TokenProbTables,

    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    /// Partitions of encoders for the macroblock coefficient data
    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,

    // what the first pass decided for each macroblock, in encoding order
    macroblocks: Vec<MacroblockInfo>,
}

impl<W: Write> Vp8Encoder<W> {
    fn new(writer: W) -> Self {
        let segment = Segment::default();

        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: [segment; MAX_SEGMENTS],
            segments_enabled: false,

            loop_filter_adjustments: false,
            macroblock_no_skip_coeff: None,
            quantization_indices: QuantizationIndices::default(),

            token_probs: Default::default(),

            top_complexity: Vec::new(),
            left_complexity: Complexity::default(),

            top_b_pred: Vec::new(),
            left_b_pred: [IntraMode::default(); 4],

            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],

            left_border_y: [0u8; 16 + 1],
            left_border_u: [0u8; 8 + 1],
            left_border_v: [0u8; 8 + 1],
            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),

            macroblocks: Vec::new(),
        }
    }

    /// Writes the uncompressed part of the frame header (9.1)
    fn write_uncompressed_frame_header(
        &mut self,
        partition_size: u32,
    ) -> Result<(), EncodingError> {
        let version = u32::from(self.frame.version);
        let for_display = if self.frame.for_display { 1 } else { 0 };

        // the first partition size occupies 19 bits of the 24 bit tag, so a
        // larger partition cannot be described by the header at all
        if partition_size >= 1 << 19 {
            return Err(EncodingError::InvalidDimensions);
        }

        let keyframe_bit = 0;
        let tag = (partition_size << 5) | (for_display << 4) | (version << 1) | (keyframe_bit);
        self.writer.write_u24::<LittleEndian>(tag)?;

        let magic_bytes_buffer: [u8; 3] = [0x9d, 0x01, 0x2a];
        self.writer.write_all(&magic_bytes_buffer)?;

        let width = self.frame.width & 0x3FFF;
        let height = self.frame.height & 0x3FFF;
        self.writer.write_u16::<LittleEndian>(width)?;
        self.writer.write_u16::<LittleEndian>(height)?;

        Ok(())
    }

    fn encode_compressed_frame_header(&mut self) {
        // if keyframe, color space must be 0
        self.encoder.write_literal(1, 0);
        // pixel type
        self.encoder.write_literal(1, 0);

        self.encoder.write_flag(self.segments_enabled);
        if self.segments_enabled {
            self.encode_segment_updates();
        }

        self.encoder.write_flag(self.frame.filter_type);
        self.encoder.write_literal(6, self.frame.filter_level);
        self.encoder.write_literal(3, self.frame.sharpness_level);

        self.encoder.write_flag(self.loop_filter_adjustments);
        if self.loop_filter_adjustments {
            self.encode_loop_filter_adjustments();
        }

        // partitions length must be 1, 2, 4 or 8, so value will be 0, 1, 2 or 3
        let partitions_value: u8 = self.partitions.len().ilog2().try_into().unwrap();
        self.encoder.write_literal(2, partitions_value);

        self.encode_quantization_indices();

        // refresh entropy probs
        self.encoder.write_literal(1, 0);

        self.encode_updated_token_probabilities();

        let mb_no_skip_coeff = if self.macroblock_no_skip_coeff.is_some() {
            1
        } else {
            0
        };
        self.encoder.write_literal(1, mb_no_skip_coeff);
        if let Some(prob_skip_false) = self.macroblock_no_skip_coeff {
            self.encoder.write_literal(8, prob_skip_false);
        }
    }

    fn write_partitions(&mut self) -> Result<(), EncodingError> {
        let partitions = std::mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions
            .into_iter()
            .map(|x| x.flush_and_get_buffer())
            .collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer
                    .write_u24::<LittleEndian>(partition.len() as u32)?;
                self.writer.write_all(partition)?;
            }
        }

        // write the final partition
        self.writer
            .write_all(&partitions_bytes[partitions_bytes.len() - 1])?;

        Ok(())
    }

    fn encode_segment_updates(&mut self) {
        // TODO: encode this as per 9.3
        todo!();
    }

    fn encode_loop_filter_adjustments(&mut self) {
        // TODO: encode this
        todo!();
    }

    fn encode_quantization_indices(&mut self) {
        self.encoder
            .write_literal(7, self.quantization_indices.yac_abs);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.ydc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2dc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2ac_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvdc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvac_delta);
    }

    // TODO: work out when we want to update these probabilities
    fn encode_updated_token_probabilities(&mut self) {
        for is in COEFF_UPDATE_PROBS.iter() {
            for js in is.iter() {
                for ks in js.iter() {
                    for prob in ks.iter() {
                        // currently just not updating these
                        self.encoder.write_bool(false, *prob);
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        if self.segments_enabled {
            if let Some(_segment_id) = macroblock_info.segment_id {
                // TODO: set segment for macroblock
                todo!();
            }
        }

        if let Some(prob) = self.macroblock_no_skip_coeff {
            self.encoder
                .write_bool(macroblock_info.coeffs_skipped, prob);
        }

        // encode macroblock info y mode using KEYFRAME_YMODE_TREE
        self.encoder.write_with_tree(
            &KEYFRAME_YMODE_TREE,
            &KEYFRAME_YMODE_PROBS,
            macroblock_info.luma_mode as i8,
        );

        if macroblock_info.luma_mode.into_intra().is_none() {
            // 11.3 code each of the subblocks. The running context each mode
            // is coded against stays local; `advance_bpred_context` below is
            // the only writer of the context the next macroblocks read.
            if let Some(bpred) = macroblock_info.luma_bpred {
                let mut top: [IntraMode; 4] = self.top_b_pred[mbx * 4..][..4].try_into().unwrap();
                for y in 0usize..4 {
                    let mut left = self.left_b_pred[y];
                    for x in 0usize..4 {
                        let probs = &KEYFRAME_BPRED_MODE_PROBS[top[x] as usize][left as usize];
                        let intra_mode = bpred[y * 4 + x];
                        self.encoder.write_with_tree(
                            &KEYFRAME_BPRED_MODE_TREE,
                            probs,
                            intra_mode as i8,
                        );
                        left = intra_mode;
                        top[x] = intra_mode;
                    }
                }
            } else {
                panic!("Invalid, can't set luma mode to B without setting preds");
            }
        }
        self.advance_bpred_context(macroblock_info, mbx);

        // encode macroblock info chroma mode
        self.encoder.write_with_tree(
            &KEYFRAME_UV_MODE_TREE,
            &KEYFRAME_UV_MODE_PROBS,
            macroblock_info.chroma_mode as i8,
        );
    }

    /// Commits the subblock mode context a macroblock leaves behind: the
    /// bottom row and right-hand column of its (effective) subblock modes.
    /// `write_macroblock_header` ends by calling this, and the first pass
    /// calls it directly, so the search context and the coded context cannot
    /// drift apart.
    fn advance_bpred_context(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        match macroblock_info.luma_mode.into_intra() {
            None => {
                let bpred = macroblock_info
                    .luma_bpred
                    .expect("Invalid, can't set luma mode to B without setting preds");
                for index in 0usize..4 {
                    // the bottom row of subblocks is what the macroblock below
                    // predicts against, the right hand column what the one to
                    // the right does
                    self.top_b_pred[mbx * 4 + index] = bpred[3 * 4 + index];
                    self.left_b_pred[index] = bpred[index * 4 + 3];
                }
            }
            Some(intra_mode) => {
                for index in 0usize..4 {
                    self.top_b_pred[mbx * 4 + index] = intra_mode;
                    self.left_b_pred[index] = intra_mode;
                }
            }
        }
    }

    /// What `encode_residual_data` would have spent on a macroblock whose
    /// coefficients are all zero, which is what skipping it saves: one end of
    /// block token per block, coded against the complexities its neighbours
    /// left behind, which are the very complexities a skipped macroblock leaves
    /// behind in turn.
    ///
    /// Reads the complexities the macroblock inherited, so it has to run before
    /// they are cleared.
    fn skipped_macroblock_cost(&self, macroblock_info: &MacroblockInfo, mbx: usize) -> u32 {
        let left = self.left_complexity;
        let top = self.top_complexity[mbx];

        let (luma_plane, first_luma_coeff, mut cost) = if macroblock_info.luma_mode == LumaMode::B {
            (Plane::YCoeff0, 0, 0)
        } else {
            (
                Plane::YCoeff1,
                1,
                self.end_of_block_cost(Plane::Y2, 0, left.y2 + top.y2),
            )
        };

        // every block but those along the top and left edges of the macroblock
        // follows an empty block of the macroblock's own, which leaves the
        // complexity it is coded against at zero
        for (plane, first_coeff, left_complexity, top_complexity) in [
            (luma_plane, first_luma_coeff, &left.y[..], &top.y[..]),
            (Plane::Chroma, 0, &left.u[..], &top.u[..]),
            (Plane::Chroma, 0, &left.v[..], &top.v[..]),
        ] {
            for (y, &left_complexity) in left_complexity.iter().enumerate() {
                for (x, &top_complexity) in top_complexity.iter().enumerate() {
                    let complexity = if x == 0 { left_complexity } else { 0 }
                        + if y == 0 { top_complexity } else { 0 };
                    cost += self.end_of_block_cost(plane, first_coeff, complexity);
                }
            }
        }

        cost
    }

    /// Cost of the end of block token a block with nothing in it is coded with,
    /// in the band the first coefficient it could have carried falls in.
    ///
    /// The token is the first leaf of `DCT_TOKEN_TREE`, so it costs the one
    /// bool at the root of that tree and nothing else.
    fn end_of_block_cost(&self, plane: Plane, first_coeff: usize, complexity: u8) -> u32 {
        let band = usize::from(COEFF_BANDS[first_coeff]);
        let probs = &self.token_probs[plane as usize][band][usize::from(complexity)];
        bit_cost(u32::from(probs[0]))
    }

    /// Whether every coefficient this macroblock would code quantizes to zero,
    /// which is what the skip flag stands for.
    ///
    /// A whole macroblock mode sends the 0th coefficient of each luma block
    /// through the Y2 block and codes the luma blocks from the 1st onwards;
    /// B_PRED has no Y2 block and codes all sixteen coefficients of each luma
    /// block. Both code all eight chroma blocks whole.
    fn macroblock_is_empty(
        &self,
        macroblock_info: &MacroblockInfo,
        y_block_data: &LumaYCoeffs,
        u_block_data: &ChromaCoeffs,
        v_block_data: &ChromaCoeffs,
    ) -> bool {
        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        let first_luma_coeff = if macroblock_info.luma_mode == LumaMode::B {
            0
        } else {
            let mut y2_coeffs = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut y2_coeffs);
            if !block_is_empty(&y2_coeffs, 0, segment.y2dc, segment.y2ac) {
                return false;
            }
            1
        };

        let luma_empty = y_block_data.chunks_exact(16).all(|block| {
            block_is_empty(
                block.try_into().unwrap(),
                first_luma_coeff,
                segment.ydc,
                segment.yac,
            )
        });

        luma_empty
            && u_block_data
                .chunks_exact(16)
                .chain(v_block_data.chunks_exact(16))
                .all(|block| {
                    block_is_empty(block.try_into().unwrap(), 0, segment.uvdc, segment.uvac)
                })
    }

    // 13 in specification, matches read_residual_data in the decoder
    fn encode_residual_data(
        &mut self,
        macroblock_info: &MacroblockInfo,
        partition_index: usize,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        // TODO: change to get index from macroblock
        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);

            // wht here on the 0th coeffs
            transform::wht4x4(&mut coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;

            let has_coeffs = self.encode_coefficients(
                &coeffs0,
                partition_index,
                plane,
                complexity.into(),
                segment.y2dc,
                segment.y2ac,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            // next encode luma coefficients without the 0th coeffs
            plane = Plane::YCoeff1;
        }

        // now encode the 16 luma 4x4 subblocks in the macroblock
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.ydc,
                    segment.yac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            // set for the next macroblock
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

        // encode the 4 u 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.uvdc,
                    segment.uvac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // encode the 4 v 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.uvdc,
                    segment.uvac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    // encodes the coefficients which is the reverse procedure of read_coefficients in the decoder
    // returns whether there was any non-zero data in the block for the complexity
    fn encode_coefficients(
        &mut self,
        block: &[i32; 16],
        partition_index: usize,
        plane: Plane,
        complexity: usize,
        dc_quant: i16,
        ac_quant: i16,
    ) -> bool {
        // transform block
        // dc is used for the 0th coefficient, ac for the others

        let encoder = &mut self.partitions[partition_index];

        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];

        assert!(complexity <= 2);
        let mut complexity = complexity;

        // convert to zigzag and quantize
        // this is the only lossy part of the encoding
        let mut zigzag_block = [0i32; 16];
        for i in first_coeff..16 {
            let zigzag_index = usize::from(ZIGZAG[i]);
            let quant = if zigzag_index > 0 { ac_quant } else { dc_quant };
            zigzag_block[i] = quantize(block[zigzag_index], quant, zigzag_index);
        }

        // get index of last coefficient that isn't 0
        let end_of_block_index =
            if let Some(last_non_zero_index) = zigzag_block.iter().rev().position(|x| *x != 0) {
                (15 - last_non_zero_index) + 1
            } else {
                // if it's all 0s then the first block is end of block
                0
            };

        let mut skip_eob = false;

        for index in first_coeff..end_of_block_index {
            let coeff = zigzag_block[index];

            let band = usize::from(COEFF_BANDS[index]);
            let probabilities = &probs[band][complexity];
            let start_index_token_tree = if skip_eob { 2 } else { 0 };
            let token_tree = &DCT_TOKEN_TREE;
            let token_probs = probabilities;

            let token = match coeff.abs() {
                0 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        DCT_0,
                        start_index_token_tree,
                    );

                    // never going to have an end of block after a 0, so skip checking next coeff
                    skip_eob = true;
                    DCT_0
                }

                // just encode as literal
                literal @ 1..=4 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        literal as i8,
                        start_index_token_tree,
                    );

                    skip_eob = false;
                    literal as i8
                }

                // encode the category
                value => {
                    let category = match value {
                        5..=6 => DCT_CAT1,
                        7..=10 => DCT_CAT2,
                        11..=18 => DCT_CAT3,
                        19..=34 => DCT_CAT4,
                        35..=66 => DCT_CAT5,
                        67..=2048 => DCT_CAT6,
                        _ => unreachable!(),
                    };

                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        category,
                        start_index_token_tree,
                    );

                    let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];

                    let extra = value - i32::from(DCT_CAT_BASE[(category - DCT_CAT1) as usize]);

                    let mut mask = if category == DCT_CAT6 {
                        1 << (11 - 1)
                    } else {
                        1 << (category - DCT_CAT1)
                    };

                    for &prob in category_probs.iter() {
                        if prob == 0 {
                            break;
                        }
                        let extra_bool = extra & mask > 0;
                        encoder.write_bool(extra_bool, prob);
                        mask >>= 1;
                    }

                    skip_eob = false;

                    category
                }
            };

            // encode sign if token is not zero
            if token != DCT_0 {
                // note flag means coeff is negative
                encoder.write_flag(!coeff.is_positive());
            }

            complexity = match token {
                DCT_0 => 0,
                DCT_1 => 1,
                _ => 2,
            };
        }

        // encode end of block
        if end_of_block_index < 16 {
            let band_index = usize::max(first_coeff, end_of_block_index);
            let band = usize::from(COEFF_BANDS[band_index]);
            let probabilities = &probs[band][complexity];
            encoder.write_with_tree(&DCT_TOKEN_TREE, probabilities, DCT_EOB);
        }

        // whether the block has a non zero coefficient
        end_of_block_index > 0
    }

    fn encode_image(
        &mut self,
        data: &[u8],
        color: ColorType,
        width: u16,
        height: u16,
        lossy_quality: u8,
    ) -> Result<(), EncodingError> {
        // `EncoderParams::lossy_quality` is public and nothing narrows it on
        // the way here, so an out-of-range value is a caller error, not a
        // broken invariant.
        if lossy_quality > 100 {
            return Err(EncodingError::InvalidQuality);
        }

        let (y_bytes, u_bytes, v_bytes) = match color {
            ColorType::Rgb8 => convert_image_yuv::<3>(data, width, height),
            ColorType::Rgba8 => convert_image_yuv::<4>(data, width, height),
            ColorType::L8 => convert_image_y::<1>(data, width, height),
            ColorType::La8 => convert_image_y::<2>(data, width, height),
        };

        let bytes_per_pixel = match color {
            ColorType::L8 => 1,
            ColorType::La8 => 2,
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
        };
        assert_eq!(
            (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
            data.len() as u64,
            "width/height doesn't match data length of {} for the color type {:?}",
            data.len(),
            color
        );

        self.setup_encoding(lossy_quality, width, height, y_bytes, u_bytes, v_bytes);

        // The coefficient partitions depend on nothing but each macroblock's
        // own decisions, so they are written as the frame is analysed. The
        // macroblock headers cannot be, because they share a partition with the
        // frame header and follow it, and the frame header carries the skip
        // probability the whole frame decides. So the first pass chooses,
        // reconstructs and writes the coefficients, and the second writes the
        // frame header and then the macroblock headers.
        let mut skipped_macroblocks = 0;
        let mut saved_cost = 0u64;
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            // reset left complexity / bpreds for left of image
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let mut macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                let y_block_data =
                    self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                    mbx.into(),
                    mby.into(),
                    macroblock_info.chroma_mode,
                );

                // a macroblock every coefficient of which quantizes to zero
                // reconstructs to its prediction alone, so the skip flag can
                // code it in one bool instead of an end of block token per
                // block. The reconstruction above is already that prediction,
                // the residual it added being zero.
                macroblock_info.coeffs_skipped = self.macroblock_is_empty(
                    &macroblock_info,
                    &y_block_data,
                    &u_block_data,
                    &v_block_data,
                );

                if macroblock_info.coeffs_skipped {
                    skipped_macroblocks += 1;
                    saved_cost +=
                        u64::from(self.skipped_macroblock_cost(&macroblock_info, mbx.into()));
                    // since coeffs are all zero, need to set all complexities to 0
                    // except if the luma mode is B then won't set Y2
                    self.left_complexity
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                    self.top_complexity[usize::from(mbx)]
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                } else {
                    self.encode_residual_data(
                        &macroblock_info,
                        partition_index,
                        mbx as usize,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                }

                self.advance_bpred_context(&macroblock_info, mbx.into());
                self.macroblocks.push(macroblock_info);
            }
        }

        self.macroblock_no_skip_coeff =
            skip_probability(skipped_macroblocks, self.macroblocks.len(), saved_cost);

        // the flag turned out not to pay for itself, so the macroblocks the
        // first pass left out have to be coded after all
        if self.macroblock_no_skip_coeff.is_none() && skipped_macroblocks > 0 {
            self.rewrite_coefficient_partitions();
        }

        self.encode_compressed_frame_header();

        // the subblock modes are coded against those of the neighbouring
        // subblocks, which the first pass walked through as it searched, so the
        // context starts over for the second pass
        self.top_b_pred.fill(IntraMode::default());
        for mby in 0..usize::from(self.macroblock_height) {
            self.left_b_pred = [IntraMode::default(); 4];

            for mbx in 0..usize::from(self.macroblock_width) {
                let macroblock_info =
                    self.macroblocks[mby * usize::from(self.macroblock_width) + mbx];
                self.write_macroblock_header(&macroblock_info, mbx);
            }
        }

        let compressed_header_encoder = std::mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32)?;

        self.writer.write_all(&compressed_header_bytes)?;

        self.write_partitions()?;

        Ok(())
    }

    /// Codes the coefficients of every macroblock again with nothing skipped,
    /// for the frame that turned out not to want the skip flag.
    ///
    /// The modes are already chosen, so this walks the reconstruction and the
    /// coder again but not the search, from the state `setup_encoding` left.
    fn rewrite_coefficient_partitions(&mut self) {
        for partition in self.partitions.iter_mut() {
            *partition = ArithmeticEncoder::new();
        }
        self.top_complexity = vec![Complexity::default(); usize::from(self.macroblock_width)];
        self.top_border_y = vec![127u8; usize::from(self.macroblock_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(self.macroblock_width) * 8];
        self.top_border_v = vec![127u8; usize::from(self.macroblock_width) * 8];

        for mby in 0..usize::from(self.macroblock_height) {
            let partition_index = mby % self.partitions.len();
            self.left_complexity = Complexity::default();

            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..usize::from(self.macroblock_width) {
                let index = mby * usize::from(self.macroblock_width) + mbx;
                self.macroblocks[index].coeffs_skipped = false;
                let macroblock_info = self.macroblocks[index];

                let y_block_data = self.transform_luma_block(mbx, mby, &macroblock_info);
                let (u_block_data, v_block_data) =
                    self.transform_chroma_blocks(mbx, mby, macroblock_info.chroma_mode);

                self.encode_residual_data(
                    &macroblock_info,
                    partition_index,
                    mbx,
                    &y_block_data,
                    &u_block_data,
                    &v_block_data,
                );
            }
        }
    }

    fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        // the borders were written by the reconstruction of the neighbouring
        // macroblocks, so every candidate is scored against the prediction the
        // decoder will produce for it
        let (luma_mode, luma_bpred) = self.choose_luma_mode(mbx, mby);
        let chroma_mode = self.choose_chroma_mode(mbx, mby);

        MacroblockInfo {
            luma_mode,
            luma_bpred,
            chroma_mode,
            segment_id: None,
            coeffs_skipped: false,
        }
    }

    /// Picks the luma mode with the lowest rate distortion cost, taking each
    /// candidate all the way through the residual pipeline. B_PRED comes with
    /// the subblock modes it was chosen with.
    fn choose_luma_mode(&self, mbx: usize, mby: usize) -> (LumaMode, Option<[IntraMode; 16]>) {
        let segment = self.segments[0];
        let lambda = lambda_for(segment.yac);
        let source_stride = usize::from(self.macroblock_width) * 16;

        let mut best = (LumaMode::DC, u64::MAX);
        for &luma_mode in &LUMA_MODE_CANDIDATES {
            if !predicts_from_reconstructed_pixels(luma_mode as i8, mbx, mby) {
                continue;
            }

            let mut predicted = self.get_predicted_luma_block_16x16(luma_mode, mbx, mby);
            let residual = self.get_luma_blocks_from_predicted_16x16(&predicted, mbx, mby);
            let mut coeffs = self.get_luma_block_coeffs_16x16(residual, &segment);

            // the 0th coefficient of every luma block travels in the Y2 block,
            // so the luma blocks themselves are coded from the 1st onwards
            let mut rate = luma_mode_rate(luma_mode) + coefficient_rate(&coeffs.y2_coeffs, 0);
            for block in coeffs.y_coeffs.chunks_exact(16) {
                rate += coefficient_rate(block.try_into().unwrap(), 1);
            }

            let dequantized = self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs);
            add_residue_blocks(&mut predicted, &dequantized, LUMA_STRIDE, 4);

            let distortion = squared_error(
                &predicted,
                LUMA_STRIDE,
                16,
                &self.frame.ybuf,
                source_stride,
                mbx * 16,
                mby * 16,
            );

            let cost = distortion + rate_cost(rate, lambda);
            if cost < best.1 {
                best = (luma_mode, cost);
            }
        }

        // a whole block mode that already describes the macroblock exactly, and
        // codes nothing but an end of block token per block, leaves nothing for
        // a subblock mode to predict better. Nothing but flat colour reaches
        // here, which on a rendered part is most of the frame, so this is also
        // where most of the search time is saved.
        let nothing_to_code = rate_cost(luma_mode_rate(best.0) + (1 + 16) * ONE_BIT, lambda);
        if best.1 <= nothing_to_code {
            return (best.0, None);
        }

        // B_PRED predicts each 4x4 subblock on its own, which describes an edge
        // running across a macroblock far better than one whole block
        // prediction can, but it pays sixteen subblock modes for it and cannot
        // use the Y2 block.
        //
        // It has to beat the whole block mode by a whole bit rather than merely
        // tie with it. `coefficient_rate` charges a bit for every end of block
        // token where the coder spends a fraction of one on the tokens that end
        // a block with nothing in it, and the whole block modes pay one such
        // token more than B_PRED does, for the Y2 block B_PRED has not got.
        // Without the correction B_PRED wins macroblocks it then spends more
        // bytes on. Ties, and the correction itself, go to the whole block
        // mode, which is the lower prediction mode number.
        let limit = best.1.saturating_sub(rate_cost(ONE_BIT, lambda));
        match self.choose_bpred_modes(mbx, mby, limit) {
            Some((bpred_modes, _)) => (LumaMode::B, Some(bpred_modes)),
            None => (best.0, None),
        }
    }

    /// Searches the sixteen 4x4 subblock modes in the raster order the decoder
    /// reconstructs them in, so that each one is predicted from the pixels its
    /// neighbours will really have, and returns them with their total cost.
    ///
    /// Returns `None` as soon as the running cost can no longer come in under
    /// `limit`, the cost of the best whole macroblock mode, since the cost only
    /// ever grows as more subblocks are added to it.
    fn choose_bpred_modes(
        &self,
        mbx: usize,
        mby: usize,
        limit: u64,
    ) -> Option<([IntraMode; 16], u64)> {
        let lambda = lambda_for(self.segments[0].yac);
        let source_stride = usize::from(self.macroblock_width) * 16;

        let mut workspace = create_border_luma(
            mbx,
            mby,
            self.macroblock_width.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        let mut modes = [IntraMode::default(); 16];
        let mut cost = rate_cost(luma_mode_rate(LumaMode::B), lambda);

        // the subblock modes are coded against the modes of the subblocks above
        // and to the left of each one, so the search tracks the same context
        // `write_macroblock_header` will code them with
        let mut left_modes = self.left_b_pred;
        let mut top_modes: [IntraMode; 4] = self.top_b_pred[mbx * 4..][..4].try_into().unwrap();

        for sby in 0..4 {
            let mut left = left_modes[sby];
            for sbx in 0..4 {
                let probs = &KEYFRAME_BPRED_MODE_PROBS[top_modes[sbx] as usize][left as usize];
                let x0 = sbx * 4 + 1;
                let y0 = sby * 4 + 1;
                let source = (mby * 16 + sby * 4) * source_stride + mbx * 16 + sbx * 4;

                let mut best = (u64::MAX, IntraMode::default(), [0u8; 16]);
                for &mode in &BPRED_MODE_CANDIDATES {
                    if !bpred_predicts_from_reconstructed_pixels(mode, mbx, mby, sbx, sby) {
                        continue;
                    }

                    // whatever the mode predicts, it pays for its own mode and
                    // for ending its block, so a mode that costs more to name
                    // than the best one costs altogether cannot win
                    let mode_rate = tree_rate(&KEYFRAME_BPRED_MODE_TREE, probs, mode as i8);
                    if rate_cost(mode_rate + ONE_BIT, lambda) >= best.0 {
                        continue;
                    }

                    let (cost, reconstruction) =
                        self.score_subblock(&mut workspace, mode, (x0, y0), source, mode_rate);
                    if cost < best.0 {
                        best = (cost, mode, reconstruction);
                    }
                }

                // B_DC_PRED is offered everywhere, so there is always a winner
                let (subblock_cost, mode, reconstruction) = best;
                cost += subblock_cost;
                if cost >= limit {
                    return None;
                }

                // the next subblocks predict from this one, so put the pixels
                // the decoder will hold there into the workspace
                for (y, row) in reconstruction.chunks_exact(4).enumerate() {
                    workspace[(y0 + y) * LUMA_STRIDE + x0..][..4].copy_from_slice(row);
                }

                modes[sby * 4 + sbx] = mode;
                left = mode;
                top_modes[sbx] = mode;
            }
            left_modes[sby] = left;
        }

        Some((modes, cost))
    }

    /// Takes one candidate subblock mode through the residual pipeline and
    /// reports its rate distortion cost along with the pixels the decoder will
    /// reconstruct for it.
    ///
    /// The prediction is left in `workspace`. Every subblock mode reads only
    /// the row above and the column left of the subblock, so overwriting the
    /// subblock itself with one candidate does not disturb the next.
    fn score_subblock(
        &self,
        workspace: &mut [u8; LUMA_BLOCK_SIZE],
        mode: IntraMode,
        (x0, y0): (usize, usize),
        source: usize,
        mode_rate: u32,
    ) -> (u64, [u8; 16]) {
        let segment = self.segments[0];
        let lambda = lambda_for(segment.yac);
        let source_stride = usize::from(self.macroblock_width) * 16;

        predict_bmode(workspace, mode, x0, y0, LUMA_STRIDE);

        let mut block = [0i32; 16];
        for y in 0..4 {
            for x in 0..4 {
                let predicted = workspace[(y0 + y) * LUMA_STRIDE + x0 + x];
                let actual = self.frame.ybuf[source + y * source_stride + x];
                block[y * 4 + x] = i32::from(actual) - i32::from(predicted);
            }
        }
        transform::dct4x4(&mut block);

        // a B_PRED macroblock has no Y2 block, so all sixteen coefficients of
        // every subblock are coded here
        for (index, coeff) in block.iter_mut().enumerate() {
            let quant = if index > 0 { segment.yac } else { segment.ydc };
            *coeff = quantize(*coeff, quant, index);
        }
        let rate = mode_rate + coefficient_rate(&block, 0);

        for (index, coeff) in block.iter_mut().enumerate() {
            let quant = if index > 0 { segment.yac } else { segment.ydc };
            *coeff *= i32::from(quant);
        }
        transform::idct4x4(&mut block);

        let mut reconstruction = [0u8; 16];
        let mut distortion = 0;
        for y in 0..4 {
            for x in 0..4 {
                let predicted = i32::from(workspace[(y0 + y) * LUMA_STRIDE + x0 + x]);
                let value = (block[y * 4 + x] + predicted).clamp(0, 255);
                reconstruction[y * 4 + x] = value as u8;

                let difference = value - i32::from(self.frame.ybuf[source + y * source_stride + x]);
                distortion += u64::from((difference * difference) as u32);
            }
        }

        (distortion + rate_cost(rate, lambda), reconstruction)
    }

    /// Picks the chroma mode, which both chroma planes share, by the summed
    /// rate distortion cost of the two planes.
    fn choose_chroma_mode(&self, mbx: usize, mby: usize) -> ChromaMode {
        let lambda = lambda_for(self.segments[0].uvac);
        let source_stride = usize::from(self.macroblock_width) * 8;

        let mut best = (ChromaMode::DC, u64::MAX);
        for &chroma_mode in &CHROMA_MODE_CANDIDATES {
            if !predicts_from_reconstructed_pixels(chroma_mode as i8, mbx, mby) {
                continue;
            }

            let mut cost = 0;
            for (top_border, left_border, source) in [
                (&self.top_border_u, &self.left_border_u, &self.frame.ubuf),
                (&self.top_border_v, &self.left_border_v, &self.frame.vbuf),
            ] {
                let mut predicted =
                    self.get_predicted_chroma_block(chroma_mode, mbx, mby, top_border, left_border);
                let residual = self.get_chroma_blocks_from_predicted(&predicted, source, mbx, mby);
                let coeffs = self.get_chroma_block_coeffs(residual);

                let mut rate = 0;
                for block in coeffs.chunks_exact(16) {
                    rate += coefficient_rate(block.try_into().unwrap(), 0);
                }

                let dequantized = self.get_dequantized_blocks_from_coeffs_chroma(&coeffs);
                add_residue_blocks(&mut predicted, &dequantized, CHROMA_STRIDE, 2);

                cost += squared_error(
                    &predicted,
                    CHROMA_STRIDE,
                    8,
                    source,
                    source_stride,
                    mbx * 8,
                    mby * 8,
                ) + rate_cost(rate, lambda);
            }
            if cost < best.1 {
                best = (chroma_mode, cost);
            }
        }

        best.0
    }

    // sets up the encoding of the encoder by setting all the encoder params based on the width and height
    fn setup_encoding(
        &mut self,
        lossy_quality: u8,
        width: u16,
        height: u16,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
    ) {
        // `encode_image`, the sole caller, has already rejected qualities
        // above 100; the table lookup would panic on one anyway.
        let quant_index: u8 = QUALITY_TO_QUANTIZER_INDEX[usize::from(lossy_quality)];
        let quant_index_usize: usize = quant_index as usize;

        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        self.macroblock_width = mb_width;
        self.macroblock_height = mb_height;
        self.frame = Frame {
            width,
            height,

            ybuf: y_buf,
            ubuf: u_buf,
            vbuf: v_buf,

            version: 0,

            keyframe: true,
            for_display: true,
            pixel_type: 0,

            filter_type: false,
            filter_level: loop_filter_level(quant_index),
            sharpness_level: 0,
        };

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];
        self.macroblocks.clear();

        self.token_probs = COEFF_PROBS;

        self.segments_enabled = false;
        let quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };
        self.quantization_indices = quantization_indices;

        let segment = Segment {
            ydc: DC_QUANT[quant_index_usize],
            yac: AC_QUANT[quant_index_usize],
            y2dc: DC_QUANT[quant_index_usize] * 2,
            y2ac: ((i32::from(AC_QUANT[quant_index_usize]) * 155 / 100) as i16).max(8),
            uvdc: DC_QUANT[quant_index_usize],
            uvac: AC_QUANT[quant_index_usize],
            ..Default::default()
        };
        self.segments[0] = segment;

        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        self.top_border_y = vec![127u8; usize::from(self.macroblock_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(self.macroblock_width) * 8];
        self.top_border_v = vec![127u8; usize::from(self.macroblock_width) * 8];
    }

    // this is for all the luma modes except B
    fn get_predicted_luma_block_16x16(
        &self,
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
    ) -> [u8; LUMA_BLOCK_SIZE] {
        let stride = LUMA_STRIDE;

        let mbw = self.macroblock_width;

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        // do the prediction
        match luma_mode {
            LumaMode::V => predict_vpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut y_with_border, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => unreachable!(),
        }

        y_with_border
    }

    // gets the luma blocks with the DCT applied to them
    fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let stride = LUMA_STRIDE;
        let width = usize::from(self.macroblock_width * 16);
        let mut luma_blocks = [0i32; 16 * 16];

        for block_y in 0..4 {
            for block_x in 0..4 {
                // the index on the luma block
                let block_index = block_y * 16 * 4 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let y_data_block_index = (mby * 16 + block_y * 4) * width + mbx * 16 + block_x * 4;

                let mut block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_y_block[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        block[y * 4 + x] = i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                // transform block before copying it into main block
                transform::dct4x4(&mut block);

                luma_blocks[block_index..][..16].copy_from_slice(&block);
            }
        }

        luma_blocks
    }

    // converts the predicted y block to the coeffs
    fn get_luma_block_coeffs_16x16(
        &self,
        mut luma_blocks: [i32; 16 * 16],
        segment: &Segment,
    ) -> Luma16x16Coeffs {
        let mut coeffs0 = get_coeffs0_from_block(&luma_blocks);
        // wht transform the y2 block and quantize it
        transform::wht4x4(&mut coeffs0);
        for (index, value) in coeffs0.iter_mut().enumerate() {
            let quant = if index > 0 {
                segment.y2ac
            } else {
                segment.y2dc
            };
            *value = quantize(*value, quant, index);
        }

        // quantize the y blocks
        for y_block in luma_blocks.chunks_exact_mut(16) {
            for (index, y_value) in y_block.iter_mut().enumerate() {
                if index == 0 {
                    *y_value = 0;
                } else {
                    *y_value = quantize(*y_value, segment.yac, index);
                }
            }
        }

        Luma16x16Coeffs {
            y2_coeffs: coeffs0,
            y_coeffs: luma_blocks,
        }
    }

    fn get_dequantized_blocks_from_coeffs_luma_16x16(
        &self,
        coeffs: &mut Luma16x16Coeffs,
    ) -> [i32; 16 * 16] {
        let mut dequantized_luma_residue = [0i32; 16 * 16];
        let segment = self.segments[0];

        for (k, y2_coeff) in coeffs.y2_coeffs.iter_mut().enumerate() {
            let quant = if k > 0 { segment.y2ac } else { segment.y2dc };
            *y2_coeff *= i32::from(quant);
        }
        transform::iwht4x4(&mut coeffs.y2_coeffs);

        // de-quantize the y blocks as well as do the inverse transform
        for (k, luma_block) in coeffs.y_coeffs.chunks_exact_mut(16).enumerate() {
            for y_value in luma_block[1..].iter_mut() {
                *y_value *= i32::from(segment.yac);
            }

            luma_block[0] = coeffs.y2_coeffs[k];

            transform::idct4x4(luma_block);

            dequantized_luma_residue[k * 16..][..16].copy_from_slice(luma_block);
        }

        dequantized_luma_residue
    }

    // Transforms the luma macroblock in the following ways
    // 1. Does the luma prediction and subtracts from the block
    // 2. Converts the block so each 4x4 subblock is contiguous within the block
    // 3. Does the DCT on each subblock
    // 4. Quantizes the block and dequantizes each subblock
    // 5. Calculates the quantized block - this can be used to calculate how accurate the
    // result is and is used to populate the borders for the next macroblock
    fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> [i32; 16 * 16] {
        if macroblock_info.luma_mode == LumaMode::B {
            if let Some(bpred_modes) = macroblock_info.luma_bpred {
                return self.transform_luma_blocks_4x4(bpred_modes, mbx, mby);
            } else {
                panic!("Invalid, need bpred modes for luma mode B");
            }
        }

        let mut y_with_border =
            self.get_predicted_luma_block_16x16(macroblock_info.luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);

        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        // get coeffs
        let mut coeffs = self.get_luma_block_coeffs_16x16(luma_blocks, &segment);

        // now we're essentially applying the same functions as the decoder in order to ensure
        // that the border is the same as the one used for the decoder in the same macroblock
        let dequantized_blocks = self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs);

        // re-use the y_with_border from earlier since the prediction is still valid
        // applies the same thing as the decoder so that the border will line up
        add_residue_blocks(&mut y_with_border, &dequantized_blocks, LUMA_STRIDE, 4);

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * LUMA_STRIDE + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * LUMA_STRIDE + x + 1];
        }

        luma_blocks
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = LUMA_STRIDE;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        let segment = self.segments[0];

        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                predict_bmode(&mut y_with_border, bpred_modes[i], x0, y0, stride);

                let block_index = sby * 16 * 4 + sbx * 16;
                let mut current_subblock = [0i32; 16];

                // subtract actual values here
                let border_subblock_index = y0 * stride + x0;
                let y_data_block_index = (mby * 16 + sby * 4) * width + mbx * 16 + sbx * 4;
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_subblock_index + y * stride + x;
                        let predicted_value = y_with_border[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        current_subblock[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut current_subblock);

                luma_blocks[block_index..][..16].copy_from_slice(&current_subblock);

                // quantize and de-quantize the subblock
                for (index, y_value) in current_subblock.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.yac } else { segment.ydc };
                    *y_value = quantize(*y_value, quant, index) * i32::from(quant);
                }
                transform::idct4x4(&mut current_subblock);
                add_residue(&mut y_with_border, &current_subblock, y0, x0, stride);
            }
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * stride + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * stride + x + 1];
        }

        luma_blocks
    }

    fn get_predicted_chroma_block(
        &self,
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
        top_border: &[u8],
        left_border: &[u8],
    ) -> [u8; CHROMA_BLOCK_SIZE] {
        let mut chroma_with_border = create_border_chroma(mbx, mby, top_border, left_border);

        match chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(
                    &mut chroma_with_border,
                    8,
                    CHROMA_STRIDE,
                    mby != 0,
                    mbx != 0,
                );
            }
            ChromaMode::V => {
                predict_vpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::H => {
                predict_hpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
        }

        chroma_with_border
    }

    fn get_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 4] {
        let mut chroma_blocks = [0i32; 16 * 4];
        let stride = CHROMA_STRIDE;

        let chroma_width = usize::from(self.macroblock_width * 8);

        for block_y in 0..2 {
            for block_x in 0..2 {
                // the index on the chroma block
                let block_index = block_y * 16 * 2 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let chroma_data_block_index =
                    (mby * 8 + block_y * 4) * chroma_width + mbx * 8 + block_x * 4;

                let mut chroma_block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_chroma[predicted_index];
                        let actual_index = chroma_data_block_index + y * chroma_width + x;
                        let actual_value = chroma_data[actual_index];
                        chroma_block[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut chroma_block);

                chroma_blocks[block_index..][..16].copy_from_slice(&chroma_block);
            }
        }

        chroma_blocks
    }

    fn get_chroma_block_coeffs(&self, chroma_blocks: [i32; 16 * 4]) -> ChromaCoeffs {
        let mut chroma_coeffs: ChromaCoeffs = [0i32; 16 * 4];
        let segment = self.segments[0];

        for (block, coeff_block) in chroma_blocks
            .chunks_exact(16)
            .zip(chroma_coeffs.chunks_exact_mut(16))
        {
            for ((index, &value), coeff) in block.iter().enumerate().zip(coeff_block.iter_mut()) {
                let quant = if index > 0 {
                    segment.uvac
                } else {
                    segment.uvdc
                };
                *coeff = quantize(value, quant, index);
            }
        }

        chroma_coeffs
    }

    fn get_dequantized_blocks_from_coeffs_chroma(
        &self,
        chroma_coeffs: &ChromaCoeffs,
    ) -> [i32; 16 * 4] {
        let mut dequantized_blocks = [0i32; 16 * 4];
        let segment = self.segments[0];

        for (coeffs_block, dequant_block) in chroma_coeffs
            .chunks_exact(16)
            .zip(dequantized_blocks.chunks_exact_mut(16))
        {
            for ((index, &coeff), dequant_value) in coeffs_block
                .iter()
                .enumerate()
                .zip(dequant_block.iter_mut())
            {
                let quant = if index > 0 {
                    segment.uvac
                } else {
                    segment.uvdc
                };
                *dequant_value = coeff * i32::from(quant);
            }

            transform::idct4x4(dequant_block);
        }

        dequantized_blocks
    }

    fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        chroma_mode: ChromaMode,
    ) -> ([i32; 16 * 4], [i32; 16 * 4]) {
        let stride = CHROMA_STRIDE;

        let mut predicted_u = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_u,
            &self.left_border_u,
        );
        let mut predicted_v = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_v,
            &self.left_border_v,
        );

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        let u_coeffs = self.get_chroma_block_coeffs(u_blocks);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks);

        let quantized_u_residue = self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs);
        let quantized_v_residue = self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs);

        add_residue_blocks(&mut predicted_u, &quantized_u_residue, stride, 2);
        add_residue_blocks(&mut predicted_v, &quantized_v_residue, stride, 2);

        // set borders
        for ((y, u_border_value), v_border_value) in self
            .left_border_u
            .iter_mut()
            .enumerate()
            .zip(self.left_border_v.iter_mut())
        {
            *u_border_value = predicted_u[y * stride + 8];
            *v_border_value = predicted_v[y * stride + 8];
        }

        for ((x, u_border_value), v_border_value) in self.top_border_u[mbx * 8..][..8]
            .iter_mut()
            .enumerate()
            .zip(self.top_border_v[mbx * 8..][..8].iter_mut())
        {
            *u_border_value = predicted_u[8 * stride + x + 1];
            *v_border_value = predicted_v[8 * stride + x + 1];
        }

        (u_blocks, v_blocks)
    }
}

/// The whole macroblock luma modes, in the order section 11.2 codes them.
/// Ties are broken towards the front of the list, which keeps the output
/// deterministic and prefers the mode the keyframe tree codes most cheaply.
const LUMA_MODE_CANDIDATES: [LumaMode; 4] = [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM];

/// The chroma modes, in the order section 11.2 codes them. VP8 has no per
/// subblock chroma mode, so this is the whole choice for both chroma planes.
const CHROMA_MODE_CANDIDATES: [ChromaMode; 4] =
    [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

/// Whether a candidate mode predicts from pixels a neighbouring macroblock
/// actually reconstructed.
///
/// Above the first macroblock row and left of the first column the border is
/// filled with the fixed 127 and 129 values of section 12.2. A directional mode
/// that copies those is predicting from nothing, and because the quantizer
/// truncates towards zero the resulting error is small enough to quantize away
/// and so is never corrected, after which every macroblock that predicts from
/// this one inherits it. DC is safe either way: it averages whichever real
/// borders exist and falls back to 128 when there are none.
fn predicts_from_reconstructed_pixels(mode: i8, mbx: usize, mby: usize) -> bool {
    match mode {
        V_PRED => mby != 0,
        H_PRED => mbx != 0,
        TM_PRED => mbx != 0 && mby != 0,
        _ => true,
    }
}

/// Weight of one bit against one unit of squared pixel error, so that the mode
/// chooser can trade the two off.
///
/// Rate distortion theory puts the multiplier at the slope of the distortion
/// curve, which for a uniform quantizer of step `quantizer` grows with the
/// square of that step: halving the step costs about a bit per sample and cuts
/// the squared error to a quarter. The constant is the one free parameter. It
/// was calibrated on rendered CAD images, where three eighths is the smallest
/// weight at which no image grew against a DC only encode and the choices stop
/// changing above it. libwebp reaches a similar magnitude from the other
/// direction, weighting with three times the square of the quantizer index
/// rather than of the step.
fn lambda_for(quantizer: i16) -> u64 {
    let quantizer = u64::from(quantizer.unsigned_abs());
    (3 * quantizer * quantizer) >> 3
}

/// Rough number of bits the coefficient coder spends on one quantized block.
///
/// The coefficients are walked in the zigzag order `encode_coefficients` writes
/// them in. Everything up to the last non zero level is coded, so the zeros
/// before it are charged too, a level costs a token that grows with its
/// magnitude plus a sign bit, and the block ends with an end of block token.
fn coefficient_rate(block: &[i32; 16], first_coeff: usize) -> u32 {
    let mut rate = 1;
    let mut pending_zeros = 0;

    for index in first_coeff..16 {
        let level = block[usize::from(ZIGZAG[index])].unsigned_abs();
        if level == 0 {
            pending_zeros += 1;
        } else {
            rate += pending_zeros + 3 + 2 * level.ilog2();
            pending_zeros = 0;
        }
    }

    rate * ONE_BIT
}

/// Rates are counted in eighths of a bit, since the mode syntax costs a
/// fraction of a bit per macroblock and the choice between one prediction mode
/// and another often comes down to nothing else.
const ONE_BIT: u32 = 8;

/// What a rate weighs against squared pixel error, for the given lambda.
fn rate_cost(rate: u32, lambda: u64) -> u64 {
    (u64::from(rate) * lambda) / u64::from(ONE_BIT)
}

/// Rate of the macroblock luma mode itself, which is what B_PRED has to pay
/// sixteen subblock modes out of: the keyframe tree codes it in 0.8 bits
/// against the 2.6 to 3.6 bits it spends on a whole macroblock mode.
fn luma_mode_rate(luma_mode: LumaMode) -> u32 {
    tree_rate(&KEYFRAME_YMODE_TREE, &KEYFRAME_YMODE_PROBS, luma_mode as i8)
}

/// Rate of coding `value` with a tree and its probabilities, by walking from
/// the leaf back to the root the way `write_with_tree` does.
fn tree_rate(tree: &[i8], probs: &[Prob], value: i8) -> u32 {
    let mut rate = 0;
    // the values are encoded as negative or zero in the tree, positive values
    // are indexes
    let mut index = tree.iter().position(|&node| node == -value).unwrap();

    loop {
        // an odd index is the right hand child of its parent, so the bit that
        // reaches it is a one
        let bit = index % 2 == 1;
        index -= usize::from(bit);
        rate += bool_rate(probs[index / 2], bit);

        if index == 0 {
            return rate;
        }
        index = tree
            .iter()
            .position(|&node| node == index as i8)
            .expect("every node of a tree but the root has a parent");
    }
}

/// Rate of coding one bool the coder gives probability `prob / 256` of being
/// false.
fn bool_rate(prob: Prob, bit: bool) -> u32 {
    let odds = if bit {
        256 - u16::from(prob)
    } else {
        u16::from(prob)
    };
    u32::from(BOOL_RATE[usize::from(odds)])
}

/// Rate of a bool of probability `p / 256`, in eighths of a bit, which is
/// `round(-log2(p / 256) * 8)`. A probability of 0 would mean the bool can
/// never be coded and appears in none of the spec's tables, so entry 0 only
/// stands in for entry 1.
#[rustfmt::skip]
const BOOL_RATE: [u8; 256] = [
    64, 64, 56, 51, 48, 45, 43, 42, 40, 39, 37, 36, 35, 34, 34, 33,
    32, 31, 31, 30, 29, 29, 28, 28, 27, 27, 26, 26, 26, 25, 25, 24,
    24, 24, 23, 23, 23, 22, 22, 22, 21, 21, 21, 21, 20, 20, 20, 20,
    19, 19, 19, 19, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 16, 16,
    16, 16, 16, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14,
    13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// The 4x4 subblock modes, in the order section 11.2 numbers them. Ties are
/// broken towards the front of the list, which keeps the output deterministic.
const BPRED_MODE_CANDIDATES: [IntraMode; 10] = [
    IntraMode::DC,
    IntraMode::TM,
    IntraMode::VE,
    IntraMode::HE,
    IntraMode::LD,
    IntraMode::RD,
    IntraMode::VR,
    IntraMode::VL,
    IntraMode::HD,
    IntraMode::HU,
];

/// Whether a subblock mode predicts from pixels something really reconstructed,
/// the same restriction `predicts_from_reconstructed_pixels` puts on the whole
/// macroblock modes, applied a subblock at a time.
///
/// A subblock past the first row of its macroblock has the subblock above it to
/// predict from whatever the macroblock's own position, and likewise to the
/// left. The four pixels above and to the right are the exception: for the
/// right hand column of subblocks `create_border_luma` copies the macroblock
/// above's row down the workspace rather than taking pixels from a macroblock
/// that has not been coded yet, so they come from above the macroblock however
/// far down the subblock sits.
///
/// B_DC_PRED is offered everywhere. It averages both edges whatever they hold,
/// so there is always a candidate to fall back on.
fn bpred_predicts_from_reconstructed_pixels(
    mode: IntraMode,
    mbx: usize,
    mby: usize,
    sbx: usize,
    sby: usize,
) -> bool {
    let above = mby != 0 || sby != 0;
    let left = mbx != 0 || sbx != 0;
    let above_right = if sbx == 3 { mby != 0 } else { above };

    match mode {
        IntraMode::DC => true,
        IntraMode::VE | IntraMode::LD | IntraMode::VL => above && above_right,
        IntraMode::HE | IntraMode::HU => left,
        IntraMode::TM | IntraMode::RD | IntraMode::VR | IntraMode::HD => above && left,
    }
}

/// Adds the dequantized residual of each 4x4 subblock back into a prediction
/// block, turning it into the reconstruction the decoder will produce.
fn add_residue_blocks(plane: &mut [u8], residual: &[i32], stride: usize, blocks_per_side: usize) {
    for y in 0..blocks_per_side {
        for x in 0..blocks_per_side {
            let index = y * blocks_per_side + x;
            // add_residue only takes a fixed size block, slices do not work
            let block: &[i32; 16] = residual[index * 16..][..16].try_into().unwrap();
            add_residue(plane, block, 1 + y * 4, 1 + x * 4, stride);
        }
    }
}

/// Sum of squared error between a reconstructed block and the source pixels.
///
/// `block` is laid out by `create_border_luma` and `create_border_chroma`, so
/// the block itself starts one row down and one column right of the origin. The
/// source planes are padded out to whole macroblocks, so the full `size` by
/// `size` block is always in bounds.
fn squared_error(
    block: &[u8],
    stride: usize,
    size: usize,
    source: &[u8],
    source_stride: usize,
    x0: usize,
    y0: usize,
) -> u64 {
    let mut error = 0u64;

    for y in 0..size {
        let block_row = &block[(y + 1) * stride + 1..][..size];
        let source_row = &source[(y0 + y) * source_stride + x0..][..size];
        for (&block, &source) in block_row.iter().zip(source_row) {
            let difference = i32::from(block) - i32::from(source);
            error += u64::from((difference * difference) as u32);
        }
    }

    error
}

/// The strength section 9.4 asks the decoder to loop filter the frame with,
/// from the quantizer index the frame is coded at.
///
/// The loop filter smooths the edges between blocks, which is worth doing in
/// proportion to how visible those edges are, and that is what the quantizer
/// sets. The encoder used to ask for 63, the most the syntax can express, at
/// every quality; over the rendered corpus that costs 0.15 to 0.22 dB against
/// the level swept out as best, at every quality from 10 to 99.
///
/// Those best levels are 1 at index 2, 10 at 12, 14 at 22, 16 at 26, 20 at 39
/// and 24 at 57: rising with the quantizer, as libwebp's own filter strength
/// does, but flattening as it goes, because this encoder predicts from the
/// unfiltered reconstruction and the filter is only ever a blur over the
/// finished image. Past a point it takes more detail than it takes blocking.
///
/// ```text
/// level(index) = 40 * index / (index + 40)
/// ```
///
/// fits all six to within a level and a hundredth of a dB, gives 0 at the
/// minimal quantizer, where there is nothing to smooth, and approaches 40
/// rather than the 63 that measured worst everywhere.
fn loop_filter_level(quant_index: u8) -> u8 {
    let index = u16::from(quant_index);
    (40 * index / (index + 40)) as u8
}

/// The quantizer index section 9.6 codes the frame with, for each of the 101
/// qualities the encoder takes. More quality, finer quantizer, lower index.
///
/// A quality is not a quantizer. The quantizer tables step in even amounts of
/// error, while what a quality buys is roughly even amounts of file size, and
/// the two are about a cube apart; mapping one straight onto the other, which
/// is what this used to do, spends the whole bottom half of the range on
/// quantizers so coarse that little is left of the image. libwebp's cube root
/// is what undoes that, and it is the shape of the whole lower branch here:
///
/// ```text
/// index(q) = 127 - 104 * cbrt(4 * (q / 100) / 3)
/// ```
///
/// which is libwebp's own `127 * (1 - cbrt(2q / 300))` with its far end moved
/// from the 26 it reaches at quality 75 to 23, the two agreeing at quality 0
/// where the quantizer runs out either way.
///
/// The joint is at 23 because that is where fine detail survives. On a shaded
/// render of a gear, whose teeth are a pixel or two wide against a dark
/// background, index 26 measures 40.56 dB and index 23 measures 45.54: five
/// decibels over three indices, with quality 75 sitting on the wrong side of
/// the cliff. Coarser than 23 the teeth quantize away wholesale.
///
/// Above 75 libwebp's own curve steepens into `cbrt(2 * q / 100 - 1)`, which
/// asks for index 9 at quality 90 where this encoder codes 13. That is 16% more
/// bytes on rendered content, and both ends cannot be had at once. So the top
/// quarter is fitted to this encoder instead, running from the 23 of the joint
/// down to 0 at 100, with the exponent picked to land quality 90 on index 12,
/// the finest quantizer that keeps every render of the corpus inside its own
/// lossless size:
///
/// ```text
/// index(q) = 23 * ((100 - q) / 25).powf(5. / 7.)
/// ```
///
/// Both are rounded to the nearest index and clamped to the 0..=127 the field
/// holds, and the two agree at 75. Quality 100 is the minimal quantizer rather
/// than lossless; the public encoder answers that quality with the lossless
/// coder instead.
#[rustfmt::skip]
const QUALITY_TO_QUANTIZER_INDEX: [u8; 101] = [
    127, 102,  96,  91,  88,  85,  82,  80,  78,  76,
     74,  72,  71,  69,  68,  66,  65,  64,  62,  61,
     60,  59,  58,  57,  56,  55,  54,  53,  52,  51,
     50,  50,  49,  48,  47,  46,  46,  45,  44,  43,
     43,  42,  41,  41,  40,  39,  39,  38,  37,  37,
     36,  36,  35,  34,  34,  33,  33,  32,  32,  31,
     30,  30,  29,  29,  28,  28,  27,  27,  26,  26,
     25,  25,  24,  24,  23,  23,  22,  22,  21,  20,
     20,  19,  18,  17,  17,  16,  15,  14,  14,  13,
     12,  11,  10,   9,   8,   7,   6,   5,   4,   2,
      0,
];

/// How far into a quantizer step an AC coefficient has to reach before it
/// rounds up to the next one, in 256ths of a step. Truncation is 256 and the
/// half step is 128.
///
/// Truncating, which is what the encoder used to do, drops everything short of
/// a whole step and reconstructs every level nearly a step low. Rounding at the
/// exact half step is the other extreme and is worse than either on flat
/// colour, for the reason [`DC_ROUND_UP_AT`] describes. A threshold a little
/// past the half step is what libwebp does, the biases in its own quantization
/// matrices putting it between 141 and 160 of these 256ths. This is where its
/// chroma coefficients round, and it measured best of the seven thresholds
/// swept over the rendered corpus and the fixtures below.
const AC_ROUND_UP_AT: i32 = 146;

/// The same for the DC coefficient of a block, which has to hold out longer.
///
/// The decoder reconstructs a block by dequantizing and then inverse
/// transforming, and the inverse transform rounds a second time, dividing the
/// DC term by eight. A flat surface reconstructed one pixel level out gives its
/// next macroblock a residual of one, which the forward transform scales to a
/// DC coefficient of eight. Round that up to one level of a quantizer of twelve
/// or more and the decoder divides the twelve back by eight and rounds it to
/// two, so the correction overshoots by exactly as much as the error it was
/// meant to remove and the surface lands one level out the other way. It never
/// settles: the macroblock never empties, the skip flag never fires, and a flat
/// image comes out both larger and noisier than it does under plain truncation.
///
/// Twelve is the finest DC quantizer the quality curve reaches where the
/// inverse transform still rounds a level up to two, so eight twelfths is the
/// largest share of a step such a correction has to be refused at, and this is
/// the first threshold past it. Refusing it costs nothing: zero and one are the
/// same distance from the surface, and zero is the one the block can skip.
/// libwebp likewise rounds its DC coefficients later than its AC ones in all
/// three of its matrices.
const DC_ROUND_UP_AT: i32 = 171;

/// Quantizes coefficient `coefficient` of a block, rounding up from its
/// threshold onwards and keeping the sign.
///
/// This is the only place the encoder quantizes. The mode chooser, the
/// reconstruction the borders are taken from, the skip decision and the
/// coefficient coder all have to agree on it exactly, or the encoder predicts
/// from pixels the decoder will not produce.
fn quantize(value: i32, quant: i16, coefficient: usize) -> i32 {
    let round_up_at = if coefficient > 0 {
        AC_ROUND_UP_AT
    } else {
        DC_ROUND_UP_AT
    };
    // `unsigned_abs` and the widening keep the arithmetic total: `abs` alone
    // would panic on `i32::MIN` in debug builds, unreachable as that is for
    // residuals of 8-bit pixels.
    let quant = u64::from(quant.unsigned_abs());
    let magnitude = (u64::from(value.unsigned_abs()) * 256 + (256 - round_up_at) as u64 * quant)
        / (256 * quant);
    let magnitude = magnitude as i32;
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Whether every coefficient `encode_coefficients` would write for this block
/// quantizes to zero, decided by the same rounding division it quantizes with.
fn block_is_empty(block: &[i32; 16], first_coeff: usize, dc_quant: i16, ac_quant: i16) -> bool {
    (first_coeff..16).all(|index| {
        let zigzag_index = usize::from(ZIGZAG[index]);
        let quant = if zigzag_index > 0 { ac_quant } else { dc_quant };
        quantize(block[zigzag_index], quant, zigzag_index) == 0
    })
}

/// The probability the frame header gives a macroblock's skip flag of being
/// false, that is of the macroblock having coefficients, out of 256 as section
/// 9.11 codes it. `None` leaves the flag out of the frame altogether.
///
/// The flag is worth carrying when the end of block tokens the skipped
/// macroblocks no longer code, `saved`, come to more than the skip flags of the
/// whole frame plus the probability itself. A frame where almost nothing skips
/// pays a flag per macroblock for savings only a few of them collect, and is
/// better off coding the empty macroblocks the long way.
fn skip_probability(skipped: usize, macroblocks: usize, saved: u64) -> Option<u8> {
    if skipped == 0 {
        return None;
    }

    // the same probability libwebp derives, which is the proportion of
    // macroblocks that are not skipped, and 0 when they all are
    let probability = (255 * (macroblocks - skipped) / macroblocks) as u8;

    let cost = 8 * u64::from(BIT_COST_ONE_BIT)
        + skipped as u64 * u64::from(bit_cost(256 - u32::from(probability)))
        + (macroblocks - skipped) as u64 * u64::from(bit_cost(u32::from(probability)));

    (saved > cost).then_some(probability)
}

/// The skip flag is weighed in 256ths of a bit rather than the eighths the mode
/// chooser trades against pixel error in, because a frame pays one flag for
/// every macroblock in it against savings of a fraction of a bit each, and a
/// fraction rounded down to nothing would make the flag look free.
const BIT_COST_ONE_BIT: u32 = 256;

/// What the coder spends on a bool it gives probability `probability / 256` of
/// coming up, in 256ths of a bit, which is `-log2(probability / 256)`.
///
/// A probability of 0 stands for the smallest chance the coder can represent,
/// the one part in 256 that costs eight bits.
fn bit_cost(probability: u32) -> u32 {
    let probability = probability.clamp(1, 256);
    let exponent = probability.ilog2();

    // the mantissa of the probability, in [1, 2), as a multiple of 2^-30. Its
    // logarithm comes out a bit at a time: squaring a number in [1, 2) either
    // leaves it there, and that bit of the logarithm is a zero, or takes it
    // past 2, and the bit is a one and halving brings it back into range.
    let mut mantissa = u64::from(probability) << (30 - exponent);
    let mut fraction = 0;
    for bit in (0..8).rev() {
        mantissa = (mantissa * mantissa) >> 30;
        if mantissa >= 2 << 30 {
            fraction |= 1 << bit;
            mantissa >>= 1;
        }
    }

    // -log2(probability / 256) = 8 - log2(probability)
    (8 - exponent) * BIT_COST_ONE_BIT - fraction
}

fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
}

pub(crate) fn encode_frame_lossy<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    lossy_quality: u8,
) -> Result<(), EncodingError> {
    let mut vp8_encoder = Vp8Encoder::new(writer);

    let width = width
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;
    let height = height
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;

    vp8_encoder.encode_image(data, color, width, height, lossy_quality)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUALITY: u8 = 90;

    /// Builds an RGB image from a function of the pixel coordinates.
    fn image(width: usize, height: usize, pixel: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&pixel(x, y));
            }
        }
        data
    }

    /// Flat colour: every candidate predicts the macroblock exactly, so the
    /// tie break has to pick DC, the mode listed first.
    fn flat_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |_, _| [96, 140, 60])
    }

    /// Ramp along the columns: the row above a macroblock is what the block
    /// itself looks like, which is exactly what V copies down.
    fn column_ramp_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, _| [(20 + 3 * x) as u8, 70, 130])
    }

    /// Ramp along the rows, predicted exactly by H copying the left column.
    fn row_ramp_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |_, y| [(20 + 3 * y) as u8, 70, 130])
    }

    /// Separable ramp along both axes. TM reproduces `left + above - corner`
    /// exactly for such a surface, while V and H each keep one axis' error.
    fn diagonal_ramp_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            [(20 + x + y) as u8, 70, (230 - x - y) as u8]
        })
    }

    /// A flat rendered part with dark outlines, the content the chooser exists
    /// for: away from an edge every macroblock is a copy of its neighbours.
    fn outlined_part_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            let inside = (24..232).contains(&x) && (20..172).contains(&y);
            let edge = inside
                && ((24..27).contains(&x)
                    || (229..232).contains(&x)
                    || (20..23).contains(&y)
                    || (169..172).contains(&y));
            match (inside, edge) {
                (_, true) => [30, 30, 34],
                (true, false) => [180, 176, 168],
                (false, _) => [245, 245, 245],
            }
        })
    }

    /// Thin strokes at an angle on white, the wireframe case: an edge crosses
    /// some 4x4 subblocks and misses the rest of the macroblock entirely, which
    /// is what one whole macroblock prediction cannot describe.
    fn line_art_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            if (x + y) % 37 < 2 || (x + height - y) % 53 < 2 {
                [40, 44, 52]
            } else {
                [250, 250, 250]
            }
        })
    }

    /// Blocky glyphs on a flat page: two colours and hard edges, the wordmark
    /// case.
    fn glyph_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            let (gx, gy) = (x % 24, y % 32);
            let stroke = (4..20).contains(&gx)
                && (4..28).contains(&gy)
                && (!(8..16).contains(&gx) || (12..16).contains(&gy));
            if stroke {
                [20, 20, 24]
            } else {
                [252, 252, 250]
            }
        })
    }

    /// A flat part with dark outlines and a hatched face: most macroblocks are
    /// a copy of their neighbours and have nothing to code, the few over a
    /// stroke want subblock modes and do.
    fn hatched_part_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            let inside = (24..width - 24).contains(&x) && (20..height - 20).contains(&y);
            let outline = inside
                && ((24..27).contains(&x)
                    || (width - 27..width - 24).contains(&x)
                    || (20..23).contains(&y)
                    || (height - 23..height - 20).contains(&y));
            let hatch = inside && (x + y) % 61 < 2;
            match (inside, outline || hatch) {
                (_, true) => [30, 30, 34],
                (true, false) => [180, 176, 168],
                (false, _) => [245, 245, 245],
            }
        })
    }

    /// Value noise, the case where every macroblock has coefficients and a skip
    /// flag would be nothing but overhead.
    fn noise_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            let mut state = (x as u32).wrapping_mul(2_654_435_761)
                ^ (y as u32).wrapping_mul(2_246_822_519).rotate_left(13);
            state ^= state >> 15;
            state = state.wrapping_mul(2_654_435_761);
            state ^= state >> 13;
            [state as u8, (state >> 8) as u8, (state >> 16) as u8]
        })
    }

    /// A checker on the 8 pixel grid, which is all hard edges but lines every
    /// 4x4 subblock up with one flat colour.
    fn checker_image(width: usize, height: usize) -> Vec<u8> {
        image(width, height, |x, y| {
            if (x / 8 + y / 8) % 2 == 0 {
                [235, 235, 240]
            } else {
                [45, 50, 60]
            }
        })
    }

    /// Runs the lossy encoder over an image and hands back the encoder, so that
    /// the tests can look at what it decided.
    fn encode_for_inspection(rgb: &[u8], width: u16, height: u16) -> Vp8Encoder<Vec<u8>> {
        let mut encoder = Vp8Encoder::new(Vec::new());
        encoder
            .encode_image(rgb, ColorType::Rgb8, width, height, QUALITY)
            .unwrap();
        encoder
    }

    /// What the chooser picked for each macroblock, in encoding order.
    fn chosen_modes(rgb: &[u8], width: u16, height: u16) -> Vec<MacroblockInfo> {
        encode_for_inspection(rgb, width, height).macroblocks
    }

    /// Asserts the expected modes were picked for every macroblock that has
    /// both a top and a left neighbour. The first macroblock row and column
    /// predict from the 127/129 border fill instead of real pixels, so the
    /// directional modes have nothing meaningful to copy there.
    ///
    /// A luma mode of `None` asks only that the macroblock predicted from a
    /// neighbour at all rather than falling back to DC, which is all that can
    /// be pinned where the plane is so nearly flat that several modes describe
    /// it and the choice between them comes down to their rate.
    fn assert_interior_modes(
        macroblocks: &[MacroblockInfo],
        width: usize,
        expected: (Option<LumaMode>, ChromaMode),
    ) {
        let macroblock_width = width.div_ceil(16);
        assert_eq!(macroblocks.len() % macroblock_width, 0);
        let mut interior = 0;
        for (index, macroblock) in macroblocks.iter().enumerate() {
            let (mbx, mby) = (index % macroblock_width, index / macroblock_width);
            if mbx == 0 || mby == 0 {
                continue;
            }
            match expected.0 {
                Some(luma_mode) => {
                    assert_eq!(macroblock.luma_mode, luma_mode, "macroblock ({mbx}, {mby})")
                }
                None => assert_ne!(
                    macroblock.luma_mode,
                    LumaMode::DC,
                    "macroblock ({mbx}, {mby})"
                ),
            }
            assert_eq!(
                macroblock.chroma_mode, expected.1,
                "macroblock ({mbx}, {mby})"
            );
            interior += 1;
        }
        assert!(interior > 0, "the image has no interior macroblocks");
    }

    /// How many of the macroblocks were coded with per subblock modes.
    fn bpred_macroblocks(macroblocks: &[MacroblockInfo]) -> usize {
        macroblocks
            .iter()
            .filter(|macroblock| macroblock.luma_mode == LumaMode::B)
            .count()
    }

    /// How many of the macroblocks had no coefficients coded for them at all.
    fn skipped_macroblocks(macroblocks: &[MacroblockInfo]) -> usize {
        macroblocks
            .iter()
            .filter(|macroblock| macroblock.coeffs_skipped)
            .count()
    }

    /// Encodes into a WebP container so that libwebp can read the result too.
    fn encode_container(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
        encode_container_at(rgb, width, height, QUALITY)
    }

    /// The same at a quality of the caller's choosing.
    fn encode_container_at(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
        let mut output = Vec::new();
        let mut encoder = crate::WebPEncoder::new(&mut output);
        encoder.set_params(crate::EncoderParams {
            use_lossy: true,
            lossy_quality: quality,
            ..Default::default()
        });
        encoder.encode(rgb, width, height, ColorType::Rgb8).unwrap();
        output
    }

    /// Decodes with this crate and with libwebp, asserts the two agree, and
    /// returns the decoded RGB pixels.
    fn decode_with_both(webp: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut decoder = crate::WebPDecoder::new(std::io::Cursor::new(webp)).unwrap();
        let mut decoded = vec![0u8; width * height * 3];
        decoder.read_image(&mut decoded).unwrap();

        let libwebp = webp::Decoder::new(webp).decode().unwrap();
        assert!(!libwebp.is_alpha());
        assert_eq!(decoded, *libwebp, "image-webp and libwebp disagree");

        decoded
    }

    /// Peak signal to noise ratio in dB over all channels. Only used to check
    /// decoded quality in tests, never in the encoder itself.
    fn psnr(original: &[u8], decoded: &[u8]) -> f64 {
        assert_eq!(original.len(), decoded.len());
        let squared_error: u64 = original
            .iter()
            .zip(decoded)
            .map(|(&a, &b)| {
                let difference = i32::from(a) - i32::from(b);
                (difference * difference) as u64
            })
            .sum();
        if squared_error == 0 {
            return f64::INFINITY;
        }
        let mean = squared_error as f64 / original.len() as f64;
        10.0 * (255.0f64 * 255.0 / mean).log10()
    }

    /// Encodes, checks both decoders agree, and reports the size and quality.
    fn round_trip(rgb: &[u8], width: usize, height: usize) -> (usize, f64) {
        let webp = encode_container(rgb, width as u32, height as u32);
        let decoded = decode_with_both(&webp, width, height);
        (webp.len(), psnr(rgb, &decoded))
    }

    // Each size bound below is what the encoder produced for the same image at
    // the same quality when every macroblock was forced to DC, so a
    // directional mode really has to win for the bound to hold.

    #[test]
    fn flat_colour_picks_dc() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let rgb = flat_image(WIDTH, HEIGHT);

        let modes = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_interior_modes(&modes, WIDTH, (Some(LumaMode::DC), ChromaMode::DC));

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 68, "{bytes} bytes");
        assert!(psnr >= 40.0, "{psnr} dB");
    }

    #[test]
    fn column_ramp_picks_v() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let rgb = column_ramp_image(WIDTH, HEIGHT);

        let modes = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_interior_modes(&modes, WIDTH, (Some(LumaMode::V), ChromaMode::V));

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 254, "{bytes} bytes");
        assert!(psnr >= 40.0, "{psnr} dB");
    }

    #[test]
    fn row_ramp_picks_h() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let rgb = row_ramp_image(WIDTH, HEIGHT);

        let modes = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_interior_modes(&modes, WIDTH, (Some(LumaMode::H), ChromaMode::H));

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 268, "{bytes} bytes");
        assert!(psnr >= 40.0, "{psnr} dB");
    }

    #[test]
    fn diagonal_ramp_picks_tm() {
        // The ramp runs in opposite directions on two of the three channels,
        // which very nearly cancels in the luma and leaves the whole of it in
        // the chroma, so TM is pinned there. The luma of the interior is flat
        // enough that the rounding quantizer leaves several modes with nothing
        // to code and the pick falls to their rate; what stays true is that it
        // is never the DC fallback.
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let rgb = diagonal_ramp_image(WIDTH, HEIGHT);

        let modes = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_interior_modes(&modes, WIDTH, (None, ChromaMode::TM));

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 188, "{bytes} bytes");
        assert!(psnr >= 40.0, "{psnr} dB");
    }

    #[test]
    fn edge_macroblocks_never_predict_from_the_border_fill() {
        // A diagonal ramp is the case that wants TM in every macroblock, and
        // line art the one that wants a subblock mode in most of them, so
        // between them anything reaching for a neighbour that does not exist
        // shows up here.
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;

        for rgb in [
            diagonal_ramp_image(WIDTH, HEIGHT),
            line_art_image(WIDTH, HEIGHT),
        ] {
            let macroblock_width = WIDTH.div_ceil(16);
            for (index, macroblock) in chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16)
                .into_iter()
                .enumerate()
            {
                let (mbx, mby) = (index % macroblock_width, index / macroblock_width);
                let (luma_mode, chroma_mode) = (macroblock.luma_mode, macroblock.chroma_mode);

                assert!(
                    predicts_from_reconstructed_pixels(luma_mode as i8, mbx, mby),
                    "{luma_mode:?} at macroblock ({mbx}, {mby})"
                );
                assert!(
                    predicts_from_reconstructed_pixels(chroma_mode as i8, mbx, mby),
                    "{chroma_mode:?} at macroblock ({mbx}, {mby})"
                );

                for (subblock, mode) in macroblock.luma_bpred.iter().flatten().enumerate() {
                    let (sbx, sby) = (subblock % 4, subblock / 4);
                    assert!(
                        bpred_predicts_from_reconstructed_pixels(*mode, mbx, mby, sbx, sby),
                        "{mode:?} at subblock ({sbx}, {sby}) of macroblock ({mbx}, {mby})"
                    );
                }
            }
        }
    }

    #[test]
    fn line_art_picks_subblock_modes() {
        // Strokes thinner than a macroblock: a subblock the stroke misses is
        // flat, and one it crosses needs a direction of its own.
        const WIDTH: usize = 128;
        const HEIGHT: usize = 96;
        let rgb = line_art_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert!(
            bpred_macroblocks(&macroblocks) * 2 > macroblocks.len(),
            "{} of {} macroblocks",
            bpred_macroblocks(&macroblocks),
            macroblocks.len()
        );

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 7250, "{bytes} bytes");
        assert!(psnr >= 37.27, "{psnr} dB");
    }

    #[test]
    fn glyphs_pick_subblock_modes() {
        // Flat colour with hard edges, the case the whole macroblock modes
        // leave a step in the residual of.
        const WIDTH: usize = 240;
        const HEIGHT: usize = 128;
        let rgb = glyph_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert!(
            bpred_macroblocks(&macroblocks) * 2 > macroblocks.len(),
            "{} of {} macroblocks",
            bpred_macroblocks(&macroblocks),
            macroblocks.len()
        );

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 4810, "{bytes} bytes");
        assert!(psnr >= 43.21, "{psnr} dB");
    }

    #[test]
    fn odd_dimensions_go_through_the_subblock_search() {
        // The last macroblock of each row and column is mostly padding, which
        // replicates the edge pixels, and the subblocks over it still have to
        // predict from the same workspace as the rest.
        const WIDTH: usize = 61;
        const HEIGHT: usize = 45;
        let rgb = line_art_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert!(bpred_macroblocks(&macroblocks) > 0);

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 1694, "{bytes} bytes");
        assert!(psnr >= 37.24, "{psnr} dB");
    }

    #[test]
    fn a_checker_on_the_subblock_grid_keeps_the_whole_macroblock_mode() {
        // Every 4x4 subblock of this one is flat, so the whole macroblock
        // residual is one value per subblock, which is exactly what the Y2
        // block carries and what B_PRED has to give up to get subblock modes.
        // The chooser only picks B_PRED where it pays for itself.
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        let rgb = checker_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_eq!(bpred_macroblocks(&macroblocks), 0);

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 596, "{bytes} bytes");
        assert!(psnr >= 41.26, "{psnr} dB");
    }

    #[test]
    fn the_subblock_modes_cost_less_than_a_whole_macroblock_mode() {
        // The keyframe tree spends the fewest bits of all on B_PRED, which is
        // most of what pays for the sixteen subblock modes it then codes.
        for luma_mode in LUMA_MODE_CANDIDATES {
            assert!(
                luma_mode_rate(LumaMode::B) < luma_mode_rate(luma_mode),
                "{luma_mode:?}"
            );
        }

        // An even bool costs a bit, one that always comes up the same way costs
        // nothing, and one that never does costs the eight bits the coder's
        // probabilities bottom out at.
        assert_eq!(bool_rate(128, false), ONE_BIT);
        assert_eq!(bool_rate(255, false), 0);
        assert_eq!(bool_rate(255, true), 8 * ONE_BIT);

        // B_DC_PRED is the first branch of the subblock tree, so it costs that
        // one bool where every other mode costs it and more.
        let probs = &KEYFRAME_BPRED_MODE_PROBS[IntraMode::DC as usize][IntraMode::DC as usize];
        let rates: Vec<u32> = BPRED_MODE_CANDIDATES
            .iter()
            .map(|&mode| tree_rate(&KEYFRAME_BPRED_MODE_TREE, probs, mode as i8))
            .collect();
        assert_eq!(rates[0], bool_rate(probs[0], false));
        assert!(rates[1..].iter().all(|&rate| rate > rates[0]), "{rates:?}");
    }

    #[test]
    fn odd_dimensions_go_through_the_chooser() {
        // The macroblock padding replicates the edge pixels, so the ramp
        // continues into it and the directional modes still win there.
        const WIDTH: usize = 37;
        const HEIGHT: usize = 23;
        let rgb = diagonal_ramp_image(WIDTH, HEIGHT);

        let modes = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_interior_modes(&modes, WIDTH, (None, ChromaMode::TM));

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 104, "{bytes} bytes");
        assert!(psnr >= 40.0, "{psnr} dB");
    }

    #[test]
    fn outlined_part_beats_dc_only() {
        // The content the mode chooser exists for: a flat rendered part with
        // dark outlines, where every macroblock away from an edge is a copy of
        // its neighbours. Both DC and the directional modes get picked here.
        const WIDTH: usize = 256;
        const HEIGHT: usize = 192;
        let rgb = outlined_part_image(WIDTH, HEIGHT);

        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes < 2514, "{bytes} bytes");
        assert!(psnr >= 41.85, "{psnr} dB");
    }

    // The size bounds below are what the encoder produced for the same image at
    // the same quality before the skip flag existed.

    #[test]
    fn a_flat_part_skips_most_of_its_macroblocks() {
        // Away from the outline every macroblock is a copy of its neighbours,
        // so its prediction is exact and there is nothing left to code.
        const WIDTH: usize = 256;
        const HEIGHT: usize = 192;
        let rgb = outlined_part_image(WIDTH, HEIGHT);

        let encoder = encode_for_inspection(&rgb, WIDTH as u16, HEIGHT as u16);
        let skipped = skipped_macroblocks(&encoder.macroblocks);
        assert!(
            skipped * 4 > encoder.macroblocks.len() * 3,
            "{skipped} of {} macroblocks",
            encoder.macroblocks.len()
        );
        assert!(encoder.macroblock_no_skip_coeff.is_some());

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 236, "{bytes} bytes");
        assert!(psnr >= 43.78, "{psnr} dB");
    }

    #[test]
    fn a_hatched_part_skips_only_the_macroblocks_between_the_strokes() {
        // The macroblocks a stroke crosses have coefficients and are coded with
        // subblock modes; the ones between the strokes are flat and skip. A
        // subblock coded macroblock has no Y2 block, so skipping it leaves the
        // Y2 complexity of its neighbours alone where skipping a whole block
        // coded one clears it, which is what section 13.3 asks for and what the
        // decoder does.
        const WIDTH: usize = 256;
        const HEIGHT: usize = 192;
        let rgb = hatched_part_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        let skipped = skipped_macroblocks(&macroblocks);
        assert!(skipped > 0 && skipped < macroblocks.len(), "{skipped}");
        assert!(macroblocks
            .iter()
            .any(|macroblock| macroblock.coeffs_skipped && macroblock.luma_mode == LumaMode::B));

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 3700, "{bytes} bytes");
        assert!(psnr >= 41.33, "{psnr} dB");
    }

    #[test]
    fn odd_dimensions_skip_the_padded_macroblocks() {
        // The last macroblock of each row and column is mostly padding, which
        // replicates the edge pixels, so out at the flat border of the image
        // they are as empty as the macroblocks beside them and skip too.
        const WIDTH: usize = 245;
        const HEIGHT: usize = 179;
        let rgb = hatched_part_image(WIDTH, HEIGHT);

        let macroblocks = chosen_modes(&rgb, WIDTH as u16, HEIGHT as u16);
        let macroblock_width = WIDTH.div_ceil(16);
        let macroblock_height = HEIGHT.div_ceil(16);
        assert_eq!(macroblocks.len(), macroblock_width * macroblock_height);
        assert!(macroblocks[macroblock_width - 1].coeffs_skipped);
        assert!(macroblocks[(macroblock_height - 1) * macroblock_width].coeffs_skipped);
        assert!(macroblocks[macroblocks.len() - 1].coeffs_skipped);

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 3252, "{bytes} bytes");
        assert!(psnr >= 41.35, "{psnr} dB");
    }

    #[test]
    fn noise_leaves_the_skip_flag_out_of_the_frame() {
        // Every macroblock of this one has coefficients, so a skip flag per
        // macroblock would buy the frame nothing at all.
        const WIDTH: usize = 128;
        const HEIGHT: usize = 128;
        let rgb = noise_image(WIDTH, HEIGHT);

        let encoder = encode_for_inspection(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_eq!(skipped_macroblocks(&encoder.macroblocks), 0);
        assert_eq!(encoder.macroblock_no_skip_coeff, None);

        // re-measured for the quality curve
        let (bytes, _) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 25442, "{bytes} bytes");
    }

    #[test]
    fn a_frame_the_flag_would_cost_more_than_it_saves_codes_every_macroblock() {
        // Six of the sixteen macroblocks of this ramp are empty, but their end
        // of block tokens come to less than sixteen skip flags and the
        // probability that codes them, so the frame leaves the flag out and
        // codes the empty macroblocks the long way after all.
        const WIDTH: usize = 64;
        const HEIGHT: usize = 64;
        let rgb = row_ramp_image(WIDTH, HEIGHT);

        let encoder = encode_for_inspection(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_eq!(encoder.macroblock_no_skip_coeff, None);
        assert_eq!(skipped_macroblocks(&encoder.macroblocks), 0);

        // re-measured for the quality curve
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 142, "{bytes} bytes");
        assert!(psnr >= 42.35, "{psnr} dB");
    }

    #[test]
    fn a_frame_that_skips_all_but_one_macroblock_still_codes_a_probability() {
        // The first macroblock of the frame has nothing but the 127 and 129
        // border fill to predict from, so it always has something to code, and
        // over a large enough frame the proportion that has rounds down to
        // zero. The coder can be given that probability, it only means the
        // macroblock that does have coefficients pays eight bits to say so.
        const WIDTH: usize = 512;
        const HEIGHT: usize = 512;
        let rgb = flat_image(WIDTH, HEIGHT);

        let encoder = encode_for_inspection(&rgb, WIDTH as u16, HEIGHT as u16);
        assert_eq!(encoder.macroblock_no_skip_coeff, Some(0));
        assert_eq!(
            skipped_macroblocks(&encoder.macroblocks),
            encoder.macroblocks.len() - 1
        );

        // The surface settles one chroma level short of the source and stays
        // there: coding the level would overshoot by as much as it corrects,
        // so the encoder leaves it and the whole frame skips, which is what
        // this is here to pin. Re-measured for the quality curve.
        let (bytes, psnr) = round_trip(&rgb, WIDTH, HEIGHT);
        assert!(bytes <= 608, "{bytes} bytes");
        assert!(psnr >= 48.13, "{psnr} dB");
    }

    #[test]
    fn an_empty_block_costs_the_bool_at_the_root_of_the_token_tree() {
        // What `end_of_block_cost` charges a skipped block relies on the end of
        // block token being the first leaf of the token tree, so that it costs
        // one bool and no more.
        for probs in COEFF_PROBS.iter().flatten().flatten() {
            assert_eq!(
                tree_rate(&DCT_TOKEN_TREE, probs, DCT_EOB),
                bool_rate(probs[0], false)
            );
        }
    }

    #[test]
    fn a_bool_costs_the_logarithm_of_its_probability() {
        // A bool that always comes up costs nothing, an even one costs a bit,
        // and the least likely one the coder can represent costs eight.
        assert_eq!(bit_cost(256), 0);
        assert_eq!(bit_cost(128), BIT_COST_ONE_BIT);
        assert_eq!(bit_cost(64), 2 * BIT_COST_ONE_BIT);
        assert_eq!(bit_cost(1), 8 * BIT_COST_ONE_BIT);
        assert_eq!(bit_cost(0), bit_cost(1));

        // and in between it is the logarithm to within the 256th of a bit it
        // counts in, never under it, since the fraction is truncated
        for probability in 1u32..=256 {
            let exact = (8.0 - f64::from(probability).log2()) * f64::from(BIT_COST_ONE_BIT);
            let cost = f64::from(bit_cost(probability));
            assert!(cost >= exact && cost < exact + 1.0, "{probability}: {cost}");
        }
    }

    #[test]
    fn the_loop_filter_follows_the_quantizer() {
        // Nothing to smooth at the minimal quantizer, and never more than the
        // six bits section 9.4 codes the level in.
        assert_eq!(loop_filter_level(0), 0);
        assert_eq!(loop_filter_level(127), 30);
        for index in 0..=127 {
            assert!(loop_filter_level(index) <= 63);
        }

        // A coarser quantizer never asks for less filtering, and asking for
        // more quality never asks for more.
        for index in 0..127 {
            assert!(loop_filter_level(index) <= loop_filter_level(index + 1));
        }
        for quality in 0..100 {
            let coarser = loop_filter_level(QUALITY_TO_QUANTIZER_INDEX[quality]);
            let finer = loop_filter_level(QUALITY_TO_QUANTIZER_INDEX[quality + 1]);
            assert!(coarser >= finer, "quality {quality}");
        }

        // and the three qualities the derivation was swept at land on the
        // levels that swept out best, to within a level
        assert_eq!(loop_filter_level(QUALITY_TO_QUANTIZER_INDEX[90]), 9);
        assert_eq!(loop_filter_level(QUALITY_TO_QUANTIZER_INDEX[75]), 14);
        assert_eq!(loop_filter_level(QUALITY_TO_QUANTIZER_INDEX[50]), 18);
    }

    #[test]
    fn the_quality_curve_never_coarsens_with_quality() {
        // The whole range is covered, from the coarsest quantizer the frame
        // header can carry to the finest.
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX.len(), 101);
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[0], 127);
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[100], 0);

        // and asking for more quality never asks for a coarser quantizer
        for quality in 0..100 {
            assert!(
                QUALITY_TO_QUANTIZER_INDEX[quality] >= QUALITY_TO_QUANTIZER_INDEX[quality + 1],
                "quality {quality}"
            );
            assert!(QUALITY_TO_QUANTIZER_INDEX[quality] <= 127);
        }

        // the two points the curve was fitted through: 75 is where libwebp's
        // own curve hands over to the segment fitted to this encoder, and 90
        // is the operating point that segment was fitted to hold
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[75], 23);
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[90], 12);

        // and the cube root spends far less of the range on coarse quantizers
        // than the straight line it replaced, which read 96, 64 and 32 here
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[25], 55);
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[50], 36);
        assert_eq!(QUALITY_TO_QUANTIZER_INDEX[75], 23);
    }

    #[test]
    fn every_quality_decodes_in_both_decoders() {
        // One image across the whole range, ends included: quality 0 is the
        // coarsest quantizer the header can carry and quality 100 the finest,
        // which the lossy coder still has to be able to write even though the
        // public encoder answers that quality with the lossless one.
        const WIDTH: usize = 128;
        const HEIGHT: usize = 96;
        const QUALITIES: [u8; 8] = [0, 10, 25, 50, 75, 90, 99, 100];
        let rgb = hatched_part_image(WIDTH, HEIGHT);

        let mut previous_index = u8::MAX;
        let mut previous_bytes = 0;
        for quality in QUALITIES {
            let index = QUALITY_TO_QUANTIZER_INDEX[usize::from(quality)];
            assert!(index < previous_index, "quality {quality}");
            previous_index = index;

            let webp = encode_container_at(&rgb, WIDTH as u32, HEIGHT as u32, quality);
            decode_with_both(&webp, WIDTH, HEIGHT);

            // and a finer quantizer never buys fewer bytes
            assert!(
                webp.len() >= previous_bytes,
                "quality {quality}: {} bytes after {previous_bytes}",
                webp.len()
            );
            previous_bytes = webp.len();
        }
    }

    #[test]
    fn the_quantizer_rounds_up_from_its_threshold_onwards() {
        for (coefficient, round_up_at) in [(1usize, AC_ROUND_UP_AT), (0, DC_ROUND_UP_AT)] {
            // A step of 256 puts the threshold exactly on a coefficient, so
            // this is the first level that rounds up and what truncation
            // would have dropped.
            assert_eq!(quantize(round_up_at, 256, coefficient), 1);
            assert_eq!(quantize(-round_up_at, 256, coefficient), -1);
            assert_eq!(quantize(256 + round_up_at, 256, coefficient), 2);

            // one short of it there is nothing to round to and the
            // coefficient truncates, and so does everything below it
            assert_eq!(quantize(round_up_at - 1, 256, coefficient), 0);
            assert_eq!(quantize(-(round_up_at - 1), 256, coefficient), 0);
            assert_eq!(quantize(128, 256, coefficient), 0);

            // over every step the quantizer tables can hand it and every
            // coefficient a residual can reach it is that same threshold,
            // never more than a step out, and negatives mirror positives
            for quant in 1i16..=512 {
                for value in 0i32..=4096 {
                    let step = i32::from(quant);
                    let steps = (value * 256 + (256 - round_up_at) * step) / (256 * step);
                    assert_eq!(
                        quantize(value, quant, coefficient),
                        steps,
                        "{value}/{quant}"
                    );
                    assert_eq!(
                        quantize(-value, quant, coefficient),
                        -steps,
                        "-{value}/{quant}"
                    );
                    assert!((steps * step - value).abs() <= step);
                }
            }
        }

        // Both thresholds sit past the half step, which is what keeps a flat
        // surface from correcting itself back and forth forever, and the DC
        // one holds out past the two thirds of a step that a flat surface one
        // pixel level out asks the finest DC quantizer of the curve for.
        const _: () = assert!(AC_ROUND_UP_AT > 128);
        const _: () = assert!(DC_ROUND_UP_AT * 12 > 256 * 8);
    }

    #[test]
    fn quality_above_100_is_an_error_not_a_panic() {
        let rgb = vec![128u8; 16 * 16 * 3];
        let mut encoder = crate::WebPEncoder::new(Vec::new());
        encoder.set_params(crate::EncoderParams {
            use_lossy: true,
            lossy_quality: 101,
            ..Default::default()
        });
        assert!(matches!(
            encoder.encode(&rgb, 16, 16, ColorType::Rgb8),
            Err(EncodingError::InvalidQuality)
        ));
    }

    #[test]
    fn oversized_first_partition_is_rejected() {
        let mut encoder = Vp8Encoder::new(Vec::new());
        assert!(encoder
            .write_uncompressed_frame_header((1 << 19) - 1)
            .is_ok());
        assert!(matches!(
            encoder.write_uncompressed_frame_header(1 << 19),
            Err(EncodingError::InvalidDimensions)
        ));
    }
}
