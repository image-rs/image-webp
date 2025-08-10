use std::io::Write;
use std::mem;

use byteorder_lite::LittleEndian;
use byteorder_lite::WriteBytesExt;

use crate::transform;
use crate::vp8_arithmetic_encoder::ArithmeticEncoder;
use crate::vp8_common::*;
use crate::vp8_prediction::*;
use crate::ColorType;
use crate::EncodingError;

/// info about the frame that's stored in the header
#[derive(Default)]
struct FrameInfo {
    // TODO: Change to option with the segments
    segments_enabled: bool,
    // TODO: Change to enum (could do for decoder as well)
    filter_type: bool,
    filter_level: u8,
    sharpness_level: u8,
    // TODO: Change to option with the loop filter adjustments
    loop_filter_adjustments: bool,

    quantization_indices: QuantizationIndices,

    macroblock_no_skip_coeff: Option<u8>,

    token_probs: TokenProbTables,

    width: u16,
    height: u16,
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

// note that in decoder it actually stores this information on the macroblock but that's super confusing
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

struct Vp8Encoder<W> {
    writer: W,
    frame_info: FrameInfo,
    encoder: ArithmeticEncoder,
    /// info on all the macroblocks in the image
    macroblock_info: Vec<MacroblockInfo>,
    segments: [Segment; MAX_SEGMENTS],
    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,
}

impl<W: Write> Vp8Encoder<W> {
    fn new(writer: W) -> Self {
        let segment = Segment::default();
        let frame_info = FrameInfo::default();

        Self {
            writer,
            frame_info,
            encoder: ArithmeticEncoder::new(),
            macroblock_info: Vec::new(),
            segments: [segment; MAX_SEGMENTS],

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
        }
    }

    /// Writes the uncompressed part of the frame header
    fn write_uncompressed_frame_header(
        &mut self,
        partition_size: u32,
    ) -> Result<(), EncodingError> {
        let keyframe = true;
        let version = 0;
        let for_display = 1;

        let keyframe_bit = if keyframe { 0 } else { 1 };

        let tag = (partition_size << 5) | (for_display << 4) | (version << 1) | (keyframe_bit);
        self.writer.write_u24::<LittleEndian>(tag)?;

        if keyframe {
            let magic_bytes_buffer: [u8; 3] = [0x9d, 0x01, 0x2a];
            self.writer.write(&magic_bytes_buffer)?;

            let width = self.frame_info.width & 0x3FFF;
            let height = self.frame_info.height & 0x3FFF;
            self.writer.write_u16::<LittleEndian>(width)?;
            self.writer.write_u16::<LittleEndian>(height)?;
        }

        Ok(())
    }

    fn encode_compressed_frame_header(&mut self) {
        // if keyframe
        // color space must be 0
        self.encoder.write_literal(1, 0);
        // pixel type
        self.encoder.write_literal(1, 0);

        self.encoder.write_flag(self.frame_info.segments_enabled);
        if self.frame_info.segments_enabled {
            self.encode_segment_updates();
        }

        self.encoder.write_flag(self.frame_info.filter_type);
        self.encoder.write_literal(6, self.frame_info.filter_level);
        self.encoder
            .write_literal(3, self.frame_info.sharpness_level);

        self.encoder
            .write_flag(self.frame_info.loop_filter_adjustments);
        if self.frame_info.loop_filter_adjustments {
            self.encode_loop_filter_adjustments();
        }

        // partitions length must be 1, 2, 4 or 8, so value will be 0, 1, 2 or 3
        let partitions_value: u8 = self.partitions.len().ilog2().try_into().unwrap();
        self.encoder.write_literal(2, partitions_value);

        self.encode_quantization_indices();

        // refresh entropy probs
        self.encoder.write_literal(1, 0);

        self.encode_updated_token_probabilities();

        let mb_no_skip_coeff = if self.frame_info.macroblock_no_skip_coeff.is_some() {
            1
        } else {
            0
        };
        self.encoder.write_literal(1, mb_no_skip_coeff);
        if let Some(prob_skip_false) = self.frame_info.macroblock_no_skip_coeff {
            self.encoder.write_literal(8, prob_skip_false);
        }
    }

    fn write_partitions(&mut self) -> Result<(), EncodingError> {
        let partitions = mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions
            .into_iter()
            .map(|x| x.flush_and_get_buffer())
            .collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer
                    .write_u24::<LittleEndian>(partition.len() as u32)?;
                self.writer.write(&partition)?;
            }
        }

        // write the final partition
        self.writer
            .write(&partitions_bytes[partitions_bytes.len() - 1])?;

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
            .write_literal(7, self.frame_info.quantization_indices.yac_abs);
        self.encoder
            .write_optional_signed_value(4, self.frame_info.quantization_indices.ydc_delta);
        self.encoder
            .write_optional_signed_value(4, self.frame_info.quantization_indices.y2dc_delta);
        self.encoder
            .write_optional_signed_value(4, self.frame_info.quantization_indices.y2ac_delta);
        self.encoder
            .write_optional_signed_value(4, self.frame_info.quantization_indices.uvdc_delta);
        self.encoder
            .write_optional_signed_value(4, self.frame_info.quantization_indices.uvac_delta);
    }

    // TODO: work out when we want to update these probabilities
    fn encode_updated_token_probabilities(&mut self) {
        for (_i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (_j, js) in is.iter().enumerate() {
                for (_k, ks) in js.iter().enumerate() {
                    for (_t, prob) in ks.iter().enumerate() {
                        // not updating these
                        self.encoder.write_bool(false, *prob);
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        if self.frame_info.segments_enabled {
            if let Some(_segment_id) = macroblock_info.segment_id {
                // TODO: set segment for macroblock
                todo!();
            }
        }

        if let Some(prob) = self.frame_info.macroblock_no_skip_coeff {
            self.encoder
                .write_bool(macroblock_info.coeffs_skipped, prob);
        }

        // encode macroblock info y mode using KEYFRAME_YMODE_TREE
        self.encoder.write_with_tree(
            &KEYFRAME_YMODE_TREE,
            &KEYFRAME_YMODE_PROBS,
            macroblock_info.luma_mode.to_i8(),
            0,
        );

        match macroblock_info.luma_mode.into_intra() {
            None => {
                // 11.3 code each of the subblocks
                if let Some(bpred) = macroblock_info.luma_bpred {
                    for y in 0usize..4 {
                        let mut left = self.left_b_pred[y];
                        for x in 0usize..4 {
                            let top = self.top_b_pred[mbx * 4 + x];
                            let probs = &KEYFRAME_BPRED_MODE_PROBS[top as usize][left as usize];
                            let intra_mode = bpred[y * 4 + x];
                            self.encoder.write_with_tree(
                                &KEYFRAME_BPRED_MODE_TREE,
                                probs,
                                intra_mode.to_i8(),
                                0,
                            );
                            left = intra_mode;
                            self.top_b_pred[mbx * 4 + x] = intra_mode;
                        }
                        self.left_b_pred[y] = left;
                    }
                } else {
                    panic!("Invalid, can't set luma mode to B without setting preds");
                }
            }
            Some(intra_mode) => {
                for (left, top) in self
                    .left_b_pred
                    .iter_mut()
                    .zip(self.top_b_pred[4 * mbx..][..4].iter_mut())
                {
                    *left = intra_mode;
                    *top = intra_mode;
                }
            }
        }

        // encode macroblock info chroma mode
        self.encoder.write_with_tree(
            &KEYFRAME_UV_MODE_TREE,
            &KEYFRAME_UV_MODE_PROBS,
            macroblock_info.chroma_mode.to_i8(),
            0,
        );
    }

    // 13
    fn encode_residual_data(
        &mut self,
        macroblock_info: &MacroblockInfo,
        partition_index: usize,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        // TODO: set residual data if coeffs are not skipped

        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        // TODO: change to get index from macroblock
        let segment = self.segments[0];

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma

            let mut coeffs0 = get_coeffs0_from_block(&y_block_data);

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
        let probs = &self.frame_info.token_probs[plane as usize];

        assert!(complexity <= 2);
        let mut complexity = complexity;

        // convert to zigzag and quantize
        // this is the only lossy part of the encoding
        let mut zigzag_block = [0i32; 16];
        for i in first_coeff..16 {
            let zigzag_index = usize::from(ZIGZAG[i]);
            let quant = if zigzag_index > 0 { ac_quant } else { dc_quant };
            zigzag_block[i] = block[zigzag_index] / i32::from(quant);
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
                    encoder.write_with_tree(token_tree, token_probs, DCT_0, start_index_token_tree);

                    // never going to have an end of block after a 0, so skip checking next coeff
                    skip_eob = true;
                    DCT_0
                }

                // just encode as literal
                literal @ 1..=4 => {
                    encoder.write_with_tree(
                        token_tree,
                        token_probs,
                        literal as i8,
                        start_index_token_tree,
                    );

                    skip_eob = false;
                    literal as i8
                }

                // encode in the category
                value @ _ => {
                    let category = match value {
                        5..=6 => DCT_CAT1,
                        7..=10 => DCT_CAT2,
                        11..=18 => DCT_CAT3,
                        19..=34 => DCT_CAT4,
                        35..=66 => DCT_CAT5,
                        67..=2048 => DCT_CAT6,
                        _ => unreachable!(),
                    };

                    encoder.write_with_tree(
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
                        let extra_bool = if extra & mask > 0 { true } else { false };
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
            encoder.write_with_tree(&DCT_TOKEN_TREE, probabilities, DCT_EOB, 0);
        }

        // whether the block has a non zero coefficient
        end_of_block_index > 0
    }

    fn encode_image(&mut self, data: &[u8], color: ColorType, width: u16, height: u16) -> Result<(), EncodingError> {
        self.frame_info.width = width;
        self.frame_info.height = height;
        self.macroblock_width = width.div_ceil(16);
        self.macroblock_height = height.div_ceil(16);

        let (y_bytes, u_bytes, v_bytes) = match color {
            ColorType::Rgb8 => convert_image_yuv::<3>(data, width, height),
            ColorType::Rgba8 => convert_image_yuv::<4>(data, width, height),
            ColorType::L8 => convert_image_y::<1>(data, width, height),
            ColorType::La8 => convert_image_y::<2>(data, width, height),
        } ;

        self.setup_encoding();

        self.encode_compressed_frame_header();

        // encode residual partitions first
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            // reset left complexity / bpreds for left of image
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let macroblock_info =
                    self.macroblock_info[(mby * self.macroblock_width + mbx) as usize];

                // write macroblock headers
                self.write_macroblock_header(&macroblock_info, mbx.into());

                if !macroblock_info.coeffs_skipped {
                    let y_block_data = self.transform_luma_block(
                        &y_bytes,
                        mbx.into(),
                        mby.into(),
                        &macroblock_info,
                    );

                    let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                        &u_bytes,
                        &v_bytes,
                        mbx.into(),
                        mby.into(),
                        (self.macroblock_width * 8).into(),
                        macroblock_info.chroma_mode,
                    );

                    self.encode_residual_data(
                        &macroblock_info,
                        partition_index,
                        mbx as usize,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                } else {
                    // since coeffs are all zero, need to set all complexities to 0
                    // except if the luma mode is B then won't set Y2
                    self.left_complexity
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                    self.top_complexity[usize::from(mbx)]
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                }
            }
        }

        let compressed_header_encoder = mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32)?;

        self.writer.write(&compressed_header_bytes)?;

        self.write_partitions()?;

        Ok(())
    }

    fn setup_encoding(&mut self) {
        let macroblock_info = MacroblockInfo {
            luma_mode: LumaMode::DC,
            luma_bpred: None,
            ..Default::default()
        };
        let macroblock_num =
            usize::from(self.macroblock_width) * usize::from(self.macroblock_height);
        self.macroblock_info = vec![macroblock_info; macroblock_num];

        self.top_complexity = vec![Complexity::default(); usize::from(self.macroblock_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(self.macroblock_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        self.frame_info.token_probs = COEFF_PROBS;

        // TODO: decide the quantization amount based on encoding parameters/the image
        const QUANT_INDEX: u8 = 10;
        const QUANT_INDEX_USIZE: usize = QUANT_INDEX as usize;

        self.frame_info.segments_enabled = false;
        let quantization_indices = QuantizationIndices {
            yac_abs: QUANT_INDEX,
            ..Default::default()
        };
        self.frame_info.quantization_indices = quantization_indices;

        let segment = Segment {
            ydc: DC_QUANT[QUANT_INDEX_USIZE],
            yac: AC_QUANT[QUANT_INDEX_USIZE],
            y2dc: DC_QUANT[QUANT_INDEX_USIZE] * 2,
            y2ac: ((i32::from(AC_QUANT[QUANT_INDEX_USIZE]) * 155 / 100) as i16).max(8),
            uvdc: DC_QUANT[QUANT_INDEX_USIZE],
            uvac: AC_QUANT[QUANT_INDEX_USIZE],
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

    /// Transforms the luma macroblock in the following ways
    /// 1. Does the luma prediction and subtracts from the block
    /// 2. Converts the block so each 4x4 subblock is contiguous within the block
    /// 3. Does the DCT on each subblock
    /// 4. Quantizes the block and dequantizes each subblock
    /// 5. Calculates the quantized block - this can be used to calculate how accurate the
    /// result is and is used to populate the borders for the next macroblock
    fn transform_luma_block(
        &mut self,
        y_data: &[u8],
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> [i32; 16 * 16] {
        let stride = 1usize + 16 + 4;

        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        // do the prediction
        match macroblock_info.luma_mode {
            LumaMode::V => predict_vpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut y_with_border, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => {
                if let Some(bpred_modes) = macroblock_info.luma_bpred {
                    return self.transform_luma_blocks_4x4(
                        y_data,
                        y_with_border,
                        bpred_modes,
                        mbx,
                        mby,
                    );
                } else {
                    panic!("Invalid, need bpred modes");
                }
            }
        }

        let mut luma_blocks = [0i32; 16 * 16];
        let segment = self.segments[0];

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
                        let predicted_value = y_with_border[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = y_data[actual_index];
                        block[y * 4 + x] = i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                // transform block before copying it into main block
                transform::dct4x4(&mut block);

                luma_blocks[block_index..][..16].copy_from_slice(&block);
            }
        }

        // now we're essentially applying the same functions as the decoder in order to ensure
        // that the border is the same as the one used for the decoder in the same macroblock

        // first, quantize and de-quantize the y2 block
        let mut coeffs0 = get_coeffs0_from_block(&luma_blocks);
        transform::wht4x4(&mut coeffs0);
        for (index, value) in coeffs0.iter_mut().enumerate() {
            let quant = if index > 0 {
                segment.y2ac
            } else {
                segment.y2dc
            };
            *value = (*value / i32::from(quant)) * i32::from(quant);
        }
        transform::iwht4x4(&mut coeffs0);

        let mut quantized_luma_residue = [0i32; 16 * 16];
        for k in 0usize..16 {
            quantized_luma_residue[16 * k] = coeffs0[k];
        }

        // quantize and de-quantize the y blocks as well as do the inverse transform
        for (y_block, luma_block) in luma_blocks
            .chunks_exact(16)
            .zip(quantized_luma_residue.chunks_exact_mut(16))
        {
            for (y_value, value) in y_block[1..].iter().zip(luma_block[1..].iter_mut()) {
                *value = (*y_value / i32::from(segment.yac)) * i32::from(segment.yac);
            }

            transform::idct4x4(luma_block);
        }

        // re-use the y_with_border from earlier since the border pixels
        // applies the same thing as the decoder so that the border
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = x + y * 4;
                // Create a reference to a [i32; 16] array for add_residue (slices of size 16 do not work).
                let rb: &[i32; 16] = quantized_luma_residue[i * 16..][..16].try_into().unwrap();
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                add_residue(&mut y_with_border, rb, y0, x0, stride);
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

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    fn transform_luma_blocks_4x4(
        &mut self,
        y_data: &[u8],
        mut y_with_border: [u8; LUMA_BLOCK_SIZE],
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = 1usize + 16 + 4;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let segment = self.segments[0];

        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                match bpred_modes[i] {
                    IntraMode::TM => predict_tmpred(&mut y_with_border, 4, x0, y0, stride),
                    IntraMode::VE => predict_bvepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HE => predict_bhepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::DC => predict_bdcpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::LD => predict_bldpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::RD => predict_brdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VR => predict_bvrpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VL => predict_bvlpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HD => predict_bhdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HU => predict_bhupred(&mut y_with_border, x0, y0, stride),
                }

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
                        let actual_value = y_data[actual_index];
                        current_subblock[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut current_subblock);

                luma_blocks[block_index..][..16].copy_from_slice(&current_subblock);

                // quantize and de-quantize the subblock
                for (index, y_value) in current_subblock.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.yac } else { segment.ydc };
                    *y_value = (*y_value / i32::from(quant)) * i32::from(quant);
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

        return luma_blocks;
    }

    fn transform_chroma_blocks(
        &mut self,
        u_data: &[u8],
        v_data: &[u8],
        mbx: usize,
        mby: usize,
        chroma_width: usize,
        prediction: ChromaMode,
    ) -> ([i32; 16 * 4], [i32; 16 * 4]) {
        let stride = 1usize + 8;

        let mut u_with_border =
            create_border_chroma(mbx, mby, &self.top_border_u, &self.left_border_u);
        let mut v_with_border =
            create_border_chroma(mbx, mby, &self.top_border_v, &self.left_border_v);

        match prediction {
            ChromaMode::DC => {
                predict_dcpred(&mut u_with_border, 8, stride, mby != 0, mbx != 0);
                predict_dcpred(&mut v_with_border, 8, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(&mut u_with_border, 8, 1, 1, stride);
                predict_vpred(&mut v_with_border, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(&mut u_with_border, 8, 1, 1, stride);
                predict_hpred(&mut v_with_border, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut u_with_border, 8, 1, 1, stride);
                predict_tmpred(&mut v_with_border, 8, 1, 1, stride);
            }
        }

        let mut u_blocks = [0i32; 16 * 4];
        let mut v_blocks = [0i32; 16 * 4];

        for block_y in 0..2 {
            for block_x in 0..2 {
                // the index on the chroma block
                let block_index = block_y * 16 * 2 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let chroma_data_block_index =
                    (mby * 8 + block_y * 4) * chroma_width + mbx * 8 + block_x * 4;

                let mut u_block = [0i32; 16];
                let mut v_block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let u_predicted_value = u_with_border[predicted_index];
                        let v_predicted_value = v_with_border[predicted_index];
                        let actual_index = chroma_data_block_index + y * chroma_width + x;
                        let u_actual_value = u_data[actual_index];
                        let v_actual_value = v_data[actual_index];
                        u_block[y * 4 + x] =
                            i32::from(u_actual_value) - i32::from(u_predicted_value);
                        v_block[y * 4 + x] =
                            i32::from(v_actual_value) - i32::from(v_predicted_value);
                    }
                }

                transform::dct4x4(&mut u_block);
                transform::dct4x4(&mut v_block);

                u_blocks[block_index..][..16].copy_from_slice(&u_block);
                v_blocks[block_index..][..16].copy_from_slice(&v_block);
            }
        }

        let segment = self.segments[0];

        let mut quantized_u_residue = [0i32; 16 * 4];
        let mut quantized_v_residue = [0i32; 16 * 4];

        for (((u_residue_block, v_residue_block), transformed_u_block), transformed_v_block) in
            quantized_u_residue
                .chunks_exact_mut(16)
                .zip(quantized_v_residue.chunks_exact_mut(16))
                .zip(u_blocks.chunks_exact(16))
                .zip(v_blocks.chunks_exact(16))
        {
            for ((((index, u_block), v_block), u_block_residue), v_block_residue) in u_residue_block
                .iter_mut()
                .enumerate()
                .zip(v_residue_block.iter_mut())
                .zip(transformed_u_block.iter())
                .zip(transformed_v_block.iter())
            {
                let quant = if index > 0 {
                    segment.uvac
                } else {
                    segment.uvdc
                };
                *u_block = (*u_block_residue / i32::from(quant)) * i32::from(quant);
                *v_block = (*v_block_residue / i32::from(quant)) * i32::from(quant);
            }

            transform::idct4x4(u_residue_block);
            transform::idct4x4(v_residue_block);
        }

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = quantized_u_residue[i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut u_with_border, urb, y0, x0, stride);

                let vrb: &[i32; 16] = quantized_v_residue[i * 16..][..16].try_into().unwrap();

                add_residue(&mut v_with_border, vrb, y0, x0, stride);
            }
        }

        // set borders
        for ((y, u_border_value), v_border_value) in self
            .left_border_u
            .iter_mut()
            .enumerate()
            .zip(self.left_border_v.iter_mut())
        {
            *u_border_value = u_with_border[y * stride + 8];
            *v_border_value = v_with_border[y * stride + 8];
        }

        for ((x, u_border_value), v_border_value) in self.top_border_u[mbx * 8..][..8]
            .iter_mut()
            .enumerate()
            .zip(self.top_border_v[mbx * 8..][..8].iter_mut())
        {
            *u_border_value = u_with_border[8 * stride + x + 1];
            *v_border_value = v_with_border[8 * stride + x + 1];
        }

        (u_blocks, v_blocks)
    }
}

fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
}

// constants used for yuv -> rgb conversion, using ones from libwebp
const YUV_FIX: i32 = 16;
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

/// converts the whole image to yuv data and adds values on the end to make it match the macroblock sizes
/// downscales the u/v data as well so it's half the width and height of the y data
fn convert_image_yuv<const BPP: usize>(image_data: &[u8], width: u16, height: u16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_width = 8 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let mut u_bytes = vec![0u8; chroma_size];
    let mut v_bytes = vec![0u8; chroma_size];

    // loop through two rows at a time so that we can calculate the average of the 2x2 pixels
    // for averaging for the chroma pixels when downscaling
    for (((image_rows, y_rows), u_row), v_row) in image_data.chunks_exact(BPP * width * 2)
        .zip(y_bytes.chunks_exact_mut(luma_width * 2))
        .zip(u_bytes.chunks_exact_mut(chroma_width)) 
        .zip(v_bytes.chunks_exact_mut(chroma_width)) 
    {
        let (image_row_1, image_row_2) = image_rows.split_at(BPP * width);
        let (y_row_1, y_row_2) = y_rows.split_at_mut(luma_width);

        for (((((row_1, row_2), y_pixels_1), y_pixels_2), u_pixel), v_pixel) in image_row_1.chunks_exact(BPP * 2)
            .zip(image_row_2.chunks_exact(BPP * 2))
            .zip(y_row_1.chunks_exact_mut(2))
            .zip(y_row_2.chunks_exact_mut(2))
            .zip(u_row.iter_mut())
            .zip(v_row.iter_mut())
        {
            let (rgb1, rgb2) = row_1.split_at(BPP);
            let (rgb3, rgb4) = row_2.split_at(BPP);

            y_pixels_1[0] = rgb_to_y(rgb1);
            y_pixels_1[1] = rgb_to_y(rgb2);
            y_pixels_2[0] = rgb_to_y(rgb3);
            y_pixels_2[1] = rgb_to_y(rgb4);

            *u_pixel = rgb_to_u_avg(rgb1, rgb2, rgb3, rgb4);
            *v_pixel = rgb_to_v_avg(rgb1, rgb2, rgb3, rgb4);
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

fn convert_image_y<const BPP: usize>(image_data: &[u8], width: u16, height: u16) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = usize::from(width);
    let height = usize::from(height);
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let luma_width = 16 * mb_width;
    let chroma_size = 8 * mb_width * 8 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    let u_bytes = vec![127u8; chroma_size];
    let v_bytes = vec![127u8; chroma_size];

    for (image_row, y_row) in image_data.chunks_exact(BPP * width)
        .zip(y_bytes.chunks_exact_mut(luma_width))
    {
        for (image_value, y_pixel) in image_row.chunks_exact(BPP)
            .zip(y_row.iter_mut())
        {
            *y_pixel = image_value[0];
        }
    }

    (y_bytes, u_bytes, v_bytes)
}

// values come from libwebp
// Y = 0.2568 * R + 0.5041 * G + 0.0979 * B + 16
// U = -0.1482 * R - 0.2910 * G + 0.4392 * B + 128
// V = 0.4392 * R - 0.3678 * G - 0.0714 * B + 128

// this is converted to 16 bit fixed point by multiplying by 2^16 
// and shifting back

fn rgb_to_y(rgb: &[u8]) -> u8 {
    let luma = 16839 * i32::from(rgb[0]) + 33059 * i32::from(rgb[1]) + 6420 * i32::from(rgb[2]);
    ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
}

// get the average of the four surrounding pixels
fn rgb_to_u_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let u1 = rgb_to_u_raw(rgb1);
    let u2 = rgb_to_u_raw(rgb2);
    let u3 = rgb_to_u_raw(rgb3);
    let u4 = rgb_to_u_raw(rgb4);

    ((u1 + u2 + u3 + u4) >> (YUV_FIX + 2)) as u8
}

    // get the average of the four surrounding pixels
fn rgb_to_v_avg(rgb1: &[u8], rgb2: &[u8], rgb3: &[u8], rgb4: &[u8]) -> u8 {
    let v1 = rgb_to_v_raw(rgb1);
    let v2 = rgb_to_v_raw(rgb2);
    let v3 = rgb_to_v_raw(rgb3);
    let v4 = rgb_to_v_raw(rgb4);

    ((v1 + v2 + v3 + v4) >> (YUV_FIX + 2)) as u8
}

fn rgb_to_u_raw(rgb: &[u8]) -> i32 {
    -9719 * i32::from(rgb[0]) - 19081 * i32::from(rgb[1]) + 28800 * i32::from(rgb[2]) + (128 << YUV_FIX)
}

fn rgb_to_v_raw(rgb: &[u8]) -> i32 {
    28800 * i32::from(rgb[0]) - 24116 * i32::from(rgb[1]) - 4684 * i32::from(rgb[2]) + (128 << YUV_FIX)
}

pub(crate) fn encode_frame_lossy<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
) -> Result<(), EncodingError> {
    let mut vp8_encoder = Vp8Encoder::new(writer);

    let width = width
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;
    let height = height
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;

    vp8_encoder.encode_image(data, color, width, height)?;

    Ok(())
}

#[cfg(test)]
mod tests {}
