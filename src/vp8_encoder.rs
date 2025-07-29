use std::io::Write;
use std::mem;

use byteorder_lite::WriteBytesExt;
use byteorder_lite::LittleEndian;

use crate::vp8::AC_QUANT;
use crate::vp8::COEFF_UPDATE_PROBS;
use crate::vp8::DCT_EOB;
use crate::vp8::DC_QUANT;
use crate::EncodingError;
use crate::{transform::{dct4x4, wht4x4}, vp8::{create_border_chroma, create_border_luma, predict_dcpred, predict_hpred, predict_tmpred, predict_vpred, ChromaMode, LumaMode, Segment, TokenProbTables, COEFF_BANDS, COEFF_PROBS, DCT_0, DCT_1, DCT_CAT1, DCT_CAT2, DCT_CAT3, DCT_CAT4, DCT_CAT5, DCT_CAT6, DCT_CAT_BASE, DCT_TOKEN_TREE, KEYFRAME_UV_MODE_PROBS, KEYFRAME_UV_MODE_TREE, KEYFRAME_YMODE_PROBS, KEYFRAME_YMODE_TREE, MAX_SEGMENTS, PROB_DCT_CAT, ZIGZAG}, vp8_arithmetic_encoder::ArithmeticEncoder, vp8_info::Plane};

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
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,
    
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

struct Vp8Encoder<W> {
    writer: W,
    frame_info: FrameInfo,
    encoder: ArithmeticEncoder,
    /// info on all the macroblocks in the image
    macroblock_info: Vec<MacroblockInfo>,
    segments: [Segment; MAX_SEGMENTS],
    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    macroblock_width: u16,
    macroblock_height: u16,

    partitions: Vec<ArithmeticEncoder>,

    // width: u32,
    // height: u32,
    // y_bytes: Vec<u8>,
    // u_bytes: Vec<u8>,
    // v_bytes: Vec<u8>,
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
            
            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],
        }
    }

    /// Writes the uncompressed part of the frame header
    fn write_uncompressed_frame_header(&mut self, partition_size: u32) -> Result<(), EncodingError> {
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
        self.encoder.write_literal(3, self.frame_info.sharpness_level);

        self.encoder.write_flag(self.frame_info.loop_filter_adjustments);
        if self.frame_info.loop_filter_adjustments {
            self.encode_loop_filter_adjustments();
        }

        let partitions_value = match self.partitions.len() {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            // number of partitions can only be one of these values
            _ => unreachable!(),
        };
        self.encoder.write_literal(2, partitions_value);

        self.encode_quantization_indices();

        // refresh entropy probs
        self.encoder.write_literal(1, 0);

        self.encode_updated_token_probabilities();

        let mb_no_skip_coeff = if self.frame_info.macroblock_no_skip_coeff.is_some() { 1 } else { 0 };
        self.encoder.write_literal(1, mb_no_skip_coeff);
        if let Some(prob_skip_false) = self.frame_info.macroblock_no_skip_coeff {
            self.encoder.write_literal(8, prob_skip_false);
        }
    }

    fn write_partitions(&mut self) -> Result<(), EncodingError> {
        let partitions = mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions.into_iter().map(|x| x.flush_and_get_buffer()).collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer.write_u24::<LittleEndian>(partition.len() as u32)?;
                self.writer.write(&partition)?;
            }
        }

        // write the final partition
        self.writer.write(&partitions_bytes[partitions_bytes.len() - 1])?;

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
        self.encoder.write_literal(7, self.frame_info.quantization_indices.yac_abs);
        self.encoder.write_optional_signed_value(4, self.frame_info.quantization_indices.ydc_delta);
        self.encoder.write_optional_signed_value(4, self.frame_info.quantization_indices.y2dc_delta);
        self.encoder.write_optional_signed_value(4, self.frame_info.quantization_indices.y2ac_delta);
        self.encoder.write_optional_signed_value(4, self.frame_info.quantization_indices.uvdc_delta);
        self.encoder.write_optional_signed_value(4, self.frame_info.quantization_indices.uvac_delta);

    }

    fn encode_updated_token_probabilities(&mut self) {
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate() {
                        // not updating these
                        self.encoder.write_bool(false, *prob);
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo) {
        if self.frame_info.segments_enabled {
            if let Some(segment_id) = macroblock_info.segment_id {
                // TODO: set segment
                todo!();
            }
        }
        
        if let Some(prob) = self.frame_info.macroblock_no_skip_coeff {
            // TODO: 11.1 set mb_skip_coeff 
            todo!();
        }

        // encode macroblock info y mode using KEYFRAME_YMODE_TREE
        self.encoder.write_with_tree(&KEYFRAME_YMODE_TREE, &KEYFRAME_YMODE_PROBS, macroblock_info.luma_mode.to_i8(), 0);
        
        if macroblock_info.luma_mode == LumaMode::B {
            // TODO: 11.3 code each of the subblocks 
            todo!();
        }

        // encode macroblock info chroma mode
        self.encoder.write_with_tree(&KEYFRAME_UV_MODE_TREE, &KEYFRAME_UV_MODE_PROBS, macroblock_info.chroma_mode.to_i8(), 0);

        
    }

    // 13
    fn encode_residual_data(&mut self, macroblock_info: &MacroblockInfo, partition_index: usize, mbx: usize, y_subblocks: &[[i32; 16]; 16], u_subblocks: &[[i32; 16]; 4], v_subblocks: &[[i32; 16]; 4]) {
        // TODO: set residual data if coeffs are not skipped

        let mut plane = if macroblock_info.luma_mode == LumaMode::B { Plane::YCoeff0 } else { Plane::Y2 };

        // TODO: change to get index from macroblock 
        let segment = self.segments[0];

        let mut transformed_y_subblocks = [[0i32; 16]; 16];
        for (transformed_block, y_block) in transformed_y_subblocks.iter_mut().zip(y_subblocks.iter()) {
            *transformed_block = dct4x4(&y_block);
        }

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma

            let mut coeffs0 = [0i32; 16];
            for (coeff, y_subblock) in coeffs0.iter_mut().zip(transformed_y_subblocks.iter()) {
                *coeff = y_subblock[0];
            }
            
            // wht here on the 0th coeffs
            let coeffs0 = wht4x4(&coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;

            let has_coeffs = self.encode_coefficients(&coeffs0, partition_index, plane, complexity.into(), segment.y2dc, segment.y2ac);

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            // next encode luma coefficients without the 0th coeffs
            plane = Plane::YCoeff1;
        }

        // now encode the 16 luma 4x4 subblocks in the macroblock
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = transformed_y_subblocks[y * 4 + x];
                // let block = dct4x4(&block);

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(&block, partition_index, plane, complexity.into(), segment.ydc, segment.yac);
                
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
                let block = u_subblocks[y * 2 + x];
                let block = dct4x4(&block);

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(&block, partition_index, plane, complexity.into(), segment.uvdc, segment.uvac);

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // encode the 4 v 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_subblocks[y * 2 + x];
                let block = dct4x4(&block);

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(&block, partition_index, plane, complexity.into(), segment.uvdc, segment.uvac);

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
        
    }

    // encodes the coefficients which is the reverse procedure of read_coefficients in the decoder
    // returns whether there was any non-zero data in the block for the complexity 
    fn encode_coefficients(&mut self, block: &[i32; 16], partition_index: usize, plane: Plane, complexity: usize, dc_quant: i16, ac_quant: i16) -> bool {
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
        let end_of_block_index = if let Some(last_non_zero_index) = zigzag_block.iter().rev().position(|x| *x != 0) {
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
                },

                // just encode as literal
                literal @ 1..=4 => { 
                    encoder.write_with_tree(token_tree, token_probs, literal as i8, start_index_token_tree);
                    
                    skip_eob = false;
                    literal as i8
                },

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

                    encoder.write_with_tree(token_tree, token_probs, category, start_index_token_tree);

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

    fn encode_image(&mut self, data: &[u8], width: u32, height: u32) -> Result<(), EncodingError> {

        self.frame_info.width = width as u16;
        self.frame_info.height = height as u16;
        self.macroblock_width = width.div_ceil(16) as u16;
        self.macroblock_height = height.div_ceil(16) as u16;

        let (y_bytes, u_bytes, v_bytes) = convert_image_yuv(data, width, height);
        
        self.setup_encoding();

        // encode residual partitions first
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            self.left_complexity = Complexity::default();
            for mbx in 0..self.macroblock_width {
                let macroblock_info = self.macroblock_info[(mby * self.macroblock_width + mbx) as usize]; 

                let predicted_luma = 
                    subtract_prediction_luma(&y_bytes, mbx as usize, mby as usize, self.macroblock_width as usize, (self.macroblock_width * 16) as usize, macroblock_info.luma_mode);
                let (predicted_u, predicted_v) = 
                    subtract_prediction_chroma(&u_bytes, &v_bytes, mbx as usize, mby as usize, (self.macroblock_width * 8) as usize, macroblock_info.chroma_mode);
                
                let y_subblocks = luma_block_to_subblock(&predicted_luma);
                let u_subblocks = chroma_block_to_subblock(&predicted_u);
                let v_subblocks = chroma_block_to_subblock(&predicted_v);

                self.encode_residual_data(&macroblock_info, partition_index, mbx as usize, &y_subblocks, &u_subblocks, &v_subblocks);
            }
        }

        self.encode_compressed_frame_header();

        // write macroblock headers
        for mby in 0..self.macroblock_height {
            for mbx in 0..self.macroblock_width {
                let macroblock_info = self.macroblock_info[(mby * self.macroblock_width + mbx) as usize]; 
                self.write_macroblock_header(&macroblock_info);
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
        let macroblock_info = MacroblockInfo::default();
        let macroblock_num = usize::from(self.macroblock_width) * usize::from(self.macroblock_height);
        self.macroblock_info = vec![macroblock_info; macroblock_num];

        self.top_complexity = vec![Complexity::default(); usize::from(self.macroblock_width)];

        self.frame_info.token_probs = COEFF_PROBS;
    
        self.frame_info.segments_enabled = false;
        let quantization_indices = QuantizationIndices {
            yac_abs: 10,
            ..Default::default()
        };
        self.frame_info.quantization_indices = quantization_indices;
        
        let segment = Segment {
            ydc: DC_QUANT[10],
            yac: AC_QUANT[10],
            y2dc: DC_QUANT[10] * 2,
            y2ac: ((i32::from(AC_QUANT[10]) * 155 / 100) as i16).max(8),
            uvdc: DC_QUANT[10],
            uvac: AC_QUANT[10],
            ..Default::default()
        };
        self.segments[0] = segment;
    }
}

fn get_macroblock_info(
    y_bytes: &[u8],
    u_bytes: &[u8],
    v_bytes: &[u8],
    macroblock_x: usize,
    macroblock_y: usize,
) -> MacroblockInfo {
    // TODO: select prediction mode properly
    let luma_prediction = LumaMode::default();
    let chroma_prediction = ChromaMode::default();
    // TODO: choose whether to use segment for this macroblock
    let segment_id = None;

    MacroblockInfo {
        luma_mode: luma_prediction,
        chroma_mode: chroma_prediction,
        segment_id,
    }
}

// gets all the subblocks out of the macroblock
fn luma_block_to_subblock(y_macroblock_data: &[i32; 16 * 16]) -> [[i32; 16]; 16] {
    let mut blocks = [[0i32; 16]; 16];
    let stride = 16;
    
    for y in 0..4 {
        for x in 0..4 {
            let start_index = y * 4 * stride + x * 4;
            let subblock = get_subblock(y_macroblock_data, start_index, stride);
            blocks[y * 4 + x] = subblock;
        }
    }

    blocks
}

// gets all the subblocks out of the macroblock
fn chroma_block_to_subblock(chroma_macroblock_data: &[i32; 8 * 8]) -> [[i32; 16]; 4] {
    let mut blocks = [[0i32; 16]; 4];
    let stride = 8;

    for y in 0..2 {
        for x in 0..2 {
            let start_index = y * 4 * stride + x * 4;
            let subblock = get_subblock(chroma_macroblock_data, start_index, stride);
            blocks[y * 2 + x] = subblock;
        }
    }

    blocks
}

fn get_subblock(block: &[i32], start_index: usize, stride: usize) -> [i32; 16] {
    let mut subblock = [0i32; 16];
    for sub_y in 0..4 {
        subblock[sub_y * 4..][..4].copy_from_slice(&block[start_index + sub_y * stride..][..4]);
    };
    subblock
}

// TODO: do we need to create a new buffer for this?
fn subtract_prediction_luma(y_data: &[u8], mbx: usize, mby: usize, mbw: usize, width: usize, pred: LumaMode) -> [i32; 16 * 16] {
    let stride = 1usize + 16 + 4;

    let top_border_y = if mby == 0 {
        // will get ignored
        &Vec::new()
    } else {
        let index = (mby * 16 - 1) * width;
        &y_data[index..][..width]
    };
    const LEFT_BORDER_SIZE: usize = 16 + 1;
    let mut left_border_y = [129u8; LEFT_BORDER_SIZE];
    if mbx != 0 {
        if mby == 0 {
            // start from point 1 to left 
            let index = mby * 16 * width + mbx * 16 - 1;
            for (index, border_val) in (index..).step_by(width).take(LEFT_BORDER_SIZE - 1)
                .zip(left_border_y[1..].iter_mut())
            {
                *border_val = y_data[index];
            }
        } else {
            // start from point 1 to top and left
            let index = (mby * 16 - 1) * width + mbx * 16 - 1;
            for (index, border_val) in (index..).step_by(width).take(LEFT_BORDER_SIZE)
                .zip(left_border_y.iter_mut())
            {
                *border_val = y_data[index];
            }
        }
    }

    let mut y_with_border = create_border_luma(mbx, mby, mbw, top_border_y, &left_border_y);

    // do the prediction
    match pred {
        LumaMode::V => predict_vpred(&mut y_with_border, 16, 1, 1, stride),
        LumaMode::H => predict_hpred(&mut y_with_border, 16, 1, 1, stride),
        LumaMode::TM => predict_tmpred(&mut y_with_border, 16, 1, 1, stride),
        LumaMode::DC => predict_dcpred(&mut y_with_border, 16, stride, mby != 0, mbx != 0),
        LumaMode::B => todo!(),
    }

    // now we need to subtract the predicted values from the actual values
    let mut new_buffer = [0i32; 16 * 16];
    for (index, buf_val) in new_buffer.iter_mut().enumerate()
    {
        let x = index % 16;
        let y = index / 16;
        let pred_value = y_with_border[stride * (y + 1) + x + 1];
        let y_value = y_data[(mby * 16 + y) * width + mbx * 16 + x];
        
        *buf_val = i32::from(y_value) - i32::from(pred_value);
    }
    new_buffer
}

fn subtract_prediction_chroma(u_data: &[u8], v_data: &[u8], mbx: usize, mby: usize, width: usize, pred: ChromaMode) -> ([i32; 8 * 8], [i32; 8 * 8]) {
    let stride = 8 + 1;
    let chroma_width = width;
    let top_border_u = create_top_border_chroma(u_data, mby, chroma_width);
    let left_border_u = create_left_border_chroma(u_data, mbx, mby, chroma_width);
    let top_border_v = create_top_border_chroma(v_data, mby, chroma_width);
    let left_border_v = create_left_border_chroma(v_data, mbx, mby, chroma_width);

    let mut u_with_border = create_border_chroma(mbx, mby, top_border_u, &left_border_u);
    let mut v_with_border = create_border_chroma(mbx, mby, top_border_v, &left_border_v);

    match pred {
        ChromaMode::DC => {
            predict_dcpred(&mut u_with_border, 8, stride, mby != 0, mbx != 0);
            predict_dcpred(&mut v_with_border, 8, stride, mby != 0, mbx != 0);
        },
        ChromaMode::V => {
            predict_vpred(&mut u_with_border, 8, 1, 1, stride);
            predict_vpred(&mut v_with_border, 8, 1, 1, stride);
        },
        ChromaMode::H => {
            predict_hpred(&mut u_with_border, 8, 1, 1, stride);
            predict_hpred(&mut v_with_border, 8, 1, 1, stride);
        },
        ChromaMode::TM => {
            predict_tmpred(&mut u_with_border, 8, 1, 1, stride);
            predict_tmpred(&mut v_with_border, 8, 1, 1, stride);
        },
    }

    let mut new_buffer_u = [0i32; 8 * 8];
    let mut new_buffer_v = [0i32; 8 * 8];

    for ((index, new_u), new_v) in new_buffer_u.iter_mut().enumerate()
        .zip(new_buffer_v.iter_mut())
    {
        let x = index % 8;
        let y = index / 8;

        let pred_value_u = u_with_border[stride * (y + 1) + x + 1];
        let u_value = u_data[(mby * 8 + y) * width + mbx * 8 + x];

        let pred_value_v = v_with_border[stride * (y + 1) + x + 1];
        let v_value = v_data[(mby * 8 + y) * width + mbx * 8 + x];

        *new_u = i32::from(u_value) - i32::from(pred_value_u);
        *new_v = i32::from(v_value) - i32::from(pred_value_v);
    }

    (new_buffer_u, new_buffer_v)
}

fn create_top_border_chroma(buffer: &[u8], mby: usize, chroma_width: usize) -> &[u8] {
    if mby == 0 {
        // won't be used
        &buffer[..]
    } else {
        let index = (mby * 8 - 1) * chroma_width; 
        &buffer[index..][..chroma_width]
    }
}

fn create_left_border_chroma(buffer: &[u8], mbx: usize, mby: usize, chroma_width: usize) -> [u8; 8 + 1] {
    let mut left_border = [129u8; 8 + 1];
    let left_border_size = 8 + 1;
    
    if mby == 0 {
        left_border[0] = 127;
    }

    if mbx != 0 {
        if mby == 0 {
            // start from point 1 to left 
            let index = mby * 8 * chroma_width + mbx * 8 - 1;
            for (index, border_val) in (index..).step_by(chroma_width).take(left_border_size - 1)
                .zip(left_border[1..].iter_mut())
            {
                *border_val = buffer[index];
            }
        } else {
            // start from point 1 to top and left
            let index = (mby * 8 - 1) * chroma_width + mbx * 8 - 1;
            for (index, border_val) in (index..).step_by(chroma_width).take(left_border_size)
                .zip(left_border.iter_mut())
            {
                *border_val = buffer[index];
            }
        }
    }

    left_border
}

fn rgb_to_y(r: u8, g: u8, b: u8) -> u8 {
    let luma = 0.2568 * f64::from(r) + 0.5041 * f64::from(g) + 0.0979 * f64::from(b) + 16.0;
    luma as u8
}

fn rgb_to_u(r: u8, g: u8, b: u8) -> u8 {
    let u = -0.1482 * f64::from(r) - 0.2910 * f64::from(g) + 0.4392 * f64::from(b) + 128.0;
    u as u8
}

fn rgb_to_v(r: u8, g: u8, b: u8) -> u8 {
    let v = 0.4392 * f64::from(r) - 0.3678 * f64::from(g) - 0.0714 * f64::from(b) + 128.0;
    v as u8
}

// converts the whole image to yuv data and adds values on the end to make it match the macroblock sizes
// downscales the u/v data as well so it's half the width and height of the y data
fn convert_image_yuv(image_data: &[u8], width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let width = width as usize;
    let height = height as usize;
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let y_size = 16 * mb_width * 16 * mb_height;
    let mut y_bytes = vec![0u8; y_size];
    for y in 0..height {
        for x in 0..width {
            if let [r, g, b] = image_data[3 * (y * width + x)..][..3] {
                y_bytes[y * 16 * mb_width + x] = rgb_to_y(r, g, b);
            }
        }
    }
    
    let (u_bytes, v_bytes) = get_uv_buffers(image_data, width, height);

    (y_bytes, u_bytes, v_bytes)
}

// need to handle u and v separately since they need to be downscaled
// TODO: something like gamma subsampling, similar to libwebp e.g.
// https://github.com/mozilla/mozjpeg/issues/193
// https://en.wikipedia.org/wiki/Chroma_subsampling#4:2:0
// currently just doing something simple where we take the average of the 4 pixels around
fn get_uv_buffers(rgb_image_data: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    const BPP: usize = 3;
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let chroma_width = mb_width * 8;
    let chroma_height = mb_height * 8;
    let chroma_size = chroma_width * chroma_height;

    let mut u_buffer = vec![0u8; chroma_size];
    let mut v_buffer = vec![0u8; chroma_size];
    
    for ((rgb_row, u_row), v_row) in rgb_image_data.chunks_exact(BPP * width * 2)
        .zip(u_buffer.chunks_exact_mut(chroma_width)) 
        .zip(v_buffer.chunks_exact_mut(chroma_width)) 
    {
        let (rgb_row1, rgb_row2) = rgb_row.split_at(BPP * width);
        
        for (((rgb1, rgb2), u), v) in (rgb_row1.chunks_exact(BPP * 2))
            .zip(rgb_row2.chunks_exact(BPP * 2)) 
            .zip(u_row.iter_mut())
            .zip(v_row.iter_mut())
        {
            let (rgb1_1, rgb1_2) = rgb1.split_at(BPP);
            let (rgb2_1, rgb2_2) = rgb2.split_at(BPP);

            let u1 = rgb_to_u(rgb1_1[0], rgb1_1[1], rgb1_1[2]);
            let u2 = rgb_to_u(rgb1_2[0], rgb1_2[1], rgb1_2[2]);
            let u3 = rgb_to_u(rgb2_1[0], rgb2_1[1], rgb2_1[2]);
            let u4 = rgb_to_u(rgb2_2[0], rgb2_2[1], rgb2_2[2]);

            let v1 = rgb_to_v(rgb1_1[0], rgb1_1[1], rgb1_1[2]);
            let v2 = rgb_to_v(rgb1_2[0], rgb1_2[1], rgb1_2[2]);
            let v3 = rgb_to_v(rgb2_1[0], rgb2_1[1], rgb2_1[2]);
            let v4 = rgb_to_v(rgb2_2[0], rgb2_2[1], rgb2_2[2]);

            let u_val = avg4(u1, u2, u3, u4);
            let v_val = avg4(v1, v2, v3, v4);

            *u = u_val;
            *v = v_val;
        }
    }

    (u_buffer, v_buffer)
}

fn avg4(val1: u8, val2: u8, val3: u8, val4: u8) -> u8 {
    let total = u16::from(val1) + u16::from(val2) + u16::from(val3) + u16::from(val4);
    (total / 4) as u8
}

pub(crate) fn encode_frame_lossy<W: Write>(writer: W, data: &[u8], width: u32, height: u32) -> Result<(), EncodingError> {
    
    let mut vp8_encoder = Vp8Encoder::new(writer);

    vp8_encoder.encode_image(data, width, height)?;

    Ok(())
}

#[cfg(test)]
mod tests {

}


