//! Optimized alpha blending routines based on libwebp
//! 
//! https://github.com/webmproject/libwebp/blob/e4f7a9f0c7c9fbfae1568bc7fa5c94b989b50872/src/demux/anim_decode.c#L215-L267

// Blend a single channel of 'src' over 'dst', given their alpha channel values.
// 'src' and 'dst' are assumed to be NOT pre-multiplied by alpha.
uint8_t BlendChannelNonPremult(uint32_t src, uint8_t src_a,
                                      uint32_t dst, uint8_t dst_a,
                                      uint32_t scale, int shift) {
  const uint8_t src_channel = (src >> shift) & 0xff;
  const uint8_t dst_channel = (dst >> shift) & 0xff;
  const uint32_t blend_unscaled = src_channel * src_a + dst_channel * dst_a;
  assert(blend_unscaled < (1ULL << 32) / scale);
  return (blend_unscaled * scale) >> CHANNEL_SHIFT(3);
}

// Blend 'src' over 'dst' assuming they are NOT pre-multiplied by alpha.
uint32_t BlendPixelNonPremult(uint32_t src, uint32_t dst) {
  const uint8_t src_a = (src >> CHANNEL_SHIFT(3)) & 0xff;

  if (src_a == 0) {
    return dst;
  } else {
    const uint8_t dst_a = (dst >> CHANNEL_SHIFT(3)) & 0xff;
    // This is the approximate integer arithmetic for the actual formula:
    // dst_factor_a = (dst_a * (255 - src_a)) / 255.
    const uint8_t dst_factor_a = (dst_a * (256 - src_a)) >> 8;
    const uint8_t blend_a = src_a + dst_factor_a;
    const uint32_t scale = (1UL << 24) / blend_a;

    const uint8_t blend_r = BlendChannelNonPremult(
        src, src_a, dst, dst_factor_a, scale, CHANNEL_SHIFT(0));
    const uint8_t blend_g = BlendChannelNonPremult(
        src, src_a, dst, dst_factor_a, scale, CHANNEL_SHIFT(1));
    const uint8_t blend_b = BlendChannelNonPremult(
        src, src_a, dst, dst_factor_a, scale, CHANNEL_SHIFT(2));
    assert(src_a + dst_factor_a < 256);

    return ((uint32_t)blend_r << CHANNEL_SHIFT(0)) |
           ((uint32_t)blend_g << CHANNEL_SHIFT(1)) |
           ((uint32_t)blend_b << CHANNEL_SHIFT(2)) |
           ((uint32_t)blend_a << CHANNEL_SHIFT(3));
  }
}

// Blend 'num_pixels' in 'src' over 'dst' assuming they are NOT pre-multiplied
// by alpha.
void BlendPixelRowNonPremult(uint32_t* const src,
                                    const uint32_t* const dst, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint8_t src_alpha = (src[i] >> CHANNEL_SHIFT(3)) & 0xff;
    if (src_alpha != 0xff) {
      src[i] = BlendPixelNonPremult(src[i], dst[i]);
    }
  }
}


pub(crate) fn do_alpha_blending(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn do_alpha_blending_reference(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
        let canvas_alpha = f64::from(canvas[3]);
        let buffer_alpha = f64::from(buffer[3]);
        let blend_alpha_f64 = buffer_alpha + canvas_alpha * (1.0 - buffer_alpha / 255.0);
        //value should be between 0 and 255, this truncates the fractional part
        let blend_alpha: u8 = blend_alpha_f64 as u8;
    
        let blend_rgb: [u8; 3] = if blend_alpha == 0 {
            [0, 0, 0]
        } else {
            let mut rgb = [0u8; 3];
            for i in 0..3 {
                let canvas_f64 = f64::from(canvas[i]);
                let buffer_f64 = f64::from(buffer[i]);
    
                let val = (buffer_f64 * buffer_alpha
                    + canvas_f64 * canvas_alpha * (1.0 - buffer_alpha / 255.0))
                    / blend_alpha_f64;
                //value should be between 0 and 255, this truncates the fractional part
                rgb[i] = val as u8;
            }
    
            rgb
        };
    
        [blend_rgb[0], blend_rgb[1], blend_rgb[2], blend_alpha]
    }

    #[test]
    fn alpha_blending_optimization() {
        for r1 in 0..u8::MAX {
            for a1 in 0..u8::MAX {
                for r2 in 0..u8::MAX {
                    for a2 in 0..u8::MAX {
                        let opt = do_alpha_blending([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        let slow = do_alpha_blending_reference([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        if opt != slow {
                            panic!("Mismatch in results! opt: {opt:?}, slow: {slow:?}, blended values: [{r1}, 0, 0, {a1}], [{r2}, 0, 0, {a2}]");
                        }
                    }
                }
            }
        }
    }
}
