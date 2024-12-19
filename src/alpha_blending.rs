//! Optimized alpha blending routines based on libwebp
//!
//! https://github.com/webmproject/libwebp/blob/e4f7a9f0c7c9fbfae1568bc7fa5c94b989b50872/src/demux/anim_decode.c#L215-L267


#[cfg(target_endian = "big")]
fn channel_shift(i: u32) -> u32 {
    24 - (i) * 8
}

#[cfg(target_endian = "little")]
fn channel_shift(i: u32) -> u32 {
    i * 8
}

/// Blend a single channel of `src` over `dst`, given their alpha channel values.
/// `src` and `dst` are assumed to be NOT pre-multiplied by alpha.
fn blend_channel_nonpremult(
    src: u32,
    src_a: u8,
    dst: u32,
    dst_a: u8,
    scale: u32,
    shift: u32,
) -> u8 {
    let src_channel = ((src >> shift) & 0xff) as u8;
    let dst_channel = ((dst >> shift) & 0xff) as u8;
    let blend_unscaled = (src_channel as u32 * src_a as u32) + (dst_channel as u32 * dst_a as u32);
    // debug_assert!(blend_unscaled < (1u64 << 32) as u32 / scale); // TODO: fails for some reason?
    ((blend_unscaled * scale) >> channel_shift(3)) as u8
}

/// Blend `src` over `dst` assuming they are NOT pre-multiplied by alpha.
fn blend_pixel_nonpremult(src: u32, dst: u32) -> u32 {
    let src_a = ((src >> channel_shift(3)) & 0xff) as u8;

    if src_a == 0 {
        dst
    } else {
        let dst_a = ((dst >> channel_shift(3)) & 0xff) as u8;
        // Approximate integer arithmetic for: dst_factor_a = (dst_a * (255 - src_a)) / 255
        // libwebp used the following formula here:
        //let dst_factor_a = (dst_a as u32 * (256 - src_a as u32)) >> 8;
        // however, we've found that we can use a more precise approximation without losing performance:
        let dst_factor_a = div_by_255(dst_a as u32 * (255 - src_a as u32));
        let blend_a = src_a as u32 + dst_factor_a as u32;
        let scale = (1u32 << 24) / blend_a;

        let blend_r =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(0));
        let blend_g =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(1));
        let blend_b =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(2));
        debug_assert!(src_a as u32 + dst_factor_a < 256);

        ((blend_r as u32) << channel_shift(0))
            | ((blend_g as u32) << channel_shift(1))
            | ((blend_b as u32) << channel_shift(2))
            | ((blend_a as u32) << channel_shift(3))
    }
}

pub(crate) fn do_alpha_blending(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    blend_pixel_nonpremult(u32::from_le_bytes(buffer), u32::from_le_bytes(canvas)).to_le_bytes()
}

/// Divides by 255, rounding to nearest (as opposed to down, like regular integer division does).
/// TODO: cannot output 256, so the output is effecitively u8. Plumb that through the code.
//
// Sources:
// https://arxiv.org/pdf/2202.02864
// https://github.com/image-rs/image-webp/issues/119#issuecomment-2544007820
#[inline]
fn div_by_255(v: u32) -> u32 {
    (((v + 0x80) >> 8) + v + 0x80) >> 8
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
            for a1 in 11..u8::MAX {
                for r2 in 0..u8::MAX {
                    for a2 in 11..u8::MAX {
                        let opt = do_alpha_blending([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        let slow = do_alpha_blending_reference([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        // libwebp doesn't do exact blending and so we don't either
                        for (o, s) in opt.iter().zip(slow.iter()) {
                            if o.abs_diff(*s) > 3 {
                                panic!("Mismatch in results! opt: {opt:?}, slow: {slow:?}, blended values: [{r1}, 0, 0, {a1}], [{r2}, 0, 0, {a2}]");
                            }
                        }
                    }
                }
            }
        }
    }
}
