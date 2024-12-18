//! Optimized alpha blending routines

/// Divides by 255, rounding to nearest (as opposed to down, like regular integer division does).
/// TODO: cannot output 256, so the output is effecitively u8. Plumb that through the code.
//
// Sources:
// https://arxiv.org/pdf/2202.02864
// https://github.com/image-rs/image-webp/issues/119#issuecomment-2544007820
#[inline]
fn div_by_255(v: u16) -> u16 {
    ((((v + 0x80) >> 8) + v + 0x80) >> 8).min(255) // TODO: .min(255) is almost certainly unnecessary
}

/// Rounding integer division (as opposed to truncating)
///
/// Works for the range of values we care about,
/// which is verified by tests against f64.round()
///
/// Breaks down when b is u16 and large due to overflow, but we don't need that!
fn rounding_div(a: u16, b: u8) -> u16 {
    let b_u16 = u16::from(b);
    (a + (b_u16 / 2)) / b_u16
}

pub(crate) fn do_alpha_blending(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    let canvas_alpha = u16::from(canvas[3]);
    let buffer_alpha = u16::from(buffer[3]);
    // repeating calculation extracted from two places and computed once. TODO: better name
    let shared_term = canvas_alpha - div_by_255(canvas_alpha * buffer_alpha);
    let blend_alpha = (buffer_alpha + shared_term) as u8;

    let blend_rgb: [u8; 3] = if blend_alpha == 0 {
        [0, 0, 0] // TODO: do we really need this special case?
    } else {
        let mut rgb = [0u8; 3];
        for i in 0..3 {
            let canvas_u16 = u16::from(canvas[i]);
            let buffer_u16 = u16::from(buffer[i]);

            // We split this arithmetic over two calls to rounding_div
            // so that the result does not overflow at any point.
            // All the values involved are 255 or lower, and 255*255 is 65,025
            // which leaves room to spare in u16 capacity of 65,535.
            // In fact, just enough room for our rounding integer division hack!
            let val = rounding_div(buffer_u16 * buffer_alpha, blend_alpha)
                + rounding_div(canvas_u16 * shared_term, blend_alpha);

            // we only need u16 for intermediate calculations, convert output back to u8
            rgb[i] = val as u8;
        }

        rgb
    };

    [blend_rgb[0], blend_rgb[1], blend_rgb[2], blend_alpha]
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
            for a1 in 15..u8::MAX {
                for r2 in 0..u8::MAX {
                    for a2 in 15..u8::MAX {
                        let opt = do_alpha_blending([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        let slow = do_alpha_blending_reference([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        // libwebp doesn't do exact blending and so we don't either
                        for (o, s) in opt.iter().zip(slow.iter()) {
                            if o.abs_diff(*s) > 5 {
                                panic!("Mismatch in results! opt: {opt:?}, slow: {slow:?}, blended values: [{r1}, 0, 0, {a1}], [{r2}, 0, 0, {a2}]");
                            }
                        }
                    }
                }
            }
        }
    }
}
