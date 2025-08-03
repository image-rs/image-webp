//! Info related to vp8 used in both the decoder and the encoder
//!
//!

/// Different planes to be encoded/decoded in DCT coefficient decoding
/// in 13.3 of the spec
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Plane {
    /// The Y plane after decoding Y2
    YCoeff1 = 0,
    /// The Y2 plane (specifies the 0th coefficient of the other Y blocks)
    Y2 = 1,
    /// The U or V plane
    Chroma = 2,
    /// The Y plane when there is no Y2 plane
    YCoeff0 = 3,
}
