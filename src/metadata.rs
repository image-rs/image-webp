//! Standalone metadata convenience functions for WebP data.
//!
//! These functions operate on already-encoded WebP bytes, extracting or
//! embedding ICC, EXIF, and XMP metadata without decoding pixels.
//!
//! For embedding metadata during encoding, use
//! [`EncodeRequest::with_metadata`](crate::EncodeRequest::with_metadata) instead.
//!
//! # Example
//!
//! ```rust,no_run
//! use zenwebp::{metadata, ImageMetadata};
//!
//! # let webp_data: &[u8] = &[];
//! // Extract
//! let icc = metadata::icc_profile(webp_data)?;
//!
//! // Embed multiple in one pass
//! let icc_bytes = vec![0u8; 10];
//! let meta = ImageMetadata::new().with_icc_profile(&icc_bytes);
//! let with_meta = metadata::embed(webp_data, &meta)?;
//!
//! // Remove
//! let stripped = metadata::remove_icc(webp_data)?;
//! # Ok::<(), whereat::At<zenwebp::MuxError>>(())
//! ```

use alloc::vec::Vec;

use crate::encoder::ImageMetadata;
use crate::mux::{MuxResult, WebPDemuxer, WebPMux};

/// Extract the ICC color profile from WebP data, if present.
#[track_caller]
pub fn icc_profile(data: &[u8]) -> MuxResult<Option<Vec<u8>>> {
    let demuxer = WebPDemuxer::new(data)?;
    Ok(demuxer.icc_profile().map(|s| s.to_vec()))
}

/// Extract EXIF metadata from WebP data, if present.
#[track_caller]
pub fn exif(data: &[u8]) -> MuxResult<Option<Vec<u8>>> {
    let demuxer = WebPDemuxer::new(data)?;
    Ok(demuxer.exif().map(|s| s.to_vec()))
}

/// Extract XMP metadata from WebP data, if present.
#[track_caller]
pub fn xmp(data: &[u8]) -> MuxResult<Option<Vec<u8>>> {
    let demuxer = WebPDemuxer::new(data)?;
    Ok(demuxer.xmp().map(|s| s.to_vec()))
}

/// Embed metadata (ICC, EXIF, XMP) into WebP data in a single pass.
///
/// More efficient than calling [`embed_icc`], [`embed_exif`], and [`embed_xmp`]
/// separately, since it only parses and reassembles the RIFF container once.
#[track_caller]
pub fn embed(data: &[u8], metadata: &ImageMetadata<'_>) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    if let Some(icc) = metadata.icc_profile {
        mux.set_icc_profile(icc.to_vec());
    }
    if let Some(exif) = metadata.exif {
        mux.set_exif(exif.to_vec());
    }
    if let Some(xmp) = metadata.xmp {
        mux.set_xmp(xmp.to_vec());
    }
    mux.assemble()
}

/// Embed an ICC color profile into WebP data.
///
/// For embedding multiple metadata types at once, use [`embed`] instead.
#[track_caller]
pub fn embed_icc(data: &[u8], icc_profile: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.set_icc_profile(icc_profile.to_vec());
    mux.assemble()
}

/// Embed EXIF metadata into WebP data.
#[track_caller]
pub fn embed_exif(data: &[u8], exif: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.set_exif(exif.to_vec());
    mux.assemble()
}

/// Embed XMP metadata into WebP data.
#[track_caller]
pub fn embed_xmp(data: &[u8], xmp: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.set_xmp(xmp.to_vec());
    mux.assemble()
}

/// Remove ICC color profile from WebP data.
#[track_caller]
pub fn remove_icc(data: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.clear_icc_profile();
    mux.assemble()
}

/// Remove EXIF metadata from WebP data.
#[track_caller]
pub fn remove_exif(data: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.clear_exif();
    mux.assemble()
}

/// Remove XMP metadata from WebP data.
#[track_caller]
pub fn remove_xmp(data: &[u8]) -> MuxResult<Vec<u8>> {
    let mut mux = WebPMux::from_data(data)?;
    mux.clear_xmp();
    mux.assemble()
}
