//! Tests for the mux/demux/animation APIs.

use zenwebp::mux::{
    AnimationConfig, AnimationEncoder, BlendMethod, DisposeMethod, MuxFrame, WebPDemuxer, WebPMux,
};
use zenwebp::{EncodeRequest, EncoderConfig, LoopCount, PixelLayout, WebPDecoder};

/// Create a solid-color RGBA frame.
fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel = [r, g, b, a];
    pixel
        .iter()
        .cycle()
        .take((width * height * 4) as usize)
        .copied()
        .collect()
}

/// Create a solid-color RGB frame.
fn solid_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    let pixel = [r, g, b];
    pixel
        .iter()
        .cycle()
        .take((width * height * 3) as usize)
        .copied()
        .collect()
}

// ============================================================================
// Demux tests
// ============================================================================

#[test]
fn demux_lossy_animated() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();
    let demuxer = WebPDemuxer::new(&data).unwrap();

    assert!(demuxer.is_animated());
    assert!(demuxer.num_frames() > 1);
    assert!(demuxer.canvas_width() > 0);
    assert!(demuxer.canvas_height() > 0);

    // Verify we can iterate all frames
    let frames: Vec<_> = demuxer.frames().collect();
    assert_eq!(frames.len(), demuxer.num_frames() as usize);

    // Verify frame metadata is reasonable
    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.bitstream.is_empty());
        assert!(frame.is_lossy);
    }

    // Verify 1-based indexing
    assert!(demuxer.frame(0).is_none());
    assert!(demuxer.frame(1).is_some());
    assert!(demuxer.frame(demuxer.num_frames()).is_some());
    assert!(demuxer.frame(demuxer.num_frames() + 1).is_none());
}

#[test]
fn demux_lossless_animated() {
    let data = std::fs::read("tests/images/animated/random_lossless.webp").unwrap();
    let demuxer = WebPDemuxer::new(&data).unwrap();

    assert!(demuxer.is_animated());
    assert!(demuxer.num_frames() > 1);

    let frames: Vec<_> = demuxer.frames().collect();
    assert_eq!(frames.len(), demuxer.num_frames() as usize);

    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.bitstream.is_empty());
        assert!(!frame.is_lossy);
    }
}

#[test]
fn demux_matches_decoder_metadata() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    let demuxer = WebPDemuxer::new(&data).unwrap();
    let decoder = WebPDecoder::new(&data).unwrap();

    assert_eq!(demuxer.canvas_width(), decoder.dimensions().0);
    assert_eq!(demuxer.canvas_height(), decoder.dimensions().1);
    assert_eq!(demuxer.num_frames(), decoder.num_frames());
    assert_eq!(demuxer.loop_count(), decoder.loop_count());
}

#[test]
fn demux_simple_lossy() {
    // Encode a simple lossy image
    let pixels = solid_rgb(64, 64, 128, 64, 192);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 64, 64)
        .encode()
        .unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert!(!demuxer.is_animated());
    assert_eq!(demuxer.num_frames(), 1);
    assert_eq!(demuxer.canvas_width(), 64);
    assert_eq!(demuxer.canvas_height(), 64);

    let frame = demuxer.frame(1).unwrap();
    assert_eq!(frame.frame_num, 1);
    assert_eq!(frame.x_offset, 0);
    assert_eq!(frame.y_offset, 0);
    assert!(frame.is_lossy);
    assert!(!frame.bitstream.is_empty());
}

#[test]
fn demux_simple_lossless() {
    // Encode a simple lossless image
    let pixels = solid_rgba(32, 32, 128, 64, 192, 255);
    let config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgba8, 32, 32)
        .encode()
        .unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert!(!demuxer.is_animated());
    assert_eq!(demuxer.num_frames(), 1);
    assert_eq!(demuxer.canvas_width(), 32);
    assert_eq!(demuxer.canvas_height(), 32);

    let frame = demuxer.frame(1).unwrap();
    assert!(!frame.is_lossy);
    assert!(!frame.bitstream.is_empty());
}

// ============================================================================
// Mux tests
// ============================================================================

#[test]
fn mux_single_image_simple() {
    // Encode a frame, then wrap it in a mux container
    let pixels = solid_rgb(64, 64, 200, 100, 50);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 64, 64)
        .encode()
        .unwrap();

    // Demux to get raw bitstream
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let frame = demuxer.frame(1).unwrap();

    // Mux it back
    let mut mux = WebPMux::new(64, 64);
    mux.set_image(MuxFrame {
        x_offset: 0,
        y_offset: 0,
        width: 64,
        height: 64,
        duration_ms: 0,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: frame.bitstream.to_vec(),
        alpha_data: None,
        is_lossless: false,
    });
    let assembled = mux.assemble().unwrap();

    // Verify it decodes
    let decoder = WebPDecoder::new(&assembled).unwrap();
    assert_eq!(decoder.dimensions(), (64, 64));
}

#[test]
fn mux_single_image_with_metadata() {
    let pixels = solid_rgb(32, 32, 100, 100, 100);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 32, 32)
        .encode()
        .unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let frame = demuxer.frame(1).unwrap();

    let icc_data = vec![1, 2, 3, 4, 5]; // fake ICC

    let mut mux = WebPMux::new(32, 32);
    mux.set_icc_profile(icc_data.clone());
    mux.set_image(MuxFrame {
        x_offset: 0,
        y_offset: 0,
        width: 32,
        height: 32,
        duration_ms: 0,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: frame.bitstream.to_vec(),
        alpha_data: None,
        is_lossless: false,
    });
    let assembled = mux.assemble().unwrap();

    // Verify ICC round-trips
    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.icc_profile().unwrap(), &icc_data[..]);

    // Verify it still decodes
    let decoder = WebPDecoder::new(&assembled).unwrap();
    assert_eq!(decoder.dimensions(), (32, 32));
}

#[test]
fn mux_from_data_roundtrip() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    // Parse into mux
    let mux = WebPMux::from_data(&data).unwrap();

    // Reassemble
    let assembled = mux.assemble().unwrap();

    // Verify it decodes with same metadata
    let decoder = WebPDecoder::new(&assembled).unwrap();
    let original_decoder = WebPDecoder::new(&data).unwrap();
    assert_eq!(decoder.dimensions(), original_decoder.dimensions());
    assert_eq!(decoder.num_frames(), original_decoder.num_frames());
}

// ============================================================================
// Animation encoder tests
// ============================================================================

#[test]
fn animation_encode_decode_lossy_roundtrip() {
    let config = AnimationConfig {
        background_color: [0, 0, 0, 0],
        loop_count: LoopCount::Forever,
        ..Default::default()
    };
    let mut anim = AnimationEncoder::new(64, 64, config).unwrap();

    let frame_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(0);

    // 3 differently-colored frames
    let frame1 = solid_rgb(64, 64, 255, 0, 0);
    let frame2 = solid_rgb(64, 64, 0, 255, 0);
    let frame3 = solid_rgb(64, 64, 0, 0, 255);

    anim.add_frame(&frame1, PixelLayout::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgb8, 100, &frame_config)
        .unwrap();
    anim.add_frame(&frame3, PixelLayout::Rgb8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(100).unwrap();

    // Decode and verify
    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.dimensions(), (64, 64));
    assert!(decoder.is_animated());
    assert_eq!(decoder.num_frames(), 3);
    assert_eq!(decoder.loop_count(), LoopCount::Forever);

    // Read all frames
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..3 {
        decoder.read_frame(&mut buf).unwrap();
    }
}

#[test]
fn animation_encode_decode_lossless_roundtrip() {
    let config = AnimationConfig {
        background_color: [0, 0, 0, 255],
        loop_count: LoopCount::Times(std::num::NonZeroU16::new(3).unwrap()),
        ..Default::default()
    };
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();

    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);

    let frame1 = solid_rgba(32, 32, 255, 0, 0, 255);
    let frame2 = solid_rgba(32, 32, 0, 255, 0, 255);

    anim.add_frame(&frame1, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgba8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(200).unwrap();

    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.dimensions(), (32, 32));
    assert!(decoder.is_animated());
    assert_eq!(decoder.num_frames(), 2);
    assert_eq!(
        decoder.loop_count(),
        LoopCount::Times(std::num::NonZeroU16::new(3).unwrap())
    );

    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];

    let d1 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d1, 200); // first frame duration = timestamp[1] - timestamp[0]

    let d2 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d2, 200); // last frame duration from finalize
}

#[test]
fn animation_frame_durations_from_timestamps() {
    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(16, 16, config).unwrap();

    let frame_config = EncoderConfig::new_lossy().with_quality(50.0).with_method(0);
    let pixels = solid_rgb(16, 16, 128, 128, 128);

    // timestamps: 0, 50, 200
    // durations should be: 50, 150, last_frame_duration
    anim.add_frame(&pixels, PixelLayout::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&pixels, PixelLayout::Rgb8, 50, &frame_config)
        .unwrap();
    anim.add_frame(&pixels, PixelLayout::Rgb8, 200, &frame_config)
        .unwrap();

    let webp = anim.finalize(300).unwrap();

    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.num_frames(), 3);

    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];

    let d1 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d1, 50);

    let d2 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d2, 150);

    let d3 = decoder.read_frame(&mut buf).unwrap();
    assert_eq!(d3, 300);
}

#[test]
fn animation_with_metadata() {
    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(16, 16, config).unwrap();

    let icc_data = vec![10, 20, 30, 40];
    let exif_data = vec![50, 60, 70];
    anim.icc_profile(icc_data.clone());
    anim.exif(exif_data.clone());

    let frame_config = EncoderConfig::new_lossy().with_quality(50.0).with_method(0);
    let pixels = solid_rgb(16, 16, 100, 100, 100);
    anim.add_frame(&pixels, PixelLayout::Rgb8, 0, &frame_config)
        .unwrap();

    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.icc_profile().unwrap(), &icc_data[..]);
    assert_eq!(demuxer.exif().unwrap(), &exif_data[..]);
    assert!(demuxer.xmp().is_none());
}

// ============================================================================
// Demux -> Mux roundtrip
// ============================================================================

#[test]
fn demux_mux_roundtrip_lossy() {
    let data = std::fs::read("tests/images/animated/random_lossy.webp").unwrap();

    // Demux
    let demuxer = WebPDemuxer::new(&data).unwrap();
    let original_frames: Vec<_> = demuxer.frames().collect();

    // Mux
    let mut mux = WebPMux::new(demuxer.canvas_width(), demuxer.canvas_height());
    mux.set_animation(demuxer.background_color(), demuxer.loop_count());

    for frame in &original_frames {
        mux.push_frame(MuxFrame {
            x_offset: frame.x_offset,
            y_offset: frame.y_offset,
            width: frame.width,
            height: frame.height,
            duration_ms: frame.duration_ms,
            dispose: frame.dispose,
            blend: frame.blend,
            bitstream: frame.bitstream.to_vec(),
            alpha_data: frame.alpha_data.map(|d| d.to_vec()),
            is_lossless: !frame.is_lossy,
        })
        .unwrap();
    }

    let assembled = mux.assemble().unwrap();

    // Verify roundtrip
    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.canvas_width(), demuxer.canvas_width());
    assert_eq!(demuxer2.canvas_height(), demuxer.canvas_height());
    assert_eq!(demuxer2.num_frames(), demuxer.num_frames());
    assert_eq!(demuxer2.loop_count(), demuxer.loop_count());

    // Verify each frame's metadata matches
    for (orig, new) in original_frames.iter().zip(demuxer2.frames()) {
        assert_eq!(orig.x_offset, new.x_offset);
        assert_eq!(orig.y_offset, new.y_offset);
        assert_eq!(orig.width, new.width);
        assert_eq!(orig.height, new.height);
        assert_eq!(orig.duration_ms, new.duration_ms);
        assert_eq!(orig.dispose, new.dispose);
        assert_eq!(orig.blend, new.blend);
        assert_eq!(orig.bitstream.len(), new.bitstream.len());
    }

    // Verify it decodes
    let mut decoder = WebPDecoder::new(&assembled).unwrap();
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..demuxer2.num_frames() {
        decoder.read_frame(&mut buf).unwrap();
    }
}

#[test]
fn demux_mux_roundtrip_lossless() {
    let data = std::fs::read("tests/images/animated/random_lossless.webp").unwrap();

    let demuxer = WebPDemuxer::new(&data).unwrap();
    let original_frames: Vec<_> = demuxer.frames().collect();

    let mut mux = WebPMux::new(demuxer.canvas_width(), demuxer.canvas_height());
    mux.set_animation(demuxer.background_color(), demuxer.loop_count());

    for frame in &original_frames {
        mux.push_frame(MuxFrame {
            x_offset: frame.x_offset,
            y_offset: frame.y_offset,
            width: frame.width,
            height: frame.height,
            duration_ms: frame.duration_ms,
            dispose: frame.dispose,
            blend: frame.blend,
            bitstream: frame.bitstream.to_vec(),
            alpha_data: frame.alpha_data.map(|d| d.to_vec()),
            is_lossless: !frame.is_lossy,
        })
        .unwrap();
    }

    let assembled = mux.assemble().unwrap();

    let demuxer2 = WebPDemuxer::new(&assembled).unwrap();
    assert_eq!(demuxer2.num_frames(), demuxer.num_frames());

    // Verify it decodes
    let mut decoder = WebPDecoder::new(&assembled).unwrap();
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..demuxer2.num_frames() {
        decoder.read_frame(&mut buf).unwrap();
    }
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn mux_no_frames_error() {
    let mux = WebPMux::new(100, 100);
    assert!(mux.assemble().is_err());
}

#[test]
fn mux_animated_no_frames_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);
    assert!(mux.assemble().is_err());
}

#[test]
fn mux_frame_outside_canvas_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);

    let result = mux.push_frame(MuxFrame {
        x_offset: 90,
        y_offset: 0,
        width: 20, // 90 + 20 = 110 > 100
        height: 10,
        duration_ms: 100,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: vec![0],
        alpha_data: None,
        is_lossless: true,
    });
    assert!(result.is_err());
}

#[test]
fn mux_odd_offset_error() {
    let mut mux = WebPMux::new(100, 100);
    mux.set_animation([0; 4], LoopCount::Forever);

    let result = mux.push_frame(MuxFrame {
        x_offset: 3, // odd!
        y_offset: 0,
        width: 10,
        height: 10,
        duration_ms: 100,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: vec![0],
        alpha_data: None,
        is_lossless: true,
    });
    assert!(result.is_err());
}

#[test]
fn animation_encoder_invalid_dimensions() {
    let config = AnimationConfig::default();
    assert!(AnimationEncoder::new(0, 100, config.clone()).is_err());
    assert!(AnimationEncoder::new(100, 0, config.clone()).is_err());
    assert!(AnimationEncoder::new(20000, 100, config).is_err());
}

#[test]
fn demux_invalid_data() {
    assert!(WebPDemuxer::new(&[]).is_err());
    assert!(WebPDemuxer::new(&[0; 12]).is_err());
    assert!(WebPDemuxer::new(b"not a webp file at all!!").is_err());
}

// ============================================================================
// Sub-frame delta compression tests
// ============================================================================

/// Create an RGBA frame that is mostly `base_color` but has a small patch at
/// `(px, py)` of size `(pw, ph)` filled with `patch_color`.
#[allow(clippy::too_many_arguments)]
fn patched_rgba(
    width: u32,
    height: u32,
    base_color: [u8; 4],
    px: u32,
    py: u32,
    pw: u32,
    ph: u32,
    patch_color: [u8; 4],
) -> Vec<u8> {
    let mut buf = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let off = ((y * width + x) * 4) as usize;
            let color = if x >= px && x < px + pw && y >= py && y < py + ph {
                patch_color
            } else {
                base_color
            };
            buf[off..off + 4].copy_from_slice(&color);
        }
    }
    buf
}

#[test]
fn subframe_reduces_file_size() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);

    let base = solid_rgba(64, 64, 100, 100, 100, 255);
    // Only change a 4x4 patch in each frame
    let frame2 = patched_rgba(
        64,
        64,
        [100, 100, 100, 255],
        10,
        10,
        4,
        4,
        [200, 50, 50, 255],
    );
    let frame3 = patched_rgba(
        64,
        64,
        [100, 100, 100, 255],
        10,
        10,
        4,
        4,
        [50, 200, 50, 255],
    );

    // With optimization (default)
    let config_opt = AnimationConfig::default();
    let mut anim_opt = AnimationEncoder::new(64, 64, config_opt).unwrap();
    anim_opt
        .add_frame(&base, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim_opt
        .add_frame(&frame2, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    anim_opt
        .add_frame(&frame3, PixelLayout::Rgba8, 200, &frame_config)
        .unwrap();
    let webp_opt = anim_opt.finalize(100).unwrap();

    // Without optimization
    let config_no = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut anim_no = AnimationEncoder::new(64, 64, config_no).unwrap();
    anim_no
        .add_frame(&base, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim_no
        .add_frame(&frame2, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    anim_no
        .add_frame(&frame3, PixelLayout::Rgba8, 200, &frame_config)
        .unwrap();
    let webp_no = anim_no.finalize(100).unwrap();

    assert!(
        webp_opt.len() < webp_no.len(),
        "optimized ({}) should be smaller than unoptimized ({})",
        webp_opt.len(),
        webp_no.len()
    );
}

#[test]
fn identical_frames_produce_tiny_subframe() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);
    let pixels = solid_rgba(64, 64, 128, 64, 192, 255);

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(64, 64, config).unwrap();
    anim.add_frame(&pixels, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&pixels, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    // Demux and check second frame is 1x1
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.num_frames(), 2);
    let f2 = demuxer.frame(2).unwrap();
    assert_eq!(f2.width, 1);
    assert_eq!(f2.height, 1);
}

#[test]
fn fully_different_frames_produce_full_canvas() {
    let frame_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(0);
    let frame1 = solid_rgb(32, 32, 255, 0, 0);
    let frame2 = solid_rgb(32, 32, 0, 255, 0);

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();
    anim.add_frame(&frame1, PixelLayout::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgb8, 100, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.num_frames(), 2);
    // Both frames should be full canvas since every pixel differs
    let f1 = demuxer.frame(1).unwrap();
    let f2 = demuxer.frame(2).unwrap();
    assert_eq!(f1.width, 32);
    assert_eq!(f1.height, 32);
    assert_eq!(f2.width, 32);
    assert_eq!(f2.height, 32);
}

#[test]
fn subframe_offsets_are_even() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);

    let base = solid_rgba(64, 64, 0, 0, 0, 255);
    // Change a single pixel at odd coordinates (3, 5)
    let mut frame2 = base.clone();
    let off = ((5 * 64 + 3) * 4) as usize;
    frame2[off] = 255;

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(64, 64, config).unwrap();
    anim.add_frame(&base, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let f2 = demuxer.frame(2).unwrap();
    assert_eq!(f2.x_offset % 2, 0, "x_offset {} must be even", f2.x_offset);
    assert_eq!(f2.y_offset % 2, 0, "y_offset {} must be even", f2.y_offset);
    // The sub-frame must cover the changed pixel at (3, 5)
    assert!(f2.x_offset <= 3);
    assert!(f2.y_offset <= 5);
    assert!(f2.x_offset + f2.width > 3);
    assert!(f2.y_offset + f2.height > 5);
}

#[test]
fn subframe_lossless_roundtrip_correctness() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);

    // Frame 1: solid green
    let frame1 = solid_rgba(32, 32, 0, 200, 0, 255);
    // Frame 2: green with a red patch at (4, 4, 8, 8)
    let frame2 = patched_rgba(32, 32, [0, 200, 0, 255], 4, 4, 8, 8, [200, 0, 0, 255]);
    // Frame 3: same as frame 2 (identical, should produce 1x1)
    let frame3 = frame2.clone();

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();
    anim.add_frame(&frame1, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    anim.add_frame(&frame3, PixelLayout::Rgba8, 200, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    // Decode and verify pixel content matches
    let mut decoder = WebPDecoder::new(&webp).unwrap();
    assert_eq!(decoder.num_frames(), 3);
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];

    // Frame 1: all green
    decoder.read_frame(&mut buf).unwrap();
    for y in 0..32u32 {
        for x in 0..32u32 {
            let off = ((y * 32 + x) * 4) as usize;
            assert_eq!(
                &buf[off..off + 4],
                &[0, 200, 0, 255],
                "frame 1 pixel ({x},{y}) mismatch"
            );
        }
    }

    // Frame 2: green with red patch
    decoder.read_frame(&mut buf).unwrap();
    for y in 0..32u32 {
        for x in 0..32u32 {
            let off = ((y * 32 + x) * 4) as usize;
            let expected = if (4..12).contains(&x) && (4..12).contains(&y) {
                [200, 0, 0, 255]
            } else {
                [0, 200, 0, 255]
            };
            assert_eq!(
                &buf[off..off + 4],
                &expected,
                "frame 2 pixel ({x},{y}) mismatch"
            );
        }
    }

    // Frame 3: identical to frame 2
    decoder.read_frame(&mut buf).unwrap();
    for y in 0..32u32 {
        for x in 0..32u32 {
            let off = ((y * 32 + x) * 4) as usize;
            let expected = if (4..12).contains(&x) && (4..12).contains(&y) {
                [200, 0, 0, 255]
            } else {
                [0, 200, 0, 255]
            };
            assert_eq!(
                &buf[off..off + 4],
                &expected,
                "frame 3 pixel ({x},{y}) mismatch"
            );
        }
    }
}

#[test]
fn add_frame_advanced_invalidates_canvas() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);
    let pixels = solid_rgba(32, 32, 100, 100, 100, 255);

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();

    // First add_frame sets canvas tracking
    anim.add_frame(&pixels, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();

    // add_frame_advanced should invalidate canvas
    anim.add_frame_advanced(
        &pixels,
        PixelLayout::Rgba8,
        32,
        32,
        0,
        0,
        100,
        &frame_config,
        DisposeMethod::Background,
        BlendMethod::Overwrite,
    )
    .unwrap();

    // Next add_frame should produce full-canvas frame (no delta optimization)
    anim.add_frame(&pixels, PixelLayout::Rgba8, 200, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.num_frames(), 3);
    // Frame 3 should be full canvas, not a 1x1 sub-frame
    let f3 = demuxer.frame(3).unwrap();
    assert_eq!(
        f3.width, 32,
        "frame 3 should be full canvas after invalidation"
    );
    assert_eq!(f3.height, 32);
}

#[test]
fn minimize_size_false_disables_optimization() {
    let frame_config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);
    let pixels = solid_rgba(32, 32, 128, 64, 192, 255);

    let config = AnimationConfig {
        minimize_size: false,
        ..Default::default()
    };
    let mut anim = AnimationEncoder::new(32, 32, config).unwrap();
    anim.add_frame(&pixels, PixelLayout::Rgba8, 0, &frame_config)
        .unwrap();
    // Identical frame, but optimization is disabled — should be full canvas
    anim.add_frame(&pixels, PixelLayout::Rgba8, 100, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.num_frames(), 2);
    let f2 = demuxer.frame(2).unwrap();
    assert_eq!(
        f2.width, 32,
        "should be full canvas with minimize_size=false"
    );
    assert_eq!(f2.height, 32);
}

#[test]
fn subframe_rgb_input_works() {
    // Test with RGB (non-alpha) input to verify the Rgb8→RGBA→Rgb8 path
    let frame_config = EncoderConfig::new_lossy().with_quality(75.0).with_method(0);

    let base = solid_rgb(64, 64, 100, 100, 100);
    // Change a small region
    let frame2_rgba = patched_rgba(64, 64, [100, 100, 100, 255], 8, 8, 8, 8, [200, 50, 50, 255]);
    // Convert back to RGB for input
    let frame2: Vec<u8> = frame2_rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();

    let config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(64, 64, config).unwrap();
    anim.add_frame(&base, PixelLayout::Rgb8, 0, &frame_config)
        .unwrap();
    anim.add_frame(&frame2, PixelLayout::Rgb8, 100, &frame_config)
        .unwrap();
    let webp = anim.finalize(100).unwrap();

    // Demux and verify sub-frame is smaller than full canvas
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.num_frames(), 2);
    let f2 = demuxer.frame(2).unwrap();
    // Sub-frame should be smaller than full 64x64
    assert!(
        f2.width < 64 || f2.height < 64,
        "expected sub-frame, got {}x{}",
        f2.width,
        f2.height
    );

    // Verify it decodes
    let mut decoder = WebPDecoder::new(&webp).unwrap();
    let buf_size = decoder.output_buffer_size().unwrap();
    let mut buf = vec![0u8; buf_size];
    for _ in 0..2 {
        decoder.read_frame(&mut buf).unwrap();
    }
}

// ============================================================================
// Metadata extraction tests
// ============================================================================

/// Encode a WebP with ICC, EXIF, and XMP metadata using the mux layer,
/// then verify the decoder's ImageInfo returns all three.
#[test]
fn decoder_info_extracts_icc_exif_xmp() {
    let pixels = solid_rgb(32, 32, 100, 150, 200);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 32, 32)
        .encode()
        .unwrap();

    // Inject all three metadata types via mux
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    let frame = demuxer.frame(1).unwrap();

    let icc_data = vec![0x49, 0x43, 0x43, 0x50]; // "ICCP"
    let exif_data = vec![0x45, 0x78, 0x69, 0x66, 0x00, 0x00]; // "Exif\0\0"
    let xmp_data = b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta/>".to_vec();

    let mut mux = WebPMux::new(32, 32);
    mux.set_icc_profile(icc_data.clone());
    mux.set_exif(exif_data.clone());
    mux.set_xmp(xmp_data.clone());
    mux.set_image(MuxFrame {
        x_offset: 0,
        y_offset: 0,
        width: 32,
        height: 32,
        duration_ms: 0,
        dispose: DisposeMethod::None,
        blend: BlendMethod::Overwrite,
        bitstream: frame.bitstream.to_vec(),
        alpha_data: None,
        is_lossless: false,
    });
    let assembled = mux.assemble().unwrap();

    // Verify via WebPDecoder.info()
    let decoder = WebPDecoder::new(&assembled).unwrap();
    let info = decoder.info();
    assert_eq!(info.icc_profile.as_deref(), Some(icc_data.as_slice()));
    assert_eq!(info.exif.as_deref(), Some(exif_data.as_slice()));
    assert_eq!(info.xmp.as_deref(), Some(xmp_data.as_slice()));
}

/// Verify ImageInfo returns None for all metadata when no chunks are present.
#[test]
fn decoder_info_no_metadata_returns_none() {
    let pixels = solid_rgb(16, 16, 50, 100, 150);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 16, 16)
        .encode()
        .unwrap();

    let decoder = WebPDecoder::new(&webp).unwrap();
    let info = decoder.info();
    assert!(info.icc_profile.is_none(), "expected no ICC profile");
    assert!(info.exif.is_none(), "expected no EXIF data");
    assert!(info.xmp.is_none(), "expected no XMP data");
}

/// Round-trip metadata through encode (with_metadata) -> decode (info).
#[test]
fn metadata_roundtrip_via_encode_request() {
    use zenwebp::ImageMetadata;

    let icc_data = vec![1u8; 128]; // 128-byte fake ICC
    let exif_data = vec![0x45, 0x78, 0x69, 0x66, 0x00, 0x00, 0xAA, 0xBB];
    let xmp_data = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>test</x:xmpmeta>".to_vec();

    let meta = ImageMetadata::new()
        .with_icc_profile(&icc_data)
        .with_exif(&exif_data)
        .with_xmp(&xmp_data);

    let pixels = solid_rgb(16, 16, 200, 100, 50);
    let config = EncoderConfig::new_lossy().with_quality(80.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 16, 16)
        .with_metadata(meta)
        .encode()
        .unwrap();

    // Verify via WebPDecoder
    let decoder = WebPDecoder::new(&webp).unwrap();
    let info = decoder.info();
    assert_eq!(info.icc_profile.as_deref(), Some(icc_data.as_slice()));
    assert_eq!(info.exif.as_deref(), Some(exif_data.as_slice()));
    assert_eq!(info.xmp.as_deref(), Some(xmp_data.as_slice()));

    // Also verify via demuxer for cross-checking
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.icc_profile().unwrap(), &icc_data[..]);
    assert_eq!(demuxer.exif().unwrap(), &exif_data[..]);
    assert_eq!(demuxer.xmp().unwrap(), &xmp_data[..]);

    // Also verify via the convenience metadata module
    let extracted_icc = zenwebp::metadata::icc_profile(&webp).unwrap();
    assert_eq!(extracted_icc.as_deref(), Some(icc_data.as_slice()));
    let extracted_exif = zenwebp::metadata::exif(&webp).unwrap();
    assert_eq!(extracted_exif.as_deref(), Some(exif_data.as_slice()));
    let extracted_xmp = zenwebp::metadata::xmp(&webp).unwrap();
    assert_eq!(extracted_xmp.as_deref(), Some(xmp_data.as_slice()));
}

/// Large ICC profile (>64KB) round-trips correctly.
#[test]
fn large_icc_profile_roundtrip() {
    use zenwebp::ImageMetadata;

    // Create a 100KB ICC profile
    let icc_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    let meta = ImageMetadata::new().with_icc_profile(&icc_data);

    let pixels = solid_rgb(8, 8, 128, 128, 128);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 8, 8)
        .with_metadata(meta)
        .encode()
        .unwrap();

    // Verify via decoder
    let decoder = WebPDecoder::new(&webp).unwrap();
    let info = decoder.info();
    let extracted = info
        .icc_profile
        .as_ref()
        .expect("ICC profile should be present");
    assert_eq!(extracted.len(), 100_000);
    assert_eq!(extracted, &icc_data);

    // Verify via demuxer
    let demuxer = WebPDemuxer::new(&webp).unwrap();
    assert_eq!(demuxer.icc_profile().unwrap(), &icc_data[..]);
}

/// Lossless WebP with metadata.
#[test]
fn lossless_webp_metadata_roundtrip() {
    use zenwebp::ImageMetadata;

    let icc_data = vec![42u8; 64];
    let exif_data = vec![0x45, 0x78, 0x69, 0x66, 0x00, 0x00];

    let meta = ImageMetadata::new()
        .with_icc_profile(&icc_data)
        .with_exif(&exif_data);

    let pixels = solid_rgba(16, 16, 100, 200, 50, 255);
    let config = EncoderConfig::new_lossy()
        .with_quality(100.0)
        .with_lossless(true);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgba8, 16, 16)
        .with_metadata(meta)
        .encode()
        .unwrap();

    let decoder = WebPDecoder::new(&webp).unwrap();
    let info = decoder.info();
    assert_eq!(info.icc_profile.as_deref(), Some(icc_data.as_slice()));
    assert_eq!(info.exif.as_deref(), Some(exif_data.as_slice()));
    assert!(info.xmp.is_none());
}

/// ImageInfo::from_bytes extracts metadata in a single call.
#[test]
fn image_info_from_bytes_extracts_metadata() {
    use zenwebp::ImageMetadata;

    let icc_data = vec![7u8; 32];
    let xmp_data = b"<rdf:RDF>test</rdf:RDF>".to_vec();

    let meta = ImageMetadata::new()
        .with_icc_profile(&icc_data)
        .with_xmp(&xmp_data);

    let pixels = solid_rgb(8, 8, 64, 128, 192);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 8, 8)
        .with_metadata(meta)
        .encode()
        .unwrap();

    let info = zenwebp::ImageInfo::from_bytes(&webp).unwrap();
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(info.icc_profile.as_deref(), Some(icc_data.as_slice()));
    assert!(info.exif.is_none());
    assert_eq!(info.xmp.as_deref(), Some(xmp_data.as_slice()));
}

/// Metadata embed/remove convenience functions work correctly.
#[test]
fn metadata_embed_and_remove() {
    let pixels = solid_rgb(8, 8, 100, 100, 100);
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp = EncodeRequest::new(&config, &pixels, PixelLayout::Rgb8, 8, 8)
        .encode()
        .unwrap();

    // Initially no metadata
    assert!(zenwebp::metadata::icc_profile(&webp).unwrap().is_none());
    assert!(zenwebp::metadata::exif(&webp).unwrap().is_none());
    assert!(zenwebp::metadata::xmp(&webp).unwrap().is_none());

    // Embed ICC
    let icc_data = vec![1, 2, 3, 4, 5];
    let with_icc = zenwebp::metadata::embed_icc(&webp, &icc_data).unwrap();
    assert_eq!(
        zenwebp::metadata::icc_profile(&with_icc)
            .unwrap()
            .as_deref(),
        Some(icc_data.as_slice())
    );

    // Embed all three at once
    let meta = zenwebp::ImageMetadata::new()
        .with_icc_profile(&icc_data)
        .with_exif(b"exif-bytes")
        .with_xmp(b"xmp-bytes");
    let with_all = zenwebp::metadata::embed(&webp, &meta).unwrap();
    assert_eq!(
        zenwebp::metadata::icc_profile(&with_all)
            .unwrap()
            .as_deref(),
        Some(icc_data.as_slice())
    );
    assert_eq!(
        zenwebp::metadata::exif(&with_all).unwrap().as_deref(),
        Some(b"exif-bytes".as_slice())
    );
    assert_eq!(
        zenwebp::metadata::xmp(&with_all).unwrap().as_deref(),
        Some(b"xmp-bytes".as_slice())
    );

    // Remove ICC
    let without_icc = zenwebp::metadata::remove_icc(&with_all).unwrap();
    assert!(
        zenwebp::metadata::icc_profile(&without_icc)
            .unwrap()
            .is_none()
    );
    // EXIF and XMP should still be there
    assert_eq!(
        zenwebp::metadata::exif(&without_icc).unwrap().as_deref(),
        Some(b"exif-bytes".as_slice())
    );

    // The resulting file still decodes correctly
    let decoder = WebPDecoder::new(&with_all).unwrap();
    assert_eq!(decoder.dimensions(), (8, 8));
}
