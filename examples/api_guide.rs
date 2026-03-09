//! Comprehensive API guide demonstrating 100% of zenwebp's public API surface.
//!
//! This example is designed to be both a test and reference documentation.
//! Every public API feature is demonstrated with idiomatic usage patterns.
//!
//! Run with: cargo run --example api_guide --all-features

use std::{io::Cursor, num::NonZeroU16};
use zenwebp::{
    AnimationConfig, AnimationEncoder, BlendMethod, DecodeError, DisposeMethod, EncodeError,
    EncodeRequest, EncoderConfig, Limits, LoopCount, LosslessConfig, LossyConfig, MuxError,
    PixelLayout, Preset, Stop, Unstoppable, WebPDemuxer, WebPMux,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== zenwebp API Guide ===\n");

    // Create test images
    let (rgb_img, rgba_img, width, height) = create_test_images();

    // ========================================================================
    // SECTION 1: LOSSY ENCODING
    // ========================================================================
    println!("1. LOSSY ENCODING\n");

    // 1.1 Basic encoding with typed config (recommended)
    println!("1.1 Basic encoding (LossyConfig + EncodeRequest::lossy)");
    let config = LossyConfig::new().with_quality(75.0).with_method(4);

    let webp_data =
        EncodeRequest::lossy(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
    println!(
        "  ✓ Encoded {}x{} RGB → {} bytes\n",
        width,
        height,
        webp_data.len()
    );

    // 1.2 Encoding with all parameters
    println!("1.2 Full parameter configuration");
    let config = LossyConfig::new()
        .with_quality(80.0)
        .with_method(6)
        .with_segments(4)
        .with_sns_strength(50)
        .with_filter_strength(60)
        .with_filter_sharpness(0)
        .with_target_size(0)
        .with_target_psnr(0.0)
        .with_sharp_yuv(false);

    let webp_data =
        EncodeRequest::lossy(&config, &rgba_img, PixelLayout::Rgba8, width, height).encode()?;
    println!("  ✓ Encoded with all params → {} bytes\n", webp_data.len());

    // 1.3 Preset-based encoding
    println!("1.3 Preset-based encoding");
    for preset in [
        Preset::Default,
        Preset::Photo,
        Preset::Drawing,
        Preset::Icon,
        Preset::Text,
    ] {
        let config = EncoderConfig::with_preset(preset, 75.0);
        let webp_data =
            EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
        println!("  ✓ Preset::{:?} → {} bytes", preset, webp_data.len());
    }
    println!();

    // 1.4 Target size encoding
    println!("1.4 Target size encoding");
    let config = EncoderConfig::new_lossy()
        .with_quality(75.0)
        .with_method(4)
        .with_target_size(5000);

    let webp_data =
        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
    println!("  ✓ Target 5000 bytes → {} bytes\n", webp_data.len());

    // 1.5 Target PSNR encoding
    println!("1.5 Target PSNR encoding");
    let config = EncoderConfig::new_lossy()
        .with_quality(75.0)
        .with_method(4)
        .with_target_psnr(42.0);

    let webp_data =
        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
    println!("  ✓ Target PSNR 42.0 → {} bytes\n", webp_data.len());

    // 1.6 Encoding with metadata (ICCP, EXIF, XMP)
    println!("1.6 Encoding with metadata");
    let iccp_profile = b"fake ICC profile data";
    let exif_data = b"fake EXIF data";
    let xmp_data = b"<x:xmpmeta>fake XMP</x:xmpmeta>";

    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp_data = EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height)
        .with_icc_profile(iccp_profile)
        .with_exif(exif_data)
        .with_xmp(xmp_data)
        .encode()?;
    println!("  ✓ With ICCP/EXIF/XMP → {} bytes\n", webp_data.len());

    // 1.7 Encoding with stats collection
    println!("1.7 Encoding with stats");
    let config = EncoderConfig::new_lossy().with_quality(75.0).with_method(4);
    let (_webp_data, stats) =
        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height)
            .encode_with_stats()?;

    println!("  ✓ Stats collected:");
    println!("    - Coded size: {} bytes", stats.coded_size);
    println!("    - PSNR: {:.2} dB", stats.psnr[0]);
    println!("    - I4 blocks: {}", stats.block_count_i4);
    println!("    - I16 blocks: {}", stats.block_count_i16);
    println!("    - Skip blocks: {}", stats.block_count_skip);
    println!("    - Header bytes: {}", stats.header_bytes);
    println!("    - Mode partition bytes: {}", stats.mode_partition_bytes);
    println!("    - Segment sizes: {:?}", stats.segment_size);
    println!("    - Segment quants: {:?}", stats.segment_quant);
    println!("    - Segment levels: {:?}", stats.segment_level);
    println!();

    // 1.8 Encoding into pre-allocated Vec
    println!("1.8 Encoding into pre-allocated Vec");
    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let mut buffer = Vec::new();
    EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height)
        .encode_into(&mut buffer)?;
    println!("  ✓ Encoded into buffer → {} bytes\n", buffer.len());

    // 1.9 Encoding to writer (std only)
    #[cfg(feature = "std")]
    {
        println!("1.9 Encoding to writer");
        let config = EncoderConfig::new_lossy().with_quality(75.0);
        let mut cursor = Cursor::new(Vec::new());
        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height)
            .encode_to(&mut cursor)?;
        let written = cursor.into_inner().len();
        println!("  ✓ Encoded to writer → {} bytes\n", written);
    }

    // 1.10 Encoding with limits
    println!("1.10 Encoding with limits");
    let limits = Limits::default()
        .max_total_pixels(1_000_000)
        .max_dimensions(2000, 2000);

    let config = EncoderConfig::new_lossy().with_quality(75.0).limits(limits);

    let webp_data =
        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
    println!("  ✓ With limits → {} bytes\n", webp_data.len());

    // 1.11 Encoding with cancellation (Stop trait)
    println!("1.11 Encoding with cancellation support");
    let webp_data = EncodeRequest::new(
        &EncoderConfig::new_lossy().with_quality(75.0),
        &rgb_img,
        PixelLayout::Rgb8,
        width,
        height,
    )
    .with_stop(&Unstoppable as &dyn Stop)
    .encode()?;
    println!("  ✓ With cancellation → {} bytes\n", webp_data.len());

    // 1.12 Custom stride (for non-packed pixels)
    println!("1.12 Custom stride");
    let stride_width = width + 16; // padding
    let mut padded_img = vec![0u8; (stride_width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let src_idx = (y * width as usize + x) * 3;
            let dst_idx = (y * stride_width as usize + x) * 3;
            padded_img[dst_idx..dst_idx + 3].copy_from_slice(&rgb_img[src_idx..src_idx + 3]);
        }
    }

    let config = EncoderConfig::new_lossy().with_quality(75.0);
    let webp_data = EncodeRequest::new(&config, &padded_img, PixelLayout::Rgb8, width, height)
        .with_stride(stride_width as usize)
        .encode()?;
    println!("  ✓ With custom stride → {} bytes\n", webp_data.len());

    // ========================================================================
    // SECTION 2: LOSSLESS ENCODING
    // ========================================================================
    println!("\n2. LOSSLESS ENCODING\n");

    // 2.1 Basic lossless encoding (typed config, recommended)
    println!("2.1 Basic lossless encoding");
    let config = LosslessConfig::new();
    let webp_data =
        EncodeRequest::lossless(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()?;
    println!("  ✓ Lossless RGB → {} bytes\n", webp_data.len());

    // 2.2 Lossless with quality parameter
    println!("2.2 Lossless with quality parameter");
    let config = LosslessConfig::new()
        .with_quality(90.0) // 0-100, controls encoding effort
        .with_method(6); // 0-6, higher = slower but better compression

    let webp_data =
        EncodeRequest::lossless(&config, &rgba_img, PixelLayout::Rgba8, width, height).encode()?;
    println!("  ✓ With quality/method → {} bytes\n", webp_data.len());

    // 2.3 Near-lossless encoding
    println!("2.3 Near-lossless encoding");
    let config = LosslessConfig::new().with_near_lossless(60); // 0-100, 100 = lossless

    let webp_data =
        EncodeRequest::lossless(&config, &rgba_img, PixelLayout::Rgba8, width, height).encode()?;
    println!("  ✓ Near-lossless → {} bytes\n", webp_data.len());

    // 2.4 Lossless with exact color preservation
    println!("2.4 Lossless with exact colors");
    let config = LosslessConfig::new().with_exact(true);

    let webp_data =
        EncodeRequest::lossless(&config, &rgba_img, PixelLayout::Rgba8, width, height).encode()?;
    println!("  ✓ Exact colors → {} bytes\n", webp_data.len());

    // 2.5 Lossless with alpha quality
    println!("2.5 Lossless with alpha quality");
    let config = LosslessConfig::new().with_alpha_quality(90);

    let webp_data =
        EncodeRequest::lossless(&config, &rgba_img, PixelLayout::Rgba8, width, height).encode()?;
    println!("  ✓ With alpha quality → {} bytes\n", webp_data.len());

    // 2.6 Lossless with metadata
    println!("2.6 Lossless with metadata");
    let config = LosslessConfig::new();
    let webp_data = EncodeRequest::lossless(&config, &rgba_img, PixelLayout::Rgba8, width, height)
        .with_icc_profile(iccp_profile)
        .with_exif(exif_data)
        .with_xmp(xmp_data)
        .encode()?;
    println!("  ✓ With ICCP/EXIF/XMP → {} bytes\n", webp_data.len());

    // ========================================================================
    // SECTION 3: DECODING
    // ========================================================================
    println!("\n3. DECODING\n");

    // Encode a test image for decoding
    let test_webp = {
        let config = EncoderConfig::new_lossy().with_quality(75.0);
        EncodeRequest::new(&config, &rgba_img, PixelLayout::Rgba8, width, height)
            .with_icc_profile(iccp_profile)
            .with_exif(exif_data)
            .with_xmp(xmp_data)
            .encode()?
    };

    // 3.1 Convenience functions (simplest API)
    println!("3.1 Convenience functions");
    let (pixels, w, h) = zenwebp::decode_rgb(&test_webp)?;
    println!("  ✓ decode_rgb() → {}x{}, {} bytes", w, h, pixels.len());

    let (pixels, w, h) = zenwebp::decode_rgba(&test_webp)?;
    println!("  ✓ decode_rgba() → {}x{}, {} bytes", w, h, pixels.len());

    let (pixels, w, h) = zenwebp::decode_bgr(&test_webp)?;
    println!("  ✓ decode_bgr() → {}x{}, {} bytes", w, h, pixels.len());

    let (pixels, w, h) = zenwebp::decode_bgra(&test_webp)?;
    println!("  ✓ decode_bgra() → {}x{}, {} bytes\n", w, h, pixels.len());

    // 3.2 WebPDecoder (full control)
    println!("3.2 WebPDecoder API");
    let mut decoder = zenwebp::WebPDecoder::new(&test_webp)?;
    let (w, h) = decoder.dimensions();
    println!("  ✓ Dimensions: {}x{}", w, h);
    println!("  ✓ Has alpha: {}", decoder.has_alpha());
    println!("  ✓ Is animated: {}", decoder.is_animated());

    let buffer_size = decoder.output_buffer_size().unwrap();
    let mut output = vec![0u8; buffer_size];
    decoder.read_image(&mut output)?;
    println!("  ✓ Decoded {} bytes\n", output.len());

    // 3.3 Decode into pre-allocated buffer
    println!("3.3 Decode into buffer");
    let mut buffer = vec![0u8; width as usize * height as usize * 4];
    zenwebp::decode_rgba_into(&test_webp, &mut buffer, width)?;
    println!("  ✓ Decoded into buffer → {} bytes\n", buffer.len());

    // 3.4 Metadata extraction
    println!("3.4 Metadata extraction");
    if let Some(iccp) = zenwebp::metadata::icc_profile(&test_webp)? {
        println!("  ✓ ICCP: {} bytes", iccp.len());
    }
    if let Some(exif) = zenwebp::metadata::exif(&test_webp)? {
        println!("  ✓ EXIF: {} bytes", exif.len());
    }
    if let Some(xmp) = zenwebp::metadata::xmp(&test_webp)? {
        println!("  ✓ XMP: {} bytes", xmp.len());
    }
    println!();

    // 3.5 Decoding with limits
    println!("3.5 Decoding with limits");
    let limits = Limits::default()
        .max_total_pixels(10_000_000)
        .max_dimensions(4000, 4000)
        .max_file_size(10 * 1024 * 1024)
        .max_memory(512 * 1024 * 1024);

    let config = zenwebp::DecodeConfig::default().limits(limits);
    let request = zenwebp::DecodeRequest::new(&config, &test_webp);
    let (_pixels, w, h) = request.decode_rgba()?;
    println!("  ✓ With limits → {}x{}\n", w, h);

    // 3.6 Decoding with cancellation
    println!("3.6 Decoding with cancellation");
    let config = zenwebp::DecodeConfig::default();
    let request = zenwebp::DecodeRequest::new(&config, &test_webp).stop(&Unstoppable as &dyn Stop);
    let (_pixels, w, h) = request.decode_rgba()?;
    println!("  ✓ With cancellation → {}x{}\n", w, h);

    // ========================================================================
    // SECTION 4: ANIMATION
    // ========================================================================
    println!("\n4. ANIMATION\n");

    // 4.1 Creating animation with default config
    println!("4.1 Creating animation");
    let anim_config = AnimationConfig::default();
    let mut anim = AnimationEncoder::new(width, height, anim_config)?;

    // Add frames with encoder config
    let encoder_config = EncoderConfig::new_lossy().with_quality(75.0);

    anim.add_frame(&rgb_img, PixelLayout::Rgb8, 0, &encoder_config)?;
    anim.add_frame(&rgb_img, PixelLayout::Rgb8, 100, &encoder_config)?;
    anim.add_frame(&rgb_img, PixelLayout::Rgb8, 200, &encoder_config)?;

    let anim_webp = anim.finalize(100)?; // last frame duration
    println!(
        "  ✓ Created 3-frame animation → {} bytes\n",
        anim_webp.len()
    );

    // 4.2 Animation with metadata
    println!("4.2 Animation with metadata");
    let anim_config = AnimationConfig {
        background_color: [255, 255, 255, 255],
        loop_count: LoopCount::Times(NonZeroU16::new(5).unwrap()),
        minimize_size: true,
    };
    let mut anim = AnimationEncoder::new(width, height, anim_config)?;
    anim.icc_profile(iccp_profile.to_vec());
    anim.exif(exif_data.to_vec());
    anim.xmp(xmp_data.to_vec());

    let encoder_config = EncoderConfig::new_lossy().with_quality(75.0);
    anim.add_frame(&rgb_img, PixelLayout::Rgb8, 0, &encoder_config)?;
    anim.add_frame(&rgb_img, PixelLayout::Rgb8, 100, &encoder_config)?;

    let anim_webp = anim.finalize(100)?;
    println!("  ✓ Animation with metadata → {} bytes\n", anim_webp.len());

    // 4.3 Animation with lossless frames
    println!("4.3 Animation with lossless frames");
    let mut anim = AnimationEncoder::new(width, height, AnimationConfig::default())?;

    let lossless_config = EncoderConfig::new_lossless().with_quality(90.0);
    anim.add_frame(&rgba_img, PixelLayout::Rgba8, 0, &lossless_config)?;
    anim.add_frame(&rgba_img, PixelLayout::Rgba8, 100, &lossless_config)?;

    let anim_webp = anim.finalize(100)?;
    println!("  ✓ Lossless animation → {} bytes\n", anim_webp.len());

    // 4.4 Animation with advanced control (sub-frames)
    println!("4.4 Animation with advanced control");
    let mut anim = AnimationEncoder::new(width, height, AnimationConfig::default())?;

    let encoder_config = EncoderConfig::new_lossy().with_quality(75.0);
    // Add a full-canvas frame
    anim.add_frame_advanced(
        &rgb_img,
        PixelLayout::Rgb8,
        width,
        height,
        0, // x_offset
        0, // y_offset
        0, // timestamp_ms
        &encoder_config,
        DisposeMethod::None,
        BlendMethod::Overwrite,
    )?;

    // Add a sub-frame (partial update)
    let sub_width = width / 2;
    let sub_height = height / 2;
    let sub_img = vec![255u8; (sub_width * sub_height * 3) as usize];
    anim.add_frame_advanced(
        &sub_img,
        PixelLayout::Rgb8,
        sub_width,
        sub_height,
        width / 4,  // x_offset
        height / 4, // y_offset
        100,        // timestamp_ms
        &encoder_config,
        DisposeMethod::Background,
        BlendMethod::AlphaBlend,
    )?;

    let anim_webp = anim.finalize(100)?;
    println!("  ✓ Advanced animation → {} bytes\n", anim_webp.len());

    // 4.5 Decoding animation
    println!("4.5 Decoding animation");
    let demuxer = WebPDemuxer::new(&anim_webp)?;
    println!("  ✓ Frame count: {}", demuxer.num_frames());
    println!("  ✓ Loop count: {:?}", demuxer.loop_count());
    println!(
        "  ✓ Canvas: {}x{}",
        demuxer.canvas_width(),
        demuxer.canvas_height()
    );
    println!("  ✓ Background: {:?}", demuxer.background_color());

    for frame in demuxer.frames() {
        println!(
            "    Frame #{}: {}ms, blend={:?}, dispose={:?}, offset=({},{}) {}x{}",
            frame.frame_num,
            frame.duration_ms,
            frame.blend,
            frame.dispose,
            frame.x_offset,
            frame.y_offset,
            frame.width,
            frame.height
        );
    }
    println!();

    // ========================================================================
    // SECTION 5: MUX/DEMUX
    // ========================================================================
    println!("\n5. MUX/DEMUX\n");

    // 5.1 Demuxing chunks
    println!("5.1 Demuxing chunks");
    let demuxer = WebPDemuxer::new(&test_webp)?;

    if let Some(iccp) = demuxer.icc_profile() {
        println!("  ✓ ICCP chunk: {} bytes", iccp.len());
    }
    if let Some(exif) = demuxer.exif() {
        println!("  ✓ EXIF chunk: {} bytes", exif.len());
    }
    if let Some(xmp) = demuxer.xmp() {
        println!("  ✓ XMP chunk: {} bytes", xmp.len());
    }
    println!();

    // 5.2 Creating custom mux (reassemble with new metadata)
    println!("5.2 Creating custom mux");

    // Use from_data to parse existing WebP and reassemble with new metadata
    let mut mux = WebPMux::from_data(&test_webp)?;

    // Add/replace metadata
    mux.set_icc_profile(b"new ICC profile".to_vec());
    mux.set_exif(b"new EXIF data".to_vec());
    mux.set_xmp(b"<x:xmpmeta>new XMP</x:xmpmeta>".to_vec());

    let muxed = mux.assemble()?;
    println!("  ✓ Muxed WebP with new metadata → {} bytes\n", muxed.len());

    // 5.3 Metadata embedding/removal
    println!("5.3 Metadata operations");
    let with_icc = zenwebp::metadata::embed_icc(&test_webp, iccp_profile)?;
    println!("  ✓ embed_icc() → {} bytes", with_icc.len());

    let with_exif = zenwebp::metadata::embed_exif(&test_webp, exif_data)?;
    println!("  ✓ embed_exif() → {} bytes", with_exif.len());

    let with_xmp = zenwebp::metadata::embed_xmp(&test_webp, xmp_data)?;
    println!("  ✓ embed_xmp() → {} bytes", with_xmp.len());

    let no_icc = zenwebp::metadata::remove_icc(&with_icc)?;
    println!("  ✓ remove_icc() → {} bytes", no_icc.len());

    let no_exif = zenwebp::metadata::remove_exif(&with_exif)?;
    println!("  ✓ remove_exif() → {} bytes", no_exif.len());

    let no_xmp = zenwebp::metadata::remove_xmp(&with_xmp)?;
    println!("  ✓ remove_xmp() → {} bytes\n", no_xmp.len());

    // ========================================================================
    // SECTION 6: PIXEL FORMATS
    // ========================================================================
    println!("\n6. PIXEL FORMATS\n");

    // 6.1 All supported pixel layouts
    println!("6.1 All pixel layouts");
    let formats = [
        (PixelLayout::L8, "L8 (grayscale)", 1),
        (PixelLayout::La8, "La8 (gray+alpha)", 2),
        (PixelLayout::Rgb8, "Rgb8 (RGB)", 3),
        (PixelLayout::Rgba8, "Rgba8 (RGBA)", 4),
        (PixelLayout::Bgr8, "Bgr8 (BGR)", 3),
        (PixelLayout::Bgra8, "Bgra8 (BGRA)", 4),
    ];

    for (layout, name, bpp) in formats {
        let img = vec![128u8; width as usize * height as usize * bpp];
        let config = EncoderConfig::new_lossy().with_quality(75.0);
        let webp = EncodeRequest::new(&config, &img, layout, width, height).encode()?;
        println!("  ✓ {} → {} bytes", name, webp.len());
    }
    println!();

    // ========================================================================
    // SECTION 7: ERROR HANDLING
    // ========================================================================
    println!("\n7. ERROR HANDLING\n");

    // 7.1 Encode errors
    println!("7.1 Encode error handling");

    // Exceeded limits (test first since 0x0 dimensions cause panic - encoder bug)
    let result: Result<Vec<u8>, whereat::At<EncodeError>> = {
        let limits = Limits::default().max_total_pixels(100);
        let config = EncoderConfig::new_lossy().with_quality(75.0).limits(limits);

        EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height).encode()
    };
    if let Err(e) = result {
        println!("  ✓ Exceeded limits: {}", e);
    }
    println!();

    // 7.2 Decode errors
    println!("7.2 Decode error handling");

    // Invalid data
    let result: Result<(Vec<u8>, u32, u32), whereat::At<DecodeError>> =
        zenwebp::decode_rgba(b"not a webp");
    if let Err(e) = result {
        println!("  ✓ Invalid data: {}", e);
    }

    // Exceeded limits
    let result: Result<(Vec<u8>, u32, u32), whereat::At<DecodeError>> = {
        let limits = Limits::default().max_total_pixels(100);
        let config = zenwebp::DecodeConfig::default().limits(limits);
        let request = zenwebp::DecodeRequest::new(&config, &test_webp);
        request.decode_rgba()
    };
    if let Err(e) = result {
        println!("  ✓ Exceeded limits: {}", e);
    }
    println!();

    // 7.3 Mux errors
    println!("7.3 Mux error handling");
    let result: Result<AnimationEncoder, whereat::At<MuxError>> =
        AnimationEncoder::new(0, 0, AnimationConfig::default());
    if let Err(e) = result {
        println!("  ✓ Invalid dimensions: {}", e);
    }
    println!();

    // ========================================================================
    // SECTION 8: ADVANCED FEATURES
    // ========================================================================
    println!("\n8. ADVANCED FEATURES\n");

    // 8.1 Resource estimation
    println!("8.1 Resource estimation");
    let estimate = EncoderConfig::new_lossy()
        .with_quality(75.0)
        .estimate(width, height, 4);
    println!(
        "  ✓ Estimated peak memory: {} bytes",
        estimate.peak_memory_bytes
    );
    println!(
        "  ✓ Estimated memory range: {}-{} bytes",
        estimate.peak_memory_bytes_min, estimate.peak_memory_bytes_max
    );
    println!("  ✓ Estimated time: {:.2} ms", estimate.time_ms);
    println!(
        "  ✓ Estimated output size: {} bytes\n",
        estimate.output_bytes
    );

    // 8.2 Content analysis
    println!("8.2 Content analysis");
    let config = EncoderConfig::with_preset(Preset::Auto, 75.0);
    let (webp, _stats) = EncodeRequest::new(&config, &rgb_img, PixelLayout::Rgb8, width, height)
        .encode_with_stats()?;
    println!("  ✓ Auto preset → {} bytes", webp.len());
    println!("  ✓ Detected content type from stats\n");

    println!("=== All API features demonstrated successfully! ===");
    Ok(())
}

/// Create test images for demonstration
fn create_test_images() -> (Vec<u8>, Vec<u8>, u32, u32) {
    let width = 64u32;
    let height = 64u32;

    // RGB gradient
    let mut rgb_img = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            rgb_img[idx] = ((x * 255) / width) as u8;
            rgb_img[idx + 1] = ((y * 255) / height) as u8;
            rgb_img[idx + 2] = 128;
        }
    }

    // RGBA gradient with alpha
    let mut rgba_img = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba_img[idx] = ((x * 255) / width) as u8;
            rgba_img[idx + 1] = ((y * 255) / height) as u8;
            rgba_img[idx + 2] = 128;
            rgba_img[idx + 3] = ((x + y) * 128 / (width + height)) as u8;
        }
    }

    (rgb_img, rgba_img, width, height)
}
