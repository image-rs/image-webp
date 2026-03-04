//! Test predictor-only (no subtract green) to isolate the mismatch.

fn libwebp_bin(name: &str) -> String {
    let dir = std::env::var("LIBWEBP_DIR").unwrap_or_else(|_| "/home/lilith/work/libwebp".into());
    format!("{dir}/examples/{name}")
}

fn wrap_vp8l_in_riff(vp8l_data: &[u8]) -> Vec<u8> {
    let mut webp = Vec::new();
    webp.extend_from_slice(b"RIFF");
    let riff_size = 4 + 8 + vp8l_data.len() + (vp8l_data.len() % 2);
    webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    webp.extend_from_slice(b"VP8L");
    webp.extend_from_slice(&(vp8l_data.len() as u32).to_le_bytes());
    webp.extend_from_slice(vp8l_data);
    if !vp8l_data.len().is_multiple_of(2) {
        webp.push(0);
    }
    webp
}

fn test_config(
    name: &str,
    rgb: &[u8],
    width: u32,
    height: u32,
    config: &zenwebp::encoder::vp8l::Vp8lConfig,
) {
    let vp8l = zenwebp::encoder::vp8l::encode_vp8l(
        rgb,
        width,
        height,
        false,
        config,
        &enough::Unstoppable,
    )
    .unwrap();
    let webp = wrap_vp8l_in_riff(&vp8l);
    let path = format!("/tmp/test_pred_{}.webp", name);
    std::fs::write(&path, &webp).unwrap();

    // Verify with dwebp
    let out = std::process::Command::new(libwebp_bin("dwebp"))
        .args([&path, "-o", &format!("{}.ppm", path)])
        .output();
    let dwebp_ok = matches!(out, Ok(ref o) if o.status.success());

    // Verify with our decoder
    match zenwebp::decode_rgba(&webp) {
        Ok((data, _w, _h)) => {
            let mut mismatches = 0;
            let total = (width * height) as usize;
            let mut first_mismatch = None;
            for i in 0..total {
                let dr = data[i * 4] != rgb[i * 3];
                let dg = data[i * 4 + 1] != rgb[i * 3 + 1];
                let db = data[i * 4 + 2] != rgb[i * 3 + 2];
                if dr || dg || db {
                    if first_mismatch.is_none() {
                        let x = i % width as usize;
                        let y = i / width as usize;
                        first_mismatch = Some(format!(
                            "({},{}) exp=({},{},{}) got=({},{},{})",
                            x,
                            y,
                            rgb[i * 3],
                            rgb[i * 3 + 1],
                            rgb[i * 3 + 2],
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2]
                        ));
                    }
                    mismatches += 1;
                }
            }
            println!(
                "  {}: {} bytes, dwebp={}, mismatches={}/{}",
                name,
                webp.len(),
                if dwebp_ok { "OK" } else { "FAIL" },
                mismatches,
                total
            );
            if let Some(fm) = first_mismatch {
                println!("    First mismatch: {}", fm);
            }
        }
        Err(e) => println!("  {}: decode error: {:?}", name, e),
    }
}

fn main() {
    let path = "/tmp/CID22/original/1001682.png";
    let file = std::fs::File::open(path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let width = info.width;
    let height = info.height;
    let rgb: Vec<u8> = if info.color_type == png::ColorType::Rgba {
        buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]].into_iter())
            .collect()
    } else {
        buf[..info.buffer_size()].to_vec()
    };

    println!("Image: {}x{}", width, height);

    // Test 1: No transforms at all
    test_config(
        "none",
        &rgb,
        width,
        height,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: false,
            use_subtract_green: false,
            use_palette: false,
            ..Default::default()
        },
    );

    // Test 2: Subtract green only
    test_config(
        "sg_only",
        &rgb,
        width,
        height,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: false,
            use_subtract_green: true,
            use_palette: false,
            ..Default::default()
        },
    );

    // Test 3: Predictor only (no subtract green)
    test_config(
        "pred_only",
        &rgb,
        width,
        height,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );

    // Test 4: Both predictor + subtract green
    test_config(
        "pred_sg",
        &rgb,
        width,
        height,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: true,
            use_palette: false,
            predictor_bits: 4,
            ..Default::default()
        },
    );

    // Test 5: Predictor with large blocks (should be trivial path)
    test_config(
        "pred_large",
        &rgb,
        width,
        height,
        &zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: 9,
            ..Default::default()
        },
    );
}
