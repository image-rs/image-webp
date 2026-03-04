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

    // Test different predictor_bits values
    for bits in (2..=9).rev() {
        let blocks = ((width + (1 << bits) - 1) >> bits) * ((height + (1 << bits) - 1) >> bits);
        let config = zenwebp::encoder::vp8l::Vp8lConfig {
            use_predictor: true,
            use_subtract_green: false,
            use_palette: false,
            predictor_bits: bits,
            ..Default::default()
        };
        let vp8l = zenwebp::encoder::vp8l::encode_vp8l(
            &rgb,
            width,
            height,
            false,
            &config,
            &enough::Unstoppable,
        )
        .unwrap();
        let webp = wrap_vp8l_in_riff(&vp8l);
        let path = format!("/tmp/test_pred_bits{}.webp", bits);
        std::fs::write(&path, &webp).unwrap();

        // Count unique modes via dwebp
        let dwebp_ok = std::process::Command::new(libwebp_bin("dwebp"))
            .args([&path, "-o", &format!("{}.ppm", path)])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        // Our decoder
        let mismatches = match zenwebp::decode_rgba(&webp) {
            Ok((data, _, _)) => {
                let total = (width * height) as usize;
                (0..total)
                    .filter(|&i| {
                        data[i * 4] != rgb[i * 3]
                            || data[i * 4 + 1] != rgb[i * 3 + 1]
                            || data[i * 4 + 2] != rgb[i * 3 + 2]
                    })
                    .count()
            }
            Err(_) => usize::MAX,
        };

        println!(
            "  bits={} ({}x{} blocks={}) size={} dwebp={} mismatches={}",
            bits,
            1 << bits,
            1 << bits,
            blocks,
            webp.len(),
            if dwebp_ok { "OK" } else { "FAIL" },
            mismatches
        );
    }
}
