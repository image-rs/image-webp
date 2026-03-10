use zencodec::encode::DynEncoderConfig;
use zenwebp::WebpEncoderConfig;

fn take_dyn(_: Box<dyn DynEncoderConfig>) {}

fn main() {
    let config = WebpEncoderConfig::lossy();
    take_dyn(Box::new(config));
    println!("works!");
}
