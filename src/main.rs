// Cargo.toml:
// [dependencies]
// cpal = "0.15"
// anyhow = "1.0"

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{I24, Sample, SampleFormat};

fn main() -> Result<(), anyhow::Error> {
    // Get the default host
    let host = cpal::default_host();

    // Get the default output device
    let device = host
        .default_output_device()
        .expect("No output device available");

    println!("Output device: {}", device.description()?);

    // Get the default output config
    let config = device.default_output_config()?;
    println!("Default output config: {:?}", config);

    // Create a sine wave generator
    let sample_rate = config.sample_rate() as f32;
    let mut sample_clock = 0f32;
    let mut next_value = move || {
        sample_clock = (sample_clock + 1.0) % sample_rate;
        (sample_clock * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin() * 0.2
    };

    // Build an output stream
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    *sample = next_value();
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_output_stream(
            &config.into(),
            move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    *sample = i16::from_sample::<f32>(next_value());
                }
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_output_stream(
            &config.into(),
            move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    *sample = u16::from_sample::<f32>(next_value());
                }
            },
            err_fn,
            None,
        )?,
        _ => return Err(anyhow::Error::msg("Unsupported sample format")),
    };

    // Play the stream
    stream.play()?;

    // Keep the program running
    println!("Playing a sine wave. Press Enter to exit...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
